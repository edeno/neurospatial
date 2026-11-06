"""Forli et al. (2025): bat hippocampus during free flight in a room (Zenodo 15738988).

What was recorded: an Egyptian fruit bat flying freely in a room for about
90 minutes (38 flights) while three Neuropixels probes recorded units in
dorsal hippocampus. Position is 3-D, in metres, sampled at 120 Hz: the
centroid of three reflective markers on the neural recording headstage,
tracked by 16 motion-capture cameras (the paper's Methods, for the indoor
room). Flights are the authors' segmentation at a 0.5 m/s velocity
threshold.

Where it lives: Zenodo record 15738988 (CC-BY-4.0), the demo session
``Dataset_3_14543_240419`` in ``Demo_Session.zip`` (3.3 GB). Only the nine
members the loader needs are read, by HTTP byte range (1.1 GB): the behavior
file, the three Kilosort 4 single-unit tables, the merge list, the three
ripple files (817 MB, which also hold each probe's ripple-channel trace and
every event's waveforms) and the best-probe choice, each verified against its
own SHA-256. The single-unit and ripple tables are MATLAB ``table`` objects,
read with ``mat-io``, which installs only on Python 3.11 or later.

Sources read: the paper (Forli, Fan, Qi & Yartsev 2025, Nature,
doi:10.1038/s41586-025-09341-z, accelerated article preview, Methods), the
record's ``NP_Bat_Demo.m`` (whose header describes every variable) and
``How to use.pdf``, the paper's code at github.com/kevin-qi/ripple-bat, commit
6812188d (``NP_RippleDetect_AF_v0.m``, which builds the ripple tables, and
``NP_Spatial_AF_v9_SWRs.m``, which reads them; read, not reused: the repository
states no license), and the files.

What the loader converts:

- positions: ``r`` as stored, in metres. The file states no unit; it is
  metres because ``r_lim`` spans 5.8 x 5.2 x 2.3 against the paper's
  5.6 x 5.2 x 2.5 m room and ``v_th`` is 0.5 against the paper's 0.5 m/s
  flight threshold (an inferred ``Conversion``).
- spike times: ``spikeTimes_usec`` (the global clock) / 1e6, as the script
  does. ``localSpikeTimes_usec`` is the probe's own clock ("DO NOT USE", the
  script) and is not used.
- putative duplicate units in ``merges_Dataset_*.mat`` are pooled under the
  first cluster's id, as the script does; the pooled train is sorted (the
  script concatenates without sorting). In every unit a spike within 1 ms of
  its predecessor is dropped, the script's rule.
- flight epochs from the 1-based sample indices ``f_smp``. The behavior
  starts before the neural recording, so the flights used for analysis are
  clipped to the recording, taken as the first to last spike of any unit
  (the files hold no recording start or stop).
- ripple events: ``RPL_out.table.t`` of each probe as stored, seconds on the
  LFP's clock, which is the spikes' (the population rate peaks within 10 ms
  before the table's times). ``t`` is the peak of the smoothed ripple power.
  ``events["ripple_candidates_probe<n>"]`` holds every row;
  ``events["swr_probe<n>"]`` the rows at the paper's minimum correlation of
  0.2 with the average event. The table's other columns are not kept.

The declared environment is the room: every voxel of a grid that tiles
``r_lim`` ("boundaries of the room/outdoor arena", the script) exactly, so
it holds the air the bat could fly through, not only the voxels it visited.
Each axis has the whole number of cells nearest to its side over 0.2 m, a
tie going to fewer, larger cells (29 x 26 x 11 cells of 0.2 x 0.2 x 0.209
m): the 2.3 m height is 11.5 cells of 0.2 m, and cells of exactly 0.2 m
would make the grid reach past the floor or the ceiling.
``r_lim`` is used because it is stated in the positions' frame; the paper
gives only the room's size. 3.7% of samples lie beyond the -y bound,
0.1% of them in the file's flights (``bflying``).

Analysis choices (``params``):

- ``bin_size`` 0.2 m: 3-D bins about the size of a bat's body;
- ``min_speed``: the file's ``v_th``, the authors' flight threshold;
- ``analysis_epochs``: flights during the neural recording, so rate maps
  describe the bat in flight; ``train_epochs`` / ``test_epochs``: even and
  odd of those flights, for held-out decoding;
- ``dt`` 0.25 s, as for the 2-D datasets;
- ``ripple_probe``: the probe whose events analyses use, the file's choice
  (``BPC_Dataset_*.mat``).
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._matio import load_mat
from neurospatial.datasets._recording import Conversion, Recording, tracked_mask

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "OWN_3D_ANALYSIS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": "most flight-active units are spatially tuned in flight",
    "quote": (
        '"a large portion of single units that were active during flight '
        "exhibited robust spatial selectivity (place cells: 990 out of 1,386 "
        'flight-active neurons (71%), n = 6 bats" (Results); spatially '
        'tuned cells have "more than one spike per flight, more than 3 Hz '
        "firing at the field peak, a minimum stability of 0.4 and a maximum "
        'pks value of 0.5" (Methods, "Identification of spatially tuned '
        'cells"), stability being "the Spearman correlation between 1D '
        'fields for the first versus the second half of the flights"'
    ),
    "source": (
        "Forli et al. 2025, Nature, Results and Methods (\u201cIdentification "
        "of spatially tuned cells\u201d), read on PubMed Central, PMC12460160"
    ),
    "paper": 0.71,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": {
        "value": 0.152,
        "criterion": (
            "more than half of the flight-active units tuned; fixed before the "
            "run, and not met"
        ),
        "parameters": (
            "the flights of the neural recording (params['analysis_epochs'], "
            "37); flight-active: more than 37 spikes in them (204 of 289 "
            "units); 3-D compute_spatial_rates (diffusion KDE, bandwidth "
            "0.2 m, our choice) on rec.env's 0.2 m voxels; tuned: peak above "
            "3 Hz and a Spearman compare_spatial_rates of at least 0.4 between "
            "the first 18 and the last 19 flights: 31 of 204"
        ),
        "differs": (
            "the paper's fields are 1-D, one per clustered flight path (0.15 m "
            "bins, a seven-bin Gaussian window); ours are one 3-D map pooling "
            "every path, whose sparsely visited voxels are too noisy to be "
            "stable (every unit peaks above 3 Hz; the median first-half "
            "against second-half Spearman correlation is 0.15 over a median "
            "183 shared voxels, and even against odd flights 0.13)",
            "the one-field condition (pks at most 0.5) needs the 1-D fields",
            "flight-active is not defined beyond the spike condition; we use "
            "more than one spike per flight on average",
            "the paper pools 23 sessions of 6 bats; this is one demo session",
        ),
    },
    "why": (
        "The paper's place cells are defined on 1-D fields of each clustered "
        "flight path, and neurospatial has no flight-path clustering or path "
        "fields. A 3-D attempt on the room's voxels is recorded in "
        "'attempted'; it is not the paper's analysis."
    ),
}

#: A test against chance of neurospatial's own 3-D analysis of the ripple events. It
#: is not the paper's replay analysis and makes no claim about it: this is why
#: it is apart from ``WORKED_RESULT``, the record of the paper's finding.
OWN_3D_ANALYSIS: dict[str, Any] = {
    "status": "not met",
    "what": (
        "a test against chance of our own analysis in the room's 3-D voxels, not "
        "the paper's replay analysis and not comparable with its numbers"
    ),
    "property": (
        "windows around ripple events are judged trajectories more often than "
        "the same windows after circularly shifting the spikes"
    ),
    "criterion": (
        "the count of judged windows exceeds the 95th percentile of the counts "
        "of each of two nulls, 20 draws each (seed 2026): every unit shifted "
        "by its own offset, and all units by one common offset, circularly "
        "over the neural recording, by at least 10 s; fixed before the run, "
        "and not met"
    ),
    "n_judged": 2,
    "n_windows": 2535,
    "n_units": 134,
    "null_own_offsets": (0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 0),
    "null_common_offset": (2, 2, 0, 0, 1, 2, 1, 6, 3, 2, 0, 2, 2, 2, 1, 2, 3, 4, 1, 1),
    "window_half_width": 0.15,
    "bandwidth": 0.2,
    "time_bin": 0.02,
    "time_step": 0.005,
    "max_jump": 0.6,
    "min_distance": 1.2,
    "parameters": (
        "events['swr_probe2'], peak +/- 0.15 s, overlapping windows merged, "
        "kept when inside the neural recording and clear of the file's "
        "flights; units with more spikes in the analysed flights than there "
        "are flights, at most 2 Hz (one over the mean interspike interval); "
        "fields from 3-D compute_spatial_rates over the flights (diffusion "
        "KDE, bandwidth 0.2 m) on rec.env's voxels; decoded in 20 ms bins "
        "stepped by 5 ms with the default prior; judged by "
        "PFEIFFER_FOSTER_2013's rule, a MAP step under 0.6 m over at least 11 "
        "bins and at least 1.2 m from start to end. Those two distances are "
        "our choice: the published 20 and 40 cm scaled from a 2 m arena to "
        "this 5.8 m room, rounded"
    ),
    "held_out_flights": {"median_error_m": 2.197, "shuffled_error_m": 2.380},
    "planted_recall": {
        "judged": 1,
        "planted": 300,
        "reading": (
            "replay planted into 300 of the windows, keeping every spike's time "
            "and drawing its cell from these rate maps along a real flight played "
            "15 times faster, is judged a trajectory 0.3% of the time. So this "
            "judge passes almost no known replay here, and 2 of 2,535 is not "
            "evidence that the bat replays nothing. This is a limit of what the "
            "number can say: planted events come from the model the decoder "
            "assumes, and compression 15 is not calibrated for bats"
        ),
    },
    "watch_out": (
        "The 3-D decode of the ripple events is neurospatial's own analysis, "
        "not the paper's: its replay analysis was 1-D, along clustered flight "
        "paths, which neurospatial cannot build yet. Its fixed test against chance "
        "was not met (2 of 2,535 windows judged trajectories, against 95th "
        "percentiles of 2 and 4.1 with shifted spikes), and this says nothing "
        "about replay in this session. 3-D place fields here barely reproduce "
        "between halves of the session (median Spearman correlation 0.15), and "
        "held-out flights decode to a median 2.2 m from the bat, against 2.4 m "
        "by chance, so these maps cannot show a trajectory."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("matio", "fsspec")

BIN_SIZE = 0.2
DT = 0.25
PROBES = (1, 2, 3)
#: Spikes of one unit closer than this are one spike counted twice, s.
CLOSE = 0.001
#: Least correlation of an event's waveform with the average event, for it to
#: count as a sharp-wave ripple (the paper's Methods).
MIN_SWR_CORRELATION = 0.2
#: The LFP's sampling rate, Hz: NP_Bat_Demo.m's '1250 Hz, default', and the
#: median step of ``t_ds``.
LFP_RATE = 1250.0
#: Mean rate above which NP_Bat_Demo.m sets a unit aside as a putative
#: interneuron, Hz.
INTERNEURON_RATE = 2.0
#: The paper's indoor flight room, m (Methods).
PAPER_ROOM = (5.6, 5.2, 2.5)
_PAPER = "the Methods of Forli et al. 2025"
_SCRIPT = "the record's NP_Bat_Demo.m"
_DETECTOR = (
    "NP_RippleDetect_AF_v0.m in the paper's code (github.com/kevin-qi/ripple-bat, "
    "commit 6812188d)"
)

_RIPPLES = (
    "The ripple events are the authors', and neurospatial did not detect them. "
    f"The files give them at 1 kHz (RPL_out.Fs; {_DETECTOR}), where {_PAPER} "
    "state the LFP was down-sampled to 500 Hz for processing. "
    "Their ripple band is 100 to 200 Hz, inside the 0.5 to 200 Hz the headstage "
    "recorded as LFP. Flight is excluded with the wing beats of the headstage's "
    "accelerometer, not the file's flights."
)
_REST = (
    "The ripple events are outside flight, and nothing here says the bat was "
    "awake: the files hold no sleep scoring, and a resting bat may doze."
)
_LFP_SCALE = (
    "rec.lfp is in arbitrary units: red_out.voltage_scaling is a "
    "'voltage scaling factor, to get the actual LFP value' "
    f"({_SCRIPT}), with no unit, and {_PAPER} state the LFP was amplified "
    "125 to 250 times, which the factor may or may not include. Compare "
    "channels of one session; do not read the values as microvolts."
)
_PROCESSED = (
    "Positions are the authors' processed behavior, not raw tracking: "
    f"{_SCRIPT} says 'Processing mainly includes smoothing of position and "
    f"flight segmentation', and {_PAPER} that positions 'were preprocessed ... "
    "to obtain continuous and smooth 3D positional data'. Speeds and "
    "held-sample counts describe that processing as much as the tracking."
)

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "Dataset_3_14543_240419": (
        r"runs of 11 or more identical position samples",
        r"samples farther from the last believable sample",
        r"lie outside the declared environment",
        r"units have no spikes in the analysed epochs",
        r"of the \d+ SWRs on probe 2 lie in the file's flight samples",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "Dataset_3_14543_240419": (
        "This is the record's one demo session; the paper reports 23 sessions "
        f"from six bats ({_PAPER}), and the record holds no other.",
        OWN_3D_ANALYSIS["watch_out"],
        "rec.lfp holds 384 channels of 6.6 million samples: one channel is 53 MB "
        "in memory and all of them 405 GB, so ask rec.lfp.fetch for the "
        "channels you need. Their times are red_out.t_ds as stored: 55 steps "
        "exceed 1.5 sampling periods, the longest 7.6 ms.",
    ),
}


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read the demo session's members into a 3-D :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The directory holding the fetched zip members.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.
    """
    import pandas as pd
    from scipy.io import loadmat

    folder = Path(path_or_stream)
    names = [Path(member.path).name for member in entry.members]
    behavior = loadmat(folder / _named(names, "Extracted_Behavior_"), squeeze_me=True)
    groups = _merge_groups(folder / _named(names, "merges_"))
    best = loadmat(folder / _named(names, "BPC_"))["best_prb_ch"]
    ripple_probe = int(best[0, 0])
    units: dict[str, NDArray[np.float64]] = {}
    dropped = 0
    for probe in PROBES:
        probe_trains, n_dropped = _probe_units(
            folder / _named(names, f"SU_kilosort4_outdir_probe{probe}."),
            probe,
            groups[probe - 1],
        )
        units.update(probe_trains)
        dropped += n_dropped

    times = np.asarray(behavior["t"], dtype=np.float64)
    positions = np.asarray(behavior["r"], dtype=np.float64)
    room = np.asarray(behavior["r_lim"], dtype=np.float64)
    flights = np.asarray(behavior["f_smp"], dtype=int) - 1
    flight_windows = np.column_stack([times[flights[:, 0]], times[flights[:, 1]]])
    trains = [train for train in units.values() if train.size]
    recorded = (
        min(float(train[0]) for train in trains),
        max(float(train[-1]) for train in trains),
    )
    neural_flights = _clip(flight_windows, recorded)
    ripples = {
        probe: _ripple_file(folder / _named(names, f"RPL_probe{probe}."))
        for probe in PROBES
    }
    events = _ripple_events(ripples)
    lfp = _lfp_source(entry, ripples, ripple_probe, folder)
    env, placement = _declared_room(positions, room, entry.session)
    ids = list(units)
    table = pd.DataFrame(
        {
            "brain_region": "dorsal hippocampus",
            "label": None,
            "probe": [int(name.split("-")[0].removeprefix("probe")) for name in ids],
        },
        index=pd.Index(ids, name="unit_id"),
    )
    v_th = float(behavior["v_th"])
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"Egyptian fruit bat {_bat_id(behavior['options'])}",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the session's synchronized clock (t, 'already synchronized with the "
            f"other data streams', {_SCRIPT}), from {times[0]:.1f} s, before the "
            "neural recording starts"
        ),
        times=times,
        spike_times=tuple(units.values()),
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            "the room, r_lim ('boundaries of the room', "
            f"{_SCRIPT}): {_sides(room)} m, every voxel of a grid tiling it; "
            f"{_PAPER} state an indoor room of "
            f"{' x '.join(f'{side:g}' for side in PAPER_ROOM)} m"
        ),
        positions=positions,
        position_units="m",
        tracking=(
            "the head: the centroid of three reflective markers on the neural "
            "recording headstage, tracked by 16 motion-capture cameras at 120 Hz "
            f"({_PAPER})"
        ),
        epochs={
            "flight": flight_windows,
            "neural_recording": np.array([recorded]),
        },
        events=events,
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": v_th,
            "analysis_epochs": neural_flights,
            "train_epochs": neural_flights[::2],
            "test_epochs": neural_flights[1::2],
            "ripple_probe": ripple_probe,
        },
        conversions=(
            Conversion(
                "positions: r as stored, in m",
                "inferred",
                "the file states no unit; r_lim spans "
                f"{_sides(room)} against the "
                f"{' x '.join(f'{side:g}' for side in PAPER_ROOM)} m room and v_th "
                f"is {v_th:g} against the 0.5 m/s flight threshold in {_PAPER}",
            ),
            Conversion(
                "spike_times: spikeTimes_usec (the global clock) / 1e6",
                "stated",
                f"{_SCRIPT}: spikeTimes_usec 'aligned to global clock' (in "
                "microseconds), divided by 1e6; localSpikeTimes_usec 'DO NOT USE'",
            ),
            Conversion(
                "spike_times: the merges_Dataset_*.mat groups pooled under the "
                "first cluster's id, sorted",
                "stated",
                f"{_SCRIPT} pools them (putative duplicates, by distance and "
                "cross-correlogram) without sorting",
            ),
            Conversion(
                f"spike_times: in every unit, a spike within {CLOSE * 1e3:g} ms of "
                f"its predecessor dropped ({dropped} spikes)",
                "stated",
                f"{_SCRIPT}: 'Clean up duplicated spikes, closer than 1 ms'",
            ),
            Conversion(
                f"epochs['flight']: times at the 1-based sample indices f_smp "
                f"({len(flight_windows)} flights)",
                "stated",
                f"{_SCRIPT}: f_smp, 'start and stop samples of each flight', "
                "indexes t in MATLAB",
            ),
            Conversion(
                "epochs['neural_recording']: the first to the last spike of any "
                f"unit, {recorded[0]:.2f} to {recorded[1]:.2f} s; "
                "params['analysis_epochs']: the flights clipped to it",
                "inferred",
                "the files hold no neural recording start or stop; "
                f"{len(trains)} units fire about "
                f"{sum(t.size for t in trains) / (recorded[1] - recorded[0]):.0f} "
                "spikes/s together, so the first and last spike lie within a "
                "fraction of a second of its edges",
            ),
            placement,
            lfp.unit_basis,
            lfp.times_basis,
            Conversion(
                "events['ripple_candidates_probe<n>']: RPL_out.table.t as stored, "
                "s; events['swr_probe<n>']: its rows with corr at least "
                f"{MIN_SWR_CORRELATION:g}",
                "stated",
                f"{_DETECTOR} takes t from findpeaks on the ripple power, in "
                "seconds on the LFP's time vector, which "
                f"{_SCRIPT} states is 'synchronized with the other data "
                f"streams'; {_PAPER} keep events of 'a minimum correlation value "
                "of 0.2'",
            ),
            Conversion(
                f"params['ripple_probe']: {ripple_probe}, best_prb_ch of "
                "BPC_Dataset_*.mat",
                "stated",
                f"{_SCRIPT}: 'Best probe and channel for LFP analysis'; "
                "NP_Spatial_AF_v9_SWRs.m in the paper's code loads that probe's "
                "ripple file",
            ),
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _before_recording(times, flight_windows, recorded),
            _interneurons(units),
            _PROCESSED,
            _swrs_in_flight(
                events[f"swr_probe{ripple_probe}"],
                len(ripples[ripple_probe].table),
                ripple_probe,
                times,
                np.asarray(behavior["bflying"], dtype=bool),
            ),
            _RIPPLES,
            _REST,
            _LFP_SCALE,
        ),
        lfp=lfp,
    )


def _named(names: list[str], prefix: str) -> str:
    """The one member name starting with ``prefix``."""
    (name,) = [name for name in names if name.startswith(prefix)]
    return name


def _merge_groups(path: Path) -> list[list[list[int]]]:
    """Per probe, the groups of cluster ids to pool (``merges{probe}{group}``)."""
    from scipy.io import loadmat

    merges = loadmat(path)["merges"]
    return [
        [
            np.asarray(cell[j, 0]).ravel().astype(int).tolist()
            for j in range(cell.shape[0])
        ]
        for cell in merges[:, 0]
    ]


def _unwrapped(value: Any) -> Any:
    """``value`` out of the one-element object arrays ``mat-io`` wraps it in."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _good_units(path: Path) -> Any:
    """The ``out.good_units`` table of a Kilosort 4 single-unit file."""
    return _unwrapped(load_mat(path)["out"]["good_units"])


class _RippleFile(NamedTuple):
    """What the loader reads of an ``RPL_probe*.mat``."""

    table: Any  # RPL_out.table: t, corr and others, one row per event
    probe_info: Any  # RPL_out.probe_info: ch, xPos, yPos, one row per LFP channel
    channel: int  # RPL_out.ch: the probe's best channel for SWRs


def _ripple_file(path: Path) -> _RippleFile:
    """The ripple table, the probe's channels and its best channel for SWRs."""
    out = load_mat(path)["RPL_out"]
    return _RippleFile(
        table=_unwrapped(out["table"]),
        probe_info=_unwrapped(out["probe_info"]),
        channel=int(np.ravel(_unwrapped(out["ch"]))[0]),
    )


def _lfp_source(
    entry: SessionEntry,
    ripples: dict[int, _RippleFile],
    ripple_probe: int,
    folder: Path,
) -> LFPSource:
    """The session's LFP: each probe's member, one channel per recorded site."""
    channels: dict[str, str] = {}
    held: dict[int, tuple[str, ...]] = {}
    for probe, ripple in ripples.items():
        info = ripple.probe_info
        names = {
            _channel_name(probe, int(ch)): (
                f"probe {probe}, channel {int(ch)}, {x:g} um across and {y:g} um "
                "from the tip"
                + (
                    ", the channel the record keeps for this probe's SWRs"
                    if int(ch) == ripple.channel
                    else ""
                )
            )
            for ch, x, y in zip(info["ch"], info["xPos"], info["yPos"], strict=True)
        }
        channels.update(names)
        held[probe] = tuple(names)
    return LFPSource(
        dataset=entry.info.dataset,
        files=tuple(
            dataclasses.replace(file, channels=held[_probe_of(file.path)])
            for file in entry.lfp_files
        ),
        sampling_rate=LFP_RATE,
        channels=channels,
        unit="arbitrary",
        unit_basis=Conversion(
            "LFP: red_out.lfp (int16) * red_out.voltage_scaling; no unit is stated",
            "stated",
            f"{_SCRIPT}: 'voltage scaling factor, to get the actual LFP value', "
            "and the same product in its code",
        ),
        times_basis=Conversion(
            "LFP times: red_out.t_ds as stored, s",
            "stated",
            f"{_SCRIPT}: 'time vector for the LFP, synchronized with the other "
            "data streams'",
        ),
        reader=read_lfp,
        ripple_channels=(_channel_name(ripple_probe, ripples[ripple_probe].channel),),
        cache=folder.parents[1],
    )


def _channel_name(probe: int, channel: int) -> str:
    """``"probe2-ch388"``: a channel's name in ``rec.lfp``."""
    return f"probe{probe}-ch{channel}"


def _probe_of(path: str | Path) -> int:
    """The probe an ``LFP_probe<n>.mat`` holds."""
    found = re.search(r"LFP_probe(\d+)\.mat$", str(path))
    assert found is not None  # the registry's file names
    return int(found.group(1))


def read_lfp(
    sources: tuple[str | Path, ...], channels: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels of ``LFP_probe*.mat``, MATLAB v7.3 (HDF5) files.

    Parameters
    ----------
    sources : tuple of path
        The probes' files.
    channels : tuple of str
        Channel names, ``probe<n>-ch<id>``.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels)
        ``red_out.lfp * red_out.voltage_scaling``, in the order of ``channels``.
    times : ndarray, shape (n_time,)
        ``red_out.t_ds``, seconds.

    Raises
    ------
    ValueError
        If the probes asked for do not share one time vector.
    """
    import h5py

    columns: dict[str, NDArray[np.float64]] = {}
    times: NDArray[np.float64] | None = None
    for source in sources:
        probe = _probe_of(source)
        with h5py.File(source, "r") as file:
            out = file["red_out"]
            # MATLAB's columns are HDF5's rows: lfp is (n_channels, n_time).
            ids = np.asarray(out["channelMap"]).ravel().astype(int).tolist()
            scale = float(np.asarray(out["voltage_scaling"]).item())
            clock = np.asarray(out["t_ds"], dtype=np.float64).ravel()
            for row, channel in enumerate(ids):
                name = _channel_name(probe, channel)
                if name in channels:
                    columns[name] = np.asarray(out["lfp"][row], np.float64) * scale
        if times is not None and not np.array_equal(times, clock):
            msg = (
                f"{Path(source).name} has its own time vector, so its channels "
                "cannot share rows with the other probe's; fetch one probe at a "
                "time"
            )
            raise ValueError(msg)
        times = clock
    assert times is not None  # fetch asks for at least one file
    return np.column_stack([columns[name] for name in channels]), times


def _ripple_events(ripples: dict[int, _RippleFile]) -> dict[str, NDArray[np.float64]]:
    """Each probe's ripple peaks: every row, and the rows that count as SWRs."""
    events = {}
    for probe, ripple in ripples.items():
        peaks = np.asarray(ripple.table["t"], dtype=np.float64)
        correlation = np.asarray(ripple.table["corr"], dtype=np.float64)
        events[f"ripple_candidates_probe{probe}"] = peaks
        events[f"swr_probe{probe}"] = peaks[correlation >= MIN_SWR_CORRELATION]
    return events


def _swrs_in_flight(
    swrs: NDArray[np.float64],
    n_candidates: int,
    probe: int,
    times: NDArray[np.float64],
    flying: NDArray[np.bool_],
) -> str:
    """How many of a probe's SWRs lie in the behavior file's flight samples."""
    nearest = np.clip(np.searchsorted(times, swrs), 0, times.size - 1)
    n_flying = int(flying[nearest].sum())
    return (
        f"{n_flying} of the {swrs.size} SWRs on probe {probe} lie in the file's "
        f"flight samples (bflying), since the detector found flight another way. "
        f"Of {n_candidates} rows of that probe's table, {swrs.size} reach the "
        f"paper's minimum correlation of {MIN_SWR_CORRELATION:g} "
        f"(events['swr_probe{probe}']); events['ripple_candidates_probe{probe}'] "
        "holds them all, including a burst of rows at about 10 Hz in the first "
        "seconds of the recording, which the files do not explain."
    )


def _probe_units(
    path: Path, probe: int, groups: list[list[int]]
) -> tuple[dict[str, NDArray[np.float64]], int]:
    """One probe's units, ``probe<n>-<cluster>``, and how many spikes were dropped."""
    table = _good_units(path)
    trains = {
        int(cluster): np.asarray(spikes, dtype=np.float64).ravel() / 1e6
        for cluster, spikes in zip(
            table["cluster_id"], table["spikeTimes_usec"], strict=True
        )
    }
    for group in groups:
        if missing := [c for c in group if c not in trains]:
            msg = (
                f"probe {probe}: the merge list pools clusters {group}, but "
                f"{path.name} has no good unit {missing}; the merge list and the "
                "unit table are from different sortings"
            )
            raise ValueError(msg)
        trains[group[0]] = np.concatenate([trains.pop(c) for c in group])
    out, dropped = {}, 0
    for cluster in sorted(trains):
        spikes = np.sort(trains[cluster])
        kept = drop_close_spikes(spikes)
        dropped += spikes.size - kept.size
        out[f"probe{probe}-{cluster}"] = kept
    return out, dropped


def drop_close_spikes(spikes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Drop each spike within 1 ms of the spike before it, as the record's script does.

    ``NP_Bat_Demo.m`` removes ``s(find(diff(s) < 0.001) + 1)``: each spike is
    compared with its predecessor in the train, whether or not that one is
    kept.

    Parameters
    ----------
    spikes : ndarray, shape (n_spikes,)
        Sorted spike times in seconds.

    Returns
    -------
    ndarray, shape (n_kept,)
    """
    close = np.flatnonzero(np.diff(spikes) < CLOSE) + 1
    return np.delete(spikes, close)


def _bat_id(options: Any) -> str:
    """The bat's id, the second entry of ``options.unique_ID``."""
    return str(np.asarray(options["unique_ID"]).item()[1])


def _clip(
    windows: NDArray[np.float64], span: tuple[float, float]
) -> NDArray[np.float64]:
    """``windows`` (n, 2) clipped to ``span``; rows left empty are dropped."""
    clipped = np.column_stack(
        [np.maximum(windows[:, 0], span[0]), np.minimum(windows[:, 1], span[1])]
    )
    kept: NDArray[np.float64] = clipped[clipped[:, 1] > clipped[:, 0]]
    return kept


def _sides(room: NDArray[np.float64]) -> str:
    """``"5.8 x 5.2 x 2.3"``: the room's sides, from ``r_lim`` (n_dims, 2)."""
    return " x ".join(f"{side:.3g}" for side in np.ptp(room, axis=1))


def _declared_room(
    positions: NDArray[np.float64], room: NDArray[np.float64], session: str
) -> tuple[Environment, Conversion]:
    """The room as an environment: a grid tiling ``r_lim``, every voxel active."""
    from neurospatial import Environment

    sides = np.ptp(room, axis=1)
    n_cells = cell_counts(sides, BIN_SIZE)
    cells = sides / n_cells
    env = Environment.from_samples(
        positions,
        bin_size=tuple(float(b) for b in cells),
        dimension_ranges=[(float(low), float(high)) for low, high in room],
        infer_active_bins=False,
        units="m",
        name=f"forli_2025 {session}",
    )
    # The paper's room, centred on r_lim in x and y, from r_lim's floor in z.
    paper = np.mean(room, axis=1)[:, None] + np.outer(PAPER_ROOM, [-0.5, 0.5])
    paper[2] = [room[2, 0], room[2, 0] + PAPER_ROOM[2]]
    residuals = ", ".join(
        f"{axis} {side / stated - 1:+.1%}"
        for axis, side, stated in zip("xyz", sides, PAPER_ROOM, strict=True)
    )
    placement = Conversion(
        "environment: every voxel of a grid tiling r_lim exactly, "
        f"{' x '.join(str(int(n)) for n in n_cells)} cells of "
        f"{' x '.join(f'{c:.3g}' for c in cells)} m; the file's frame, not fitted",
        "stated",
        f"r_lim, 'boundaries of the room/outdoor arena' ({_SCRIPT}); its sides "
        f"against the paper's room: {residuals}. "
        f"{_share_outside(positions, room):.2%} of tracked samples lie outside "
        f"r_lim{_where_outside(positions, room)}; the paper's room centred on "
        "r_lim (from its floor in z) would leave "
        f"{_share_outside(positions, paper):.2%} outside",
    )
    return env, placement


def cell_counts(sides: NDArray[np.float64], bin_size: float) -> NDArray[np.float64]:
    """Whole numbers of cells per axis nearest to ``bin_size``, ties to fewer.

    ``sides / bin_size`` is rounded to 6 decimals first, so float noise
    (2.3 / 0.2 is 11.499999999999998) does not decide a tie.

    Parameters
    ----------
    sides : ndarray, shape (n_dims,)
    bin_size : float

    Returns
    -------
    ndarray, shape (n_dims,)
        At least one cell per axis.

    Examples
    --------
    >>> cell_counts(np.array([5.8, 2.3, 2.35]), 0.2).tolist()
    [29.0, 11.0, 12.0]
    """
    ratio = np.round(np.asarray(sides, dtype=np.float64) / bin_size, 6)
    counts: NDArray[np.float64] = np.maximum(np.ceil(ratio - 0.5), 1.0)
    return counts


def _share_outside(positions: NDArray[np.float64], box: NDArray[np.float64]) -> float:
    """Share of tracked samples outside the box ``box``, shape (n_dims, 2)."""
    tracked = positions[tracked_mask(positions)]
    inside = np.all((tracked >= box[:, 0]) & (tracked <= box[:, 1]), axis=1)
    return float(1 - inside.mean())


def _where_outside(positions: NDArray[np.float64], room: NDArray[np.float64]) -> str:
    """``" (all past -y, up to 1.63 m past it; ...)"``; empty if none is outside."""
    tracked = tracked_mask(positions)
    below = (positions < room[:, 0]) & tracked[:, None]
    above = (positions > room[:, 1]) & tracked[:, None]
    if not (below.any() or above.any()):
        return ""
    sides = ", ".join(
        f"{int(mask[:, axis].sum())} past {sign}{'xyz'[axis]}"
        for axis in range(room.shape[0])
        for sign, mask in (("-", below), ("+", above))
        if mask[:, axis].any()
    )
    beyond = np.maximum(room[:, 0] - positions, positions - room[:, 1])
    farthest = float(np.nanmax(np.where(below | above, beyond, np.nan)))
    return (
        f" ({sides}, up to {farthest:.2f} m past a bound: either r_lim leaves "
        "out a place the bat lands, or the tracking there is off, and the files "
        "do not say which)"
    )


def _before_recording(
    times: NDArray[np.float64],
    flights: NDArray[np.float64],
    recorded: tuple[float, float],
) -> str:
    """How the behavior's span sits against the neural recording's."""
    n_out = int(np.sum((flights[:, 1] <= recorded[0]) | (flights[:, 0] >= recorded[1])))
    return (
        f"The behavior runs from {times[0]:.1f} to {times[-1]:.1f} s and the "
        f"spikes from {recorded[0]:.1f} to {recorded[1]:.1f} s: {n_out} of the "
        f"{len(flights)} flights in epochs['flight'] "
        f"{'lies' if n_out == 1 else 'lie'} wholly outside the neural recording, "
        "so analyses use params['analysis_epochs'], the flights clipped to it."
    )


def _interneurons(units: dict[str, NDArray[np.float64]]) -> str:
    """The units NP_Bat_Demo.m would set aside as putative interneurons."""
    rates = [
        1 / float(np.mean(np.diff(train))) if train.size > 1 else 0.0
        for train in units.values()
    ]
    n_fast = int(np.sum(np.asarray(rates) > INTERNEURON_RATE))
    return (
        f"{n_fast} of {len(units)} units fire above {INTERNEURON_RATE:g} Hz on "
        f"average (one over the mean interspike interval, after the 1 ms rule; "
        f"{_SCRIPT} computes it before); the script sets such units aside as "
        "putative interneurons before its place analyses. All are kept here, "
        "with label None."
    )
