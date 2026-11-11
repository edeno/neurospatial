"""Diba & Buzsáki (2007): rat CA1 and CA3 on linear tracks, from CRCNS hc-3.

What was recorded: Kamran Diba's three rats, ``gor01``, ``pin01`` and
``vvp01``, implanted with 32- or 64-site silicon probes (4 or 8 shanks of 8
sites) in dorsal CA1 and CA3, recorded with a Neuralynx DigitaLynx at
32,552 Hz while they ran back and forth on a linear track for water at each
end (the Supplementary Methods: tracks of 79, 125 and 170 cm, 6.2 cm wide,
with 21 x 21 cm platforms at the ends). CRCNS hc-3 (Mizuseki, Sirota,
Pastalkova, Diba & Buzsáki 2013) publishes 27 of their sessions, recorded on
seven days: linear tracks (hc-3's ``linearOne`` and ``linearTwo``), a T-maze,
an open field, a wheel and one sleep session. Every one is offered here.

Which sessions are the paper's is inferred. No source lists the 2007 paper's
sessions. Maboudi et al. (2018), whose authors include Diba, analysed 18
linear-track sessions "previously used by [Diba and Buzsáki, 2007]", and the
output of their ``Figure2.ipynb`` (github.com/kemerelab/
UncoveringTemporalStructureHippocampus) lists them; 6 of those are in hc-3
(``PAPER_SESSIONS``). The others hc-3 holds are Diba's too, but nothing read
ties them to the paper.

hc-3 holds no position for any of these sessions: their archives have no
``.whl`` file (the 27 were read member by member, 2026-11-11). One session,
``2006-6-7_16-40-19`` (the default), has positions from another source:
Maboudi et al. 2018 released it as a "sample recording epoch"
(``data/fig1.nel``, a dill pickle of nelpy objects, in eLife's archived copy
of their repository), with the rat's 2-D position at 60 Hz. Its spikes are
on hc-3's clock: 95.3% of the file's 258,427 spike times fall within 1 µs of
an hc-3 spike of the same session (18% at a 0.1 s lag). The other 26 sessions
load with spikes, units and LFP only.

Sources read: the paper and its Supplementary Methods (Nature Neuroscience
10:1241-1242), Maboudi et al. 2018 (eLife 7:e34467) and its repository's
notebooks, hc-3's data description (``crcns-hc3-data-description.pdf``,
version 0.9) and metadata tables, its original docs (``README.txt``), nelpy's
``nelpy/io/hc3.py`` (MIT), and the files.

What the loader converts:

- ``.res.N``: spike samples / the ``.xml``'s 32,552 Hz, seconds from the
  start of the session's file.
- units: every cluster from 2 up on each shank (hc-3: cluster 0 is
  artefact, 1 unsortable spikes), id ``"<shank>.<cluster>"``. hc-3 sorted
  all sessions of a day together, so a cluster is the same unit in every
  session of its day. ``brain_region`` is hc-3's electrode table's region of
  the shank; ``label`` is hc-3's cell table's type: ``pyramidal`` ('p'),
  ``interneuron`` ('i') or ``unidentified`` ('n').
- ``2006-6-7_16-40-19``'s positions: the file's ``pos2d`` (x and y, cm), the
  position Maboudi et al. used for place fields, as stored. Its unit is
  inferred (see ``rec.conversions``).
- LFP (on request, ``rec.lfp.fetch``): the ``.eeg`` member of the session's
  archive, int16 interleaved over every channel at 1,252 Hz, times
  ``k / 1252`` s, in µV (the ``.xml``'s 20 V range over 16 bits, gain 1,000:
  0.305 µV per step).
- times, without positions: the LFP's sample clock, 1,252 Hz over the
  ``.eeg`` file, so that the session's span is known; there is no tracking.

Analysis choices (``params``, the session with positions): ``bin_size`` 2 cm;
``max_speed`` 10 cm/s, the paper's immobility ("When the animal's speed was
≤ 10 cm s-1"); ``platform`` 21 cm, the paper's end platforms;
``train_epochs`` and ``test_epochs``, alternating 60 s blocks.
"""

from __future__ import annotations

import csv
import functools
import io
import itertools
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._checks import _count
from neurospatial.datasets._lfp import LFPSource
from neurospatial.datasets._neuroscope import (
    read_acquisition,
    read_eeg,
    read_res_clu,
)
from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._pickled import inert_fields, read_dill_pickle
from neurospatial.datasets._recording import Conversion, Recording, tracked_mask
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "PAPER_SESSIONS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

_PAPER = "Diba & Buzsáki 2007"
_SUPPLEMENT = "the Supplementary Methods of Diba & Buzsáki 2007"
_MABOUDI = "Maboudi et al. 2018"

#: Modules of the ``datasets`` extra the loader imports: none.
REQUIRES: tuple[str, ...] = ()

#: The hc-3 sessions among the 18 linear-track sessions Maboudi et al. (2018)
#: analysed as "previously used by [Diba and Buzsáki, 2007]" (their
#: ``Figure2.ipynb`` output, by animal and start time). The link is inferred.
PAPER_SESSIONS = frozenset(
    {
        "2006-6-7_16-40-19",
        "2006-6-12_15-55-31",
        "2006-6-12_16-53-46",
        "2006-6-13_15-22-3",
        "2006-4-9_16-40-54",
        "2006-4-9_17-29-30",
    }
)
#: The six Maboudi sessions that hc-3 does not hold, by animal and start
#: time as their table gives them (gor01 6-8 and 6-9, pin01 11-1, vvp01 4-16
#: and 4-25); hc-3's original docs say gor01-6-8 was removed from the list.
ABSENT_PAPER_SESSIONS = (
    "gor01 6-8 21-16-25",
    "gor01 6-9 1-22-43",
    "gor01 6-9 22-24-40",
    "pin01 11-1 12-58-54",
    "vvp01 4-16 15-12-23",
    "vvp01 4-25 14-28-51",
)

#: The session with positions from Maboudi et al. 2018's sample file.
POSITION_SESSION = "2006-6-7_16-40-19"
#: The file's name, and the session key it holds.
POSITION_FILE = "fig1.nel"
_POSITION_KEY = "16-40-19"
#: The nelpy classes the file pickles, each read as an inert container.
NELPY_CLASSES = frozenset(
    {
        "hmmlearn.base.ConvergenceMonitor",
        "nelpy.auxiliary._results.ResultsContainer",
        "nelpy.auxiliary._tuningcurve.TuningCurve1D",
        "nelpy.core._analogsignalarray.AnalogSignalArray",
        "nelpy.core._analogsignalarray.EpochSignalSlicer",
        "nelpy.core._epocharray.EpochArray",
        "nelpy.core._spiketrain.BinnedSpikeTrainArray",
        "nelpy.core._spiketrain.EpochUnitSlicer",
        "nelpy.core._spiketrain.ItemGetter_iloc",
        "nelpy.core._spiketrain.ItemGetter_loc",
        "nelpy.core._spiketrain.SpikeTrainArray",
        "nelpy.hmmutils.PoissonHMM",
        "scipy.interpolate.interpolate.interp1d",
    }
)
#: Functions, types, containers and module namespaces the file names (in its
#: HMM and tuning-curve objects), each read as an inert stand-in.
STAND_INS = frozenset(
    {
        "collections.deque",
        "dill.dill._create_function",
        "dill.dill._load_type",
        "hmmlearn.base.__dict__",
        "nelpy.auxiliary._tuningcurve.__dict__",
        "numpy.float64",
        "scipy.interpolate.interpolate.__dict__",
    }
)

#: hc-3's metadata tables, a member of every session: units' types, shanks'
#: regions and sessions' behaviour.
METADATA_FILE = "crcns-hc3-metadata-tables.zip"
_LABELS = {"p": "pyramidal", "i": "interneuron", "n": "unidentified"}
#: hc-3's behaviour names, as the apparatus.
_BEHAVIOR = {
    "linearOne": "linear track ('linearOne')",
    "linearTwo": "linear track ('linearTwo')",
    "Tmaze": "T-maze",
    "Open": "open field",
    "wheel": "operant running wheel",
    "sleep": "sleep in the home cage",
}

#: Position bins, cm.
BIN_SIZE = 2.0
#: The paper's immobility: speed at or below this, cm/s.
MAX_SPEED = 10.0
#: The paper's end platforms, 21 x 21 cm: the reward areas at each end.
PLATFORM = 21.0
#: The stated tracks, cm (79, 125 and 170 cm), for the watch-outs.
TRACK_LENGTHS = (79.0, 125.0, 170.0)
#: Train and test blocks for decoding the rat's position, s.
BLOCK = 60.0
#: A step between position samples longer than this, s, is a gap.
MAX_GAP = 1.0
#: The track's placement tolerance is twice this quantile of the samples'
#: distance from it.
TOLERANCE_QUANTILE = 0.95
#: The two ends of the inferred track are these quantiles of the long
#: segment's positions along it, so a few samples past an end (the head over
#: the platform's edge) do not lengthen it.
END_QUANTILES = (0.005, 0.995)

#: The worked result: the criterion was fixed before the analysis was run
#: (commit "diba_2007: worked-result criterion fixed before the run");
#: ``ours`` and ``parameters`` were filled in after it.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "significant forward-correlated events happen at the start end of the "
        "track before the run (preplay), reverse-correlated ones at the far end "
        "after it (replay)"
    ),
    "quote": (
        '"the majority (841 out of 887, P < 10^-190) of positively correlated '
        "significant events occurred at the start end of the track before "
        "running (preplay), whereas the negatively correlated significant "
        "events (395 out of 464; P < 10^-57) occurred at the other end "
        'following the run (replay; Fig. 1c)"; Fig. 1c: "95% of significant '
        "forward correlated events took place before the run (preplay), "
        "whereas 85% of significant reverse correlated events took place after "
        'the run (replay)."'
    ),
    "source": (
        "Diba & Buzsáki 2007, main text p. 1241 and Fig. 1c; Supplementary "
        "Methods for the event detection and the test"
    ),
    "paper": "95% (841/887) forward before the run; 85% (395/464) reverse after it",
    "ours": None,
    "criterion": (
        "fixed before the run. On every track segment of every session with "
        "positions (2006-6-7_16-40-19: the long-track and the short-track "
        "segment), for each running direction: place cells are pyramidal "
        "units whose rate map over that direction's laps peaks at 5 Hz or more; "
        "candidate events are ripple_detection.literature_methods.diba_2007 "
        "(60 ms of silence of that direction's place cells, then max(5, 30%) "
        "of them firing in the next 300 ms, speed at or below 10 cm/s at both "
        "ends), kept when wholly inside a reward area (within 21 cm of a track "
        "end); each event is judged with replay.methods.DIBA_BUZSAKI_2007 "
        "(first spike, 500 cell-identity shuffles, two-sided p < 0.1) against "
        "its direction's template, and an event found for both directions "
        "(overlapping windows) is assigned as the preset's template_choice "
        "does: among templates on which it is significant, the one with more "
        "active cells, then larger |rho|. A significant event with rho > 0 is "
        "forward, with rho < 0 reverse; it is 'before the run' when it lies at "
        "the start end of its template's direction and 'after the run' at the "
        "far end. Reproduced when, pooled over the segments, more than half "
        "the significant forward events are before the run and more than half "
        "the significant reverse events are after it, each with a one-sided "
        "binomial p < 0.05 against one half; seed 2007. The shares are "
        "reported beside the paper's 95% and 85%, per segment and pooled."
    ),
    "parameters": None,
    "unmatched": (
        "one session of the paper's 69 has positions; the paper pools three rats",
        "units are hc-3's sorting, not the paper's: Maboudi et al.'s copy of "
        "the default session (Diba's own sorting) has 103 units against hc-3's "
        "87 clusters, and only 62% of its spikes are hc-3 sorted spikes",
        "'stable single place fields' are not selected: every pyramidal unit "
        "whose direction's rate map peaks at 5 Hz or more is a template cell",
        "the reward areas are the 21 cm platforms placed at the ends of the "
        "samples along the track, an inference; the paper does not say where "
        "on the platform an event must be",
    ),
    "attempted": None,
    "why": "the criterion is fixed; the analysis has not run yet",
}

#: Findings computed from the data that each verified session must report.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    POSITION_SESSION: (
        r"tracked position samples of \d+ \([\d.]+%\) lie outside the inferred environment",
        r"hc-3's cell table lists 3 units of topdir gor01-6-7 that have no spike",
        r"The positions end at [\d.]+ s and the last spike is at",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    POSITION_SESSION: (
        (
            f"Positions come from {_MABOUDI}'s sample file data/fig1.nel (eLife's "
            "archived copy of github.com/kemerelab/UncoveringTemporalStructureHippocampus, "
            "commit f1b021d), not from hc-3. No data license is stated: the paper's Data "
            "availability says 'Data for Figures 1-5 has been previously reported in "
            "Diba and Buzsáki (2007, Nature Neuroscience). These data ... are available "
            "from Kamran Diba on request. ... All analysis code and sample recording "
            "epochs for Figures 1-7 are available on' that repository, whose MIT "
            "license covers its code. The file is downloaded from the archive and never "
            "redistributed. It is a pickle, read without running any of it."
        ),
        "The positions are processed, not raw tracking: no sample is missing "
        "and steps are smooth to a hundredth of a cm, so lost LED frames were "
        "filled in before release. How is not documented.",
    ),
}


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one hc-3 session of Diba's rats.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The session's folder in the cache: its ``.res.N``, ``.clu.N`` and
        ``.xml`` members, hc-3's metadata tables and, for
        ``2006-6-7_16-40-19``, Maboudi et al.'s ``fig1.nel``.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the ``.xml`` states another wideband rate than 32,552 Hz, the
        metadata tables do not list the session, or the position file holds
        another session or layout.
    pickle.UnpicklingError
        If the position file names a global outside the allow-list.
    """
    import pandas as pd

    folder = Path(path_or_stream)
    session = entry.session
    tables = _metadata(folder / METADATA_FILE)
    if session not in tables.sessions:
        msg = f"{session}: hc-3's session table does not list it"
        raise ValueError(msg)
    topdir, behavior, duration = tables.sessions[session]
    acquisition = read_acquisition(folder / f"{session}.xml")
    if acquisition.wideband_hz != WIDEBAND_HZ:
        msg = (
            f"{session}.xml states {acquisition.wideband_hz:g} Hz, not the "
            f"{WIDEBAND_HZ:g} Hz of {_SUPPLEMENT}"
        )
        raise ValueError(msg)
    regions = tables.regions[topdir]
    trains, ids, rows, n_clusters = [], [], [], 0
    shanks = sorted(
        int(path.name.rsplit(".", 1)[1]) for path in folder.glob(f"{session}.res.*")
    )
    for shank in shanks:
        samples, clusters = read_res_clu(
            folder / f"{session}.res.{shank}", folder / f"{session}.clu.{shank}"
        )
        for cluster in np.unique(clusters[clusters >= 2]).tolist():
            n_clusters += 1
            trains.append(
                np.sort(samples[clusters == cluster]) / acquisition.wideband_hz
            )
            ids.append(f"{shank}.{cluster}")
            label = tables.cell_types.get((topdir, shank, cluster))
            rows.append(
                {
                    "brain_region": regions[shank - 1],
                    "label": _LABELS.get(label, np.nan) if label else np.nan,
                }
            )
    units = pd.DataFrame(rows, index=pd.Index(ids, name="unit_id"))
    listed = {key for key in tables.cell_types if key[0] == topdir}
    absent = sorted(listed - {(topdir, *map(int, uid.split("."))) for uid in ids})
    unlisted = int(units["label"].isna().sum())
    lfp = _lfp_source(entry, folder, acquisition, regions)
    animal = topdir.split("-", 1)[0]
    common: dict[str, Any] = {
        **entry.identity(folder),
        "animal": f"rat {animal}",
        "apparatus": _BEHAVIOR.get(behavior, behavior),
        "brain_regions": tuple(dict.fromkeys(regions)),
        "spike_times": tuple(trains),
        "units": units,
        "lfp": lfp,
    }
    unit_conversions = (
        Conversion(
            f"spike_times: .res samples / {acquisition.wideband_hz:g} Hz, seconds "
            "from the start of the session's file",
            "stated",
            f"{session}.xml: samplingRate {acquisition.wideband_hz:g}; hc-3's data "
            "description: '.res.N' holds the time of each detected spike",
        ),
        Conversion(
            f"units: the {n_clusters} clusters from 2 up on {_count(len(shanks), 'shank')}, "
            "id '<shank>.<cluster>'; brain_region from hc-3's electrode table "
            f"(topdir {topdir}); label from hc-3's cell table cellType "
            "('p' pyramidal, 'i' interneuron, 'n' unidentified)",
            "stated",
            "hc-3's data description: clusters 0 (noise) and 1 (unsortable) "
            "'should not be used'; the cell table's cellType comment; every "
            "cell-table region of these rats equals its shank's electrode-table "
            "region",
        ),
    )
    paper_link = _paper_link(session)
    watch_outs = [
        paper_link,
        _SORTING,
        *(
            [
                f"hc-3's cell table lists {_count(len(absent), 'unit')} of topdir "
                f"{topdir} that have no spike in this session ({_shown(absent)}): "
                "the day's sorting covers every session of the day."
            ]
            if absent
            else []
        ),
        *(
            [
                f"{_count(unlisted, 'cluster')} from 2 up {'is' if unlisted == 1 else 'are'} "
                "not in hc-3's cell table, which lists the units that passed the "
                "authors' check: their label is missing."
            ]
            if unlisted
            else []
        ),
    ]
    if session != POSITION_SESSION:
        if (folder / POSITION_FILE).exists():
            msg = f"{session}: {POSITION_FILE} holds {POSITION_SESSION} only"
            raise ValueError(msg)
        n_samples = lfp.files[0].size_bytes // (2 * acquisition.n_channels)
        times = np.arange(n_samples) / acquisition.lfp_hz
        return Recording(
            **common,
            clock=(
                "the session file's clock, seconds from its first sample; "
                "times are the LFP's sample clock, since hc-3 holds no tracking "
                "for this session"
            ),
            times=times,
            epochs={"session": np.array([[0.0, float(times[-1])]])},
            params={"analysis_epochs": np.array([[0.0, float(times[-1])]])},
            conversions=(
                *unit_conversions,
                Conversion(
                    f"times: the .eeg's sample clock, k / {acquisition.lfp_hz:g} s for "
                    f"its {n_samples} samples; no positions",
                    "stated",
                    f"{session}.xml: lfpSamplingRate {acquisition.lfp_hz:g}, "
                    f"nChannels {acquisition.n_channels}; the .eeg member is "
                    f"{lfp.files[0].size_bytes} bytes of int16. hc-3's session "
                    f"table gives {duration:g} s",
                ),
            ),
            watch_outs=(
                _NO_TRACKING.format(behavior=_BEHAVIOR.get(behavior, behavior)),
                *watch_outs,
            ),
        )
    return _with_positions(folder, common, unit_conversions, watch_outs)


#: The wideband rate the Supplementary Methods state, Hz.
WIDEBAND_HZ = 32552.0

_SORTING = (
    "Units are hc-3's sorting, not the paper's. nelpy's hc-3 reader (Ackermann) "
    "notes of Diba's own copy of these data: 'Kamran just has a different (much "
    "better) unit sorting than what's available on CRCNS.' For "
    f"{POSITION_SESSION}, Maboudi et al.'s file (Diba's sorting) holds 103 "
    "units with interneurons left out ('spikes_noIN') against hc-3's 87 "
    "clusters; 95.3% of its 258,427 spikes are hc-3 detected spikes (clusters "
    "0 and 1 included), 62.3% hc-3 sorted ones."
)

_NO_TRACKING = (
    "hc-3 holds no position for this session ({behavior}): its archive has no "
    ".whl file, nor do any of Diba's 27 sessions. There are no positions, "
    "environment or speed; times are the LFP's sample clock. The spikes, units "
    "and LFP are complete."
)


def _paper_link(session: str) -> str:
    """Whether the session is one of the paper's, and the evidence."""
    if session in PAPER_SESSIONS:
        return (
            f"Inferred to be one of {_PAPER}'s sessions: {_MABOUDI} list it among "
            "the 18 linear-track sessions of Diba's rats they analysed, data "
            f"'previously used by [Diba and Buzsáki, 2007]' (their Figure2.ipynb "
            "output). No source lists the 2007 paper's own sessions (69 in all: "
            "9, 40 and 20 per rat)."
        )
    return (
        f"Not tied to {_PAPER} by any source read: Diba's rat, but not among the "
        f"18 linear-track sessions {_MABOUDI} analysed as the paper's data. The "
        "paper used 69 sessions of three rats, which no source lists."
    )


class _Tables:
    """hc-3's metadata tables, for Diba's three rats."""

    def __init__(self, path: Path) -> None:
        with zipfile.ZipFile(path) as archive:

            def rows(name: str) -> list[list[str]]:
                text = archive.read(f"hc3-metadata-tables/{name}").decode()
                return [
                    row
                    for row in csv.reader(io.StringIO(text))
                    if row and row[1].split("-", 1)[0] in _ANIMALS
                ]

            #: session -> (topdir, behaviour, duration in s)
            self.sessions = {
                row[2]: (row[1], row[3], float(row[5]))
                for row in rows("hc3-session.csv")
            }
            #: topdir -> region of each shank
            self.regions = {
                row[0]: tuple(region for region in row[2:] if region)
                for row in csv.reader(
                    io.StringIO(
                        archive.read("hc3-metadata-tables/hc3-epos.csv").decode()
                    )
                )
                if row and row[0].split("-", 1)[0] in _ANIMALS
            }
            #: (topdir, shank, cluster) -> cellType
            self.cell_types = {
                (row[1], int(row[3]), int(row[4])): row[-1]
                for row in rows("hc3-cell.csv")
            }


_ANIMALS = ("gor01", "pin01", "vvp01")


@functools.lru_cache(maxsize=4)
def _metadata(path: Path) -> _Tables:
    """hc-3's metadata tables at ``path``, read once."""
    return _Tables(path)


def _shown(keys: list[tuple[str, int, int]]) -> str:
    """``"1.6, 1.7"``: unit ids of (topdir, shank, cluster) keys, at most eight."""
    ids = [f"{shank}.{cluster}" for _, shank, cluster in keys]
    return ", ".join(ids[:8]) + (", ..." if len(ids) > 8 else "")


def read_lfp(
    sources: tuple[str | Path, ...],
    channels: tuple[str, ...],
    *,
    n_channels: int,
    microvolts_per_bit: float,
    rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read channels ``"ch<k>"`` of a session's ``.eeg`` file, in µV, on its clock.

    Parameters
    ----------
    sources : tuple of path
        The one ``.eeg`` file.
    channels : tuple of str
        ``"ch<k>"`` names, ``k`` from 0.
    n_channels : int
        The channels interleaved in the file.
    microvolts_per_bit : float
        The voltage of one stored step.
    rate : float
        Samples per second.

    Returns
    -------
    samples : ndarray, shape (n_time, n_channels_read)
    times : ndarray, shape (n_time,)
        ``k / rate``, seconds.
    """
    (path,) = sources
    picks = tuple(int(name.removeprefix("ch")) for name in channels)
    stored = read_eeg(path, n_channels, picks)
    return stored * microvolts_per_bit, np.arange(len(stored)) / rate


def _lfp_source(
    entry: SessionEntry,
    folder: Path,
    acquisition: Any,
    regions: tuple[str, ...],
) -> LFPSource:
    """The session's ``.eeg``, fetched from its archive on request."""
    shank_of = {
        channel: shank
        for shank, group in enumerate(acquisition.channel_groups, start=1)
        for channel in group
    }
    channels = {
        f"ch{channel}": (
            f"shank {shank_of[channel]}, {regions[shank_of[channel] - 1]}"
            if channel in shank_of and shank_of[channel] <= len(regions)
            else "not in an electrode group of the .xml"
        )
        for channel in range(acquisition.n_channels)
    }
    return LFPSource(
        dataset=entry.info.dataset,
        files=entry.lfp_files,
        sampling_rate=acquisition.lfp_hz,
        channels=channels,
        unit="uV",
        unit_basis=Conversion(
            f"LFP: stored int16 x {acquisition.microvolts_per_bit:.4f} µV, the "
            f".xml's voltageRange {acquisition.voltage_range:g} V / 2^"
            f"{acquisition.n_bits} / amplification {acquisition.amplification:g}",
            "stated",
            f"{entry.session}.xml's acquisitionSystem; hc-3's data description: "
            "'.eeg - raw data, down sampled to 1.25 KHZ'",
        ),
        times_basis=Conversion(
            f"LFP times: sample k at k / {acquisition.lfp_hz:g} s, the clock of the "
            "spikes",
            "stated",
            f"{entry.session}.xml: lfpSamplingRate {acquisition.lfp_hz:g}, a "
            f"26th of the {acquisition.wideband_hz:g} Hz wideband rate the .res "
            "samples count; NeuroSuite writes both from the same file",
        ),
        reader=functools.partial(
            read_lfp,
            n_channels=acquisition.n_channels,
            microvolts_per_bit=acquisition.microvolts_per_bit,
            rate=acquisition.lfp_hz,
        ),
        cache=folder.parents[1],
    )


def _position_file(path: Path) -> dict[str, Any]:
    """The 2-D position and its segments from Maboudi et al.'s ``fig1.nel``."""
    root = inert_fields(read_dill_pickle(path, NELPY_CLASSES, STAND_INS))
    sessions = root.get("exp_data", {})
    if list(sessions) != [_POSITION_KEY]:
        msg = f"{path.name}: expected one session {_POSITION_KEY!r}, found {list(sessions)}"
        raise ValueError(msg)
    session = sessions[_POSITION_KEY]
    position = inert_fields(session["pos2d"])
    times = np.asarray(position["_time"], dtype=np.float64)
    positions = np.asarray(position["_ydata"], dtype=np.float64)
    if positions.shape != (2, times.size):
        msg = f"{path.name}: pos2d has shape {positions.shape}, not (2, {times.size})"
        raise ValueError(msg)
    support = np.asarray(
        inert_fields(position["_support"])["_time"], dtype=np.float64
    ).reshape(-1, 2)
    return {"times": times, "positions": positions.T, "support": support}


def _with_positions(
    folder: Path,
    common: dict[str, Any],
    unit_conversions: tuple[Conversion, ...],
    watch_outs: list[str],
) -> Recording:
    """The session Maboudi et al. released with positions."""
    stored = _position_file(folder / POSITION_FILE)
    times, positions = stored["times"], stored["positions"]
    support = stored["support"]
    if len(support) != 2:
        msg = f"{POSITION_FILE}: pos2d has {len(support)} support intervals, not 2"
        raise ValueError(msg)
    long_rows, short_rows = support[:1], support[1:]
    track = _track(times, positions, long_rows)
    epochs: dict[str, NDArray[np.float64]] = {
        "long_track": long_rows,
        "short_track": short_rows,
    }
    lap_text = []
    for name, rows in (("long_track", long_rows), ("short_track", short_rows)):
        inside = (times >= rows[0, 0]) & (times <= rows[0, 1])
        along = track.along(positions[inside])
        low, high = np.quantile(along[np.isfinite(along)], END_QUANTILES)
        laps = _laps(times[inside], along, low + PLATFORM, high - PLATFORM)
        for direction, intervals in laps.items():
            epochs[f"{name}_laps_{direction}"] = intervals
        lap_text.append(
            f"{name}: the samples span {high - low:.1f} cm along the track, "
            f"{len(laps['increasing'])} laps of increasing and "
            f"{len(laps['decreasing'])} of decreasing position"
        )
    train, test = alternating_blocks(float(times[0]), float(times[-1]), BLOCK)
    gap = float(short_rows[0, 0] - long_rows[0, 1])
    return Recording(
        **common,
        clock=(
            "the session file's clock, seconds from its first sample; the "
            "positions are on it (see conversions)"
        ),
        times=times,
        positions=positions,
        position_units="cm",
        tracking=(
            f"the head, from 'an LED' ({_SUPPLEMENT}); {_MABOUDI}'s processed "
            "position, one point per sample at 60 Hz"
        ),
        env=track.env,
        env_kind="inferred",
        env_source=track.source,
        track_width_quantile=track.width_quantile,
        epochs=epochs,
        params={
            "bin_size": BIN_SIZE,
            "max_speed": MAX_SPEED,
            "platform": PLATFORM,
            "max_gap": MAX_GAP,
            "analysis_epochs": support,
            "train_epochs": train,
            "test_epochs": test,
        },
        conversions=(
            *unit_conversions,
            Conversion(
                f"positions: exp_data['16-40-19']['pos2d'] of {_MABOUDI}'s "
                f"{POSITION_FILE}, x and y as stored, in cm",
                "inferred",
                "the file states no unit; the authors' figures label it 'position "
                "[cm]'. It is 3.299 times the file's 'pos' (residual SD 0.15) and "
                "the short-track segment spans "
                f"{track.spans['short_track']:.0f} cm, against the 125 cm track and "
                f"two 21 cm platforms ({_SUPPLEMENT}: 167 cm); the long segment "
                f"spans {track.spans['long_track']:.0f} cm, which matches no stated "
                "track",
            ),
            Conversion(
                "times: pos2d's sample times, seconds, on the clock of the spikes",
                "inferred",
                "the file's spike times (Diba's sorting of the same recording) "
                "coincide with hc-3's: 95.3% of its 258,427 spikes fall within 1 µs "
                "of an hc-3 detected spike of this session, against 18% with a "
                "0.1 s shift either way",
            ),
            Conversion(
                f"epochs: 'long_track' and 'short_track', pos2d's two support "
                f"intervals, {long_rows[0, 0]:.2f}-{long_rows[0, 1]:.2f} s and "
                f"{short_rows[0, 0]:.2f}-{short_rows[0, 1]:.2f} s",
                "stated",
                "the file's support; nelpy's hc-3 table (get_hc3_dataframe) gives "
                "this session's segments as 'long' (0, 1180) and 'short' "
                f"(1250, 2580) s, and pos2d has no sample in the {gap:.0f} s between",
            ),
            Conversion(
                "epochs '<segment>_laps_<direction>': from leaving one end "
                f"platform ({PLATFORM:g} cm from the samples' 0.5% or 99.5% "
                "quantile along the track) to entering the other, by the sign of "
                "the change in position along the track",
                "inferred",
                f"{_SUPPLEMENT}: 'A trajectory was marked when the animal left one "
                "reward area platform, traversed the track, and entered the other "
                f"reward area platform', platforms 21 x 21 cm. {'; '.join(lap_text)}",
            ),
            *track.conversions,
        ),
        watch_outs=(
            *watch_outs,
            *WATCH_OUTS[POSITION_SESSION],
            f"The track was shortened in the {gap:.0f} s between the segments, "
            "which have no position: "
            f"'long_track' spans {track.spans['long_track']:.0f} cm along the track "
            f"and 'short_track' {track.spans['short_track']:.0f} cm (platforms "
            f"included). {_SUPPLEMENT} state tracks of "
            f"{', '.join(f'{x:g}' for x in TRACK_LENGTHS)} cm with 21 cm platforms; "
            "the short segment matches 125 cm, the long one none (170 cm would "
            "span 212 cm). rec.env is the long track; place fields of the two "
            "segments differ and are analysed apart.",
            f"The positions end at {times[-1]:.2f} s and the last spike is at "
            f"{max(float(t[-1]) for t in common['spike_times'] if t.size):.2f} s: "
            "spikes after the last position sample have none.",
        ),
    )


class _InferredTrack:
    """A straight track along the long segment's samples, and how it was placed."""

    def __init__(
        self,
        env: Environment,
        start: NDArray[np.float64],
        axis: NDArray[np.float64],
        source: str,
        conversions: tuple[Conversion, ...],
        width_quantile: float | None,
        spans: dict[str, float],
    ) -> None:
        self.env = env
        self.start = start
        self.axis = axis
        self.source = source
        self.conversions = conversions
        self.width_quantile = width_quantile
        self.spans = spans

    def along(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each point's position along the track from its start, cm."""
        result: NDArray[np.float64] = (points - self.start) @ self.axis
        return result


def _track(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    long_rows: NDArray[np.float64],
) -> _InferredTrack:
    """The long track: a straight line through the long segment's samples."""
    import networkx as nx

    from neurospatial import Environment

    tracked = tracked_mask(positions)
    long = tracked & (times >= long_rows[0, 0]) & (times <= long_rows[0, 1])
    points = positions[long]
    centre = np.median(points, axis=0)
    # The line through the samples near it: the rat starts off the track.
    near = np.abs(points[:, 1] - centre[1]) < 4 * np.median(
        np.abs(points[:, 1] - centre[1])
    )
    _, _, vectors = np.linalg.svd(
        points[near] - points[near].mean(axis=0), full_matrices=False
    )
    axis = vectors[0] * np.sign(vectors[0][0])
    origin = points[near].mean(axis=0)
    along = (points - origin) @ axis
    low, high = np.quantile(along, END_QUANTILES)
    start, stop = origin + low * axis, origin + high * axis
    graph = nx.Graph()
    graph.add_node("low_end", pos=(float(start[0]), float(start[1])))
    graph.add_node("high_end", pos=(float(stop[0]), float(stop[1])))
    graph.add_edge("low_end", "high_end", distance=float(high - low))
    distance = nearest_segment(
        positions[tracked], start[None, :], stop[None, :]
    ).distance
    width, width_quantile, evidence = placement_tolerance(
        distance,
        TOLERANCE_QUANTILE,
        floor=BIN_SIZE,
        measured="no source read states where the track lay; the samples' distance from the fitted line",
        reason=(
            "The 95th, not the 99th, percentile: the rat starts the session off "
            "the track."
        ),
    )
    env = Environment.from_graph(
        graph,
        [("low_end", "high_end")],
        bin_size=BIN_SIZE,
        units="cm",
        name=f"diba_2007 {POSITION_SESSION}",
        track_width=width,
    )
    spans = {"long_track": float(high - low)}
    short = tracked & (times > long_rows[0, 1])
    short_along = (positions[short] - origin) @ axis
    spans["short_track"] = float(np.ptp(np.quantile(short_along, END_QUANTILES)))
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return _InferredTrack(
        env=env,
        start=start,
        axis=axis,
        source=(
            f"a straight {high - low:.0f} cm track through the long segment's "
            "samples, inferred: no source read states the long track's length or "
            f"placement, {BIN_SIZE:g} cm bins"
        ),
        conversions=(
            Conversion(
                f"environment: one straight edge from ({start[0]:.1f}, {start[1]:.1f}) "
                f"to ({stop[0]:.1f}, {stop[1]:.1f}) cm, {angle:.2f} degrees from +x",
                "inferred",
                "the principal axis of the long segment's samples within four median "
                "absolute deviations of its median y; its ends are the 0.5% and "
                "99.5% quantiles of the samples along it, platforms included",
            ),
            Conversion(
                f"environment: track_width {width:.3g} cm, a placement tolerance, "
                "not the track's 6.2 cm width",
                "inferred",
                evidence,
            ),
        ),
        width_quantile=width_quantile,
        spans=spans,
    )


def _laps(
    times: NDArray[np.float64],
    along: NDArray[np.float64],
    low_edge: float,
    high_edge: float,
) -> dict[str, NDArray[np.float64]]:
    """Runs from one platform to the other, [leave, enter) rows by direction.

    A lap starts at the last sample on one platform (``along`` at or below
    ``low_edge``, or at or above ``high_edge``) before the rat next reaches the
    other platform, and ends at the first sample on the other.
    """
    side = np.where(along <= low_edge, -1, np.where(along >= high_edge, 1, 0))
    on = np.flatnonzero(side != 0)
    laps: dict[str, list[list[float]]] = {"increasing": [], "decreasing": []}
    for a, b in itertools.pairwise(on):
        if side[a] != side[b]:
            direction = "increasing" if side[b] > side[a] else "decreasing"
            laps[direction].append([float(times[a]), float(times[b])])
    return {
        key: np.array(rows, dtype=np.float64).reshape(-1, 2)
        for key, rows in laps.items()
    }
