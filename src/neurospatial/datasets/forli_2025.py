"""Forli et al. (2025): bat hippocampus during free flight in a room (Zenodo 15738988).

What was recorded: an Egyptian fruit bat flying freely in a room for about
90 minutes (38 flights) while three Neuropixels probes recorded units in
dorsal hippocampus. Position is 3-D, in metres, sampled at 120 Hz: the
centroid of three reflective markers on the neural recording headstage,
tracked by 16 motion-capture cameras (the paper's Methods, for the indoor
room). Flights are the authors' segmentation at a 0.5 m/s velocity
threshold.

Where it lives: Zenodo record 15738988 (CC-BY-4.0), the demo session
``Dataset_3_14543_240419`` in ``Demo_Session.zip`` (3.3 GB). Only the five
members the loader needs are read, by HTTP byte range (260 MB): the behavior
file, the three Kilosort 4 single-unit tables and the merge list, each
verified against its own SHA-256. The single-unit tables are MATLAB
``table`` objects, read with ``mat-io``, which installs only on Python 3.11
or later.

Sources read: the paper (Forli, Fan, Qi & Yartsev 2025, Nature,
doi:10.1038/s41586-025-09341-z, accelerated article preview, Methods), the
record's ``NP_Bat_Demo.m`` (whose header describes every variable) and
``How to use.pdf``, and the files.

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

The declared environment is the room: every voxel of a grid that tiles
``r_lim`` ("boundaries of the room/outdoor arena", the script) exactly, so
it holds the air the bat could fly through, not only the voxels it visited.
Each axis has a whole number of cells close to 0.2 m (29 x 26 x 11 cells of
0.2 x 0.2 x 0.209 m): with 0.2 m cells on every axis the 2.3 m height is
11.5 cells and the grid would reach 5 cm past the floor and the ceiling.
``r_lim`` is used because it is stated in the positions' frame; the paper
gives only the room's size. Some resting samples lie beyond the -y bound.

Analysis choices (``params``):

- ``bin_size`` 0.2 m: 3-D bins about the size of a bat's body;
- ``min_speed``: the file's ``v_th``, the authors' flight threshold;
- ``analysis_epochs``: flights during the neural recording, so rate maps
  describe the bat in flight; ``train_epochs`` / ``test_epochs``: even and
  odd of those flights, for held-out decoding;
- ``dt`` 0.25 s, as for the 2-D datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._recording import Conversion, Recording

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = ["REQUIRES", "WATCH_OUTS", "load_session"]

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("matio", "fsspec")

BIN_SIZE = 0.2
DT = 0.25
PROBES = (1, 2, 3)
#: Spikes of one unit closer than this are one spike counted twice, s.
CLOSE = 0.001
#: Mean rate above which NP_Bat_Demo.m sets a unit aside as a putative
#: interneuron, Hz.
INTERNEURON_RATE = 2.0
#: The paper's indoor flight room, m (Methods).
PAPER_ROOM = (5.6, 5.2, 2.5)
_PAPER = "the Methods of Forli et al. 2025"
_SCRIPT = "the record's NP_Bat_Demo.m"

_PROCESSED = (
    "Positions are the authors' processed behavior (smoothed, with flights "
    f"segmented; {_SCRIPT}), so held-sample counts describe that processing as "
    "much as the tracking: at rest the stored position stays exactly the same "
    "for minutes."
)

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "Dataset_3_14543_240419": (
        "This is the record's one demo session; the paper reports 23 sessions "
        f"from six bats ({_PAPER}), and the record holds no other.",
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
    env, placement = _declared_room(positions, room, entry.session)
    flying = np.asarray(behavior["bflying"], dtype=bool)
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
        params={
            "bin_size": BIN_SIZE,
            "dt": DT,
            "min_speed": v_th,
            "analysis_epochs": neural_flights,
            "train_epochs": neural_flights[::2],
            "test_epochs": neural_flights[1::2],
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
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            *_outside_room(positions, room, flying),
            _before_recording(times, flight_windows, recorded),
            _interneurons(units),
            _PROCESSED,
        ),
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


def _good_units(path: Path) -> Any:
    """The ``out.good_units`` table of a Kilosort 4 single-unit file."""
    import matio

    value: Any = matio.load_from_mat(str(path))["out"]["good_units"]
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


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
    n_cells = np.floor(sides / BIN_SIZE + 0.5)
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
        "r_lim; the paper's room centred on r_lim (from its floor in z) would "
        f"leave {_share_outside(positions, paper):.2%} outside",
    )
    return env, placement


def _share_outside(positions: NDArray[np.float64], box: NDArray[np.float64]) -> float:
    """Share of tracked samples outside the box ``box``, shape (n_dims, 2)."""
    tracked = positions[np.all(np.isfinite(positions), axis=1)]
    inside = np.all((tracked >= box[:, 0]) & (tracked <= box[:, 1]), axis=1)
    return float(1 - inside.mean())


def _outside_room(
    positions: NDArray[np.float64], room: NDArray[np.float64], flying: NDArray[np.bool_]
) -> tuple[str, ...]:
    """Where the samples outside ``r_lim`` lie: by axis and side, and in flight."""
    tracked = np.all(np.isfinite(positions), axis=1)
    below = (positions < room[:, 0]) & tracked[:, None]
    above = (positions > room[:, 1]) & tracked[:, None]
    outside = below.any(axis=1) | above.any(axis=1)
    if not outside.any():
        return ()
    sides = ", ".join(
        f"{int(mask[:, axis].sum())} past {sign}{'xyz'[axis]}"
        for axis in range(room.shape[0])
        for sign, mask in (("-", below), ("+", above))
        if mask[:, axis].any()
    )
    beyond = np.maximum(room[:, 0] - positions, positions - room[:, 1])
    farthest = float(np.nanmax(np.where(below | above, beyond, np.nan)))
    return (
        f"The tracked samples outside the room bounds r_lim lie {sides}, up to "
        f"{farthest:.2f} m past a bound, and {np.mean(flying[outside]):.1%} of "
        "them during flight (the file's bflying). Either r_lim leaves out a "
        "place the bat lands, or the tracking there is off; the files do not say which. The samples are kept as "
        "stored and fall in no bin of the environment.",
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
