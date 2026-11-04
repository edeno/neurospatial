"""Shipley et al. (2026): mouse CA1 on an 8-arm radial maze, with rest (UCL RDR 30902972).

What was recorded: tetrode units (16 tetrodes, 64 channels) in the CA1
pyramidal layer of App NL-G-F/ChAT-cre mice and wild-type littermates
while they searched an 8-arm radial maze for four baited arms. Each test
day began with an 8-minute exploration trial, then two sessions of five
trials (4 minutes at most) separated by a 45-minute rest recording in a
cylindrical box on the centre platform; a 2-minute holding-box interval
separated the trials. The arms are 80 x 10 cm, joined by an octagonal centre
platform 24.7 cm across. Position is the headstage's LEDs, filmed by an
overhead camera at "approximately 30Hz with some variability" (the paper's
Methods).

Where it lives: UCL Research Data Repository 30902972 (CC-BY-4.0),
``Shipley2025_Data.zip`` (4.4 GB), one folder per mouse and day. The loader
reads four members of the day's ``PhyOutput/OldData`` folder by HTTP byte
range: ``spike_times.npy`` and ``spike_clusters.npy`` (Kilosort, read with
``np.load(..., allow_pickle=False)``), ``cluster_info.tsv`` (the Phy labels)
and ``trialData_newMUA.mat`` (MATLAB v5: the day's positions, and per
tetrode the curated spikes' times and position samples). The archive holds
50 days of 15 mice; the 36 days (14 mice) the loader can read are offered,
and the default is mouse m4005 on 2020-09-26.

Not offered: 14 days whose position samples cannot be put on one increasing
clock, so loading them would raise ``ValueError``. In 12 a trial's fitted
line starts before the previous trial's ends. In the 9 where the overlap is
under one frame interval (33 ms), the file's spike-to-sample map gives the
spikes at the boundary to both trials; the 3 overlaps of 0.8 to 10.6 s,
after a rest, are more than that can explain, and no source read says why.
In 2 a trial has no spikes mapped to it, so its samples cannot be timed:

- ``m4005_2020-09-28``: trial 8 starts 0.53 ms before trial 7 ends;
- ``m4101_2020-11-03``: trial 2 starts 0.34 ms before trial 1 ends;
- ``m4230_2020-12-18``: trial 9 starts 0.42 ms before trial 8 ends;
- ``m4230_2020-12-19``: trial 5 starts 0.36 ms before trial 4 ends;
- ``m4232_2020-12-17``: trial 2 starts 0.38 ms before trial 1 ends;
- ``m4232_2020-12-18``: trial 9 starts 0.11 ms before trial 8 ends;
- ``m4232_2020-12-19``: trial 5 starts 0.04 ms before trial 4 ends;
- ``m4376_2021-08-16``: trial 9 starts 0.75 ms before trial 8 ends;
- ``m4604_2022-03-02``: trial 1 starts 28.95 ms before trial 0 ends;
- ``m4201_2019-12-17``: trial 6 starts 796.89 ms before rest S1 ends;
- ``m4202_2019-12-16``: trial 6 starts 1381.64 ms before rest S1 ends;
- ``m4020_2020-08-18``: trial 6 starts 10583.43 ms before rest S1 ends;
- ``m4020_2020-08-21``: trials 0 to 3 have no spikes mapped to them;
- ``m4578_2021-10-18``: trial 0 has no spikes mapped to it.

Sources read: the paper (Shipley, Abrate, Hayman, Chan & Barry 2026,
Current Biology 36, 859-871.e5, on Europe PMC, PMC7619514), the record's
description and its ``MouseStats.xlsx``, and the files.

What the loader converts:

- spike times: ``spike_times.npy`` sample indices / 30 kHz (the Methods'
  sampling rate), for the clusters ``cluster_info.tsv`` labels ``good``.
  They must equal the curated spike times ``trialData.tetrode(k).timestamp``
  holds, or the loader refuses the day.
- positions: ``trialData.pos.xy`` (pixels) / the header's
  ``pixels_per_metre`` x 100, in cm.
- position times. ``trialData.pos`` is the day's trials concatenated, on the
  clock of the day's recordings combined for spike sorting (Methods:
  "entire daily recordings were combined"), and holds no sample times. The
  file maps every curated spike to its position sample
  (``tetrode(k).pos_sample``, 1-based). Within each trial
  (``trialFinalInd``) the loader fits a line from sample index to spike time
  and times the samples with it: an inferred ``Conversion``. On the default
  day the fitted rates are 30.047 to 30.064 Hz, not the header's 30 Hz, and
  the largest residual is 41 ms (about one frame); the residuals are
  reported for every day. The loader refuses a day where the fitted times
  do not increase (a trial's line starts before the previous trial's ends)
  or a trial has too few spikes to fit; the days that do are not offered
  (above).
- headings: none. The file has one LED (``n_leds`` 1) though the Methods
  fit two, and ``pos.dir`` equals ``dir_disp``, the direction of
  displacement, not of the head.

The declared environment is the radial maze as a star graph from the stated
sizes: a centre node, a 12.35 cm segment to the platform's edge and an 80
cm arm for each of the eight arms, 2 cm bins along the track. Where the
maze lies comes from the maze trials' samples (an inferred ``Conversion``):
eight directions 45 degrees apart are turned to fit the samples, each arm
visited enough then takes the principal axis of its samples beyond the
platform, and the centre is the point nearest those axes in least squares. The track's width is a placement tolerance measured from the
samples, at least the stated 10 cm arm width.

Analysis choices (``params``):

- ``bin_size`` 2 cm and ``min_speed`` 2 cm/s, the paper's rate-map bins and
  running-speed filter;
- ``analysis_epochs``: the maze trials (exploration and test trials);
  ``train_epochs`` / ``test_epochs``: their even and odd rows, for held-out
  decoding at ``dt`` 0.25 s.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._recording import Conversion, Recording, tracked_mask
from neurospatial.datasets._tracks import nearest_segment, placement_tolerance

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = [
    "EXPECTED_WATCH_OUTS",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": None,
    "quote": None,
    "source": None,
    "paper": None,
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (),
    "attempted": None,
    "why": (
        "The paper's numbers pool many mice and days (for example the peak "
        "firing rates of 232 wild-type and 222 NL-G-F place cells, or "
        "spatial information from the second session of each day with "
        "adaptive binning), and its replay results need a replay analysis "
        "of the rest recordings; none is a single-day number this session "
        "can reproduce. A pooled result over the 36 offered days is left for later."
    ),
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES = ("fsspec",)

#: The Methods' acquisition rate of the spike times, Hz.
SAMPLING_HZ = 30_000.0
#: The stated maze: arm length and width, and the platform's diameter, cm.
ARM_LENGTH = 80.0
ARM_WIDTH = 10.0
PLATFORM = 24.7
N_ARMS = 8
BIN_SIZE = 2.0
MIN_SPEED = 2.0
DT = 0.25
#: A residual beyond this (s) from its trial's line, about one and a half
#: frames, is counted in the watch-out on the position clock.
RESIDUAL = 0.05
#: Fewest spikes a trial's line is fitted on.
MIN_FIT_SPIKES = 100
#: Fewest samples beyond the platform an arm's direction is fitted on.
MIN_ARM_SAMPLES = 100
#: Samples within this distance (cm) of the platform's edge are left out of
#: an arm's direction fit, where the platform's samples would bend it.
PLATFORM_MARGIN = 5.0
#: Half-width of the angular sector (as a share of the angle between arms)
#: whose samples fit an arm.
SECTOR = 0.5
#: The share of samples the placement tolerance leaves outside the track.
TOLERANCE_QUANTILE = 0.99
_PAPER = "Shipley et al. 2026, Methods, PMC7619514"
_NAMES = ("spike_times.npy", "spike_clusters.npy", "cluster_info.tsv")
_TRIAL_DATA = "trialData_newMUA.mat"

#: Each mouse's genotype, from the record's MouseStats.xlsx (sheet
#: Exp1Sessions, columns AD and WT).
GENOTYPES = {
    **dict.fromkeys(
        ("m4005", "m4020", "m4202", "m4232", "m4602", "m4609", "m4610"),
        "App NL-G-F/ChAT-cre",
    ),
    **dict.fromkeys(
        ("m4098", "m4101", "m4201", "m4230", "m4376", "m4578", "m4604", "m4605"),
        "wild type",
    ),
}

#: Findings computed from the data (by the checks or this loader) that each
#: verified session must report, as patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "m4005_2020-09-26": (
        r"Position times are rebuilt per trial from the spike-to-sample map: "
        r"fitted rates 30\.047 to 30\.064 Hz against the header's 30 Hz; the "
        r"largest residual is 41 ms",
        r"pos\.dir equals dir_disp in all 146101 samples",
        r"runs of 11 or more identical position samples",
    ),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "m4005_2020-09-26": (
        "The day has one rest recording, S2 after the last trial; the first "
        "rest (S1, between trials 5 and 6) is not in the archive, though the "
        "folders' start times leave 52 minutes between trial 5 (13:58:54) and "
        "trial 6 (14:51:20).",
    ),
}

_PROCESSED = (
    "Positions are the authors' processed tracking: 'Post-hoc removal of "
    "jumps exceeding one standard deviation from mean position change was "
    f"performed in Matlab, with interpolation replacing missing data points' "
    f"({_PAPER}), so held and fast samples describe that processing as much as "
    "the tracking."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read one day's members into a :class:`Recording`.

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

    Raises
    ------
    ValueError
        If the good clusters' spikes differ from the file's curated spikes,
        a trial has too few spikes to time its samples, the rebuilt times do
        not increase, or fewer than two arms have enough samples to place the
        maze.
    """
    from scipy.io import loadmat

    folder = Path(path_or_stream)
    samples, clusters, rows = _read_sorting(folder)
    trial = loadmat(folder / _TRIAL_DATA, squeeze_me=True, struct_as_record=False)[
        "trialData"
    ]
    tetrodes = np.atleast_1d(trial.tetrode)
    good = np.array(
        [_cluster_id(row) for row in rows if row["group"].strip() == "good"]
    )
    spike_seconds = samples.astype(np.float64) / SAMPLING_HZ
    order = np.argsort(spike_seconds, kind="stable")
    spike_seconds, clusters = spike_seconds[order], clusters[order]
    spikes = tuple(spike_seconds[clusters == c] for c in good)
    curated = np.concatenate([np.atleast_1d(t.timestamp) for t in tetrodes])
    if not np.array_equal(np.sort(np.concatenate(spikes)), np.sort(curated)):
        msg = (
            f"{entry.session}: the clusters cluster_info.tsv labels good have "
            f"{sum(s.size for s in spikes)} spikes, which differ from the "
            f"{curated.size} curated spikes of trialData's tetrodes"
        )
        raise ValueError(msg)
    pos_sample = np.concatenate([np.atleast_1d(t.pos_sample) for t in tetrodes])
    ends = np.atleast_1d(np.asarray(trial.pos.trialFinalInd, dtype=np.int64))
    names = [str(name) for name in np.atleast_1d(trial.pos.trialNames)]
    times, fit = position_times(pos_sample, curated, ends, names)
    if not np.all(np.diff(times) > 0):
        k = int(np.argmax(np.diff(times) <= 0))
        trial_at = int(np.searchsorted(ends, k + 1, side="right"))
        overlap = float(times[k] - times[k + 1])
        frame = float(np.median(np.diff(times)))
        cause = (
            "under one frame interval: the file's spike-to-sample map puts "
            "spikes of both trials at the boundary"
            if overlap < frame
            else f"more than one frame interval ({frame * 1e3:.1f} ms), which "
            "spikes at the boundary mapped to both trials cannot explain; no "
            "source read says why"
        )
        msg = (
            f"{entry.session}: the rebuilt position times do not increase at "
            f"sample {k + 1}, where trial {names[trial_at]!r} starts "
            f"{overlap * 1e3:.2f} ms before trial {names[trial_at - 1]!r} ends, "
            f"{cause}; so the samples cannot be put on one clock"
        )
        raise ValueError(msg)
    header = dict(np.atleast_2d(trial.pos.header))
    ppm = float(header["pixels_per_metre"])
    positions = np.asarray(trial.pos.xy, dtype=np.float64) / ppm * 100.0
    starts = np.r_[0, ends[:-1]]
    windows = np.column_stack([times[starts], times[ends - 1]])
    rest = np.array([name.startswith("S") for name in names])
    maze = windows[~rest]
    exploration = np.array([name == "0" for name in names])
    in_maze = np.zeros(len(times), dtype=bool)
    for start, stop in zip(starts[~rest], ends[~rest], strict=True):
        in_maze[start:stop] = True
    graph, placement_evidence = star_graph(positions[in_maze])
    env, placement = _declared_star(graph, positions, placement_evidence, entry.session)
    table = unit_table(rows, good)
    mouse = entry.session.split("_")[0]
    dirs = np.asarray(trial.pos.dir, dtype=np.float64)
    displacement = np.asarray(trial.pos.dir_disp, dtype=np.float64)
    from neurospatial import Intervals

    trial_names = [name for name in names if not name.startswith("S")]
    return Recording(
        **entry.identity(path_or_stream),
        animal=f"mouse {mouse} ({GENOTYPES.get(mouse, 'genotype not listed')})",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the day's recordings combined for spike sorting, seconds from the "
            "first ('entire daily recordings were combined', "
            f"{_PAPER}): the 2-minute holding-box intervals are not in it, so "
            "each trial starts within a few frames of the previous one's end"
        ),
        times=times,
        spike_times=spikes,
        units=table,
        env=env,
        env_kind="declared",
        track_width_quantile=placement.width_quantile,
        env_source=(
            f"the stated radial maze, {N_ARMS} arms of {ARM_LENGTH:g} x "
            f"{ARM_WIDTH:g} cm on an octagonal centre platform {PLATFORM:g} cm "
            f"across ({_PAPER}), as a star graph with {BIN_SIZE:g} cm bins along "
            "the track"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the headstage's LEDs, filmed by an overhead camera at about 30 Hz "
            f"({_PAPER}); the file stores one LED (n_leds 1)"
        ),
        epochs={
            "exploration": windows[exploration],
            "trials": windows[~rest & ~exploration],
            "maze": maze,
            "rest": windows[rest],
        },
        trials=Intervals.from_arrays(
            maze[:, 0], maze[:, 1], trial=np.array(trial_names, dtype=object)
        ),
        params={
            "bin_size": BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "analysis_epochs": maze,
            "train_epochs": maze[::2],
            "test_epochs": maze[1::2],
        },
        conversions=(
            Conversion(
                f"spike_times: spike_times.npy / {SAMPLING_HZ:g} Hz for the "
                f"{len(good)} clusters cluster_info.tsv labels good",
                "stated",
                f"{_PAPER}: signals 'sampled at 30kHz', spike sorted with "
                "Kilosort 2.0 and refined in Phy; the spikes equal the "
                f"{curated.size} curated spike times of trialData's tetrodes",
            ),
            Conversion(
                f"positions: cm = pos.xy (pixels) / {ppm:g} pixels per metre x 100",
                "stated",
                f"trialData.pos.header: pixels_per_metre {ppm:g}; the camera "
                "frame's handedness is not stated (whether pixel y grows up or "
                "down the room), so the positions keep the file's axes and may "
                "be the room's mirror image; the arms are numbered by angle in "
                "this frame",
            ),
            Conversion(
                "times: per trial, a least-squares line from position sample "
                "index to the time of the spikes mapped to it (tetrode(k)."
                "pos_sample), evaluated at every sample",
                "inferred",
                f"the file stores no sample times and its header states "
                f"{header['sample_rate']}; {fit.evidence}",
            ),
            Conversion(
                "epochs: trialFinalInd's trials named by trialNames: '0' is "
                "'exploration', the other numbers 'trials', 'maze' both, and "
                "S* 'rest'",
                "stated",
                f"{_PAPER}: each day begins with an 8-minute 'exploration' phase, "
                "then trials, with a 45-minute rest recording between two "
                f"sessions; this day's trials {names}",
            ),
            placement.conversion,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            fit.watch_out,
            *_genotype_watch_out(mouse),
            _direction_watch_out(dirs, displacement),
            _PROCESSED,
        ),
    )


def _read_sorting(
    folder: Path,
) -> tuple[NDArray[np.uint64], NDArray[np.int64], list[dict[str, str]]]:
    """The Kilosort spike samples and clusters, and cluster_info.tsv's rows."""
    samples = np.load(folder / _NAMES[0], allow_pickle=False).ravel()
    clusters = np.load(folder / _NAMES[1], allow_pickle=False).ravel()
    with (folder / _NAMES[2]).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return samples, clusters.astype(np.int64), rows


def _cluster_id(row: dict[str, str]) -> int:
    """A cluster_info.tsv row's cluster id: column ``id``, or ``cluster_id``."""
    return int(row["id"] if "id" in row else row["cluster_id"])


class PositionClock(NamedTuple):
    """How the position samples were timed: the conversion's evidence and a watch-out."""

    evidence: str
    watch_out: str


def position_times(
    pos_sample: NDArray[np.integer[Any]],
    spike_time: NDArray[np.float64],
    trial_ends: NDArray[np.int64],
    names: list[str],
) -> tuple[NDArray[np.float64], PositionClock]:
    """Time each position sample from the spikes the file maps to it.

    Parameters
    ----------
    pos_sample : ndarray, shape (n_spikes,)
        1-based position sample of each spike.
    spike_time : ndarray, shape (n_spikes,)
        The spikes' times (s).
    trial_ends : ndarray, shape (n_trials,)
        1-based index of each trial's last position sample.
    names : list of str
        Each trial's name, for the messages.

    Returns
    -------
    times : ndarray, shape (trial_ends[-1],)
        Per trial, the least-squares line from sample index to spike time,
        evaluated at each of the trial's samples.
    clock : PositionClock
        The fitted rates and residuals, as evidence and as a watch-out.

    Raises
    ------
    ValueError
        If a trial has fewer than ``MIN_FIT_SPIKES`` spikes mapped to it.
    """
    index = np.asarray(pos_sample, dtype=np.float64) - 1
    times = np.empty(int(trial_ends[-1]))
    rates, worst, n_far, n_fitted = [], 0.0, 0, 0
    starts = np.r_[0, trial_ends[:-1]]
    for name, start, stop in zip(names, starts, trial_ends, strict=True):
        inside = (index >= start) & (index < stop)
        if (n := int(inside.sum())) < MIN_FIT_SPIKES:
            msg = (
                f"trial {name!r} (position samples {start + 1} to {stop}) has {n} "
                f"spikes mapped to it, fewer than the {MIN_FIT_SPIKES} its clock "
                "is fitted on, so its samples cannot be timed"
            )
            raise ValueError(msg)
        slope, intercept = np.polyfit(index[inside], spike_time[inside], 1)
        residual = np.abs(spike_time[inside] - (intercept + slope * index[inside]))
        worst = max(worst, float(residual.max()))
        n_far += int(np.sum(residual > RESIDUAL))
        n_fitted += n
        times[start:stop] = intercept + slope * np.arange(start, stop)
        rates.append(1.0 / slope)
    low, high = min(rates), max(rates)
    measured = (
        f"fitted rates {low:.3f} to {high:.3f} Hz against the header's 30 Hz; the "
        f"largest residual is {worst * 1e3:.0f} ms, and {n_far} of {n_fitted} "
        f"spikes lie more than {RESIDUAL * 1e3:.0f} ms from their trial's line"
    )
    exact = (
        " Every fitted rate is the header's 30 Hz: the file's map was likely "
        "built from that rate, so the fit reproduces it and does not measure "
        "the camera."
        if np.allclose(rates, 30.0, atol=1e-4)
        else ""
    )
    watch_out = (
        "Position times are rebuilt per trial from the spike-to-sample map: "
        f"{measured}.{exact} Sample index / 30 Hz, the header's rate, would drift "
        "from the spikes over a day. Times within a trial are a straight line: "
        "dropped frames, if any, are not located."
    )
    return times, PositionClock(f"{measured} (over {len(rates)} trials)", watch_out)


def unit_table(rows: list[dict[str, str]], good: NDArray[np.int64]) -> pd.DataFrame:
    """The good clusters' table, with cluster_info.tsv's Phy columns.

    Parameters
    ----------
    rows : list of dict
        ``cluster_info.tsv``'s rows.
    good : ndarray of int, shape (n_units,)
        The good clusters' ids, in ``spike_times`` order.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``unit_id`` (the cluster id): ``brain_region`` CA1,
        ``label`` None, and the Phy columns ``KSLabel``, ``ContamPct``,
        ``Amplitude`` and ``ch`` (NaN or None where the cell is empty).
    """
    import pandas as pd

    by_id = {_cluster_id(row): row for row in rows}

    def column(name: str, kind: type) -> list[Any]:
        # An empty cell is a missing value, kept as NaN (or None for text).
        values = [by_id[c][name].strip() for c in good.tolist()]
        if kind is str:
            return [value or None for value in values]
        return [kind(value) if value else np.nan for value in values]

    kinds = {"KSLabel": str, "ContamPct": float, "Amplitude": float, "ch": float}
    return pd.DataFrame(
        {
            "brain_region": "CA1",
            "label": None,
            **{name: column(name, kind) for name, kind in kinds.items()},
        },
        index=pd.Index(good, name="unit_id"),
    )


def star_graph(maze_positions: NDArray[np.float64]) -> tuple[nx.Graph, str]:
    """The stated radial maze placed on the maze trials' samples.

    Eight directions 45 degrees apart (the arms of a regular octagon) are
    turned to fit the samples' angular histogram around the middle of their
    extent. Each arm with at least ``MIN_ARM_SAMPLES`` samples in its sector
    beyond the platform then takes the principal axis of those samples, and
    the centre is the point nearest those axes in least squares; the sectors
    are taken again around that centre and the arms refitted once. An arm
    with fewer samples keeps its direction from the octagon. From the centre
    each arm has a node at the platform's edge (half the stated platform
    diameter) and one at the arm's end (the stated arm length past it).

    Parameters
    ----------
    maze_positions : ndarray, shape (n_samples, 2)
        Samples of the maze trials, cm.

    Returns
    -------
    graph : networkx.Graph
        Nodes ``centre``, ``platform_k`` and ``arm_k`` (``k`` from 0, by
        angle) with ``pos``; edges with ``distance``.
    evidence : str
        The placement's evidence: the arms' directions, how far the fitted
        axes pass from the centre and how far the samples reach along each
        arm.

    Raises
    ------
    ValueError
        If fewer than two arms have ``MIN_ARM_SAMPLES`` samples, so no
        centre can be fitted.
    """
    import networkx as nx

    samples = maze_positions[tracked_mask(maze_positions)]
    low, high = np.percentile(samples, [0.5, 99.5], axis=0)
    centre = (low + high) / 2
    directions = _octagon(samples, centre)
    for _ in range(2):
        units, fitted = _arm_axes(samples, centre, directions)
        lines = [(mean, units[k]) for k, mean in fitted.items()]
        if len(lines) < 2:
            msg = (
                f"{len(lines)} arms have {MIN_ARM_SAMPLES} or more samples beyond "
                "the platform; at least two are needed to place the maze's centre"
            )
            raise ValueError(msg)
        projector = sum(np.eye(2) - np.outer(u, u) for _, u in lines)
        target = sum((np.eye(2) - np.outer(u, u)) @ m for m, u in lines)
        centre = np.linalg.solve(projector, target)
        directions = [float(np.arctan2(u[1], u[0])) for u in units]
    radius = PLATFORM / 2
    graph = nx.Graph()
    graph.add_node("centre", pos=(float(centre[0]), float(centre[1])))
    reaches = []
    for k, u in enumerate(units):
        edge = centre + radius * u
        end = edge + ARM_LENGTH * u
        graph.add_node(f"platform_{k}", pos=(float(edge[0]), float(edge[1])))
        graph.add_node(f"arm_{k}", pos=(float(end[0]), float(end[1])))
        graph.add_edge("centre", f"platform_{k}", distance=radius)
        graph.add_edge(f"platform_{k}", f"arm_{k}", distance=ARM_LENGTH)
        rel = samples - centre
        along = rel @ u
        on_arm = (np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]) < ARM_WIDTH) & (
            along > radius
        )
        if on_arm.any():
            reaches.append(float(np.quantile(along[on_arm], 0.99)))
    misses = [
        float(abs((mean - centre) @ np.array([units[k][1], -units[k][0]])))
        for k, mean in fitted.items()
    ]
    angles = ", ".join(f"{np.degrees(np.arctan2(u[1], u[0])):.0f}" for u in units)
    unfitted = [k for k in range(N_ARMS) if k not in fitted]
    evidence = (
        f"the {len(samples)} tracked samples of the maze trials: arms at {angles} "
        f"degrees, {len(fitted)} of them the principal axes of their samples "
        "beyond the platform"
        + (
            f" (arms {unfitted} have fewer than {MIN_ARM_SAMPLES} samples there "
            "and keep the octagon's direction, 45 degrees from their neighbours')"
            if unfitted
            else ""
        )
        + f"; centre ({centre[0]:.1f}, {centre[1]:.1f}) cm, the point nearest "
        f"those axes, which pass {min(misses):.1f} to {max(misses):.1f} cm from "
        "it; along each visited arm the samples' 99th percentile reaches "
        f"{min(reaches):.1f} to {max(reaches):.1f} cm from the centre against the "
        f"stated {radius + ARM_LENGTH:.2f} cm ({ARM_LENGTH:g} cm arms from a "
        f"{PLATFORM:g} cm platform)"
    )
    return graph, evidence


def _octagon(samples: NDArray[np.float64], centre: NDArray[np.float64]) -> list[float]:
    """Eight directions 45 degrees apart, turned to the samples' angular histogram.

    The turn maximizes the summed counts (1 degree bins, smoothed over 9) of
    the samples beyond their median distance from ``centre`` at the eight
    directions.
    """
    offset = samples - centre
    distance = np.linalg.norm(offset, axis=1)
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    outer = distance > np.quantile(distance, 0.5)
    counts, _ = np.histogram(angle[outer], bins=360, range=(-np.pi, np.pi))
    smooth = np.convolve(np.tile(counts, 3), np.ones(9) / 9, mode="same")[360:720]
    step = 360 // N_ARMS
    turn = int(np.argmax([smooth[t::step].sum() for t in range(step)]))
    return [np.radians(turn + 0.5 + step * k - 180.0) for k in range(N_ARMS)]


def _arm_axes(
    samples: NDArray[np.float64], centre: NDArray[np.float64], directions: list[float]
) -> tuple[list[NDArray[np.float64]], dict[int, NDArray[np.float64]]]:
    """Each arm's unit direction, and the mean of the samples of the arms fitted.

    An arm's samples lie in its sector (``SECTOR`` of the angle between arms
    either side of its direction) beyond the platform's edge and
    ``PLATFORM_MARGIN``; with ``MIN_ARM_SAMPLES`` or more its direction is
    their principal axis, otherwise it keeps ``directions[k]``.
    """
    offset = samples - centre
    distance = np.linalg.norm(offset, axis=1)
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    width = SECTOR * np.pi / N_ARMS
    units, fitted = [], {}
    for k, direction in enumerate(directions):
        mine = (np.abs(np.angle(np.exp(1j * (angle - direction)))) < width) & (
            distance > PLATFORM / 2 + PLATFORM_MARGIN
        )
        u = np.array([np.cos(direction), np.sin(direction)])
        if mine.sum() >= MIN_ARM_SAMPLES:
            points = samples[mine]
            mean = points.mean(axis=0)
            axis = np.linalg.svd(points - mean, full_matrices=False)[2][0]
            u = axis if axis @ (mean - centre) >= 0 else -axis
            fitted[k] = mean
        units.append(u)
    return units, fitted


class _Placement(NamedTuple):
    """The star's placement conversion and the width quantile it used."""

    conversion: Conversion
    width_quantile: float | None


def _declared_star(
    graph: nx.Graph,
    positions: NDArray[np.float64],
    placed: str,
    session: str,
) -> tuple[Environment, _Placement]:
    """The star as a track environment, its width a placement tolerance."""
    from neurospatial import Environment

    order = [
        edge
        for k in range(N_ARMS)
        for edge in (("centre", f"platform_{k}"), (f"platform_{k}", f"arm_{k}"))
    ]
    pos = {node: np.asarray(graph.nodes[node]["pos"]) for node in graph.nodes}
    tracked = positions[tracked_mask(positions)]
    distance = nearest_segment(
        tracked,
        np.array([pos[a] for a, _ in order]),
        np.array([pos[b] for _, b in order]),
    ).distance
    width, quantile, tolerance = placement_tolerance(
        distance,
        TOLERANCE_QUANTILE,
        floor=ARM_WIDTH,
        measured="the tracked LED's distance from the star",
        reason=(
            "The head-mounted LED swings past the arms' edges and the rest box "
            f"on the platform holds the mouse off the graph; the stated "
            f"{ARM_WIDTH:g} cm arm width is the narrowest the track is."
        ),
    )
    env = Environment.from_graph(
        graph,
        order,
        bin_size=BIN_SIZE,
        units="cm",
        name=f"shipley_2026 {session}",
        track_width=width,
    )
    sized = (
        f"{width:.3g} cm, the stated arm width (twice the samples' "
        f"{TOLERANCE_QUANTILE:.0%} quantile distance from the star is narrower)"
        if quantile is None
        else f"{width:.3g} cm, a placement tolerance wider than the stated "
        f"{ARM_WIDTH:g} cm arm width"
    )
    conversion = Conversion(
        f"environment: the stated star placed on the samples, track_width {sized}",
        "inferred",
        f"{placed}; {tolerance}",
    )
    return env, _Placement(conversion, quantile)


def _genotype_watch_out(mouse: str) -> tuple[str, ...]:
    """The mouse's genotype, which analyses comparing mice need."""
    genotype = GENOTYPES.get(mouse)
    if genotype is None:
        return (f"The record's MouseStats.xlsx lists no genotype for {mouse}.",)
    return (
        f"Genotype: {mouse} is {genotype} (the record's MouseStats.xlsx, sheet "
        "Exp1Sessions). The paper compares App NL-G-F/ChAT-cre mice with "
        "wild-type littermates; pool sessions by genotype, not across it.",
    )


def _direction_watch_out(
    dirs: NDArray[np.float64], displacement: NDArray[np.float64]
) -> str:
    """Why pos.dir is not a head direction, with the count that shows it."""
    same = int(np.sum(np.isclose(dirs, displacement, equal_nan=True)))
    share = f"all {dirs.size}" if same == dirs.size else f"{same} of {dirs.size}"
    return (
        f"No headings: the file stores one LED (n_leds 1), though the paper's Methods fit "
        f"two, and pos.dir equals dir_disp in {share} samples: it is the "
        "direction of displacement, not of the head, so head direction cannot be "
        "found from the file."
    )
