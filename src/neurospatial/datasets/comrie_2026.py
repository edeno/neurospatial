"""Comrie et al. (2026): rat hippocampus in a spatial bandit maze (Zenodo 23020757).

What was recorded: rat j16 on 2021-07-10, run epoch ``02_r1`` of the
spatial-bandit task of Comrie et al. (2026), with 203 spike-sorted
hippocampal units (the record's word; the preprint's tetrodes are in dorsal
CA1). The maze has three Y-shaped patches of hallways joined at
a centre, two reward ports per patch: nine hallways 6.5 cm wide, each 53 cm
long, meeting at 120 degrees. The head position is the centroid of a front
and a back LED on the implant, filmed at 30 frames/s and, in the authors'
pipeline, converted to cm, smoothed, interpolated and upsampled to 500 Hz.
The file also holds the authors' linearized position and the track graph
they linearized with (10 nodes, 9 edges, nodes at the reward ports and the
hallway intersections).

Where it lives: Zenodo record 23020757 (CC-BY-4.0), one ``.npz`` of numeric
and string arrays, 75.1 MB, published with Zeng, Comrie, Frank, Eden and
Denovellis's *Local goodness-of-fit measures for neural decoding* (its
Figure 4 input). The record states it has no pickled objects; it is read
with ``np.load(..., allow_pickle=False)``, which refuses any object array.
The raw recording is DANDI 001942; the spike sorting is only in this file.

Sources read: the Zenodo record's description, the file, and the bioRxiv
preprint of Comrie et al. (doi:10.1101/2024.09.23.613567, version 1, read on
PubMed Central, PMC11463554). The published Neuron article was not
available to read; the maze and tracking facts here are the preprint's.

What the loader converts:

- positions: ``head_position_x`` and ``head_position_y`` as stored, in cm.
  Neither the record nor the file states the unit; it is cm because the
  graph's edges, in the positions' unit, are 52 to 57 long against the
  53 cm hallways the preprint states, and the preprint says the positions
  were converted to cm (an inferred ``Conversion``). The file's
  ``head_speed`` matches the speed of the stored positions (|step| / dt)
  in shape (correlation at least 0.8) and in scale (the ratio of their
  median speeds within a factor of 2 of the 1.26 measured on this file),
  so speed and position share that unit; the loader refuses
  the file if they stop agreeing.
- times stay as stored: seconds on the Unix clock (1.6259e9 s is
  2021-07-10, the session's date); float64 resolves them to about 0.2 µs.
- spike trains are cut from ``spike_times`` at ``spike_offsets``, as the
  record describes. 21 of the 203 units have no spikes in this epoch; they
  are kept as empty trains, as the file lists them.
- headings: none. ``head_orientation`` (radians, [-π, π]) is not used as
  head direction: minus the direction of movement while running it has a
  circular mean of about +13 degrees. The
  spread (about 20 degrees) says it is the head's direction; the constant
  offset is either the LEDs' mounting or the rat's head posture, and the
  file cannot tell which. ``params["head_orientation"]`` keeps it.

The declared environment is the authors' track graph from the file, every
edge kept, laid out in the file's own ``linear_edge_order`` with its
``linear_edge_spacing`` (15 cm), so ``env.to_linear`` reproduces the file's
``linear_position``. Bins are 2 cm along the track. About 64% of head
samples lie beyond half the 6.5 cm hallway width, most when the rat is
stopped: the head swings past a hallway's low walls and a raised head
projects outward, while the graph sits where the samples are. So the track
is built with a placement tolerance measured from the samples
(``track_width``, an inferred ``Conversion``), not the hallway width, which
would leave most samples in no bin.

Analysis choices (``params``):

- ``bin_size`` 4 cm, for a 2-D grid over the maze, and ``graph_bin_size``
  2 cm along the track, the environment's bins;
- ``min_speed`` 10 cm/s, the preprint's Methods' threshold for locomotion
  ("valid times of high speed (>10 cm/s)"); also the speed above which
  heading is compared with the direction of movement;
- ``dt`` 0.25 s, the decoding bin, and ``train_epochs`` / ``test_epochs``,
  alternating 60 s blocks;
- ``head_orientation``: the file's head orientation, for comparison.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._recording import (
    Conversion,
    Recording,
    session_speed,
    tracked_mask,
)
from neurospatial.datasets._tracks import NearestSegment, nearest_segment

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets._catalog import SessionEntry

__all__ = ["REQUIRES", "WATCH_OUTS", "WORKED_RESULT", "load_session"]

#: A level B sanity property of the data and decoder, not a finding of the
#: preprint, tested against measured baselines
#: (``tests/datasets/test_worked_results.py``).
WORKED_RESULT: dict[str, Any] = {
    "status": "sanity property",
    "level": "B",
    "finding": (
        "a sanity property of the data and decoder, not a finding of the "
        "preprint: while the rat runs, the decoded position usually lies on "
        "the rat's own track segment"
    ),
    "quote": (
        '"Non-local representation times were identified as times in which '
        "the most likely decoded position (hereafter \u201cdecoded "
        "position\u201d) was in a track segment distinct to the animal\u2019s "
        'actual position." (Methods, "Non-local representation of Stay and '
        'Switch paths"); place cells\u2019 activity "typically signals the '
        'actual location of the animal" (Introduction)'
    ),
    "source": (
        "Comrie et al., bioRxiv 2024.09.23.613567 v1 (PMC11463554), "
        "Introduction and Methods; the published Neuron article was not "
        "available to read"
    ),
    "paper": "no share stated",
    "ours": 0.924,
    "criterion": (
        "the share of held-out running decoding bins on the rat's own segment "
        "exceeds both a constant decoder naming the segment the rat runs on "
        "most (35.1% of those bins) and the 95th percentile of the same share "
        "after shifting every unit's spikes by one common circular offset "
        "(20 draws, seed 2026: 18.1% to 29.4%, 95th percentile 28.1%); fixed "
        "before the run"
    ),
    "parameters": (
        "BayesianDecoder on rec.env (the authors' track graph, 2 cm bins, "
        "default diffusion KDE), dt 0.25 s, fitted on params['train_epochs'] "
        "with min_speed 10 cm/s and decoded on params['test_epochs']; a bin "
        "counts when the speed at its centre is above 10 cm/s; the segment "
        "is the graph edge (source_edge_id) of the MAP bin and of the rat's "
        "bin"
    ),
    "unmatched": (
        "the decoder: the preprint's is clusterless and state-space, in "
        "2 ms bins, fitted and decoded on the same data; ours uses the "
        "file's sorted units, 0.25 s bins, held-out time",
        "the preprint's confidence criterion (the highest 50% of the "
        "posterior within 50 cm) is not applied",
    ),
    "attempted": None,
    "why": None,
}

#: Modules of the ``datasets`` extra the loader imports.
REQUIRES: tuple[str, ...] = ()

#: The hallway width and length the preprint's Methods state, in cm.
HALLWAY_WIDTH = 6.5
HALLWAY_LENGTH = 53.0
BIN_SIZE = 4.0
GRAPH_BIN_SIZE = 2.0
#: The preprint's locomotion threshold, cm/s: its Methods analyse "valid
#: times of high speed (>10 cm/s)".
MIN_SPEED = 10.0
DT = 0.25
BLOCK = 60.0
#: Below this correlation of the file's head_speed with the speed of the
#: stored positions, the positions are not in the unit of head_speed.
MIN_SPEED_AGREEMENT = 0.8
#: The ratio of the median speed of the stored positions (|step| / dt) to the
#: file's median head_speed, measured on the pinned file: 8.04 / 6.39 cm/s.
#: head_speed is smoothed, so it runs low.
REFERENCE_SPEED_RATIO = 1.26
#: The largest factor, either way, between that ratio on a loaded file and
#: REFERENCE_SPEED_RATIO. A correlation ignores scale, so this bound catches a
#: unit error: the nearest units to cm move the ratio by 2.54 (inch), 10 (mm)
#: and 100 (m), all outside a factor of 2 on either side of the reference.
MAX_SPEED_RATIO = 2.0
#: The middle of each edge, as shares of its length from its first node:
#: its samples' lateral bias is reported beside the whole edge's, leaving out
#: the ends, where the head's posture at a port or junction shifts it. The
#: window was chosen after seeing the whole-edge bias.
MID_EDGE = (0.2, 0.8)
#: The share of samples the placement tolerance leaves outside the track:
#: its half-width is the samples' 99th percentile distance from the graph.
TOLERANCE_QUANTILE = 0.99

_PREPRINT = "the Methods of Comrie et al.'s preprint, bioRxiv 2024.09.23.613567"

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "j1620210710_02_r1_figure04_inputs": (
        "The units are the authors' spike sorting of this one run epoch; the raw "
        "recording is DANDI 001942, which does not hold this sorting (the Zenodo "
        "record), so these units cannot be matched to another epoch's.",
    ),
}

_INTERPOLATED = (
    "Positions are the authors' smoothed, interpolated head positions, "
    "upsampled to 500 Hz from a 30 frames/s camera "
    f"({_PREPRINT}), about 16 samples per video frame, so speeds and held-sample "
    "counts are those of smoothed, interpolated positions, not of raw tracking."
)


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read the Zenodo 23020757 ``.npz`` into a :class:`Recording`.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached ``.npz`` file.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    ValueError
        If the file's ``head_speed`` no longer matches the speed of the stored
        positions (correlation below 0.8, or the ratio of their median speeds
        more than a factor of 2 from the 1.26 measured on the pinned file), or
        its ``linear_edge_order`` leaves out an edge of its
        track graph.
    """
    with np.load(os.fspath(path_or_stream), allow_pickle=False) as arrays:
        z = {name: arrays[name] for name in arrays.files}
    times = np.asarray(z["position_time"], dtype=np.float64)
    positions = np.column_stack(
        [z["position/head_position_x"], z["position/head_position_y"]]
    ).astype(np.float64)
    cm = _cm_evidence(times, positions, z)
    orientation = np.asarray(z["position/head_orientation"], dtype=np.float64)
    offset, spread = heading_vs_movement(
        times, positions, orientation, min_speed=MIN_SPEED
    )
    offsets = np.asarray(z["spike_offsets"])
    spike_times = tuple(
        np.asarray(z["spike_times"][offsets[i] : offsets[i + 1]], dtype=np.float64)
        for i in range(len(offsets) - 1)
    )
    n_empty = sum(train.size == 0 for train in spike_times)
    graph = track_graph(z)
    env, distance, placement, width_quantile = _declared_track(
        graph, z, positions, entry.session
    )
    train, test = alternating_blocks(times[0], times[-1], BLOCK)
    import pandas as pd

    units = pd.DataFrame(
        {"brain_region": "hippocampus", "label": None},
        index=pd.Index(np.arange(len(spike_times)), name="unit_id"),
    )
    edges = np.asarray(z["track_edge_distance"], dtype=np.float64)
    return Recording(
        **entry.identity(path_or_stream),
        animal="rat j16",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the Unix clock, seconds since 1970-01-01 UTC, as stored "
            f"({times[0]:.10g} s falls on 2021-07-10, the session's date)"
        ),
        times=times,
        spike_times=spike_times,
        units=units,
        env=env,
        env_kind="declared",
        track_width_quantile=width_quantile,
        env_source=(
            f"the authors' track graph from the file ({graph.number_of_nodes()} "
            f"nodes, {graph.number_of_edges()} edges, every edge kept), "
            f"{GRAPH_BIN_SIZE:g} cm bins along the track; the preprint's Methods "
            f"state {HALLWAY_WIDTH:g} cm wide, {HALLWAY_LENGTH:g} cm hallways "
            "meeting at 120 degrees"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: the centroid of the front and back LEDs on the implant, "
            f"filmed at 30 frames/s and upsampled to 500 Hz ({_PREPRINT})"
        ),
        linearized=np.asarray(z["position/linear_position"], dtype=np.float64),
        params={
            "bin_size": BIN_SIZE,
            "graph_bin_size": GRAPH_BIN_SIZE,
            "min_speed": MIN_SPEED,
            "dt": DT,
            "train_epochs": train,
            "test_epochs": test,
            "head_orientation": orientation,
        },
        conversions=(
            Conversion(
                "positions: head_position_x and head_position_y as stored, in cm",
                "inferred",
                cm,
            ),
            Conversion(
                f"spike_times: spike_times split at spike_offsets into "
                f"{len(spike_times)} units, {n_empty} with no spikes",
                "stated",
                "the Zenodo record: 'the spike times of 203 units, 870,018 spikes "
                "(spike_times, split into units by spike_offsets)'",
            ),
            Conversion(
                "environment: Environment.from_graph on the file's track_nodes, "
                "track_node_positions, track_edges and track_edge_distance, edges "
                f"in the file's linear_edge_order, linear_edge_spacing "
                f"{float(z['linear_edge_spacing']):g} cm",
                "stated",
                f"the file's track graph; {_PREPRINT}: nodes at the reward ports "
                "and hallway intersections, 15 cm gaps between edges in 1-D; "
                f"edge lengths {edges.min():.1f} to {edges.max():.1f} cm against "
                f"the stated {HALLWAY_LENGTH:g} cm; env.to_linear reproduces the "
                f"file's linear_position to {distance:.2g} cm",
            ),
            placement,
        ),
        watch_outs=(
            *WATCH_OUTS.get(entry.session, ()),
            _heading_watch_out(offset, spread),
            _beyond_hallway(times, positions, graph),
            _INTERPOLATED,
        ),
    )


def track_graph(arrays: dict[str, NDArray[Any]]) -> nx.Graph:
    """The authors' track graph: node ``pos`` in cm, edge ``distance`` in cm.

    Parameters
    ----------
    arrays : dict of str to ndarray
        The file's arrays: ``track_nodes`` (n_nodes,), ``track_node_positions``
        (n_nodes, 2), ``track_edges`` (n_edges, 2) and ``track_edge_distance``
        (n_edges,).
    """
    import networkx as nx

    graph = nx.Graph()
    for node, xy in zip(
        arrays["track_nodes"], arrays["track_node_positions"], strict=True
    ):
        graph.add_node(int(node), pos=(float(xy[0]), float(xy[1])))
    for (a, b), distance in zip(
        arrays["track_edges"], arrays["track_edge_distance"], strict=True
    ):
        graph.add_edge(int(a), int(b), distance=float(distance))
    return graph


def heading_vs_movement(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    min_speed: float,
) -> tuple[float, float]:
    """Compare a heading series with the direction of movement while running.

    A heading in the positions' allocentric frame points along the path
    while the animal runs, so the circular mean of heading minus movement
    direction is near 0 and the spread is tens of degrees. A constant mean
    away from 0 is an offset (a mounting or posture); near ±π/2 or π a
    rotated or mirrored frame.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    positions : ndarray, shape (n_time, 2)
        In the length unit of ``min_speed``.
    headings : ndarray, shape (n_time,)
        Radians.
    min_speed : float
        Steps faster than this (length per second) count as running.

    Returns
    -------
    mean, spread : float
        Circular mean of heading minus the direction of the step to the next
        sample, and the median absolute deviation from that mean, in radians.
    """
    from neurospatial._angles import wrap_angle
    from neurospatial.stats.circular import circular_mean

    step = np.diff(positions, axis=0)
    speed = np.linalg.norm(step, axis=1) / np.diff(times)
    moving = np.arctan2(step[:, 1], step[:, 0])
    start = headings[:-1]
    running = (speed > min_speed) & np.isfinite(start)
    diff = wrap_angle(start[running] - moving[running])
    mean = circular_mean(diff)
    spread = float(np.median(np.abs(wrap_angle(diff - mean))))
    return mean, spread


def _cm_evidence(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    arrays: dict[str, NDArray[Any]],
) -> str:
    """Why the stored positions are cm; raises if head_speed no longer agrees."""
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    stored = np.asarray(arrays["position/head_speed"], dtype=np.float64)[1:]
    finite = np.isfinite(step_speed) & np.isfinite(stored)
    agreement = (
        float(np.corrcoef(step_speed[finite], stored[finite])[0, 1])
        if finite.sum() > 1
        else float("nan")
    )
    if not agreement >= MIN_SPEED_AGREEMENT:  # NaN fails too
        msg = (
            f"the file's head_speed no longer matches the speed of the stored "
            f"positions (r = {agreement:.2f}, below {MIN_SPEED_AGREEMENT:g}), so "
            "the positions may not be in cm"
        )
        raise ValueError(msg)
    stored_median = float(np.median(stored[finite]))
    step_median = float(np.median(step_speed[finite]))
    ratio = step_median / stored_median if stored_median > 0 else float("inf")
    low, high = (
        REFERENCE_SPEED_RATIO / MAX_SPEED_RATIO,
        REFERENCE_SPEED_RATIO * MAX_SPEED_RATIO,
    )
    if not low <= ratio <= high:
        msg = (
            "the speed of the stored positions is not in the unit of the file's "
            f"head_speed: their medians are {step_median:.3g} and "
            f"{stored_median:.3g} per s, a ratio of {ratio:.3g}, outside "
            f"{low:.3g} to {high:.3g} (within a factor of {MAX_SPEED_RATIO:g} of "
            f"the {REFERENCE_SPEED_RATIO:g} measured on the pinned file). A unit "
            "error (mm, m or inch instead of cm) moves the ratio by a factor of "
            "10, 100 or 2.54, so the positions may not be in cm. Check that the "
            "file is Zenodo 23020757's and that its sha256 matches the registry"
        )
        raise ValueError(msg)
    edges = np.asarray(arrays["track_edge_distance"], dtype=np.float64)
    return (
        "the file states no unit. Its head_speed matches |step| / dt of the "
        f"stored positions (correlation {agreement:.2f}; medians "
        f"{stored_median:.2f} and {step_median:.2f} per s, a factor of "
        f"{ratio:.2f}, within a factor of {MAX_SPEED_RATIO:g} of the "
        f"{REFERENCE_SPEED_RATIO:g} measured on the pinned file), so both "
        "are in one unit; the track graph's edges, in that unit, are "
        f"{edges.min():.1f} to {edges.max():.1f} long against the "
        f"{HALLWAY_LENGTH:g} cm hallways, and {_PREPRINT} says positions were "
        "converted from pixels to cm"
    )


def _declared_track(
    graph: nx.Graph,
    arrays: dict[str, NDArray[Any]],
    positions: NDArray[np.float64],
    session: str,
) -> tuple[Environment, float, Conversion, float | None]:
    """The track graph as an environment, with every edge kept.

    ``Environment.from_graph`` bins only the edges in ``edge_order``, so an
    order that leaves one out would lose it silently; the file's order is
    checked to hold every edge.

    Returns
    -------
    env : Environment
    distance : float
        The largest difference between ``env.to_linear`` and the file's
        ``linear_position``, in cm, over the samples placed on the track.
    placement : Conversion
        The placement tolerance used as ``track_width``, and its evidence.
    width_quantile : float or None
        ``TOLERANCE_QUANTILE`` when it set ``track_width``; None when the
        hallway width or one bin floored it.
    """
    from neurospatial import Environment

    order = [(int(a), int(b)) for a, b in arrays["linear_edge_order"]]
    if {frozenset(edge) for edge in order} != {frozenset(e) for e in graph.edges}:
        msg = (
            f"{session}: linear_edge_order lists {len(order)} edges, not the "
            f"{graph.number_of_edges()} edges of track_edges, so the environment "
            "would leave edges out"
        )
        raise ValueError(msg)
    offsets = _distance_to_graph(positions, graph)
    whole = _edge_bias(positions, graph, (0.0, 1.0))
    middle = _edge_bias(positions, graph, MID_EDGE)
    reach = float(np.quantile(offsets, TOLERANCE_QUANTILE))
    width = max(2 * reach, HALLWAY_WIDTH, GRAPH_BIN_SIZE)
    env = Environment.from_graph(
        graph,
        order,
        bin_size=GRAPH_BIN_SIZE,
        edge_spacing=float(arrays["linear_edge_spacing"]),
        units="cm",
        name=f"comrie_2026 {session}",
        track_width=width,
    )
    placed = np.asarray(env.bin_at(positions)) >= 0
    linear = np.asarray(env.to_linear(positions[placed]))
    stored = np.asarray(arrays["position/linear_position"], dtype=np.float64)
    distance = float(np.max(np.abs(linear - stored[placed])))
    placement = Conversion(
        f"environment: track_width {width:.3g} cm, a placement tolerance, not the "
        f"{HALLWAY_WIDTH:g} cm hallway width: twice the samples' "
        f"{TOLERANCE_QUANTILE:.0%} quantile distance from the graph",
        "inferred",
        f"the tracked head's distance from the graph: median {np.median(offsets):.2f} "
        f"cm, {TOLERANCE_QUANTILE:.0%} quantile {reach:.3g} cm; "
        f"{np.mean(offsets > width / 2):.1%} of samples lie beyond the tolerance, "
        "by construction, and fall in no bin. Each edge's samples sit a median "
        f"{whole[0]:+.2f} to {whole[1]:+.2f} cm to one side of it over the whole "
        f"edge, inside the {HALLWAY_WIDTH / 2:g} cm half-width of a hallway, and "
        f"{middle[0]:+.2f} to {middle[1]:+.2f} cm on its middle "
        f"{MID_EDGE[1] - MID_EDGE[0]:.0%}, inside one {GRAPH_BIN_SIZE:g} cm bin. "
        "The ends are left out of the second figure because the head's posture "
        "while stopped at a reward port or junction, and samples of a "
        "neighbouring hallway near a junction, shift the median there; that "
        "window was chosen after seeing the whole-edge figure. Neither figure "
        "puts an edge's samples beyond its hallway, so the graph is not drawn "
        "beside them. track_width decides only which samples are placed, not "
        "the bins, their lengths or connectivity",
    )
    width_quantile = TOLERANCE_QUANTILE if width == 2 * reach else None
    return env, distance, placement, width_quantile


def _distance_to_graph(
    positions: NDArray[np.float64], graph: nx.Graph
) -> NDArray[np.float64]:
    """Each tracked sample's distance to the nearest edge, shape (n_tracked,)."""
    return _nearest_edge(positions, graph).distance


def _nearest_edge(positions: NDArray[np.float64], graph: nx.Graph) -> NearestSegment:
    """Each tracked sample's nearest edge (segment between its nodes) and offsets.

    ``segment`` indexes ``graph.edges`` order; ``lateral`` is positive to the
    left of the direction from an edge's first node to its second.
    """
    tracked = positions[tracked_mask(positions)]
    pos = {node: np.asarray(graph.nodes[node]["pos"]) for node in graph.nodes}
    starts = np.array([pos[a] for a, _ in graph.edges])
    ends = np.array([pos[b] for _, b in graph.edges])
    return nearest_segment(tracked, starts, ends)


def _edge_bias(
    positions: NDArray[np.float64],
    graph: nx.Graph,
    window: tuple[float, float],
) -> tuple[float, float]:
    """The lowest and highest per-edge median signed lateral offset, in cm.

    A graph drawn beside its samples has the samples of some edge consistently
    to one side of it; head swings about a correct graph do not.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
    graph : networkx.Graph
    window : (float, float)
        The part of each edge whose samples count, as shares of its length
        from its first node: ``(0, 1)`` for the whole edge.
    """
    _, edge, along, lateral = _nearest_edge(positions, graph)
    low, high = window
    middle = (along >= low) & (along <= high)
    medians = [
        float(np.median(lateral[(edge == k) & middle]))
        for k in range(graph.number_of_edges())
        if np.any((edge == k) & middle)
    ]
    return min(medians), max(medians)


def _heading_watch_out(offset: float, spread: float) -> str:
    """Why ``head_orientation`` is not ``headings``, with the measured offset."""
    return (
        "No headings: the file's head_orientation (from the front and back LEDs) "
        "minus the direction of movement above "
        f"{MIN_SPEED:g} cm/s has a circular mean of {np.degrees(offset):+.1f} "
        f"degrees (median spread {np.degrees(spread):.1f} degrees): the head's "
        "direction with a constant offset, from the LEDs' mounting or the rat's "
        "head posture, which the file cannot tell apart. It is kept in "
        "params['head_orientation'] (radians, [-π, π]); subtract the offset only "
        "if it is the mounting."
    )


def _beyond_hallway(
    times: NDArray[np.float64], positions: NDArray[np.float64], graph: nx.Graph
) -> str:
    """The share of head samples beyond the hallway's half-width, by speed."""
    from neurospatial.datasets._checks import speed_split

    beyond = _distance_to_graph(positions, graph) > HALLWAY_WIDTH / 2
    speed = session_speed(times, positions)
    split = (
        None
        if speed is None
        else speed_split(speed[tracked_mask(positions)], beyond, MIN_SPEED, "cm")
    )
    return (
        f"{beyond.mean():.1%} of tracked head samples lie farther than half the "
        f"{HALLWAY_WIDTH:g} cm hallway width from the track graph"
        + (f": {split}" if split else "")
        + ". This is tracking spread, not misfit: the head-mounted LEDs swing "
        f"past the hallways' walls (3 cm high, {_PREPRINT}) and a raised head "
        "projects outward, "
        "while each edge's samples are centred on it (see the environment's "
        "conversions). The positions are kept as stored."
    )
