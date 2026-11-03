"""Track graphs for declared track and maze environments, placed on the samples.

A loader that knows a track's layout (a straight track of stated length, a
W-track, a radial maze, a loop) builds its graph here and declares it with
``declare``: ``rec.geometry["environment"]`` becomes a zero-argument callable
returning ``Environment.from_graph`` on that graph with every edge kept
(``_geometry.graph_environment``). The source gives the topology and, where it
states them, lengths and the track width; the samples only place the nodes.

Placement is fitted on the clean samples inside ``load()``, so a hazard probe
that corrupts the positions meets the environment of the clean data. Every
fit reports its residual (``graph_residual``): the samples' distance to the
nearest track segment and, for a stated width, the share of samples within
half of it. ``declare`` refuses a fit whose median sample distance exceeds a
stated bound; the recording then stays ``inferred``.

Node fitting (``fit_nodes``) minimises a truncated squared distance from a
subsample of the samples to the graph's segments over the free node positions
(L-BFGS-B, seeded subsample). Truncation at ``cap`` keeps far-off samples
(tracking jumps, a rat off the track) from pulling the track toward them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FIT_SAMPLES = 6000
FIT_SEED = 0
EDGE_POINTS = 25
# Refusal of a fitted layout, on misfit evidence (``edge_fit``), not scatter:
# a segment whose samples sit consistently to one side by more than a bin, or
# a quarter of a segment's length with no sample within a bin.
MAX_EDGE_BIAS_BINS = 1.0
MAX_UNSUPPORTED = 0.25
BIAS_WINDOW_BINS = 3.0
REACH_QUANTILE = 0.995
# A stretch as long as a maze arm (about 50 cm at 2.5 cm bins): local
# features (a reward area, a junction, a rounded corner) are shorter. Set
# after seeing the data: it was 10 bins until the 000044 circle was refused
# on one short side under its reward area (commit 4b865baca), then widened.
# These limits are in bins, so they vary with each recording's bin: from 0.5
# cell (the labyrinth) and 2-4 cm to 5 px (hc-18). How sensitive they are to
# misfits of known size is measured in the survey's simulation ladder (phase
# 6), not here.
MIN_PIECE_BINS = 20.0
BASE_OFFSETS = (0.05, 0.1, 0.15, 0.2)


def graph_arrays(graph: Any) -> tuple[list[Any], NDArray[np.float64], NDArray[np.intp]]:
    """Node names, node positions ``(n_nodes, n_dims)`` and edges ``(n_edges, 2)``."""
    names = list(graph.nodes)
    lookup = {name: k for k, name in enumerate(names)}
    nodes = np.array(
        [np.atleast_1d(graph.nodes[name]["pos"]) for name in names], dtype=float
    )
    edges = np.array([[lookup[u], lookup[v]] for u, v in graph.edges], dtype=np.intp)
    return names, nodes, edges.reshape(-1, 2)


def segment_distance(
    points: NDArray[np.float64], nodes: NDArray[np.float64], edges: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Distance from each point to the nearest segment, shape ``(n_points,)``."""
    a = nodes[edges[:, 0]]
    ab = nodes[edges[:, 1]] - a
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    best = np.full(len(points), np.inf)
    for k in range(len(a)):  # few segments, vectorized over points
        t = np.clip((points - a[k]) @ ab[k] / length2[k], 0.0, 1.0)
        best = np.minimum(
            best, np.linalg.norm(points - (a[k] + t[:, None] * ab[k]), axis=1)
        )
    return best


def tracked(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    positions = np.asarray(positions, dtype=float)
    return positions[np.isfinite(positions).all(axis=1)]


def graph_residual(
    graph: Any, positions: NDArray[np.float64], track_width: float | None
) -> dict[str, float]:
    """How well the tracked samples lie on the graph.

    Returns the median and 95th percentile distance to the nearest segment,
    the total track length and, for a stated ``track_width``, the share of
    samples within half of it.
    """
    _, nodes, edges = graph_arrays(graph)
    dist = segment_distance(tracked(positions), nodes, edges)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    out = {
        "median_offset": float(np.median(dist)),
        "p95_offset": float(np.percentile(dist, 95)),
        "track_length": float(lengths.sum()),
    }
    if track_width is not None:
        out["share_within_half_width"] = float(np.mean(dist <= track_width / 2))
    return out


def edge_offsets(
    graph: Any, positions: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """Each tracked sample's nearest segment and its signed offsets from it.

    Returns
    -------
    segment : ndarray, shape (n_samples,)
        Index of the nearest segment (``graph.edges`` order).
    lateral : ndarray, shape (n_samples,)
        Signed distance across the segment's line, positive to the left of
        its direction from its first node to its second.
    along : ndarray, shape (n_samples,)
        Position of the projection along the segment as a fraction of its
        length, unclipped: outside [0, 1] beyond an end.
    """
    _, nodes, edges = graph_arrays(graph)
    samples = tracked(positions)
    a = nodes[edges[:, 0]]
    ab = nodes[edges[:, 1]] - a
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    best = np.full(len(samples), np.inf)
    segment = np.zeros(len(samples), dtype=np.intp)
    lateral = np.zeros(len(samples))
    along = np.zeros(len(samples))
    for k in range(len(a)):
        rel = samples - a[k]
        t = rel @ ab[k] / length2[k]
        dist = np.linalg.norm(rel - np.clip(t, 0, 1)[:, None] * ab[k], axis=1)
        closer = dist < best
        best[closer] = dist[closer]
        segment[closer] = k
        along[closer] = t[closer]
        if samples.shape[1] == 2:
            cross = ab[k, 0] * rel[:, 1] - ab[k, 1] * rel[:, 0]
            lateral[closer] = cross[closer] / np.sqrt(length2[k])
        else:
            lateral[closer] = 0.0
    return segment, lateral, along


def _straight_pieces(
    nodes: NDArray[np.float64],
    edges: NDArray[np.intp],
    *,
    min_length: float,
    max_turn: float = np.radians(20.0),
) -> list[list[tuple[int, float]]]:
    """For each segment, the nearly straight stretch of track around it.

    A segment is extended, through nodes joining exactly two segments, into
    neighbours that continue it within ``max_turn``, until the stretch is at
    least ``min_length`` long (or cannot grow). Each entry pairs a segment
    with +1 or -1, the sign that orients it along the stretch. A curve drawn
    as a polygon of short segments (a loop) is judged over stretches as long
    as an arm, not over each short side; a corner is not crossed.
    """
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    by_node: dict[int, list[int]] = {}
    for k, (u, v) in enumerate(edges):
        by_node.setdefault(int(u), []).append(k)
        by_node.setdefault(int(v), []).append(k)

    def direction(k: int, sign: float) -> NDArray[np.float64]:
        d = (nodes[edges[k, 1]] - nodes[edges[k, 0]]) * sign
        return d / max(np.linalg.norm(d), 1e-12)

    pieces = []
    for k in range(len(edges)):
        piece = [(k, 1.0)]
        total = lengths[k]
        # (segment, sign, node at the open end) for the head and the tail.
        ends = [(k, 1.0, int(edges[k, 1])), (k, -1.0, int(edges[k, 0]))]
        grown = True
        while total < min_length and grown:
            grown = False
            for i, (seg, sign, node) in enumerate(ends):
                nxt = [j for j in by_node[node] if j != seg]
                if (
                    len(by_node[node]) != 2
                    or not nxt
                    or any(j == q for j, _ in piece for q in nxt)
                ):
                    continue
                j = nxt[0]
                # Orient j to continue away from ``node``.
                sign_j = 1.0 if int(edges[j, 0]) == node else -1.0
                turn = np.arccos(
                    np.clip(direction(seg, sign) @ direction(j, sign_j), -1, 1)
                )
                if turn > max_turn:
                    continue
                far = int(edges[j, 1]) if sign_j > 0 else int(edges[j, 0])
                # On the tail the stretch runs backwards: keep one orientation.
                piece.append((j, sign_j if i == 0 else -sign_j))
                ends[i] = (j, sign_j, far)
                total += lengths[j]
                grown = True
        pieces.append(piece)
    return pieces


def edge_fit(
    graph: Any, positions: NDArray[np.float64], *, tolerance: float
) -> dict[str, float]:
    """Misfit evidence for a layout on its samples, independent of any width.

    ``tolerance`` is one bin along the track.

    - ``median_abs_lateral``: median distance of samples from the track's line,
      over samples that project inside a segment.
    - ``max_edge_bias``: the largest absolute median of the signed lateral
      offset over a stretch of nearly straight track at least
      ``MIN_PIECE_BINS`` bins long around each segment (``_straight_pieces``:
      the segment itself on a maze, several sides of a polygon on a loop),
      using samples within ``BIAS_WINDOW_BINS`` bins of it (so tracking jumps
      between arms do not count): track drawn beside its samples. A layout
      moved, scaled or rotated off its samples shifts most of its stretches
      sideways; a bend or a rounded corner the straight segments cut, or a
      reward area where the head leans out, is local and does not move a
      stretch's median.
    - ``max_unsupported``: the largest per-segment share of points along it
      with no sample within ``tolerance``: a segment where the animal never
      was.
    - ``end_overshoot``: how far the samples reach past the track's free ends
      (each end's ``REACH_QUANTILE`` reach, summed over the ends), as a share
      of the track's total length: a track drawn shorter than its samples,
      which the lateral and support measures do not see. ``max_end_reach``
      is the largest single end's reach, in the positions' units.
    """
    from scipy.spatial import cKDTree

    _, nodes, edges = graph_arrays(graph)
    segment, lateral, along = edge_offsets(graph, positions)
    inside = (along > 0) & (along < 1)
    near = inside & (np.abs(lateral) <= BIAS_WINDOW_BINS * tolerance)
    biases = []
    for piece in _straight_pieces(nodes, edges, min_length=MIN_PIECE_BINS * tolerance):
        offsets = np.concatenate(
            [sign * lateral[near & (segment == j)] for j, sign in piece]
        )
        if len(offsets) >= 10:
            biases.append(float(np.median(offsets)))
    tree = cKDTree(tracked(positions))
    steps = np.linspace(0.0, 1.0, EDGE_POINTS)
    unsupported = []
    for u, v in edges:
        points = nodes[u] + steps[:, None] * (nodes[v] - nodes[u])
        unsupported.append(float(np.mean(tree.query(points)[0] > tolerance)))
    # How far the samples reach past the track's free ends (nodes of degree
    # one), summed over the ends, as a share of the track's length: a track
    # drawn shorter than its samples. Summed over the whole track, not per
    # segment, so a dead end drawn as a short segment (a cell of the
    # labyrinth) is not judged on its own length.
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    reaches = []
    for k, (u, v) in enumerate(edges):
        mine = segment == k
        for free, past in (
            (degree[u] == 1, -along[mine]),
            (degree[v] == 1, along[mine] - 1),
        ):
            if free and mine.any():
                share = np.quantile(np.clip(past, 0, None), REACH_QUANTILE)
                reaches.append(float(share * lengths[k]))
    return {
        "end_overshoot": float(np.sum(reaches) / np.sum(lengths)),
        "max_end_reach": max(reaches, default=0.0),
        "median_abs_lateral": float(np.median(np.abs(lateral[inside])))
        if inside.any()
        else float("nan"),
        "max_edge_bias": float(np.max(np.abs(biases))) if biases else float("nan"),
        "max_unsupported": float(np.max(unsupported)),
    }


def with_positions(graph: Any, positions: Mapping[Any, Sequence[float]]) -> Any:
    """A copy of ``graph`` with node ``pos`` replaced and edge ``distance`` recomputed."""
    out = graph.copy()
    for name, pos in positions.items():
        out.nodes[name]["pos"] = tuple(float(v) for v in pos)
    for u, v in out.edges:
        a = np.asarray(out.nodes[u]["pos"], dtype=float)
        b = np.asarray(out.nodes[v]["pos"], dtype=float)
        out.edges[u, v]["distance"] = float(np.linalg.norm(b - a))
    return out


def graph_from_points(
    points: Mapping[Any, Sequence[float]], edges: Iterable[tuple[Any, Any]]
) -> Any:
    """A track graph with node ``pos`` from ``points`` and Euclidean ``distance``."""
    import networkx as nx

    graph = nx.Graph()
    for name, pos in points.items():
        graph.add_node(name, pos=tuple(float(v) for v in pos))
    graph.add_edges_from(edges)
    return with_positions(graph, points)


def polyline_graph(
    points: NDArray[np.float64],
    *,
    closed: bool,
    names: Sequence[str] | None = None,
) -> Any:
    """A path (or, ``closed``, a cycle) through ordered points.

    Nodes are ``names`` when given, else ``p0``, ``p1``, ….
    """
    names = list(names) if names is not None else [f"p{k}" for k in range(len(points))]
    edges = list(itertools.pairwise(names))
    if closed:
        edges.append((names[-1], names[0]))
    return graph_from_points(dict(zip(names, points, strict=True)), edges)


def fit_nodes(
    graph: Any,
    positions: NDArray[np.float64],
    *,
    cap: float,
    fixed: Iterable[Any] = (),
) -> tuple[Any, float]:
    """Move the free nodes of ``graph`` to fit the samples.

    The objective has two truncated squared terms: the samples' distance to
    the nearest segment (the track must pass where the samples are) and the
    distance from points along each segment to the nearest sample, weighted by
    segment length (the track must not extend where no sample is; without it
    an arm end could run off past the samples). Both use a seeded subsample of
    the tracked samples and are truncated at ``cap``.

    Returns
    -------
    graph : networkx.Graph
        A copy with fitted node positions and recomputed edge distances.
    loss : float
        The objective at the fit.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    names, nodes, edges = graph_arrays(graph)
    samples = tracked(positions)
    rng = np.random.default_rng(FIT_SEED)
    if len(samples) > FIT_SAMPLES:
        samples = samples[rng.choice(len(samples), FIT_SAMPLES, replace=False)]
    tree = cKDTree(samples)
    steps = np.linspace(0.0, 1.0, EDGE_POINTS)
    keep = set(fixed)
    free = np.array([name not in keep for name in names])

    def loss(flat: NDArray[np.float64]) -> float:
        moved = nodes.copy()
        moved[free] = flat.reshape(-1, nodes.shape[1])
        to_track = segment_distance(samples, moved, edges)
        a, b = moved[edges[:, 0]], moved[edges[:, 1]]
        along = a[:, None, :] + steps[None, :, None] * (b - a)[:, None, :]
        to_samples = tree.query(along.reshape(-1, nodes.shape[1]))[0]
        lengths = np.linalg.norm(b - a, axis=1)
        weight = np.repeat(lengths / max(lengths.sum(), 1e-12), EDGE_POINTS)
        return float(
            np.mean(np.minimum(to_track, cap) ** 2)
            + np.sum(weight * np.minimum(to_samples, cap) ** 2) / EDGE_POINTS
        )

    result = minimize(loss, nodes[free].ravel(), method="L-BFGS-B")
    fitted = nodes.copy()
    fitted[free] = result.x.reshape(-1, nodes.shape[1])
    out = with_positions(graph, dict(zip(names, fitted, strict=True)))
    return out, float(result.fun)


def best_fit(
    candidates: Iterable[Any], positions: NDArray[np.float64], *, cap: float
) -> tuple[Any, float]:
    """Fit each candidate layout and keep the one with the lowest loss."""
    fits = [fit_nodes(graph, positions, cap=cap) for graph in candidates]
    return min(fits, key=lambda fit: fit[1])


def principal_frame(
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centre of the samples and their principal axes (rows, long axis first)."""
    samples = tracked(positions)
    centre = samples.mean(axis=0)
    axes = np.linalg.svd(samples - centre, full_matrices=False)[2]
    return centre, axes


def straight_track(
    positions: NDArray[np.float64],
    length: float | None,
    *,
    trim: float = 0.005,
) -> tuple[Any, dict[str, float]]:
    """A straight track along the samples' principal axis.

    The ends are the ``trim`` and ``1 - trim`` quantiles of the samples along
    the axis; with a stated ``length``, the track keeps that length, centred
    between them. Returns the graph (nodes ``"start"``, ``"end"``) and the
    fit: axis angle (rad), the samples' own span, and the share of samples
    beyond either end.
    """
    samples = tracked(positions)
    centre, axes = principal_frame(samples)
    axis = axes[0]
    along = (samples - centre) @ axis
    low, high = np.quantile(along, [trim, 1 - trim])
    span = float(high - low)
    if length is not None:
        middle = 0.5 * (low + high)
        low, high = middle - length / 2, middle + length / 2
    ends = {"start": centre + low * axis, "end": centre + high * axis}
    beyond = float(np.mean((along < low) | (along > high)))
    fit = {
        "samples_span": span,
        "share_beyond_ends": beyond,
    }
    if samples.shape[1] == 2:
        fit["axis_angle_rad"] = float(np.arctan2(axis[1], axis[0]))
    return graph_from_points(ends, [("start", "end")]), fit


def parallel_arm_candidates(
    positions: NDArray[np.float64], n_arms: int = 3
) -> list[Any]:
    """Starting layouts for ``n_arms`` parallel arms joined by a base (a W or E).

    The base runs along one side of the samples' minimum rotated rectangle, at
    each of ``BASE_OFFSETS`` (shares of the rectangle's depth) inside it, and
    the arms leave it at equal spacing; each side and offset is a candidate
    (a base that curves into the arms, as on a 3-arm maze, fits a deeper
    start).
    Nodes: ``base_0`` … ``base_{n-1}`` and ``arm_0`` … ``arm_{n-1}``.
    """
    from shapely.geometry import MultiPoint

    samples = tracked(positions)
    rect = MultiPoint(samples).minimum_rotated_rectangle
    corners = np.asarray(rect.exterior.coords)[:4]
    candidates = []
    for k in range(4):
        b0, b1 = corners[k], corners[(k + 1) % 4]
        far = corners[(k + 2) % 4]
        out = far - b1
        for offset in BASE_OFFSETS:
            points: dict[str, NDArray[np.float64]] = {}
            for i, f in enumerate(np.linspace(0.1, 0.9, n_arms)):
                start = b0 + f * (b1 - b0)
                points[f"base_{i}"] = start + offset * out
                points[f"arm_{i}"] = start + 0.95 * out
            edges = [(f"base_{i}", f"base_{i + 1}") for i in range(n_arms - 1)]
            edges += [(f"base_{i}", f"arm_{i}") for i in range(n_arms)]
            candidates.append(graph_from_points(points, edges))
    return candidates


def figure8_candidates(positions: NDArray[np.float64]) -> list[Any]:
    """Starting layouts for a figure-8: a rectangle with a bar across its middle.

    Nodes ``corner_0`` … ``corner_3`` and the bar's ends ``bar_0``, ``bar_1``;
    the bar is parallel to either pair of sides.
    """
    from shapely.geometry import MultiPoint

    samples = tracked(positions)
    rect = MultiPoint(samples).minimum_rotated_rectangle
    c = np.asarray(rect.exterior.coords)[:4]
    candidates = []
    for k in range(2):
        c0, c1, c2, c3 = (c[(k + j) % 4] for j in range(4))
        # Sides c0-c1 and c2-c3 are cut in half by the bar from bar_0 to bar_1.
        points = {
            "corner_0": c0,
            "corner_1": c1,
            "corner_2": c2,
            "corner_3": c3,
            "bar_0": 0.5 * (c0 + c3),
            "bar_1": 0.5 * (c1 + c2),
        }
        edges = [
            ("corner_0", "corner_1"),
            ("corner_1", "bar_1"),
            ("bar_1", "corner_2"),
            ("corner_2", "corner_3"),
            ("corner_3", "bar_0"),
            ("bar_0", "corner_0"),
            ("bar_0", "bar_1"),
        ]
        candidates.append(graph_from_points(points, edges))
    return candidates


def star_candidates(
    positions: NDArray[np.float64], n_arms: int, centre: NDArray[np.float64]
) -> list[Any]:
    """A starting star layout: ``centre`` joined to ``n_arms`` arm ends.

    Each arm end is the samples' 99th percentile distance from the centre in
    one of ``n_arms`` angular sectors, found as the peaks of the samples'
    angular histogram around the centre (sectors centred on the peaks).
    Nodes ``centre`` and ``arm_0`` … ``arm_{n-1}`` in angular order.
    """
    samples = tracked(positions)
    offset = samples - centre
    radius = np.linalg.norm(offset, axis=1)
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    outer = radius > np.quantile(radius, 0.5)
    hist, edges = np.histogram(angle[outer], bins=360, range=(-np.pi, np.pi))
    smooth = np.convolve(np.tile(hist, 3), np.ones(9) / 9, mode="same")[360:720]
    peaks: list[float] = []
    for index in np.argsort(smooth)[::-1]:
        a = 0.5 * (edges[index] + edges[index + 1])
        if all(abs(np.angle(np.exp(1j * (a - p)))) > np.pi / n_arms for p in peaks):
            peaks.append(a)
        if len(peaks) == n_arms:
            break
    peaks.sort()
    points: dict[str, NDArray[np.float64]] = {"centre": centre}
    for k, a in enumerate(peaks):
        sector = np.abs(np.angle(np.exp(1j * (angle - a)))) < np.pi / (2 * n_arms)
        r = np.quantile(radius[sector], 0.99)
        points[f"arm_{k}"] = centre + r * np.array([np.cos(a), np.sin(a)])
    return [graph_from_points(points, [("centre", f"arm_{k}") for k in range(n_arms)])]


def z_track_candidates(positions: NDArray[np.float64]) -> list[Any]:
    """Starting layouts for a Z-track: two parallel sections joined by a diagonal.

    The parallel sections lie along either pair of opposite sides of the
    samples' minimum rotated rectangle and the diagonal joins them along
    either diagonal: four candidates. Nodes ``end_0``, ``turn_0``, ``turn_1``,
    ``end_1`` along the track.
    """
    from shapely.geometry import MultiPoint

    samples = tracked(positions)
    rect = MultiPoint(samples).minimum_rotated_rectangle
    c = np.asarray(rect.exterior.coords)[:4]
    candidates = []
    for k in range(2):
        c0, c1, c2, c3 = (c[(k + j) % 4] for j in range(4))
        # Parallel sections c0-c1 and c3-c2; the diagonal is c1-c3 or c0-c2.
        for path in ((c0, c1, c3, c2), (c1, c0, c2, c3)):
            points = dict(
                zip(("end_0", "turn_0", "turn_1", "end_1"), path, strict=True)
            )
            edges = [("end_0", "turn_0"), ("turn_0", "turn_1"), ("turn_1", "end_1")]
            candidates.append(graph_from_points(points, edges))
    return candidates


def radial_arms(
    star: Any, positions: NDArray[np.float64], *, min_radius: float, window: float
) -> Any:
    """A radial maze's arms as lines through their own samples, joined at the centre.

    Each arm of the fitted ``star`` is refitted as the principal axis of the
    samples nearest it that lie more than ``min_radius`` from the centre and
    within ``window`` of the arm: its end is their 99th percentile along the
    axis and its base the point of the axis nearest the centre, so an arm
    need not point exactly at the centre (the arms leave the edge of a
    platform). Nodes ``centre``, ``base_k`` and ``arm_k``.
    """
    samples = tracked(positions)
    centre = np.asarray(star.nodes["centre"]["pos"], dtype=float)
    segment, lateral, _ = edge_offsets(star, samples)
    radius = np.linalg.norm(samples - centre, axis=1)
    points: dict[str, NDArray[np.float64]] = {"centre": centre}
    edges = []
    for k in range(star.number_of_edges()):
        mine = (segment == k) & (radius > min_radius) & (np.abs(lateral) < window)
        arm_samples = samples[mine]
        mean = arm_samples.mean(axis=0)
        axis = np.linalg.svd(arm_samples - mean, full_matrices=False)[2][0]
        if axis @ (mean - centre) < 0:
            axis = -axis
        along = (arm_samples - mean) @ axis
        points[f"base_{k}"] = mean + ((centre - mean) @ axis) * axis
        points[f"arm_{k}"] = mean + np.quantile(along, 0.99) * axis
        edges += [("centre", f"base_{k}"), (f"base_{k}", f"arm_{k}")]
    return graph_from_points(points, edges)


def x_maze_candidates(positions: NDArray[np.float64]) -> list[Any]:
    """Starting layouts for an X maze: two junctions joined by a centre bar, two arms each.

    The bar runs along either principal axis of the samples through their
    centre; each junction is joined to the two arm ends on its side, the
    samples' farthest points (99th percentile distance) in the two quadrants
    there. Nodes ``junction_0``, ``junction_1``, ``arm_0`` … ``arm_3``.
    """
    samples = tracked(positions)
    centre = 0.5 * (samples.min(axis=0) + samples.max(axis=0))
    _, axes = principal_frame(samples)
    candidates = []
    for k in range(2):
        u, v = axes[k], axes[1 - k]
        local = np.column_stack([(samples - centre) @ u, (samples - centre) @ v])
        points: dict[str, NDArray[np.float64]] = {}
        edges = [("junction_0", "junction_1")]
        span = np.ptp(local[:, 0])
        for j, side in enumerate((-1.0, 1.0)):
            points[f"junction_{j}"] = centre + side * 0.05 * span * u
            for i, sign in enumerate((-1.0, 1.0)):
                quadrant = (np.sign(local[:, 0]) == side) & (
                    np.sign(local[:, 1]) == sign
                )
                radius = np.linalg.norm(local[quadrant], axis=1)
                far = local[quadrant][radius >= np.quantile(radius, 0.99)].mean(axis=0)
                name = f"arm_{2 * j + i}"
                points[name] = centre + far[0] * u + far[1] * v
                edges.append((f"junction_{j}", name))
        candidates.append(graph_from_points(points, edges))
    return candidates


def loop_centreline(
    positions: NDArray[np.float64], n_points: int = 72
) -> NDArray[np.float64]:
    """A closed centreline: the median sample in each angular sector around the centre.

    The centre is the middle of the samples' bounding box; each of
    ``n_points`` equal sectors contributes the point at its median angle and
    median radius. For a loop whose every ray from the centre crosses the
    track once (a circle, an oval, a rounded rectangle).
    """
    samples = tracked(positions)
    centre = 0.5 * (samples.min(axis=0) + samples.max(axis=0))
    offset = samples - centre
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    radius = np.linalg.norm(offset, axis=1)
    sector = np.floor((angle + np.pi) / (2 * np.pi) * n_points).astype(int) % n_points
    points = []
    for k in range(n_points):
        inside = sector == k
        if inside.sum() == 0:
            msg = f"no sample in sector {k} of {n_points}: not a loop around its centre"
            raise ValueError(msg)
        a, r = np.median(angle[inside]), np.median(radius[inside])
        points.append(centre + r * np.array([np.cos(a), np.sin(a)]))
    return np.asarray(points)


def simplify_polyline(
    points: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64]:
    """Douglas-Peucker simplification of an open polyline (shapely)."""
    from shapely.geometry import LineString

    return np.asarray(LineString(points).simplify(tolerance).coords)


def node_table(graph: Any, digits: int = 1) -> str:
    """Node positions as ``name (x, y)`` text for ``conversions``."""
    return "; ".join(
        f"{name} ({', '.join(f'{v:.{digits}f}' for v in np.atleast_1d(pos))})"
        for name, pos in graph.nodes(data="pos")
    )


def edge_lengths(graph: Any, digits: int = 1) -> str:
    """Edge lengths as ``u-v L`` text for ``conversions``."""
    return "; ".join(
        f"{u}-{v} {d:.{digits}f}" for u, v, d in graph.edges(data="distance")
    )


def declare(
    graph: Any,
    *,
    bin_size: float,
    track_width: float,
    units: str | None,
    name: str,
    regions: Mapping[str, Any] | None = None,
) -> Callable[[], Any]:
    """A zero-argument builder of the track environment; it caches what it builds."""
    from _geometry import graph_environment

    built: list[Any] = []

    def environment() -> Any:
        if not built:
            env = graph_environment(
                graph,
                bin_size=bin_size,
                track_width=track_width,
                units=units,
                name=name,
            )
            built.append(env.with_regions(regions) if regions else env)
        return built[0]

    return environment


MEASURED_WIDTH_QUANTILE = 0.99
# Where a declared track's placement comes from; only FITTED ones are refused.
PLACED_BY = (
    "fitted to the samples",
    "samples' principal axis",
    "authors' graph",
    "authors' centrelines",
    "file's linearized coordinate",
    "file's coordinates",
)
FITTED = ("fitted to the samples", "samples' principal axis")


def misfit_reasons(fit: Mapping[str, float], bin_size: float) -> list[str]:
    """Evidence that a fitted layout does not match its samples (``edge_fit``)."""
    from _geometry import MAX_SIZE_RESIDUAL

    reasons = []
    if fit["max_edge_bias"] > MAX_EDGE_BIAS_BINS * bin_size:
        reasons.append(
            f"a segment's samples lie {fit['max_edge_bias']:.3g} to one side of it"
            f" (limit {MAX_EDGE_BIAS_BINS:g} bin, {MAX_EDGE_BIAS_BINS * bin_size:.3g})"
        )
    if fit["max_unsupported"] > MAX_UNSUPPORTED:
        reasons.append(
            f"{fit['max_unsupported']:.0%} of a segment has no sample within a bin"
            f" (limit {MAX_UNSUPPORTED:.0%})"
        )
    for key, value in fit.items():
        if key.startswith("size_residual") and abs(value) > MAX_SIZE_RESIDUAL:
            reasons.append(
                f"{key.removeprefix('size_residual_')} {value:+.0%} of the stated"
                f" length (limit {MAX_SIZE_RESIDUAL:.0%})"
            )
    return reasons


def declared_geometry(
    graph: Any,
    positions: NDArray[np.float64],
    *,
    times: NDArray[np.float64],
    kind: str,
    source: str,
    bin_size: float,
    physical_width: float | None,
    units: str | None,
    name: str,
    placement: str,
    placed_by: str,
    tracked_point: str,
    width_kind: str = "stated",
    width_note: str = "",
    speed_thresholds: Sequence[float] = (),
    stated_lengths: Mapping[str, tuple[float, float]] | None = None,
    regions: Mapping[str, Any] | None = None,
    width_quantile: float = MEASURED_WIDTH_QUANTILE,
) -> tuple[dict[str, Any], str]:
    """Geometry entries declaring the track, and its ``conversions`` line.

    Two widths are kept apart (G-152). ``physical_width`` is the apparatus's
    width, ``width_kind`` ``"stated"`` by a source read or ``"chosen"`` by
    the loader (``width_note`` says why), or None when no source states it.
    The environment is built with a placement tolerance instead, measured
    from the samples: twice their ``width_quantile`` distance from the graph,
    and at least the physical width and one bin, so that about
    ``1 - width_quantile`` of the samples are dropped by construction. The
    tracked point (``tracked_point``, recorded as ``geometry["tracked_point"]``)
    is usually a head-mounted LED above an unwalled track, which swings past
    its edge; ``track_width`` only decides which positions are placed and the
    footprint (``TrackArena``), not the bins, their measure, connectivity or
    any rate, so the tolerance changes no analysis of the samples it places.
    The share of samples beyond the physical width is reported, split by
    speed (``speed_thresholds``, the positions' units per second), as a
    tracking measure.

    ``placed_by`` (one of ``PLACED_BY``) says where the placement comes from.
    Only a layout ``FITTED`` to the samples is refused, on misfit evidence
    (``misfit_reasons``): a stretch of track drawn beside its samples, a
    segment with no samples near it, or a length beyond
    ``_geometry.MAX_SIZE_RESIDUAL`` of a stated one (``stated_lengths``:
    label -> (fitted, stated)). Scatter across the track is not misfit. The
    authors' geometry and the file's own coordinates are not fitted, so they
    are not refused here; the fidelity study judges them.

    Returns the entries (``tracked_point``, ``declared_physical_width`` and
    its kind always; when declared also ``environment``, ``declared_graph``,
    ``declared_track_width``, the tolerance, ``declared_from`` and
    ``environment_fit``) and a line naming the layout, its source, the
    widths, the placement and the fit; when the fit is refused, a line saying
    why the recording stays inferred.
    """
    from _geometry import share_by_speed

    if placed_by not in PLACED_BY:
        msg = f"placed_by must be one of {PLACED_BY}, not {placed_by!r}"
        raise ValueError(msg)
    label = units or "units"
    _, nodes, edges = graph_arrays(graph)
    samples = tracked(positions)
    offsets = segment_distance(samples, nodes, edges)
    measured = 2 * float(np.quantile(offsets, width_quantile))
    tolerance = max(measured, physical_width or 0.0, bin_size)
    fit = edge_fit(graph, positions, tolerance=bin_size)
    fit["max_edge_bias_limit"] = MAX_EDGE_BIAS_BINS * bin_size
    for key, (fitted, stated) in (stated_lengths or {}).items():
        fit[f"size_residual_{key}"] = fitted / stated - 1
    fit["share_outside_tolerance"] = float(np.mean(offsets > tolerance / 2))
    entries: dict[str, Any] = {
        "tracked_point": tracked_point,
        "declared_physical_width": physical_width,
        "declared_physical_width_kind": width_kind
        if physical_width is not None
        else None,
    }
    if physical_width is not None:
        by_speed = share_by_speed(
            times, positions, offsets > physical_width / 2, speed_thresholds
        )
        fit["outside_physical_width"] = by_speed["all"]
        fit["outside_physical_width_by_speed"] = by_speed
        physical = (
            f"physical width {physical_width:g} {label} ({width_kind}"
            + (f": {width_note}" if width_note else "")
            + f"), {by_speed['all']:.1%} of samples beyond half of it"
            + (
                " ("
                + ", ".join(
                    f"{v:.0%} at {k} {label}/s"
                    for k, v in by_speed.items()
                    if k != "all"
                )
                + ")"
                if len(by_speed) > 1
                else ""
            )
        )
    else:
        physical = "physical width not stated"
    reasons = misfit_reasons(fit, bin_size) if placed_by in FITTED else []
    if reasons:
        if physical_width is not None:
            entries["outside_physical_width_by_speed"] = fit[
                "outside_physical_width_by_speed"
            ]
        return entries, (
            f"environment: inferred; the {kind} fit ({source}; {placement}) was"
            f" refused: {'; '.join(reasons)}; tracked point: {tracked_point};"
            f" on the refused layout, {physical}"
        )
    entries |= {
        "environment": declare(
            graph,
            bin_size=bin_size,
            track_width=tolerance,
            units=units,
            name=name,
            regions=regions,
        ),
        "declared_graph": graph,
        "declared_track_width": tolerance,
        "declared_track_width_kind": "placement tolerance",
        "declared_from": placed_by,
        "declared_speed_thresholds": tuple(speed_thresholds),
        "environment_fit": fit,
    }
    lengths = "".join(
        f", {key.removeprefix('size_residual_')} {value:+.0%} of the stated"
        for key, value in fit.items()
        if key.startswith("size_residual")
    )
    return entries, (
        f"environment: declared {kind} ({source}); tracked point: {tracked_point};"
        f" {physical}; placement tolerance {tolerance:.3g} {label} (track_width: twice"
        f" the samples' {width_quantile:.0%} quantile distance from the track,"
        " at least the physical width and one bin; a tracked head swings past an"
        f" unwalled track's edge, so {fit['share_outside_tolerance']:.1%} of samples"
        " are dropped by construction); placement:"
        f" {placement} ({placed_by}); fit: median |lateral offset|"
        f" {fit['median_abs_lateral']:.2f} {label}, worst stretch bias"
        f" {fit['max_edge_bias']:.2f} {label} (limit {fit['max_edge_bias_limit']:.3g}),"
        f" worst unsupported share of a segment {fit['max_unsupported']:.0%}{lengths};"
        f" track length {graph_residual(graph, positions, None)['track_length']:.1f}"
        f" {label}"
    )


def straight_placement(fit: Mapping[str, float], units: str) -> str:
    """``straight_track``'s placement for ``conversions``."""
    angle = (
        f" at {np.degrees(fit['axis_angle_rad']):.1f} deg"
        if "axis_angle_rad" in fit
        else ""
    )
    return (
        f"along the samples' principal axis{angle}, centred between their 0.5th and"
        f" 99.5th percentiles along it (span {fit['samples_span']:.1f} {units});"
        f" {fit['share_beyond_ends']:.1%} of samples lie beyond the ends"
    )


def with_declared(rec: Any, entries: Mapping[str, Any], line: str) -> Any:
    """``rec`` with the declared-geometry ``entries`` and their ``conversions`` line."""
    from dataclasses import replace

    return replace(
        rec,
        geometry={**rec.geometry, **entries},
        conversions=(*rec.conversions, line),
    )


def fit_circle_centre(
    positions: NDArray[np.float64], *, scale: float
) -> tuple[NDArray[np.float64], float]:
    """Centre and radius of a circle through the samples (robust least squares).

    The residual is each sample's distance from the centre minus the radius,
    with a soft-L1 loss of scale ``scale`` so samples off the ring count less.
    """
    from scipy.optimize import least_squares

    samples = tracked(positions)
    start = 0.5 * (samples.min(axis=0) + samples.max(axis=0))
    radius = float(np.median(np.linalg.norm(samples - start, axis=1)))
    fit = least_squares(
        lambda x: np.linalg.norm(samples - x[:2], axis=1) - x[2],
        [*start, radius],
        loss="soft_l1",
        f_scale=scale,
    )
    return np.asarray(fit.x[:2]), float(fit.x[2])


def circle_points(
    centre: NDArray[np.float64], radius: float, n_points: int
) -> NDArray[np.float64]:
    """``n_points`` vertices of a circle, counter-clockwise from east."""
    angle = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    return centre + radius * np.column_stack([np.cos(angle), np.sin(angle)])


def cut_region(
    name: str,
    point: Sequence[float],
    along: Sequence[float],
    *,
    half_length: float,
    half_width: float,
) -> Any:
    """A rectangular ``Region`` across a track at ``point``.

    ``along`` is the track's direction there; the rectangle is ``2 *
    half_length`` along it and ``2 * half_width`` across it, so a lap must
    pass through it.
    """
    from shapely.geometry import Polygon

    from neurospatial import Region

    u = np.asarray(along, dtype=float)
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    p = np.asarray(point, dtype=float)
    corners = [
        p + su * half_length * u + sv * half_width * v
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return Region(
        name=name,
        kind="polygon",
        data=Polygon(corners),
        metadata={"role": "lap_start"},
    )


def arms_text(graph: Any, units: str) -> str:
    """Fitted base and arm lengths of a parallel-arm (W or E) graph."""
    base = sum(
        d for u, v, d in graph.edges(data="distance") if u[:4] == v[:4] == "base"
    )
    arms = [d for u, v, d in graph.edges(data="distance") if "arm" in u + v]
    return (
        f"fitted base {base:.1f} {units}, arms {min(arms):.1f}-{max(arms):.1f} {units}"
    )


def fitted_maze(
    candidates: Iterable[Any], positions: NDArray[np.float64], *, cap: float
) -> Any:
    """The best-fitting candidate layout (``best_fit``), its loss dropped."""
    return best_fit(candidates, positions, cap=cap)[0]


def densest_point(
    positions: NDArray[np.float64], bins: int = 60
) -> NDArray[np.float64]:
    """Centre of the most occupied cell of a ``bins`` x ``bins`` histogram of the samples."""
    samples = tracked(positions)
    counts, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins)
    i, j = np.unravel_index(np.argmax(counts), counts.shape)
    return np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])


def parallel_arm_lengths(graph: Any) -> tuple[float, float]:
    """A parallel-arm (W or E) graph's base length and mean arm length."""
    base = sum(
        d for u, v, d in graph.edges(data="distance") if u[:4] == v[:4] == "base"
    )
    arms = [d for u, v, d in graph.edges(data="distance") if "arm" in u + v]
    return float(base), float(np.mean(arms))
