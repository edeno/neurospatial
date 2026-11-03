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

    The base runs along one side of the samples' minimum rotated rectangle and
    the arms leave it at equal spacing; each of the four sides is a candidate.
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
        points: dict[str, NDArray[np.float64]] = {}
        for i, f in enumerate(np.linspace(0.1, 0.9, n_arms)):
            base = b0 + f * (b1 - b0) + 0.05 * out
            points[f"base_{i}"] = base
            points[f"arm_{i}"] = base + 0.9 * out
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
    positions: NDArray[np.float64],
    *,
    bin_size: float,
    track_width: float | None,
    units: str | None,
    name: str,
    max_median_offset: float,
    regions: Mapping[str, Any] | None = None,
) -> tuple[Callable[[], Any] | None, dict[str, float]]:
    """The declared environment's builder and the fit residual, or None if refused.

    The fit is refused when the samples' median distance to the graph is above
    ``max_median_offset``. The builder caches the environment it returns.
    """
    from _geometry import graph_environment

    residual = graph_residual(graph, positions, track_width)
    if residual["median_offset"] > max_median_offset:
        return None, residual
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

    return environment, residual


def residual_text(residual: Mapping[str, float], units: str) -> str:
    """The fit residual for ``conversions``."""
    text = (
        f"samples a median {residual['median_offset']:.2f} {units} from the track"
        f" (95th percentile {residual['p95_offset']:.2f}), track length"
        f" {residual['track_length']:.1f} {units}"
    )
    if "share_within_half_width" in residual:
        text += (
            f", {residual['share_within_half_width']:.1%} of samples within half"
            " the track width"
        )
    return text


MEASURED_WIDTH_QUANTILE = 0.99


def declared_geometry(
    graph: Any,
    positions: NDArray[np.float64],
    *,
    kind: str,
    source: str,
    bin_size: float,
    track_width: float | None,
    units: str | None,
    name: str,
    placement: str,
    max_median_offset: float | None = None,
    regions: Mapping[str, Any] | None = None,
    width_quantile: float = MEASURED_WIDTH_QUANTILE,
) -> tuple[dict[str, Any], str]:
    """Geometry entries declaring the track, and its ``conversions`` line.

    ``track_width`` is the stated full width. When no source states one, the
    width is measured: twice the samples' ``width_quantile`` (default
    ``MEASURED_WIDTH_QUANTILE``) distance from the graph, so that share of samples lies on the track
    (neurospatial refuses to place positions on a track without a width),
    and at least one bin (positions on a 1-D track lie on it exactly).

    Returns ``{"environment": builder, "declared_graph": graph}`` and a line
    naming the layout, its source, how it was placed and the fit residual; or,
    when the fit is refused (median sample distance above
    ``max_median_offset``, default half the stated width or else two bins),
    no entries and a line saying why the recording stays inferred.
    """
    label = units or "units"
    if track_width is None:
        _, nodes, edges = graph_arrays(graph)
        offsets = segment_distance(tracked(positions), nodes, edges)
        measured = 2 * float(np.quantile(offsets, width_quantile))
        width = max(measured, bin_size)
        width_text = (
            f"track width not stated; measured {width:.3g} {label}, twice the"
            f" samples' {width_quantile:.0%} quantile distance from the track"
            + ("" if measured >= bin_size else ", raised to one bin")
        )
    else:
        width = track_width
        width_text = f"track width {track_width:g} {label} (stated)"
    bound = (
        max_median_offset
        if max_median_offset is not None
        else (track_width / 2 if track_width is not None else 2 * bin_size)
    )
    builder, residual = declare(
        graph,
        positions,
        bin_size=bin_size,
        track_width=width,
        units=units,
        name=name,
        max_median_offset=bound,
        regions=regions,
    )
    if builder is None:
        return {}, (
            f"environment: inferred; the {kind} fit ({source}) was refused: samples a"
            f" median {residual['median_offset']:.2f} {label} from it, above the"
            f" {bound:g} allowed"
        )
    entries = {
        "environment": builder,
        "declared_graph": graph,
        "declared_track_width": width,
        "declared_track_width_stated": track_width is not None,
    }
    return entries, (
        f"environment: declared {kind} ({source}); {width_text}; placement:"
        f" {placement}; fit: {residual_text(residual, label)}"
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
