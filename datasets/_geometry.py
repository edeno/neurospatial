"""Build declared environments from a source's stated apparatus.

A loader that knows the apparatus (an arena's outline and size, a track graph,
a lattice of bars, a room) declares it in ``rec.geometry["environment"]`` as a
zero-argument callable returning the ``Environment``. The samples only place a
stated shape: centre and rotation are fitted, size comes from the source. The
fit's residual goes into the loader's ``conversions``.

``Environment.from_graph`` keeps only the edges listed in ``edge_order`` and
drops the rest without a word, while its track footprint uses every edge, so a
depth-first order on a graph with a cycle loses edges and misplaces positions
on them. ``graph_edge_order`` lists every edge.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# A stated size more than this share away from the samples' own fitted size
# is refused: the stated apparatus, the unit or the session is then wrong.
# One tolerance for every size (lengths, diameters, sides, the samples'
# reach past a track's end or short of a wall), shared with the arena
# loaders: it allows "about" in stated sizes ("~80 x 80 cm") and a head that
# leans past a track's end, and is not tuned to any recording.
MAX_SIZE_RESIDUAL = 0.15


def graph_edge_order(graph: Any, root: Any = None) -> list[tuple[Any, Any]]:
    """Every edge of ``graph``: a depth-first walk from ``root``, then the rest.

    The walk lays a tree out branch by branch; the remaining edges are those
    that close a cycle (a loop's closing edge, a lattice's cross bars).
    """
    import networkx as nx

    start = next(iter(graph.nodes)) if root is None else root
    order = list(nx.dfs_edges(graph, source=start))
    seen = {frozenset(edge) for edge in order}
    order += [edge for edge in graph.edges if frozenset(edge) not in seen]
    return order


def graph_environment(
    graph: Any,
    *,
    bin_size: float,
    track_width: float | None,
    units: str | None,
    name: str,
    root: Any = None,
) -> Any:
    """``Environment.from_graph`` with every edge of the graph kept."""
    from neurospatial import Environment

    order = graph_edge_order(graph, root)
    return Environment.from_graph(
        graph,
        order,
        bin_size=bin_size,
        units=units,
        name=name,
        track_width=track_width,
    )


def _tracked(positions: NDArray[np.float64], trim: float) -> NDArray[np.float64]:
    tracked = positions[np.isfinite(positions).all(axis=1)]
    if trim <= 0:
        return tracked
    lo, hi = np.percentile(tracked, [100 * trim, 100 * (1 - trim)], axis=0)
    keep = ((tracked >= lo) & (tracked <= hi)).all(axis=1)
    return tracked[keep]


def fit_rectangle(
    positions: NDArray[np.float64],
    width: float,
    height: float,
    *,
    trim: float = 0.001,
) -> tuple[Any, dict[str, float]]:
    """Place a ``width`` × ``height`` rectangle on the samples.

    Centre and rotation come from the minimum rotated rectangle around the
    samples (after trimming ``trim`` of each tail per axis); the size is the
    stated one. Returns the polygon and the fit: centre, rotation (rad, of the
    side matched to ``width``), and the samples' own rectangle sides.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
    width, height : float
        Stated sides, in the positions' units.
    trim : float
        Fraction of each tail dropped per axis before fitting.
    """
    from shapely import affinity
    from shapely.geometry import MultiPoint, box

    points = _tracked(positions, trim)
    fitted = MultiPoint(points).minimum_rotated_rectangle
    corners = np.asarray(fitted.exterior.coords)[:4]
    sides = np.diff(np.vstack([corners, corners[:1]]), axis=0)[:2]
    lengths = np.linalg.norm(sides, axis=1)
    # Match the stated width to the fitted side closer to it in length.
    pairs = (
        abs(lengths[0] - width) + abs(lengths[1] - height),
        abs(lengths[1] - width) + abs(lengths[0] - height),
    )
    along = sides[0] if pairs[0] <= pairs[1] else sides[1]
    angle = float(np.arctan2(along[1], along[0]))
    centre = np.asarray(fitted.centroid.coords[0])
    rect = box(-width / 2, -height / 2, width / 2, height / 2)
    rect = affinity.rotate(rect, angle, origin=(0, 0), use_radians=True)
    rect = affinity.translate(rect, *centre)
    fit = {
        "centre_x": float(centre[0]),
        "centre_y": float(centre[1]),
        "rotation_rad": angle,
        "samples_side_a": float(max(lengths)),
        "samples_side_b": float(min(lengths)),
    }
    return rect, fit


def fit_circle(
    positions: NDArray[np.float64],
    diameter: float,
    *,
    trim: float = 0.001,
    resolution: int = 64,
) -> tuple[Any, dict[str, float]]:
    """Place a circle of the stated ``diameter`` on the samples.

    The centre is a least-squares circle fit to the convex hull of the samples
    (after trimming ``trim`` of each tail per axis); the radius is the stated
    one. Returns the polygon and the fit: centre and the samples' own fitted
    diameter.
    """
    from shapely.geometry import MultiPoint, Point

    points = _tracked(positions, trim)
    hull = np.asarray(MultiPoint(points).convex_hull.exterior.coords)[:-1]
    # Kasa fit: x² + y² = 2 a x + 2 b y + c.
    design = np.column_stack([2 * hull, np.ones(len(hull))])
    target = (hull**2).sum(axis=1)
    (a, b, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    radius = float(np.sqrt(c + a**2 + b**2))
    circle = Point(a, b).buffer(diameter / 2, quad_segs=resolution // 4)
    return circle, {
        "centre_x": float(a),
        "centre_y": float(b),
        "samples_diameter": 2 * radius,
    }


def placement_line(kind: str, stated: str, fit: dict[str, float], source: str) -> str:
    """One ``conversions`` line describing a fitted placement of a stated shape."""
    values = ", ".join(f"{key} {value:.3g}" for key, value in fit.items())
    return (
        f"environment: declared {kind} ({stated}, {source}); placement fitted: {values}"
    )


def sample_speeds(
    times: NDArray[np.float64], positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Speed at each sample, the mean of the steps to and from it (NaN where untracked).

    Units are the positions' per second.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(len(times), -1)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    padded = np.concatenate([[np.nan], step, [np.nan]])
    pair = np.vstack([padded[:-1], padded[1:]])
    finite = np.isfinite(pair)
    count = finite.sum(axis=0)
    total = np.where(finite, pair, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def share_by_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    flag: NDArray[np.bool_],
    thresholds: Sequence[float],
) -> dict[str, float]:
    """Share of tracked samples with ``flag`` in each speed band.

    ``flag`` has one entry per tracked sample (finite position); the bands
    are below ``thresholds[0]``, between consecutive thresholds and above
    the last, in the positions' units per second. Keys are ``"all"`` and
    ``"<lo>-<hi>"``; a band with no sample is left out.
    """
    tracked = (
        np.isfinite(np.asarray(positions, dtype=float))
        .reshape(len(times), -1)
        .all(axis=1)
    )
    speed = sample_speeds(times, positions)[tracked]
    flag = np.asarray(flag, dtype=bool)
    out = {"all": float(np.mean(flag))}
    edges = [0.0, *thresholds, np.inf]
    for low, high in itertools.pairwise(edges):
        band = (speed >= low) & (speed < high)
        if band.any():
            out[f"{low:g}-{high:g}"] = float(np.mean(flag[band]))
    return out


# A declared environment's misfit, judged by kind (``judge_declared``): its
# samples consistently to one side of a track stretch or beyond a wall, by
# more than this many bins.
MAX_SIDE_BIAS_BINS = 1.0
OUTER_QUANTILE = 0.95


def judge_declared(
    env: Any,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    physical_width: float | None = None,
    speed_thresholds: Sequence[float] = (),
    stated_residuals: Mapping[str, float] | None = None,
    bin_size: float | None = None,
) -> dict[str, Any]:
    """One verdict for any declared environment: misfit evidence, and tracking.

    Misfit is judged by the environment's kind:

    - a track graph (``TrackArena``): the worst straight stretch's median
      signed lateral offset (over ``_tracks.MIN_PIECE_BINS`` bins, limit one
      bin) and segments with no sample within a bin over more than
      ``_tracks.MAX_UNSUPPORTED`` of their length (``_tracks.edge_fit``);
    - an outline (a polygonal domain): per side, how far the samples nearest
      it reach past it (their ``OUTER_QUANTILE`` signed distance, positive
      beyond the wall; limit ``MAX_SIDE_BIAS_BINS`` bins: an outline moved off
      its samples has them run past one wall), the samples' extent across the
      outline (they may fall short of a wall, but not by more than
      ``MAX_SIZE_RESIDUAL`` of the side: an outline drawn too large), and the
      share of each side with no sample within three bins (reported);
    - a track, also how far its samples reach past its free ends
      (``max_end_overshoot``, limit ``MAX_SIZE_RESIDUAL`` of the segment: a
      track drawn too short);
    - either: a stated length or size off by more than ``MAX_SIZE_RESIDUAL``
      (``stated_residuals``: label -> fitted / stated - 1).

    The share of tracked samples beyond the physical apparatus (outside half
    of ``physical_width`` on a track, outside the outline) is reported as a
    tracking measure, not judged: a head-mounted LED above an unwalled track
    swings past its edge, and a 2-D camera projects a raised head outward.
    It is split by speed (``speed_thresholds``, the positions' units per
    second) to show whether it comes from stopped posture or movement.

    ``bin_size`` is the nominal bin the environment was built with (its
    bins along a graph's edges can be shorter); by default the median bin's
    length, or its measure's root for a grid or outline.

    Returns ``{"kind", "misfit": [reasons], "fit": {...}, "outside": share or
    None, "outside_by_speed": {...}}``.
    """
    from _tracks import edge_fit, misfit_reasons, segment_distance

    positions = np.asarray(positions, dtype=float)
    tracked = np.isfinite(positions).all(axis=1)
    samples = positions[tracked]
    arena = env.arena
    is_track = type(arena).__name__ == "TrackArena"
    # A track's bin measure is a length; a grid's or outline's an area or volume.
    if bin_size is None:
        measure = float(np.median(np.asarray(env.bin_sizes)))
        bin_size = measure if is_track else measure ** (1 / env.n_dims)
    out: dict[str, Any] = {
        "misfit": [],
        "fit": {},
        "outside": None,
        "outside_by_speed": {},
    }
    beyond = None
    if is_track:
        out["kind"] = "track"
        segments = np.asarray(arena.as_track().segments, dtype=float)
        graph = _segments_graph(segments)
        fit = edge_fit(graph, samples, tolerance=bin_size)
        fit["max_edge_bias_limit"] = bin_size
        for key, value in (stated_residuals or {}).items():
            fit[f"size_residual_{key}"] = value
        out["fit"] = fit
        out["misfit"] = misfit_reasons(fit, bin_size)
        if fit["max_end_overshoot"] > MAX_SIZE_RESIDUAL:
            out["misfit"].append(
                f"samples reach {fit['max_end_overshoot']:.0%} of a segment's length"
                f" past its free ends (limit {MAX_SIZE_RESIDUAL:.0%})"
            )
        if physical_width is not None:
            nodes = segments.reshape(-1, segments.shape[2])
            edges = np.arange(len(nodes)).reshape(-1, 2)
            beyond = segment_distance(samples, nodes, edges) > physical_width / 2
    else:
        polygon = (
            arena.polygonal_domain() if hasattr(arena, "polygonal_domain") else None
        )
        if polygon is None:
            out["kind"] = "unknown"
        else:
            import shapely
            from scipy.spatial import cKDTree

            out["kind"] = "outline"
            ring = np.asarray(polygon.exterior.coords)
            points = shapely.points(samples)
            beyond = ~shapely.contains_xy(polygon, samples[:, 0], samples[:, 1])
            sides = [shapely.LineString([a, b]) for a, b in itertools.pairwise(ring)]
            dist = np.column_stack([shapely.distance(side, points) for side in sides])
            nearest = np.argmin(dist, axis=1)
            signed = np.where(beyond, 1.0, -1.0) * dist[np.arange(len(dist)), nearest]
            biases, unsupported = [], []
            for k, (a, b) in enumerate(itertools.pairwise(ring)):
                mine = nearest == k
                if mine.sum() >= 10:
                    # How far the outermost samples reach past this wall.
                    biases.append(float(np.quantile(signed[mine], OUTER_QUANTILE)))
                along = np.linspace(0, 1, 25)[:, None] * (b - a) + a
                unsupported.append(
                    float(np.mean(cKDTree(samples).query(along)[0] > 3 * bin_size))
                )
            # Samples may fall short of a wall but not reach past it: the
            # samples' extent (0.5th to 99.5th percentile) along each axis of
            # the outline's minimum rotated rectangle, over its side.
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # GEOS on axis-aligned hulls
                rectangle = polygon.minimum_rotated_rectangle
            corners = np.asarray(rectangle.exterior.coords)[:3]
            shortfall = []
            for side in (corners[1] - corners[0], corners[2] - corners[1]):
                length = float(np.linalg.norm(side))
                along = samples @ (side / length)
                low, high = np.quantile(along, [0.005, 0.995])
                shortfall.append(float((high - low) / length - 1))
            fit = {
                "max_side_bias": max(biases) if biases else float("nan"),
                "max_side_bias_limit": MAX_SIDE_BIAS_BINS * bin_size,
                "max_side_unsupported": max(unsupported) if unsupported else 0.0,
                "size_residual_extent": min(shortfall),
            }
            for key, value in (stated_residuals or {}).items():
                fit[f"size_residual_{key}"] = value
            out["fit"] = fit
            if fit["size_residual_extent"] < -MAX_SIZE_RESIDUAL:
                out["misfit"].append(
                    f"samples span {fit['size_residual_extent']:+.0%} of the"
                    f" outline's side (limit -{MAX_SIZE_RESIDUAL:.0%})"
                )
            if fit["max_side_bias"] > fit["max_side_bias_limit"]:
                out["misfit"].append(
                    f"samples reach {fit['max_side_bias']:.3g} beyond one side of the"
                    f" outline (limit {fit['max_side_bias_limit']:.3g})"
                )
            for key, value in (stated_residuals or {}).items():
                if abs(value) > MAX_SIZE_RESIDUAL:
                    out["misfit"].append(f"{key} {value:+.0%} of the stated")
    if beyond is not None:
        out["outside"] = float(np.mean(beyond))
        out["outside_by_speed"] = share_by_speed(
            times, positions, beyond, speed_thresholds
        )
    return out


def _segments_graph(segments: NDArray[np.float64]) -> Any:
    """A graph of the segments, joining endpoints that coincide."""
    import networkx as nx

    graph = nx.Graph()
    names: dict[tuple[float, ...], int] = {}
    for a, b in segments:
        ends = []
        for point in (a, b):
            key = tuple(np.round(point, 9))
            if key not in names:
                names[key] = len(names)
                graph.add_node(names[key], pos=tuple(float(v) for v in point))
            ends.append(names[key])
        graph.add_edge(*ends)
    return graph
