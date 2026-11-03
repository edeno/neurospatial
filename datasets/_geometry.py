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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
from _record import Recording
from numpy.typing import NDArray

# A stated size more than this share away from the samples' own fitted size
# is refused: the stated apparatus, the unit or the session is then wrong.
# One tolerance for every size (lengths, diameters, sides, the samples'
# reach past a track's end or short of a wall), shared by the arena and
# track loaders: it allows "about" in stated sizes ("~80 x 80 cm") and a head
# that leans past a track's end, and is not tuned to any recording.
MAX_SIZE_RESIDUAL = 0.15
# A stated centre farther than this share of the radius from the samples'
# centre is refused: the positions are then in another frame or unit.
MAX_CENTRE_OFFSET = 0.15


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

    The centre is that of the samples' minimum bounding circle (after
    trimming ``trim`` of each tail per axis), as ``fit_rectangle`` places a
    rectangle by the samples' minimum rotated rectangle; the radius is the
    stated one. A least-squares circle through the convex hull's vertices
    would weight the hull's unevenly spaced vertices and leave more samples
    outside. Returns the polygon and the fit: centre and the samples' own
    diameter (their bounding circle's).
    """
    import shapely
    from shapely.geometry import MultiPoint, Point

    points = _tracked(positions, trim)
    bounding = shapely.minimum_bounding_circle(MultiPoint(points))
    centre = np.asarray(bounding.centroid.coords[0])
    radius = float(shapely.minimum_bounding_radius(MultiPoint(points)))
    circle = Point(*centre).buffer(diameter / 2, quad_segs=resolution // 4)
    return circle, {
        "centre_x": float(centre[0]),
        "centre_y": float(centre[1]),
        "samples_diameter": 2 * radius,
    }


def placement_line(kind: str, stated: str, fit: dict[str, float], source: str) -> str:
    """One ``conversions`` line describing a fitted placement of a stated shape."""
    values = ", ".join(f"{key} {value:.3g}" for key, value in fit.items())
    return (
        f"environment: declared {kind} ({stated}, {source}); placement fitted: {values}"
    )


def _check_residuals(
    residuals: dict[str, float], long_limits: dict[str, float], max_short: float
) -> None:
    """Refuse a stated size the samples do not fit.

    A negative residual (the samples' fitted size short of the stated one) is
    refused beyond ``max_short``: animals often keep off the walls. A positive
    one (samples reaching past the stated wall) is refused beyond its entry
    in ``long_limits``, one bin over the side: the grid resolves a wall only
    to half a bin at each end.
    """
    short = {k: v for k, v in residuals.items() if v < -max_short}
    long = {k: v for k, v in residuals.items() if v > long_limits[k]}
    if short or long:
        detail = ", ".join(f"{k} {v:+.1%}" for k, v in {**short, **long}.items())
        msg = (
            f"the samples' fitted size differs from the stated size ({detail}; limits"
            f" -{max_short:.0%} short, one bin long): check the stated size, the"
            " position unit, or which apparatus this session used"
        )
        raise ValueError(msg)


def declared_rectangle(
    positions: NDArray[np.float64],
    width: float,
    height: float,
    *,
    bin_size: float,
    units: str | None,
    name: str,
    source: str,
    outline: str = "wall",
    max_residual: float = MAX_SIZE_RESIDUAL,
) -> tuple[Callable[[], Any], str, dict[str, float]]:
    """A stated ``width`` x ``height`` arena placed on the samples.

    Centre and rotation are fitted (``fit_rectangle``); the size is the
    stated one. The residual of each side is the samples' fitted side over
    the stated side, minus 1 (negative when the animal stays off the walls).

    Returns
    -------
    environment : callable
        Zero-argument ``Environment.from_polygon`` on the placed rectangle at
        ``bin_size``; its outline is ``outline`` (``"wall"``, or ``"drop"``
        for an elevated platform, ``OpenArena(..., outline="drop")``).
    conversion : str
        The ``conversions`` line: shape, size, source, placement and residuals.
    fit : dict
        The placement and residuals, for ``geometry["environment_fit"]``.

    Raises
    ------
    ValueError
        When a side is more than ``max_residual`` short of the stated side, or
        longer than it by more than one bin (``_check_residuals``).
    """
    from neurospatial import Environment
    from neurospatial.arena import OpenArena

    polygon, fit = fit_rectangle(positions, width, height)
    stated = sorted((width, height), reverse=True)
    fit["size_residual_a"] = fit["samples_side_a"] / stated[0] - 1
    fit["size_residual_b"] = fit["samples_side_b"] / stated[1] - 1
    _check_residuals(
        {k: fit[k] for k in ("size_residual_a", "size_residual_b")},
        {
            "size_residual_a": bin_size / stated[0],
            "size_residual_b": bin_size / stated[1],
        },
        max_residual,
    )
    shape = "square" if width == height else "rectangle"
    arena = None if outline == "wall" else OpenArena(polygon, outline=outline)
    factory = partial(
        Environment.from_polygon,
        polygon,
        bin_size=bin_size,
        units=units,
        name=name,
        arena=arena,
    )
    stated = f"{width:g} x {height:g}" + (
        "" if outline == "wall" else f", {outline} edge"
    )
    line = placement_line(shape, stated, fit, source)
    return factory, line, fit


def declared_circle(
    positions: NDArray[np.float64],
    diameter: float,
    *,
    bin_size: float,
    units: str | None,
    name: str,
    source: str,
    centre: tuple[float, float] | None = None,
    max_residual: float = MAX_SIZE_RESIDUAL,
    max_centre_offset: float = MAX_CENTRE_OFFSET,
) -> tuple[Callable[[], Any], str, dict[str, float]]:
    """A stated circular arena of ``diameter``, centred on the samples or at ``centre``.

    Without ``centre`` the centre is fitted (``fit_circle``); with it (when the
    source states where the centre is) the fitted centre's distance from it
    is recorded as ``centre_offset`` and refused beyond ``max_centre_offset``
    of the radius. The residual is the samples' fitted diameter over the stated
    one, minus 1. Returns and raises as ``declared_rectangle``.
    """
    from shapely.geometry import Point

    from neurospatial import Environment

    polygon, fit = fit_circle(positions, diameter)
    if centre is not None:
        offset = float(
            np.hypot(fit["centre_x"] - centre[0], fit["centre_y"] - centre[1])
        )
        if offset > max_centre_offset * diameter / 2:
            msg = (
                f"the samples' centre lies {offset:.3g} from the stated centre"
                f" {tuple(centre)}, more than {max_centre_offset:.0%} of the radius"
                f" ({diameter / 2:g}): the positions are in another frame or unit"
                " than the stated centre; check both"
            )
            raise ValueError(msg)
        polygon = Point(*centre).buffer(diameter / 2, quad_segs=16)
        fit = {
            "stated_centre_x": float(centre[0]),
            "stated_centre_y": float(centre[1]),
            "centre_offset": offset,
            "samples_diameter": fit["samples_diameter"],
        }
    fit["size_residual"] = fit["samples_diameter"] / diameter - 1
    _check_residuals(
        {"size_residual": fit["size_residual"]},
        {"size_residual": bin_size / diameter},
        max_residual,
    )
    factory = partial(
        Environment.from_polygon, polygon, bin_size=bin_size, units=units, name=name
    )
    line = placement_line("circle", f"diameter {diameter:g}", fit, source)
    return factory, line, fit


def with_environment(
    rec: Recording,
    environment: Callable[[], Any],
    conversion: str,
    fit: dict[str, float] | None = None,
    *,
    tracked_point: str,
) -> Recording:
    """``rec`` with its declared environment, its placement fit and the conversion line.

    ``tracked_point`` is what the positions track, as a source states it (a
    head LED, a body part), or "not stated": a head LED on a headstage falls
    short of walls a body point would reach.
    """
    geometry = {
        **rec.geometry,
        "environment": environment,
        "tracked_point": tracked_point,
    }
    if fit is not None:
        geometry["environment_fit"] = fit
    return replace(rec, geometry=geometry, conversions=(*rec.conversions, conversion))


def lattice_graph(bars: NDArray[np.float64], n_cubes: int) -> Any:
    """A climbing lattice's bars as a graph of segments between their crossings.

    Each full-length bar is cut at its ``n_cubes + 1`` evenly spaced
    crossings; crossings shared by several bars (closer than 5 % of the
    spacing) become one node.

    Parameters
    ----------
    bars : ndarray, shape (n_bars, 2, n_dims)
        Start and end of each full-length bar.
    n_cubes : int
        Cubes along each bar.

    Returns
    -------
    networkx.Graph
        Nodes carry ``pos`` (the crossing), edges ``distance`` (the segment's
        length).
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    steps = np.linspace(0.0, 1.0, n_cubes + 1)
    points = bars[:, :1] + steps[None, :, None] * (bars[:, 1:] - bars[:, :1])
    points = points.reshape(-1, bars.shape[2])
    spacing = float(np.linalg.norm(bars[0, 1] - bars[0, 0])) / n_cubes
    label = -np.ones(len(points), dtype=np.intp)
    nodes = []
    for k, group in enumerate(cKDTree(points).query_ball_point(points, 0.05 * spacing)):
        if label[k] < 0:
            members = [g for g in group if label[g] < 0]
            label[members] = len(nodes)
            nodes.append(points[members].mean(axis=0))
    per_bar = label.reshape(len(bars), n_cubes + 1)
    graph = nx.Graph()
    for k, node in enumerate(nodes):
        graph.add_node(k, pos=tuple(float(v) for v in node))
    for u, v in zip(per_bar[:, :-1].ravel(), per_bar[:, 1:].ravel(), strict=True):
        graph.add_edge(
            int(u), int(v), distance=float(np.linalg.norm(nodes[u] - nodes[v]))
        )
    return graph


def distance_to_segments(
    points: NDArray[np.float64], segments: NDArray[np.float64], *, chunk: int = 20000
) -> NDArray[np.float64]:
    """Each point's distance to the nearest of ``segments``.

    Parameters
    ----------
    points : ndarray, shape (n_points, n_dims)
    segments : ndarray, shape (n_segments, 2, n_dims)
        Start and end of each segment.

    Returns
    -------
    ndarray, shape (n_points,)
    """
    start, step = segments[:, 0], segments[:, 1] - segments[:, 0]
    length2 = np.einsum("sd,sd->s", step, step)
    out = np.empty(len(points))
    for lo in range(0, len(points), chunk):
        block = points[lo : lo + chunk, None, :] - start[None]
        t = np.clip(np.einsum("psd,sd->ps", block, step) / length2, 0.0, 1.0)
        nearest = block - t[..., None] * step[None]
        out[lo : lo + chunk] = np.sqrt(
            np.einsum("psd,psd->ps", nearest, nearest).min(1)
        )
    return out


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
# The share of a wall with no sample within ``WALL_SUPPORT_BINS`` bins is
# reported, not judged: a head-mounted LED on a headstage above the animal
# keeps off walls and bars, so a declared outline or lattice is expected to
# have unsupported stretches. A misplaced or wrongly sized outline is caught
# by samples past the opposite wall, the samples' fitted size and the stated
# centre instead.
WALL_SUPPORT_BINS = 3.0
# Consecutive outline edges turning less than this at their shared vertex
# are one side: a circle drawn as a polygon is one wall, a square four.
SIDE_TURN_DEGREES = 10.0


def judge_declared(
    env: Any,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    physical_width: float | None = None,
    speed_thresholds: Sequence[float] = (),
    stated_residuals: Mapping[str, float] | None = None,
    centre_offset: float | None = None,
    bin_size: float | None = None,
    lattice: bool = False,
) -> dict[str, Any]:
    """One verdict for any declared environment: misfit evidence, and tracking.

    Misfit is judged by the environment's kind:

    - a track graph (``TrackArena``): the worst straight stretch's median
      signed lateral offset (over ``_tracks.MIN_PIECE_BINS`` bins, limit one
      bin) and segments with no sample within a bin over more than
      ``_tracks.MAX_UNSUPPORTED`` of their length (``_tracks.edge_fit``);
      for a climbing ``lattice`` that segment support is reported, not
      judged (the tracked head stays a few cm off the bars), and in 3-D a
      side is not defined, so the lattice's placement is not judged;
    - an outline (a polygonal domain): per side, how far the samples nearest
      it reach past it (their ``OUTER_QUANTILE`` signed distance, positive
      beyond the wall; limit ``MAX_SIDE_BIAS_BINS`` bins: an outline moved off
      its samples has them run past one wall), and the samples' extent across
      the outline (they may fall short of a wall, but not by more than
      ``MAX_SIZE_RESIDUAL`` of the side: an outline drawn too large). The
      share of each side (``outline_sides``) with no sample within
      ``WALL_SUPPORT_BINS`` bins is reported, not judged: a headstage LED
      keeps off the walls;
    - a room or other box without an outline (a 3-D grid): per face, as an
      outline's side, how far the samples assigned to it (the face each lies
      farthest past, inside the nearest) reach past it (their
      ``OUTER_QUANTILE`` signed distance; limit ``MAX_SIDE_BIAS_BINS``
      bins), and the stated sizes;
    - a track, also how far its samples reach past its free ends
      (``end_overshoot``, summed over the ends, limit ``MAX_SIZE_RESIDUAL``
      of the track's length: a track drawn too short);
    - any: a stated length or size off by more than ``MAX_SIZE_RESIDUAL``
      (``stated_residuals``: label -> fitted / stated - 1), and the samples'
      centre farther from a stated centre than ``MAX_CENTRE_OFFSET`` of the
      radius (``centre_offset``, that distance over the stated radius).

    The share of tracked samples beyond the physical apparatus (outside half
    of ``physical_width`` on a track, outside the outline or the room) is
    reported as a
    tracking measure, not judged: a head-mounted LED above an unwalled track
    swings past its edge, and a 2-D camera projects a raised head outward.
    It is split by speed (``speed_thresholds``, the positions' units per
    second) to show whether it comes from stopped posture or movement.

    ``bin_size`` is the nominal bin the environment was built with (its
    bins along a graph's edges can be shorter); by default the median bin's
    length, or its measure's root for a grid or outline.

    No sample near the declared geometry (no stretch, side or face with
    samples near it) is a misfit, not a pass.

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
        out["misfit"] = misfit_reasons(fit, bin_size, judge_support=not lattice)
        if fit["end_overshoot"] > MAX_SIZE_RESIDUAL:
            out["misfit"].append(
                f"samples reach {fit['end_overshoot']:.0%} of the track's length"
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
        if polygon is None and getattr(env, "grid_edges", None) is not None:
            out["kind"] = "room"
            edges = [np.asarray(e, dtype=float) for e in env.grid_edges]
            low = np.array([e[0] for e in edges])
            high = np.array([e[-1] for e in edges])
            beyond = np.asarray(env.bin_at(samples)) < 0
            reach = np.quantile(
                np.maximum(np.maximum(low - samples, samples - high), 0.0),
                0.995,
                axis=0,
            )
            # Each face as an outline's side: each sample's signed distance
            # past every face (positive outside), the sample assigned to the
            # face it lies farthest past (inside, the nearest face), and per
            # face the ``OUTER_QUANTILE`` of its samples' distance.
            past = np.hstack([low - samples, samples - high])
            nearest = np.argmax(past, axis=1)
            signed = past[np.arange(len(past)), nearest]
            faces = [
                float(np.quantile(signed[nearest == k], OUTER_QUANTILE))
                for k in range(past.shape[1])
                if np.sum(nearest == k) >= 10
            ]
            out["fit"] = {
                "room_size": (high - low).tolist(),
                "samples_extent": (
                    np.quantile(samples, 0.995, axis=0)
                    - np.quantile(samples, 0.005, axis=0)
                ).tolist(),
                "max_reach_past_bounds": float(reach.max()),
                "max_side_bias": max(faces) if faces else float("nan"),
                "max_side_bias_limit": MAX_SIDE_BIAS_BINS * bin_size,
            }
            if not faces:
                out["misfit"].append("no samples near the declared geometry")
            elif out["fit"]["max_side_bias"] > out["fit"]["max_side_bias_limit"]:
                out["misfit"].append(
                    f"samples reach {out['fit']['max_side_bias']:.3g} beyond one face"
                    f" of the room (limit {out['fit']['max_side_bias_limit']:.3g})"
                )
            for key, value in (stated_residuals or {}).items():
                out["fit"][f"size_residual_{key}"] = value
                if abs(value) > MAX_SIZE_RESIDUAL:
                    out["misfit"].append(f"{key} {value:+.0%} of the stated")
        elif polygon is None:
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
            biases = []
            for k in range(len(sides)):
                mine = nearest == k
                if mine.sum() >= 10:
                    # How far the outermost samples reach past this wall.
                    biases.append(float(np.quantile(signed[mine], OUTER_QUANTILE)))
            tree = cKDTree(samples)
            unsupported = []
            for side in outline_sides(ring):
                points = _points_along(side, bin_size / 2)
                unsupported.append(
                    float(np.mean(tree.query(points)[0] > WALL_SUPPORT_BINS * bin_size))
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
            if not biases:
                out["misfit"].append("no samples near the declared geometry")
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
    if centre_offset is not None:
        out["fit"]["centre_offset_share"] = centre_offset
        if centre_offset > MAX_CENTRE_OFFSET:
            out["misfit"].append(
                f"samples' centre {centre_offset:.0%} of the radius from the stated"
                f" centre (limit {MAX_CENTRE_OFFSET:.0%})"
            )
    if beyond is not None:
        out["outside"] = float(np.mean(beyond))
        out["outside_by_speed"] = share_by_speed(
            times, positions, beyond, speed_thresholds
        )
    return out


def outline_sides(ring: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """An outline's sides: runs of edges turning less than ``SIDE_TURN_DEGREES``.

    ``ring`` is the closed exterior, shape (n_vertices + 1, 2). Each side is
    its vertices in order, shape (n_side_vertices, 2); a ring with no corner
    (a circle drawn as a polygon) is one side.
    """
    vertices = np.asarray(ring, dtype=float)[:-1]
    step = np.roll(vertices, -1, axis=0) - vertices
    heading = np.arctan2(step[:, 1], step[:, 0])
    # Turn at vertex k, between the edge ending there and the edge leaving it.
    turn = np.abs(np.angle(np.exp(1j * (heading - np.roll(heading, 1)))))
    corners = np.flatnonzero(turn >= np.radians(SIDE_TURN_DEGREES))
    if not len(corners):
        return [np.vstack([vertices, vertices[:1]])]
    sides = []
    for start, stop in zip(corners, np.roll(corners, -1), strict=True):
        index = np.arange(start, stop + (len(vertices) if stop <= start else 0) + 1)
        sides.append(vertices[index % len(vertices)])
    return sides


def _points_along(polyline: NDArray[np.float64], step: float) -> NDArray[np.float64]:
    """Points along ``polyline`` (shape (n, 2)) at most ``step`` apart, ends included."""
    pieces = []
    for a, b in itertools.pairwise(polyline):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        pieces.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    return np.vstack([*pieces, polyline[-1:]])


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
