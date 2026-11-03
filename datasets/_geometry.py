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

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
from _record import Recording
from numpy.typing import NDArray

# A stated size more than this share away from the samples' own fitted size
# is refused: the stated apparatus, the unit or the session is then wrong.
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
) -> Recording:
    """``rec`` with its declared environment, its placement fit and the conversion line."""
    geometry = {**rec.geometry, "environment": environment}
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
