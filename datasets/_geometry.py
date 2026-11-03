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

from typing import Any

import numpy as np
from numpy.typing import NDArray


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
