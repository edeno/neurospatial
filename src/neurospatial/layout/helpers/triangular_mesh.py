"""Helpers for ``TriangularMeshLayout``.

Sample points on a lattice inside a polygon and along each of its rings,
triangulate them, keep the triangles whose centroid lies in the polygon,
and pair triangles that share a side.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Polygon

from neurospatial._exceptions import _format_error

# Ring segments shorter than this fraction of the point spacing are not split.
_SHORTEST_SPLIT = 1e-6
# Refinement may add points up to this multiple of the starting points.
_MAX_GROWTH = 50
# A triangle whose doubled area is below this fraction of its longest side
# squared is flat: its corners are collinear to within rounding.
_FLAT_TRIANGLE = 1e-12


@dataclass(frozen=True)
class _MeshTriangulation:
    """Triangles over ``points``: what the mesh engine keeps of a Delaunay.

    Attributes
    ----------
    points : ndarray of float64, shape (n_points, 2)
        Triangle corners, in the polygon's coordinates.
    simplices : ndarray of int, shape (n_triangles, 3)
        Corner indices of each triangle.
    neighbors : ndarray of int, shape (n_triangles, 3)
        Triangle across each side, -1 on the hull, as in
        ``scipy.spatial.Delaunay.neighbors``.
    """

    points: NDArray[np.float64]
    simplices: NDArray[np.intp]
    neighbors: NDArray[np.intp]


# --------------------------------------------------------------------------
# Module-level Private Helper Functions
# --------------------------------------------------------------------------
def _generate_interior_points_for_mesh(
    boundary_polygon: Polygon,
    point_spacing: float,
) -> NDArray[np.float64]:
    """Generate a uniform grid of points spaced by `point_spacing` that lie inside `boundary_polygon`.

    Parameters
    ----------
    boundary_polygon : shapely.geometry.Polygon
        The polygon defining the boundary.
    point_spacing : float
        Spacing between generated points.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (M, 2) representing the interior sample points.

    """
    minx, miny, maxx, maxy = boundary_polygon.bounds
    xs = np.arange(minx + point_spacing / 2, maxx, point_spacing)
    ys = np.arange(miny + point_spacing / 2, maxy, point_spacing)

    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.column_stack((xv.ravel(), yv.ravel()))

    if candidates.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    mask_pts = shapely.covers(boundary_polygon, shapely.points(candidates))
    return np.asarray(candidates[mask_pts], dtype=np.float64)


def _triangulate_points(sample_points: NDArray[np.float64]) -> Delaunay:
    """Attempt a Delaunay triangulation on `sample_points`.

    Parameters
    ----------
    sample_points : NDArray[np.float64]
        Points to triangulate, shape (N, 2).

    Returns
    -------
    scipy.spatial.Delaunay
        The Delaunay triangulation object.

    Raises
    ------
    ValueError
        If triangulation fails (e.g., points are colinear or insufficient).

    """
    if sample_points.shape[0] < 3:
        raise ValueError(
            f"Triangulation requires at least 3 points, got {sample_points.shape[0]}.",
        )
    try:
        return Delaunay(sample_points)
    except QhullError as e:
        raise ValueError(
            "Delaunay triangulation failed. Points may be colinear, "
            "insufficient, or have other geometric issues.",
        ) from e
    except TypeError as e:  # Can happen if points array is not float/double
        raise ValueError(
            f"Delaunay triangulation failed due to input type: {e}. Ensure points are float.",
        ) from e


def _filter_active_simplices_by_centroid(
    triangulation: _MeshTriangulation | Delaunay,
    boundary_polygon: Polygon,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Keep the triangles with area whose centroid lies in or on the polygon.

    Collinear samples along the outline can form flat triangles on the
    triangulation's hull; they cover no area and are not bins.

    Parameters
    ----------
    triangulation : _MeshTriangulation or scipy.spatial.Delaunay
        Triangulation of the lattice and ring sample points.
    boundary_polygon : shapely.geometry.Polygon
        Polygon the mesh covers.

    Returns
    -------
    active_indices : ndarray of int, shape (n_active,)
        Increasing indices of the kept triangles.
    centroids : ndarray of float64, shape (n_triangles, 2)
        Centroid of every triangle, kept or not.
    """
    simplices = triangulation.simplices  # shape (M, 3)
    points = triangulation.points  # shape (N, 2)

    if simplices.shape[0] == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=float)

    triangle_vertices = points[simplices]  # (M, 3, 2)
    all_centroids = np.mean(triangle_vertices, axis=1)  # (M, 2)
    first, second = np.moveaxis(
        triangle_vertices[:, 1:] - triangle_vertices[:, :1], 1, 0
    )
    doubled_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    sides = np.linalg.norm(
        triangle_vertices - np.roll(triangle_vertices, 1, axis=1), axis=2
    )
    flat = doubled_area <= _FLAT_TRIANGLE * sides.max(axis=1) ** 2
    active_mask = ~flat & shapely.covers(
        boundary_polygon, shapely.points(all_centroids)
    )

    active_original_indices = np.flatnonzero(active_mask)
    return active_original_indices, all_centroids


def mesh_pairs(
    active_simplices: NDArray[np.int_], simplex_neighbors: NDArray[np.int_]
) -> NDArray[np.intp]:
    """Pair active triangles with the active triangles sharing one of their sides.

    Bins are the active triangles in ``active_simplices`` order. Each bin
    pairs with its higher-numbered active neighbors, in the order of its row
    of ``simplex_neighbors``.

    Parameters
    ----------
    active_simplices : ndarray of int, shape (n_bins,)
        Increasing triangulation indices of the active triangles.
    simplex_neighbors : ndarray of int, shape (n_simplices, 3)
        Neighboring triangle across each side, or -1 on the hull, as in
        ``scipy.spatial.Delaunay.neighbors``.

    Returns
    -------
    ndarray of intp, shape (n_pairs, 2)
        Pairs ``(lower, higher)`` of bin IDs in authoritative order.

    Examples
    --------
    >>> neighbors = np.array([[1, -1, -1], [2, 0, -1], [-1, 1, -1]])
    >>> mesh_pairs(np.array([0, 1, 2]), neighbors).tolist()
    [[0, 1], [1, 2]]
    """
    active_simplices = np.asarray(active_simplices, dtype=np.intp)
    n_bins = len(active_simplices)
    bin_of_simplex = np.full(len(simplex_neighbors) + 1, -1, dtype=np.intp)
    bin_of_simplex[active_simplices] = np.arange(n_bins, dtype=np.intp)
    # Index -1 (hull side) reads the extra trailing -1 entry.
    other = bin_of_simplex[np.asarray(simplex_neighbors)[active_simplices]]
    lower = np.broadcast_to(np.arange(n_bins, dtype=np.intp)[:, None], other.shape)
    keep = other > lower
    return np.column_stack((lower[keep], other[keep]))


def _sample_polygon_boundary(
    poly: Polygon, point_spacing: float
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Sample every ring of a polygon at about ``point_spacing`` intervals.

    The exterior and each hole's ring are sampled, vertices included, so
    the triangulation follows the holes' edges as well as the outline.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Polygon whose rings are sampled.
    point_spacing : float
        Largest distance between consecutive samples along a ring edge.

    Returns
    -------
    points : ndarray of float64, shape (n_points, 2)
        Distinct sample points, sorted.
    segments : ndarray of intp, shape (n_segments, 2)
        Consecutive samples along each ring, as ``(lower, higher)`` indices
        into ``points``.
    """
    samples = []
    ring_of_sample = []
    for ring_number, ring in enumerate((poly.exterior, *poly.interiors)):
        coords = np.asarray(ring.coords, dtype=np.float64)
        starts, steps = coords[:-1], np.diff(coords, axis=0)
        n_segments = np.maximum(
            np.ceil(np.hypot(steps[:, 0], steps[:, 1]) / point_spacing), 1
        ).astype(int)
        edge = np.repeat(np.arange(len(starts)), n_segments + 1)
        first = np.repeat(np.cumsum(n_segments + 1) - (n_segments + 1), n_segments + 1)
        step = np.arange(len(edge)) - first
        # np.linspace's arithmetic: k * (1 / n). The last sample of an edge
        # is the polygon vertex itself, so neighboring edges share it exactly.
        last = step == n_segments[edge]
        sampled = (
            starts[edge] + (step * (1.0 / n_segments[edge]))[:, None] * steps[edge]
        )
        sampled[last] = coords[edge[last] + 1]
        samples.append(sampled)
        ring_of_sample.append(np.full(len(edge), ring_number))
    # Consecutive edges share endpoints, and each ring ends where it starts.
    points, order = np.unique(np.concatenate(samples), axis=0, return_inverse=True)
    order = order.ravel()
    same_ring = np.diff(np.concatenate(ring_of_sample)) == 0
    pairs = np.column_stack((order[:-1], order[1:]))[same_ring]
    pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1).astype(np.intp)
    return points, np.unique(pairs, axis=0)


def _conforming_triangulation(
    interior_points: NDArray[np.float64],
    boundary_polygon: Polygon,
    point_spacing: float,
    triangulate: Callable[[NDArray[np.float64]], Delaunay],
    max_rounds: int = 64,
) -> _MeshTriangulation:
    """Triangulate so that every ring segment of the polygon is a triangle side.

    A Delaunay triangulation of the lattice and ring samples can bridge a
    hole or notch narrower than ``point_spacing``: a triangle with corners
    on both sides then straddles the outline, and keeping triangles by
    their centroid gets the covered area wrong. Each ring segment missing
    from the triangulation is split at its midpoint and the points are
    triangulated again, until every segment is a side. Then every triangle
    lies inside or outside the polygon, and its centroid tells which.

    Parameters
    ----------
    interior_points : ndarray of float64, shape (n_interior, 2)
        Lattice points inside the polygon.
    boundary_polygon : shapely.geometry.Polygon
        Polygon the mesh covers.
    point_spacing : float
        Spacing of the ring samples.
    triangulate : callable
        Delaunay triangulation of an ``(n_points, 2)`` array, called on the
        points shifted so the polygon's lower-left bound is the origin.
    max_rounds : int, default 64
        Rounds of segment splitting before giving up.

    Returns
    -------
    _MeshTriangulation
        Triangles over the original (unshifted) points whose sides include
        every ring segment.

    Raises
    ------
    ValueError
        If refinement makes no progress (Qhull drops a split point), would
        grow past ``_MAX_GROWTH`` times the starting points, or segments are
        still missing after ``max_rounds`` rounds.

    Notes
    -----
    A missing segment shorter than a millionth of ``point_spacing`` is not
    split: Qhull merges points that close, so splitting could not add it,
    and a triangle crossing it misplaces no measurable area. Each segment is
    therefore halved at most about 20 times.
    """
    ring_points, segments = _sample_polygon_boundary(boundary_polygon, point_spacing)
    points = np.vstack([interior_points, ring_points])
    segments = segments + len(interior_points)
    # Triangulate near the origin: far from it, float spacing is too coarse
    # for Qhull to place the midpoints, and refinement makes no progress.
    origin = np.asarray(boundary_polygon.bounds[:2], dtype=np.float64)
    budget = _MAX_GROWTH * len(points)
    added = np.empty(0, dtype=np.intp)
    for _ in range(max_rounds):
        triangulation = triangulate(points - origin)
        # Compare points by their first identical copy: Qhull keeps one of
        # a lattice point and a ring sample at the same place.
        _, first, inverse = np.unique(
            points, axis=0, return_index=True, return_inverse=True
        )
        same = first[inverse.ravel()]
        vertices = np.zeros(len(points), dtype=bool)
        vertices[same[triangulation.simplices]] = True
        if not vertices[same[added]].all():
            raise ValueError(_unfollowed_outline("a split point was dropped"))
        sides = same[triangulation.simplices[:, [[0, 1], [1, 2], [2, 0]]]]
        sides = np.sort(sides.reshape(-1, 2), axis=1)
        wanted = np.sort(same[segments], axis=1)
        n = len(points)
        missing = ~np.isin(
            wanted[:, 0] * n + wanted[:, 1], sides[:, 0] * n + sides[:, 1]
        )
        lengths = np.linalg.norm(
            points[segments[:, 1]] - points[segments[:, 0]], axis=1
        )
        missing &= lengths > _SHORTEST_SPLIT * point_spacing
        if not missing.any():
            return _MeshTriangulation(
                points, triangulation.simplices, triangulation.neighbors
            )
        low, high = segments[missing].T
        if n + len(low) > budget:
            raise ValueError(
                _unfollowed_outline(f"it would need more than {budget} points")
            )
        added = np.arange(n, n + len(low), dtype=np.intp)
        points = np.vstack([points, (points[low] + points[high]) / 2])
        segments = np.vstack(
            [
                segments[~missing],
                np.column_stack((low, added)),
                np.column_stack((added, high)),
            ]
        )
    raise ValueError(_unfollowed_outline(f"{max_rounds} rounds were not enough"))


def _unfollowed_outline(reason: str) -> str:
    """The refusal for an outline the mesh could not follow."""
    return _format_error(
        f"The mesh could not follow the polygon's outline: {reason}.",
        why="Why: a very sharp corner or a gap far narrower than point_spacing "
        "needs ever smaller triangles along it.",
        fix="use a smaller point_spacing, or simplify the polygon, e.g. "
        "polygon.simplify(point_spacing / 10)",
    )
