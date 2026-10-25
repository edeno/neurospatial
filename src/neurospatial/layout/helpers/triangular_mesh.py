"""Helpers for ``TriangularMeshLayout``.

Sample points on a lattice inside a polygon and along each of its rings,
triangulate them, keep the triangles whose centroid lies in the polygon,
and pair triangles that share a side.
"""

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Polygon


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
    triangulation: Delaunay,
    boundary_polygon: Polygon,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Keep the Delaunay triangles whose centroid lies in or on the polygon.

    Parameters
    ----------
    triangulation : scipy.spatial.Delaunay
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

    # (1) compute centroids of each triangle
    triangle_vertices = points[simplices]  # (M, 3, 2)
    all_centroids = np.mean(triangle_vertices, axis=1)  # (M, 2)
    active_mask = shapely.covers(boundary_polygon, shapely.points(all_centroids))

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
) -> NDArray[np.float64]:
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
    ndarray of float64, shape (n_points, 2)
        Distinct sample points, sorted.
    """
    samples = []
    for ring in (poly.exterior, *poly.interiors):
        coords = np.asarray(ring.coords, dtype=np.float64)
        starts, steps = coords[:-1], np.diff(coords, axis=0)
        n_segments = np.maximum(
            np.ceil(np.hypot(steps[:, 0], steps[:, 1]) / point_spacing), 1
        ).astype(int)
        edge = np.repeat(np.arange(len(starts)), n_segments + 1)
        first = np.repeat(np.cumsum(n_segments + 1) - (n_segments + 1), n_segments + 1)
        step = np.arange(len(edge)) - first
        # np.linspace's arithmetic: k * (1 / n), with the last sample at 1.
        t = np.where(step == n_segments[edge], 1.0, step * (1.0 / n_segments[edge]))
        samples.append(starts[edge] + t[:, None] * steps[edge])
    # Consecutive edges share endpoints.
    return np.unique(np.concatenate(samples), axis=0)
