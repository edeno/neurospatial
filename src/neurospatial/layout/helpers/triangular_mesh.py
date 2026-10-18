"""TriangularMeshLayout: a LayoutEngine that discretizes a 2D corridor region via a Delaunay triangulation.
Instead of requiring the user to supply sample points, this layout generates a uniform grid of interior points
(spaced by `point_spacing`) clipped to the provided boundary polygon. Each triangle whose centroid lies
inside the polygon is kept as an active bin. Connectivity is inferred by shared faces.

Conforms to the LayoutEngine protocol:
  - Attributes:    bin_centers, connectivity (derived), dimension_ranges,
                   grid_edges, grid_shape, active_mask,
                   layout_type, _build_params_used
  - Properties:    is_linearized_track
  - Methods:       build(boundary_polygon, point_spacing), binning(),
                   plot(ax, ...), bin_sizes()
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
    """Filter Delaunay simplices by requiring:
    1) The triangle's centroid lies inside or on the boundary (using `covers`).
    2) All three vertices lie inside or on the boundary.
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
    """Given a Shapely Polygon, return an (M,2)-array of points sampled along
    the exterior boundary (including the vertices) at approximately 'point_spacing' intervals.
    """
    coords = np.array(poly.exterior.coords)  # shape = (N_vertices+1, 2)
    boundary_pts: list[list[float]] = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        edge_len = np.hypot(x1 - x0, y1 - y0)
        if edge_len == 0:
            continue

        # How many samples along this edge? 1 point every ~point_spacing (plus both endpoints).
        n_segs = max(int(np.ceil(edge_len / point_spacing)), 1)
        # Generate n_segs+1 points equally spaced including both ends.
        for t in np.linspace(0.0, 1.0, n_segs + 1):
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            boundary_pts.append([float(px), float(py)])

    # Deduplicate (because consecutive edges share endpoints):
    boundary_pts_array = np.unique(np.array(boundary_pts), axis=0)
    return np.asarray(boundary_pts_array, dtype=np.float64)  # shape = (M, 2)
