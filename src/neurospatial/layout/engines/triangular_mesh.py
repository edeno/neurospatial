from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry import (
    Binning,
    MeshLocator,
    Outline,
    TriangleShapes,
    component_extent,
    mask_active_map,
)
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._labels import LayoutType
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.neighbor_pairs import euclidean_pair_distances
from neurospatial.layout.helpers.triangular_mesh import (
    _filter_active_simplices_by_centroid,
    _generate_interior_points_for_mesh,
    _sample_polygon_boundary,
    _triangulate_points,
    mesh_pairs,
)
from neurospatial.layout.mixins import _NeighborPairsMixin
from neurospatial.layout.validation import validate_cell_size


class TriangularMeshLayout(_NeighborPairsMixin):
    """A LayoutEngine that builds a triangular mesh over interior points
    (auto-generated) clipped to a boundary polygon. Each triangle whose centroid
    lies inside the polygon is kept as an active bin. Connectivity by shared faces.
    """

    layout_type: ClassVar[str] = LayoutType.TRIANGULAR_MESH.value
    size_parameter: ClassVar[str | None] = "point_spacing"
    _build_params_used: dict[str, Any]

    bin_centers: NDArray[np.float64]
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[()]  # Explicitly empty tuple for non-grid
    grid_shape: tuple[int, ...] | None
    active_mask: NDArray[np.bool_] | None

    # Internal state
    _full_delaunay_tri: Delaunay | None
    _active_original_simplex_indices: (
        NDArray[np.int_] | None
    )  # Store original indices of active ones
    _boundary_polygon_stored: Polygon | None  # Store the actual polygon object
    _locator: MeshLocator | None = None

    def __init__(self) -> None:
        self.bin_centers = np.empty((0, 2), dtype=float)
        self.dimension_ranges = None
        self.grid_edges = ()  # For non-grid layouts, this is typically empty
        self.grid_shape = None
        self.active_mask = None  # Will be updated in build
        self._build_params_used = {}

        self._full_delaunay_tri = None
        self._active_original_simplex_indices = None
        self._boundary_polygon_stored = None
        self._locator = None

    @capture_build_params(data=("boundary_polygon",))
    def build(self, boundary_polygon: Polygon, *, point_spacing: float) -> None:
        """Build the triangular mesh layout.

        Parameters
        ----------
        boundary_polygon : shapely.geometry.Polygon
            The polygon defining the boundary. Triangles with centroids
            outside this polygon are discarded.
        point_spacing : float
            Desired spacing (in same units as polygon) between generated sample points
            used for triangulation.

        """
        self._begin_build()
        if not isinstance(boundary_polygon, Polygon):
            raise TypeError("boundary_polygon must be a Shapely Polygon.")
        if boundary_polygon.is_empty:
            raise ValueError("boundary_polygon cannot be empty.")
        point_spacing = float(
            validate_cell_size(point_spacing, name="point_spacing", scalar_only=True)[0]
        )

        self._boundary_polygon_stored = (
            boundary_polygon  # Store the actual object for use
        )

        # 1. Generate sample points for triangulation
        sample_points = _generate_interior_points_for_mesh(
            boundary_polygon,
            point_spacing,
        )
        if sample_points.shape[0] < 3:  # Delaunay needs at least N+1 points in N-D
            raise ValueError(
                f"Not enough interior sample points ({sample_points.shape[0]}) generated "
                "to form any triangle. Try decreasing point_spacing or ensuring "
                "the polygon is large enough relative to the spacing.",
            )
        exterior_coords = _sample_polygon_boundary(boundary_polygon, point_spacing)
        # Stack the interior grid points with the boundary vertices.
        sample_points = np.vstack([sample_points, exterior_coords])

        # 2. Perform Delaunay triangulation
        self._full_delaunay_tri = _triangulate_points(sample_points)

        # 3. Filter active simplices (triangles)
        active_original_indices, all_centroids = _filter_active_simplices_by_centroid(
            self._full_delaunay_tri,
            boundary_polygon,
        )
        n_total_delaunay_triangles = self._full_delaunay_tri.simplices.shape[0]

        if active_original_indices.size == 0:
            raise ValueError(
                "No triangles found with centroids inside the boundary polygon. "
                "Check boundary_polygon shape, point_spacing, or point generation strategy.",
            )

        self._active_original_simplex_indices = active_original_indices

        # 4. Populate core attributes for active triangles
        self.bin_centers = all_centroids[active_original_indices]

        # 5. Pair active triangles that share a side
        pairs = mesh_pairs(active_original_indices, self._full_delaunay_tri.neighbors)
        self._set_neighbor_pairs(
            pairs, euclidean_pair_distances(self.bin_centers, pairs)
        )

        # 6. Cell extent: the mesh boundary's bounds
        self.dimension_ranges = component_extent(
            outline=self.outline(), shapes=self.bin_shapes()
        )

        # 7. Set grid-related attributes for protocol conformance
        # The "conceptual grid" here is the list of all Delaunay triangles.
        self.grid_shape = (n_total_delaunay_triangles,)
        self.active_mask = np.zeros(n_total_delaunay_triangles, dtype=bool)
        self.active_mask[active_original_indices] = True
        # self.grid_edges remains an empty tuple as it's not a rectilinear grid.
        self._own_arrays("bin_centers", "active_mask")
        self._build_locator()

    def _build_locator(self) -> None:
        """Snapshot triangulation, active-bin mapping and the boundary."""
        if self._full_delaunay_tri is None or self.active_mask is None:
            raise LayoutNotBuiltError(type(self).__name__, "locator")
        self._locator = MeshLocator(
            self._full_delaunay_tri.points,
            self._full_delaunay_tri.simplices,
            mask_active_map(self.active_mask),
            self._boundary_polygon_stored,
        )

    def locator(self) -> MeshLocator:
        """Return the owned locator, building it on first use."""
        if self._locator is None:
            self._build_locator()
        assert self._locator is not None
        return self._locator

    def binning(self) -> Binning:
        """Return owned triangles, lookup and lazy shared-edge faces.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        tri = self._full_delaunay_tri
        if tri is None or self._active_original_simplex_indices is None:
            raise LayoutNotBuiltError(type(self).__name__, "binning")
        pairs, distances = self._neighbor_pairs()
        return assemble_binning(
            pairs,
            distances,
            kind=self.layout_type,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=self.bin_centers.shape[1],
            locator=self.locator(),
            faces="mesh",
            face_arrays=(
                tri.points,
                tri.simplices,
                np.asarray(self._active_original_simplex_indices),
            ),
            shapes=self.bin_shapes(),
            outline=self.outline(),
            dimension_ranges=self.dimension_ranges,
        )

    @property
    def is_linearized_track(self) -> bool:
        """Always False, as this is a 2D mesh layout.

        Returns
        -------
        bool
            Always False, indicating a 2D layout.

        """
        return False

    def bin_shapes(self) -> TriangleShapes:
        """Return mesh points and the active triangles in bin order."""
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "bin_shapes")
        return TriangleShapes(
            self._full_delaunay_tri.points,
            self._full_delaunay_tri.simplices[self._active_original_simplex_indices],
        )

    def outline(self) -> Outline | None:
        """Return the mesh boundary, including its interior rings when present."""
        return (
            Outline(self._boundary_polygon_stored)
            if self._boundary_polygon_stored is not None
            else None
        )

    def bin_sizes(self) -> NDArray[np.float64]:
        """Return the N-dimensional volume of each active N-simplex (bin).

        For 2D, this is area. For 3D, this is volume, and so on.
        The volume of an N-simplex with vertices v0, v1, ..., vn is calculated as:
        V = (1 / n!) * |det([v1-v0, v2-v0, ..., vn-v0])|
        where n is the number of dimensions.

        Returns
        -------
        NDArray[np.float64]
            Array of N-dimensional volumes, shape (n_active_simplices,).

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.

        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "bin_sizes")

        # Get the vertices of all active N-simplices
        # .points has shape (total_points, n_dim)
        # .simplices has shape (total_simplices, n_dim + 1)
        all_mesh_points = self._full_delaunay_tri.points
        active_simplices_vertex_indices = self._full_delaunay_tri.simplices[
            self._active_original_simplex_indices
        ]

        # nsimplex_vertices will have shape:
        # (num_active_simplices, num_vertices_per_simplex, n_dim)
        nsimplex_vertices = all_mesh_points[active_simplices_vertex_indices]

        n_dim = all_mesh_points.shape[1]

        # Select one vertex from each simplex as the origin (v0)
        # v0 will have shape (num_active_simplices, n_dim)
        v0 = nsimplex_vertices[:, 0, :]

        # Create vectors from v0 to all other vertices (v1-v0, v2-v0, ..., vn-v0)
        # These vectors form the rows (or columns) of the matrix for the determinant.
        # nsimplex_vertices[:, 1:, :] has shape (n_active, n_dim, n_dim)
        # v0[:, np.newaxis, :] broadcasts v0 to match for subtraction
        # matrix_for_determinant will have shape (n_active, n_dim, n_dim)
        matrix_for_determinant = nsimplex_vertices[:, 1:, :] - v0[:, np.newaxis, :]

        # Calculate the determinant for each simplex's matrix
        # np.linalg.det operates on the last two axes by default.
        determinants = np.linalg.det(matrix_for_determinant)

        # Volume = abs(determinant) / n!
        n_factorial = float(math.factorial(n_dim))
        volumes = np.abs(determinants) / n_factorial

        return np.asarray(volumes, dtype=np.float64)
