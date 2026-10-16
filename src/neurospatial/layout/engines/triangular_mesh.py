from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.axes
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._components import Outline, TriangleShapes
from neurospatial.geometry._locators import MeshLocator
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._labels import LayoutType
from neurospatial.layout._plotting import plot_triangles
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.triangular_mesh import (
    _build_mesh_connectivity_graph,
    _compute_mesh_dimension_ranges,
    _filter_active_simplices_by_centroid,
    _generate_interior_points_for_mesh,
    _sample_polygon_boundary,
    _triangulate_points,
)
from neurospatial.layout.validation import validate_connectivity_graph

_MESH_SKEW_ANGLE_DEG = 30.0
_MESH_SKEW_FRACTION = 0.05


class TriangularMeshLayout:
    """A LayoutEngine that builds a triangular mesh over interior points
    (auto-generated) clipped to a boundary polygon. Each triangle whose centroid
    lies inside the polygon is kept as an active bin. Connectivity by shared faces.
    """

    layout_type: str = LayoutType.TRIANGULAR_MESH.value
    _build_params_used: dict[str, Any]

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[()]  # Explicitly empty tuple for non-grid
    grid_shape: tuple[int, ...] | None
    active_mask: NDArray[np.bool_] | None

    # Internal state
    _full_delaunay_tri: Delaunay | None
    _original_simplex_to_active_idx_map: dict[int, int] | None
    _active_original_simplex_indices: (
        NDArray[np.int_] | None
    )  # Store original indices of active ones
    _boundary_polygon_stored: Polygon | None  # Store the actual polygon object
    _locator: MeshLocator | None = None

    def __init__(self) -> None:
        self.bin_centers = np.empty((0, 2), dtype=float)
        self.connectivity = nx.Graph()
        self.dimension_ranges = None
        self.grid_edges = ()  # For non-grid layouts, this is typically empty
        self.grid_shape = None
        self.active_mask = None  # Will be updated in build
        self._build_params_used = {}

        self._full_delaunay_tri = None
        self._original_simplex_to_active_idx_map = None
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
        if not isinstance(boundary_polygon, Polygon):
            raise TypeError("boundary_polygon must be a Shapely Polygon.")
        if boundary_polygon.is_empty:
            raise ValueError("boundary_polygon cannot be empty.")
        if boundary_polygon.geom_type == "MultiPolygon":
            raise ValueError(
                "MultiPolygon boundaries are not directly supported. "
                "Please provide a single Polygon component.",
            )
        if point_spacing <= 0:
            raise ValueError(f"point_spacing must be positive, got {point_spacing}.")

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
        # 4) Stack the interior grid points with the boundary vertices.
        if sample_points.size == 0:
            sample_points = exterior_coords.copy()
        else:
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

        # 4. Create mapping from original Delaunay simplex index to active triangle index
        self._original_simplex_to_active_idx_map = {
            orig_idx: active_idx
            for active_idx, orig_idx in enumerate(active_original_indices)
        }

        # 5. Populate core attributes for active triangles
        self.bin_centers = all_centroids[active_original_indices]

        # 6. Build connectivity graph for active triangles
        self.connectivity = _build_mesh_connectivity_graph(
            active_original_indices,
            all_centroids,
            self._original_simplex_to_active_idx_map,
            self._full_delaunay_tri,
        )

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(self.connectivity, n_dims=2)

        # 7. Compute dimension_ranges based on active bin centers
        self.dimension_ranges = _compute_mesh_dimension_ranges(self.bin_centers)

        # 8. Set grid-related attributes for protocol conformance
        # The "conceptual grid" here is the list of all Delaunay triangles.
        self.grid_shape = (n_total_delaunay_triangles,)
        self.active_mask = np.zeros(n_total_delaunay_triangles, dtype=bool)
        self.active_mask[active_original_indices] = True
        # self.grid_edges remains an empty tuple as it's not a rectilinear grid.
        self._build_locator()

    def _build_locator(self) -> None:
        """Snapshot triangulation, active-bin mapping and the boundary."""
        if (
            self._full_delaunay_tri is None
            or self._original_simplex_to_active_idx_map is None
        ):
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in (
                        "_full_delaunay_tri",
                        "_original_simplex_to_active_idx_map",
                    )
                    if getattr(self, name) is None
                ),
            )
        simplex_to_active = np.full(
            len(self._full_delaunay_tri.simplices), -1, dtype=np.intp
        )
        for original, active in self._original_simplex_to_active_idx_map.items():
            simplex_to_active[original] = active
        self._locator = MeshLocator(
            self._full_delaunay_tri.points,
            self._full_delaunay_tri.simplices,
            simplex_to_active,
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
            raise LayoutNotBuiltError(type(self).__name__, "_full_delaunay_tri")
        return assemble_binning(
            self.connectivity,
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

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show_triangles: bool = True,
        show_centroids: bool = True,
        show_connectivity: bool = True,
        show_boundary: bool = True,
        triangle_kwargs: dict[str, Any] | None = None,
        centroid_kwargs: dict[str, Any] | None = None,
        connectivity_kwargs: dict[str, Any] | None = None,
        boundary_kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the triangular mesh layout.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, a new figure and axes are created.
        show_triangles : bool, optional
            Whether to draw the filled active triangles. Defaults to True.
        show_centroids : bool, optional
            Whether to draw the centroids of active triangles. Defaults to True.
        show_connectivity : bool, optional
            Whether to draw edges of the connectivity graph. Defaults to True.
        show_boundary : bool, optional
            Whether to draw the original boundary polygon. Defaults to True.
        triangle_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for `matplotlib.collections.PatchCollection` of triangles.
        centroid_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for `ax.scatter` plotting centroids.
        connectivity_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for plotting connectivity edges.
        boundary_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for plotting the boundary polygon.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout was plotted.

        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
            or self.bin_centers is None
            or self.connectivity is None
            or self.dimension_ranges is None
        ):
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in (
                        "_full_delaunay_tri",
                        "_active_original_simplex_indices",
                        "bin_centers",
                        "connectivity",
                        "dimension_ranges",
                    )
                    if getattr(self, name) is None
                ),
            )

        return plot_triangles(
            ax,
            title=self.layout_type,
            shapes=self.bin_shapes(),
            outline=self.outline(),
            connectivity=self.connectivity,
            dimension_ranges=self.dimension_ranges,
            bin_centers=self.bin_centers,
            show_triangles=show_triangles,
            show_centroids=show_centroids,
            show_connectivity=show_connectivity,
            show_boundary=show_boundary,
            triangle_kwargs=triangle_kwargs,
            centroid_kwargs=centroid_kwargs,
            connectivity_kwargs=connectivity_kwargs,
            boundary_kwargs=boundary_kwargs,
        )

    def bin_shapes(self) -> TriangleShapes:
        """Return mesh points and the active triangles in bin order."""
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "_full_delaunay_tri")
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
        RuntimeError
            If the layout is not built (e.g., `_full_delaunay_tri` or
            `_active_original_simplex_indices` is None).
        ValueError
            If the dimensionality is less than 1.

        """
        if (
            self._full_delaunay_tri is None
            or self._active_original_simplex_indices is None
        ):  # pragma: no cover
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in (
                        "_full_delaunay_tri",
                        "_active_original_simplex_indices",
                    )
                    if getattr(self, name) is None
                ),
            )

        # Get the vertices of all active N-simplices
        # .points has shape (total_points, n_dim)
        # .simplices has shape (total_simplices, n_dim + 1)
        all_mesh_points = self._full_delaunay_tri.points
        active_simplices_vertex_indices = self._full_delaunay_tri.simplices[
            self._active_original_simplex_indices
        ]

        if active_simplices_vertex_indices.shape[0] == 0:
            return np.array([], dtype=float)  # No active simplices

        # nsimplex_vertices will have shape:
        # (num_active_simplices, num_vertices_per_simplex, n_dim)
        nsimplex_vertices = all_mesh_points[active_simplices_vertex_indices]

        n_dim = all_mesh_points.shape[1]

        if n_dim == 1:
            # For 1D simplices (line segments), volume is length
            # Vertices are (n_active, 2, 1)
            # v0 is (n_active, 1), v1 is (n_active, 1)
            v0 = nsimplex_vertices[:, 0, :]  # First vertex of each simplex
            v1 = nsimplex_vertices[:, 1, :]  # Second vertex of each simplex
            lengths = np.abs(v1 - v0).squeeze(axis=-1)
            return np.asarray(lengths, dtype=np.float64)

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
