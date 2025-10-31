"""
Tests for TriangularMeshLayout engine and helper functions.

This module tests the triangular mesh layout engine which:
- Generates interior points for triangulation
- Performs Delaunay triangulation
- Filters active triangles by centroid location
- Builds connectivity graphs
- Maps points to triangle bins
- Computes triangle areas (bin sizes)
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout
from neurospatial.layout.helpers.triangular_mesh import (
    _build_mesh_connectivity_graph,
    _compute_mesh_dimension_ranges,
    _filter_active_simplices_by_centroid,
    _generate_interior_points_for_mesh,
    _sample_polygon_boundary,
    _triangulate_points,
)


class TestGenerateInteriorPoints:
    """Tests for _generate_interior_points_for_mesh helper function."""

    def test_basic_rectangular_polygon(self):
        """Test generating interior points for a simple rectangle."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        points = _generate_interior_points_for_mesh(poly, point_spacing=2.0)

        # Should generate a grid of interior points
        assert points.shape[1] == 2  # 2D points
        assert points.shape[0] > 0  # Some points generated
        # Check all points are inside
        for pt in points:
            assert poly.contains(shapely.geometry.Point(pt))

    def test_small_spacing_generates_more_points(self):
        """Test that smaller spacing generates more interior points."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        points_coarse = _generate_interior_points_for_mesh(poly, point_spacing=5.0)
        points_fine = _generate_interior_points_for_mesh(poly, point_spacing=2.0)

        assert points_fine.shape[0] > points_coarse.shape[0]

    def test_empty_result_for_tiny_polygon(self):
        """Test that very small polygon with large spacing returns empty array."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        points = _generate_interior_points_for_mesh(poly, point_spacing=10.0)

        assert points.shape[0] == 0  # No interior points fit
        assert points.shape == (0, 2)

    def test_triangle_polygon(self):
        """Test generating interior points for a triangular polygon."""
        poly = Polygon([(0, 0), (10, 0), (5, 10)])
        points = _generate_interior_points_for_mesh(poly, point_spacing=2.0)

        assert points.shape[1] == 2
        assert points.shape[0] > 0


class TestSamplePolygonBoundary:
    """Tests for _sample_polygon_boundary helper function."""

    def test_simple_square(self):
        """Test sampling boundary of a simple square."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        boundary_pts = _sample_polygon_boundary(poly, point_spacing=2.0)

        # Should have multiple points along the boundary
        assert boundary_pts.shape[0] > 4  # More than just the vertices
        assert boundary_pts.shape[1] == 2

    def test_boundary_includes_vertices(self):
        """Test that boundary sampling includes original vertices."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        boundary_pts = _sample_polygon_boundary(poly, point_spacing=15.0)

        # With large spacing, should at least have the 4 corner vertices
        assert boundary_pts.shape[0] >= 4

    def test_deduplication(self):
        """Test that duplicate points are removed."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        boundary_pts = _sample_polygon_boundary(poly, point_spacing=5.0)

        # Check no exact duplicates
        unique_pts = np.unique(boundary_pts, axis=0)
        assert unique_pts.shape[0] == boundary_pts.shape[0]


class TestTriangulatePoints:
    """Tests for _triangulate_points helper function."""

    def test_basic_triangulation(self):
        """Test basic Delaunay triangulation."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tri = _triangulate_points(points)

        assert isinstance(tri, Delaunay)
        assert tri.points.shape == (4, 2)
        assert tri.simplices.shape[0] == 2  # 2 triangles for a square

    def test_insufficient_points_raises_error(self):
        """Test that < 3 points raises ValueError."""
        points = np.array([[0, 0], [1, 0]], dtype=float)

        with pytest.raises(ValueError, match="at least 3 points"):
            _triangulate_points(points)

    def test_collinear_points_raises_error(self):
        """Test that collinear points raise ValueError."""
        points = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)

        with pytest.raises(ValueError, match="triangulation failed"):
            _triangulate_points(points)


class TestFilterActiveSimplicesByCentroid:
    """Tests for _filter_active_simplices_by_centroid helper function."""

    def test_all_triangles_inside(self):
        """Test when all triangle centroids are inside the polygon."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tri = Delaunay(points)
        poly = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])  # Large bounding box

        active_indices, centroids = _filter_active_simplices_by_centroid(tri, poly)

        # Both triangles should be active
        assert active_indices.shape[0] == 2
        assert centroids.shape[0] == 2

    def test_some_triangles_outside(self):
        """Test when some triangle centroids are outside the polygon."""
        points = np.array([[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2]], dtype=float)
        tri = Delaunay(points)
        # Small polygon that only covers left side
        poly = Polygon([(0, 0), (2.5, 0), (2.5, 3), (0, 3)])

        active_indices, _centroids = _filter_active_simplices_by_centroid(tri, poly)

        # Only some triangles should be active
        assert active_indices.shape[0] < tri.simplices.shape[0]
        assert active_indices.shape[0] > 0

    def test_no_triangles_inside(self):
        """Test when no triangle centroids are inside the polygon."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tri = Delaunay(points)
        # Polygon far away from triangulation
        poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])

        active_indices, _centroids = _filter_active_simplices_by_centroid(tri, poly)

        assert active_indices.shape[0] == 0


class TestBuildMeshConnectivityGraph:
    """Tests for _build_mesh_connectivity_graph helper function."""

    def test_basic_connectivity(self):
        """Test building connectivity graph for simple triangulation."""
        points = np.array([[0, 0], [2, 0], [1, 2], [3, 2]], dtype=float)
        tri = Delaunay(points)
        poly = Polygon([(-1, -1), (4, -1), (4, 3), (-1, 3)])

        active_indices, centroids = _filter_active_simplices_by_centroid(tri, poly)
        orig_to_active = {orig: act for act, orig in enumerate(active_indices)}

        graph = _build_mesh_connectivity_graph(
            active_indices, centroids, orig_to_active, tri
        )

        # Should have one node per active triangle
        assert graph.number_of_nodes() == len(active_indices)
        # Nodes should have required attributes
        for node in graph.nodes():
            assert "pos" in graph.nodes[node]
            assert "source_grid_flat_index" in graph.nodes[node]
            assert "original_grid_nd_index" in graph.nodes[node]

    def test_edge_attributes(self):
        """Test that edges have required attributes."""
        points = np.array([[0, 0], [2, 0], [1, 1.5], [3, 1.5]], dtype=float)
        tri = Delaunay(points)
        poly = Polygon([(-1, -1), (4, -1), (4, 3), (-1, 3)])

        active_indices, centroids = _filter_active_simplices_by_centroid(tri, poly)
        orig_to_active = {orig: act for act, orig in enumerate(active_indices)}

        graph = _build_mesh_connectivity_graph(
            active_indices, centroids, orig_to_active, tri
        )

        # Check edge attributes
        for u, v in graph.edges():
            assert "distance" in graph.edges[u, v]
            assert "vector" in graph.edges[u, v]
            assert "angle_2d" in graph.edges[u, v]
            assert graph.edges[u, v]["distance"] > 0

    def test_no_duplicate_edges(self):
        """Test that graph has no duplicate edges (undirected)."""
        points = np.array([[0, 0], [2, 0], [1, 2], [3, 2]], dtype=float)
        tri = Delaunay(points)
        poly = Polygon([(-1, -1), (4, -1), (4, 3), (-1, 3)])

        active_indices, centroids = _filter_active_simplices_by_centroid(tri, poly)
        orig_to_active = {orig: act for act, orig in enumerate(active_indices)}

        graph = _build_mesh_connectivity_graph(
            active_indices, centroids, orig_to_active, tri
        )

        # Check that no reverse edges exist (since graph is undirected)
        edges_set = {tuple(sorted([u, v])) for u, v in graph.edges()}
        assert len(edges_set) == graph.number_of_edges()


class TestComputeMeshDimensionRanges:
    """Tests for _compute_mesh_dimension_ranges helper function."""

    def test_basic_dimension_ranges(self):
        """Test computing dimension ranges from bin centers."""
        bin_centers = np.array([[0, 0], [10, 5], [5, 10]], dtype=float)
        ranges = _compute_mesh_dimension_ranges(bin_centers)

        assert ranges is not None
        assert len(ranges) == 2  # 2D
        assert ranges[0] == (0.0, 10.0)  # X range
        assert ranges[1] == (0.0, 10.0)  # Y range

    def test_empty_bin_centers(self):
        """Test that empty bin centers returns None."""
        bin_centers = np.empty((0, 2), dtype=float)
        ranges = _compute_mesh_dimension_ranges(bin_centers)

        assert ranges is None

    def test_single_point(self):
        """Test dimension ranges with single point."""
        bin_centers = np.array([[5.0, 7.0]])
        ranges = _compute_mesh_dimension_ranges(bin_centers)

        assert ranges is not None
        assert ranges[0] == (5.0, 5.0)
        assert ranges[1] == (7.0, 7.0)


class TestTriangularMeshLayoutBuild:
    """Tests for TriangularMeshLayout.build() method."""

    def test_basic_build(self):
        """Test basic build with rectangular polygon."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        layout.build(poly, point_spacing=3.0)

        # Check that core attributes are populated
        assert layout.bin_centers.shape[0] > 0
        assert layout.bin_centers.shape[1] == 2
        assert layout.connectivity.number_of_nodes() == layout.bin_centers.shape[0]
        assert layout.dimension_ranges is not None
        assert len(layout.dimension_ranges) == 2

    def test_invalid_boundary_polygon_type(self):
        """Test that non-Polygon type raises TypeError."""
        layout = TriangularMeshLayout()

        with pytest.raises(TypeError, match="must be a Shapely Polygon"):
            layout.build("not a polygon", point_spacing=1.0)

    def test_empty_polygon_raises_error(self):
        """Test that empty polygon raises ValueError."""
        layout = TriangularMeshLayout()
        poly = Polygon()

        with pytest.raises(ValueError, match="cannot be empty"):
            layout.build(poly, point_spacing=1.0)

    def test_negative_point_spacing_raises_error(self):
        """Test that negative point_spacing raises ValueError."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        with pytest.raises(ValueError, match="must be positive"):
            layout.build(poly, point_spacing=-1.0)

    def test_insufficient_points_raises_error(self):
        """Test that tiny polygon with large spacing raises ValueError."""
        layout = TriangularMeshLayout()
        # Very small polygon with very large spacing
        poly = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])

        with pytest.raises(ValueError, match="Not enough interior sample points"):
            layout.build(poly, point_spacing=10.0)

    def test_build_params_stored(self):
        """Test that build parameters are stored correctly."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        layout.build(poly, point_spacing=2.5)

        assert layout._build_params_used["point_spacing"] == 2.5
        assert "boundary_exterior_coords" in layout._build_params_used
        assert "boundary_interior_coords_list" in layout._build_params_used

    def test_grid_shape_and_active_mask(self):
        """Test that grid_shape and active_mask are set correctly."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        layout.build(poly, point_spacing=3.0)

        assert layout.grid_shape is not None
        assert len(layout.grid_shape) == 1  # 1D shape for triangle list
        assert layout.active_mask is not None
        assert layout.active_mask.shape[0] == layout.grid_shape[0]
        assert np.sum(layout.active_mask) == layout.bin_centers.shape[0]


class TestTriangularMeshLayoutPointToBinIndex:
    """Tests for TriangularMeshLayout.point_to_bin_index() method."""

    def test_points_inside_triangles(self):
        """Test mapping points that fall inside triangles."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        # Points inside the polygon
        test_points = np.array([[5.0, 5.0], [2.0, 2.0], [8.0, 8.0]])
        bin_indices = layout.point_to_bin_index(test_points)

        assert bin_indices.shape[0] == 3
        # All points should map to valid bins (not -1)
        assert np.all(bin_indices >= 0)
        assert np.all(bin_indices < layout.bin_centers.shape[0])

    def test_points_outside_polygon(self):
        """Test that points outside polygon map to -1."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        # Points outside the polygon
        test_points = np.array([[20.0, 20.0], [-5.0, -5.0]])
        bin_indices = layout.point_to_bin_index(test_points)

        assert bin_indices.shape[0] == 2
        assert np.all(bin_indices == -1)

    def test_mixed_inside_outside_points(self):
        """Test mixture of points inside and outside."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        test_points = np.array([[5.0, 5.0], [20.0, 20.0], [2.0, 2.0]])
        bin_indices = layout.point_to_bin_index(test_points)

        assert bin_indices.shape[0] == 3
        # First and third should be valid, second should be -1
        assert bin_indices[0] >= 0
        assert bin_indices[1] == -1
        assert bin_indices[2] >= 0

    def test_single_point_input(self):
        """Test point_to_bin_index with single point."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        test_point = np.array([5.0, 5.0])
        bin_index = layout.point_to_bin_index(test_point)

        assert bin_index.shape == (1,)
        assert bin_index[0] >= 0

    def test_unbuild_layout_raises_error(self):
        """Test that calling point_to_bin_index before build raises RuntimeError."""
        layout = TriangularMeshLayout()
        test_points = np.array([[5.0, 5.0]])

        with pytest.raises(RuntimeError, match="not built"):
            layout.point_to_bin_index(test_points)

    def test_invalid_point_shape_raises_error(self):
        """Test that invalid point shape raises ValueError."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        # 3D points instead of 2D
        test_points = np.array([[5.0, 5.0, 5.0]])

        with pytest.raises(ValueError, match="shape \\(M, 2\\)"):
            layout.point_to_bin_index(test_points)


class TestTriangularMeshLayoutProperties:
    """Tests for TriangularMeshLayout properties."""

    def test_is_1d_property(self):
        """Test that is_1d always returns False."""
        layout = TriangularMeshLayout()
        assert layout.is_1d is False

        # Even after building
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)
        assert layout.is_1d is False

    def test_layout_type_tag(self):
        """Test that _layout_type_tag is set correctly."""
        layout = TriangularMeshLayout()
        assert layout._layout_type_tag == "TriangularMesh"


class TestTriangularMeshLayoutBinSizes:
    """Tests for TriangularMeshLayout.bin_sizes() method."""

    def test_bin_sizes_basic(self):
        """Test that bin_sizes returns areas for all active triangles."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        sizes = layout.bin_sizes()

        assert sizes.shape[0] == layout.bin_centers.shape[0]
        assert np.all(sizes > 0)  # All triangles should have positive area

    def test_bin_sizes_unbuild_raises_error(self):
        """Test that calling bin_sizes before build raises RuntimeError."""
        layout = TriangularMeshLayout()

        with pytest.raises(RuntimeError, match="not built"):
            layout.bin_sizes()

    def test_bin_sizes_reasonable_values(self):
        """Test that bin sizes are reasonable given the spacing."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        spacing = 2.0
        layout.build(poly, point_spacing=spacing)

        sizes = layout.bin_sizes()

        # Triangle areas should be roughly on the order of spacing^2
        # (exact values depend on triangulation, but order of magnitude check)
        assert np.all(sizes < spacing**2 * 10)  # Upper bound
        assert np.all(sizes > 0.01)  # Lower bound (not too tiny)


class TestTriangularMeshLayoutPlot:
    """Tests for TriangularMeshLayout.plot() method."""

    def test_basic_plot(self):
        """Test basic plotting functionality."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        fig, ax = plt.subplots()
        returned_ax = layout.plot(ax=ax)

        assert returned_ax is ax
        # Check that something was plotted
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        plt.close(fig)

    def test_plot_creates_axes_if_none(self):
        """Test that plot creates axes if none provided."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        returned_ax = layout.plot()

        assert returned_ax is not None
        plt.close()

    def test_plot_unbuild_raises_error(self):
        """Test that calling plot before build raises RuntimeError."""
        layout = TriangularMeshLayout()

        with pytest.raises(RuntimeError, match="not built"):
            layout.plot()

    def test_plot_with_custom_kwargs(self):
        """Test plot with custom styling kwargs."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        fig, ax = plt.subplots()
        layout.plot(
            ax=ax,
            show_triangles=True,
            show_centroids=True,
            show_connectivity=True,
            show_boundary=True,
            triangle_kwargs={"alpha": 0.2},
            centroid_kwargs={"color": "red"},
        )

        # Should not raise any errors
        plt.close(fig)

    def test_plot_selective_display(self):
        """Test plot with selective display options."""
        layout = TriangularMeshLayout()
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(poly, point_spacing=3.0)

        fig, ax = plt.subplots()
        layout.plot(
            ax=ax,
            show_triangles=False,
            show_centroids=True,
            show_connectivity=False,
            show_boundary=False,
        )

        # Should only show centroids
        plt.close(fig)
