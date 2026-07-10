"""Tests for error paths in Environment class.

This module tests error handling and edge cases that were previously untested,
particularly for methods like path_between() and bins_in_region().
"""

import numpy as np
import pytest

from neurospatial.environment import Environment

# Test data for creating environments
_rng = np.random.default_rng(42)
SAMPLE_DATA_2D = _rng.standard_normal((100, 2)) * 10


class TestPathBetweenErrorPaths:
    """Test error handling in Environment.path_between()."""

    def test_shortest_path_no_path_disconnected_components(self):
        """Test path_between when nodes are in disconnected components."""
        # Create environment with two disconnected regions
        data1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        data2 = np.array([[100, 100], [101, 100], [100, 101], [101, 101]])
        data = np.vstack([data1, data2])

        env = Environment.from_samples(data, bin_size=1.5)

        # Find bins in each disconnected component
        bin1 = env.bin_at(np.array([[0.5, 0.5]]))[0]
        bin2 = env.bin_at(np.array([[100.5, 100.5]]))[0]

        # Verify bins are valid but in different components
        assert bin1 >= 0
        assert bin2 >= 0
        assert bin1 != bin2

        # Should warn and return empty path when no path exists
        with pytest.warns(UserWarning, match="No path found"):
            path = env.path_between(bin1, bin2)

        assert path == []
        assert isinstance(path, list)

    def test_shortest_path_invalid_source_node(self):
        """Test path_between with invalid source node index."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        invalid_source = env.n_bins + 100  # Way out of range
        valid_target = 0

        with pytest.raises(IndexError, match=r"out of range"):
            env.path_between(invalid_source, valid_target)

    def test_shortest_path_invalid_target_node(self):
        """Test path_between with invalid target node index."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        valid_source = 0
        invalid_target = -999  # Negative index

        with pytest.raises(IndexError, match=r"out of range"):
            env.path_between(valid_source, invalid_target)

    def test_shortest_path_both_nodes_invalid(self):
        """Test path_between when both nodes are invalid."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        invalid_source = env.n_bins + 50
        invalid_target = env.n_bins + 100

        with pytest.raises(IndexError):
            env.path_between(invalid_source, invalid_target)

    def test_shortest_path_same_node(self):
        """Test path_between when source and target are the same."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Path from node to itself should work
        path = env.path_between(5, 5)
        assert path == [5]

    def test_shortest_path_adjacent_nodes(self):
        """Test path_between between adjacent nodes."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Get first node and one of its neighbors
        source = 0
        neighbors = env.neighbors(source)

        if len(neighbors) > 0:
            target = neighbors[0]
            path = env.path_between(source, target)

            assert len(path) == 2
            assert path[0] == source
            assert path[1] == target


class TestBinsInRegionErrorPaths:
    """Test error handling in Environment.bins_in_region()."""

    def test_bins_in_region_polygon_3d_environment(self):
        """Test polygon region query on 3D environment raises clear error."""
        from shapely.geometry import Polygon as ShapelyPolygon

        rng = np.random.default_rng(42)
        # Create 3D environment
        data_3d = rng.standard_normal((100, 3)) * 10
        env = Environment.from_samples(data_3d, bin_size=2.0)

        # Add polygon region (only valid for 2D)
        polygon_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shapely_poly = ShapelyPolygon(polygon_coords)
        env.regions.add("test_poly", polygon=shapely_poly)

        # Should raise ValueError for 3D environment
        with pytest.raises(ValueError, match=r"Polygon regions.*only supported.*2D"):
            env.bins_in_region("test_poly")

    def test_bins_in_region_polygon_without_shapely(self, monkeypatch):
        """Test polygon region query when shapely not available."""
        from shapely.geometry import Polygon as ShapelyPolygon

        # Create 2D environment
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Add polygon region
        polygon_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        shapely_poly = ShapelyPolygon(polygon_coords)
        env.regions.add("test_poly", polygon=shapely_poly)

        # Mock shapely as unavailable
        import neurospatial.environment as env_module

        monkeypatch.setattr(env_module, "_HAS_SHAPELY", False)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Polygon region queries require"):
            env.bins_in_region("test_poly")

    def test_bins_in_region_nonexistent_region(self):
        """Test bins_in_region with region name that doesn't exist."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent_region"):
            env.bins_in_region("nonexistent_region")

    def test_bins_in_region_point_region(self):
        """Test bins_in_region with point region (should work)."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Use a point that's definitely in an active bin (first bin center)
        point_in_bin = env.bin_centers[0]
        env.regions.add("center", point=point_in_bin.tolist())

        # Should return bins near the point
        bins = env.bins_in_region("center")
        assert isinstance(bins, np.ndarray)
        assert bins.dtype == np.int_
        # Point in active bin should return at least one bin
        assert len(bins) >= 1

    def test_bins_in_region_polygon_2d_valid(self):
        """Test bins_in_region with valid polygon region in 2D."""
        from shapely.geometry import Polygon as ShapelyPolygon

        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Add polygon region
        polygon_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        shapely_poly = ShapelyPolygon(polygon_coords)
        env.regions.add("center_box", polygon=shapely_poly)

        # Should return bins within the polygon
        bins = env.bins_in_region("center_box")
        assert isinstance(bins, np.ndarray)
        assert bins.dtype == np.int_
        assert len(bins) > 0  # Should have some bins in this region


class TestMaskForRegionErrorPaths:
    """Test error handling in Environment.region_mask()."""

    def test_mask_for_region_nonexistent_region(self):
        """Test region_mask with region that doesn't exist."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        with pytest.raises(KeyError, match="nonexistent"):
            env.region_mask("nonexistent")

    def test_mask_for_region_point_region(self):
        """Test region_mask with point region."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Use a point that's definitely in an active bin
        point_in_bin = env.bin_centers[0]
        env.regions.add("test_point", point=point_in_bin.tolist())

        mask = env.region_mask("test_point")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (env.n_bins,)
        assert np.sum(mask) >= 1  # At least one bin selected

    def test_mask_for_region_polygon_region(self):
        """Test region_mask with polygon region."""
        from shapely.geometry import Polygon as ShapelyPolygon

        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        polygon_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        shapely_poly = ShapelyPolygon(polygon_coords)
        env.regions.add("box", polygon=shapely_poly)

        mask = env.region_mask("box")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (env.n_bins,)


class TestContainsErrorPaths:
    """Test error handling in Environment.contains()."""

    def test_contains_wrong_dimensionality_raises(self):
        """Test contains with points of wrong dimensionality raises ValueError."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Try 1D point for 2D environment - this now raises loudly
        point_1d = np.array([[5.0]])

        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            env.contains(point_1d)

    def test_contains_wrong_shape(self):
        """Test contains with malformed point array."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # 1D array instead of 2D
        point_wrong = np.array([5.0, 10.0])

        # Should handle gracefully
        result = env.contains(point_wrong)
        assert isinstance(result, (bool, np.ndarray))


class TestNeighborsErrorPaths:
    """Test error handling in Environment.neighbors()."""

    def test_neighbors_invalid_bin_index(self):
        """Test neighbors with invalid bin index."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        invalid_idx = env.n_bins + 100

        with pytest.raises(IndexError, match=r"out of range"):
            env.neighbors(invalid_idx)

    def test_neighbors_negative_index(self):
        """Test neighbors with negative bin index."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        with pytest.raises(IndexError, match=r"out of range"):
            env.neighbors(-1)

    def test_neighbors_valid_isolated_node(self):
        """Test neighbors for node with no neighbors (if such exists)."""
        # This is an edge case - most environments don't have isolated nodes
        # but we test the interface works
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Get any valid node
        valid_node = 0
        neighbors = env.neighbors(valid_node)

        assert isinstance(neighbors, list)
        # For regular grids, nodes typically have neighbors


class TestGraphQueriesAcceptCoordinates:
    """neighbors/path_between/reachable_from accept a point OR a bin index.

    Resolves the coordinate-vs-index footgun: these graph queries now accept a
    coordinate array (mapped via bin_at) just like the geometric methods, while
    still accepting integer bin indices. A point outside the environment raises.
    """

    def test_coordinate_and_index_agree(self):
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)
        # Use a CONNECTED pair so path_between returns a non-empty path -- an
        # equivalence over two empty lists (disconnected bins) would be vacuous
        # and prove nothing about coordinate resolution.
        idx_a = 3
        idx_b = env.neighbors(idx_a)[0]  # a bin directly connected to idx_a
        coord_a = env.bin_centers[idx_a]
        coord_b = env.bin_centers[idx_b]

        assert sorted(env.neighbors(coord_a)) == sorted(env.neighbors(idx_a))

        path_from_coords = env.path_between(coord_a, coord_b)
        assert path_from_coords == env.path_between(idx_a, idx_b)
        assert len(path_from_coords) >= 2  # a real path, not vacuous [] == []

        assert np.array_equal(
            env.reachable_from(coord_a, radius=2),
            env.reachable_from(idx_a, radius=2),
        )

    def test_point_outside_environment_raises(self):
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)
        far = np.array([1e6, 1e6])
        with pytest.raises(ValueError, match=r"not inside any active bin"):
            env.neighbors(far)

    def test_batch_of_coordinates_raises(self):
        """A multi-point coordinate array is rejected, not silently truncated.

        Regression: ``_resolve_point_or_index`` resolved the whole batch via
        ``bin_at`` but kept only the first bin, so ``neighbors(bin_centers[[0,
        3]])`` silently returned bin 0's neighbors. A batch must fail loud.
        """
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)
        batch = env.bin_centers[[0, 3]]  # two points -> shape (2, n_dims)
        with pytest.raises(ValueError, match=r"single coordinate point"):
            env.neighbors(batch)


class TestDistanceBetweenErrorPaths:
    """Test error handling in Environment.distance_between()."""

    def test_distance_between_points_outside_environment(self):
        """Test distance_between with points outside environment returns inf."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Points far outside the environment
        far_point1 = np.array([1000.0, 1000.0])
        far_point2 = np.array([1100.0, 1100.0])

        # Should return infinity when points map to invalid bins
        dist = env.distance_between(far_point1, far_point2)
        assert np.isinf(dist)


class TestBinAtErrorPaths:
    """Test error handling in Environment.bin_at()."""

    def test_bin_at_wrong_dimensionality_raises(self):
        """Test bin_at with points of wrong dimensionality raises ValueError."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Try 1D points for 2D environment - now raises loudly
        points_1d = np.array([[5.0], [10.0]])

        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            env.bin_at(points_1d)

    def test_bin_at_linearized_positions_on_graph_env_raises(self):
        """bin_at on a graph/track env raises on 1-D linearized positions.

        Regression: a linearized-track environment routes bin_at through the
        KDTree path, which swallowed a dimension mismatch to a warning and
        returned -1 for every point -- so feeding ``to_linear()`` output (1-D)
        back in silently produced an empty/all-zero field. It must fail loud,
        like the grid path, and point at the real fix.
        """
        env = Environment.linear_track(
            endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=5.0
        )
        assert env.is_linearized_track and env.n_dims == 2

        positions_2d = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        linear = np.asarray(env.to_linear(positions_2d)).reshape(-1, 1)  # (50, 1)

        with pytest.raises(ValueError, match=r"(?i)dimension|to_linear"):
            env.bin_at(linear)

    def test_bin_at_empty_array(self):
        """Test bin_at with empty point array."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        empty_points = np.empty((0, 2))
        result = env.bin_at(empty_points)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_bin_at_outside_environment(self):
        """Test bin_at with points far outside environment."""
        env = Environment.from_samples(SAMPLE_DATA_2D, bin_size=2.0)

        # Points very far from environment
        far_points = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
        result = env.bin_at(far_points)

        # Should return -1 for points outside
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        # Points outside should have -1
        assert all(idx == -1 for idx in result)
