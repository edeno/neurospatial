"""Tests for neurospatial.spatial query utilities."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.spatial import clear_kdtree_cache, map_points_to_bins


class TestMapPointsToBins:
    """Test map_points_to_bins function."""

    @pytest.fixture
    def grid_env(self):
        """Create a simple grid environment."""
        np.random.seed(42)
        # Create data on a regular grid
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=2.0, name="grid")
        return env

    def test_map_points_basic(self, grid_env):
        """Test basic point mapping."""
        points = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (3,)
        assert bins.dtype == np.int_
        # All points should map to valid bins (>= 0)
        assert np.all(bins >= 0)

    def test_map_points_with_distances(self, grid_env):
        """Test that return_dist=True returns distances."""
        points = np.array([[5.0, 5.0]])
        bins, dists = map_points_to_bins(points, grid_env, return_dist=True)

        assert bins.shape == (1,)
        assert dists.shape == (1,)
        assert dists[0] < 2.0  # Should be close to bin center

    def test_tie_break_lowest_index(self, grid_env):
        """Test deterministic tie-breaking."""
        # Point exactly on boundary between bins
        points = np.array([[1.0, 1.0]])

        bins1 = map_points_to_bins(points, grid_env, tie_break="lowest_index")
        bins2 = map_points_to_bins(points, grid_env, tie_break="lowest_index")

        # Should be deterministic
        assert bins1[0] == bins2[0]

    def test_tie_break_closest_center(self, grid_env):
        """Test closest_center tie-breaking mode."""
        points = np.array([[5.0, 5.0]])
        bins = map_points_to_bins(points, grid_env, tie_break="closest_center")

        assert bins.shape == (1,)
        assert bins[0] >= 0

    def test_invalid_tie_break_raises_error(self, grid_env):
        """Test that invalid tie_break mode raises error."""
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match="Invalid tie_break mode"):
            map_points_to_bins(points, grid_env, tie_break="invalid")

    def test_kdtree_caching(self, grid_env):
        """Test that KD-tree is cached on environment."""
        assert not hasattr(grid_env, "_kdtree_cache") or grid_env._kdtree_cache is None

        points = np.array([[5.0, 5.0]])
        map_points_to_bins(points, grid_env)

        # Cache should now exist
        assert hasattr(grid_env, "_kdtree_cache")
        assert grid_env._kdtree_cache is not None

    def test_clear_kdtree_cache(self, grid_env):
        """Test clearing KD-tree cache."""
        points = np.array([[5.0, 5.0]])
        map_points_to_bins(points, grid_env)

        assert grid_env._kdtree_cache is not None

        clear_kdtree_cache(grid_env)
        assert grid_env._kdtree_cache is None

    def test_out_of_bounds_points(self, grid_env):
        """Test that far out-of-bounds points are marked as -1."""
        # Points very far from environment
        points = np.array([[1000.0, 1000.0], [5.0, 5.0]])
        bins = map_points_to_bins(points, grid_env)

        # Far point should be -1, close point should be valid
        assert bins[0] == -1
        assert bins[1] >= 0

    def test_batch_mapping_performance(self, grid_env):
        """Test that batch mapping works with many points."""
        np.random.seed(42)
        points = np.random.rand(1000, 2) * 10

        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (1000,)
        assert np.all((bins >= -1) & (bins < grid_env.n_bins))

    def test_single_point(self, grid_env):
        """Test mapping a single point."""
        points = np.array([[5.0, 5.0]])
        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (1,)
        assert bins[0] >= 0
