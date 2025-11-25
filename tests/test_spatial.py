"""Tests for neurospatial.spatial query utilities."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.spatial import map_points_to_bins


class TestMapPointsToBins:
    """Test map_points_to_bins function."""

    @pytest.fixture
    def grid_env(self):
        """Create a simple grid environment."""
        # Create data on a regular grid (deterministic, no RNG needed)
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

        with pytest.raises(ValueError, match="Invalid tie_break value"):
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
        """Test clearing KD-tree cache using env.clear_cache()."""
        points = np.array([[5.0, 5.0]])
        map_points_to_bins(points, grid_env)

        assert grid_env._kdtree_cache is not None

        # Use env.clear_cache() with selective clearing
        grid_env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
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
        rng = np.random.default_rng(42)
        points = rng.random((1000, 2)) * 10

        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (1000,)
        assert np.all((bins >= -1) & (bins < grid_env.n_bins))

    def test_single_point(self, grid_env):
        """Test mapping a single point."""
        points = np.array([[5.0, 5.0]])
        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (1,)
        assert bins[0] >= 0


class TestEnvironmentClearCache:
    """Test Environment.clear_cache() method with selective clearing."""

    @pytest.fixture
    def env_with_caches(self):
        """Create environment and populate various caches."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 100
        env = Environment.from_samples(data, bin_size=5.0)

        # Populate KDTree cache
        points = np.array([[50.0, 50.0]])
        map_points_to_bins(points, env)

        # Populate kernel cache
        _ = env.compute_kernel(bandwidth=10.0)

        # Populate cached properties
        _ = env.differential_operator
        _ = env.boundary_bins
        _ = env.bin_sizes

        return env

    def test_clear_cache_clears_all_by_default(self, env_with_caches):
        """Test that clear_cache() with no args clears all caches."""
        env = env_with_caches

        # Verify caches exist
        assert hasattr(env, "_kdtree_cache") and env._kdtree_cache is not None
        assert len(env._kernel_cache) > 0
        assert "differential_operator" in env.__dict__
        assert "boundary_bins" in env.__dict__

        # Clear all caches
        env.clear_cache()

        # Verify all caches cleared
        assert env._kdtree_cache is None
        assert len(env._kernel_cache) == 0
        assert "differential_operator" not in env.__dict__
        assert "boundary_bins" not in env.__dict__

    def test_clear_cache_kdtree_only(self, env_with_caches):
        """Test selective clearing of KDTree cache only."""
        env = env_with_caches

        # Verify caches exist
        assert env._kdtree_cache is not None
        assert len(env._kernel_cache) > 0
        assert "differential_operator" in env.__dict__

        # Clear only KDTree cache
        env.clear_cache(kdtree=True, kernels=False, cached_properties=False)

        # Verify only KDTree cache cleared
        assert env._kdtree_cache is None
        assert len(env._kernel_cache) > 0  # Should still exist
        assert "differential_operator" in env.__dict__  # Should still exist

    def test_clear_cache_kernels_only(self, env_with_caches):
        """Test selective clearing of kernel cache only."""
        env = env_with_caches

        # Verify caches exist
        assert env._kdtree_cache is not None
        assert len(env._kernel_cache) > 0
        assert "differential_operator" in env.__dict__

        # Clear only kernel cache
        env.clear_cache(kdtree=False, kernels=True, cached_properties=False)

        # Verify only kernel cache cleared
        assert env._kdtree_cache is not None  # Should still exist
        assert len(env._kernel_cache) == 0  # Should be cleared
        assert "differential_operator" in env.__dict__  # Should still exist

    def test_clear_cache_cached_properties_only(self, env_with_caches):
        """Test selective clearing of cached properties only."""
        env = env_with_caches

        # Verify caches exist
        assert env._kdtree_cache is not None
        assert len(env._kernel_cache) > 0
        assert "differential_operator" in env.__dict__
        assert "boundary_bins" in env.__dict__

        # Clear only cached properties
        env.clear_cache(kdtree=False, kernels=False, cached_properties=True)

        # Verify only cached properties cleared
        assert env._kdtree_cache is not None  # Should still exist
        assert len(env._kernel_cache) > 0  # Should still exist
        assert "differential_operator" not in env.__dict__  # Should be cleared
        assert "boundary_bins" not in env.__dict__  # Should be cleared

    def test_clear_cache_multiple_selective(self, env_with_caches):
        """Test clearing multiple cache types selectively."""
        env = env_with_caches

        # Clear KDTree and kernels, keep cached properties
        env.clear_cache(kdtree=True, kernels=True, cached_properties=False)

        # Verify correct caches cleared
        assert env._kdtree_cache is None
        assert len(env._kernel_cache) == 0
        assert "differential_operator" in env.__dict__  # Should still exist

    def test_clear_cache_none_selected(self, env_with_caches):
        """Test that clear_cache() with all False doesn't clear anything."""
        env = env_with_caches

        # Store initial state
        kdtree_before = env._kdtree_cache
        kernel_count_before = len(env._kernel_cache)
        has_diff_op_before = "differential_operator" in env.__dict__

        # Clear nothing
        env.clear_cache(kdtree=False, kernels=False, cached_properties=False)

        # Verify nothing changed
        assert env._kdtree_cache is kdtree_before
        assert len(env._kernel_cache) == kernel_count_before
        assert ("differential_operator" in env.__dict__) == has_diff_op_before

    def test_clear_cache_multiple_calls(self, env_with_caches):
        """Test that multiple clear_cache() calls work correctly."""
        env = env_with_caches

        # First clear: kdtree only
        env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
        assert env._kdtree_cache is None
        assert len(env._kernel_cache) > 0

        # Second clear: kernels only
        env.clear_cache(kdtree=False, kernels=True, cached_properties=False)
        assert env._kdtree_cache is None  # Still None
        assert len(env._kernel_cache) == 0  # Now cleared

        # Third clear: cached properties
        env.clear_cache(kdtree=False, kernels=False, cached_properties=True)
        assert "differential_operator" not in env.__dict__

    def test_clear_cache_on_fresh_environment(self):
        """Test clear_cache() on environment with no caches populated."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 2)) * 50
        env = Environment.from_samples(data, bin_size=5.0)

        # Should not raise error on fresh environment
        env.clear_cache()

        # Verify state is clean
        assert not hasattr(env, "_kdtree_cache") or env._kdtree_cache is None
        assert len(env._kernel_cache) == 0

    def test_clear_cache_recomputes_cached_properties(self, env_with_caches):
        """Test that cached properties are recomputed after clearing."""
        env = env_with_caches

        # Get values before clearing
        diff_op_before = env.differential_operator
        boundary_before = env.boundary_bins.copy()

        # Clear cached properties
        env.clear_cache(cached_properties=True)
        assert "differential_operator" not in env.__dict__

        # Access should recompute
        diff_op_after = env.differential_operator
        boundary_after = env.boundary_bins

        # Values should be the same (recomputed correctly)
        assert np.array_equal(diff_op_before.toarray(), diff_op_after.toarray())
        assert np.array_equal(boundary_before, boundary_after)

    def test_clear_cache_different_environments_independent(self):
        """Test that clearing cache on one environment doesn't affect another."""
        rng = np.random.default_rng(42)
        data1 = rng.random((50, 2)) * 50
        env1 = Environment.from_samples(data1, bin_size=5.0)
        data2 = rng.random((50, 2)) * 50
        env2 = Environment.from_samples(data2, bin_size=5.0)

        # Populate caches on both
        map_points_to_bins(np.array([[25.0, 25.0]]), env1)
        map_points_to_bins(np.array([[25.0, 25.0]]), env2)

        assert env1._kdtree_cache is not None
        assert env2._kdtree_cache is not None

        # Clear only env1
        env1.clear_cache()

        # env1 should be cleared, env2 should still have cache
        assert env1._kdtree_cache is None
        assert env2._kdtree_cache is not None
