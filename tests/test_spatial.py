"""Tests for neurospatial.spatial query utilities."""

import numpy as np
import pytest

from neurospatial import Environment


class TestNearestBin:
    """Test Environment.nearest_bin."""

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
        bins = grid_env.nearest_bin(points)

        assert bins.shape == (3,)
        assert bins.dtype == np.int_
        # All points should map to valid bins (>= 0)
        assert np.all(bins >= 0)

    def test_map_points_with_distances(self, grid_env):
        """Test that return_dist=True returns distances."""
        points = np.array([[5.0, 5.0]])
        bins, dists = grid_env.nearest_bin(points, return_dist=True)

        assert bins.shape == (1,)
        assert dists.shape == (1,)
        assert dists[0] < 2.0  # Should be close to bin center

    def test_tie_break_lowest_index(self, grid_env):
        """Test deterministic tie-breaking."""
        # Point exactly on boundary between bins
        points = np.array([[1.0, 1.0]])

        bins1 = grid_env.nearest_bin(points, tie_break="lowest_index")
        bins2 = grid_env.nearest_bin(points, tie_break="lowest_index")

        # Should be deterministic
        assert bins1[0] == bins2[0]

    def test_tie_break_closest_center(self, grid_env):
        """Test closest_center tie-breaking mode."""
        points = np.array([[5.0, 5.0]])
        bins = grid_env.nearest_bin(points, tie_break="closest_center")

        assert bins.shape == (1,)
        assert bins[0] >= 0

    def test_invalid_tie_break_raises_error(self, grid_env):
        """Test that invalid tie_break mode raises error."""
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match="Invalid tie_break value"):
            grid_env.nearest_bin(points, tie_break="invalid")

    def test_kdtree_caching_behavior(self, grid_env):
        """Test that repeated calls produce identical results (caching behavior)."""
        points = np.array([[5.0, 5.0], [2.0, 3.0], [8.0, 7.0]])

        # First call
        bins1 = grid_env.nearest_bin(points)

        # Second call should return identical results
        bins2 = grid_env.nearest_bin(points)

        np.testing.assert_array_equal(bins1, bins2)

    def test_clear_cache_behavior(self, grid_env):
        """Test that clear_cache() allows recomputation with identical results."""
        points = np.array([[5.0, 5.0], [2.0, 3.0], [8.0, 7.0]])

        # First mapping
        bins_before = grid_env.nearest_bin(points)

        # Clear cache
        grid_env.clear_cache()

        # Mapping after clear should return identical results
        bins_after = grid_env.nearest_bin(points)

        np.testing.assert_array_equal(bins_before, bins_after)

    def test_out_of_bounds_points(self, grid_env):
        """Test that far out-of-bounds points are marked as -1."""
        # Points very far from environment
        points = np.array([[1000.0, 1000.0], [5.0, 5.0]])
        bins = grid_env.nearest_bin(points)

        # Far point should be -1, close point should be valid
        assert bins[0] == -1
        assert bins[1] >= 0

    def test_batch_mapping_performance(self, grid_env):
        """Test that batch mapping works with many points."""
        rng = np.random.default_rng(42)
        points = rng.random((1000, 2)) * 10

        bins = grid_env.nearest_bin(points)

        assert bins.shape == (1000,)
        assert np.all((bins >= -1) & (bins < grid_env.n_bins))

    def test_single_point(self, grid_env):
        """Test mapping a single point."""
        points = np.array([[5.0, 5.0]])
        bins = grid_env.nearest_bin(points)

        assert bins.shape == (1,)
        assert bins[0] >= 0


class TestEnvironmentClearCache:
    """Test the behavioral contract of Environment.clear_cache().

    - After clearing, cached computations are recomputed on next access
    - Results after clearing are identical to results before clearing
    """

    @pytest.fixture
    def env_with_caches(self):
        """Create environment and populate various caches."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 100
        env = Environment.from_samples(data, bin_size=5.0)

        # Populate KDTree cache via nearest_bin
        points = np.array([[50.0, 50.0]])
        env.nearest_bin(points)

        # Populate kernel cache
        _ = env.compute_kernel(bandwidth=10.0, mode="density")

        # Populate cached properties
        _ = env.get_differential_operator()
        _ = env.boundary_bins
        _ = env.bin_sizes

        return env

    def _to_array(self, m):
        """Convert sparse or dense matrix to dense array for comparison."""
        import scipy.sparse

        if scipy.sparse.issparse(m):
            return m.toarray()
        return np.asarray(m)

    def test_clear_cache_all_recomputes_correctly(self, env_with_caches):
        """Test that clearing all caches allows correct recomputation."""
        env = env_with_caches
        points = np.array([[50.0, 50.0], [30.0, 70.0]])

        # Get values before clearing
        bins_before = env.nearest_bin(points)
        kernel_before = env.compute_kernel(bandwidth=10.0, mode="density")
        diff_op_before = env.get_differential_operator().copy()
        boundary_before = env.boundary_bins.copy()

        # Clear all caches
        env.clear_cache()

        # Recompute and verify identical results
        bins_after = env.nearest_bin(points)
        kernel_after = env.compute_kernel(bandwidth=10.0, mode="density")
        diff_op_after = env.get_differential_operator()
        boundary_after = env.boundary_bins

        np.testing.assert_array_equal(bins_before, bins_after)
        np.testing.assert_array_almost_equal(
            self._to_array(kernel_before), self._to_array(kernel_after)
        )
        np.testing.assert_array_almost_equal(
            self._to_array(diff_op_before), self._to_array(diff_op_after)
        )
        np.testing.assert_array_equal(boundary_before, boundary_after)

    def test_clear_cache_on_fresh_environment(self):
        """Test clear_cache() on environment with no caches populated."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 2)) * 50
        env = Environment.from_samples(data, bin_size=5.0)

        # Should not raise error on fresh environment
        env.clear_cache()

        # Environment should still function correctly after clearing
        points = np.array([[25.0, 25.0]])
        bins = env.nearest_bin(points)
        assert bins[0] >= 0  # Valid bin mapping

    def test_clear_cache_different_environments_independent(self):
        """Test that clearing cache on one environment doesn't affect another."""
        rng = np.random.default_rng(42)
        data1 = rng.random((50, 2)) * 50
        env1 = Environment.from_samples(data1, bin_size=5.0)
        data2 = rng.random((50, 2)) * 50
        env2 = Environment.from_samples(data2, bin_size=5.0)

        points = np.array([[25.0, 25.0]])

        # Populate caches on both
        bins1_before = env1.nearest_bin(points)
        bins2_before = env2.nearest_bin(points)

        # Clear only env1
        env1.clear_cache()

        # env1 should recompute correctly, env2 should be unaffected
        bins1_after = env1.nearest_bin(points)
        bins2_after = env2.nearest_bin(points)

        np.testing.assert_array_equal(bins1_before, bins1_after)
        np.testing.assert_array_equal(bins2_before, bins2_after)
