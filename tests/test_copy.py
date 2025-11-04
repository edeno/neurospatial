"""
Tests for Environment.copy() method.

Following TDD - these tests are written BEFORE implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.regions import Regions

# ============================================================================
# Test copy() - Basic Functionality
# ============================================================================


class TestCopyBasic:
    """Test basic copy() functionality."""

    def test_copy_creates_new_instance(self, sample_2d_grid):
        """copy() creates a new Environment instance."""
        env = sample_2d_grid
        env_copy = env.copy()

        assert isinstance(env_copy, Environment)
        assert env_copy is not env  # Different objects

    def test_copy_preserves_attributes(self, sample_2d_grid):
        """copy() preserves all environment attributes."""
        env = sample_2d_grid
        env_copy = env.copy()

        # Basic attributes
        assert env_copy.n_bins == env.n_bins
        assert env_copy.n_dims == env.n_dims
        assert env_copy.name == env.name
        assert env_copy.units == env.units
        assert env_copy.frame == env.frame

        # Bin centers (values equal, but arrays may be different objects)
        assert_array_equal(env_copy.bin_centers, env.bin_centers)

        # Dimension ranges
        assert len(env_copy.dimension_ranges) == len(env.dimension_ranges)
        for i in range(len(env.dimension_ranges)):
            assert_array_equal(env_copy.dimension_ranges[i], env.dimension_ranges[i])

    def test_copy_connectivity_graph(self, sample_2d_grid):
        """copy() preserves connectivity graph."""
        env = sample_2d_grid
        env_copy = env.copy()

        # Graph structure preserved
        assert (
            env_copy.connectivity.number_of_nodes()
            == env.connectivity.number_of_nodes()
        )
        assert (
            env_copy.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        # Check a few node attributes
        for node in list(env.connectivity.nodes())[:5]:  # Sample a few nodes
            assert node in env_copy.connectivity.nodes()
            for attr in ["pos", "source_grid_flat_index"]:
                if attr == "pos":
                    assert_array_equal(
                        env_copy.connectivity.nodes[node][attr],
                        env.connectivity.nodes[node][attr],
                    )
                else:
                    assert (
                        env_copy.connectivity.nodes[node][attr]
                        == env.connectivity.nodes[node][attr]
                    )

    def test_copy_preserves_regions(self, sample_2d_grid):
        """copy() preserves regions."""
        env = sample_2d_grid
        from shapely.geometry import Point

        # Add a region
        env.regions.add("test", point=Point(5.0, 5.0))

        env_copy = env.copy()

        # Regions preserved
        assert len(env_copy.regions) == len(env.regions)
        assert "test" in env_copy.regions
        assert env_copy.regions["test"].kind == "point"


class TestCopyDeepVsShallow:
    """Test deep vs shallow copy behavior."""

    def test_deep_copy_default(self, sample_2d_grid):
        """deep=True is the default."""
        env = sample_2d_grid
        env_copy = env.copy()  # No explicit deep parameter

        # Modify copy's bin_centers array
        env_copy.bin_centers[0, 0] = 999.0

        # Original should be unchanged (deep copy)
        assert env.bin_centers[0, 0] != 999.0

    def test_deep_copy_explicit(self, sample_2d_grid):
        """deep=True creates independent copy."""
        env = sample_2d_grid
        env_copy = env.copy(deep=True)

        # Modify copy's bin_centers
        env_copy.bin_centers[0, 0] = 999.0

        # Original unchanged
        assert env.bin_centers[0, 0] != 999.0

    def test_deep_copy_graph_independent(self, sample_2d_grid):
        """Deep copy: modifying copy's graph doesn't affect original."""
        env = sample_2d_grid
        env_copy = env.copy(deep=True)

        # Add node to copy's graph (shouldn't affect original)
        new_node = env_copy.connectivity.number_of_nodes()
        env_copy.connectivity.add_node(new_node, pos=(999.0, 999.0))

        # Original graph unchanged
        assert new_node not in env.connectivity.nodes()

    def test_deep_copy_regions_independent(self, sample_2d_grid):
        """Deep copy: modifying copy's regions doesn't affect original."""
        env = sample_2d_grid
        from shapely.geometry import Point

        env.regions.add("original", point=Point(5.0, 5.0))
        env_copy = env.copy(deep=True)

        # Add region to copy
        env_copy.regions.add("copy_only", point=Point(10.0, 10.0))

        # Original regions unchanged
        assert "copy_only" not in env.regions
        assert len(env.regions) == 1
        assert len(env_copy.regions) == 2

    def test_shallow_copy_shares_arrays(self, sample_2d_grid):
        """deep=False creates shallow copy (shares data)."""
        env = sample_2d_grid
        env_copy = env.copy(deep=False)

        # Modify copy's bin_centers array
        original_value = env.bin_centers[0, 0]
        env_copy.bin_centers[0, 0] = 999.0

        # Original should be changed (shallow copy shares arrays)
        assert env.bin_centers[0, 0] == 999.0

        # Restore for other tests
        env.bin_centers[0, 0] = original_value

    def test_shallow_copy_shares_graph(self, sample_2d_grid):
        """Shallow copy: graph is shared."""
        env = sample_2d_grid
        env_copy = env.copy(deep=False)

        # Graphs should be the same object
        assert env_copy.connectivity is env.connectivity


class TestCopyCacheInvalidation:
    """Test that caches are cleared after copy."""

    def test_copy_clears_kdtree_cache(self, sample_2d_grid):
        """copy() clears KDTree cache."""
        env = sample_2d_grid

        # Trigger KDTree cache creation (by calling a method that uses it)
        from neurospatial.spatial import map_points_to_bins

        points = np.array([[5.0, 5.0]])
        _ = map_points_to_bins(points, env)

        # Check cache exists on the environment object
        assert hasattr(env, "_kdtree_cache")
        assert env._kdtree_cache is not None

        # Copy environment
        env_copy = env.copy()

        # Copy should have no kdtree cache
        assert not hasattr(env_copy, "_kdtree_cache") or env_copy._kdtree_cache is None

    def test_copy_clears_kernel_cache(self, sample_2d_grid):
        """copy() clears kernel cache."""
        env = sample_2d_grid

        # Build a kernel to populate cache
        kernel1 = env.compute_kernel(bandwidth=2.0, mode="transition")
        assert kernel1 is not None

        # Cache should exist
        assert hasattr(env, "_kernel_cache")
        assert len(env._kernel_cache) > 0

        # Copy environment
        env_copy = env.copy()

        # Copy should have empty kernel cache
        assert hasattr(env_copy, "_kernel_cache")
        assert len(env_copy._kernel_cache) == 0

    def test_copy_with_deep_clears_both_caches(self, sample_2d_grid):
        """deep=True copy clears all caches."""
        env = sample_2d_grid
        from neurospatial.spatial import map_points_to_bins

        # Populate both caches
        points = np.array([[5.0, 5.0]])
        _ = map_points_to_bins(points, env)
        _ = env.compute_kernel(bandwidth=2.0, mode="transition")

        # Both caches should exist
        assert hasattr(env, "_kdtree_cache") and env._kdtree_cache is not None
        assert len(env._kernel_cache) > 0

        # Deep copy
        env_copy = env.copy(deep=True)

        # Copy should have no cache entries
        assert not hasattr(env_copy, "_kdtree_cache") or env_copy._kdtree_cache is None
        assert len(env_copy._kernel_cache) == 0


class TestCopyEdgeCases:
    """Test edge cases and special scenarios."""

    def test_copy_empty_regions(self, sample_2d_grid):
        """copy() with no regions works correctly."""
        env = sample_2d_grid
        # Ensure no regions
        env.regions = Regions()

        env_copy = env.copy()

        assert len(env_copy.regions) == 0

    def test_copy_with_custom_units_and_frame(self, sample_2d_grid):
        """copy() preserves custom units and frame."""
        env = sample_2d_grid
        env.units = "cm"
        env.frame = "session1"

        env_copy = env.copy()

        assert env_copy.units == "cm"
        assert env_copy.frame == "session1"

    def test_copy_preserves_fitted_state(self, sample_2d_grid):
        """copy() preserves fitted state."""
        env = sample_2d_grid
        # Environment from fixture should be fitted
        assert env._is_fitted

        env_copy = env.copy()

        # Copy should also be fitted
        assert env_copy._is_fitted

    def test_copy_multiple_times(self, sample_2d_grid):
        """Can create multiple copies."""
        env = sample_2d_grid

        copy1 = env.copy()
        copy2 = env.copy()
        copy3 = copy1.copy()

        # All should be different instances
        assert copy1 is not env
        assert copy2 is not env
        assert copy3 is not env
        assert copy1 is not copy2
        assert copy1 is not copy3

        # All should have same structure
        assert copy1.n_bins == copy2.n_bins == copy3.n_bins == env.n_bins


class TestCopyDifferentLayouts:
    """Test copy() works across different layout types."""

    def test_copy_graph_layout(self, graph_env):
        """copy() works for GraphLayout environments."""
        env = graph_env
        env_copy = env.copy()

        assert env_copy.n_bins == env.n_bins
        assert env_copy.is_1d == env.is_1d
        assert_array_equal(env_copy.bin_centers, env.bin_centers)

    def test_copy_grid_from_samples(self, grid_env_from_samples):
        """copy() works for grid created from samples."""
        env = grid_env_from_samples
        env_copy = env.copy()

        assert env_copy.n_bins == env.n_bins
        assert_array_equal(env_copy.bin_centers, env.bin_centers)

    def test_copy_masked_grid(self, env_all_active_2x2):
        """copy() works for masked grid."""
        env = env_all_active_2x2
        env_copy = env.copy()

        assert env_copy.n_bins == env.n_bins
        assert_array_equal(env_copy.bin_centers, env.bin_centers)


class TestCopyIntegration:
    """Test copy() integration with other Environment methods."""

    def test_copy_then_modify_regions(self, sample_2d_grid):
        """Can modify copy's regions without affecting original."""
        env = sample_2d_grid
        from shapely.geometry import Point

        env.regions.add("original", point=Point(5.0, 5.0))

        env_copy = env.copy(deep=True)
        env_copy.regions.add("new", point=Point(10.0, 10.0))

        # Original unchanged
        assert "new" not in env.regions

    def test_copy_then_spatial_operations(self, sample_2d_grid):
        """Spatial operations work on copied environment."""
        env = sample_2d_grid
        env_copy = env.copy()

        # Test various operations on copy
        point = env_copy.bin_centers[0]
        bin_idx = env_copy.bin_at(point)
        assert bin_idx >= 0

        neighbors = list(env_copy.neighbors(0))
        assert len(neighbors) > 0

    def test_copy_then_compute_kernel(self, sample_2d_grid):
        """Can compute kernel on copied environment."""
        env = sample_2d_grid
        env_copy = env.copy()

        # Compute kernel on copy
        kernel = env_copy.compute_kernel(bandwidth=2.0, mode="transition")

        assert kernel.shape == (env_copy.n_bins, env_copy.n_bins)
        assert np.allclose(kernel.sum(axis=0), 1.0)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_2d_grid():
    """Create a simple 2D grid environment for testing."""
    # Create a 10x10 grid
    n = 10
    points = [[i, j] for i in range(n) for j in range(n)]
    data = np.array(points, dtype=np.float64)
    env = Environment.from_samples(data, bin_size=1.0)
    return env
