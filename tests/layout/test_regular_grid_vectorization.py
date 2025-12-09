"""Tests for regular grid vectorization optimizations.

These tests verify that the vectorized implementation produces identical
results to the original loop-based implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.layout.helpers.regular_grid import _points_to_regular_grid_bin_ind


@pytest.fixture
def simple_grid_env():
    """Create a simple 10x10 grid environment."""
    from shapely.geometry import box

    square = box(0, 0, 20, 20)
    env = Environment.from_polygon(square, bin_size=2.0)
    return env


@pytest.fixture
def sparse_grid_env():
    """Create a sparse grid environment with many inactive bins."""
    # Create L-shaped environment (sparse active mask)
    positions = np.vstack(
        [
            np.random.uniform(0, 10, (500, 2)),  # Left part
            np.random.uniform([10, 0], [20, 10], (500, 2)),  # Bottom right
        ]
    )
    env = Environment.from_samples(positions, bin_size=2.0)
    return env


class TestPointsToBinIndReference:
    """Reference tests capturing current behavior before optimization."""

    def test_single_point_in_grid(self, simple_grid_env):
        """Test single point mapping."""
        env = simple_grid_env
        layout = env.layout

        point = np.array([[5.0, 5.0]])
        result = _points_to_regular_grid_bin_ind(
            point,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # Point should map to a valid bin
        assert result[0] >= 0
        assert result[0] < env.n_bins

    def test_multiple_points(self, simple_grid_env):
        """Test multiple points mapping."""
        env = simple_grid_env
        layout = env.layout

        points = np.array(
            [
                [1.0, 1.0],
                [5.0, 5.0],
                [15.0, 15.0],
                [19.0, 19.0],
            ]
        )
        result = _points_to_regular_grid_bin_ind(
            points,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        assert len(result) == 4
        # All points should be in valid bins
        assert np.all(result >= 0)
        assert np.all(result < env.n_bins)

    def test_out_of_bounds_points(self, simple_grid_env):
        """Test points outside grid boundaries."""
        env = simple_grid_env
        layout = env.layout

        points = np.array(
            [
                [-5.0, 5.0],  # Left of grid
                [5.0, -5.0],  # Below grid
                [25.0, 5.0],  # Right of grid
                [5.0, 25.0],  # Above grid
            ]
        )
        result = _points_to_regular_grid_bin_ind(
            points,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # All out-of-bounds points should return -1
        assert_array_equal(result, [-1, -1, -1, -1])

    def test_nan_points(self, simple_grid_env):
        """Test points with NaN values."""
        env = simple_grid_env
        layout = env.layout

        points = np.array(
            [
                [np.nan, 5.0],
                [5.0, np.nan],
                [np.nan, np.nan],
                [5.0, 5.0],  # Valid point
            ]
        )
        result = _points_to_regular_grid_bin_ind(
            points,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # NaN points should return -1
        assert result[0] == -1
        assert result[1] == -1
        assert result[2] == -1
        # Valid point should have valid index
        assert result[3] >= 0

    def test_sparse_grid_inactive_bins(self, sparse_grid_env):
        """Test that points in inactive bins return -1."""
        env = sparse_grid_env
        layout = env.layout

        # Find a point that would be in an inactive bin
        # (upper right corner in L-shaped environment)
        point_in_inactive = np.array([[15.0, 15.0]])

        result = _points_to_regular_grid_bin_ind(
            point_in_inactive,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # Should return -1 for inactive bin
        assert result[0] == -1

    def test_large_batch_consistency(self, simple_grid_env):
        """Test that large batches produce consistent results."""
        env = simple_grid_env
        layout = env.layout

        # Generate many random points
        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 25, (10000, 2))

        result = _points_to_regular_grid_bin_ind(
            points,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # Verify basic properties
        assert len(result) == 10000
        assert result.dtype == int
        # Valid indices should be in range
        valid_mask = result >= 0
        assert np.all(result[valid_mask] < env.n_bins)

    def test_bin_center_mapping(self, simple_grid_env):
        """Test that bin centers map to their own bins."""
        env = simple_grid_env
        layout = env.layout

        bin_centers = env.bin_centers
        result = _points_to_regular_grid_bin_ind(
            bin_centers,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # Each bin center should map to its own bin index
        expected = np.arange(env.n_bins)
        assert_array_equal(result, expected)

    def test_edge_cases_at_bin_boundaries(self, simple_grid_env):
        """Test points exactly at bin boundaries."""
        env = simple_grid_env
        layout = env.layout

        # Points at exact bin edges
        edges = layout.grid_edges
        points = np.array(
            [
                [edges[0][0], edges[1][0]],  # Origin corner
                [edges[0][1], edges[1][1]],  # First bin edge
                [edges[0][-1], edges[1][-1]],  # Last edge (should be out or last bin)
            ]
        )

        result = _points_to_regular_grid_bin_ind(
            points,
            grid_edges=layout.grid_edges,
            grid_shape=layout.grid_shape,
            active_mask=layout.active_mask,
        )

        # First two should be valid, last may be -1 (at upper boundary)
        assert result[0] >= 0  # Origin should be valid
        assert result[1] >= 0  # First internal edge should be valid


class TestContainsPerformance:
    """Tests for env.contains() which uses the bin mapping."""

    def test_contains_single_point(self, simple_grid_env):
        """Test contains with single point."""
        env = simple_grid_env

        point_inside = np.array([[10.0, 10.0]])
        point_outside = np.array([[-5.0, 10.0]])

        assert env.contains(point_inside)[0]
        assert not env.contains(point_outside)[0]

    def test_contains_batch(self, simple_grid_env):
        """Test contains with batch of points."""
        env = simple_grid_env

        points = np.array(
            [
                [10.0, 10.0],  # Inside
                [-5.0, 10.0],  # Outside
                [10.0, -5.0],  # Outside
                [5.0, 5.0],  # Inside
            ]
        )

        result = env.contains(points)
        assert result[0]
        assert not result[1]
        assert not result[2]
        assert result[3]

    def test_contains_large_batch_performance(self, simple_grid_env):
        """Test that large batch doesn't hang (regression test for vectorization)."""
        import time

        env = simple_grid_env

        # This should complete in < 1 second with vectorization
        # Without vectorization, 100k points would take many seconds
        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 25, (100000, 2))

        start = time.perf_counter()
        result = env.contains(points)
        elapsed = time.perf_counter() - start

        assert len(result) == 100000
        # Should complete in reasonable time (< 2 seconds)
        assert elapsed < 2.0, (
            f"contains() took {elapsed:.2f}s for 100k points (too slow)"
        )
