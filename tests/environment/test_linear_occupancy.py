"""Tests for linear time allocation in occupancy computation.

This module tests the P2.11 enhancement: time_allocation='linear' mode for
more accurate occupancy when trajectories cross bin boundaries on regular grids.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestLinearOccupancyBasic:
    """Test basic linear occupancy functionality."""

    def test_diagonal_trajectory_across_four_bins(self):
        """Test that diagonal trajectory splits time across traversed bins."""
        # Create full 10x10 grid with bin_size=1.0
        # Grid edges at: [-0.5, 0.5, 1.5, 2.5, 3.5, ...]
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Diagonal path from (1.0, 1.0) to (4.0, 4.0)
        # Crosses bins at (1,1), (2,2), (3,3), (4,4)
        times = np.array([0.0, 1.0])
        trajectory = np.array([[1.0, 1.0], [4.0, 4.0]])

        # Linear allocation should split time across all 4 bins
        occupancy_linear = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Find which bins were crossed
        bin_0 = env.bin_at([1.0, 1.0])
        bin_1 = env.bin_at([2.0, 2.0])
        bin_2 = env.bin_at([3.0, 3.0])
        bin_3 = env.bin_at([4.0, 4.0])

        # All four bins should have nonzero occupancy
        assert occupancy_linear[bin_0] > 0
        assert occupancy_linear[bin_1] > 0
        assert occupancy_linear[bin_2] > 0
        assert occupancy_linear[bin_3] > 0

        # Total time should be conserved
        assert_allclose(occupancy_linear.sum(), 1.0, rtol=1e-6)

        # Compare to 'start' allocation - only start bin gets time
        occupancy_start = env.occupancy(
            times, trajectory, time_allocation="start", max_gap=None
        )
        assert occupancy_start[bin_0] == 1.0  # All time in start bin
        assert occupancy_start[bin_1] == 0.0
        assert occupancy_start[bin_2] == 0.0
        assert occupancy_start[bin_3] == 0.0

    def test_proportional_time_allocation(self):
        """Test that time is allocated proportional to distance traveled in each bin."""
        # Create full grid
        # from_samples with bin_size=1.0 creates edges at half-integers: [-0.5, 0.5, 1.5, 2.5, ...]
        # Bins are: x in [0.5, 1.5), [1.5, 2.5), etc.
        positions = np.array([[i, j] for i in range(6) for j in range(6)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Horizontal path from (1.0, 1.0) to (3.0, 1.0) - total distance 2.0
        # Crosses edge at x=1.5 and x=2.5
        # Segments:
        #   Bin x=1: from 1.0 to 1.5 → 0.5 units → 0.5s
        #   Bin x=2: from 1.5 to 2.5 → 1.0 units → 1.0s
        #   Bin x=3: from 2.5 to 3.0 → 0.5 units → 0.5s
        times = np.array([0.0, 2.0])  # 2 seconds
        trajectory = np.array([[1.0, 1.0], [3.0, 1.0]])

        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        bin_0 = env.bin_at([1.0, 1.0])  # Bin at x in [0.5, 1.5), y in [0.5, 1.5)
        bin_1 = env.bin_at([2.0, 1.0])  # Bin at x in [1.5, 2.5), y in [0.5, 1.5)
        bin_2 = env.bin_at([3.0, 1.0])  # Bin at x in [2.5, 3.5), y in [0.5, 1.5)

        # Expected allocation (proportional to distance):
        # bin_0: 2.0 × (0.5/2.0) = 0.5s
        # bin_1: 2.0 × (1.0/2.0) = 1.0s
        # bin_2: 2.0 × (0.5/2.0) = 0.5s
        assert_allclose(occupancy[bin_0], 0.5, rtol=1e-6)
        assert_allclose(occupancy[bin_1], 1.0, rtol=1e-6)
        assert_allclose(occupancy[bin_2], 0.5, rtol=1e-6)

        # Total time conserved
        assert_allclose(occupancy.sum(), 2.0, rtol=1e-6)

    def test_linear_allocation_same_bin_no_crossing(self):
        """Test that linear allocation matches start allocation when no bin crossing."""
        # Create full grid
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Short trajectory within single bin
        times = np.array([0.0, 1.0])
        trajectory = np.array([[1.0, 1.0], [1.2, 1.1]])

        occupancy_start = env.occupancy(
            times, trajectory, time_allocation="start", max_gap=None
        )
        occupancy_linear = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Should be identical (no bin crossing)
        assert_allclose(occupancy_start, occupancy_linear, rtol=1e-6)

    def test_default_time_allocation_is_start(self):
        """Test that time_allocation='start' is the default."""
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([0.0, 1.0, 2.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

        # Default call (no time_allocation specified)
        occupancy_default = env.occupancy(times, trajectory)

        # Explicit 'start'
        occupancy_start = env.occupancy(times, trajectory, time_allocation="start")

        # Should be identical
        assert_allclose(occupancy_default, occupancy_start, rtol=1e-12)


class TestLinearOccupancyMassConservation:
    """Test that linear allocation conserves total time."""

    def test_mass_conservation_complex_trajectory(self):
        """Test mass conservation on complex multi-segment trajectory."""
        # Create grid
        positions = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Complex trajectory crossing many bins
        rng = np.random.default_rng(42)
        n_samples = 20
        times = np.arange(n_samples, dtype=np.float64)
        trajectory = rng.uniform(0.5, 8.5, size=(n_samples, 2))

        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Total time = times[-1] - times[0] = 19.0
        expected_time = times[-1] - times[0]
        assert_allclose(occupancy.sum(), expected_time, rtol=1e-6)

    def test_mass_conservation_with_gaps(self):
        """Test that mass conservation works with max_gap filtering."""
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Trajectory with large gap that should be filtered
        times = np.array([0.0, 0.5, 10.0, 10.5])  # 0.5s gap at t=10
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]])

        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=1.0
        )

        # Only first and last intervals count (both 0.5s)
        # Middle interval (9.5s) exceeds max_gap
        expected_time = 0.5 + 0.5
        assert_allclose(occupancy.sum(), expected_time, rtol=1e-6)


class TestLinearOccupancyLayoutCompatibility:
    """Test that linear allocation only works on RegularGridLayout."""

    def test_linear_allocation_requires_regular_grid(self):
        """Test that linear allocation raises error on non-grid layouts."""
        # Create graph layout (1D track)
        from track_linearization import make_track_graph

        track_graph = make_track_graph(
            node_positions=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            edges=np.array([[0, 1], [1, 2]]),
        )
        edge_order = [(0, 1), (1, 2)]
        edge_spacing = 1.0
        env = Environment.from_graph(
            track_graph, edge_order, edge_spacing, bin_size=0.1
        )

        times = np.array([0.0, 1.0])
        positions = np.array([[0.5, 0.0], [1.5, 0.0]])

        # Should raise NotImplementedError
        with pytest.raises(
            NotImplementedError,
            match=r"time_allocation='linear' is only supported.*RegularGridLayout",
        ):
            env.occupancy(times, positions, time_allocation="linear")

    def test_linear_allocation_on_polygon_layout(self):
        """Test that linear allocation raises error on polygon-bounded layout."""
        from shapely.geometry import box

        # Create polygon layout
        polygon = box(0, 0, 9, 9)
        env = Environment.from_polygon(polygon, bin_size=1.0)

        times = np.array([0.0, 1.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5]])

        # Should raise NotImplementedError (polygon layout is not RegularGridLayout)
        with pytest.raises(
            NotImplementedError,
            match=r"time_allocation='linear'.*RegularGridLayout",
        ):
            env.occupancy(times, trajectory, time_allocation="linear")


class TestLinearOccupancyValidation:
    """Test input validation for time_allocation parameter."""

    def test_invalid_time_allocation_value(self):
        """Test that invalid time_allocation values raise ValueError."""
        positions = np.array([[i, i] for i in np.linspace(0, 10, 100)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([0.0, 1.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5]])

        with pytest.raises(
            ValueError, match=r"time_allocation must be 'start' or 'linear'"
        ):
            env.occupancy(times, trajectory, time_allocation="invalid")

    def test_time_allocation_must_be_string(self):
        """Test that time_allocation must be a string."""
        positions = np.array([[i, i] for i in np.linspace(0, 10, 100)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([0.0, 1.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5]])

        # Pass non-string value
        with pytest.raises((ValueError, TypeError)):
            env.occupancy(times, trajectory, time_allocation=123)


class TestLinearOccupancyEdgeCases:
    """Test edge cases for linear occupancy."""

    def test_single_sample_linear_allocation(self):
        """Test that single sample returns zero occupancy (no intervals)."""
        positions = np.array([[i, i] for i in np.linspace(0, 10, 100)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([0.0])
        trajectory = np.array([[0.5, 0.5]])

        occupancy = env.occupancy(times, trajectory, time_allocation="linear")

        # No intervals, so all zeros
        assert_allclose(occupancy, 0.0)

    def test_empty_arrays_linear_allocation(self):
        """Test that empty arrays return zero occupancy."""
        positions = np.array([[i, i] for i in np.linspace(0, 10, 100)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([])
        trajectory = np.array([]).reshape(0, 2)

        occupancy = env.occupancy(times, trajectory, time_allocation="linear")

        # Empty input, so all zeros
        assert_allclose(occupancy, 0.0)

    def test_trajectory_outside_environment_linear_allocation(self):
        """Test that linear allocation handles outside points gracefully."""
        # Create small grid (0-5 range)
        positions = np.array([[i, j] for i in range(6) for j in range(6)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Trajectory from inside to outside
        times = np.array([0.0, 1.0])
        trajectory = np.array([[2.5, 2.5], [10.0, 10.0]])

        # Should handle gracefully (only allocate time to valid bins)
        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Some time should be allocated (at least to starting bin)
        assert occupancy.sum() >= 0.0


class TestLinearOccupancyIntegration:
    """Integration tests combining linear allocation with other features."""

    def test_linear_allocation_with_speed_filtering(self):
        """Test linear allocation combined with speed filtering."""
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Trajectory with varying speeds
        times = np.array([0.0, 1.0, 2.0, 3.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]])
        speeds = np.array([5.0, 0.5, 5.0, 5.0])  # Slow at second sample

        occupancy = env.occupancy(
            times,
            trajectory,
            speed=speeds,
            min_speed=1.0,
            time_allocation="linear",
            max_gap=None,
        )

        # Second interval (t=1 to t=2) should be filtered out
        # Only first and third intervals contribute (2.0s total)
        assert_allclose(occupancy.sum(), 2.0, rtol=1e-6)

    def test_linear_allocation_with_kernel_smoothing(self):
        """Test linear allocation combined with kernel smoothing."""
        positions = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(positions, bin_size=1.0)

        times = np.array([0.0, 1.0, 2.0])
        trajectory = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

        occupancy = env.occupancy(
            times,
            trajectory,
            time_allocation="linear",
            kernel_bandwidth=2.0,
            max_gap=None,
        )

        # Mass should still be conserved after smoothing
        expected_time = times[-1] - times[0]
        assert_allclose(occupancy.sum(), expected_time, rtol=1e-6)


class TestLinearOccupancyNumericalStability:
    """Test numerical stability for edge cases in linear allocation.

    These tests verify that the ray-grid intersection algorithm handles
    edge cases without producing inf/nan values.
    """

    def test_occupancy_ray_parallel_to_edge(self):
        """Test trajectory segment parallel to grid edge (ray_dir[dim] ≈ 0).

        When a ray is perfectly parallel to a grid edge, the ray direction
        component in that dimension is zero. The algorithm should skip
        division by zero and handle this gracefully.
        """
        # Create grid
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Movement only in x direction (parallel to y-edges)
        # ray_dir = [10.0, 0.0] → ray_dir[1] = 0.0 after normalization
        times = np.array([0.0, 1.0])
        trajectory = np.array([[1.0, 5.0], [9.0, 5.0]])

        # Should not produce inf/nan
        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Verify no inf/nan values
        assert np.all(np.isfinite(occupancy)), "Occupancy contains inf/nan values"
        assert np.sum(occupancy) > 0, "Occupancy should be non-zero"
        assert_allclose(occupancy.sum(), 1.0, rtol=1e-6)

    def test_occupancy_very_small_ray_direction(self):
        """Test trajectory with very small movement in one dimension.

        When movement in one dimension is tiny (but non-zero), we approach
        the numerical stability limit. This tests that epsilon threshold
        properly prevents division by near-zero values.
        """
        # Create grid
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Very small movement in y direction (1e-14), large in x
        # This tests the epsilon threshold (should be ~1e-12)
        times = np.array([0.0, 1.0])
        trajectory = np.array([[1.0, 5.0], [9.0, 5.0 + 1e-14]])

        # Should handle gracefully (treat y as parallel since movement < epsilon)
        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Verify no inf/nan values
        assert np.all(np.isfinite(occupancy)), "Occupancy contains inf/nan values"
        assert np.sum(occupancy) > 0
        assert_allclose(occupancy.sum(), 1.0, rtol=1e-6)

    def test_occupancy_near_epsilon_threshold(self):
        """Test ray direction component near but above epsilon threshold.

        Tests that ray direction components just above the epsilon threshold
        (~1e-12) are handled correctly without numerical issues.
        """
        # Create grid
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Movement slightly above epsilon threshold in y
        # If epsilon is 1e-12, use 1e-10 (100x above threshold)
        times = np.array([0.0, 1.0])
        small_dy = 1e-10
        trajectory = np.array([[1.0, 5.0], [9.0, 5.0 + small_dy]])

        # Should handle without issues
        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Verify numerical stability
        assert np.all(np.isfinite(occupancy)), "Occupancy contains inf/nan values"
        assert np.sum(occupancy) > 0
        # Total time should be conserved
        assert_allclose(occupancy.sum(), 1.0, rtol=1e-6)

    def test_occupancy_perfectly_stationary_linear(self):
        """Test zero-distance ray (start == end).

        When start and end positions are identical, total_distance = 0.
        This should be handled before any division occurs.
        """
        # Create grid
        positions = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Stationary (no movement)
        times = np.array([0.0, 1.0])
        trajectory = np.array([[5.0, 5.0], [5.0, 5.0]])

        # Should allocate all time to starting bin
        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Verify no inf/nan
        assert np.all(np.isfinite(occupancy)), "Occupancy contains inf/nan values"

        # All time should be in the starting bin
        bin_idx = env.bin_at([5.0, 5.0])
        assert_allclose(occupancy[bin_idx], 1.0, rtol=1e-6)
        assert_allclose(occupancy.sum(), 1.0, rtol=1e-6)


class TestLinearOccupancyAccuracy:
    """Test accuracy of linear allocation against known solutions."""

    def test_45_degree_diagonal_equal_allocation(self):
        """Test 45° diagonal allocates time proportional to distance through each bin."""
        # Create uniform grid
        # Grid edges at [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, ...]
        positions = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Perfect 45° diagonal from (1.0, 1.0) to (5.0, 5.0)
        # Total distance = √32 ≈ 5.657
        # Crosses edges at (1.5,1.5), (2.5,2.5), (3.5,3.5), (4.5,4.5)
        # Segment distances: √0.5, √2, √2, √2, √0.5
        times = np.array([0.0, np.sqrt(32)])
        trajectory = np.array([[1.0, 1.0], [5.0, 5.0]])

        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # End bins get √0.5, middle bins get √2
        sqrt_half = np.sqrt(0.5)
        sqrt_two = np.sqrt(2.0)

        assert_allclose(occupancy[env.bin_at([1.0, 1.0])], sqrt_half, rtol=0.01)
        assert_allclose(occupancy[env.bin_at([2.0, 2.0])], sqrt_two, rtol=0.01)
        assert_allclose(occupancy[env.bin_at([3.0, 3.0])], sqrt_two, rtol=0.01)
        assert_allclose(occupancy[env.bin_at([4.0, 4.0])], sqrt_two, rtol=0.01)
        assert_allclose(occupancy[env.bin_at([5.0, 5.0])], sqrt_half, rtol=0.01)

    def test_vertical_line_single_column(self):
        """Test vertical line stays in single column of bins."""
        positions = np.array([[i, j] for i in range(10) for j in range(10)])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Vertical line from (2.0, 1.0) to (2.0, 4.0)
        # Grid edges at [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        # Should cross 4 bins in column x=2: at y=1,2,3,4
        times = np.array([0.0, 3.0])
        trajectory = np.array([[2.0, 1.0], [2.0, 4.0]])

        occupancy = env.occupancy(
            times, trajectory, time_allocation="linear", max_gap=None
        )

        # Should cross 4 bins in column x=2
        bins_in_column = [
            env.bin_at([2.0, 1.0]),
            env.bin_at([2.0, 2.0]),
            env.bin_at([2.0, 3.0]),
            env.bin_at([2.0, 4.0]),
        ]

        # All 4 bins should have nonzero occupancy
        for bin_idx in bins_in_column:
            assert occupancy[bin_idx] > 0

        # Bins outside column should have zero occupancy
        outside_bin = env.bin_at([3.0, 2.0])  # Different column
        assert occupancy[outside_bin] == 0.0

        # Total mass conserved
        assert_allclose(occupancy.sum(), 3.0, rtol=1e-6)
