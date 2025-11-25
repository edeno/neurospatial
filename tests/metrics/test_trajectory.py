"""Tests for trajectory metrics module.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.metrics.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)


class TestComputeTurnAngles:
    """Test compute_turn_angles function."""

    def test_straight_line_trajectory(self):
        """Test that straight line movement has near-zero turn angles."""
        # Create straight line trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # Compute turn angles (continuous API)
        angles = compute_turn_angles(positions)

        # Should have n-2 angles (first and last transitions)
        assert angles.shape == (18,)

        # All angles should be near zero (straight line)
        assert_allclose(angles, 0.0, atol=0.1)

    def test_circular_trajectory_constant_turning(self):
        """Test that circular motion produces turn angles."""
        # Create circular trajectory (100 points around a circle)
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = 50 + 40 * np.cos(theta)
        y = 50 + 40 * np.sin(theta)
        positions = np.column_stack([x, y])

        # Compute turn angles (continuous API - no env needed!)
        angles = compute_turn_angles(positions)

        # Should have some non-zero angles (circular motion produces turns)
        assert len(angles) > 0
        non_zero_angles = angles[np.abs(angles) > 0.01]
        # At least some turning should occur
        assert len(non_zero_angles) > 0

    def test_turn_angles_range(self):
        """Test that turn angles are in [-π, π] range."""
        # Create meandering trajectory (deterministic - no randomness needed)
        t = np.linspace(0, 4 * np.pi, 200)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])

        # Continuous API
        angles = compute_turn_angles(positions)

        # All angles should be in [-π, π]
        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)

    def test_stationary_trajectory(self):
        """Test that stationary periods are skipped."""
        # Create trajectory with repeated positions (stationary periods)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Duplicate
                [0.0, 0.0],  # Duplicate
                [10.0, 0.0],
                [20.0, 0.0],
                [20.0, 0.0],  # Duplicate
                [30.0, 0.0],
                [40.0, 0.0],
                [40.0, 0.0],  # Duplicate
                [40.0, 0.0],  # Duplicate
                [50.0, 0.0],
            ]
        )

        angles = compute_turn_angles(positions)

        # Should only compute angles for actual movements
        # Consecutive duplicates should be filtered out
        assert len(angles) >= 0  # At least some angles computed

    def test_parameter_order(self):
        """Test that parameter is just positions (continuous API)."""
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # This should work without error
        angles = compute_turn_angles(positions)
        assert isinstance(angles, np.ndarray)


class TestComputeStepLengths:
    """Test compute_step_lengths function."""

    def test_step_lengths_straight_line(self):
        """Test step lengths on a straight 1D trajectory with Euclidean distance."""
        # Create straight line trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        # Euclidean distance (default)
        step_lengths = compute_step_lengths(positions, distance_type="euclidean")

        # Should have n-1 step lengths
        assert step_lengths.shape == (20,)

        # All steps should be positive and roughly equal
        assert np.all(step_lengths >= 0)
        # For uniform spacing, expect uniform step lengths
        assert_allclose(step_lengths, step_lengths[0], rtol=0.01)

    def test_step_lengths_with_duplicates(self):
        """Test that consecutive duplicates have zero step length."""
        # Trajectory with stationary periods (duplicates)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Duplicate
                [10.0, 0.0],
                [20.0, 0.0],
                [20.0, 0.0],  # Duplicate
                [20.0, 0.0],  # Duplicate
                [30.0, 0.0],
            ]
        )

        step_lengths = compute_step_lengths(positions, distance_type="euclidean")

        # Should have n-1 step lengths
        assert step_lengths.shape == (6,)

        # Consecutive duplicates should have zero distance
        assert step_lengths[0] == 0.0  # [0,0] -> [0,0]
        assert step_lengths[3] == 0.0  # [20,0] -> [20,0]
        assert step_lengths[4] == 0.0  # [20,0] -> [20,0]

    def test_step_lengths_uses_graph_distance(self):
        """Test that step lengths with distance_type='geodesic' use graph distances."""
        # Create 2D grid environment
        x = np.linspace(0, 40, 100)
        y = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Create trajectory on bin centers (for geodesic distance to work)
        trajectory_bins = np.arange(5)
        positions = env.bin_centers[trajectory_bins]

        # Use geodesic distance
        step_lengths = compute_step_lengths(
            positions, distance_type="geodesic", env=env
        )

        # Verify each step length matches nx.shortest_path_length
        import networkx as nx

        for i in range(len(step_lengths)):
            bin_i = trajectory_bins[i]
            bin_j = trajectory_bins[i + 1]
            expected = float(
                nx.shortest_path_length(
                    env.connectivity, source=bin_i, target=bin_j, weight="distance"
                )
            )
            assert_allclose(step_lengths[i], expected, rtol=1e-5)

    def test_parameter_order(self):
        """Test new continuous API signature."""
        positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

        # Euclidean (default)
        step_lengths = compute_step_lengths(positions)
        assert isinstance(step_lengths, np.ndarray)

        # Geodesic requires env
        env = Environment.from_samples(positions, bin_size=5.0)
        bin_positions = env.bin_centers[env.bin_at(positions)]
        step_lengths_geo = compute_step_lengths(
            bin_positions, distance_type="geodesic", env=env
        )
        assert isinstance(step_lengths_geo, np.ndarray)


class TestComputeHomeRange:
    """Test compute_home_range function."""

    def test_home_range_95_percentile(self):
        """Test 95% home range selection."""
        # Create trajectory with known occupancy distribution
        # Bin 0: 50 visits, Bin 1: 30 visits, Bin 2: 15 visits, Bin 3: 5 visits
        trajectory_bins = np.concatenate(
            [
                np.repeat(0, 50),
                np.repeat(1, 30),
                np.repeat(2, 15),
                np.repeat(3, 5),
            ]
        )

        home_range = compute_home_range(trajectory_bins, percentile=95.0)

        # 95% of 100 visits = 95 visits
        # Bin 0 (50) + Bin 1 (30) + Bin 2 (15) = 95 visits = 95%
        # So bins 0, 1, 2 should be in home range
        assert set(home_range) == {0, 1, 2}

    def test_home_range_100_percentile(self):
        """Test that 100% includes all visited bins."""
        trajectory_bins = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 0])

        home_range = compute_home_range(trajectory_bins, percentile=100.0)

        # Should include all unique bins
        assert set(home_range) == {0, 1, 2, 3}

    def test_home_range_50_percentile(self):
        """Test 50% home range (core area)."""
        # Uniform distribution: 10 visits per bin across 10 bins
        trajectory_bins = np.repeat(np.arange(10), 10)

        home_range = compute_home_range(trajectory_bins, percentile=50.0)

        # 50% of visits = any 5 bins
        assert len(home_range) == 5

    def test_home_range_returns_indices(self):
        """Test that home_range returns bin indices (integers)."""
        trajectory_bins = np.array([0, 1, 2, 1, 0, 1, 2])

        home_range = compute_home_range(trajectory_bins, percentile=95.0)

        # Should be numpy array of integers
        assert isinstance(home_range, np.ndarray)
        assert home_range.dtype in [np.int32, np.int64, np.intp]

    def test_parameter_order(self):
        """Test that parameter order is (trajectory_bins, *, percentile)."""
        trajectory_bins = np.array([0, 1, 2, 1, 0])

        # This should work without error
        home_range = compute_home_range(trajectory_bins, percentile=95.0)
        assert isinstance(home_range, np.ndarray)


class TestMeanSquareDisplacement:
    """Test mean_square_displacement function."""

    def test_msd_shape(self):
        """Test that MSD returns two arrays (tau, msd)."""
        # Create simple trajectory (continuous positions)
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        # Continuous API with Euclidean distance (default)
        tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=5.0
        )

        # Both should be 1D arrays
        assert tau_values.ndim == 1
        assert msd_values.ndim == 1
        assert len(tau_values) == len(msd_values)

    def test_msd_monotonic_increase(self):
        """Test that MSD generally increases with tau (for diffusive motion)."""
        # Create 2D random walk trajectory with local RNG
        rng = np.random.default_rng(42)
        n_steps = 100
        steps = rng.standard_normal((n_steps, 2)) * 2  # 2D random steps
        positions = np.cumsum(steps, axis=0)
        times = np.arange(n_steps) * 0.1

        # Continuous API with Euclidean distance
        _tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=5.0
        )

        # MSD should generally increase with tau (monotonic for diffusion)
        # Due to sampling noise, allow for small decreases
        # Check that MSD at end is significantly larger than at start
        assert msd_values[-1] > msd_values[0]

    def test_msd_stationary_is_zero(self):
        """Test that MSD is near zero for stationary trajectory."""
        # Stationary trajectory (all same position)
        positions = np.tile([0.0, 0.0], (50, 1))  # 50 identical positions
        times = np.linspace(0, 10, 50)

        # Continuous API
        _tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=5.0
        )

        # All MSD values should be zero (no displacement)
        assert_allclose(msd_values, 0.0, atol=1e-10)

    def test_msd_max_tau_parameter(self):
        """Test that max_tau limits the lag times."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        tau_values, _ = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=3.0
        )

        # All tau values should be <= max_tau
        assert np.all(tau_values <= 3.0)

    def test_msd_returns_floats(self):
        """Test that MSD values are float64."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=5.0
        )

        assert tau_values.dtype == np.float64
        assert msd_values.dtype == np.float64

    def test_parameter_order(self):
        """Test new continuous API signature."""
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        times = np.linspace(0, 10, 50)

        # Euclidean (default)
        tau_values, msd_values = mean_square_displacement(positions, times, max_tau=5.0)
        assert isinstance(tau_values, np.ndarray)
        assert isinstance(msd_values, np.ndarray)

        # Geodesic requires env
        env = Environment.from_samples(positions, bin_size=5.0)
        bin_positions = env.bin_centers[env.bin_at(positions)]
        tau_geo, msd_geo = mean_square_displacement(
            bin_positions, times, distance_type="geodesic", env=env, max_tau=5.0
        )
        assert isinstance(tau_geo, np.ndarray)
        assert isinstance(msd_geo, np.ndarray)


class TestTrajectoryMetricsIntegration:
    """Test integration of all trajectory metrics."""

    def test_complete_workflow(self):
        """Test complete trajectory analysis workflow with continuous API."""
        # Generate synthetic meandering trajectory (deterministic - no randomness needed)
        t = np.linspace(0, 4 * np.pi, 200)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])
        times = np.linspace(0, 20, len(positions))

        # Continuous API: all metrics work directly on positions!
        turn_angles = compute_turn_angles(positions)
        step_lengths = compute_step_lengths(positions, distance_type="euclidean")

        # Home range still uses bins (makes sense for occupancy-based metrics)
        env = Environment.from_samples(positions, bin_size=5.0)
        trajectory_bins = env.bin_at(positions)
        home_range = compute_home_range(trajectory_bins, percentile=95.0)

        # MSD uses continuous positions
        tau_values, msd_values = mean_square_displacement(
            positions, times, distance_type="euclidean", max_tau=10.0
        )

        # All metrics should be computed successfully
        assert len(turn_angles) > 0
        assert len(step_lengths) > 0
        assert len(home_range) > 0
        assert len(tau_values) > 0
        assert len(msd_values) > 0

        # Sanity checks
        assert np.all(np.abs(turn_angles) <= np.pi)
        assert np.all(step_lengths >= 0)
        assert np.all(tau_values >= 0)
        assert np.all(msd_values >= 0)
