"""Tests for path efficiency metrics module.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestTraveledPathLength:
    """Test traveled_path_length function."""

    def test_straight_line_euclidean(self):
        """Test that straight line path gives expected total length."""
        from neurospatial.behavior.navigation import traveled_path_length

        # Create straight line trajectory from 0 to 100 in x
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        length = traveled_path_length(positions, metric="euclidean")

        # Total length should be 100 (20 steps of 5 each)
        assert_allclose(length, 100.0, rtol=0.01)

    def test_single_position_returns_zero(self):
        """Test that single position returns 0 path length."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([[50.0, 50.0]])

        length = traveled_path_length(positions, metric="euclidean")

        assert length == 0.0

    def test_two_positions(self):
        """Test path length with exactly 2 positions."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([[0.0, 0.0], [30.0, 40.0]])  # 3-4-5 triangle

        length = traveled_path_length(positions, metric="euclidean")

        assert_allclose(length, 50.0, rtol=0.01)

    def test_geodesic_requires_env(self):
        """Test that geodesic metric raises without env."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        with pytest.raises(ValueError, match="env parameter is required"):
            traveled_path_length(positions, metric="geodesic")

    def test_geodesic_with_env(self):
        """Test geodesic path length with environment."""
        from neurospatial.behavior.navigation import traveled_path_length

        # Create grid environment
        x = np.linspace(0, 40, 100)
        y = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Create trajectory on bin centers
        trajectory_bins = np.arange(5)
        positions = env.bin_centers[trajectory_bins]

        length = traveled_path_length(positions, metric="geodesic", env=env)

        # Should be positive and finite
        assert length > 0
        assert np.isfinite(length)


class TestShortestPathLength:
    """Test shortest_path_length function."""

    def test_euclidean_straight_line(self):
        """Test Euclidean shortest path is straight line distance."""
        from neurospatial.behavior.navigation import shortest_path_length

        # Create simple grid environment
        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        env = Environment.from_samples(positions, bin_size=5.0)

        start = np.array([0.0, 0.0])
        goal = np.array([30.0, 40.0])

        length = shortest_path_length(env, start, goal, metric="euclidean")

        # Should be Euclidean distance: sqrt(30^2 + 40^2) = 50
        assert_allclose(length, 50.0, rtol=0.01)

    def test_geodesic_on_grid(self):
        """Test geodesic shortest path on grid environment."""
        from neurospatial.behavior.navigation import shortest_path_length

        # Create 2D grid environment
        x = np.linspace(0, 40, 100)
        y = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        start = env.bin_centers[0]
        goal = env.bin_centers[5]

        length = shortest_path_length(env, start, goal, metric="geodesic")

        # Should be positive and finite
        assert length > 0
        assert np.isfinite(length)

    def test_same_start_and_goal(self):
        """Test that same start and goal gives zero distance."""
        from neurospatial.behavior.navigation import shortest_path_length

        positions = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        env = Environment.from_samples(positions, bin_size=5.0)

        start = np.array([50.0, 0.0])
        goal = np.array([50.0, 0.0])

        length = shortest_path_length(env, start, goal, metric="euclidean")

        assert_allclose(length, 0.0, atol=1e-10)


class TestPathEfficiency:
    """Test path_efficiency function."""

    def test_straight_path_efficiency_is_one(self):
        """Test that straight path from start to goal has efficiency 1.0."""
        from neurospatial.behavior.navigation import path_efficiency

        # Create grid environment
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Straight line trajectory
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        goal = np.array([50.0, 0.0])

        eff = path_efficiency(env, positions, goal, metric="euclidean")

        # Straight path should be close to 1.0
        assert_allclose(eff, 1.0, rtol=0.05)

    def test_u_turn_path_efficiency(self):
        """Test that U-turn path has efficiency ~0.5."""
        from neurospatial.behavior.navigation import path_efficiency

        # Create grid environment
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # U-turn trajectory: 0 -> 50 -> 0 -> 25 (goal)
        # Total traveled: 50 + 50 + 25 = 125
        # Shortest: 25
        # Efficiency: 25/125 = 0.2
        x_coords = np.concatenate(
            [
                np.linspace(0, 50, 11),
                np.linspace(50, 0, 11)[1:],
                np.linspace(0, 25, 6)[1:],
            ]
        )
        positions = np.column_stack([x_coords, np.zeros_like(x_coords)])
        goal = np.array([25.0, 0.0])

        eff = path_efficiency(env, positions, goal, metric="euclidean")

        # Efficiency should be low due to backtracking
        assert eff < 0.5

    def test_less_than_two_positions_returns_nan(self):
        """Test that < 2 positions returns NaN efficiency."""
        from neurospatial.behavior.navigation import path_efficiency

        positions = np.array([[50.0, 50.0]])
        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        goal = np.array([60.0, 0.0])

        eff = path_efficiency(env, positions, goal, metric="euclidean")

        assert np.isnan(eff)

    def test_zero_traveled_returns_nan(self):
        """Test that zero traveled distance returns NaN efficiency."""
        from neurospatial.behavior.navigation import path_efficiency

        # All positions identical
        positions = np.tile([50.0, 50.0], (5, 1))
        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        goal = np.array([60.0, 0.0])

        eff = path_efficiency(env, positions, goal, metric="euclidean")

        assert np.isnan(eff)


class TestAngularEfficiency:
    """Test angular_efficiency function."""

    def test_straight_to_goal_is_one(self):
        """Test that heading directly to goal gives efficiency ~1.0."""
        from neurospatial.behavior.navigation import angular_efficiency

        # Straight line toward goal at (100, 0)
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        goal = np.array([100.0, 0.0])

        eff = angular_efficiency(positions, goal)

        # Should be very close to 1.0 (always heading toward goal)
        assert eff > 0.9

    def test_less_than_3_positions_returns_one(self):
        """Test that < 3 positions returns 1.0 (no turns possible)."""
        from neurospatial.behavior.navigation import angular_efficiency

        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        goal = np.array([50.0, 0.0])

        eff = angular_efficiency(positions, goal)

        assert eff == 1.0

    def test_identical_positions_returns_nan(self):
        """Test that all identical positions returns NaN."""
        from neurospatial.behavior.navigation import angular_efficiency

        positions = np.tile([50.0, 50.0], (10, 1))
        goal = np.array([100.0, 0.0])

        eff = angular_efficiency(positions, goal)

        assert np.isnan(eff)

    def test_returns_value_in_zero_one(self):
        """Test that angular efficiency is in [0, 1] range."""
        from neurospatial.behavior.navigation import angular_efficiency

        # Meandering trajectory
        t = np.linspace(0, 4 * np.pi, 100)
        x = t * 5 + 20 * np.sin(t)
        y = 20 * np.cos(t)
        positions = np.column_stack([x, y])
        goal = np.array([100.0, 0.0])

        eff = angular_efficiency(positions, goal)

        assert 0.0 <= eff <= 1.0


class TestTimeEfficiency:
    """Test time_efficiency function."""

    def test_with_reference_speed(self):
        """Test time efficiency computation with reference speed."""
        from neurospatial.behavior.navigation import time_efficiency

        # Trajectory: 100 units in 10 seconds = 10 units/s actual speed
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.linspace(0, 10, 21)
        goal = np.array([100.0, 0.0])
        reference_speed = 20.0  # units/s

        # Shortest path = 100, at reference speed 20 = 5 seconds optimal
        # Actual time = 10 seconds
        # Efficiency = 5/10 = 0.5
        eff = time_efficiency(positions, times, goal, reference_speed=reference_speed)

        assert_allclose(eff, 0.5, rtol=0.1)


class TestSubgoalEfficiency:
    """Test subgoal_efficiency function."""

    def test_two_segment_path(self):
        """Test efficiency with two segments via a subgoal."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        # Create grid environment
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Straight line trajectory: start -> subgoal -> goal
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        subgoals = np.array([[50.0, 0.0], [100.0, 0.0]])

        result = subgoal_efficiency(env, positions, subgoals, metric="euclidean")

        # Both segments should be efficient
        assert len(result.segment_results) == 2
        assert result.mean_efficiency > 0.8


class TestPathEfficiencyResult:
    """Test PathEfficiencyResult dataclass."""

    def test_is_efficient_method(self):
        """Test is_efficient() helper method."""
        from neurospatial.behavior.navigation import PathEfficiencyResult

        result = PathEfficiencyResult(
            traveled_length=100.0,
            shortest_length=90.0,
            efficiency=0.9,
            time_efficiency=None,
            angular_efficiency=0.85,
            start_position=np.array([0.0, 0.0]),
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        assert result.is_efficient(threshold=0.8)
        assert not result.is_efficient(threshold=0.95)

    def test_is_efficient_with_nan(self):
        """Test that is_efficient() returns False for NaN efficiency."""
        from neurospatial.behavior.navigation import PathEfficiencyResult

        result = PathEfficiencyResult(
            traveled_length=0.0,
            shortest_length=50.0,
            efficiency=np.nan,
            time_efficiency=None,
            angular_efficiency=1.0,
            start_position=np.array([0.0, 0.0]),
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        assert not result.is_efficient(threshold=0.5)

    def test_summary_method(self):
        """Test summary() returns formatted string."""
        from neurospatial.behavior.navigation import PathEfficiencyResult

        result = PathEfficiencyResult(
            traveled_length=45.2,
            shortest_length=32.1,
            efficiency=0.71,
            time_efficiency=None,
            angular_efficiency=0.85,
            start_position=np.array([0.0, 0.0]),
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        summary = result.summary()

        assert "45.2" in summary
        assert "32.1" in summary
        assert "71" in summary  # 71% efficiency


class TestComputePathEfficiency:
    """Test compute_path_efficiency function (combines all metrics)."""

    def test_returns_result_dataclass(self):
        """Test that compute_path_efficiency returns PathEfficiencyResult."""
        from neurospatial.behavior.navigation import (
            PathEfficiencyResult,
            compute_path_efficiency,
        )

        # Create grid environment
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Straight line trajectory
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 11)
        goal = np.array([50.0, 0.0])

        result = compute_path_efficiency(
            env, positions, times, goal, metric="euclidean"
        )

        assert isinstance(result, PathEfficiencyResult)
        assert result.traveled_length > 0
        assert result.shortest_length >= 0
        assert 0 < result.efficiency <= 1.0
        assert 0 <= result.angular_efficiency <= 1.0
        assert result.metric == "euclidean"


class TestErrorHandling:
    """Test error handling and messages."""

    def test_mismatched_array_lengths_error(self):
        """Test helpful error for mismatched positions/times."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 10)  # Wrong length!
        goal = np.array([50.0, 0.0])

        with pytest.raises(
            ValueError, match="positions and times must have same length"
        ):
            compute_path_efficiency(env, positions, times, goal)

    def test_empty_trajectory_error(self):
        """Test helpful error for empty trajectory."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([]).reshape(0, 2)

        with pytest.raises(ValueError, match="positions array is empty"):
            traveled_path_length(positions)
