"""Tests for goal-directed navigation metrics module.

Following TDD: Tests written FIRST, then implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestGoalVector:
    """Test goal_vector function."""

    def test_basic_goal_vector(self):
        """Test computing vector from positions to goal."""
        from neurospatial.metrics.goal_directed import goal_vector

        positions = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        goal = np.array([50.0, 0.0])

        result = goal_vector(positions, goal)

        # Goal vector should be goal - position
        assert result.shape == (3, 2)
        assert_allclose(result[0], [50.0, 0.0])
        assert_allclose(result[1], [40.0, 0.0])
        assert_allclose(result[2], [30.0, 0.0])

    def test_2d_goal_vector(self):
        """Test goal vector in 2D."""
        from neurospatial.metrics.goal_directed import goal_vector

        positions = np.array([[0.0, 0.0], [30.0, 40.0]])
        goal = np.array([100.0, 100.0])

        result = goal_vector(positions, goal)

        assert_allclose(result[0], [100.0, 100.0])
        assert_allclose(result[1], [70.0, 60.0])

    def test_dimension_mismatch_error(self):
        """Test error when goal dimensions don't match positions."""
        from neurospatial.metrics.goal_directed import goal_vector

        positions = np.array([[0.0, 0.0], [10.0, 0.0]])  # 2D
        goal = np.array([50.0, 0.0, 0.0])  # 3D

        with pytest.raises(ValueError, match="dimensions"):
            goal_vector(positions, goal)


class TestGoalDirection:
    """Test goal_direction function."""

    def test_basic_goal_direction(self):
        """Test computing direction (angle) to goal."""
        from neurospatial.metrics.goal_directed import goal_direction

        positions = np.array([[0.0, 0.0]])
        goal = np.array([1.0, 0.0])  # East

        result = goal_direction(positions, goal)

        assert result.shape == (1,)
        assert_allclose(result[0], 0.0, atol=1e-10)  # 0 radians = East

    def test_goal_direction_various_angles(self):
        """Test goal direction for various angles."""
        from neurospatial.metrics.goal_directed import goal_direction

        # Position at origin, goals in different directions
        positions = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        goals_individual = [
            np.array([1.0, 0.0]),  # East -> 0
            np.array([0.0, 1.0]),  # North -> pi/2
            np.array([-1.0, 0.0]),  # West -> pi
            np.array([0.0, -1.0]),  # South -> -pi/2
        ]

        # Test each individually since goal is single position
        expected_angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]

        for pos, goal, expected in zip(
            positions, goals_individual, expected_angles, strict=True
        ):
            result = goal_direction(pos.reshape(1, -1), goal)
            assert_allclose(result[0], expected, atol=1e-10)


class TestInstantaneousGoalAlignment:
    """Test instantaneous_goal_alignment function."""

    def test_direct_approach_to_goal(self):
        """Test alignment is ~1.0 when moving directly toward goal."""
        from neurospatial.metrics.goal_directed import instantaneous_goal_alignment

        # Moving directly East toward goal at (100, 0)
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = instantaneous_goal_alignment(positions, times, goal, min_speed=0.0)

        # Should be ~1.0 everywhere (ignoring edge effects)
        valid_alignment = result[~np.isnan(result)]
        assert len(valid_alignment) > 0
        assert np.mean(valid_alignment) > 0.9

    def test_moving_away_from_goal(self):
        """Test alignment is negative when moving away from goal."""
        from neurospatial.metrics.goal_directed import instantaneous_goal_alignment

        # Moving West, away from goal at (100, 0)
        n_samples = 21
        positions = np.column_stack(
            [
                np.linspace(50, 0, n_samples),  # Moving West
                np.zeros(n_samples),
            ]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = instantaneous_goal_alignment(positions, times, goal, min_speed=0.0)

        # Should be ~-1.0 (moving away)
        valid_alignment = result[~np.isnan(result)]
        assert len(valid_alignment) > 0
        assert np.mean(valid_alignment) < -0.8

    def test_orthogonal_movement(self):
        """Test alignment is ~0 when moving in circle around goal.

        Note: Moving in a straight line perpendicular to goal from origin
        doesn't stay perpendicular - the goal direction changes as position
        changes. For truly orthogonal movement, we need circular motion around
        the goal where velocity is always tangent to the circle.
        """
        from neurospatial.metrics.goal_directed import instantaneous_goal_alignment

        # Circular path around goal at (50, 50), always tangent to goal direction
        n_samples = 100
        t = np.linspace(0, 2 * np.pi, n_samples)
        radius = 30.0
        center = np.array([50.0, 50.0])
        positions = np.column_stack(
            [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = center  # Goal at center of circle

        result = instantaneous_goal_alignment(positions, times, goal, min_speed=0.0)

        # Should be ~0 (always orthogonal to radial direction)
        valid_alignment = result[~np.isnan(result)]
        assert len(valid_alignment) > 0
        # Circular motion is always perpendicular to goal direction (radial)
        assert abs(np.mean(valid_alignment)) < 0.15

    def test_stationary_periods_are_nan(self):
        """Test that stationary periods (below min_speed) are NaN."""
        from neurospatial.metrics.goal_directed import instantaneous_goal_alignment

        # Position that doesn't change
        n_samples = 10
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = instantaneous_goal_alignment(positions, times, goal, min_speed=5.0)

        # All should be NaN due to zero speed
        assert np.all(np.isnan(result))


class TestGoalBias:
    """Test goal_bias function."""

    def test_strong_goal_directed(self):
        """Test goal_bias > 0.8 for direct approach."""
        from neurospatial.metrics.goal_directed import goal_bias

        # Moving directly toward goal
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = goal_bias(positions, times, goal, min_speed=0.0)

        assert result > 0.8

    def test_negative_goal_bias_moving_away(self):
        """Test goal_bias < -0.5 when moving away from goal."""
        from neurospatial.metrics.goal_directed import goal_bias

        # Moving away from goal
        n_samples = 21
        positions = np.column_stack(
            [
                np.linspace(50, 0, n_samples),  # Moving West
                np.zeros(n_samples),
            ]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = goal_bias(positions, times, goal, min_speed=0.0)

        assert result < -0.5

    def test_circular_path_around_goal(self):
        """Test goal_bias ~0 for circular path around goal."""
        from neurospatial.metrics.goal_directed import goal_bias

        # Circular path around goal at (50, 50)
        n_samples = 100
        t = np.linspace(0, 2 * np.pi, n_samples)
        radius = 20.0
        center = np.array([50.0, 50.0])
        positions = np.column_stack(
            [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = center  # Goal at center of circle

        result = goal_bias(positions, times, goal, min_speed=0.0)

        # Should be close to 0 (equal time moving toward and away)
        assert abs(result) < 0.3

    def test_stationary_returns_nan(self):
        """Test that all-stationary trajectory returns NaN."""
        from neurospatial.metrics.goal_directed import goal_bias

        n_samples = 10
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = goal_bias(positions, times, goal, min_speed=5.0)

        assert np.isnan(result)


class TestApproachRate:
    """Test approach_rate function."""

    def test_approaching_goal_negative_rate(self):
        """Test that approaching goal gives negative approach rate."""
        from neurospatial.metrics.goal_directed import approach_rate

        # Moving toward goal
        n_samples = 11
        positions = np.column_stack(
            [np.linspace(0, 50, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = approach_rate(positions, times, goal, metric="euclidean")

        # Should be negative (distance decreasing)
        assert np.nanmean(result) < 0

    def test_retreating_from_goal_positive_rate(self):
        """Test that moving away gives positive approach rate."""
        from neurospatial.metrics.goal_directed import approach_rate

        # Moving away from goal
        n_samples = 11
        positions = np.column_stack(
            [
                np.linspace(50, 0, n_samples),  # Moving West
                np.zeros(n_samples),
            ]
        )
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])  # Goal in East

        result = approach_rate(positions, times, goal, metric="euclidean")

        # Should be positive (distance increasing)
        assert np.nanmean(result) > 0

    def test_approach_rate_magnitude(self):
        """Test that approach rate magnitude matches speed for direct approach."""
        from neurospatial.metrics.goal_directed import approach_rate

        # Moving 10 units/s directly toward goal
        n_samples = 11
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)  # 100 units in 10 seconds = 10 units/s
        goal = np.array([100.0, 0.0])

        result = approach_rate(positions, times, goal, metric="euclidean")

        # Approach rate magnitude should be ~10 units/s
        valid_rates = result[~np.isnan(result)]
        assert_allclose(np.mean(np.abs(valid_rates)), 10.0, rtol=0.2)


class TestGoalDirectedMetrics:
    """Test GoalDirectedMetrics dataclass."""

    def test_is_goal_directed_method(self):
        """Test is_goal_directed() helper method."""
        from neurospatial.metrics.goal_directed import GoalDirectedMetrics

        result = GoalDirectedMetrics(
            goal_bias=0.6,
            mean_approach_rate=-10.0,
            time_to_goal=5.0,
            min_distance_to_goal=2.0,
            goal_distance_at_start=50.0,
            goal_distance_at_end=2.0,
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        assert result.is_goal_directed(threshold=0.5)
        assert not result.is_goal_directed(threshold=0.7)

    def test_summary_method(self):
        """Test summary() returns formatted string."""
        from neurospatial.metrics.goal_directed import GoalDirectedMetrics

        result = GoalDirectedMetrics(
            goal_bias=0.65,
            mean_approach_rate=-10.5,
            time_to_goal=5.2,
            min_distance_to_goal=2.0,
            goal_distance_at_start=50.0,
            goal_distance_at_end=2.0,
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        summary = result.summary()

        assert "Goal bias" in summary or "goal_bias" in summary.lower()
        assert "0.65" in summary or "0.6" in summary
        assert "5.2" in summary or "time" in summary.lower()


class TestComputeGoalDirectedMetrics:
    """Test compute_goal_directed_metrics function."""

    def test_returns_dataclass(self):
        """Test that compute_goal_directed_metrics returns GoalDirectedMetrics."""
        from neurospatial.metrics.goal_directed import (
            GoalDirectedMetrics,
            compute_goal_directed_metrics,
        )

        # Create environment
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Create trajectory toward goal
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(10, 90, n_samples), np.linspace(10, 90, n_samples)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([90.0, 90.0])

        result = compute_goal_directed_metrics(
            env, positions, times, goal, metric="euclidean"
        )

        assert isinstance(result, GoalDirectedMetrics)
        assert -1.0 <= result.goal_bias <= 1.0
        assert result.min_distance_to_goal >= 0
        assert result.goal_distance_at_start > 0
        assert result.metric == "euclidean"

    def test_direct_approach_metrics(self):
        """Test metrics for direct approach to goal."""
        from neurospatial.metrics.goal_directed import compute_goal_directed_metrics

        # Create environment
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Direct approach to goal
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(10, 90, n_samples), np.zeros(n_samples) + 50]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([90.0, 50.0])

        result = compute_goal_directed_metrics(
            env, positions, times, goal, metric="euclidean", min_speed=0.0
        )

        # Should show strong goal-directed behavior
        assert result.goal_bias > 0.5
        assert result.mean_approach_rate < 0  # Negative = approaching
        assert result.goal_distance_at_end < result.goal_distance_at_start


class TestErrorHandling:
    """Test error handling and messages."""

    def test_mismatched_array_lengths(self):
        """Test helpful error for mismatched positions/times."""
        from neurospatial.metrics.goal_directed import compute_goal_directed_metrics

        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 10)  # Wrong length!
        goal = np.array([50.0, 0.0])

        with pytest.raises(
            ValueError, match="positions and times must have same length"
        ):
            compute_goal_directed_metrics(env, positions, times, goal)

    def test_single_position(self):
        """Test edge case with single position."""
        from neurospatial.metrics.goal_directed import goal_bias

        positions = np.array([[50.0, 50.0]])
        times = np.array([0.0])
        goal = np.array([100.0, 0.0])

        # Should return NaN since can't compute velocity
        result = goal_bias(positions, times, goal, min_speed=0.0)

        assert np.isnan(result)
