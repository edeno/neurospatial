"""Tests for goal-directed navigation metrics module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestGoalVector:
    """Test goal_vector function."""

    def test_basic_goal_vector(self):
        """Test computing vector from positions to goal."""
        from neurospatial.behavior.navigation import goal_vector

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
        from neurospatial.behavior.navigation import goal_vector

        positions = np.array([[0.0, 0.0], [30.0, 40.0]])
        goal = np.array([100.0, 100.0])

        result = goal_vector(positions, goal)

        assert_allclose(result[0], [100.0, 100.0])
        assert_allclose(result[1], [70.0, 60.0])

    def test_dimension_mismatch_error(self):
        """Test error when goal dimensions don't match positions."""
        from neurospatial.behavior.navigation import goal_vector

        positions = np.array([[0.0, 0.0], [10.0, 0.0]])  # 2D
        goal = np.array([50.0, 0.0, 0.0])  # 3D

        with pytest.raises(
            ValueError, match=r"goal has 3 coordinates but positions have 2"
        ):
            goal_vector(positions, goal)


class TestGoalDirection:
    """Test goal_direction function."""

    def test_basic_goal_direction(self):
        """Test computing direction (angle) to goal."""
        from neurospatial.behavior.navigation import goal_direction

        positions = np.array([[0.0, 0.0]])
        goal = np.array([1.0, 0.0])  # East

        result = goal_direction(positions, goal)

        assert result.shape == (1,)
        assert_allclose(result[0], 0.0, atol=1e-10)  # 0 radians = East

    def test_goal_direction_various_angles(self):
        """Test goal direction for various angles."""
        from neurospatial.behavior.navigation import goal_direction

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
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

        # Moving directly East toward goal at (100, 0)
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)

        # Exactly 1 at every sample but the last, which is at the goal.
        np.testing.assert_allclose(result[:-1], 1.0, rtol=1e-12)
        assert np.isnan(result[-1])

    def test_westward_arrival_is_aligned(self):
        """Running West onto the goal scores 1 up to it and NaN on it.

        At the goal the direction to the goal was arctan2(0, 0) = East, so a
        westward arrival scored -1 there.
        """
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

        positions = np.column_stack([np.linspace(100, 0, 21), np.zeros(21)])
        times = np.linspace(0, 10, 21)
        goal = np.array([0.0, 0.0])

        result = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)

        np.testing.assert_allclose(result[:-1], 1.0, rtol=1e-12)
        assert np.isnan(result[-1])

    def test_moving_away_from_goal(self):
        """Test alignment is negative when moving away from goal."""
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

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

        result = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)

        np.testing.assert_allclose(result, -1.0, rtol=1e-12)

    def test_orthogonal_movement(self):
        """Test alignment is ~0 when moving in circle around goal.

        Note: Moving in a straight line perpendicular to goal from origin
        doesn't stay perpendicular - the goal direction changes as position
        changes. For truly orthogonal movement, we need circular motion around
        the goal where velocity is always tangent to the circle.
        """
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

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

        result = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)

        # Should be ~0 (always orthogonal to radial direction)
        valid_alignment = result[~np.isnan(result)]
        assert len(valid_alignment) > 0
        # Circular motion is always perpendicular to goal direction (radial)
        assert abs(np.mean(valid_alignment)) < 0.15

    def test_stationary_periods_are_nan(self):
        """Test that stationary periods (below min_speed) are NaN."""
        from neurospatial.behavior.navigation import instantaneous_goal_alignment

        # Position that doesn't change
        n_samples = 10
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = instantaneous_goal_alignment(
            times, positions, goal, min_speed=5.0, max_gap=None
        )

        # All should be NaN due to zero speed
        assert np.all(np.isnan(result))


class TestGoalBias:
    """Test goal_bias function."""

    def test_strong_goal_directed(self):
        """A direct approach has goal bias 1."""
        from neurospatial.behavior.navigation import goal_bias

        # Moving directly toward goal
        n_samples = 21
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)
        goal = np.array([100.0, 0.0])

        result = goal_bias(times, positions, goal, min_speed=0.0)

        assert result == pytest.approx(1.0, rel=1e-12)

    def test_westward_approach_has_bias_one(self):
        """The arrival sample used to score -1, pulling the bias to 0.9."""
        from neurospatial.behavior.navigation import goal_bias

        positions = np.column_stack([np.linspace(100, 0, 21), np.zeros(21)])
        times = np.linspace(0, 10, 21)

        result = goal_bias(times, positions, np.array([0.0, 0.0]), min_speed=0.0)

        assert result == pytest.approx(1.0, rel=1e-12)

    def test_negative_goal_bias_moving_away(self):
        """Moving straight away from the goal has goal bias -1."""
        from neurospatial.behavior.navigation import goal_bias

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

        result = goal_bias(times, positions, goal, min_speed=0.0)

        assert result == pytest.approx(-1.0, rel=1e-12)

    def test_circular_path_around_goal(self):
        """Test goal_bias ~0 for circular path around goal."""
        from neurospatial.behavior.navigation import goal_bias

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

        result = goal_bias(times, positions, goal, min_speed=0.0)

        # Should be close to 0 (equal time moving toward and away)
        assert abs(result) < 0.3

    def test_stationary_returns_nan(self):
        """Test that all-stationary trajectory returns NaN."""
        from neurospatial.behavior.navigation import goal_bias

        n_samples = 10
        positions = np.tile([50.0, 50.0], (n_samples, 1))
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = goal_bias(times, positions, goal, min_speed=5.0, max_gap=None)

        assert np.isnan(result)


class TestApproachRate:
    """Test approach_rate function."""

    def test_approaching_goal_negative_rate(self):
        """Test that approaching goal gives negative approach rate."""
        from neurospatial.behavior.navigation import approach_rate

        # Moving toward goal
        n_samples = 11
        positions = np.column_stack(
            [np.linspace(0, 50, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 5, n_samples)
        goal = np.array([100.0, 0.0])

        result = approach_rate(times, positions, goal, metric="euclidean")

        # Should be negative (distance decreasing)
        assert np.nanmean(result) < 0

    def test_retreating_from_goal_positive_rate(self):
        """Test that moving away gives positive approach rate."""
        from neurospatial.behavior.navigation import approach_rate

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

        result = approach_rate(times, positions, goal, metric="euclidean")

        # Should be positive (distance increasing)
        assert np.nanmean(result) > 0

    def test_approach_rate_magnitude(self):
        """Test that approach rate magnitude matches speed for direct approach."""
        from neurospatial.behavior.navigation import approach_rate

        # Moving 10 units/s directly toward goal
        n_samples = 11
        positions = np.column_stack(
            [np.linspace(0, 100, n_samples), np.zeros(n_samples)]
        )
        times = np.linspace(0, 10, n_samples)  # 100 units in 10 seconds = 10 units/s
        goal = np.array([100.0, 0.0])

        result = approach_rate(times, positions, goal, metric="euclidean", max_gap=None)

        # Approach rate magnitude should be ~10 units/s
        valid_rates = result[~np.isnan(result)]
        assert_allclose(np.mean(np.abs(valid_rates)), 10.0, rtol=0.2)

    def test_approach_rate_geodesic_branch(self):
        """metric='geodesic' returns negative rates when approaching, NaN first."""
        from neurospatial.behavior.navigation import approach_rate

        rng = np.random.default_rng(0)
        env = Environment.from_samples(rng.uniform(0, 100, (2000, 2)), bin_size=5.0)

        # Straight walk from (10,50) toward goal at (90,50).
        xs = np.linspace(10.0, 80.0, 20)
        positions = np.column_stack([xs, np.full_like(xs, 50.0)])
        times = np.linspace(0.0, 4.0, 20)
        goal = np.array([90.0, 50.0])

        rates = approach_rate(times, positions, goal, metric="geodesic", env=env)

        assert np.isnan(rates[0])  # first value is NaN by contract
        assert np.nanmean(rates) < 0  # approaching => distance decreasing

    def test_approach_rate_geodesic_requires_env(self):
        """metric='geodesic' without env raises a clear ValueError."""
        from neurospatial.behavior.navigation import approach_rate

        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 11)
        with pytest.raises(
            ValueError, match=r"approach_rate\(metric='geodesic'\) needs env="
        ):
            approach_rate(times, positions, np.array([100.0, 0.0]), metric="geodesic")

    def test_geodesic_rate_around_a_wall_is_smooth(self):
        """Walking up a U-maze arm, away from a goal hidden behind the wall.

        The path to the goal (45, 5) leaves the left arm over the wall's
        corner, so walking North at 7 cm/s shortens it at
        7 * cos(angle to the corner): between 0 and -7 cm/s, changing
        smoothly, never the bin-quantized zeros and spikes.
        """
        from neurospatial.behavior.navigation import approach_rate

        cells = np.array(
            [(x + 1.0, y + 1.0) for x in range(0, 50, 2) for y in range(0, 60, 2)]
        )
        u_maze = cells[(cells[:, 0] < 10) | (cells[:, 0] > 40) | (cells[:, 1] > 50)]
        env = Environment.from_samples(u_maze, bin_size=2.0)
        times = np.arange(0.0, 5.0, 1 / 30)
        positions = np.column_stack([np.full(times.size, 5.0), 5.0 + 7.0 * times])

        rates = approach_rate(
            times, positions, np.array([45.0, 5.0]), metric="geodesic", env=env
        )

        assert np.all((rates[1:] >= -7.0 - 1e-9) & (rates[1:] < 0.0))
        assert np.max(np.abs(np.diff(rates[1:]))) < 0.5

    def test_misspelled_metric_raises(self):
        """A misspelled metric used to run the geodesic branch silently."""
        from neurospatial.behavior.navigation import approach_rate

        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 11)
        with pytest.raises(ValueError, match="got metric='euclidian'"):
            approach_rate(times, positions, np.array([100.0, 0.0]), metric="euclidian")

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    @pytest.mark.parametrize("heading", [0.0, 0.5])
    def test_constant_speed_approach_has_constant_rate(self, metric, heading):
        """20 cm/s straight toward the goal reads -20 cm/s at every sample.

        The geodesic rate used to be quantized by the 2 cm bins: 86% exact
        zeros and spikes of -150 cm/s.
        """
        from neurospatial.behavior.navigation import approach_rate

        x, y = np.meshgrid(np.arange(0.25, 100.0, 0.5), np.arange(0.25, 100.0, 0.5))
        env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        times = np.arange(0.0, 3.0, 1 / 30)  # 30 Hz, 3 s
        direction = np.array([np.cos(heading), np.sin(heading)])
        start = np.array([10.0, 10.0])
        goal = start + 80.0 * direction
        positions = start + 20.0 * times[:, None] * direction  # 20 cm/s

        rates = approach_rate(times, positions, goal, metric=metric, env=env)

        assert np.isnan(rates[0])
        np.testing.assert_allclose(rates[1:], -20.0, rtol=1e-9)

    @pytest.fixture
    def jump(self):
        """Two runs 48 units apart on one continuous 8 Hz clock."""
        times = np.arange(6) / 8
        positions = np.column_stack(
            [np.array([0.0, 1.0, 2.0, 50.0, 51.0, 52.0]), np.zeros(6)]
        )
        return times, positions, np.array([100.0, 0.0])

    def test_undeclared_jump_is_a_rate(self, jump):
        """On a continuous clock, a jump is measured as fast approach."""
        from neurospatial.behavior.navigation import approach_rate

        rates = approach_rate(*jump)
        np.testing.assert_array_equal(rates, [np.nan, -8, -8, -384, -8, -8])

    def test_break_before_leaves_the_join_nan(self, jump):
        """No rate is measured across a declared recording break."""
        from neurospatial.behavior.navigation import approach_rate

        break_before = np.zeros(6, dtype=bool)
        break_before[3] = True
        rates = approach_rate(*jump, break_before=break_before)
        np.testing.assert_array_equal(rates, [np.nan, -8, -8, np.nan, -8, -8])

    def test_nan_row_is_bridged_like_a_deleted_row(self, jump):
        """A missing position has no rate; the rate after it spans the gap."""
        from neurospatial.behavior.navigation import approach_rate

        times, positions, goal = jump
        positions = positions.copy()
        positions[1] = np.nan
        rates = approach_rate(times, positions, goal)
        kept = np.delete(np.arange(len(times)), 1)
        deleted = approach_rate(times[kept], positions[kept], goal)
        np.testing.assert_array_equal(rates[kept], deleted)
        np.testing.assert_array_equal(rates, [np.nan, np.nan, -8, -384, -8, -8])
        # The 0.25 s gap is a tracking loss when max_gap is shorter.
        rates = approach_rate(times, positions, goal, max_gap=0.2)
        np.testing.assert_array_equal(rates, [np.nan, np.nan, np.nan, -384, -8, -8])

    def test_empty_input_raises(self):
        from neurospatial.behavior.navigation import approach_rate

        with pytest.raises(ValueError, match="got no samples"):
            approach_rate(np.array([]), np.zeros((0, 2)), np.array([1.0, 0.0]))

    def test_single_sample_is_nan(self):
        from neurospatial.behavior.navigation import approach_rate

        rates = approach_rate(np.array([0.0]), np.array([[0.0, 0.0]]), [1.0, 0.0])
        np.testing.assert_array_equal(rates, [np.nan])


class TestGoalDirectedMetrics:
    """Test GoalDirectedMetrics dataclass."""

    def test_is_goal_directed_method(self):
        """Test is_goal_directed() helper method."""
        from neurospatial.behavior.navigation import GoalDirectedMetrics

        result = GoalDirectedMetrics(
            goal_bias=0.6,
            mean_approach_rate=-10.0,
            time_to_reach_goal=5.0,
            min_distance_to_goal=2.0,
            goal_distance_at_start=50.0,
            goal_distance_at_end=2.0,
            goal_position=np.array([100.0, 0.0]),
            metric="euclidean",
        )

        assert result.is_goal_directed(threshold=0.5)
        assert not result.is_goal_directed(threshold=0.7)


class TestComputeGoalDirectedMetrics:
    """Test compute_goal_directed_metrics function."""

    def test_direct_approach_metrics(self):
        """Test metrics for direct approach to goal."""
        from neurospatial.behavior.navigation import compute_goal_directed_metrics

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
            env, times, positions, goal, metric="euclidean", min_speed=0.0
        )

        # Should show strong goal-directed behavior
        assert result.goal_bias > 0.5
        assert result.mean_approach_rate < 0  # Negative = approaching
        assert result.goal_distance_at_end < result.goal_distance_at_start


class TestErrorHandling:
    """Test error handling and messages."""

    def test_mismatched_array_lengths(self):
        """Test helpful error for mismatched positions/times."""
        from neurospatial.behavior.navigation import compute_goal_directed_metrics

        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0, 5, 10)  # Wrong length!
        goal = np.array([50.0, 0.0])

        with pytest.raises(ValueError, match="same length"):
            compute_goal_directed_metrics(env, times, positions, goal)

    def test_single_position(self):
        """Test edge case with single position."""
        from neurospatial.behavior.navigation import goal_bias

        positions = np.array([[50.0, 50.0]])
        times = np.array([0.0])
        goal = np.array([100.0, 0.0])

        # Should return NaN since can't compute velocity
        result = goal_bias(times, positions, goal, min_speed=0.0)

        assert np.isnan(result)


def _goal_metrics(**overrides):
    from neurospatial.behavior.navigation import GoalDirectedMetrics

    values = {
        "goal_bias": 0.3,
        "mean_approach_rate": -10.0,
        "time_to_reach_goal": 5.0,
        "min_distance_to_goal": 2.0,
        "goal_distance_at_start": 50.0,
        "goal_distance_at_end": 2.0,
        "goal_position": np.array([100.0, 0.0]),
        "metric": "euclidean",
    }
    values.update(overrides)
    return GoalDirectedMetrics(**values)


def test_is_goal_directed_is_inclusive_at_threshold():
    assert _goal_metrics(goal_bias=0.3).is_goal_directed(threshold=0.3)
    assert not _goal_metrics(goal_bias=np.nan).is_goal_directed(threshold=-1.0)


def test_goal_directed_metrics_invariants():
    with pytest.raises(ValueError, match="metric='taxicab'"):
        _goal_metrics(metric="taxicab")
    with pytest.raises(ValueError, match=r"goal_distance_at_start=-1\.0 is negative"):
        _goal_metrics(goal_distance_at_start=-1.0)
    with pytest.raises(ValueError, match="goal_position has shape"):
        _goal_metrics(goal_position=np.zeros((1, 2)))
    assert type(_goal_metrics(goal_bias=np.float64(0.5)).goal_bias) is float


class TestGoalDirectedDistances:
    """Distance summaries and arrival use the metric and the epochs."""

    @staticmethod
    def _run():
        """10 cm/s East from x = 0 to x = 100 cm, goal at x = 100."""
        x, y = np.meshgrid(np.arange(0.0, 101.0), np.arange(-5.0, 6.0))
        env = Environment.from_samples(
            np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm"
        )
        times = np.arange(101) / 10
        positions = np.column_stack([10.0 * times, np.zeros(101)])
        return env, times, positions, np.array([100.0, 0.0])

    def test_arrival_without_and_with_epochs(self):
        from neurospatial.behavior import compute_goal_directed_metrics

        env, times, positions, goal = self._run()
        whole = compute_goal_directed_metrics(
            env, times, positions, goal, goal_radius=50.0
        )
        # Inside [2, 8) s the first sample is x = 20 and the last x = 79.
        window = compute_goal_directed_metrics(
            env, times, positions, goal, goal_radius=50.0, epochs=(2.0, 8.0)
        )
        assert whole.time_to_reach_goal == pytest.approx(5.0)
        assert (whole.goal_distance_at_start, whole.goal_distance_at_end) == (
            100.0,
            0.0,
        )
        assert window.time_to_reach_goal == pytest.approx(3.0)
        assert window.goal_distance_at_start == pytest.approx(80.0)
        assert window.goal_distance_at_end == pytest.approx(21.0)
        assert window.min_distance_to_goal == pytest.approx(21.0)

    def test_not_reached_is_nan_and_not_tested_is_none(self):
        from neurospatial.behavior import compute_goal_directed_metrics

        env, times, positions, goal = self._run()
        not_reached = compute_goal_directed_metrics(
            env, times, positions, goal, goal_radius=1.0, epochs=(2.0, 8.0)
        )
        not_tested = compute_goal_directed_metrics(env, times, positions, goal)
        assert np.isnan(not_reached.time_to_reach_goal)
        assert not_tested.time_to_reach_goal is None
        assert "not reached" in str(not_reached)
        assert "not tested" in str(not_tested)

    @pytest.mark.parametrize("goal_radius", [-1.0, np.nan])
    def test_goal_radius_must_be_finite_and_non_negative(self, goal_radius):
        from neurospatial.behavior import compute_goal_directed_metrics

        env, times, positions, goal = self._run()
        with pytest.raises(ValueError, match="finite distance >= 0"):
            compute_goal_directed_metrics(
                env, times, positions, goal, goal_radius=goal_radius
            )

    def test_geodesic_distances_go_around_the_wall(self):
        """With metric='geodesic' the summaries are geodesic, not straight-line.

        In a U-maze the goal (45, 5) is 40 cm away in a straight line but
        2 * hypot(5, 45) + 30 = 120.55 cm around the wall (plus at most one
        2 cm bin per corner); it used to report 40.
        """
        from neurospatial.behavior import (
            compute_goal_directed_metrics,
            shortest_path_length,
        )

        cells = np.array(
            [(x + 1.0, y + 1.0) for x in range(0, 50, 2) for y in range(0, 60, 2)]
        )
        u_maze = cells[(cells[:, 0] < 10) | (cells[:, 0] > 40) | (cells[:, 1] > 50)]
        env = Environment.from_samples(u_maze, bin_size=2.0, units="cm")
        times = np.arange(50) / 30
        positions = np.column_stack([np.full(50, 5.0), np.linspace(5, 30, 50)])
        goal = np.array([45.0, 5.0])

        geodesic = compute_goal_directed_metrics(
            env, times, positions, goal, metric="geodesic"
        )
        euclidean = compute_goal_directed_metrics(env, times, positions, goal)

        exact = 2 * np.hypot(5.0, 45.0) + 30.0
        assert euclidean.goal_distance_at_start == 40.0
        assert exact <= geodesic.goal_distance_at_start <= exact + 4.0
        assert geodesic.goal_distance_at_start == shortest_path_length(
            env, positions[0], goal
        )
        assert geodesic.min_distance_to_goal < geodesic.goal_distance_at_start
