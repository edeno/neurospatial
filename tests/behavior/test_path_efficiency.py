"""Tests for path efficiency metrics module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestTraveledPathLength:
    """Test traveled_path_length function."""

    TWO_RECORDINGS = np.vstack([[[0.0, 0.0], [1.0, 0.0]], [[50.0, 0.0], [51.0, 0.0]]])

    def test_straight_line_euclidean(self):
        """Straight line path gives the expected total length."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.arange(21) * 0.1

        length = traveled_path_length(times, positions, metric="euclidean")

        assert_allclose(length, 100.0, rtol=1e-12)

    def test_two_positions(self):
        """Path length with exactly 2 positions (3-4-5 triangle)."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([[0.0, 0.0], [30.0, 40.0]])

        length = traveled_path_length(np.array([0.0, 0.1]), positions)

        assert_allclose(length, 50.0, rtol=1e-12)

    def test_python_list_input_normalized(self):
        """Python-list inputs are coerced; no AttributeError at the boundary."""
        from neurospatial.behavior.navigation import traveled_path_length

        length = traveled_path_length([0.0, 0.1], [[0.0, 0.0], [30.0, 40.0]])

        assert_allclose(length, 50.0, rtol=1e-12)

    @pytest.mark.parametrize(
        "positions",
        [[[0.0, 1.0], [2.0], [3.0, 4.0]], [["a", "b"], ["c", "d"], ["e", "f"]]],
        ids=["ragged", "strings"],
    )
    def test_malformed_input_raises_domain_error(self, positions):
        """Ragged or non-numeric positions get the validator's what/why/fix."""
        from neurospatial.behavior.navigation import traveled_path_length

        with pytest.raises(
            ValueError,
            match=r"(?s)Invalid times/positions passed to traveled_path_length\(\):"
            r"\n- positions must contain numeric values convertible to float64.*"
            r"\nWhy: .*\nFix: pass times as a sorted 1-D array",
        ):
            traveled_path_length([0.0, 0.1, 0.2], positions)

    def test_one_dimensional_positions_are_one_coordinate(self):
        """A 1-D array is one coordinate (for example a linearized track)."""
        from neurospatial.behavior.navigation import traveled_path_length

        length = traveled_path_length([0.0, 0.1, 0.2], [0.0, 1.0, 3.0])

        assert_allclose(length, 3.0)

    def test_geodesic_requires_env(self):
        """Geodesic metric raises without env."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])

        with pytest.raises(ValueError, match="env parameter is required"):
            traveled_path_length(np.arange(21) * 0.1, positions, metric="geodesic")

    def test_unknown_metric_raises(self):
        from neurospatial.behavior.navigation import traveled_path_length

        with pytest.raises(
            ValueError,
            match=r"got metric='manhattan'; it must be .*\n.*\nFix: use metric='euclidean'",
        ):
            traveled_path_length(
                np.array([0.0, 0.1]), np.zeros((2, 2)), metric="manhattan"
            )

    @staticmethod
    def _grid_env():
        x = np.linspace(0, 40, 100)
        xx, yy = np.meshgrid(x, x)
        return Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )

    def test_geodesic_with_env(self):
        """Geodesic path length on bin centers is positive and finite."""
        from neurospatial.behavior.navigation import traveled_path_length

        env = self._grid_env()
        positions = env.bin_centers[np.arange(5)]

        length = traveled_path_length(
            np.arange(5) * 0.1, positions, metric="geodesic", env=env
        )

        assert length > 0
        assert np.isfinite(length)

    def test_geodesic_excludes_interval_outside_environment(self):
        """A sample outside the environment drops its intervals, not the total."""
        from neurospatial.behavior.navigation import traveled_path_length
        from neurospatial.behavior.trajectory import compute_step_lengths

        env = self._grid_env()
        positions = env.bin_centers[np.arange(6)].copy()
        positions[2] = [500.0, 500.0]
        times = np.arange(6) * 0.1

        steps = compute_step_lengths(positions, metric="geodesic", env=env)
        assert np.isnan(steps[1]) and np.isnan(steps[2])
        expected = steps[[0, 3, 4]].sum()

        length = traveled_path_length(times, positions, metric="geodesic", env=env)

        assert np.isfinite(length)
        assert_allclose(length, expected, rtol=1e-12)

    def test_gap_between_recordings_is_not_counted(self):
        """The jump between two recordings (time gap) adds no distance."""
        from neurospatial.behavior.navigation import traveled_path_length

        times = np.array([0.0, 0.1, 100.0, 100.1])

        assert traveled_path_length(times, self.TWO_RECORDINGS) == 2.0

    def test_declared_break_is_not_counted(self):
        from neurospatial.behavior.navigation import traveled_path_length

        times = np.array([0.0, 0.1, 0.2, 0.3])
        breaks = np.array([False, False, True, False])

        length = traveled_path_length(times, self.TWO_RECORDINGS, break_before=breaks)

        assert length == 2.0

    def test_undeclared_jump_on_continuous_clock_is_movement(self):
        from neurospatial.behavior.navigation import traveled_path_length

        times = np.array([0.0, 0.1, 0.2, 0.3])

        assert traveled_path_length(times, self.TWO_RECORDINGS) == 51.0

    def test_nan_row_is_bridged_like_a_deleted_row(self):
        from neurospatial.behavior.navigation import traveled_path_length

        times = np.arange(5) * 0.1
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [2.0, 0.0], [3.0, 0.0]]
        )
        kept = [0, 1, 3, 4]

        length = traveled_path_length(times, positions)

        assert length == traveled_path_length(times[kept], positions[kept]) == 3.0

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_nan_row_gap_longer_than_max_gap_breaks_path(self, metric):
        from neurospatial import Environment
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [2.0, 0.0], [3.0, 0.0]]
        )
        grid = np.c_[np.arange(-1.0, 5.0, 0.5), np.zeros(12)]
        env = Environment.from_samples(grid, bin_size=1.0)

        length = traveled_path_length(
            np.arange(5) * 0.1, positions, metric=metric, env=env, max_gap=0.15
        )

        assert length == 2.0

    def test_epochs_exclude_middle_interval(self):
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        times = np.array([0.0, 0.1, 0.2, 0.3])

        length = traveled_path_length(
            times, positions, epochs=[(0.0, 0.15), (0.18, 0.35)]
        )

        assert length == 2.0

    def test_no_usable_interval_is_nan(self):
        """Single sample and all-missing positions give NaN, not 0.0."""
        from neurospatial.behavior.navigation import traveled_path_length

        assert np.isnan(traveled_path_length(np.array([0.0]), np.array([[1.0, 2.0]])))
        assert np.isnan(
            traveled_path_length(np.arange(3) * 0.1, np.full((3, 2), np.nan))
        )

    def test_stationary_observed_travel_is_zero(self):
        from neurospatial.behavior.navigation import traveled_path_length

        length = traveled_path_length(np.arange(4) * 0.1, np.ones((4, 2)))

        assert length == 0.0

    def test_matches_step_length_sum_on_continuous_recording(
        self, continuous_recording
    ):
        """Gap-free recording: identical to the plain sum of step lengths."""
        from neurospatial.behavior.navigation import traveled_path_length
        from neurospatial.behavior.trajectory import compute_step_lengths

        rec = continuous_recording

        assert_allclose(
            traveled_path_length(rec.times, rec.positions),
            np.sum(compute_step_lengths(rec.positions)),
            rtol=1e-12,
        )
        assert_allclose(
            traveled_path_length(
                rec.times, rec.positions, metric="geodesic", env=rec.env
            ),
            np.sum(compute_step_lengths(rec.positions, metric="geodesic", env=rec.env)),
            rtol=1e-12,
        )


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
    """Test the efficiency ratio reported by compute_path_efficiency."""

    def _efficiency(self, env, positions, goal):
        from neurospatial.behavior.navigation import compute_path_efficiency

        times = np.linspace(0.0, 1.0, len(positions))
        return compute_path_efficiency(
            env, times, positions, goal, metric="euclidean"
        ).efficiency

    def test_straight_path_efficiency_is_one(self):
        """Test that straight path from start to goal has efficiency 1.0."""
        # Create grid environment
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        xx, yy = np.meshgrid(x, y)
        sample_positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Straight line trajectory
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        goal = np.array([50.0, 0.0])

        eff = self._efficiency(env, positions, goal)

        # Straight path should be close to 1.0
        assert_allclose(eff, 1.0, rtol=0.05)

    def test_u_turn_path_efficiency(self):
        """Test that U-turn path has efficiency ~0.5."""
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

        eff = self._efficiency(env, positions, goal)

        # Efficiency should be low due to backtracking
        assert eff < 0.5

    def test_less_than_two_positions_returns_nan(self):
        """Test that < 2 positions returns NaN efficiency."""
        positions = np.array([[50.0, 50.0]])
        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        goal = np.array([60.0, 0.0])

        eff = self._efficiency(env, positions, goal)

        assert np.isnan(eff)

    def test_zero_traveled_returns_nan(self):
        """Test that zero traveled distance returns NaN efficiency."""
        # All positions identical
        positions = np.tile([50.0, 50.0], (5, 1))
        env = Environment.from_samples(
            np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]), bin_size=5.0
        )
        goal = np.array([60.0, 0.0])

        eff = self._efficiency(env, positions, goal)

        assert np.isnan(eff)


class TestPathEfficiencyObservedSupport:
    """Traveled length and efficiency under lost frames, declared breaks, single samples and invalid input."""

    @pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
    def test_gap_free_lengths_equal_step_length_sum(self, continuous_recording, metric):
        from neurospatial.behavior.navigation import (
            angular_efficiency,
            compute_path_efficiency,
        )
        from neurospatial.behavior.trajectory import compute_step_lengths

        r = continuous_recording
        goal = r.positions[-1]
        result = compute_path_efficiency(
            r.env, r.times, r.positions, goal, metric=metric
        )
        assert result.traveled_length == float(
            np.sum(compute_step_lengths(r.positions, metric=metric, env=r.env))
        )
        assert result.efficiency == result.shortest_length / result.traveled_length
        assert result.angular_efficiency == angular_efficiency(
            r.times, r.positions, goal
        )

    POSITIONS = np.column_stack([np.arange(11.0), np.zeros(11)])  # cm
    TIMES = np.arange(11) / 10  # seconds

    @classmethod
    def _straight_run(cls):
        positions = cls.POSITIONS.copy()
        env = Environment.from_samples(positions, bin_size=1.0)
        return env, cls.TIMES.copy(), positions, positions[-1].copy()

    def test_lost_frame_is_bridged(self):
        """One lost frame inside max_gap is bridged like a deleted row."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env, times, positions, goal = self._straight_run()
        positions[5] = np.nan
        kept = np.delete(np.arange(len(times)), 5)

        result = compute_path_efficiency(
            env, times, positions, goal, metric="euclidean"
        )
        deleted = compute_path_efficiency(
            env, times[kept], positions[kept], goal, metric="euclidean"
        )

        assert result.traveled_length == deleted.traveled_length == 10.0
        assert result.efficiency == deleted.efficiency == 1.0
        assert result.angular_efficiency == deleted.angular_efficiency

    def test_lost_stretch_longer_than_max_gap_leaves_ratios_nan(self):
        """A loss longer than max_gap leaves the observed travel, ratios NaN."""
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            traveled_path_length,
        )

        env, times, positions, goal = self._straight_run()
        positions[5] = np.nan
        max_gap = 0.75 * (times[6] - times[4])

        result = compute_path_efficiency(
            env, times, positions, goal, metric="euclidean", max_gap=max_gap
        )

        assert result.traveled_length == 8.0
        assert result.traveled_length == traveled_path_length(
            times, positions, max_gap=max_gap
        )
        # 10 / 8 = 1.25 would claim a path shorter than the shortest one.
        assert np.isnan(result.efficiency)
        # Every observed step heads straight to the goal.
        assert result.angular_efficiency == 1.0
        assert result.shortest_length == 10.0

    def test_declared_break_equals_separate_calls(self):
        """A declared break measures the same travel as two separate calls."""
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            traveled_path_length,
        )

        env, times, positions, goal = self._straight_run()
        break_before = np.arange(11) == 5

        result = compute_path_efficiency(
            env, times, positions, goal, metric="euclidean", break_before=break_before
        )
        first = compute_path_efficiency(
            env, times[:5], positions[:5], goal, metric="euclidean"
        )
        second = compute_path_efficiency(
            env, times[5:], positions[5:], goal, metric="euclidean"
        )

        assert (first.traveled_length, second.traveled_length) == (4.0, 5.0)
        assert result.traveled_length == 9.0
        assert result.traveled_length == (
            first.traveled_length + second.traveled_length
        )
        assert result.traveled_length == traveled_path_length(
            times, positions, break_before=break_before
        )
        assert np.isnan(result.efficiency)
        assert result.angular_efficiency == 1.0

    def test_single_sample_has_no_observed_path(self):
        """One sample is no observed path: NaN, not a stationary or direct one."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env, times, positions, goal = self._straight_run()

        result = compute_path_efficiency(
            env,
            times[:1],
            positions[:1],
            goal,
            metric="euclidean",
            reference_speed=10.0,
        )

        assert np.isnan(result.traveled_length)
        assert np.isnan(result.efficiency)
        assert np.isnan(result.angular_efficiency)
        assert result.time_efficiency is None

    def test_complete_path_keeps_its_ratios(self):
        """A completely observed straight run is perfectly efficient."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env, times, positions, goal = self._straight_run()

        result = compute_path_efficiency(
            env, times, positions, goal, metric="euclidean"
        )

        assert (
            result.traveled_length,
            result.efficiency,
            result.angular_efficiency,
        ) == (10.0, 1.0, 1.0)

    def test_geodesic_step_outside_environment_is_left_out(self):
        """A geodesic step leaving the environment is unobserved travel."""
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            traveled_path_length,
        )

        env = TestTraveledPathLength._grid_env()
        positions = env.bin_centers[np.arange(6)].copy()
        positions[2] = [500.0, 500.0]
        times = np.arange(6) * 0.1
        goal = positions[-1]

        result = compute_path_efficiency(env, times, positions, goal, metric="geodesic")

        assert np.isfinite(result.traveled_length)
        assert result.traveled_length == traveled_path_length(
            times, positions, metric="geodesic", env=env
        )
        assert np.isnan(result.efficiency)

    def test_repeated_timestamp_and_empty_input_raise(self):
        """The clock must be strictly increasing and hold at least one sample."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env, _, _, goal = self._straight_run()
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_path_efficiency(
                env,
                np.array([0.0, 0.1, 0.1, 0.2]),
                self.POSITIONS[:4],
                goal,
                metric="euclidean",
            )
        with pytest.raises(
            ValueError, match=r"compute_path_efficiency\(\) got no samples"
        ):
            compute_path_efficiency(
                env, np.array([]), np.zeros((0, 2)), goal, metric="euclidean"
            )

    def test_malformed_break_before_raises(self):
        """break_before must be a Boolean array, one entry per sample."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        env, times, positions, goal = self._straight_run()
        with pytest.raises(ValueError, match="Boolean array"):
            compute_path_efficiency(
                env,
                times,
                positions,
                goal,
                metric="euclidean",
                break_before=np.zeros(11, dtype=int),
            )


class TestAngularEfficiency:
    """angular_efficiency scores each step's direction against the goal's."""

    TIMES = np.arange(21) / 10  # s
    EAST_RUN = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])  # cm

    def test_run_toward_goal_is_one(self):
        from neurospatial.behavior.navigation import angular_efficiency

        eff = angular_efficiency(self.TIMES, self.EAST_RUN, np.array([150.0, 0.0]))
        assert eff == 1.0

    def test_run_away_from_goal_is_zero(self):
        """The same straight run scores 0 when the goal is behind it."""
        from neurospatial.behavior.navigation import angular_efficiency

        eff = angular_efficiency(self.TIMES, self.EAST_RUN, np.array([-500.0, 0.0]))
        assert eff == 0.0

    def test_steps_perpendicular_to_goal_are_one_half(self):
        """Each step East with the goal due North of its start scores 0.5."""
        from neurospatial.behavior.navigation import angular_efficiency

        times = np.array([0.0, 0.1])
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        eff = angular_efficiency(times, positions, np.array([0.0, 5.0]))
        assert eff == pytest.approx(0.5, abs=1e-15)

    def test_angles_are_weighted_by_step_length(self):
        """A 10 cm step toward (angle 0) and a 1 cm step at pi/2.

        Per unit of distance: 1 - (1 * pi/2) / (pi * 11) = 1 - 1/22.
        """
        from neurospatial.behavior.navigation import angular_efficiency

        times = np.array([0.0, 0.1, 0.2])
        # Goal at (1, 10): from (1, 0) it is due North, the second step East.
        positions = np.array([[1.0, -10.0], [1.0, 0.0], [2.0, 0.0]])
        eff = angular_efficiency(times, positions, np.array([1.0, 10.0]))
        assert eff == pytest.approx(1.0 - 1.0 / 22.0, abs=1e-15)

    def test_a_short_wiggle_after_a_run_weighs_little(self):
        """100 cm toward the goal, then ten 0.1 cm steps across its direction.

        The wiggle is 1 cm of 101 cm at pi/2: 1 - 0.5 / 101. Counting steps
        equally it would be 1 - 0.5 * 10 / 30.
        """
        from neurospatial.behavior.navigation import angular_efficiency

        run = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        wiggle = np.column_stack([np.full(10, 100.0), 0.1 * (np.arange(1, 11) % 2)])
        positions = np.vstack([run, wiggle])
        eff = angular_efficiency(
            np.arange(len(positions)) / 10, positions, np.array([1e7, 0.0])
        )
        assert eff == pytest.approx(1.0 - 0.5 / 101.0, abs=1e-6)

    def test_independent_of_goal_distance_along_heading(self):
        """The old turn-angle score could not tell these goals apart."""
        from neurospatial.behavior.navigation import angular_efficiency

        toward = angular_efficiency(self.TIMES, self.EAST_RUN, np.array([150.0, 0.0]))
        away = angular_efficiency(self.TIMES, self.EAST_RUN, np.array([-150.0, 0.0]))
        assert (toward, away) == (1.0, 0.0)

    def test_two_samples_define_one_step(self):
        from neurospatial.behavior.navigation import angular_efficiency

        eff = angular_efficiency(
            np.array([0.0, 0.1]), self.EAST_RUN[:2], np.array([150.0, 0.0])
        )
        assert eff == 1.0

    def test_identical_positions_return_nan(self):
        """No movement means no direction: NaN, not a perfect score."""
        from neurospatial.behavior.navigation import angular_efficiency

        positions = np.tile([50.0, 50.0], (10, 1))
        eff = angular_efficiency(np.arange(10) / 10, positions, np.array([0.0, 0.0]))
        assert np.isnan(eff)

    def test_long_tracking_loss_drops_its_step(self):
        """A gap longer than max_gap is no step; a backtrack across it is ignored."""
        from neurospatial.behavior.navigation import angular_efficiency

        times = np.array([0.0, 0.1, 5.0, 5.1])
        # The pickup jumps back West across a 4.9 s gap: not a step.
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [-20.0, 0.0], [-19.0, 0.0]])
        goal = np.array([100.0, 0.0])
        assert angular_efficiency(times, positions, goal) == 1.0
        # Bridged: 1 cm toward, 21 cm straight away (pi), 1 cm toward, so
        # 1 - 21 / 23 per unit of distance.
        assert angular_efficiency(times, positions, goal, max_gap=None) == (
            pytest.approx(2.0 / 23.0)
        )

    def test_lost_frame_is_bridged_like_a_deleted_row(self):
        from neurospatial.behavior.navigation import angular_efficiency

        positions = self.EAST_RUN.copy()
        positions[[5, 6]] = [[50.0, 50.0], [np.nan, np.nan]]
        kept = np.delete(np.arange(21), 6)
        goal = np.array([150.0, 0.0])
        assert angular_efficiency(self.TIMES, positions, goal) == angular_efficiency(
            self.TIMES[kept], positions[kept], goal
        )

    def test_goal_dimension_mismatch_raises(self):
        from neurospatial.behavior.navigation import angular_efficiency

        with pytest.raises(ValueError, match="one coordinate per position"):
            angular_efficiency(self.TIMES, self.EAST_RUN, np.array([1.0, 2.0, 3.0]))


class TestTimeEfficiency:
    """Test time_efficiency function."""

    def test_with_reference_speed(self):
        """Test time efficiency computation with reference speed."""
        from neurospatial.behavior.navigation import time_efficiency

        # Trajectory: 100 units in 10 seconds = 10 units/s actual speed
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.linspace(0, 10, 21)
        reference_speed = 20.0  # units/s

        # Shortest path = 100, at reference speed 20 = 5 seconds optimal
        # Actual time = 10 seconds
        # Efficiency = 5/10 = 0.5
        eff = time_efficiency(
            times,
            positions,
            reference_speed=reference_speed,
            optimal_distance=100.0,
        )

        assert_allclose(eff, 0.5, rtol=0.1)


class TestSubgoalEfficiency:
    """subgoal_efficiency cuts the path at each subgoal's closest approach."""

    @staticmethod
    def _corridor():
        x, y = np.meshgrid(np.arange(0.0, 101.0), np.arange(-10.0, 11.0))
        return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)

    def test_two_segment_straight_path(self):
        """A straight run through both subgoals: two perfectly efficient halves."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.arange(21) / 10
        subgoals = np.array([[50.0, 0.0], [100.0, 0.0]])

        result = subgoal_efficiency(env, times, positions, subgoals, metric="euclidean")

        assert [seg.traveled_length for seg in result.segment_results] == [50.0, 50.0]
        assert [seg.shortest_length for seg in result.segment_results] == [50.0, 50.0]
        assert (result.mean_efficiency, result.weighted_efficiency) == (1.0, 1.0)
        assert [seg.angular_efficiency for seg in result.segment_results] == [1.0, 1.0]

    def test_detour_segment_known_efficiency(self):
        """A 10 cm detour North and back before the subgoal: 50 / 70."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.array(
            [[0.0, 0.0], [25.0, 0.0], [25.0, 10.0], [25.0, 0.0], [50.0, 0.0]]
        )
        positions = np.vstack([positions, [[75.0, 0.0], [100.0, 0.0]]])
        times = np.arange(7) / 10
        subgoals = np.array([[50.0, 0.0], [100.0, 0.0]])

        result = subgoal_efficiency(env, times, positions, subgoals, metric="euclidean")

        first, second = result.segment_results
        assert first.traveled_length == 70.0
        assert first.efficiency == pytest.approx(50.0 / 70.0)
        assert second.efficiency == 1.0
        assert result.weighted_efficiency == pytest.approx((50 / 70 * 50 + 50) / 100)

    def test_missing_row_does_not_hide_the_arrival(self):
        """argmin over a NaN row used to return it: every segment was NaN."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        positions[3] = np.nan
        times = np.arange(21) / 10
        subgoals = np.array([[50.0, 0.0], [100.0, 0.0]])

        result = subgoal_efficiency(env, times, positions, subgoals, metric="euclidean")

        assert [seg.traveled_length for seg in result.segment_results] == [50.0, 50.0]
        assert result.mean_efficiency == 1.0

    def test_long_tracking_loss_leaves_the_segment_incomplete(self):
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.arange(21) / 10
        times[5:] += 10.0  # a 10 s loss inside the first segment
        subgoals = np.array([[50.0, 0.0], [100.0, 0.0]])

        result = subgoal_efficiency(env, times, positions, subgoals, metric="euclidean")

        first, second = result.segment_results
        assert not first.fully_observed
        assert np.isnan(first.efficiency)
        assert first.traveled_length == 45.0
        assert second.efficiency == 1.0
        assert result.mean_efficiency == 1.0  # only the observed second segment

    def test_never_approached_subgoal_warns_and_is_nan(self):
        """A subgoal behind the start: no path, NaN (not 0.0 and 1.0 placeholders)."""
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.column_stack([np.linspace(50, 100, 11), np.zeros(11)])
        times = np.arange(11) / 10
        subgoals = np.array([[0.0, 0.0], [100.0, 0.0]])

        with pytest.warns(UserWarning, match="subgoal 0 .* is never approached"):
            result = subgoal_efficiency(
                env, times, positions, subgoals, metric="euclidean"
            )

        behind, ahead = result.segment_results
        assert np.isnan(behind.traveled_length)
        assert np.isnan(behind.efficiency)
        assert np.isnan(behind.angular_efficiency)
        assert behind.shortest_length == 50.0  # measured from the start
        assert ahead.efficiency == 1.0
        assert result.mean_efficiency == 1.0

    def test_geodesic_matches_compute_path_efficiency_for_one_subgoal(self):
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            subgoal_efficiency,
        )

        env = self._corridor()
        s = np.linspace(0, 1, 50)
        positions = np.column_stack([5 + 90 * s, -8 + 16 * s])
        times = s * 5
        goal = positions[-1]

        sub = subgoal_efficiency(env, times, positions, goal)
        direct = compute_path_efficiency(env, times, positions, goal)

        (segment,) = sub.segment_results
        assert segment.traveled_length == pytest.approx(direct.traveled_length)
        assert segment.shortest_length == pytest.approx(direct.shortest_length)
        assert segment.efficiency == pytest.approx(1.0)

    def test_subgoals_of_wrong_shape_raise(self):
        from neurospatial.behavior.navigation import subgoal_efficiency

        env = self._corridor()
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        with pytest.raises(ValueError, match="one coordinate per position dimension"):
            subgoal_efficiency(env, np.arange(21) / 10, positions, np.array([[50.0]]))


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

        with pytest.raises(ValueError, match="same length"):
            compute_path_efficiency(env, times, positions, goal)

    def test_unknown_metric_raises(self):
        """An unrecognized metric raises instead of falling back to geodesic."""
        from neurospatial.behavior.navigation import compute_path_efficiency

        positions = np.column_stack([np.arange(11.0), np.zeros(11)])
        env = Environment.from_samples(positions, bin_size=1.0)

        with pytest.raises(
            ValueError,
            match=r"compute_path_efficiency\(\) got metric='manhattan'; it must be "
            r".*\n.*\nFix: use metric='euclidean'",
        ):
            compute_path_efficiency(
                env, np.arange(11) / 10, positions, positions[-1], metric="manhattan"
            )

    def test_empty_trajectory_error(self):
        """Test helpful error for empty trajectory."""
        from neurospatial.behavior.navigation import traveled_path_length

        positions = np.array([]).reshape(0, 2)

        with pytest.raises(
            ValueError,
            match=r"traveled_path_length\(\) got no samples: times and positions "
            r"are empty",
        ):
            traveled_path_length(np.array([]), positions)

    @pytest.mark.parametrize("goal", [[10.0], [10.0, 0.0, 0.0], [[10.0, 0.0]]])
    def test_goal_of_wrong_shape_raises(self, goal):
        """A one-coordinate goal used to broadcast: efficiency 1.414 on a 2-D run."""
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            shortest_path_length,
        )

        positions = np.column_stack([np.arange(11.0), np.zeros(11)])
        env = Environment.from_samples(positions, bin_size=1.0)
        with pytest.raises(ValueError, match="one coordinate per position dimension"):
            compute_path_efficiency(
                env, np.arange(11) / 10, positions, goal, metric="euclidean"
            )
        with pytest.raises(ValueError, match="one coordinate per position dimension"):
            shortest_path_length(env, positions[0], goal, metric="euclidean")

    def test_non_finite_goal_raises(self):
        from neurospatial.behavior.navigation import compute_path_efficiency

        positions = np.column_stack([np.arange(11.0), np.zeros(11)])
        env = Environment.from_samples(positions, bin_size=1.0)
        with pytest.raises(ValueError, match="goal with a non-finite coordinate"):
            compute_path_efficiency(
                env, np.arange(11) / 10, positions, [np.nan, 0.0], metric="euclidean"
            )

    @pytest.mark.parametrize("reference_speed", [0.0, -2.0, np.nan, np.inf])
    def test_reference_speed_must_be_positive_and_finite(self, reference_speed):
        """A non-positive speed gave -2.0 or ZeroDivisionError, not an error."""
        from neurospatial.behavior.navigation import (
            compute_path_efficiency,
            time_efficiency,
        )

        positions = np.column_stack([np.arange(11.0), np.zeros(11)])
        times = np.arange(11) / 10
        env = Environment.from_samples(positions, bin_size=1.0)
        with pytest.raises(ValueError, match="finite speed greater than zero"):
            compute_path_efficiency(
                env,
                times,
                positions,
                positions[-1],
                metric="euclidean",
                reference_speed=reference_speed,
            )
        with pytest.raises(ValueError, match="finite speed greater than zero"):
            time_efficiency(
                times,
                positions,
                reference_speed=reference_speed,
                optimal_distance=10.0,
            )


def _segment(traveled, shortest, *, fully_observed=True, metric="euclidean"):
    from neurospatial.behavior.navigation import PathEfficiencyResult

    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest,
        time_efficiency=None,
        angular_efficiency=1.0,
        start_position=np.zeros(2),
        goal_position=np.ones(2),
        metric=metric,
        fully_observed=fully_observed,
    )


class TestPathEfficiencyResultContract:
    """Derived fields and invariants of the path-efficiency results."""

    def test_efficiency_is_derived_from_the_lengths(self):
        assert _segment(12.5, 10.0).efficiency == 0.8
        assert np.isnan(_segment(12.5, 10.0, fully_observed=False).efficiency)
        assert np.isnan(_segment(0.0, 10.0).efficiency)
        assert np.isnan(_segment(12.5, np.inf).efficiency)
        assert _segment(np.inf, 10.0).efficiency == 0.0

    def test_efficiency_cannot_be_passed(self):
        from neurospatial.behavior.navigation import PathEfficiencyResult

        with pytest.raises(TypeError, match="efficiency"):
            PathEfficiencyResult(
                traveled_length=1.0,
                shortest_length=1.0,
                efficiency=0.5,
                time_efficiency=None,
                angular_efficiency=1.0,
                start_position=np.zeros(2),
                goal_position=np.ones(2),
                metric="euclidean",
                fully_observed=True,
            )

    def test_is_efficient_is_inclusive_at_threshold(self):
        result = _segment(12.5, 10.0)
        assert result.is_efficient(threshold=0.8)
        assert not result.is_efficient(threshold=0.81)
        assert not _segment(12.5, 10.0, fully_observed=False).is_efficient(0.0)

    def test_invalid_metric_and_negative_length_raise(self):
        with pytest.raises(ValueError, match="metric='taxicab'"):
            _segment(1.0, 1.0, metric="taxicab")
        with pytest.raises(ValueError, match=r"traveled_length=-1\.0 is negative"):
            _segment(-1.0, 1.0)

    def test_record_fields_are_plain_python_scalars(self):
        result = _segment(np.float64(12.5), np.float32(10.0))
        assert type(result.traveled_length) is float
        assert type(result.shortest_length) is float
        assert type(result.efficiency) is float

    def test_subgoal_means_are_derived(self):
        """Weighted by shortest length: (0.5 * 10 + 1.0 * 30) / 40 = 0.875."""
        from neurospatial.behavior.navigation import SubgoalEfficiencyResult

        result = SubgoalEfficiencyResult(
            segment_results=[
                _segment(20.0, 10.0),
                _segment(30.0, 30.0),
                _segment(np.nan, 5.0),
            ],
            subgoal_positions=np.zeros((3, 2)),
        )
        assert isinstance(result.segment_results, tuple)
        assert result.mean_efficiency == 0.75
        assert result.weighted_efficiency == 0.875

    def test_subgoal_weighted_efficiency_is_nan_when_weights_sum_to_zero(self):
        """It used to fall back silently to the unweighted mean."""
        from neurospatial.behavior.navigation import SubgoalEfficiencyResult

        result = SubgoalEfficiencyResult(
            segment_results=(_segment(5.0, 0.0),),
            subgoal_positions=np.zeros((1, 2)),
        )
        assert result.mean_efficiency == 0.0
        assert np.isnan(result.weighted_efficiency)

    def test_subgoal_count_must_match_segments(self):
        from neurospatial.behavior.navigation import SubgoalEfficiencyResult

        with pytest.raises(ValueError, match="one segment result per row"):
            SubgoalEfficiencyResult(
                segment_results=(_segment(1.0, 1.0),),
                subgoal_positions=np.zeros((2, 2)),
            )


class TestGeodesicPathLengthEstimands:
    """Geodesic traveled and shortest lengths are lengths inside the environment.

    The bin-graph estimate made a straight oblique run read 0.79-0.83
    efficient, depending on the bin size and the sampling rate.
    """

    @staticmethod
    def _open_field(bin_size):
        x = np.arange(0.0, 100.0, 0.25)
        xx, yy = np.meshgrid(x, x)
        return Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=bin_size
        )

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    @pytest.mark.parametrize("n_samples", [151, 600, 3000])
    def test_oblique_straight_run_is_perfectly_efficient(self, bin_size, n_samples):
        from neurospatial.behavior.navigation import compute_path_efficiency

        env = self._open_field(bin_size)
        s = np.linspace(0.0, 1.0, n_samples)
        positions = np.column_stack([10 + 70 * s, 10 + 40 * s])
        times = 10 * s

        result = compute_path_efficiency(
            env, times, positions, positions[-1], metric="geodesic"
        )

        assert result.traveled_length == pytest.approx(np.hypot(70, 40), rel=1e-9)
        assert result.shortest_length == pytest.approx(np.hypot(70, 40), rel=1e-12)
        assert result.efficiency == pytest.approx(1.0, rel=1e-9)

    def test_u_maze_run_around_the_wall(self):
        """Up the left arm, across, down the right arm of a U-maze.

        The run is 50 + 40 + 50 = 140 cm. The environment's free space is
        the union of its 2 cm bins, whose inner wall spans x in [10, 40] up
        to y = 50; the shortest path from (5, 5) to (45, 5) touches the
        wall's corners: 2 * hypot(5, 45) + 30 = 120.55 cm. The taut path
        through bin centers may round each corner by up to one bin.
        """
        from neurospatial.behavior.navigation import compute_path_efficiency

        cells = np.array(
            [(x + 1.0, y + 1.0) for x in range(0, 50, 2) for y in range(0, 60, 2)]
        )
        u_maze = cells[(cells[:, 0] < 10) | (cells[:, 0] > 40) | (cells[:, 1] > 50)]
        env = Environment.from_samples(u_maze, bin_size=2.0)
        up = np.column_stack([np.full(100, 5.0), np.linspace(5, 55, 100)])
        across = np.column_stack([np.linspace(5, 45, 80), np.full(80, 55.0)])
        down = np.column_stack([np.full(100, 45.0), np.linspace(55, 5, 100)])
        positions = np.vstack([up, across[1:], down[1:]])
        times = np.arange(len(positions)) / 30

        result = compute_path_efficiency(
            env, times, positions, positions[-1], metric="geodesic"
        )

        exact = 2 * np.hypot(5.0, 45.0) + 30.0
        assert result.traveled_length == pytest.approx(140.0, rel=1e-12)
        assert exact <= result.shortest_length <= exact + 2 * 2.0
        assert exact / 140 <= result.efficiency <= (exact + 4.0) / 140
