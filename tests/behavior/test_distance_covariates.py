"""Tests for distance_to_reward and distance_to_boundary."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Regions

# =============================================================================
# Test distance_to_reward
# =============================================================================


@pytest.fixture
def simple_grid_env():
    """Create a simple 10x10 grid environment for testing."""
    from neurospatial import Environment

    # Create positions covering 10x10 grid
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(positions, bin_size=1.0)
    return env


class TestDistanceToReward:
    """Tests for distance_to_reward function."""

    def test_basic_functionality(self, simple_grid_env):
        """Test basic distance to reward calculation."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        # Trajectory moving from (0,0) to (9,9)
        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (midpoint)
        reward_times = np.array([5.0])

        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean"
        )

        assert result.shape == (n_samples,)
        assert not np.all(np.isnan(result))
        # At t=5, animal is at midpoint, reward is there, distance ~0
        mid_idx = n_samples // 2
        assert result[mid_idx] < result[0]  # Closer at reward time

    def test_mode_nearest(self, simple_grid_env):
        """Test mode='nearest' selects closest reward in time."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.full(n_samples, 5.0),  # Stay at center
                np.full(n_samples, 5.0),
            ]
        )

        # Two rewards at different locations and times
        reward_times = np.array([2.0, 8.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="nearest",
            metric="euclidean",
        )

        # At t=2, should be distance to (0,5) which is 5 units
        idx_at_2 = 20  # t=2.0
        # At t=8, should be distance to (9,5) which is 4 units
        idx_at_8 = 80  # t=8.0

        assert result[idx_at_2] == pytest.approx(5.0, abs=0.5)
        assert result[idx_at_8] == pytest.approx(4.0, abs=0.5)

    def test_mode_last(self, simple_grid_env):
        """Test mode='last' uses most recent reward."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="last",
            metric="euclidean",
        )

        # Before first reward (t<3): NaN
        assert np.isnan(result[0])
        assert np.isnan(result[10])

        # After first reward (3<=t<7): distance to (0,5)
        assert not np.isnan(result[30])  # t=3.0
        assert not np.isnan(result[50])  # t=5.0

        # After second reward (t>=7): distance to (9,5)
        assert not np.isnan(result[70])  # t=7.0
        assert not np.isnan(result[90])  # t=9.0

    def test_mode_next(self, simple_grid_env):
        """Test mode='next' uses upcoming reward."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="next",
            metric="euclidean",
        )

        # Before first reward: distance to (0,5)
        assert not np.isnan(result[0])
        assert not np.isnan(result[20])

        # Between rewards (3<=t<7): distance to (9,5)
        assert not np.isnan(result[50])

        # After last reward: NaN
        assert np.isnan(result[80])
        assert np.isnan(result[99])

    def test_empty_rewards_returns_nan(self, simple_grid_env):
        """Test empty reward_times returns all NaN."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        reward_times = np.array([])

        result = distance_to_reward(env, times, positions, reward_times)

        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        times = np.array([])
        positions = np.empty((0, 2))
        reward_times = np.array([1.0, 2.0])

        result = distance_to_reward(env, times, positions, reward_times)

        assert len(result) == 0

    def test_reward_positions_interpolated(self, simple_grid_env):
        """Test reward positions are interpolated when not provided."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        # Trajectory from (0,0) to (9,9)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (position should be ~(4.5, 4.5))
        reward_times = np.array([5.0])

        # Without explicit reward_positions
        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean"
        )

        # At t=5, animal is at ~(4.5, 4.5), reward interpolated to same
        mid_idx = n_samples // 2
        assert result[mid_idx] < 1.0  # Close to zero at reward time

    def test_reward_time_outside_session_has_unknown_location(self, simple_grid_env):
        """A reward outside the recording is not clipped to the session edge."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=20.0 is past the end of the session (times max = 10.0).
        reward_times = np.array([20.0])

        with pytest.warns(UserWarning, match=r"1 of 1 reward[\s\S]*1 outside the"):
            result = distance_to_reward(
                env, times, positions, reward_times, metric="euclidean"
            )
        assert result.shape == (n_samples,)
        assert np.isnan(result).all()

    def test_reward_time_within_session_does_not_warn(self, simple_grid_env):
        """Reward times inside the session must not trigger the clip warning."""
        import warnings

        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )
        reward_times = np.array([3.0, 7.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            distance_to_reward(env, times, positions, reward_times, metric="euclidean")

    def test_position_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched positions and times raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        times = np.array([0.0, 1.0])  # Wrong length
        reward_times = np.array([0.5])

        with pytest.raises(ValueError, match=r"positions and times.*same length"):
            distance_to_reward(env, times, positions, reward_times)

    def test_reward_positions_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched reward_positions and reward_times raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        times = np.array([0.0, 1.0])
        reward_times = np.array([0.5, 0.8])
        reward_positions = np.array([[1.0, 1.0]])  # Wrong length

        with pytest.raises(ValueError, match=r"reward_positions and reward_times"):
            distance_to_reward(
                env, times, positions, reward_times, reward_positions=reward_positions
            )

    def test_geodesic_respects_graph(self, simple_grid_env):
        """Test geodesic metric uses graph distances."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        # Position at (0, 0), reward at (1, 1)
        positions = np.array([[0.0, 0.0]])
        times = np.array([0.0])
        reward_times = np.array([0.0])
        reward_positions = np.array([[1.0, 1.0]])

        # Geodesic distance (Manhattan-like on grid)
        result_geodesic = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="geodesic",
        )

        # Euclidean distance (sqrt(2) ~= 1.41)
        result_euclidean = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="euclidean",
        )

        # Geodesic should be >= Euclidean (diagonal not directly connected)
        assert result_geodesic[0] >= result_euclidean[0]


class TestDistanceToRewardValidation:
    """Tests for non-finite and unsorted-time guards in distance_to_reward."""

    def test_distance_to_reward_nan_reward_time_raises(self, simple_grid_env):
        """A NaN in reward_times must raise, naming the argument."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.linspace(0, 9, n_samples)]
        )
        reward_times = np.array([2.0, np.nan])

        with pytest.raises(ValueError, match="reward_times"):
            distance_to_reward(env, times, positions, reward_times)

    def test_distance_to_reward_nan_position_is_missing(self, simple_grid_env):
        """A NaN position is a missing sample: NaN distance, no error."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.linspace(0, 9, n_samples)]
        )
        positions[3, 0] = np.nan
        reward_times = np.array([2.0])  # exactly on sample 2

        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean", max_gap=None
        )

        assert np.isnan(result[3])
        assert np.isfinite(np.delete(result, 3)).all()
        assert result[2] == 0.0

    def test_distance_to_reward_infinite_position_raises(self, simple_grid_env):
        """An infinite position is a corrupt value, not a missing sample."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        times = np.linspace(0, 0.9, 10)
        positions = np.full((10, 2), 5.0)
        positions[3, 0] = np.inf

        with pytest.raises(ValueError, match="positions"):
            distance_to_reward(env, times, positions, np.array([0.2]))

    def test_distance_to_reward_unsorted_times_raises(self, simple_grid_env):
        """Descending times with inferred reward positions must raise."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        # Descending step at index 2 -> 3.
        times = np.array([0.0, 1.0, 5.0, 2.0, 6.0])
        positions = np.column_stack([np.linspace(0, 9, 5), np.linspace(0, 9, 5)])
        reward_times = np.array([3.0])

        with pytest.raises(ValueError, match=r"ascending|sorted"):
            distance_to_reward(env, times, positions, reward_times)

    def test_distance_to_reward_unsorted_times_ok_with_explicit_positions(
        self, simple_grid_env
    ):
        """Unsorted times with explicit reward_positions must not raise."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        times = np.array([0.0, 1.0, 5.0, 2.0, 6.0])
        positions = np.full((5, 2), 5.0)
        reward_times = np.array([3.0])
        reward_positions = np.array([[5.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="euclidean",
        )

        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_distance_to_reward_finite_inputs_unchanged(self, simple_grid_env):
        """A finite, sorted example returns correct, unchanged distances."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        # Animal stays on the y=5 line, moving x from 0 to 9.
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.full(n_samples, 5.0)]
        )
        # Reward inferred at t=4.5 -> x=4.5, y=5.
        reward_times = np.array([4.5])

        # 1 Hz demonstration samples are continuous tracking here.
        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean", max_gap=None
        )

        assert result.shape == (n_samples,)
        assert np.all(np.isfinite(result))
        # Distance decreases monotonically toward the reward, then increases.
        reward_idx = int(np.argmin(result))
        assert np.all(np.diff(result[: reward_idx + 1]) <= 1e-9)
        assert np.all(np.diff(result[reward_idx:]) >= -1e-9)
        # First sample is at x=0; reward at x=4.5 -> distance 4.5.
        assert result[0] == pytest.approx(4.5, abs=0.6)


class TestDistanceToRewardSupport:
    """Reward locations and reward selection follow the observation-support rules."""

    @pytest.fixture
    def run(self):
        """A 10 Hz run along y = 5 from x = 0 to x = 9 over 1 s."""
        times = np.arange(10) * 0.1
        positions = np.column_stack([np.arange(10.0), np.full(10, 5.0)])
        return times, positions

    @staticmethod
    def _distance(env, times, positions, reward_times, *args, **kwargs):
        from neurospatial.behavior.navigation import distance_to_reward

        return distance_to_reward(
            env, times, positions, reward_times, *args, metric="euclidean", **kwargs
        )

    def test_reward_exactly_on_a_sample_uses_that_sample(self, simple_grid_env, run):
        times, positions = run
        positions = positions.copy()
        positions[5] = np.nan  # the next sample is missing; no interpolation needed

        result = self._distance(simple_grid_env, times, positions, times[[4]])

        expected = np.abs(np.arange(10.0) - 4.0)
        expected[5] = np.nan
        assert_array_equal(result, expected)

    def test_reward_between_samples_is_interpolated(self, simple_grid_env, run):
        times, positions = run
        result = self._distance(simple_grid_env, times, positions, np.array([0.25]))
        # Interpolated to x = 2.5, between the bins centered at x = 2 and 3;
        # the reward's bin, not the raw coordinate, sets the distance.
        assert np.isfinite(result).all()
        assert result[[2, 3]].min() == 0.0
        assert result[[2, 3]].max() == 1.0

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            ("gap", "in a gap longer than max_gap"),
            ("break", "across a declared break"),
            ("missing", "before the first or after the last observed position"),
        ],
    )
    def test_reward_in_an_unusable_interval_has_unknown_location(
        self, simple_grid_env, run, change, reason
    ):
        """The samples that select the reward get NaN; no fallback to another reward."""
        times, positions = run
        times, positions = times.copy(), positions.copy()
        kwargs = {}
        reward_times = np.array([0.15, 0.55])  # inside intervals 1 and 5
        if change == "gap":
            times[6:] += 2.0  # interval 5 is a 2 s tracking pause
            reward_times[1] = times[5] + 1.0
        elif change == "break":
            kwargs["break_before"] = np.arange(10) == 6
        else:
            positions[6:] = np.nan

        with pytest.warns(UserWarning, match=rf"1 of 2 reward[\s\S]*1 {reason}"):
            result = self._distance(
                simple_grid_env, times, positions, reward_times, mode="next", **kwargs
            )

        # Samples 0 and 1 use the first reward, at x = 1.5.
        assert np.isfinite(result[:2]).all()
        # Samples 2..5 select the second reward, whose location is unknown.
        assert np.isnan(result[2:6]).all()

    def test_reward_in_a_bridged_gap_is_interpolated_across_missing_samples(
        self, simple_grid_env, run
    ):
        times, positions = run
        positions = positions.copy()
        positions[5:7] = np.nan  # x = 4 at 0.4 s, then x = 7 at 0.7 s
        kept = np.delete(np.arange(10), [5, 6])
        reward_times = np.array([0.55])

        result = self._distance(simple_grid_env, times, positions, reward_times)
        deleted = self._distance(
            simple_grid_env, times[kept], positions[kept], reward_times
        )

        assert_array_equal(result[kept], deleted)
        assert np.isfinite(deleted).all()

    def test_unknown_reward_is_not_replaced_by_a_known_one(self, simple_grid_env, run):
        times, positions = run
        times = times.copy()
        times[6:] += 2.0
        reward_times = np.array([times[1], times[5] + 1.0])  # second in the pause

        with pytest.warns(UserWarning, match="could not be inferred"):
            result = self._distance(
                simple_grid_env, times, positions, reward_times, mode="nearest"
            )

        # Samples 6..9 are nearest in time to the unknown reward: NaN, not the
        # distance to the first reward.
        assert np.isnan(result[6:]).all()
        assert_array_equal(result[:2], [1.0, 0.0])

    def test_nan_positions_give_nan_distance(self, simple_grid_env, run):
        times, positions = run
        positions = positions.copy()
        positions[[2, 7]] = np.nan
        reward_positions = np.array([[4.0, 5.0]])

        result = self._distance(
            simple_grid_env, times, positions, np.array([0.4]), reward_positions
        )

        expected = np.abs(np.arange(10.0) - 4.0)
        expected[[2, 7]] = np.nan
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", ["nearest", "last", "next"])
    @pytest.mark.parametrize("explicit", [True, False])
    def test_declared_break_equals_the_same_data_shifted_past_max_gap(
        self, simple_grid_env, run, mode, explicit
    ):
        """A declared break blocks interpolation exactly as a long gap does.

        The rewards are chosen so the time shift does not change which reward
        each sample selects: one at the start, one inside the join between
        samples 5 and 6, and one exactly on the first sample after it.
        """
        times, positions = run
        breaks = np.arange(10) == 6
        shifted = times.copy()
        shifted[6:] += 1.0
        reward_times = np.array([0.0, 0.55, times[6]])
        shifted_rewards = np.array([0.0, 0.55, shifted[6]])
        reward_positions = (
            np.array([[0.0, 5.0], [5.0, 5.0], [6.0, 5.0]]) if explicit else None
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            declared = self._distance(
                simple_grid_env,
                times,
                positions,
                reward_times,
                reward_positions,
                mode=mode,
                break_before=breaks,
            )
            gapped = self._distance(
                simple_grid_env,
                shifted,
                positions,
                shifted_rewards,
                reward_positions,
                mode=mode,
            )

        assert_array_equal(declared, gapped)
        assert np.isfinite(declared).any()
        # Inferred: the reward in the join has an unknown location in both.
        assert len(caught) == (0 if explicit else 2)
        if not explicit and mode == "next":
            assert np.isnan(declared[1:6]).all()

    def test_a_reward_in_another_recording_is_never_selected(
        self, simple_grid_env, run
    ):
        times, positions = run
        breaks = np.arange(10) == 5
        reward_times = np.array([0.2, 0.8])
        reward_positions = np.array([[2.0, 5.0], [8.0, 5.0]])
        distances = {
            mode: self._distance(
                simple_grid_env,
                times,
                positions,
                reward_times,
                reward_positions,
                mode=mode,
                break_before=breaks,
            )
            for mode in ("last", "nearest", "next")
        }

        x = np.arange(10.0)
        nan = np.nan
        # Recording 2 (samples 5..9) has no reward of its own before 0.8 s.
        assert_array_equal(distances["last"], [nan, nan, 0, 1, 2, nan, nan, nan, 0, 1])
        # Sample 5 starts recording 2 and may only use the second reward.
        assert_array_equal(
            distances["nearest"], np.r_[np.abs(x[:5] - 2), np.abs(x[5:] - 8)]
        )
        # Recording 1 has no reward of its own after 0.2 s.
        assert_array_equal(distances["next"], [2, 1, 0, nan, nan, 3, 2, 1, 0, nan])

    def test_without_breaks_a_gap_does_not_restrict_selection(
        self, simple_grid_env, run
    ):
        """A tracking dropout does not change which reward a sample uses."""
        times, positions = run
        times = times.copy()
        times[5:] += 5.0

        last = self._distance(
            simple_grid_env,
            times,
            positions,
            np.array([0.2]),
            np.array([[2.0, 5.0]]),
            mode="last",
        )

        assert_array_equal(last[5:], np.abs(np.arange(5.0, 10.0) - 2))

    def test_samples_outside_epochs_are_nan(self, simple_grid_env, run):
        times, positions = run

        result = self._distance(
            simple_grid_env,
            times,
            positions,
            np.array([0.0]),
            np.array([[0.0, 5.0]]),
            epochs=[(0.2, 0.5)],
        )

        expected = np.full(10, np.nan)
        expected[2:5] = [2.0, 3.0, 4.0]  # half-open [0.2, 0.5): sample 5 is out
        assert_array_equal(result, expected)

    def test_reward_outside_running_epochs_keeps_its_inferred_location(
        self, simple_grid_env
    ):
        """Rewards come while the animal sits at the port, outside running epochs."""
        times = np.arange(10) * 0.1
        # Runs from x = 0 to x = 5 in 0.5 s, then sits at the port at x = 5.
        x = np.r_[np.arange(5.0), np.full(5, 5.0)]
        positions = np.column_stack([x, np.full(10, 5.0)])
        running = [(0.0, 0.5)]
        reward_times = np.array([0.75])  # between two stationary samples

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the location is known: no warning
            result = self._distance(
                simple_grid_env,
                times,
                positions,
                reward_times,
                mode="next",
                epochs=running,
            )

        # In-epoch samples select the reward and measure to the port at x = 5.
        assert_array_equal(result[:5], [5.0, 4.0, 3.0, 2.0, 1.0])
        # Samples outside the running epochs still give NaN.
        assert np.isnan(result[5:]).all()

    def test_invalid_break_before_raises_the_shared_error(self, simple_grid_env, run):
        times, positions = run
        with pytest.raises(ValueError, match=r"break_before to be a Boolean array"):
            self._distance(
                simple_grid_env,
                times,
                positions,
                np.array([0.2]),
                break_before=np.zeros(10, dtype=int),
            )

    def test_break_before_needs_sorted_times(self, simple_grid_env):
        times = np.array([0.0, 0.1, 0.3, 0.2])
        positions = np.full((4, 2), 5.0)
        with pytest.raises(ValueError, match=r"ascending"):
            self._distance(
                simple_grid_env,
                times,
                positions,
                np.array([0.1]),
                np.array([[5.0, 5.0]]),
                break_before=np.array([False, False, True, False]),
            )

    @pytest.mark.parametrize("mode", ["nearest", "last", "next"])
    @pytest.mark.parametrize("metric", ["geodesic", "euclidean"])
    def test_continuous_tracking_matches_whole_session_interpolation(
        self, simple_grid_env, mode, metric
    ):
        """Gap-free, NaN-free tracking keeps the earlier whole-session result.

        The reference interpolates every reward over the whole session and
        picks each sample's reward by brute force.
        """
        from neurospatial.behavior.navigation import distance_to_reward
        from neurospatial.ops.distance import distance_field

        env = simple_grid_env
        rng = np.random.default_rng(5)
        times = np.cumsum(np.r_[0.0, rng.uniform(0.02, 0.04, 299)])
        positions = np.clip(4.5 + np.cumsum(rng.normal(0, 0.3, (300, 2)), 0), 0, 9)
        # Five rewards inside the session, one exactly on a sample.
        reward_times = np.sort(np.r_[rng.uniform(times[0], times[-1], 4), times[120]])

        result = distance_to_reward(
            env, times, positions, reward_times, metric=metric, mode=mode
        )

        reward_xy = np.column_stack(
            [np.interp(reward_times, times, positions[:, d]) for d in range(2)]
        )
        reward_bins = env.bin_at(reward_xy)
        position_bins = env.bin_at(positions)
        expected = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            before = np.flatnonzero(reward_times <= t)
            after = np.flatnonzero(reward_times >= t)
            if mode == "last":
                pick = before[-1] if before.size else None
            elif mode == "next":
                pick = after[0] if after.size else None
            else:
                strictly_before = np.flatnonzero(reward_times < t)
                left = strictly_before[-1] if strictly_before.size else None
                right = after[0] if after.size else None
                pick = (
                    left
                    if right is None
                    or (
                        left is not None
                        and t - reward_times[left] <= reward_times[right] - t
                    )
                    else right
                )
            if pick is None or reward_bins[pick] < 0 or position_bins[i] < 0:
                continue
            field = distance_field(
                env.connectivity,
                [int(reward_bins[pick])],
                metric=metric,
                bin_centers=env.bin_centers if metric == "euclidean" else None,
            )
            expected[i] = field[position_bins[i]]

        assert_array_equal(result, expected)
        assert np.isfinite(result).sum() > 100


# =============================================================================
# Test distance_to_boundary
# =============================================================================


class TestDistanceToBoundary:
    """Tests for distance_to_boundary function."""

    def test_basic_edge_distance(self, simple_grid_env):
        """Test basic distance to edge calculation."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Positions: center and corner
        positions = np.array([[5.0, 5.0], [0.0, 0.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        # (5, 5) is four 1 cm bins from the rim at x = 9 or y = 9; the corner
        # bin is on the rim.
        np.testing.assert_allclose(result, [4.0, 0.0])
        np.testing.assert_allclose(
            distance_to_boundary(env, positions, metric="euclidean"), [4.0, 0.0]
        )

    def test_region_boundary_distance(self, simple_grid_env):
        """Test distance to region boundary."""
        from shapely.geometry import box as shapely_box

        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Add a region (simple polygon using shapely)
        goal_polygon = shapely_box(7.0, 7.0, 9.0, 9.0)  # minx, miny, maxx, maxy
        region_builder = Regions(env.regions.values())
        region_builder.add("goal", polygon=goal_polygon)
        env = env.with_regions(region_builder)

        # Positions: one far from region, one near region
        positions = np.array([[1.0, 1.0], [6.0, 6.0]])

        result = distance_to_boundary(
            env, positions, boundary_type="region", region_name="goal"
        )

        assert result.shape == (2,)
        # Position at (1,1) is farther from goal region than (6,6)
        assert result[0] > result[1]

    def test_missing_region_raises(self, simple_grid_env):
        """Test missing region name raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"Region.*not found"):
            distance_to_boundary(
                env, positions, boundary_type="region", region_name="nonexistent"
            )

    def test_region_name_required_for_region_type(self, simple_grid_env):
        """Test region_name is required when boundary_type='region'."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"region_name.*required"):
            distance_to_boundary(env, positions, boundary_type="region")

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.empty((0, 2))

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert len(result) == 0

    def test_geodesic_vs_euclidean(self, simple_grid_env):
        """Test geodesic and euclidean metrics give different results."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        result_geodesic = distance_to_boundary(
            env, positions, boundary_type="edge", metric="geodesic"
        )

        result_euclidean = distance_to_boundary(
            env, positions, boundary_type="edge", metric="euclidean"
        )

        # Both should return valid distances
        assert not np.isnan(result_geodesic[0])
        assert not np.isnan(result_euclidean[0])

    def test_invalid_boundary_type_raises(self, simple_grid_env):
        """Test invalid boundary_type raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"boundary_type"):
            distance_to_boundary(env, positions, boundary_type="invalid")
        # Regions carry no obstacle marker, so there is no obstacle boundary.
        with pytest.raises(ValueError, match=r"'edge' or 'region'"):
            distance_to_boundary(env, positions, boundary_type="obstacle")

    def test_position_outside_env_returns_nan(self, simple_grid_env):
        """Test positions outside environment return NaN."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Position far outside the 10x10 grid
        positions = np.array([[100.0, 100.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        # Position outside env → invalid bin → NaN
        assert np.isnan(result[0])

    def test_t_maze_edge_is_the_three_dead_ends(self):
        """On a T-maze track the edge is the three arm ends, not the arms.

        Each arm is 50 cm of 5 cm bins. A point 20 cm along the stem from its
        dead end is 20 cm from the edge; a point at the junction is 50 cm from
        every dead end.
        """
        import networkx as nx

        from neurospatial import Environment
        from neurospatial.behavior.navigation import distance_to_boundary

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, -50.0))
        graph.add_node(1, pos=(0.0, 0.0))
        graph.add_node(2, pos=(-50.0, 0.0))
        graph.add_node(3, pos=(50.0, 0.0))
        for u, v in [(0, 1), (1, 2), (1, 3)]:
            graph.add_edge(u, v, distance=50.0)
        env = Environment.from_graph(
            graph,
            edge_order=[(0, 1), (1, 2), (1, 3)],
            edge_spacing=0.0,
            bin_size=5.0,
            track_width=5.0,
        )

        positions = np.array([[0.0, -27.5], [0.0, -47.5], [-27.5, 0.0]])
        dist = distance_to_boundary(env, positions, boundary_type="edge")

        np.testing.assert_allclose(dist, [20.0, 0.0, 20.0])

    def test_edge_bins_1d_track_are_the_two_ends(self):
        """On a 1D linear track the boundary is the two end bins.

        A path-graph track has interior degree 2 and endpoint degree 1, so
        the edge bins are exactly the two ends of the track. Distance to the
        boundary must be largest in the middle of the track.
        """
        import networkx as nx

        from neurospatial import Environment

        graph = nx.path_graph(8)
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]["pos"] = (float(i) * 5.0, 0.0)
        for u, v in graph.edges:
            graph.edges[u, v]["distance"] = 5.0
        env = Environment.from_graph(
            graph,
            edge_order=list(graph.edges),
            edge_spacing=0.0,
            bin_size=5.0,
            track_width=5.0,
        )

        edge_bins = env.boundary_bins.tolist()

        # Exactly the two ends of the linearized track.
        assert len(edge_bins) == 2
        assert set(edge_bins) == {0, env.n_bins - 1}

        # The middle of the track is the farthest point from either end.
        from neurospatial.behavior.navigation import distance_to_boundary

        # bin_centers are along the linearized axis; pick the geometric
        # midpoint and the two ends.
        mid_pos = env.bin_centers[env.n_bins // 2]
        end_pos = env.bin_centers[0]
        positions = np.vstack([mid_pos, end_pos])
        dist = distance_to_boundary(env, positions, boundary_type="edge")
        assert dist[0] > dist[1]

    def test_edge_distance_counts_an_inner_hole(self):
        """Distance to the edge is zero beside an inner hole, not only at the rim.

        In a 7 x 7 grid of 1 cm bins with the centre bin missing, the four bins
        orthogonally next to the hole are on the boundary (distance 0), and
        the bin at (1, 1) is one bin from the outer rim (distance 1).
        """
        from neurospatial import Environment
        from neurospatial.behavior.navigation import distance_to_boundary

        xx, yy = np.meshgrid(np.arange(7.0), np.arange(7.0))
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        keep = ~((positions[:, 0] == 3.0) & (positions[:, 1] == 3.0))
        env = Environment.from_samples(positions[keep], bin_size=1.0)

        queries = np.array([[3.0, 2.0], [3.0, 4.0], [2.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
        dist = distance_to_boundary(env, queries, boundary_type="edge")

        np.testing.assert_allclose(dist, [0.0, 0.0, 0.0, 0.0, 1.0])
