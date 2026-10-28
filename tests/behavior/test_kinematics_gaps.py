"""Kinematic intervals preserve observations without bridging recording gaps."""

import numpy as np
import pytest

from neurospatial import Regions
from neurospatial._intervals import observed_interval_mask
from neurospatial.behavior import Trial, compute_planar_turn_angles
from neurospatial.behavior.decisions import (
    compute_pre_decision_metrics,
    decision_region_entry_time,
    pre_decision_heading_stats,
    pre_decision_speed_stats,
)
from neurospatial.behavior.navigation import (
    approach_rate,
    compute_goal_directed_metrics,
    compute_path_efficiency,
    heading_direction_labels,
    instantaneous_goal_alignment,
)
from neurospatial.behavior.trajectory import compute_home_range
from neurospatial.behavior.vte import (
    compute_vte_session,
    compute_vte_trial,
    head_sweep_from_positions,
)


def test_interval_velocity_has_physical_units_and_masks_gap():
    from neurospatial.behavior._kinematics import interval_velocity

    times = np.array([0.0, 0.2, 0.4, 2.4, 2.6])
    positions = np.column_stack([[0.0, 2.0, 6.0, 100.0, 101.0], np.zeros(5)])
    mask = observed_interval_mask(times, max_gap=0.5, epochs=None)
    result = interval_velocity(times, positions, mask)
    assert result.shape == (4, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [[10, 0], [20, 0], [np.nan, np.nan], [5, 0]])


def test_interval_velocity_is_empty_without_intervals():
    from neurospatial.behavior._kinematics import interval_velocity

    for n_samples in (0, 1):
        result = interval_velocity(
            np.zeros(n_samples), np.zeros((n_samples, 2)), np.zeros(0, dtype=bool)
        )
        assert result.shape == (0, 2)


def test_speed_stats_exclude_pause(two_epoch_recording):
    r = two_epoch_recording
    from neurospatial.behavior import compute_speed

    expected = np.concatenate(
        [
            compute_speed(r.times[s], r.positions[s])
            for s in (slice(0, 5000), slice(5000, 10000))
        ]
    )
    mean, minimum = pre_decision_speed_stats(r.times, r.positions)
    assert mean == pytest.approx(np.mean(expected), rel=1e-12, abs=0)
    assert minimum == pytest.approx(np.min(expected), rel=1e-12, abs=0)


def test_head_sweep_sums_runs(two_epoch_recording):
    r = two_epoch_recording
    expected = sum(
        head_sweep_from_positions(r.times[s], r.positions[s], min_speed=5.0)
        for s in (slice(0, 5000), slice(5000, 10000))
    )
    actual = head_sweep_from_positions(r.times, r.positions, min_speed=5.0)
    assert expected > 0
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


def test_path_efficiency_counts_observed_travel_across_pause(two_epoch_recording):
    from neurospatial.behavior.navigation import (
        angular_efficiency,
        traveled_path_length,
    )

    r = two_epoch_recording
    result = compute_path_efficiency(
        r.env,
        r.times,
        r.positions,
        r.positions[-1],
        metric="euclidean",
        reference_speed=10,
    )
    first = compute_path_efficiency(
        r.env, r.times[:5000], r.positions[:5000], r.positions[4999], metric="euclidean"
    )
    second = compute_path_efficiency(
        r.env, r.times[5000:], r.positions[5000:], r.positions[-1], metric="euclidean"
    )
    assert result.traveled_length == traveled_path_length(r.times, r.positions)
    assert result.traveled_length == pytest.approx(
        first.traveled_length + second.traveled_length, rel=1e-12
    )
    assert np.isnan(result.efficiency)
    # A mean over the observed steps of both epochs; the pause is no step.
    assert result.angular_efficiency == angular_efficiency(
        r.times, r.positions, r.positions[-1]
    )
    assert np.isfinite(result.angular_efficiency)
    assert np.isfinite(result.shortest_length)
    assert np.isfinite(result.time_efficiency)
    assert np.isfinite(first.traveled_length)
    assert np.isfinite(first.efficiency)


def test_home_range_dwell_excludes_pause(two_epoch_recording):
    from neurospatial.behavior.trajectory import _sample_dwell_times

    r = two_epoch_recording
    mask = observed_interval_mask(r.times, max_gap=0.5, epochs=None)
    weights = _sample_dwell_times(r.times, mask, np.ones(r.times.size, dtype=bool))
    assert weights.sum() == pytest.approx(200.0, rel=0, abs=1e-9)
    assert weights[4999] == pytest.approx(0.02, rel=0, abs=1e-12)
    assert weights[-1] == pytest.approx(0.02, rel=0, abs=1e-12)
    bins = r.env.bin_at(r.positions)
    totals = np.bincount(bins, weights=weights, minlength=r.env.n_bins)
    visited = np.unique(bins)
    ordered = visited[np.argsort(totals[visited])[::-1]]
    count = np.searchsorted(np.cumsum(totals[ordered]) / weights.sum() * 100, 95) + 1
    # The home range is the occupancy level set at the 95 % cutoff: every bin
    # with as much dwell as the cutoff bin (equal sample counts at 50 Hz give
    # dwell equal up to roundoff).
    level = totals[ordered[count - 1]]
    expected = visited[totals[visited] >= level * (1 - 1e-9)]
    np.testing.assert_array_equal(
        np.sort(compute_home_range(bins, times=r.times)), expected
    )


def test_heading_direction_labels_post_pause_sample(two_epoch_recording):
    r = two_epoch_recording
    labels = heading_direction_labels(r.times, r.positions, min_speed=0)
    # The first sample after the pause is labeled from its own run's motion,
    # never from the jump across the pause.
    after = heading_direction_labels(r.times[5000:], r.positions[5000:], min_speed=0)
    assert labels[5000] == after[0] != "stationary"
    assert labels[5001] != "stationary"


def test_goal_alignment_nan_outside_runs(two_epoch_recording):
    r = two_epoch_recording
    selected = [0, 1, 2, 4999, 5000, 5001, 5002]  # 4999 sits alone
    actual = instantaneous_goal_alignment(
        r.times[selected], r.positions[selected], r.positions[-1], min_speed=0
    )
    assert np.isnan(actual[3])
    assert np.isfinite(actual[[0, 1, 2, 4, 5, 6]]).all()
    full = instantaneous_goal_alignment(
        r.times, r.positions, r.positions[-1], min_speed=0
    )
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING, smoothed_velocity

    # The run's own smoothed velocity at its last sample, never across the gap.
    run = slice(0, 5000)
    velocity = smoothed_velocity(
        r.times[run],
        r.positions[run],
        np.ones(4999, dtype=bool),
        smooth_window=DEFAULT_SPEED_SMOOTHING,
    )[-1]
    goal_vector = r.positions[-1] - r.positions[4999]
    expected = np.dot(velocity, goal_vector) / (
        np.linalg.norm(velocity) * np.linalg.norm(goal_vector)
    )
    assert full[4999] == pytest.approx(expected, rel=1e-12, abs=0)


def test_approach_rate_masks_backward_gap(two_epoch_recording):
    r = two_epoch_recording
    actual = approach_rate(r.times, r.positions, r.positions[-1])
    assert np.isnan(actual[5000])
    assert np.isfinite(actual[[4999, 5001]]).all()


def test_smoothed_turns_are_per_run(two_epoch_recording):
    r = two_epoch_recording
    expected = np.r_[
        compute_planar_turn_angles(
            r.times[:5000], r.positions[:5000], smooth_window=0.2
        ),
        compute_planar_turn_angles(
            r.times[5000:], r.positions[5000:], smooth_window=0.2
        ),
    ]
    actual = compute_planar_turn_angles(r.times, r.positions, smooth_window=0.2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_smoothed_turns_declared_break_equals_separate_calls(two_epoch_recording):
    """A declared break keeps two recordings on one clock from smoothing together."""
    r = two_epoch_recording
    n = 5000
    # Join the second recording directly after the first: no time gap remains.
    times = np.r_[r.times[:n], r.times[n - 1] + 0.02 + (r.times[n:] - r.times[n])]
    new_recording = np.arange(times.size) == n
    joined = compute_planar_turn_angles(
        times, r.positions, break_before=new_recording, smooth_window=0.2
    )
    # Slice the same arrays so the time differences are exactly the same.
    np.testing.assert_array_equal(
        joined,
        np.r_[
            compute_planar_turn_angles(times[:n], r.positions[:n], smooth_window=0.2),
            compute_planar_turn_angles(times[n:], r.positions[n:], smooth_window=0.2),
        ],
    )
    undeclared = compute_planar_turn_angles(times, r.positions, smooth_window=0.2)
    near_join = (times > times[n] - 0.6) & (times < times[n] + 0.6)
    assert not np.allclose(
        undeclared[near_join], joined[near_join], rtol=0, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize(
    "function", [pre_decision_speed_stats, compute_planar_turn_angles]
)
def test_kinematics_excluded_epochs_have_nan(two_epoch_recording, function):
    r = two_epoch_recording
    result = function(positions=r.positions, times=r.times, epochs=(500, 600))
    assert np.isnan(result).all()


def test_home_range_without_observations_is_empty(two_epoch_recording):
    r = two_epoch_recording
    result = compute_home_range(
        r.env.bin_at(r.positions), times=r.times, epochs=(500, 600)
    )
    assert result.size == 0


def test_full_home_range_excludes_zero_dwell_bins():
    times = np.r_[np.arange(215) * 0.02, 1000.0]
    bins = np.arange(len(times))
    observed = compute_home_range(bins, times=times, percentile=100)
    np.testing.assert_array_equal(np.sort(observed), bins[:-1])
    # The isolated sample is still a visit when timestamps are absent.
    untimed = compute_home_range(bins, percentile=100)
    np.testing.assert_array_equal(np.sort(untimed), bins)


def test_smoothed_turns_isolated_sample_nan(two_epoch_recording):
    r = two_epoch_recording
    selected = [0, 1, 2, 4999, 5000, 5001, 5002]  # 4999 sits alone
    turns = compute_planar_turn_angles(
        r.times[selected], r.positions[selected], smooth_window=0.2
    )
    assert np.isnan(turns[3])
    # Each observed run gets the turns it has on its own.
    for run in ([0, 1, 2], [5000, 5001, 5002]):
        alone = compute_planar_turn_angles(
            r.times[run], r.positions[run], smooth_window=0.2
        )
        rows = [selected.index(i) for i in run]
        np.testing.assert_allclose(turns[rows], alone, rtol=1e-12)
        # A three-sample run has one turn, at its middle sample.
        np.testing.assert_array_equal(np.isfinite(alone), [False, True, False])


def test_goal_metrics_keep_known_wall_clock_time(two_epoch_recording):
    """Time to reach the goal is wall-clock time, known across the pause."""
    r = two_epoch_recording
    goal = r.positions[5000]  # first sample of the second recording, t = 1100 s
    across_pause = compute_goal_directed_metrics(
        r.env, r.times, r.positions, goal, goal_radius=0
    )
    second_recording = compute_goal_directed_metrics(
        r.env, r.times, r.positions, goal, goal_radius=0, epochs=(1100, 1200)
    )
    assert across_pause.time_to_reach_goal == 1100
    assert second_recording.time_to_reach_goal == 0.0


def test_goal_metrics_with_no_sample_in_epochs_are_nan(two_epoch_recording):
    """Epochs in the pause hold no sample: nothing is measured or reached.

    Arrival used to be searched outside the epochs, giving 1100 s.
    """
    r = two_epoch_recording
    with pytest.warns(UserWarning):
        result = compute_goal_directed_metrics(
            r.env,
            r.times,
            r.positions,
            r.positions[5000],
            goal_radius=0,
            epochs=(500, 600),
        )
    assert np.isnan(result.goal_bias)
    assert np.isnan(result.mean_approach_rate)
    assert np.isnan(result.goal_distance_at_start)
    assert np.isnan(result.time_to_reach_goal)


def test_composites_forward_coarse_sampling_optout(continuous_recording):
    r = continuous_recording
    positions, times = r.positions[:601:50], r.times[:601:50]
    entry, duration = 10.0, 6.0
    selected = (times >= entry - duration) & (times < entry)
    expected_mean, expected_min = pre_decision_speed_stats(
        times[selected], positions[selected], max_gap=None
    )
    expected_heading = pre_decision_heading_stats(
        times[selected],
        positions[selected],
        max_gap=None,
        min_speed=5.0,
    )
    metrics = compute_pre_decision_metrics(
        times,
        positions,
        entry,
        window_duration=duration,
        max_gap=None,
        min_speed=5.0,
    )
    assert metrics.mean_speed == expected_mean
    assert metrics.min_observed_speed == expected_min
    assert metrics.heading_mean_resultant_length == expected_heading[2]
    trial = compute_vte_trial(
        times, positions, entry, window_duration=duration, max_gap=None, min_speed=5.0
    )
    assert trial.mean_speed == expected_mean
    expected_sweep = head_sweep_from_positions(
        times[selected],
        positions[selected],
        max_gap=None,
        min_speed=5.0,
    )
    assert expected_sweep > 0
    assert trial.head_sweep_magnitude == expected_sweep


def test_vte_session_forwards_coarse_sampling_optout(continuous_recording):
    r = continuous_recording
    positions, times = r.positions[:601:50], r.times[:601:50]
    env = r.env
    bins = env.bin_at(positions)
    region_builder = Regions(env.regions.values())
    region_builder.add("decision", point=tuple(env.bin_centers[bins[-1]]))
    env = env.with_regions(region_builder)
    entry = decision_region_entry_time(bins, times, env, region="decision")
    duration = 6.0
    selected = (times >= entry - duration) & (times < entry)
    assert np.sum(selected) >= 3
    expected_speed = pre_decision_speed_stats(
        times[selected], positions[selected], max_gap=None
    )[0]
    expected_sweep = head_sweep_from_positions(
        times[selected],
        positions[selected],
        max_gap=None,
        min_speed=5.0,
    )
    assert expected_sweep > 0
    with pytest.warns(UserWarning, match="No variation"):
        result = compute_vte_session(
            env,
            times,
            positions,
            decision_region="decision",
            trials=[Trial(times[0], times[-1], "source", "target", "reached")],
            window_duration=duration,
            max_gap=None,
        )
    assert len(result.trial_results) == 1
    assert result.trial_results[0].mean_speed == expected_speed
    assert result.trial_results[0].head_sweep_magnitude == expected_sweep


def test_duplicate_timestamps_give_undefined_not_infinite_speed():
    """A zero-length interval has no velocity; it must not become inf."""
    import warnings

    from neurospatial.behavior._kinematics import interval_velocity
    from neurospatial.behavior.decisions import pre_decision_speed_stats

    times = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [2.5, 0.0], [3.5, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        velocity = interval_velocity(times, positions, np.ones(4, dtype=bool))
    assert np.all(np.isnan(velocity[1]))
    np.testing.assert_allclose(velocity[[0, 2, 3], 0], 10.0)
    # The public speed statistics refuse a repeated timestamp outright.
    with pytest.raises(ValueError, match="strictly increasing"):
        pre_decision_speed_stats(times, positions)


def test_one_dimensional_positions_give_track_speed():
    """1-D (linear-track) positions are one coordinate per sample, not a row."""
    from neurospatial.behavior._kinematics import interval_velocity
    from neurospatial.behavior.decisions import pre_decision_speed_stats

    times = np.arange(0.0, 1.0, 0.1)
    x = 3.0 * np.arange(times.size)  # 30 units per second
    velocity = interval_velocity(times, x, np.ones(times.size - 1, dtype=bool))
    assert velocity.shape == (times.size - 1, 1)
    np.testing.assert_allclose(velocity[:, 0], 30.0)
    mean_speed, min_speed = pre_decision_speed_stats(times, x)
    assert mean_speed == pytest.approx(30.0)
    assert min_speed == pytest.approx(30.0)
