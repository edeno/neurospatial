"""Detectors analyze observed intervals without inventing transitions in pauses."""

import warnings

import numpy as np
import pytest

from neurospatial.behavior import (
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_runs_between_regions,
    running_direction_labels,
    segment_by_velocity,
    segment_trials,
)
from neurospatial.behavior.decisions import (
    compute_decision_analysis,
    compute_pre_decision_metrics,
    detect_boundary_crossings,
    extract_pre_decision_window,
)
from neurospatial.behavior.vte import compute_vte_session, compute_vte_trial


def test_no_crossing_reported_across_pause(pause_track):
    r = pause_track
    assert (
        detect_region_crossings(r.position_bins, r.times, r.env, region_name="target")
        == []
    )


def test_crossing_epochs_equal_slicing(continuous_pause_track):
    r = continuous_pause_track
    selected = r.times <= 100
    expected = detect_region_crossings(
        r.position_bins[selected], r.times[selected], r.env, region_name="target"
    )
    actual = detect_region_crossings(
        r.position_bins, r.times, r.env, region_name="target", epochs=(0, 100)
    )
    assert actual == expected


def test_compatibility_warning_is_emitted_once(pause_track):
    r = pause_track
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detect_region_crossings(r.position_bins, r.times, "target", r.env)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1


def test_invalid_epochs_raise_without_observed_samples(pause_track):
    r = pause_track
    with pytest.raises(ValueError, match="epochs"):
        detect_region_crossings(
            np.array([], dtype=np.int64),
            np.array([]),
            r.env,
            region_name="target",
            epochs=(2, 1),
        )


def test_runs_do_not_span_pause(pause_track):
    r = pause_track
    runs = detect_runs_between_regions(
        r.position_bins,
        r.times,
        r.env,
        source="source",
        target="target",
        max_duration=2000,
    )
    assert len(runs) == 1
    assert 98 <= runs[0].start_time < 99
    assert runs[0].end_time <= 99.9
    assert not runs[0].success


def test_trials_do_not_span_pause(pause_track):
    r = pause_track
    trials = segment_trials(
        r.position_bins,
        r.times,
        r.env,
        start_region="source",
        end_regions=["target"],
        max_duration=2000,
    )
    assert len(trials) == 1
    assert trials[0].start_time == 0
    assert trials[0].end_time <= 99.9
    assert not trials[0].success


def test_velocity_epochs_do_not_span_pause(pause_track):
    r = pause_track
    epochs = segment_by_velocity(r.positions, r.times, min_speed=5, min_duration=0.1)
    assert epochs
    assert all(run.end_time <= 99.9 or run.start_time >= 1100 for run in epochs)


def test_region_laps_do_not_pair_entries_across_pause(lap_track):
    r = lap_track
    laps = detect_laps(
        r.position_bins, r.times, r.env, method="region", start_region="start"
    )
    assert len(laps) == 4
    assert all(lap.end_time < 100 or lap.start_time >= 1100 for lap in laps)
    assert sum(lap.start_time < 100 for lap in laps) == 2
    assert sum(lap.start_time >= 1100 for lap in laps) == 2


def test_reference_laps_do_not_span_pause(lap_track):
    r = lap_track
    laps = detect_laps(
        r.position_bins,
        r.times,
        r.env,
        method="reference",
        reference_lap=r.position_bins[100:501],
    )
    assert laps
    assert all(lap.end_time < 100 or lap.start_time >= 1100 for lap in laps)
    assert any(lap.start_time < 100 for lap in laps)
    assert any(lap.start_time >= 1100 for lap in laps)


def test_auto_laps_keep_one_global_template(lap_track):
    r = lap_track
    template_size = r.times.size // 10
    actual = detect_laps(r.position_bins, r.times, r.env, method="auto")
    expected = []
    for recording in (slice(0, 1000), slice(1000, 2000)):
        start = max(recording.start, template_size)
        expected.extend(
            detect_laps(
                r.position_bins[start : recording.stop],
                r.times[start : recording.stop],
                r.env,
                method="reference",
                reference_lap=r.position_bins[:template_size],
            )
        )
    assert actual == expected
    assert any(lap.start_time >= 1100 for lap in actual)


def test_boundary_crossing_not_in_pause(pause_track):
    r = pause_track
    labels = np.where(r.env.bin_centers[:, 0] < 50, 0, 1)
    times, directions = detect_boundary_crossings(r.position_bins, labels, r.times)
    assert len(times) == len(directions)
    assert times == []


def test_goal_directed_runs_are_candidates_per_recording(pause_track):
    r = pause_track
    runs = detect_goal_directed_runs(
        r.position_bins, r.times, r.env, goal_region="target"
    )
    assert len(runs) == 1
    assert runs[0].start_time == 0
    assert runs[0].end_time <= 99.9


def test_running_labels_inherit_observed_runs(pause_track):
    r = pause_track
    labels = running_direction_labels(
        r.position_bins,
        r.times,
        r.env,
        start_region="source",
        end_regions="target",
        max_duration=2000,
    )
    assert labels.shape == r.times.shape
    assert np.all(labels == "other")


@pytest.mark.parametrize("kind", ["trials", "velocity"])
def test_segment_epochs_equal_slicing(continuous_pause_track, kind):
    r = continuous_pause_track
    selected = r.times <= 100
    if kind == "trials":
        function = segment_trials
        data = r.position_bins
        arguments = (r.env,)
        options = {
            "start_region": "source",
            "end_regions": ["target"],
            "max_duration": 2000,
        }
    else:
        function = segment_by_velocity
        data = r.positions
        arguments = ()
        options = {"min_speed": 5, "min_duration": 0.1}
    actual = function(data, r.times, *arguments, epochs=(0, 100), **options)
    expected = function(data[selected], r.times[selected], *arguments, **options)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert a.start_time == b.start_time
        assert a.end_time == b.end_time
        if hasattr(a, "success"):
            assert a.success == b.success
        if hasattr(a, "bins"):
            np.testing.assert_array_equal(a.bins, b.bins)


def test_pre_decision_window_stays_in_run(pause_track):
    r = pause_track
    positions, times = extract_pre_decision_window(
        r.positions, r.times, entry_time=1100.5, window_duration=1001
    )
    np.testing.assert_array_equal(times, r.times[1000:1005])
    np.testing.assert_array_equal(positions, r.positions[1000:1005])
    metrics = compute_pre_decision_metrics(
        r.positions, r.times, entry_time=1100.5, window_duration=1001
    )
    assert metrics.n_samples == 5
    assert metrics.window_duration <= 0.5
    assert metrics.mean_speed == 0


@pytest.mark.parametrize("entry", [500.0, 1200.0])
def test_entry_outside_observed_runs_has_empty_window(pause_track, entry):
    r = pause_track
    positions, times = extract_pre_decision_window(
        r.positions, r.times, entry_time=entry, window_duration=1001
    )
    assert positions.shape == (0, 2)
    assert times.shape == (0,)


@pytest.mark.parametrize(
    "function",
    [extract_pre_decision_window, compute_pre_decision_metrics, compute_vte_trial],
)
def test_window_callers_forward_epochs(pause_track, function):
    r = pause_track
    result = function(
        r.positions,
        r.times,
        entry_time=1100.5,
        window_duration=1001,
        epochs=(0, 100),
    )
    if function is extract_pre_decision_window:
        assert result[1].size == 0
    elif function is compute_pre_decision_metrics:
        assert result.n_samples == 0
    else:
        assert result.mean_speed == 0
        assert result.head_sweep_magnitude == 0


def test_vte_trial_uses_only_entry_run(pause_track):
    r = pause_track
    result = compute_vte_trial(
        r.positions, r.times, entry_time=1100.5, window_duration=1001
    )
    assert result.mean_speed == 0
    assert result.min_speed == 0
    assert result.head_sweep_magnitude == 0


def test_vte_session_keeps_trial_clamp_and_observed_window(pause_track):
    from neurospatial.behavior import Trial

    r = pause_track
    # Decision-region entry is 1100.5; the five preceding samples are stationary.
    positions = r.positions.copy()
    positions[1000:1005, 0] = 80
    trial = Trial(99.5, 1109.9, "source", "target", True)
    with pytest.warns(UserWarning, match="No variation"):
        result = compute_vte_session(
            r.env,
            positions,
            r.times,
            decision_region="target",
            trials=[trial],
            window_duration=1001,
        )
    assert len(result.trial_results) == 1
    assert result.trial_results[0].mean_speed == 0
    assert result.trial_results[0].window_start >= trial.start_time


def test_decision_analysis_forwards_windows(pause_track):
    r = pause_track
    result = compute_decision_analysis(
        r.env,
        r.positions,
        r.times,
        decision_region="target",
        goal_regions=["source", "target"],
        pre_window=1001,
        epochs=(0, 100),
    )
    assert result.pre_decision.n_samples == 0
    assert result.boundary.crossing_times == []


def test_max_gap_none_preserves_requested_window(pause_track):
    r = pause_track
    _, times = extract_pre_decision_window(
        r.positions, r.times, entry_time=1100.5, window_duration=1001, max_gap=None
    )
    assert times.size == 10
    assert times[0] == 99.5
