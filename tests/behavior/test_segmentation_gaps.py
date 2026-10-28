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


def test_canonical_crossings_do_not_warn_for_multiple_runs(pause_track):
    r = pause_track
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detect_region_crossings(r.position_bins, r.times, r.env, region_name="target")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations


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
    # The trial starts when the animal leaves the source, before the pause.
    in_source = r.env.region_mask("source")[r.position_bins]
    departure = r.times[np.flatnonzero(in_source & (r.times < 100))[-1]]
    assert trials[0].start_time == departure
    assert trials[0].end_time <= 99.9
    assert not trials[0].success


def test_velocity_epochs_do_not_span_pause(pause_track):
    r = pause_track
    epochs = segment_by_velocity(r.times, r.positions, min_speed=5, min_duration=0.1)
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
    assert times.shape == (0,)
    assert directions.shape == (0, 2)


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
    assert all(label is None for label in labels)


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
    pair = (r.times, data) if kind == "velocity" else (data, r.times)
    sliced_pair = tuple(array[selected] for array in pair)
    actual = function(*pair, *arguments, epochs=(0, 100), **options)
    expected = function(*sliced_pair, *arguments, **options)
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
    times, positions = extract_pre_decision_window(
        r.times, r.positions, entry_time=1100.5, window_duration=1001
    )
    np.testing.assert_array_equal(times, r.times[1000:1005])
    np.testing.assert_array_equal(positions, r.positions[1000:1005])
    metrics = compute_pre_decision_metrics(
        r.times,
        r.positions,
        entry_time=1100.5,
        window_duration=1001,
        min_speed=5.0,
    )
    assert metrics.n_samples == 5
    assert metrics.window_duration <= 0.5
    assert metrics.mean_speed == 0


def test_entry_in_tracking_gap_has_empty_window(pause_track):
    r = pause_track
    times, positions = extract_pre_decision_window(
        r.times, r.positions, entry_time=500.0, window_duration=1001
    )
    assert positions.shape == (0, 2)
    assert times.shape == (0,)


def test_entry_after_the_recording_raises(pause_track):
    r = pause_track
    with pytest.raises(ValueError, match="outside the recording"):
        extract_pre_decision_window(
            r.times, r.positions, entry_time=1200.0, window_duration=1001
        )


@pytest.mark.parametrize(
    "function",
    [extract_pre_decision_window, compute_pre_decision_metrics, compute_vte_trial],
)
def test_window_callers_forward_epochs(pause_track, function):
    r = pause_track
    speed_gate = {} if function is extract_pre_decision_window else {"min_speed": 5.0}
    result = function(
        r.times,
        r.positions,
        entry_time=1100.5,
        window_duration=1001,
        epochs=(0, 100),
        **speed_gate,
    )
    if function is extract_pre_decision_window:
        assert result[0].size == 0
    elif function is compute_pre_decision_metrics:
        assert result.n_samples == 0
    else:
        # Entry lies outside the epochs: the window is empty, nothing measured.
        assert np.isnan(result.mean_speed)
        assert np.isnan(result.head_sweep_magnitude)


def test_vte_trial_uses_only_entry_run(pause_track):
    r = pause_track
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vte_trial(
            r.times,
            r.positions,
            entry_time=1100.5,
            window_duration=1001,
            min_speed=5.0,
        )
    # The entry run is stationary: an observed zero speed and no heading.
    assert result.mean_speed == 0
    assert result.min_observed_speed == 0
    assert np.isnan(result.head_sweep_magnitude)


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
            r.times,
            positions,
            decision_region="target",
            trials=[trial],
            window_duration=1001,
        )
    assert len(result.trial_results) == 1
    (trial_result,) = result.trial_results
    assert trial_result.mean_speed == 0
    assert trial_result.window_start >= trial.start_time
    # A stationary window has no heading, so the trial is unclassified.
    assert np.isnan(trial_result.head_sweep_magnitude)
    assert trial_result.is_vte is None
    assert np.isnan(result.vte_fraction)


def test_decision_analysis_forwards_windows(pause_track):
    """The target is entered only after 1100 s, outside epochs (0, 100) s."""
    r = pause_track
    with pytest.raises(ValueError, match="never enters region 'target' inside"):
        compute_decision_analysis(
            r.env,
            r.times,
            r.positions,
            decision_region="target",
            goal_regions=["source", "target"],
            pre_window=1001,
            epochs=(0, 100),
        )


def test_max_gap_none_preserves_requested_window(pause_track):
    r = pause_track
    times, _ = extract_pre_decision_window(
        r.times, r.positions, entry_time=1100.5, window_duration=1001, max_gap=None
    )
    assert times.size == 10
    assert times[0] == 99.5


def test_auto_laps_with_missing_template_rows_equal_deleted_rows():
    """NaN rows in the first 10% change neither the template nor the search.

    The automatic template is the first 10% of the located samples, and the
    search begins after it, exactly as when the missing rows are deleted.
    """
    from neurospatial import Environment

    theta = np.linspace(0, 20 * np.pi, 1801)
    positions = np.column_stack([50 + 30 * np.cos(theta), 50 + 30 * np.sin(theta)])
    times = np.linspace(0, 60, 1801)
    env = Environment.from_samples(positions, bin_size=3.0)
    missing = np.zeros(times.size, dtype=bool)
    missing[50:70] = True  # 0.67 s of lost tracking, bridged by max_gap=1
    missing[150:170] = True
    positions[missing] = np.nan
    position_bins = env.bin_at(positions)
    kept = ~missing
    expected = detect_laps(position_bins[kept], times[kept], env, method="auto")
    assert expected
    assert detect_laps(position_bins, times, env, method="auto") == expected
