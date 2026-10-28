"""Detectors analyze observed intervals without inventing transitions in pauses."""

import warnings

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment, Regions
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
    # [0, 100) excludes the sample at 100 s, where the target is entered.
    selected = r.times < 100
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
    assert len(epochs)
    starts, stops = epochs.T
    assert np.all((stops <= 99.9) | (starts >= 1100))


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
    """The first circuit is the template for every recording."""
    r = lap_track
    actual = detect_laps(r.position_bins, r.times, r.env, method="auto")
    first = actual[0]
    in_first = (r.times >= first.start_time) & (r.times < first.end_time)
    expected = detect_laps(
        r.position_bins,
        r.times,
        r.env,
        method="reference",
        reference_lap=r.position_bins[in_first],
    )
    assert actual == expected
    assert first.start_time == r.times[0]
    # One 40 s lap; the boundary is the first sample in the start bin, which
    # the animal takes 1 s to cross.
    assert first.end_time - first.start_time == pytest.approx(40.0, abs=1.0)
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
    assert runs.shape == (1, 2)
    assert runs[0, 0] == 0
    assert runs[0, 1] <= 99.9


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
    # epochs=(0, 100) is half-open: no trial event at the 100 s sample, but
    # the interval ending there is inside the epoch, so a movement epoch can
    # run up to 100 s.
    selected = r.times < 100 if kind == "trials" else r.times <= 100
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
    if kind == "velocity":
        np.testing.assert_array_equal(actual, expected)
        return
    for a, b in zip(actual, expected, strict=True):
        assert a == b


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
    trial = Trial(99.5, 1109.9, "source", "target", "reached")
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


def _corridor_with_wells():
    from shapely.geometry import box

    xs = np.linspace(0.0, 100.0, 401)
    env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
    regions = Regions(env.regions.values())
    regions.add("start", polygon=box(-5.0, -5.0, 5.0, 5.0))  # bins 0-4 cm
    regions.add("goal", polygon=box(95.0, -5.0, 105.0, 5.0))  # bins 96-100 cm
    return env.with_regions(regions)


def test_no_event_at_an_epoch_stop_sample():
    """An arrival exactly at an epoch's stop lies outside [start, stop).

    Leave the start well at 1 s and arrive at the goal at 5 s; with
    epochs=(0, 5) the arrival is not observed, so there is no goal entry,
    the run and the trial are censored at 4 s, the last sample inside.
    """
    env = _corridor_with_wells()
    x = np.array([0.0, 0.0, 20.0, 50.0, 80.0, 98.0, 98.0])
    times = np.arange(x.size, dtype=float)
    bins = env.bin_at(np.c_[x, np.zeros_like(x)])
    common = {"max_gap": None, "epochs": (0.0, 5.0)}

    assert detect_region_crossings(bins, times, env, region_name="goal", **common) == []
    (run,) = detect_runs_between_regions(
        bins, times, env, source="start", target="goal", **common
    )
    assert (run.start_time, run.end_time, run.outcome) == (2.0, 4.0, "censored")
    (trial,) = segment_trials(
        bins, times, env, start_region="start", end_regions=["goal"], **common
    )
    assert (trial.start_time, trial.end_time, trial.outcome) == (1.0, 4.0, "censored")

    # Without the epoch the same arrival is observed at 5 s.
    (entry,) = detect_region_crossings(
        bins, times, env, region_name="goal", direction="entry", max_gap=None
    )
    assert entry.time == 5.0


def test_region_lap_does_not_end_at_an_epoch_stop_sample():
    """A start-region entry at an epoch's stop does not close a lap."""
    theta = np.linspace(0.0, 4 * np.pi, 81)  # two laps, 40 samples each
    positions = np.c_[50 + 30 * np.cos(theta), 50 + 30 * np.sin(theta)]
    env = Environment.from_samples(positions, bin_size=3.0)
    regions = Regions(env.regions.values())
    regions.add("start", polygon=Point(80.0, 50.0).buffer(4.0))
    env = env.with_regions(regions)
    bins = env.bin_at(positions)
    times = np.arange(theta.size, dtype=float)
    laps = detect_laps(bins, times, env, method="region", start_region="start")
    entries = sorted({lap.start_time for lap in laps} | {lap.end_time for lap in laps})
    second_entry = entries[1]

    clipped = detect_laps(
        bins,
        times,
        env,
        method="region",
        start_region="start",
        epochs=(0.0, second_entry),
    )

    assert clipped == []
