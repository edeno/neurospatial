"""Observed cumulative distance never crosses a recording break."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial.behavior import compute_cumulative_distance, compute_step_lengths

T4 = np.array([0.0, 0.1, 0.2, 0.3])
TWO_RECORDINGS = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])


def test_declared_break_between_recordings():
    result = compute_cumulative_distance(
        T4, TWO_RECORDINGS, break_before=np.array([False, False, True, False])
    )
    np.testing.assert_allclose(result.step_lengths, [1, np.nan, 1])
    np.testing.assert_allclose(result.run_distance, [0, 1, 0, 1])
    np.testing.assert_allclose(result.cumulative_distance, [0, 1, 1, 2])
    np.testing.assert_allclose(result.run_path_lengths, [1, 1])
    assert result.total_distance == 2.0


def test_break_from_time_gap():
    times = np.array([0.0, 0.1, 100.0, 100.1])
    result = compute_cumulative_distance(times, TWO_RECORDINGS)
    assert result.total_distance == 2.0
    assert result.run_ids.tolist() == [0, 0, 1, 1]


def test_break_invariance_on_two_epoch_recording(two_epoch_recording):
    times, positions = two_epoch_recording.times, two_epoch_recording.positions
    whole = compute_cumulative_distance(times, positions)
    first = compute_cumulative_distance(times[:5000], positions[:5000])
    second = compute_cumulative_distance(times[5000:], positions[5000:])
    np.testing.assert_allclose(
        whole.total_distance,
        first.total_distance + second.total_distance,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        whole.run_path_lengths,
        [first.total_distance, second.total_distance],
        rtol=1e-12,
    )


def test_no_declared_break_counts_jump_as_motion():
    assert compute_cumulative_distance(T4, TWO_RECORDINGS).total_distance == 51.0


def test_nan_row_is_bridged_like_a_deleted_row():
    times = np.arange(5) * 0.1
    positions = np.array([[0, 0], [1, 0], [np.nan, np.nan], [2, 0], [3, 0]], float)
    result = compute_cumulative_distance(times, positions)
    kept = [0, 1, 3, 4]
    deleted = compute_cumulative_distance(times[kept], positions[kept])
    np.testing.assert_allclose(result.step_lengths, [1, np.nan, 1, 1])
    assert result.run_ids.tolist() == [0, 0, -1, 0, 0]
    np.testing.assert_allclose(result.cumulative_distance, [0, 1, np.nan, 2, 3])
    np.testing.assert_allclose(
        result.cumulative_distance[kept], deleted.cumulative_distance
    )
    assert result.total_distance == deleted.total_distance == 3.0


def test_nan_row_gap_longer_than_max_gap_breaks_path_without_poisoning():
    positions = np.array([[0, 0], [1, 0], [np.nan, np.nan], [2, 0], [3, 0]], float)
    result = compute_cumulative_distance(np.arange(5) * 0.1, positions, max_gap=0.15)
    np.testing.assert_allclose(result.step_lengths, [1, np.nan, np.nan, 1])
    assert result.run_ids.tolist() == [0, 0, -1, 1, 1]
    np.testing.assert_allclose(result.cumulative_distance, [0, 1, np.nan, 1, 2])
    assert result.total_distance == 2.0


def test_stationary_run_is_zero_not_nan():
    result = compute_cumulative_distance(T4, np.ones((4, 2)))
    assert result.total_distance == 0.0
    np.testing.assert_allclose(result.run_time_bounds, [[0, 0.3]])
    assert result.summary()["observed_duration"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("times", "positions"),
    [
        (np.array([0.0]), np.zeros((1, 2))),
        (np.arange(3) * 0.1, np.full((3, 2), np.nan)),
    ],
)
def test_no_usable_interval_is_nan(times, positions):
    result = compute_cumulative_distance(times, positions)
    assert np.isnan(result.total_distance)
    assert (result.run_ids == -1).all()
    assert np.isnan(result.cumulative_distance).all()
    assert np.isnan(result.run_distance).all()


def test_physical_scaling():
    break_before = np.array([False, False, True, False])
    base = compute_cumulative_distance(T4, TWO_RECORDINGS, break_before=break_before)
    scaled = compute_cumulative_distance(
        T4, 2.5 * TWO_RECORDINGS, break_before=break_before
    )
    np.testing.assert_allclose(
        scaled.cumulative_distance, 2.5 * base.cumulative_distance
    )
    np.testing.assert_allclose(scaled.run_distance, 2.5 * base.run_distance)
    assert scaled.total_distance == pytest.approx(2.5 * base.total_distance)


def test_three_dimensional_positions_use_all_columns():
    result = compute_cumulative_distance(
        np.array([0.0, 0.1]), np.array([[0.0, 0, 0], [1, 2, 2]])
    )
    assert result.total_distance == 3.0


def test_one_dimensional_positions():
    result = compute_cumulative_distance(np.arange(3) * 0.1, np.array([0.0, 1, 3]))
    assert result.total_distance == 3.0


def test_gap_free_equals_step_length_sum(continuous_recording):
    times, positions = continuous_recording.times, continuous_recording.positions
    result = compute_cumulative_distance(times, positions)
    np.testing.assert_allclose(
        result.total_distance, np.sum(compute_step_lengths(positions)), rtol=1e-12
    )
    assert result.run_ids.max() == 0


def test_epochs_gate_and_normalized_storage():
    times = np.arange(6) * 0.1
    positions = np.c_[np.arange(6.0), np.zeros(6)]
    result = compute_cumulative_distance(times, positions, epochs=[(0.0, 0.25)])
    np.testing.assert_allclose(result.epochs, [[0.0, 0.25]])
    assert result.total_distance == 2.0
    assert result.max_gap == DEFAULT_MAX_GAP


def test_arrays_read_only_and_summary_scalars():
    result = compute_cumulative_distance(
        T4, TWO_RECORDINGS, break_before=np.array([False, False, True, False])
    )
    with pytest.raises(ValueError, match="read-only"):
        result.cumulative_distance[0] = 1.0
    summary = result.summary()
    assert set(summary) == {
        "n_samples",
        "n_runs",
        "n_isolated_samples",
        "total_distance",
        "observed_duration",
    }
    assert summary["n_samples"] == 4
    assert summary["n_runs"] == 2
    assert summary["n_isolated_samples"] == 0
    assert summary["observed_duration"] == pytest.approx(0.2)
    assert all(type(v) in (int, float) for v in summary.values())


def test_consecutive_non_finite_rows_do_not_warn():
    positions = np.array(
        [[0.0, 0.0], [np.inf, 0.0], [np.inf, 0.0], [1.0, 0.0], [2.0, 0.0]]
    )
    times = np.arange(5) * 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_cumulative_distance(times, positions)

    # The two missing rows are bridged: 1 cm across the 0.3 s gap, then 1 cm.
    assert result.total_distance == 2.0
