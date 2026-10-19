"""Distance resampling places samples along observed arclength, never across a break."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.behavior import (
    ResampledPathResult,
    compute_cumulative_distance,
    resample_by_distance,
)


def _xline(x, times):
    """Positions on the x axis, shape (n_samples, 2)."""
    return np.c_[np.asarray(x, dtype=float), np.zeros(len(times))]


def test_plateau_departs_at_last_stationary_sample():
    times = np.array([0.0, 1.0, 10.0, 11.0])
    result = resample_by_distance(
        times, _xline([0, 1, 1, 2], times), spacing=0.5, max_gap=None
    )
    assert isinstance(result, ResampledPathResult)
    np.testing.assert_allclose(result.run_distance, [0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(result.cumulative_distance, [0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(result.times, [0, 0.5, 10, 10.5, 11])
    np.testing.assert_allclose(result.positions[:, 0], [0, 0.5, 1, 1.5, 2])
    np.testing.assert_array_equal(result.positions[:, 1], 0.0)
    assert result.source_left.tolist() == [0, 0, 2, 2, 3]
    assert result.source_right.tolist() == [0, 1, 2, 3, 3]
    np.testing.assert_allclose(result.weights, [0, 0.5, 0, 0.5, 0])
    np.testing.assert_array_equal(result.run_time_bounds, [[0.0, 11.0]])
    np.testing.assert_array_equal(result.run_sample_bounds, [[0, 3]])
    np.testing.assert_allclose(result.run_path_lengths, [2.0])
    assert result.run_ids.tolist() == [0] * 5
    assert not result.off_grid_endpoint.any()
    assert result.n_source_samples == 4
    assert result.spacing == 0.5
    assert result.include_endpoint is False
    assert result.max_gap is None
    assert result.epochs is None


def test_output_dtypes():
    times = np.array([0.0, 1.0, 10.0, 11.0])
    result = resample_by_distance(
        times, _xline([0, 1, 1, 2], times), spacing=0.5, max_gap=None
    )
    for name in ("source_left", "source_right", "run_ids", "run_sample_bounds"):
        assert getattr(result, name).dtype == np.intp, name
    for name in ("times", "positions", "weights", "run_distance", "run_time_bounds"):
        assert getattr(result, name).dtype == np.float64, name
    assert result.off_grid_endpoint.dtype == np.bool_


def test_floating_point_cumsum_snaps_to_exact_sample():
    times = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    result = resample_by_distance(
        times, _xline([0, 0.1, 0.3, 0.3, 0.6], times), spacing=0.3, max_gap=None
    )
    at = int(np.flatnonzero(np.isclose(result.run_distance, 0.3))[0])
    assert result.times[at] == 10.0
    assert result.source_left[at] == result.source_right[at] == 3
    assert result.weights[at] == 0.0


def test_endpoint_rule():
    times = np.arange(4) * 0.1
    positions = _xline([0, 1, 2, 2.3], times)
    default = resample_by_distance(times, positions, spacing=1.0)
    np.testing.assert_allclose(default.run_distance, [0, 1, 2])
    assert not default.off_grid_endpoint.any()

    with_end = resample_by_distance(
        times, positions, spacing=1.0, include_endpoint=True
    )
    np.testing.assert_allclose(with_end.run_distance, [0, 1, 2, 2.3])
    assert with_end.off_grid_endpoint.tolist() == [False, False, False, True]
    assert with_end.source_left[-1] == with_end.source_right[-1] == 3
    assert with_end.times[-1] == times[-1]
    assert with_end.summary()["n_off_grid_endpoints"] == 1

    on_grid = resample_by_distance(
        times[:3], positions[:3], spacing=1.0, include_endpoint=True
    )
    np.testing.assert_allclose(on_grid.run_distance, [0, 1, 2])
    assert not on_grid.off_grid_endpoint.any()


def _recording(times, phase):
    return np.c_[10 * np.cos(times + phase), 10 * np.sin(2 * times + phase)]


def test_declared_break_matches_separate_calls():
    t1 = np.arange(50) * 0.1
    t2 = 5.0 + np.arange(40) * 0.1
    p1 = _recording(t1, 0.0)
    p2 = 100 + _recording(t2, 1.0)
    first = resample_by_distance(t1, p1, spacing=0.7)
    second = resample_by_distance(t2, p2, spacing=0.7)
    whole = resample_by_distance(
        np.r_[t1, t2],
        np.r_[p1, p2],
        spacing=0.7,
        break_before=np.r_[np.zeros(50, bool), True, np.zeros(39, bool)],
    )
    n1 = first.times.size
    np.testing.assert_allclose(
        whole.times, np.r_[first.times, second.times], rtol=1e-12
    )
    np.testing.assert_allclose(
        whole.positions, np.r_[first.positions, second.positions], rtol=1e-12
    )
    np.testing.assert_allclose(
        whole.run_distance, np.r_[first.run_distance, second.run_distance], rtol=1e-12
    )
    assert whole.run_ids.tolist() == [0] * n1 + [1] * second.times.size
    np.testing.assert_array_equal(whole.source_left[n1:], second.source_left + 50)
    np.testing.assert_array_equal(whole.source_right[n1:], second.source_right + 50)
    np.testing.assert_allclose(
        whole.cumulative_distance[n1:],
        first.run_path_lengths[0] + second.cumulative_distance,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(whole.run_sample_bounds, [[0, 49], [50, 89]])


def test_nan_row_is_never_bridged():
    times = np.arange(20) * 0.1
    positions = _xline(np.arange(20.0), times)
    positions[10] = np.nan
    result = resample_by_distance(times, positions, spacing=0.3)
    steps = compute_cumulative_distance(times, positions).step_lengths

    inexact = result.source_left != result.source_right
    assert inexact.any()
    np.testing.assert_array_equal(
        result.source_right[inexact], result.source_left[inexact] + 1
    )
    assert np.isfinite(steps[result.source_left[inexact]]).all()
    assert not ((result.times > times[9]) & (result.times < times[11])).any()
    assert 10 not in result.source_left and 10 not in result.source_right
    np.testing.assert_array_equal(result.run_sample_bounds, [[0, 9], [11, 19]])


def test_same_path_at_different_sampling_gives_same_samples():
    dense = np.linspace(0, 2, 201)
    rng = np.random.default_rng(0)
    sparse = np.sort(np.r_[0.0, rng.uniform(0, 2, 40), 2.0])
    results = [
        resample_by_distance(t, _xline(3 * t, t), spacing=0.25, max_gap=None)
        for t in (dense, sparse)
    ]
    np.testing.assert_allclose(results[0].positions, results[1].positions, atol=1e-12)
    for result in results:
        assert result.times.size == 25
        np.testing.assert_allclose(result.times, result.positions[:, 0] / 3, atol=1e-12)


def test_samples_are_reconstructed_from_source_brackets(continuous_recording):
    times = continuous_recording.times
    positions = continuous_recording.positions
    result = resample_by_distance(times, positions, spacing=5.0)
    left, right, w = result.source_left, result.source_right, result.weights
    assert result.times.size > 100
    assert (w >= 0).all() and (w < 1).all()
    np.testing.assert_array_equal(w[left == right], 0.0)
    np.testing.assert_array_equal(right[left != right], left[left != right] + 1)
    np.testing.assert_allclose(
        result.positions,
        (1 - w)[:, None] * positions[left] + w[:, None] * positions[right],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.times, (1 - w) * times[left] + w * times[right], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.diff(result.run_distance), 5.0, rtol=1e-9
    )  # one run, no endpoint


def test_wholly_stationary_run_gives_one_sample():
    times = np.arange(5) * 0.1
    result = resample_by_distance(times, np.ones((5, 2)), spacing=1.0)
    np.testing.assert_array_equal(result.times, [0.0])
    assert result.source_left.tolist() == result.source_right.tolist() == [0]
    np.testing.assert_allclose(result.run_time_bounds, [[0.0, 0.4]])
    np.testing.assert_array_equal(result.run_path_lengths, [0.0])
    summary = result.summary()
    assert summary["n_stationary_runs"] == 1
    assert summary["total_distance"] == 0.0


def test_isolated_sample_produces_no_output():
    times = np.arange(6) * 0.1
    positions = np.array(
        [[0, 0], [np.nan, np.nan], [5, 5], [np.nan, np.nan], [1, 0], [2, 0]], float
    )
    result = resample_by_distance(times, positions, spacing=0.5)
    assert 2 not in result.source_left and 2 not in result.source_right
    np.testing.assert_array_equal(result.run_sample_bounds, [[4, 5]])
    np.testing.assert_allclose(result.positions[:, 0], [1, 1.5, 2])


@pytest.mark.parametrize(
    ("times", "positions"),
    [
        (np.array([0.0]), np.array([[1.0, 2.0]])),
        (np.arange(4) * 0.1, np.full((4, 2), np.nan)),
    ],
    ids=["single-sample", "all-missing"],
)
def test_no_runs_give_empty_arrays(times, positions):
    result = resample_by_distance(times, positions, spacing=1.0)
    for name in (
        "times",
        "run_distance",
        "cumulative_distance",
        "run_ids",
        "source_left",
        "source_right",
        "weights",
        "off_grid_endpoint",
        "run_path_lengths",
    ):
        assert getattr(result, name).shape == (0,), name
    assert result.positions.shape == (0, 2)
    assert result.run_sample_bounds.shape == (0, 2)
    assert result.run_time_bounds.shape == (0, 2)
    assert result.source_left.dtype == np.intp
    assert result.off_grid_endpoint.dtype == np.bool_
    summary = result.summary()
    assert summary["n_samples"] == 0 and summary["n_runs"] == 0
    assert np.isnan(summary["total_distance"])


def test_scaling_positions_and_spacing_together(continuous_recording):
    times = continuous_recording.times
    positions = continuous_recording.positions
    base = resample_by_distance(times, positions, spacing=3.0)
    scaled = resample_by_distance(times, 2 * positions, spacing=6.0)
    np.testing.assert_allclose(scaled.times, base.times, rtol=1e-12)
    np.testing.assert_allclose(scaled.positions, 2 * base.positions, rtol=1e-12)


def test_one_dimensional_positions():
    times = np.arange(3) * 0.1
    result = resample_by_distance(times, np.array([0.0, 1.0, 2.0]), spacing=0.5)
    assert result.positions.shape == (5, 1)
    np.testing.assert_allclose(result.positions[:, 0], [0, 0.5, 1, 1.5, 2])


@pytest.mark.parametrize("spacing", [0, -1.0, np.nan, np.inf])
def test_invalid_spacing_raises(spacing):
    times = np.arange(3) * 0.1
    with pytest.raises(ValueError, match="spacing") as caught:
        resample_by_distance(times, _xline([0, 1, 2], times), spacing=spacing)
    assert "Fix:" in str(caught.value)


def test_repeated_timestamps_raise():
    times = np.array([0.0, 0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_by_distance(times, _xline([0, 1, 2, 3], times), spacing=1.0)


def test_output_times_strictly_increasing_across_runs(two_epoch_recording):
    times = two_epoch_recording.times.copy()
    positions = two_epoch_recording.positions.copy()
    positions[[100, 2000, 2001, 7000]] = np.nan
    result = resample_by_distance(times, positions, spacing=2.0, include_endpoint=True)
    assert result.run_path_lengths.size >= 4
    assert (np.diff(result.times) > 0).all()
    assert (np.diff(result.cumulative_distance) >= 0).all()
    assert np.unique(result.run_ids).tolist() == list(
        range(result.run_path_lengths.size)
    )


def test_epochs_are_stored_normalized():
    times = np.arange(30) * 0.1
    result = resample_by_distance(
        times, _xline(np.arange(30.0), times), spacing=1.0, epochs=(0.0, 1.05)
    )
    np.testing.assert_array_equal(result.epochs, [[0.0, 1.05]])
    np.testing.assert_allclose(result.run_time_bounds, [[0.0, 1.0]])
    np.testing.assert_allclose(result.run_distance, np.arange(11.0))
