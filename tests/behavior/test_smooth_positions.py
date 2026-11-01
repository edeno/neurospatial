"""Smoothing tracked positions before measuring path length."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior import (
    compute_cumulative_distance,
    compute_path_efficiency,
    resample_by_distance,
    smooth_positions,
    traveled_path_length,
)


def _noisy_stop(hz: float = 30.0):
    """Run 0 -> 50 cm at 25 cm/s, stop 5 s, run on to 100 cm; 0.5 cm jitter."""
    rng = np.random.default_rng(0)
    dt = 1 / hz
    run_in, stop, run_out = (
        np.arange(0.0, 2.0, dt),
        np.arange(2.0, 7.0, dt),
        np.arange(7.0, 9.0, dt),
    )
    times = np.r_[run_in, stop, run_out]
    x = np.r_[run_in * 25.0, np.full(stop.size, 50.0), 50.0 + (run_out - 7.0) * 25.0]
    positions = np.c_[x, np.zeros_like(x)] + rng.normal(0, 0.5, (times.size, 2))
    return times, positions


@pytest.mark.parametrize("hz", [30.0, 125.0])
def test_smoothing_removes_the_jitter_raw_steps_count_as_running(hz):
    times, positions = _noisy_stop(hz)
    smoothed = smooth_positions(times, positions)
    assert traveled_path_length(times, positions) > 250.0  # jitter as running
    assert traveled_path_length(times, smoothed) == pytest.approx(100.0, abs=10.0)
    samples = resample_by_distance(times, smoothed, spacing=10.0)
    assert samples.times.size in (10, 11)


def test_constant_velocity_is_unchanged_also_at_the_ends():
    """The local line fit has no lag where the window runs off a run's end."""
    times = np.arange(0.0, 3.0, 1 / 30)
    positions = np.c_[25.0 * times, 5.0 - 10.0 * times]
    np.testing.assert_allclose(smooth_positions(times, positions), positions, atol=1e-9)


def test_no_smoothing_across_a_tracking_loss_or_break():
    times = np.r_[np.arange(0.0, 2.0, 0.1), np.arange(5.0, 7.0, 0.1)]
    positions = np.r_[np.zeros((20, 2)), np.full((20, 2), 100.0)]
    # A 3 s gap is a tracking loss at the default max_gap: the two sides stay apart.
    smoothed = smooth_positions(times, positions, smooth_window=1.0)
    np.testing.assert_allclose(smoothed, positions)
    # Bridged, the step is smoothed; a declared break keeps it.
    bridged = smooth_positions(times, positions, smooth_window=1.0, max_gap=None)
    assert not np.allclose(bridged, positions)
    breaks = np.zeros(times.size, dtype=bool)
    breaks[20] = True
    kept = smooth_positions(
        times, positions, smooth_window=1.0, max_gap=None, break_before=breaks
    )
    np.testing.assert_allclose(kept, positions)


def test_missing_rows_stay_missing_and_shapes_are_kept():
    times = np.arange(0.0, 2.0, 0.1)
    positions = np.c_[times * 10, np.zeros_like(times)]
    positions[5] = np.nan
    smoothed = smooth_positions(times, positions)
    assert np.isnan(smoothed[5]).all()
    assert np.isfinite(np.delete(smoothed, 5, 0)).all()
    line = smooth_positions(times, times * 10)
    assert line.shape == times.shape


def test_zero_window_returns_the_positions():
    times, positions = _noisy_stop()
    np.testing.assert_array_equal(
        smooth_positions(times, positions, smooth_window=0.0), positions
    )


def test_smoothed_positions_feed_every_path_analysis():
    """One smoothing step serves path length, resampling and efficiency alike."""
    times, positions = _noisy_stop()
    smoothed = smooth_positions(times, positions)
    env = Environment.from_samples(
        np.c_[np.linspace(-10, 110, 600), np.linspace(-6, 6, 600)], bin_size=2.0
    )
    goal = np.array([100.0, 0.0])
    length = traveled_path_length(times, smoothed)
    assert compute_cumulative_distance(times, smoothed).run_path_lengths.sum() == (
        pytest.approx(length)
    )
    efficiency = compute_path_efficiency(env, times, smoothed, goal, metric="euclidean")
    assert efficiency.traveled_length == pytest.approx(length)


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, True])
def test_refuses_a_smooth_window_that_is_not_seconds(value):
    times, positions = _noisy_stop()
    with pytest.raises(ValueError, match="smooth_window"):
        smooth_positions(times, positions, smooth_window=value)
