"""A missing sample is a deleted row, in every analysis.

Video and pose tracking lose samples two ways: a dropped frame removes the
row, and a low-confidence keypoint leaves the row with NaN coordinates. Both
are the same missing observation, so every analysis must give the same result
for either. A gap between consecutive valid samples no longer than
``max_gap`` is bridged; a longer one is a tracking loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._sampling import observed_rows, sample_at_times
from neurospatial.behavior import compute_speed
from neurospatial.behavior.segmentation import segment_by_velocity
from neurospatial.encoding import compute_directional_rate, compute_spatial_rate
from neurospatial.simulation import generate_poisson_spikes

FS = 30.0


@pytest.fixture(scope="module")
def tracking():
    """60 s at 30 Hz: running laps with pauses, DeepLabCut-style NaN bursts.

    Returns the clean recording and the indices of rows lost to isolated
    dropped frames and to low-confidence bursts of 1-6 frames.
    """
    rng = np.random.default_rng(20)
    times = np.arange(0, 60, 1 / FS)
    phase = 2 * np.pi * times / 12.0
    x = 50 + 40 * np.sin(phase) * (np.abs(np.sin(phase)) > 0.2)
    positions = np.column_stack([x, 50 + 10 * np.cos(phase)])
    positions += rng.normal(0, 0.3, positions.shape)
    lost = np.zeros(times.size, dtype=bool)
    lost[rng.random(times.size) < 0.03] = True
    for start in rng.choice(np.arange(5, times.size - 10), 25, replace=False):
        lost[start : start + rng.integers(1, 7)] = True
    lost[[0, -1]] = False
    headings = np.unwrap(
        np.arctan2(np.gradient(positions[:, 1]), 1e-9 + np.gradient(x))
    )
    spikes = generate_poisson_spikes(
        2 + 10 * np.exp(-((x - 70) ** 2) / 200), times, seed=21
    )
    return times, positions, np.angle(np.exp(1j * headings)), spikes, lost


@pytest.fixture(scope="module")
def env():
    grid = np.stack(np.meshgrid(np.arange(0, 101.0), np.arange(30, 71.0)), -1)
    return Environment.from_samples(grid.reshape(-1, 2), bin_size=4.0, units="cm")


def _with_nan(values, lost):
    values = np.array(values, dtype=float, copy=True)
    values[lost] = np.nan
    return values


@pytest.mark.parametrize("time_allocation", ["start", "linear"])
def test_occupancy_with_nan_rows_equals_deleted_rows(env, tracking, time_allocation):
    times, positions, _, _, lost = tracking
    kept = ~lost
    with_nan = env.occupancy(
        times, _with_nan(positions, lost), time_allocation=time_allocation
    )
    deleted = env.occupancy(
        times[kept], positions[kept], time_allocation=time_allocation
    )
    np.testing.assert_allclose(with_nan, deleted, rtol=0, atol=1e-12)
    # Every gap is shorter than max_gap, so no tracked time is lost.
    np.testing.assert_allclose(with_nan.sum(), times[-1] - times[0], rtol=1e-12)


def test_spatial_rate_with_nan_rows_equals_deleted_rows(env, tracking):
    times, positions, _, spikes, lost = tracking
    kept = ~lost
    with_nan = compute_spatial_rate(
        env, spikes, times, _with_nan(positions, lost), method="binned"
    )
    deleted = compute_spatial_rate(
        env, spikes, times[kept], positions[kept], method="binned"
    )
    np.testing.assert_allclose(with_nan.occupancy, deleted.occupancy, atol=1e-12)
    np.testing.assert_allclose(
        with_nan.firing_rate, deleted.firing_rate, rtol=1e-12, atol=0
    )
    assert (
        with_nan.analysis_settings["observed_intervals"]
        == deleted.analysis_settings["observed_intervals"]
    )


def test_directional_rate_with_nan_headings_equals_deleted_rows(tracking):
    times, _, headings, spikes, lost = tracking
    kept = ~lost
    with_nan = compute_directional_rate(spikes, times, _with_nan(headings, lost))
    deleted = compute_directional_rate(spikes, times[kept], headings[kept])
    np.testing.assert_allclose(with_nan.occupancy, deleted.occupancy, atol=1e-12)
    np.testing.assert_allclose(with_nan.firing_rate, deleted.firing_rate, rtol=1e-12)


def test_speed_and_running_bouts_with_nan_rows_equal_deleted_rows(tracking):
    times, positions, _, _, lost = tracking
    kept = ~lost
    with_nan = _with_nan(positions, lost)
    speed = compute_speed(times, with_nan)
    np.testing.assert_allclose(speed[kept], compute_speed(times[kept], positions[kept]))
    assert np.isnan(speed[lost]).all()
    bouts = segment_by_velocity(times, with_nan, min_speed=10.0)
    expected = segment_by_velocity(times[kept], positions[kept], min_speed=10.0)
    assert [(b.start_time, b.end_time) for b in bouts] == [
        (b.start_time, b.end_time) for b in expected
    ]
    assert len(bouts) > 4


def test_speed_across_a_bridged_gap_uses_the_elapsed_time():
    times = np.arange(5) / 10
    positions = np.column_stack([10.0 * np.arange(5), np.zeros(5)])  # 100 cm/s
    positions[2] = np.nan
    speed = compute_speed(times, positions, smooth_window=0.0)
    np.testing.assert_allclose(speed[[0, 1, 3, 4]], 100.0)
    assert np.isnan(speed[2])


def test_gap_longer_than_max_gap_is_a_tracking_loss(env, tracking):
    times, positions, _, _, lost = tracking
    with_nan = _with_nan(positions, lost)
    occupancy = env.occupancy(times, with_nan, max_gap=1.5 / FS)
    # Every gap that lost a frame is excluded at this max_gap.
    valid = np.flatnonzero(~lost)
    gaps = np.diff(times[valid])
    np.testing.assert_allclose(occupancy.sum(), gaps[gaps <= 1.5 / FS].sum())


def test_animation_samples_a_bridged_gap_between_its_valid_rows(tracking):
    times, positions, _, _, lost = tracking
    kept = ~lost
    rows, valid, intervals = observed_rows(
        times, _with_nan(positions, lost), max_gap=1.0, interp="linear"
    )
    query = np.linspace(times[0], times[-1], 997)
    sampled = sample_at_times(
        times,
        rows,
        query,
        interp="linear",
        sample_valid=valid,
        interval_valid=intervals,
    )
    expected = np.column_stack(
        [np.interp(query, times[kept], positions[kept, d]) for d in range(2)]
    )
    np.testing.assert_allclose(sampled, expected, rtol=0, atol=1e-9)
