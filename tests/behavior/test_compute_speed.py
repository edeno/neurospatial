"""Speed comes from Gaussian-smoothed velocity, so tracking jitter is not running.

Differencing raw positions turns a few millimetres of camera jitter into tens
of centimetres per second, more at higher frame rates. Every speed gate in the
package uses one definition: the velocity vector smoothed with a Gaussian of
``smooth_window`` seconds, then its magnitude.
"""

import numpy as np
import pytest
import shapely

from neurospatial import Environment, compute_spatial_rates
from neurospatial.behavior import compute_speed, segment_by_velocity

JITTER_CM = 0.5


def _still(fs, seconds=60.0, seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, seconds, 1.0 / fs)
    positions = 50.0 + rng.normal(0.0, JITTER_CM, (len(times), 2))
    return times, positions


def _running(fs, speed=20.0, seconds=60.0, seed=1):
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, seconds, 1.0 / fs)
    positions = np.column_stack([speed * times, np.zeros_like(times)])
    return times, positions + rng.normal(0.0, JITTER_CM, positions.shape)


@pytest.mark.parametrize("fs", [30.0, 500.0])
def test_tracking_jitter_is_not_running(fs):
    times, positions = _still(fs)
    speed = compute_speed(times, positions)
    assert np.nanmedian(speed) < 2.0
    assert np.mean(speed > 4.0) < 0.02


@pytest.mark.parametrize("fs", [30.0, 500.0])
def test_steady_running_speed_is_preserved(fs):
    times, positions = _running(fs)
    speed = compute_speed(times, positions)
    inner = slice(int(fs), -int(fs))  # away from the recording's ends
    np.testing.assert_allclose(np.median(speed[inner]), 20.0, atol=0.5)


def test_without_smoothing_speed_is_the_central_difference():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    speed = compute_speed(times, positions, smooth_window=0.0, max_gap=None)
    np.testing.assert_allclose(speed, np.abs(np.gradient(positions[:, 0], times)))


def test_speed_does_not_smooth_across_a_tracking_gap():
    """Each observed run is smoothed separately; a lone sample has no speed."""
    times = np.concatenate([np.arange(0, 2, 0.1), [5.0], 10 + np.arange(0, 2, 0.1)])
    positions = np.column_stack([np.where(times < 5, 0.0, 100.0), np.zeros_like(times)])
    speed = compute_speed(times, positions, max_gap=0.5)
    assert np.isnan(speed[times == 5.0]).all()
    np.testing.assert_allclose(speed[times != 5.0], 0.0, atol=1e-12)


def test_rate_maps_drop_the_jittering_rest_from_speed_filtered_occupancy():
    env = Environment.from_polygon(
        shapely.box(0, 0, 100, 100), bin_size=4.0, units="cm"
    )
    rest_t, rest_p = _still(30.0, seconds=30.0)
    run_t, run_p = _running(30.0, speed=20.0, seconds=4.0)
    times = np.concatenate([rest_t, 30.0 + run_t])
    positions = np.concatenate([rest_p, run_p + np.array([10.0, 50.0])])
    result = compute_spatial_rates(
        env, [np.array([1.0, 31.0])], times, positions, min_speed=4.0, bandwidth=5.0
    )
    assert result.occupancy.sum() == pytest.approx(4.0, abs=0.5)


def test_rate_maps_and_compute_speed_share_one_definition():
    env = Environment.from_polygon(
        shapely.box(0, 0, 100, 100), bin_size=4.0, units="cm"
    )
    times, positions = _still(30.0, seconds=20.0)
    positions = positions + np.column_stack(
        [np.linspace(0, 30, len(times)), np.zeros(len(times))]
    )
    spikes = [np.linspace(0.5, 19.5, 40)]
    implicit = compute_spatial_rates(env, spikes, times, positions, min_speed=1.0)
    explicit = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        min_speed=1.0,
        speed=compute_speed(times, positions),
    )
    np.testing.assert_array_equal(implicit.occupancy, explicit.occupancy)


def test_velocity_segments_ignore_jitter_at_rest():
    times, positions = _still(30.0)
    assert segment_by_velocity(times, positions, 4.0) == []
