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


def test_speed_matches_rate_map_gate_with_missing_rows_and_gaps():
    from neurospatial.encoding._spatial_counts import resolve_speed

    times = np.r_[np.arange(0, 4, 1 / 30), 10 + np.arange(0, 4, 1 / 30)]
    rng = np.random.default_rng(2)
    positions = np.column_stack([20 * times, np.zeros_like(times)])
    positions = positions + rng.normal(0, JITTER_CM, positions.shape)
    positions[[7, 8, 200]] = np.nan
    np.testing.assert_array_equal(
        compute_speed(times, positions),
        resolve_speed(times, positions, None, min_speed=1.0, max_gap=0.5),
    )


def test_epochs_speed_equals_speed_of_the_selected_samples():
    times = np.arange(0, 4, 1 / 30)
    _, positions = _running(30.0, seconds=4.0)
    result = compute_speed(times, positions, epochs=(1.01, 2.99))
    inside = (times > 1.01) & (times < 2.99)
    assert np.isnan(result[~inside]).all()
    np.testing.assert_array_equal(
        result[inside], compute_speed(times[inside], positions[inside])
    )


def test_missing_row_is_nan_and_splits_the_run():
    times = np.arange(10) / 4
    x = np.arange(10.0)
    x[5] = np.nan
    np.testing.assert_array_equal(
        compute_speed(times, x, smooth_window=0.0),
        [4, 4, 4, 4, 4, np.nan, 4, 4, 4, 4],
    )


def test_lone_sample_has_no_speed():
    np.testing.assert_array_equal(
        compute_speed(np.array([0.0]), np.array([[1.0, 2.0]])), [np.nan]
    )


def test_declared_break_equals_separate_calls():
    """A declared break keeps two recordings on one clock from smoothing together."""
    t_a, p_a = _running(30.0, seconds=2.0, seed=1)
    t_b, p_b = _running(30.0, seconds=2.0, seed=3)
    times = np.r_[t_a, t_a[-1] + 1 / 30 + t_b]
    positions = np.r_[p_a, p_b + np.array([0.0, 30.0])]  # put back 30 cm away
    n = len(t_a)
    new_recording = np.arange(len(times)) == n
    joined = compute_speed(times, positions, break_before=new_recording)
    # Slice the same arrays: re-zeroing the second clock changes the time
    # differences by rounding, so the comparison would no longer be exact.
    np.testing.assert_array_equal(
        joined,
        np.r_[
            compute_speed(times[:n], positions[:n]),
            compute_speed(times[n:], positions[n:]),
        ],
    )
    around_join = slice(n - 10, n + 10)
    assert np.nanmax(compute_speed(times, positions)[around_join]) > 60
    assert np.nanmax(joined[around_join]) < 30


def test_repeated_timestamp_raises():
    times = np.array([0.0, 0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_speed(times, np.arange(8.0).reshape(4, 2))


def test_malformed_break_before_raises():
    times = np.arange(5) / 10
    with pytest.raises(ValueError, match="Boolean array"):
        compute_speed(
            times, np.arange(10.0).reshape(5, 2), break_before=np.zeros(5, dtype=int)
        )


def test_negative_smooth_window_raises():
    times = np.arange(5) / 10
    with pytest.raises(ValueError, match="smooth_window must be a finite number"):
        compute_speed(times, np.arange(10.0).reshape(5, 2), smooth_window=-0.1)
