"""Planar turning angle and geometric curvature on observed movement."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.behavior import (
    compute_movement_angular_velocity,
    compute_planar_curvature,
    compute_planar_turn_angles,
    resample_by_distance,
)

RADIUS = 7.0
ANGULAR_RATE = 0.8  # rad/s, counterclockwise
PLANAR = [compute_planar_turn_angles, compute_planar_curvature]
SUPPORTED = [*PLANAR, compute_movement_angular_velocity]


def circle(times: np.ndarray) -> np.ndarray:
    """Counterclockwise circle of radius 7 centred at the origin, shape (n, 2)."""
    return RADIUS * np.c_[np.cos(ANGULAR_RATE * times), np.sin(ANGULAR_RATE * times)]


@pytest.fixture(scope="module")
def uniform_circle():
    times = np.arange(0, 5, 1 / 30)
    return times, circle(times)


@pytest.mark.parametrize("function", PLANAR)
def test_three_columns_raise_with_horizontal_plane_fix(function):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="got shape \\(3, 3\\)") as caught:
        function([0.0, 0.1, 0.2], positions)
    assert "positions[:, :2]" in str(caught.value)
    assert "no sign without a reference axis" in str(caught.value)


@pytest.mark.parametrize("function", PLANAR)
def test_one_dimensional_positions_raise(function):
    with pytest.raises(ValueError, match="positions\\[:, :2\\]"):
        function([0.0, 0.1, 0.2], np.array([0.0, 1.0, 2.0]))


def test_horizontal_projection_of_a_3d_path_is_a_left_turn():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    angles = compute_planar_turn_angles([0.0, 0.1, 0.2], positions[:, :2])
    np.testing.assert_allclose(angles, [np.nan, np.pi / 2, np.nan])


@pytest.mark.parametrize("function", PLANAR)
def test_straight_line_interior_zero_endpoints_nan(function):
    times = np.arange(10) / 30
    positions = np.c_[np.linspace(0, 9, 10), np.linspace(0, 4.5, 10)]
    values = function(times, positions)
    assert values.shape == (10,) and values.dtype == np.float64
    assert np.isnan(values[[0, -1]]).all()
    np.testing.assert_allclose(values[1:-1], 0.0, atol=1e-12)


@pytest.mark.parametrize("function", PLANAR)
def test_fewer_than_three_samples_are_all_nan(function):
    values = function([0.0, 0.1], [[0.0, 0.0], [1.0, 0.0]])
    assert values.shape == (2,) and np.isnan(values).all()


def test_circle_curvature_exact_under_irregular_sampling():
    times = np.sort(np.random.default_rng(0).uniform(0, 5, 60))
    positions = circle(times)
    curvature = compute_planar_curvature(times, positions, max_gap=None)
    assert np.isnan(curvature[[0, -1]]).all()
    np.testing.assert_allclose(curvature[1:-1], 1 / RADIUS, rtol=1e-9)

    clockwise = compute_planar_curvature(times, positions[::-1].copy(), max_gap=None)
    np.testing.assert_allclose(clockwise[1:-1], -1 / RADIUS, rtol=1e-9)


def test_curvature_does_not_depend_on_frame_rate_but_turn_angle_does():
    slow_times = np.arange(0, 5, 1 / 30)
    fast_times = np.arange(0, 5, 1 / 120)
    slow_curvature = compute_planar_curvature(slow_times, circle(slow_times))
    fast_curvature = compute_planar_curvature(fast_times, circle(fast_times))
    np.testing.assert_allclose(slow_curvature[1:-1], 1 / RADIUS, rtol=1e-9)
    np.testing.assert_allclose(fast_curvature[1:-1], 1 / RADIUS, rtol=1e-9)

    slow_turn = compute_planar_turn_angles(slow_times, circle(slow_times))
    fast_turn = compute_planar_turn_angles(fast_times, circle(fast_times))
    np.testing.assert_allclose(
        np.nanmedian(np.abs(slow_turn)) / np.nanmedian(np.abs(fast_turn)),
        4.0,
        rtol=1e-6,
    )


def test_scaling_positions_scales_curvature_not_turn_angles(uniform_circle):
    times, positions = uniform_circle
    np.testing.assert_allclose(
        compute_planar_curvature(times, 3 * positions),
        compute_planar_curvature(times, positions) / 3,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        compute_planar_turn_angles(times, 3 * positions),
        compute_planar_turn_angles(times, positions),
        rtol=1e-12,
    )


@pytest.mark.parametrize("function", PLANAR)
def test_scaling_times_changes_nothing(function, uniform_circle):
    times, positions = uniform_circle
    np.testing.assert_array_equal(
        function(2 * times, positions), function(times, positions)
    )


@pytest.mark.parametrize("function", SUPPORTED)
def test_missing_sample_equals_deleted_row(function, uniform_circle):
    times, positions = uniform_circle[0][:20], uniform_circle[1][:20].copy()
    positions[5] = np.nan
    values = function(times, positions)
    kept = np.delete(np.arange(20), 5)
    assert np.isnan(values[5])
    np.testing.assert_allclose(
        values[kept], function(times[kept], positions[kept]), rtol=1e-12
    )
    assert np.isfinite(values[[4, 6]]).all()


@pytest.mark.parametrize("function", PLANAR)
def test_missing_sample_beyond_max_gap_blanks_it_and_its_neighbours(
    function, uniform_circle
):
    times, positions = uniform_circle[0][:20], uniform_circle[1][:20].copy()
    complete = function(times, positions)
    positions[5] = np.nan
    values = function(times, positions, max_gap=1.5 / 30)
    assert np.isnan(values[[4, 5, 6]]).all()
    keep = np.setdiff1d(np.arange(20), [4, 5, 6])
    np.testing.assert_array_equal(values[keep], complete[keep])


@pytest.mark.parametrize("function", PLANAR)
def test_declared_break_equals_separate_calls(function, uniform_circle):
    times, positions = uniform_circle[0][:20], uniform_circle[1][:20]
    break_before = np.zeros(20, dtype=bool)
    break_before[5] = True
    values = function(times, positions, break_before=break_before)
    assert np.isnan(values[[4, 5]]).all()
    separate = np.r_[
        function(times[:5], positions[:5]), function(times[5:], positions[5:])
    ]
    np.testing.assert_array_equal(values, separate)


@pytest.mark.parametrize("function", PLANAR)
def test_time_gap_longer_than_max_gap_acts_as_break(function, uniform_circle):
    times, positions = uniform_circle[0][:20].copy(), uniform_circle[1][:20]
    times[5:] += 10.0
    values = function(times, positions)
    break_before = np.zeros(20, dtype=bool)
    break_before[5] = True
    np.testing.assert_array_equal(
        values, function(times, positions, break_before=break_before, max_gap=None)
    )


@pytest.mark.parametrize("function", PLANAR)
def test_stationary_sample_is_nan(function):
    times = np.arange(5) / 30
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    values = function(times, positions)
    # Sample 1 has zero outgoing and sample 2 zero incoming displacement.
    assert np.isnan(values[[0, 1, 2, 4]]).all()
    assert np.isfinite(values[3])


@pytest.mark.parametrize(
    "positions",
    [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],  # cross product +0.0
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        [[2.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # cross product -0.0
        [[0.0, 2.0], [0.0, 1.0], [0.0, 2.0]],
    ],
)
def test_exact_reversal_turns_pi_and_has_no_curvature(positions):
    times = [0.0, 0.1, 0.2]
    assert compute_planar_turn_angles(times, positions)[1] == np.pi
    assert np.isnan(compute_planar_curvature(times, positions)).all()


@pytest.mark.parametrize(
    "positions",
    [
        [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [-2.0, 0.0]],
        [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, -3.0], [0.0, -1.0]],
    ],
)
def test_unequal_length_reversal_turns_pi_and_has_no_curvature(positions):
    times = [0.0, 0.1, 0.2]
    assert compute_planar_turn_angles(times, positions)[1] == np.pi
    assert np.isnan(compute_planar_curvature(times, positions)[1])


@pytest.mark.parametrize(
    "positions",
    [[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]],
)
def test_unequal_forward_straight_segment_has_zero_curvature(positions):
    times = [0.0, 0.1, 0.2]
    assert compute_planar_turn_angles(times, positions)[1] == 0.0
    assert compute_planar_curvature(times, positions)[1] == 0.0


def test_turns_recover_the_heading_changes_of_a_random_walk():
    rng = np.random.default_rng(42)
    heading_changes = rng.normal(0, 0.3, 200)
    step_lengths = rng.uniform(0.5, 2.0, 200)
    heading = np.cumsum(heading_changes)
    steps = step_lengths[:, None] * np.c_[np.cos(heading), np.sin(heading)]
    positions = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    turns = compute_planar_turn_angles(np.arange(201) / 30, positions)
    # The turn at sample i is the heading change between steps i - 1 and i.
    np.testing.assert_allclose(turns[1:-1], heading_changes[1:], atol=1e-9)


def test_right_turn_is_negative():
    times = [0.0, 0.1, 0.2]
    positions = [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0]]
    np.testing.assert_allclose(
        compute_planar_turn_angles(times, positions)[1], -np.pi / 2
    )
    assert compute_planar_curvature(times, positions)[1] < 0


def test_curvature_of_distance_resampled_path():
    times = np.arange(0, 5, 1e-3)
    resampled = resample_by_distance(times, circle(times), spacing=0.5, max_gap=None)
    curvature = compute_planar_curvature(
        resampled.times,
        resampled.positions,
        break_before=resampled.break_before,
        max_gap=None,
    )
    assert np.isnan(curvature[[0, -1]]).all()
    np.testing.assert_allclose(curvature[1:-1], 1 / RADIUS, rtol=1e-3)


def test_angular_velocity_recovers_rate_under_irregular_sampling():
    times = np.sort(np.random.default_rng(0).uniform(0, 5, 60))
    positions = circle(times)
    omega = compute_movement_angular_velocity(times, positions, max_gap=None)
    assert omega.shape == (60,) and omega.dtype == np.float64
    assert np.isnan(omega[[0, -1]]).all()
    np.testing.assert_allclose(omega[1:-1], ANGULAR_RATE, rtol=1e-9)

    clockwise = compute_movement_angular_velocity(
        times, positions * [1.0, -1.0], max_gap=None
    )
    np.testing.assert_allclose(clockwise[1:-1], -ANGULAR_RATE, rtol=1e-9)


def test_scaling_times_divides_angular_velocity_but_not_curvature(uniform_circle):
    times, positions = uniform_circle
    np.testing.assert_allclose(
        compute_movement_angular_velocity(2 * times, positions, max_gap=None),
        compute_movement_angular_velocity(times, positions) / 2,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        compute_planar_curvature(2 * times, positions, max_gap=None),
        compute_planar_curvature(times, positions),
    )


def test_angular_velocity_is_the_same_for_dense_and_sparse_sampling():
    dense = np.arange(0, 5, 1 / 120)
    sparse = np.arange(0, 5, 1 / 20)
    for times in (dense, sparse):
        omega = compute_movement_angular_velocity(times, circle(times))
        np.testing.assert_allclose(omega[1:-1], ANGULAR_RATE, rtol=1e-9)


def test_angular_velocity_is_turn_over_mean_neighbour_spacing():
    times = np.array([0.0, 0.1, 0.35, 0.4])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    omega = compute_movement_angular_velocity(times, positions)
    turns = compute_planar_turn_angles(times, positions)
    np.testing.assert_allclose(omega[1], turns[1] / 0.175)
    np.testing.assert_allclose(omega[2], turns[2] / 0.15)


def test_straight_line_angular_velocity_is_zero_in_the_interior():
    times = np.arange(10) / 30
    positions = np.c_[np.linspace(0, 9, 10), np.linspace(0, 4.5, 10)]
    omega = compute_movement_angular_velocity(times, positions)
    assert np.isnan(omega[[0, -1]]).all()
    np.testing.assert_allclose(omega[1:-1], 0.0, atol=1e-12)


def test_angular_velocity_is_nan_exactly_where_the_turn_angle_is(uniform_circle):
    times, positions = uniform_circle[0][:20], uniform_circle[1][:20].copy()
    positions[5] = np.nan
    positions[12] = positions[11]  # stationary
    break_before = np.zeros(20, dtype=bool)
    break_before[16] = True
    kwargs = {"break_before": break_before, "max_gap": 1.5 / 30}
    omega = compute_movement_angular_velocity(times, positions, **kwargs)
    turns = compute_planar_turn_angles(times, positions, **kwargs)
    np.testing.assert_array_equal(np.isnan(omega), np.isnan(turns))
    assert np.isnan(omega[[4, 5, 6, 11, 12, 15, 16]]).all()
    assert np.isfinite(omega[[1, 8, 13, 18]]).all()


def test_declared_break_angular_velocity_equals_separate_calls(uniform_circle):
    times, positions = uniform_circle[0][:20], uniform_circle[1][:20]
    break_before = np.zeros(20, dtype=bool)
    break_before[5] = True
    values = compute_movement_angular_velocity(
        times, positions, break_before=break_before
    )
    separate = np.r_[
        compute_movement_angular_velocity(times[:5], positions[:5]),
        compute_movement_angular_velocity(times[5:], positions[5:]),
    ]
    np.testing.assert_array_equal(values, separate)


@pytest.mark.parametrize("positions", [np.zeros((4, 3)), np.arange(4.0)])
def test_angular_velocity_rejects_non_planar_positions(positions):
    with pytest.raises(ValueError, match="positions\\[:, :2\\]"):
        compute_movement_angular_velocity(np.arange(4) / 30, positions)


@pytest.mark.parametrize("function", SUPPORTED)
def test_planar_error_names_the_called_function(function):
    with pytest.raises(ValueError, match=f"{function.__name__}\\(\\)"):
        function([0.0, 0.1, 0.2], np.zeros((3, 3)))


# Smoothing: a Gaussian in time over the defined turns of each run.


def _reference_smoothing(times, turns, run_ids, sigma):
    """Dense Gaussian-weighted mean of the defined turns in the same run."""
    lag = times[:, None] - times[None, :]
    defined = np.isfinite(turns)
    weight = np.exp(-0.5 * (lag / sigma) ** 2) * (np.abs(lag) <= 4 * sigma)
    weight *= (run_ids[:, None] == run_ids[None, :]) & defined[None, :]
    smoothed = weight @ np.where(defined, turns, 0.0) / weight.sum(axis=1)
    return np.where(defined, smoothed, np.nan)


def test_zero_smooth_window_is_bitwise_the_raw_turns(uniform_circle):
    times, positions = uniform_circle[0][:40], uniform_circle[1][:40].copy()
    positions[7] = np.nan
    raw = compute_planar_turn_angles(times, positions)
    for window in (0.0, 0, np.float32(0.0)):
        np.testing.assert_array_equal(
            compute_planar_turn_angles(times, positions, smooth_window=window), raw
        )


def test_smoothing_leaves_a_constant_turn_unchanged(uniform_circle):
    times, positions = uniform_circle
    turns = compute_planar_turn_angles(times, positions, smooth_window=0.2)
    assert np.isnan(turns[[0, -1]]).all()
    np.testing.assert_allclose(turns[1:-1], ANGULAR_RATE / 30, rtol=1e-9)


def test_smoothing_matches_a_gaussian_in_time_under_irregular_sampling():
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0, 4, 150))
    heading = np.cumsum(rng.normal(0, 0.3, 150))
    positions = np.cumsum(np.c_[np.cos(heading), np.sin(heading)], axis=0)
    positions[[20, 21, 90]] = np.nan
    positions[60] = positions[59]  # stationary: no turn at 59 or 60
    break_before = np.zeros(150, dtype=bool)
    break_before[100] = True
    kwargs = {"max_gap": None, "break_before": break_before}
    raw = compute_planar_turn_angles(times, positions, **kwargs)
    smoothed = compute_planar_turn_angles(times, positions, smooth_window=0.1, **kwargs)
    run_ids = (np.arange(150) >= 100).astype(int)
    np.testing.assert_allclose(
        smoothed, _reference_smoothing(times, raw, run_ids, 0.1), rtol=1e-12
    )


def test_undefined_turns_stay_nan_and_do_not_count_as_zero():
    # Turning left 0.1 rad per step, with a 10-sample pause in the middle.
    heading = 0.1 * np.arange(20)
    steps = np.c_[np.cos(heading), np.sin(heading)]
    path = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    path = np.insert(path, 10, np.repeat(path[10:11], 10, axis=0), axis=0)
    times = np.arange(len(path)) * 0.1
    raw = compute_planar_turn_angles(times, path)
    smoothed = compute_planar_turn_angles(times, path, smooth_window=0.3)
    np.testing.assert_array_equal(np.isnan(smoothed), np.isnan(raw))
    assert np.isnan(smoothed[10:20]).all()
    np.testing.assert_allclose(smoothed[np.isfinite(smoothed)], 0.1, rtol=1e-9)


def test_smoothed_missing_sample_equals_deleted_row():
    rng = np.random.default_rng(2)
    times = np.arange(60) / 30
    positions = np.cumsum(rng.normal(0, 1, (60, 2)), axis=0)
    positions[[10, 11, 40]] = np.nan
    kept = np.setdiff1d(np.arange(60), [10, 11, 40])
    values = compute_planar_turn_angles(times, positions, smooth_window=0.2)
    assert np.isnan(values[[10, 11, 40]]).all()
    np.testing.assert_allclose(
        values[kept],
        compute_planar_turn_angles(times[kept], positions[kept], smooth_window=0.2),
        rtol=1e-12,
    )


def test_smoothed_break_equals_separate_calls(uniform_circle):
    times = uniform_circle[0][:60]
    positions = np.r_[uniform_circle[1][:30], uniform_circle[1][:30][::-1]]
    break_before = np.zeros(60, dtype=bool)
    break_before[30] = True
    values = compute_planar_turn_angles(
        times, positions, break_before=break_before, smooth_window=0.5
    )
    separate = np.r_[
        compute_planar_turn_angles(times[:30], positions[:30], smooth_window=0.5),
        compute_planar_turn_angles(times[30:], positions[30:], smooth_window=0.5),
    ]
    np.testing.assert_array_equal(values, separate)
    # Left turns before the break, right turns after: nothing mixes across it.
    assert (values[1:29] > 0).all() and (values[31:59] < 0).all()


@pytest.mark.parametrize(
    "gate",
    [
        {"max_gap": 1.5 / 30, "missing": 30},
        {"epochs": [[0.0, 0.99], [1.0, 2.0]]},
    ],
)
def test_smoothing_never_crosses_a_long_gap_or_an_epoch_edge(gate, uniform_circle):
    times = uniform_circle[0][:60]
    positions = np.r_[uniform_circle[1][:30], uniform_circle[1][:30][::-1]]
    gate = dict(gate)
    missing = gate.pop("missing", None)
    if missing is not None:
        positions[missing] = np.nan
    first = slice(0, 30)
    second = slice(30 if missing is None else 31, 60)
    values = compute_planar_turn_angles(times, positions, smooth_window=0.5, **gate)
    np.testing.assert_array_equal(
        values[first],
        compute_planar_turn_angles(times[first], positions[first], smooth_window=0.5),
    )
    np.testing.assert_array_equal(
        values[second],
        compute_planar_turn_angles(times[second], positions[second], smooth_window=0.5),
    )


def test_smoothing_does_not_reach_across_a_bridged_gap_longer_than_the_kernel():
    # Straight along x, a 0.9 s tracking gap (bridged at max_gap=1.0), then
    # straight again for 0.2 s before a counterclockwise circle. Weights use
    # sample times, so turning after the gap never leaks into the straight run
    # before it, although the gap is a single sample interval.
    before = np.arange(0, 1, 1 / 30)
    after = 1.9 + np.arange(0, 3, 1 / 30)
    times = np.r_[before, after]
    x = 10.0 * np.r_[before, np.minimum(after, 2.1)]
    arc = np.clip(after - 2.1, 0, None)
    positions = np.c_[x, np.zeros(times.size)]
    positions[before.size :] += 7.0 * np.c_[np.sin(arc / 0.7), 1 - np.cos(arc / 0.7)]
    turns = compute_planar_turn_angles(times, positions, smooth_window=0.1)
    assert np.isfinite(turns[1 : before.size]).all()
    np.testing.assert_allclose(turns[1 : before.size], 0.0, atol=1e-12)
    assert np.nanmax(turns) > 0.01


@pytest.mark.parametrize("window", [-0.1, np.nan, np.inf, None, True, "0.2"])
def test_invalid_smooth_window_raises_with_a_fix(window, uniform_circle):
    times, positions = uniform_circle
    with pytest.raises(ValueError, match="smooth_window") as caught:
        compute_planar_turn_angles(times, positions, smooth_window=window)
    assert "compute_planar_turn_angles()" in str(caught.value)
    assert "smooth_window=0.0" in str(caught.value)


def test_smoothing_averages_out_tracking_jitter():
    times = np.arange(0, 10, 1 / 30)
    path = np.c_[20.0 * times, np.zeros(times.size)]  # straight at 20 cm/s
    jittered = path + np.random.default_rng(0).normal(0, 0.1, path.shape)
    raw = compute_planar_turn_angles(times, jittered)
    smoothed = compute_planar_turn_angles(times, jittered, smooth_window=0.2)
    assert np.nanstd(smoothed) < 0.05 * np.nanstd(raw)
    assert abs(np.nanmean(smoothed)) < 0.01


def test_smoothed_stationary_path_has_no_turn():
    times = np.array([0.0, 1.0, 2.0])
    turns = compute_planar_turn_angles(
        times, np.zeros((3, 2)), smooth_window=0.2, max_gap=None
    )
    assert turns.shape == (3,) and np.isnan(turns).all()
