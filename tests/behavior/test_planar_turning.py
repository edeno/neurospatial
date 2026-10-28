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


def test_horizontal_projection_of_legacy_example_is_a_left_turn():
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
    [[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]],
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
