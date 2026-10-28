"""Pin the numerical output of the sample-based turning and step-length functions.

These values are guarded so that the gap-aware planar functions can be added
beside them without changing what existing callers receive.
"""

import numpy as np

from neurospatial.behavior.trajectory import (
    compute_step_lengths,
    compute_trajectory_curvature,
    compute_turn_angles,
)

_TOL = {"rtol": 0, "atol": 1e-15}

_THREE_D = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
_WITH_DUPLICATE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
_THREE_D_TIMES = np.arange(len(_THREE_D)) * 0.1
_WITH_DUPLICATE_TIMES = np.arange(len(_WITH_DUPLICATE)) * 0.1


# Steps (1, 0, 1) and (0, 1, 1): cos(angle) = 1 / (sqrt(2) sqrt(2)), so pi / 3.
# The golden was pi / 4 while the turn mixed an x-y cross product with a 3-D
# dot product, which is neither the planar nor the 3-D angle.
def test_three_dimensional_turn_is_the_unsigned_angle_between_steps():
    np.testing.assert_allclose(
        compute_turn_angles(_THREE_D), [np.pi / 3], rtol=1e-15, atol=0
    )


def test_three_dimensional_curvature_is_the_unsigned_angle_between_steps():
    np.testing.assert_allclose(
        compute_trajectory_curvature(_THREE_D_TIMES, _THREE_D, smooth_window=0.0),
        [np.nan, np.pi / 3, np.nan],
        rtol=1e-15,
        atol=1e-15,
        equal_nan=True,
    )


def test_three_dimensional_reversal_and_straight_step():
    positions = np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 1]])
    np.testing.assert_allclose(
        compute_turn_angles(positions), [0.0, np.pi], rtol=1e-15, atol=1e-15
    )


def test_dropping_the_third_coordinate_gives_the_planar_right_angle():
    np.testing.assert_allclose(
        compute_turn_angles(_THREE_D[:, :2]), [np.pi / 2], **_TOL
    )


def test_turn_angles_drop_duplicate_positions():
    np.testing.assert_allclose(
        compute_turn_angles(_WITH_DUPLICATE), [0.0, np.pi / 2], **_TOL
    )


# The run's ends and the stationary repeat have no turn: NaN (they were 0.0,
# which smoothing then spread into the real turns).
def test_curvature_without_smoothing_places_angle_at_retained_middle_sample():
    np.testing.assert_allclose(
        compute_trajectory_curvature(
            _WITH_DUPLICATE_TIMES, _WITH_DUPLICATE, smooth_window=0.0
        ),
        [np.nan, 0.0, np.nan, np.pi / 2, np.nan],
        equal_nan=True,
        **_TOL,
    )


def test_turn_angle_is_independent_of_step_length():
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    np.testing.assert_allclose(compute_turn_angles(positions), [np.pi / 2], **_TOL)


def test_nan_sample_poisons_every_turn_angle():
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [2.0, 0.0], [3.0, 0.0]]
    )
    result = compute_turn_angles(positions)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, np.full(3, np.nan), equal_nan=True, **_TOL)


def test_step_lengths_keep_zero_length_steps():
    positions = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(compute_step_lengths(positions), [5.0, 0.0, 5.0], **_TOL)


def test_step_lengths_propagate_nan_to_both_adjacent_steps():
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    np.testing.assert_allclose(
        compute_step_lengths(positions), [np.nan, np.nan], equal_nan=True, **_TOL
    )
