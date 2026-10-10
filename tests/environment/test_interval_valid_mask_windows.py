"""Shared trajectory masks restrict complete intervals to recording windows."""

import numpy as np
import pytest

from neurospatial.environment.trajectory import interval_valid_mask


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"epochs": np.array([[0.0, 1.0]])}, [True, True, False]),
        ({"spike_window": np.array([[0.0, 1.0]])}, [True, True, False]),
        (
            {"epochs": np.array([[0.0, 1.0]]), "spike_window": np.array([[0.5, 2.0]])},
            [False, True, False],
        ),
        ({"start_bin": np.array([0, -1, 0, 0])}, [True, False, True]),
        (
            {"speed": np.array([1.0, 0.0, 1.0, 1.0]), "min_speed": 0.5},
            [True, False, True],
        ),
        ({"min_speed": 0.5}, [True, True, True]),
        ({}, [True, True, True]),
    ],
)
def test_interval_valid_mask_windows(kwargs, expected):
    times = np.array([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(interval_valid_mask(times, **kwargs), expected)


def test_interval_valid_mask_rejects_partial_window():
    times = np.array([0.0, 0.5, 1.0, 1.5])
    actual = interval_valid_mask(times, epochs=np.array([[0.25, 1.25]]))
    np.testing.assert_array_equal(actual, [False, True, False])


def test_interval_valid_mask_gap_without_positions():
    np.testing.assert_array_equal(
        interval_valid_mask(np.array([0.0, 0.1, 10.0, 10.1])), [True, False, True]
    )
    assert interval_valid_mask(np.array([0.0])).shape == (0,)


def test_observed_runs(two_epoch_recording):
    from neurospatial.environment.trajectory import observed_runs

    times = two_epoch_recording.times
    assert observed_runs(times, max_gap=0.5, epochs=None) == [
        slice(0, 5000),
        slice(5000, 10000),
    ]
    assert observed_runs(times, max_gap=0.5, epochs=[(0.0, 50.0)]) == [slice(0, 2501)]
    assert observed_runs(
        np.array([0.0, 0.1, 10.0, 20.0, 20.1]), max_gap=0.5, epochs=None
    ) == [slice(0, 2), slice(3, 5)]


def test_start_allocated_occupancy():
    from neurospatial.environment.trajectory import start_allocated_occupancy

    bins = np.array([0, -1, 1, 0])
    dt = np.array([0.1, 10.0, 0.2])
    mask = np.array([True, False, True])
    np.testing.assert_array_equal(
        start_allocated_occupancy(bins, dt, mask, 3), [0.1, 0.2, 0.0]
    )
    np.testing.assert_array_equal(
        start_allocated_occupancy(bins, dt, mask, 3, return_seconds=False), [1, 1, 0]
    )
