"""Shared trajectory masks restrict complete intervals to recording windows."""

import numpy as np
import pytest

from neurospatial._intervals import (
    interval_valid_mask,
    start_allocated_occupancy,
    warn_excluded_intervals,
)


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
        ({}, [True, True, True]),
    ],
)
def test_interval_valid_mask_windows(kwargs, expected):
    times = np.array([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(interval_valid_mask(times, **kwargs), expected)


def test_min_speed_without_speed_is_refused():
    with pytest.raises(ValueError, match="without speed"):
        interval_valid_mask(np.array([0.0, 0.5, 1.0]), min_speed=0.5)


@pytest.mark.parametrize("bad", [0, -1.0, np.nan, np.inf, "0.5", True])
def test_shared_gate_refuses_invalid_max_gap(bad):
    with pytest.raises(ValueError, match="max_gap must be a positive finite"):
        interval_valid_mask(np.array([0.0, 0.5, 1.0]), max_gap=bad)


def test_encoding_refuses_invalid_max_gap():
    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rate

    times = np.linspace(0.0, 10.0, 201)
    positions = np.column_stack([np.linspace(0, 20, 201), np.zeros(201)])
    env = Environment.from_samples(positions, bin_size=5.0)
    with pytest.raises(ValueError, match="max_gap must be a positive finite"):
        compute_spatial_rate(env, np.array([1.0, 2.0]), times, positions, max_gap=-1.0)


def test_inclusive_gap_contract():
    np.testing.assert_array_equal(
        interval_valid_mask(np.array([0.0, 0.5, 1.0]), max_gap=0.5), [True, True]
    )


@pytest.mark.parametrize("rate", [30.0, 1000.0, 29.97])
def test_max_gap_at_the_sampling_interval_warns_about_rounding(rate):
    times = np.arange(10_000) / rate
    mask = interval_valid_mask(times, max_gap=1 / rate)
    assert 0 < np.count_nonzero(~mask) < mask.size  # rounding drops a share
    with pytest.warns(UserWarning, match="only by float rounding") as record:
        warn_excluded_intervals(times, mask, max_gap=1 / rate)
    assert "not in seconds" not in str(record[0].message)


def test_max_gap_above_the_sampling_interval_is_silent(recwarn):
    times = np.arange(10_000) / 30.0
    mask = interval_valid_mask(times, max_gap=1.5 / 30.0)
    assert mask.all()
    warn_excluded_intervals(times, mask, max_gap=1.5 / 30.0)
    assert not recwarn.list


def test_all_excluded_without_named_gates_reads_cleanly():
    with pytest.warns(UserWarning) as record:
        warn_excluded_intervals(np.arange(4.0), np.zeros(3, bool), max_gap=None)
    message = str(record[0].message)
    assert "gate(s) )" not in message and "Fix: ." not in message
    assert "Fix: check that the samples are finite" in message


def test_start_allocated_occupancy_refuses_bins_outside_the_environment():
    with pytest.raises(ValueError, match="only 3 bins"):
        start_allocated_occupancy(
            np.array([0, 5, 1, 0]),
            np.full(3, 0.1),
            np.ones(3, dtype=bool),
            n_bins=3,
        )


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
    from neurospatial._intervals import observed_runs

    def bounds(runs):
        return [(int(run[0]), int(run[-1]) + 1) for run in runs]

    times = two_epoch_recording.times
    assert bounds(observed_runs(times, max_gap=0.5, epochs=None)) == [
        (0, 5000),
        (5000, 10000),
    ]
    assert bounds(observed_runs(times, max_gap=0.5, epochs=[(0.0, 50.0)])) == [
        (0, 2501)
    ]
    assert bounds(
        observed_runs(
            np.array([0.0, 0.1, 0.2, 10.0, 20.0, 20.1, 20.2]), max_gap=0.5, epochs=None
        )
    ) == [(0, 3), (4, 7)]
    # Missing samples are skipped inside a run, as if their rows were deleted.
    runs = observed_runs(
        np.arange(6) * 0.1,
        max_gap=0.5,
        epochs=None,
        sample_valid=np.array([True, True, False, True, True, False]),
    )
    assert [run.tolist() for run in runs] == [[0, 1, 3, 4]]


def test_start_allocated_occupancy():
    from neurospatial._intervals import start_allocated_occupancy

    bins = np.array([0, -1, 1, 0])
    dt = np.array([0.1, 10.0, 0.2])
    mask = np.array([True, False, True])
    np.testing.assert_array_equal(
        start_allocated_occupancy(bins, dt, mask, 3), [0.1, 0.2, 0.0]
    )
    np.testing.assert_array_equal(
        start_allocated_occupancy(bins, dt, mask, 3, return_seconds=False), [1, 1, 0]
    )
