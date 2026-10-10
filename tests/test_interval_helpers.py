"""Time-window normalization, containment, intersections and valid runs."""

import sys
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class IntervalSetLike:
    """A minimal interval holder with no dependency on pynapple."""

    start: np.ndarray
    end: np.ndarray


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((0, 10), [[0, 10]]),
        ([[20, 30], [0, 10]], [[0, 10], [20, 30]]),
        ([[0, 10], [10, 20]], [[0, 20]]),
        ([[0, 10], [5, 15]], [[0, 15]]),
        ([[0, 30], [5, 10], [20, 25]], [[0, 30]]),
    ],
)
def test_as_intervals_accepted_forms(value, expected):
    from neurospatial._intervals import as_intervals

    actual = as_intervals(value, name="epochs")
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    assert as_intervals(None, name="epochs") is None


def test_as_intervals_duck_types_intervalset():
    from neurospatial._intervals import as_intervals

    already_imported = "pynapple" in sys.modules
    value = IntervalSetLike(np.array([20.0, 0.0]), np.array([30.0, 10.0]))
    np.testing.assert_array_equal(
        as_intervals(value, name="epochs"), [[0, 10], [20, 30]]
    )
    if not already_imported:
        assert "pynapple" not in sys.modules


@pytest.mark.pynapple
def test_as_intervals_real_intervalset():
    nap = pytest.importorskip("pynapple")
    from neurospatial._intervals import as_intervals

    value = nap.IntervalSet(start=[0, 20], end=[10, 30])
    np.testing.assert_array_equal(
        as_intervals(value, name="epochs"), [[0, 10], [20, 30]]
    )


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        ([0, 1, 2], "shape (3,)"),
        (np.empty((0, 2)), "no rows"),
        ([[0, np.nan]], "row(s) 0 contain NaN or inf"),
        ([[5, 5]], "row(s) 0 have stop <= start"),
        (IntervalSetLike(np.array([0, 1]), np.array([2])), "shapes (2,) and (1,)"),
        ("bad", "could not be read as numbers"),
    ],
)
def test_as_intervals_errors_follow_contract(value, detail):
    from neurospatial._intervals import as_intervals

    with pytest.raises(ValueError) as exc:
        as_intervals(value, name="epochs")
    message = str(exc.value)
    assert "epochs" in message
    assert detail in message
    assert "Why:" in message
    assert "\nFix:" in message


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("container", [tuple, list])
def test_as_intervals_rejects_parallel_start_stop_arrays(n, container):
    """Two 1-D arrays are the retired (starts, ends) form, never two rows.

    At n=2 the pair also has shape (2, 2), so reading it as rows would silently
    analyze [[0, 10], [5, 15]] instead of [[0, 5], [10, 15]].
    """
    from neurospatial._intervals import as_intervals

    starts = np.arange(n) * 10.0
    value = container([starts, starts + 5.0])
    with pytest.raises(ValueError) as exc:
        as_intervals(value, name="epochs")
    message = str(exc.value)
    assert "parallel" in message
    assert "np.column_stack([starts, stops])" in message
    assert "Why:" in message
    assert "\nFix:" in message


def test_as_intervals_reports_every_problem():
    from neurospatial._intervals import as_intervals, resolve_time_windows

    with pytest.raises(ValueError) as exc:
        as_intervals([[1, 0], [np.nan, 1]], name="epochs")
    assert "row(s) 0 have stop <= start" in str(exc.value)
    assert "row(s) 1 contain NaN or inf" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        resolve_time_windows([[1, 0]], [[np.inf, 2]])
    assert "epochs" in str(exc.value)
    assert "spike_window" in str(exc.value)


@pytest.mark.parametrize(
    "value",
    [
        IntervalSetLike(np.array(["bad"]), np.array([10])),
        IntervalSetLike(np.array([0]), np.array(["bad"])),
        IntervalSetLike(np.array([object()]), np.array([10])),
    ],
)
def test_intervalset_numeric_errors_follow_contract(value):
    from neurospatial._intervals import as_intervals

    with pytest.raises(ValueError) as exc:
        as_intervals(value, name="epochs")
    message = str(exc.value)
    assert "epochs.start and epochs.end" in message
    assert "could not be read as numbers" in message
    assert "Why:" in message
    assert "\nFix:" in message


def test_intervalset_numeric_error_does_not_hide_other_window_errors():
    from neurospatial._intervals import resolve_time_windows

    value = IntervalSetLike(np.array(["bad"]), np.array([10]))
    with pytest.raises(ValueError) as exc:
        resolve_time_windows(value, [[5, 1]])
    message = str(exc.value)
    assert "epochs.start and epochs.end" in message
    assert "spike_window row(s) 0 have stop <= start" in message
    assert "Why:" in message
    assert "\nFix:" in message


def test_intervals_contain():
    from neurospatial._intervals import intervals_contain

    queries = np.array(
        [
            [0, 10],
            [5, 10],
            [9, 11],
            [10, 20],
            [-1, 0],
            [20, 30],
            [25, 31],
            [29.5, 30],
            [np.nan, 1],
            [1, np.nan],
        ]
    )
    actual = intervals_contain(
        np.array([[0, 10], [20, 30]]), queries[:, 0], queries[:, 1]
    )
    np.testing.assert_array_equal(
        actual, [True, True, False, False, False, True, False, True, False, False]
    )
    np.testing.assert_array_equal(
        intervals_contain(np.empty((0, 2)), queries[:, 0], queries[:, 1]),
        np.zeros(10, dtype=bool),
    )


def test_interval_operations_agree_with_reference():
    from neurospatial._intervals import (
        as_intervals,
        intersect_intervals,
        intervals_contain,
    )

    rng = np.random.default_rng(0)
    for _ in range(200):
        a = as_intervals(np.sort(rng.uniform(-20, 20, (10, 2)), axis=1), name="epochs")
        b = as_intervals(
            np.sort(rng.uniform(-20, 20, (10, 2)), axis=1), name="spike_window"
        )
        queries = np.sort(rng.uniform(-25, 25, (50, 2)), axis=1)
        queries[::10, 0] = np.nan
        expected = [
            any(lo <= start and stop <= hi for lo, hi in a) for start, stop in queries
        ]
        np.testing.assert_array_equal(
            intervals_contain(a, queries[:, 0], queries[:, 1]), expected
        )
        intersections = np.array(
            [
                (max(alo, blo), min(ahi, bhi))
                for alo, ahi in a
                for blo, bhi in b
                if max(alo, blo) < min(ahi, bhi)
            ]
        ).reshape(-1, 2)
        np.testing.assert_array_equal(intersect_intervals(a, b), intersections)


def test_intersect_intervals():
    from neurospatial._intervals import intersect_intervals

    a = np.array([[0, 10], [20, 30]])
    np.testing.assert_array_equal(
        intersect_intervals(a, np.array([[5, 25]])), [[5, 10], [20, 25]]
    )
    assert intersect_intervals(a, np.array([[40, 50]])).shape == (0, 2)
    assert intersect_intervals(np.empty((0, 2)), a).shape == (0, 2)
    assert intersect_intervals(a, np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([True, True, False, True], [[0, 2], [3, 4]]),
        ([False, False], []),
        ([], []),
        ([True, True], [[0, 2]]),
    ],
)
def test_run_bounds(mask, expected):
    from neurospatial._intervals import run_sample_bounds, run_time_bounds

    mask = np.asarray(mask, dtype=bool)
    expected = np.asarray(expected, dtype=np.intp).reshape(-1, 2)
    np.testing.assert_array_equal(run_sample_bounds(mask), expected)
    times = 100 + np.arange(mask.size + 1) / 10
    np.testing.assert_array_equal(run_time_bounds(times, mask), times[expected])
