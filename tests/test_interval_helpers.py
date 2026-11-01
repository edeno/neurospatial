"""Time-window normalization, containment, intersections and valid runs."""

import re
import sys
import warnings
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
    rows = as_intervals(value, name="epochs")
    # pynapple intervals include their end; the half-open stop is the next
    # float, so a sample exactly at 10.0 stays inside.
    np.testing.assert_array_equal(rows[:, 0], [0, 20])
    np.testing.assert_array_equal(rows[:, 1], np.nextafter([10.0, 30.0], np.inf))
    t = np.array([10.0, 30.0])
    np.testing.assert_array_equal(value.in_interval(nap.Ts(t=t)), [0, 1])
    inside = (t[:, None] >= rows[:, 0]) & (t[:, None] < rows[:, 1])
    np.testing.assert_array_equal(inside.any(axis=1), [True, True])


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        ([0, 1, 2], "shape (3,)"),
        (np.empty((0, 2)), "no rows"),
        ([[0, np.nan]], "row(s) 0 contain NaN or inf"),
        ([[5, 5]], "row(s) 0 have stop <= start"),
        (IntervalSetLike(np.array([0, 1]), np.array([2])), "shapes (2,) and (1,)"),
        ("bad", "holds text"),
        ([{}, 1.0], "could not be read as numbers"),
    ],
)
def test_as_intervals_errors_follow_contract(value, detail):
    from neurospatial._intervals import as_intervals

    with pytest.raises(
        ValueError, match=r"Why: epochs and spike_window are half-open"
    ) as exc:
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
    with pytest.raises(
        ValueError, match=r"Invalid time window: epochs is a pair of 1-D arrays, which"
    ) as exc:
        as_intervals(value, name="epochs")
    message = str(exc.value)
    assert "parallel" in message
    assert "np.column_stack([starts, stops])" in message
    assert "Why:" in message
    assert "\nFix:" in message


def test_as_intervals_reports_every_problem():
    from neurospatial._intervals import as_intervals, resolve_time_windows

    with pytest.raises(ValueError, match=r"Invalid time window: epochs row") as exc:
        as_intervals([[1, 0], [np.nan, 1]], name="epochs")
    assert "row(s) 0 have stop <= start" in str(exc.value)
    assert "row(s) 1 contain NaN or inf" in str(exc.value)
    with pytest.raises(ValueError, match=r"Invalid time window: epochs row") as exc:
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

    with pytest.raises(
        ValueError,
        match=r"Invalid time window: epochs\.start and epochs\.end could not",
    ) as exc:
        as_intervals(value, name="epochs")
    message = str(exc.value)
    assert "epochs.start and epochs.end" in message
    assert "could not be read as numbers" in message
    assert "Why:" in message
    assert "\nFix:" in message


def test_intervalset_numeric_error_does_not_hide_other_window_errors():
    from neurospatial._intervals import resolve_time_windows

    value = IntervalSetLike(np.array(["bad"]), np.array([10]))
    with pytest.raises(
        ValueError,
        match=r"Invalid time window: epochs\.start and epochs\.end could not",
    ) as exc:
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
    ("mask", "expected"),
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


def test_break_before_drops_only_the_interval_it_starts():
    from neurospatial._intervals import interval_valid_mask
    from neurospatial._observation import resolve_sample_support

    times = np.arange(6) * 0.1
    breaks = np.array([True, False, False, True, False, False])

    # Entry 0 is ignored; the break at sample 3 drops interval (2, 3) only.
    mask = interval_valid_mask(times, max_gap=0.5, break_before=breaks)
    np.testing.assert_array_equal(mask, [True, True, False, True, True])
    runs = resolve_sample_support(
        times,
        np.ones(times.size, dtype=bool),
        max_gap=0.5,
        epochs=None,
        break_before=breaks,
        result="the result is empty",
    ).run_samples()
    assert [run.tolist() for run in runs] == [[0, 1, 2], [3, 4, 5]]


def test_break_before_matches_an_equivalent_gap():
    from neurospatial._intervals import interval_valid_mask

    # A declared break and a gap longer than max_gap exclude the same interval.
    regular = np.arange(6) * 0.1
    gapped = np.r_[regular[:3], regular[3:] + 10.0]
    breaks = np.zeros(6, dtype=bool)
    breaks[3] = True
    np.testing.assert_array_equal(
        interval_valid_mask(regular, max_gap=0.5, break_before=breaks),
        interval_valid_mask(gapped, max_gap=0.5),
    )


def test_all_excluding_breaks_are_named_in_the_warning():
    from neurospatial._intervals import observed_interval_mask

    times = np.arange(3) * 0.1
    with pytest.warns(UserWarning, match="break_before"):
        mask = observed_interval_mask(
            times,
            max_gap=0.5,
            epochs=None,
            break_before=np.array([False, True, True]),
        )
    assert not mask.any()


@pytest.mark.parametrize("how", ["direct", "replace", "deepcopy", "pickle"])
def test_time_support_arrays_are_read_only_on_every_path(how):
    import copy
    import dataclasses
    import pickle

    from neurospatial._intervals import TimeSupport

    mask = np.array([True, False, True])
    support = TimeSupport(mask, np.array([[0.0, 1.0]]), None, None, 0.5)
    if how == "replace":
        support = dataclasses.replace(support, runs=np.array([[0.0, 2.0]]))
    elif how == "deepcopy":
        support = copy.deepcopy(support)
    elif how == "pickle":
        support = pickle.loads(pickle.dumps(support))
    mask[0] = False
    assert support.mask[0]
    assert not support.mask.flags.writeable
    assert not support.runs.flags.writeable


def test_time_support_records_fractional_runs_when_given():
    from neurospatial._intervals import TimeSupport

    times = np.array([0.0, 1.0, 2.0])
    support = TimeSupport.from_mask(
        times,
        np.array([True, True]),
        epochs=None,
        spike_window=None,
        max_gap=None,
        runs=np.array([[0.25, 1.75]]),
    )
    assert support.temporal_settings()["observed_intervals"] == [[0.25, 1.75]]


def test_touching_interval_sets_do_not_intersect():
    from neurospatial._intervals import intersect_intervals

    actual = intersect_intervals(np.array([[0.0, 10.0]]), np.array([[10.0, 20.0]]))
    assert actual.shape == (0, 2)


@pytest.mark.parametrize("container", [tuple, list])
def test_pair_of_one_dimensional_arrays_is_ambiguous(container) -> None:
    """Two Series (or arrays) could be two rows or starts and stops; with two
    windows both readings have shape (2, 2), so either guess could be wrong."""
    pd = pytest.importorskip("pandas")
    from neurospatial._intervals import as_intervals

    value = container([pd.Series([0.0, 10.0]), pd.Series([5.0, 15.0])])
    with pytest.raises(ValueError, match=re.escape("could be two [start, stop) rows")):
        as_intervals(value, name="epochs")


def test_pair_of_plain_lists_is_two_rows() -> None:
    from neurospatial._intervals import as_intervals

    rows = as_intervals(([0.0, 5.0], [10.0, 15.0]), name="epochs")
    np.testing.assert_array_equal(rows, [[0.0, 5.0], [10.0, 15.0]])


def test_starts_and_stops_as_rows_get_a_column_stack_hint() -> None:
    from neurospatial._intervals import as_intervals

    with pytest.raises(
        ValueError, match=re.escape("pass np.column_stack([starts, stops])")
    ):
        as_intervals(np.array([[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]]), name="epochs")


def test_context_warnings_survive_an_error_in_the_block() -> None:
    """A warning raised before an error often explains it, so it is kept."""

    from neurospatial._stacklevel import _warnings_with_context

    def emit_then_fail() -> None:
        with _warnings_with_context("ctx"):
            warnings.warn("all intervals excluded", UserWarning, stacklevel=1)
            msg = "nothing left"
            raise ValueError(msg)

    with (
        pytest.warns(UserWarning, match="ctx: all intervals excluded"),
        pytest.raises(ValueError, match="nothing left"),
    ):
        emit_then_fail()


def test_break_declared_on_a_missing_row_moves_to_the_next_kept_row():
    from neurospatial._intervals import drop_missing_samples

    index, breaks = drop_missing_samples(
        np.array([True, True, False, True, True]),
        np.array([False, False, True, False, False]),
    )
    np.testing.assert_array_equal(index, [0, 1, 3, 4])
    np.testing.assert_array_equal(breaks, [False, False, True, False])


def test_break_before_the_first_kept_row_is_dropped():
    """No recording precedes the first kept row, so there is nothing to split."""
    from neurospatial._intervals import drop_missing_samples

    index, breaks = drop_missing_samples(
        np.array([False, True, True]), np.array([False, True, False])
    )
    np.testing.assert_array_equal(index, [1, 2])
    np.testing.assert_array_equal(breaks, [False, False])


@pytest.mark.parametrize("seed", range(20))
def test_merge_covers_exactly_the_union_of_its_rows(seed):
    """A time is in the merged rows iff it is in some input row; rows stay apart."""
    from neurospatial._intervals import _merge

    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 12))
    # Integer grid endpoints make touching and nested rows common.
    starts = rng.integers(0, 20, n).astype(float)
    rows = np.column_stack([starts, starts + rng.integers(1, 6, n)])
    merged = _merge(rows)

    queries = np.unique(np.r_[rows.ravel(), rng.uniform(-1, 26, 200)])
    inside_input = ((rows[:, :1] <= queries) & (queries < rows[:, 1:])).any(axis=0)
    inside_merged = ((merged[:, :1] <= queries) & (queries < merged[:, 1:])).any(axis=0)
    np.testing.assert_array_equal(inside_merged, inside_input)
    assert (merged[:, 0] < merged[:, 1]).all()
    assert (merged[:-1, 1] < merged[1:, 0]).all()


def test_windows_protocol():
    from neurospatial import Interval, Intervals
    from neurospatial._intervals import as_intervals

    intervals = Intervals(
        [
            Interval(20.0, 30.0),
            Interval(0.0, 10.0),
            Interval(12.0, 12.0),  # observed, no time
        ]
    )
    np.testing.assert_array_equal(
        as_intervals(intervals, name="epochs"),
        as_intervals(intervals.windows, name="epochs"),
    )
    np.testing.assert_array_equal(
        as_intervals(intervals, name="epochs"), [[0.0, 10.0], [20.0, 30.0]]
    )

    # No windows is an empty selection; it stays empty when normalized again.
    empty = as_intervals(intervals[[]], name="epochs")
    assert empty.shape == (0, 2)
    assert as_intervals(empty, name="epochs").shape == (0, 2)
    assert as_intervals(intervals[[1]], name="epochs").shape == (0, 2)

    # A plain array without rows is still refused as a probable bug.
    with pytest.raises(ValueError, match="no rows"):
        as_intervals(np.empty((0, 2)), name="epochs")


@pytest.mark.pynapple
def test_windows_protocol_keeps_pynapple_inclusive_end():
    nap = pytest.importorskip("pynapple")
    from neurospatial._intervals import as_intervals, pynapple_end_to_stop

    iset = nap.IntervalSet(start=[0.0, 5.0], end=[2.0, 7.0])
    np.testing.assert_array_equal(
        as_intervals(iset, name="epochs")[:, 1],
        pynapple_end_to_stop(np.array([2.0, 7.0])),
    )


def test_linear_allocated_occupancy_by_hand():
    """Constant speed between samples, worked by hand on 1 cm bins [0, 4).

    0.5 -> 2.5 cm in 4 s moves at 0.5 cm/s: 1 s in [0, 1) (0.5 cm), 2 s in
    [1, 2) and 1 s in [2, 3). Then 3 s standing at 2.5 cm all go to [2, 3).
    The 10 s step back to 0 cm is masked out.
    """
    from neurospatial._intervals import linear_allocated_occupancy

    edges = np.arange(5.0)
    values = np.array([0.5, 2.5, 2.5, 0.0])
    dt = np.array([4.0, 3.0, 10.0])
    mask = np.array([True, True, False])
    np.testing.assert_array_equal(
        linear_allocated_occupancy(values, dt, edges, mask), [1.0, 2.0, 4.0, 0.0]
    )
    # Reversing the direction of travel splits the time the same way.
    np.testing.assert_array_equal(
        linear_allocated_occupancy(values[1::-1], dt[:1], edges, mask[:1]),
        [1.0, 2.0, 1.0, 0.0],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.0, [0.0, 0.0, 3.0, 0.0]),  # an inner edge starts its bin
        (0.0, [3.0, 0.0, 0.0, 0.0]),  # the first edge is inside
        (4.0, [0.0, 0.0, 0.0, 0.0]),  # the last edge is outside (half-open)
        (-1.0, [0.0, 0.0, 0.0, 0.0]),
        (np.nan, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_linear_allocated_occupancy_standing_still(value, expected):
    """A zero-length step puts all its time in the half-open bin of its value."""
    from neurospatial._intervals import linear_allocated_occupancy

    np.testing.assert_array_equal(
        linear_allocated_occupancy(
            np.array([value, value]), np.array([3.0]), np.arange(5.0), np.array([True])
        ),
        expected,
    )


def test_linear_allocated_occupancy_clips_a_step_beyond_the_edges():
    """-2 -> 6 cm in 8 s: 1 s in each of the four 1 cm bins, the rest outside."""
    from neurospatial._intervals import linear_allocated_occupancy

    np.testing.assert_array_equal(
        linear_allocated_occupancy(
            np.array([-2.0, 6.0]), np.array([8.0]), np.arange(5.0), np.array([True])
        ),
        [1.0, 1.0, 1.0, 1.0],
    )


def test_starts_and_stops_stacked_as_two_rows_are_ambiguous() -> None:
    from neurospatial._intervals import as_intervals

    starts, stops = np.array([0.0, 10.0]), np.array([5.0, 15.0])
    # Read as rows, [[0, 10], [5, 15]] would merge into one window [0, 15).
    with pytest.raises(
        ValueError, match=re.escape("pass np.column_stack([starts, stops])")
    ):
        as_intervals(np.vstack([starts, stops]), name="epochs")
    np.testing.assert_array_equal(
        as_intervals(np.column_stack([starts, stops]), name="epochs"),
        [[0.0, 5.0], [10.0, 15.0]],
    )
    # Rows that do not overlap, or whose transpose is not two ordered
    # windows, are read as rows.
    np.testing.assert_array_equal(
        as_intervals(np.array([[0.0, 5.0], [10.0, 15.0]]), name="epochs"),
        [[0.0, 5.0], [10.0, 15.0]],
    )
    np.testing.assert_array_equal(
        as_intervals(np.array([[0.0, 30.0], [5.0, 10.0]]), name="epochs"),
        [[0.0, 30.0]],
    )


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (np.array([[0, 5]], dtype="timedelta64[s]"), "total_seconds()"),
        (np.array([[0, 5]], dtype="datetime64[s]"), "datetime"),
        (np.array([[False, True]]), "bool"),
        (np.array([["0", "5"]]), "text"),
        (np.array([["0", "5"]], dtype=object), "text"),
        ([("0", "5")], "text"),
    ],
)
def test_non_numeric_epochs_are_refused(value, detail) -> None:
    from neurospatial._intervals import as_intervals

    with pytest.raises(ValueError, match="float seconds") as exc:
        as_intervals(value, name="epochs")
    assert detail in str(exc.value)


def test_timedelta_epochs_name_the_conversion() -> None:
    pd = pytest.importorskip("pandas")
    from neurospatial._intervals import as_intervals

    table = pd.DataFrame(
        {
            "start_time": pd.to_timedelta([0, 10], unit="s"),
            "stop_time": pd.to_timedelta([5, 15], unit="s"),
        }
    )
    with pytest.raises(ValueError, match=re.escape("np.timedelta64(1, 's')")):
        as_intervals(table, name="epochs")


def test_a_table_is_read_by_its_start_and_stop_column_names() -> None:
    pd = pytest.importorskip("pandas")
    from neurospatial._intervals import as_intervals

    table = pd.DataFrame(
        {
            "outcome": ["correct", "error"],
            "stop_time": [5.0, 15.0],
            "trial": [3, 4],
            "start_time": [0.0, 10.0],
        }
    )
    np.testing.assert_array_equal(
        as_intervals(table, name="epochs"), [[0.0, 5.0], [10.0, 15.0]]
    )


def test_a_table_without_bound_columns_gets_a_table_hint() -> None:
    pd = pytest.importorskip("pandas")
    from neurospatial._intervals import as_intervals

    table = pd.DataFrame({"a": [0.0, 10.0, 20.0], "b": [5.0, 15.0, 25.0]})
    np.testing.assert_array_equal(
        as_intervals(table, name="epochs"), [[0.0, 5.0], [10.0, 15.0], [20.0, 25.0]]
    )
    with pytest.raises(ValueError, match="start_time and stop_time") as exc:
        as_intervals(table.T, name="epochs")
    assert "column_stack" not in str(exc.value)


@pytest.mark.nwb
def test_an_nwb_time_intervals_table_points_to_read_intervals() -> None:
    epoch = pytest.importorskip("pynwb.epoch")
    from neurospatial._intervals import as_intervals

    table = epoch.TimeIntervals(name="trials", description="trials")
    table.add_interval(start_time=0.0, stop_time=1.0)
    table.add_interval(start_time=2.0, stop_time=3.0)
    with pytest.raises(ValueError, match="read_intervals") as exc:
        as_intervals(table, name="epochs")
    assert "Intervals.from_dataframe" in str(exc.value)
