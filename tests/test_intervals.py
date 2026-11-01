"""Tests for neurospatial.Intervals and neurospatial.Interval."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from neurospatial import Interval, Intervals
from neurospatial.behavior import Trial


@pytest.fixture
def trials() -> Intervals[Trial]:
    """Four trials given out of time order, with non-contiguous ids."""
    records = [
        Trial(20.0, 29.0, "home", "left", "reached"),
        Trial(0.0, 8.0, "home", "right", "reached"),
        Trial(10.0, 17.0, "home", None, "returned"),
        Trial(30.0, 30.0, "home", None, "censored"),
    ]
    return Intervals(records, interval_ids=[7, 3, 5, 9])


def _random_intervals(rng: np.random.Generator, n: int) -> Intervals[Interval]:
    """``n`` non-overlapping intervals, some touching, with shuffled stable ids."""
    edges = np.cumsum(rng.uniform(0.0, 2.0, size=2 * n))
    start, stop = edges[0::2], edges[1::2]
    touching = rng.random(n - 1) < 0.3
    stop[:-1][touching] = start[1:][touching]
    ids = rng.permutation(10 * n)[:n]
    return Intervals.from_arrays(start, stop, interval_ids=ids)


def test_sample_interval_ids_match_loop() -> None:
    rng = np.random.default_rng(0)
    intervals = _random_intervals(rng, 200)
    times = np.sort(rng.uniform(-1.0, intervals.stop.max() + 1.0, size=10_000))
    # Put samples exactly on boundaries too.
    times = np.unique(np.concatenate([times, intervals.start, intervals.stop]))

    expected = np.full(times.size, -1, dtype=np.int64)
    for j, t in enumerate(times):
        for start, stop, interval_id in zip(
            intervals.start, intervals.stop, intervals.interval_ids, strict=True
        ):
            if start <= t < stop:
                expected[j] = interval_id
    np.testing.assert_array_equal(intervals.sample_interval_ids(times), expected)

    # A sample at a stop belongs to the next touching interval, or none.
    at_stop = np.searchsorted(times, intervals.stop)
    got = intervals.sample_interval_ids(times)[at_stop]
    touches = np.isin(intervals.stop, intervals.start)
    assert np.all(got[~touches] == -1)
    assert np.all(got[touches] != intervals.interval_ids[touches])
    assert np.all(got[touches] >= 0)


def test_ids_survive_selection(trials: Intervals[Trial]) -> None:
    assert trials.interval_ids.tolist() == [3, 5, 7, 9]  # sorted by time
    reached = trials[trials.outcome == "reached"]
    assert reached.interval_ids.tolist() == [3, 7]
    assert trials[np.array([3, 0])].interval_ids.tolist() == [3, 9]
    assert trials[1:3].interval_ids.tolist() == [5, 7]
    assert trials.sel(interval_id=[9, 5]).interval_ids.tolist() == [5, 9]
    restored = pickle.loads(pickle.dumps(reached))
    assert restored.interval_ids.tolist() == [3, 7]
    assert not restored.interval_ids.flags.writeable

    times = np.arange(0.0, 31.0, 0.5)
    ids = reached.sample_interval_ids(times)
    assert set(np.unique(ids).tolist()) == {-1, 3, 7}
    np.testing.assert_array_equal(restored.sample_interval_ids(times), ids)
    # The same samples get the same ids from the full table.
    full = trials.sample_interval_ids(times)
    np.testing.assert_array_equal(ids[ids >= 0], full[ids >= 0])


def test_sel_vs_position(trials: Intervals[Trial]) -> None:
    assert trials[0].start_time == 0.0  # first by time
    assert trials[-1].start_time == 30.0
    assert trials.sel(interval_id=7) == Trial(20.0, 29.0, "home", "left", "reached")
    with pytest.raises(KeyError, match=r"interval_id 1, 2 not in"):
        trials.sel(interval_id=[1, 2, 3])
    with pytest.raises(IndexError):
        trials[4]


def test_empty_selection_keeps_schema(trials: Intervals[Trial]) -> None:
    empty = trials[trials.outcome == "missing"]
    assert len(empty) == 0
    assert empty.fields == ("start_region", "end_region", "outcome")
    assert empty.record_type is Trial
    assert empty.outcome.dtype == object
    assert empty.outcome.shape == (0,)
    assert empty.windows.shape == (0, 2)
    table = empty.intervals_table()
    assert table.empty
    assert list(table.columns) == [
        "start_time",
        "stop_time",
        "start_region",
        "end_region",
        "outcome",
    ]
    assert trials[[]].fields == trials.fields

    numeric = Intervals.from_arrays([0.0], [1.0], rate=[2.5], count=[3])
    none = numeric[np.zeros(1, dtype=bool)]
    assert none.rate.dtype == np.float64
    assert none.count.dtype == np.int64


def test_zero_duration_observation(trials: Intervals[Trial]) -> None:
    assert len(trials) == 4
    assert trials.durations.tolist() == [8.0, 7.0, 9.0, 0.0]
    assert trials.windows.tolist() == [[0.0, 8.0], [10.0, 17.0], [20.0, 29.0]]
    assert 9 in trials.intervals_table().index
    times = np.arange(0.0, 32.0, 0.5)
    assert 9 not in trials.sample_interval_ids(times)
    assert trials.sample_slices(times)[3] == slice(60, 60)
    # A zero-duration row inside another row does not overlap it.
    nested = Intervals.from_arrays([0.0, 1.0], [2.0, 1.0])
    assert nested.sample_interval_ids(np.array([0.5, 1.0, 1.5])).tolist() == [0, 0, 0]


def test_overlap_refused_where_samples_assigned() -> None:
    overlapping = Intervals.from_arrays(
        [0.0, 10.0, 5.0], [20.0, 12.0, 6.0], interval_ids=[4, 8, 6]
    )
    times = np.arange(0.0, 25.0)
    message = r"intervals 4 and 6 overlap \(\[0, 20\) and \[5, 6\) s\)"
    with pytest.raises(ValueError, match=message):
        overlapping.sample_interval_ids(times)
    with pytest.raises(ValueError, match=message):
        overlapping.sample_values(times, overlapping.durations)
    slices = overlapping.sample_slices(times)
    assert slices == [slice(0, 20), slice(5, 6), slice(10, 12)]

    touching = Intervals.from_arrays([0.0, 5.0], [5.0, 9.0])
    assert touching.sample_interval_ids(np.array([4.0, 5.0, 9.0])).tolist() == [
        0,
        1,
        -1,
    ]


def test_sample_values_missing_outside(trials: Intervals[Trial]) -> None:
    times = np.array([1.0, 9.0, 21.0, 40.0])
    durations = trials.sample_values(times, trials.durations)
    np.testing.assert_array_equal(durations, [8.0, np.nan, 9.0, np.nan])
    assert trials.sample_values(times, "end_region").tolist() == [
        "right",
        None,
        "left",
        None,
    ]
    pairs = trials.sample_values(times, ("start_region", "end_region"))
    assert pairs.tolist() == [("home", "right"), None, ("home", "left"), None]
    ids = trials.sample_values(times, trials.interval_ids)
    assert ids.dtype == object
    assert ids.tolist() == [3, None, 7, None]
    by_id = pd.Series({7: "b", 3: "a", 5: "c", 9: "d"})
    assert trials.sample_values(times, by_id).tolist() == ["a", None, "b", None]


def test_sample_methods_warn_when_no_sample_inside(trials: Intervals[Trial]) -> None:
    times_ms = np.arange(1000.0, 2000.0)
    with pytest.warns(UserWarning, match="same clock"):
        ids = trials.sample_interval_ids(times_ms)
    assert np.all(ids == -1)
    # No warning without intervals that hold time.
    assert trials[[3]].sample_interval_ids(times_ms).tolist() == [-1] * 1000


def test_interval_record_pickles_and_hashes() -> None:
    record = Interval(1.0, 2.0, {"goal": "left", "rewarded": True, "n": 3})
    assert record.goal == "left"
    assert pickle.loads(pickle.dumps(record)) == record
    assert copy.deepcopy(record) == record
    assert hash(record) == hash(
        Interval(1.0, 2.0, {"goal": "left", "rewarded": True, "n": 3})
    )
    with pytest.raises(AttributeError, match=r"no field 'cue'"):
        _ = record.cue

    intervals = Intervals(
        [record, Interval(3.0, 4.0, {"goal": None, "rewarded": False, "n": 1})]
    )
    for clone in (pickle.loads(pickle.dumps(intervals)), copy.deepcopy(intervals)):
        assert list(clone) == list(intervals)
        assert clone.goal.tolist() == ["left", None]
        assert not clone.start.flags.writeable

    with pytest.raises(ValueError, match="empty string"):
        Interval(0.0, 1.0, {"goal": ""})
    with pytest.raises(ValueError, match="empty string"):
        Intervals.from_arrays([0.0], [1.0], goal=[""])
    with pytest.raises(TypeError, match="not hashable"):
        Interval(0.0, 1.0, {"path": [1, 2]})


def test_construction_refusals(trials: Intervals[Trial]) -> None:
    with pytest.raises(ValueError, match=r"repeated ids \[1\]"):
        Intervals.from_arrays([0.0, 1.0], [1.0, 2.0], interval_ids=[1, 1])
    with pytest.raises(ValueError, match=r"negative ids \[-1\]"):
        Intervals.from_arrays([0.0], [1.0], interval_ids=[-1])
    with pytest.raises(ValueError, match="start_time <= stop_time"):
        Interval(2.0, 1.0)
    with pytest.raises(ValueError, match="already an attribute"):
        Intervals.from_arrays([0.0], [1.0], windows=[1.0])
    with pytest.raises(TypeError, match="record_type"):
        Intervals([])
    with pytest.raises(ValueError, match="has fields"):
        Intervals([Interval(0.0, 1.0, {"a": 1}), Interval(1.0, 2.0, {"b": 1})])
    with pytest.raises(TypeError, match="Interval records too"):
        Intervals([*trials, Interval(0.0, 1.0)])
    with pytest.raises(ValueError, match="cannot repeat a row"):
        trials[[0, 0]]
    with pytest.raises(AttributeError, match=r"its fields are \['start_region'"):
        _ = trials.goal


def test_table_round_trip_keeps_schema(trials: Intervals[Trial]) -> None:
    reloaded = Intervals.from_dataframe(
        trials.intervals_table().reset_index(), id_column="interval_id"
    )
    assert reloaded.fields == trials.fields
    assert list(reloaded.intervals_table().columns) == list(
        trials.intervals_table().columns
    )
    np.testing.assert_array_equal(reloaded.interval_ids, trials.interval_ids)
    assert reloaded.end_region.tolist() == trials.end_region.tolist()


def test_from_dataframe() -> None:
    log = pd.DataFrame(
        {
            "trial": [12, 11],
            "begin": [10.0, 0.0],
            "end": [17.0, 8.0],
            "instructed_goal": ["right", np.nan],
            "reward_ml": [0.1, 0.0],
        }
    )
    trials = Intervals.from_dataframe(
        log, start_column="begin", stop_column="end", id_column="trial"
    )
    assert trials.fields == ("instructed_goal", "reward_ml")
    assert trials.interval_ids.tolist() == [11, 12]
    assert trials.instructed_goal.tolist() == [None, "right"]
    assert trials.reward_ml.dtype == np.float64
    assert trials.analysis_settings == {}

    with pytest.raises(ValueError, match="no column start_column='start_time'"):
        Intervals.from_dataframe(log)
    with pytest.raises(ValueError, match="id_column='trial_id'"):
        Intervals.from_dataframe(
            log, start_column="begin", stop_column="end", id_column="trial_id"
        )

    empty = Intervals.from_dataframe(
        log.iloc[:0], start_column="begin", stop_column="end", id_column="trial"
    )
    assert len(empty) == 0
    assert empty.fields == ("instructed_goal", "reward_ml")
    assert empty.reward_ml.dtype == np.float64


def test_analysis_settings_are_copies() -> None:
    intervals = Intervals(
        [Interval(0.0, 1.0)], analysis_settings={"epochs": [[0.0, 5.0]]}
    )
    settings = intervals.analysis_settings
    settings["epochs"][0][1] = 99.0
    assert intervals.analysis_settings == {"epochs": [[0.0, 5.0]]}
    assert intervals[[]].analysis_settings == {"epochs": [[0.0, 5.0]]}


def test_fields_are_copies_and_bounds_read_only(trials: Intervals[Trial]) -> None:
    outcome = trials.outcome
    outcome.iloc[0] = "timeout"
    assert trials.outcome.iloc[0] != "timeout"
    with pytest.raises(ValueError, match="read-only"):
        trials.start[0] = 1.0


def test_field_is_series_by_interval_id(trials: Intervals[Trial]) -> None:
    outcome = trials.outcome
    assert outcome.name == "outcome"
    assert outcome.index.name == "interval_id"
    assert outcome.index.tolist() == trials.interval_ids.tolist()
    subset = trials[[2, 0]]
    assert subset.outcome.index.tolist() == subset.interval_ids.tolist()
    empty = trials[trials.outcome == "never"]
    assert len(empty.outcome) == 0
    assert empty.outcome.dtype == trials.outcome.dtype


def test_boolean_series_selection_aligns_by_id() -> None:
    log = Intervals.from_arrays(
        [20.0, 0.0, 10.0], [29.0, 8.0, 17.0], interval_ids=[5, 9, 2], goal=list("aba")
    )
    # Rows are in time order (ids 9, 2, 5); a mask built in another order
    # selects by id.
    # By id this keeps id 5; by position it would keep the first row (id 9).
    mask = pd.Series([True, False, False], index=[5, 2, 9])
    assert log[mask].interval_ids.tolist() == [5]
    assert log[log.goal == "a"].interval_ids.tolist() == [2, 5]
    # A plain Boolean array still selects by position.
    assert log[np.array([True, False, False])].interval_ids.tolist() == [9]
    with pytest.raises(ValueError, match=r"ids \[7\]"):
        log[pd.Series([True, True, True, True], index=[5, 9, 2, 7])]


def test_with_fields_adds_and_replaces() -> None:
    trials = Intervals.from_arrays(
        [10.0, 0.0], [17.0, 8.0], interval_ids=[4, 1], end_region=["b", "a"]
    )
    labelled = trials.with_fields(
        outcome=pd.Series(["error", "correct"], index=[4, 1]), end_region=["x", "y"]
    )
    assert labelled.fields == ("end_region", "outcome")
    assert labelled.interval_ids.tolist() == [1, 4]
    assert labelled.outcome.tolist() == ["correct", "error"]
    assert labelled.end_region.tolist() == ["x", "y"]
    np.testing.assert_array_equal(labelled.start, trials.start)
    for name in ("interval_id", "start_time", "with_fields", "_hidden"):
        with pytest.raises(ValueError, match="field"):
            trials.with_fields(**{name: [1, 2]})
    with pytest.raises(ValueError, match="empty string"):
        trials.with_fields(label=["", "a"])
    with pytest.raises(ValueError, match="one entry per interval"):
        trials.with_fields(label=["a"])


def test_series_selection_must_be_boolean_with_unique_ids() -> None:
    log = Intervals.from_arrays([0.0, 10.0], [8.0, 17.0], n=[1, 2])
    with pytest.raises(TypeError, match="Boolean mask"):
        log[log.n - 1]
    with pytest.raises(ValueError, match=r"ids \[0\] repeat"):
        log[pd.Series([True, False, True], index=[0, 0, 1])]


def test_with_fields_reads_missing_labels_as_none() -> None:
    trials = Intervals.from_arrays([0.0, 10.0], [8.0, 17.0], interval_ids=[3, 4])
    labelled = trials.with_fields(
        goal=pd.Series(["a", np.nan], index=[3, 4]),
        cue=pd.Series(["x", pd.NA], index=[3, 4], dtype="string"),
    )
    assert labelled.goal.tolist() == ["a", None]
    assert labelled.cue.tolist() == ["x", None]
    with pytest.raises(TypeError, match="one value per interval"):
        trials.with_fields(goal="a")


def test_with_fields_keeps_analysis_settings() -> None:
    trials = Intervals(
        [Interval(0.0, 1.0)], analysis_settings={"detector": "test"}
    ).with_fields(label=["a"])
    assert trials.analysis_settings == {"detector": "test"}


def test_repr(trials: Intervals[Trial]) -> None:
    assert repr(trials) == (
        "Intervals[Trial]: 4 intervals in [0, 30) s; "
        "fields: start_region, end_region, outcome"
    )
    assert repr(trials[[]]) == (
        "Intervals[Trial]: 0 intervals; fields: start_region, end_region, outcome"
    )


def test_negative_step_slice_keeps_time_order(trials: Intervals[Trial]) -> None:
    reversed_rows = trials[::-1]
    assert reversed_rows.interval_ids.tolist() == trials.interval_ids.tolist()
    assert reversed_rows[0].start_time == 0.0
    np.testing.assert_array_equal(
        reversed_rows.sample_interval_ids(np.arange(0.0, 31.0)),
        trials.sample_interval_ids(np.arange(0.0, 31.0)),
    )


def test_series_by_id_must_cover_every_interval(trials: Intervals[Trial]) -> None:
    times = np.array([1.0, 11.0])
    partial = pd.Series({3: "a", 5: "b", 7: "c"})  # no value for id 9
    with pytest.raises(ValueError, match=r"lacks ids \[9\]"):
        trials.sample_values(times, partial)
    extra = pd.Series({3: "a", 5: "b", 7: "c", 9: "d", 4: "e"})
    with pytest.raises(ValueError, match=r"ids \[4\] that are not"):
        trials.sample_values(times, extra)


def test_from_dataframe_nullable_columns() -> None:
    log = pd.DataFrame(
        {
            "start_time": [0.0, 2.0, 4.0],
            "stop_time": [1.0, 3.0, 5.0],
            "rewarded": pd.array([True, None, False], dtype="boolean"),
            "lick_count": pd.array([3, None, 1], dtype="Int64"),
            "complete": pd.array([1, 2, 3], dtype="Int64"),
            "goal": pd.array(["left", None, "right"], dtype="string"),
        }
    )
    intervals = Intervals.from_dataframe(log)
    assert intervals.rewarded.tolist() == [True, None, False]
    assert intervals.lick_count.tolist() == [3, None, 1]
    assert intervals.complete.dtype == np.int64
    assert intervals.goal.tolist() == ["left", None, "right"]


def test_unhashable_value_inside_a_tuple_is_refused() -> None:
    with pytest.raises(TypeError, match="not hashable"):
        Interval(0.0, 1.0, {"path": (1, [2, 3])})


def test_no_rows_assign_no_sample() -> None:
    empty = Intervals.from_arrays([], [])
    assert empty.sample_interval_ids(np.array([0.0, 1.0])).tolist() == [-1, -1]
    assert np.isnan(empty.sample_values(np.array([0.0, 1.0]), empty.durations)).all()


def test_interval_id_is_not_a_field_name() -> None:
    with pytest.raises(ValueError, match="'interval_id'"):
        Intervals.from_arrays([0.0, 1.0], [1.0, 2.0], interval_id=[100, 200])
    log = pd.DataFrame({"start_time": [0.0], "stop_time": [1.0], "interval_id": [5]})
    with pytest.raises(ValueError, match="id_column='interval_id'"):
        Intervals.from_dataframe(log)


def test_ids_beyond_int64_are_refused() -> None:
    with pytest.raises(ValueError, match="int64"):
        Intervals.from_arrays(
            [0.0], [1.0], interval_ids=np.array([2**64 - 1], dtype=np.uint64)
        )


def test_spike_counts_by_hand_with_overlapping_rows():
    trials = Intervals.from_arrays(
        [0.0, 0.5, 2.0], [1.0, 1.5, 2.0], interval_ids=[4, 9, 2]
    )
    # A spike at a start counts, at a stop does not; the spike at 0.75 lies
    # in two overlapping rows and counts in both. Trains need not be sorted.
    counts = trials.spike_counts(
        {"a": np.array([1.0, 0.0, 0.75]), "b": np.array([1.5, 0.5])}
    )
    assert counts.index.tolist() == [4, 9, 2]
    assert counts.columns.tolist() == ["a", "b"]
    assert counts["a"].tolist() == [2, 2, 0]
    assert counts["b"].tolist() == [1, 1, 0]
    assert counts.dtypes.tolist() == [np.int64, np.int64]


def test_spike_counts_sum_to_the_spikes_inside_separate_rows():
    rng = np.random.default_rng(1)
    spikes = np.sort(rng.uniform(0, 100, 500))
    trials = Intervals.from_arrays([5.0, 20.0, 47.5, 80.0], [12.0, 33.3, 61.0, 99.0])
    inside = ((spikes[:, None] >= trials.start) & (spikes[:, None] < trials.stop)).any(
        axis=1
    )
    assert trials.spike_counts([spikes])[0].sum() == inside.sum()


def test_one_interval_restricts_an_analysis_like_its_row():
    from neurospatial._intervals import as_intervals

    trials = Intervals.from_arrays([1.0, 4.0, 6.0], [3.0, 5.0, 6.0])
    one = trials.sel(interval_id=1)
    np.testing.assert_array_equal(as_intervals(one, name="epochs"), [[4.0, 5.0]])
    np.testing.assert_array_equal(
        as_intervals(one, name="epochs"),
        as_intervals(trials[trials.interval_ids == 1], name="epochs"),
    )
    # Without duration it holds no time: an empty selection, as for Intervals.
    assert as_intervals(trials.sel(interval_id=2), name="epochs").shape == (0, 2)


def test_a_table_with_start_and_stop_columns_is_read_as_rows():
    from neurospatial._intervals import as_intervals

    table = pd.DataFrame({"start_time": [0.0, 2.0], "stop_time": [1.0, 3.0]})
    np.testing.assert_array_equal(as_intervals(table, name="epochs"), [[0, 1], [2, 3]])
    np.testing.assert_array_equal(as_intervals(table.iloc[:1], name="epochs"), [[0, 1]])


def test_a_record_with_a_missing_bound_is_refused():
    from neurospatial._intervals import as_intervals

    with pytest.raises(ValueError, match="NaN or inf"):
        as_intervals(pd.Series({"start_time": 0.0, "stop_time": np.nan}), name="epochs")


def test_from_dataframe_ids_are_the_table_index() -> None:
    log = pd.DataFrame(
        {
            "start_time": [10.0, 0.0, 20.0],
            "stop_time": [15.0, 5.0, 25.0],
            "outcome": ["error", "correct", "correct"],
        }
    )
    trials = Intervals.from_dataframe(log.sort_values("start_time"))
    # The log's own rows 1 and 2 are correct: they start at 0 and 20 s.
    assert trials[log.outcome == "correct"].start.tolist() == [0.0, 20.0]
    rewarded = trials.with_fields(reward=log.outcome == "correct")
    assert rewarded.reward.tolist() == [True, False, True]  # time order
    assert trials.interval_ids.tolist() == [1, 0, 2]
    # A default index numbers the rows 0 to n - 1 in table order, as before.
    assert Intervals.from_dataframe(log).interval_ids.tolist() == [1, 0, 2]


@pytest.mark.parametrize(
    "index",
    [["a", "b", "c"], [0.0, 1.0, 2.0], pd.date_range("2026-01-01", periods=3)],
)
def test_from_dataframe_refuses_an_index_that_is_not_integer(index) -> None:
    log = pd.DataFrame(
        {"start_time": [0.0, 1.0, 2.0], "stop_time": [0.5, 1.5, 2.5]}, index=index
    )
    with pytest.raises(TypeError, match=r"id_column=.*reset_index\(drop=True\)"):
        Intervals.from_dataframe(log)
    by_column = Intervals.from_dataframe(log.reset_index(drop=True))
    assert by_column.interval_ids.tolist() == [0, 1, 2]


def test_from_dataframe_refuses_a_repeated_index() -> None:
    log = pd.DataFrame({"start_time": [0.0, 1.0], "stop_time": [0.5, 1.5]})
    with pytest.raises(ValueError, match=r"reset_index\(drop=True\)"):
        Intervals.from_dataframe(pd.concat([log, log]))


def test_spike_counts_warn_for_spikes_on_another_clock() -> None:
    trials = Intervals.from_arrays([0.0, 20.0], [10.0, 30.0])
    with pytest.warns(UserWarning, match="spike_counts: none of the 2 interval"):
        counts = trials.spike_counts([np.array([1000.0, 1001.0]), np.array([1500.0])])
    assert counts.to_numpy().sum() == 0
    # No spikes at all is a silent population, not another clock.
    assert trials.spike_counts([np.array([])]).to_numpy().sum() == 0


def test_spike_counts_of_an_empty_intervals_and_a_zero_duration_row() -> None:
    empty = Intervals.from_arrays([], [])
    counts = empty.spike_counts([np.array([0.5, 1.0])])
    assert counts.shape == (0, 1)
    assert counts.index.name == "interval_id"
    # A row without duration holds no time, so a spike exactly on it is not
    # counted there; the touching row [5, 8) counts it.
    rows = Intervals.from_arrays([0.0, 5.0, 5.0], [5.0, 5.0, 8.0])
    assert rows.spike_counts([np.array([5.0])])[0].tolist() == [0, 0, 1]


def test_spike_counts_refuse_a_nan_spike() -> None:
    trials = Intervals.from_arrays([0.0], [10.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        trials.spike_counts([np.array([1.0, np.nan])])


@pytest.mark.pynapple
def test_spike_counts_of_a_tsgroup_keep_its_labels() -> None:
    nap = pytest.importorskip("pynapple")
    trials = Intervals.from_arrays([0.0, 5.0], [5.0, 10.0], interval_ids=[4, 8])
    # pynapple drops a unit with one spike, so each has at least two.
    group = nap.TsGroup(
        {3: nap.Ts(np.array([1.0, 6.0, 7.0])), 9: nap.Ts(np.array([2.0, 3.0]))}
    )
    counts = trials.spike_counts(group)
    assert counts.columns.tolist() == [3, 9]
    assert counts.loc[4].tolist() == [1, 2]
    assert counts.loc[8].tolist() == [2, 0]


def test_every_record_needs_finite_ordered_times() -> None:
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Stay:
        start_time: float
        stop_time: float

    assert Intervals([Stay(0.0, 1.0)]).stop.tolist() == [1.0]
    for start, stop in [(5.0, 1.0), (np.nan, 1.0), (0.0, np.inf)]:
        with pytest.raises(ValueError, match="start_time <= stop_time"):
            Intervals([Stay(start, stop)])


def test_records_must_be_frozen_dataclasses() -> None:
    import dataclasses

    @dataclasses.dataclass
    class Editable:
        start_time: float
        stop_time: float

    with pytest.raises(TypeError, match="frozen"):
        Intervals([Editable(0.0, 1.0)])
    with pytest.raises(TypeError, match="frozen"):
        Intervals([], record_type=Editable)


def test_with_fields_keeps_the_record_class_rules(trials: Intervals[Trial]) -> None:
    # Rows in time order: reached (right), returned, reached (left), censored.
    with pytest.raises(ValueError, match="end_region=None does not match"):
        trials.with_fields(outcome=["reached", "reached", "reached", "censored"])
    with pytest.raises(ValueError, match="outcome"):
        trials.with_fields(outcome=["won", "returned", "reached", "censored"])
    # A valid replacement and a new label are kept.
    relabelled = trials.with_fields(
        end_region=["left", None, "right", None], goal=["a", "b", "a", "b"]
    )
    assert relabelled.end_region.tolist() == ["left", None, "right", None]
    assert relabelled.goal.tolist() == ["a", "b", "a", "b"]
