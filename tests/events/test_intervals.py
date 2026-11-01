"""Tests for the conversions between point events and Intervals."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Intervals
from neurospatial.events import (
    events_to_intervals,
    filter_by_intervals,
    intervals_to_events,
)


def _events(*times: float, **columns) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": list(times), **columns})


# =============================================================================
# Inputs shared by intervals_to_events and filter_by_intervals
# =============================================================================


def _call(name, intervals):
    if name == "intervals_to_events":
        return intervals_to_events(intervals)
    return filter_by_intervals(_events(1.0), intervals)


@pytest.mark.parametrize("name", ["intervals_to_events", "filter_by_intervals"])
class TestIntervalsArgument:
    def test_a_dataframe_points_to_intervals_from_dataframe(self, name):
        """A table's column names and stop meaning are the user's to state."""
        table = pd.DataFrame({"start_time": [0.0], "stop_time": [2.0]})

        with pytest.raises(TypeError, match=r"Intervals\.from_dataframe"):
            _call(name, table)

    def test_other_inputs_are_refused_with_the_accepted_types(self, name):
        with pytest.raises(TypeError, match="Intervals or a pynapple IntervalSet"):
            _call(name, np.array([[0.0, 2.0]]))


@pytest.mark.parametrize(("start", "stop"), [(0.0, np.nan), (0.0, -5.0)])
def test_nan_or_inverted_rows_are_refused_before_any_conversion(start, stop):
    """An inverted or NaN interval would silently select nothing.

    Such rows are refused when a table becomes Intervals, so no conversion
    or filter can see them.
    """
    table = pd.DataFrame({"start_time": [start], "stop_time": [stop]})

    with pytest.raises(ValueError, match="start_time <= stop_time"):
        Intervals.from_dataframe(table)


# =============================================================================
# intervals_to_events
# =============================================================================


class TestIntervalsToEvents:
    def test_extract_start_times(self):
        intervals = Intervals.from_arrays(
            [1.0, 3.0, 5.0], [2.0, 4.0, 6.0], label=["a", "b", "c"]
        )

        result = intervals_to_events(intervals, which="start")

        assert len(result) == 3
        assert_allclose(result["timestamp"].to_numpy(), [1.0, 3.0, 5.0])

    def test_extract_stop_times(self):
        intervals = Intervals.from_arrays([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])

        result = intervals_to_events(intervals, which="stop")

        assert_allclose(result["timestamp"].to_numpy(), [2.0, 4.0, 6.0])

    def test_extract_both_times(self):
        intervals = Intervals.from_arrays([1.0, 5.0], [2.0, 6.0])

        result = intervals_to_events(intervals, which="both")

        assert_allclose(result["timestamp"].to_numpy(), [1.0, 2.0, 5.0, 6.0])
        assert result["boundary"].tolist() == ["start", "stop", "start", "stop"]

    def test_each_event_carries_its_interval_id(self):
        """Ids follow the interval, not its position."""
        intervals = Intervals.from_arrays([5.0, 1.0], [6.0, 2.0], interval_ids=[12, 30])

        result = intervals_to_events(intervals, which="both")

        assert_allclose(result["timestamp"], [1.0, 2.0, 5.0, 6.0])
        assert result["interval_id"].tolist() == [30, 30, 12, 12]
        assert result["interval_id"].dtype == np.int64
        assert "boundary" not in intervals_to_events(intervals).columns

    def test_preserve_fields(self):
        intervals = Intervals.from_arrays(
            [1.0, 3.0], [2.0, 4.0], trial_id=[1, 2], condition=["A", "B"]
        )

        result = intervals_to_events(
            intervals, which="start", preserve_fields=["trial_id", "condition"]
        )

        assert list(result.columns) == [
            "timestamp",
            "interval_id",
            "trial_id",
            "condition",
        ]
        assert result["trial_id"].tolist() == [1, 2]
        assert result["condition"].tolist() == ["A", "B"]

    def test_preserve_fields_with_both(self):
        """Each interval's fields go on both its boundaries."""
        intervals = Intervals.from_arrays([1.0, 5.0], [2.0, 6.0], trial_id=[1, 2])

        result = intervals_to_events(
            intervals, which="both", preserve_fields=["trial_id"]
        )

        assert result["trial_id"].tolist() == [1, 1, 2, 2]

    def test_fields_are_not_copied_unless_asked(self):
        intervals = Intervals.from_arrays([1.0], [2.0], trial_id=[1])

        assert "trial_id" not in intervals_to_events(intervals).columns

    def test_empty_intervals(self):
        result = intervals_to_events(Intervals.from_arrays([], []), which="start")

        assert len(result) == 0
        assert result["timestamp"].dtype == np.float64

    def test_abutting_intervals_list_stop_before_start(self):
        intervals = Intervals.from_arrays([0.0, 1.0], [1.0, 2.0])

        result = intervals_to_events(intervals, which="both")

        assert result["boundary"].tolist() == ["start", "stop", "start", "stop"]
        assert_allclose(result["timestamp"], [0.0, 1.0, 1.0, 2.0])
        assert result["interval_id"].tolist() == [0, 0, 1, 1]

    def test_single_interval(self):
        result = intervals_to_events(Intervals.from_arrays([5.0], [10.0]), which="both")

        assert_allclose(result["timestamp"].to_numpy(), [5.0, 10.0])
        assert result["boundary"].tolist() == ["start", "stop"]

    def test_overlapping_intervals(self):
        intervals = Intervals.from_arrays([1.0, 2.0], [3.0, 4.0])

        result = intervals_to_events(intervals, which="both")

        assert_allclose(result["timestamp"].to_numpy(), [1.0, 2.0, 3.0, 4.0])
        assert result["interval_id"].tolist() == [0, 1, 0, 1]

    def test_invalid_which_raises(self):
        with pytest.raises(ValueError, match=r"which.*must be"):
            intervals_to_events(Intervals.from_arrays([1.0], [2.0]), which="invalid")

    def test_preserve_field_not_found_raises(self):
        intervals = Intervals.from_arrays([1.0], [2.0], trial_id=[1])

        with pytest.raises(
            ValueError, match=r"(?s)nonexistent.*choose from \['trial_id'\]"
        ):
            intervals_to_events(intervals, preserve_fields=["nonexistent"])

    def test_index_is_reset(self):
        intervals = Intervals.from_arrays([3.0, 1.0, 2.0], [4.0, 2.0, 3.0])

        result = intervals_to_events(intervals, which="start")

        assert list(result.index) == [0, 1, 2]
        assert_allclose(result["timestamp"], [1.0, 2.0, 3.0])

    def test_a_task_log_with_its_own_column_names(self):
        """A table with other names becomes Intervals by naming its columns."""
        log = pd.DataFrame({"begin": [3.0, 1.0], "end": [4.0, 2.0], "trial": [8, 7]})
        trials = Intervals.from_dataframe(
            log, start_column="begin", stop_column="end", id_column="trial"
        )

        result = intervals_to_events(trials, which="start")

        assert_allclose(result["timestamp"], [1.0, 3.0])
        assert result["interval_id"].tolist() == [7, 8]

    def test_typical_trial_use_case(self):
        """Trial starts of one condition, for PSTH alignment."""
        trials = Intervals.from_arrays(
            [0.0, 10.0, 20.0, 30.0],
            [8.0, 18.0, 28.0, 38.0],
            trial_type=["go", "nogo", "go", "nogo"],
            correct=[True, True, False, True],
        )

        trial_starts = intervals_to_events(
            trials, which="start", preserve_fields=["trial_type", "correct"]
        )

        go_starts = trial_starts[trial_starts["trial_type"] == "go"]
        assert_allclose(go_starts["timestamp"].to_numpy(), [0.0, 20.0])
        assert trial_starts["correct"].tolist() == [True, True, False, True]
        # Selecting trials first gives the same events.
        go_only = intervals_to_events(trials[trials.trial_type == "go"])
        assert_allclose(go_only["timestamp"], go_starts["timestamp"])


# =============================================================================
# events_to_intervals
# =============================================================================


class TestEventsToIntervals:
    def test_basic_sequential_pairing(self):
        result = events_to_intervals(_events(1.0, 5.0, 10.0), _events(3.0, 8.0, 15.0))

        assert isinstance(result, Intervals)
        assert_allclose(result.start, [1.0, 5.0, 10.0])
        assert_allclose(result.stop, [3.0, 8.0, 15.0])
        assert_allclose(result.durations, [2.0, 3.0, 5.0])
        assert result.interval_ids.tolist() == [0, 1, 2]
        assert result.fields == ()

    def test_match_by_column(self):
        start_events = _events(1.0, 2.0, 3.0, trial_id=[1, 2, 3])
        # Out of order, but matched by trial_id.
        stop_events = _events(5.0, 4.0, 6.0, trial_id=[2, 1, 3])

        result = events_to_intervals(start_events, stop_events, match_by="trial_id")

        assert_allclose(result.start, [1.0, 2.0, 3.0])
        assert_allclose(result.stop, [4.0, 5.0, 6.0])
        assert result.fields == ("trial_id",)
        assert result._field_array("trial_id").tolist() == [1, 2, 3]

    def test_max_duration_filter(self):
        # Sorted starts [1, 5, 10] pair with sorted stops [3, 8, 20].
        result = events_to_intervals(
            _events(1.0, 5.0, 10.0), _events(3.0, 20.0, 8.0), max_duration=5.0
        )

        assert_allclose(result.start, [1.0, 5.0])
        assert_allclose(result.stop, [3.0, 8.0])

    def test_empty_events_keep_the_field_schema(self):
        result = events_to_intervals(
            pd.DataFrame({"timestamp": [], "region": []}), _events()
        )

        assert len(result) == 0
        assert result.fields == ("region",)

    def test_single_interval(self):
        result = events_to_intervals(_events(5.0), _events(10.0))

        assert result.start.tolist() == [5.0]
        assert result.stop.tolist() == [10.0]

    def test_mismatched_counts_raises(self):
        with pytest.raises(ValueError, match=r"count.*match"):
            events_to_intervals(_events(1.0, 2.0, 3.0), _events(4.0, 5.0))

    def test_match_by_missing_column_raises(self):
        with pytest.raises(ValueError, match=r"trial_id"):
            events_to_intervals(_events(1.0), _events(2.0), match_by="trial_id")

    def test_match_by_unmatched_values_raises(self):
        starts = _events(1.0, 2.0, trial_id=[1, 2])
        stops = _events(3.0, 4.0, trial_id=[1, 3])  # 3 has no start

        with pytest.raises(ValueError, match=r"unmatched"):
            events_to_intervals(starts, stops, match_by="trial_id")

    def test_stop_before_start_is_refused(self):
        """A negative duration means a start and stop were mismatched."""
        with pytest.raises(ValueError, match="before its start"):
            events_to_intervals(_events(5.0), _events(3.0))

    def test_missed_and_extra_event_is_refused(self):
        """A missed exit plus an extra one keeps the counts equal.

        Pairing by rank would return the overlapping [30, 58) and [50, 60).
        """
        with pytest.raises(ValueError, match="do not alternate"):
            events_to_intervals(_events(10.0, 30.0, 50.0), _events(15.0, 58.0, 60.0))

    def test_abutting_intervals_pair_sequentially(self):
        result = events_to_intervals(_events(0.0, 1.0), _events(1.0, 2.0))

        assert_allclose(np.column_stack([result.start, result.stop]), [[0, 1], [1, 2]])

    @pytest.mark.parametrize("side", ["start_events", "stop_events"])
    def test_nan_timestamp_is_refused(self, side):
        """Sorted last, a NaN stop would hand the next trial's stop to this one."""
        frames = {
            "start_events": _events(10.0, 30.0),
            "stop_events": _events(15.0, 35.0),
        }
        frames[side].loc[0, "timestamp"] = np.nan

        with pytest.raises(ValueError, match=rf"{side}.*NaN"):
            events_to_intervals(frames["start_events"], frames["stop_events"])

    def test_match_by_stop_before_start_is_refused(self):
        starts = _events(1.0, 2.0, trial=[1, 2])
        stops = _events(5.0, 0.5, trial=[1, 2])

        with pytest.raises(ValueError, match="stop before they start"):
            events_to_intervals(starts, stops, match_by="trial")

    def test_match_by_missing_key_is_refused(self):
        starts = _events(1.0, 2.0, trial=[1.0, np.nan])
        stops = _events(1.5, 2.5, trial=[1.0, np.nan])

        with pytest.raises(ValueError, match="missing values"):
            events_to_intervals(starts, stops, match_by="trial")

    @pytest.mark.parametrize("max_duration", [np.nan, -1.0])
    def test_invalid_max_duration_is_refused(self, max_duration):
        """NaN would silently drop every interval."""
        with pytest.raises(ValueError, match="max_duration"):
            events_to_intervals(_events(0.0), _events(1.0), max_duration=max_duration)

    def test_missing_timestamp_column_is_explained(self):
        with pytest.raises(ValueError, match=r"Missing required columns.*timestamp"):
            events_to_intervals(pd.DataFrame({"time": [0.0]}), _events(1.0))

    def test_start_event_columns_become_fields(self):
        result = events_to_intervals(
            _events(1.0, 5.0, condition=["A", "B"]), _events(3.0, 8.0)
        )

        assert result.fields == ("condition",)
        assert result.condition.tolist() == ["A", "B"]

    def test_output_sorted_by_start_time(self):
        result = events_to_intervals(_events(10.0, 1.0, 5.0), _events(12.0, 3.0, 8.0))

        assert_allclose(result.start, [1.0, 5.0, 10.0])
        assert_allclose(result.stop, [3.0, 8.0, 12.0])
        assert result.interval_ids.tolist() == [0, 1, 2]

    def test_an_interval_id_column_gives_the_ids(self):
        result = events_to_intervals(
            _events(10.0, 1.0, interval_id=[4, 9]), _events(12.0, 3.0)
        )

        assert result.interval_ids.tolist() == [9, 4]
        assert result.fields == ()

    def test_typical_zone_crossing_use_case(self):
        """Zone entries and exits become dwell intervals."""
        entries = _events(10.0, 30.0, 50.0, region=["goal"] * 3)
        exits = _events(15.0, 35.0, 58.0, region=["goal"] * 3)

        dwell = events_to_intervals(entries, exits)

        assert_allclose(dwell.durations, [5.0, 5.0, 8.0])
        assert dwell.durations.sum() == 18.0  # total dwell time
        assert dwell.region.tolist() == ["goal"] * 3


class TestEventsToIntervalsMatchByUniqueness:
    """match_by requires one-to-one (non-duplicate) keys."""

    def test_duplicate_match_key_raises(self):
        """Duplicate keys on both sides must raise, not cross-join."""
        starts = _events(1.0, 2.0, 3.0, trial_id=[1, 1, 2])
        stops = _events(4.0, 5.0, 6.0, trial_id=[1, 1, 2])

        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            events_to_intervals(starts, stops, match_by="trial_id")

    def test_duplicate_start_only_raises(self):
        """A duplicate on the start side alone must raise.

        The set-difference (unmatched-values) check cannot catch this:
        ``{1, 2}`` equals ``{1, 2}`` even though ``trial_id=1`` repeats on
        the start side.
        """
        starts = _events(1.0, 2.0, 3.0, trial_id=[1, 1, 2])
        stops = _events(4.0, 6.0, trial_id=[1, 2])

        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            events_to_intervals(starts, stops, match_by="trial_id")

    def test_unique_match_key_pairs_each_start_with_its_stop(self):
        starts = _events(1.0, 2.0, 3.0, trial_id=[1, 2, 3])
        stops = _events(5.0, 4.0, 6.0, trial_id=[2, 1, 3])

        intervals = events_to_intervals(starts, stops, match_by="trial_id")

        assert_allclose(
            np.column_stack([intervals.start, intervals.stop, intervals.durations]),
            [[1.0, 4.0, 3.0], [2.0, 5.0, 3.0], [3.0, 6.0, 3.0]],
        )


# =============================================================================
# filter_by_intervals
# =============================================================================


class TestFilterByIntervals:
    def test_basic_inclusion_filter(self):
        events = _events(1.0, 3.0, 5.0, 7.0, 9.0, value=["a", "b", "c", "d", "e"])
        intervals = Intervals.from_arrays([2.0, 8.0], [4.0, 10.0])

        result = filter_by_intervals(events, intervals, include=True)

        assert_allclose(result["timestamp"].to_numpy(), [3.0, 9.0])
        assert result["value"].tolist() == ["b", "e"]

    def test_basic_exclusion_filter(self):
        intervals = Intervals.from_arrays([2.0, 8.0], [4.0, 10.0])

        result = filter_by_intervals(
            _events(1.0, 3.0, 5.0, 7.0, 9.0), intervals, include=False
        )

        assert_allclose(result["timestamp"].to_numpy(), [1.0, 5.0, 7.0])

    def test_interval_is_half_open(self):
        """An event at the interval start is kept; one at the stop is not."""
        result = filter_by_intervals(
            _events(2.0, 4.0), Intervals.from_arrays([2.0], [4.0])
        )

        assert_allclose(result["timestamp"].to_numpy(), [2.0])

    def test_overlapping_intervals(self):
        intervals = Intervals.from_arrays([2.0, 4.0], [5.0, 7.0])

        result = filter_by_intervals(_events(1.0, 3.0, 5.0, 7.0), intervals)

        # 3 and 5 are within [2, 5) or [4, 7); 7 is a stop.
        assert_allclose(result["timestamp"].to_numpy(), [3.0, 5.0])

    @pytest.mark.parametrize(
        "intervals",
        [Intervals.from_arrays([], []), Intervals.from_arrays([1.0], [1.0])],
        ids=["no rows", "zero-length row"],
    )
    def test_no_time_selects_no_event(self, intervals):
        events = _events(1.0, 2.0, 3.0)

        assert len(filter_by_intervals(events, intervals, include=True)) == 0
        assert len(filter_by_intervals(events, intervals, include=False)) == 3

    def test_empty_events(self):
        intervals = Intervals.from_arrays([1.0, 5.0], [3.0, 7.0])

        assert len(filter_by_intervals(_events(), intervals)) == 0

    def test_custom_timestamp_column(self):
        events = pd.DataFrame({"time": [1.0, 3.0, 5.0]})

        result = filter_by_intervals(
            events, Intervals.from_arrays([2.0], [4.0]), timestamp_column="time"
        )

        assert result["time"].tolist() == [3.0]

    def test_preserves_all_columns(self):
        events = _events(1.0, 3.0, 5.0, value=[10, 20, 30], label=["a", "b", "c"])

        result = filter_by_intervals(events, Intervals.from_arrays([2.0], [4.0]))

        assert list(result.columns) == ["timestamp", "value", "label"]
        assert result["value"].iloc[0] == 20
        assert result["label"].iloc[0] == "b"

    def test_preserves_index(self):
        events = pd.DataFrame({"timestamp": [1.0, 3.0, 5.0]}, index=[100, 200, 300])

        result = filter_by_intervals(events, Intervals.from_arrays([2.0], [4.0]))

        assert result.index.tolist() == [200]

    def test_single_interval_multiple_events(self):
        result = filter_by_intervals(
            _events(1.0, 2.0, 3.0, 4.0, 5.0), Intervals.from_arrays([1.5], [4.5])
        )

        assert_allclose(result["timestamp"].to_numpy(), [2.0, 3.0, 4.0])

    def test_missing_timestamp_column_raises(self):
        events = pd.DataFrame({"time": [1.0, 2.0]})

        with pytest.raises(ValueError, match=r"timestamp"):
            filter_by_intervals(events, Intervals.from_arrays([0.0], [3.0]))

    def test_nan_timestamp_is_refused_when_excluding(self):
        """Removing artifacts must not keep an event with no time as clean."""
        artifacts = Intervals.from_arrays([0.0], [10.0])

        with pytest.raises(ValueError, match="NaN"):
            filter_by_intervals(_events(5.0, np.nan, 25.0), artifacts, include=False)

    def test_zero_length_interval_holds_no_event(self):
        intervals = Intervals.from_arrays([1.0, 1.5], [1.0, 3.0])

        kept = filter_by_intervals(_events(1.0, 2.0), intervals)

        assert kept["timestamp"].tolist() == [2.0]

    def test_many_events_and_intervals_match_a_brute_force_check(self):
        rng = np.random.default_rng(0)
        times = rng.uniform(0, 100, 2000)
        starts = rng.uniform(0, 100, 50)
        stops = starts + rng.uniform(0, 3, 50)

        kept = filter_by_intervals(
            pd.DataFrame({"timestamp": times}), Intervals.from_arrays(starts, stops)
        )

        inside = ((times[:, None] >= starts) & (times[:, None] < stops)).any(axis=1)
        assert_array_equal(kept.index, np.flatnonzero(inside))

    def test_typical_artifact_rejection_use_case(self):
        spikes = pd.DataFrame({"timestamp": np.arange(0, 100, 0.5), "unit_id": 1})
        artifacts = Intervals.from_arrays([10.0, 50.0], [15.0, 55.0])

        clean = filter_by_intervals(spikes, artifacts, include=False)

        # Ten spikes in each 5 s artifact, every 0.5 s.
        assert len(clean) == len(spikes) - 20
        for start, stop in artifacts.windows:
            t = clean["timestamp"]
            assert not ((t >= start) & (t < stop)).any()

    def test_typical_trial_selection_use_case(self):
        """Licks in the trials of one condition."""
        licks = _events(5.0, 15.0, 25.0, 35.0, 45.0)
        trials = Intervals.from_arrays(
            [0.0, 20.0, 40.0], [10.0, 30.0, 50.0], outcome=["reached", None, "reached"]
        )

        assert_allclose(filter_by_intervals(licks, trials)["timestamp"], [5, 25, 45])
        reached = trials[trials.outcome == "reached"]
        assert_allclose(filter_by_intervals(licks, reached)["timestamp"], [5, 45])


# =============================================================================
# Round trip
# =============================================================================


class TestRoundTrip:
    def test_intervals_to_events_to_intervals(self):
        """Times, ids and preserved fields survive events and back."""
        original = Intervals.from_arrays(
            [1.0, 5.0, 10.0],
            [3.0, 8.0, 15.0],
            interval_ids=[7, 3, 5],
            label=["a", "b", "c"],
        )

        events = intervals_to_events(original, which="both", preserve_fields=["label"])
        start_events = events[events["boundary"] == "start"].drop(columns="boundary")
        stop_events = events[events["boundary"] == "stop"]
        recovered = events_to_intervals(start_events, stop_events)

        assert_array_equal(recovered.start, original.start)
        assert_array_equal(recovered.stop, original.stop)
        assert_array_equal(recovered.interval_ids, original.interval_ids)
        assert recovered.fields == ("label",)
        pd.testing.assert_series_equal(recovered.label, original.label)
