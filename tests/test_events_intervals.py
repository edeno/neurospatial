"""Tests for events module interval utilities."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Test intervals_to_events
# =============================================================================


class TestIntervalsToEvents:
    """Tests for intervals_to_events function."""

    def test_extract_start_times(self):
        """Test extracting start times from intervals."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 3.0, 5.0],
                "stop_time": [2.0, 4.0, 6.0],
                "label": ["a", "b", "c"],
            }
        )

        result = intervals_to_events(intervals, which="start")

        assert "timestamp" in result.columns
        assert len(result) == 3
        assert_allclose(result["timestamp"].values, [1.0, 3.0, 5.0])

    def test_extract_stop_times(self):
        """Test extracting stop times from intervals."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 3.0, 5.0],
                "stop_time": [2.0, 4.0, 6.0],
            }
        )

        result = intervals_to_events(intervals, which="stop")

        assert "timestamp" in result.columns
        assert len(result) == 3
        assert_allclose(result["timestamp"].values, [2.0, 4.0, 6.0])

    def test_extract_both_times(self):
        """Test extracting both start and stop times."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "stop_time": [2.0, 6.0],
            }
        )

        result = intervals_to_events(intervals, which="both")

        assert "timestamp" in result.columns
        assert "boundary" in result.columns
        assert len(result) == 4  # 2 starts + 2 stops

        # Should be sorted by timestamp
        timestamps = result["timestamp"].values
        assert_allclose(timestamps, [1.0, 2.0, 5.0, 6.0])

        # Check boundary labels
        boundaries = result["boundary"].values
        assert list(boundaries) == ["start", "stop", "start", "stop"]

    def test_custom_column_names(self):
        """Test custom start/stop column names."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "begin": [1.0, 3.0],
                "end": [2.0, 4.0],
            }
        )

        result = intervals_to_events(
            intervals,
            which="start",
            start_column="begin",
            stop_column="end",
        )

        assert len(result) == 2
        assert_allclose(result["timestamp"].values, [1.0, 3.0])

    def test_preserve_columns(self):
        """Test preserving additional columns from intervals."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 3.0],
                "stop_time": [2.0, 4.0],
                "trial_id": [1, 2],
                "condition": ["A", "B"],
            }
        )

        result = intervals_to_events(
            intervals,
            which="start",
            preserve_columns=["trial_id", "condition"],
        )

        assert "trial_id" in result.columns
        assert "condition" in result.columns
        assert list(result["trial_id"]) == [1, 2]
        assert list(result["condition"]) == ["A", "B"]

    def test_preserve_columns_with_both(self):
        """Test preserving columns when extracting both boundaries."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "stop_time": [2.0, 6.0],
                "trial_id": [1, 2],
            }
        )

        result = intervals_to_events(
            intervals,
            which="both",
            preserve_columns=["trial_id"],
        )

        assert len(result) == 4
        assert "trial_id" in result.columns
        # Each interval contributes 2 events (start and stop)
        # trial_id should be duplicated for each boundary
        assert list(result["trial_id"]) == [1, 1, 2, 2]

    def test_empty_intervals(self):
        """Test empty intervals DataFrame returns empty result."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [],
                "stop_time": [],
            }
        )

        result = intervals_to_events(intervals, which="start")

        assert len(result) == 0
        assert "timestamp" in result.columns

    def test_single_interval(self):
        """Test with single interval."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [5.0],
                "stop_time": [10.0],
            }
        )

        result = intervals_to_events(intervals, which="both")

        assert len(result) == 2
        assert_allclose(result["timestamp"].values, [5.0, 10.0])
        assert list(result["boundary"]) == ["start", "stop"]

    def test_overlapping_intervals(self):
        """Test with overlapping intervals."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],  # Overlapping
                "stop_time": [3.0, 4.0],
            }
        )

        result = intervals_to_events(intervals, which="both")

        assert len(result) == 4
        # Should be sorted by timestamp
        assert_allclose(result["timestamp"].values, [1.0, 2.0, 3.0, 4.0])

    def test_missing_start_column_raises(self):
        """Test missing start column raises ValueError."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "stop_time": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match=r"start_time"):
            intervals_to_events(intervals, which="start")

    def test_missing_stop_column_raises(self):
        """Test missing stop column raises ValueError."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match=r"stop_time"):
            intervals_to_events(intervals, which="stop")

    def test_invalid_which_raises(self):
        """Test invalid 'which' parameter raises ValueError."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0],
                "stop_time": [2.0],
            }
        )

        with pytest.raises(ValueError, match=r"which.*must be"):
            intervals_to_events(intervals, which="invalid")

    def test_not_dataframe_raises(self):
        """Test non-DataFrame input raises TypeError."""
        from neurospatial.events.intervals import intervals_to_events

        with pytest.raises(TypeError, match=r"DataFrame"):
            intervals_to_events([1.0, 2.0], which="start")

    def test_preserve_columns_not_found_raises(self):
        """Test preserving non-existent column raises ValueError."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0],
                "stop_time": [2.0],
            }
        )

        with pytest.raises(ValueError, match=r"nonexistent"):
            intervals_to_events(
                intervals,
                which="start",
                preserve_columns=["nonexistent"],
            )

    def test_output_is_new_dataframe(self):
        """Test that output is a new DataFrame (not modifying input)."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 2.0],
                "stop_time": [3.0, 4.0],
            }
        )
        original_cols = list(intervals.columns)

        result = intervals_to_events(intervals, which="start")

        # Original should be unchanged
        assert list(intervals.columns) == original_cols
        # Result should be different object
        assert result is not intervals

    def test_index_is_reset(self):
        """Test that output has reset index."""
        from neurospatial.events.intervals import intervals_to_events

        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 2.0, 3.0],
                "stop_time": [2.0, 3.0, 4.0],
            },
            index=[10, 20, 30],
        )  # Custom index

        result = intervals_to_events(intervals, which="start")

        # Result should have default RangeIndex
        assert list(result.index) == [0, 1, 2]

    def test_typical_trial_use_case(self):
        """Test typical use case: converting trials to trial start events."""
        from neurospatial.events.intervals import intervals_to_events

        # NWB-style trials table
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 10.0, 20.0, 30.0],
                "stop_time": [8.0, 18.0, 28.0, 38.0],
                "trial_type": ["go", "nogo", "go", "nogo"],
                "correct": [True, True, False, True],
            }
        )

        # Get trial start events for PSTH alignment
        trial_starts = intervals_to_events(
            trials,
            which="start",
            preserve_columns=["trial_type", "correct"],
        )

        assert len(trial_starts) == 4
        assert "timestamp" in trial_starts.columns
        assert "trial_type" in trial_starts.columns
        assert "correct" in trial_starts.columns

        # Filter to just 'go' trials
        go_starts = trial_starts[trial_starts["trial_type"] == "go"]
        assert len(go_starts) == 2
        assert_allclose(go_starts["timestamp"].values, [0.0, 20.0])


# =============================================================================
# Test events_to_intervals
# =============================================================================


class TestEventsToIntervals:
    """Tests for events_to_intervals function."""

    def test_basic_sequential_pairing(self):
        """Test basic sequential pairing of start and stop events."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [1.0, 5.0, 10.0]})
        stop_events = pd.DataFrame({"timestamp": [3.0, 8.0, 15.0]})

        result = events_to_intervals(start_events, stop_events)

        assert "start_time" in result.columns
        assert "stop_time" in result.columns
        assert "duration" in result.columns
        assert len(result) == 3

        assert_allclose(result["start_time"].values, [1.0, 5.0, 10.0])
        assert_allclose(result["stop_time"].values, [3.0, 8.0, 15.0])
        assert_allclose(result["duration"].values, [2.0, 3.0, 5.0])

    def test_match_by_column(self):
        """Test matching by column value."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "trial_id": [1, 2, 3],
            }
        )
        stop_events = pd.DataFrame(
            {
                "timestamp": [5.0, 4.0, 6.0],  # Out of order
                "trial_id": [2, 1, 3],  # But matching by trial_id
            }
        )

        result = events_to_intervals(start_events, stop_events, match_by="trial_id")

        assert len(result) == 3
        # Should be sorted by start time
        assert_allclose(result["start_time"].values, [1.0, 2.0, 3.0])
        assert_allclose(result["stop_time"].values, [4.0, 5.0, 6.0])
        assert_allclose(result["duration"].values, [3.0, 3.0, 3.0])

    def test_max_duration_filter(self):
        """Test filtering by maximum duration."""
        from neurospatial.events.intervals import events_to_intervals

        # Note: sequential pairing pairs sorted starts with sorted stops
        # Starts sorted: [1.0, 5.0, 10.0]
        # Stops sorted: [3.0, 8.0, 20.0]
        # Pairs: (1.0, 3.0)=2s, (5.0, 8.0)=3s, (10.0, 20.0)=10s
        start_events = pd.DataFrame({"timestamp": [1.0, 5.0, 10.0]})
        stop_events = pd.DataFrame({"timestamp": [3.0, 20.0, 8.0]})

        result = events_to_intervals(start_events, stop_events, max_duration=5.0)

        # Only intervals with duration <= 5.0 should remain
        assert len(result) == 2
        assert_allclose(result["start_time"].values, [1.0, 5.0])
        assert_allclose(result["stop_time"].values, [3.0, 8.0])

    def test_empty_events(self):
        """Test empty events returns empty intervals."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": []})
        stop_events = pd.DataFrame({"timestamp": []})

        result = events_to_intervals(start_events, stop_events)

        assert len(result) == 0
        assert "start_time" in result.columns
        assert "stop_time" in result.columns
        assert "duration" in result.columns

    def test_single_interval(self):
        """Test with single start/stop pair."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [5.0]})
        stop_events = pd.DataFrame({"timestamp": [10.0]})

        result = events_to_intervals(start_events, stop_events)

        assert len(result) == 1
        assert result["start_time"].iloc[0] == 5.0
        assert result["stop_time"].iloc[0] == 10.0
        assert result["duration"].iloc[0] == 5.0

    def test_mismatched_counts_raises(self):
        """Test mismatched event counts raises ValueError."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0]})
        stop_events = pd.DataFrame({"timestamp": [4.0, 5.0]})

        with pytest.raises(ValueError, match=r"count.*match"):
            events_to_intervals(start_events, stop_events)

    def test_match_by_missing_column_raises(self):
        """Test missing match_by column raises ValueError."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [1.0]})
        stop_events = pd.DataFrame({"timestamp": [2.0]})

        with pytest.raises(ValueError, match=r"trial_id"):
            events_to_intervals(start_events, stop_events, match_by="trial_id")

    def test_match_by_unmatched_values_raises(self):
        """Test unmatched values in match_by column raises ValueError."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "trial_id": [1, 2],
            }
        )
        stop_events = pd.DataFrame(
            {
                "timestamp": [3.0, 4.0],
                "trial_id": [1, 3],  # trial_id=3 has no matching start
            }
        )

        with pytest.raises(ValueError, match=r"unmatched"):
            events_to_intervals(start_events, stop_events, match_by="trial_id")

    def test_negative_duration_warning(self):
        """Test warning when stop time is before start time."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [5.0]})
        stop_events = pd.DataFrame({"timestamp": [3.0]})  # Before start!

        with pytest.warns(UserWarning, match=r"negative.*duration"):
            result = events_to_intervals(start_events, stop_events)

        # Should still return the interval (user might want to filter later)
        assert len(result) == 1
        assert result["duration"].iloc[0] == -2.0

    def test_preserves_additional_columns(self):
        """Test that additional columns from start_events are preserved."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame(
            {
                "timestamp": [1.0, 5.0],
                "condition": ["A", "B"],
            }
        )
        stop_events = pd.DataFrame({"timestamp": [3.0, 8.0]})

        result = events_to_intervals(start_events, stop_events)

        assert "condition" in result.columns
        assert list(result["condition"]) == ["A", "B"]

    def test_output_sorted_by_start_time(self):
        """Test output is sorted by start time."""
        from neurospatial.events.intervals import events_to_intervals

        start_events = pd.DataFrame({"timestamp": [10.0, 1.0, 5.0]})
        stop_events = pd.DataFrame({"timestamp": [12.0, 3.0, 8.0]})

        result = events_to_intervals(start_events, stop_events)

        assert_allclose(result["start_time"].values, [1.0, 5.0, 10.0])

    def test_typical_zone_crossing_use_case(self):
        """Test typical use case: zone entry/exit to dwell intervals."""
        from neurospatial.events.intervals import events_to_intervals

        # Zone crossings
        entries = pd.DataFrame(
            {
                "timestamp": [10.0, 30.0, 50.0],
                "region": ["goal", "goal", "goal"],
            }
        )
        exits = pd.DataFrame(
            {
                "timestamp": [15.0, 35.0, 58.0],
                "region": ["goal", "goal", "goal"],
            }
        )

        # Convert to dwell intervals
        dwell = events_to_intervals(entries, exits)

        assert len(dwell) == 3
        assert_allclose(dwell["duration"].values, [5.0, 5.0, 8.0])
        assert dwell["duration"].sum() == 18.0  # Total dwell time


# =============================================================================
# Test filter_by_intervals
# =============================================================================


class TestFilterByIntervals:
    """Tests for filter_by_intervals function."""

    def test_basic_inclusion_filter(self):
        """Test basic filtering to events within intervals."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 3.0, 5.0, 7.0, 9.0],
                "value": ["a", "b", "c", "d", "e"],
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0, 8.0],
                "stop_time": [4.0, 10.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert len(result) == 2  # Events at 3.0 and 9.0
        assert_allclose(result["timestamp"].values, [3.0, 9.0])
        assert list(result["value"]) == ["b", "e"]

    def test_basic_exclusion_filter(self):
        """Test filtering to events outside intervals."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 3.0, 5.0, 7.0, 9.0],
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0, 8.0],
                "stop_time": [4.0, 10.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=False)

        assert len(result) == 3  # Events at 1.0, 5.0, 7.0
        assert_allclose(result["timestamp"].values, [1.0, 5.0, 7.0])

    def test_event_at_interval_boundary_included(self):
        """Test events at exact interval boundaries are included."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [2.0, 4.0],  # At start and stop boundaries
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0],
                "stop_time": [4.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert len(result) == 2  # Both boundary events included

    def test_overlapping_intervals(self):
        """Test filtering with overlapping intervals."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 3.0, 5.0, 7.0],
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0, 4.0],  # Overlapping: 2-5 and 4-7
                "stop_time": [5.0, 7.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        # Events at 3.0, 5.0, 7.0 are within at least one interval
        assert len(result) == 3
        assert_allclose(result["timestamp"].values, [3.0, 5.0, 7.0])

    def test_empty_intervals_with_include(self):
        """Test empty intervals with include=True returns empty."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0]})
        intervals = pd.DataFrame({"start_time": [], "stop_time": []})

        result = filter_by_intervals(events, intervals, include=True)

        assert len(result) == 0

    def test_empty_intervals_with_exclude(self):
        """Test empty intervals with include=False returns all events."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0]})
        intervals = pd.DataFrame({"start_time": [], "stop_time": []})

        result = filter_by_intervals(events, intervals, include=False)

        assert len(result) == 3

    def test_empty_events(self):
        """Test empty events returns empty result."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"timestamp": []})
        intervals = pd.DataFrame(
            {
                "start_time": [1.0, 5.0],
                "stop_time": [3.0, 7.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert len(result) == 0

    def test_custom_column_names(self):
        """Test custom column names."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"time": [1.0, 3.0, 5.0]})
        intervals = pd.DataFrame(
            {
                "begin": [2.0],
                "end": [4.0],
            }
        )

        result = filter_by_intervals(
            events,
            intervals,
            include=True,
            timestamp_column="time",
            start_column="begin",
            stop_column="end",
        )

        assert len(result) == 1
        assert result["time"].iloc[0] == 3.0

    def test_preserves_all_columns(self):
        """Test all event columns are preserved."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 3.0, 5.0],
                "value": [10, 20, 30],
                "label": ["a", "b", "c"],
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0],
                "stop_time": [4.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert list(result.columns) == ["timestamp", "value", "label"]
        assert result["value"].iloc[0] == 20
        assert result["label"].iloc[0] == "b"

    def test_preserves_index(self):
        """Test original index is preserved."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 3.0, 5.0],
            },
            index=[100, 200, 300],
        )
        intervals = pd.DataFrame(
            {
                "start_time": [2.0],
                "stop_time": [4.0],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert result.index.tolist() == [200]

    def test_single_interval_multiple_events(self):
        """Test single interval containing multiple events."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        intervals = pd.DataFrame(
            {
                "start_time": [1.5],
                "stop_time": [4.5],
            }
        )

        result = filter_by_intervals(events, intervals, include=True)

        assert len(result) == 3  # Events at 2.0, 3.0, 4.0
        assert_allclose(result["timestamp"].values, [2.0, 3.0, 4.0])

    def test_missing_timestamp_column_raises(self):
        """Test missing timestamp column raises ValueError."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"time": [1.0, 2.0]})
        intervals = pd.DataFrame(
            {
                "start_time": [0.0],
                "stop_time": [3.0],
            }
        )

        with pytest.raises(ValueError, match=r"timestamp"):
            filter_by_intervals(events, intervals)

    def test_missing_interval_columns_raises(self):
        """Test missing interval columns raises ValueError."""
        from neurospatial.events.intervals import filter_by_intervals

        events = pd.DataFrame({"timestamp": [1.0, 2.0]})
        intervals = pd.DataFrame({"start": [0.0], "end": [3.0]})

        with pytest.raises(ValueError, match=r"start_time"):
            filter_by_intervals(events, intervals)

    def test_typical_artifact_rejection_use_case(self):
        """Test typical use case: rejecting events during artifact periods."""
        from neurospatial.events.intervals import filter_by_intervals

        # Neural events
        spikes = pd.DataFrame(
            {
                "timestamp": np.arange(0, 100, 0.5),  # Spikes every 0.5s
                "unit_id": [1] * 200,
            }
        )

        # Artifact periods to exclude
        artifacts = pd.DataFrame(
            {
                "start_time": [10.0, 50.0],
                "stop_time": [15.0, 55.0],
            }
        )

        # Remove spikes during artifacts
        clean_spikes = filter_by_intervals(spikes, artifacts, include=False)

        # Should have removed ~10 spikes from each artifact period
        assert len(clean_spikes) < len(spikes)
        # Check no spikes in artifact periods
        for _, row in artifacts.iterrows():
            mask = (clean_spikes["timestamp"] >= row["start_time"]) & (
                clean_spikes["timestamp"] <= row["stop_time"]
            )
            assert not mask.any()

    def test_typical_trial_selection_use_case(self):
        """Test typical use case: selecting events within trials."""
        from neurospatial.events.intervals import filter_by_intervals

        # Lick events
        licks = pd.DataFrame(
            {
                "timestamp": [5.0, 15.0, 25.0, 35.0, 45.0],
            }
        )

        # Trial intervals
        trials = pd.DataFrame(
            {
                "start_time": [0.0, 20.0, 40.0],
                "stop_time": [10.0, 30.0, 50.0],
            }
        )

        # Get licks within trials only
        trial_licks = filter_by_intervals(licks, trials, include=True)

        assert len(trial_licks) == 3  # Licks at 5.0, 25.0, 45.0
        assert_allclose(trial_licks["timestamp"].values, [5.0, 25.0, 45.0])


# =============================================================================
# Test round-trip conversions
# =============================================================================


class TestRoundTrip:
    """Tests for round-trip conversions between events and intervals."""

    def test_intervals_to_events_to_intervals(self):
        """Test round-trip: intervals → events → intervals."""
        from neurospatial.events.intervals import (
            events_to_intervals,
            intervals_to_events,
        )

        # Original intervals
        original = pd.DataFrame(
            {
                "start_time": [1.0, 5.0, 10.0],
                "stop_time": [3.0, 8.0, 15.0],
            }
        )

        # Convert to events and back
        events = intervals_to_events(original, which="both")
        start_events = events[events["boundary"] == "start"].copy()
        stop_events = events[events["boundary"] == "stop"].copy()

        # Reset index for pairing
        start_events = start_events.reset_index(drop=True)
        stop_events = stop_events.reset_index(drop=True)

        recovered = events_to_intervals(start_events, stop_events)

        # Should match original
        assert_allclose(
            recovered["start_time"].values,
            original["start_time"].values,
        )
        assert_allclose(
            recovered["stop_time"].values,
            original["stop_time"].values,
        )
