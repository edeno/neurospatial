"""
Tests for NWB events and intervals reading functions.

Tests the read_events() function for reading EventsTable data from NWB files,
and the read_intervals() function for reading TimeIntervals (trials, epochs, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# pynwb is required for all tests
pynwb = pytest.importorskip("pynwb")

from ._nwb_helpers import create_roundtrip_nwb  # noqa: E402

# Check if ndx_events is available for EventsTable tests
try:
    import ndx_events  # noqa: F401

    HAS_NDX_EVENTS = True
except ImportError:
    HAS_NDX_EVENTS = False


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestReadEvents:
    """Tests for read_events() function (requires ndx-events)."""

    def test_basic_events_reading(self, sample_nwb_with_events):
        """Test reading events data from NWB file."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, table_name="laps")

        # Check return type
        assert isinstance(events, pd.DataFrame)

        # Check shape (20 lap events from fixture)
        assert len(events) == 20

        # Check timestamp column exists
        assert "timestamp" in events.columns

        # Check timestamps are valid
        assert events["timestamp"].dtype == np.float64
        assert all(np.isfinite(events["timestamp"]))
        assert all(events["timestamp"] >= 0)

    def test_events_data_matches_original(self, sample_nwb_with_events):
        """Test that read data matches the original data in the NWB file."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, table_name="laps")

        # Get original data directly from NWB
        original_table = sample_nwb_with_events.processing["behavior"]["laps"]

        # Check timestamps match
        np.testing.assert_array_almost_equal(
            events["timestamp"].values, original_table["timestamp"][:]
        )

    def test_with_explicit_processing_module(self, empty_nwb):
        """Test reading with explicit processing_module parameter."""
        from ndx_events import EventsTable

        from neurospatial.io.nwb import read_events

        nwbfile = empty_nwb

        # Create EventsTable in a custom module
        custom_module = nwbfile.create_processing_module(
            name="analysis", description="Analysis results"
        )
        events_table = EventsTable(name="custom_events", description="Custom events")
        events_table.add_row(timestamp=1.5)
        events_table.add_row(timestamp=3.0)
        custom_module.add(events_table)

        # Read with explicit module
        events = read_events(
            nwbfile, table_name="custom_events", processing_module="analysis"
        )

        assert len(events) == 2
        assert events["timestamp"].iloc[0] == pytest.approx(1.5)
        assert events["timestamp"].iloc[1] == pytest.approx(3.0)

    def test_error_when_table_not_found(self, sample_nwb_with_events):
        """Test KeyError when EventsTable not found."""
        from neurospatial.io.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_table"):
            read_events(sample_nwb_with_events, table_name="nonexistent_table")

    def test_error_when_module_not_found(self, empty_nwb):
        """Test KeyError when processing module not found."""
        from neurospatial.io.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_module"):
            read_events(
                empty_nwb, table_name="events", processing_module="nonexistent_module"
            )

    def test_dataframe_output_with_timestamp_column(self, sample_nwb_with_events):
        """Test DataFrame output includes timestamp column."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, table_name="laps")

        # Timestamp must be a column
        assert "timestamp" in events.columns

        # Timestamps should be sorted (as created in fixture)
        timestamps = events["timestamp"].values
        assert all(timestamps[:-1] <= timestamps[1:])

    def test_additional_columns_preserved(self, sample_nwb_with_events):
        """Test that additional columns from EventsTable are preserved."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, table_name="laps")

        # Fixture adds 'direction' and 'duration' columns
        assert "direction" in events.columns
        assert "duration" in events.columns

        # Check direction values (0 or 1 from fixture)
        assert all(events["direction"].isin([0, 1]))

        # Check duration values are positive
        assert all(events["duration"] > 0)

    def test_empty_events_table(self, empty_nwb):
        """Test reading an empty EventsTable."""
        from ndx_events import EventsTable

        from neurospatial.io.nwb import read_events

        nwbfile = empty_nwb

        # Create empty EventsTable
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavioral data"
        )
        events_table = EventsTable(name="empty_events", description="Empty table")
        behavior_module.add(events_table)

        # Read empty table
        events = read_events(nwbfile, table_name="empty_events")

        assert isinstance(events, pd.DataFrame)
        assert len(events) == 0
        assert "timestamp" in events.columns

    def test_events_with_string_columns(self, empty_nwb):
        """Test reading EventsTable with string columns."""
        from ndx_events import EventsTable

        from neurospatial.io.nwb import read_events

        nwbfile = empty_nwb

        # Create EventsTable with string column
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavioral data"
        )
        events_table = EventsTable(name="region_events", description="Region events")
        events_table.add_column(name="region", description="Region name")
        events_table.add_column(name="event_type", description="Event type")
        events_table.add_row(timestamp=1.0, region="start", event_type="enter")
        events_table.add_row(timestamp=2.5, region="goal", event_type="enter")
        events_table.add_row(timestamp=3.0, region="goal", event_type="exit")
        behavior_module.add(events_table)

        # Read table
        events = read_events(nwbfile, table_name="region_events")

        assert len(events) == 3
        assert "region" in events.columns
        assert "event_type" in events.columns
        assert list(events["region"]) == ["start", "goal", "goal"]
        assert list(events["event_type"]) == ["enter", "enter", "exit"]

    def test_error_when_table_is_wrong_type(self, empty_nwb):
        """Test TypeError when trying to read non-EventsTable with read_events."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_events

        nwbfile = empty_nwb

        # Create a Position container (not EventsTable)
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavioral data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((10, 2)),
                timestamps=np.arange(10) / 10.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position)

        # Try to read Position with read_events - should raise TypeError
        with pytest.raises(TypeError, match="is not an EventsTable"):
            read_events(nwbfile, table_name="Position")


class TestReadIntervals:
    """Tests for read_intervals() function (built-in NWB TimeIntervals)."""

    def test_read_trials(self, empty_nwb):
        """Test reading trials table from NWB file."""
        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Add trials to NWB file
        nwbfile.add_trial_column(name="trial_type", description="Type of trial")
        nwbfile.add_trial(start_time=0.0, stop_time=1.0, trial_type="go")
        nwbfile.add_trial(start_time=2.0, stop_time=3.5, trial_type="nogo")
        nwbfile.add_trial(start_time=5.0, stop_time=6.0, trial_type="go")

        # Read trials
        trials = read_intervals(nwbfile, interval_name="trials")

        assert isinstance(trials, pd.DataFrame)
        assert len(trials) == 3
        assert "start_time" in trials.columns
        assert "stop_time" in trials.columns
        assert "trial_type" in trials.columns

    def test_read_epochs(self, empty_nwb):
        """Test reading epochs table from NWB file."""
        from pynwb.epoch import TimeIntervals

        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Create epochs table
        epochs = TimeIntervals(name="epochs", description="Experimental epochs")
        epochs.add_column(name="epoch_type", description="Type of epoch")
        epochs.add_row(start_time=0.0, stop_time=60.0, epoch_type="baseline")
        epochs.add_row(start_time=60.0, stop_time=180.0, epoch_type="stimulus")
        epochs.add_row(start_time=180.0, stop_time=240.0, epoch_type="recovery")
        nwbfile.add_time_intervals(epochs)

        # Read epochs
        result = read_intervals(nwbfile, interval_name="epochs")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "start_time" in result.columns
        assert "stop_time" in result.columns
        assert "epoch_type" in result.columns
        assert list(result["epoch_type"]) == ["baseline", "stimulus", "recovery"]

    def test_read_custom_intervals(self, empty_nwb):
        """Test reading custom intervals table from NWB file."""
        from pynwb.epoch import TimeIntervals

        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Create custom intervals
        laps = TimeIntervals(name="laps", description="Lap intervals")
        laps.add_column(name="direction", description="Lap direction")
        laps.add_row(start_time=10.0, stop_time=25.0, direction="outbound")
        laps.add_row(start_time=30.0, stop_time=42.0, direction="inbound")
        nwbfile.add_time_intervals(laps)

        # Read custom intervals
        result = read_intervals(nwbfile, interval_name="laps")

        assert len(result) == 2
        assert result["start_time"].iloc[0] == pytest.approx(10.0)
        assert result["stop_time"].iloc[0] == pytest.approx(25.0)
        assert result["direction"].iloc[0] == "outbound"

    def test_start_stop_time_columns(self, empty_nwb):
        """Test that start_time and stop_time columns are present."""
        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Add minimal trials
        nwbfile.add_trial(start_time=0.0, stop_time=1.0)

        trials = read_intervals(nwbfile, interval_name="trials")

        # Required columns must be present
        assert "start_time" in trials.columns
        assert "stop_time" in trials.columns

        # Check types
        assert trials["start_time"].dtype == np.float64
        assert trials["stop_time"].dtype == np.float64

        # Check stop > start
        assert all(trials["stop_time"] > trials["start_time"])

    def test_error_when_interval_not_found(self, empty_nwb):
        """Test KeyError when interval table not found."""
        from neurospatial.io.nwb import read_intervals

        with pytest.raises(KeyError, match="nonexistent"):
            read_intervals(empty_nwb, interval_name="nonexistent")

    def test_additional_columns_preserved(self, empty_nwb):
        """Test that additional columns from TimeIntervals are preserved."""
        from pynwb.epoch import TimeIntervals

        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Create intervals with multiple custom columns
        intervals = TimeIntervals(
            name="behavioral_states", description="Behavioral states"
        )
        intervals.add_column(name="state", description="Behavioral state name")
        intervals.add_column(name="confidence", description="State confidence score")
        intervals.add_row(start_time=0.0, stop_time=10.0, state="rest", confidence=0.95)
        intervals.add_row(start_time=10.0, stop_time=30.0, state="run", confidence=0.88)
        nwbfile.add_time_intervals(intervals)

        result = read_intervals(nwbfile, interval_name="behavioral_states")

        assert "state" in result.columns
        assert "confidence" in result.columns
        assert list(result["state"]) == ["rest", "run"]
        assert result["confidence"].iloc[0] == pytest.approx(0.95)

    def test_empty_intervals_table(self, empty_nwb):
        """Test reading an empty TimeIntervals table."""
        from pynwb.epoch import TimeIntervals

        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Create empty intervals table
        empty_intervals = TimeIntervals(
            name="empty_table", description="Empty intervals"
        )
        nwbfile.add_time_intervals(empty_intervals)

        result = read_intervals(nwbfile, interval_name="empty_table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "start_time" in result.columns
        assert "stop_time" in result.columns

    def test_intervals_data_matches_original(self, empty_nwb):
        """Test that read data matches the original data in the NWB file."""
        from neurospatial.io.nwb import read_intervals

        nwbfile = empty_nwb

        # Add trials with specific values
        nwbfile.add_trial(start_time=1.5, stop_time=3.5)
        nwbfile.add_trial(start_time=5.0, stop_time=8.0)

        result = read_intervals(nwbfile, interval_name="trials")

        # Get original data
        original = nwbfile.trials

        np.testing.assert_array_almost_equal(
            result["start_time"].values, original["start_time"][:]
        )
        np.testing.assert_array_almost_equal(
            result["stop_time"].values, original["stop_time"][:]
        )


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteRegionCrossings:
    """Tests for write_region_crossings() function (requires ndx-events)."""

    def test_basic_crossing_times_writing(self, empty_nwb):
        """Test writing basic region crossing times to NWB file."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, 2.5, 5.0, 8.2])
        region_names = np.array(["start", "goal", "start", "goal"])
        event_types = np.array(["enter", "enter", "exit", "exit"])

        # Write crossings
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        # Verify EventsTable was created
        assert "behavior" in nwbfile.processing
        assert "region_crossings" in nwbfile.processing["behavior"].data_interfaces

        # Verify data
        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert len(crossings_table) == 4

        # Check timestamps
        np.testing.assert_array_almost_equal(
            crossings_table["timestamp"][:], crossing_times
        )

    def test_region_name_column(self, empty_nwb):
        """Test that region name column is properly stored."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, 2.0, 3.0])
        region_names = np.array(["start", "goal", "reward_zone"])
        event_types = np.array(["enter", "enter", "enter"])

        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        crossings_table = nwbfile.processing["behavior"]["region_crossings"]

        # Check region column exists
        assert "region" in crossings_table.colnames

        # Check region values
        assert list(crossings_table["region"][:]) == ["start", "goal", "reward_zone"]

    def test_event_type_column(self, empty_nwb):
        """Test that event_type column (enter/exit) is properly stored."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, 2.0, 3.0, 4.0])
        region_names = np.array(["goal", "goal", "start", "start"])
        event_types = np.array(["enter", "exit", "enter", "exit"])

        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        crossings_table = nwbfile.processing["behavior"]["region_crossings"]

        # Check event_type column exists
        assert "event_type" in crossings_table.colnames

        # Check event_type values
        assert list(crossings_table["event_type"][:]) == [
            "enter",
            "exit",
            "enter",
            "exit",
        ]

    def test_events_table_creation_in_behavior(self, empty_nwb):
        """Test EventsTable is created in processing/behavior/ module."""
        from ndx_events import EventsTable

        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])

        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        # Verify processing/behavior exists
        assert "behavior" in nwbfile.processing

        # Verify region_crossings is an EventsTable
        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert isinstance(crossings_table, EventsTable)

    def test_custom_description(self, empty_nwb):
        """Test custom description parameter."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])

        write_region_crossings(
            nwbfile,
            crossing_times,
            region_names,
            event_types,
            description="Custom crossing description",
        )

        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert crossings_table.description == "Custom crossing description"

    def test_default_description(self, empty_nwb):
        """Test default description is used."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])

        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert "Region crossing events" in crossings_table.description

    def test_custom_name(self, empty_nwb):
        """Test writing crossings with custom table name."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])

        write_region_crossings(
            nwbfile,
            crossing_times,
            region_names,
            event_types,
            name="maze_crossings",
        )

        # Verify custom name used
        assert "maze_crossings" in nwbfile.processing["behavior"].data_interfaces
        assert "region_crossings" not in nwbfile.processing["behavior"].data_interfaces

    def test_duplicate_name_error(self, empty_nwb):
        """Test error when crossings table already exists without overwrite."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])

        # Write crossings first time
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        # Try to write again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_overwrite_replaces_existing(self, empty_nwb):
        """Test overwrite=True replaces existing crossings table."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        original_times = np.array([1.0])
        original_regions = np.array(["start"])
        original_types = np.array(["enter"])

        new_times = np.array([5.0, 10.0, 15.0])
        new_regions = np.array(["goal", "start", "goal"])
        new_types = np.array(["enter", "exit", "exit"])

        # Write original crossings
        write_region_crossings(
            nwbfile, original_times, original_regions, original_types
        )

        # Overwrite with new data
        write_region_crossings(
            nwbfile, new_times, new_regions, new_types, overwrite=True
        )

        # Verify new data
        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert len(crossings_table) == 3
        np.testing.assert_array_almost_equal(crossings_table["timestamp"][:], new_times)

    def test_empty_crossing_times(self, empty_nwb):
        """Test writing empty crossing times array."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([])
        region_names = np.array([])
        event_types = np.array([])

        # Write empty crossings - should succeed
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        crossings_table = nwbfile.processing["behavior"]["region_crossings"]
        assert len(crossings_table) == 0

    def test_crossing_times_must_be_1d(self, empty_nwb):
        """Test error when crossing_times is not 1D."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        region_names = np.array(["goal", "goal"])
        event_types = np.array(["enter", "exit"])

        with pytest.raises(ValueError, match=r"1D|one-dimensional"):
            write_region_crossings(
                nwbfile, crossing_times_2d, region_names, event_types
            )

    def test_region_names_length_mismatch(self, empty_nwb):
        """Test error when region_names length doesn't match crossing_times."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, 2.0, 3.0])
        region_names = np.array(["goal", "start"])  # Wrong length
        event_types = np.array(["enter", "exit", "enter"])

        with pytest.raises(ValueError, match=r"length|shape"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_event_types_length_mismatch(self, empty_nwb):
        """Test error when event_types length doesn't match crossing_times."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, 2.0, 3.0])
        region_names = np.array(["goal", "start", "goal"])
        event_types = np.array(["enter", "exit"])  # Wrong length

        with pytest.raises(ValueError, match=r"length|shape"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_crossing_times_with_nan_raises_error(self, empty_nwb):
        """Test error when crossing_times contains NaN values."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, np.nan, 3.0])
        region_names = np.array(["goal", "start", "goal"])
        event_types = np.array(["enter", "exit", "enter"])

        with pytest.raises(ValueError, match=r"non-finite|NaN"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_crossing_times_with_inf_raises_error(self, empty_nwb):
        """Test error when crossing_times contains Inf values."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.0, np.inf, 3.0])
        region_names = np.array(["goal", "start", "goal"])
        event_types = np.array(["enter", "exit", "enter"])

        with pytest.raises(ValueError, match=r"non-finite|Inf"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_crossing_times_negative_raises_error(self, empty_nwb):
        """Test error when crossing_times contains negative timestamps."""
        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([-1.0, 1.0, 3.0])
        region_names = np.array(["goal", "start", "goal"])
        event_types = np.array(["enter", "exit", "enter"])

        with pytest.raises(ValueError, match="negative"):
            write_region_crossings(nwbfile, crossing_times, region_names, event_types)

    def test_data_integrity(self, empty_nwb):
        """Test data integrity via round-trip with read_events()."""
        from neurospatial.io.nwb import read_events, write_region_crossings

        nwbfile = empty_nwb
        crossing_times = np.array([1.5, 3.2, 7.8, 12.1])
        region_names = np.array(["start", "goal", "goal", "start"])
        event_types = np.array(["enter", "enter", "exit", "exit"])

        # Write crossings
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        # Read back using read_events
        events = read_events(nwbfile, table_name="region_crossings")

        # Verify data matches
        np.testing.assert_array_almost_equal(events["timestamp"].values, crossing_times)
        assert list(events["region"]) == list(region_names)
        assert list(events["event_type"]) == list(event_types)

    def test_behavior_module_reuse(self, empty_nwb):
        """Test that existing behavior module is reused."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import write_region_crossings

        nwbfile = empty_nwb

        # Create behavior module with Position data first
        behavior = nwbfile.create_processing_module(
            name="behavior", description="Behavioral data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((10, 2)),
                timestamps=np.arange(10) / 10.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior.add(position)

        # Write crossings - should reuse existing module
        crossing_times = np.array([1.0])
        region_names = np.array(["goal"])
        event_types = np.array(["enter"])
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        # Verify both Position and region_crossings exist in same module
        assert "Position" in nwbfile.processing["behavior"].data_interfaces
        assert "region_crossings" in nwbfile.processing["behavior"].data_interfaces


@pytest.mark.slow
@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteRegionCrossingsDiskRoundTrip:
    """write_region_crossings survives a real disk write -> reopen cycle."""

    def test_region_crossings_survive_disk_roundtrip(self, tmp_path):
        """Region-crossing events survive write -> close -> reopen."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_events, write_region_crossings

        # 8 enter/exit events across 2 regions.
        crossing_times = np.array([1.0, 2.5, 5.0, 8.2, 9.0, 11.0, 12.5, 14.0])
        region_names = np.array(
            ["start", "goal", "start", "goal", "start", "goal", "start", "goal"]
        )
        event_types = np.array(
            ["enter", "enter", "exit", "exit", "enter", "enter", "exit", "exit"]
        )

        nwbfile = create_roundtrip_nwb()
        write_region_crossings(nwbfile, crossing_times, region_names, event_types)

        nwb_path = tmp_path / "crossings.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            recovered = read_events(reopened, table_name="region_crossings")

        expected = pd.DataFrame(
            {
                "timestamp": crossing_times,
                "region": region_names,
                "event_type": event_types,
            }
        )
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True)[["timestamp", "region", "event_type"]],
            expected,
            check_dtype=False,
        )


@pytest.mark.slow
@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteEventsDiskRoundTrip:
    """write_events (generic DataFrame writer) survives a real disk round-trip.

    write_events serializes via the ndx-events EventsTable extension, so these
    tests are gated on ndx_events. There is no fallback path when ndx-events is
    unavailable (write_events raises ImportError via _require_ndx_events), so a
    "without ndx-events" round-trip is not applicable.
    """

    def test_events_with_ndx_events_survive_disk_roundtrip(self, tmp_path):
        """Generic events DataFrame survives write -> close -> reopen."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_events, write_events

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.5, 5.0, 8.2],
                "label": ["reward", "lick", "reward", "lick"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        nwbfile = create_roundtrip_nwb()
        write_events(nwbfile, events, name="behavioral_events")

        nwb_path = tmp_path / "events.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            recovered = read_events(reopened, table_name="behavioral_events")

        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True)[["timestamp", "label", "value"]],
            events,
            check_dtype=False,
        )


def _write_rewards(nwbfile, times, *, overwrite=False):
    from neurospatial.io.nwb import write_events

    write_events(
        nwbfile,
        pd.DataFrame({"timestamp": times}),
        name="rewards",
        overwrite=overwrite,
    )


def _write_crossings(nwbfile, times, *, overwrite=False):
    from neurospatial.io.nwb import write_region_crossings

    n = len(times)
    write_region_crossings(
        nwbfile,
        times,
        np.array(["goal"] * n),
        np.array(["entry"] * n),
        name="rewards",
        overwrite=overwrite,
    )


_EVENT_WRITERS = [
    pytest.param(_write_rewards, id="write_events"),
    pytest.param(_write_crossings, id="write_region_crossings"),
]


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestEventsOverwrite:
    """overwrite=True never loses the existing table to a failed write."""

    @pytest.mark.parametrize("write", _EVENT_WRITERS)
    def test_invalid_replacement_keeps_the_original(self, empty_nwb, write):
        from neurospatial.io.nwb import read_events

        write(empty_nwb, np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="non-finite"):
            write(empty_nwb, np.array([3.0, np.nan]), overwrite=True)

        kept = read_events(empty_nwb, table_name="rewards")
        np.testing.assert_array_equal(kept["timestamp"].to_numpy(), [1.0, 2.0])

    @pytest.mark.parametrize("mode", ["r+", "a"])
    @pytest.mark.parametrize("write", _EVENT_WRITERS)
    def test_in_place_overwrite_is_refused(self, tmp_path, write, mode):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_events

        path = tmp_path / "session.nwb"
        nwbfile = create_roundtrip_nwb()
        write(nwbfile, np.array([1.0, 2.0]))
        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(path), mode) as io:
            opened = io.read()
            with pytest.raises(NotImplementedError, match=f"mode='{mode}'"):
                write(opened, np.array([5.0, 6.0, 7.0]), overwrite=True)
            kept = read_events(opened, table_name="rewards")
            np.testing.assert_array_equal(kept["timestamp"].to_numpy(), [1.0, 2.0])

    @pytest.mark.parametrize("write", _EVENT_WRITERS)
    def test_overwrite_then_export_round_trips(self, tmp_path, write):
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_events

        source = tmp_path / "session.nwb"
        copy = tmp_path / "session_new.nwb"
        nwbfile = create_roundtrip_nwb()
        write(nwbfile, np.array([1.0, 2.0]))
        with NWBHDF5IO(str(source), "w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(str(source), "r") as io:
            opened = io.read()
            write(opened, np.array([5.0, 6.0, 7.0]), overwrite=True)
            with NWBHDF5IO(str(copy), "w") as export_io:
                export_io.export(src_io=io, nwbfile=opened)

        with NWBHDF5IO(str(copy), "r") as io:
            replaced = read_events(io.read(), table_name="rewards")
        np.testing.assert_array_equal(replaced["timestamp"].to_numpy(), [5.0, 6.0, 7.0])


def _disk_round_trip(nwbfile, path):
    """Write ``nwbfile`` to ``path`` and return a reader for the reopened file."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return NWBHDF5IO(str(path), "r")


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestEventTablesOnDisk:
    """Event tables keep their columns and dtypes through a file."""

    def test_zero_region_crossings_round_trip(self, tmp_path):
        from neurospatial.io.nwb import read_events, write_region_crossings

        nwbfile = create_roundtrip_nwb()
        write_region_crossings(nwbfile, np.array([]), np.array([]), np.array([]))

        with _disk_round_trip(nwbfile, tmp_path / "crossings.nwb") as io:
            recovered = read_events(io.read(), table_name="region_crossings")
        assert len(recovered) == 0
        assert set(recovered.columns) == {"timestamp", "region", "event_type"}

    def test_zero_events_keep_their_columns(self, tmp_path):
        from neurospatial.io.nwb import read_events, write_events

        nwbfile = create_roundtrip_nwb()
        events = pd.DataFrame(
            {
                "timestamp": np.array([], dtype=float),
                "label": np.array([], dtype=object),
                "value": np.array([], dtype=float),
            }
        )
        write_events(nwbfile, events, name="rewards")

        with _disk_round_trip(nwbfile, tmp_path / "events.nwb") as io:
            recovered = read_events(io.read(), table_name="rewards")
        assert len(recovered) == 0
        assert set(recovered.columns) == {"timestamp", "label", "value"}

    def test_boolean_column_round_trips_as_bool(self, tmp_path):
        from neurospatial.io.nwb import read_events, write_events

        nwbfile = create_roundtrip_nwb()
        events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "rewarded": [True, False, True],
                "count": [1, 2, 3],
            }
        )
        write_events(nwbfile, events, name="rewards")

        with _disk_round_trip(nwbfile, tmp_path / "events.nwb") as io:
            recovered = read_events(io.read(), table_name="rewards")
        assert recovered["rewarded"].dtype == np.bool_
        np.testing.assert_array_equal(recovered["rewarded"], [True, False, True])
        assert np.issubdtype(recovered["count"].dtype, np.integer)
        np.testing.assert_array_equal(recovered["count"], [1, 2, 3])
