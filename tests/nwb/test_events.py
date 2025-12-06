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

        events = read_events(sample_nwb_with_events, "laps")

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

        events = read_events(sample_nwb_with_events, "laps")

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
        events = read_events(nwbfile, "custom_events", processing_module="analysis")

        assert len(events) == 2
        assert events["timestamp"].iloc[0] == pytest.approx(1.5)
        assert events["timestamp"].iloc[1] == pytest.approx(3.0)

    def test_error_when_table_not_found(self, sample_nwb_with_events):
        """Test KeyError when EventsTable not found."""
        from neurospatial.io.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_table"):
            read_events(sample_nwb_with_events, "nonexistent_table")

    def test_error_when_module_not_found(self, empty_nwb):
        """Test KeyError when processing module not found."""
        from neurospatial.io.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_module"):
            read_events(empty_nwb, "events", processing_module="nonexistent_module")

    def test_dataframe_output_with_timestamp_column(self, sample_nwb_with_events):
        """Test DataFrame output includes timestamp column."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, "laps")

        # Timestamp must be a column
        assert "timestamp" in events.columns

        # Timestamps should be sorted (as created in fixture)
        timestamps = events["timestamp"].values
        assert all(timestamps[:-1] <= timestamps[1:])

    def test_additional_columns_preserved(self, sample_nwb_with_events):
        """Test that additional columns from EventsTable are preserved."""
        from neurospatial.io.nwb import read_events

        events = read_events(sample_nwb_with_events, "laps")

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
        events = read_events(nwbfile, "empty_events")

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
        events = read_events(nwbfile, "region_events")

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
            read_events(nwbfile, "Position")


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestReadEventsImportError:
    """Tests for import error handling in read_events()."""

    def test_import_error_message(self, monkeypatch):
        """Test ImportError message when ndx-events is not installed."""
        # This test verifies the _require_ndx_events() function
        # by mocking the import to fail
        import sys

        # Save and remove ndx_events from sys.modules
        saved_modules = {}
        modules_to_remove = [k for k in sys.modules if k.startswith("ndx_events")]
        for mod in modules_to_remove:
            saved_modules[mod] = sys.modules.pop(mod)

        # Mock the import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "ndx_events" or name.startswith("ndx_events."):
                raise ImportError("No module named 'ndx_events'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        try:
            from neurospatial.io.nwb._core import _require_ndx_events

            with pytest.raises(ImportError, match="ndx-events is required"):
                _require_ndx_events()
        finally:
            # Restore modules
            sys.modules.update(saved_modules)


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
        trials = read_intervals(nwbfile, "trials")

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
        result = read_intervals(nwbfile, "epochs")

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
        result = read_intervals(nwbfile, "laps")

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

        trials = read_intervals(nwbfile, "trials")

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
            read_intervals(empty_nwb, "nonexistent")

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

        result = read_intervals(nwbfile, "behavioral_states")

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

        result = read_intervals(nwbfile, "empty_table")

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

        result = read_intervals(nwbfile, "trials")

        # Get original data
        original = nwbfile.trials

        np.testing.assert_array_almost_equal(
            result["start_time"].values, original["start_time"][:]
        )
        np.testing.assert_array_almost_equal(
            result["stop_time"].values, original["stop_time"][:]
        )


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteLaps:
    """Tests for write_laps() function (requires ndx-events)."""

    def test_basic_lap_times_writing(self, empty_nwb):
        """Test writing basic lap times to NWB file."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])

        # Write laps
        write_laps(nwbfile, lap_times)

        # Verify EventsTable was created
        assert "behavior" in nwbfile.processing
        assert "laps" in nwbfile.processing["behavior"].data_interfaces

        # Verify data
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert len(laps_table) == 4

        # Check timestamps
        np.testing.assert_array_almost_equal(laps_table["timestamp"][:], lap_times)

    def test_with_lap_types_direction(self, empty_nwb):
        """Test writing laps with direction/lap_types column."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])
        lap_types = np.array([0, 1, 0, 1])  # 0=outbound, 1=inbound

        # Write laps with directions
        write_laps(nwbfile, lap_times, lap_types=lap_types)

        # Verify direction column
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "direction" in laps_table.colnames

        # Check direction values
        np.testing.assert_array_equal(laps_table["direction"][:], lap_types)

    def test_events_table_creation_in_behavior(self, empty_nwb):
        """Test EventsTable is created in processing/behavior/ module."""
        from ndx_events import EventsTable

        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 2.0, 3.0])

        # Write laps
        write_laps(nwbfile, lap_times)

        # Verify processing/behavior exists
        assert "behavior" in nwbfile.processing

        # Verify laps is an EventsTable
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert isinstance(laps_table, EventsTable)

    def test_custom_description(self, empty_nwb):
        """Test custom description parameter."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 2.0])

        # Write with custom description
        write_laps(nwbfile, lap_times, description="Custom lap description")

        laps_table = nwbfile.processing["behavior"]["laps"]
        assert laps_table.description == "Custom lap description"

    def test_default_description(self, empty_nwb):
        """Test default description is used."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0])

        write_laps(nwbfile, lap_times)

        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "Detected lap events" in laps_table.description

    def test_duplicate_name_error(self, empty_nwb):
        """Test error when laps table already exists without overwrite."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 2.0])

        # Write laps first time
        write_laps(nwbfile, lap_times)

        # Try to write again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            write_laps(nwbfile, lap_times)

    def test_overwrite_replaces_existing(self, empty_nwb):
        """Test overwrite=True replaces existing laps table."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        original_times = np.array([1.0, 2.0])
        new_times = np.array([5.0, 10.0, 15.0])

        # Write original laps
        write_laps(nwbfile, original_times)

        # Overwrite with new data
        write_laps(nwbfile, new_times, overwrite=True)

        # Verify new data
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert len(laps_table) == 3
        np.testing.assert_array_almost_equal(laps_table["timestamp"][:], new_times)

    def test_empty_lap_times(self, empty_nwb):
        """Test writing empty lap times array."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([])

        # Write empty laps - should succeed
        write_laps(nwbfile, lap_times)

        laps_table = nwbfile.processing["behavior"]["laps"]
        assert len(laps_table) == 0

    def test_lap_times_must_be_1d(self, empty_nwb):
        """Test error when lap_times is not 1D."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match=r"1D|one-dimensional"):
            write_laps(nwbfile, lap_times_2d)

    def test_lap_types_length_mismatch(self, empty_nwb):
        """Test error when lap_types length doesn't match lap_times."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 2.0, 3.0])
        lap_types = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError, match=r"length|shape"):
            write_laps(nwbfile, lap_times, lap_types=lap_types)

    def test_data_integrity(self, empty_nwb):
        """Test data integrity after writing."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.5, 3.2, 7.8, 12.1])
        lap_types = np.array([1, 0, 1, 0])

        # Write laps
        write_laps(nwbfile, lap_times, lap_types=lap_types)

        # Read back using read_events
        events = read_events(nwbfile, "laps")

        # Verify data matches
        np.testing.assert_array_almost_equal(events["timestamp"].values, lap_times)
        np.testing.assert_array_equal(events["direction"].values, lap_types)

    def test_behavior_module_reuse(self, empty_nwb):
        """Test that existing behavior module is reused."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import write_laps

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

        # Write laps - should reuse existing module
        lap_times = np.array([1.0, 2.0])
        write_laps(nwbfile, lap_times)

        # Verify both Position and laps exist in same module
        assert "Position" in nwbfile.processing["behavior"].data_interfaces
        assert "laps" in nwbfile.processing["behavior"].data_interfaces

    def test_custom_name(self, empty_nwb):
        """Test writing laps with custom table name."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 2.0])

        # Write with custom name
        write_laps(nwbfile, lap_times, name="track_laps")

        # Verify custom name used
        assert "track_laps" in nwbfile.processing["behavior"].data_interfaces
        assert "laps" not in nwbfile.processing["behavior"].data_interfaces

    def test_lap_times_with_nan_raises_error(self, empty_nwb):
        """Test error when lap_times contains NaN values."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match=r"non-finite|NaN"):
            write_laps(nwbfile, lap_times)

    def test_lap_times_with_inf_raises_error(self, empty_nwb):
        """Test error when lap_times contains Inf values."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match=r"non-finite|Inf"):
            write_laps(nwbfile, lap_times)

    def test_lap_times_negative_raises_error(self, empty_nwb):
        """Test error when lap_times contains negative timestamps."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([-1.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="negative"):
            write_laps(nwbfile, lap_times)


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteLapsImportError:
    """Tests for import error handling in write_laps()."""

    def test_import_error_message(self, monkeypatch):
        """Test ImportError message when ndx-events is not installed."""
        import sys

        # Save and remove ndx_events from sys.modules
        saved_modules = {}
        modules_to_remove = [k for k in sys.modules if k.startswith("ndx_events")]
        for mod in modules_to_remove:
            saved_modules[mod] = sys.modules.pop(mod)

        # Mock the import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "ndx_events" or name.startswith("ndx_events."):
                raise ImportError("No module named 'ndx_events'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        try:
            from neurospatial.io.nwb._core import _require_ndx_events

            with pytest.raises(ImportError, match="ndx-events is required"):
                _require_ndx_events()
        finally:
            # Restore modules
            sys.modules.update(saved_modules)


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
        events = read_events(nwbfile, "region_crossings")

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


@pytest.mark.skipif(not HAS_NDX_EVENTS, reason="ndx_events not installed")
class TestWriteLapsRegionColumns:
    """Tests for write_laps() region columns extension (requires ndx-events)."""

    def test_write_laps_with_start_regions(self, empty_nwb):
        """Test writing laps with start_regions column."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])
        start_regions = ["home", "goal", "home", "goal"]

        # Write laps with start_regions
        write_laps(nwbfile, lap_times, start_regions=start_regions)

        # Verify data
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "start_region" in laps_table.colnames

        # Check values via read_events
        events = read_events(nwbfile, "laps")
        assert list(events["start_region"]) == start_regions

    def test_write_laps_with_end_regions(self, empty_nwb):
        """Test writing laps with end_regions column."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])
        end_regions = ["goal", "home", "goal", "home"]

        # Write laps with end_regions
        write_laps(nwbfile, lap_times, end_regions=end_regions)

        # Verify data
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "end_region" in laps_table.colnames

        # Check values via read_events
        events = read_events(nwbfile, "laps")
        assert list(events["end_region"]) == end_regions

    def test_write_laps_with_stop_times(self, empty_nwb):
        """Test writing laps with stop_times column (interval events)."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])
        stop_times = np.array([4.5, 9.8, 14.5, 20.0])

        # Write laps with stop_times
        write_laps(nwbfile, lap_times, stop_times=stop_times)

        # Verify data
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "stop_time" in laps_table.colnames

        # Check values via read_events
        events = read_events(nwbfile, "laps")
        np.testing.assert_array_almost_equal(events["stop_time"].values, stop_times)

    def test_write_laps_with_all_optional(self, empty_nwb):
        """Test writing laps with all optional columns."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        lap_types = np.array([0, 1, 0])
        start_regions = ["home", "goal", "home"]
        end_regions = ["goal", "home", "goal"]
        stop_times = np.array([4.5, 9.8, 14.5])

        # Write laps with all columns
        write_laps(
            nwbfile,
            lap_times,
            lap_types=lap_types,
            start_regions=start_regions,
            end_regions=end_regions,
            stop_times=stop_times,
        )

        # Verify all columns exist
        laps_table = nwbfile.processing["behavior"]["laps"]
        assert "direction" in laps_table.colnames
        assert "start_region" in laps_table.colnames
        assert "end_region" in laps_table.colnames
        assert "stop_time" in laps_table.colnames

        # Check all values via read_events
        events = read_events(nwbfile, "laps")
        np.testing.assert_array_equal(events["direction"].values, lap_types)
        assert list(events["start_region"]) == start_regions
        assert list(events["end_region"]) == end_regions
        np.testing.assert_array_almost_equal(events["stop_time"].values, stop_times)

    def test_write_laps_start_regions_length_mismatch(self, empty_nwb):
        """Test error when start_regions length doesn't match lap_times."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        start_regions = ["home", "goal"]  # Wrong length

        with pytest.raises(ValueError, match=r"length"):
            write_laps(nwbfile, lap_times, start_regions=start_regions)

    def test_write_laps_end_regions_length_mismatch(self, empty_nwb):
        """Test error when end_regions length doesn't match lap_times."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        end_regions = ["goal", "home"]  # Wrong length

        with pytest.raises(ValueError, match=r"length"):
            write_laps(nwbfile, lap_times, end_regions=end_regions)

    def test_write_laps_stop_times_length_mismatch(self, empty_nwb):
        """Test error when stop_times length doesn't match lap_times."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        stop_times = np.array([4.5, 9.8])  # Wrong length

        with pytest.raises(ValueError, match=r"length"):
            write_laps(nwbfile, lap_times, stop_times=stop_times)

    def test_write_laps_stop_times_less_than_lap_times(self, empty_nwb):
        """Test error when stop_times < lap_times for any entry."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        stop_times = np.array([4.5, 3.0, 14.5])  # Second stop_time < lap_time

        with pytest.raises(ValueError, match=r"stop_time.*>=.*lap_time|start_time"):
            write_laps(nwbfile, lap_times, stop_times=stop_times)

    def test_write_laps_stop_times_with_nan_raises_error(self, empty_nwb):
        """Test error when stop_times contains NaN values."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        stop_times = np.array([4.5, np.nan, 14.5])

        with pytest.raises(ValueError, match=r"non-finite|NaN"):
            write_laps(nwbfile, lap_times, stop_times=stop_times)

    def test_write_laps_stop_times_negative_raises_error(self, empty_nwb):
        """Test error when stop_times contains negative timestamps."""
        from neurospatial.io.nwb import write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2])
        stop_times = np.array([-1.0, 9.8, 14.5])

        with pytest.raises(ValueError, match="negative"):
            write_laps(nwbfile, lap_times, stop_times=stop_times)

    def test_write_laps_regions_with_overwrite(self, empty_nwb):
        """Test overwrite replaces laps with region columns."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb

        # Write original laps (no regions)
        original_times = np.array([1.0, 2.0])
        write_laps(nwbfile, original_times)

        # Overwrite with regions
        new_times = np.array([5.0, 10.0, 15.0])
        new_start_regions = ["home", "goal", "home"]
        new_end_regions = ["goal", "home", "goal"]

        write_laps(
            nwbfile,
            new_times,
            start_regions=new_start_regions,
            end_regions=new_end_regions,
            overwrite=True,
        )

        # Verify new data
        events = read_events(nwbfile, "laps")
        assert len(events) == 3
        assert "start_region" in events.columns
        assert "end_region" in events.columns
        assert list(events["start_region"]) == new_start_regions
        assert list(events["end_region"]) == new_end_regions

    def test_write_laps_backwards_compatible(self, empty_nwb):
        """Test that write_laps without new parameters still works."""
        from neurospatial.io.nwb import read_events, write_laps

        nwbfile = empty_nwb
        lap_times = np.array([1.0, 5.5, 10.2, 15.8])
        lap_types = np.array([0, 1, 0, 1])

        # Write using original API (no new params)
        write_laps(nwbfile, lap_times, lap_types=lap_types)

        # Verify original behavior preserved
        events = read_events(nwbfile, "laps")
        np.testing.assert_array_almost_equal(events["timestamp"].values, lap_times)
        np.testing.assert_array_equal(events["direction"].values, lap_types)

        # New columns should NOT be present
        assert "start_region" not in events.columns
        assert "end_region" not in events.columns
        assert "stop_time" not in events.columns
