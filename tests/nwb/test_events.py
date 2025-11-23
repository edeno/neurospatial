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
        from neurospatial.nwb import read_events

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
        from neurospatial.nwb import read_events

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

        from neurospatial.nwb import read_events

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
        from neurospatial.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_table"):
            read_events(sample_nwb_with_events, "nonexistent_table")

    def test_error_when_module_not_found(self, empty_nwb):
        """Test KeyError when processing module not found."""
        from neurospatial.nwb import read_events

        with pytest.raises(KeyError, match="nonexistent_module"):
            read_events(empty_nwb, "events", processing_module="nonexistent_module")

    def test_dataframe_output_with_timestamp_column(self, sample_nwb_with_events):
        """Test DataFrame output includes timestamp column."""
        from neurospatial.nwb import read_events

        events = read_events(sample_nwb_with_events, "laps")

        # Timestamp must be a column
        assert "timestamp" in events.columns

        # Timestamps should be sorted (as created in fixture)
        timestamps = events["timestamp"].values
        assert all(timestamps[:-1] <= timestamps[1:])

    def test_additional_columns_preserved(self, sample_nwb_with_events):
        """Test that additional columns from EventsTable are preserved."""
        from neurospatial.nwb import read_events

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

        from neurospatial.nwb import read_events

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

        from neurospatial.nwb import read_events

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

        from neurospatial.nwb import read_events

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
            from neurospatial.nwb._core import _require_ndx_events

            with pytest.raises(ImportError, match="ndx-events is required"):
                _require_ndx_events()
        finally:
            # Restore modules
            sys.modules.update(saved_modules)


class TestReadIntervals:
    """Tests for read_intervals() function (built-in NWB TimeIntervals)."""

    def test_read_trials(self, empty_nwb):
        """Test reading trials table from NWB file."""
        from neurospatial.nwb import read_intervals

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

        from neurospatial.nwb import read_intervals

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

        from neurospatial.nwb import read_intervals

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
        from neurospatial.nwb import read_intervals

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
        from neurospatial.nwb import read_intervals

        with pytest.raises(KeyError, match="nonexistent"):
            read_intervals(empty_nwb, "nonexistent")

    def test_additional_columns_preserved(self, empty_nwb):
        """Test that additional columns from TimeIntervals are preserved."""
        from pynwb.epoch import TimeIntervals

        from neurospatial.nwb import read_intervals

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

        from neurospatial.nwb import read_intervals

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
        from neurospatial.nwb import read_intervals

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
