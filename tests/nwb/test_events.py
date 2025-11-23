"""
Tests for NWB events reading functions.

Tests the read_events() function for reading EventsTable data from NWB files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip all tests if ndx_events is not installed
ndx_events = pytest.importorskip("ndx_events")
pynwb = pytest.importorskip("pynwb")


class TestReadEvents:
    """Tests for read_events() function."""

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
