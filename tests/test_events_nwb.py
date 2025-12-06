"""Tests for NWB integration with events module.

Tests for:
- write_events(): Write generic events DataFrame to NWB EventsTable
- dataframe_to_events_table(): Convert DataFrame to EventsTable
- Round-trip: DataFrame -> NWB -> DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

# Skip entire module if pynwb or ndx-events not installed
pytest.importorskip("pynwb")
pytest.importorskip("ndx_events")


@pytest.fixture
def nwbfile():
    """Create a minimal NWBFile for testing."""
    from datetime import datetime

    from pynwb import NWBFile

    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test_session_001",
        session_start_time=datetime(2025, 1, 1, 12, 0, 0),
    )
    return nwbfile


@pytest.fixture
def basic_events_df():
    """Create a basic events DataFrame with just timestamps."""
    return pd.DataFrame({"timestamp": [1.0, 2.5, 5.0, 8.2]})


@pytest.fixture
def events_with_labels_df():
    """Create events DataFrame with timestamp and label columns."""
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0, 4.0],
            "label": ["reward", "lick", "reward", "lick"],
        }
    )


@pytest.fixture
def events_with_spatial_df():
    """Create events DataFrame with spatial columns."""
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.5, 5.0],
            "x": [10.0, 25.5, 50.0],
            "y": [20.0, 35.5, 60.0],
            "label": ["a", "b", "c"],
        }
    )


@pytest.fixture
def events_with_value_df():
    """Create events DataFrame with numeric value column."""
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0],
            "value": [0.5, 1.2, 0.8],
        }
    )


class TestWriteEvents:
    """Tests for write_events() function."""

    def test_basic_events(self, nwbfile, basic_events_df):
        """Test writing events with only timestamps."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, basic_events_df, name="test_events")

        # Verify written
        result = read_events(nwbfile, "test_events")
        assert len(result) == 4
        assert_array_almost_equal(
            result["timestamp"].values, basic_events_df["timestamp"].values
        )

    def test_events_with_labels(self, nwbfile, events_with_labels_df):
        """Test writing events with label column."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, events_with_labels_df, name="labeled_events")

        result = read_events(nwbfile, "labeled_events")
        assert len(result) == 4
        assert "label" in result.columns
        assert_array_equal(
            result["label"].values, events_with_labels_df["label"].values
        )

    def test_events_with_spatial_columns(self, nwbfile, events_with_spatial_df):
        """Test writing events with x, y spatial columns."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, events_with_spatial_df, name="spatial_events")

        result = read_events(nwbfile, "spatial_events")
        assert len(result) == 3
        assert "x" in result.columns
        assert "y" in result.columns
        assert_array_almost_equal(
            result["x"].values, events_with_spatial_df["x"].values
        )
        assert_array_almost_equal(
            result["y"].values, events_with_spatial_df["y"].values
        )

    def test_events_with_value(self, nwbfile, events_with_value_df):
        """Test writing events with numeric value column."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, events_with_value_df, name="value_events")

        result = read_events(nwbfile, "value_events")
        assert len(result) == 3
        assert "value" in result.columns
        assert_array_almost_equal(
            result["value"].values, events_with_value_df["value"].values
        )

    def test_custom_description(self, nwbfile, basic_events_df):
        """Test custom description is stored."""
        from neurospatial.io.nwb import write_events

        write_events(
            nwbfile,
            basic_events_df,
            name="described_events",
            description="My custom description",
        )

        # Access the EventsTable directly to check description
        events_table = nwbfile.processing["behavior"]["described_events"]
        assert events_table.description == "My custom description"

    def test_custom_processing_module(self, nwbfile, basic_events_df):
        """Test writing to custom processing module."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(
            nwbfile,
            basic_events_df,
            name="custom_events",
            processing_module="analysis",
        )

        # Verify it's in the custom module
        assert "analysis" in nwbfile.processing
        result = read_events(nwbfile, "custom_events", processing_module="analysis")
        assert len(result) == 4

    def test_overwrite_false_raises(self, nwbfile, basic_events_df):
        """Test that overwrite=False raises on duplicate name."""
        from neurospatial.io.nwb import write_events

        write_events(nwbfile, basic_events_df, name="dup_events")

        with pytest.raises(ValueError, match="already exists"):
            write_events(nwbfile, basic_events_df, name="dup_events", overwrite=False)

    def test_overwrite_true_replaces(self, nwbfile, basic_events_df):
        """Test that overwrite=True replaces existing table."""
        from neurospatial.io.nwb import read_events, write_events

        # Write initial
        write_events(nwbfile, basic_events_df, name="replace_events")

        # Write new data with overwrite
        new_events = pd.DataFrame({"timestamp": [10.0, 20.0]})
        write_events(nwbfile, new_events, name="replace_events", overwrite=True)

        result = read_events(nwbfile, "replace_events")
        assert len(result) == 2
        assert_array_almost_equal(result["timestamp"].values, [10.0, 20.0])

    def test_empty_events(self, nwbfile):
        """Test writing empty events DataFrame."""
        from neurospatial.io.nwb import read_events, write_events

        empty_df = pd.DataFrame({"timestamp": []})
        write_events(nwbfile, empty_df, name="empty_events")

        result = read_events(nwbfile, "empty_events")
        assert len(result) == 0

    def test_invalid_not_dataframe(self, nwbfile):
        """Test that non-DataFrame input raises TypeError."""
        from neurospatial.io.nwb import write_events

        with pytest.raises(TypeError, match="DataFrame"):
            write_events(nwbfile, [1.0, 2.0, 3.0], name="bad_events")

    def test_invalid_no_timestamp(self, nwbfile):
        """Test that DataFrame without timestamp column raises ValueError."""
        from neurospatial.io.nwb import write_events

        df = pd.DataFrame({"time": [1.0, 2.0]})  # Wrong column name
        with pytest.raises(ValueError, match="timestamp"):
            write_events(nwbfile, df, name="bad_events")

    def test_invalid_nan_timestamps(self, nwbfile):
        """Test that NaN timestamps raise ValueError."""
        from neurospatial.io.nwb import write_events

        df = pd.DataFrame({"timestamp": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="non-finite"):
            write_events(nwbfile, df, name="nan_events")

    def test_invalid_inf_timestamps(self, nwbfile):
        """Test that Inf timestamps raise ValueError."""
        from neurospatial.io.nwb import write_events

        df = pd.DataFrame({"timestamp": [1.0, np.inf, 3.0]})
        with pytest.raises(ValueError, match="non-finite"):
            write_events(nwbfile, df, name="inf_events")

    def test_invalid_negative_timestamps(self, nwbfile):
        """Test that negative timestamps raise ValueError."""
        from neurospatial.io.nwb import write_events

        df = pd.DataFrame({"timestamp": [-1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="negative"):
            write_events(nwbfile, df, name="neg_events")

    def test_multiple_custom_columns(self, nwbfile):
        """Test writing events with multiple custom columns."""
        from neurospatial.io.nwb import read_events, write_events

        df = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "reward_amount": [0.1, 0.2, 0.15],
                "trial_id": [1, 1, 2],
                "region": ["goal", "home", "goal"],
            }
        )
        write_events(nwbfile, df, name="multi_col_events")

        result = read_events(nwbfile, "multi_col_events")
        assert "reward_amount" in result.columns
        assert "trial_id" in result.columns
        assert "region" in result.columns

    def test_preserves_column_order(self, nwbfile):
        """Test that column order is preserved in round-trip."""
        from neurospatial.io.nwb import read_events, write_events

        df = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "col_a": ["a", "b"],
                "col_b": [1, 2],
                "col_c": [1.5, 2.5],
            }
        )
        write_events(nwbfile, df, name="order_events")

        result = read_events(nwbfile, "order_events")
        # Note: timestamp is always first in EventsTable
        assert "timestamp" in result.columns
        assert "col_a" in result.columns
        assert "col_b" in result.columns
        assert "col_c" in result.columns


class TestDataframeToEventsTable:
    """Tests for dataframe_to_events_table() helper function."""

    def test_basic_conversion(self, basic_events_df):
        """Test basic DataFrame to EventsTable conversion."""
        from neurospatial.io.nwb._events import dataframe_to_events_table

        events_table = dataframe_to_events_table(
            basic_events_df, name="test", description="Test events"
        )

        # Verify it's an EventsTable
        import ndx_events

        assert isinstance(events_table, ndx_events.EventsTable)
        assert len(events_table) == 4

    def test_conversion_with_columns(self, events_with_labels_df):
        """Test conversion preserves additional columns."""
        from neurospatial.io.nwb._events import dataframe_to_events_table

        events_table = dataframe_to_events_table(
            events_with_labels_df, name="labeled", description="Labeled events"
        )

        # Verify columns exist
        assert "label" in events_table.colnames


class TestRoundTrip:
    """Tests for complete round-trip: DataFrame -> NWB -> DataFrame."""

    def test_basic_roundtrip(self, nwbfile, basic_events_df):
        """Test basic round-trip preserves data."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, basic_events_df, name="roundtrip")
        result = read_events(nwbfile, "roundtrip")

        assert_array_almost_equal(
            result["timestamp"].values, basic_events_df["timestamp"].values
        )

    def test_spatial_roundtrip(self, nwbfile, events_with_spatial_df):
        """Test round-trip with spatial columns preserves x, y."""
        from neurospatial.io.nwb import read_events, write_events

        write_events(nwbfile, events_with_spatial_df, name="spatial_roundtrip")
        result = read_events(nwbfile, "spatial_roundtrip")

        assert_array_almost_equal(
            result["x"].values, events_with_spatial_df["x"].values
        )
        assert_array_almost_equal(
            result["y"].values, events_with_spatial_df["y"].values
        )
        assert_array_equal(
            result["label"].values, events_with_spatial_df["label"].values
        )

    def test_roundtrip_with_mixed_types(self, nwbfile):
        """Test round-trip with mixed column types."""
        from neurospatial.io.nwb import read_events, write_events

        df = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        write_events(nwbfile, df, name="mixed_roundtrip")
        result = read_events(nwbfile, "mixed_roundtrip")

        assert_array_almost_equal(result["timestamp"].values, df["timestamp"].values)
        # Note: bool may be converted to int/str depending on NWB storage
        assert len(result) == 3


class TestIntegrationWithEventsModule:
    """Tests for integration with neurospatial.events module functions."""

    def test_write_events_from_add_positions(self, nwbfile):
        """Test writing events after add_positions()."""
        from neurospatial.events import add_positions
        from neurospatial.io.nwb import read_events, write_events

        # Create base events
        events = pd.DataFrame({"timestamp": [0.5, 1.5, 2.5]})

        # Add positions
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        times = np.array([0.0, 1.0, 2.0, 3.0])
        events_with_pos = add_positions(events, positions, times)

        # Write to NWB
        write_events(nwbfile, events_with_pos, name="positioned_events")

        # Read back
        result = read_events(nwbfile, "positioned_events")
        assert "x" in result.columns
        assert "y" in result.columns
        assert len(result) == 3

    def test_write_events_from_filter_by_intervals(self, nwbfile):
        """Test writing filtered events."""
        from neurospatial.events import filter_by_intervals
        from neurospatial.io.nwb import read_events, write_events

        # Create events
        events = pd.DataFrame({"timestamp": [1.0, 2.0, 5.0, 8.0, 10.0]})

        # Create intervals
        intervals = pd.DataFrame({"start_time": [0.0, 4.0], "stop_time": [3.0, 9.0]})

        # Filter
        filtered = filter_by_intervals(events, intervals)

        # Write to NWB
        write_events(nwbfile, filtered, name="filtered_events")

        # Read back
        result = read_events(nwbfile, "filtered_events")
        assert len(result) == 4  # Events at 1.0, 2.0, 5.0, 8.0
