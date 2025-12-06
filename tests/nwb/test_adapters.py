"""
Tests for NWB adapter utilities.

Tests the internal adapter functions that extract data from pynwb/ndx containers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class MockTimeSeries:
    """Mock time series object for testing timestamps_from_series."""

    def __init__(
        self,
        data: np.ndarray,
        timestamps: np.ndarray | None = None,
        rate: float | None = None,
        starting_time: float | None = None,
    ):
        self.data = data
        self._timestamps = timestamps
        self.rate = rate
        self.starting_time = starting_time

    @property
    def timestamps(self):
        return self._timestamps


class MockEventsTable:
    """Mock events table for testing events_table_to_dataframe."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df


class TestTimestampsFromSeries:
    """Tests for timestamps_from_series() adapter function."""

    def test_returns_explicit_timestamps(self):
        """Test that explicit timestamps are returned when present."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        expected_timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        mock_series = MockTimeSeries(
            data=np.zeros((5, 2)),
            timestamps=expected_timestamps,
        )

        result = timestamps_from_series(mock_series)

        np.testing.assert_array_equal(result, expected_timestamps)
        assert result.dtype == np.float64

    def test_computes_timestamps_from_rate(self):
        """Test timestamp computation from sampling rate."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        mock_series = MockTimeSeries(
            data=np.zeros((5, 2)),
            timestamps=None,
            rate=10.0,  # 10 Hz
            starting_time=0.0,
        )

        result = timestamps_from_series(mock_series)

        expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_computes_timestamps_with_starting_time(self):
        """Test timestamp computation with non-zero starting time."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        mock_series = MockTimeSeries(
            data=np.zeros((5, 2)),
            timestamps=None,
            rate=10.0,  # 10 Hz
            starting_time=1.0,  # Start at 1 second
        )

        result = timestamps_from_series(mock_series)

        expected = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_none_starting_time(self):
        """Test that None starting_time defaults to 0.0."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        mock_series = MockTimeSeries(
            data=np.zeros((3, 2)),
            timestamps=None,
            rate=100.0,
            starting_time=None,
        )

        result = timestamps_from_series(mock_series)

        expected = np.array([0.0, 0.01, 0.02])
        np.testing.assert_array_almost_equal(result, expected)

    def test_raises_when_neither_timestamps_nor_rate(self):
        """Test ValueError when neither timestamps nor rate are available."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        mock_series = MockTimeSeries(
            data=np.zeros((5, 2)),
            timestamps=None,
            rate=None,
        )

        with pytest.raises(ValueError, match="neither 'timestamps' nor 'rate'"):
            timestamps_from_series(mock_series)

    def test_prefers_explicit_timestamps_over_rate(self):
        """Test that explicit timestamps take precedence over rate."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        explicit_timestamps = np.array([0.0, 0.5, 1.5, 3.0])  # Non-uniform
        mock_series = MockTimeSeries(
            data=np.zeros((4, 2)),
            timestamps=explicit_timestamps,
            rate=10.0,  # Would produce different result
            starting_time=0.0,
        )

        result = timestamps_from_series(mock_series)

        np.testing.assert_array_equal(result, explicit_timestamps)

    def test_output_dtype_is_float64(self):
        """Test that output is always float64."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        # Even with float32 input
        mock_series = MockTimeSeries(
            data=np.zeros((3, 2)),
            timestamps=np.array([0, 1, 2], dtype=np.float32),
        )

        result = timestamps_from_series(mock_series)

        assert result.dtype == np.float64


class TestEventsTableToDataframe:
    """Tests for events_table_to_dataframe() adapter function."""

    def test_returns_dataframe_with_timestamp(self):
        """Test successful conversion with timestamp column."""
        from neurospatial.io.nwb._adapters import events_table_to_dataframe

        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "label": ["a", "b", "c"],
            }
        )
        mock_table = MockEventsTable(df)

        result = events_table_to_dataframe(mock_table, table_name="test_table")

        pd.testing.assert_frame_equal(result, df)

    def test_raises_when_timestamp_missing(self):
        """Test KeyError when timestamp column is missing."""
        from neurospatial.io.nwb._adapters import events_table_to_dataframe

        df = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],  # Wrong column name
                "label": ["a", "b", "c"],
            }
        )
        mock_table = MockEventsTable(df)

        with pytest.raises(KeyError, match="does not have a 'timestamp' column"):
            events_table_to_dataframe(mock_table, table_name="test_table")

    def test_error_message_includes_table_name(self):
        """Test that error message includes the table name."""
        from neurospatial.io.nwb._adapters import events_table_to_dataframe

        df = pd.DataFrame({"value": [1, 2, 3]})
        mock_table = MockEventsTable(df)

        with pytest.raises(KeyError, match="my_events"):
            events_table_to_dataframe(mock_table, table_name="my_events")

    def test_preserves_additional_columns(self):
        """Test that additional columns are preserved."""
        from neurospatial.io.nwb._adapters import events_table_to_dataframe

        df = pd.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                "label": ["start", "stop"],
                "duration": [0.1, 0.2],
                "data": [42, 99],
            }
        )
        mock_table = MockEventsTable(df)

        result = events_table_to_dataframe(mock_table, table_name="test")

        assert list(result.columns) == ["timestamp", "label", "duration", "data"]

    def test_preserves_index(self):
        """Test that DataFrame index is preserved."""
        from neurospatial.io.nwb._adapters import events_table_to_dataframe

        df = pd.DataFrame(
            {"timestamp": [0.0, 1.0, 2.0], "label": ["a", "b", "c"]},
            index=[10, 20, 30],
        )
        mock_table = MockEventsTable(df)

        result = events_table_to_dataframe(mock_table, table_name="test")

        np.testing.assert_array_equal(result.index, [10, 20, 30])


class TestIntegrationWithPynwb:
    """Integration tests using real pynwb objects."""

    def test_timestamps_from_spatial_series(self, sample_nwb_with_position):
        """Test timestamps_from_series with real SpatialSeries."""
        from neurospatial.io.nwb._adapters import timestamps_from_series

        spatial_series = sample_nwb_with_position.processing["behavior"]["Position"][
            "position"
        ]
        timestamps = timestamps_from_series(spatial_series)

        assert timestamps.shape == (1000,)
        assert timestamps.dtype == np.float64
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0.0

    def test_timestamps_from_rate_based_series(self, empty_nwb):
        """Test timestamps computation from rate-based SpatialSeries."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb._adapters import timestamps_from_series

        rng = np.random.default_rng(42)

        # Create position data with rate instead of timestamps
        behavior_module = empty_nwb.create_processing_module(
            "behavior", "Behavioral data"
        )
        position = Position(name="Position")
        spatial_series = SpatialSeries(
            name="position",
            data=rng.random((100, 2)),
            reference_frame="arena",
            unit="cm",
            rate=30.0,  # 30 Hz sampling rate
            starting_time=0.0,
        )
        position.add_spatial_series(spatial_series)
        behavior_module.add(position)

        timestamps = timestamps_from_series(spatial_series)

        assert timestamps.shape == (100,)
        assert timestamps[0] == 0.0
        np.testing.assert_almost_equal(timestamps[1], 1.0 / 30.0)
        np.testing.assert_almost_equal(timestamps[-1], 99.0 / 30.0)
