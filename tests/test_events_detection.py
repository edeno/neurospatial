"""Tests for events module spatial detection utilities."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Test add_positions
# =============================================================================


class TestAddPositions:
    """Tests for add_positions function."""

    def test_basic_interpolation_2d(self):
        """Test basic position interpolation for 2D trajectory."""
        from neurospatial.events.detection import add_positions

        # Events at t=1.5 and t=3.5
        events = pd.DataFrame({"timestamp": [1.5, 3.5], "label": ["A", "B"]})

        # Linear trajectory from (0,0) to (10,10) over 5 seconds
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array(
            [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0], [10.0, 10.0]]
        )

        result = add_positions(events, positions, times)

        # Check x, y columns added
        assert "x" in result.columns
        assert "y" in result.columns

        # Check interpolated values (linear interp at t=1.5 -> (3,3), t=3.5 -> (7,7))
        assert_allclose(result["x"].values, [3.0, 7.0])
        assert_allclose(result["y"].values, [3.0, 7.0])

        # Check original columns preserved
        assert "timestamp" in result.columns
        assert "label" in result.columns
        assert list(result["label"]) == ["A", "B"]

    def test_basic_interpolation_1d(self):
        """Test basic position interpolation for 1D trajectory."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.5, 3.0]})

        # 1D trajectory
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([[0.0], [10.0], [20.0], [30.0], [40.0]])

        result = add_positions(events, positions, times)

        # Only x column for 1D
        assert "x" in result.columns
        assert "y" not in result.columns

        assert_allclose(result["x"].values, [15.0, 30.0])

    def test_basic_interpolation_3d(self):
        """Test basic position interpolation for 3D trajectory."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [0.5]})

        # 3D trajectory
        times = np.array([0.0, 1.0])
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])

        result = add_positions(events, positions, times)

        # x, y, z columns for 3D
        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns

        assert_allclose(result["x"].values, [5.0])
        assert_allclose(result["y"].values, [10.0])
        assert_allclose(result["z"].values, [15.0])

    def test_event_at_trajectory_time(self):
        """Test event exactly at a trajectory sample time."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [2.0]})

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = add_positions(events, positions, times)

        # Should get exact position at t=2.0
        assert_allclose(result["x"].values, [3.0])
        assert_allclose(result["y"].values, [4.0])

    def test_returns_new_dataframe(self):
        """Test that original DataFrame is not modified."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.5]})
        original_columns = list(events.columns)

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, positions, times)

        # Original should be unchanged
        assert list(events.columns) == original_columns
        assert "x" not in events.columns

        # Result should have new columns
        assert "x" in result.columns
        assert "y" in result.columns

    def test_empty_events(self):
        """Test with empty events DataFrame."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": []})

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        result = add_positions(events, positions, times)

        assert len(result) == 0
        assert "x" in result.columns
        assert "y" in result.columns

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 2.0])
        positions = np.array([[0.0, 0.0], [4.0, 8.0]])

        result = add_positions(events, positions, times)

        assert len(result) == 1
        assert_allclose(result["x"].values, [2.0])
        assert_allclose(result["y"].values, [4.0])

    def test_custom_timestamp_column(self):
        """Test with custom timestamp column name."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"event_time": [1.5], "label": ["reward"]})

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, positions, times, timestamp_column="event_time")

        assert_allclose(result["x"].values, [1.5])
        assert_allclose(result["y"].values, [1.5])

    def test_preserves_all_columns(self):
        """Test that all original columns are preserved."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "label": ["A", "B"],
                "value": [10.0, 20.0],
                "trial_id": [1, 2],
            }
        )

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, positions, times)

        # All original columns preserved
        assert "timestamp" in result.columns
        assert "label" in result.columns
        assert "value" in result.columns
        assert "trial_id" in result.columns

        # Values unchanged
        assert list(result["label"]) == ["A", "B"]
        assert list(result["value"]) == [10.0, 20.0]

    def test_preserves_index(self):
        """Test that DataFrame index is preserved."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0, 2.0]}, index=["event_a", "event_b"])

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, positions, times)

        assert list(result.index) == ["event_a", "event_b"]

    def test_does_not_add_bin_index_or_region(self):
        """Test that only x, y (and z) are added - no derived columns."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        result = add_positions(events, positions, times)

        # Only x, y added (plus original timestamp)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "bin_index" not in result.columns
        assert "region" not in result.columns

    def test_event_before_trajectory_extrapolates(self):
        """Test event before trajectory start extrapolates from first segment."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [-1.0]})  # Before trajectory

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

        result = add_positions(events, positions, times)

        # Linear extrapolation: at t=-1, x=-1, y=-2
        assert_allclose(result["x"].values, [-1.0])
        assert_allclose(result["y"].values, [-2.0])

    def test_event_after_trajectory_extrapolates(self):
        """Test event after trajectory end extrapolates from last segment."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [3.0]})  # After trajectory

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

        result = add_positions(events, positions, times)

        # Linear extrapolation: at t=3, x=3, y=6
        assert_allclose(result["x"].values, [3.0])
        assert_allclose(result["y"].values, [6.0])

    def test_missing_timestamp_column_raises(self):
        """Test that missing timestamp column raises ValueError."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"time": [1.0]})  # Wrong column name

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(ValueError, match=r"timestamp.*column"):
            add_positions(events, positions, times)

    def test_non_dataframe_raises(self):
        """Test that non-DataFrame events raises TypeError."""
        from neurospatial.events.detection import add_positions

        events = {"timestamp": [1.0]}  # Dict, not DataFrame

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(TypeError, match=r"DataFrame"):
            add_positions(events, positions, times)

    def test_mismatched_times_positions_raises(self):
        """Test that mismatched times and positions raises ValueError."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 1.0, 2.0])  # 3 samples
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples

        with pytest.raises(ValueError, match=r"times.*positions"):
            add_positions(events, positions, times)

    def test_nan_in_event_timestamps_handled(self):
        """Test NaN timestamps in events are propagated to positions."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0, np.nan, 2.0]})

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, positions, times)

        # First and third should be valid
        assert_allclose(result["x"].values[0], 1.0)
        assert_allclose(result["x"].values[2], 2.0)

        # Second (NaN timestamp) should produce NaN position
        assert np.isnan(result["x"].values[1])
        assert np.isnan(result["y"].values[1])

    def test_output_dtype_is_float64(self):
        """Test that x, y columns have float64 dtype."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0, 0], [1, 1], [2, 2]])  # Integer positions

        result = add_positions(events, positions, times)

        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.float64

    def test_unsorted_trajectory_times_handled(self):
        """Test that unsorted trajectory times are handled correctly."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.5]})

        # Unsorted times and positions
        times = np.array([2.0, 0.0, 1.0, 3.0])
        positions = np.array([[4.0, 4.0], [0.0, 0.0], [2.0, 2.0], [6.0, 6.0]])

        result = add_positions(events, positions, times)

        # Should interpolate correctly after internal sorting
        # t=1.5 is between t=1 (pos 2,2) and t=2 (pos 4,4) -> (3,3)
        assert_allclose(result["x"].values, [3.0])
        assert_allclose(result["y"].values, [3.0])

    def test_typical_neuroscience_use_case(self):
        """Test typical use case: adding positions to reward events."""
        from neurospatial.events.detection import add_positions

        # Reward events from behavioral task
        rewards = pd.DataFrame(
            {
                "timestamp": [10.5, 25.3, 40.1, 55.8],
                "reward_size": [1.0, 2.0, 1.0, 2.0],
                "trial": [1, 2, 3, 4],
            }
        )

        # Position tracking at 30 Hz over 60 seconds
        n_samples = 1800
        times = np.linspace(0, 60, n_samples)
        # Circular trajectory
        positions = np.column_stack(
            [50 + 30 * np.cos(times * 0.1), 50 + 30 * np.sin(times * 0.1)]
        )

        result = add_positions(rewards, positions, times)

        # All rewards should have positions
        assert len(result) == 4
        assert not result["x"].isna().any()
        assert not result["y"].isna().any()

        # Original columns preserved
        assert list(result["reward_size"]) == [1.0, 2.0, 1.0, 2.0]
        assert list(result["trial"]) == [1, 2, 3, 4]

        # Positions should be within environment bounds
        assert np.all(result["x"] >= 20)
        assert np.all(result["x"] <= 80)
        assert np.all(result["y"] >= 20)
        assert np.all(result["y"] <= 80)
