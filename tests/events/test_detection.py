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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(events, times=times, positions=positions, max_gap=None)

        assert len(result) == 0
        assert "x" in result.columns
        assert "y" in result.columns

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 2.0])
        positions = np.array([[0.0, 0.0], [4.0, 8.0]])

        result = add_positions(events, times=times, positions=positions, max_gap=None)

        assert len(result) == 1
        assert_allclose(result["x"].values, [2.0])
        assert_allclose(result["y"].values, [4.0])

    def test_custom_timestamp_column(self):
        """Test with custom timestamp column name."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"event_time": [1.5], "label": ["reward"]})

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(
            events,
            times=times,
            positions=positions,
            timestamp_column="event_time",
            max_gap=None,
        )

        assert_allclose(result["x"].values, [1.5])
        assert_allclose(result["y"].values, [1.5])

    def test_preserves_index(self):
        """Test that DataFrame index is preserved."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0, 2.0]}, index=["event_a", "event_b"])

        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = add_positions(events, times=times, positions=positions, max_gap=None)

        assert list(result.index) == ["event_a", "event_b"]

    def test_event_before_trajectory_gets_nan(self):
        """Outside-span events are missing; observed events retain positions."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [-1.0, 1.0]})

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

        result = add_positions(events, times=times, positions=positions, max_gap=None)

        assert_allclose(result["x"].values, [np.nan, 1.0])
        assert_allclose(result["y"].values, [np.nan, 2.0])

    def test_event_after_trajectory_gets_nan(self):
        """Outside-span events are missing; observed events retain positions."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [3.0, 1.0]})

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

        result = add_positions(events, times=times, positions=positions, max_gap=None)

        assert_allclose(result["x"].values, [np.nan, 1.0])
        assert_allclose(result["y"].values, [np.nan, 2.0])

    def test_missing_timestamp_column_raises(self):
        """Test that missing timestamp column raises ValueError."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"time": [1.0]})  # Wrong column name

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(ValueError, match=r"Missing required columns.*timestamp"):
            add_positions(events, times=times, positions=positions)

    def test_non_dataframe_raises(self):
        """Test that non-DataFrame events raises TypeError."""
        from neurospatial.events.detection import add_positions

        events = {"timestamp": [1.0]}  # Dict, not DataFrame

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(TypeError, match=r"DataFrame"):
            add_positions(events, times=times, positions=positions)

    def test_mismatched_times_positions_raises(self):
        """Test that mismatched times and positions raises ValueError."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0]})

        times = np.array([0.0, 1.0, 2.0])  # 3 samples
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples

        with pytest.raises(ValueError, match=r"times.*positions"):
            add_positions(events, times=times, positions=positions)

    def test_swapped_positions_times_raises_clear_error(self):
        """Swapping times/positions fails loud, not silently -- even in 1-D.

        ``times`` and ``positions`` are keyword-only, so the old positional order
        ``add_positions(events, positions, times)`` raises ``TypeError`` instead
        of silently mis-interpolating. This matters most for a bare 1-D
        trajectory, where both arrays are 1-D and a positional swap would be
        shape-indistinguishable (the length + shape guards cannot catch it). A
        mislabeled 2-D array handed to ``times=`` is still caught by the shape
        guard.
        """
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [0.1]})
        times = np.array([0.0, 0.1, 0.2])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        # Correct (keyword) call works.
        result = add_positions(events, times=times, positions=positions)
        np.testing.assert_allclose(result[["x", "y"]].to_numpy(), [[1.0, 1.0]])

        # Old positional order is now a loud TypeError, not a wrong result.
        with pytest.raises(TypeError):
            add_positions(events, positions, times)

        # The dangerous 1-D case: both arrays 1-D, so a positional swap would be
        # shape-indistinguishable -- keyword-only makes it a TypeError too.
        positions_1d = np.array([15.0, 20.0, 25.0])
        with pytest.raises(TypeError):
            add_positions(events, positions_1d, times)

        # A mislabeled 2-D array passed to times= is caught by the shape guard.
        with pytest.raises(ValueError, match=r"(?i)1-D|swap"):
            add_positions(events, times=positions, positions=times)

    def test_nan_event_timestamp_raises(self):
        """Like every events function, a NaN event time is refused."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0, np.nan, 2.0]})
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        with pytest.raises(ValueError, match=r"NaN or inf time.*row 1"):
            add_positions(events, times=times, positions=positions, max_gap=None)

    def test_unsorted_trajectory_times_raise(self):
        """Trajectory times must increase, as for every per-sample analysis."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.5]})
        times = np.array([2.0, 0.0, 1.0, 3.0])
        positions = np.array([[4.0, 4.0], [0.0, 0.0], [2.0, 2.0], [6.0, 6.0]])

        with pytest.raises(ValueError, match="decreases"):
            add_positions(events, times=times, positions=positions, max_gap=None)

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

        result = add_positions(rewards, times=times, positions=positions)

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


class TestAddPositionsDegenerateTrajectory:
    """Tests that add_positions rejects trajectories interp1d cannot use."""

    def test_add_positions_single_sample_is_a_point_observation(self):
        """One sample positions an event at its time and nothing else."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.0, 1.5]})
        times = np.array([1.0])
        positions = np.array([[4.0, 5.0]])

        result = add_positions(events, times=times, positions=positions)

        assert_allclose(result.loc[0, ["x", "y"]].to_numpy(dtype=float), [4.0, 5.0])
        assert result.loc[1, ["x", "y"]].isna().all()

    def test_add_positions_identical_times_raises(self):
        """All-identical trajectory times leave the interpolant undefined."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [2.0]})
        times = np.array([2.0, 2.0, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(ValueError, match="strictly increasing"):
            add_positions(events, times=times, positions=positions)

    def test_add_positions_duplicate_time_raises(self):
        """Which of two positions at one instant wins depended on sort order."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [2.0, 2.5]})
        times = np.array([0.0, 1.0, 2.0, 2.0, 3.0])
        positions = np.array([[0.0], [1.0], [2.0], [50.0], [3.0]])

        with pytest.raises(ValueError, match=r"strictly increasing.*== 2 at index 2"):
            add_positions(events, times=times, positions=positions)

    @pytest.mark.parametrize("n_dims", [0, 4])
    def test_add_positions_refuses_other_than_1_to_3_dims(self, n_dims):
        """Only x, y and z are written; a 4th column was silently dropped."""
        from neurospatial.events.detection import add_positions

        times = np.arange(5.0)
        with pytest.raises(ValueError, match=f"{n_dims} columns"):
            add_positions(
                pd.DataFrame({"timestamp": [1.5]}),
                times=times,
                positions=np.zeros((5, n_dims)),
            )

    def test_add_positions_refuses_text_timestamps(self):
        from neurospatial.events.detection import add_positions

        with pytest.raises(ValueError, match="non-numeric"):
            add_positions(
                pd.DataFrame({"timestamp": ["1.5"]}),
                times=np.arange(3.0),
                positions=np.zeros((3, 2)),
            )

    def test_add_positions_nan_trajectory_time_raises(self):
        """A NaN trajectory timestamp must raise, naming the argument."""
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": [1.5]})
        times = np.array([0.0, np.nan, 2.0])
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(ValueError, match="times"):
            add_positions(events, times=times, positions=positions)

    def test_add_positions_empty_events_get_empty_columns(self):
        from neurospatial.events.detection import add_positions

        events = pd.DataFrame({"timestamp": []})
        times = np.array([0.0])
        positions = np.array([[0.0, 0.0]])

        result = add_positions(events, times=times, positions=positions)

        assert len(result) == 0
        assert "x" in result.columns
        assert "y" in result.columns

    def test_add_positions_valid_unchanged(self):
        """The module doctest example still returns [[3, 3], [7, 7]]."""
        from neurospatial.events.detection import add_positions

        rewards = pd.DataFrame({"timestamp": [1.5, 3.5], "size": [1, 2]})
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]], dtype=np.float64)

        result = add_positions(rewards, times=times, positions=positions, max_gap=None)

        assert_allclose(result[["x", "y"]].values, [[3.0, 3.0], [7.0, 7.0]])
