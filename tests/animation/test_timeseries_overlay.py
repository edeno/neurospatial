"""Tests for TimeSeriesOverlay dataclass and TimeSeriesData container.

This module tests the public API dataclass (TimeSeriesOverlay) and its
internal data container (TimeSeriesData), including validation, conversion,
and window extraction functionality.

Following TDD: These tests are written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal


class TestTimeSeriesOverlayCreation:
    """Test TimeSeriesOverlay dataclass creation and defaults."""

    def test_basic_creation(self):
        """Test creating TimeSeriesOverlay with required fields only."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        assert_array_equal(overlay.data, data)
        assert_array_equal(overlay.times, times)
        # Check defaults
        assert overlay.label == ""
        assert overlay.color == "white"
        assert overlay.window_seconds == 2.0
        assert overlay.linewidth == 1.0
        assert overlay.alpha == 1.0
        assert overlay.group is None
        assert overlay.normalize is False
        assert overlay.show_cursor is True
        assert overlay.cursor_color == "red"
        assert overlay.vmin is None
        assert overlay.vmax is None
        assert overlay.interp == "linear"

    def test_creation_with_all_parameters(self):
        """Test TimeSeriesOverlay with all optional parameters."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=3.0,
            linewidth=2.0,
            alpha=0.8,
            group="kinematics",
            normalize=True,
            show_cursor=False,
            cursor_color="yellow",
            vmin=0.0,
            vmax=100.0,
            interp="nearest",
        )

        assert overlay.label == "Speed (cm/s)"
        assert overlay.color == "cyan"
        assert overlay.window_seconds == 3.0
        assert overlay.linewidth == 2.0
        assert overlay.alpha == 0.8
        assert overlay.group == "kinematics"
        assert overlay.normalize is True
        assert overlay.show_cursor is False
        assert overlay.cursor_color == "yellow"
        assert overlay.vmin == 0.0
        assert overlay.vmax == 100.0
        assert overlay.interp == "nearest"


class TestTimeSeriesOverlayValidation:
    """Test TimeSeriesOverlay validation in __post_init__."""

    def test_data_must_be_1d(self):
        """Test that data must be 1D array."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        times = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match=r"data.*1D|1-dimensional"):
            TimeSeriesOverlay(data=data_2d, times=times)

    def test_times_must_be_1d(self):
        """Test that times must be 1D array."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0])
        times_2d = np.array([[0.0], [1.0]])

        with pytest.raises(ValueError, match=r"times.*1D|1-dimensional"):
            TimeSeriesOverlay(data=data, times=times_2d)

    def test_data_and_times_must_have_same_length(self):
        """Test that data and times must have matching lengths."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0])  # Mismatched length

        with pytest.raises(ValueError, match="different lengths"):
            TimeSeriesOverlay(data=data, times=times)

    def test_times_must_be_monotonically_increasing(self):
        """Test that times must be strictly monotonically increasing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times_non_monotonic = np.array([0.0, 1.0, 0.5])  # Not monotonic

        with pytest.raises(ValueError, match="monotonic"):
            TimeSeriesOverlay(data=data, times=times_non_monotonic)

    def test_window_seconds_must_be_positive(self):
        """Test that window_seconds must be positive."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match=r"window_seconds.*positive"):
            TimeSeriesOverlay(data=data, times=times, window_seconds=-1.0)

        with pytest.raises(ValueError, match=r"window_seconds.*positive"):
            TimeSeriesOverlay(data=data, times=times, window_seconds=0.0)

    def test_alpha_must_be_in_valid_range(self):
        """Test that alpha must be in [0, 1]."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match=r"alpha.*0.*1"):
            TimeSeriesOverlay(data=data, times=times, alpha=-0.1)

        with pytest.raises(ValueError, match=r"alpha.*0.*1"):
            TimeSeriesOverlay(data=data, times=times, alpha=1.5)

    def test_times_must_be_finite(self):
        """Test that times must not contain NaN or Inf."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times_with_nan = np.array([0.0, np.nan, 1.0])

        with pytest.raises(ValueError, match=r"finite|NaN|Inf"):
            TimeSeriesOverlay(data=data, times=times_with_nan)

    def test_data_allows_nan_but_not_inf(self):
        """Test that data allows NaN (gaps) but not Inf."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        # NaN should be allowed (creates gaps in line)
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        overlay = TimeSeriesOverlay(data=data_with_nan, times=times)
        assert np.isnan(overlay.data[2])

        # Inf should raise error
        data_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        with pytest.raises(ValueError, match="Inf"):
            TimeSeriesOverlay(data=data_with_inf, times=times)

    def test_data_must_have_at_least_one_sample(self):
        """Test that data must have at least one sample."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        empty_data = np.array([])
        empty_times = np.array([])

        with pytest.raises(ValueError, match="at least 1 sample"):
            TimeSeriesOverlay(data=empty_data, times=empty_times)

    def test_linewidth_must_be_positive(self):
        """Test that linewidth must be positive."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match=r"linewidth.*positive"):
            TimeSeriesOverlay(data=data, times=times, linewidth=-1.0)

        with pytest.raises(ValueError, match=r"linewidth.*positive"):
            TimeSeriesOverlay(data=data, times=times, linewidth=0.0)


class TestTimeSeriesDataContainer:
    """Test TimeSeriesData internal container."""

    def test_get_window_slice_basic(self):
        """Test basic window extraction."""
        from neurospatial.animation.overlays import TimeSeriesData

        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        times = np.arange(10.0)
        start_indices = np.array([0, 2, 4])  # Precomputed for 3 frames
        end_indices = np.array([4, 6, 8])  # Window of 4 samples each

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=start_indices,
            end_indices=end_indices,
            label="Test",
            color="white",
            window_seconds=4.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=9.0,
            use_global_limits=True,
            interp="linear",
        )

        # Frame 0: indices [0:4]
        y_slice, t_slice = ts_data.get_window_slice(0)
        assert_array_equal(y_slice, np.array([0.0, 1.0, 2.0, 3.0]))
        assert_array_equal(t_slice, np.array([0.0, 1.0, 2.0, 3.0]))

        # Frame 1: indices [2:6]
        y_slice, t_slice = ts_data.get_window_slice(1)
        assert_array_equal(y_slice, np.array([2.0, 3.0, 4.0, 5.0]))
        assert_array_equal(t_slice, np.array([2.0, 3.0, 4.0, 5.0]))

        # Frame 2: indices [4:8]
        y_slice, t_slice = ts_data.get_window_slice(2)
        assert_array_equal(y_slice, np.array([4.0, 5.0, 6.0, 7.0]))
        assert_array_equal(t_slice, np.array([4.0, 5.0, 6.0, 7.0]))

    def test_get_cursor_value_linear(self):
        """Test linear interpolation for cursor value."""
        from neurospatial.animation.overlays import TimeSeriesData

        data = np.array([0.0, 10.0, 20.0])
        times = np.array([0.0, 1.0, 2.0])

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=np.array([0]),
            end_indices=np.array([3]),
            label="Test",
            color="white",
            window_seconds=2.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=20.0,
            use_global_limits=True,
            interp="linear",
        )

        # At exact sample points
        assert ts_data.get_cursor_value(0.0) == pytest.approx(0.0)
        assert ts_data.get_cursor_value(1.0) == pytest.approx(10.0)
        assert ts_data.get_cursor_value(2.0) == pytest.approx(20.0)

        # Linear interpolation between points
        assert ts_data.get_cursor_value(0.5) == pytest.approx(5.0)
        assert ts_data.get_cursor_value(1.5) == pytest.approx(15.0)

    def test_get_cursor_value_nearest(self):
        """Test nearest-neighbor interpolation for cursor value."""
        from neurospatial.animation.overlays import TimeSeriesData

        data = np.array([0.0, 10.0, 20.0])
        times = np.array([0.0, 1.0, 2.0])

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=np.array([0]),
            end_indices=np.array([3]),
            label="Test",
            color="white",
            window_seconds=2.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=20.0,
            use_global_limits=True,
            interp="nearest",
        )

        # At exact sample points
        assert ts_data.get_cursor_value(0.0) == pytest.approx(0.0)
        assert ts_data.get_cursor_value(1.0) == pytest.approx(10.0)
        assert ts_data.get_cursor_value(2.0) == pytest.approx(20.0)

        # Nearest neighbor (0.4 closer to 0.0, 0.6 closer to 1.0)
        assert ts_data.get_cursor_value(0.4) == pytest.approx(0.0)
        assert ts_data.get_cursor_value(0.6) == pytest.approx(10.0)
        assert ts_data.get_cursor_value(1.4) == pytest.approx(10.0)
        assert ts_data.get_cursor_value(1.6) == pytest.approx(20.0)

    def test_get_cursor_value_out_of_range(self):
        """Test cursor value returns None when outside data range."""
        from neurospatial.animation.overlays import TimeSeriesData

        data = np.array([0.0, 10.0, 20.0])
        times = np.array([1.0, 2.0, 3.0])

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=np.array([0]),
            end_indices=np.array([3]),
            label="Test",
            color="white",
            window_seconds=2.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=20.0,
            use_global_limits=True,
            interp="linear",
        )

        # Before data range
        assert ts_data.get_cursor_value(0.5) is None
        # After data range
        assert ts_data.get_cursor_value(3.5) is None


class TestTimeSeriesOverlayConvertToData:
    """Test TimeSeriesOverlay.convert_to_data() method."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims = 2
            dimension_ranges = np.array([[0.0, 100.0], [0.0, 100.0]])

        return MockEnv()

    def test_convert_to_data_basic(self, mock_env):
        """Test basic conversion to TimeSeriesData."""
        from neurospatial.animation.overlays import (
            TimeSeriesData,
            TimeSeriesOverlay,
        )

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed",
            window_seconds=0.5,
        )

        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = 3

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert isinstance(result, TimeSeriesData)
        assert_array_equal(result.data, data)
        assert_array_equal(result.times, times)
        assert result.label == "Speed"
        assert result.window_seconds == 0.5
        assert len(result.start_indices) == n_frames
        assert len(result.end_indices) == n_frames

    def test_convert_to_data_precomputes_indices(self, mock_env):
        """Test that convert_to_data precomputes window indices correctly."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # 10 samples spanning 0-1 seconds
        data = np.arange(10.0)
        times = np.linspace(0.0, 1.0, 10)  # 0.0, 0.111, 0.222, ..., 1.0
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            window_seconds=0.4,  # ±0.2 seconds
        )

        # 5 frames at 0.0, 0.25, 0.5, 0.75, 1.0
        frame_times = np.linspace(0.0, 1.0, 5)
        n_frames = 5

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Verify indices are computed for each frame
        assert len(result.start_indices) == 5
        assert len(result.end_indices) == 5

        # At frame_time=0.5, window is [0.3, 0.7]
        # Check that slicing gives correct subset
        mid_frame_idx = 2
        y_slice, t_slice = result.get_window_slice(mid_frame_idx)

        # Window should contain samples with times in [0.3, 0.7]
        # Times[3] ≈ 0.333, times[6] ≈ 0.667
        assert len(y_slice) > 0
        assert np.all((t_slice >= 0.3 - 1e-9) & (t_slice <= 0.7 + 1e-9))

    def test_convert_to_data_computes_global_limits(self, mock_env):
        """Test that global limits are computed from data."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        frame_times = np.array([0.0, 0.5, 1.0])
        result = overlay.convert_to_data(frame_times, 3, mock_env)

        assert result.global_vmin == pytest.approx(10.0)
        assert result.global_vmax == pytest.approx(50.0)

    def test_convert_to_data_respects_explicit_limits(self, mock_env):
        """Test that explicit vmin/vmax override computed limits."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([10.0, 20.0, 30.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            vmin=0.0,
            vmax=100.0,
        )

        frame_times = np.array([0.0, 0.5, 1.0])
        result = overlay.convert_to_data(frame_times, 3, mock_env)

        assert result.global_vmin == pytest.approx(0.0)
        assert result.global_vmax == pytest.approx(100.0)

    def test_convert_to_data_with_normalization(self, mock_env):
        """Test that normalize=True scales data to [0, 1]."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([0.0, 50.0, 100.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            normalize=True,
        )

        frame_times = np.array([0.0, 0.5, 1.0])
        result = overlay.convert_to_data(frame_times, 3, mock_env)

        # After normalization, data should be [0.0, 0.5, 1.0]
        assert_array_almost_equal(result.data, np.array([0.0, 0.5, 1.0]))
        # Limits should be [0, 1] after normalization
        assert result.global_vmin == pytest.approx(0.0)
        assert result.global_vmax == pytest.approx(1.0)

    def test_convert_to_data_nan_handling(self, mock_env):
        """Test that NaN values are preserved (create gaps)."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, np.nan, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        frame_times = np.array([0.0, 0.5, 1.0])
        result = overlay.convert_to_data(frame_times, 3, mock_env)

        # NaN should be preserved
        assert np.isnan(result.data[1])

        # Global limits should be computed from finite values only
        assert result.global_vmin == pytest.approx(1.0)
        assert result.global_vmax == pytest.approx(3.0)


class TestTimeSeriesDataInOverlayData:
    """Test that TimeSeriesData is properly integrated into OverlayData."""

    def test_overlay_data_has_timeseries_field(self):
        """Test that OverlayData has a timeseries field."""
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData()
        assert hasattr(overlay_data, "timeseries")
        assert isinstance(overlay_data.timeseries, list)
        assert len(overlay_data.timeseries) == 0

    def test_convert_overlays_dispatches_timeseries(self):
        """Test that _convert_overlays_to_data handles TimeSeriesOverlay."""
        from neurospatial.animation.overlays import (
            TimeSeriesData,
            TimeSeriesOverlay,
            _convert_overlays_to_data,
        )

        class MockEnv:
            n_dims = 2
            dimension_ranges = np.array([[0.0, 100.0], [0.0, 100.0]])

            def __init__(self) -> None:
                self.regions: dict[str, object] = {}

        env = MockEnv()

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times, label="Speed")

        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = 3

        result = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=env,
        )

        assert len(result.timeseries) == 1
        assert isinstance(result.timeseries[0], TimeSeriesData)
        assert result.timeseries[0].label == "Speed"


class TestTimeSeriesGroupingHelpers:
    """Test helper functions for grouping time series overlays."""

    def test_group_timeseries_single_no_group(self):
        """Test grouping a single overlay without explicit group."""
        from neurospatial.animation._timeseries import _group_timeseries
        from neurospatial.animation.overlays import TimeSeriesData

        ts_data = TimeSeriesData(
            data=np.array([1.0, 2.0, 3.0]),
            times=np.array([0.0, 0.5, 1.0]),
            start_indices=np.array([0]),
            end_indices=np.array([3]),
            label="Speed",
            color="cyan",
            window_seconds=2.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,  # No group
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=3.0,
            use_global_limits=True,
            interp="linear",
        )

        groups = _group_timeseries([ts_data])

        # Single overlay without group should be in its own group
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] is ts_data

    def test_group_timeseries_multiple_no_groups(self):
        """Test that overlays without groups create separate rows."""
        from neurospatial.animation._timeseries import _group_timeseries
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0]),
                end_indices=np.array([3]),
                label=label,
                color="cyan",
                window_seconds=2.0,
                linewidth=1.0,
                alpha=1.0,
                group=None,  # No group
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        ts_data1 = make_ts_data("Speed")
        ts_data2 = make_ts_data("Accel")

        groups = _group_timeseries([ts_data1, ts_data2])

        # Two overlays without groups should be in separate groups
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1

    def test_group_timeseries_same_group_overlaid(self):
        """Test that overlays with same group are placed together."""
        from neurospatial.animation._timeseries import _group_timeseries
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, group):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0]),
                end_indices=np.array([3]),
                label=label,
                color="cyan",
                window_seconds=2.0,
                linewidth=1.0,
                alpha=1.0,
                group=group,
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        ts_data1 = make_ts_data("Speed", "kinematics")
        ts_data2 = make_ts_data("Accel", "kinematics")

        groups = _group_timeseries([ts_data1, ts_data2])

        # Two overlays with same group should be in single group
        assert len(groups) == 1
        assert len(groups[0]) == 2
        # Use identity check (any with is) instead of 'in' for numpy array members
        assert any(item is ts_data1 for item in groups[0])
        assert any(item is ts_data2 for item in groups[0])

    def test_group_timeseries_mixed_groups(self):
        """Test mixed grouping: some overlaid, some stacked."""
        from neurospatial.animation._timeseries import _group_timeseries
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, group):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0]),
                end_indices=np.array([3]),
                label=label,
                color="cyan",
                window_seconds=2.0,
                linewidth=1.0,
                alpha=1.0,
                group=group,
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        ts_data1 = make_ts_data("Speed", "kinematics")
        ts_data2 = make_ts_data("Accel", "kinematics")
        ts_data3 = make_ts_data("LFP", None)  # Separate

        groups = _group_timeseries([ts_data1, ts_data2, ts_data3])

        # Should have 2 groups: kinematics (2 items), None (1 item)
        assert len(groups) == 2

    def test_get_group_index(self):
        """Test _get_group_index returns correct index."""
        from neurospatial.animation._timeseries import (
            _get_group_index,
            _group_timeseries,
        )
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, group):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0]),
                end_indices=np.array([3]),
                label=label,
                color="cyan",
                window_seconds=2.0,
                linewidth=1.0,
                alpha=1.0,
                group=group,
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        ts_data1 = make_ts_data("Speed", "kinematics")
        ts_data2 = make_ts_data("Accel", "kinematics")
        ts_data3 = make_ts_data("LFP", None)

        groups = _group_timeseries([ts_data1, ts_data2, ts_data3])

        # Items in same group should have same index
        idx1 = _get_group_index(ts_data1, groups)
        idx2 = _get_group_index(ts_data2, groups)
        idx3 = _get_group_index(ts_data3, groups)

        assert idx1 == idx2
        assert idx1 != idx3


class TestTimeSeriesArtistManager:
    """Test TimeSeriesArtistManager for matplotlib rendering."""

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample TimeSeriesData for testing."""
        from neurospatial.animation.overlays import TimeSeriesData

        return TimeSeriesData(
            data=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            start_indices=np.array([0, 1, 2]),
            end_indices=np.array([3, 4, 5]),
            label="Speed",
            color="cyan",
            window_seconds=0.5,
            linewidth=1.5,
            alpha=0.8,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=40.0,
            use_global_limits=True,
            interp="linear",
        )

    def test_artist_manager_create(self, sample_timeseries_data):
        """Test TimeSeriesArtistManager.create() creates figure and artists."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Check manager was created with correct structure
        assert len(manager.axes) == 1  # One group = one axes
        assert len(manager.lines) == 1  # One line per overlay
        assert len(manager.cursors) == 1  # One cursor per axes
        assert "Speed" in manager.lines or 0 in manager.lines

        plt.close(fig)

    def test_artist_manager_create_multiple_rows(self):
        """Test manager creates multiple axes for ungrouped overlays."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0, 1, 2]),
                end_indices=np.array([1, 2, 3]),
                label=label,
                color="cyan",
                window_seconds=1.0,
                linewidth=1.0,
                alpha=1.0,
                group=None,  # Separate groups
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        fig = plt.figure(figsize=(4, 6))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[make_ts_data("Speed"), make_ts_data("Accel")],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Two ungrouped overlays should create 2 axes
        assert len(manager.axes) == 2
        assert len(manager.cursors) == 2

        plt.close(fig)

    def test_artist_manager_create_overlaid(self):
        """Test overlays with same group share axes."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, color):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0, 1, 2]),
                end_indices=np.array([1, 2, 3]),
                label=label,
                color=color,
                window_seconds=1.0,
                linewidth=1.0,
                alpha=1.0,
                group="kinematics",  # Same group
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[
                make_ts_data("Speed", "cyan"),
                make_ts_data("Accel", "orange"),
            ],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Two overlays in same group should create only 1 axes
        assert len(manager.axes) == 1
        # But two lines
        assert len(manager.lines) == 2
        assert len(manager.cursors) == 1

        plt.close(fig)

    def test_artist_manager_update(self, sample_timeseries_data):
        """Test update() changes line data."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Update to frame 0
        manager.update(0, [sample_timeseries_data])

        # Get line data
        line_key = (
            "Speed" if "Speed" in manager.lines else next(iter(manager.lines.keys()))
        )
        line = manager.lines[line_key]
        x_data, y_data = line.get_data()

        # Should have data from window slice
        assert len(x_data) > 0
        assert len(y_data) > 0

        # Update to frame 1
        manager.update(1, [sample_timeseries_data])
        x_data2, y_data2 = line.get_data()

        # Data should have changed
        assert not np.array_equal(x_data, x_data2) or not np.array_equal(
            y_data, y_data2
        )

        plt.close(fig)

    def test_artist_manager_cursor_update(self, sample_timeseries_data):
        """Test that cursor position updates with frame."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Update to frame 0 (time=0.0)
        manager.update(0, [sample_timeseries_data])
        cursor_x0 = manager.cursors[0].get_xdata()

        # Update to frame 2 (time=1.0)
        manager.update(2, [sample_timeseries_data])
        cursor_x2 = manager.cursors[0].get_xdata()

        # Cursor should be at different x positions
        assert not np.allclose(cursor_x0, cursor_x2)

        plt.close(fig)

    def test_artist_manager_dark_theme(self, sample_timeseries_data):
        """Test dark theme styling is applied."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Check figure background is dark
        assert fig.get_facecolor()[:3] != (1.0, 1.0, 1.0)  # Not white

        # Check axes background
        ax = manager.axes[0]
        ax_facecolor = ax.get_facecolor()[:3]
        assert ax_facecolor != (1.0, 1.0, 1.0)  # Not white

        plt.close(fig)


class TestTimeSeriesGroupConflictWarnings:
    """Test warnings for conflicting parameters in same group."""

    def test_conflicting_window_seconds_warning(self):
        """Test warning when same group has different window_seconds."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, window_seconds):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0, 1, 2]),
                end_indices=np.array([1, 2, 3]),
                label=label,
                color="cyan",
                window_seconds=window_seconds,
                linewidth=1.0,
                alpha=1.0,
                group="kinematics",  # Same group
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeSeriesArtistManager.create(
                fig=fig,
                timeseries_data=[
                    make_ts_data("Speed", 2.0),
                    make_ts_data("Accel", 5.0),
                ],
                frame_times=frame_times,
                dark_theme=True,
            )

            # Should warn about conflicting window_seconds
            assert len(w) >= 1
            assert any(
                "window_seconds" in str(warning.message).lower() for warning in w
            )

        plt.close(fig)

    def test_mixed_normalize_warning(self):
        """Test warning when same group has mixed normalize settings."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label, normalize):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0]),
                times=np.array([0.0, 0.5, 1.0]),
                start_indices=np.array([0, 1, 2]),
                end_indices=np.array([1, 2, 3]),
                label=label,
                color="cyan",
                window_seconds=2.0,
                linewidth=1.0,
                alpha=1.0,
                group="kinematics",  # Same group
                normalize=normalize,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.array([0.0, 0.5, 1.0])

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeSeriesArtistManager.create(
                fig=fig,
                timeseries_data=[
                    make_ts_data("Speed", True),
                    make_ts_data("Accel", False),
                ],
                frame_times=frame_times,
                dark_theme=True,
            )

            # Should warn about mixed normalize
            assert len(w) >= 1
            assert any("normalize" in str(warning.message).lower() for warning in w)

        plt.close(fig)


@pytest.fixture
def _napari_viewer():
    """Create a napari viewer for testing, skip if unavailable."""
    napari = pytest.importorskip("napari")
    try:
        from qtpy.QtWidgets import QApplication

        # Create application if needed
        app = QApplication.instance()
        if app is None:
            pytest.skip("No Qt application available")
    except ImportError:
        pytest.skip("Qt not available")

    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


class TestTimeSeriesNapariDockWidget:
    """Test napari dock widget integration for time series overlays."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(50, 2) * 100
        env = Environment.from_samples(positions, bin_size=10.0)
        return env

    @pytest.fixture
    def sample_fields(self, simple_env):
        """Create sample field data for testing."""
        n_frames = 10
        n_bins = simple_env.n_bins
        return [np.random.rand(n_bins) for _ in range(n_frames)]

    @pytest.fixture
    def sample_timeseries_overlay(self):
        """Create sample time series overlay for testing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 100, 100)
        times = np.linspace(0, 1, 100)
        return TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=0.5,
        )

    def test_timeseries_dock_widget_added(
        self, simple_env, sample_fields, sample_timeseries_overlay, _napari_viewer
    ):
        """Test that time series dock widget is added to napari viewer."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        viewer = _napari_viewer

        # Convert overlays to data
        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        # Import and call the dock widget function
        from neurospatial.animation.backends.napari_backend import (
            _add_timeseries_dock,
        )

        _add_timeseries_dock(
            viewer=viewer,
            timeseries_data=overlay_data.timeseries,
            frame_times=frame_times,
        )

        # Check dock widget was added
        dock_widgets = viewer.window._dock_widgets
        assert len(dock_widgets) > 0
        # Check that Time Series widget exists
        widget_names = [dw.title() for dw in dock_widgets.values()]
        assert "Time Series" in widget_names

    def test_timeseries_dock_updates_on_frame_change(
        self, simple_env, sample_fields, sample_timeseries_overlay, _napari_viewer
    ):
        """Test that time series plot updates when frame changes."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        viewer = _napari_viewer

        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        from neurospatial.animation.backends.napari_backend import (
            _add_timeseries_dock,
        )

        _add_timeseries_dock(
            viewer=viewer,
            timeseries_data=overlay_data.timeseries,
            frame_times=frame_times,
        )

        # Simulate adding an image layer so dims has time dimension
        viewer.add_image(np.random.rand(len(sample_fields), 50, 50))

        # Change frame and verify no errors
        viewer.dims.current_step = (0,)
        viewer.dims.current_step = (5,)
        viewer.dims.current_step = (9,)

        # If we get here without error, the callback is working

    def test_render_napari_with_timeseries(
        self, simple_env, sample_fields, sample_timeseries_overlay
    ):
        """Test that render_napari correctly adds time series dock widget."""
        pytest.importorskip("napari")
        try:
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is None:
                pytest.skip("No Qt application available")
        except ImportError:
            pytest.skip("Qt not available")

        from neurospatial.animation.overlays import _convert_overlays_to_data

        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            env=simple_env,
            fields=sample_fields,
            overlay_data=overlay_data,
        )

        try:
            # Check dock widget was added
            dock_widgets = viewer.window._dock_widgets
            widget_names = [dw.title() for dw in dock_widgets.values()]
            assert "Time Series" in widget_names

        finally:
            viewer.close()

    def test_no_timeseries_dock_when_empty(self, simple_env, sample_fields):
        """Test that no dock widget is added when no time series overlays."""
        pytest.importorskip("napari")
        try:
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is None:
                pytest.skip("No Qt application available")
        except ImportError:
            pytest.skip("Qt not available")

        from neurospatial.animation.overlays import OverlayData

        # Create empty overlay data
        overlay_data = OverlayData()

        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            env=simple_env,
            fields=sample_fields,
            overlay_data=overlay_data,
        )

        try:
            # Check dock widgets - should not have Time Series
            dock_widgets = viewer.window._dock_widgets
            widget_names = [dw.title() for dw in dock_widgets.values()]
            # Time Series should NOT be present
            assert "Time Series" not in widget_names

        finally:
            viewer.close()
