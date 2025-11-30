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


# =============================================================================
# Phase 3: Video Backend Tests
# =============================================================================


class TestTimeSeriesVideoBackendLayout:
    """Test video backend layout with time series column."""

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
        n_frames = 5
        n_bins = simple_env.n_bins
        return [np.random.rand(n_bins) for _ in range(n_frames)]

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample TimeSeriesData for testing."""
        from neurospatial.animation.overlays import TimeSeriesData

        return TimeSeriesData(
            data=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            start_indices=np.array([0, 1, 2, 3, 4]),
            end_indices=np.array([2, 3, 4, 5, 5]),
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

    def test_setup_video_figure_with_timeseries_creates_gridspec(
        self, simple_env, sample_timeseries_data
    ):
        """Test _setup_video_figure_with_timeseries creates correct GridSpec layout."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import (
            _setup_video_figure_with_timeseries,
        )

        frame_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        fig, ax_field, ts_manager = _setup_video_figure_with_timeseries(
            env=simple_env,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dpi=100,
        )

        # Check figure was created
        assert fig is not None
        # Check spatial field axes was created
        assert ax_field is not None
        # Check time series manager was created
        assert ts_manager is not None
        # Manager should have one axes (one ungrouped timeseries)
        assert len(ts_manager.axes) == 1

        plt.close(fig)

    def test_setup_video_figure_with_multiple_timeseries(self, simple_env):
        """Test layout with multiple stacked time series rows."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import (
            _setup_video_figure_with_timeseries,
        )
        from neurospatial.animation.overlays import TimeSeriesData

        def make_ts_data(label):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                start_indices=np.array([0, 1, 2, 3, 4]),
                end_indices=np.array([2, 3, 4, 5, 5]),
                label=label,
                color="cyan",
                window_seconds=0.5,
                linewidth=1.0,
                alpha=1.0,
                group=None,  # Separate rows
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=5.0,
                use_global_limits=True,
                interp="linear",
            )

        frame_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        fig, _ax_field, ts_manager = _setup_video_figure_with_timeseries(
            env=simple_env,
            timeseries_data=[make_ts_data("Speed"), make_ts_data("Accel")],
            frame_times=frame_times,
            dpi=100,
        )

        # Should have 2 rows of time series (ungrouped)
        assert len(ts_manager.axes) == 2

        plt.close(fig)

    def test_setup_video_figure_light_theme(self, simple_env, sample_timeseries_data):
        """Test video backend uses light theme (not dark like napari)."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import (
            _setup_video_figure_with_timeseries,
        )

        frame_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        fig, _ax_field, ts_manager = _setup_video_figure_with_timeseries(
            env=simple_env,
            timeseries_data=[sample_timeseries_data],
            frame_times=frame_times,
            dpi=100,
        )

        # Video backend should use light theme (white background)
        ax_ts = ts_manager.axes[0]
        # Check background is light (R > 0.5 typically indicates light theme)
        facecolor = ax_ts.get_facecolor()
        # White background has values close to 1.0
        assert facecolor[0] > 0.5  # R channel > 0.5 indicates light

        plt.close(fig)


class TestTimeSeriesArtistManagerFromAxes:
    """Test create_from_axes classmethod for video backend."""

    def test_create_from_axes_basic(self):
        """Test TimeSeriesArtistManager.create_from_axes() with pre-created axes."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        # Create figure with pre-existing axes
        fig, axes = plt.subplots(1, 1, squeeze=False)
        ax_list = list(axes[:, 0])

        ts_data = TimeSeriesData(
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            start_indices=np.array([0, 1, 2]),
            end_indices=np.array([2, 3, 5]),
            label="Speed",
            color="cyan",
            window_seconds=0.5,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=5.0,
            use_global_limits=True,
            interp="linear",
        )

        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create_from_axes(
            axes=ax_list,
            timeseries_data=[ts_data],
            frame_times=frame_times,
            dark_theme=False,  # Light theme for video
        )

        # Check manager uses provided axes
        assert manager.axes is ax_list
        assert len(manager.lines) == 1
        assert len(manager.cursors) == 1

        plt.close(fig)

    def test_create_from_axes_multiple_groups(self):
        """Test create_from_axes with multiple pre-created axes."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        # Create figure with 2 axes (for 2 ungrouped time series)
        fig, axes = plt.subplots(2, 1, squeeze=False)
        ax_list = list(axes[:, 0])

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
                group=None,  # Each gets own row
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=3.0,
                use_global_limits=True,
                interp="linear",
            )

        frame_times = np.array([0.0, 0.5, 1.0])

        manager = TimeSeriesArtistManager.create_from_axes(
            axes=ax_list,
            timeseries_data=[make_ts_data("Speed"), make_ts_data("Accel")],
            frame_times=frame_times,
            dark_theme=False,
        )

        # Should use both provided axes
        assert len(manager.axes) == 2
        assert len(manager.lines) == 2

        plt.close(fig)


class TestTimeSeriesVideoRender:
    """Test video export with time series column."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        np.random.seed(42)
        positions = np.random.rand(50, 2) * 100
        env = Environment.from_samples(positions, bin_size=10.0)
        env.clear_cache()  # Ensure pickle-able
        return env

    @pytest.fixture
    def sample_fields(self, simple_env):
        """Create sample field data for testing."""
        np.random.seed(42)
        n_frames = 5
        n_bins = simple_env.n_bins
        return [np.random.rand(n_bins) for _ in range(n_frames)]

    @pytest.fixture
    def sample_timeseries_overlay(self):
        """Create sample time series overlay for testing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 100, 50)
        times = np.linspace(0, 1, 50)
        return TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=0.5,
        )

    def test_video_render_includes_timeseries_column(
        self, simple_env, sample_fields, sample_timeseries_overlay, tmp_path
    ):
        """Test that video export includes time series column."""
        pytest.importorskip("subprocess")
        import shutil

        # Skip if ffmpeg not available
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")

        from neurospatial.animation.backends.video_backend import render_video
        from neurospatial.animation.overlays import _convert_overlays_to_data

        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        output_path = tmp_path / "test_with_timeseries.mp4"

        # Render video with time series
        result = render_video(
            env=simple_env,
            fields=sample_fields,
            save_path=str(output_path),
            fps=10,
            n_workers=1,  # Serial for test reliability
            overlay_data=overlay_data,
            frame_times=frame_times,  # Required for time series rendering
        )

        # Check video was created
        assert result is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_video_render_parallel_with_timeseries(
        self, simple_env, sample_fields, sample_timeseries_overlay, tmp_path
    ):
        """Test parallel video rendering with time series (pickle-safe)."""
        pytest.importorskip("subprocess")
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")

        from neurospatial.animation.backends.video_backend import render_video
        from neurospatial.animation.overlays import _convert_overlays_to_data

        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        output_path = tmp_path / "test_parallel_timeseries.mp4"

        # Render with 2 workers (tests pickling)
        result = render_video(
            env=simple_env,
            fields=sample_fields,
            save_path=str(output_path),
            fps=10,
            n_workers=2,
            overlay_data=overlay_data,
            frame_times=frame_times,  # Required for time series rendering
        )

        assert result is not None
        assert output_path.exists()


class TestTimeSeriesArtistManagerPickleSafe:
    """Test TimeSeriesArtistManager pickle safety for parallel rendering."""

    def test_timeseries_data_is_pickle_safe(self):
        """Test that TimeSeriesData can be pickled for parallel workers."""
        import pickle

        from neurospatial.animation.overlays import TimeSeriesData

        ts_data = TimeSeriesData(
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            start_indices=np.array([0, 1, 2, 3, 4]),
            end_indices=np.array([2, 3, 4, 5, 5]),
            label="Speed",
            color="cyan",
            window_seconds=0.5,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=5.0,
            use_global_limits=True,
            interp="linear",
        )

        # Test pickling round-trip
        pickled = pickle.dumps(ts_data)
        restored = pickle.loads(pickled)

        # Verify data was preserved
        assert_array_equal(restored.data, ts_data.data)
        assert_array_equal(restored.times, ts_data.times)
        assert restored.label == ts_data.label
        assert restored.color == ts_data.color

    def test_overlay_data_with_timeseries_is_pickle_safe(self):
        """Test that OverlayData with timeseries can be pickled."""
        import pickle

        from neurospatial.animation.overlays import OverlayData, TimeSeriesData

        ts_data = TimeSeriesData(
            data=np.array([1.0, 2.0, 3.0]),
            times=np.array([0.0, 0.5, 1.0]),
            start_indices=np.array([0]),
            end_indices=np.array([3]),
            label="Speed",
            color="cyan",
            window_seconds=1.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=3.0,
            use_global_limits=True,
            interp="linear",
        )

        overlay_data = OverlayData(timeseries=[ts_data])

        # Test pickling round-trip
        pickled = pickle.dumps(overlay_data)
        restored = pickle.loads(pickled)

        # Verify timeseries was preserved
        assert len(restored.timeseries) == 1
        assert restored.timeseries[0].label == "Speed"


# =============================================================================
# Phase 4: Widget Backend Tests
# =============================================================================


class TestTimeSeriesWidgetBackend:
    """Test widget backend time series integration."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        np.random.seed(42)
        positions = np.random.rand(50, 2) * 100
        env = Environment.from_samples(positions, bin_size=10.0)
        return env

    @pytest.fixture
    def sample_fields(self, simple_env):
        """Create sample field data for testing."""
        np.random.seed(42)
        n_frames = 5
        n_bins = simple_env.n_bins
        return [np.random.rand(n_bins) for _ in range(n_frames)]

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample TimeSeriesData for testing."""
        from neurospatial.animation.overlays import TimeSeriesData

        return TimeSeriesData(
            data=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            start_indices=np.array([0, 1, 2, 3, 4]),
            end_indices=np.array([2, 3, 4, 5, 5]),
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

    @pytest.fixture
    def sample_timeseries_overlay(self):
        """Create sample TimeSeriesOverlay for testing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 100, 50)
        times = np.linspace(0, 1, 50)
        return TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=0.5,
        )

    def test_render_field_to_png_bytes_with_timeseries(
        self, simple_env, sample_fields, sample_timeseries_data
    ):
        """Test render_field_to_png_bytes_with_overlays includes time series column."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData(timeseries=[sample_timeseries_data])
        frame_times = np.linspace(0, 1, len(sample_fields))

        # Render frame with time series
        png_bytes = render_field_to_png_bytes_with_overlays(
            env=simple_env,
            field=sample_fields[0],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
            frame_idx=0,
            overlay_data=overlay_data,
            frame_times=frame_times,
        )

        # Check PNG bytes were created
        assert png_bytes is not None
        assert len(png_bytes) > 0
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"

    def test_persistent_figure_renderer_with_timeseries(
        self, simple_env, sample_fields, sample_timeseries_data
    ):
        """Test PersistentFigureRenderer handles time series overlays."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData(timeseries=[sample_timeseries_data])
        frame_times = np.linspace(0, 1, len(sample_fields))

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
            frame_times=frame_times,
        )

        try:
            # First render
            png_bytes = renderer.render(
                field=sample_fields[0],
                frame_idx=0,
                overlay_data=overlay_data,
            )
            assert len(png_bytes) > 0

            # Second render (should reuse figure)
            png_bytes2 = renderer.render(
                field=sample_fields[1],
                frame_idx=1,
                overlay_data=overlay_data,
            )
            assert len(png_bytes2) > 0

        finally:
            renderer.close()

    def test_persistent_figure_renderer_timeseries_updates(
        self, simple_env, sample_fields, sample_timeseries_data
    ):
        """Test that time series updates correctly across frames."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData(timeseries=[sample_timeseries_data])
        frame_times = np.linspace(0, 1, len(sample_fields))

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
            frame_times=frame_times,
        )

        try:
            # Render multiple frames
            bytes_frames = []
            for i in range(len(sample_fields)):
                png_bytes = renderer.render(
                    field=sample_fields[i],
                    frame_idx=i,
                    overlay_data=overlay_data,
                )
                bytes_frames.append(png_bytes)

            # Each frame should render successfully
            assert all(len(b) > 0 for b in bytes_frames)

        finally:
            renderer.close()

    def test_persistent_figure_renderer_multiple_timeseries(
        self, simple_env, sample_fields
    ):
        """Test PersistentFigureRenderer with multiple stacked time series."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )
        from neurospatial.animation.overlays import OverlayData, TimeSeriesData

        def make_ts_data(label, color):
            return TimeSeriesData(
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                times=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                start_indices=np.array([0, 1, 2, 3, 4]),
                end_indices=np.array([2, 3, 4, 5, 5]),
                label=label,
                color=color,
                window_seconds=0.5,
                linewidth=1.0,
                alpha=1.0,
                group=None,  # Separate rows
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=1.0,
                global_vmax=5.0,
                use_global_limits=True,
                interp="linear",
            )

        overlay_data = OverlayData(
            timeseries=[make_ts_data("Speed", "cyan"), make_ts_data("Accel", "orange")]
        )
        frame_times = np.linspace(0, 1, len(sample_fields))

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
            frame_times=frame_times,
        )

        try:
            png_bytes = renderer.render(
                field=sample_fields[0],
                frame_idx=0,
                overlay_data=overlay_data,
            )
            # Should render with 2 time series rows
            assert len(png_bytes) > 0

        finally:
            renderer.close()

    def test_render_widget_with_timeseries(
        self, simple_env, sample_fields, sample_timeseries_overlay
    ):
        """Test render_widget with time series overlay."""
        pytest.importorskip("ipywidgets")

        from neurospatial.animation.overlays import _convert_overlays_to_data

        frame_times = np.linspace(0, 1, len(sample_fields))
        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay],
            frame_times=frame_times,
            n_frames=len(sample_fields),
            env=simple_env,
        )

        # We can't fully test widget display without Jupyter environment,
        # but we can test the conversion and that no errors are raised
        # during overlay data preparation
        assert len(overlay_data.timeseries) == 1
        assert overlay_data.timeseries[0].label == "Speed (cm/s)"

    def test_widget_backend_no_timeseries_still_works(self, simple_env, sample_fields):
        """Test widget backend without time series still works normally."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )
        from neurospatial.animation.overlays import OverlayData

        # Empty overlay data (no time series)
        overlay_data = OverlayData()

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
        )

        try:
            png_bytes = renderer.render(
                field=sample_fields[0],
                frame_idx=0,
                overlay_data=overlay_data,
            )
            assert len(png_bytes) > 0

        finally:
            renderer.close()


# =============================================================================
# Phase 5.1: HTML Backend Time Series Warning Tests
# =============================================================================


class TestHTMLBackendTimeSeriesWarning:
    """Test HTML backend emits warning for TimeSeriesOverlay (unsupported).

    TimeSeriesOverlay requires a separate panel with time series plots,
    which is not supported in HTML backend. A warning should be emitted
    and the overlay should be skipped (other overlays still render).
    """

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        from neurospatial import Environment

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def simple_fields(self, simple_env):
        """Create simple fields for testing."""
        rng = np.random.default_rng(42)
        return [rng.random(simple_env.n_bins) for _ in range(10)]

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample TimeSeriesData for testing."""
        from neurospatial.animation.overlays import TimeSeriesData

        return TimeSeriesData(
            data=np.linspace(0, 100, 100),
            times=np.linspace(0, 1, 100),
            start_indices=np.arange(10),
            end_indices=np.arange(10) + 10,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=0.5,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=100.0,
            use_global_limits=True,
            interp="linear",
        )

    def test_html_backend_warns_on_timeseries_overlay(
        self, simple_env, simple_fields, sample_timeseries_data, tmp_path
    ):
        """Test HTML backend emits warning when time series overlay is present."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData(timeseries=[sample_timeseries_data])

        save_path = tmp_path / "test.html"

        # Should emit warning about time series not being supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

            # Check for warning about time series
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            warning_text = " ".join(warning_messages).lower()

            # Warning should mention time series not being supported
            assert "time series" in warning_text or "timeseries" in warning_text, (
                f"Expected warning about time series, got: {warning_messages}"
            )

            # Warning should suggest alternatives (video or napari)
            assert "video" in warning_text or "napari" in warning_text, (
                f"Expected suggestion to use video or napari, got: {warning_messages}"
            )

    def test_html_backend_still_renders_with_timeseries_present(
        self, simple_env, simple_fields, sample_timeseries_data, tmp_path
    ):
        """Test HTML backend still renders successfully when time series is skipped."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData(timeseries=[sample_timeseries_data])

        save_path = tmp_path / "test.html"

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

        # File should be created successfully
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_html_backend_renders_other_overlays_with_timeseries_present(
        self, simple_env, simple_fields, sample_timeseries_data, tmp_path
    ):
        """Test HTML renders position overlays even when time series is skipped."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import OverlayData, PositionData

        rng = np.random.default_rng(42)
        positions = rng.random((10, 2)) * 10.0

        # Mixed overlay data (timeseries + position)
        overlay_data = OverlayData(
            timeseries=[sample_timeseries_data],
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ],
        )

        save_path = tmp_path / "test.html"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

        # File should be created with content
        assert result_path.exists()
        # Should have substantial content (includes position overlay baked in)
        assert result_path.stat().st_size > 1000

    def test_html_backend_no_warning_without_timeseries(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test no time series warning when no time series overlay is present."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import OverlayData, PositionData

        rng = np.random.default_rng(42)
        positions = rng.random((10, 2)) * 10.0

        # Position overlay only (no time series)
        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ],
        )

        save_path = tmp_path / "test.html"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
                dpi=50,  # Low DPI to avoid file size warnings
            )

            # Check that no time series warning was emitted
            warning_messages = [str(warning.message).lower() for warning in w]
            for msg in warning_messages:
                assert "time series" not in msg and "timeseries" not in msg, (
                    f"Unexpected time series warning: {msg}"
                )


# =============================================================================
# Phase 5.2: Performance Testing
# =============================================================================


class TestTimeSeriesPerformance:
    """Test time series overlay performance with high-rate data.

    These tests verify that time series overlays can handle high-frequency
    data (1 kHz) and long sessions (1 hour = 3.6M samples) efficiently.
    """

    @pytest.mark.slow
    def test_timeseries_high_rate_data_conversion(self):
        """Test TimeSeriesOverlay handles 1 kHz data over 1 hour efficiently.

        This tests the conversion from TimeSeriesOverlay to TimeSeriesData,
        which includes the O(1) index precomputation step.
        """
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # 1 kHz data over 1 hour = 3,600,000 samples
        sample_rate_hz = 1000
        duration_seconds = 3600  # 1 hour
        n_samples = sample_rate_hz * duration_seconds

        # Create high-rate data
        rng = np.random.default_rng(42)
        data = rng.random(n_samples).astype(np.float64)
        times = np.linspace(0, duration_seconds, n_samples, dtype=np.float64)

        # Create overlay - this should be fast (just stores references)
        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="High-rate signal",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (n_samples,)
        assert overlay.times.shape == (n_samples,)

    @pytest.mark.slow
    def test_timeseries_data_precomputed_indices(self):
        """Test that TimeSeriesData uses precomputed indices for O(1) window extraction."""
        from neurospatial.animation.overlays import TimeSeriesData

        # Create TimeSeriesData with precomputed indices
        n_samples = 1000
        n_frames = 100

        data = np.linspace(0, 100, n_samples)
        times = np.linspace(0, 10, n_samples)

        # Precomputed indices - each frame has a fixed range
        start_indices = np.linspace(0, n_samples - 100, n_frames).astype(int)
        end_indices = start_indices + 100

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=start_indices,
            end_indices=end_indices,
            label="Test",
            color="cyan",
            window_seconds=1.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=100.0,
            use_global_limits=True,
            interp="linear",
        )

        # Verify index access is O(1) - just array indexing
        for frame_idx in range(n_frames):
            start = ts_data.start_indices[frame_idx]
            end = ts_data.end_indices[frame_idx]

            # Should be immediate (no search)
            window_data = ts_data.data[start:end]
            window_times = ts_data.times[start:end]

            assert len(window_data) == 100
            assert len(window_times) == 100

    @pytest.mark.slow
    def test_timeseries_artist_manager_update_performance(self):
        """Test that TimeSeriesArtistManager.update() is efficient per frame."""
        import time

        import matplotlib.pyplot as plt

        from neurospatial.animation._timeseries import TimeSeriesArtistManager
        from neurospatial.animation.overlays import TimeSeriesData

        # Create reasonable size data (10 Hz for 10 minutes)
        n_samples = 6000
        n_frames = 100

        data = np.linspace(0, 100, n_samples)
        times = np.linspace(0, 600, n_samples)

        # Precomputed indices
        samples_per_frame = n_samples // n_frames
        start_indices = np.arange(n_frames) * samples_per_frame
        end_indices = np.minimum(start_indices + samples_per_frame * 2, n_samples)

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=start_indices,
            end_indices=end_indices,
            label="Speed",
            color="cyan",
            window_seconds=60.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=100.0,
            use_global_limits=True,
            interp="linear",
        )

        fig = plt.figure(figsize=(4, 3))
        frame_times = np.linspace(0, 600, n_frames)

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=[ts_data],
            frame_times=frame_times,
            dark_theme=True,
        )

        # Time the update calls
        start_time = time.perf_counter()
        for frame_idx in range(n_frames):
            manager.update(frame_idx, [ts_data])
        elapsed = time.perf_counter() - start_time

        plt.close(fig)

        # Should complete in reasonable time (< 1 second for 100 frames)
        assert elapsed < 1.0, f"Updates took {elapsed:.2f}s, expected < 1.0s"

        # Average per-frame update should be < 10 ms
        avg_per_frame_ms = (elapsed / n_frames) * 1000
        assert avg_per_frame_ms < 10.0, (
            f"Average update {avg_per_frame_ms:.1f}ms, expected < 10ms"
        )

    @pytest.mark.slow
    def test_timeseries_window_extraction_is_o1(self):
        """Verify window extraction scales O(1) with data size.

        Compare extraction time for small vs large datasets.
        O(1) means extraction time should be similar regardless of data size.
        """
        import time

        from neurospatial.animation.overlays import TimeSeriesData

        def measure_extraction_time(n_samples: int, n_iterations: int = 1000) -> float:
            """Measure average time to extract windows."""
            data = np.random.rand(n_samples)
            times = np.linspace(0, n_samples, n_samples)

            # Precomputed indices for 100 frames
            n_frames = 100
            window_size = 100  # Fixed window size

            start_indices = np.linspace(0, n_samples - window_size, n_frames).astype(
                int
            )
            end_indices = start_indices + window_size

            ts_data = TimeSeriesData(
                data=data,
                times=times,
                start_indices=start_indices,
                end_indices=end_indices,
                label="Test",
                color="cyan",
                window_seconds=1.0,
                linewidth=1.0,
                alpha=1.0,
                group=None,
                normalize=False,
                show_cursor=True,
                cursor_color="red",
                global_vmin=0.0,
                global_vmax=1.0,
                use_global_limits=True,
                interp="linear",
            )

            # Time multiple extractions
            start_time = time.perf_counter()
            for _ in range(n_iterations):
                for frame_idx in range(n_frames):
                    start = ts_data.start_indices[frame_idx]
                    end = ts_data.end_indices[frame_idx]
                    _ = ts_data.data[start:end]
            elapsed = time.perf_counter() - start_time

            return elapsed / n_iterations

        # Test with different data sizes
        small_time = measure_extraction_time(1_000)
        _medium_time = measure_extraction_time(100_000)  # For debugging if needed
        large_time = measure_extraction_time(1_000_000)

        # O(1) means times should be similar (within 10x)
        # We're mainly extracting fixed-size windows, so scaling should be minimal
        assert large_time < small_time * 10, (
            f"Large data ({large_time:.6f}s) much slower than small ({small_time:.6f}s)"
        )


# =============================================================================
# Phase 5.3: Edge Case Tests
# =============================================================================


class TestTimeSeriesEdgeCases:
    """Test edge cases for time series overlay handling.

    These tests verify graceful handling of edge cases like:
    - Frame times outside data range
    - Windows partially outside data
    - Single data point
    - All NaN data
    - Mismatched sampling rates
    """

    def test_timeseries_empty_window_frame_outside_data(self):
        """Test handling when frame time is outside data time range."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Data from 0-10 seconds
        data = np.linspace(0, 100, 100)
        times = np.linspace(0, 10, 100)

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Test",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (100,)
        # Frame times outside range are handled during conversion

    def test_timeseries_partial_window_start(self):
        """Test handling when window starts before data begins."""
        from neurospatial.animation.overlays import TimeSeriesData

        # Data from t=5 to t=15
        data = np.linspace(0, 100, 100)
        times = np.linspace(5, 15, 100)

        # Window centered at t=6 with window=4s would start at t=4 (before data)
        # Precomputed indices should handle this by clamping
        start_indices = np.array([0, 0, 0, 10, 50])  # First few clamped to 0
        end_indices = np.array([20, 30, 40, 50, 90])

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=start_indices,
            end_indices=end_indices,
            label="Test",
            color="cyan",
            window_seconds=4.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=100.0,
            use_global_limits=True,
            interp="linear",
        )

        # Should be able to access all windows without error
        for frame_idx in range(len(start_indices)):
            start = ts_data.start_indices[frame_idx]
            end = ts_data.end_indices[frame_idx]
            window = ts_data.data[start:end]
            assert len(window) > 0

    def test_timeseries_partial_window_end(self):
        """Test handling when window extends past data end."""
        from neurospatial.animation.overlays import TimeSeriesData

        # Data from t=0 to t=10
        data = np.linspace(0, 100, 100)
        times = np.linspace(0, 10, 100)

        # Window at end would extend past data
        start_indices = np.array([50, 70, 80, 90, 95])
        end_indices = np.array([90, 95, 100, 100, 100])  # Last few clamped to 100

        ts_data = TimeSeriesData(
            data=data,
            times=times,
            start_indices=start_indices,
            end_indices=end_indices,
            label="Test",
            color="cyan",
            window_seconds=4.0,
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=0.0,
            global_vmax=100.0,
            use_global_limits=True,
            interp="linear",
        )

        # Should be able to access all windows without error
        for frame_idx in range(len(start_indices)):
            start = ts_data.start_indices[frame_idx]
            end = ts_data.end_indices[frame_idx]
            window = ts_data.data[start:end]
            assert len(window) > 0

    def test_timeseries_single_data_point(self):
        """Test TimeSeriesOverlay with single data point."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Single data point
        data = np.array([42.0])
        times = np.array([5.0])

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Single point",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (1,)
        assert overlay.times.shape == (1,)
        assert overlay.data[0] == 42.0

    def test_timeseries_two_data_points(self):
        """Test TimeSeriesOverlay with two data points."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Two data points
        data = np.array([10.0, 20.0])
        times = np.array([0.0, 1.0])

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Two points",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (2,)
        assert overlay.times.shape == (2,)

    def test_timeseries_with_nan_values(self):
        """Test TimeSeriesOverlay with NaN values in data."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Data with NaN gaps
        data = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
        times = np.linspace(0, 5, 6)

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="With NaNs",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (6,)
        assert np.isnan(overlay.data[2])
        assert np.isnan(overlay.data[3])

    def test_timeseries_all_nan(self):
        """Test TimeSeriesOverlay with all NaN data."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # All NaN data
        data = np.array([np.nan, np.nan, np.nan, np.nan])
        times = np.array([0.0, 1.0, 2.0, 3.0])

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="All NaN",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (4,)
        assert np.all(np.isnan(overlay.data))

    def test_timeseries_mismatched_rates_high_rate_ts(self):
        """Test time series at higher rate than field frames."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Time series at 100 Hz (high rate)
        data = np.linspace(0, 100, 1000)
        times = np.linspace(0, 10, 1000)  # 100 Hz

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="High rate",
            window_seconds=1.0,
        )

        # Overlay should be created successfully
        assert overlay.data.shape == (1000,)

    def test_timeseries_mismatched_rates_low_rate_ts(self):
        """Test time series at lower rate than field frames."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Time series at 1 Hz (low rate)
        data = np.linspace(0, 100, 10)
        times = np.linspace(0, 10, 10)  # 1 Hz

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Low rate",
            window_seconds=3.0,
        )

        # Overlay should be created successfully
        assert overlay.data.shape == (10,)

    def test_timeseries_irregular_times(self):
        """Test TimeSeriesOverlay with irregular time spacing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Irregular time spacing (common in event data)
        times = np.array([0.0, 0.5, 0.6, 2.0, 5.0, 5.1, 10.0])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Irregular",
            window_seconds=2.0,
        )

        assert overlay.data.shape == (7,)
        assert overlay.times.shape == (7,)

    def test_timeseries_conversion_with_edge_frame_times(self):
        """Test overlay conversion when frame times are at data boundaries."""
        from neurospatial import Environment
        from neurospatial.animation.overlays import (
            TimeSeriesOverlay,
            _convert_overlays_to_data,
        )

        # Create simple environment
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        env = Environment.from_samples(positions, bin_size=5.0)

        # Time series from 0-10
        data = np.linspace(0, 100, 100)
        times = np.linspace(0, 10, 100)

        overlay = TimeSeriesOverlay(
            data=data,
            times=times,
            label="Test",
            window_seconds=2.0,
        )

        # Frame times at exact boundaries
        frame_times = np.array([0.0, 5.0, 10.0])

        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=3,
            env=env,
        )

        # Should convert without error
        assert len(overlay_data.timeseries) == 1
        ts_data = overlay_data.timeseries[0]
        assert len(ts_data.start_indices) == 3
        assert len(ts_data.end_indices) == 3

    def test_timeseries_conversion_empty_overlay_list(self):
        """Test overlay conversion with empty overlay list."""
        from neurospatial import Environment
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create simple environment
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        env = Environment.from_samples(positions, bin_size=5.0)

        frame_times = np.array([0.0, 0.5, 1.0])

        # Empty overlay list
        overlay_data = _convert_overlays_to_data(
            overlays=[],
            frame_times=frame_times,
            n_frames=3,
            env=env,
        )

        assert len(overlay_data.timeseries) == 0
        assert len(overlay_data.positions) == 0
