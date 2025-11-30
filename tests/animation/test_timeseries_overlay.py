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
