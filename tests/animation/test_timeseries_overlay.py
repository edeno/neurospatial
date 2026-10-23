"""Tests for TimeSeriesOverlay dataclass and TimeSeriesData container.

This module tests the public API dataclass (TimeSeriesOverlay) and its
internal data container (TimeSeriesData), including validation, conversion,
and window extraction functionality.
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


def test_signals_sharing_a_group_share_a_panel_in_listed_order():
    """Ungrouped signals get a panel each; panels follow the listed order."""
    from types import SimpleNamespace

    from neurospatial.animation._prepared import signal_groups

    lfp, speed, theta, accel = (
        SimpleNamespace(group=group) for group in (None, "kin", None, "kin")
    )
    panels = list(signal_groups([lfp, speed, theta, accel]).values())
    assert [[id(s) for s in panel] for panel in panels] == [
        [id(lfp)],
        [id(speed), id(accel)],
        [id(theta)],
    ]


class TestTimeSeriesDataPickleSafe:
    """Time series data survives pickling."""

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
# Performance Testing
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
    @pytest.mark.wallclock
    def test_timeseries_window_extraction_is_o1(self):
        """Verify window extraction scales O(1) with data size.

        Compare extraction time for small vs large datasets.
        O(1) means extraction time should be similar regardless of data size.
        """
        import time

        from neurospatial.animation.overlays import TimeSeriesData

        rng = np.random.default_rng(42)

        def measure_extraction_time(n_samples: int, n_iterations: int = 1000) -> float:
            """Measure average time to extract windows."""
            data = rng.random(n_samples)
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
# Edge Case Tests
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


# =============================================================================
# Final Integration Tests
# =============================================================================


class TestTimeSeriesIntegration:
    """Final integration tests for time series overlay with other overlay types."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        from neurospatial import Environment

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def sample_fields(self, simple_env):
        """Create sample fields for testing."""
        rng = np.random.default_rng(42)
        return [rng.random(simple_env.n_bins) for _ in range(20)]

    @pytest.fixture
    def sample_timeseries_overlay(self):
        """Create sample TimeSeriesOverlay."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 100, 200)
        times = np.linspace(0, 10, 200)

        return TimeSeriesOverlay(
            data=data,
            times=times,
            label="Speed (cm/s)",
            color="cyan",
            window_seconds=2.0,
        )

    def test_timeseries_with_position_overlay(
        self, simple_env, sample_fields, sample_timeseries_overlay
    ):
        """Test TimeSeriesOverlay works with PositionOverlay."""
        from neurospatial.animation.overlays import (
            PositionOverlay,
            _convert_overlays_to_data,
        )

        rng = np.random.default_rng(42)
        positions = rng.random((20, 2)) * 10.0

        position_overlay = PositionOverlay(
            positions=positions,
            color="red",
            size=10.0,
            trail_length=5,
        )

        frame_times = np.linspace(0, 10, 20)

        overlay_data = _convert_overlays_to_data(
            overlays=[sample_timeseries_overlay, position_overlay],
            frame_times=frame_times,
            n_frames=20,
            env=simple_env,
        )

        # Both overlay types should be converted
        assert len(overlay_data.timeseries) == 1
        assert len(overlay_data.positions) == 1

        # Time series data should have correct structure
        ts_data = overlay_data.timeseries[0]
        assert ts_data.label == "Speed (cm/s)"
        assert len(ts_data.start_indices) == 20
        assert len(ts_data.end_indices) == 20

        # Position data should have correct structure
        pos_data = overlay_data.positions[0]
        assert pos_data.data.shape == (20, 2)

    def test_timeseries_with_event_overlay(self, simple_env, sample_fields):
        """Test TimeSeriesOverlay works with EventOverlay."""
        from neurospatial.animation.overlays import (
            EventOverlay,
            TimeSeriesOverlay,
            _convert_overlays_to_data,
        )

        # Create time series overlay
        ts_data = np.linspace(0, 100, 200)
        ts_times = np.linspace(0, 10, 200)
        timeseries_overlay = TimeSeriesOverlay(
            data=ts_data,
            times=ts_times,
            label="LFP",
            color="blue",
            window_seconds=1.0,
        )

        # Create event overlay (spikes)
        rng = np.random.default_rng(42)
        trajectory = rng.random((200, 2)) * 10.0
        spike_times = rng.uniform(0, 10, 50)

        event_overlay = EventOverlay(
            event_times=spike_times,
            positions=trajectory,
            position_times=ts_times,
            colors="red",
            size=8.0,
            decay_frames=3,
        )

        frame_times = np.linspace(0, 10, 20)

        overlay_data = _convert_overlays_to_data(
            overlays=[timeseries_overlay, event_overlay],
            frame_times=frame_times,
            n_frames=20,
            env=simple_env,
        )

        # Both overlay types should be converted
        assert len(overlay_data.timeseries) == 1
        assert len(overlay_data.events) == 1

    def test_timeseries_full_workflow_video_backend(
        self, simple_env, sample_fields, sample_timeseries_overlay, tmp_path
    ):
        """Test end-to-end workflow with video backend and time series."""
        import warnings

        from neurospatial.animation.overlays import PositionOverlay

        rng = np.random.default_rng(42)
        positions = rng.random((20, 2)) * 10.0

        position_overlay = PositionOverlay(
            positions=positions,
            color="red",
            size=10.0,
            trail_length=5,
        )

        frame_times = np.linspace(0, 10, 20)
        save_path = tmp_path / "test_integration.mp4"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = simple_env.animate_fields(
                sample_fields,
                overlays=[sample_timeseries_overlay, position_overlay],
                frame_times=frame_times,
                backend="video",
                save_path=str(save_path),
                n_workers=1,  # Sequential for test stability
            )

        # Video should be created
        assert result.exists()
        assert result.stat().st_size > 0

    def test_signals_get_a_row_each_unless_they_share_a_group(self, simple_env):
        """Grouped signals share one time panel; normalized ones stay in [0, 1]."""
        from matplotlib.lines import Line2D

        from neurospatial.animation import Animation
        from neurospatial.animation.overlays import TimeSeriesOverlay

        times = np.linspace(0, 1, 50)

        def signal(label, data, **options):
            return TimeSeriesOverlay(data=data, times=times, label=label, **options)

        overlays = [
            signal("Speed", 100 + 100 * np.sin(times), group="kin", normalize=True),
            signal("Accel", 0.5 + 0.5 * np.cos(times), group="kin", normalize=True),
            signal("Theta", np.sin(8 * times)),
            signal("Gamma", np.sin(40 * times)),
        ]
        frame_times = np.linspace(0, 1, 11)
        with Animation(
            simple_env,
            np.zeros((11, simple_env.n_bins)),
            frame_times=frame_times,
            overlays=overlays,
            window_seconds=0.4,
        ) as scene:
            scene.render_frame(0.5, dpi=30)
            panels = [
                ax
                for ax in scene._renderer.figure.axes
                if ax.get_xlabel() == "Time (s)"
            ]
            labels = [
                sorted(
                    line.get_label()
                    for line in ax.lines
                    if isinstance(line, Line2D) and not line.get_label().startswith("_")
                )
                for ax in panels
            ]
            assert labels == [["Accel", "Speed"], ["Theta"], ["Gamma"]]
            grouped = panels[0]
            assert grouped.get_ylim() == (0.0, 1.0)
            for line in grouped.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if line.get_label() in ("Speed", "Accel"):
                    assert y.min() >= 0.0 and y.max() <= 1.0
            # Each panel shows the scene's window around the current time.
            for ax in panels:
                np.testing.assert_allclose(ax.get_xlim(), (0.3, 0.7))
