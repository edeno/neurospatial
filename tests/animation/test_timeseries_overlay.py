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
        assert overlay.linewidth == 1.0
        assert overlay.alpha == 1.0
        assert overlay.group is None
        assert overlay.normalize is False
        assert overlay.show_cursor is True
        assert overlay.cursor_color == "red"
        assert overlay.vmin is None
        assert overlay.vmax is None

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
            linewidth=2.0,
            alpha=0.8,
            group="kinematics",
            normalize=True,
            show_cursor=False,
            cursor_color="yellow",
            vmin=0.0,
            vmax=100.0,
        )

        assert overlay.label == "Speed (cm/s)"
        assert overlay.color == "cyan"
        assert overlay.linewidth == 2.0
        assert overlay.alpha == 0.8
        assert overlay.group == "kinematics"
        assert overlay.normalize is True
        assert overlay.show_cursor is False
        assert overlay.cursor_color == "yellow"
        assert overlay.vmin == 0.0
        assert overlay.vmax == 100.0


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
        )

        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = 3

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert isinstance(result, TimeSeriesData)
        assert_array_equal(result.data, data)
        assert_array_equal(result.times, times)
        assert result.label == "Speed"

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

        result, _ = _convert_overlays_to_data(
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
            label="Speed",
            color="cyan",
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=5.0,
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
            label="Speed",
            color="cyan",
            linewidth=1.0,
            alpha=1.0,
            group=None,
            normalize=False,
            show_cursor=True,
            cursor_color="red",
            global_vmin=1.0,
            global_vmax=3.0,
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


# =============================================================================
# Edge Case Tests
# =============================================================================


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

        overlay_data, _ = _convert_overlays_to_data(
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

        overlay_data, _ = _convert_overlays_to_data(
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

        pytest.importorskip("imageio_ffmpeg")

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


class TestSignalWindows:
    """Signal panels show the samples within the scene's window around now."""

    @pytest.fixture
    def scene(self):
        from neurospatial import Environment
        from neurospatial.animation import Animation
        from neurospatial.animation.overlays import TimeSeriesOverlay

        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
        )
        data = np.arange(101.0)
        data[50] = np.nan  # a missing sample stays missing in its window
        signal = TimeSeriesOverlay(data, np.linspace(5, 15, 101), label="LFP")
        frames = np.linspace(0, 20, 41)
        with Animation(
            env,
            np.ones((41, env.n_bins)),
            frame_times=frames,
            overlays=[signal],
            window_seconds=4.0,
        ) as animation:
            yield animation

    @pytest.mark.parametrize(
        ("time", "first", "last"),
        [(6.0, 5.0, 8.0), (14.0, 12.0, 15.0), (10.0, 8.0, 12.0)],
    )
    def test_window_is_clipped_to_the_recorded_samples(self, scene, time, first, last):
        from neurospatial.animation._prepared import PreparedScene

        prepared = PreparedScene(scene)
        times, _ = prepared.signal_window(scene._overlay_data.timeseries[0], time)
        assert times[0] == pytest.approx(first)
        assert times[-1] == pytest.approx(last)

    def test_window_outside_the_recording_is_empty(self, scene):
        from neurospatial.animation._prepared import PreparedScene

        prepared = PreparedScene(scene)
        times, values = prepared.signal_window(scene._overlay_data.timeseries[0], 19.0)
        assert len(times) == len(values) == 0

    def test_missing_samples_stay_missing(self, scene):
        from neurospatial.animation._prepared import PreparedScene

        prepared = PreparedScene(scene)
        times, values = prepared.signal_window(scene._overlay_data.timeseries[0], 10.0)
        at_gap = np.isclose(times, 10.0)
        assert at_gap.sum() == 1
        assert np.isnan(values[at_gap]).all()
        assert np.isfinite(values[~at_gap]).all()
