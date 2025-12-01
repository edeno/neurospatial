"""Tests for TimeSeriesOverlay update_mode parameter and behavior.

This module tests the update_mode feature that controls when the time series
dock widget updates during animation playback.

Following TDD: These tests are written BEFORE implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestTimeSeriesOverlayUpdateModeParameter:
    """Test update_mode parameter on TimeSeriesOverlay."""

    def test_default_update_mode_is_live(self):
        """Test that default update_mode is 'live'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        assert overlay.update_mode == "live"

    def test_update_mode_live(self):
        """Test setting update_mode='live'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="live")

        assert overlay.update_mode == "live"

    def test_update_mode_on_pause(self):
        """Test setting update_mode='on_pause'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="on_pause")

        assert overlay.update_mode == "on_pause"

    def test_update_mode_manual(self):
        """Test setting update_mode='manual'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="manual")

        assert overlay.update_mode == "manual"

    def test_invalid_update_mode_raises_error(self):
        """Test that invalid update_mode raises ValueError."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(
            ValueError, match=r"update_mode.*must be.*live.*on_pause.*manual"
        ):
            TimeSeriesOverlay(data=data, times=times, update_mode="invalid")


class TestTimeSeriesDataUpdateModeField:
    """Test update_mode field on TimeSeriesData container."""

    def test_timeseries_data_has_update_mode_field(self):
        """Test that TimeSeriesData has update_mode attribute."""
        # Check the field exists in the dataclass
        import dataclasses

        from neurospatial.animation.overlays import TimeSeriesData

        field_names = [f.name for f in dataclasses.fields(TimeSeriesData)]
        assert "update_mode" in field_names

    def test_convert_to_data_preserves_update_mode_live(self):
        """Test that convert_to_data preserves update_mode='live'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        frame_times = np.array([0.0, 0.5, 1.0])

        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="live")
        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,
        )

        assert ts_data.update_mode == "live"

    def test_convert_to_data_preserves_update_mode_on_pause(self):
        """Test that convert_to_data preserves update_mode='on_pause'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        frame_times = np.array([0.0, 0.5, 1.0])

        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="on_pause")
        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,
        )

        assert ts_data.update_mode == "on_pause"

    def test_convert_to_data_preserves_update_mode_manual(self):
        """Test that convert_to_data preserves update_mode='manual'."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        frame_times = np.array([0.0, 0.5, 1.0])

        overlay = TimeSeriesOverlay(data=data, times=times, update_mode="manual")
        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,
        )

        assert ts_data.update_mode == "manual"


class TestTimeSeriesDockUpdateModeBehavior:
    """Test update_mode behavior in _add_timeseries_dock.

    These tests verify that the dock widget respects the update_mode setting
    and integrates with PlaybackController for on_pause mode.
    """

    @pytest.fixture
    def frame_times(self) -> NDArray[np.float64]:
        """Create frame times for 30 fps over 1 second."""
        return np.linspace(0.0, 1.0, 30)

    @pytest.fixture
    def sample_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create sample time series data."""
        times = np.linspace(0.0, 1.0, 100)
        data = np.sin(2 * np.pi * times)
        return data, times

    @pytest.mark.skip(reason="Requires napari viewer - integration test")
    def test_live_mode_updates_on_frame_change(
        self, frame_times: NDArray[np.float64], sample_data: tuple
    ):
        """Test that live mode updates the plot on frame changes.

        In 'live' mode (default), the time series dock should update
        when the viewer's current frame changes (throttled to 20 Hz).
        """
        # This is a behavioral test - would require mocking napari viewer
        pass

    @pytest.mark.skip(reason="Requires napari viewer - integration test")
    def test_on_pause_mode_skips_updates_during_playback(
        self, frame_times: NDArray[np.float64], sample_data: tuple
    ):
        """Test that on_pause mode skips updates during playback.

        In 'on_pause' mode, the time series dock should NOT update
        when the viewer's current frame changes during playback,
        only when playback pauses.
        """
        pass

    @pytest.mark.skip(reason="Requires napari viewer - integration test")
    def test_manual_mode_never_auto_updates(
        self, frame_times: NDArray[np.float64], sample_data: tuple
    ):
        """Test that manual mode never auto-updates.

        In 'manual' mode, the time series dock should never update
        automatically, regardless of frame changes or playback state.
        """
        pass


class TestTimeSeriesDockWithPlaybackController:
    """Test integration between time series dock and PlaybackController.

    These tests verify that on_pause mode correctly hooks into
    PlaybackController's play/pause events.
    """

    @pytest.mark.skip(reason="Requires napari viewer - integration test")
    def test_on_pause_mode_updates_when_controller_pauses(self):
        """Test that on_pause mode updates when PlaybackController pauses.

        When PlaybackController.pause() is called and update_mode='on_pause',
        the time series dock should immediately update to the current frame.
        """
        pass

    @pytest.mark.skip(reason="Requires napari viewer - integration test")
    def test_on_pause_mode_registers_with_controller(self):
        """Test that on_pause mode registers callback with PlaybackController.

        When update_mode='on_pause', the dock should register a callback
        with the PlaybackController to be notified on pause.
        """
        pass


class TestMixedUpdateModes:
    """Test behavior when multiple TimeSeriesOverlays have different update_modes.

    The dock widget should respect each overlay's update_mode independently
    or use the most restrictive mode for the group.
    """

    def test_all_overlays_same_mode_respected(self):
        """Test that when all overlays have the same mode, it's respected."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        overlay1 = TimeSeriesOverlay(
            data=data, times=times, label="Speed", update_mode="on_pause"
        )
        overlay2 = TimeSeriesOverlay(
            data=data, times=times, label="Accel", update_mode="on_pause"
        )

        assert overlay1.update_mode == "on_pause"
        assert overlay2.update_mode == "on_pause"

    def test_different_overlays_can_have_different_modes(self):
        """Test that different overlays can have different update_modes."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        overlay1 = TimeSeriesOverlay(
            data=data, times=times, label="Speed", update_mode="live"
        )
        overlay2 = TimeSeriesOverlay(
            data=data, times=times, label="Accel", update_mode="on_pause"
        )
        overlay3 = TimeSeriesOverlay(
            data=data, times=times, label="LFP", update_mode="manual"
        )

        assert overlay1.update_mode == "live"
        assert overlay2.update_mode == "on_pause"
        assert overlay3.update_mode == "manual"
