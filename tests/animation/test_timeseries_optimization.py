"""TimeSeriesOverlay throttle settings for playback and scrubbing."""

from __future__ import annotations

import numpy as np
import pytest


class TestPlaybackThrottleFrequency:
    """Test configurable throttle frequency during playback."""

    def test_timeseries_overlay_has_playback_throttle_hz_parameter(self):
        """Test that TimeSeriesOverlay accepts playback_throttle_hz parameter."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        # Should accept playback_throttle_hz without error
        overlay = TimeSeriesOverlay(
            data=data, times=times, playback_throttle_hz=10, label="Test"
        )

        assert overlay.playback_throttle_hz == 10

    def test_playback_throttle_hz_default_is_10(self):
        """Test that default playback_throttle_hz is 10."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        assert overlay.playback_throttle_hz == 10

    def test_playback_throttle_hz_validation_positive(self):
        """Test that playback_throttle_hz must be positive."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match=r"playback_throttle_hz.*positive"):
            TimeSeriesOverlay(data=data, times=times, playback_throttle_hz=0)

        with pytest.raises(ValueError, match=r"playback_throttle_hz.*positive"):
            TimeSeriesOverlay(data=data, times=times, playback_throttle_hz=-5)

    def test_timeseries_data_has_playback_throttle_hz_field(self):
        """Test that TimeSeriesData has playback_throttle_hz attribute."""
        import dataclasses

        from neurospatial.animation.overlays import TimeSeriesData

        field_names = [f.name for f in dataclasses.fields(TimeSeriesData)]
        assert "playback_throttle_hz" in field_names

    def test_convert_to_data_preserves_playback_throttle_hz(self):
        """Test that convert_to_data preserves playback_throttle_hz."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 10, 100)
        times = np.linspace(0, 10, 100)
        frame_times = np.linspace(0, 10, 50)

        overlay = TimeSeriesOverlay(
            data=data, times=times, playback_throttle_hz=15, label="Test"
        )
        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,
        )

        assert ts_data.playback_throttle_hz == 15


class TestScrubThrottleHzParameter:
    """Test scrub_throttle_hz parameter for scrubbing responsiveness."""

    def test_timeseries_overlay_has_scrub_throttle_hz_parameter(self):
        """Test that TimeSeriesOverlay accepts scrub_throttle_hz parameter."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        overlay = TimeSeriesOverlay(
            data=data, times=times, scrub_throttle_hz=25, label="Test"
        )

        assert overlay.scrub_throttle_hz == 25

    def test_scrub_throttle_hz_default_is_20(self):
        """Test that default scrub_throttle_hz is 20."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])
        overlay = TimeSeriesOverlay(data=data, times=times)

        assert overlay.scrub_throttle_hz == 20

    def test_scrub_throttle_hz_validation_positive(self):
        """Test that scrub_throttle_hz must be positive."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match=r"scrub_throttle_hz.*positive"):
            TimeSeriesOverlay(data=data, times=times, scrub_throttle_hz=0)

    def test_timeseries_data_has_scrub_throttle_hz_field(self):
        """Test that TimeSeriesData has scrub_throttle_hz attribute."""
        import dataclasses

        from neurospatial.animation.overlays import TimeSeriesData

        field_names = [f.name for f in dataclasses.fields(TimeSeriesData)]
        assert "scrub_throttle_hz" in field_names

    def test_convert_to_data_preserves_scrub_throttle_hz(self):
        """Test that convert_to_data preserves scrub_throttle_hz."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 10, 100)
        times = np.linspace(0, 10, 100)
        frame_times = np.linspace(0, 10, 50)

        overlay = TimeSeriesOverlay(
            data=data, times=times, scrub_throttle_hz=30, label="Test"
        )
        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,
        )

        assert ts_data.scrub_throttle_hz == 30
