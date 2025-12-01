"""Tests for Phase 3.2: Reduce Matplotlib Draw Calls optimizations.

This module tests:
1. Configurable throttle frequency during playback (10 Hz vs 20 Hz)
2. xlim caching to avoid redundant set_xlim() calls

Following TDD: These tests are written BEFORE implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from neurospatial.animation._timeseries import TimeSeriesArtistManager


class TestTimeSeriesArtistManagerXlimCaching:
    """Test xlim caching in TimeSeriesArtistManager.update()."""

    @pytest.fixture
    def sample_timeseries_data(self) -> list[Any]:
        """Create sample TimeSeriesData for testing."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        data = np.linspace(0, 10, 100)
        times = np.linspace(0, 10, 100)
        overlay = TimeSeriesOverlay(data=data, times=times, label="Test")
        frame_times = np.linspace(0, 10, 50)

        ts_data = overlay.convert_to_data(
            frame_times=frame_times,
            n_frames=len(frame_times),
            env=None,  # env is not used by TimeSeriesOverlay
        )
        return [ts_data]

    @pytest.fixture
    def artist_manager(
        self, sample_timeseries_data: list[Any]
    ) -> TimeSeriesArtistManager:
        """Create TimeSeriesArtistManager for testing."""
        from matplotlib.figure import Figure

        from neurospatial.animation._timeseries import TimeSeriesArtistManager

        fig = Figure(figsize=(4, 2), dpi=72)
        frame_times = np.linspace(0, 10, 50)

        manager = TimeSeriesArtistManager.create(
            fig=fig,
            timeseries_data=sample_timeseries_data,
            frame_times=frame_times,
            dark_theme=True,
        )
        return manager

    def test_manager_has_xlim_cache_attribute(
        self, artist_manager: TimeSeriesArtistManager
    ):
        """Test that TimeSeriesArtistManager has _last_xlim_bounds cache."""
        # After first call, manager should have cached bounds
        assert hasattr(artist_manager, "_last_xlim_bounds")
        # Initially should be empty dict (no axes have been updated yet)
        assert isinstance(artist_manager._last_xlim_bounds, dict)

    def test_first_update_sets_xlim(
        self, artist_manager: TimeSeriesArtistManager, sample_timeseries_data: list
    ):
        """Test that first update() call does set xlim."""
        # Get the first axis
        ax = artist_manager.axes[0]

        # Mock set_xlim to track calls
        original_set_xlim = ax.set_xlim
        call_count = [0]

        def mock_set_xlim(*args, **kwargs):
            call_count[0] += 1
            return original_set_xlim(*args, **kwargs)

        ax.set_xlim = mock_set_xlim

        # First update should call set_xlim
        artist_manager.update(0, sample_timeseries_data)
        assert call_count[0] == 1

    def test_same_xlim_skips_set_xlim_call(
        self, artist_manager: TimeSeriesArtistManager, sample_timeseries_data: list
    ):
        """Test that update() with same xlim bounds skips set_xlim call."""
        ax = artist_manager.axes[0]

        # First update to establish bounds
        artist_manager.update(0, sample_timeseries_data)

        # Mock set_xlim to track calls after first update
        call_count = [0]
        original_set_xlim = ax.set_xlim

        def mock_set_xlim(*args, **kwargs):
            call_count[0] += 1
            return original_set_xlim(*args, **kwargs)

        ax.set_xlim = mock_set_xlim

        # Same frame should not call set_xlim again (bounds unchanged)
        artist_manager.update(0, sample_timeseries_data)
        assert call_count[0] == 0, (
            "set_xlim should not be called when bounds haven't changed"
        )

    def test_different_xlim_calls_set_xlim(
        self, artist_manager: TimeSeriesArtistManager, sample_timeseries_data: list
    ):
        """Test that update() with different xlim bounds does call set_xlim."""
        ax = artist_manager.axes[0]

        # First update to establish bounds at frame 0
        artist_manager.update(0, sample_timeseries_data)

        # Mock set_xlim to track calls after first update
        call_count = [0]
        original_set_xlim = ax.set_xlim

        def mock_set_xlim(*args, **kwargs):
            call_count[0] += 1
            return original_set_xlim(*args, **kwargs)

        ax.set_xlim = mock_set_xlim

        # Jump to a frame with different xlim bounds
        # Frame 49 (last frame) should have different bounds than frame 0
        artist_manager.update(49, sample_timeseries_data)
        assert call_count[0] == 1, "set_xlim should be called when bounds change"

    def test_xlim_cache_stores_bounds_correctly(
        self, artist_manager: TimeSeriesArtistManager, sample_timeseries_data: list
    ):
        """Test that _last_xlim_bounds stores the correct bounds."""
        # Update to frame 25 (middle of animation)
        artist_manager.update(25, sample_timeseries_data)

        # Check cache has entry for first axis (index 0)
        assert 0 in artist_manager._last_xlim_bounds

        # Cache should contain (xmin, xmax) tuple
        cached_bounds = artist_manager._last_xlim_bounds[0]
        assert isinstance(cached_bounds, tuple)
        assert len(cached_bounds) == 2

    def test_xlim_tolerance_for_floating_point(
        self, artist_manager: TimeSeriesArtistManager, sample_timeseries_data: list
    ):
        """Test that xlim comparison handles floating point tolerance."""
        # This tests that very small differences don't trigger unnecessary updates
        artist_manager.update(0, sample_timeseries_data)

        # Manually set cache to have tiny floating point difference
        if artist_manager.axes:
            ax_idx = 0
            original_bounds = artist_manager._last_xlim_bounds.get(ax_idx)
            if original_bounds:
                # Add tiny epsilon that shouldn't trigger update
                epsilon = 1e-10
                artist_manager._last_xlim_bounds[ax_idx] = (
                    original_bounds[0] + epsilon,
                    original_bounds[1] + epsilon,
                )

        # Count set_xlim calls
        call_count = [0]
        ax = artist_manager.axes[0]
        original_set_xlim = ax.set_xlim

        def mock_set_xlim(*args, **kwargs):
            call_count[0] += 1
            return original_set_xlim(*args, **kwargs)

        ax.set_xlim = mock_set_xlim

        # Same frame with epsilon smaller than xlim_tolerance (1e-6) should skip update
        artist_manager.update(0, sample_timeseries_data)

        # Epsilon (1e-10) is much smaller than xlim_tolerance (1e-6), so cache
        # should consider the bounds unchanged and skip set_xlim call
        assert call_count[0] == 0, (
            "set_xlim should be skipped for sub-tolerance changes"
        )


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


class TestDynamicThrottleFrequency:
    """Test that throttle frequency changes based on playback state."""

    def test_uses_lower_frequency_during_playback(self):
        """Test that during playback, lower throttle (10 Hz) is used."""
        # This is an integration test that would require napari viewer
        # Mark as skip if napari not available
        pytest.importorskip("napari")

        # Implementation will use PlaybackController.is_playing to determine
        # which throttle rate to use:
        # - is_playing=True: use playback_throttle_hz (default 10)
        # - is_playing=False: use scrub_throttle_hz (default 20)
        pass  # Integration test - see test_timeseries_dock_integration

    def test_uses_higher_frequency_when_scrubbing(self):
        """Test that during scrubbing (paused), higher throttle (20 Hz) is used."""
        # This is an integration test that would require napari viewer
        pytest.importorskip("napari")
        pass  # Integration test - see test_timeseries_dock_integration


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
