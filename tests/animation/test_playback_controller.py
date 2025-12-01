"""Tests for PlaybackController class.

This test module validates the PlaybackController:
- Frame navigation (go_to_frame, current_frame property)
- Playback control (play, pause, step, is_playing)
- Frame skipping based on elapsed time
- Callback registration and notification
- Metrics tracking (frames_rendered, frames_skipped)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestPlaybackControllerInit:
    """Tests for PlaybackController initialization."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_playback_controller_exists(self):
        """PlaybackController class should exist in napari_backend module."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        assert PlaybackController is not None

    def test_init_basic_attributes(self, mock_viewer: MagicMock):
        """PlaybackController should initialize with basic attributes."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

        assert controller.viewer is mock_viewer
        assert controller.n_frames == 100
        assert controller.fps == 30.0
        assert controller.current_frame == 0
        assert controller.is_playing is False

    def test_init_with_frame_times(self, mock_viewer: MagicMock):
        """PlaybackController should accept optional frame_times array."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        frame_times: NDArray[np.float64] = np.arange(100) / 30.0

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            frame_times=frame_times,
        )

        assert controller.frame_times is not None
        assert len(controller.frame_times) == 100

    def test_init_with_allow_frame_skip(self, mock_viewer: MagicMock):
        """PlaybackController should accept allow_frame_skip parameter."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            allow_frame_skip=False,
        )

        assert controller.allow_frame_skip is False


class TestPlaybackControllerNavigation:
    """Tests for frame navigation."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def controller(self, mock_viewer: MagicMock):
        """Create a PlaybackController instance."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        return PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

    def test_go_to_frame_valid(self, controller, mock_viewer: MagicMock):
        """go_to_frame should update current frame and call viewer.dims.set_current_step."""
        controller.go_to_frame(50)

        assert controller.current_frame == 50
        mock_viewer.dims.set_current_step.assert_called_with(0, 50)

    def test_go_to_frame_clamps_to_zero(self, controller, mock_viewer: MagicMock):
        """go_to_frame should clamp negative frame indices to 0."""
        controller.go_to_frame(-10)

        assert controller.current_frame == 0
        mock_viewer.dims.set_current_step.assert_called_with(0, 0)

    def test_go_to_frame_clamps_to_max(self, controller, mock_viewer: MagicMock):
        """go_to_frame should clamp frame index to n_frames - 1."""
        controller.go_to_frame(200)

        assert controller.current_frame == 99  # n_frames - 1
        mock_viewer.dims.set_current_step.assert_called_with(0, 99)

    def test_go_to_frame_notifies_callbacks(self, controller, mock_viewer: MagicMock):
        """go_to_frame should notify all registered callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        controller.register_callback(callback1)
        controller.register_callback(callback2)

        controller.go_to_frame(25)

        callback1.assert_called_once_with(25)
        callback2.assert_called_once_with(25)


class TestPlaybackControllerPlayPause:
    """Tests for play/pause functionality."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def controller(self, mock_viewer: MagicMock):
        """Create a PlaybackController instance."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        return PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

    def test_play_starts_playback(self, controller):
        """play() should set is_playing to True."""
        assert controller.is_playing is False

        controller.play()

        assert controller.is_playing is True

    def test_pause_stops_playback(self, controller):
        """pause() should set is_playing to False."""
        controller.play()
        assert controller.is_playing is True

        controller.pause()

        assert controller.is_playing is False

    def test_play_records_start_time(self, controller):
        """play() should record start time for frame skip calculations."""
        controller.play()

        # Start time should be set (approximately now)
        assert controller._start_time is not None
        assert abs(controller._start_time - time.perf_counter()) < 0.1

    def test_play_records_start_frame(self, controller):
        """play() should record current frame as start frame."""
        controller.go_to_frame(25)
        controller.play()

        assert controller._start_frame == 25


class TestPlaybackControllerStep:
    """Tests for frame stepping and skipping."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def controller(self, mock_viewer: MagicMock):
        """Create a PlaybackController instance with debounce disabled."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        return PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=0,  # Disable debounce for step tests
        )

    def test_step_does_nothing_when_not_playing(
        self, controller, mock_viewer: MagicMock
    ):
        """step() should do nothing if not playing."""
        controller.step()

        # Should not have called set_current_step
        mock_viewer.dims.set_current_step.assert_not_called()

    def test_step_advances_frame_based_on_elapsed_time(
        self, controller, mock_viewer: MagicMock
    ):
        """step() should advance to target frame based on elapsed time."""
        controller.play()

        # Simulate elapsed time of ~0.1 seconds at 30 fps = 3 frames
        controller._start_time = time.perf_counter() - 0.1

        controller.step()

        # Should have advanced (elapsed * fps = 0.1 * 30 = 3 frames)
        assert controller.current_frame >= 2  # At least 2 frames (timing tolerance)

    def test_step_skips_frames_when_behind(self, controller, mock_viewer: MagicMock):
        """step() should skip frames if significantly behind schedule."""
        controller.play()

        # Simulate being 10 frames behind (0.33 seconds at 30 fps)
        controller._start_time = time.perf_counter() - 0.33

        controller.step()

        # Should have jumped to ~frame 10, not just frame 1
        assert controller.current_frame >= 8  # With timing tolerance

    def test_step_pauses_at_end(self, controller, mock_viewer: MagicMock):
        """step() should pause when reaching the last frame."""
        controller.go_to_frame(98)
        controller.play()

        # Simulate enough time to reach end
        controller._start_time = time.perf_counter() - 0.1

        controller.step()

        # Should have paused at end
        assert controller.current_frame == 99
        assert controller.is_playing is False


class TestPlaybackControllerCallbacks:
    """Tests for callback registration and notification."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def controller(self, mock_viewer: MagicMock):
        """Create a PlaybackController instance."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        return PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

    def test_register_callback(self, controller):
        """register_callback() should add callback to list."""
        callback = MagicMock()

        controller.register_callback(callback)

        assert callback in controller._callbacks

    def test_callbacks_called_on_frame_change(self, controller):
        """Callbacks should be called when frame changes."""
        callback = MagicMock()
        controller.register_callback(callback)

        controller.go_to_frame(50)

        callback.assert_called_once_with(50)

    def test_multiple_callbacks(self, controller):
        """Multiple callbacks should all be called."""
        callbacks = [MagicMock() for _ in range(5)]
        for cb in callbacks:
            controller.register_callback(cb)

        controller.go_to_frame(30)

        for cb in callbacks:
            cb.assert_called_once_with(30)


class TestPlaybackControllerMetrics:
    """Tests for playback metrics tracking."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def controller(self, mock_viewer: MagicMock):
        """Create a PlaybackController instance with debounce disabled."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        return PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=0,  # Disable debounce for metrics tests
        )

    def test_frames_rendered_initial_zero(self, controller):
        """frames_rendered should start at 0."""
        assert controller.frames_rendered == 0

    def test_frames_skipped_initial_zero(self, controller):
        """frames_skipped should start at 0."""
        assert controller.frames_skipped == 0

    def test_frames_rendered_increments(self, controller):
        """frames_rendered should increment on each frame change."""
        controller.go_to_frame(10)
        controller.go_to_frame(20)
        controller.go_to_frame(30)

        assert controller.frames_rendered == 3

    def test_frames_skipped_tracks_skipped_frames(self, controller):
        """frames_skipped should track frames skipped during playback."""
        controller.play()

        # Simulate being 10 frames behind
        controller._start_time = time.perf_counter() - 0.33

        controller.step()

        # Should have skipped some frames (went from 0 to ~10, skipping intermediate)
        assert controller.frames_skipped >= 8  # ~10 frames skipped with tolerance


class TestPlaybackControllerAllowFrameSkip:
    """Tests for frame skip enable/disable."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_no_skip_when_disabled(self, mock_viewer: MagicMock):
        """When allow_frame_skip=False, step() should advance by 1 frame only."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            allow_frame_skip=False,
        )

        controller.play()

        # Simulate being significantly behind schedule
        controller._start_time = time.perf_counter() - 1.0  # 30 frames behind

        controller.step()

        # With skip disabled, should only advance by 1 frame
        assert controller.current_frame == 1
