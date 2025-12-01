"""Integration tests for PlaybackController with render_napari().

This test module validates the integration of PlaybackController into
the napari rendering pipeline:
- Controller creation after viewer setup
- Storage in viewer.metadata["playback_controller"]
- Wiring to play/pause widget
- Keeping existing dims.events callbacks (for migration in Phase 5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Skip all tests if napari is not available
pytest.importorskip("napari")


class TestPlaybackControllerCreation:
    """Tests for PlaybackController creation in render_napari()."""

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def fields(self, env):
        """Create minimal field data for testing."""
        n_frames = 20
        return [np.random.rand(env.n_bins) for _ in range(n_frames)]

    def test_playback_controller_created_in_render_napari(self, env, fields):
        """render_napari() should create a PlaybackController instance."""
        from neurospatial.animation.backends.napari_backend import (
            PlaybackController,
            render_napari,
        )

        viewer = render_napari(env, fields, fps=30)
        try:
            # Controller should exist as viewer attribute
            assert hasattr(viewer, "playback_controller")
            controller = viewer.playback_controller
            assert isinstance(controller, PlaybackController)
        finally:
            viewer.close()

    def test_playback_controller_has_correct_n_frames(self, env, fields):
        """PlaybackController should have correct n_frames from fields."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.n_frames == len(fields)
        finally:
            viewer.close()

    def test_playback_controller_has_correct_fps(self, env, fields):
        """PlaybackController should have correct fps from render_napari()."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=25)
        try:
            controller = viewer.playback_controller
            assert controller.fps == 25.0
        finally:
            viewer.close()

    def test_playback_controller_viewer_reference(self, env, fields):
        """PlaybackController should have reference to the viewer."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.viewer is viewer
        finally:
            viewer.close()


class TestPlaybackControllerWithFrameTimes:
    """Tests for PlaybackController with frame_times from overlay_data."""

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def fields(self, env):
        """Create minimal field data for testing."""
        n_frames = 20
        return [np.random.rand(env.n_bins) for _ in range(n_frames)]

    @pytest.fixture
    def overlay_data_with_frame_times(self, env, fields):
        """Create overlay_data with frame_times."""
        from neurospatial.animation.overlays import OverlayData

        n_frames = len(fields)
        frame_times: NDArray[np.float64] = np.arange(n_frames) / 30.0
        return OverlayData(frame_times=frame_times)

    def test_playback_controller_receives_frame_times(
        self, env, fields, overlay_data_with_frame_times
    ):
        """PlaybackController should receive frame_times from overlay_data."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(
            env, fields, fps=30, overlay_data=overlay_data_with_frame_times
        )
        try:
            controller = viewer.playback_controller
            assert controller.frame_times is not None
            np.testing.assert_array_equal(
                controller.frame_times, overlay_data_with_frame_times.frame_times
            )
        finally:
            viewer.close()


class TestPlaybackControllerWithArrayFields:
    """Tests for PlaybackController with 2D array fields (memmap path)."""

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def array_fields(self, env):
        """Create 2D array field data for testing."""
        n_frames = 20
        return np.random.rand(n_frames, env.n_bins)

    def test_playback_controller_created_with_array_fields(self, env, array_fields):
        """PlaybackController should be created for 2D array fields."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, array_fields, fps=30)
        try:
            assert hasattr(viewer, "playback_controller")
            controller = viewer.playback_controller
            assert controller.n_frames == array_fields.shape[0]
        finally:
            viewer.close()


class TestPlaybackControllerWidgetWiring:
    """Tests for PlaybackController wiring to play/pause widget.

    These tests verify that the PlaybackController is properly wired to the
    playback widget, so that play/pause actions go through the controller.

    Note: Full widget integration testing with Qt events is complex due to
    event loop requirements. These tests focus on the controller API working
    correctly when used from render_napari().
    """

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def fields(self, env):
        """Create minimal field data for testing."""
        n_frames = 20
        return [np.random.rand(env.n_bins) for _ in range(n_frames)]

    def test_controller_play_updates_is_playing(self, env, fields):
        """Calling controller.play() should set is_playing to True."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.is_playing is False
            controller.play()
            assert controller.is_playing is True
        finally:
            viewer.close()

    def test_controller_pause_updates_is_playing(self, env, fields):
        """Calling controller.pause() should set is_playing to False."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            controller.play()
            assert controller.is_playing is True
            controller.pause()
            assert controller.is_playing is False
        finally:
            viewer.close()

    def test_controller_go_to_frame_updates_viewer_dims(self, env, fields):
        """Calling controller.go_to_frame() should update viewer.dims."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            controller.go_to_frame(10)

            # Verify viewer dims were updated
            assert viewer.dims.current_step[0] == 10
        finally:
            viewer.close()


class TestPlaybackControllerMultiField:
    """Tests for PlaybackController with multi-field animations."""

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def multi_field_sequences(self, env):
        """Create multiple field sequences for testing."""
        n_frames = 15
        seq1 = [np.random.rand(env.n_bins) for _ in range(n_frames)]
        seq2 = [np.random.rand(env.n_bins) for _ in range(n_frames)]
        return [seq1, seq2]

    def test_playback_controller_created_for_multi_field(
        self, env, multi_field_sequences
    ):
        """PlaybackController should be created for multi-field animations."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, multi_field_sequences, fps=30, layout="horizontal")
        try:
            assert hasattr(viewer, "playback_controller")
            controller = viewer.playback_controller
            assert controller.n_frames == len(multi_field_sequences[0])
        finally:
            viewer.close()

    def test_playback_controller_correct_fps_multi_field(
        self, env, multi_field_sequences
    ):
        """PlaybackController should have correct fps for multi-field."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, multi_field_sequences, fps=20, layout="vertical")
        try:
            controller = viewer.playback_controller
            assert controller.fps == 20.0
        finally:
            viewer.close()


class TestPlaybackControllerInitialState:
    """Tests for PlaybackController initial state in render_napari()."""

    @pytest.fixture
    def env(self):
        """Create a minimal Environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    @pytest.fixture
    def fields(self, env):
        """Create minimal field data for testing."""
        n_frames = 20
        return [np.random.rand(env.n_bins) for _ in range(n_frames)]

    def test_controller_starts_at_frame_zero(self, env, fields):
        """PlaybackController should start at frame 0."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.current_frame == 0
        finally:
            viewer.close()

    def test_controller_starts_not_playing(self, env, fields):
        """PlaybackController should start in paused state."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.is_playing is False
        finally:
            viewer.close()

    def test_controller_allow_frame_skip_default_true(self, env, fields):
        """PlaybackController should have allow_frame_skip=True by default."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(env, fields, fps=30)
        try:
            controller = viewer.playback_controller
            assert controller.allow_frame_skip is True
        finally:
            viewer.close()
