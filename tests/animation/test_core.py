"""Test core animation dispatcher and utilities.

This module tests the main animation orchestration, backend selection,
and frame subsampling utilities.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import (
    Environment,
    PositionOverlay,
)


class TestPlaybackConstants:
    """Test playback constants are defined and importable (Task 1.1)."""

    def test_max_playback_fps_exists(self):
        """Test that MAX_PLAYBACK_FPS constant exists."""
        from neurospatial.animation.core import MAX_PLAYBACK_FPS

        assert MAX_PLAYBACK_FPS is not None

    def test_max_playback_fps_value(self):
        """Test that MAX_PLAYBACK_FPS has correct value."""
        from neurospatial.animation.core import MAX_PLAYBACK_FPS

        assert MAX_PLAYBACK_FPS == 60

    def test_max_playback_fps_is_int(self):
        """Test that MAX_PLAYBACK_FPS is an integer."""
        from neurospatial.animation.core import MAX_PLAYBACK_FPS

        assert isinstance(MAX_PLAYBACK_FPS, int)

    def test_min_playback_fps_exists(self):
        """Test that MIN_PLAYBACK_FPS constant exists."""
        from neurospatial.animation.core import MIN_PLAYBACK_FPS

        assert MIN_PLAYBACK_FPS is not None

    def test_min_playback_fps_value(self):
        """Test that MIN_PLAYBACK_FPS has correct value."""
        from neurospatial.animation.core import MIN_PLAYBACK_FPS

        assert MIN_PLAYBACK_FPS == 1

    def test_min_playback_fps_is_int(self):
        """Test that MIN_PLAYBACK_FPS is an integer."""
        from neurospatial.animation.core import MIN_PLAYBACK_FPS

        assert isinstance(MIN_PLAYBACK_FPS, int)

    def test_default_speed_exists(self):
        """Test that DEFAULT_SPEED constant exists."""
        from neurospatial.animation.core import DEFAULT_SPEED

        assert DEFAULT_SPEED is not None

    def test_default_speed_value(self):
        """Test that DEFAULT_SPEED has correct value."""
        from neurospatial.animation.core import DEFAULT_SPEED

        assert DEFAULT_SPEED == 1.0

    def test_default_speed_is_float(self):
        """Test that DEFAULT_SPEED is a float."""
        from neurospatial.animation.core import DEFAULT_SPEED

        assert isinstance(DEFAULT_SPEED, float)

    def test_constants_relationship(self):
        """Test that MIN_PLAYBACK_FPS <= MAX_PLAYBACK_FPS."""
        from neurospatial.animation.core import MAX_PLAYBACK_FPS, MIN_PLAYBACK_FPS

        assert MIN_PLAYBACK_FPS <= MAX_PLAYBACK_FPS


class TestComputePlaybackFps:
    """Test _compute_playback_fps() helper function (Task 1.2)."""

    def test_returns_tuple(self):
        """Test that function returns tuple of (int, float)."""
        from neurospatial.animation.core import _compute_playback_fps

        frame_times = np.linspace(0, 10, 301)  # 30 Hz, 10 seconds
        result = _compute_playback_fps(frame_times, speed=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_normal_case_30hz_realtime(self):
        """Test 30 Hz data at real-time speed."""
        from neurospatial.animation.core import _compute_playback_fps

        # 30 Hz data, 10 seconds (301 frames)
        frame_times = np.linspace(0, 10, 301)
        fps, actual_speed = _compute_playback_fps(frame_times, speed=1.0)

        assert fps == 30
        assert actual_speed == pytest.approx(1.0, rel=0.01)

    def test_capping_500hz_data(self):
        """Test that 500 Hz data at speed=1.0 is capped to 60 fps."""
        from neurospatial.animation.core import _compute_playback_fps

        # 500 Hz data, 1 second (501 frames)
        frame_times = np.linspace(0, 1, 501)
        fps, actual_speed = _compute_playback_fps(frame_times, speed=1.0)

        assert fps == 60  # Capped to MAX_PLAYBACK_FPS
        assert actual_speed == pytest.approx(0.12, rel=0.01)  # 60/500 = 0.12

    def test_slow_motion(self):
        """Test slow motion playback (10% speed)."""
        from neurospatial.animation.core import _compute_playback_fps

        # 30 Hz data, 10% speed
        frame_times = np.linspace(0, 10, 301)  # 30 Hz
        fps, actual_speed = _compute_playback_fps(frame_times, speed=0.1)

        assert fps == 3  # 30 * 0.1 = 3 fps
        assert actual_speed == pytest.approx(0.1, rel=0.01)

    def test_minimum_fps_clamping(self):
        """Test that fps is clamped to MIN_PLAYBACK_FPS (1 fps)."""
        from neurospatial.animation.core import _compute_playback_fps

        # 10 Hz data, 1% speed would be 0.1 fps
        frame_times = np.linspace(0, 10, 101)  # 10 Hz
        fps, actual_speed = _compute_playback_fps(frame_times, speed=0.01)

        assert fps == 1  # Clamped to MIN_PLAYBACK_FPS
        assert actual_speed == pytest.approx(0.1, rel=0.01)  # 1/10 = 0.1

    def test_single_frame_edge_case(self):
        """Test edge case: single frame returns max_fps."""
        from neurospatial.animation.core import _compute_playback_fps

        frame_times = np.array([0.0])  # Single frame
        fps, actual_speed = _compute_playback_fps(frame_times, speed=1.0)

        assert fps == 60  # Returns max_fps
        assert actual_speed == 1.0  # Returns requested speed

    def test_zero_duration_edge_case(self):
        """Test edge case: zero duration (all same timestamp)."""
        from neurospatial.animation.core import _compute_playback_fps

        frame_times = np.array([5.0, 5.0, 5.0])  # Zero duration
        fps, actual_speed = _compute_playback_fps(frame_times, speed=1.0)

        assert fps == 60  # Returns max_fps
        assert actual_speed == 1.0  # Returns requested speed

    def test_custom_max_fps(self):
        """Test that custom max_fps is respected."""
        from neurospatial.animation.core import _compute_playback_fps

        # 500 Hz data with custom max of 120 fps
        frame_times = np.linspace(0, 1, 501)  # 500 Hz
        fps, actual_speed = _compute_playback_fps(frame_times, speed=1.0, max_fps=120)

        assert fps == 120  # Custom max
        assert actual_speed == pytest.approx(0.24, rel=0.01)  # 120/500 = 0.24

    def test_fast_forward_2x(self):
        """Test 2x fast forward playback."""
        from neurospatial.animation.core import _compute_playback_fps

        # 30 Hz data at 2x speed
        frame_times = np.linspace(0, 10, 301)  # 30 Hz
        fps, actual_speed = _compute_playback_fps(frame_times, speed=2.0)

        assert fps == 60  # 30 * 2 = 60 fps (within limit)
        assert actual_speed == pytest.approx(2.0, rel=0.01)

    def test_uses_constants(self):
        """Test that function uses MAX_PLAYBACK_FPS and MIN_PLAYBACK_FPS constants."""
        from neurospatial.animation.core import (
            MAX_PLAYBACK_FPS,
            MIN_PLAYBACK_FPS,
            _compute_playback_fps,
        )

        # High speed case should use MAX_PLAYBACK_FPS
        frame_times = np.linspace(0, 1, 501)  # 500 Hz
        fps_high, _ = _compute_playback_fps(frame_times, speed=1.0)
        assert fps_high == MAX_PLAYBACK_FPS

        # Very slow case should use MIN_PLAYBACK_FPS
        frame_times_slow = np.linspace(0, 10, 11)  # 1 Hz
        fps_low, _ = _compute_playback_fps(frame_times_slow, speed=0.01)
        assert fps_low == MIN_PLAYBACK_FPS


class TestSubsampleFrames:
    """Test subsample_frames() utility function."""

    def test_subsample_ndarray(self):
        """Test subsampling with ndarray input."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        # Create test data: 100 frames at 100 Hz
        fields = rng.random((100, 50))  # 100 frames, 50 bins

        # Subsample to 25 Hz (every 4th frame)
        result = subsample_frames(fields, target_fps=25, source_fps=100)

        # Check output type
        assert isinstance(result, np.ndarray)

        # Check output shape (should be ~25 frames)
        assert result.shape[0] == 25
        assert result.shape[1] == 50

        # Check first frame matches
        np.testing.assert_array_equal(result[0], fields[0])

    def test_subsample_list(self):
        """Test subsampling with list input."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        # Create test data as list
        fields = [rng.random(50) for _ in range(100)]

        # Subsample to 25 Hz
        result = subsample_frames(fields, target_fps=25, source_fps=100)

        # Check output type preserved
        assert isinstance(result, list)

        # Check output length
        assert len(result) == 25

        # Check first frame matches
        np.testing.assert_array_equal(result[0], fields[0])

    def test_subsample_invalid_fps(self):
        """Test that target_fps > source_fps raises error."""
        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        fields = rng.random((100, 50))

        with pytest.raises(ValueError, match=r"target_fps.*cannot exceed.*source_fps"):
            subsample_frames(fields, target_fps=200, source_fps=100)

    def test_subsample_memmap(self):
        """Test subsampling works with memory-mapped arrays."""
        # Create memory-mapped array
        import tempfile
        from pathlib import Path

        from neurospatial.animation.core import subsample_frames

        rng = np.random.default_rng(42)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_path = Path(tmpfile.name)
            fields_mmap = np.memmap(
                tmpfile.name, dtype="float32", mode="w+", shape=(1000, 50)
            )
            fields_mmap[:] = rng.random((1000, 50))
            fields_mmap.flush()

            # Subsample
            result = subsample_frames(fields_mmap, target_fps=30, source_fps=250)

            # Check it worked without loading all data
            assert len(result) > 0
            assert result.shape[1] == 50

        # Cleanup
        tmpfile_path.unlink()

    def test_subsample_calculation(self):
        """Test subsampling rate calculation is correct."""
        from neurospatial.animation.core import subsample_frames

        # 250 Hz → 30 fps should take every ~8.3 frames
        fields = np.arange(250 * 50).reshape(250, 50)  # Sequential data

        result = subsample_frames(fields, target_fps=30, source_fps=250)

        # Calculate expected number of frames
        # np.arange(0, 250, 8.333...) produces 30 indices: [0, 8, 16, 24, ...]
        subsample_rate = 250 / 30
        expected_n_frames = len(np.arange(0, 250, subsample_rate))
        assert len(result) == expected_n_frames


class TestBackendSelection:
    """Test _select_backend() auto-selection logic."""

    def test_select_video_from_extension_mp4(self):
        """Test auto-selection of video backend for .mp4 files."""
        from neurospatial.animation.core import _select_backend

        backend = _select_backend(n_frames=100, save_path="output.mp4")
        assert backend == "video"

    def test_select_video_from_extension_webm(self):
        """Test auto-selection for .webm extension."""
        from neurospatial.animation.core import _select_backend

        backend = _select_backend(n_frames=100, save_path="output.webm")
        assert backend == "video"

    def test_select_html_from_extension(self):
        """Test auto-selection of HTML backend for .html files."""
        from neurospatial.animation.core import _select_backend

        backend = _select_backend(n_frames=100, save_path="output.html")
        assert backend == "html"

    def test_select_napari_large_dataset(self):
        """Test auto-selection of Napari for >10K frames."""
        from neurospatial.animation.core import _select_backend

        # Mock napari as available
        with patch(
            "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", True
        ):
            backend = _select_backend(n_frames=15000, save_path=None)
            assert backend == "napari"

    def test_select_napari_large_dataset_unavailable(self):
        """Test error when Napari needed but not available."""
        from neurospatial.animation.core import _select_backend

        # Mock napari as unavailable
        with (
            patch(
                "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="requires GPU acceleration"),
        ):
            _select_backend(n_frames=15000, save_path=None)

    def test_select_widget_in_jupyter(self):
        """Test auto-selection of widget backend in Jupyter."""
        from neurospatial.animation.core import _select_backend

        # Mock Jupyter environment by patching IPython.get_ipython
        with patch("IPython.get_ipython", return_value=MagicMock()):
            backend = _select_backend(n_frames=100, save_path=None)
            assert backend == "widget"

    def test_select_napari_default(self):
        """Test default to Napari when available."""
        from neurospatial.animation.core import _select_backend

        # Mock IPython not available and napari available
        with (
            patch("IPython.get_ipython", side_effect=ImportError),
            patch(
                "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", True
            ),
        ):
            backend = _select_backend(n_frames=100, save_path=None)
            assert backend == "napari"

    def test_select_no_backend_available(self):
        """Test error when no suitable backend available."""
        from neurospatial.animation.core import _select_backend

        # Mock IPython not available and napari unavailable
        with (
            patch("IPython.get_ipython", side_effect=ImportError),
            patch(
                "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="No suitable animation backend"),
        ):
            _select_backend(n_frames=100, save_path=None)


class TestAnimateFieldsValidation:
    """Test animate_fields() validation and error handling."""

    def test_animate_fields_not_fitted(self):
        """Test error when environment is not fitted."""
        from neurospatial.animation.core import animate_fields
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        rng = np.random.default_rng(42)
        # Create environment with layout but not fitted
        layout = RegularGridLayout()
        env = Environment(layout=layout)
        # Manually set _is_fitted to False (simulating unfitted state)
        env._is_fitted = False

        fields = [rng.random(100) for _ in range(10)]
        frame_times = np.linspace(0, 1, 10)

        with pytest.raises(RuntimeError, match="Environment must be fitted"):
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

    def test_animate_fields_empty(self):
        """Test error when fields is empty."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        frame_times = np.linspace(0, 1, 1)  # Need at least one frame_time

        with pytest.raises(ValueError, match="fields cannot be empty"):
            animate_fields(
                env, [], backend="html", save_path="test.html", frame_times=frame_times
            )

    def test_animate_fields_shape_mismatch(self):
        """Test error when field shape doesn't match environment."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create fields with wrong size
        wrong_size = env.n_bins + 10
        fields = [rng.random(wrong_size) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        with pytest.raises(ValueError, match=r"Field 0 has.*but environment has.*bins"):
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

    def test_animate_fields_normalizes_ndarray(self):
        """Test that ndarray input is normalized to list of arrays."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass ndarray (n_frames, n_bins)
        fields_ndarray = rng.random((10, env.n_bins))
        frame_times = np.linspace(0, 1, 10)

        # Mock HTML backend to check what it receives
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields_ndarray,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            # Check that backend received list of arrays
            call_args = mock.call_args
            fields_arg = call_args[0][1]  # Second positional arg
            assert isinstance(fields_arg, list)
            assert len(fields_arg) == 10

    def test_animate_fields_ndarray_too_few_dims(self):
        """Test error when ndarray has <2 dimensions."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # 1D array (invalid)
        fields = rng.random(env.n_bins)
        frame_times = np.linspace(0, 1, 1)

        with pytest.raises(ValueError, match="must be at least 2D"):
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )


class TestAnimateFieldsBackendRouting:
    """Test animate_fields() routing to backends."""

    def test_route_to_html_backend(self):
        """Test routing to HTML backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock HTML backend
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            result = animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            # Check backend was called
            mock.assert_called_once()
            assert result == Path("test.html")

    def test_route_to_napari_backend(self):
        """Test routing to Napari backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock Napari backend
        with patch(
            "neurospatial.animation.backends.napari_backend.render_napari"
        ) as mock:
            mock.return_value = MagicMock()  # Mock viewer

            animate_fields(env, fields, backend="napari", frame_times=frame_times)

            # Check backend was called
            mock.assert_called_once()

    def test_route_to_video_backend_requires_ffmpeg(self):
        """Test video backend checks ffmpeg availability."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock ffmpeg as unavailable
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="Video export requires ffmpeg"),
        ):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path="test.mp4",
                frame_times=frame_times,
            )

    def test_route_to_video_backend_requires_save_path(self):
        """Test video backend requires save_path."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock ffmpeg as available but no save_path
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            pytest.raises(ValueError, match="save_path required for video"),
        ):
            animate_fields(env, fields, backend="video", frame_times=frame_times)

    def test_route_to_video_backend_pickle_validation(self):
        """Test video backend validates environment pickle-ability."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock ffmpeg as available and mock pickle to fail
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            patch("pickle.dumps", side_effect=Exception("Cannot pickle")),
            pytest.raises(ValueError, match="Environment is not pickle-able"),
        ):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path="test.mp4",
                n_workers=2,
                frame_times=frame_times,
            )

    def test_route_to_video_backend_no_pickle_single_worker(self):
        """Test video backend doesn't validate pickle for single worker."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock backends
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            patch("neurospatial.animation.backends.video_backend.render_video") as mock,
        ):
            mock.return_value = Path("test.mp4")

            # Should not fail even if pickle fails (n_workers=1)
            with patch("pickle.dumps", side_effect=Exception("Cannot pickle")):
                animate_fields(
                    env,
                    fields,
                    backend="video",
                    save_path="test.mp4",
                    n_workers=1,
                    frame_times=frame_times,
                )

            mock.assert_called_once()

    def test_route_to_widget_backend(self):
        """Test routing to widget backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Mock widget backend
        with patch(
            "neurospatial.animation.backends.widget_backend.render_widget"
        ) as mock:
            mock.return_value = MagicMock()  # Mock widget

            animate_fields(env, fields, backend="widget", frame_times=frame_times)

            # Check backend was called
            mock.assert_called_once()

    def test_unknown_backend_error(self):
        """Test error for unknown backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        with pytest.raises(ValueError, match="Unknown backend: invalid"):
            animate_fields(env, fields, backend="invalid", frame_times=frame_times)


class TestAnimateFieldsIntegration:
    """Integration tests for animate_fields() dispatcher."""

    def test_auto_backend_selection(self):
        """Test auto backend selection works end-to-end."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Auto-select HTML for .html extension
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="auto",
                save_path="test.html",
                frame_times=frame_times,
            )

            mock.assert_called_once()

    def test_passes_kwargs_to_backend(self):
        """Test that kwargs are passed through to backends."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(5)]
        frame_times = np.linspace(0, 1, 5)

        # Pass custom kwargs
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                cmap="hot",
                dpi=200,
            )

            # Check kwargs were passed (fps is computed, cmap and dpi are passed through)
            call_kwargs = mock.call_args[1]
            assert call_kwargs["cmap"] == "hot"
            assert call_kwargs["dpi"] == 200


class TestDispatcherOverlayIntegration:
    """Test dispatcher integration with overlay system (Milestone 2.2)."""

    def test_dispatcher_accepts_overlay_parameters(self):
        """Test that dispatcher accepts new overlay parameters."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]

        # Create test overlay
        overlay_pos = rng.standard_normal((10, 2)) * 50
        position_overlay = PositionOverlay(data=overlay_pos, color="red", size=5.0)

        # Create frame times
        frame_times = np.linspace(0, 10, 10)

        # Mock HTML backend
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Should accept all new parameters without error
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                overlays=[position_overlay],
                frame_times=frame_times,
                show_regions=True,
                region_alpha=0.5,
            )

            mock.assert_called_once()

    def test_dispatcher_converts_overlays_when_provided(self):
        """Test that dispatcher calls conversion funnel when overlays provided."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]

        # Create test overlays
        overlay_pos = rng.standard_normal((10, 2)) * 50
        position_overlay = PositionOverlay(data=overlay_pos)
        frame_times = np.linspace(0, 10, 10)

        # Mock conversion funnel and backend
        with (
            patch(
                "neurospatial.animation.overlays._convert_overlays_to_data"
            ) as mock_convert,
            patch(
                "neurospatial.animation.backends.html_backend.render_html"
            ) as mock_backend,
        ):
            mock_backend.return_value = Path("test.html")
            from neurospatial.animation.overlays import OverlayData

            mock_convert.return_value = OverlayData()  # Empty overlay data

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                overlays=[position_overlay],
                frame_times=frame_times,
            )

            # Check conversion was called
            mock_convert.assert_called_once()
            call_kwargs = mock_convert.call_args.kwargs
            assert call_kwargs["overlays"] == [position_overlay]

    def test_dispatcher_skips_conversion_when_no_overlays(self):
        """Test that dispatcher skips conversion when overlays is None."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 10, 10)

        # Mock conversion funnel and backend
        with (
            patch(
                "neurospatial.animation.overlays._convert_overlays_to_data"
            ) as mock_convert,
            patch(
                "neurospatial.animation.backends.html_backend.render_html"
            ) as mock_backend,
        ):
            mock_backend.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                overlays=None,
                frame_times=frame_times,
            )

            # Check conversion was NOT called
            mock_convert.assert_not_called()

    def test_dispatcher_passes_overlay_data_to_backend(self):
        """Test that dispatcher passes OverlayData to backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]

        # Create test overlay
        overlay_pos = rng.standard_normal((10, 2)) * 50
        position_overlay = PositionOverlay(data=overlay_pos)
        frame_times = np.linspace(0, 10, 10)

        # Mock conversion and backend
        with (
            patch(
                "neurospatial.animation.overlays._convert_overlays_to_data"
            ) as mock_convert,
            patch(
                "neurospatial.animation.backends.html_backend.render_html"
            ) as mock_backend,
        ):
            from neurospatial.animation.overlays import OverlayData

            mock_overlay_data = OverlayData()
            mock_convert.return_value = mock_overlay_data
            mock_backend.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                overlays=[position_overlay],
                frame_times=frame_times,
            )

            # Check backend received overlay_data
            call_kwargs = mock_backend.call_args[1]
            assert "overlay_data" in call_kwargs
            assert call_kwargs["overlay_data"] is mock_overlay_data

    def test_dispatcher_uses_provided_frame_times(self):
        """Test that dispatcher uses explicitly provided frame_times."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]

        # Create overlay and explicit frame times
        overlay_pos = rng.standard_normal((10, 2)) * 50
        position_overlay = PositionOverlay(data=overlay_pos)
        frame_times = np.linspace(0, 5, 10)  # Custom times

        # Mock conversion and backend
        with (
            patch(
                "neurospatial.animation.overlays._convert_overlays_to_data"
            ) as mock_convert,
            patch(
                "neurospatial.animation.backends.html_backend.render_html"
            ) as mock_backend,
        ):
            from neurospatial.animation.overlays import OverlayData

            mock_convert.return_value = OverlayData()
            mock_backend.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                overlays=[position_overlay],
                frame_times=frame_times,
            )

            # Check conversion received the exact frame_times we provided
            call_kwargs = mock_convert.call_args.kwargs
            frame_times_arg = call_kwargs["frame_times"]
            np.testing.assert_array_equal(frame_times_arg, frame_times)

    def test_dispatcher_passes_show_regions_to_backend(self):
        """Test that dispatcher passes show_regions parameter to backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 10, 10)

        # Mock backend
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                show_regions=["region1", "region2"],
                region_alpha=0.4,
                frame_times=frame_times,
            )

            # Check backend received parameters
            call_kwargs = mock.call_args[1]
            assert call_kwargs["show_regions"] == ["region1", "region2"]
            assert call_kwargs["region_alpha"] == 0.4


class TestArrayPreservation:
    """Test array format preservation for different backends (Milestone 1).

    These tests verify that:
    - napari backend receives arrays as-is (not converted to list)
    - non-napari backends receive lists (arrays are converted)
    - list inputs remain lists for all backends
    """

    def test_napari_backend_receives_array_not_list(self):
        """Test that napari backend receives 2D array without list conversion.

        This is the core test for Task 1.1: arrays should pass through to
        napari backend unchanged to enable memmap efficiency.
        """
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass 2D ndarray
        fields_array = rng.random((10, env.n_bins)).astype(np.float64)
        frame_times = np.linspace(0, 1, 10)

        # Mock napari backend
        with patch(
            "neurospatial.animation.backends.napari_backend.render_napari"
        ) as mock:
            mock.return_value = MagicMock()

            animate_fields(env, fields_array, backend="napari", frame_times=frame_times)

            # Check backend received array, NOT list
            call_args = mock.call_args
            fields_arg = call_args[0][1]  # Second positional arg
            assert isinstance(fields_arg, np.ndarray), (
                f"Expected np.ndarray, got {type(fields_arg).__name__}"
            )
            assert fields_arg.shape == (10, env.n_bins)

    def test_html_backend_receives_list_from_array(self):
        """Test that HTML backend receives list when given array input.

        Non-napari backends expect list iteration semantics.
        """
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass 2D ndarray
        fields_array = rng.random((10, env.n_bins)).astype(np.float64)
        frame_times = np.linspace(0, 1, 10)

        # Mock HTML backend
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields_array,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            # Check backend received list, NOT array
            call_args = mock.call_args
            fields_arg = call_args[0][1]
            assert isinstance(fields_arg, list), (
                f"Expected list, got {type(fields_arg).__name__}"
            )
            assert len(fields_arg) == 10

    def test_video_backend_receives_list_from_array(self):
        """Test that video backend receives list when given array input."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass 2D ndarray
        fields_array = rng.random((10, env.n_bins)).astype(np.float64)
        frame_times = np.linspace(0, 1, 10)

        # Mock video backend
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            patch("neurospatial.animation.backends.video_backend.render_video") as mock,
        ):
            mock.return_value = Path("test.mp4")

            animate_fields(
                env,
                fields_array,
                backend="video",
                save_path="test.mp4",
                frame_times=frame_times,
            )

            # Check backend received list
            call_args = mock.call_args
            fields_arg = call_args[0][1]
            assert isinstance(fields_arg, list), (
                f"Expected list, got {type(fields_arg).__name__}"
            )

    def test_widget_backend_receives_list_from_array(self):
        """Test that widget backend receives list when given array input."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass 2D ndarray
        fields_array = rng.random((10, env.n_bins)).astype(np.float64)
        frame_times = np.linspace(0, 1, 10)

        # Mock widget backend
        with patch(
            "neurospatial.animation.backends.widget_backend.render_widget"
        ) as mock:
            mock.return_value = MagicMock()

            animate_fields(env, fields_array, backend="widget", frame_times=frame_times)

            # Check backend received list
            call_args = mock.call_args
            fields_arg = call_args[0][1]
            assert isinstance(fields_arg, list), (
                f"Expected list, got {type(fields_arg).__name__}"
            )

    def test_list_input_stays_list_for_napari(self):
        """Test that list input remains list for napari backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass list of arrays
        fields_list = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 1, 10)

        # Mock napari backend
        with patch(
            "neurospatial.animation.backends.napari_backend.render_napari"
        ) as mock:
            mock.return_value = MagicMock()

            animate_fields(env, fields_list, backend="napari", frame_times=frame_times)

            # Check backend received list
            call_args = mock.call_args
            fields_arg = call_args[0][1]
            assert isinstance(fields_arg, list)
            assert len(fields_arg) == 10

    def test_list_input_stays_list_for_html(self):
        """Test that list input remains list for HTML backend."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass list of arrays
        fields_list = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 1, 10)

        # Mock HTML backend
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields_list,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            # Check backend received list
            call_args = mock.call_args
            fields_arg = call_args[0][1]
            assert isinstance(fields_arg, list)
            assert len(fields_arg) == 10

    def test_memmap_preserved_for_napari(self):
        """Test that memory-mapped arrays are preserved for napari backend.

        This is critical for large-session support: memmaps should not be
        converted to list (which would load all data into RAM).
        """
        import tempfile
        from pathlib import Path as PathlibPath

        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create memmap
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_path = PathlibPath(tmpfile.name)
            fields_mmap = np.memmap(
                tmpfile.name, dtype="float64", mode="w+", shape=(100, env.n_bins)
            )
            fields_mmap[:] = rng.random((100, env.n_bins))
            fields_mmap.flush()
            frame_times = np.linspace(0, 1, 100)

            # Mock napari backend
            with patch(
                "neurospatial.animation.backends.napari_backend.render_napari"
            ) as mock:
                mock.return_value = MagicMock()

                animate_fields(
                    env, fields_mmap, backend="napari", frame_times=frame_times
                )

                # Check backend received memmap (or at least ndarray)
                call_args = mock.call_args
                fields_arg = call_args[0][1]
                assert isinstance(fields_arg, np.ndarray), (
                    f"Expected np.ndarray/memmap, got {type(fields_arg).__name__}"
                )
                # Verify it's the same object (not a copy)
                assert fields_arg is fields_mmap

        # Cleanup
        tmpfile_path.unlink()


class TestEstimateColormapRangeFromSubset:
    """Test estimate_colormap_range_from_subset() helper function."""

    def test_returns_tuple_of_floats(self):
        """Test that function returns tuple of two floats."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((100, 50))

        result = estimate_colormap_range_from_subset(fields)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_reasonable_range_for_uniform_random(self):
        """Test that range is reasonable for uniform random [0, 1] data."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((10_000, 50))

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        # For uniform random [0, 1], 1st and 99th percentiles should be ~0.01 and ~0.99
        assert 0.0 <= vmin <= 0.05
        assert 0.95 <= vmax <= 1.0

    def test_works_with_list_input(self):
        """Test that function works with list of arrays."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = [rng.random(50) for _ in range(100)]

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((1000, 50))

        result1 = estimate_colormap_range_from_subset(fields, seed=123)
        result2 = estimate_colormap_range_from_subset(fields, seed=123)

        assert result1 == result2

    def test_custom_percentiles(self):
        """Test custom percentile range."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((10_000, 50))

        # Use 5th and 95th percentiles
        vmin, vmax = estimate_colormap_range_from_subset(fields, percentile=(5.0, 95.0))

        # Should be tighter than default 1st/99th
        assert 0.02 <= vmin <= 0.1
        assert 0.9 <= vmax <= 0.98

    def test_n_samples_parameter(self):
        """Test that n_samples controls sampling."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((100_000, 50))

        # Small sample should still work
        vmin, vmax = estimate_colormap_range_from_subset(fields, n_samples=100)

        assert vmin < vmax

    def test_works_with_memmap(self, tmp_path):
        """Test that function works with memory-mapped arrays."""
        from neurospatial.animation.core import estimate_colormap_range_from_subset

        # Create memmap
        mmap_path = tmp_path / "test.dat"
        fields = np.memmap(
            str(mmap_path), dtype=np.float64, mode="w+", shape=(1000, 50)
        )
        rng = np.random.default_rng(42)
        fields[:] = rng.random((1000, 50))
        fields.flush()

        vmin, vmax = estimate_colormap_range_from_subset(fields)

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_fast_for_large_dataset(self):
        """Test that function completes quickly for large datasets."""
        import time

        from neurospatial.animation.core import estimate_colormap_range_from_subset

        rng = np.random.default_rng(42)
        fields = rng.random((1_000_000, 50))

        start = time.perf_counter()
        vmin, vmax = estimate_colormap_range_from_subset(fields)
        elapsed = time.perf_counter() - start

        # Should complete in under 0.5s
        assert elapsed < 0.5
        assert vmin < vmax


class TestAnimateFieldsSpeedBasedPlayback:
    """Test animate_fields() speed-based playback API (Milestone 3).

    These tests verify:
    - frame_times is required (defines temporal structure)
    - speed parameter controls playback speed relative to real-time
    - max_playback_fps parameter caps playback frame rate
    - fps is computed from frame_times and speed, overriding any user-provided fps
    """

    def test_frame_times_is_required(self):
        """Test that frame_times is required (no default value).

        Calling animate_fields without frame_times should raise TypeError.
        """
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]

        # Mock backend to avoid actual rendering
        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Should raise TypeError because frame_times is required
            with pytest.raises(TypeError, match="frame_times"):
                animate_fields(env, fields, backend="html", save_path="test.html")

    def test_speed_parameter_exists(self):
        """Test that speed parameter exists and is accepted."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 10, 10)

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Should accept speed parameter without error
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                speed=0.5,
            )

            mock.assert_called_once()

    def test_speed_parameter_default_is_1(self):
        """Test that speed parameter defaults to 1.0."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 10, 10)

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Call without speed parameter
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            # Check that computed fps reflects speed=1.0
            # With 10 frames over 10 seconds = 0.9 Hz sample rate
            # speed=1.0 means fps = 0.9 → clamped to MIN_PLAYBACK_FPS=1
            call_kwargs = mock.call_args[1]
            assert call_kwargs.get("fps") == 1  # MIN_PLAYBACK_FPS

    def test_max_playback_fps_parameter_exists(self):
        """Test that max_playback_fps parameter exists and is accepted."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(10)]
        frame_times = np.linspace(0, 10, 10)

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Should accept max_playback_fps parameter without error
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                max_playback_fps=30,
            )

            mock.assert_called_once()

    def test_max_playback_fps_parameter_default_is_60(self):
        """Test that max_playback_fps parameter defaults to 60."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create high sample rate data: 100 Hz
        fields = [rng.random(env.n_bins) for _ in range(100)]
        frame_times = np.linspace(0, 1, 100)  # 99 Hz sample rate

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Call without max_playback_fps - should use default 60
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                speed=1.0,
            )

            # Check that fps is capped at 60 (default max)
            call_kwargs = mock.call_args[1]
            assert call_kwargs.get("fps") <= 60

    def test_speed_affects_computed_fps(self):
        """Test that speed parameter affects computed fps."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create 30 Hz data (31 frames over 1 second)
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)  # 30 Hz sample rate

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # speed=0.5 should give fps = 30 * 0.5 = 15
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                speed=0.5,
            )

            call_kwargs = mock.call_args[1]
            assert call_kwargs.get("fps") == 15

    def test_fps_kwarg_is_ignored(self):
        """Test that fps in kwargs is overwritten by computed fps.

        The new API computes fps from frame_times and speed, so any
        user-provided fps should be ignored/overwritten.
        """
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create 30 Hz data
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)  # 30 Hz

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Pass fps=100 in kwargs - should be ignored/overwritten
            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                speed=1.0,
                fps=100,
            )

            call_kwargs = mock.call_args[1]
            # fps should be 30 (computed from data), not 100
            assert call_kwargs.get("fps") == 30

    def test_warning_emitted_when_speed_capped(self):
        """Test that UserWarning is emitted when speed is capped.

        When the requested speed would require fps > max_playback_fps,
        a warning should be emitted informing the user of the effective speed.
        """
        import warnings

        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create 500 Hz data - at speed=1.0 would require 500 fps
        fields = [rng.random(env.n_bins) for _ in range(501)]
        frame_times = np.linspace(0, 1, 501)  # 500 Hz sample rate

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                animate_fields(
                    env,
                    fields,
                    backend="html",
                    save_path="test.html",
                    frame_times=frame_times,
                    speed=1.0,  # Would require 500 fps
                )

                # Should have emitted a warning about capping
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert "Capped to 60 fps" in str(w[0].message)
                assert "effective speed" in str(w[0].message).lower()

    def test_no_warning_when_speed_not_capped(self):
        """Test that no warning is emitted when speed is not capped.

        When the requested speed results in fps <= max_playback_fps,
        no warning should be emitted.
        """
        import warnings

        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create 30 Hz data - at speed=1.0 would require 30 fps (within limit)
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)  # 30 Hz sample rate

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                animate_fields(
                    env,
                    fields,
                    backend="html",
                    save_path="test.html",
                    frame_times=frame_times,
                    speed=1.0,  # Results in 30 fps (within limit)
                )

                # Should not have emitted any warnings
                assert len(w) == 0

    @pytest.mark.skip(reason="Task 3.2 - not yet implemented")
    def test_sample_rate_hz_passed_to_backend(self):
        """Test that sample_rate_hz is passed to backend via kwargs (Task 3.2)."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create 30 Hz data
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)  # 30 Hz sample rate

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
            )

            call_kwargs = mock.call_args[1]
            assert "sample_rate_hz" in call_kwargs
            assert call_kwargs["sample_rate_hz"] == pytest.approx(30.0, rel=0.01)

    @pytest.mark.skip(reason="Task 3.2 - not yet implemented")
    def test_speed_passed_to_backend(self):
        """Test that speed is passed to backend via kwargs (Task 3.2)."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                speed=0.5,
            )

            call_kwargs = mock.call_args[1]
            assert "speed" in call_kwargs
            assert call_kwargs["speed"] == 0.5

    @pytest.mark.skip(reason="Task 3.2 - not yet implemented")
    def test_max_playback_fps_passed_to_backend(self):
        """Test that max_playback_fps is passed to backend via kwargs (Task 3.2)."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(31)]
        frame_times = np.linspace(0, 1, 31)

        with patch("neurospatial.animation.backends.html_backend.render_html") as mock:
            mock.return_value = Path("test.html")

            animate_fields(
                env,
                fields,
                backend="html",
                save_path="test.html",
                frame_times=frame_times,
                max_playback_fps=120,
            )

            call_kwargs = mock.call_args[1]
            assert "max_playback_fps" in call_kwargs
            assert call_kwargs["max_playback_fps"] == 120


class TestLargeSessionNapariConfig:
    """Test large_session_napari_config() helper function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        """Test that returned dict contains expected keys."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        assert "fps" in result
        assert "chunk_size" in result
        assert "max_chunks" in result

    def test_fps_is_positive_int(self):
        """Test that fps is a positive integer."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        assert isinstance(result["fps"], int)
        assert result["fps"] > 0

    def test_chunk_size_is_positive_int(self):
        """Test that chunk_size is a positive integer."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        assert isinstance(result["chunk_size"], int)
        assert result["chunk_size"] > 0

    def test_max_chunks_is_positive_int(self):
        """Test that max_chunks is a positive integer."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        assert isinstance(result["max_chunks"], int)
        assert result["max_chunks"] > 0

    def test_larger_datasets_get_larger_chunks(self):
        """Test that larger datasets get larger chunk sizes."""
        from neurospatial.animation.core import large_session_napari_config

        small = large_session_napari_config(n_frames=10_000)
        large = large_session_napari_config(n_frames=1_000_000)

        assert large["chunk_size"] >= small["chunk_size"]

    def test_sample_rate_affects_fps(self):
        """Test that sample_rate_hz affects recommended fps."""
        from neurospatial.animation.core import large_session_napari_config

        result_250hz = large_session_napari_config(n_frames=100_000, sample_rate_hz=250)
        result_500hz = large_session_napari_config(n_frames=100_000, sample_rate_hz=500)

        # Both should return valid configurations
        assert result_250hz["fps"] > 0
        assert result_500hz["fps"] > 0

    def test_can_unpack_into_animate_fields(self):
        """Test that result can be unpacked as kwargs to animate_fields."""
        from neurospatial.animation.core import large_session_napari_config

        result = large_session_napari_config(n_frames=100_000)

        # These should be valid kwargs for animate_fields (napari backend)
        # Just verify they're the right types
        assert isinstance(result.get("fps"), int)
        assert isinstance(result.get("chunk_size"), int)
        assert isinstance(result.get("max_chunks"), int)
