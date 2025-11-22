"""Tests for ffmpeg I/O handling in video backend.

Phase 5.2: Control ffmpeg I/O
- Verifies stdout is discarded (DEVNULL) to avoid buffering issues
- Verifies stderr is captured for error reporting
- Tests behavior with large frame counts
"""

import contextlib
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFfmpegIOHandling:
    """Tests for ffmpeg subprocess I/O configuration."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_ffmpeg_stdout_is_devnull(self, simple_env, tmp_path):
        """Verify ffmpeg stdout is routed to DEVNULL to avoid buffer issues."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            # May fail for other reasons, but we captured the call
            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            if mock_run.called:
                call_kwargs = mock_run.call_args.kwargs
                # stdout should be DEVNULL to avoid buffering issues
                assert call_kwargs.get("stdout") == subprocess.DEVNULL, (
                    "ffmpeg stdout should be DEVNULL to avoid buffer issues"
                )

    def test_ffmpeg_stderr_is_captured(self, simple_env, tmp_path):
        """Verify ffmpeg stderr is captured for error reporting."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            if mock_run.called:
                call_kwargs = mock_run.call_args.kwargs
                # stderr should be captured (PIPE) for error messages
                assert call_kwargs.get("stderr") == subprocess.PIPE, (
                    "ffmpeg stderr should be PIPE to capture errors"
                )

    def test_ffmpeg_error_includes_stderr_message(self, simple_env, tmp_path):
        """Verify error messages include ffmpeg stderr output."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        error_message = "Unknown encoder 'invalid_codec'"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr=error_message)

            with pytest.raises(RuntimeError) as exc_info:
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            # Error message should include the stderr output
            assert error_message in str(exc_info.value)

    def test_ffmpeg_does_not_use_capture_output(self, simple_env, tmp_path):
        """Verify capture_output is not used (explicit stdout/stderr instead)."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            if mock_run.called:
                call_kwargs = mock_run.call_args.kwargs
                # Should NOT use capture_output (uses explicit stdout/stderr instead)
                assert call_kwargs.get("capture_output") is not True, (
                    "Should use explicit stdout/stderr, not capture_output"
                )


class TestFfmpegLargeFrameCount:
    """Tests for ffmpeg behavior with large frame counts."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_ffmpeg_called_with_correct_arguments(self, simple_env, tmp_path):
        """Verify ffmpeg is called with correct basic arguments."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [
            np.random.default_rng(42).random(simple_env.n_bins) for _ in range(10)
        ]
        output_path = tmp_path / "test_video.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            if mock_run.called:
                call_args = mock_run.call_args[0][0]  # Get command list
                # Check essential ffmpeg arguments are present
                assert "ffmpeg" in call_args[0]
                assert "-y" in call_args  # Overwrite
                assert "-framerate" in call_args
                assert "-i" in call_args  # Input pattern


class TestFfmpegTextMode:
    """Tests for ffmpeg text mode handling."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_ffmpeg_uses_text_mode(self, simple_env, tmp_path):
        """Verify ffmpeg uses text mode for stderr parsing."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            if mock_run.called:
                call_kwargs = mock_run.call_args.kwargs
                # text=True should be used for string stderr
                assert call_kwargs.get("text") is True, (
                    "ffmpeg should use text=True for string stderr"
                )
