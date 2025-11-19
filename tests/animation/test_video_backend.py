"""Test video export backend with parallel rendering.

This module tests the video backend implementation, including:
- ffmpeg availability checking
- Dry run mode (estimation without rendering)
- Video export with serial rendering (n_workers=1)
- Video export with parallel rendering (n_workers>1)
- Error cases (missing ffmpeg, invalid parameters)
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.core import animate_fields


class TestFfmpegAvailability:
    """Test ffmpeg availability checking."""

    def test_check_ffmpeg_available_true(self):
        """Test ffmpeg availability check when ffmpeg is installed."""
        from neurospatial.animation.backends.video_backend import (
            check_ffmpeg_available,
        )

        # Mock successful ffmpeg call
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert check_ffmpeg_available() is True
            mock_run.assert_called_once()
            # Check that ffmpeg -version was called
            args = mock_run.call_args[0][0]
            assert "ffmpeg" in args
            assert "-version" in args

    def test_check_ffmpeg_available_false_not_found(self):
        """Test ffmpeg check when command not found."""
        from neurospatial.animation.backends.video_backend import (
            check_ffmpeg_available,
        )

        # Mock FileNotFoundError
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_ffmpeg_available() is False

    def test_check_ffmpeg_available_false_error(self):
        """Test ffmpeg check when command returns error."""
        from neurospatial.animation.backends.video_backend import (
            check_ffmpeg_available,
        )

        # Mock error return code
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
        ):
            assert check_ffmpeg_available() is False


class TestVideoDryRun:
    """Test dry run mode (estimation without rendering)."""

    def test_dry_run_prints_estimate(self, tmp_path, capsys):
        """Test dry run prints time and size estimate."""
        # Create environment
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create simple field sequence
        fields = [np.random.rand(env.n_bins) for _ in range(20)]

        output_path = tmp_path / "test.mp4"

        # Mock ffmpeg availability
        with patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ):
            # Call with dry_run=True
            result = animate_fields(
                env,
                fields,
                backend="video",
                save_path=str(output_path),
                fps=30,
                n_workers=2,
                dry_run=True,
            )

            # Should return None (no file created)
            assert result is None

            # Should NOT create file
            assert not output_path.exists()

            # Check printed output
            captured = capsys.readouterr()
            assert "dry run" in captured.out.lower()
            assert "frames:" in captured.out.lower()
            assert "workers:" in captured.out.lower()
            assert "est. total time:" in captured.out.lower()
            assert "est. file size:" in captured.out.lower()

    def test_dry_run_does_not_spawn_workers(self, tmp_path):
        """Test dry run doesn't spawn parallel workers."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        output_path = tmp_path / "test.mp4"

        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            patch(
                "neurospatial.animation._parallel.parallel_render_frames"
            ) as mock_parallel,
        ):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path=str(output_path),
                dry_run=True,
                n_workers=4,
            )

            # parallel_render_frames should NOT be called
            mock_parallel.assert_not_called()


class TestVideoExportSerial:
    """Test video export with serial rendering (n_workers=1)."""

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_video_export_small_serial(self, tmp_path):
        """Test video export with n_workers=1 (serial rendering)."""
        # Create environment
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create simple field sequence (small for speed)
        fields = []
        for i in range(5):
            field = np.sin(np.linspace(0, 2 * np.pi, env.n_bins) + i * 0.5)
            fields.append(field)

        # Export video
        output_path = tmp_path / "test_serial.mp4"

        result = animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=5,
            n_workers=1,  # Serial rendering
        )

        # Check file created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Check result is Path
        assert isinstance(result, Path)
        assert result == output_path

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_video_export_with_labels(self, tmp_path):
        """Test video export with frame labels."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        fields = [np.random.rand(env.n_bins) for _ in range(5)]
        labels = [f"Trial {i + 1}" for i in range(5)]

        output_path = tmp_path / "labeled.mp4"

        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=5,
            frame_labels=labels,
            n_workers=1,
        )

        assert output_path.exists()

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_video_export_custom_parameters(self, tmp_path):
        """Test video export with custom codec, dpi, bitrate."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "custom.mp4"

        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=10,
            codec="h264",
            dpi=50,  # Lower resolution for speed
            bitrate=3000,
            n_workers=1,
        )

        assert output_path.exists()


class TestVideoExportParallel:
    """Test video export with parallel rendering (n_workers>1)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_video_export_parallel(self, tmp_path):
        """Test parallel rendering with n_workers=2."""
        positions = np.random.randn(200, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # 20 frames for parallel test
        fields = [np.random.rand(env.n_bins) for _ in range(20)]

        output_path = tmp_path / "parallel.mp4"

        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=10,
            n_workers=2,  # Parallel rendering
        )

        # Check file created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_video_export_auto_workers(self, tmp_path):
        """Test automatic worker count selection (n_workers=None)."""
        positions = np.random.randn(200, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        output_path = tmp_path / "auto_workers.mp4"

        # n_workers=None should auto-select based on CPU count
        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=10,
            n_workers=None,  # Auto-select
        )

        assert output_path.exists()


class TestVideoErrors:
    """Test error handling in video backend."""

    def test_video_missing_ffmpeg(self, tmp_path):
        """Test error when ffmpeg not available."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "test.mp4"

        # Mock ffmpeg not available
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="ffmpeg"),
        ):
            animate_fields(
                env, fields, backend="video", save_path=str(output_path), fps=10
            )

    def test_video_pickle_failure(self, tmp_path):
        """Test error when environment not pickle-able with n_workers>1."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "test.mp4"

        # Mock ffmpeg available but pickle fails
        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            patch("pickle.dumps", side_effect=Exception("Cannot pickle")),
            pytest.raises(ValueError, match="pickle-able"),
        ):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path=str(output_path),
                fps=10,
                n_workers=2,  # Parallel requires pickle
            )

    def test_video_no_pickle_check_serial(self, tmp_path):
        """Test pickle check skipped when n_workers=1."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "test.mp4"

        with (
            patch(
                "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
                return_value=True,
            ),
            # Mock render_video to avoid actual rendering
            patch(
                "neurospatial.animation.backends.video_backend.render_video",
                return_value=output_path,
            ),
            patch("pickle.dumps") as mock_pickle,
        ):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path=str(output_path),
                fps=10,
                n_workers=1,  # Serial - no pickle check
            )

            # pickle.dumps should NOT be called (happens in core.py only for n_workers > 1)
            mock_pickle.assert_not_called()

    def test_video_ffmpeg_encoding_failure(self, tmp_path):
        """Test error when ffmpeg encoding fails."""
        from neurospatial.animation.backends.video_backend import render_video

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "test.mp4"

        # Mock parallel rendering to succeed
        with (
            patch(
                "neurospatial.animation._parallel.parallel_render_frames",
                return_value=str(tmp_path / "frame_%05d.png"),
            ),
            # Mock ffmpeg to fail
            patch(
                "subprocess.run",
                return_value=Mock(returncode=1, stderr="Encoding failed"),
            ),
            pytest.raises(RuntimeError, match="ffmpeg encoding failed"),
        ):
            render_video(env, fields, str(output_path), fps=10, n_workers=1, dpi=50)

    def test_video_negative_workers(self, tmp_path):
        """Test error for negative n_workers."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "test.mp4"

        with pytest.raises(ValueError, match="n_workers must be positive"):
            animate_fields(
                env,
                fields,
                backend="video",
                save_path=str(output_path),
                fps=10,
                n_workers=-1,
            )


class TestParallelRendering:
    """Test parallel frame rendering utilities."""

    def test_parallel_render_frames_partitioning(self, tmp_path):
        """Test frame partitioning across workers."""
        from neurospatial.animation._parallel import parallel_render_frames

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # 10 frames, 3 workers
        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        # Mock ProcessPoolExecutor
        with patch("neurospatial.animation._parallel.ProcessPoolExecutor") as mock_pool:
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_executor
            mock_executor.map.return_value = [None, None, None]  # 3 workers

            pattern = parallel_render_frames(
                env=env,
                fields=fields,
                output_dir=str(tmp_path),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=3,
            )

            # Check pattern returned
            assert "frame_" in pattern
            assert ".png" in pattern

            # Check executor.map was called with tasks
            mock_executor.map.assert_called_once()
            args = mock_executor.map.call_args[0]
            tasks = list(args[1])  # Get task list

            # Should have 3 tasks (one per worker)
            assert len(tasks) == 3

            # Check task structure
            assert all("env" in task for task in tasks)
            assert all("fields" in task for task in tasks)
            assert all("start_frame_idx" in task for task in tasks)

    def test_parallel_render_frames_unpicklable_env(self, tmp_path):
        """Test error when environment cannot be pickled."""
        from neurospatial.animation._parallel import parallel_render_frames

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        # Mock pickle to fail
        with (
            patch("pickle.dumps", side_effect=Exception("Cannot pickle")),
            pytest.raises(ValueError, match="pickle-able"),
        ):
            parallel_render_frames(
                env=env,
                fields=fields,
                output_dir=str(tmp_path),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                frame_labels=None,
                dpi=100,
                n_workers=2,
            )

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_worker_frame_rendering(self, tmp_path):
        """Test worker function renders frames correctly."""
        from neurospatial.animation._parallel import _render_worker_frames

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # 5 frames for this worker
        fields = [np.random.rand(env.n_bins) for _ in range(5)]
        labels = [f"Frame {i}" for i in range(5)]

        task = {
            "env": env,
            "fields": fields,
            "start_frame_idx": 0,
            "output_dir": str(tmp_path),
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "frame_labels": labels,
            "dpi": 50,
        }

        # Call worker function
        _render_worker_frames(task)

        # Check frames were created
        png_files = list(tmp_path.glob("frame_*.png"))
        assert len(png_files) == 5

        # Check files are valid PNGs (non-zero size)
        for png_file in png_files:
            assert png_file.stat().st_size > 0

    def test_worker_frame_rendering_no_labels(self, tmp_path):
        """Test worker rendering without frame labels."""
        from neurospatial.animation._parallel import _render_worker_frames

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        fields = [np.random.rand(env.n_bins) for _ in range(3)]

        task = {
            "env": env,
            "fields": fields,
            "start_frame_idx": 10,  # Non-zero start index
            "output_dir": str(tmp_path),
            "cmap": "hot",
            "vmin": 0.0,
            "vmax": 1.0,
            "frame_labels": None,  # No labels
            "dpi": 50,
        }

        # Should not raise error
        _render_worker_frames(task)

        # Check frames created with correct indices (11, 12, 13)
        png_files = sorted(tmp_path.glob("frame_*.png"))
        assert len(png_files) == 3


class TestCodecSelection:
    """Test video codec selection."""

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_h264_codec(self, tmp_path):
        """Test H264 codec selection."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "h264.mp4"

        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=5,
            codec="h264",
            n_workers=1,
        )

        assert output_path.exists()

    @pytest.mark.skipif(
        os.system("ffmpeg -version > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed",
    )
    def test_mpeg4_codec(self, tmp_path):
        """Test MPEG4 codec selection."""
        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(5)]

        output_path = tmp_path / "mpeg4.mp4"

        animate_fields(
            env,
            fields,
            backend="video",
            save_path=str(output_path),
            fps=5,
            codec="mpeg4",
            n_workers=1,
        )

        assert output_path.exists()
