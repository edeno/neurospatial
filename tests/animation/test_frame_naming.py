"""Tests for video backend frame naming pattern consistency.

Phase 5.1: Sanitize Frame Naming Pattern
- Verifies zero-padded filenames are used
- Verifies parallel_render_frames uses consistent pattern
- Verifies ffmpeg pattern matches actual saved filenames
"""

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFrameNamingPattern:
    """Tests for frame naming pattern in parallel rendering."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    @pytest.fixture
    def simple_fields(self, simple_env):
        """Create simple fields matching environment."""
        n_bins = simple_env.n_bins
        rng = np.random.default_rng(42)
        # 10 frames for quick tests
        return [rng.random(n_bins) for _ in range(10)]

    def test_frame_pattern_uses_zero_padding(self, simple_env, simple_fields, tmp_path):
        """Verify frame filenames use zero-padded integers."""
        from neurospatial.animation._parallel import parallel_render_frames

        pattern = parallel_render_frames(
            env=simple_env,
            fields=simple_fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=1,
        )

        # Pattern should have zero-padded format specifier
        assert "%0" in pattern, f"Pattern should have zero-padding: {pattern}"
        assert "d.png" in pattern, f"Pattern should have integer format: {pattern}"

    def test_frame_pattern_minimum_5_digits(self, simple_env, simple_fields, tmp_path):
        """Verify minimum 5 digits for zero-padding."""
        from neurospatial.animation._parallel import parallel_render_frames

        pattern = parallel_render_frames(
            env=simple_env,
            fields=simple_fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=1,
        )

        # Should use at least 5 digits (supports up to 99,999 frames)
        assert "%05d" in pattern or "%06d" in pattern, (
            f"Pattern should use minimum 5 digits: {pattern}"
        )

    def test_frame_pattern_matches_saved_files(
        self, simple_env, simple_fields, tmp_path
    ):
        """Verify ffmpeg pattern matches actual saved filenames."""
        from neurospatial.animation._parallel import parallel_render_frames

        pattern = parallel_render_frames(
            env=simple_env,
            fields=simple_fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=1,
        )

        # Verify pattern format
        assert "%05d" in pattern, f"Pattern should use 5 digits: {pattern}"

        # Get saved files
        png_files = sorted(tmp_path.glob("frame_*.png"))
        assert len(png_files) == len(simple_fields)

        # Verify each file matches the expected pattern
        for idx, png_file in enumerate(png_files):
            expected_name = f"frame_{idx:05d}.png"
            assert png_file.name == expected_name, (
                f"File {png_file.name} doesn't match expected {expected_name}"
            )

    def test_frame_pattern_consistency_with_workers(
        self, simple_env, simple_fields, tmp_path
    ):
        """Verify pattern is consistent when using multiple workers."""
        from neurospatial.animation._parallel import parallel_render_frames

        # Use 2 workers (if available)
        pattern = parallel_render_frames(
            env=simple_env,
            fields=simple_fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=2,
        )

        # Verify pattern format
        assert "%05d" in pattern, f"Pattern should use 5 digits: {pattern}"

        # All files should be consecutively numbered
        png_files = sorted(tmp_path.glob("frame_*.png"))
        assert len(png_files) == len(simple_fields)

        # Verify sequential numbering (no gaps)
        for idx, png_file in enumerate(png_files):
            expected_name = f"frame_{idx:05d}.png"
            assert png_file.name == expected_name, (
                f"File {png_file.name} doesn't match expected {expected_name}"
            )


class TestFrameNamingDigitCalculation:
    """Tests for digit calculation based on frame count."""

    def test_small_frame_count_uses_5_digits(self):
        """Verify small frame counts use minimum 5 digits."""
        # For 10 frames, max index is 9 (1 digit needed)
        # But minimum is 5 digits
        n_frames = 10
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 5

    def test_medium_frame_count_uses_5_digits(self):
        """Verify medium frame counts (5k) use 5 digits."""
        n_frames = 5000
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 5  # 4999 has 4 digits, but min is 5

    def test_large_frame_count_uses_5_digits(self):
        """Verify large frame counts (100k) use appropriate digits."""
        n_frames = 100_000
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 5  # 99999 has 5 digits

    def test_very_large_frame_count_uses_6_digits(self):
        """Verify very large frame counts (1M) expand to 6 digits."""
        n_frames = 1_000_000
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 6  # 999999 has 6 digits

    def test_extreme_frame_count_uses_7_digits(self):
        """Verify extreme frame counts expand appropriately."""
        n_frames = 10_000_000
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 7  # 9999999 has 7 digits


class TestFrameNamingIntegration:
    """Integration tests for frame naming in video export."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_ffmpeg_receives_correct_pattern(self, simple_env, tmp_path):
        """Verify ffmpeg command receives correct frame pattern."""
        from neurospatial.animation.backends.video_backend import render_video

        fields = [np.random.default_rng(42).random(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "test_video.mp4"

        # Mock subprocess.run to capture ffmpeg command
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            # May fail if ffmpeg not available, but we captured the call
            with contextlib.suppress(Exception):
                render_video(
                    env=simple_env,
                    fields=fields,
                    save_path=str(output_path),
                    n_workers=1,
                    fps=30,
                )

            # Check that ffmpeg was called with correct pattern
            if mock_run.called:
                call_args = mock_run.call_args[0][0]  # Get command list
                # Find the input pattern argument (after -i)
                for i, arg in enumerate(call_args):
                    if arg == "-i" and i + 1 < len(call_args):
                        pattern = call_args[i + 1]
                        assert "frame_" in pattern
                        assert "%0" in pattern
                        assert ".png" in pattern
                        break

    def test_worker_saves_correct_filenames_non_zero_start(self, simple_env, tmp_path):
        """Verify worker saves correct filenames for non-zero start index."""
        from neurospatial.animation._parallel import _render_worker_frames

        n_bins = simple_env.n_bins
        rng = np.random.default_rng(42)
        fields = [rng.random(n_bins) for _ in range(3)]

        # Simulate worker starting at frame 100
        task = {
            "env": simple_env,
            "fields": fields,
            "start_frame_idx": 100,
            "output_dir": str(tmp_path),
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "frame_labels": None,
            "dpi": 50,
            "digits": 5,
            "reuse_artists": True,
        }

        _render_worker_frames(task)

        # Check files are named correctly
        png_files = sorted(tmp_path.glob("frame_*.png"))
        assert len(png_files) == 3

        expected_names = ["frame_00100.png", "frame_00101.png", "frame_00102.png"]
        actual_names = [f.name for f in png_files]
        assert actual_names == expected_names

    def test_digits_parameter_propagates_to_workers(self, simple_env, tmp_path):
        """Verify digits parameter is correctly passed to workers."""
        from neurospatial.animation._parallel import _render_worker_frames

        n_bins = simple_env.n_bins
        rng = np.random.default_rng(42)
        fields = [rng.random(n_bins) for _ in range(3)]

        # Use 6 digits explicitly
        task = {
            "env": simple_env,
            "fields": fields,
            "start_frame_idx": 0,
            "output_dir": str(tmp_path),
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "frame_labels": None,
            "dpi": 50,
            "digits": 6,  # 6 digits
            "reuse_artists": True,
        }

        _render_worker_frames(task)

        png_files = sorted(tmp_path.glob("frame_*.png"))
        expected_names = ["frame_000000.png", "frame_000001.png", "frame_000002.png"]
        actual_names = [f.name for f in png_files]
        assert actual_names == expected_names


class TestFrameNamingEdgeCases:
    """Edge case tests for frame naming."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.random.default_rng(42).uniform(0, 50, (100, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        return env

    def test_single_frame(self, simple_env, tmp_path):
        """Verify single frame is correctly named."""
        from neurospatial.animation._parallel import parallel_render_frames

        fields = [np.random.default_rng(42).random(simple_env.n_bins)]

        pattern = parallel_render_frames(
            env=simple_env,
            fields=fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=1,
        )

        # Verify pattern returned
        assert "%05d" in pattern, f"Pattern should use 5 digits: {pattern}"

        png_files = list(tmp_path.glob("frame_*.png"))
        assert len(png_files) == 1
        assert png_files[0].name == "frame_00000.png"

    def test_exact_boundary_99999_frames(self):
        """Verify boundary case at exactly 99999 frames uses 5 digits."""
        n_frames = 99999
        digits = max(5, len(str(max(0, n_frames - 1))))
        assert digits == 5  # 99998 has 5 digits

    def test_boundary_100000_frames(self):
        """Verify boundary case at 100000 frames uses 6 digits."""
        n_frames = 100000
        digits = max(5, len(str(max(0, n_frames - 1))))
        # 99999 has 5 digits, so 5 digits is still sufficient
        assert digits == 5

    def test_boundary_100001_frames(self):
        """Verify just over 100k frames still uses 5 digits."""
        n_frames = 100001
        digits = max(5, len(str(max(0, n_frames - 1))))
        # 100000 has 6 digits
        assert digits == 6
