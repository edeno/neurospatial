"""Tests for the calibrate_video convenience function.

This module tests the calibrate_video() function which provides a
high-level API for calibrating video coordinates to environment space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.transforms import VideoCalibration

# We'll import calibrate_video from the animation module when implemented
# For TDD, tests are written first before implementation exists


@pytest.fixture
def tiny_video_path(tmp_path: Path) -> Path:
    """Create a tiny test video (16x16 pixels, 10 frames, 10 fps).

    Frame size is (16, 16) for simple coordinate math.
    """
    import cv2

    video_path = tmp_path / "test_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    frame_size = (16, 16)  # width, height
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    for i in range(10):
        frame = np.full((16, 16, 3), i * 25, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def simple_env_16cm() -> Environment:
    """Create a 16x16 cm environment (1 cm per pixel at video resolution).

    Note: We use positions inset by 0.5 to account for bin_size=1.0 which
    creates bins extending half a bin size beyond positions. This ensures
    the environment dimension_ranges match [0, 16] exactly.
    """
    positions = np.array(
        [
            [0.5, 0.5],
            [15.5, 0.5],
            [15.5, 15.5],
            [0.5, 15.5],
        ]
    )
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture
def large_env_100cm() -> Environment:
    """Create a 100x80 cm environment (larger than video coverage)."""
    positions = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 80.0],
            [0.0, 80.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=5.0)


class TestCalibrateVideoScaleBar:
    """Tests for calibrate_video() with scale_bar method."""

    def test_scale_bar_calibration(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Scale bar method produces correct VideoCalibration."""
        from neurospatial.animation import calibrate_video

        # Scale bar: 8 pixels (from x=0 to x=8) = 8 cm
        # So cm_per_px = 1.0
        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            scale_bar=((0.0, 8.0), (8.0, 8.0), 8.0),  # ((x1, y1), (x2, y2), length_cm)
        )

        assert isinstance(calibration, VideoCalibration)
        assert calibration.frame_size_px == (16, 16)
        assert calibration.cm_per_px == pytest.approx(1.0, rel=0.01)

    def test_scale_bar_with_different_scale(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Scale bar with 0.5 cm/px produces correct scale."""
        from neurospatial.animation import calibrate_video

        # 8 pixels = 4 cm -> cm_per_px = 0.5
        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            scale_bar=((0.0, 8.0), (8.0, 8.0), 4.0),
        )

        assert calibration.cm_per_px == pytest.approx(0.5, rel=0.01)


class TestCalibrateVideoLandmarks:
    """Tests for calibrate_video() with landmarks method."""

    def test_landmarks_calibration(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Landmark method produces correct VideoCalibration."""
        from neurospatial.animation import calibrate_video

        # Map video corners to environment corners
        # Video (16x16 px) -> Env (16x16 cm)
        # Note: Y is flipped (video y=0 is top, env y=0 is bottom)
        landmarks_px = np.array(
            [
                [0.0, 16.0],  # video bottom-left -> env (0, 0)
                [16.0, 16.0],  # video bottom-right -> env (16, 0)
                [16.0, 0.0],  # video top-right -> env (16, 16)
                [0.0, 0.0],  # video top-left -> env (0, 16)
            ]
        )
        landmarks_env = np.array(
            [
                [0.0, 0.0],
                [16.0, 0.0],
                [16.0, 16.0],
                [0.0, 16.0],
            ]
        )

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            landmarks_px=landmarks_px,
            landmarks_env=landmarks_env,
        )

        assert isinstance(calibration, VideoCalibration)
        assert calibration.frame_size_px == (16, 16)

    def test_landmarks_with_different_scale(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Landmarks with 2x scale produce correct transform."""
        from neurospatial.animation import calibrate_video

        # Video (16x16 px) -> Env (32x32 cm) = 2 cm/px
        landmarks_px = np.array(
            [
                [0.0, 16.0],
                [16.0, 16.0],
                [16.0, 0.0],
                [0.0, 0.0],
            ]
        )
        landmarks_env = np.array(
            [
                [0.0, 0.0],
                [32.0, 0.0],
                [32.0, 32.0],
                [0.0, 32.0],
            ]
        )

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            landmarks_px=landmarks_px,
            landmarks_env=landmarks_env,
        )

        assert calibration.cm_per_px == pytest.approx(2.0, rel=0.05)


class TestCalibrateVideoCmPerPx:
    """Tests for calibrate_video() with direct cm_per_px method."""

    def test_cm_per_px_calibration(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Direct cm_per_px produces correct VideoCalibration."""
        from neurospatial.animation import calibrate_video

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            cm_per_px=1.0,
        )

        assert isinstance(calibration, VideoCalibration)
        assert calibration.cm_per_px == pytest.approx(1.0, rel=0.01)
        assert calibration.frame_size_px == (16, 16)

    def test_cm_per_px_with_different_scale(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Different cm_per_px values work correctly."""
        from neurospatial.animation import calibrate_video

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            cm_per_px=0.5,
        )

        assert calibration.cm_per_px == pytest.approx(0.5, rel=0.01)


class TestCalibrateVideoBoundsValidation:
    """Tests for bounds coverage validation."""

    def test_warns_when_env_exceeds_video(
        self, tiny_video_path: Path, large_env_100cm: Environment
    ):
        """Warns when environment bounds exceed calibrated video coverage."""
        from neurospatial.animation import calibrate_video

        # Video is 16x16 px with 1 cm/px = 16x16 cm coverage
        # Env is 100x80 cm - much larger
        with pytest.warns(UserWarning, match="extend beyond"):
            calibrate_video(
                tiny_video_path,
                large_env_100cm,
                cm_per_px=1.0,
            )

    def test_no_warning_when_video_covers_env(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """No warning when video coverage includes full environment."""
        import warnings

        from neurospatial.animation import calibrate_video

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warnings
            calibration = calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                cm_per_px=1.0,
            )
            assert calibration is not None


class TestCalibrateVideoErrors:
    """Tests for error handling."""

    def test_no_method_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises ValueError when no calibration method is specified."""
        from neurospatial.animation import calibrate_video

        with pytest.raises(ValueError, match="calibration method"):
            calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                # No calibration parameters provided
            )

    def test_conflicting_methods_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises ValueError when multiple calibration methods specified."""
        from neurospatial.animation import calibrate_video

        with pytest.raises(ValueError, match="mutually exclusive"):
            calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                scale_bar=((0.0, 0.0), (10.0, 0.0), 10.0),
                cm_per_px=1.0,  # Conflicting!
            )

    def test_invalid_video_path_raises(self, simple_env_16cm: Environment):
        """Raises error for non-existent video file."""
        from neurospatial.animation import calibrate_video

        with pytest.raises((FileNotFoundError, ValueError)):
            calibrate_video(
                "/nonexistent/video.mp4",
                simple_env_16cm,
                cm_per_px=1.0,
            )

    def test_invalid_cm_per_px_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises ValueError for non-positive cm_per_px."""
        from neurospatial.animation import calibrate_video

        with pytest.raises(ValueError, match="positive"):
            calibrate_video(tiny_video_path, simple_env_16cm, cm_per_px=-1.0)

        with pytest.raises(ValueError, match="positive"):
            calibrate_video(tiny_video_path, simple_env_16cm, cm_per_px=0.0)

    def test_landmarks_mismatched_length_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises when landmarks_px and landmarks_env have different lengths."""
        from neurospatial.animation import calibrate_video

        landmarks_px = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        landmarks_env = np.array([[0.0, 0.0], [10.0, 0.0]])  # Only 2 points!

        with pytest.raises(ValueError, match="same number"):
            calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                landmarks_px=landmarks_px,
                landmarks_env=landmarks_env,
            )

    def test_landmarks_only_px_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises when only landmarks_px provided without landmarks_env."""
        from neurospatial.animation import calibrate_video

        landmarks_px = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="landmarks_env"):
            calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                landmarks_px=landmarks_px,
                # No landmarks_env!
            )

    def test_landmarks_only_env_raises(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Raises when only landmarks_env provided without landmarks_px."""
        from neurospatial.animation import calibrate_video

        landmarks_env = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError, match="landmarks_px"):
            calibrate_video(
                tiny_video_path,
                simple_env_16cm,
                landmarks_env=landmarks_env,
                # No landmarks_px!
            )


class TestCalibrateVideoTransformAccuracy:
    """Tests for transform accuracy."""

    def test_roundtrip_accuracy(
        self, tiny_video_path: Path, simple_env_16cm: Environment
    ):
        """Pixel -> cm -> pixel roundtrip is accurate."""
        from neurospatial.animation import calibrate_video

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            cm_per_px=1.0,
        )

        # Test point
        point_px = np.array([[8.0, 8.0]])

        # Forward transform
        point_cm = calibration.transform_px_to_cm(point_px)

        # Inverse transform
        point_px_back = calibration.transform_cm_to_px(point_cm)

        # Should match original within tolerance
        assert_allclose(point_px_back, point_px, atol=1e-10)

    def test_y_flip_applied(self, tiny_video_path: Path, simple_env_16cm: Environment):
        """Calibration includes Y-flip (video top-left to env bottom-left)."""
        from neurospatial.animation import calibrate_video

        calibration = calibrate_video(
            tiny_video_path,
            simple_env_16cm,
            cm_per_px=1.0,
        )

        # Video top-left (0, 0) should map to env top (0, 16)
        top_left_px = np.array([[0.0, 0.0]])
        top_left_cm = calibration.transform_px_to_cm(top_left_px)
        assert top_left_cm[0, 1] == pytest.approx(16.0, rel=0.01)  # Y = frame_height

        # Video bottom-left (0, 16) should map to env bottom (0, 0)
        bottom_left_px = np.array([[0.0, 16.0]])
        bottom_left_cm = calibration.transform_px_to_cm(bottom_left_px)
        assert bottom_left_cm[0, 1] == pytest.approx(0.0, rel=0.01)
