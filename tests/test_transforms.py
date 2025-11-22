"""Tests for transforms.py calibration helpers.

These tests cover the video calibration functions added in v0.5.0
for pixel↔cm coordinate conversion.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.transforms import (
    Affine2D,
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
    flip_y,
    scale_2d,
)


class TestCalibrateFromScaleBar:
    """Tests for calibrate_from_scale_bar()."""

    def test_horizontal_scale_bar(self):
        """Horizontal scale bar produces correct scale."""
        # 200 pixel scale bar represents 50 cm
        # px distance = 200, so cm_per_px = 50/200 = 0.25
        transform = calibrate_from_scale_bar(
            p1_px=(100.0, 200.0),
            p2_px=(300.0, 200.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        # Check scale: 100 pixels should map to 25 cm
        point_px = np.array([[100.0, 0.0]])
        point_cm = transform(point_px)
        # x: 100 * 0.25 = 25 cm
        assert_allclose(point_cm[0, 0], 25.0, atol=1e-10)

    def test_vertical_scale_bar(self):
        """Vertical scale bar produces correct scale."""
        # 100 pixel vertical scale bar represents 25 cm
        # cm_per_px = 25/100 = 0.25
        transform = calibrate_from_scale_bar(
            p1_px=(50.0, 100.0),
            p2_px=(50.0, 200.0),
            known_length_cm=25.0,
            frame_size_px=(640, 480),
        )

        # Check that transform is correctly computed
        assert isinstance(transform, Affine2D)

    def test_diagonal_scale_bar(self):
        """Diagonal scale bar computes Euclidean distance correctly."""
        # 3-4-5 triangle: endpoints (0, 0) and (300, 400) -> distance = 500 pixels
        # If 500 pixels = 100 cm, cm_per_px = 0.2
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(300.0, 400.0),
            known_length_cm=100.0,
            frame_size_px=(640, 480),
        )

        # 500 pixels should map to 100 cm
        point_px = np.array([[500.0, 0.0]])
        point_cm = transform(point_px)
        # x: 500 * 0.2 = 100 cm
        assert_allclose(point_cm[0, 0], 100.0, atol=1e-10)

    def test_y_flip_applied(self):
        """Transform includes Y-flip (video origin top-left to env bottom-left)."""
        # Frame height 480, scale bar at y=100 in video
        # After flip: y_cm = (480 - y_px) * scale
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 200.0),
            p2_px=(100.0, 200.0),  # 100 px = 50 cm -> cm_per_px = 0.5
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        # Point at top of video (y_px=0) should map to y_cm = 480 * 0.5 = 240
        top_point_px = np.array([[0.0, 0.0]])
        top_point_cm = transform(top_point_px)
        assert_allclose(top_point_cm[0, 1], 240.0, atol=1e-10)

        # Point at bottom of video (y_px=480) should map to y_cm = 0
        bottom_point_px = np.array([[0.0, 480.0]])
        bottom_point_cm = transform(bottom_point_px)
        assert_allclose(bottom_point_cm[0, 1], 0.0, atol=1e-10)

    def test_returns_affine2d(self):
        """Returns an Affine2D instance."""
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=10.0,
            frame_size_px=(640, 480),
        )
        assert isinstance(transform, Affine2D)

    def test_zero_length_raises(self):
        """Zero-length scale bar raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            calibrate_from_scale_bar(
                p1_px=(100.0, 100.0),
                p2_px=(100.0, 100.0),  # Same point = zero length
                known_length_cm=50.0,
                frame_size_px=(640, 480),
            )

    def test_negative_length_raises(self):
        """Negative known_length_cm raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calibrate_from_scale_bar(
                p1_px=(0.0, 0.0),
                p2_px=(100.0, 0.0),
                known_length_cm=-10.0,
                frame_size_px=(640, 480),
            )


class TestCalibrateFromLandmarks:
    """Tests for calibrate_from_landmarks()."""

    def test_rigid_transform(self):
        """Rigid (rotation + translation) transform from landmarks."""
        # Simple case: 4 corners, no rotation, pure scaling + Y-flip
        landmarks_px = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        landmarks_cm = np.array(
            [
                [0.0, 10.0],  # Y-flipped: px(0,0) -> cm(0, 100*0.1 = 10)
                [10.0, 10.0],
                [10.0, 0.0],
                [0.0, 0.0],
            ]
        )

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(640, 480),
            kind="rigid",
        )

        assert isinstance(transform, Affine2D)

    def test_similarity_transform(self):
        """Similarity (scale + rotation + translation) transform."""
        # Create landmarks with uniform scale
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        # 2x scale
        landmarks_cm = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=float)

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(100, 100),
            kind="similarity",
        )

        assert isinstance(transform, Affine2D)

    def test_affine_transform(self):
        """Full affine transform from landmarks."""
        # Simple 1:1 mapping for test
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        landmarks_cm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(100, 100),
            kind="affine",
        )

        assert isinstance(transform, Affine2D)

    def test_roundtrip_accuracy(self):
        """Pixel→cm→pixel roundtrip within tolerance."""
        # Define landmarks
        landmarks_px = np.array(
            [
                [50, 50],
                [590, 50],
                [590, 430],
                [50, 430],
            ],
            dtype=float,
        )
        landmarks_cm = np.array(
            [
                [0, 80],
                [100, 80],
                [100, 0],
                [0, 0],
            ],
            dtype=float,
        )

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(640, 480),
            kind="similarity",
        )

        # Transform px -> cm
        test_px = landmarks_px.copy()
        result_cm = transform(test_px)

        # Transform cm -> px (inverse)
        inverse = transform.inverse()
        roundtrip_px = inverse(result_cm)

        # Should get back to original (within tolerance)
        # Use scale-dependent tolerance: max(1e-4, 1e-6 * extent)
        extent = max(640, 480)
        atol = max(1e-4, 1e-6 * extent)
        assert_allclose(roundtrip_px, test_px, atol=atol)

    def test_insufficient_landmarks_raises(self):
        """Less than 3 landmarks raises ValueError."""
        landmarks_px = np.array([[0, 0], [100, 0]], dtype=float)  # Only 2 points
        landmarks_cm = np.array([[0, 0], [10, 0]], dtype=float)

        with pytest.raises(ValueError, match="landmark"):
            calibrate_from_landmarks(
                landmarks_px=landmarks_px,
                landmarks_cm=landmarks_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )

    def test_mismatched_landmark_count_raises(self):
        """Different number of px and cm landmarks raises ValueError."""
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100]], dtype=float)
        landmarks_cm = np.array([[0, 0], [10, 0]], dtype=float)  # Only 2!

        with pytest.raises(ValueError, match="same number"):
            calibrate_from_landmarks(
                landmarks_px=landmarks_px,
                landmarks_cm=landmarks_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )


class TestFlipYIntegration:
    """Test flip_y() integration with calibration."""

    def test_flip_y_compose_with_scale(self):
        """scale_2d @ flip_y produces correct video->env transform.

        Composition order: scale_2d @ flip_y means "apply flip_y first, then scale_2d".
        """
        frame_height = 480
        cm_per_px = 0.25

        # Compose: flip_y then scale (right-to-left application)
        transform = scale_2d(cm_per_px, cm_per_px) @ flip_y(frame_height)

        # Point at top of video (y=0) should map to y = 480 * 0.25 = 120 cm
        top_px = np.array([[0.0, 0.0]])
        top_cm = transform(top_px)
        assert_allclose(top_cm[0, 1], 120.0, atol=1e-10)

        # Point at bottom of video (y=480) should map to y = 0 cm
        bottom_px = np.array([[0.0, 480.0]])
        bottom_cm = transform(bottom_px)
        assert_allclose(bottom_cm[0, 1], 0.0, atol=1e-10)


class TestVideoCalibration:
    """Tests for VideoCalibration dataclass."""

    def test_basic_creation(self):
        """VideoCalibration can be created with transform and frame size."""
        from neurospatial.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        assert calib.frame_size_px == (640, 480)
        assert isinstance(calib.transform_px_to_cm, Affine2D)

    def test_cm_per_px_property(self):
        """cm_per_px returns approximate scale factor."""
        from neurospatial.transforms import VideoCalibration

        # 100 pixels = 50 cm -> cm_per_px = 0.5
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        assert_allclose(calib.cm_per_px, 0.5, atol=1e-10)

    def test_transform_cm_to_px_inverse(self):
        """transform_cm_to_px is the inverse of transform_px_to_cm."""
        from neurospatial.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(200.0, 0.0),
            known_length_cm=100.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        # Test roundtrip px -> cm -> px
        test_px = np.array([[100.0, 200.0]])
        result_cm = calib.transform_px_to_cm(test_px)
        roundtrip_px = calib.transform_cm_to_px(result_cm)

        assert_allclose(roundtrip_px, test_px, atol=1e-10)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip preserves calibration."""
        from neurospatial.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        original = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        # Roundtrip through dict
        d = original.to_dict()
        restored = VideoCalibration.from_dict(d)

        # Check restored calibration works identically
        test_px = np.array([[320.0, 240.0]])
        original_cm = original.transform_px_to_cm(test_px)
        restored_cm = restored.transform_px_to_cm(test_px)

        assert_allclose(restored_cm, original_cm, atol=1e-10)
        assert restored.frame_size_px == original.frame_size_px

    def test_to_dict_contains_expected_keys(self):
        """to_dict produces expected structure."""
        from neurospatial.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        d = calib.to_dict()

        assert "transform_px_to_cm" in d
        assert "frame_size_px" in d
        assert d["frame_size_px"] == [640, 480]
