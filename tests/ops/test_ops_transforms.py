"""Tests for ops/transforms module (new location).

This test file verifies that transforms functions can be imported from
their new location at neurospatial.ops.transforms.

The canonical implementations remain in the original test files
(tests/test_transforms.py, tests/test_calibration.py).
This file tests the new import paths work correctly.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Test imports from new location
from neurospatial.ops.transforms import (
    # Core classes
    Affine2D,
    Affine3D,
    AffineND,
    SpatialTransform,
    VideoCalibration,
    # Transform estimation
    apply_transform_to_environment,
    # Calibration functions
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
    # Helper functions
    convert_to_cm,
    convert_to_pixels,
    estimate_transform,
    # 2D factory functions
    flip_y,
    flip_y_data,
    # 3D factory functions
    from_rotation_matrix,
    identity,
    identity_nd,
    scale_2d,
    scale_3d,
    # From calibration.py (merged)
    simple_scale,
    translate,
    translate_3d,
)


class TestNewImportPaths:
    """Test that all exports are accessible from ops.transforms."""

    def test_core_classes_importable(self):
        """Core transform classes are importable."""
        assert Affine2D is not None
        assert AffineND is not None
        assert Affine3D is not None
        assert SpatialTransform is not None
        assert VideoCalibration is not None

    def test_2d_factories_importable(self):
        """2D factory functions are importable."""
        assert identity is not None
        assert scale_2d is not None
        assert translate is not None
        assert flip_y is not None

    def test_3d_factories_importable(self):
        """3D factory functions are importable."""
        assert translate_3d is not None
        assert scale_3d is not None
        assert identity_nd is not None
        assert from_rotation_matrix is not None

    def test_calibration_functions_importable(self):
        """Calibration functions are importable."""
        assert calibrate_from_scale_bar is not None
        assert calibrate_from_landmarks is not None
        assert simple_scale is not None

    def test_helper_functions_importable(self):
        """Helper functions are importable."""
        assert flip_y_data is not None
        assert convert_to_cm is not None
        assert convert_to_pixels is not None

    def test_transform_estimation_importable(self):
        """Transform estimation functions are importable."""
        assert estimate_transform is not None
        assert apply_transform_to_environment is not None


class TestAffine2DBasic:
    """Basic tests for Affine2D from new location."""

    def test_identity_transform(self):
        """Identity transform returns same points."""
        transform = identity()
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = transform(points)
        assert_allclose(result, points)

    def test_translation(self):
        """Translation moves points correctly."""
        transform = translate(10.0, 20.0)
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = transform(points)
        expected = np.array([[10.0, 20.0], [11.0, 21.0]])
        assert_allclose(result, expected)

    def test_scaling(self):
        """Scaling multiplies coordinates correctly."""
        transform = scale_2d(2.0)
        points = np.array([[1.0, 2.0]])
        result = transform(points)
        expected = np.array([[2.0, 4.0]])
        assert_allclose(result, expected)

    def test_composition(self):
        """Transform composition works correctly."""
        t1 = translate(10.0, 0.0)
        t2 = scale_2d(2.0)
        combined = t1 @ t2
        points = np.array([[1.0, 1.0]])
        result = combined(points)
        # First scale (2, 2), then translate (12, 2)
        expected = np.array([[12.0, 2.0]])
        assert_allclose(result, expected)

    def test_inverse(self):
        """Inverse transform reverses transformation."""
        transform = translate(10.0, 20.0)
        inverse = transform.inverse()
        points = np.array([[10.0, 20.0]])
        result = inverse(points)
        expected = np.array([[0.0, 0.0]])
        assert_allclose(result, expected)


class TestAffineNDBasic:
    """Basic tests for AffineND (3D) from new location."""

    def test_3d_translation(self):
        """3D translation moves points correctly."""
        transform = translate_3d(10.0, 20.0, 30.0)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = transform(points)
        expected = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
        assert_allclose(result, expected)

    def test_3d_scaling(self):
        """3D scaling multiplies coordinates correctly."""
        transform = scale_3d(2.0)
        points = np.array([[1.0, 2.0, 3.0]])
        result = transform(points)
        expected = np.array([[2.0, 4.0, 6.0]])
        assert_allclose(result, expected)

    def test_3d_identity(self):
        """3D identity transform returns same points."""
        transform = identity_nd(n_dims=3)
        points = np.array([[1.0, 2.0, 3.0]])
        result = transform(points)
        assert_allclose(result, points)


class TestSimpleScale:
    """Tests for simple_scale function (merged from calibration.py)."""

    def test_basic_scaling(self):
        """Basic scaling without offset."""
        transform = simple_scale(px_per_cm=10.0)
        assert isinstance(transform, Affine2D)

        points = np.array([[10.0, 20.0]])
        result = transform(points)
        expected = np.array([[1.0, 2.0]])
        assert_allclose(result, expected)

    def test_scaling_with_offset(self):
        """Scaling with pixel offset."""
        transform = simple_scale(px_per_cm=10.0, offset_px=(5.0, 10.0))

        # Point at offset should become origin
        points = np.array([[5.0, 10.0]])
        result = transform(points)
        expected = np.array([[0.0, 0.0]])
        assert_allclose(result, expected)

    def test_zero_px_per_cm_raises(self):
        """Zero px_per_cm raises ValueError."""
        with pytest.raises(ValueError, match="px_per_cm must be nonzero"):
            simple_scale(px_per_cm=0.0)


class TestCalibrateFromScaleBar:
    """Tests for calibrate_from_scale_bar from new location."""

    def test_horizontal_scale_bar(self):
        """Horizontal scale bar produces correct scale."""
        transform = calibrate_from_scale_bar(
            p1_px=(100.0, 200.0),
            p2_px=(300.0, 200.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        assert isinstance(transform, Affine2D)

    def test_zero_length_raises(self):
        """Zero-length scale bar raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            calibrate_from_scale_bar(
                p1_px=(100.0, 100.0),
                p2_px=(100.0, 100.0),
                known_length_cm=50.0,
                frame_size_px=(640, 480),
            )


class TestEstimateTransform:
    """Tests for estimate_transform from new location."""

    def test_rigid_2d(self):
        """Estimate rigid 2D transform."""
        src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        # Translate by (5, 5)
        dst = src + np.array([5.0, 5.0])

        transform = estimate_transform(src, dst, kind="rigid")
        result = transform(src)
        assert_allclose(result, dst, atol=1e-10)

    def test_similarity_2d(self):
        """Estimate similarity 2D transform."""
        src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        # Scale by 2 and translate
        dst = src * 2.0 + np.array([5.0, 5.0])

        transform = estimate_transform(src, dst, kind="similarity")
        result = transform(src)
        assert_allclose(result, dst, atol=1e-10)


class TestVideoCalibration:
    """Tests for VideoCalibration class from new location."""

    def test_from_scale_bar(self):
        """VideoCalibration can store transform from scale bar."""
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        calib = VideoCalibration(transform, frame_size_px=(640, 480))

        assert calib.transform_px_to_cm is not None
        assert calib.frame_size_px == (640, 480)
        assert calib.cm_per_px > 0

    def test_serialization(self):
        """VideoCalibration can be serialized and restored."""
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        calib = VideoCalibration(transform, frame_size_px=(640, 480))

        # Serialize
        d = calib.to_dict()
        assert "transform_px_to_cm" in d
        assert "frame_size_px" in d

        # Restore
        restored = VideoCalibration.from_dict(d)
        assert restored.frame_size_px == calib.frame_size_px
        assert_allclose(restored.transform_px_to_cm.A, calib.transform_px_to_cm.A)
