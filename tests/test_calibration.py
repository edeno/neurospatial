"""Tests for calibration module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.calibration import simple_scale
from neurospatial.transforms import Affine2D


class TestSimpleScale:
    """Tests for simple_scale function."""

    def test_basic_scaling_no_offset(self):
        """Test basic scaling without offset."""
        px_per_cm = 10.0  # 10 pixels per cm
        transform = simple_scale(px_per_cm)

        # Test that it's an Affine2D
        assert isinstance(transform, Affine2D)

        # Pixel coordinates
        points_px = np.array([[10.0, 20.0], [30.0, 40.0]])

        # Apply transform (should divide by 10)
        points_cm = transform(points_px)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])

        assert_allclose(points_cm, expected)

    def test_scaling_with_offset(self):
        """Test scaling with pixel offset."""
        px_per_cm = 10.0
        offset_px = (5.0, 10.0)

        transform = simple_scale(px_per_cm, offset_px)

        # Point at offset should become origin
        point_at_offset = np.array([[5.0, 10.0]])
        result = transform(point_at_offset)
        assert_allclose(result, np.array([[0.0, 0.0]]))

        # Point at (15, 30) should be (1, 2) cm from offset
        point = np.array([[15.0, 30.0]])
        result = transform(point)
        assert_allclose(result, np.array([[1.0, 2.0]]))

    def test_high_resolution(self):
        """Test with high pixel density."""
        px_per_cm = 100.0  # High resolution
        transform = simple_scale(px_per_cm)

        points_px = np.array([[100.0, 200.0]])
        points_cm = transform(points_px)
        expected = np.array([[1.0, 2.0]])

        assert_allclose(points_cm, expected)

    def test_low_resolution(self):
        """Test with low pixel density."""
        px_per_cm = 1.0  # 1 pixel per cm
        transform = simple_scale(px_per_cm)

        points_px = np.array([[5.0, 10.0]])
        points_cm = transform(points_px)
        expected = np.array([[5.0, 10.0]])

        assert_allclose(points_cm, expected)

    def test_fractional_pixels_per_cm(self):
        """Test with fractional pixels per centimeter."""
        px_per_cm = 2.5
        transform = simple_scale(px_per_cm)

        points_px = np.array([[5.0, 10.0]])
        points_cm = transform(points_px)
        expected = np.array([[2.0, 4.0]])

        assert_allclose(points_cm, expected)

    def test_negative_offset(self):
        """Test with negative offset."""
        px_per_cm = 10.0
        offset_px = (-5.0, -10.0)

        transform = simple_scale(px_per_cm, offset_px)

        # Negative offset should shift in positive direction
        points_px = np.array([[0.0, 0.0]])
        points_cm = transform(points_px)
        expected = np.array([[0.5, 1.0]])  # (0 - (-5))/10, (0 - (-10))/10

        assert_allclose(points_cm, expected)

    def test_zero_px_per_cm_raises(self):
        """Test that zero px_per_cm raises ValueError."""
        with pytest.raises(ValueError, match="px_per_cm must be nonzero"):
            simple_scale(0.0)

    def test_invalid_offset_type(self):
        """Test that invalid offset type raises ValueError."""
        with pytest.raises(
            ValueError, match="offset_px must be a tuple of two numeric values"
        ):
            simple_scale(10.0, offset_px="invalid")  # pyright: ignore[reportArgumentType]

    def test_invalid_offset_length(self):
        """Test that offset with wrong number of elements raises ValueError."""
        with pytest.raises(ValueError, match="offset_px must be a tuple"):
            simple_scale(10.0, offset_px=(1.0,))  # pyright: ignore[reportArgumentType]

    def test_inverse_transform(self):
        """Test that inverse transform works correctly."""
        px_per_cm = 10.0
        offset_px = (5.0, 10.0)

        transform = simple_scale(px_per_cm, offset_px)
        inverse = transform.inverse()

        # Apply forward then inverse should give back original
        points_px = np.array([[15.0, 30.0], [25.0, 50.0]])
        points_cm = transform(points_px)
        points_px_recovered = inverse(points_cm)

        assert_allclose(points_px_recovered, points_px, rtol=1e-10)

    def test_single_point(self):
        """Test with a single point."""
        px_per_cm = 10.0
        transform = simple_scale(px_per_cm)

        point_px = np.array([[100.0, 200.0]])
        point_cm = transform(point_px)

        assert point_cm.shape == (1, 2)
        assert_allclose(point_cm, np.array([[10.0, 20.0]]))

    def test_multiple_points(self):
        """Test with multiple points."""
        px_per_cm = 10.0
        transform = simple_scale(px_per_cm)

        points_px = np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0],
            ]
        )
        points_cm = transform(points_px)

        expected = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )

        assert_allclose(points_cm, expected)

    def test_matrix_structure(self):
        """Test that the transformation matrix has correct structure."""
        px_per_cm = 10.0
        offset_px = (5.0, 10.0)

        transform = simple_scale(px_per_cm, offset_px)
        A = transform.A

        # Check bottom row is [0, 0, 1]
        assert_allclose(A[2, :], [0.0, 0.0, 1.0])

        # Check diagonal scaling
        expected_scale = 1.0 / px_per_cm
        assert_allclose(A[0, 0], expected_scale)
        assert_allclose(A[1, 1], expected_scale)

        # Check off-diagonal elements are zero
        assert A[0, 1] == 0.0
        assert A[1, 0] == 0.0

    def test_composition_with_other_transforms(self):
        """Test that calibration transform can compose with other transforms."""
        from neurospatial.transforms import translate

        px_per_cm = 10.0
        calibration = simple_scale(px_per_cm)

        # Translate 5 cm in both directions (in cm space)
        shift = translate(5.0, 5.0)

        # Compose: first calibrate, then shift
        combined = shift @ calibration

        # Point at (10, 10) pixels -> (1, 1) cm -> (6, 6) cm after shift
        point_px = np.array([[10.0, 10.0]])
        result = combined(point_px)
        expected = np.array([[6.0, 6.0]])

        assert_allclose(result, expected)

    def test_zero_coordinates(self):
        """Test with zero coordinates."""
        px_per_cm = 10.0
        transform = simple_scale(px_per_cm)

        points_px = np.array([[0.0, 0.0]])
        points_cm = transform(points_px)

        assert_allclose(points_cm, np.array([[0.0, 0.0]]))

    def test_large_offset(self):
        """Test with large offset values."""
        px_per_cm = 10.0
        offset_px = (1000.0, 2000.0)

        transform = simple_scale(px_per_cm, offset_px)

        # Point at offset should be origin
        points_px = np.array([[1000.0, 2000.0]])
        points_cm = transform(points_px)

        assert_allclose(points_cm, np.array([[0.0, 0.0]]))

    def test_asymmetric_offset(self):
        """Test with different x and y offsets."""
        px_per_cm = 10.0
        offset_px = (10.0, 50.0)

        transform = simple_scale(px_per_cm, offset_px)

        points_px = np.array([[20.0, 60.0]])
        points_cm = transform(points_px)
        expected = np.array([[1.0, 1.0]])  # (20-10)/10, (60-50)/10

        assert_allclose(points_cm, expected)
