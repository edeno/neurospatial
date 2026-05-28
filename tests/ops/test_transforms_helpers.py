"""Tests for the quick coordinate-conversion helpers in ops.transforms.

These pin the actual API of ``flip_y_data``, ``convert_to_cm``, and
``convert_to_pixels``. The functions operate on pixel/centimeter video
coordinates with a y-axis flip about the frame height; they do not take a
unit-string argument.
"""

import numpy as np
import pytest

from neurospatial.ops.transforms import (
    convert_to_cm,
    convert_to_pixels,
    flip_y_data,
)


class TestFlipYData:
    """flip_y_data flips y about the frame height, preserving x."""

    def test_flip_y_data_inverts_y(self):
        """y becomes (frame_height - y); x is unchanged."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = flip_y_data(data, frame_size_px=(10.0, 10.0))
        # y_new = H - y = 10 - y.
        np.testing.assert_allclose(result, [[1.0, 8.0], [3.0, 6.0]], atol=1e-10)

    def test_flip_y_data_preserves_x(self):
        """The x column is identical before and after the flip."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [7.5, 0.0]])
        result = flip_y_data(data, frame_size_px=(20.0, 12.0))
        np.testing.assert_allclose(result[:, 0], data[:, 0], atol=1e-10)

    def test_flip_y_data_preserves_batched_shape(self):
        """A (n_time, n_points, 2) array keeps its shape through the flip."""
        batch = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
        result = flip_y_data(batch, frame_size_px=(10.0, 10.0))
        assert result.shape == (2, 3, 2)


class TestConvertToCm:
    """convert_to_cm scales pixels to centimeters with a y-flip."""

    @pytest.mark.parametrize(
        "cm_per_px, point_px, expected_cm",
        [
            # x_cm = cm_per_px * x_px; y_cm = cm_per_px * (H - y_px).
            (1.0, [3.0, 4.0], [3.0, 6.0]),
            (2.0, [3.0, 4.0], [6.0, 12.0]),
            (0.5, [8.0, 2.0], [4.0, 4.0]),
        ],
    )
    def test_convert_to_cm_known_scale(self, cm_per_px, point_px, expected_cm):
        """Pixel coordinates convert to cm with the expected scale and y-flip."""
        result = convert_to_cm(
            np.array([point_px]), frame_size_px=(10.0, 10.0), cm_per_px=cm_per_px
        )
        np.testing.assert_allclose(result[0], expected_cm, atol=1e-10)

    def test_convert_to_cm_default_scale_is_identity_up_to_flip(self):
        """With cm_per_px=1.0 the conversion is just the y-flip."""
        point = np.array([[3.0, 4.0]])
        cm = convert_to_cm(point, frame_size_px=(10.0, 10.0))
        flipped = flip_y_data(point, frame_size_px=(10.0, 10.0))
        np.testing.assert_allclose(cm, flipped, atol=1e-10)


class TestConvertToPixels:
    """convert_to_pixels is the inverse of convert_to_cm."""

    @pytest.mark.parametrize("cm_per_px", [1.0, 2.0, 0.5])
    def test_convert_to_pixels_round_trip(self, cm_per_px):
        """cm -> px -> cm recovers the original coordinates exactly."""
        cm = np.array([[12.0, 8.0], [3.0, 5.0]])
        frame = (10.0, 10.0)
        px = convert_to_pixels(cm, frame_size_px=frame, cm_per_px=cm_per_px)
        back = convert_to_cm(px, frame_size_px=frame, cm_per_px=cm_per_px)
        np.testing.assert_allclose(back, cm, atol=1e-10)
