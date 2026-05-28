"""Behavioral tests for ImageMaskLayout driven directly through the engine."""

import numpy as np

from neurospatial.layout.engines.image_mask import ImageMaskLayout


def test_loads_from_synthetic_mask(synthetic_16x16_mask):
    """Active bin count equals the number of True pixels in the mask.

    The engine exposes no ``n_bins`` attribute, so the active count is read
    from ``bin_centers.shape[0]``.
    """
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, bin_size=1.0)

    assert layout.bin_centers.shape[0] == int(synthetic_16x16_mask.sum())


def test_point_outside_image_bounds_returns_negative_one(synthetic_16x16_mask):
    """A point well outside the image extent maps to -1."""
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, bin_size=1.0)

    result = layout.point_to_bin_index(np.array([[1000.0, 1000.0]]))
    assert result[0] == -1


def test_point_in_active_pixel_returns_valid_bin(synthetic_16x16_mask):
    """A point inside an active (True) pixel returns a valid bin index."""
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, bin_size=1.0)

    # The disk is centered, so the image center (7.5, 7.5) is active.
    result = layout.point_to_bin_index(np.array([[7.5, 7.5]]))
    assert result[0] >= 0


def test_pixel_size_scaling(synthetic_16x16_mask):
    """Doubling bin_size scales every bin center linearly by the same factor."""
    layout_1 = ImageMaskLayout()
    layout_1.build(image_mask=synthetic_16x16_mask, bin_size=1.0)

    layout_2 = ImageMaskLayout()
    layout_2.build(image_mask=synthetic_16x16_mask, bin_size=2.0)

    # Same active mask -> same number/ordering of bins, centers scaled by 2.
    assert layout_2.bin_centers.shape == layout_1.bin_centers.shape
    np.testing.assert_allclose(
        layout_2.bin_centers, 2.0 * layout_1.bin_centers, atol=1e-10
    )
