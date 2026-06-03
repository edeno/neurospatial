"""Behavioral tests for ImageMaskLayout driven directly through the engine."""

import numpy as np
import pytest

from neurospatial.layout.engines.image_mask import ImageMaskLayout


def test_loads_from_synthetic_mask(synthetic_16x16_mask):
    """Active bin count equals the number of True pixels in the mask.

    The engine exposes no ``n_bins`` attribute, so the active count is read
    from ``bin_centers.shape[0]``.
    """
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, pixel_size=1.0)

    assert layout.bin_centers.shape[0] == int(synthetic_16x16_mask.sum())


def test_point_outside_image_bounds_returns_negative_one(synthetic_16x16_mask):
    """A point well outside the image extent maps to -1."""
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, pixel_size=1.0)

    result = layout.point_to_bin_index(np.array([[1000.0, 1000.0]]))
    assert result[0] == -1


def test_point_in_active_pixel_returns_valid_bin(synthetic_16x16_mask):
    """A point inside an active (True) pixel returns a valid bin index."""
    layout = ImageMaskLayout()
    layout.build(image_mask=synthetic_16x16_mask, pixel_size=1.0)

    # The disk is centered, so the image center (7.5, 7.5) is active.
    result = layout.point_to_bin_index(np.array([[7.5, 7.5]]))
    assert result[0] >= 0


def test_pixel_size_scaling(synthetic_16x16_mask):
    """Doubling pixel_size scales every bin center linearly by the same factor."""
    layout_1 = ImageMaskLayout()
    layout_1.build(image_mask=synthetic_16x16_mask, pixel_size=1.0)

    layout_2 = ImageMaskLayout()
    layout_2.build(image_mask=synthetic_16x16_mask, pixel_size=2.0)

    # Same active mask -> same number/ordering of bins, centers scaled by 2.
    assert layout_2.bin_centers.shape == layout_1.bin_centers.shape
    np.testing.assert_allclose(
        layout_2.bin_centers, 2.0 * layout_1.bin_centers, atol=1e-10
    )


def test_image_mask_nonsquare_pixel_bin_lookup():
    """A non-square mask with anisotropic pixels digitizes x against x edges.

    Build a 2-row x 5-col mask with pixel width 3 (x) and height 1 (y). The
    center of pixel ``[row=0, col=4]`` is world ``(x=13.5, y=0.5)``. With the
    (x, y) convention the query maps to the bin whose center is ``[13.5, 0.5]``;
    with a swapped (y, x) order x=13.5 would be digitized against the y edges
    (max 2.0) and return -1.
    """
    image_mask = np.ones((2, 5), dtype=bool)  # 2 rows (y), 5 cols (x)
    layout = ImageMaskLayout()
    layout.build(image_mask=image_mask, pixel_size=(3.0, 1.0))

    assert layout.dimension_ranges == ((0.0, 15.0), (0.0, 2.0))
    assert layout.grid_edges[0].shape == (6,)  # x edges
    assert layout.grid_edges[1].shape == (3,)  # y edges

    result = layout.point_to_bin_index(np.array([[13.5, 0.5]]))
    assert result[0] >= 0
    np.testing.assert_allclose(layout.bin_centers[result[0]], [13.5, 0.5])


def test_image_mask_bin_centers_match_dimension_ranges():
    """Every bin center lies within its own (x, y) dimension range."""
    image_mask = np.ones((2, 5), dtype=bool)
    layout = ImageMaskLayout()
    layout.build(image_mask=image_mask, pixel_size=(3.0, 1.0))

    (x_lo, x_hi), (y_lo, y_hi) = layout.dimension_ranges
    assert np.all(layout.bin_centers[:, 0] >= x_lo)
    assert np.all(layout.bin_centers[:, 0] <= x_hi)
    assert np.all(layout.bin_centers[:, 1] >= y_lo)
    assert np.all(layout.bin_centers[:, 1] <= y_hi)


def test_image_mask_pixel_size_param_and_bin_size_alias():
    """pixel_size and the deprecated bin_size alias are equivalent; both errors."""
    image_mask = np.ones((3, 4), dtype=bool)

    layout_pixel = ImageMaskLayout()
    layout_pixel.build(image_mask=image_mask, pixel_size=2.0)

    layout_alias = ImageMaskLayout()
    with pytest.warns(DeprecationWarning):
        layout_alias.build(image_mask=image_mask, bin_size=2.0)

    np.testing.assert_allclose(layout_alias.bin_centers, layout_pixel.bin_centers)

    layout_both = ImageMaskLayout()
    with pytest.raises(ValueError):
        layout_both.build(image_mask=image_mask, pixel_size=2.0, bin_size=2.0)


def test_image_mask_square_pixel_unchanged():
    """Regression guard: square mask + scalar pixel_size matches known values.

    The square / scalar case was correct by symmetry before the (x, y) fix, so
    bin counts, centers, and connectivity edge counts are pinned here.
    """
    image_mask = np.array([[True, True], [True, False]], dtype=bool)  # 2x2, one hole
    layout = ImageMaskLayout()
    layout.build(image_mask=image_mask, pixel_size=1.0)

    assert layout.bin_centers.shape[0] == 3
    # Active pixels (row, col): (0,0), (0,1), (1,0) -> (x, y) centers.
    expected = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]])
    np.testing.assert_allclose(
        np.sort(layout.bin_centers, axis=0), np.sort(expected, axis=0)
    )
    # Diagonal connections on by default: all three active pixels are mutually
    # adjacent (two orthogonal + one diagonal) -> 3 edges.
    assert layout.connectivity.number_of_edges() == 3
