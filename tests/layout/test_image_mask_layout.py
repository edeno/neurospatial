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

    result = layout.binning().locator.locate(np.array([[1000.0, 1000.0]]))
    assert result[0] == -1


def test_point_lookup_returns_the_pixel_that_holds_it():
    """A point maps to the active pixel it falls in, or -1 on an inactive one.

    Pixel ``[r, c]`` spans ``[c * px, (c + 1) * px) x [r * py, (r + 1) * py)``.
    The mask and pixels are asymmetric, so swapping rows and columns or the
    two pixel sides puts points in the wrong bin.
    """
    image_mask = np.array(
        [
            [True, True, False, False],
            [False, True, True, True],
            [True, False, True, False],
        ]
    )
    px, py = 1.0, 2.0
    layout = ImageMaskLayout()
    layout.build(image_mask=image_mask, pixel_size=(px, py))

    points = np.random.default_rng(0).uniform((0, 0), (4 * px, 3 * py), (400, 2))
    rows = np.floor(points[:, 1] / py).astype(int)
    cols = np.floor(points[:, 0] / px).astype(int)
    active = image_mask[rows, cols]
    assert active.any() and not active.all()

    result = layout.binning().locator.locate(points)
    np.testing.assert_array_equal(result[~active], -1)
    np.testing.assert_array_equal(
        layout.bin_centers[result[active]],
        np.column_stack(((cols + 0.5) * px, (rows + 0.5) * py))[active],
    )


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

    result = layout.binning().locator.locate(np.array([[13.5, 0.5]]))
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


def test_image_mask_rejects_removed_bin_size_keyword():
    layout = ImageMaskLayout()
    with pytest.raises(TypeError, match="bin_size"):
        layout.build(
            image_mask=np.ones((3, 4), dtype=bool), pixel_size=1.0, bin_size=2.0
        )


def test_image_mask_requires_pixel_size():
    """A pixel has no physical size without calibration, so none is assumed."""
    layout = ImageMaskLayout()
    with pytest.raises(TypeError, match="pixel_size"):
        layout.build(image_mask=np.ones((3, 4), dtype=bool))


def test_image_mask_records_the_pixel_size_used():
    layout = ImageMaskLayout()
    layout.build(image_mask=np.ones((3, 4), dtype=bool), pixel_size=0.5)
    assert layout._build_params_used["pixel_size"] == 0.5


def test_image_mask_pixel_centers_in_bin_order():
    """Pixel ``[r, c]`` becomes the bin centered at ``((c + .5) px, (r + .5) py)``.

    Bins are numbered x-major, like every grid engine: by column, then by
    row. The mask is not symmetric under a transpose or a flip, so the
    unsorted rows pin both the coordinates and the order.
    """
    image_mask = np.array([[True, True, False], [False, True, True]])
    layout = ImageMaskLayout()
    layout.build(image_mask=image_mask, pixel_size=(1.0, 2.0))

    # Active (row, col), column by column: (0,0), (0,1), (1,1), (1,2).
    expected = np.array([[0.5, 1.0], [1.5, 1.0], [1.5, 3.0], [2.5, 3.0]])
    np.testing.assert_array_equal(layout.bin_centers, expected)
    # With diagonal neighbors: three side-sharing pairs and two corner pairs.
    assert sorted(layout.connectivity.edges()) == [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
