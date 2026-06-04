"""Behavioral tests for MaskedGridLayout driven directly through the engine."""

import numpy as np
import pytest

from neurospatial.layout.engines.masked_grid import MaskedGridLayout


def _build_masked_layout():
    """Build a 4x4 unit grid (edges 0..4) with the central 2x2 block masked out.

    Active bins are the 12 border cells; the central cells (rows 1-2, cols 1-2)
    are inactive.
    """
    grid_edges = (np.arange(5.0), np.arange(5.0))
    active_mask = np.ones((4, 4), dtype=bool)
    active_mask[1:3, 1:3] = False  # mask out the central 2x2 block

    layout = MaskedGridLayout()
    layout.build(active_mask=active_mask, grid_edges=grid_edges)
    return layout, active_mask


def test_point_inside_bbox_but_masked_returns_negative_one():
    """A point inside the bounding box but in a masked cell returns -1."""
    layout, _ = _build_masked_layout()

    # Cell center (1.5, 1.5) lies inside the bounding box but is masked out.
    result = layout.point_to_bin_index(np.array([[1.5, 1.5]]))
    assert result[0] == -1


def test_point_in_active_region():
    """A point in an active cell returns the corresponding active bin index."""
    layout, _ = _build_masked_layout()

    # Cell center (0.5, 0.5) is the first active bin.
    result = layout.point_to_bin_index(np.array([[0.5, 0.5]]))
    assert result[0] >= 0
    np.testing.assert_allclose(layout.bin_centers[result[0]], [0.5, 0.5])


def test_point_on_mask_boundary():
    """A point on the grid edge between an active and masked cell.

    Pinned to current behavior: grid-index lookup places x=1.0 into the cell
    spanning [1, 2), so the point maps to the active border bin at that index
    rather than to the masked interior cell.
    """
    layout, _ = _build_masked_layout()

    result = layout.point_to_bin_index(np.array([[1.0, 0.5]]))
    assert result[0] >= 0
    # The assigned cell is in the active bottom row (y in [0, 1)).
    assert layout.bin_centers[result[0]][1] == 0.5


def test_n_bins_excludes_masked():
    """Active bin count equals the number of True cells, not the full grid size.

    The engine exposes no ``n_bins`` attribute, so the active count is taken
    from ``bin_centers.shape[0]`` per the LayoutEngine protocol.
    """
    layout, active_mask = _build_masked_layout()

    assert layout.bin_centers.shape[0] == int(active_mask.sum())
    # 16-cell grid with a 4-cell hole leaves 12 active bins.
    assert layout.bin_centers.shape[0] == 12


def test_masked_grid_rejects_float_mask():
    """A float or int mask raises ValueError naming the dtype; bool still builds."""
    grid_edges = (np.arange(4.0), np.arange(3.0))  # 3x2 bins
    bool_mask = np.ones((3, 2), dtype=bool)

    layout_float = MaskedGridLayout()
    with pytest.raises(ValueError, match="dtype"):
        layout_float.build(active_mask=bool_mask.astype(float), grid_edges=grid_edges)

    layout_int = MaskedGridLayout()
    with pytest.raises(ValueError, match="dtype"):
        layout_int.build(active_mask=bool_mask.astype(int), grid_edges=grid_edges)

    # A genuine boolean mask still builds.
    layout_ok = MaskedGridLayout()
    layout_ok.build(active_mask=bool_mask, grid_edges=grid_edges)
    assert layout_ok.bin_centers.shape[0] == int(bool_mask.sum())


def test_masked_grid_rejects_non_array_mask():
    """A Python list mask raises TypeError."""
    grid_edges = (np.arange(4.0), np.arange(3.0))
    list_mask = [[True, True], [True, False], [False, True]]

    layout = MaskedGridLayout()
    with pytest.raises(TypeError):
        layout.build(active_mask=list_mask, grid_edges=grid_edges)
