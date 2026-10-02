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


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0, 1.0, 10.0]),  # nonuniform
        np.array([0.0, 1.0, 2.001]),  # nonuniform by 0.1%
        np.array([2.0, 1.0, 0.0]),  # decreasing
        np.array([0.0, 0.0, 1.0]),  # duplicate edge
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, 1.0, np.inf]),
        np.array([[0.0, 1.0, 2.0]]),  # 2-D
        np.array([0.0]),  # a single edge defines no bin
    ],
    ids=[
        "nonuniform",
        "slightly-nonuniform",
        "decreasing",
        "duplicate",
        "nan",
        "inf",
        "2d",
        "one-edge",
    ],
)
def test_rejects_invalid_edges(edges):
    """Edges must be finite, 1-D, strictly increasing and uniformly spaced."""
    layout = MaskedGridLayout()
    with pytest.raises(ValueError, match=r"grid_edges\[0\]"):
        layout.build(active_mask=np.ones(2, dtype=bool), grid_edges=(edges,))


def test_rejects_unrepresentable_edges():
    """Edges at 1e15 cannot resolve a unit bin width; the error says to offset."""
    # float64 spacing at 1e15 is 0.125, so 1e15 + [0, 1, 3] has widths [1, 2].
    edges = np.array([1e15, 1e15 + 1, 1e15 + 3])
    layout = MaskedGridLayout()
    with pytest.raises(ValueError, match=r"grid_edges\[0\]") as excinfo:
        layout.build(active_mask=np.ones(2, dtype=bool), grid_edges=(edges,))
    fix_line = next(
        line for line in str(excinfo.value).splitlines() if line.startswith("Fix:")
    )
    assert "origin offset" in fix_line


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([0.0, 0.01, 0.0201]),  # widths [0.01, 0.0101]
        np.array([0.0, 0.01, 0.02 + 1e-8]),  # one width off by 1e-8
    ],
    ids=["one-percent", "1e-8"],
)
def test_rejects_nonuniform_at_large_offset(offsets):
    """At a 1e7 offset the uniformity tolerance is ~8.45e-9, not looser."""
    edges = 1e7 + offsets
    layout = MaskedGridLayout()
    with pytest.raises(ValueError, match="uniformly spaced"):
        layout.build(active_mask=np.ones(2, dtype=bool), grid_edges=(edges,))


@pytest.mark.parametrize(("offset", "bin_size"), [(1e7, 0.01), (1e9, 1.0)])
def test_accepts_fine_bins_at_large_offsets(offset, bin_size):
    """Rounded but uniform grids far from the origin still build and subset."""
    from neurospatial import Environment

    rng = np.random.default_rng(0)
    positions = offset + rng.uniform(0, 45 * bin_size, (20_000, 2))
    env = Environment.from_samples(positions, bin_size=bin_size)
    keep = np.zeros(env.n_bins, dtype=bool)
    keep[::2] = True

    sub = env.subset(bins=keep)

    assert isinstance(sub.layout, MaskedGridLayout)
    assert sub.n_bins == keep.sum()


def test_accepts_uniform_anisotropic_edges():
    """Each axis needs one width, but the axes may differ."""
    from neurospatial import Environment

    edges = (np.linspace(0.0, 1.0, 11), np.linspace(0.0, 30.0, 11))
    env = Environment.from_grid_mask(np.ones((10, 10), dtype=bool), grid_edges=edges)
    np.testing.assert_allclose(env.bin_sizes, 0.3, rtol=1e-12)
