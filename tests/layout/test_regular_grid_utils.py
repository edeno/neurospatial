import numpy as np
import pytest

from neurospatial.geometry import points_to_regular_grid_bin_ind
from neurospatial.layout.helpers.regular_grid import (
    _create_regular_grid,
    _infer_active_bins_from_regular_grid,
    regular_grid_pairs,
)
from neurospatial.layout.helpers.utils import get_n_bins


def test_create_regular_grid_2d_basic():
    # Simple 2D data, no boundary bins
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    bin_size = 1.0
    edges, centers, shape = _create_regular_grid(data, bin_size)
    assert len(edges) == 2
    assert all(isinstance(e, np.ndarray) for e in edges)
    assert centers.shape[1] == 2
    assert np.prod(shape) == centers.shape[0]
    # Check that the grid covers the data
    assert np.all(data.min(axis=0) >= [e[0] for e in edges])
    assert np.all(data.max(axis=0) <= [e[-1] for e in edges])


def test_create_regular_grid_with_dimension_range_and_boundary():
    # 1D grid, explicit range, with boundary bins
    edges, centers, shape = _create_regular_grid(
        positions=None,
        bin_size=1.0,
        dimension_range=[(0, 2)],
        add_boundary_bins=True,
    )
    # Should have 4 bins (2 core + 2 boundary)
    assert shape == (4,)
    assert edges[0].shape == (5,)
    # Centers should be within the extended range
    assert np.all(centers[:, 0] >= edges[0][0])
    assert np.all(centers[:, 0] <= edges[0][-1])


def test_infer_active_bins_from_regular_grid_simple():
    # 2D: points in a 3x3 grid, threshold=0
    data = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    edges = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    mask = _infer_active_bins_from_regular_grid(data, edges)
    assert mask.shape == (3, 3)
    # Only diagonal bins should be active
    assert np.all(mask.diagonal())
    assert np.sum(mask) == 3


def test_infer_active_bins_with_threshold_and_morphology():
    # 2D: points in a cross, threshold=1, dilate
    data = np.array(
        [
            [1.5, 0.5],
            [1.5, 1.5],
            [1.5, 2.5],  # vertical
            [0.5, 1.5],
            [2.5, 1.5],  # horizontal
        ]
    )
    edges = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    mask = _infer_active_bins_from_regular_grid(
        data, edges, bin_count_threshold=0, dilate=True
    )
    # Dilation should expand the cross
    assert mask.sum() > 5
    # All original cross points should be active
    for pt in data:
        i, j = int(pt[0]), int(pt[1])
        assert mask[i, j]


def test_infer_active_bins_boundary_exists():
    # 1D: boundary bins should be set to False
    data = np.array([[0.5], [1.5], [2.5]])
    edges = (np.array([0, 1, 2, 3]),)
    mask = _infer_active_bins_from_regular_grid(data, edges, boundary_exists=True)
    assert mask.shape == (3,)
    assert not mask[0]
    assert not mask[-1]
    assert mask[1]


def test_infer_active_bins_boundary_exists_2d():
    """boundary_exists trims the outer shell of a 2D grid on every axis."""
    # Fully populate a 4x4 grid so all interior bins would otherwise be active.
    edges = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
    centers = np.array([[i + 0.5, j + 0.5] for i in range(4) for j in range(4)])
    mask = _infer_active_bins_from_regular_grid(
        centers, edges, bin_count_threshold=0, boundary_exists=True
    )
    assert mask.shape == (4, 4)
    # Outer rows/cols inactive on both axes.
    assert not mask[0, :].any()
    assert not mask[-1, :].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()
    # Interior preserved.
    assert mask[1:-1, 1:-1].all()


def test_infer_active_bins_boundary_exists_3d():
    """boundary_exists trims the outer shell of a 3D grid on every axis."""
    e = np.array([0, 1, 2, 3])
    edges = (e, e, e)  # 3x3x3 grid
    centers = np.array(
        [
            [i + 0.5, j + 0.5, k + 0.5]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ]
    )
    mask = _infer_active_bins_from_regular_grid(
        centers, edges, bin_count_threshold=0, boundary_exists=True
    )
    assert mask.shape == (3, 3, 3)
    # Every outer slab on all three axes is inactive.
    assert not mask[0, :, :].any() and not mask[-1, :, :].any()
    assert not mask[:, 0, :].any() and not mask[:, -1, :].any()
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()
    # The single interior bin (center) is the only one that can survive.
    assert mask[1, 1, 1]


def test_regular_grid_pairs_2d_orthogonal():
    # 2x2 grid, all bins active: every bin has its two face neighbors.
    pairs = regular_grid_pairs(np.ones((2, 2), dtype=bool))
    assert pairs.dtype == np.intp
    assert pairs.tolist() == [[0, 2], [0, 1], [1, 3], [2, 3]]


def test_regular_grid_pairs_2d_diagonal():
    # Diagonals add the two corner-sharing pairs of a 2x2 grid.
    pairs = regular_grid_pairs(np.ones((2, 2), dtype=bool), connect_diagonal=True)
    assert pairs.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


@pytest.mark.parametrize("connect_diagonal", [False, True])
@pytest.mark.parametrize("shape", [(9,), (5, 7), (3, 4, 5)])
def test_regular_grid_pairs_match_adjacent_active_cells(shape, connect_diagonal):
    """Pairs are exactly the adjacent active cells, each once, low to high."""
    mask = np.random.default_rng(3).random(shape) < 0.7
    cells = np.argwhere(mask)
    steps = np.abs(cells[:, None, :] - cells[None, :, :])
    adjacent = steps.max(axis=-1) == 1 if connect_diagonal else steps.sum(axis=-1) == 1
    expected = {tuple(pair) for pair in np.argwhere(np.triu(adjacent))}

    pairs = regular_grid_pairs(mask, connect_diagonal=connect_diagonal)

    assert {tuple(pair) for pair in pairs.tolist()} == expected
    assert len(pairs) == len(expected)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert np.all(np.diff(pairs[:, 0]) >= 0)


def test_points_to_regular_grid_bin_ind_basic():
    # 2D grid, no active mask
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    shape = (2, 2)
    points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [np.nan, 1]])
    inds = points_to_regular_grid_bin_ind(points, edges, shape)
    # First two points are in-bounds, third is out, fourth is nan
    assert inds[0] == 0
    assert inds[1] == 3
    assert inds[2] == -1
    assert inds[3] == -1


def test_points_to_regular_grid_bin_ind_with_active_mask():
    # 2D grid, only one bin active
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    shape = (2, 2)
    points = np.array([[0.5, 0.5], [1.5, 1.5]])
    active_mask = np.zeros((2, 2), dtype=bool)
    active_mask[1, 1] = True
    inds = points_to_regular_grid_bin_ind(points, edges, shape, active_mask)
    # Only the second point is in the active bin
    assert inds[0] == -1
    assert inds[1] == 0


def test_points_to_bin_dimensionality_mismatch_raises():
    # 3-D points against a 2-D grid must raise rather than silently
    # returning all -1 (which would look like "no bins found").
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    shape = (2, 2)
    points = np.array([[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        points_to_regular_grid_bin_ind(points, edges, shape)


def test_create_and_connectivity_for_3d_grid():
    """
    Build a small 3D dataset (points forming the 8 corners of a unit cube)
    and verify:

      1) _create_regular_grid produces exactly one bin along each dimension
      2) bin_centers is a single point at (0.5, 0.5, 0.5)
      3) The single active bin has no neighbor pairs
      4) Per-dimension bin sizes are [1.0, 1.0, 1.0]
    """
    # 1) Eight corner points of [0,1]^3
    cube_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    # 2) Create the grid with bin_size = 1.0 (no boundary bins)
    edges_tuple, bin_centers, centers_shape = _create_regular_grid(
        positions=cube_pts,
        bin_size=1.0,
        dimension_range=None,
        add_boundary_bins=False,
    )

    # We expect 3 dimensions, each edge array = [0.0, 1.0]
    assert len(edges_tuple) == 3
    for dim_edges in edges_tuple:
        assert isinstance(dim_edges, np.ndarray)
        assert dim_edges.shape == (2,)
        assert np.allclose(dim_edges, np.array([0.0, 1.0], dtype=float))

    # Confirm get_n_bins returns [1,1,1] given points at the exact endpoints
    n_bins = get_n_bins(
        cube_pts,
        np.array([1.0, 1.0, 1.0]),
        dimension_range=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )
    assert np.array_equal(n_bins, np.array([1, 1, 1], dtype=int))

    # bin_centers: exactly one point at (0.5, 0.5, 0.5)
    assert bin_centers.shape == (1, 3)
    assert np.allclose(bin_centers, np.array([[0.5, 0.5, 0.5]]))

    # centers_shape should be (1, 1, 1)
    assert centers_shape == (1, 1, 1)

    # 3) Infer the active_mask without any morphological ops.
    #    Signature: _infer_active_bins_from_regular_grid(data_samples, edges, ...)
    active_mask = _infer_active_bins_from_regular_grid(
        positions=cube_pts,
        edges=edges_tuple,
        close_gaps=False,
        fill_holes=False,
        dilate=False,
        bin_count_threshold=0,
        boundary_exists=False,
    )
    # Should be shape (1, 1, 1), with exactly one True
    assert active_mask.ndim == 3
    assert active_mask.shape == (1, 1, 1)
    assert active_mask[0, 0, 0]

    # There’s exactly 1 active bin, so it has no neighbor pairs.
    assert regular_grid_pairs(active_mask).shape == (0, 2)

    # 4) Compute per-dimension bin sizes by differencing each edge array:
    diffs = [float(np.diff(e)[0]) for e in edges_tuple]
    assert diffs == [1.0, 1.0, 1.0]


def test_points_to_regular_grid_bin_ind_3d_and_oob():
    """
    Test that `points_to_regular_grid_bin_ind` correctly handles 3D points:

      - (0.5,0.5,0.5) should map to some valid bin index ≥ 0.
      - (2.0,2.0,2.0) should map to -1 (out-of-bounds).
      - (NaN, 0.5, 0.5) should map to -1.
    """
    # Edges for a single bin in each of 3 dims:
    edges_tuple = (
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    # Only one bin total (shape = (1,1,1)), flat index = 0
    grid_shape = (1, 1, 1)
    active_mask = np.ones(grid_shape, dtype=bool)

    pts = np.array(
        [
            [0.5, 0.5, 0.5],  # inside
            [2.0, 2.0, 2.0],  # outside
            [np.nan, 0.5, 0.5],  # NaN → out-of-bounds
        ],
        dtype=float,
    )

    # Now call with the correct signature: (points, grid_edges, grid_shape, active_mask)
    idxs = points_to_regular_grid_bin_ind(pts, edges_tuple, grid_shape, active_mask)

    assert isinstance(idxs, np.ndarray)
    assert idxs.shape == (3,)
    # The first point should map to a valid bin index (≥ 0). The others → -1.
    assert idxs[0] >= 0
    assert idxs[1] == -1
    assert idxs[2] == -1
