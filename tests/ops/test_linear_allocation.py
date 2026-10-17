"""Straight-path time allocation across rectangular cells."""

import functools
import itertools

import numpy as np
import pytest

from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.ops import _linear_allocation as _allocation
from neurospatial.ops._linear_allocation import _linear_grid_occupancy, ray_grid_pieces


@pytest.fixture
def unit_grid() -> RectilinearGrid:
    """Four by three unit cells covering [0, 4) x [0, 3)."""
    edges = (np.arange(5.0), np.arange(4.0))
    return RectilinearGrid(edges, (4, 3), np.ones((4, 3), dtype=bool))


def test_pieces_follow_the_path_in_proportion_to_length(unit_grid):
    # From x=0.5 to x=3.0 at y=1.5: half a cell, then two whole cells.
    interval, cell, time = ray_grid_pieces(
        np.array([[0.5, 1.5]]),
        np.array([[3.0, 1.5]]),
        unit_grid.edges,
        unit_grid.shape,
        np.array([5.0]),
    )
    np.testing.assert_array_equal(interval, [0, 0, 0])
    np.testing.assert_array_equal(cell, [1, 4, 7])  # cells (0,1), (1,1), (2,1)
    np.testing.assert_allclose(time, [1.0, 2.0, 2.0])


def test_diagonal_path_conserves_duration(unit_grid):
    interval, cell, time = ray_grid_pieces(
        np.array([[0.2, 0.1], [3.9, 2.9]]),
        np.array([[3.7, 2.8], [0.1, 0.3]]),
        unit_grid.edges,
        unit_grid.shape,
        np.array([2.0, 3.0]),
    )
    np.testing.assert_allclose(np.bincount(interval, weights=time), [2.0, 3.0])
    # Each interval's pieces are listed in order along its path.
    assert np.all(np.diff(interval) >= 0)
    assert cell[interval == 0][0] == 0
    assert cell[interval == 1][0] == 11


def test_stationary_and_unusable_intervals(unit_grid):
    starts = np.array([[1.5, 1.5], [np.nan, 1.0], [-5.0, -5.0]])
    stops = np.array([[1.5, 1.5], [2.0, 1.0], [-4.0, -5.0]])
    interval, cell, time = ray_grid_pieces(
        starts, stops, unit_grid.edges, unit_grid.shape, np.array([1.0, 1.0, 1.0])
    )
    # A stationary sample keeps the whole interval in its cell; a non-finite
    # endpoint or a path outside the grid gives nothing.
    np.testing.assert_array_equal(interval, [0])
    np.testing.assert_array_equal(cell, [4])
    np.testing.assert_array_equal(time, [1.0])


def test_inactive_cells_drop_their_share():
    edges = (np.arange(4.0), np.array([0.0, 1.0]))
    active = np.array([[True], [False], [True]])
    grid = RectilinearGrid(edges, (3, 1), active)
    positions = np.array([[0.5, 0.5], [2.5, 0.5]])
    bins = np.array([0, 1])  # active-bin ids of cells 0 and 2
    seconds = _linear_grid_occupancy(
        2, grid, positions, np.array([4.0]), np.array([True]), bins, return_seconds=True
    )
    counts = _linear_grid_occupancy(
        2,
        grid,
        positions,
        np.array([4.0]),
        np.array([True]),
        bins,
        return_seconds=False,
    )
    # The middle (inactive) cell holds half the path; its time is not moved.
    np.testing.assert_allclose(seconds, [1.0, 1.0])
    np.testing.assert_allclose(counts, [0.25, 0.25])


def test_interval_inside_one_bin_and_invalid_intervals(unit_grid):
    positions = np.array([[0.2, 0.2], [0.8, 0.7], [3.5, 2.5]])
    bins = np.array([0, 0, 11])
    occupancy = _linear_grid_occupancy(
        12,
        unit_grid,
        positions,
        np.array([2.0, 3.0]),
        np.array([True, False]),
        bins,
        return_seconds=True,
    )
    expected = np.zeros(12)
    expected[0] = 2.0
    np.testing.assert_array_equal(occupancy, expected)


def test_crossing_blocks_cover_intervals_in_order_within_budget():
    rng = np.random.default_rng(1)
    edges = (np.arange(101.0), np.arange(101.0))
    starts = rng.uniform(0, 100, (500, 2))
    stops = rng.uniform(0, 100, (500, 2))
    blocks = _allocation.crossing_blocks(starts, stops, edges, max_crossings=300)
    assert blocks[0].start == 0
    assert blocks[-1].stop == 500
    assert all(a.stop == b.start for a, b in itertools.pairwise(blocks))
    assert len(blocks) > 1
    # One interval with more candidates than the budget is a block on its own.
    single = _allocation.crossing_blocks(starts[:3], stops[:3], edges, max_crossings=1)
    assert single == [slice(0, 1), slice(1, 2), slice(2, 3)]


@pytest.mark.parametrize("return_seconds", [True, False])
def test_block_size_does_not_change_occupancy(monkeypatch, return_seconds):
    rng = np.random.default_rng(2)
    edges = (np.arange(31.0), np.arange(21.0), np.arange(11.0))
    active = rng.random((30, 20, 10)) < 0.8
    grid = RectilinearGrid(edges, (30, 20, 10), active)
    positions = rng.uniform(-1, 31, (400, 3)) * np.array([1.0, 0.7, 0.35])
    dt = rng.exponential(0.1, 399)
    valid = rng.random(399) < 0.9
    bins = np.full(400, -1, dtype=np.intp)  # every interval takes the traversal
    whole = _linear_grid_occupancy(
        int(active.sum()),
        grid,
        positions,
        dt,
        valid,
        bins,
        return_seconds=return_seconds,
    )
    small = functools.partial(_allocation.crossing_blocks, max_crossings=5)
    monkeypatch.setattr(_allocation, "crossing_blocks", small)
    blocked = _linear_grid_occupancy(
        int(active.sum()),
        grid,
        positions,
        dt,
        valid,
        bins,
        return_seconds=return_seconds,
    )
    np.testing.assert_array_equal(blocked, whole)


def test_path_through_cell_corners_keeps_zero_length_pieces(unit_grid):
    # The diagonal (0, 0) -> (3, 3) meets the x and y edges at the same lattice
    # points, (1, 1) and (2, 2): each gives a zero-length piece owned by the
    # cell above-right of the corner (half-open cells).
    interval, cell, time = ray_grid_pieces(
        np.array([[0.0, 0.0]]),
        np.array([[3.0, 3.0]]),
        unit_grid.edges,
        unit_grid.shape,
        np.array([3.0]),
    )
    np.testing.assert_array_equal(interval, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(cell, [0, 4, 4, 8, 8])  # (0,0), (1,1), (2,2)
    np.testing.assert_array_equal(time[[1, 3]], [0.0, 0.0])
    np.testing.assert_allclose(time[[0, 2, 4]], [1.0, 1.0, 1.0], rtol=1e-15)


def test_path_along_a_cell_edge_belongs_to_the_cells_above(unit_grid):
    # Along y = 1 the path lies on the edge between rows 0 and 1; half-open
    # cells [y_i, y_{i+1}) give it to row 1. Along the grid's top edge, y = 3,
    # it lies outside every cell.
    interval, cell, time = ray_grid_pieces(
        np.array([[0.5, 1.0], [0.5, 3.0]]),
        np.array([[3.5, 1.0], [3.5, 3.0]]),
        unit_grid.edges,
        unit_grid.shape,
        np.array([6.0, 6.0]),
    )
    np.testing.assert_array_equal(interval, [0, 0, 0, 0])
    np.testing.assert_array_equal(cell, [1, 4, 7, 10])  # (0,1) ... (3,1)
    np.testing.assert_allclose(time, [1.0, 2.0, 2.0, 1.0], rtol=1e-15)


def test_one_step_across_many_cells():
    edges = (np.arange(1001.0), np.array([0.0, 1.0]))
    interval, cell, time = ray_grid_pieces(
        np.array([[0.5, 0.5]]),
        np.array([[999.5, 0.5]]),
        edges,
        (1000, 1),
        np.array([999.0]),
    )
    np.testing.assert_array_equal(cell, np.arange(1000))
    expected = np.ones(1000)
    expected[[0, -1]] = 0.5
    np.testing.assert_allclose(time, expected, rtol=1e-12)
    assert np.all(interval == 0)


def test_pieces_of_paths_inside_the_grid_add_up_to_each_duration(unit_grid):
    # Inside the grid every piece is kept, so an interval's pieces sum to its
    # duration up to floating-point rounding of the piece fractions.
    rng = np.random.default_rng(4)
    upper = np.array([4.0, 3.0]) - 1e-9
    starts = rng.uniform(0, upper, (2000, 2))
    stops = rng.uniform(0, upper, (2000, 2))
    durations = rng.exponential(1.0, 2000)
    interval, _, time = ray_grid_pieces(
        starts, stops, unit_grid.edges, unit_grid.shape, durations
    )
    totals = np.bincount(interval, weights=time, minlength=2000)
    np.testing.assert_allclose(totals, durations, rtol=1e-13, atol=0)
