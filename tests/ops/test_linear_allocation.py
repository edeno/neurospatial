"""Straight-path time allocation across rectangular cells."""

import numpy as np
import pytest

from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.ops._linear_allocation import linear_grid_occupancy, ray_grid_pieces


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
    seconds = linear_grid_occupancy(
        2, grid, positions, np.array([4.0]), np.array([True]), bins, return_seconds=True
    )
    counts = linear_grid_occupancy(
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
    occupancy = linear_grid_occupancy(
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
