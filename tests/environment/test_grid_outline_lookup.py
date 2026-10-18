"""A point on the outline of a grid's active cells belongs to the cell it bounds.

Cells are half-open inside the arena, so a face shared by two active cells has
one owner. A point on a face with no active cell beyond it -- the arena's edge,
or the tip of an arm -- belongs to the active cell it bounds, as in
``np.histogram``, whose last bin includes its right edge.
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment


@pytest.fixture(scope="module")
def square():
    """A 10 x 10 cm arena with 2 cm bins (centers 1, 3, ..., 9)."""
    return Environment.from_polygon(box(0, 0, 10, 10), bin_size=2.0, units="cm")


@pytest.mark.parametrize(
    ("point", "center"),
    [
        ((10.0, 5.0), (9.0, 5.0)),  # right edge
        ((5.0, 10.0), (5.0, 9.0)),  # top edge
        ((10.0, 10.0), (9.0, 9.0)),  # top-right corner
        ((0.0, 10.0), (1.0, 9.0)),  # top-left corner
        ((0.0, 0.0), (1.0, 1.0)),  # lower edges are already owned
    ],
)
def test_point_on_arena_edge_maps_to_the_edge_bin(square, point, center):
    query = np.array([point])
    (index,) = square.bin_at(query)
    assert index >= 0
    np.testing.assert_allclose(square.bin_centers[index], center)
    assert square.contains(query)[0]


def test_shared_face_inside_the_arena_keeps_one_owner(square):
    (index,) = square.bin_at(np.array([[4.0, 5.0]]))
    assert index >= 0
    np.testing.assert_allclose(square.bin_centers[index], [5.0, 5.0])


def test_point_beyond_the_edge_stays_outside(square):
    assert square.bin_at(np.array([[10.0 + 1e-9, 5.0]]))[0] == -1


def test_arm_tip_inside_the_bounding_box_closes_too():
    """An L-shaped mask: the arm tip at x = 1 borders an inactive cell."""
    mask = np.array([[True, True], [True, False]])  # cell (1, 1) inactive
    edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    env = Environment.from_grid_mask(mask, grid_edges=edges)
    (index,) = env.bin_at(np.array([[1.5, 1.0]]))  # top face of cell (1, 0)
    assert index >= 0
    np.testing.assert_allclose(env.bin_centers[index], [1.5, 0.5])


def test_animal_waiting_on_the_outline_is_counted_in_occupancy(square):
    times = np.arange(31) / 30.0
    positions = np.tile([10.0, 5.0], (31, 1))  # sitting against the right wall
    assert square.occupancy(times, positions).sum() == pytest.approx(1.0)


def test_maze_wells_on_arm_tips_cover_a_bin():
    from neurospatial.simulation.mazes import make_w_maze

    env = make_w_maze().env_2d
    for name in env.regions:
        assert len(env.bins_in_region(name)) == 1, name
