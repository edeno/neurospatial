"""Known answers for the geodesic metric of neurospatial.behavior.

``metric="geodesic"`` is the length of the shortest path between two
positions that stays inside the environment. Expected values come from the
geometry of each environment, not from the code under test.
"""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.behavior import shortest_path_length, traveled_path_length

# A 100 x 100 cm arena of 5 cm bins with a wall along x = 50 from y = 0 to
# y = 70: the two sides meet only above the wall's end at (50, 70).
BIN = 5.0
WALL_END = np.array([50.0, 70.0])


@pytest.fixture(scope="module")
def divider_arena():
    edges = np.arange(0.0, 101.0, BIN)
    return Environment.from_grid_mask(
        np.ones((20, 20), dtype=bool),
        grid_edges=(edges, edges),
        units="cm",
        arena=OpenArena(
            box(0, 0, 100, 100), walls={"divider": LineString([(50, 0), (50, 70)])}
        ),
    )


@pytest.fixture(scope="module")
def slit_arena():
    """A 100 x 50 cm room with a 0.2 cm slit up from the floor to y = 40."""
    outline = Polygon(
        [
            (0, 0),
            (49.9, 0),
            (49.9, 40),
            (50.1, 40),
            (50.1, 0),
            (100, 0),
            (100, 50),
            (0, 50),
        ]
    )
    return Environment.from_polygon(outline, bin_size=2.0)


def _around_wall_end(start, goal, wall_end):
    """Shortest length from start to goal on opposite sides of a thin wall."""
    return float(np.linalg.norm(wall_end - start) + np.linalg.norm(goal - wall_end))


class TestWallsBlockGeodesicPaths:
    def test_path_across_a_declared_wall_goes_around_its_end(self, divider_arena):
        start, goal = np.array([42.5, 12.5]), np.array([57.5, 12.5])
        length = shortest_path_length(divider_arena, start, goal, metric="geodesic")
        # The straight line (15 cm) crosses the wall. No path is shorter than
        # the taut string around the wall end; bin-center waypoints round
        # that end about half a bin out on each side, so the path through
        # (47.5, 72.5) and (52.5, 72.5) bounds it from above.
        taut = _around_wall_end(start, goal, WALL_END)
        via_centers = 2 * np.hypot(5.0, 60.0) + 5.0
        assert taut <= length <= via_centers + 1e-9

    def test_walking_through_a_wall_is_not_counted_as_a_short_step(self, divider_arena):
        # 0.15 cm steps along y = 12.5 across x = 50. The step that crosses
        # the wall must go around its end, 57.5 cm up and back down, so the
        # total is at least 2 * 57.5 cm (the straight walk is 15 cm).
        x = np.linspace(42.5, 57.5, 101)
        positions = np.c_[x, np.full(x.size, 12.5)]
        length = traveled_path_length(
            np.arange(x.size) / 10, positions, metric="geodesic", env=divider_arena
        )
        assert length >= 2 * (WALL_END[1] - 12.5)

    def test_path_across_an_outline_slit_goes_around_its_tip(self, slit_arena):
        start, goal = np.array([45.0, 5.0]), np.array([55.0, 5.0])
        length = shortest_path_length(slit_arena, start, goal, metric="geodesic")
        taut = 2 * np.hypot(4.9, 35.0) + 0.2  # around the slit's two tip corners
        assert taut <= length <= taut + 2 * 2.0  # within a bin per tip corner
