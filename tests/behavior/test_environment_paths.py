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

    def test_path_around_a_wall_is_the_shortest_through_bin_centers(
        self, divider_arena
    ):
        # Waypoints are bin centers: no center left of the wall sees one
        # right of it below y = 70, so the path climbs to the centers just
        # above the wall end, (47.5, 72.5) and (52.5, 72.5), and back down.
        start, goal = np.array([42.5, 12.5]), np.array([57.5, 12.5])
        length = shortest_path_length(divider_arena, start, goal, metric="geodesic")
        assert length == pytest.approx(2 * np.hypot(5.0, 60.0) + 5.0, abs=1e-9)

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


class TestOpenEnvironments:
    def test_any_two_positions_in_a_convex_arena_are_a_straight_line_apart(self):
        edges = np.arange(0.0, 101.0, 2.0)
        env = Environment.from_grid_mask(
            np.ones((50, 50), dtype=bool), grid_edges=(edges, edges)
        )
        rng = np.random.default_rng(3)
        starts = rng.uniform(0.5, 99.5, (200, 2))
        ends = rng.uniform(0.5, 99.5, (200, 2))
        traveled = traveled_path_length(
            np.arange(2) / 10,
            np.vstack([starts[0], ends[0]]),
            metric="geodesic",
            env=env,
        )
        assert traveled == pytest.approx(np.linalg.norm(ends[0] - starts[0]))
        for start, end in zip(starts[:20], ends[:20], strict=True):
            assert shortest_path_length(env, start, end) == pytest.approx(
                np.linalg.norm(end - start), rel=1e-12
            )


@pytest.mark.wallclock
@pytest.mark.parametrize(
    ("maze", "budget"),
    [("make_hampton_court_maze", 0.5), ("make_rat_hexmaze", 1.0)],
)
def test_geodesic_approach_rate_of_a_ten_minute_session_is_fast(maze, budget):
    """One goal, 10 min at 30 Hz: one field per goal, then lookups."""
    import time

    from neurospatial.behavior import approach_rate
    from neurospatial.simulation import mazes

    env = getattr(mazes, maze)().env_2d
    rng = np.random.default_rng(0)
    positions = env.bin_centers[rng.integers(0, env.n_bins, 18_000)]
    times = np.arange(18_000) / 30
    goal = env.bin_centers[np.argmax(env.bin_centers[:, 0])]
    start = time.perf_counter()
    approach_rate(times, positions, goal, metric="geodesic", env=env, max_gap=None)
    assert time.perf_counter() - start <= budget
