"""Known answers for the geodesic metric of neurospatial.behavior.

``metric="geodesic"`` is the length of the shortest path between two
positions that stays inside the environment. Expected values come from the
geometry of each environment, not from the code under test.
"""

import networkx as nx
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


def _w_track(bin_size, track_width):
    """A W-track: a 40 cm center stem and two 40 cm outer arms joined by
    30 cm crossbars at y = 40."""
    graph = nx.Graph()
    nodes = [
        (0.0, 0.0),
        (0.0, 40.0),
        (-30.0, 40.0),
        (30.0, 40.0),
        (-30.0, 0.0),
        (30.0, 0.0),
    ]
    for node, pos in enumerate(nodes):
        graph.add_node(node, pos=pos)
    for u, v in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]:
        graph.add_edge(
            u, v, distance=float(np.linalg.norm(np.subtract(nodes[u], nodes[v])))
        )
    return Environment.from_graph(
        graph,
        [(0, 1), (1, 2), (2, 4), (1, 3), (3, 5)],
        bin_size=bin_size,
        track_width=track_width,
    )


class TestTrackEnvironments:
    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    @pytest.mark.parametrize("track_width", [4.0, 7.0, 10.0])
    def test_arm_end_to_arm_end_is_the_centerline_length(self, bin_size, track_width):
        # (-30, 1) is 39 cm down the left arm from its top; along the
        # centerline: 39 up + 30 + 30 across + 39 down the right arm.
        env = _w_track(bin_size, track_width)
        start, goal = np.array([-30.0, 1.0]), np.array([30.0, 1.0])
        assert shortest_path_length(env, start, goal) == pytest.approx(138.0)
        times = np.arange(2) / 10
        assert traveled_path_length(
            times, np.vstack([start, goal]), metric="geodesic", env=env
        ) == pytest.approx(138.0)

    def test_position_off_the_centerline_is_measured_from_its_projection(self):
        env = _w_track(2.0, 10.0)
        # 3 cm beside the left arm at y = 20: projects to (-30, 20).
        length = shortest_path_length(
            env, np.array([-33.0, 20.0]), np.array([0.0, 20.0])
        )
        assert length == pytest.approx(20.0 + 30.0 + 20.0)


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


class TestBinInputsUseBinCenters:
    """Functions that take bins measure from each bin's center."""

    LEFT, RIGHT, ABOVE = (42.5, 12.5), (57.5, 12.5), (47.5, 72.5)
    # Between the left and right centers the path climbs to the centers just
    # above the wall end: hypot(5, 60) up, 5 across, hypot(5, 60) down.
    CLIMB = float(np.hypot(5.0, 60.0))

    def _bins(self, env, *points):
        return env.bin_at(np.array(points, dtype=float))

    def test_distance_to_region_goes_around_the_wall(self, divider_arena):
        from neurospatial.behavior import distance_to_region

        left, right = self._bins(divider_arena, self.LEFT, self.RIGHT)
        distances = distance_to_region(
            np.array([left, right]), divider_arena, target_bins=int(right)
        )
        np.testing.assert_allclose(distances, [2 * self.CLIMB + 5.0, 0.0], atol=1e-9)

    def test_path_progress_above_the_wall_end(self, divider_arena):
        from neurospatial.behavior import path_progress

        left, right, above = self._bins(
            divider_arena, self.LEFT, self.RIGHT, self.ABOVE
        )
        progress = path_progress(
            np.array([above]), divider_arena, start_bins=left, goal_bins=right
        )
        # Covered: straight up the left side; left: across and down.
        expected = self.CLIMB / (self.CLIMB + 5.0 + self.CLIMB)
        np.testing.assert_allclose(progress, [expected], atol=1e-12)

    def test_uniform_cost_is_a_multiple_of_the_geodesic_distance(self, divider_arena):
        from neurospatial.behavior import cost_to_goal

        left, right = self._bins(divider_arena, self.LEFT, self.RIGHT)
        cost = cost_to_goal(
            np.array([left]),
            divider_arena,
            goal_bins=int(right),
            cost_map=np.ones(divider_arena.n_bins),
        )
        np.testing.assert_allclose(cost, [2 * (2 * self.CLIMB + 5.0)], rtol=1e-12)

    def test_oblique_cost_in_an_open_field_is_not_a_staircase(self):
        from neurospatial.behavior import cost_to_goal

        edges = np.arange(0.0, 51.0, 1.0)
        env = Environment.from_grid_mask(
            np.ones((50, 50), dtype=bool), grid_edges=(edges, edges)
        )
        start, goal = env.bin_at(np.array([[0.5, 0.5], [30.5, 40.5]]))
        cost = cost_to_goal(
            np.array([start]), env, goal_bins=int(goal), cost_map=np.ones(env.n_bins)
        )
        # Twice the 50 cm straight line; the bin graph would read 2 * 52.4.
        np.testing.assert_allclose(cost, [100.0], rtol=1e-12)

    def test_polar_environments_are_refused(self):
        from neurospatial._exceptions import CartesianOnlyError
        from neurospatial.behavior import path_progress

        polar = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 6,
        )
        with pytest.raises(CartesianOnlyError, match="path_progress"):
            path_progress(np.array([0, 1]), polar, start_bins=0, goal_bins=5)


class TestPositionInputsUseThePositionItself:
    def test_distance_to_reward_goes_around_the_wall(self, divider_arena):
        from neurospatial.behavior import distance_to_reward

        positions = np.array([[42.5, 12.5], [43.0, 12.5]])
        reward = np.array([[57.5, 12.5]])
        distances = distance_to_reward(
            divider_arena,
            np.array([0.0, 0.1]),
            positions,
            np.array([0.05]),
            reward,
            metric="geodesic",
        )
        # Up to the center above the wall end, across, and down to the
        # reward; the second position starts 0.5 cm further right.
        climb_from = np.linalg.norm(positions - [47.5, 72.5], axis=1)
        np.testing.assert_allclose(
            distances, climb_from + 5.0 + np.hypot(5.0, 60.0), atol=1e-9
        )

    @pytest.mark.parametrize("metric", ["geodesic", "euclidean"])
    def test_distance_to_boundary_is_not_quantized_to_bins(self, metric):
        from neurospatial.behavior import distance_to_boundary

        edges = np.arange(0.0, 11.0, 1.0)
        env = Environment.from_grid_mask(
            np.ones((10, 10), dtype=bool), grid_edges=(edges, edges)
        )
        # The rim bin centers nearest (5.2, 5.0) are (9.5, 4.5) and
        # (9.5, 5.5), hypot(4.3, 0.5) away along a straight line inside the
        # arena. Its bin's center (5.5, 5.5) is 4.0 cm from the rim.
        distances = distance_to_boundary(env, np.array([[5.2, 5.0]]), metric=metric)
        np.testing.assert_allclose(distances, [np.hypot(4.3, 0.5)], rtol=1e-12)
