"""Known answers for the path engines in ``neurospatial.ops._paths``.

The any-angle engine measures the shortest path inside the environment; the
bin-graph engine walks the connections between bin centers. Expected values
come from geometry (straight lines, taut strings) or from an independent
shortest-path search (NetworkX), not from the code under test.
"""

import warnings

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.ops import _bin_graph_paths, _environment_paths


def _grid(n, size, walls=None):
    edges = np.linspace(0.0, n * size, n + 1)
    arena = None
    if walls is not None:
        arena = OpenArena(box(0, 0, n * size, n * size), walls=walls)
    with warnings.catch_warnings():
        # A slanted wall snaps to the grid edges; that is the wall measured.
        warnings.simplefilter("ignore", UserWarning)
        return Environment.from_grid_mask(
            np.ones((n, n), dtype=bool),
            grid_edges=(edges, edges),
            units="cm",
            arena=arena,
        )


@pytest.fixture(scope="module")
def walled():
    """100 x 100 cm of 2 cm bins with a divider and a slanted wall."""
    return _grid(
        50,
        2.0,
        walls={
            "divider": LineString([(50, 0), (50, 70)]),
            "slant": LineString([(10, 80), (40, 90)]),
        },
    )


def _inside(env, n, rng):
    lo, hi = env.bin_centers.min(0) - 1, env.bin_centers.max(0) + 1
    points = rng.uniform(lo, hi, (4 * n, env.n_dims))
    return points[env.bin_at(points) >= 0][:n]


class TestSymmetricPairs:
    def test_shorter_of_both_directions_near_walls(self, walled):
        rng = np.random.default_rng(0)
        starts, ends = _inside(walled, 300, rng), _inside(walled, 300, rng)
        paths = _environment_paths(walled, call="test")
        forward = paths.between(starts, ends)
        backward = paths.between(ends, starts)
        np.testing.assert_array_equal(forward, backward)
        # Never below the straight line, and never above the path along the
        # field grown from either end.
        a, b = paths.bins(starts), paths.bins(ends)
        straight = np.linalg.norm(ends - starts, axis=1)
        assert np.all(forward >= straight - 1e-9)
        assert np.all(forward <= paths._toward(starts, ends, a, b) + 1e-9)
        assert np.all(forward <= paths._toward(ends, starts, b, a) + 1e-9)

    def test_the_two_directions_differ_where_theta_star_misses(self, walled):
        # Bins (11, 83) and (91, 77): the field grown from (91, 77) goes
        # round the left end of the slanted wall (86.50 cm), the one grown
        # from (11, 83) round its right end (82.97 cm). Both legs of the
        # shorter path are visible, so it is a path inside the room.
        start, end = np.array([[11.0, 83.0]]), np.array([[91.0, 77.0]])
        paths = _environment_paths(walled, call="test")
        a, b = paths.bins(start), paths.bins(end)
        one, other = paths._toward(start, end, a, b), paths._toward(end, start, b, a)
        assert abs(one[0] - other[0]) > 3.0
        waypoint = np.array([[41.0, 91.0]])
        assert paths.sees(start, waypoint)[0]
        assert paths.sees(waypoint, end)[0]
        via = np.linalg.norm(waypoint - start) + np.linalg.norm(end - waypoint)
        np.testing.assert_allclose(paths.between(start, end), [via], rtol=1e-12)

    def test_pair_tables_are_symmetric(self, walled):
        rng = np.random.default_rng(1)
        points = _inside(walled, 300, rng)
        pairs = _environment_paths(walled, call="test").pairwise(points)
        first, second = rng.integers(0, 300, 2000), rng.integers(0, 300, 2000)
        np.testing.assert_array_equal(
            pairs.between(first, second), pairs.between(second, first)
        )


def _ring(rng):
    """Hexagonal bins on a ring: the hole hides opposite sides from each other."""
    samples = rng.uniform(-20, 20, (40_000, 2))
    radius = np.linalg.norm(samples, axis=1)
    return Environment.from_samples(
        samples[(radius > 8) & (radius < 20)], bin_size=2.0, layout="hexagonal"
    )


@pytest.mark.parametrize("layout", ["walled", "open", "ring"])
def test_pair_tables_measure_what_single_pairs_measure(layout, walled):
    # ``pairwise(points).between(i, j)`` and ``between(points[i], points[j])``
    # are two routes to one length: the straight line where the segment stays
    # inside, else the shorter grown path.
    rng = np.random.default_rng(7)
    env = {"walled": walled, "open": _grid(20, 2.0), "ring": _ring(rng)}[layout]
    points = _inside(env, 400, rng)
    first, second = rng.integers(0, 400, 6000), rng.integers(0, 400, 6000)
    paths = _environment_paths(env, call="test")
    single = paths.between(points[first], points[second])
    np.testing.assert_allclose(
        paths.pairwise(points).between(first, second), single, rtol=1e-9
    )
    visible = paths.sees(points[first], points[second])
    if layout != "open":
        # Both kinds of pair are compared.
        assert 0.1 < visible.mean() < 0.9
    np.testing.assert_allclose(
        single[visible],
        np.linalg.norm(points[second[visible]] - points[first[visible]], axis=1),
        rtol=1e-12,
    )


class TestStraightLines:
    def test_a_visible_segment_is_the_straight_line(self, walled):
        rng = np.random.default_rng(2)
        starts, ends = _inside(walled, 3000, rng), _inside(walled, 3000, rng)
        paths = _environment_paths(walled, call="test")
        visible = paths.sees(starts, ends)
        assert 0.2 < visible.mean() < 0.9
        np.testing.assert_allclose(
            paths.between(starts[visible], ends[visible]),
            np.linalg.norm(ends[visible] - starts[visible], axis=1),
            rtol=1e-12,
        )

    def test_a_path_across_the_divider_is_the_taut_string(self, walled):
        # Either side of the divider, below its end at (50, 70): no path is
        # shorter than the string around the end; bin-center waypoints keep
        # the path within one bin diagonal of it.
        start, end = np.array([[45.0, 20.0]]), np.array([[55.0, 20.0]])
        tip = np.array([50.0, 70.0])
        taut = np.linalg.norm(tip - start) + np.linalg.norm(end - tip)
        length = _environment_paths(walled, call="test").between(start, end)[0]
        assert taut <= length <= taut + 2 * np.hypot(2.0, 2.0)


class TestLimitedFields:
    @pytest.mark.parametrize("limit", [3.0, 10.0, 25.0])
    def test_limited_field_is_the_full_field_cut_at_the_limit(self, walled, limit):
        paths = _environment_paths(walled, call="test")
        for sources in (np.array([1210]), np.array([0, 1275, 2499])):
            walled._cache.clear()
            paths = _environment_paths(walled, call="test")
            limited = paths.from_bins(sources, limit=limit)
            full = paths.from_bins(sources)
            expected = np.where(full <= limit, full, np.inf)
            np.testing.assert_array_equal(limited, expected)
            assert np.isfinite(limited).sum() > len(sources)

    def test_limited_field_on_a_hexagonal_layout(self):
        rng = np.random.default_rng(3)
        env = Environment.from_samples(
            rng.uniform(0, 40, (20_000, 2)), bin_size=2.0, layout="hexagonal"
        )
        paths = _environment_paths(env, call="test")
        source = np.array([env.n_bins // 2])
        limited = paths.from_bins(source, limit=8.0)
        env._cache.clear()
        full = _environment_paths(env, call="test").from_bins(source)
        np.testing.assert_array_equal(limited, np.where(full <= 8.0, full, np.inf))


def test_fields_grow_in_batches_bounded_by_layout_size():
    from neurospatial.ops import _paths

    rng = np.random.default_rng(4)
    cube = Environment.from_samples(rng.uniform(0, 30, (100_000, 3)), bin_size=2.0)
    paths = _environment_paths(cube, call="test")
    degree = paths._neighbors.shape[1]
    assert degree == 26
    assert paths._batch * cube.n_bins * degree <= _paths._GROW_EDGE_ENTRIES
    assert paths._batch < _paths._FIELD_BATCH
    small = _environment_paths(_grid(10, 2.0), call="test")
    assert small._batch == _paths._FIELD_BATCH


class TestBinGraphPaths:
    @pytest.fixture(scope="class")
    def graph_lengths(self, walled):
        """NetworkX path lengths from a bin, computed when first asked for."""

        class Lengths(dict):
            def __missing__(self, source):
                self[source] = nx.single_source_dijkstra_path_length(
                    walled.connectivity, source, weight="distance"
                )
                return self[source]

        return Lengths()

    def test_pairs_are_graph_distances_between_their_bins(self, walled, graph_lengths):
        rng = np.random.default_rng(5)
        starts, ends = _inside(walled, 150, rng), _inside(walled, 150, rng)
        paths = _bin_graph_paths(walled)
        a, b = walled.bin_at(starts), walled.bin_at(ends)
        expected = [graph_lengths[int(i)][int(j)] for i, j in zip(a, b, strict=True)]
        np.testing.assert_allclose(paths.between(starts, ends), expected, rtol=1e-12)
        np.testing.assert_allclose(
            paths.pairwise(np.vstack([starts, ends])).between(
                np.arange(150), np.arange(150, 300)
            ),
            expected,
            rtol=1e-12,
        )

    def test_positions_in_one_bin_are_zero_apart(self, walled):
        center = walled.bin_centers[100]
        paths = _bin_graph_paths(walled)
        assert paths.between(center[None] - 0.4, center[None] + 0.4)[0] == 0.0

    def test_field_and_direction_follow_the_graph(self, walled, graph_lengths):
        paths = _bin_graph_paths(walled)
        source = 1210
        field = paths.from_bins(np.array([source]))
        np.testing.assert_allclose(
            field, [graph_lengths[source][b] for b in range(walled.n_bins)], rtol=1e-12
        )
        # Each bin's direction points at a neighbor one edge nearer the goal.
        points = walled.bin_centers[::37]
        heading = paths.directions(points, walled.bin_centers[source])
        bins = walled.bin_at(points)
        for point, b, unit in zip(points, bins, heading, strict=True):
            if b == source:
                continue
            step = point + unit * np.min(
                [
                    np.linalg.norm(walled.bin_centers[n] - point)
                    for n in walled.connectivity.neighbors(int(b))
                ]
            )
            nxt = int(np.argmin(np.linalg.norm(walled.bin_centers - step, axis=1)))
            assert nxt in walled.connectivity[int(b)]
            edge = walled.connectivity.edges[int(b), nxt]["distance"]
            np.testing.assert_allclose(
                graph_lengths[source][nxt] + edge, graph_lengths[source][int(b)]
            )

    def test_polar_environments_are_allowed(self):
        polar = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=5.0,
            angle_bin_size=np.pi / 6,
        )
        field = _bin_graph_paths(polar).from_bins(np.array([0]))
        expected = nx.single_source_dijkstra_path_length(
            polar.connectivity, 0, weight="distance"
        )
        np.testing.assert_allclose(
            field, [expected[b] for b in range(polar.n_bins)], rtol=1e-12
        )


@pytest.mark.slow
def test_a_long_session_keeps_whole_fields_with_the_same_lengths():
    """Many pairs per bin grow whole fields once; few pairs grow partial ones."""
    room = _grid(20, 5.0, walls={"divider": LineString([(50, 0), (50, 70)])})
    rng = np.random.default_rng(6)
    bins = rng.choice(room.n_bins, 20, replace=False)

    def near(k):
        jitter = rng.uniform(-2.0, 2.0, (k, 2))
        return room.bin_centers[rng.choice(bins, k)] + jitter

    starts, ends = near(1000), near(1000)
    session = _environment_paths(room, call="test").between(starts, ends)
    assert _environment_paths(room, call="test")._fields.n_kept > 0
    room._cache.clear()
    paths = _environment_paths(room, call="test")
    pieces = np.concatenate(
        [paths.between(starts[k : k + 2], ends[k : k + 2]) for k in range(0, 1000, 2)]
    )
    assert paths._fields.n_kept == 0
    np.testing.assert_array_equal(session, pieces)


class TestCornersOfFourConnectedGrids:
    """A line through a cell corner passes only when every cell around it is free."""

    @staticmethod
    def _room(n, missing=(), walls=None):
        mask = np.ones((n, n), dtype=bool)
        for cell in missing:
            mask[cell] = False
        edges = np.arange(n + 1) * 2.0
        arena = (
            None if walls is None else OpenArena(box(0, 0, 2 * n, 2 * n), walls=walls)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Environment.from_grid_mask(
                mask,
                grid_edges=(edges, edges),
                connect_diagonal_neighbors=False,
                arena=arena,
            )

    def test_an_open_room_is_the_straight_line_between_every_two_centers(self):
        # Many center-to-center lines run exactly through cell corners here.
        room = self._room(12)
        paths = _environment_paths(room, call="test")
        a, b = np.triu_indices(room.n_bins, k=1)
        straight = np.linalg.norm(room.bin_centers[a] - room.bin_centers[b], axis=1)
        np.testing.assert_allclose(paths.between_bins(a, b), straight, rtol=1e-12)

    @pytest.mark.parametrize(
        "obstacle",
        [
            {"missing": [(1, 0)]},  # the cell [2, 4] x [0, 2] is not a bin
            {"walls": {"edge": LineString([(2, 0), (2, 2)])}},  # a wall up to (2, 2)
        ],
        ids=["missing-cell", "wall-to-the-corner"],
    )
    def test_a_corner_touching_an_obstacle_blocks(self, obstacle):
        # From the center (1, 1) to (5, 5) the line runs through the corner
        # (2, 2), which the missing cell or the wall touches.
        room = self._room(4, **obstacle)
        start, end = np.array([[1.0, 1.0]]), np.array([[5.0, 5.0]])
        paths = _environment_paths(room, call="test")
        assert not paths.sees(start, end)[0]
        assert paths.between(start, end)[0] > np.hypot(4.0, 4.0) + 1e-6
        # Shifted off the corner, the line passes beside it.
        assert paths.sees(np.array([[1.0, 1.3]]), end)[0]
