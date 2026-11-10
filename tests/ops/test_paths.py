"""Known answers for the path engines in ``neurospatial.ops._paths``.

The any-angle engine measures the shortest path inside the environment; the
bin-graph engine walks the connections between bin centers. Expected values
come from geometry (straight lines, paths through named bin centers), from
Shapely's segment-against-wall test (``path_oracles``) or from an
independent shortest-path search (NetworkX).

A few tests compare two routes of the engine with each other instead: both
directions of a pair, pair tables against single pairs, a limited field
against the full one, a long query against the same pairs in pieces. Their
names say so. They check that the routes agree, not that either is right.
"""

import itertools
import warnings

import networkx as nx
import numpy as np
import pytest
import shapely
from path_oracles import clear_of_walls
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
    @pytest.mark.parametrize("ends_at", ["positions", "bin centers"])
    def test_line_of_sight_is_a_segment_clear_of_the_walls(self, walled, ends_at):
        # Shapely decides which segments touch a wall of the arena (the
        # walls as fitted to the bin edges, ``walled.arena.walls``).
        rng = np.random.default_rng(8)
        starts, ends = _inside(walled, 20_000, rng), _inside(walled, 20_000, rng)
        if ends_at == "bin centers":
            # Lines between centers run exactly through cell corners.
            starts = walled.bin_centers[walled.bin_at(starts)]
            ends = walled.bin_centers[walled.bin_at(ends)]
        clear = clear_of_walls(walled, starts, ends)
        assert 0.2 < clear.mean() < 0.9
        # A segment through the very end of a wall is neither: left out.
        wall_ends = shapely.multipoints(
            [end for wall in walled.arena.walls for end in wall.boundary.geoms]
        )
        segments = shapely.linestrings(np.stack([starts, ends], axis=1))
        decided = shapely.distance(segments, wall_ends) > 1e-9
        assert decided.mean() > 0.99
        sees = _environment_paths(walled, call="test").sees(starts, ends)
        np.testing.assert_array_equal(sees[decided], clear[decided])

    def test_a_segment_clear_of_the_walls_is_the_straight_line(self, walled):
        rng = np.random.default_rng(2)
        starts, ends = _inside(walled, 3000, rng), _inside(walled, 3000, rng)
        clear = clear_of_walls(walled, starts, ends)
        assert 0.2 < clear.mean() < 0.9
        paths = _environment_paths(walled, call="test")
        np.testing.assert_allclose(
            paths.between(starts[clear], ends[clear]),
            np.linalg.norm(ends[clear] - starts[clear], axis=1),
            rtol=1e-12,
        )
        # A segment through a wall is longer than the straight line.
        blocked = paths.between(starts[~clear], ends[~clear])
        assert np.all(
            blocked > np.linalg.norm(ends[~clear] - starts[~clear], axis=1) + 1e-9
        )

    def test_a_path_across_the_divider_turns_at_the_centers_past_its_end(self, walled):
        # Either side of the divider, which ends at (50, 70). A path turns
        # at bin centers, and the nearest ones that see past the wall's end
        # are (49, 71) and (51, 71): up 51 and across 4, 2 over the end,
        # and down again. The string pulled taut around the end itself is
        # 0.3 cm shorter; the staircase along the bin graph is 1 cm longer.
        start, end = np.array([[45.0, 20.0]]), np.array([[55.0, 20.0]])
        around = 2 * np.hypot(4.0, 51.0) + 2.0
        length = _environment_paths(walled, call="test").between(start, end)[0]
        assert length == pytest.approx(around, rel=1e-12)
        assert 2 * np.hypot(5.0, 50.0) < around


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


class TestCellCorners:
    """A line through a cell corner passes only when the corner is free."""

    @staticmethod
    def _room(shape, sizes, missing=(), walls=None, diagonal=False):
        """A grid of ``shape`` cells of ``sizes``, joining face neighbors only."""
        mask = np.ones(shape, dtype=bool)
        for cell in missing:
            mask[cell] = False
        edges = tuple(
            np.arange(n + 1) * size for n, size in zip(shape, sizes, strict=True)
        )
        arena = (
            None
            if walls is None
            else OpenArena(box(0, 0, edges[0][-1], edges[1][-1]), walls=walls)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Environment.from_grid_mask(
                mask,
                grid_edges=edges,
                connect_diagonal_neighbors=diagonal,
                arena=arena,
            )

    @pytest.mark.parametrize(
        ("shape", "sizes"),
        [
            ((12, 12), (2.0, 2.0)),
            ((10, 8), (2.0, 3.0)),
            ((5, 5, 5), (2.0, 2.0, 2.0)),
            ((5, 4, 3), (2.0, 3.0, 5.0)),
        ],
        ids=["squares", "rectangles", "cubes", "boxes"],
    )
    def test_an_open_room_is_the_straight_line_between_every_two_centers(
        self, shape, sizes
    ):
        # Many center-to-center lines run exactly through cell corners (and,
        # in 3-D, cell edges) here; every cell around them is a bin.
        room = self._room(shape, sizes)
        paths = _environment_paths(room, call="test")
        a, b = np.triu_indices(room.n_bins, k=1)
        straight = np.linalg.norm(room.bin_centers[a] - room.bin_centers[b], axis=1)
        np.testing.assert_allclose(paths.between_bins(a, b), straight, rtol=1e-12)

    @pytest.mark.parametrize("sizes", [(2.0, 2.0), (2.0, 3.0)])
    @pytest.mark.parametrize(
        "obstacle",
        ["missing-cell", "wall-to-the-corner"],
    )
    def test_a_corner_touching_an_obstacle_blocks(self, obstacle, sizes):
        # From the center of cell (0, 0) to that of cell (2, 2) the line runs
        # through the corner shared by cells (0, 0), (1, 0), (0, 1) and
        # (1, 1), which the missing cell (1, 0) or a wall along the face
        # between cells (0, 0) and (1, 0) touches.
        width, height = sizes
        if obstacle == "missing-cell":
            room = self._room((4, 4), sizes, missing=[(1, 0)])
        else:
            wall = LineString([(width, 0), (width, height)])
            room = self._room((4, 4), sizes, walls={"edge": wall})
        start = np.array([[0.5 * width, 0.5 * height]])
        end = np.array([[2.5 * width, 2.5 * height]])
        paths = _environment_paths(room, call="test")
        assert not paths.sees(start, end)[0]
        # The path turns at one bin center beside the corner: up one cell
        # to the center of cell (0, 1) and on to the end (it crosses cells
        # (1, 1) and (1, 2)), or across the top of cell (1, 1) from the
        # center of cell (1, 2), whichever is shorter.
        around = min(
            height + np.hypot(2 * width, height), width + np.hypot(width, 2 * height)
        )
        assert around > np.linalg.norm(end - start) + 0.5
        assert paths.between(start, end)[0] == pytest.approx(around, rel=1e-12)
        # Shifted off the corner, the line passes beside it.
        assert paths.sees(start + np.array([0.0, 0.15 * height]), end)[0]

    @pytest.mark.parametrize(
        "missing",
        [c for c in itertools.product((0, 1), repeat=3) if 0 < sum(c) < 3],
    )
    def test_a_missing_cell_at_a_vertex_blocks_the_body_diagonal(self, missing):
        # The line between the centers of cubes (0, 0, 0) and (1, 1, 1) runs
        # through the vertex (2, 2, 2); the other six cubes around it each
        # touch that vertex.
        room = self._room((3, 3, 3), (2.0, 2.0, 2.0), missing=[missing])
        start, end = np.array([[1.0, 1.0, 1.0]]), np.array([[3.0, 3.0, 3.0]])
        paths = _environment_paths(room, call="test")
        assert not paths.sees(start, end)[0]
        assert paths.between(start, end)[0] > np.sqrt(12.0) + 1e-6

    def test_only_cells_around_a_3d_cell_edge_block_a_line_through_it(self):
        # Between the centers of cubes (0, 0, 0) and (1, 1, 0) the line runs
        # through the cell edge x = y = 2 at height z = 1. The four cubes
        # around it are in layer 0; the cube (1, 0, 1) above does not touch
        # the line.
        start, end = np.array([[1.0, 1.0, 1.0]]), np.array([[3.0, 3.0, 1.0]])
        above = self._room((3, 3, 3), (2.0, 2.0, 2.0), missing=[(1, 0, 1)])
        assert _environment_paths(above, call="test").sees(start, end)[0]
        beside = self._room((3, 3, 3), (2.0, 2.0, 2.0), missing=[(1, 0, 0)])
        assert not _environment_paths(beside, call="test").sees(start, end)[0]

    def test_a_wall_through_a_corner_closes_the_diagonal_connection(self):
        # On a grid that joins diagonal neighbors, the wall from (9, 11) to
        # (11, 9) through the corner (10, 10) closes the connection between
        # the bins at (9, 9) and (11, 11) and no face between bins. The
        # shortest path between their centers goes around the wall's end at
        # (9, 11) or (11, 9): 2 + 2 cm.
        wall = LineString([(9, 11), (11, 9)])
        room = self._room((10, 10), (2.0, 2.0), walls={"wall": wall}, diagonal=True)
        start, end = np.array([[9.0, 9.0]]), np.array([[11.0, 11.0]])
        a, b = room.bin_at(start)[0], room.bin_at(end)[0]
        assert not room.connectivity.has_edge(a, b)
        corner_cells = room.bin_at(np.array([[9.0, 11.0], [11.0, 9.0]]))
        for cell in corner_cells:
            assert room.connectivity.has_edge(a, cell)
            assert room.connectivity.has_edge(cell, b)
        paths = _environment_paths(room, call="test")
        assert not paths.sees(start, end)[0]
        np.testing.assert_allclose(paths.between(start, end), [4.0], rtol=1e-12)
        # A corner no wall touches still passes.
        np.testing.assert_allclose(
            paths.between(start - 4.0, end - 4.0), [np.hypot(2.0, 2.0)], rtol=1e-12
        )
