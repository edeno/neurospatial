"""Known answers for the distance primitives in ``neurospatial.ops.distance``.

Expected values come from geometry, not from the code under test: straight
lines between bin centers in a convex room, the taut string around a wall
end, octile (staircase) lengths on an 8-connected grid, and NetworkX
shortest paths for the bin graph.
"""

import warnings

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose
from path_oracles import FIELD_EXCESS, PAIR_EXCESS, through_centers
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.arena import OpenArena
from neurospatial.ops.distance import (
    _bin_adjacency,
    _geodesic_pair_distances,
    distance_field,
    euclidean_distance_matrix,
    geodesic_distance_matrix,
    neighbors_within,
    pairwise_distances,
)


def _grid(mask, size, walls=None, diagonal=True):
    """Grid of ``mask``'s cells; ``size`` is one cell size or one per axis."""
    mask = np.asarray(mask, dtype=bool)
    sizes = np.broadcast_to(size, (mask.ndim,))
    edges = [
        np.arange(n + 1) * width for n, width in zip(mask.shape, sizes, strict=True)
    ]
    arena = None
    if walls is not None:
        arena = OpenArena(box(0, 0, edges[0][-1], edges[1][-1]), walls=walls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Environment.from_grid_mask(
            mask,
            grid_edges=tuple(edges),
            units="cm",
            arena=arena,
            connect_diagonal_neighbors=diagonal,
        )


_DIVIDER = {"divider": LineString([(50, 0), (50, 70)])}


def _hexagons_around_a_barrier():
    """3 cm hexagons over 60 x 60 cm, none in a 5 cm wide strip up to y = 42."""
    samples = np.random.default_rng(0).uniform(0, 60, (60_000, 2))
    barrier = (np.abs(samples[:, 0] - 30) < 2.5) & (samples[:, 1] < 42)
    return Environment.from_samples(samples[~barrier], bin_size=3.0, layout="hexagonal")


def _cubes_around_a_slab(diagonal):
    """10 x 10 x 5 cubes of 2 cm, without the slab x in [10, 12], y < 14."""
    mask = np.ones((10, 10, 5), dtype=bool)
    mask[5, :7, :] = False
    return _grid(mask, 2.0, diagonal=diagonal)


# Layouts with something to go around, and the most a one-direction field
# exceeded the pair distance over every pair of their bins (as a share of
# the pair distance), as the Notes of ``distance_field`` quote it.
_OBSTRUCTED = {
    "square-8-connected": (lambda: _grid(np.ones((20, 20)), 5.0, _DIVIDER), 0.056),
    "square-4-connected": (
        lambda: _grid(np.ones((20, 20)), 5.0, _DIVIDER, diagonal=False),
        0.056,
    ),
    "rectangular": (
        lambda: _grid(
            np.ones((20, 25)), (5.0, 4.0), {"divider": LineString([(50, 0), (50, 68)])}
        ),
        0.073,
    ),
    "hexagonal": (_hexagons_around_a_barrier, 0.099),
    "cubes-26-connected": (lambda: _cubes_around_a_slab(True), 0.031),
    "cubes-6-connected": (lambda: _cubes_around_a_slab(False), 0.064),
}


@pytest.fixture(scope="module")
def line():
    """Five 1 cm bins in a row: bin i is centred at x = i + 0.5."""
    return _grid(np.ones((5, 1)), 1.0)


@pytest.fixture(scope="module")
def square():
    """A convex 5 x 5 grid of 2 cm bins."""
    return _grid(np.ones((5, 5)), 2.0)


@pytest.fixture(scope="module")
def walled():
    """100 x 100 cm of 5 cm bins, divided along x = 50 up to y = 70."""
    return _grid(
        np.ones((20, 20)), 5.0, walls={"divider": LineString([(50, 0), (50, 70)])}
    )


@pytest.fixture(scope="module")
def through_walled_centers(walled):
    """Shortest paths of ``walled`` that turn only at bin centers (not the engine)."""
    return through_centers(walled)


@pytest.fixture(scope="module")
def two_rooms():
    """Two 2 x 2 rooms of 1 cm bins with an empty column between them."""
    mask = np.ones((5, 2), dtype=bool)
    mask[2] = False
    return _grid(mask, 1.0)


@pytest.fixture(scope="module")
def polar():
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )


def _straight(env):
    return euclidean_distance_matrix(np.asarray(env.bin_centers))


def _assert_pair_distances(values, shortest):
    """Symmetric pair distances against the shortest paths through bin centers.

    The bin graph cannot pass: its staircases are up to 8.2% longer, and it
    equals the shortest path through centers for one pair in eight.
    """
    assert np.all(values >= shortest - 1e-9)
    assert np.all(values <= (1 + PAIR_EXCESS) * shortest + 1e-9)
    assert np.mean(np.isclose(values, shortest, rtol=1e-12, atol=1e-12)) > 0.75


def _assert_field_lengths(values, shortest):
    """One-direction field lengths against the shortest paths through bin centers."""
    assert np.all(values >= shortest - 1e-9)
    assert np.all(values <= (1 + FIELD_EXCESS) * shortest + 1e-9)
    assert np.mean(np.isclose(values, shortest, rtol=1e-12, atol=1e-12)) > 0.7


def _octile(env):
    """Staircase length on an 8-connected square grid: diagonals then straights."""
    d = np.abs(env.bin_centers[:, None, :] - env.bin_centers[None, :, :])
    low, high = d.min(axis=2), d.max(axis=2)
    return np.sqrt(2) * low + (high - low)


def _networkx(env, sources):
    lengths = nx.multi_source_dijkstra_path_length(
        env.connectivity, {int(s) for s in sources}, weight="distance"
    )
    return np.array([lengths.get(b, np.inf) for b in range(env.n_bins)])


class TestEuclideanDistanceMatrix:
    """Tests for euclidean_distance_matrix function."""

    def test_basic_2d_distances(self):
        """Test basic 2D Euclidean distance calculation."""
        centers = np.array([[0, 0], [3, 4], [6, 8]])

        result = euclidean_distance_matrix(centers)

        # Distance from [0,0] to [3,4] is 5.0
        # Distance from [0,0] to [6,8] is 10.0
        # Distance from [3,4] to [6,8] is 5.0
        expected = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]])

        assert_allclose(result, expected)

    def test_single_point(self):
        """Test distance matrix for single point."""
        centers = np.array([[1, 2]])

        result = euclidean_distance_matrix(centers)
        expected = np.array([[0.0]])

        assert_allclose(result, expected)
        assert result.shape == (1, 1)

    def test_two_points(self):
        """Test distance matrix for two points."""
        centers = np.array([[0, 0], [3, 4]])

        result = euclidean_distance_matrix(centers)

        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0
        assert_allclose(result[0, 1], 5.0)
        assert_allclose(result[1, 0], 5.0)

    def test_3d_distances(self):
        """Test 3D Euclidean distance calculation."""
        centers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = euclidean_distance_matrix(centers)

        # Diagonal should be zero
        assert_allclose(np.diag(result), [0.0, 0.0, 0.0, 0.0])

        # All off-diagonal unit distances should be 1.0
        assert_allclose(result[0, 1], 1.0)
        assert_allclose(result[0, 2], 1.0)
        assert_allclose(result[0, 3], 1.0)

        # Distance between [1,0,0] and [0,1,0] is sqrt(2)
        assert_allclose(result[1, 2], np.sqrt(2.0))

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        rng = np.random.default_rng(42)
        centers = rng.random((5, 2))

        result = euclidean_distance_matrix(centers)

        assert_allclose(result, result.T)

    def test_empty_points(self):
        """Test with empty points array."""
        centers = np.empty((0, 2))

        result = euclidean_distance_matrix(centers)

        assert result.shape == (0, 0)

    def test_colinear_points(self):
        """Test with co-linear points."""
        centers = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

        result = euclidean_distance_matrix(centers)

        # Distances should be difference in x-coordinates
        expected = np.array(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float
        )

        assert_allclose(result, expected)

    def test_identical_points(self):
        """Test with identical points."""
        centers = np.array([[1, 2], [1, 2], [1, 2]])

        result = euclidean_distance_matrix(centers)

        # All distances should be zero
        expected = np.zeros((3, 3))
        assert_allclose(result, expected)

    def test_large_distances(self):
        """Test with large coordinate values."""
        centers = np.array([[0, 0], [1000, 0], [0, 1000]])

        result = euclidean_distance_matrix(centers)

        assert_allclose(result[0, 1], 1000.0)
        assert_allclose(result[0, 2], 1000.0)
        assert_allclose(result[1, 2], 1000.0 * np.sqrt(2))

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        centers = np.array([[-1, -2], [3, 4]])

        result = euclidean_distance_matrix(centers)

        # Distance is sqrt((3-(-1))^2 + (4-(-2))^2) = sqrt(16 + 36) = sqrt(52)
        expected_dist = np.sqrt(52)
        assert_allclose(result[0, 1], expected_dist)
        assert_allclose(result[1, 0], expected_dist)


class TestDistanceField:
    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_along_a_row_of_bins(self, line, metric):
        assert_allclose(distance_field(line, [2], metric=metric), [2, 1, 0, 1, 2])
        assert_allclose(distance_field(line, [0, 4], metric=metric), [0, 1, 2, 1, 0])

    def test_geodesic_is_the_straight_line_in_a_convex_room(self, square):
        assert_allclose(distance_field(square, [0]), _straight(square)[0], rtol=1e-12)
        assert_allclose(
            distance_field(square, [0, 24]),
            np.minimum(_straight(square)[0], _straight(square)[24]),
            rtol=1e-12,
        )

    def test_bin_graph_is_the_staircase(self, square):
        assert_allclose(
            distance_field(square, [0], metric="bin_graph"),
            _octile(square)[0],
            rtol=1e-12,
        )

    def test_bin_graph_is_the_networkx_shortest_path(self, walled):
        for sources in ([0], [37, 250, 399]):
            assert_allclose(
                distance_field(walled, sources, metric="bin_graph"),
                _networkx(walled, sources),
                rtol=1e-12,
            )

    def test_geodesic_goes_around_the_wall_end(self, walled, through_walled_centers):
        # From the center (37.5, 12.5) to (62.5, 12.5) across the divider,
        # which ends at (50, 70). A path turns at bin centers, and the
        # nearest ones that see past the wall's end are (47.5, 72.5) and
        # (52.5, 72.5): up 60 and across 10, 5 over the end, and down again.
        source = int(walled.bin_at(np.array([[37.5, 12.5]]))[0])
        target = int(walled.bin_at(np.array([[62.5, 12.5]]))[0])
        around = 2 * np.hypot(10.0, 60.0) + 5.0
        assert through_walled_centers[source, target] == pytest.approx(
            around, rel=1e-12
        )
        field = distance_field(walled, [source])
        assert field[target] == pytest.approx(around, rel=1e-12)
        # The staircase along the bin graph is longer; the straight line
        # runs through the wall.
        assert distance_field(walled, [source], metric="bin_graph")[
            target
        ] == pytest.approx(2 * (10 * np.sqrt(2) + 50.0) + 5.0, rel=1e-12)
        assert distance_field(walled, [source], metric="euclidean")[target] == 25.0

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_cutoff(self, square, metric):
        full = distance_field(square, [12], metric=metric)
        for cutoff in (0.0, 2.0, 3.0, 4.5):
            assert_allclose(
                distance_field(square, [12], metric=metric, cutoff=cutoff),
                np.where(full <= cutoff, full, np.inf),
            )

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph"])
    def test_disconnected_bins_are_infinitely_far(self, two_rooms, metric):
        field = distance_field(two_rooms, [0], metric=metric)
        assert np.all(np.isfinite(field[:4]))
        assert np.all(np.isinf(field[4:]))

    def test_refusals(self, line):
        with pytest.raises(ValueError, match="at least one"):
            distance_field(line, [])
        with pytest.raises(ValueError, match="'geodesic', 'bin_graph' or 'euclidean'"):
            distance_field(line, [0], metric="manhattan")
        with pytest.raises(ValueError, match="cutoff must be non-negative"):
            distance_field(line, [0], cutoff=-1.0)

    def test_polar_takes_only_the_bin_graph(self, polar):
        field = distance_field(polar, [0], metric="bin_graph")
        assert_allclose(field, _networkx(polar, [0]), rtol=1e-12)
        for metric in ("geodesic", "euclidean"):
            with pytest.raises(CartesianOnlyError, match="metric='bin_graph'"):
                distance_field(polar, [0], metric=metric)


class TestPairwiseAndMatrix:
    def test_geodesic_matrix_is_the_straight_line_in_a_convex_room(self, square):
        assert_allclose(geodesic_distance_matrix(square), _straight(square), rtol=1e-12)

    def test_bin_graph_matrix_is_the_staircase(self, square):
        assert_allclose(
            geodesic_distance_matrix(square, metric="bin_graph"),
            _octile(square),
            rtol=1e-12,
        )

    def test_geodesic_matrix_near_a_wall_is_the_path_through_bin_centers(
        self, walled, through_walled_centers
    ):
        matrix = geodesic_distance_matrix(walled)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)
        # Against the shortest path that turns only at bin centers, found
        # by Dijkstra over a visibility graph built in the tests: never
        # shorter, the same for most pairs, and nowhere longer than a path
        # grown greedily from both ends was measured to be.
        _assert_pair_distances(matrix, through_walled_centers)

    def test_pairwise_is_the_matrix_restricted_to_its_bins(self, walled):
        bins = np.array([3, 77, 150, 151, 299, 398])
        matrix = geodesic_distance_matrix(walled)
        assert_allclose(pairwise_distances(walled, bins), matrix[np.ix_(bins, bins)])
        assert_allclose(
            pairwise_distances(walled, bins, metric="bin_graph"),
            geodesic_distance_matrix(walled, metric="bin_graph")[np.ix_(bins, bins)],
        )
        assert_allclose(
            pairwise_distances(walled, bins, metric="euclidean"),
            _straight(walled)[np.ix_(bins, bins)],
        )

    def test_pairwise_shapes(self, line):
        assert pairwise_distances(line, []).shape == (0, 0)
        assert_allclose(pairwise_distances(line, [3]), [[0.0]])

    def test_matrix_takes_path_metrics_only(self, line):
        with pytest.raises(ValueError, match="'geodesic' or 'bin_graph'"):
            geodesic_distance_matrix(line, metric="euclidean")

    def test_disconnected_pairs_are_infinitely_far(self, two_rooms):
        matrix = geodesic_distance_matrix(two_rooms)
        assert np.isinf(matrix[0, 5])
        assert np.isfinite(matrix[0, 3])


@pytest.mark.parametrize("layout", list(_OBSTRUCTED))
def test_a_field_lies_between_the_pair_distance_and_the_bin_graph(layout):
    """The documented relation of the three routes, for every pair of bins.

    ``distance_field`` grows each path from the source only; the pair
    distance is the shorter of the paths grown from either end, so it is
    never longer; and a field never exceeds the walk along the bin graph,
    which is one of the paths it chooses from.
    """
    build, one_way_excess = _OBSTRUCTED[layout]
    env = build()
    pair = geodesic_distance_matrix(env)
    graph = geodesic_distance_matrix(env, metric="bin_graph")
    # field[s, t]: from bin t toward source s.
    field = np.stack([distance_field(env, [s]) for s in range(env.n_bins)])
    reached = np.isfinite(pair)
    np.testing.assert_array_equal(np.isfinite(field), reached)
    np.testing.assert_array_equal(pair, pair.T)
    rounding = 1e-9
    assert np.all(pair[reached] <= field[reached] + rounding)
    assert np.all(field[reached] <= graph[reached] + rounding)
    assert np.all(field[reached] <= (1 + one_way_excess) * pair[reached] + rounding)
    # Something is one-way here, and most paths beat the staircase.
    assert np.any(field[reached] > pair[reached] + rounding)
    assert np.mean(field[reached] < graph[reached] - rounding) > 0.5


class TestNeighborsWithin:
    def test_geodesic_ball_in_a_convex_room_is_round(self, square):
        straight = _straight(square)
        for radius in (2.0, 3.0, 4.5, 6.0):
            (geodesic,) = neighbors_within(square, [12], radius=radius)
            expected = np.flatnonzero(straight[12] <= radius)
            np.testing.assert_array_equal(np.sort(geodesic), expected)

    def test_bin_graph_ball_is_the_staircase_ball(self, square):
        (graph,) = neighbors_within(square, [12], radius=4.5, metric="bin_graph")
        np.testing.assert_array_equal(
            np.sort(graph), np.flatnonzero(_octile(square)[12] <= 4.5)
        )

    def test_limited_growth_equals_the_full_field(self, walled):
        # A center's neighborhood is the ball of the field grown from it,
        # for every center and whether the radius stops short of the wall's
        # end or reaches around it.
        centers = np.arange(walled.n_bins)
        fields = np.stack([distance_field(walled, [c]) for c in centers])
        for radius in (7.5, 40.0, 90.0):
            balls = neighbors_within(walled, centers, radius=radius)
            for center, ball in zip(centers, balls, strict=True):
                np.testing.assert_array_equal(
                    np.sort(ball), np.flatnonzero(fields[center] <= radius)
                )

    def test_a_geodesic_ball_is_one_way_near_a_wall(self, walled):
        # Bins 14 and 385 are on either side of the divider. The path grown
        # from 14 reaches 385 in 113.64 cm, the one grown from 385 reaches
        # 14 in 112.27 cm (the pair distance), so at a radius between the
        # two each bin's neighborhood holds the other only one way round.
        radius = 113.0
        to_385 = distance_field(walled, [385])[14]
        to_14 = distance_field(walled, [14])[385]
        assert min(to_385, to_14) < radius < max(to_385, to_14)
        around_14, around_385 = neighbors_within(walled, [14, 385], radius=radius)
        assert (385 in around_14) == (to_14 <= radius)
        assert (14 in around_385) == (to_385 <= radius)
        assert (385 in around_14) != (14 in around_385)
        reachable = walled.reachable_from(bins=14, radius=radius, metric="geodesic")
        assert reachable[385] == (385 in around_14)

    def test_the_wall_splits_a_geodesic_neighborhood(self, walled):
        center = int(walled.bin_at(np.array([[47.5, 12.5]]))[0])
        across = int(walled.bin_at(np.array([[52.5, 12.5]]))[0])
        (geodesic,) = neighbors_within(walled, [center], radius=10.0)
        (straight,) = neighbors_within(
            walled, [center], radius=10.0, metric="euclidean"
        )
        assert across in straight
        assert across not in geodesic

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_center_options(self, line, metric):
        (with_center,) = neighbors_within(line, [2], radius=1.0, metric=metric)
        (without,) = neighbors_within(
            line, [2], radius=1.0, metric=metric, include_center=False
        )
        assert sorted(with_center.tolist()) == [1, 2, 3]
        assert sorted(without.tolist()) == [1, 3]
        assert neighbors_within(line, [2], radius=0.0, metric=metric)[0].tolist() == [2]
        assert neighbors_within(line, [], radius=1.0, metric=metric) == []

    def test_refusals(self, line):
        with pytest.raises(ValueError, match="'geodesic', 'bin_graph' or 'euclidean'"):
            neighbors_within(line, [0], radius=1.0, metric="hops")
        with pytest.raises(ValueError, match="radius must be non-negative"):
            neighbors_within(line, [0], radius=-1.0)


def _networkx_matrix(env):
    lengths = dict(
        nx.all_pairs_dijkstra_path_length(env.connectivity, weight="distance")
    )
    return np.array(
        [
            [lengths[a].get(b, np.inf) for b in range(env.n_bins)]
            for a in range(env.n_bins)
        ]
    )


class TestBinGraphAgainstNetworkX:
    """``metric="bin_graph"`` is NetworkX's shortest path on ``env.connectivity``."""

    @pytest.mark.parametrize(
        "layout", ["walled", "two_rooms", "hexagonal", "cubes-26-connected"]
    )
    def test_matrix_field_pairs_and_neighborhoods(self, layout, walled, two_rooms):
        env = (
            {"walled": walled, "two_rooms": two_rooms}[layout]
            if layout in ("walled", "two_rooms")
            else _OBSTRUCTED[layout][0]()
        )
        expected = _networkx_matrix(env)
        assert_allclose(
            geodesic_distance_matrix(env, metric="bin_graph"), expected, rtol=1e-12
        )
        centers = np.random.default_rng(13).choice(
            env.n_bins, min(6, env.n_bins), replace=False
        )
        assert_allclose(
            pairwise_distances(env, centers, metric="bin_graph"),
            expected[np.ix_(centers, centers)],
            rtol=1e-12,
        )
        assert_allclose(
            distance_field(env, centers[:3], metric="bin_graph"),
            expected[centers[:3]].min(axis=0),
            rtol=1e-12,
        )
        # A radius between two of the distances, so rounding moves no bin.
        finite = np.unique(np.round(expected[np.isfinite(expected)], 9))
        radius = float(finite[len(finite) // 3 : len(finite) // 3 + 2].mean())
        balls = neighbors_within(env, centers, radius=radius, metric="bin_graph")
        for center, ball in zip(centers, balls, strict=True):
            np.testing.assert_array_equal(
                np.sort(ball), np.flatnonzero(expected[center] <= radius)
            )

    def test_disconnected_rooms_are_infinitely_far(self, two_rooms):
        matrix = geodesic_distance_matrix(two_rooms, metric="bin_graph")
        assert np.all(np.isinf(matrix[:4, 4:]))
        assert np.all(np.isfinite(matrix[:4, :4]))


class TestNeighborsWithinSets:
    def test_euclidean_balls_in_two_and_three_dimensions(self, square):
        cubes = _grid(np.ones((4, 4, 4)), 2.0)
        for env, centers in ((square, [0, 12, 24]), (cubes, [0, 21, 63])):
            straight = _straight(env)
            for radius in (2.0, 3.0, 4.5):
                balls = neighbors_within(
                    env, centers, radius=radius, metric="euclidean"
                )
                assert len(balls) == len(centers)
                for center, ball in zip(centers, balls, strict=True):
                    np.testing.assert_array_equal(
                        np.sort(ball), np.flatnonzero(straight[center] <= radius)
                    )

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_a_path_ball_stays_in_its_room_and_a_straight_one_does_not(
        self, two_rooms, metric
    ):
        # Two 2 x 2 rooms of 1 cm bins, 1 cm apart: bin 2 at (1.5, 0.5) and
        # bin 4 at (3.5, 0.5) are 2 cm apart across the gap.
        (ball,) = neighbors_within(two_rooms, [2], radius=2.5, metric=metric)
        if metric == "euclidean":
            assert sorted(ball.tolist()) == [0, 1, 2, 3, 4, 5]
        else:
            assert sorted(ball.tolist()) == [0, 1, 2, 3]

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_each_ball_is_an_array_of_distinct_bin_indices(self, walled, metric):
        centers = [0, 150, 150, 399]  # a center may repeat
        balls = neighbors_within(walled, centers, radius=12.0, metric=metric)
        assert len(balls) == len(centers)
        for center, ball in zip(centers, balls, strict=True):
            assert isinstance(ball, np.ndarray)
            assert ball.dtype == np.int_
            assert ball.ndim == 1
            assert len(np.unique(ball)) == len(ball)
            assert center in ball
        np.testing.assert_array_equal(np.sort(balls[1]), np.sort(balls[2]))


class TestDistanceFieldBeyondTheBasics:
    def test_euclidean_in_three_dimensions(self):
        cubes = _grid(np.ones((4, 4, 4)), 2.0)
        straight = _straight(cubes)
        assert_allclose(
            distance_field(cubes, [0, 63], metric="euclidean"),
            np.minimum(straight[0], straight[63]),
            rtol=1e-12,
        )
        # Bin 21 is at (3, 3, 3) and bin 0 at (1, 1, 1).
        assert distance_field(cubes, [21], metric="euclidean")[0] == pytest.approx(
            np.sqrt(12.0), rel=1e-12
        )

    @pytest.mark.parametrize("metric", ["geodesic", "bin_graph", "euclidean"])
    def test_cutoff_with_several_sources(self, walled, metric):
        sources = [0, 210, 399]
        full = distance_field(walled, sources, metric=metric)
        for cutoff in (0.0, 12.0, 30.0, 75.0):
            limited = distance_field(walled, sources, metric=metric, cutoff=cutoff)
            np.testing.assert_array_equal(
                limited, np.where(full <= cutoff, full, np.inf)
            )
            assert np.isfinite(limited).sum() >= len(sources)
        assert np.isinf(
            distance_field(walled, sources, metric=metric, cutoff=12.0)
        ).any()


def _path_inserted_out_of_order():
    """Bins 0-1-2-3-4 in a row, 1 cm apart, added to the graph out of order."""
    graph = nx.Graph()
    graph.add_nodes_from([3, 0, 4, 2, 1])
    for u in (2, 0, 3, 1):
        graph.add_edge(u, u + 1, distance=1.0)
    return graph


def _reciprocal_digraph():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=2.0)
    graph.add_edge(1, 0, distance=2.0)
    graph.add_edge(1, 2, distance=2.0)
    return graph


def _path_multigraph(directed=False):
    graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
    graph.add_edge(0, 1, distance=2.0)
    graph.add_edge(1, 2, distance=2.0)
    return graph


_GRAPH_READERS = {
    "adjacency": lambda graph, weight: _bin_adjacency(
        graph, weight=weight, operation="test()"
    ),
    "pair distances": lambda graph, weight: _geodesic_pair_distances(
        graph,
        np.array([0]),
        np.array([2]),
        weight=weight,
        operation="test()",
    ),
}


@pytest.mark.parametrize("read", list(_GRAPH_READERS))
class TestBinGraphsAreValidated:
    """Every bin-graph distance reads its graph through these checks.

    No public function takes a graph of its own any more; they pass
    ``env.connectivity``. The checks still stand between any graph and the
    distance arrays, which are indexed by bin.
    """

    def test_rows_are_bins_whatever_the_node_order(self, read):
        graph = _path_inserted_out_of_order()
        if read == "adjacency":
            expected = np.zeros((5, 5))
            for u in range(4):
                expected[u, u + 1] = expected[u + 1, u] = 1.0
            np.testing.assert_array_equal(
                _GRAPH_READERS[read](graph, "distance").toarray(), expected
            )
        else:
            assert_allclose(_GRAPH_READERS[read](graph, "distance"), [2.0])

    def test_a_node_that_is_not_a_bin_index_is_refused(self, read):
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 5, distance=1.0)  # three nodes, but no bin 2
        with pytest.raises(
            ValueError,
            match=r"^test\(\): the graph's nodes must be the bin indices 0 to 2; "
            r"nodes \[5\] are not\.",
        ):
            _GRAPH_READERS[read](graph, "distance")

    def test_an_edge_without_the_weight_is_refused(self, read):
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=2.5)
        graph.add_edge(1, 2, distance=2.5)
        with pytest.raises(
            ValueError,
            match=r"^test\(\): 2 edge\(s\) have no 'dist' attribute, for example "
            r"edge \(0, 1\) \(its attributes: \['distance'\]\)\.",
        ):
            _GRAPH_READERS[read](graph, "dist")
        graph.edges[0, 1]["dist"] = 1.0
        with pytest.raises(
            ValueError,
            match=r"1 edge\(s\) have no 'dist' attribute, for example edge \(1, 2\)",
        ):
            _GRAPH_READERS[read](graph, "dist")

    def test_no_weight_counts_edges(self, read):
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=2.5)
        graph.add_edge(1, 2, distance=2.5)
        if read == "adjacency":
            np.testing.assert_array_equal(
                _GRAPH_READERS[read](graph, None).toarray(),
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            )
        else:
            assert_allclose(_GRAPH_READERS[read](graph, None), [2.0])
            assert_allclose(_GRAPH_READERS[read](graph, "distance"), [5.0])

    @pytest.mark.parametrize(
        ("graph", "kind"),
        [
            (_reciprocal_digraph(), "DiGraph"),
            (_path_multigraph(), "MultiGraph"),
            (_path_multigraph(directed=True), "MultiDiGraph"),
        ],
        ids=["directed", "multi", "directed-multi"],
    )
    @pytest.mark.parametrize("weight", ["distance", None])
    def test_a_directed_or_multigraph_is_refused(self, read, graph, kind, weight):
        with pytest.raises(
            ValueError,
            match=rf"^test\(\): the graph must be an undirected simple nx\.Graph, "
            rf"got {kind}\.",
        ):
            _GRAPH_READERS[read](graph, weight)


_BIN_QUERIES = {
    f"{name}-{metric}": (call, metric)
    for name, call in {
        "field": lambda env, bins, metric: distance_field(env, bins, metric=metric),
        "pairwise": lambda env, bins, metric: pairwise_distances(
            env, bins, metric=metric
        ),
        "neighbors": lambda env, bins, metric: neighbors_within(
            env, bins, radius=1.0, metric=metric
        ),
    }.items()
    for metric in ("geodesic", "bin_graph", "euclidean")
}


class TestInvalidBinsAreRefused:
    """Every distance function refuses an index that is not a bin, for every metric.

    -1 marks a sample outside the environment: reading it as the last bin,
    or dropping it with a warning, would give a distance it does not have.
    """

    @pytest.mark.parametrize("query", list(_BIN_QUERIES))
    @pytest.mark.parametrize("bins", [[0, -1], [999]], ids=["outside", "unknown"])
    def test_non_bin_index_is_refused(self, line, query, bins):
        call, metric = _BIN_QUERIES[query]
        with pytest.raises(nx.NodeNotFound, match="are not bins of the environment"):
            call(line, bins, metric)

    @pytest.mark.parametrize("query", list(_BIN_QUERIES))
    @pytest.mark.parametrize(
        "bins",
        [np.array([0.0, 4.9]), np.array([True, False])],
        ids=["fractional", "boolean"],
    )
    def test_non_integer_bins_are_refused(self, line, query, bins):
        call, metric = _BIN_QUERIES[query]
        with pytest.raises(TypeError, match="must be integer bin indices"):
            call(line, bins, metric)


class TestReachableFrom:
    def test_geodesic_radius_is_round_and_bin_graph_radius_is_the_staircase(
        self, square
    ):
        for radius in (2.0, 4.5, 6.0):
            geodesic = square.reachable_from(bins=12, radius=radius, metric="geodesic")
            graph = square.reachable_from(bins=12, radius=radius, metric="bin_graph")
            np.testing.assert_array_equal(geodesic, _straight(square)[12] <= radius)
            np.testing.assert_array_equal(graph, _octile(square)[12] <= radius)

    def test_hops_count_edges(self, square):
        within = square.reachable_from(bins=12, radius=1, metric="hops")
        assert within.sum() == 9

    def test_geodesic_radius_stops_at_the_wall(self, walled):
        center = int(walled.bin_at(np.array([[47.5, 12.5]]))[0])
        across = int(walled.bin_at(np.array([[52.5, 12.5]]))[0])
        assert not walled.reachable_from(bins=center, radius=10.0, metric="geodesic")[
            across
        ]


class TestDistanceBetween:
    def test_points_are_measured_from_themselves(self, walled):
        # 6 cm right and 6 cm up; their bin centers are only 5 cm apart each way.
        points = np.array([[41.0, 3.0], [47.0, 9.0]])
        assert walled.distance_between(points=points) == pytest.approx(
            np.hypot(6.0, 6.0), rel=1e-12
        )
        bins = walled.bin_at(points)
        assert walled.distance_between(points=points, metric="bin_graph") == (
            pytest.approx(
                nx.shortest_path_length(
                    walled.connectivity, int(bins[0]), int(bins[1]), weight="distance"
                ),
                rel=1e-12,
            )
        )

    def test_the_same_either_way_round_near_a_wall(self, walled):
        rng = np.random.default_rng(3)
        for _ in range(30):
            a, b = rng.integers(0, walled.n_bins, 2)
            assert walled.distance_between(bins=[a, b]) == walled.distance_between(
                bins=[b, a]
            )


def _hexagons_in_a_box():
    samples = np.random.default_rng(0).uniform(0, 30, (40_000, 2))
    return Environment.from_samples(samples, bin_size=3.0, layout="hexagonal")


def _mesh_in_a_box():
    from neurospatial.layout import create_layout

    return Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 30, 30), point_spacing=3.0
        )
    )


# Convex rooms with nothing in them, on every kind of bins. A grid joins
# its diagonal neighbors or its face neighbors only.
_CONVEX = {
    f"{name}-{'diagonal' if diagonal else 'faces'}": (
        lambda mask=mask, size=size, diagonal=diagonal: _grid(
            mask, size, diagonal=diagonal
        )
    )
    for name, mask, size in [
        ("squares", np.ones((8, 8)), 2.0),
        ("rectangles", np.ones((10, 4)), (2.0, 5.0)),
        ("cubes", np.ones((5, 5, 5)), 2.0),
        ("boxes", np.ones((5, 4, 3)), (2.0, 3.0, 5.0)),
    ]
    for diagonal in (True, False)
} | {"hexagons": _hexagons_in_a_box, "triangles": _mesh_in_a_box}


@pytest.mark.parametrize("layout", list(_CONVEX))
def test_every_route_is_the_straight_line_in_a_convex_room(layout):
    """Nothing is in the way, so the shortest path is the straight line.

    Lines between bin centers run through cell corners and along cell
    edges; on every layout and connectivity they still pass.
    """
    env = _CONVEX[layout]()
    straight = _straight(env)
    exact = {"rtol": 1e-12, "atol": 1e-12}
    assert_allclose(geodesic_distance_matrix(env), straight, **exact)
    bins = np.random.default_rng(12).choice(env.n_bins, 12, replace=False)
    assert_allclose(
        pairwise_distances(env, bins), straight[np.ix_(bins, bins)], **exact
    )
    for source in bins[:4]:
        assert_allclose(distance_field(env, [int(source)]), straight[source], **exact)
        assert_allclose(env.distance_to(bins=[int(source)]), straight[source], **exact)
    for a, b in zip(bins[:6], bins[6:], strict=True):
        assert env.distance_between(bins=[int(a), int(b)]) == pytest.approx(
            straight[a, b], rel=1e-12
        )
    # A radius between two of the distances from the first bin, so that
    # rounding cannot move a bin across it.
    distances = np.unique(np.round(straight[bins[0]], 9))
    radius = float(distances[len(distances) // 2 : len(distances) // 2 + 2].mean())
    inside = straight[bins] <= radius
    for center, ball, expected in zip(
        bins, neighbors_within(env, bins, radius=radius), inside, strict=True
    ):
        np.testing.assert_array_equal(np.sort(ball), np.flatnonzero(expected))
        np.testing.assert_array_equal(
            env.reachable_from(bins=int(center), radius=radius, metric="geodesic"),
            expected,
        )


@pytest.mark.xfail(
    reason="With several sources a bin joins the field through a neighbor "
    "nearer on the bin graph, so it can keep a source that is farther in a "
    "straight line: (11, 1) reads 10.0 cm, its distance to (1, 1), though "
    "(5, 7) is 8.49 cm away with nothing between them."
)
def test_a_field_from_several_sources_is_the_distance_to_the_nearest():
    """In an open room every bin is the straight line from its nearest source."""
    room = _grid(np.ones((6, 6)), 2.0, diagonal=False)
    sources = room.bin_at(np.array([[1.0, 1.0], [5.0, 7.0]]))
    assert_allclose(
        distance_field(room, sources),
        _straight(room)[sources].min(axis=0),
        rtol=1e-12,
    )


def test_every_route_runs_along_the_centerline_of_a_track():
    """On a track the distance is along its centerline, whatever its width."""
    nodes = [(0.0, 0.0), (0.0, 40.0), (-30.0, 40.0), (30.0, 40.0), (-30.0, 0.0)]
    nodes.append((30.0, 0.0))
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
    graph = nx.Graph()
    for node, position in enumerate(nodes):
        graph.add_node(node, pos=position)
    for u, v in edges:
        graph.add_edge(
            u,
            v,
            distance=float(
                np.linalg.norm(np.subtract(*map(nodes.__getitem__, (u, v))))
            ),
        )
    track = Environment.from_graph(
        graph, [(0, 1), (1, 2), (2, 4), (1, 3), (3, 5)], bin_size=2.0, track_width=10.0
    )
    # Near the ends of the two outer arms, which hang down from the bar at
    # y = 40: up one arm, 60 cm along the bar, down the other.
    a, b = (int(i) for i in track.bin_at(np.array([[-30.0, 1.0], [30.0, 1.0]])))
    (xa, ya), (xb, yb) = track.bin_centers[a], track.bin_centers[b]
    assert (xa, xb) == (-30.0, 30.0)
    along = (40.0 - ya) + 60.0 + (40.0 - yb)
    assert along == pytest.approx(138.0)
    exact = pytest.approx(along, rel=1e-12)
    assert geodesic_distance_matrix(track)[a, b] == exact
    assert pairwise_distances(track, [a, b])[0, 1] == exact
    assert distance_field(track, [a])[b] == exact
    assert track.distance_to(bins=[a])[b] == exact
    assert track.distance_between(bins=[a, b]) == exact
    (ball,) = neighbors_within(track, [a], radius=along + 0.5)
    assert b in ball
    (ball,) = neighbors_within(track, [a], radius=along - 0.5)
    assert b not in ball
    assert distance_field(track, [a], metric="euclidean")[b] == pytest.approx(60.0)


class TestEachRouteNearAWall:
    """Every geodesic route of ``walled`` against the independent shortest paths.

    The routes are compared with ``through_centers`` (Shapely visibility and
    SciPy's Dijkstra over bin centers), not with each other.
    """

    @pytest.fixture(scope="class")
    def bins(self, walled):
        return np.random.default_rng(11).choice(walled.n_bins, 80, replace=False)

    def test_pairwise_distances(self, walled, through_walled_centers, bins):
        _assert_pair_distances(
            pairwise_distances(walled, bins),
            through_walled_centers[np.ix_(bins, bins)],
        )

    def test_distance_between_bins(self, walled, through_walled_centers, bins):
        first, second = bins[:40], bins[40:]
        between = np.array(
            [
                float(np.ravel(walled.distance_between(bins=[int(a), int(b)]))[0])
                for a in first
                for b in second
            ]
        )
        _assert_pair_distances(
            between, through_walled_centers[np.ix_(first, second)].ravel()
        )

    def test_distance_field(self, walled, through_walled_centers, bins):
        fields = np.stack([distance_field(walled, [int(b)]) for b in bins])
        _assert_field_lengths(fields, through_walled_centers[bins])

    def test_distance_field_to_the_nearest_of_several_sources(
        self, walled, through_walled_centers, bins
    ):
        sources = bins[:5]
        _assert_field_lengths(
            distance_field(walled, sources),
            through_walled_centers[sources].min(axis=0),
        )

    def test_distance_to(self, walled, through_walled_centers, bins):
        fields = np.stack([walled.distance_to(bins=[int(b)]) for b in bins])
        _assert_field_lengths(fields, through_walled_centers[bins])

    @pytest.mark.parametrize("radius", [24.0, 48.0, 75.0, 110.0])
    def test_neighbors_within_and_reachable_from(
        self, walled, through_walled_centers, bins, radius
    ):
        # A ball of the field holds every bin whose shortest path through
        # centers is within the radius even when grown greedily, and no bin
        # whose shortest path is beyond it. The short radii lie between the
        # straight length and the staircase length of a bin two (four)
        # across and one (two) up: 22.4 and 24.1 cm, 44.7 and 48.3 cm.
        surely = (1 + FIELD_EXCESS) * through_walled_centers[bins] <= radius
        possibly = through_walled_centers[bins] <= radius + 1e-9
        balls = neighbors_within(walled, bins, radius=radius)
        for center, ball, inside, allowed in zip(
            bins, balls, surely, possibly, strict=True
        ):
            member = np.zeros(walled.n_bins, dtype=bool)
            member[ball] = True
            assert np.all(member[inside])
            assert np.all(allowed[member])
            np.testing.assert_array_equal(
                walled.reachable_from(
                    bins=int(center), radius=radius, metric="geodesic"
                ),
                member,
            )


def test_geodesic_diameter_is_the_longest_symmetric_path(walled, two_rooms, square):
    from shapely.geometry import Point

    # In a disk the bin farthest out along the bin graph is not an end of the
    # longest any-angle path, so the search must look past its first guess.
    disk = Environment.from_polygon(Point(0, 0).buffer(20), bin_size=2.0)
    for env in (walled, two_rooms, square, disk):
        matrix = geodesic_distance_matrix(env)
        longest = matrix[np.isfinite(matrix)].max()
        assert env._geodesic_diameter("geodesic") == longest
        graph = geodesic_distance_matrix(env, metric="bin_graph")
        assert env._geodesic_diameter("bin_graph") == graph[np.isfinite(graph)].max()


@pytest.mark.parametrize("layout", ["disk", "walled"])
def test_geodesic_diameter_grows_its_fields_in_bounded_blocks(layout, monkeypatch):
    """However many bins could end the longest path, a block of fields is held."""
    from shapely.geometry import Point

    from neurospatial.ops import _paths, distance

    def build():
        if layout == "disk":
            return Environment.from_polygon(Point(0, 0).buffer(20), bin_size=2.0)
        divider = {"divider": LineString([(50, 0), (50, 70)])}
        return _grid(np.ones((20, 20)), 5.0, walls=divider)

    matrix = geodesic_distance_matrix(build())
    env = build()  # nothing cached
    block = 8
    monkeypatch.setattr(distance, "_DIJKSTRA_BLOCK_VALUES", block * env.n_bins)
    grown_at_once = []
    to_bins = _paths._OpenPaths.to_bins

    def recorded(self, bins, **kwargs):
        grown_at_once.append(len(bins))
        return to_bins(self, bins, **kwargs)

    monkeypatch.setattr(_paths._OpenPaths, "to_bins", recorded)
    assert env._geodesic_diameter("geodesic") == matrix[np.isfinite(matrix)].max()
    assert sum(grown_at_once) > block  # more fields than one block holds
    assert max(grown_at_once) <= block


def test_both_diameters_share_one_search_from_every_bin(monkeypatch):
    from neurospatial.ops import distance

    env = _grid(np.ones((5, 5)), 2.0)  # nothing cached
    searched = []
    blocks = distance._dijkstra_blocks

    def recorded(adjacency, sources, **kwargs):
        searched.append(len(sources))
        return blocks(adjacency, sources, **kwargs)

    monkeypatch.setattr(distance, "_dijkstra_blocks", recorded)
    # Corner to corner of the 5 x 5 grid of 2 cm bins: four diagonal steps
    # on the bin graph, the same straight line at any angle.
    corner_to_corner = pytest.approx(np.hypot(8.0, 8.0), rel=1e-12)
    assert env._geodesic_diameter("geodesic") == corner_to_corner
    assert env._geodesic_diameter("bin_graph") == corner_to_corner
    assert searched == [env.n_bins]


def test_distances_between_two_bin_sets_are_the_symmetric_matrix(walled):
    """Route tables grow one field per bin and keep the shorter direction."""
    from neurospatial.ops.distance import _between_bin_sets

    rng = np.random.default_rng(8)
    first = rng.choice(walled.n_bins, 60, replace=False)
    second = rng.choice(walled.n_bins, 45, replace=False)
    table = _between_bin_sets(walled, first, second, "test")
    matrix = geodesic_distance_matrix(walled)
    np.testing.assert_array_equal(table, matrix[np.ix_(first, second)])
    np.testing.assert_array_equal(
        table, _between_bin_sets(walled, second, first, "test").T
    )
