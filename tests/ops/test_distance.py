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
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.arena import OpenArena
from neurospatial.ops.distance import (
    distance_field,
    euclidean_distance_matrix,
    geodesic_distance_matrix,
    neighbors_within,
    pairwise_distances,
)


def _grid(mask, size, walls=None):
    mask = np.asarray(mask, dtype=bool)
    edges = [np.arange(n + 1) * size for n in mask.shape]
    arena = None
    if walls is not None:
        arena = OpenArena(
            box(0, 0, mask.shape[0] * size, mask.shape[1] * size), walls=walls
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Environment.from_grid_mask(
            mask, grid_edges=tuple(edges), units="cm", arena=arena
        )


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

    def test_geodesic_goes_around_the_wall_end(self, walled):
        source = int(walled.bin_at(np.array([[37.5, 12.5]]))[0])
        target = int(walled.bin_at(np.array([[62.5, 12.5]]))[0])
        tip = np.array([50.0, 70.0])
        a, b = walled.bin_centers[source], walled.bin_centers[target]
        taut = np.linalg.norm(a - tip) + np.linalg.norm(b - tip)
        field = distance_field(walled, [source])
        assert taut <= field[target] <= taut + 2 * np.hypot(5.0, 5.0)
        assert (
            field[target] < distance_field(walled, [source], metric="bin_graph")[target]
        )
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

    def test_geodesic_matrix_near_a_wall_is_symmetric_and_bounded(self, walled):
        matrix = geodesic_distance_matrix(walled)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)
        assert np.all(matrix >= _straight(walled) - 1e-9)
        assert np.all(
            matrix <= geodesic_distance_matrix(walled, metric="bin_graph") + 1e-9
        )

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
        centers = np.array([0, 150, 155, 260, 399])
        matrix = geodesic_distance_matrix(walled)
        for radius in (7.5, 20.0, 40.0):
            balls = neighbors_within(walled, centers, radius=radius)
            for center, ball in zip(centers, balls, strict=True):
                np.testing.assert_array_equal(
                    np.sort(ball), np.flatnonzero(matrix[center] <= radius)
                )

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


def test_both_diameters_share_one_search_from_every_bin(square, monkeypatch):
    from neurospatial.ops import distance

    env = _grid(np.ones((5, 5)), 2.0)  # nothing cached
    searched = []
    blocks = distance._dijkstra_blocks

    def recorded(adjacency, sources, **kwargs):
        searched.append(len(sources))
        return blocks(adjacency, sources, **kwargs)

    monkeypatch.setattr(distance, "_dijkstra_blocks", recorded)
    assert env._geodesic_diameter("geodesic") == square._geodesic_diameter("geodesic")
    assert env._geodesic_diameter("bin_graph") == square._geodesic_diameter("bin_graph")
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
