"""Queries take an explicit target: ``bins=``, ``points=`` or ``region=``.

A bin index and a coordinate cannot be told apart from their values, so each
kind has its own keyword. A point means the bin that contains it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from neurospatial import BinIndexOutOfRangeError, Environment, RegionNotFoundError
from neurospatial.layout.factories import create_layout
from neurospatial.regions import Regions


def _track() -> Environment:
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=20.0)
    graph.add_edge(1, 2, distance=20.0)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )


_LAYOUTS = {
    "grid": lambda: Environment.from_samples(
        np.random.default_rng(0).uniform(0, 30, (4000, 2)), bin_size=3.0
    ),
    "polygon": lambda: Environment.from_polygon(Point(15, 15).buffer(15), bin_size=3.0),
    "hexagonal": lambda: Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 30)]
        )
    ),
    "mesh": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]),
            point_spacing=4.0,
        )
    ),
    "track": _track,
    "polar": lambda: Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    ),
}


@pytest.fixture(params=sorted(_LAYOUTS))
def env(request) -> Environment:
    return _LAYOUTS[request.param]()


@pytest.fixture
def grid() -> Environment:
    return _LAYOUTS["grid"]()


def _goal(env: Environment) -> tuple[np.ndarray, int]:
    """A point just off a bin center, and that bin."""
    index = env.n_bins // 3
    return env.bin_centers[[index]] + 0.05, index


class TestPointEqualsItsBin:
    def test_distance_to_a_point_equals_distance_to_its_bin(self, env):
        point, index = _goal(env)
        from_point = env.distance_to(points=point)
        np.testing.assert_array_equal(from_point, env.distance_to(bins=index))
        assert from_point[index] == 0.0

    def test_point_region_gives_exactly_its_bin(self, env):
        point, index = _goal(env)
        regions = Regions()
        regions.add("reward", point=point[0])
        env = env.with_regions(regions)
        np.testing.assert_array_equal(env.bins_in_region("reward"), [index])
        np.testing.assert_array_equal(
            env.distance_to(region="reward"), env.distance_to(points=point)
        )

    def test_graph_queries_agree_for_a_point_and_its_bin(self, env):
        point, index = _goal(env)
        assert env.neighbors(points=point) == env.neighbors(bins=index)
        np.testing.assert_array_equal(
            env.reachable_from(points=point, radius=2),
            env.reachable_from(bins=index, radius=2),
        )
        for ring_point, ring_bin in zip(
            env.rings(points=point, hops=2), env.rings(bins=index, hops=2), strict=True
        ):
            np.testing.assert_array_equal(ring_point, ring_bin)
        last = env.bin_centers[[-1]] + 0.05
        assert env.path_between(points=np.vstack([point, last])) == env.path_between(
            bins=[index, env.n_bins - 1]
        )

    def test_distance_between_points_equals_distance_between_their_bins(self, grid):
        points = grid.bin_centers[[0, -1]] + 0.1
        assert grid.distance_between(points=points) == grid.distance_between(
            bins=grid.bin_at(points)
        )


class TestExactlyOneTarget:
    def test_no_target_names_all_three_keywords(self, grid):
        with pytest.raises(TypeError, match=r"bins=, points= or region=; got none"):
            grid.distance_to()

    def test_two_targets_name_both(self, grid):
        with pytest.raises(TypeError, match=r"got bins= and points="):
            grid.distance_to(bins=0, points=grid.bin_centers[:1])

    def test_queries_without_region_name_only_their_keywords(self, grid):
        with pytest.raises(TypeError, match=r"bins= or points=; got none"):
            grid.neighbors()

    def test_positional_target_is_refused(self, grid):
        with pytest.raises(TypeError, match="positional"):
            grid.distance_to([0])  # type: ignore[misc]


class TestBinsTakeIntegers:
    @pytest.mark.parametrize(
        "bins",
        [50.7, [50.7, 12.3], np.array([1.0, 2.0])],
        ids=["float", "floats", "array"],
    )
    def test_float_bins_point_to_points(self, grid, bins):
        """A coordinate passed as bins is refused, not truncated to bin indices."""
        with pytest.raises(TypeError, match=r"integer bin indices[\s\S]*Fix:.*points="):
            grid.distance_to(bins=bins)

    @pytest.mark.parametrize(
        "bins", [True, np.ones(3, dtype=bool)], ids=["bool", "mask"]
    )
    def test_boolean_bins_are_refused(self, grid, bins):
        with pytest.raises(TypeError, match=r"np\.flatnonzero\(mask\)[\s\S]*points="):
            grid.neighbors(bins=bins)

    def test_empty_bins_raise(self, grid):
        with pytest.raises(ValueError, match="empty bins="):
            grid.distance_to(bins=[])

    @pytest.mark.parametrize("index", [-1, 10_000])
    def test_out_of_range_bins_raise(self, grid, index):
        with pytest.raises(BinIndexOutOfRangeError):
            grid.distance_to(bins=index)

    def test_numpy_integer_scalar_is_one_bin(self, grid):
        np.testing.assert_array_equal(
            grid.distance_to(bins=np.int64(3)), grid.distance_to(bins=[3])
        )

    def test_bin_center_of_refuses_floats(self, grid):
        with pytest.raises(TypeError, match=r"bin_center_of\(\)[\s\S]*env\.bin_at"):
            grid.bin_center_of(np.array([1.5, 2.0]))


class TestFloatHintFollowsGeometry:
    """The fix for a float target names a lookup that works on the geometry."""

    def test_polar_query_hint_names_distance_and_angle(self):
        polar = _LAYOUTS["polar"]()
        with pytest.raises(TypeError, match=r"\(distance, angle\) rows with points="):
            polar.distance_to(bins=1.5)

    def test_polar_bin_center_of_points_to_a_polar_lookup(self):
        polar = _LAYOUTS["polar"]()
        with pytest.raises(TypeError) as caught:
            polar.bin_center_of(1.5)
        assert "(distance, angle) rows, use env.bin_at(points)" in str(caught.value)
        # The suggested lookup finds the bin containing each row, including
        # an angle given a whole turn outside the angle range.
        points = polar.bin_centers[[3, 3]] + [[0.01, 0.01], [0.01, 0.01 + 2 * np.pi]]
        np.testing.assert_array_equal(polar.bin_at(points), [3, 3])

    def test_cartesian_bin_center_of_points_to_bin_at(self, grid):
        with pytest.raises(TypeError, match=r"env\.bin_at\(points\)"):
            grid.bin_center_of(1.5)


class TestOutOfRangeHintFollowsGeometry:
    """An out-of-range bin index names a lookup that works on the geometry."""

    def test_polar_out_of_range_names_a_distance_angle_lookup(self):
        polar = _LAYOUTS["polar"]()
        with pytest.raises(BinIndexOutOfRangeError) as caught:
            polar.distance_to(bins=polar.n_bins)
        suggestion = "int(env.bin_at([[distance, angle]])[0])"
        assert suggestion in str(caught.value)
        distance, angle = polar.bin_centers[5]
        assert (
            eval(suggestion, {"env": polar, "distance": distance, "angle": angle}) == 5
        )

    def test_cartesian_out_of_range_names_bin_at(self, grid):
        with pytest.raises(BinIndexOutOfRangeError, match=r"env\.bin_at\(\[point\]\)"):
            grid.distance_to(bins=grid.n_bins)


class TestPointsAreValidated:
    def test_one_dimensional_points_raise_with_fix(self, grid):
        with pytest.raises(ValueError, match=r"shape \(2,\)[\s\S]*points=\[\[x, y\]\]"):
            grid.distance_to(points=[15.0, 15.0])

    def test_wrong_width_raises(self, grid):
        with pytest.raises(ValueError, match=r"expected \(n_points, 2\)"):
            grid.distance_to(points=[[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_points_raise(self, grid, bad):
        with pytest.raises(ValueError, match=r"NaN or inf[\s\S]*Fix:"):
            grid.distance_to(points=[[15.0, 15.0], [bad, 15.0]])

    def test_points_outside_raise_instead_of_minus_one(self, grid):
        with pytest.raises(ValueError, match=r"1 point\(s\) outside[\s\S]*row 1"):
            grid.distance_to(points=[[15.0, 15.0], [500.0, 500.0]])

    def test_non_numeric_points_raise(self, grid):
        with pytest.raises(TypeError, match="not numbers"):
            grid.distance_to(points=[["a", "b"]])

    def test_single_start_queries_take_one_point(self, grid):
        with pytest.raises(ValueError, match=r"takes one bin; got 2"):
            grid.reachable_from(points=grid.bin_centers[:2], radius=1)

    def test_two_point_queries_take_two(self, grid):
        with pytest.raises(
            ValueError,
            match=r"takes two bins, a start and an end; got 1[\s\S]*bins=\[start, end\]",
        ):
            grid.path_between(points=grid.bin_centers[:1])


class TestRegionTarget:
    def test_region_targets_are_the_region_bins(self, grid):
        regions = Regions()
        regions.add("zone", polygon=box(0, 0, 10, 30))
        grid = grid.with_regions(regions)
        np.testing.assert_array_equal(
            grid.distance_to(region="zone"),
            grid.distance_to(bins=grid.bins_in_region("zone")),
        )

    def test_region_covering_no_bins_raises(self, grid):
        regions = Regions()
        regions.add("far", point=(500.0, 500.0))
        grid = grid.with_regions(regions)
        with pytest.raises(ValueError, match=r"covers no bins[\s\S]*Fix:"):
            grid.distance_to(region="far")

    def test_region_object_is_refused_with_a_fix(self, grid):
        from neurospatial.regions import Region

        reward = Region("reward", "point", np.array([15.0, 15.0]))
        with pytest.raises(TypeError, match=r"region name in region=[\s\S]*Fix:"):
            grid.distance_to(region=reward)  # type: ignore[arg-type]

    def test_unknown_region_suggests_the_keyword(self, grid):
        with pytest.raises(RegionNotFoundError, match="region="):
            grid.distance_to(region="nest")


def test_distance_to_reward_workflow():
    """Distance from each tracked position to a reward port, by the graph."""
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 100, (5000, 2))  # cm
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    regions = Regions()
    regions.add("reward", point=(82.0, 77.0))
    env = env.with_regions(regions)

    to_reward = env.distance_to(region="reward")
    sample_bins = env.bin_at(positions)
    per_sample = to_reward[sample_bins]

    (reward_bin,) = env.bins_in_region("reward")
    assert to_reward[reward_bin] == 0.0
    # Each sample's distance is the graph distance from its bin to the reward bin.
    for row in (0, 1, 2):
        expected = env.distance_between(bins=[sample_bins[row], reward_bin])
        assert per_sample[row] == pytest.approx(expected)
    # The graph distance is never shorter than the straight line.
    straight = env.distance_to(region="reward", metric="euclidean")
    assert np.all(to_reward >= straight - 1e-9)
