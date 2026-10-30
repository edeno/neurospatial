"""Every region query covers the same bins, by one rule, on every layout.

A point region covers the bin that contains the point; a polygon region covers
the bins whose centers lie inside it, and a center exactly on its edge when
nudging it a hair north-east moves it inside.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment
from neurospatial.layout.factories import create_layout
from neurospatial.ops.mapping import regions_to_mask
from neurospatial.regions import Region, Regions


def _track() -> Environment:
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=20.0)
    graph.add_edge(1, 2, distance=20.0)
    return Environment.from_graph(
        graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
        bin_size=2.0,
        track_width=4.0,
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
    # Regions on a polar environment are read in (distance, angle) coordinates.
    "polar": lambda: Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    ),
}
# Layouts that subset() accepts: Cartesian bins with cell geometry.
_SUBSET_LAYOUTS = ["grid", "hexagonal", "mesh", "polygon"]


def _with_regions(layout: str) -> Environment:
    env = _LAYOUTS[layout]()
    centers = env.bin_centers
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    mid = (lo + hi) / 2.0
    regions = Regions()
    # A point just off a bin center, so the point region is not a bin center.
    regions.add("reward", point=centers[len(centers) // 3] + 0.1)
    # A box whose lower edges pass exactly through the lowest bin centers.
    regions.add("zone", polygon=box(lo[0], lo[1], mid[0], hi[1]))
    return env.with_regions(regions)


@pytest.fixture(params=sorted(_LAYOUTS))
def env_with_regions(request) -> Environment:
    return _with_regions(request.param)


@pytest.fixture(params=_SUBSET_LAYOUTS)
def subset_env(request) -> Environment:
    return _with_regions(request.param)


def _containing_bin(env: Environment, point: np.ndarray) -> int:
    """The bin containing ``point``, found without the environment's locator."""
    if env.coordinate_kind == "polar":
        # A polar bin is a rectangle in (distance, angle): find its cell by edges.
        cell_center = [
            (edges[i] + edges[i + 1]) / 2.0
            for edges, value in zip(env.grid_edges, point, strict=True)
            for i in [int(np.searchsorted(edges, value, side="right")) - 1]
        ]
        (index,) = np.flatnonzero(
            np.all(np.isclose(env.bin_centers, cell_center), axis=1)
        )
        return int(index)
    return int(env.bin_at(point[np.newaxis])[0])


def _consumers(env: Environment, name: str) -> dict:
    """The bins each region query reports for ``name``."""
    column = list(env.regions).index(name)
    membership = env.region_membership()
    return {
        "bins_in_region": env.bins_in_region(name),
        "region_mask": np.flatnonzero(env.region_mask(name)),
        "region_membership": np.flatnonzero(membership[:, column]),
        "regions_to_mask": np.flatnonzero(regions_to_mask(env, name)),
    }


@pytest.mark.parametrize("name", ["reward", "zone"])
def test_region_queries_agree_on_every_layout(env_with_regions, name):
    env = env_with_regions
    reported = _consumers(env, name)
    reference = reported.pop("bins_in_region")
    # Track centers lie on the zone's south edge, which the zone holds.
    assert reference.size > 0
    for consumer, bins in reported.items():
        np.testing.assert_array_equal(bins, reference, err_msg=consumer)


def test_polygon_region_matches_an_independent_shapely_oracle(env_with_regions):
    """Inside by shapely's contains, plus edge centers whose nudge is inside."""
    env = env_with_regions
    polygon = env.regions["zone"].data
    centers = shapely.points(env.bin_centers)
    lo, hi = env.bin_centers.min(axis=0), env.bin_centers.max(axis=0)
    # A nudge north-east, far smaller than the bins but resolvable here.
    nudge = 1e-6 * np.abs(np.r_[lo, hi]).max() * np.array([1.0, 0.75])
    on_edge = shapely.touches(polygon, centers)
    nudged = shapely.points(env.bin_centers + nudge)
    expected = np.flatnonzero(
        shapely.contains(polygon, centers)
        | (on_edge & shapely.contains(polygon, nudged))
    )
    np.testing.assert_array_equal(env.bins_in_region("zone"), expected)
    np.testing.assert_array_equal(
        np.flatnonzero(regions_to_mask(env, "zone")), expected
    )


@pytest.mark.parametrize("name", ["reward", "zone"])
def test_distance_to_region_is_zero_exactly_on_its_bins(env_with_regions, name):
    env = env_with_regions
    distances = env.distance_to(region=name, metric="geodesic")
    np.testing.assert_array_equal(
        np.flatnonzero(distances == 0.0), env.bins_in_region(name)
    )


def test_point_region_covers_exactly_the_bin_containing_it(env_with_regions):
    env = env_with_regions
    point = env.regions["reward"].data
    np.testing.assert_array_equal(
        env.bins_in_region("reward"), [_containing_bin(env, point)]
    )


def test_point_region_outside_the_environment_covers_no_bin():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 30, (4000, 2)), bin_size=3.0
    )
    regions = Regions()
    regions.add("far", point=(500.0, 500.0))
    env = env.with_regions(regions)
    assert env.bins_in_region("far").size == 0
    assert not env.region_mask("far").any()
    assert not env.region_membership().any()


def test_box_through_centers_holds_its_south_and_west_edges():
    """Centers on a box's south and west edges are in it; north and east not."""
    edges = np.arange(0.0, 11.0, 2.0)  # centers 1, 3, 5, 7, 9
    env = Environment.from_grid_mask(np.ones((5, 5), dtype=bool), (edges, edges))
    regions = Regions()
    regions.add("zone", polygon=box(3.0, 3.0, 7.0, 7.0))
    env = env.with_regions(regions)

    np.testing.assert_array_equal(
        env.bin_centers[env.bins_in_region("zone")],
        [[3.0, 3.0], [3.0, 5.0], [5.0, 3.0], [5.0, 5.0]],
    )


def test_boxes_that_share_edges_partition_the_bins():
    """Four boxes meeting on a row and column of centers cover each bin once."""
    edges = np.arange(0.0, 11.0, 2.0)  # centers 1, 3, 5, 7, 9
    env = Environment.from_grid_mask(np.ones((5, 5), dtype=bool), (edges, edges))
    regions = Regions()
    corners = {"sw": (0.0, 0.0), "se": (5.0, 0.0), "nw": (0.0, 5.0), "ne": (5.0, 5.0)}
    for name, (x0, y0) in corners.items():
        regions.add(name, polygon=box(x0, y0, x0 + 5.0, y0 + 5.0))
    env = env.with_regions(regions)
    membership = env.region_membership()
    np.testing.assert_array_equal(membership.sum(axis=1), 1)
    # The center (5, 5) is on all four boxes: it goes to the north-east one.
    (center,) = np.flatnonzero((env.bin_centers == 5.0).all(axis=1))
    assert membership[center].tolist() == [False, False, False, True]


@pytest.mark.parametrize("layout", ["polygon", "mesh"])
def test_arena_outline_region_keeps_every_bin(layout):
    """A region drawn as the arena's outline covers the whole environment."""
    env = _LAYOUTS[layout]()
    regions = Regions()
    regions.add("arena", polygon=env.arena.polygon)
    env = env.with_regions(regions)
    assert env.region_mask("arena").all()


def test_outline_region_through_edge_centers_covers_its_area():
    """Edge bins of a region drawn along the outermost centers go to one side.

    Bins centered on the region's outline are half outside it. The region
    holds those on its south and west edges, so it covers as many bins as fit
    in its area.
    """
    lattice = np.arange(0.0, 31.0, 1.0)
    samples = np.stack(np.meshgrid(lattice, lattice), axis=-1).reshape(-1, 2)
    env = Environment.from_samples(samples, bin_size=2.0)
    np.testing.assert_array_equal(np.unique(env.bin_centers), np.arange(0, 31, 2))
    regions = Regions()
    regions.add("arena", polygon=box(0.0, 0.0, 30.0, 30.0))
    env = env.with_regions(regions)
    covered = env.bin_centers[env.bins_in_region("arena")]
    assert len(covered) * 2.0**2 == 30.0**2
    assert covered.min() == 0.0
    assert covered.max() == 28.0


def test_subset_by_region_keeps_exactly_the_region_bins(subset_env):
    env = subset_env
    for name in ("reward", "zone"):
        sub = env.subset(region_names=[name])
        np.testing.assert_array_equal(
            sub.bin_centers, env.bin_centers[env.bins_in_region(name)]
        )


def test_subset_by_polygon_matches_the_polygon_region(subset_env):
    env = subset_env
    polygon = env.regions["zone"].data
    np.testing.assert_array_equal(
        env.subset(polygon=polygon).bin_centers,
        env.subset(region_names=["zone"]).bin_centers,
    )


def test_subset_uses_the_edge_rule():
    """subset(region_names=) and subset(polygon=) split edge centers alike."""
    edges = np.arange(0.0, 11.0, 2.0)  # centers 1, 3, 5, 7, 9
    env = Environment.from_grid_mask(np.ones((5, 5), dtype=bool), (edges, edges))
    zone = box(3.0, 3.0, 7.0, 7.0)  # edges through the centers at 3 and 7
    regions = Regions()
    regions.add("zone", polygon=zone)
    env = env.with_regions(regions)
    expected = env.bin_centers[env.bins_in_region("zone")]
    for sub in (env.subset(region_names=["zone"]), env.subset(polygon=zone)):
        np.testing.assert_array_equal(sub.bin_centers, expected)
    assert len(expected) == 4


def test_polygon_region_in_a_1d_environment_raises_with_fix():
    env = Environment.from_samples(np.linspace(0, 20, 50)[:, None], bin_size=2.0)
    zone = Region("zone", "polygon", box(0, 0, 5, 5))
    with pytest.raises(ValueError, match=r"needs 2-D bins[\s\S]*Fix:"):
        env.region_mask(zone)


def test_point_region_with_wrong_dimension_raises_with_fix():
    env = Environment.from_samples(np.linspace(0, 20, 50)[:, None], bin_size=2.0)
    reward = Region("reward", "point", np.array([5.0, 5.0]))
    with pytest.raises(ValueError, match=r"has 2 coordinates, but the bins have 1"):
        env.region_mask(reward)
