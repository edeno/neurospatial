"""Every region query covers the same bins, by one rule, on every layout.

A point region covers the bin that contains the point; a polygon region covers
the bins whose centers lie inside it, with ``include_boundary`` deciding a
center on the edge.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from neurospatial import Environment
from neurospatial.layout.factories import create_layout
from neurospatial.ops.binning import regions_to_mask
from neurospatial.regions import Region, Regions


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
}


@pytest.fixture(params=sorted(_LAYOUTS))
def env_with_regions(request) -> Environment:
    env = _LAYOUTS[request.param]()
    centers = env.bin_centers
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    mid = (lo + hi) / 2.0
    regions = Regions()
    # A point just off a bin center, so the point region is not a bin center.
    regions.add("reward", point=centers[len(centers) // 3] + 0.1)
    # A box whose lower edges pass exactly through the lowest bin centers.
    regions.add("zone", polygon=box(lo[0], lo[1], mid[0], hi[1]))
    return env.with_regions(regions)


def _consumers(env: Environment, name: str, include_boundary: bool) -> dict:
    """The bins each region query reports for ``name``."""
    column = list(env.regions).index(name)
    membership = env.region_membership(include_boundary=include_boundary)
    return {
        "bins_in_region": env.bins_in_region(name, include_boundary=include_boundary),
        "region_mask": np.flatnonzero(
            env.region_mask(name, include_boundary=include_boundary)
        ),
        "region_membership": np.flatnonzero(membership[:, column]),
        "regions_to_mask": np.flatnonzero(
            regions_to_mask(env, name, include_boundary=include_boundary)
        ),
    }


@pytest.mark.parametrize("include_boundary", [True, False])
@pytest.mark.parametrize("name", ["reward", "zone"])
def test_region_queries_agree_on_every_layout(env_with_regions, name, include_boundary):
    env = env_with_regions
    reported = _consumers(env, name, include_boundary)
    reference = reported.pop("bins_in_region")
    # Track centers all lie on the zone's lower edge, so only its interior is empty.
    assert reference.size > 0 or not include_boundary
    for consumer, bins in reported.items():
        np.testing.assert_array_equal(bins, reference, err_msg=consumer)


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
        env.bins_in_region("reward"), env.bin_at(point[np.newaxis])
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


def test_polygon_edge_through_centers_follows_include_boundary():
    """A center exactly on the polygon's edge counts only with include_boundary."""
    edges = np.arange(0.0, 11.0, 2.0)  # centers 1, 3, 5, 7, 9
    env = Environment.from_grid_mask(np.ones((5, 5), dtype=bool), (edges, edges))
    regions = Regions()
    regions.add("zone", polygon=box(3.0, 3.0, 7.0, 7.0))
    env = env.with_regions(regions)

    with_edge = env.bin_centers[env.bins_in_region("zone")]
    interior = env.bin_centers[env.bins_in_region("zone", include_boundary=False)]

    assert len(with_edge) == 9  # centers 3, 5, 7 in x and y
    np.testing.assert_array_equal(interior, [[5.0, 5.0]])


def test_subset_by_region_keeps_exactly_the_region_bins(env_with_regions):
    env = env_with_regions
    if env.is_linearized_track:
        pytest.skip("subset is a Cartesian-grid operation")
    for name in ("reward", "zone"):
        sub = env.subset(region_names=[name])
        np.testing.assert_array_equal(
            sub.bin_centers, env.bin_centers[env.bins_in_region(name)]
        )


def test_subset_by_polygon_matches_the_polygon_region(env_with_regions):
    env = env_with_regions
    if env.is_linearized_track:
        pytest.skip("subset is a Cartesian-grid operation")
    polygon = env.regions["zone"].data
    np.testing.assert_array_equal(
        env.subset(polygon=polygon).bin_centers,
        env.subset(region_names=["zone"]).bin_centers,
    )


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
