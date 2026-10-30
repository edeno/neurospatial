"""Tests for Honeycomb Maze implementation.

The published apparatus (Wood et al. 2018, doi:10.1038/nature25433) has 37
tessellated hexagonal platforms, 11.5 cm along each side, 1 cm apart, in a
hexagonal configuration 145.5 cm across.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
import shapely

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.honeycomb import HoneycombDims, make_honeycomb_maze

N_PLATFORMS = 37


@pytest.fixture(scope="module")
def maze():
    return make_honeycomb_maze()


@pytest.fixture(scope="module")
def track_graph(maze):
    return maze.env_track._binning.linearization.graph


@pytest.fixture(scope="module")
def platforms(maze):
    return [maze.env_2d.regions[f"platform_{i}"].data for i in range(N_PLATFORMS)]


class TestPublishedGeometry:
    def test_default_dims(self):
        dims = HoneycombDims()
        assert dims.platform_side == 11.5
        assert dims.gap == 1.0
        assert dims.n_rings == 3
        assert dims.spacing == pytest.approx(11.5 * np.sqrt(3) + 1.0)

    def test_thirty_seven_platforms_of_published_side(self, platforms):
        assert len(platforms) == N_PLATFORMS
        for platform in platforms:
            sides = np.hypot(*np.diff(np.asarray(platform.exterior.coords), axis=0).T)
            np.testing.assert_allclose(sides, 11.5)

    def test_neighbours_face_across_a_one_cm_gap(self, platforms, track_graph):
        # Flat sides face each other: the nearest points of two neighbouring
        # platforms are a whole side apart in parallel, 1 cm away.
        for u, v in track_graph.edges():
            a = platforms[int(u.split("_")[1])]
            b = platforms[int(v.split("_")[1])]
            assert a.distance(b) == pytest.approx(1.0)
            shared = a.buffer(0.51).intersection(b.buffer(0.51))
            assert shared.length > 2 * 11.5 * 0.9  # a full side, not a corner

    def test_platforms_do_not_overlap(self, platforms):
        union = shapely.union_all(platforms)
        assert union.area == pytest.approx(sum(p.area for p in platforms))

    def test_total_diameter_is_published(self, platforms):
        # 145.5 cm published; seven platforms flat to flat give 145.43 cm.
        x0, _, x1, _ = shapely.union_all(platforms).bounds
        assert x1 - x0 == pytest.approx(145.5, abs=0.1)

    def test_centred_on_origin(self, platforms):
        assert np.allclose(platforms[0].centroid.coords[0], (0.0, 0.0))
        x0, y0, x1, y1 = shapely.union_all(platforms).bounds
        assert (x0 + x1) / 2 == pytest.approx(0.0, abs=1e-9)
        assert (y0 + y1) / 2 == pytest.approx(0.0, abs=1e-9)

    def test_track_graph_is_the_hexagonal_lattice(self, track_graph):
        assert track_graph.number_of_nodes() == N_PLATFORMS
        assert track_graph.number_of_edges() == 90
        assert Counter(dict(track_graph.degree()).values()) == {6: 19, 4: 12, 3: 6}
        assert track_graph.degree("platform_0") == 6
        assert nx.is_connected(track_graph)

    def test_custom_rings_and_side(self):
        dims = HoneycombDims(platform_side=10.0, gap=2.0, n_rings=2)
        maze = make_honeycomb_maze(dims=dims, bin_size=3.0)
        names = [r for r in maze.env_2d.regions if r.startswith("platform_")]
        assert len(names) == 19
        second = maze.env_2d.regions["platform_1"].data.centroid
        assert np.hypot(second.x, second.y) == pytest.approx(10.0 * np.sqrt(3) + 2.0)


class TestFootprint:
    def test_floor_is_one_piece_without_holes(self, maze):
        floor = maze.env_2d.arena.polygon
        assert floor.geom_type == "Polygon"
        assert len(floor.interiors) == 0

    def test_every_track_edge_lies_inside_the_maze(self, maze, track_graph):
        env = maze.env_2d
        for u, v in track_graph.edges():
            p = np.asarray(track_graph.nodes[u]["pos"])
            q = np.asarray(track_graph.nodes[v]["pos"])
            points = p + np.linspace(0.0, 1.0, 200)[:, None] * (q - p)
            assert env.arena.contains(points).all(), (u, v)
            assert (env.bin_at(points) >= 0).all(), (u, v)

    def test_floor_is_platforms_plus_gaps(self, maze, platforms):
        # Each platform grown by half a gap, keeping its corners: the floor
        # adds only the 1 cm gaps (narrower than a bin) and a half-gap rim.
        floor = maze.env_2d.arena.polygon
        grown = shapely.union_all(
            [p.buffer(0.5 + 1e-6, join_style="mitre") for p in platforms]
        )
        assert floor.difference(grown).area == pytest.approx(0.0, abs=1e-6)
        platform_area = sum(p.area for p in platforms)
        assert platform_area / floor.area == pytest.approx(0.907, abs=0.001)

    def test_most_bins_lie_on_platforms(self, maze, platforms):
        centers = maze.env_2d.bin_centers
        on_platform = shapely.contains_xy(
            shapely.union_all(platforms), centers[:, 0], centers[:, 1]
        )
        assert on_platform.mean() > 0.88
        # The rest are in a gap or the rim: at most half a gap from a
        # platform's side, or half a gap / cos(30 deg) where three gaps meet.
        off = shapely.points(centers[~on_platform])
        distances = shapely.distance(shapely.union_all(platforms), off)
        assert distances.max() <= 0.5 / np.cos(np.pi / 6) + 1e-6


class TestRegions:
    def test_platform_regions_are_the_platforms(self, maze, platforms):
        assert list(maze.env_2d.regions) == [
            f"platform_{i}" for i in range(N_PLATFORMS)
        ]
        for i, platform in enumerate(platforms):
            assert maze.env_2d.regions[f"platform_{i}"].kind == "polygon"
            assert len(maze.env_2d.bins_in_region(f"platform_{i}")) > 0
            assert platform.area == pytest.approx(1.5 * np.sqrt(3) * 11.5**2)

    def test_no_apparatus_landmarks(self, maze):
        # Wood et al. (2018) fix no goal, start, reward port or cue on the
        # apparatus: the schedule chooses the goal and start platforms.
        assert len(maze.env_2d.landmarks) == 0
        assert len(maze.env_track.landmarks) == 0


class TestEnvironments:
    def test_returns_maze_environments(self, maze):
        assert isinstance(maze, MazeEnvironments)
        assert maze.env_2d.units == "cm"
        assert maze.env_track.units == "cm"
        assert maze.env_track.is_linearized_track

    def test_include_track_false_no_env_track(self):
        assert make_honeycomb_maze(include_track=False).env_track is None

    def test_custom_bin_size(self):
        fine = make_honeycomb_maze(bin_size=1.0, include_track=False)
        coarse = make_honeycomb_maze(bin_size=5.0, include_track=False)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins


class TestLargeMazes:
    def test_twelve_rings_build_their_track(self):
        """A large honeycomb linearizes without recursing once per edge."""
        dims = HoneycombDims(n_rings=12)
        maze = make_honeycomb_maze(dims, bin_size=10.0)
        graph = maze.env_track.binning.linearization.graph
        n_platforms = 1 + 3 * 12 * 13
        assert graph.number_of_nodes() == n_platforms
        # Each platform pair facing across a side is one edge.
        assert graph.number_of_edges() == 3 * 12 * (3 * 12 + 1)

    def test_track_is_linearized_breadth_first_from_the_centre(self, maze, track_graph):
        edge_order = maze.env_track.binning.linearization.edge_order
        assert list(edge_order) == list(nx.edge_bfs(track_graph, "platform_0"))
