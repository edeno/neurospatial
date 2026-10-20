"""Tests for Hamlet Maze implementation.

The apparatus (Crouzier et al. 2018, doi:10.1016/j.nlm.2018.02.014) is 1.6 m
across: a central pentagonal ring of alleys, five alleys radiating from it,
each splitting into two arms that lead to two different houses (Wijnen et
al. 2024, doi:10.1007/s00429-024-02771-x). Alley positions and widths are
estimates; these tests pin the topology, the published diameter and that the
floor is exactly the alleys and houses.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
import shapely

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

HOUSES = [f"house_{k}" for k in range(5)]
RING = [f"ring_{k}" for k in range(5)]
SPLITS = [f"split_{k}" for k in range(5)]


@pytest.fixture(scope="module")
def maze():
    return make_hamlet_maze()


@pytest.fixture(scope="module")
def track_graph(maze):
    return maze.env_track._binning.linearization.graph


def _pos(graph, node):
    return np.asarray(graph.nodes[node]["pos"])


def _segments(graph):
    return shapely.union_all(
        [shapely.LineString([_pos(graph, u), _pos(graph, v)]) for u, v in graph.edges()]
    )


class TestTopology:
    def test_node_and_edge_counts(self, track_graph):
        assert set(track_graph) == set(HOUSES + RING + SPLITS)
        assert track_graph.number_of_edges() == 20
        assert nx.is_connected(track_graph)

    def test_pentagonal_ring_of_five_alleys(self, track_graph):
        ring = track_graph.subgraph(RING)
        assert ring.number_of_edges() == 5
        assert all(d == 2 for _, d in ring.degree())
        assert [len(cycle) for cycle in nx.cycle_basis(ring)] == [5]

    def test_each_radial_alley_splits_into_arms_to_two_houses(self, track_graph):
        for k in range(5):
            assert set(track_graph[f"ring_{k}"]) - set(RING) == {f"split_{k}"}
            assert set(track_graph[f"split_{k}"]) == {
                f"ring_{k}",
                f"house_{k}",
                f"house_{(k + 1) % 5}",
            }

    def test_degrees(self, track_graph):
        degrees = dict(track_graph.degree())
        assert Counter(degrees.values()) == {3: 10, 2: 5}
        assert all(degrees[h] == 2 for h in HOUSES)

    def test_no_central_hub(self, track_graph):
        positions = np.array([_pos(track_graph, n) for n in track_graph])
        assert np.linalg.norm(positions, axis=1).min() > 15.0


class TestGeometry:
    def test_default_dims(self):
        dims = HamletDims()
        assert (dims.ring_radius, dims.split_radius, dims.house_radius) == (
            22.0,
            43.0,
            70.0,
        )
        assert dims.corridor_width == 9.0
        assert dims.house_size == 16.0
        assert dims.n_arms == 5

    def test_diameter_is_about_one_point_six_metres(self, maze):
        floor = maze.env_2d.arena.polygon
        radii = np.hypot(*np.asarray(floor.exterior.coords).T)
        assert 2 * radii.max() == pytest.approx(160.0, abs=5.0)

    def test_centred_with_houses_at_star_points(self, track_graph):
        for k, name in enumerate(HOUSES):
            angle = np.pi / 2 + 2 * np.pi * k / 5
            expected = 70.0 * np.array([np.cos(angle), np.sin(angle)])
            np.testing.assert_allclose(_pos(track_graph, name), expected, atol=1e-9)
        centroid = np.mean([_pos(track_graph, n) for n in track_graph], axis=0)
        np.testing.assert_allclose(centroid, 0.0, atol=1e-9)

    def test_alleys_lie_between_houses(self, track_graph):
        for k in range(5):
            alley = np.arctan2(*_pos(track_graph, f"split_{k}")[::-1])
            expected = np.pi / 2 + 2 * np.pi * (k + 0.5) / 5
            assert np.angle(np.exp(1j * (alley - expected))) == pytest.approx(0.0)


class TestFootprint:
    def test_every_track_edge_lies_inside_the_maze(self, maze, track_graph):
        env = maze.env_2d
        for u, v in track_graph.edges():
            p, q = _pos(track_graph, u), _pos(track_graph, v)
            points = p + np.linspace(0.0, 1.0, 200)[:, None] * (q - p)
            assert env.arena.contains(points).all(), (u, v)
            assert (env.bin_at(points) >= 0).all(), (u, v)

    def test_floor_is_only_alleys_and_houses(self, maze, track_graph):
        floor = maze.env_2d.arena.polygon
        houses = shapely.union_all([maze.env_2d.regions[h].data for h in HOUSES])
        alleys = _segments(track_graph).buffer(4.5)
        assert floor.difference(alleys.union(houses)).area == pytest.approx(
            0.0, abs=1e-6
        )
        centers = maze.env_2d.bin_centers
        on_alley = (
            shapely.distance(_segments(track_graph), shapely.points(centers))
            <= 4.5 + 1e-9
        )
        in_house = shapely.contains_xy(houses, centers[:, 0], centers[:, 1])
        assert (on_alley | in_house).all()

    def test_alleys_keep_their_full_width(self, maze, track_graph):
        # The floor covers the whole 9 cm band around every alley centerline,
        # junctions included.
        floor = maze.env_2d.arena.polygon
        band = _segments(track_graph).buffer(4.5 - 1e-6, cap_style="flat")
        assert band.difference(floor).area == pytest.approx(0.0, abs=1e-6)

    def test_blocks_are_solid(self, maze, track_graph):
        # Six blocks: the center of the ring, and one between each house and
        # the ring. Their centers are not floor and no bin lies in them.
        floor = maze.env_2d.arena.polygon
        assert floor.geom_type == "Polygon"
        assert len(floor.interiors) == 6
        ring = np.array([_pos(track_graph, n) for n in RING])
        block_centers = [ring.mean(axis=0)]
        for k in range(5):
            face = [
                ring[k - 1],
                _pos(track_graph, f"split_{(k - 1) % 5}"),
                _pos(track_graph, f"house_{k}"),
                _pos(track_graph, f"split_{k}"),
                ring[k],
            ]
            block_centers.append(shapely.Polygon(face).centroid.coords[0])
        assert not maze.env_2d.arena.contains(np.array(block_centers)).any()


class TestRegions:
    def test_region_names(self, maze):
        assert list(maze.env_2d.regions) == HOUSES + RING + SPLITS

    def test_houses_are_square_goal_boxes(self, maze, track_graph):
        for name in HOUSES:
            region = maze.env_2d.regions[name]
            assert region.kind == "polygon"
            assert region.data.area == pytest.approx(16.0**2)
            assert region.data.contains(shapely.Point(_pos(track_graph, name)))

    def test_junction_points_on_the_track_nodes(self, maze, track_graph):
        for name in RING + SPLITS:
            region = maze.env_2d.regions[name]
            assert region.kind == "point"
            np.testing.assert_allclose(region.data, _pos(track_graph, name))


class TestEnvironments:
    def test_returns_maze_environments(self, maze):
        assert isinstance(maze, MazeEnvironments)
        assert maze.env_2d.units == "cm"
        assert maze.env_track.units == "cm"
        assert maze.env_track.is_linearized_track
        assert maze.env_track.arena.width == pytest.approx(9.0)

    def test_include_track_false_no_env_track(self):
        assert make_hamlet_maze(include_track=False).env_track is None

    def test_custom_dims_scale_the_maze(self):
        dims = HamletDims(split_radius=50.0, house_radius=90.0, corridor_width=10.0)
        maze = make_hamlet_maze(dims=dims, bin_size=3.0)
        graph = maze.env_track._binning.linearization.graph
        assert np.linalg.norm(_pos(graph, "house_0")) == pytest.approx(90.0)
        assert np.linalg.norm(_pos(graph, "split_0")) == pytest.approx(50.0)

    def test_custom_bin_size(self):
        fine = make_hamlet_maze(bin_size=1.0, include_track=False)
        coarse = make_hamlet_maze(bin_size=5.0, include_track=False)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins
