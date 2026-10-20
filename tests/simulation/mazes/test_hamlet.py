"""Tests for Hamlet Maze implementation.

The apparatus (Crouzier et al. 2018, doi:10.1016/j.nlm.2018.02.014) is 1.6 m
across: a central agora inside a pentagonal ring of alleys, five alleys
radiating from it, each splitting into two arms that lead to two different
houses (Wijnen et al. 2024, doi:10.1007/s00429-024-02771-x). The agora is the
start box and the five houses are the Hide, Drink, Eat, Interact and Run
houses, clockwise from the top, 28 x 16 cm each (Crouzier et al. 2020,
doi:10.1016/j.euroneuro.2020.08.010; Couly 2018, tel-01925318). Alley
positions and widths are estimates; these tests pin the topology, the
published diameter, house sizes and order, and that the floor is exactly the
agora, alleys and houses.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
import shapely

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

# Counterclockwise from the top.
HOUSES = ["hide_house", "run_house", "interact_house", "eat_house", "drink_house"]
RING = [f"ring_{k}" for k in range(5)]
AGORA = ["agora", "agora_north_exit"]
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
        assert set(track_graph) == set(AGORA + HOUSES + RING + SPLITS)
        assert track_graph.number_of_edges() == 23
        assert nx.is_connected(track_graph)

    def test_pentagonal_ring_of_alleys_around_the_agora(self, track_graph):
        # The top exit of the agora joins the ring alley between corners 4
        # and 0.
        ring = track_graph.subgraph([*RING, "agora_north_exit"])
        assert ring.number_of_edges() == 6
        assert all(d == 2 for _, d in ring.degree())
        assert [len(cycle) for cycle in nx.cycle_basis(ring)] == [6]
        assert set(track_graph["agora_north_exit"]) == {"agora", "ring_4", "ring_0"}

    def test_agora_has_two_exits_onto_the_ring(self, track_graph):
        assert set(track_graph["agora"]) == {"agora_north_exit", "ring_2"}
        # One through the side facing the Hide house, one through the
        # opposite corner.
        np.testing.assert_allclose(
            _pos(track_graph, "agora_north_exit"),
            (_pos(track_graph, "ring_4") + _pos(track_graph, "ring_0")) / 2,
        )
        for node in track_graph["agora"]:
            assert _pos(track_graph, node)[0] == pytest.approx(0.0, abs=1e-9)

    def test_each_radial_alley_splits_into_arms_to_two_houses(self, track_graph):
        for k in range(5):
            assert set(track_graph[f"ring_{k}"]) - set(RING + AGORA) == {f"split_{k}"}
            assert set(track_graph[f"split_{k}"]) == {
                f"ring_{k}",
                HOUSES[k],
                HOUSES[(k + 1) % 5],
            }

    def test_degrees(self, track_graph):
        degrees = dict(track_graph.degree())
        assert Counter(degrees.values()) == {3: 10, 2: 6, 4: 1}
        assert all(degrees[h] == 2 for h in HOUSES)
        assert degrees["agora"] == 2
        assert degrees["ring_2"] == 4

    def test_only_the_agora_is_inside_the_ring(self, track_graph):
        inner = [n for n in track_graph if np.linalg.norm(_pos(track_graph, n)) < 15.0]
        assert inner == ["agora"]


class TestGeometry:
    def test_default_dims(self):
        dims = HamletDims()
        assert (dims.ring_radius, dims.split_radius, dims.house_radius) == (
            22.0,
            43.0,
            70.0,
        )
        assert dims.corridor_width == 8.0
        # Couly (2018): houses "28 × 16 cm".
        assert (dims.house_width, dims.house_depth) == (28.0, 16.0)
        assert dims.agora_wall_thickness == 3.0
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
        centroid = np.mean([_pos(track_graph, n) for n in HOUSES], axis=0)
        np.testing.assert_allclose(centroid, 0.0, atol=1e-9)
        np.testing.assert_allclose(_pos(track_graph, "agora"), 0.0)

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

    def test_floor_is_only_agora_alleys_and_houses(self, maze, track_graph):
        floor = maze.env_2d.arena.polygon
        houses = shapely.union_all(
            [maze.env_2d.regions[h].data for h in [*HOUSES, "agora"]]
        )
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
        # The floor covers the whole alley-wide band around every centerline,
        # junctions included.
        floor = maze.env_2d.arena.polygon
        half = HamletDims().corridor_width / 2
        band = _segments(track_graph).buffer(half - 1e-6, cap_style="flat")
        assert band.difference(floor).area == pytest.approx(0.0, abs=1e-6)

    def test_blocks_are_solid(self, maze, track_graph):
        # Five blocks, one between each house and the ring, and the agora's
        # wall in two pieces between its exits. Block centers are not floor.
        floor = maze.env_2d.arena.polygon
        assert floor.geom_type == "Polygon"
        assert len(floor.interiors) == 7
        ring = np.array([_pos(track_graph, n) for n in RING])
        block_centers = []
        for k in range(5):
            face = [
                ring[k - 1],
                _pos(track_graph, f"split_{(k - 1) % 5}"),
                _pos(track_graph, HOUSES[k]),
                _pos(track_graph, f"split_{k}"),
                ring[k],
            ]
            block_centers.append(shapely.Polygon(face).centroid.coords[0])
        assert not maze.env_2d.arena.contains(np.array(block_centers)).any()

    def test_agora_is_walled_except_at_its_two_exits(self, maze):
        # Every 2-D step out of the agora goes through one of its exits, the
        # alley-wide passages straight up and straight down the y axis.
        env = maze.env_2d
        agora = set(env.bins_in_region("agora").tolist())
        leaving = [v for u in agora for v in env.connectivity[u] if v not in agora]
        assert leaving
        outside = env.bin_centers[leaving]
        assert np.abs(outside[:, 0]).max() < HamletDims().corridor_width / 2
        assert {np.sign(y) for y in outside[:, 1]} == {-1.0, 1.0}


class TestRegions:
    def test_region_names(self, maze):
        assert list(maze.env_2d.regions) == ["agora", *HOUSES, *RING, *SPLITS]

    def test_houses_are_28_by_16_cm_long_side_to_the_center(self, maze, track_graph):
        for name in HOUSES:
            region = maze.env_2d.regions[name]
            centre = _pos(track_graph, name)
            assert region.kind == "polygon"
            assert region.data.area == pytest.approx(28.0 * 16.0)
            assert region.data.contains(shapely.Point(centre))
            # 8 cm from the center to the side facing the maze center.
            inward = centre - 8.0 * centre / np.linalg.norm(centre)
            assert region.data.exterior.distance(shapely.Point(inward)) < 1e-9
            assert len(maze.env_2d.bins_in_region(name)) > 0

    def test_agora_region_is_the_floor_inside_the_ring(self, maze, track_graph):
        region = maze.env_2d.regions["agora"]
        assert region.kind == "polygon"
        floor = maze.env_2d.arena.polygon
        assert region.data.difference(floor).area == pytest.approx(0.0, abs=1e-9)
        assert region.data.contains(shapely.Point(0.0, 0.0))
        # Half an alley and the agora wall inside the ring's centerline.
        ring = shapely.Polygon([_pos(track_graph, n) for n in RING])
        dims = HamletDims()
        inside = ring.buffer(
            -dims.corridor_width / 2 - dims.agora_wall_thickness, join_style="mitre"
        )
        assert inside.symmetric_difference(region.data).area == pytest.approx(
            0.0, abs=1e-6
        )
        assert len(maze.env_2d.bins_in_region("agora")) > 0
        assert len(maze.env_track.bins_in_region("agora")) > 0

    def test_junction_points_on_the_track_nodes(self, maze, track_graph):
        for name in RING + SPLITS:
            region = maze.env_2d.regions[name]
            assert region.kind == "point"
            np.testing.assert_allclose(region.data, _pos(track_graph, name))


class TestLandmarks:
    def test_agora_is_the_start_box_and_houses_are_goal_boxes(self, maze):
        landmarks = maze.env_2d.landmarks
        assert list(landmarks) == ["agora", *HOUSES]
        roles = {name: lm.metadata["role"] for name, lm in landmarks.items()}
        assert roles == {"agora": "start_box"} | dict.fromkeys(HOUSES, "goal_box")

    def test_houses_hold_what_crouzier_2020_lists(self, maze):
        contents = {
            name: maze.env_2d.landmarks[name].metadata["contents"] for name in HOUSES
        }
        assert contents == {
            "hide_house": "a novomaze (Viewpoint)",
            "run_house": "a running wheel",
            "interact_house": "a stranger mouse behind a wire mesh screen",
            "eat_house": "a food dispenser",
            "drink_house": "two water dispensers",
        }

    def test_houses_clockwise_from_the_top(self, maze):
        # Crouzier et al. (2020, Fig. 1b): Hide at the top, then clockwise
        # Drink, Eat, Interact and Run.
        angles = {
            name: np.degrees(np.arctan2(*maze.env_2d.landmarks[name].data[::-1]))
            for name in HOUSES
        }
        assert angles["hide_house"] == pytest.approx(90.0)
        clockwise = ["hide_house", "drink_house", "eat_house", "interact_house"]
        clockwise.append("run_house")
        steps = np.diff([angles[name] for name in clockwise]) % 360
        np.testing.assert_allclose(steps, 288.0)  # -72 degrees each

    def test_landmarks_at_the_region_centres(self, maze, track_graph):
        for name, landmark in maze.env_2d.landmarks.items():
            region = maze.env_2d.regions[name].data
            np.testing.assert_allclose(
                landmark.data, region.centroid.coords[0], atol=1e-9
            )
            np.testing.assert_allclose(landmark.data, _pos(track_graph, name))

    def test_other_house_counts_are_numbered(self):
        maze = make_hamlet_maze(HamletDims(n_arms=6), include_track=False)
        houses = [n for n in maze.env_2d.landmarks if n.startswith("house_")]
        assert houses == [f"house_{k}" for k in range(6)]
        assert all(
            dict(maze.env_2d.landmarks[n].metadata) == {"role": "goal_box"}
            for n in houses
        )


class TestEnvironments:
    def test_returns_maze_environments(self, maze):
        assert isinstance(maze, MazeEnvironments)
        assert maze.env_2d.units == "cm"
        assert maze.env_track.units == "cm"
        assert maze.env_track.is_linearized_track
        assert maze.env_track.arena.width == pytest.approx(8.0)

    def test_include_track_false_no_env_track(self):
        assert make_hamlet_maze(include_track=False).env_track is None

    def test_custom_dims_scale_the_maze(self):
        dims = HamletDims(split_radius=50.0, house_radius=90.0, corridor_width=10.0)
        maze = make_hamlet_maze(dims=dims, bin_size=2.0)
        graph = maze.env_track._binning.linearization.graph
        assert np.linalg.norm(_pos(graph, "hide_house")) == pytest.approx(90.0)
        assert np.linalg.norm(_pos(graph, "split_0")) == pytest.approx(50.0)

    def test_custom_bin_size(self):
        fine = make_hamlet_maze(bin_size=1.0, include_track=False)
        coarse = make_hamlet_maze(bin_size=2.1, include_track=False)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins

    def test_bins_coarser_than_the_agora_wall_warn(self):
        with pytest.warns(UserWarning, match="slanted wall of the Hamlet maze"):
            make_hamlet_maze(bin_size=2.2, include_track=False)

    @pytest.mark.parametrize("thickness", [0.0, 20.0])
    def test_agora_wall_must_leave_an_agora(self, thickness):
        with pytest.raises(ValueError, match="no walled agora"):
            make_hamlet_maze(HamletDims(agora_wall_thickness=thickness))
