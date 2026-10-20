"""Tests for Crossword Maze implementation.

The published apparatus (McNamara et al. 2014, doi:10.1038/nn.3843) has four
start boxes and eight intersecting 5 cm tracks forming fourteen
intersections, five tracks protruding from the maze, and measures 95 cm square
excluding the start boxes.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from neurospatial.simulation.mazes.crossword import CrosswordDims, make_crossword_maze

BOXES = [
    "start_box_top_left",
    "start_box_top_right",
    "start_box_bottom_left",
    "start_box_bottom_right",
]
CAPS = ["cap_west", "cap_east", "cap_north", "cap_south_left", "cap_south_right"]


@pytest.fixture(scope="module")
def maze():
    return make_crossword_maze()


@pytest.fixture(scope="module")
def track_graph(maze):
    return maze.env_track._binning.linearization.graph


def _edge_samples(graph, n=200):
    for u, v in graph.edges():
        p, q = np.asarray(graph.nodes[u]["pos"]), np.asarray(graph.nodes[v]["pos"])
        yield (u, v), p + np.linspace(0.0, 1.0, n)[:, None] * (q - p)


class TestPublishedGeometry:
    def test_default_dims(self):
        dims = CrosswordDims()
        assert dims.corridor_width == 5.0  # published
        assert dims.grid_spacing == 19.0  # estimate: 95 cm / 5
        assert dims.box_size == 16.0  # estimate

    def test_fourteen_intersections(self, track_graph):
        junctions = [n for n in track_graph if n.startswith("junction_")]
        assert len(junctions) == 14
        assert all(track_graph.degree(n) in (3, 4) for n in junctions)

    def test_eight_tracks(self, track_graph):
        # Each track is a maximal straight run of collinear edges: count the
        # distinct (axis, coordinate) lines the edges lie on.
        lines = set()
        for u, v in track_graph.edges():
            (x0, y0), (_, y1) = track_graph.nodes[u]["pos"], track_graph.nodes[v]["pos"]
            lines.add(("row", round(y0, 6)) if y0 == y1 else ("col", round(x0, 6)))
        assert len(lines) == 8
        assert Counter(axis for axis, _ in lines) == {"row": 4, "col": 4}

    def test_node_and_edge_counts(self, track_graph):
        assert track_graph.number_of_nodes() == 23  # 14 crossings + 4 boxes + 5 ends
        assert track_graph.number_of_edges() == 29
        degrees = Counter(dict(track_graph.degree()).values())
        assert degrees == {1: 9, 3: 7, 4: 7}
        assert nx.is_connected(track_graph)

    def test_dead_ends_are_boxes_and_five_protruding_tracks(self, track_graph):
        dead_ends = {n for n, d in track_graph.degree() if d == 1}
        assert dead_ends == set(BOXES) | set(CAPS)

    def test_ninety_five_cm_square_excluding_boxes_centred(self, maze):
        boxes = [maze.env_2d.regions[name].data for name in BOXES]
        floor = maze.env_2d.arena.polygon
        tracks = floor.difference(
            boxes[0].union(boxes[1]).union(boxes[2]).union(boxes[3])
        )
        x0, y0, x1, y1 = tracks.bounds
        assert x1 - x0 == pytest.approx(95.0)
        assert y1 - y0 == pytest.approx(95.0)
        assert (x0 + x1) / 2 == pytest.approx(0.0)
        assert (y0 + y1) / 2 == pytest.approx(0.0)

    def test_protruding_tracks_reach_the_square_edge(self, track_graph):
        ends = np.array([track_graph.nodes[n]["pos"] for n in CAPS])
        assert np.all(np.isclose(np.abs(ends).max(axis=1), 47.5))

    def test_scales_with_grid_spacing(self):
        maze = make_crossword_maze(CrosswordDims(grid_spacing=30.0), bin_size=3.0)
        graph = maze.env_track._binning.linearization.graph
        ends = np.array([graph.nodes[n]["pos"] for n in CAPS])
        assert np.all(np.isclose(np.abs(ends).max(axis=1), 75.0))


class TestFootprint:
    def test_every_track_edge_lies_inside_the_maze(self, maze, track_graph):
        env = maze.env_2d
        for edge, points in _edge_samples(track_graph):
            assert env.arena.contains(points).all(), edge
            assert (env.bin_at(points) >= 0).all(), edge

    def test_no_floor_off_the_tracks_or_boxes(self, maze, track_graph):
        # Every bin center is on a track (within half a track width of its
        # centerline) or inside a start box; no open floor between tracks.
        half_width = CrosswordDims().corridor_width / 2
        segments = [
            LineString([track_graph.nodes[u]["pos"], track_graph.nodes[v]["pos"]])
            for u, v in track_graph.edges()
        ]
        boxes = [maze.env_2d.regions[name].data for name in BOXES]
        for center in maze.env_2d.bin_centers:
            point = Point(center)
            on_track = min(s.distance(point) for s in segments) <= half_width + 1e-9
            assert on_track or any(b.covers(point) for b in boxes), center

    def test_blocks_between_tracks_are_not_floor(self, maze):
        # The middle of each enclosed lattice cell is a hole, not floor.
        s = CrosswordDims().grid_spacing
        cell_centers = np.array(
            [[i * s, j * s] for i in (-1, 0, 1) for j in (-1, 0, 1)]
        )
        assert not maze.env_2d.arena.contains(cell_centers).any()


class TestRegions:
    def test_start_boxes_are_square_polygons(self, maze, track_graph):
        for name in BOXES:
            region = maze.env_2d.regions[name]
            assert region.kind == "polygon"
            assert region.data.area == pytest.approx(16.0**2)
            # The box is centred on the end of its track.
            centre = region.data.centroid.coords[0]
            np.testing.assert_allclose(centre, track_graph.nodes[name]["pos"])
            assert len(maze.env_2d.bins_in_region(name)) > 0

    def test_caps_at_protruding_track_ends(self, maze, track_graph):
        for name in CAPS:
            region = maze.env_2d.regions[name]
            assert region.kind == "point"
            end = np.asarray(track_graph.nodes[name]["pos"])
            # Half a track width back from the free end, on the track.
            assert np.linalg.norm(np.asarray(region.data) - end) == pytest.approx(2.5)
            assert len(maze.env_2d.bins_in_region(name)) > 0

    def test_region_names(self, maze):
        assert list(maze.env_2d.regions) == BOXES + CAPS


class TestLandmarks:
    def test_start_boxes_and_caps_are_landmarks(self, maze):
        landmarks = maze.env_2d.landmarks
        assert list(landmarks) == BOXES + CAPS
        roles = {name: lm.metadata["role"] for name, lm in landmarks.items()}
        assert roles == dict.fromkeys(BOXES, "start_box") | dict.fromkeys(
            CAPS, "reward_port"
        )

    def test_start_box_landmarks_at_the_box_centres(self, maze):
        expected = {
            "start_box_top_left": (-28.5, 28.5),
            "start_box_top_right": (47.5, 28.5),
            "start_box_bottom_left": (-28.5, -28.5),
            "start_box_bottom_right": (47.5, -28.5),
        }
        for name, centre in expected.items():
            np.testing.assert_allclose(maze.env_2d.landmarks[name].data, centre)

    def test_cap_landmarks_are_the_cap_regions(self, maze):
        for name in CAPS:
            np.testing.assert_allclose(
                maze.env_2d.landmarks[name].data, maze.env_2d.regions[name].data
            )


class TestEnvironments:
    def test_units_and_track(self, maze):
        assert maze.env_2d.units == "cm"
        assert maze.env_track.units == "cm"
        assert maze.env_track.is_linearized_track
        assert nx.is_connected(maze.env_track.connectivity)

    def test_include_track_false_no_env_track(self):
        assert make_crossword_maze(include_track=False).env_track is None

    def test_custom_bin_size(self):
        fine = make_crossword_maze(bin_size=1.0, include_track=False)
        coarse = make_crossword_maze(bin_size=4.0, include_track=False)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins

    def test_none_dims_uses_defaults(self):
        default = make_crossword_maze(dims=None, include_track=False)
        explicit = make_crossword_maze(dims=CrosswordDims(), include_track=False)
        assert default.env_2d.n_bins == explicit.env_2d.n_bins
