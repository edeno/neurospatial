"""Tests for the repeated T-maze: Blodgett's six-unit T maze (Tolman &
Nyswander 1927, Fig. 5)."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.repeated_t import (
    RepeatedTDims,
    make_repeated_t_maze,
)

INCH = 2.54
CHOICES = [f"choice_{k}" for k in range(1, 7)]
BLINDS = [f"blind_{k}" for k in range(1, 7)]


@pytest.fixture(scope="module")
def maze():
    return make_repeated_t_maze()


@pytest.fixture(scope="module")
def floor(maze):
    return maze.env_2d.arena.polygonal_domain()


@pytest.fixture(scope="module")
def track_graph(maze):
    """The track's nodes and edges, rebuilt from its segments."""
    graph = nx.Graph()
    for start, end in np.asarray(maze.env_track.arena.segments):
        graph.add_edge(
            tuple(np.round(start, 6)),
            tuple(np.round(end, 6)),
            length=float(np.linalg.norm(end - start)),
        )
    return graph


def _point(maze, name):
    """A point region's position, or a box landmark's centre."""
    env = maze.env_2d
    data = env.landmarks[name].data if name in env.landmarks else env.regions[name].data
    return tuple(np.round(np.asarray(data, float), 6))


def _dead_end_near(track_graph, node, point):
    """The dead end next to track node ``node`` that lies closest to ``point``."""
    return min(
        (n for n in track_graph.neighbors(node) if track_graph.degree(n) == 1),
        key=lambda n: np.linalg.norm(np.subtract(n, point)),
    )


def test_returns_maze_environments_in_cm(maze):
    assert isinstance(maze, MazeEnvironments)
    assert maze.env_2d.units == "cm"
    assert maze.env_track.units == "cm"
    assert maze.env_track.is_linearized_track


def test_default_dims_are_the_figures_numbers():
    # Tolman & Nyswander (1927) Fig. 5: 4-in alleys, 16-in and 18-in segments.
    dims = RepeatedTDims()
    assert dims.corridor_width == pytest.approx(4 * INCH)
    assert dims.arm_length == pytest.approx(16 * INCH)
    assert dims.long_arm_length == pytest.approx(18 * INCH)


def test_footprint_is_78_by_56_inches_centred_on_the_origin(floor):
    # Blind alleys 2 and 3 and the goal arm set the extent, not the start box.
    np.testing.assert_allclose(
        floor.bounds, (-39 * INCH, -28 * INCH, 39 * INCH, 28 * INCH), atol=1e-9
    )


def test_regions_name_start_six_choices_six_blinds_and_goal(maze):
    assert list(maze.env_2d.regions) == ["start_box", *CHOICES, *BLINDS, "food_box"]


def test_six_t_choice_points_and_six_blind_alleys(maze, track_graph):
    degrees = Counter(dict(track_graph.degree()).values())
    assert degrees == {3: 6, 1: 8}
    assert nx.is_tree(track_graph)
    assert {_point(maze, name) for name in CHOICES} == {
        n for n, d in track_graph.degree() if d == 3
    }


def test_true_path_meets_the_choice_points_in_order(maze, track_graph):
    last = _point(maze, "choice_6")
    goal = _dead_end_near(track_graph, last, _point(maze, "food_box"))
    route = nx.shortest_path(track_graph, _point(maze, "start_box"), goal)
    assert route[1:-1] == [_point(maze, name) for name in CHOICES]


def test_start_box_feeds_choice_1_from_the_left(maze, track_graph):
    start, first = _point(maze, "start_box"), _point(maze, "choice_1")
    assert list(track_graph.neighbors(start)) == [first]
    assert start[1] == pytest.approx(first[1])
    assert start[0] < first[0]


def test_choice_points_are_one_segment_plus_an_alley_width_apart(maze):
    gaps = [
        np.linalg.norm(np.subtract(_point(maze, a), _point(maze, b)))
        for a, b in pairwise(CHOICES)
    ]
    np.testing.assert_allclose(gaps, np.array([20, 20, 20, 22, 22]) * INCH, atol=1e-6)


def test_blind_alleys_and_goal_arm_are_16_inches_along_their_walls(maze, track_graph):
    dims = RepeatedTDims()
    half = dims.corridor_width / 2
    lengths = {}
    for k, name in enumerate(BLINDS, start=1):
        choice = _point(maze, f"choice_{k}")
        end = _dead_end_near(track_graph, choice, _point(maze, name))
        lengths[name] = np.linalg.norm(np.subtract(end, choice)) - half
    choice_6 = _point(maze, "choice_6")
    goal_end = _dead_end_near(track_graph, choice_6, _point(maze, "food_box"))
    goal_arm = np.linalg.norm(np.subtract(goal_end, choice_6)) - half
    for name, length in lengths.items():
        # Blind alley 4 stops a partition short of alley 1.
        expected = 16 * INCH - (dims.wall_thickness if name == "blind_4" else 0.0)
        assert length == pytest.approx(expected), name
    assert goal_arm == pytest.approx(16 * INCH)


def test_alleys_are_4_inches_wide(maze, floor, track_graph):
    start = _point(maze, "start_box")
    widths = [
        2.0 * floor.boundary.distance(Point((np.array(a) + np.array(b)) / 2))
        for a, b in track_graph.edges
        if start not in (a, b)
    ]
    np.testing.assert_allclose(widths, 4 * INCH)


def test_blind_alley_4_ends_a_partition_short_of_alley_1(maze, floor, track_graph):
    dims = RepeatedTDims()
    choice_1, choice_4 = _point(maze, "choice_1"), _point(maze, "choice_4")
    end = _dead_end_near(track_graph, choice_4, _point(maze, "blind_4"))
    assert end[0] - choice_1[0] == pytest.approx(
        dims.corridor_width / 2 + dims.wall_thickness
    )
    across = LineString([(choice_1[0], end[1]), end])
    assert not floor.contains(across)


def test_start_and_food_boxes_are_landmarks(maze):
    landmarks = maze.env_2d.landmarks
    assert {name: lm.metadata["role"] for name, lm in landmarks.items()} == {
        "start_box": "start_box",
        "food_box": "goal_box",
    }


def test_start_box_is_7_5_by_6_inches_before_choice_1(maze, track_graph):
    dims = RepeatedTDims()
    region = maze.env_2d.regions["start_box"]
    x0, y0, x1, y1 = region.data.bounds
    assert region.kind == "polygon"
    assert (x1 - x0, y1 - y0) == pytest.approx(
        (dims.start_box_depth, dims.start_box_width)
    )
    # The box ends one stem length and half an alley short of choice point 1.
    assert _point(maze, "choice_1")[0] - x1 == pytest.approx(
        dims.start_stem_length + dims.corridor_width / 2
    )
    # The landmark is the box's centre, where the track starts.
    centre = region.data.centroid.coords[0]
    np.testing.assert_allclose(_point(maze, "start_box"), centre, atol=1e-6)
    assert track_graph.degree(_point(maze, "start_box")) == 1


def test_food_box_is_the_end_of_the_last_alley(maze, floor, track_graph):
    dims = RepeatedTDims()
    region = maze.env_2d.regions["food_box"]
    x0, y0, x1, y1 = region.data.bounds
    assert region.kind == "polygon"
    assert (x1 - x0, y1 - y0) == pytest.approx(
        (dims.food_box_depth, dims.corridor_width)
    )
    assert floor.covers(region.data)
    # It closes the alley from choice point 6: Fig. 5's F at the far end.
    choice_6 = _point(maze, "choice_6")
    goal_end = _dead_end_near(track_graph, choice_6, _point(maze, "food_box"))
    assert x1 == pytest.approx(goal_end[0]) == pytest.approx(floor.bounds[2])
    assert (y0 + y1) / 2 == pytest.approx(choice_6[1])
    np.testing.assert_allclose(
        _point(maze, "food_box"), region.data.centroid.coords[0], atol=1e-6
    )


@pytest.mark.parametrize("name", ["start_box", "food_box"])
def test_box_regions_cover_bins_on_both_environments(maze, name):
    assert len(maze.env_2d.bins_in_region(name)) > 0
    assert len(maze.env_track.bins_in_region(name)) > 0


def test_include_track_false_has_no_track():
    assert make_repeated_t_maze(include_track=False).env_track is None


def test_long_arm_not_longer_than_arm_raises():
    with pytest.raises(ValueError, match="so the alleys touch"):
        RepeatedTDims(long_arm_length=40.64)


def test_food_box_longer_than_the_alley_raises():
    with pytest.raises(ValueError, match="does not fit"):
        RepeatedTDims(food_box_depth=50.0)
