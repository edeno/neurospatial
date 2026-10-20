"""Tests for the Hampton Court maze: Small's (1901) layout and dimensions."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.hampton_court import (
    HamptonCourtDims,
    make_hampton_court_maze,
)

FEET = 30.48
INCH = 2.54


@pytest.fixture(scope="module")
def maze():
    return make_hampton_court_maze()


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
    return tuple(np.round(np.asarray(maze.env_2d.regions[name].data, float), 6))


def test_returns_maze_environments_in_cm(maze):
    assert isinstance(maze, MazeEnvironments)
    assert maze.env_2d.units == "cm"
    assert maze.env_track.units == "cm"
    assert maze.env_track.is_linearized_track


def test_default_dims_are_smalls_published_numbers():
    # Small (1901, p. 208): "6 by 8 feet", galleries 4 inches wide.
    dims = HamptonCourtDims()
    assert dims.width == pytest.approx(8 * FEET)
    assert dims.height == pytest.approx(6 * FEET)
    assert dims.corridor_width == pytest.approx(4 * INCH)


def test_floor_fills_the_8_by_6_ft_frame_centred_on_the_origin(floor):
    np.testing.assert_allclose(
        floor.bounds, (-4 * FEET, -3 * FEET, 4 * FEET, 3 * FEET), atol=1e-9
    )


def test_galleries_are_4_inches_wide(maze, floor, track_graph):
    start, goal = _point(maze, "start"), _point(maze, "goal")
    widths = [
        2.0 * floor.boundary.distance(Point((np.array(a) + np.array(b)) / 2))
        for a, b in track_graph.edges
        if not {start, goal} & {a, b}
    ]
    np.testing.assert_allclose(widths, 4 * INCH)


def test_central_open_space_is_entered_from_below(maze, floor, track_graph):
    goal = _point(maze, "goal")
    # The open space spans about 65 x 77 cm round the goal.
    assert floor.contains(box(goal[0] - 32, goal[1] - 38, goal[0] + 32, goal[1] + 38))
    assert not floor.contains(
        box(goal[0] - 34, goal[1] - 40, goal[0] + 34, goal[1] + 40)
    )
    # A single stem leads in, straight up from the south.
    (stem_end,) = track_graph.neighbors(goal)
    assert stem_end[0] == pytest.approx(goal[0])
    assert stem_end[1] < goal[1]


def test_entrance_in_the_south_wall_below_the_goal(maze):
    start, goal = _point(maze, "start"), _point(maze, "goal")
    assert start[1] == pytest.approx(-3 * FEET)
    assert abs(start[0]) < 4 * INCH
    assert goal == pytest.approx((start[0], 0.0), abs=1e-9)


def test_track_has_smalls_choice_points_dead_ends_and_one_loop(maze, track_graph):
    degrees = Counter(dict(track_graph.degree()).values())
    # Small's seven numbered choice points plus the fork he marks x.
    assert degrees[3] == 8
    # Start, goal and six cul-de-sacs.
    assert degrees[1] == 8
    assert set(degrees) == {1, 2, 3}
    # The circuit round the centre is the only loop.
    assert len(nx.cycle_basis(track_graph)) == 1
    dead_ends = {n for n, d in track_graph.degree() if d == 1}
    assert {_point(maze, "start"), _point(maze, "goal")} <= dead_ends


def test_seven_blind_alleys_hold_the_six_cul_de_sacs(maze, floor, track_graph):
    regions = maze.env_2d.regions
    blinds = [f"blind_{k}" for k in range(1, 8)]
    assert [name for name in regions if name.startswith("blind_")] == blinds
    ends = {n for n, d in track_graph.degree() if d == 1} - {
        _point(maze, "start"),
        _point(maze, "goal"),
    }
    for name in blinds:
        alley = regions[name].data
        assert regions[name].kind == "polygon"
        assert floor.covers(alley)
        held = [end for end in ends if alley.covers(Point(end))]
        # Alley 4 leads round the centre to the choice point of alley 5.
        assert len(held) == (0 if name == "blind_4" else 1), name


def test_shortest_route_avoids_every_blind_alley(maze, track_graph):
    route = nx.shortest_path(
        track_graph, _point(maze, "start"), _point(maze, "goal"), weight="length"
    )
    line = LineString(route)
    for k in range(1, 8):
        assert line.intersection(maze.env_2d.regions[f"blind_{k}"].data).length == 0


def test_include_track_false_has_no_track():
    assert make_hampton_court_maze(include_track=False).env_track is None


def test_custom_dims_scale_the_frame():
    dims = HamptonCourtDims(width=200.0, height=150.0, corridor_width=8.0)
    floor = make_hampton_court_maze(dims, include_track=False).env_2d.arena
    np.testing.assert_allclose(
        floor.polygonal_domain().bounds, (-100, -75, 100, 75), atol=1e-9
    )


def test_frame_too_small_for_the_galleries_raises():
    with pytest.raises(ValueError, match="cannot hold 18 x 14 galleries"):
        make_hampton_court_maze(HamptonCourtDims(width=150.0))


def test_bins_coarser_than_the_walls_warn():
    with pytest.warns(UserWarning, match="walls of the Hampton Court maze"):
        make_hampton_court_maze(bin_size=4.0, include_track=False)
