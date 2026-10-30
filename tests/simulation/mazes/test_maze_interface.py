"""Every maze factory returns the same kind of result.

Each builds a ``MazeEnvironments`` in cm whose 2-D floor is one connected
piece; corridor mazes add a connected linearized track, which
``include_track=False`` leaves out, and open fields have none.
"""

from __future__ import annotations

import networkx as nx
import pytest

from neurospatial.simulation import mazes
from neurospatial.simulation.mazes import MazeEnvironments

OPEN_FIELDS = {
    "make_barnes_maze",
    "make_cheeseboard_maze",
    "make_square_maze",
    "make_watermaze",
}


def test_returns_a_maze_in_cm_on_one_connected_floor(default_maze):
    assert isinstance(default_maze, MazeEnvironments)
    env = default_maze.env_2d
    assert env.units == "cm"
    assert env.n_bins > 0
    assert len(env.components()) == 1
    assert len(env.regions) > 0


def test_corridor_mazes_have_a_connected_track_and_open_fields_none(
    maze_name, default_maze
):
    track = default_maze.env_track
    if maze_name in OPEN_FIELDS:
        assert track is None
        return
    assert track.is_linearized_track
    assert track.units == "cm"
    assert nx.is_connected(track.connectivity)


TRACK_MAZES = sorted(
    name
    for name in dir(mazes)
    if name.startswith("make_")
    and (name.endswith("maze") or name == "make_linear_track")
    and name not in OPEN_FIELDS
)


@pytest.mark.parametrize("name", TRACK_MAZES)
def test_include_track_false_leaves_the_track_out(name):
    assert getattr(mazes, name)(bin_size=4.0, include_track=False).env_track is None
