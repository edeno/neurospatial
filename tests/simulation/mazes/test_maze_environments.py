"""MazeEnvironments holds a 2-D maze and its linearized track, and keeps them."""

from __future__ import annotations

import dataclasses

import pytest

from neurospatial.simulation.mazes import MazeEnvironments, make_t_maze


@pytest.fixture(scope="module")
def maze():
    return make_t_maze(bin_size=4.0)


def test_fields_cannot_be_replaced(maze):
    with pytest.raises(dataclasses.FrozenInstanceError):
        maze.env_track = maze.env_2d
    with pytest.raises(dataclasses.FrozenInstanceError):
        maze.env_2d = maze.env_track


def test_track_must_be_a_linearized_track(maze):
    with pytest.raises(ValueError, match="env_track must be a linearized track"):
        MazeEnvironments(env_2d=maze.env_2d, env_track=maze.env_2d)


def test_track_must_share_the_2d_units(maze):
    in_metres = maze.env_track.with_metadata(units="m")
    with pytest.raises(ValueError, match="units"):
        MazeEnvironments(env_2d=maze.env_2d, env_track=in_metres)
