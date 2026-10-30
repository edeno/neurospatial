"""A built-in maze names its apparatus as landmarks with matching regions.

Landmarks give encoding functions the ports, holes, platforms and boxes by
name; regions of the same name give visit and occupancy analyses their real
extent. Both must describe the same thing in the same place.
"""

import numpy as np
import pytest
import shapely

from neurospatial.simulation.mazes import MAZE_LANDMARK_ROLES


@pytest.fixture
def maze(default_maze):
    return default_maze


def test_landmark_roles_are_known(maze):
    roles = {lm.metadata.get("role") for lm in maze.env_2d.landmarks.values()}
    assert roles <= MAZE_LANDMARK_ROLES


# The reward hardware a port landmark's ``port_type`` may name.
PORT_TYPES = {"food_well", "food_cup", "plastic_cap", "liquid_port"}


def test_every_reward_port_names_its_hardware(maze):
    for name, lm in maze.env_2d.landmarks.items():
        if lm.metadata["role"] == "reward_port":
            assert lm.metadata["port_type"] in PORT_TYPES, name


def test_each_apparatus_landmark_has_a_region_around_it(maze):
    env = maze.env_2d
    for name, lm in env.landmarks.items():
        if lm.metadata["role"] == "cue":
            continue
        assert name in env.regions, name
        region = env.regions[name]
        if lm.kind == "point" and region.kind == "polygon":
            assert (
                shapely.Polygon(region.data)
                .buffer(1e-9)
                .contains(shapely.Point(np.asarray(lm.data)))
            ), name
        elif lm.kind == "point" and region.kind == "point":
            np.testing.assert_allclose(region.data, lm.data, err_msg=name)


def test_track_carries_the_landmarks(maze):
    if maze.env_track is None:
        pytest.skip("open-field maze without a track")
    assert list(maze.env_track.landmarks) == list(maze.env_2d.landmarks)
