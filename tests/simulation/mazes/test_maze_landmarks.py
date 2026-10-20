"""A built-in maze names its apparatus as landmarks with matching regions.

Landmarks give encoding functions the ports, holes, platforms and boxes by
name; regions of the same name give visit and occupancy analyses their real
extent. Both must describe the same thing in the same place.
"""

import numpy as np
import pytest
import shapely

import neurospatial.simulation.mazes as mazes
from neurospatial.simulation.mazes import MAZE_LANDMARK_ROLES

MAKERS = sorted(
    name
    for name in dir(mazes)
    if name.startswith("make_")
    and (name.endswith("maze") or name == "make_linear_track")
)


@pytest.fixture(scope="module", params=MAKERS)
def maze(request):
    return getattr(mazes, request.param)()


def test_landmark_roles_are_known(maze):
    roles = {lm.metadata.get("role") for lm in maze.env_2d.landmarks.values()}
    assert roles <= MAZE_LANDMARK_ROLES


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
