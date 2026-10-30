"""A built-in maze's track and point regions lie on its 2-D floor.

The linearized track stands for the corridors of the 2-D maze. A track segment
that leaves the floor crosses a wall, so linearized positions and geodesic
distances along it describe paths the animal cannot take.
"""

import numpy as np
import pytest


@pytest.fixture
def maze(default_maze):
    return default_maze


def test_track_lies_on_the_2d_floor(maze):
    if maze.env_track is None:
        pytest.skip("open-field maze without a track")
    # Track bins are spaced one bin_size apart along every track edge.
    samples = maze.env_track.bin_centers
    off_floor = samples[~maze.env_2d.contains(samples)]
    assert off_floor.shape[0] == 0, np.round(off_floor[:5], 1)


def test_point_regions_lie_on_the_2d_floor(maze):
    env = maze.env_2d
    off_floor = [
        name
        for name, region in env.regions.items()
        if region.kind == "point"
        and not env.contains(np.asarray(region.data, dtype=float)[np.newaxis])[0]
    ]
    assert off_floor == []
