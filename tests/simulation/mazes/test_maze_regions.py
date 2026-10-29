"""Every named region of a built-in maze covers at least one bin.

A region covering no bins makes trial, visit and run analyses refuse it, so a
built-in maze must place its wells, arm ends and goals where its bins are, in
both its 2-D and its linearized track environment.
"""

import pytest


@pytest.fixture
def maze(default_maze):
    return default_maze


def test_2d_regions_cover_bins(maze):
    env = maze.env_2d
    assert len(env.regions) > 0
    empty = [name for name in env.regions if len(env.bins_in_region(name)) == 0]
    assert empty == []


def test_track_carries_the_2d_regions_and_they_cover_bins(maze):
    if maze.env_track is None:
        pytest.skip("open-field maze without a track")
    track = maze.env_track
    assert list(track.regions) == list(maze.env_2d.regions)
    empty = [name for name in track.regions if len(track.bins_in_region(name)) == 0]
    assert empty == []
