"""Simulated mazes publish labels and complete immutable region snapshots."""

import pytest

from neurospatial.simulation.mazes import (
    make_barnes_maze,
    make_cheeseboard_maze,
    make_linear_track,
    make_radial_arm_maze,
    make_rat_hexmaze,
    make_repeated_y_maze,
)


@pytest.mark.parametrize(
    "factory",
    [make_linear_track, make_radial_arm_maze, make_repeated_y_maze, make_rat_hexmaze],
)
def test_maze_snapshots_keep_units_and_named_regions_in_both_representations(factory):
    maze = factory(bin_size=10.0)
    assert maze.env_2d.units == "cm"
    assert maze.env_2d.regions
    assert maze.env_track.units == "cm"
    with pytest.raises(AttributeError, match="read-only"):
        maze.env_2d.units = "m"
    with pytest.raises(TypeError, match="read-only"):
        maze.env_2d.regions.clear()


def test_open_field_maze_publishes_all_reward_locations():
    maze = make_cheeseboard_maze(bin_size=10.0)
    assert maze.env_2d.units == "cm"
    assert len(maze.env_2d.regions) > 1
    assert all(name.startswith("well_") for name in maze.env_2d.regions)
    with pytest.raises(TypeError, match="read-only"):
        maze.env_2d.regions.popitem()


def test_barnes_escape_hole_uses_the_complete_region_builder():
    maze = make_barnes_maze(bin_size=10.0, escape_hole_index=9)
    names = maze.env_2d.regions.list_names()
    assert names[9] == "escape_hole"
    assert names[-1] == "start_box"
    assert maze.env_2d.landmarks["escape_hole"].metadata["index"] == 9
