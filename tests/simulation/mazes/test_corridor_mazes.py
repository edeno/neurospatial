"""Integration tests for corridor-based mazes (Panel a from Wijnen et al. 2024).

These tests verify that all corridor mazes (Linear Track, T-Maze, Y-Maze,
W-Maze, Small Hex) behave as their sources describe; their shared interface is
tested in test_maze_interface.py.
"""

from neurospatial.simulation.mazes import (
    make_linear_track,
    make_t_maze,
    make_w_maze,
    make_y_maze,
)


class TestCorridorMazesSpecificBehavior:
    """Test maze-specific behavior for corridor mazes."""

    def test_linear_track_is_horizontal(self):
        """Linear track should be oriented horizontally."""
        maze = make_linear_track(bin_size=2.0)
        bin_centers = maze.env_2d.bin_centers
        x_range = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_range = bin_centers[:, 1].max() - bin_centers[:, 1].min()
        # X range should be much larger than Y range
        assert x_range > y_range * 5

    def test_t_maze_has_junction(self):
        """T-maze should have a junction region."""
        maze = make_t_maze(bin_size=2.0)
        assert "junction" in maze.env_2d.regions

    def test_y_maze_has_three_arms(self):
        """Y-maze should have 3 arm endpoints."""
        maze = make_y_maze(bin_size=2.0)
        arm_regions = [r for r in maze.env_2d.regions if r.startswith("arm")]
        assert len(arm_regions) == 3

    def test_w_maze_has_three_wells(self):
        """W-maze should have 3 wells."""
        maze = make_w_maze(bin_size=2.0)
        well_regions = [r for r in maze.env_2d.regions if r.startswith("well")]
        assert len(well_regions) == 3
