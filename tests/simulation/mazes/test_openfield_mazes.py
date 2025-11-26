"""Integration tests for open-field mazes (Panel c from Wijnen et al. 2024).

These tests verify that all open-field mazes (Watermaze, Barnes, Cheeseboard,
Radial Arm) work correctly together and share consistent interfaces.
"""

import pytest

from neurospatial.simulation.mazes import (
    BarnesDims,
    CheeseboardDims,
    MazeEnvironments,
    RadialArmDims,
    WatermazeDims,
    make_barnes_maze,
    make_cheeseboard_maze,
    make_radial_arm_maze,
    make_watermaze,
)

# Parametrize over all open-field maze factories
OPENFIELD_MAZES = [
    ("watermaze", make_watermaze, WatermazeDims),
    ("barnes", make_barnes_maze, BarnesDims),
    ("cheeseboard", make_cheeseboard_maze, CheeseboardDims),
    ("radial_arm", make_radial_arm_maze, RadialArmDims),
]


class TestOpenFieldMazesCommonInterface:
    """Test that all open-field mazes share a consistent interface."""

    @pytest.mark.parametrize("name,factory,dims_cls", OPENFIELD_MAZES)
    def test_returns_maze_environments(self, name, factory, dims_cls):
        """All factories return MazeEnvironments."""
        maze = factory(bin_size=4.0)
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", OPENFIELD_MAZES)
    def test_env_2d_has_units_cm(self, name, factory, dims_cls):
        """All 2D environments have units='cm'."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.units == "cm"

    @pytest.mark.parametrize("name,factory,dims_cls", OPENFIELD_MAZES)
    def test_env_2d_has_bins(self, name, factory, dims_cls):
        """All 2D environments have bins."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.n_bins > 0

    @pytest.mark.parametrize("name,factory,dims_cls", OPENFIELD_MAZES)
    def test_has_regions(self, name, factory, dims_cls):
        """All open-field mazes have regions defined."""
        maze = factory(bin_size=4.0)
        assert len(maze.env_2d.regions) > 0

    @pytest.mark.parametrize("name,factory,dims_cls", OPENFIELD_MAZES)
    def test_dims_is_frozen_dataclass(self, name, factory, dims_cls):
        """All Dims classes are frozen dataclasses."""
        import dataclasses

        assert dataclasses.is_dataclass(dims_cls)


class TestCircularMazes:
    """Test circular arena mazes (Watermaze, Barnes, Cheeseboard)."""

    @pytest.mark.parametrize(
        "factory,dims_cls",
        [
            (make_watermaze, WatermazeDims),
            (make_barnes_maze, BarnesDims),
            (make_cheeseboard_maze, CheeseboardDims),
        ],
    )
    def test_no_track_graph(self, factory, dims_cls):
        """Circular arenas have no track graph (open field)."""
        maze = factory(bin_size=4.0)
        assert maze.env_track is None

    @pytest.mark.parametrize(
        "factory,dims_cls",
        [
            (make_watermaze, WatermazeDims),
            (make_barnes_maze, BarnesDims),
            (make_cheeseboard_maze, CheeseboardDims),
        ],
    )
    def test_roughly_circular_extent(self, factory, dims_cls):
        """Circular mazes should have roughly equal x and y extent."""
        maze = factory(bin_size=4.0)
        bin_centers = maze.env_2d.bin_centers
        x_range = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_range = bin_centers[:, 1].max() - bin_centers[:, 1].min()
        # Should be within 20% of each other
        assert abs(x_range - y_range) / max(x_range, y_range) < 0.2


class TestOpenFieldMazesSpecificBehavior:
    """Test maze-specific behavior for open-field mazes."""

    def test_watermaze_has_quadrants(self):
        """Watermaze should have NE, NW, SE, SW quadrant regions."""
        maze = make_watermaze(bin_size=4.0)
        for quadrant in ["NE", "NW", "SE", "SW"]:
            assert quadrant in maze.env_2d.regions

    def test_watermaze_has_platform(self):
        """Watermaze should have a platform region."""
        maze = make_watermaze(bin_size=4.0)
        assert "platform" in maze.env_2d.regions

    def test_barnes_has_escape_hole(self):
        """Barnes maze should have an escape hole region."""
        maze = make_barnes_maze(bin_size=4.0)
        assert "escape_hole" in maze.env_2d.regions

    def test_barnes_has_18_holes_default(self):
        """Barnes maze should have 18 holes by default (original design)."""
        maze = make_barnes_maze(bin_size=4.0)
        hole_regions = [r for r in maze.env_2d.regions if r.startswith("hole_")]
        assert len(hole_regions) == 18

    def test_cheeseboard_has_many_rewards(self):
        """Cheeseboard should have reward locations distributed across surface."""
        maze = make_cheeseboard_maze(bin_size=4.0)
        reward_regions = [r for r in maze.env_2d.regions if r.startswith("reward_")]
        # Should have many reward locations (>50 for default dimensions)
        assert len(reward_regions) > 50

    def test_radial_arm_has_8_arms_default(self):
        """Radial arm maze should have 8 arms by default."""
        maze = make_radial_arm_maze(bin_size=3.0)
        arm_regions = [r for r in maze.env_2d.regions if r.startswith("arm_")]
        assert len(arm_regions) == 8

    def test_radial_arm_has_center(self):
        """Radial arm maze should have a center region."""
        maze = make_radial_arm_maze(bin_size=3.0)
        assert "center" in maze.env_2d.regions

    def test_radial_arm_has_track_graph(self):
        """Radial arm maze has track graph (star topology)."""
        maze = make_radial_arm_maze(bin_size=3.0)
        assert maze.env_track is not None
