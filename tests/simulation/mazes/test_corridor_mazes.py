"""Integration tests for corridor-based mazes (Panel a from Wijnen et al. 2024).

These tests verify that all corridor mazes (Linear Track, T-Maze, Y-Maze,
W-Maze, Small Hex) work correctly together and share consistent interfaces.
"""

import pytest

from neurospatial.simulation.mazes import (
    LinearTrackDims,
    MazeEnvironments,
    SmallHexDims,
    TMazeDims,
    WMazeDims,
    YMazeDims,
    make_linear_track,
    make_small_hex_maze,
    make_t_maze,
    make_w_maze,
    make_y_maze,
)

# Parametrize over all corridor maze factories
CORRIDOR_MAZES = [
    ("linear_track", make_linear_track, LinearTrackDims),
    ("t_maze", make_t_maze, TMazeDims),
    ("y_maze", make_y_maze, YMazeDims),
    ("w_maze", make_w_maze, WMazeDims),
    ("small_hex", make_small_hex_maze, SmallHexDims),
]


class TestCorridorMazesCommonInterface:
    """Test that all corridor mazes share a consistent interface."""

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_returns_maze_environments(self, name, factory, dims_cls):
        """All factories return MazeEnvironments."""
        maze = factory(bin_size=3.0)
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_env_2d_has_units_cm(self, name, factory, dims_cls):
        """All 2D environments have units='cm'."""
        maze = factory(bin_size=3.0)
        assert maze.env_2d.units == "cm"

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_env_2d_has_bins(self, name, factory, dims_cls):
        """All 2D environments have bins."""
        maze = factory(bin_size=3.0)
        assert maze.env_2d.n_bins > 0

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_env_track_exists(self, name, factory, dims_cls):
        """All corridor mazes have track graphs."""
        maze = factory(bin_size=3.0)
        assert maze.env_track is not None

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_track_graph_is_connected(self, name, factory, dims_cls):
        """All track graphs are connected."""
        import networkx as nx

        maze = factory(bin_size=3.0)
        graph = maze.env_track.connectivity
        assert nx.is_connected(graph)

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_dims_is_frozen_dataclass(self, name, factory, dims_cls):
        """All Dims classes are frozen dataclasses."""
        import dataclasses

        assert dataclasses.is_dataclass(dims_cls)
        # Frozen check - try to modify attribute directly
        dims = dims_cls()
        field_name = next(iter(dataclasses.fields(dims_cls))).name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(dims, field_name, 999)

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_custom_dims_work(self, name, factory, dims_cls):
        """Can pass custom dimensions to factory."""
        dims = dims_cls()  # Default dims
        maze = factory(dims=dims, bin_size=3.0)
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_include_track_false(self, name, factory, dims_cls):
        """Can disable track graph creation."""
        maze = factory(bin_size=3.0, include_track=False)
        assert maze.env_track is None


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


class TestCorridorMazesBinSizeScaling:
    """Test that bin size affects environment resolution correctly."""

    @pytest.mark.parametrize("name,factory,dims_cls", CORRIDOR_MAZES)
    def test_smaller_bin_size_more_bins(self, name, factory, dims_cls):
        """Smaller bin size should create more bins."""
        maze_coarse = factory(bin_size=5.0)
        maze_fine = factory(bin_size=2.0)
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins
