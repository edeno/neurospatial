"""Tests for W-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestWMazeDims:
    """Tests for WMazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """WMazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims

        dims = WMazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """WMazeDims should have correct default values."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims

        dims = WMazeDims()
        assert dims.width == 120.0
        assert dims.height == 80.0
        assert dims.corridor_width == 10.0
        assert dims.n_wells == 3

    def test_is_frozen(self):
        """WMazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims

        dims = WMazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.width = 150.0  # type: ignore[misc]

    def test_custom_values(self):
        """WMazeDims should accept custom values."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims

        dims = WMazeDims(width=150.0, height=100.0, corridor_width=15.0, n_wells=4)
        assert dims.width == 150.0
        assert dims.height == 100.0
        assert dims.corridor_width == 15.0
        assert dims.n_wells == 4


class TestMakeWMaze:
    """Tests for make_w_maze factory function."""

    def test_returns_maze_environments(self):
        """make_w_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_w_shape_extent(self):
        """env_2d should have a W-shaped spatial extent."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims(width=120.0, height=80.0, corridor_width=10.0)
        maze = make_w_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover full width
        x_extent = x_max - x_min
        expected_x_extent = dims.width
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover the height
        y_extent = y_max - y_min
        expected_y_extent = dims.height
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_well_1_region(self):
        """env_2d should have a 'well_1' region at first well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_1" in maze.env_2d.regions

    def test_env_2d_has_well_2_region(self):
        """env_2d should have a 'well_2' region at second well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_2" in maze.env_2d.regions

    def test_env_2d_has_well_3_region(self):
        """env_2d should have a 'well_3' region at third well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_3" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for W-maze geometry."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims(width=120.0, height=80.0, n_wells=3)
        maze = make_w_maze(dims=dims)

        # Get regions
        well_1 = maze.env_2d.regions["well_1"]
        well_2 = maze.env_2d.regions["well_2"]
        well_3 = maze.env_2d.regions["well_3"]

        # All should be point regions
        assert well_1.kind == "point"
        assert well_2.kind == "point"
        assert well_3.kind == "point"

        # Wells should be at top of vertical corridors
        # For 3 wells, x positions are: -width/2, 0, +width/2 (at edges and center)
        assert np.isclose(well_1.data[0], -dims.width / 2, atol=1.0)
        assert np.isclose(well_1.data[1], dims.height, atol=1.0)

        assert np.isclose(well_2.data[0], 0.0, atol=1.0)
        assert np.isclose(well_2.data[1], dims.height, atol=1.0)

        assert np.isclose(well_3.data[0], dims.width / 2, atol=1.0)
        assert np.isclose(well_3.data[1], dims.height, atol=1.0)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims()

        maze_fine = make_w_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_w_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims(width=150.0, height=100.0, corridor_width=15.0)
        maze = make_w_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Y should be roughly 100 cm (height)
        assert y_extent > 80  # Longer than default 80

        # X should be roughly 150 cm (width)
        assert x_extent > 100  # Wider than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        maze_default = make_w_maze(dims=None)
        maze_explicit = make_w_maze(dims=WMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestWMazeTrackGraph:
    """Tests for the track graph structure of W-Maze."""

    def test_track_follows_w_pattern(self):
        """Track graph should follow the W-shaped corridor pattern."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full W-maze extent."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims(width=120.0, height=80.0)
        maze = make_w_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover full width
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = dims.width
        assert x_extent >= expected_x * 0.8  # At least 80%

        # Y extent should cover the height
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = dims.height
        assert y_extent >= expected_y * 0.8  # At least 80%


class TestWMazeDocstrings:
    """Tests for docstrings and examples."""

    def test_make_w_maze_has_docstring(self):
        """make_w_maze should have a docstring."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        assert make_w_maze.__doc__ is not None
        assert len(make_w_maze.__doc__) > 100  # Reasonable length

    def test_w_maze_dims_has_docstring(self):
        """WMazeDims should have a docstring."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims

        assert WMazeDims.__doc__ is not None
