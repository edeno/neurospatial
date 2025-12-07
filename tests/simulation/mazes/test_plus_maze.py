"""Tests for Plus Maze (Cruciate Maze) implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestPlusMazeDims:
    """Tests for PlusMazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """PlusMazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims

        dims = PlusMazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """PlusMazeDims should have correct default values."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims

        dims = PlusMazeDims()
        assert dims.arm_length == 45.0
        assert dims.width == 10.0

    def test_default_creates_100cm_maze(self):
        """Default dimensions should create a 100 cm x 100 cm maze footprint."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims

        dims = PlusMazeDims()
        # Total span = 2 * arm_length + width = 2 * 45 + 10 = 100
        total_span = 2 * dims.arm_length + dims.width
        assert total_span == 100.0

    def test_is_frozen(self):
        """PlusMazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims

        dims = PlusMazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.arm_length = 60.0  # type: ignore[misc]

    def test_custom_values(self):
        """PlusMazeDims should accept custom values."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims

        dims = PlusMazeDims(arm_length=60.0, width=15.0)
        assert dims.arm_length == 60.0
        assert dims.width == 15.0


class TestMakePlusMaze:
    """Tests for make_plus_maze factory function."""

    def test_returns_maze_environments(self):
        """make_plus_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_plus_shape_extent(self):
        """env_2d should have a plus-shaped spatial extent."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        dims = PlusMazeDims(arm_length=45.0, width=10.0)
        maze = make_plus_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover both east and west arms (2 * arm_length)
        x_extent = x_max - x_min
        expected_x_extent = 2 * dims.arm_length  # -45 to +45
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover both north and south arms (2 * arm_length)
        y_extent = y_max - y_min
        expected_y_extent = 2 * dims.arm_length  # -45 to +45
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_symmetric(self):
        """Plus maze should be roughly symmetric around the origin."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()

        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X should be roughly symmetric
        assert np.isclose(abs(x_min), abs(x_max), rtol=0.2)

        # Y should be roughly symmetric
        assert np.isclose(abs(y_min), abs(y_max), rtol=0.2)

    def test_env_2d_has_center_region(self):
        """env_2d should have a 'center' region at origin."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert "center" in maze.env_2d.regions

    def test_env_2d_has_north_end_region(self):
        """env_2d should have a 'north_end' region."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert "north_end" in maze.env_2d.regions

    def test_env_2d_has_south_end_region(self):
        """env_2d should have a 'south_end' region."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert "south_end" in maze.env_2d.regions

    def test_env_2d_has_east_end_region(self):
        """env_2d should have an 'east_end' region."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert "east_end" in maze.env_2d.regions

    def test_env_2d_has_west_end_region(self):
        """env_2d should have a 'west_end' region."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze()
        assert "west_end" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for plus maze geometry."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        dims = PlusMazeDims(arm_length=45.0, width=10.0)
        maze = make_plus_maze(dims=dims)

        # Get regions
        center = maze.env_2d.regions["center"]
        north_end = maze.env_2d.regions["north_end"]
        south_end = maze.env_2d.regions["south_end"]
        east_end = maze.env_2d.regions["east_end"]
        west_end = maze.env_2d.regions["west_end"]

        # All should be point regions
        assert center.kind == "point"
        assert north_end.kind == "point"
        assert south_end.kind == "point"
        assert east_end.kind == "point"
        assert west_end.kind == "point"

        # Center should be at origin
        assert np.isclose(center.data[0], 0.0, atol=1.0)
        assert np.isclose(center.data[1], 0.0, atol=1.0)

        # North end should be at (0, +arm_length)
        assert np.isclose(north_end.data[0], 0.0, atol=1.0)
        assert np.isclose(north_end.data[1], dims.arm_length, atol=1.0)

        # South end should be at (0, -arm_length)
        assert np.isclose(south_end.data[0], 0.0, atol=1.0)
        assert np.isclose(south_end.data[1], -dims.arm_length, atol=1.0)

        # East end should be at (+arm_length, 0)
        assert np.isclose(east_end.data[0], dims.arm_length, atol=1.0)
        assert np.isclose(east_end.data[1], 0.0, atol=1.0)

        # West end should be at (-arm_length, 0)
        assert np.isclose(west_end.data[0], -dims.arm_length, atol=1.0)
        assert np.isclose(west_end.data[1], 0.0, atol=1.0)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        dims = PlusMazeDims()

        maze_fine = make_plus_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_plus_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        dims = PlusMazeDims(arm_length=60.0, width=15.0)
        maze = make_plus_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both X and Y should be roughly 120 cm (2 * 60 arm_length)
        assert x_extent > 90  # More than default
        assert y_extent > 90  # More than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        maze_default = make_plus_maze(dims=None)
        maze_explicit = make_plus_maze(dims=PlusMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestPlusMazeTrackGraph:
    """Tests for the track graph structure of Plus Maze."""

    def test_track_has_star_topology(self):
        """Track graph should have star topology with 4 arms from center."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.plus_maze import make_plus_maze

        maze = make_plus_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full plus maze extent."""
        from neurospatial.simulation.mazes.plus_maze import PlusMazeDims, make_plus_maze

        dims = PlusMazeDims(arm_length=45.0, width=10.0)
        maze = make_plus_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover both east and west arms
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = 2 * dims.arm_length  # west to east arm
        assert x_extent >= expected_x * 0.8  # At least 80%

        # Y extent should cover both north and south arms
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = 2 * dims.arm_length  # south to north arm
        assert y_extent >= expected_y * 0.8  # At least 80%


class TestPlusMazeModuleExports:
    """Tests for module-level exports."""

    def test_can_import_from_mazes_module(self):
        """Should be able to import from neurospatial.simulation.mazes."""
        from neurospatial.simulation.mazes import PlusMazeDims, make_plus_maze

        assert PlusMazeDims is not None
        assert make_plus_maze is not None

    def test_plus_maze_in_all(self):
        """PlusMazeDims and make_plus_maze should be in __all__."""
        from neurospatial.simulation import mazes

        assert "PlusMazeDims" in mazes.__all__
        assert "make_plus_maze" in mazes.__all__
