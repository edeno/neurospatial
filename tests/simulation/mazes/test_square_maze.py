"""Tests for Square Maze (Open Field Box) implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestSquareMazeDims:
    """Tests for SquareMazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """SquareMazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.square_maze import SquareMazeDims

        dims = SquareMazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """SquareMazeDims should have correct default values."""
        from neurospatial.simulation.mazes.square_maze import SquareMazeDims

        dims = SquareMazeDims()
        assert dims.side_length == 30.0

    def test_default_creates_30cm_maze(self):
        """Default dimensions should create a 30 cm x 30 cm maze."""
        from neurospatial.simulation.mazes.square_maze import SquareMazeDims

        dims = SquareMazeDims()
        assert dims.side_length == 30.0

    def test_is_frozen(self):
        """SquareMazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.square_maze import SquareMazeDims

        dims = SquareMazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.side_length = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """SquareMazeDims should accept custom values."""
        from neurospatial.simulation.mazes.square_maze import SquareMazeDims

        dims = SquareMazeDims(side_length=50.0)
        assert dims.side_length == 50.0


class TestMakeSquareMaze:
    """Tests for make_square_maze factory function."""

    def test_returns_maze_environments(self):
        """make_square_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_square_shape_extent(self):
        """env_2d should have a square spatial extent."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        dims = SquareMazeDims(side_length=30.0)
        maze = make_square_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should be approximately side_length
        x_extent = x_max - x_min
        expected_extent = dims.side_length
        assert x_extent > expected_extent * 0.8  # At least 80% of expected

        # Y extent should be approximately side_length
        y_extent = y_max - y_min
        assert y_extent > expected_extent * 0.8  # At least 80% of expected

        # Should be roughly square
        assert np.isclose(x_extent, y_extent, rtol=0.1)

    def test_env_2d_centered_at_origin(self):
        """Square maze should be centered at the origin."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()

        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X should be roughly symmetric around 0
        assert np.isclose(abs(x_min), abs(x_max), rtol=0.2)

        # Y should be roughly symmetric around 0
        assert np.isclose(abs(y_min), abs(y_max), rtol=0.2)

    def test_env_2d_has_center_region(self):
        """env_2d should have a 'center' region at origin."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        assert "center" in maze.env_2d.regions

    def test_env_2d_has_quadrant_regions(self):
        """env_2d should have quadrant regions NE, NW, SE, SW."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        assert "NE" in maze.env_2d.regions
        assert "NW" in maze.env_2d.regions
        assert "SE" in maze.env_2d.regions
        assert "SW" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for square maze geometry."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        dims = SquareMazeDims(side_length=30.0)
        maze = make_square_maze(dims=dims)

        # Get regions
        center = maze.env_2d.regions["center"]
        ne = maze.env_2d.regions["NE"]
        nw = maze.env_2d.regions["NW"]
        se = maze.env_2d.regions["SE"]
        sw = maze.env_2d.regions["SW"]

        # All should be point regions
        assert center.kind == "point"
        assert ne.kind == "point"
        assert nw.kind == "point"
        assert se.kind == "point"
        assert sw.kind == "point"

        # Center should be at origin
        assert np.isclose(center.data[0], 0.0, atol=1.0)
        assert np.isclose(center.data[1], 0.0, atol=1.0)

        # Quarter position = side_length / 4
        quarter = dims.side_length / 4.0

        # NE should be at (+quarter, +quarter)
        assert np.isclose(ne.data[0], quarter, atol=1.0)
        assert np.isclose(ne.data[1], quarter, atol=1.0)

        # NW should be at (-quarter, +quarter)
        assert np.isclose(nw.data[0], -quarter, atol=1.0)
        assert np.isclose(nw.data[1], quarter, atol=1.0)

        # SE should be at (+quarter, -quarter)
        assert np.isclose(se.data[0], quarter, atol=1.0)
        assert np.isclose(se.data[1], -quarter, atol=1.0)

        # SW should be at (-quarter, -quarter)
        assert np.isclose(sw.data[0], -quarter, atol=1.0)
        assert np.isclose(sw.data[1], -quarter, atol=1.0)

    def test_no_track_graph(self):
        """Square maze should not have a track graph (open field)."""
        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()
        assert maze.env_track is None

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        dims = SquareMazeDims()

        maze_fine = make_square_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_square_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        dims = SquareMazeDims(side_length=50.0)
        maze = make_square_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both X and Y should be roughly 50 cm
        assert x_extent > 40  # More than default 30
        assert y_extent > 40  # More than default 30

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        maze_default = make_square_maze(dims=None)
        maze_explicit = make_square_maze(dims=SquareMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestSquareMazeOpenField:
    """Tests specific to open-field behavior."""

    def test_bins_cover_full_area(self):
        """Bins should cover the full square area without large gaps."""
        from neurospatial.simulation.mazes.square_maze import (
            SquareMazeDims,
            make_square_maze,
        )

        dims = SquareMazeDims(side_length=30.0)
        maze = make_square_maze(dims=dims, bin_size=2.0)

        # Expected number of bins roughly (30/2)^2 = 225
        # Allow some tolerance for edge effects
        expected_min = (dims.side_length / 2.0) ** 2 * 0.8
        assert maze.env_2d.n_bins > expected_min

    def test_all_bins_connected(self):
        """All bins should be part of a connected graph (open field)."""
        import networkx as nx

        from neurospatial.simulation.mazes.square_maze import make_square_maze

        maze = make_square_maze()

        # Open field should be fully connected
        assert nx.is_connected(maze.env_2d.connectivity)


class TestSquareMazeModuleExports:
    """Tests for module-level exports."""

    def test_can_import_from_mazes_module(self):
        """Should be able to import from neurospatial.simulation.mazes."""
        from neurospatial.simulation.mazes import SquareMazeDims, make_square_maze

        assert SquareMazeDims is not None
        assert make_square_maze is not None

    def test_square_maze_in_all(self):
        """SquareMazeDims and make_square_maze should be in __all__."""
        from neurospatial.simulation import mazes

        assert "SquareMazeDims" in mazes.__all__
        assert "make_square_maze" in mazes.__all__
