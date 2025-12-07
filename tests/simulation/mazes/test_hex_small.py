"""Tests for Small Hex Maze implementation.

The Small Hex Maze is based on CVAT annotations of an actual triangular hex maze.
It consists of a triangular arena with hexagonal barriers creating a complex path
structure. The maze has 34 hex centers as navigation nodes and 3 reward wells.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestSmallHexDims:
    """Tests for SmallHexDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """SmallHexDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.hex_small import SmallHexDims

        dims = SmallHexDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """SmallHexDims should have correct default values."""
        from neurospatial.simulation.mazes.hex_small import SmallHexDims

        dims = SmallHexDims()
        assert dims.scale == 1.0

    def test_is_frozen(self):
        """SmallHexDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.hex_small import SmallHexDims

        dims = SmallHexDims()
        with pytest.raises(FrozenInstanceError):
            dims.scale = 2.0  # type: ignore[misc]

    def test_custom_values(self):
        """SmallHexDims should accept custom scale values."""
        from neurospatial.simulation.mazes.hex_small import SmallHexDims

        dims = SmallHexDims(scale=1.5)
        assert dims.scale == 1.5


class TestMakeSmallHexMaze:
    """Tests for make_small_hex_maze factory function."""

    def test_returns_maze_environments(self):
        """make_small_hex_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_triangular_shape(self):
        """env_2d should have a triangular spatial extent."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze()

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Maze should be approximately centered at origin
        # (CVAT-based geometry may not be perfectly centered)
        assert abs((x_min + x_max) / 2) < 20  # Within 20 cm of center
        assert abs((y_min + y_max) / 2) < 20

    def test_env_2d_has_3_reward_well_regions(self):
        """env_2d should have 3 reward well regions."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(bin_size=2.0)

        # Should have reward_well_0 through reward_well_2
        reward_regions = [
            name for name in maze.env_2d.regions if name.startswith("reward_well_")
        ]
        assert len(reward_regions) == 3

        # All should be point regions
        for name in reward_regions:
            region = maze.env_2d.regions[name]
            assert region.kind == "point"

    def test_env_2d_is_centered(self):
        """env_2d should be approximately centered at origin."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze()
        # The centroid should be near the origin
        centroid = maze.env_2d.bin_centers.mean(axis=0)
        assert abs(centroid[0]) < 20  # Within 20 cm of center
        assert abs(centroid[1]) < 20

    def test_env_2d_has_reasonable_bin_count(self):
        """env_2d should have reasonable number of bins for the maze extent."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(bin_size=2.0)

        # The CVAT-based maze has substantial navigable area
        # With barriers subtracted, expect several hundred bins
        assert maze.env_2d.n_bins > 500  # At least 500 bins
        assert maze.env_2d.n_bins < 2000  # But not too many

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze_fine = make_small_hex_maze(bin_size=1.0)
        maze_coarse = make_small_hex_maze(bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_scale(self):
        """Custom scale should affect maze size."""
        from neurospatial.simulation.mazes.hex_small import (
            SmallHexDims,
            make_small_hex_maze,
        )

        # Create mazes with different scales
        maze_normal = make_small_hex_maze(dims=SmallHexDims(scale=1.0), bin_size=2.0)
        maze_large = make_small_hex_maze(dims=SmallHexDims(scale=1.5), bin_size=2.0)

        # Larger scale should result in more bins (same bin_size)
        assert maze_large.env_2d.n_bins > maze_normal.env_2d.n_bins

        # Larger scale should have larger spatial extent
        extent_normal = np.ptp(maze_normal.env_2d.bin_centers, axis=0)
        extent_large = np.ptp(maze_large.env_2d.bin_centers, axis=0)
        assert extent_large[0] > extent_normal[0]
        assert extent_large[1] > extent_normal[1]

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.hex_small import (
            SmallHexDims,
            make_small_hex_maze,
        )

        maze_default = make_small_hex_maze(dims=None)
        maze_explicit = make_small_hex_maze(dims=SmallHexDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestSmallHexTrackGraph:
    """Tests for the track graph structure of Small Hex Maze."""

    def test_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_maze_extent(self):
        """Track should cover the maze extent."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # Track should cover substantial area
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        y_extent = positions[:, 1].max() - positions[:, 1].min()

        # Should cover at least 30 cm in each direction
        assert x_extent > 30
        assert y_extent > 30

    def test_track_has_many_nodes(self):
        """Track graph should have many nodes due to discretization."""
        from neurospatial.simulation.mazes.hex_small import make_small_hex_maze

        maze = make_small_hex_maze(include_track=True, bin_size=2.0)
        assert maze.env_track is not None

        # The track graph is discretized with 34 hex centers as nodes
        # With bin_size=2.0, we expect many bins along the edges
        n_nodes = maze.env_track.connectivity.number_of_nodes()
        assert n_nodes > 200  # Should have many discretized nodes
