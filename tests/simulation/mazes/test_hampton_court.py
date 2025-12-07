"""Tests for Hampton Court Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestHamptonCourtDims:
    """Tests for HamptonCourtDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """HamptonCourtDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.hampton_court import HamptonCourtDims

        dims = HamptonCourtDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """HamptonCourtDims should have correct default values."""
        from neurospatial.simulation.mazes.hampton_court import HamptonCourtDims

        dims = HamptonCourtDims()
        assert dims.size == 300.0
        assert dims.corridor_width == 11.0

    def test_is_frozen(self):
        """HamptonCourtDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.hampton_court import HamptonCourtDims

        dims = HamptonCourtDims()
        with pytest.raises(FrozenInstanceError):
            dims.size = 400.0  # type: ignore[misc]

    def test_custom_values(self):
        """HamptonCourtDims should accept custom values."""
        from neurospatial.simulation.mazes.hampton_court import HamptonCourtDims

        dims = HamptonCourtDims(size=400.0, corridor_width=15.0)
        assert dims.size == 400.0
        assert dims.corridor_width == 15.0


class TestMakeHamptonCourtMaze:
    """Tests for make_hampton_court_maze factory function."""

    def test_returns_maze_environments(self):
        """make_hampton_court_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_approximate_size(self):
        """env_2d should have approximate 300 × 300 cm spatial extent."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        dims = HamptonCourtDims(size=300.0)
        maze = make_hampton_court_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Both X and Y extents should be roughly 300 cm
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Allow for some tolerance (corridors might not reach exact edges)
        assert x_extent > 200.0  # At least 200 cm wide
        assert y_extent > 200.0  # At least 200 cm tall
        assert x_extent < 400.0  # Not larger than 400 cm
        assert y_extent < 400.0  # Not larger than 400 cm

    def test_env_2d_has_start_region(self):
        """env_2d should have a 'start' region at edge."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        assert "start" in maze.env_2d.regions

    def test_env_2d_has_goal_region(self):
        """env_2d should have a 'goal' region at center."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        assert "goal" in maze.env_2d.regions

    def test_start_region_is_point(self):
        """Start region should be a point region."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        start = maze.env_2d.regions["start"]
        assert start.kind == "point"

    def test_goal_region_is_point(self):
        """Goal region should be a point region."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze()
        goal = maze.env_2d.regions["goal"]
        assert goal.kind == "point"

    def test_goal_region_near_center(self):
        """Goal region should be near the center of the maze."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        dims = HamptonCourtDims(size=300.0)
        maze = make_hampton_court_maze(dims=dims)

        goal = maze.env_2d.regions["goal"]
        goal_pos = goal.data

        # Goal should be near the center (0, 0) with some tolerance
        assert abs(goal_pos[0]) < 50.0  # Within 50 cm of center X
        assert abs(goal_pos[1]) < 50.0  # Within 50 cm of center Y

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        dims = HamptonCourtDims()

        maze_fine = make_hampton_court_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_hampton_court_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        dims = HamptonCourtDims(size=400.0, corridor_width=15.0)
        maze = make_hampton_court_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Should be larger than default (300 cm)
        assert x_extent > 250.0  # Larger than 250 cm
        assert y_extent > 250.0  # Larger than 250 cm

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        maze_default = make_hampton_court_maze(dims=None)
        maze_explicit = make_hampton_court_maze(dims=HamptonCourtDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestHamptonCourtTrackGraph:
    """Tests for the track graph structure of Hampton Court Maze."""

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_maze_extent(self):
        """Track should cover the Hampton Court Maze extent."""
        from neurospatial.simulation.mazes.hampton_court import (
            HamptonCourtDims,
            make_hampton_court_maze,
        )

        dims = HamptonCourtDims(size=300.0)
        maze = make_hampton_court_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # Both X and Y extents should be substantial
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        y_extent = positions[:, 1].max() - positions[:, 1].min()

        # At least 50% of the size
        assert x_extent >= dims.size * 0.5
        assert y_extent >= dims.size * 0.5

    def test_path_from_start_to_goal_exists(self):
        """There should be a path from start to goal in the track graph."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity

        # Find nodes closest to start and goal regions
        start_region = maze.env_2d.regions["start"]
        goal_region = maze.env_2d.regions["goal"]

        start_pos = start_region.data
        goal_pos = goal_region.data

        # Find nearest nodes
        positions = {n: graph.nodes[n]["pos"] for n in graph.nodes()}

        start_node = min(
            positions.keys(),
            key=lambda n: np.linalg.norm(np.array(positions[n]) - np.array(start_pos)),
        )
        goal_node = min(
            positions.keys(),
            key=lambda n: np.linalg.norm(np.array(positions[n]) - np.array(goal_pos)),
        )

        # There should be a path between them
        assert nx.has_path(graph, start_node, goal_node)

    def test_maze_has_multiple_paths_complexity(self):
        """Hampton Court maze should have multiple dead ends (complex structure)."""
        from neurospatial.simulation.mazes.hampton_court import (
            make_hampton_court_maze,
        )

        maze = make_hampton_court_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity

        # Count nodes with degree 1 (dead ends)
        dead_ends = [n for n in graph.nodes() if graph.degree(n) == 1]

        # Hampton Court maze should have at least 3 dead ends
        # (start, goal, and at least one other dead end to show complexity)
        assert len(dead_ends) >= 3
