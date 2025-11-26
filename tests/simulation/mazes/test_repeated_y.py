"""Tests for Repeated Y-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestRepeatedYDims:
    """Tests for RepeatedYDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """RepeatedYDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """RepeatedYDims should have correct default values."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims()
        assert dims.n_junctions == 3
        assert dims.segment_length == 50.0
        assert dims.width == 10.0

    def test_is_frozen(self):
        """RepeatedYDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims()
        with pytest.raises(FrozenInstanceError):
            dims.n_junctions = 5  # type: ignore[misc]

    def test_custom_values(self):
        """RepeatedYDims should accept custom values."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims(n_junctions=5, segment_length=40.0, width=12.0)
        assert dims.n_junctions == 5
        assert dims.segment_length == 40.0
        assert dims.width == 12.0


class TestMakeRepeatedYMaze:
    """Tests for make_repeated_y_maze factory function."""

    def test_returns_maze_environments(self):
        """make_repeated_y_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_start_region(self):
        """env_2d should have a 'start' region."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()
        assert "start" in maze.env_2d.regions

    def test_env_2d_has_goal_region(self):
        """env_2d should have a 'goal' region at final endpoint."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()
        assert "goal" in maze.env_2d.regions

    def test_env_2d_has_junction_regions(self):
        """env_2d should have junction regions for each Y-junction."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=3)
        maze = make_repeated_y_maze(dims=dims)

        # Should have junction_1, junction_2, junction_3
        for i in range(1, dims.n_junctions + 1):
            assert f"junction_{i}" in maze.env_2d.regions

    def test_env_2d_has_dead_end_regions(self):
        """env_2d should have dead end regions (Warner-Warden forks)."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=3)
        maze = make_repeated_y_maze(dims=dims)

        # Each junction has one dead end (split into two corridors)
        # Expect dead_1_left, dead_1_right, dead_2_left, dead_2_right, etc.
        for i in range(1, dims.n_junctions):  # n_junctions - 1 dead ends
            assert f"dead_{i}_left" in maze.env_2d.regions
            assert f"dead_{i}_right" in maze.env_2d.regions

    def test_three_sequential_junctions_by_default(self):
        """Default maze should have 3 Y-junctions in series."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()

        # Count junction regions
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 3

    def test_custom_number_of_junctions(self):
        """Should support custom number of junctions."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=5)
        maze = make_repeated_y_maze(dims=dims)

        # Count junction regions
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 5

    def test_warner_warden_dead_ends_are_forked(self):
        """Dead ends should be split into two small corridors (Warner-Warden)."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()

        # Check that we have paired dead end regions
        assert "dead_1_left" in maze.env_2d.regions
        assert "dead_1_right" in maze.env_2d.regions

        # The two dead ends should be spatially separated
        dead_1_left = maze.env_2d.regions["dead_1_left"]
        dead_1_right = maze.env_2d.regions["dead_1_right"]

        # Both should be point regions
        assert dead_1_left.kind == "point"
        assert dead_1_right.kind == "point"

        # They should be distinct positions
        dist = np.linalg.norm(np.array(dead_1_left.data) - np.array(dead_1_right.data))
        assert dist > 5.0  # At least 5 cm apart

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims()

        maze_fine = make_repeated_y_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_repeated_y_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=4, segment_length=60.0, width=15.0)
        maze = make_repeated_y_maze(dims=dims)

        # Should have 4 junctions
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 4

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        maze_default = make_repeated_y_maze(dims=None)
        maze_explicit = make_repeated_y_maze(dims=RepeatedYDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestRepeatedYTrackGraph:
    """Tests for the track graph structure of Repeated Y-Maze."""

    def test_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_has_branching_structure(self):
        """Track should have correct branching structure for Y-junctions."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=3)
        maze = make_repeated_y_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity

        # Each junction should have 3 connections (1 entry, 2 exits)
        # Except the first junction (2 exits only if start is separate)
        # Count degree-3 nodes (junctions)
        degree_3_nodes = [n for n in graph.nodes() if graph.degree(n) == 3]
        # Should have at least n_junctions degree-3 nodes
        assert len(degree_3_nodes) >= dims.n_junctions - 1


class TestRepeatedYDocstrings:
    """Tests for docstrings and examples."""

    def test_make_repeated_y_maze_has_docstring(self):
        """make_repeated_y_maze should have a docstring."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        assert make_repeated_y_maze.__doc__ is not None
        assert len(make_repeated_y_maze.__doc__) > 100  # Reasonable length

    def test_repeated_y_dims_has_docstring(self):
        """RepeatedYDims should have a docstring."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        assert RepeatedYDims.__doc__ is not None
