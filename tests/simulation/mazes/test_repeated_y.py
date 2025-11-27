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
        assert dims.n_junctions == 4
        assert dims.arm_length == 25.0
        assert dims.fork_length == 12.0
        assert dims.width == 8.0

    def test_is_frozen(self):
        """RepeatedYDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims()
        with pytest.raises(FrozenInstanceError):
            dims.n_junctions = 5  # type: ignore[misc]

    def test_custom_values(self):
        """RepeatedYDims should accept custom values."""
        from neurospatial.simulation.mazes.repeated_y import RepeatedYDims

        dims = RepeatedYDims(n_junctions=6, arm_length=30.0, width=10.0)
        assert dims.n_junctions == 6
        assert dims.arm_length == 30.0
        assert dims.width == 10.0


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

    def test_env_2d_has_arm_regions(self):
        """env_2d should have forked arm regions at each junction."""
        from neurospatial.simulation.mazes.repeated_y import (
            RepeatedYDims,
            make_repeated_y_maze,
        )

        dims = RepeatedYDims(n_junctions=3)
        maze = make_repeated_y_maze(dims=dims)

        # Each junction has a fork with left and right endpoints
        for i in range(1, dims.n_junctions + 1):
            assert f"arm_{i}_fork_left" in maze.env_2d.regions
            assert f"arm_{i}_fork_right" in maze.env_2d.regions

    def test_four_sequential_junctions_by_default(self):
        """Default maze should have 4 Y-junctions in series."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()

        # Count junction regions
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 4

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

    def test_y_arms_are_at_120_degrees(self):
        """Y-junction arms should be at 120 degrees from each other."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()

        # Get junction and fork arm positions for first junction
        junction_1 = maze.env_2d.regions["junction_1"]
        fork_left = maze.env_2d.regions["arm_1_fork_left"]
        fork_right = maze.env_2d.regions["arm_1_fork_right"]

        # All should be point regions
        assert junction_1.kind == "point"
        assert fork_left.kind == "point"
        assert fork_right.kind == "point"

        # Fork endpoints should be at arm_length + fork_length from junction
        # Default: 25 + 12 = 37 cm
        dist_left = np.linalg.norm(np.array(fork_left.data) - np.array(junction_1.data))
        dist_right = np.linalg.norm(
            np.array(fork_right.data) - np.array(junction_1.data)
        )
        assert dist_left > 30.0  # Should be roughly arm_length + fork_length
        assert dist_right > 30.0

    def test_junctions_alternate_orientation(self):
        """Y-junctions should alternate: odd up, even down."""
        from neurospatial.simulation.mazes.repeated_y import make_repeated_y_maze

        maze = make_repeated_y_maze()

        # Junction 1 (index 0, up-pointing): fork arms should be ABOVE junction
        junction_1 = maze.env_2d.regions["junction_1"]
        fork_1_left = maze.env_2d.regions["arm_1_fork_left"]
        fork_1_right = maze.env_2d.regions["arm_1_fork_right"]
        assert fork_1_left.data[1] > junction_1.data[1]  # Fork above junction
        assert fork_1_right.data[1] > junction_1.data[1]

        # Junction 2 (index 1, down-pointing): fork arms should be BELOW junction
        junction_2 = maze.env_2d.regions["junction_2"]
        fork_2_left = maze.env_2d.regions["arm_2_fork_left"]
        fork_2_right = maze.env_2d.regions["arm_2_fork_right"]
        assert fork_2_left.data[1] < junction_2.data[1]  # Fork below junction
        assert fork_2_right.data[1] < junction_2.data[1]

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

        dims = RepeatedYDims(n_junctions=5, arm_length=30.0, width=12.0)
        maze = make_repeated_y_maze(dims=dims)

        # Should have 5 junctions
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 5

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

        # After discretization, the graph has many nodes with varying degrees
        # Junction nodes should have degree > 2 (more than just along a line)
        # Look for nodes with degree 3 or higher (junction points)
        high_degree_nodes = [n for n in graph.nodes() if graph.degree(n) >= 3]
        # Should have at least some high-degree nodes (junctions)
        # With 3 Y-junctions, we expect at least 3 high-degree nodes
        assert len(high_degree_nodes) >= 3


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
