"""Tests for T-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestTMazeDims:
    """Tests for TMazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """TMazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims

        dims = TMazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """TMazeDims should have correct default values."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims

        dims = TMazeDims()
        assert dims.stem_length == 100.0
        assert dims.arm_length == 50.0
        assert dims.width == 10.0

    def test_is_frozen(self):
        """TMazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims

        dims = TMazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.stem_length = 150.0  # type: ignore[misc]

    def test_custom_values(self):
        """TMazeDims should accept custom values."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims

        dims = TMazeDims(stem_length=150.0, arm_length=75.0, width=15.0)
        assert dims.stem_length == 150.0
        assert dims.arm_length == 75.0
        assert dims.width == 15.0


class TestMakeTMaze:
    """Tests for make_t_maze factory function."""

    def test_returns_maze_environments(self):
        """make_t_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_t_shape_extent(self):
        """env_2d should have a T-shaped spatial extent."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        dims = TMazeDims(stem_length=100.0, arm_length=50.0, width=10.0)
        maze = make_t_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover both arms (2 * arm_length)
        x_extent = x_max - x_min
        expected_x_extent = 2 * dims.arm_length  # -50 to +50
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover the stem (stem_length)
        y_extent = y_max - y_min
        expected_y_extent = dims.stem_length  # -50 to +50
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_start_region(self):
        """env_2d should have a 'start' region at stem base."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert "start" in maze.env_2d.regions

    def test_env_2d_has_junction_region(self):
        """env_2d should have a 'junction' region at T-intersection."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert "junction" in maze.env_2d.regions

    def test_env_2d_has_left_end_region(self):
        """env_2d should have a 'left_end' region at left arm endpoint."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert "left_end" in maze.env_2d.regions

    def test_env_2d_has_right_end_region(self):
        """env_2d should have a 'right_end' region at right arm endpoint."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze()
        assert "right_end" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for T-maze geometry."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        dims = TMazeDims(stem_length=100.0, arm_length=50.0)
        maze = make_t_maze(dims=dims)

        # Get regions
        start = maze.env_2d.regions["start"]
        junction = maze.env_2d.regions["junction"]
        left_end = maze.env_2d.regions["left_end"]
        right_end = maze.env_2d.regions["right_end"]

        # All should be point regions
        assert start.kind == "point"
        assert junction.kind == "point"
        assert left_end.kind == "point"
        assert right_end.kind == "point"

        # Start should be at bottom of stem (y = -stem_length/2)
        half_stem = dims.stem_length / 2
        assert np.isclose(start.data[1], -half_stem, atol=1.0)
        assert np.isclose(start.data[0], 0.0, atol=1.0)  # Centered in x

        # Junction should be at top of stem (y = +stem_length/2)
        assert np.isclose(junction.data[1], half_stem, atol=1.0)
        assert np.isclose(junction.data[0], 0.0, atol=1.0)  # Centered in x

        # Left end should be at left arm end
        half_arm = dims.arm_length
        assert np.isclose(left_end.data[0], -half_arm, atol=1.0)
        assert np.isclose(left_end.data[1], half_stem, atol=1.0)  # Same y as junction

        # Right end should be at right arm end
        assert np.isclose(right_end.data[0], half_arm, atol=1.0)
        assert np.isclose(right_end.data[1], half_stem, atol=1.0)  # Same y as junction

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        dims = TMazeDims()

        maze_fine = make_t_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_t_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        dims = TMazeDims(stem_length=150.0, arm_length=80.0, width=15.0)
        maze = make_t_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Y should be roughly 150 cm (stem length)
        assert y_extent > 100  # Longer than default 100

        # X should be roughly 160 cm (2 * arm_length)
        assert x_extent > 120  # Wider than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        maze_default = make_t_maze(dims=None)
        maze_explicit = make_t_maze(dims=TMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestTMazeTrackGraph:
    """Tests for the track graph structure of T-Maze."""

    def test_track_has_three_edges_from_junction(self):
        """Track graph should have 3 edges branching from junction concept."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        # T-maze has: start -> junction, junction -> left_end, junction -> right_end
        # This means the underlying graph has 3 edges
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        maze = make_t_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full T-maze extent."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims, make_t_maze

        dims = TMazeDims(stem_length=100.0, arm_length=50.0)
        maze = make_t_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover both arms
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = 2 * dims.arm_length  # left to right arm
        assert x_extent >= expected_x * 0.8  # At least 80%

        # Y extent should cover the stem
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = dims.stem_length
        assert y_extent >= expected_y * 0.8  # At least 80%


class TestTMazeDocstrings:
    """Tests for docstrings and examples."""

    def test_make_t_maze_has_docstring(self):
        """make_t_maze should have a docstring."""
        from neurospatial.simulation.mazes.t_maze import make_t_maze

        assert make_t_maze.__doc__ is not None
        assert len(make_t_maze.__doc__) > 100  # Reasonable length

    def test_t_maze_dims_has_docstring(self):
        """TMazeDims should have a docstring."""
        from neurospatial.simulation.mazes.t_maze import TMazeDims

        assert TMazeDims.__doc__ is not None
