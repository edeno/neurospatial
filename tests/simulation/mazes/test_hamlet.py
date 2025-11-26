"""Tests for Hamlet Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestHamletDims:
    """Tests for HamletDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """HamletDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """HamletDims should have correct default values."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims()
        assert dims.central_radius == 30.0
        assert dims.arm_length == 40.0
        assert dims.corridor_width == 10.0
        assert dims.n_peripheral_arms == 5

    def test_is_frozen(self):
        """HamletDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims()
        with pytest.raises(FrozenInstanceError):
            dims.central_radius = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """HamletDims should accept custom values."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims(
            central_radius=40.0,
            arm_length=60.0,
            corridor_width=15.0,
            n_peripheral_arms=6,
        )
        assert dims.central_radius == 40.0
        assert dims.arm_length == 60.0
        assert dims.corridor_width == 15.0
        assert dims.n_peripheral_arms == 6


class TestMakeHamletMaze:
    """Tests for make_hamlet_maze factory function."""

    def test_returns_maze_environments(self):
        """make_hamlet_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_pentagonal_ring_extent(self):
        """env_2d should have a pentagonal ring with radiating arms."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(central_radius=30.0, arm_length=40.0, corridor_width=10.0)
        maze = make_hamlet_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Total extent should cover central radius + arm length + fork
        # Approximate total radius = central_radius + arm_length + corridor_width*2
        total_radius = dims.central_radius + dims.arm_length + dims.corridor_width * 2
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Both extents should be approximately 2 * total_radius (diameter)
        expected_extent = 2 * total_radius
        assert x_extent > expected_extent * 0.6  # At least 60% (conservative)
        assert y_extent > expected_extent * 0.6  # At least 60% (conservative)

    def test_env_2d_has_ring_regions(self):
        """env_2d should have 5 ring regions (pentagon vertices)."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        # Should have ring_0 through ring_4 (5 regions)
        for i in range(5):
            assert f"ring_{i}" in maze.env_2d.regions

    def test_env_2d_has_goal_regions(self):
        """env_2d should have 10 goal regions (2 per arm)."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        # Should have goal_0 through goal_9 (10 regions)
        for i in range(10):
            assert f"goal_{i}" in maze.env_2d.regions

    def test_ring_regions_are_point_type(self):
        """All ring regions should be point type."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        for i in range(5):
            region = maze.env_2d.regions[f"ring_{i}"]
            assert region.kind == "point"

    def test_goal_regions_are_point_type(self):
        """All goal regions should be point type."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        for i in range(10):
            region = maze.env_2d.regions[f"goal_{i}"]
            assert region.kind == "point"

    def test_ring_positions_form_pentagon(self):
        """Ring regions should be positioned at pentagon vertices."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(central_radius=30.0)
        maze = make_hamlet_maze(dims=dims)

        # Ring positions should be approximately at distance central_radius from origin
        for i in range(5):
            ring_pos = maze.env_2d.regions[f"ring_{i}"].data
            distance_from_origin = np.linalg.norm(ring_pos)
            assert np.isclose(distance_from_origin, dims.central_radius, atol=2.0)

    def test_goal_positions_are_at_arm_ends(self):
        """Goal regions should be at the ends of forked arms."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(central_radius=30.0, arm_length=40.0, corridor_width=10.0)
        maze = make_hamlet_maze(dims=dims)

        # Goals should be further from origin than ring vertices
        min_distance = dims.central_radius + dims.arm_length

        for i in range(10):
            goal_pos = maze.env_2d.regions[f"goal_{i}"].data
            distance_from_origin = np.linalg.norm(goal_pos)
            assert distance_from_origin > min_distance

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims()

        maze_fine = make_hamlet_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_hamlet_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(central_radius=40.0, arm_length=60.0, corridor_width=15.0)
        maze = make_hamlet_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()

        # Should be larger than default
        default_maze = make_hamlet_maze()
        default_x_extent = (
            default_maze.env_2d.bin_centers[:, 0].max()
            - default_maze.env_2d.bin_centers[:, 0].min()
        )

        assert x_extent > default_x_extent

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        maze_default = make_hamlet_maze(dims=None)
        maze_explicit = make_hamlet_maze(dims=HamletDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestHamletTrackGraph:
    """Tests for the track graph structure of Hamlet Maze."""

    def test_track_has_pentagon_ring(self):
        """Track graph should have pentagon ring connectivity."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full Hamlet maze extent."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(central_radius=30.0, arm_length=40.0)
        maze = make_hamlet_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # Extent should cover central ring + arms
        extent = positions.max(axis=0) - positions.min(axis=0)
        expected_extent = 2 * (dims.central_radius + dims.arm_length)

        # Both x and y should cover most of the expected extent
        assert extent[0] > expected_extent * 0.6  # At least 60%
        assert extent[1] > expected_extent * 0.6  # At least 60%

    def test_track_has_ring_nodes(self):
        """Track graph should have nodes for pentagon ring."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity
        # Should have ring nodes (at least 5 for pentagon vertices)
        # Implementation may add more nodes for connectivity
        assert graph.number_of_nodes() >= 5

    def test_track_has_goal_nodes(self):
        """Track graph should have nodes for goal positions."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity
        # Should have nodes for 10 goals + 5 ring vertices + connections
        # At minimum: 5 ring + 10 goals = 15 nodes
        assert graph.number_of_nodes() >= 15


class TestHamletDocstrings:
    """Tests for docstrings and examples."""

    def test_make_hamlet_maze_has_docstring(self):
        """make_hamlet_maze should have a docstring."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        assert make_hamlet_maze.__doc__ is not None
        assert len(make_hamlet_maze.__doc__) > 100  # Reasonable length

    def test_hamlet_dims_has_docstring(self):
        """HamletDims should have a docstring."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        assert HamletDims.__doc__ is not None
