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
        assert dims.inner_radius == 40.0
        assert dims.outer_radius == 80.0
        assert dims.corridor_width == 10.0
        assert dims.n_arms == 5

    def test_is_frozen(self):
        """HamletDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims()
        with pytest.raises(FrozenInstanceError):
            dims.inner_radius = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """HamletDims should accept custom values."""
        from neurospatial.simulation.mazes.hamlet import HamletDims

        dims = HamletDims(
            inner_radius=50.0,
            outer_radius=100.0,
            corridor_width=15.0,
            n_arms=6,
        )
        assert dims.inner_radius == 50.0
        assert dims.outer_radius == 100.0
        assert dims.corridor_width == 15.0
        assert dims.n_arms == 6


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

    def test_env_2d_has_correct_extent(self):
        """env_2d should cover center-inner-outer structure."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(inner_radius=40.0, outer_radius=80.0, corridor_width=10.0)
        maze = make_hamlet_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # Total extent should cover outer_radius + corridor_width buffer
        total_radius = dims.outer_radius + dims.corridor_width
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Both extents should be approximately 2 * total_radius (diameter)
        expected_extent = 2 * total_radius
        assert x_extent > expected_extent * 0.6  # At least 60% (conservative)
        assert y_extent > expected_extent * 0.6  # At least 60% (conservative)

    def test_env_2d_has_center_region(self):
        """env_2d should have center region."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        assert "center" in maze.env_2d.regions

    def test_env_2d_has_inner_regions(self):
        """env_2d should have 5 inner regions (pentagon vertices)."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        # Should have inner_0 through inner_4 (5 regions)
        for i in range(5):
            assert f"inner_{i}" in maze.env_2d.regions

    def test_env_2d_has_outer_regions(self):
        """env_2d should have 5 outer regions."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        # Should have outer_0 through outer_4 (5 regions)
        for i in range(5):
            assert f"outer_{i}" in maze.env_2d.regions

    def test_center_region_is_point_type(self):
        """Center region should be point type."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        region = maze.env_2d.regions["center"]
        assert region.kind == "point"

    def test_inner_regions_are_point_type(self):
        """All inner regions should be point type."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        for i in range(5):
            region = maze.env_2d.regions[f"inner_{i}"]
            assert region.kind == "point"

    def test_outer_regions_are_point_type(self):
        """All outer regions should be point type."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze()
        for i in range(5):
            region = maze.env_2d.regions[f"outer_{i}"]
            assert region.kind == "point"

    def test_inner_positions_form_pentagon(self):
        """Inner regions should be positioned at pentagon vertices."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(inner_radius=40.0)
        maze = make_hamlet_maze(dims=dims)

        # Inner positions should be approximately at distance inner_radius from origin
        for i in range(5):
            inner_pos = maze.env_2d.regions[f"inner_{i}"].data
            distance_from_origin = np.linalg.norm(inner_pos)
            assert np.isclose(distance_from_origin, dims.inner_radius, atol=2.0)

    def test_outer_positions_are_at_outer_radius(self):
        """Outer regions should be at the outer radius."""
        from neurospatial.simulation.mazes.hamlet import HamletDims, make_hamlet_maze

        dims = HamletDims(inner_radius=40.0, outer_radius=80.0)
        maze = make_hamlet_maze(dims=dims)

        # Outer positions should be at outer_radius from origin
        for i in range(5):
            outer_pos = maze.env_2d.regions[f"outer_{i}"].data
            distance_from_origin = np.linalg.norm(outer_pos)
            assert np.isclose(distance_from_origin, dims.outer_radius, atol=2.0)

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

        dims = HamletDims(inner_radius=50.0, outer_radius=100.0, corridor_width=15.0)
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

    def test_track_has_center_inner_outer_structure(self):
        """Track graph should have center-inner-outer connectivity."""
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

        dims = HamletDims(inner_radius=40.0, outer_radius=80.0)
        maze = make_hamlet_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # Extent should cover outer_radius
        extent = positions.max(axis=0) - positions.min(axis=0)
        expected_extent = 2 * dims.outer_radius

        # Both x and y should cover most of the expected extent
        assert extent[0] > expected_extent * 0.6  # At least 60%
        assert extent[1] > expected_extent * 0.6  # At least 60%

    def test_track_has_center_node(self):
        """Track graph should have nodes at center position."""
        import numpy as np

        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity
        # After discretization, nodes have integer IDs but positions
        # Check that there's a node near the center (0, 0)
        center_pos = np.array([0.0, 0.0])
        min_dist = float("inf")
        for node in graph.nodes():
            pos = np.array(graph.nodes[node]["pos"])
            dist = np.linalg.norm(pos - center_pos)
            min_dist = min(min_dist, dist)
        # Should have a node within reasonable distance of center
        assert min_dist < 10.0  # Within 10 cm of center

    def test_track_has_inner_and_outer_nodes(self):
        """Track graph should have inner and outer nodes."""
        from neurospatial.simulation.mazes.hamlet import make_hamlet_maze

        maze = make_hamlet_maze(include_track=True)
        assert maze.env_track is not None

        graph = maze.env_track.connectivity
        # Should have inner and outer nodes (5 each = 10 + center = 11)
        assert graph.number_of_nodes() >= 11
