"""Tests for Linear Track maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestLinearTrackDims:
    """Tests for LinearTrackDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """LinearTrackDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.linear_track import LinearTrackDims

        dims = LinearTrackDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """LinearTrackDims should have correct default values."""
        from neurospatial.simulation.mazes.linear_track import LinearTrackDims

        dims = LinearTrackDims()
        assert dims.length == 150.0
        assert dims.width == 10.0

    def test_is_frozen(self):
        """LinearTrackDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.linear_track import LinearTrackDims

        dims = LinearTrackDims()
        with pytest.raises(FrozenInstanceError):
            dims.length = 200.0  # type: ignore[misc]

    def test_custom_values(self):
        """LinearTrackDims should accept custom values."""
        from neurospatial.simulation.mazes.linear_track import LinearTrackDims

        dims = LinearTrackDims(length=200.0, width=15.0)
        assert dims.length == 200.0
        assert dims.width == 15.0


class TestMakeLinearTrack:
    """Tests for make_linear_track factory function."""

    def test_returns_maze_environments(self):
        """make_linear_track should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_correct_spatial_extent(self):
        """env_2d should cover the track dimensions."""
        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        dims = LinearTrackDims(length=150.0, width=10.0)
        maze = make_linear_track(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should roughly match length (some tolerance for binning)
        # Track is centered at origin, so x from -75 to 75
        assert x_max - x_min > dims.length * 0.8  # At least 80% of length covered

        # Y extent should roughly match width
        assert y_max - y_min < dims.width * 2  # Not too much wider than corridor

    def test_env_2d_has_reward_left_region(self):
        """env_2d should have a 'reward_left' region."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track()
        assert "reward_left" in maze.env_2d.regions

    def test_env_2d_has_reward_right_region(self):
        """env_2d should have a 'reward_right' region."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track()
        assert "reward_right" in maze.env_2d.regions

    def test_reward_regions_at_correct_positions(self):
        """Reward regions should be at the ends of the track."""
        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        dims = LinearTrackDims(length=150.0)
        maze = make_linear_track(dims=dims)

        # Get region positions (Region.data contains the point coordinates)
        reward_left = maze.env_2d.regions["reward_left"]
        reward_right = maze.env_2d.regions["reward_right"]

        # Both should be point regions
        assert reward_left.kind == "point"
        assert reward_right.kind == "point"

        # Left should have smaller x than right
        assert reward_left.data[0] < reward_right.data[0]

        # They should be approximately length apart
        x_diff = reward_right.data[0] - reward_left.data[0]
        assert abs(x_diff - dims.length) < dims.length * 0.1  # Within 10%

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        dims = LinearTrackDims(length=150.0, width=10.0)

        maze_fine = make_linear_track(dims=dims, bin_size=1.0)
        maze_coarse = make_linear_track(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        dims = LinearTrackDims(length=200.0, width=20.0)
        maze = make_linear_track(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()

        # Should be roughly 200 cm long (with some tolerance)
        assert x_extent > 150  # Longer than default 150

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        maze_default = make_linear_track(dims=None)
        maze_explicit = make_linear_track(dims=LinearTrackDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestLinearTrackTrackGraph:
    """Tests for the track graph structure of Linear Track."""

    def test_track_has_start_and_end_semantics(self):
        """Track should represent a linear path from start to end."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None

        # Linear track should be 1D and connected
        assert maze.env_track.is_1d

        # All nodes should be reachable from each other
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.linear_track import make_linear_track

        maze = make_linear_track(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_length_matches_dims(self):
        """Track length should approximately match the dims.length."""

        from neurospatial.simulation.mazes.linear_track import (
            LinearTrackDims,
            make_linear_track,
        )

        dims = LinearTrackDims(length=150.0)
        maze = make_linear_track(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # The extent in x should be approximately the track length
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        assert abs(x_extent - dims.length) < dims.length * 0.1  # Within 10%
