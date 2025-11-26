"""Tests for Repeated T-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestRepeatedTDims:
    """Tests for RepeatedTDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """RepeatedTDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """RepeatedTDims should have correct default values."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        assert dims.spine_length == 150.0
        assert dims.arm_length == 40.0
        assert dims.n_junctions == 3
        assert dims.width == 10.0

    def test_is_frozen(self):
        """RepeatedTDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims()
        with pytest.raises(FrozenInstanceError):
            dims.spine_length = 200.0  # type: ignore[misc]

    def test_custom_values(self):
        """RepeatedTDims should accept custom values."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims(
            spine_length=200.0, arm_length=50.0, n_junctions=4, width=12.0
        )
        assert dims.spine_length == 200.0
        assert dims.arm_length == 50.0
        assert dims.n_junctions == 4
        assert dims.width == 12.0


class TestMakeRepeatedTMaze:
    """Tests for make_repeated_t_maze factory function."""

    def test_returns_maze_environments(self):
        """make_repeated_t_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_comb_shape_extent(self):
        """env_2d should have a comb/rake spatial extent."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(spine_length=150.0, arm_length=40.0, n_junctions=3)
        maze = make_repeated_t_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover spine length
        x_extent = x_max - x_min
        expected_x_extent = dims.spine_length
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover the arms (upward from spine)
        y_extent = y_max - y_min
        expected_y_extent = dims.arm_length  # Arms extend upward from spine
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_start_region(self):
        """env_2d should have a 'start' region."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze()
        assert "start" in maze.env_2d.regions

    def test_env_2d_has_junction_regions(self):
        """env_2d should have junction regions for each T-junction."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(n_junctions=3)
        maze = make_repeated_t_maze(dims=dims)

        # Should have junction_0, junction_1, junction_2
        assert "junction_0" in maze.env_2d.regions
        assert "junction_1" in maze.env_2d.regions
        assert "junction_2" in maze.env_2d.regions

    def test_env_2d_has_arm_end_regions(self):
        """env_2d should have arm_end regions for each arm."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(n_junctions=3)
        maze = make_repeated_t_maze(dims=dims)

        # Should have arm_0_end, arm_1_end, arm_2_end
        assert "arm_0_end" in maze.env_2d.regions
        assert "arm_1_end" in maze.env_2d.regions
        assert "arm_2_end" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for repeated T-maze geometry."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(spine_length=150.0, arm_length=40.0, n_junctions=3)
        maze = make_repeated_t_maze(dims=dims)

        # Start should be at the left end of the spine
        start = maze.env_2d.regions["start"]
        assert start.kind == "point"
        assert np.isclose(start.data[0], -dims.spine_length / 2, atol=1.0)
        assert np.isclose(start.data[1], 0.0, atol=1.0)  # On spine centerline

        # Junctions should be evenly spaced along spine
        junction_spacing = dims.spine_length / (dims.n_junctions + 1)
        for i in range(dims.n_junctions):
            junction = maze.env_2d.regions[f"junction_{i}"]
            assert junction.kind == "point"
            expected_x = -dims.spine_length / 2 + (i + 1) * junction_spacing
            assert np.isclose(junction.data[0], expected_x, atol=1.0)
            assert np.isclose(junction.data[1], 0.0, atol=1.0)  # On spine centerline

        # Arm ends should be at the top of each perpendicular arm
        for i in range(dims.n_junctions):
            arm_end = maze.env_2d.regions[f"arm_{i}_end"]
            assert arm_end.kind == "point"
            junction_x = -dims.spine_length / 2 + (i + 1) * junction_spacing
            assert np.isclose(arm_end.data[0], junction_x, atol=1.0)
            assert np.isclose(arm_end.data[1], dims.arm_length, atol=1.0)

    def test_arms_are_perpendicular_to_spine(self):
        """Arms should extend perpendicular (90 degrees) from spine."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(spine_length=150.0, arm_length=40.0, n_junctions=3)
        maze = make_repeated_t_maze(dims=dims)

        # Check that each arm is vertical (perpendicular to horizontal spine)
        for i in range(dims.n_junctions):
            junction = maze.env_2d.regions[f"junction_{i}"]
            arm_end = maze.env_2d.regions[f"arm_{i}_end"]

            # Both should have same x-coordinate (arm is vertical)
            assert np.isclose(junction.data[0], arm_end.data[0], atol=1.0)

            # Y-coordinates should differ by arm_length
            y_diff = arm_end.data[1] - junction.data[1]
            assert np.isclose(y_diff, dims.arm_length, atol=1.0)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims()

        maze_fine = make_repeated_t_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_repeated_t_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(spine_length=200.0, arm_length=50.0, n_junctions=4)
        maze = make_repeated_t_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # X should be roughly 200 cm (spine length)
        assert x_extent > 160  # Longer than default 150

        # Y should be roughly 50 cm (arm length)
        assert y_extent > 32  # Longer than default 40

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        maze_default = make_repeated_t_maze(dims=None)
        maze_explicit = make_repeated_t_maze(dims=RepeatedTDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins

    def test_different_n_junctions(self):
        """Different n_junctions should create different numbers of arms."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        maze_2 = make_repeated_t_maze(dims=RepeatedTDims(n_junctions=2))
        maze_5 = make_repeated_t_maze(dims=RepeatedTDims(n_junctions=5))

        # Check regions count
        # maze_2 should have: start + 2 junctions + 2 arm_ends = 5 regions
        assert len(maze_2.env_2d.regions) == 5
        assert "junction_0" in maze_2.env_2d.regions
        assert "junction_1" in maze_2.env_2d.regions
        assert "arm_0_end" in maze_2.env_2d.regions
        assert "arm_1_end" in maze_2.env_2d.regions

        # maze_5 should have: start + 5 junctions + 5 arm_ends = 11 regions
        assert len(maze_5.env_2d.regions) == 11
        assert "junction_4" in maze_5.env_2d.regions
        assert "arm_4_end" in maze_5.env_2d.regions


class TestRepeatedTTrackGraph:
    """Tests for the track graph structure of Repeated T-Maze."""

    def test_track_has_branches_from_spine(self):
        """Track graph should have branches from spine to arm ends."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(n_junctions=3)
        maze = make_repeated_t_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        maze = make_repeated_t_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full repeated T-maze extent."""
        from neurospatial.simulation.mazes.repeated_t import (
            RepeatedTDims,
            make_repeated_t_maze,
        )

        dims = RepeatedTDims(spine_length=150.0, arm_length=40.0, n_junctions=3)
        maze = make_repeated_t_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover spine length
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = dims.spine_length
        assert x_extent >= expected_x * 0.8  # At least 80%

        # Y extent should cover the arms
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = dims.arm_length
        assert y_extent >= expected_y * 0.8  # At least 80%


class TestRepeatedTDocstrings:
    """Tests for docstrings and examples."""

    def test_make_repeated_t_maze_has_docstring(self):
        """make_repeated_t_maze should have a docstring."""
        from neurospatial.simulation.mazes.repeated_t import make_repeated_t_maze

        assert make_repeated_t_maze.__doc__ is not None
        assert len(make_repeated_t_maze.__doc__) > 100  # Reasonable length

    def test_repeated_t_dims_has_docstring(self):
        """RepeatedTDims should have a docstring."""
        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        assert RepeatedTDims.__doc__ is not None
