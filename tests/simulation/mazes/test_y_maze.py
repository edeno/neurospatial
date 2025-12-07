"""Tests for Y-Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestYMazeDims:
    """Tests for YMazeDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """YMazeDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims

        dims = YMazeDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """YMazeDims should have correct default values."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims

        dims = YMazeDims()
        assert dims.arm_length == 50.0
        assert dims.width == 10.0

    def test_is_frozen(self):
        """YMazeDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims

        dims = YMazeDims()
        with pytest.raises(FrozenInstanceError):
            dims.arm_length = 100.0  # type: ignore[misc]

    def test_custom_values(self):
        """YMazeDims should accept custom values."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims

        dims = YMazeDims(arm_length=75.0, width=15.0)
        assert dims.arm_length == 75.0
        assert dims.width == 15.0


class TestMakeYMaze:
    """Tests for make_y_maze factory function."""

    def test_returns_maze_environments(self):
        """make_y_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_y_shape_extent(self):
        """env_2d should have a Y-shaped spatial extent."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims(arm_length=50.0, width=10.0)
        maze = make_y_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover arms at 210° and 330° (down-left and down-right)
        # At 330°: x = arm_length * cos(330°) ≈ 0.866 * arm_length
        # At 210°: x = arm_length * cos(210°) ≈ -0.866 * arm_length
        x_extent = x_max - x_min
        expected_x_extent = 2 * dims.arm_length * np.cos(np.radians(30))  # ~86.6 cm
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover from center to top arm (90°) and bottom arms (210°, 330°)
        # At 90°: y = +arm_length
        # At 210°/330°: y = -arm_length * sin(30°) = -0.5 * arm_length
        y_extent = y_max - y_min
        expected_y_extent = 1.5 * dims.arm_length  # From -0.5*arm to +arm
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_center_region(self):
        """env_2d should have a 'center' region at origin."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert "center" in maze.env_2d.regions

    def test_env_2d_has_arm1_end_region(self):
        """env_2d should have an 'arm1_end' region."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert "arm1_end" in maze.env_2d.regions

    def test_env_2d_has_arm2_end_region(self):
        """env_2d should have an 'arm2_end' region."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert "arm2_end" in maze.env_2d.regions

    def test_env_2d_has_arm3_end_region(self):
        """env_2d should have an 'arm3_end' region."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert "arm3_end" in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for Y-maze geometry."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims(arm_length=50.0)
        maze = make_y_maze(dims=dims)

        # Get regions
        center = maze.env_2d.regions["center"]
        arm1_end = maze.env_2d.regions["arm1_end"]
        arm2_end = maze.env_2d.regions["arm2_end"]
        arm3_end = maze.env_2d.regions["arm3_end"]

        # All should be point regions
        assert center.kind == "point"
        assert arm1_end.kind == "point"
        assert arm2_end.kind == "point"
        assert arm3_end.kind == "point"

        # Center should be at origin
        assert np.isclose(center.data[0], 0.0, atol=1.0)
        assert np.isclose(center.data[1], 0.0, atol=1.0)

        # Arm 1 endpoint at 90° (up)
        expected_arm1 = (
            dims.arm_length * np.cos(np.radians(90)),
            dims.arm_length * np.sin(np.radians(90)),
        )
        assert np.isclose(arm1_end.data[0], expected_arm1[0], atol=1.0)
        assert np.isclose(arm1_end.data[1], expected_arm1[1], atol=1.0)

        # Arm 2 endpoint at 210° (down-left)
        expected_arm2 = (
            dims.arm_length * np.cos(np.radians(210)),
            dims.arm_length * np.sin(np.radians(210)),
        )
        assert np.isclose(arm2_end.data[0], expected_arm2[0], atol=1.0)
        assert np.isclose(arm2_end.data[1], expected_arm2[1], atol=1.0)

        # Arm 3 endpoint at 330° (down-right)
        expected_arm3 = (
            dims.arm_length * np.cos(np.radians(330)),
            dims.arm_length * np.sin(np.radians(330)),
        )
        assert np.isclose(arm3_end.data[0], expected_arm3[0], atol=1.0)
        assert np.isclose(arm3_end.data[1], expected_arm3[1], atol=1.0)

    def test_arm_angles_are_120_degrees_apart(self):
        """Arms should be separated by 120° angles."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims(arm_length=50.0)
        maze = make_y_maze(dims=dims)

        # Get arm endpoints
        center = maze.env_2d.regions["center"].data
        arm1_end = maze.env_2d.regions["arm1_end"].data
        arm2_end = maze.env_2d.regions["arm2_end"].data
        arm3_end = maze.env_2d.regions["arm3_end"].data

        # Calculate angles from center to each arm
        def get_angle(endpoint, center_point):
            dx = endpoint[0] - center_point[0]
            dy = endpoint[1] - center_point[1]
            return np.degrees(np.arctan2(dy, dx)) % 360

        angle1 = get_angle(arm1_end, center)
        angle2 = get_angle(arm2_end, center)
        angle3 = get_angle(arm3_end, center)

        # Normalize angles to [0, 360)
        angles = sorted([angle1, angle2, angle3])

        # Check 120° separation (with wrap-around)
        diff1 = angles[1] - angles[0]
        diff2 = angles[2] - angles[1]
        diff3 = (360 + angles[0]) - angles[2]

        assert np.isclose(diff1, 120.0, atol=5.0)
        assert np.isclose(diff2, 120.0, atol=5.0)
        assert np.isclose(diff3, 120.0, atol=5.0)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims()

        maze_fine = make_y_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_y_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims(arm_length=80.0, width=15.0)
        maze = make_y_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both extents should be larger than default (50 cm arms)
        assert x_extent > 60  # Larger than default
        assert y_extent > 60  # Larger than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        maze_default = make_y_maze(dims=None)
        maze_explicit = make_y_maze(dims=YMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestYMazeTrackGraph:
    """Tests for the track graph structure of Y-Maze."""

    def test_track_has_three_edges_from_center(self):
        """Track graph should have 3 edges branching from center."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        # Y-maze has: center -> arm1_end, center -> arm2_end, center -> arm3_end
        # This means the underlying graph has 3 edges from center
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_1d

        # Find the center node in the connectivity graph
        # The center should have degree 3 (connected to 3 arms)
        graph = maze.env_track.connectivity
        center_nodes = [n for n, d in graph.degree() if d == 3]
        assert len(center_nodes) >= 1  # At least one node with degree 3

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full Y-maze extent."""
        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims(arm_length=50.0)
        maze = make_y_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover arms at 210° and 330°
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = 2 * dims.arm_length * np.cos(np.radians(30))  # ~86.6 cm
        assert x_extent >= expected_x * 0.8  # At least 80%

        # Y extent should cover from -0.5*arm_length to +arm_length
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = 1.5 * dims.arm_length  # 75 cm for arm_length=50
        assert y_extent >= expected_y * 0.8  # At least 80%
