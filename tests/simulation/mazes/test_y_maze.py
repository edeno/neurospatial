"""Tests for Y-Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point


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

        polygon = maze.env_2d.arena.polygon
        vertices = np.asarray(polygon.exterior.coords)

        # Each arm reaches exactly arm_length from the centre along its axis.
        for angle in (270.0, 30.0, 150.0):
            axis = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            assert (vertices @ axis).max() == pytest.approx(dims.arm_length)

        # The single arm points down; the other two point up and out.
        x_min, y_min, x_max, y_max = polygon.bounds
        assert y_min == pytest.approx(-dims.arm_length)
        assert y_max < dims.arm_length
        assert x_max == pytest.approx(-x_min)

    def test_arm_ends_are_square_and_one_corridor_wide(self):
        """Halfway along, each arm is one 10 cm corridor."""
        from shapely.geometry import LineString

        from neurospatial.simulation.mazes.y_maze import YMazeDims, make_y_maze

        dims = YMazeDims()
        polygon = make_y_maze(dims=dims, include_track=False).env_2d.arena.polygon
        # A horizontal cut through the down arm at half its length.
        y = -dims.arm_length / 2
        section = polygon.intersection(LineString([(-100.0, y), (100.0, y)]))
        assert section.length == pytest.approx(dims.width)

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

        np.testing.assert_allclose(center.data, [0.0, 0.0])

        # Arm ends are centred in each arm's end square, half a corridor width
        # back from the far end: arm 1 down (270°), arm 2 at 30°, arm 3 at 150°.
        distance = dims.arm_length - dims.width / 2
        polygon = maze.env_2d.arena.polygon
        for region, angle in ((arm1_end, 270), (arm2_end, 30), (arm3_end, 150)):
            expected = distance * np.array(
                [np.cos(np.radians(angle)), np.sin(np.radians(angle))]
            )
            np.testing.assert_allclose(region.data, expected, atol=1e-9)
            point = Point(region.data)
            assert polygon.contains(point)
            assert polygon.boundary.distance(point) == pytest.approx(dims.width / 2)

    def test_track_regions_stay_where_they_are_in_2d(self):
        """Each region on the linearized track lies where it does in 2-D."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze(bin_size=1.0)
        for name, region in maze.env_2d.regions.items():
            np.testing.assert_allclose(
                maze.env_track.regions[name].data, region.data, atol=1e-9
            )

    def test_spontaneous_alternation_maze_has_no_reward_ports(self):
        """The unrewarded Y-maze has no landmarks; arm ends stay regions."""
        from neurospatial.simulation.mazes.y_maze import make_y_maze

        maze = make_y_maze()
        assert len(maze.env_2d.landmarks) == 0
        assert len(maze.env_track.landmarks) == 0
        assert list(maze.env_2d.regions) == [
            "center",
            "arm1_end",
            "arm2_end",
            "arm3_end",
        ]

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
        assert maze.env_track.is_linearized_track

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
        assert maze.env_track.is_linearized_track

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

        # The track runs from the centre to each arm's far end.
        radii = np.linalg.norm(positions, axis=1)
        assert radii.max() <= dims.arm_length + 1e-9
        assert radii.max() >= dims.arm_length - maze.env_track.bin_sizes.max()

        # Arms at 30° and 150° span 2 * 0.866 * arm_length in x; the down arm
        # reaches -arm_length and the others +0.5 * arm_length in y.
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        assert x_extent >= 2 * dims.arm_length * np.cos(np.radians(30)) * 0.9
        assert positions[:, 1].min() <= -0.9 * dims.arm_length
        assert positions[:, 1].max() <= 0.5 * dims.arm_length + 1e-9
