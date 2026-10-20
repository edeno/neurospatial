"""Tests for Radial Arm Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest


class TestMakeRadialArmMaze:
    """Tests for make_radial_arm_maze factory function."""

    def test_returns_maze_environments(self):
        """make_radial_arm_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_radial_extent(self):
        """env_2d should have correct radial spatial extent."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(center_radius=15.0, arm_length=50.0)
        maze = make_radial_arm_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        distances = np.linalg.norm(bin_centers, axis=1)

        # Max distance should be approximately center_radius + arm_length
        max_distance = distances.max()
        expected_max = dims.center_radius + dims.arm_length  # 65 cm
        assert max_distance >= expected_max * 0.8  # At least 80%

    def test_env_2d_has_center_region(self):
        """env_2d should have a 'center' region."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze()
        assert "center" in maze.env_2d.regions

    def test_env_2d_has_all_arm_regions(self):
        """env_2d should have regions for all arms (arm_0 through arm_7)."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze()
        for i in range(8):
            assert f"arm_{i}" in maze.env_2d.regions

    def test_env_2d_arm_regions_for_custom_n_arms(self):
        """env_2d should have correct number of arm regions for custom n_arms."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(n_arms=6)
        maze = make_radial_arm_maze(dims=dims)

        # Should have arm_0 through arm_5
        for i in range(6):
            assert f"arm_{i}" in maze.env_2d.regions

        # Should not have arm_6 or arm_7
        assert "arm_6" not in maze.env_2d.regions
        assert "arm_7" not in maze.env_2d.regions

    def test_region_positions_correct(self):
        """Regions should be at expected positions for radial arm geometry."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(center_radius=15.0, arm_length=50.0, n_arms=8)
        maze = make_radial_arm_maze(dims=dims)

        # Get regions
        center = maze.env_2d.regions["center"]
        assert center.kind == "point"
        assert np.isclose(center.data[0], 0.0, atol=1.0)
        assert np.isclose(center.data[1], 0.0, atol=1.0)

        # Check arm endpoints are at correct distances and angles
        for i in range(8):
            arm_region = maze.env_2d.regions[f"arm_{i}"]
            assert arm_region.kind == "point"

            # Distance from center: the arm end's square, half an arm width
            # inside the tip, where the well sits
            distance = np.linalg.norm(arm_region.data)
            expected_distance = (
                dims.center_radius + dims.arm_length - dims.arm_width / 2
            )
            assert np.isclose(distance, expected_distance, atol=2.0)

            # Angle
            angle = np.arctan2(arm_region.data[1], arm_region.data[0])
            expected_angle = 2 * np.pi * i / 8
            # Normalize angles to [0, 2π)
            angle = angle % (2 * np.pi)
            expected_angle = expected_angle % (2 * np.pi)
            assert np.isclose(angle, expected_angle, atol=0.1)

    def test_default_is_the_published_eight_arm_rat_maze(self):
        """Zoladek & Roberts (1978): 35.5 cm platform, 8 arms of 76 x 9 cm."""
        from shapely.geometry import LineString

        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()
        assert 2 * dims.center_radius == 35.5
        assert (dims.arm_length, dims.arm_width, dims.n_arms) == (76.0, 9.0, 8)

        polygon = make_radial_arm_maze(include_track=False).env_2d.arena.polygon
        reach = dims.center_radius + dims.arm_length
        # Arm 0 points along +x: it ends 17.75 + 76 cm out and is 9 cm wide
        # halfway along.
        assert polygon.bounds[2] == pytest.approx(reach)
        middle = dims.center_radius + dims.arm_length / 2
        section = polygon.intersection(LineString([(middle, -20.0), (middle, 20.0)]))
        assert section.length == pytest.approx(dims.arm_width)

    def test_arm_regions_sit_inside_their_arm_ends(self):
        """Each arm region is half an arm width from the end and side walls."""
        from shapely.geometry import Point

        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()
        env = make_radial_arm_maze(dims=dims, include_track=False).env_2d
        polygon = env.arena.polygon
        for i in range(dims.n_arms):
            point = Point(env.regions[f"arm_{i}"].data)
            assert polygon.contains(point)
            assert polygon.boundary.distance(point) == pytest.approx(dims.arm_width / 2)

    def test_track_arm_ends_agree_with_the_arm_regions(self):
        """Track leaves are the arm ends; arm regions keep their 2-D places."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()
        maze = make_radial_arm_maze(dims=dims, bin_size=1.0)
        track = maze.env_track
        graph = track.connectivity
        leaves = [n for n in graph.nodes if graph.degree(n) == 1]
        assert len(leaves) == dims.n_arms
        reach = dims.center_radius + dims.arm_length
        np.testing.assert_allclose(
            np.linalg.norm(track.bin_centers[leaves], axis=1), reach, atol=1.0
        )
        for name, region in maze.env_2d.regions.items():
            np.testing.assert_allclose(track.regions[name].data, region.data, atol=1e-9)

    def test_arms_at_equal_angular_spacing(self):
        """Arms should be equally spaced around the center."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(n_arms=8)
        maze = make_radial_arm_maze(dims=dims)

        # Get angles of all arm endpoints
        angles = []
        for i in range(8):
            arm_region = maze.env_2d.regions[f"arm_{i}"]
            angle = np.arctan2(arm_region.data[1], arm_region.data[0])
            angles.append(angle)

        angles = np.array(angles)
        angles = np.sort(angles)

        # Check angular spacing is uniform (45° = π/4 radians for 8 arms)
        expected_spacing = 2 * np.pi / 8
        for i in range(len(angles) - 1):
            spacing = angles[i + 1] - angles[i]
            assert np.isclose(spacing, expected_spacing, atol=0.1)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_linearized_track

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()

        maze_fine = make_radial_arm_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_radial_arm_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(center_radius=20.0, arm_length=80.0)
        maze = make_radial_arm_maze(dims=dims)

        # Check that max distance from center is approximately correct
        bin_centers = maze.env_2d.bin_centers
        max_distance = np.linalg.norm(bin_centers, axis=1).max()

        expected_max = dims.center_radius + dims.arm_length  # 100 cm
        assert max_distance >= expected_max * 0.8

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        maze_default = make_radial_arm_maze(dims=None)
        maze_explicit = make_radial_arm_maze(dims=RadialArmDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestRadialArmTrackGraph:
    """Tests for the track graph structure of Radial Arm Maze."""

    def test_track_has_star_topology(self):
        """Track graph should have star topology (center connected to all arms)."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_linearized_track

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        maze = make_radial_arm_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_full_maze(self):
        """Track should cover the full radial arm extent."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(center_radius=15.0, arm_length=50.0)
        maze = make_radial_arm_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # Max distance from origin should be approximately center_radius + arm_length
        distances = np.linalg.norm(positions, axis=1)
        max_distance = distances.max()
        expected_max = dims.center_radius + dims.arm_length
        assert max_distance >= expected_max * 0.8
