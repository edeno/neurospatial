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

    def test_each_arm_has_a_food_cup_reward_port(self):
        """Zoladek & Roberts (1978): a food cup at each arm's peripheral end."""
        from neurospatial.simulation.mazes.radial_arm import make_radial_arm_maze

        env = make_radial_arm_maze(include_track=False).env_2d
        names = [f"food_cup_{i}" for i in range(8)]
        # The rat starts on the central platform, so there is no start box.
        assert list(env.landmarks) == names
        assert list(env.regions) == ["center", *names]
        for name in names:
            cup = env.landmarks[name]
            assert cup.kind == "point"
            assert dict(cup.metadata) == {
                "role": "reward_port",
                "port_type": "food_cup",
            }

    def test_food_cups_for_custom_n_arms(self):
        """A six-arm maze has six food cups."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(n_arms=6)
        maze = make_radial_arm_maze(dims=dims)
        names = [f"food_cup_{i}" for i in range(6)]
        assert list(maze.env_2d.landmarks) == names
        assert list(maze.env_2d.regions) == ["center", *names]

    def test_food_cup_positions_correct(self):
        """Each cup is centred half its length in from its arm's far end."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(center_radius=15.0, arm_length=50.0, n_arms=8)
        maze = make_radial_arm_maze(dims=dims)

        center = maze.env_2d.regions["center"]
        assert center.kind == "point"
        np.testing.assert_allclose(center.data, [0.0, 0.0])

        distance = dims.center_radius + dims.arm_length - dims.food_cup_length / 2
        for i in range(8):
            angle = 2 * np.pi * i / 8
            np.testing.assert_allclose(
                maze.env_2d.landmarks[f"food_cup_{i}"].data,
                [distance * np.cos(angle), distance * np.sin(angle)],
                atol=1e-9,
            )

    def test_food_cups_have_the_published_size_at_the_arm_ends(self):
        """Each cup region is the 7.5 x 5 cm cup, flush with its arm's end."""
        from shapely.geometry import Point, Polygon

        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()
        assert (dims.food_cup_length, dims.food_cup_width) == (7.5, 5.0)
        env = make_radial_arm_maze(dims=dims, include_track=False).env_2d
        floor = env.arena.polygon
        reach = dims.center_radius + dims.arm_length
        for i in range(dims.n_arms):
            region = env.regions[f"food_cup_{i}"]
            assert region.kind == "polygon"
            cup = Polygon(region.data)
            assert cup.area == pytest.approx(7.5 * 5.0)
            # Its length runs along the arm, out to the arm's far end.
            radii = np.linalg.norm(np.asarray(cup.exterior.coords), axis=1)
            assert radii.max() == pytest.approx(np.hypot(reach, 2.5))
            assert floor.buffer(1e-9).contains(cup)
            assert cup.contains(Point(env.landmarks[f"food_cup_{i}"].data))
            assert len(env.bins_in_region(f"food_cup_{i}")) > 0

    def test_default_is_the_published_eight_arm_rat_maze(self):
        """Zoladek & Roberts (1978): 35.5 cm platform, 8 arms of 76 x 9 cm."""
        from shapely.geometry import LineString

        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims()
        assert 2 * dims.center_radius == 35.5
        assert (dims.arm_length, dims.corridor_width, dims.n_arms) == (76.0, 9.0, 8)

        polygon = make_radial_arm_maze(include_track=False).env_2d.arena.polygon
        reach = dims.center_radius + dims.arm_length
        # Arm 0 points along +x: it ends 17.75 + 76 cm out and is 9 cm wide
        # halfway along.
        assert polygon.bounds[2] == pytest.approx(reach)
        middle = dims.center_radius + dims.arm_length / 2
        section = polygon.intersection(LineString([(middle, -20.0), (middle, 20.0)]))
        assert section.length == pytest.approx(dims.corridor_width)

    def test_track_arm_ends_agree_with_the_food_cups(self):
        """Track leaves are the arm ends, named like the cups there."""
        from shapely.geometry import Polygon

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
        track_graph = track.binning.linearization.graph
        assert sorted(n for n in track_graph if track_graph.degree(n) == 1) == [
            f"food_cup_{i}" for i in range(dims.n_arms)
        ]
        # The track keeps each cup's footprint and landmark.
        np.testing.assert_allclose(track.regions["center"].data, [0.0, 0.0])
        assert list(track.landmarks) == list(maze.env_2d.landmarks)
        for name, cup in maze.env_2d.landmarks.items():
            assert Polygon(track.regions[name].data).equals(
                Polygon(maze.env_2d.regions[name].data)
            )
            np.testing.assert_allclose(track.landmarks[name].data, cup.data)
            assert len(track.bins_in_region(name)) > 0

    def test_arms_at_equal_angular_spacing(self):
        """Arms should be equally spaced around the center."""
        from neurospatial.simulation.mazes.radial_arm import (
            RadialArmDims,
            make_radial_arm_maze,
        )

        dims = RadialArmDims(n_arms=8)
        maze = make_radial_arm_maze(dims=dims)

        # Get angles of all arms' food cups
        angles = []
        for i in range(8):
            cup = maze.env_2d.landmarks[f"food_cup_{i}"]
            angle = np.arctan2(cup.data[1], cup.data[0])
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
