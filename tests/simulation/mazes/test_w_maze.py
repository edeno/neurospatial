"""Tests for W-Maze implementation."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point


class TestMakeWMaze:
    """Tests for make_w_maze factory function."""

    def test_returns_maze_environments(self):
        """make_w_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert maze.env_2d.units == "cm"

    def test_default_footprint_is_the_published_76_by_76_cm_w_track(self):
        """Karlsson & Frank (2009): 76 x 76 cm W-tracks, 7 cm wide sections."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims()
        assert dims.corridor_width == 7.0
        assert dims.n_wells == 3

        polygon = make_w_maze(include_track=False).env_2d.arena.polygon
        x_min, y_min, x_max, y_max = polygon.bounds
        assert x_max - x_min == pytest.approx(76.0)
        assert y_max - y_min == pytest.approx(76.0)

        # Halfway up, the maze is three 7 cm wide arms, 69 cm apart outer to outer.
        section = polygon.intersection(LineString([(x_min, 40.0), (x_max, 40.0)]))
        arms = sorted(section.geoms, key=lambda arm: arm.centroid.x)
        np.testing.assert_allclose([arm.length for arm in arms], 7.0)
        np.testing.assert_allclose([arm.centroid.x for arm in arms], [-34.5, 0, 34.5])

    def test_env_2d_has_well_1_region(self):
        """env_2d should have a 'well_1' region at first well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_1" in maze.env_2d.regions

    def test_env_2d_has_well_2_region(self):
        """env_2d should have a 'well_2' region at second well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_2" in maze.env_2d.regions

    def test_env_2d_has_well_3_region(self):
        """env_2d should have a 'well_3' region at third well top."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        assert "well_3" in maze.env_2d.regions

    def test_wells_sit_in_the_end_square_of_each_arm(self):
        """Each arm's reward well is centred in the arm's end square."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims()
        maze = make_w_maze(dims=dims, include_track=False)
        polygon = maze.env_2d.arena.polygon

        expected_x = [-dims.outer_arm_spacing / 2, 0.0, dims.outer_arm_spacing / 2]
        for i, x in enumerate(expected_x, start=1):
            well = maze.env_2d.regions[f"well_{i}"]
            assert well.kind == "point"
            np.testing.assert_allclose(
                well.data, [x, dims.arm_length - dims.corridor_width / 2]
            )
            # Half a corridor width from the arm's end and side walls.
            assert polygon.contains(Point(well.data))
            assert polygon.boundary.distance(Point(well.data)) == pytest.approx(
                dims.corridor_width / 2
            )

    def test_each_arm_end_well_is_a_reward_port(self):
        """Karlsson & Frank (2009): one reward site at the endpoint of each arm."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        env = make_w_maze(include_track=False).env_2d
        assert list(env.landmarks) == ["well_1", "well_2", "well_3"]
        for name, well in env.landmarks.items():
            assert well.kind == "point"
            assert dict(well.metadata) == {
                "role": "reward_port",
                "port_type": "food_well",
            }
            # The well size is unpublished, so the region is the well's point.
            np.testing.assert_allclose(env.regions[name].data, well.data)

    def test_track_carries_the_wells_and_ends_at_them(self):
        """The track's landmarks are the 2-D wells; its arm ends share their names."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze()
        track = maze.env_track
        assert list(track.landmarks) == list(maze.env_2d.landmarks)
        for name, well in maze.env_2d.landmarks.items():
            np.testing.assert_allclose(track.landmarks[name].data, well.data)
        graph = track.binning.linearization.graph
        leaves = sorted(n for n in graph.nodes if graph.degree(n) == 1)
        assert leaves == ["well_1", "well_2", "well_3"]

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_linearized_track

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims()

        maze_fine = make_w_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_w_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims(outer_arm_spacing=150.0, arm_length=100.0, corridor_width=15.0)
        maze = make_w_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Y should be roughly 100 cm (arm_length)
        assert y_extent > 80  # Longer than default 80

        # X should be roughly 150 cm (outer_arm_spacing)
        assert x_extent > 100  # Wider than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        maze_default = make_w_maze(dims=None)
        maze_explicit = make_w_maze(dims=WMazeDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestWMazeTrackGraph:
    """Tests for the track graph structure of W-Maze."""

    def test_track_follows_w_pattern(self):
        """Track graph should follow the W-shaped corridor pattern."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected with proper topology
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)
        assert maze.env_track.is_linearized_track

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_is_a_tree_whose_leaves_are_the_three_arm_ends(self):
        """The base ends at the outer arms: no dead-end stubs past them."""
        import networkx as nx

        from neurospatial.simulation.mazes.w_maze import WMazeDims, make_w_maze

        dims = WMazeDims()
        track = make_w_maze(dims=dims, bin_size=1.0).env_track
        assert nx.is_tree(track.binning.linearization.graph)
        # The arm ends meeting at the middle junction are pairwise adjacent,
        # so the bin graph has one triangle there and is not itself a tree.
        graph = track.connectivity

        leaves = sorted(n for n in graph.nodes if graph.degree(n) == 1)
        assert len(leaves) == 3
        leaf_positions = track.bin_centers[leaves]
        np.testing.assert_allclose(
            np.sort(leaf_positions[:, 0]),
            [-dims.outer_arm_spacing / 2, 0.0, dims.outer_arm_spacing / 2],
            atol=1.0,
        )
        np.testing.assert_allclose(leaf_positions[:, 1], dims.arm_length, atol=1.0)

        # The base spans only the outer-arm junctions.
        base = track.bin_centers[np.abs(track.bin_centers[:, 1]) < 1e-9]
        assert base[:, 0].min() >= -dims.outer_arm_spacing / 2 - 1e-9
        assert base[:, 0].max() <= dims.outer_arm_spacing / 2 + 1e-9

    def test_track_wells_stay_on_their_own_arms(self):
        """Each well on the linearized track lies where it does in 2-D."""
        from neurospatial.simulation.mazes.w_maze import make_w_maze

        maze = make_w_maze(bin_size=1.0)
        for name in ("well_1", "well_2", "well_3"):
            np.testing.assert_allclose(
                maze.env_track.regions[name].data,
                maze.env_2d.regions[name].data,
                atol=1e-9,
            )
