"""Tests for Crossword Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestCrosswordDims:
    """Tests for CrosswordDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """CrosswordDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """CrosswordDims should have correct default values."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims()
        assert dims.grid_spacing == 30.0
        assert dims.corridor_width == 10.0
        assert dims.box_size == 15.0

    def test_is_frozen(self):
        """CrosswordDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims()
        with pytest.raises(FrozenInstanceError):
            dims.grid_spacing = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """CrosswordDims should accept custom values."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, box_size=20.0)
        assert dims.grid_spacing == 40.0
        assert dims.corridor_width == 12.0
        assert dims.box_size == 20.0


class TestMakeCrosswordMaze:
    """Tests for make_crossword_maze factory function."""

    def test_returns_maze_environments(self):
        """make_crossword_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_has_grid_extent(self):
        """env_2d should cover the grid plus corner boxes."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, corridor_width=10.0, box_size=15.0)
        maze = make_crossword_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent: from -box_size to 3*spacing + box_size (boxes extend outward)
        # Grid spans 0 to 3*spacing = 90 cm, boxes add ~15 cm on each side
        x_extent = x_max - x_min
        expected_x_extent = 3 * dims.grid_spacing + dims.box_size  # ~105 cm
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent: from -box_size to 2*spacing + box_size
        y_extent = y_max - y_min
        expected_y_extent = 2 * dims.grid_spacing + dims.box_size  # ~75 cm
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_four_corner_boxes(self):
        """env_2d should have 4 corner box regions as polygons."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        assert "box_top_left" in maze.env_2d.regions
        assert "box_top_right" in maze.env_2d.regions
        assert "box_bottom_left" in maze.env_2d.regions
        assert "box_bottom_right" in maze.env_2d.regions

    def test_corner_boxes_are_polygons(self):
        """Corner box regions should be polygon type (not points)."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        for name in [
            "box_top_left",
            "box_top_right",
            "box_bottom_left",
            "box_bottom_right",
        ]:
            region = maze.env_2d.regions[name]
            assert region.kind == "polygon", f"{name} should be a polygon region"

    def test_env_2d_has_junction_regions(self):
        """env_2d should have junction_X_Y regions at intersections."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        # Check junction regions at key intersections
        expected_junctions = [
            "junction_0_0",
            "junction_0_1",
            "junction_0_2",
            "junction_0_3",
            "junction_1_1",
            "junction_1_2",
            "junction_2_0",
            "junction_2_1",
            "junction_2_2",
            "junction_2_3",
        ]
        for junction in expected_junctions:
            assert junction in maze.env_2d.regions, f"{junction} should exist"

    def test_junction_regions_are_points(self):
        """Junction regions should be point type."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        for name, region in maze.env_2d.regions.items():
            if name.startswith("junction_"):
                assert region.kind == "point", f"{name} should be a point region"

    def test_corner_box_positions_correct(self):
        """Corner boxes should extend outward from grid corners."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, box_size=15.0)
        maze = make_crossword_maze(dims=dims)

        # Get corner box regions (all should be polygon regions)
        box_tl = maze.env_2d.regions["box_top_left"]
        box_tr = maze.env_2d.regions["box_top_right"]
        box_bl = maze.env_2d.regions["box_bottom_left"]
        box_br = maze.env_2d.regions["box_bottom_right"]

        # All boxes should be polygons
        assert box_tl.kind == "polygon"
        assert box_tr.kind == "polygon"
        assert box_bl.kind == "polygon"
        assert box_br.kind == "polygon"

        # Check centroids are in expected positions (approximately)
        # Top-left extends left and up from (0, 2*s)
        tl_centroid = box_tl.data.centroid
        assert tl_centroid.x < 0  # Extends left
        assert tl_centroid.y > 2 * dims.grid_spacing  # Extends up

        # Top-right extends right and up from (3*s, 2*s)
        tr_centroid = box_tr.data.centroid
        assert tr_centroid.x > 3 * dims.grid_spacing  # Extends right
        assert tr_centroid.y > 2 * dims.grid_spacing  # Extends up

        # Bottom-left extends left and down from (0, 0)
        bl_centroid = box_bl.data.centroid
        assert bl_centroid.x < 0  # Extends left
        assert bl_centroid.y < 0  # Extends down

        # Bottom-right extends right and down from (3*s, 0)
        br_centroid = box_br.data.centroid
        assert br_centroid.x > 3 * dims.grid_spacing  # Extends right
        assert br_centroid.y < 0  # Extends down

    def test_junction_positions_correct(self):
        """Junction regions should be at expected grid positions."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0)
        maze = make_crossword_maze(dims=dims)

        # Check junction_0_0 (bottom-left grid corner)
        j_0_0 = maze.env_2d.regions["junction_0_0"]
        assert j_0_0.kind == "point"
        assert np.isclose(j_0_0.data[0], 0.0, atol=1.0)
        assert np.isclose(j_0_0.data[1], 0.0, atol=1.0)

        # Check junction_1_1 (middle)
        j_1_1 = maze.env_2d.regions["junction_1_1"]
        assert j_1_1.kind == "point"
        assert np.isclose(j_1_1.data[0], dims.grid_spacing, atol=1.0)
        assert np.isclose(j_1_1.data[1], dims.grid_spacing, atol=1.0)

        # Check junction_2_3 (top-right grid corner)
        j_2_3 = maze.env_2d.regions["junction_2_3"]
        assert j_2_3.kind == "point"
        assert np.isclose(j_2_3.data[0], 3 * dims.grid_spacing, atol=1.0)
        assert np.isclose(j_2_3.data[1], 2 * dims.grid_spacing, atol=1.0)

    def test_include_track_true_creates_env_track(self):
        """include_track=True should create env_track."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None

    def test_include_track_false_no_env_track(self):
        """include_track=False should result in env_track=None."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=False)
        assert maze.env_track is None

    def test_env_track_is_1d(self):
        """env_track should be a 1D linearized environment."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.is_1d

    def test_env_track_is_connected(self):
        """Track graph should be connected."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

    def test_env_track_has_correct_units(self):
        """env_track should have units set to 'cm'."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None
        assert maze.env_track.units == "cm"

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims()

        maze_fine = make_crossword_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_crossword_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=50.0, corridor_width=15.0, box_size=20.0)
        maze = make_crossword_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # X should span: -box_size to 3*spacing + box_size = -20 to 170 = 190 cm
        # Y should span: -box_size to 2*spacing + box_size = -20 to 120 = 140 cm
        assert x_extent > 150  # Larger than default
        assert y_extent > 100  # Larger than default

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        maze_default = make_crossword_maze(dims=None)
        maze_explicit = make_crossword_maze(dims=CrosswordDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins


class TestCrosswordTrackGraph:
    """Tests for the track graph structure of Crossword Maze."""

    def test_track_has_manhattan_connectivity(self):
        """Track graph should have 4-connectivity (Manhattan grid)."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None

        # The track graph should be connected
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

        # Check that the connectivity is preserved (no diagonal connections)
        # After discretization, most bins should have degree 2 (along an edge)
        # Junction bins may have degree 3 or 4 (where edges meet)
        graph = maze.env_track.connectivity

        # Count degrees
        degrees = [graph.degree(n) for n in graph.nodes()]
        # Maximum degree should be at most 4 (at junction points in Manhattan grid)
        assert max(degrees) <= 4  # No bin should have more than 4 neighbors
        # Minimum degree should be at least 1 (no isolated bins)
        assert min(degrees) >= 1

    def test_track_nodes_have_positions(self):
        """All track graph nodes should have position attributes."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None

        # Each node should have a 'pos' attribute
        for node in maze.env_track.connectivity.nodes():
            assert "pos" in maze.env_track.connectivity.nodes[node]

    def test_track_covers_grid_and_boxes(self):
        """Track should cover the grid extent plus corner boxes."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, box_size=15.0)
        maze = make_crossword_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should include boxes: from -box/2 to 3*spacing + box/2
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        assert x_min < 0  # Extends left from grid
        assert x_max > 3 * dims.grid_spacing  # Extends right from grid

        # Y extent should include boxes: from -box/2 to 2*spacing + box/2
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        assert y_min < 0  # Extends down from grid
        assert y_max > 2 * dims.grid_spacing  # Extends up from grid

    def test_track_has_many_bins(self):
        """Track graph should have many bins (discretized from graph edges)."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, box_size=15.0)
        maze = make_crossword_maze(dims=dims, include_track=True, bin_size=2.0)
        assert maze.env_track is not None

        # The track is discretized into bins
        # Should have at least 100 bins
        assert maze.env_track.n_bins > 100  # Many bins from discretization

    def test_track_edge_connectivity_preserved(self):
        """Track graph should preserve the Manhattan connectivity structure."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze(include_track=True)
        assert maze.env_track is not None

        # The underlying graph should be connected (all bins reachable)
        import networkx as nx

        assert nx.is_connected(maze.env_track.connectivity)

        # Bins should have low degree (Manhattan structure preserved)
        degrees = [
            maze.env_track.connectivity.degree(n)
            for n in maze.env_track.connectivity.nodes()
        ]
        # Most bins should have degree 2 (along edges), some have degree > 2 at junctions
        assert max(degrees) <= 4  # No bin should have more than 4 neighbors


class TestCrosswordDocstrings:
    """Tests for docstrings and examples."""

    def test_make_crossword_maze_has_docstring(self):
        """make_crossword_maze should have a docstring."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        assert make_crossword_maze.__doc__ is not None
        assert len(make_crossword_maze.__doc__) > 100  # Reasonable length

    def test_crossword_dims_has_docstring(self):
        """CrosswordDims should have a docstring."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        assert CrosswordDims.__doc__ is not None
