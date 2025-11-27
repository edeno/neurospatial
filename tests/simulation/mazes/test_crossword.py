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
        assert dims.n_rows == 4
        assert dims.n_cols == 4

    def test_is_frozen(self):
        """CrosswordDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims()
        with pytest.raises(FrozenInstanceError):
            dims.grid_spacing = 50.0  # type: ignore[misc]

    def test_custom_values(self):
        """CrosswordDims should accept custom values."""
        from neurospatial.simulation.mazes.crossword import CrosswordDims

        dims = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, n_rows=5, n_cols=5)
        assert dims.grid_spacing == 40.0
        assert dims.corridor_width == 12.0
        assert dims.n_rows == 5
        assert dims.n_cols == 5


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
        """env_2d should have a 4×4 grid spatial extent."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, corridor_width=10.0, n_rows=4, n_cols=4)
        maze = make_crossword_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X extent should cover 4 columns (0 to 3*spacing)
        x_extent = x_max - x_min
        expected_x_extent = 3 * dims.grid_spacing  # 0 to 90 cm
        assert x_extent > expected_x_extent * 0.8  # At least 80% of expected

        # Y extent should cover 4 rows (0 to 3*spacing)
        y_extent = y_max - y_min
        expected_y_extent = 3 * dims.grid_spacing  # 0 to 90 cm
        assert y_extent > expected_y_extent * 0.8  # At least 80% of expected

    def test_env_2d_has_corner_boxes(self):
        """env_2d should have 4 corner box regions (box_0 through box_3)."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        assert "box_0" in maze.env_2d.regions  # Top-left
        assert "box_1" in maze.env_2d.regions  # Top-right
        assert "box_2" in maze.env_2d.regions  # Bottom-right
        assert "box_3" in maze.env_2d.regions  # Bottom-left

    def test_env_2d_has_node_regions(self):
        """env_2d should have node_i_j regions at intersections."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        # Check a few node regions exist
        assert "node_0_0" in maze.env_2d.regions  # Bottom-left corner
        assert "node_1_1" in maze.env_2d.regions  # Interior node
        assert "node_3_3" in maze.env_2d.regions  # Top-right corner

    def test_sparse_node_regions_present(self):
        """Sparse grid nodes should have regions (not all 16 nodes present)."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        # Crossword maze is a sparse/incomplete grid - not all nodes present
        # The actual nodes present in the sparse grid:
        expected_nodes = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),  # Bottom row (full)
            (1, 0),
            (1, 1),
            (1, 2),  # Row 1 (partial - missing 1,3)
            (2, 1),
            (2, 2),
            (2, 3),  # Row 2 (partial - missing 2,0)
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),  # Top row (full)
        ]
        for row, col in expected_nodes:
            region_name = f"node_{row}_{col}"
            assert region_name in maze.env_2d.regions

    def test_missing_nodes_in_sparse_grid(self):
        """Some nodes should be missing in the incomplete grid."""
        from neurospatial.simulation.mazes.crossword import make_crossword_maze

        maze = make_crossword_maze()
        # These nodes should NOT be present in the sparse grid
        # (based on incomplete corridor structure)
        assert "node_1_3" not in maze.env_2d.regions  # Row 1 right gap
        assert "node_2_0" not in maze.env_2d.regions  # Row 2 left gap

    def test_corner_box_positions_correct(self):
        """Corner boxes should be at expected positions."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, n_rows=4, n_cols=4)
        maze = make_crossword_maze(dims=dims)

        # Get corner box regions (all should be point regions)
        box_0 = maze.env_2d.regions["box_0"]
        box_1 = maze.env_2d.regions["box_1"]
        box_2 = maze.env_2d.regions["box_2"]
        box_3 = maze.env_2d.regions["box_3"]

        assert box_0.kind == "point"
        assert box_1.kind == "point"
        assert box_2.kind == "point"
        assert box_3.kind == "point"

        # box_0: top-left (0, 3*spacing)
        assert np.isclose(box_0.data[0], 0.0, atol=1.0)
        assert np.isclose(box_0.data[1], 3 * dims.grid_spacing, atol=1.0)

        # box_1: top-right (3*spacing, 3*spacing)
        assert np.isclose(box_1.data[0], 3 * dims.grid_spacing, atol=1.0)
        assert np.isclose(box_1.data[1], 3 * dims.grid_spacing, atol=1.0)

        # box_2: bottom-right (3*spacing, 0)
        assert np.isclose(box_2.data[0], 3 * dims.grid_spacing, atol=1.0)
        assert np.isclose(box_2.data[1], 0.0, atol=1.0)

        # box_3: bottom-left (0, 0)
        assert np.isclose(box_3.data[0], 0.0, atol=1.0)
        assert np.isclose(box_3.data[1], 0.0, atol=1.0)

    def test_node_positions_correct(self):
        """Node regions should be at expected grid positions."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, n_rows=4, n_cols=4)
        maze = make_crossword_maze(dims=dims)

        # Check node_0_0 (bottom-left)
        node_0_0 = maze.env_2d.regions["node_0_0"]
        assert node_0_0.kind == "point"
        assert np.isclose(node_0_0.data[0], 0.0, atol=1.0)
        assert np.isclose(node_0_0.data[1], 0.0, atol=1.0)

        # Check node_2_1 (row=2, col=1)
        node_2_1 = maze.env_2d.regions["node_2_1"]
        assert node_2_1.kind == "point"
        assert np.isclose(node_2_1.data[0], 1 * dims.grid_spacing, atol=1.0)
        assert np.isclose(node_2_1.data[1], 2 * dims.grid_spacing, atol=1.0)

        # Check node_3_3 (top-right corner)
        node_3_3 = maze.env_2d.regions["node_3_3"]
        assert node_3_3.kind == "point"
        assert np.isclose(node_3_3.data[0], 3 * dims.grid_spacing, atol=1.0)
        assert np.isclose(node_3_3.data[1], 3 * dims.grid_spacing, atol=1.0)

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

        dims = CrosswordDims(grid_spacing=50.0, corridor_width=15.0, n_rows=5, n_cols=5)
        maze = make_crossword_maze(dims=dims)

        # Check that bin_centers span approximately the expected range
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Both should be roughly (n-1)*grid_spacing = 4*50 = 200 cm
        assert x_extent > 180  # Larger than default
        assert y_extent > 180  # Larger than default

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

    def test_track_covers_full_maze(self):
        """Track should cover the full 4×4 grid extent."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, n_rows=4, n_cols=4)
        maze = make_crossword_maze(dims=dims, include_track=True)
        assert maze.env_track is not None

        # Get the positions of all nodes
        graph = maze.env_track.connectivity
        positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

        # X extent should cover the full grid (0 to 3*spacing)
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        expected_x = 3 * dims.grid_spacing  # 0 to 90 cm
        assert x_extent >= expected_x * 0.95  # At least 95%

        # Y extent should cover the full grid (0 to 3*spacing)
        y_extent = positions[:, 1].max() - positions[:, 1].min()
        expected_y = 3 * dims.grid_spacing  # 0 to 90 cm
        assert y_extent >= expected_y * 0.95  # At least 95%

    def test_track_has_many_bins(self):
        """Track graph should have many bins (discretized from graph edges)."""
        from neurospatial.simulation.mazes.crossword import (
            CrosswordDims,
            make_crossword_maze,
        )

        dims = CrosswordDims(grid_spacing=30.0, n_rows=4, n_cols=4)
        maze = make_crossword_maze(dims=dims, include_track=True, bin_size=2.0)
        assert maze.env_track is not None

        # The track is discretized into bins (many more than the 16 original nodes)
        # With grid_spacing=30.0 and bin_size=2.0, we expect roughly:
        # Total edge length = 12 horizontal edges × 30 + 12 vertical edges × 30 = 720 cm
        # 720 cm / 2 cm per bin ≈ 360 bins
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
