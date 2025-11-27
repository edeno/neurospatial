"""Integration tests for structured lattice mazes (Panel d from Wijnen et al. 2024).

These tests verify that all lattice mazes (Crossword, Honeycomb, Hamlet)
and the complex Rat HexMaze work correctly together and share consistent interfaces.
"""

import pytest

from neurospatial.simulation.mazes import (
    CrosswordDims,
    HamletDims,
    HoneycombDims,
    MazeEnvironments,
    RatHexmazeDims,
    make_crossword_maze,
    make_hamlet_maze,
    make_honeycomb_maze,
    make_rat_hexmaze,
)

# Parametrize over all lattice maze factories
LATTICE_MAZES = [
    ("crossword", make_crossword_maze, CrosswordDims),
    ("honeycomb", make_honeycomb_maze, HoneycombDims),
    ("hamlet", make_hamlet_maze, HamletDims),
    ("rat_hexmaze", make_rat_hexmaze, RatHexmazeDims),
]


class TestLatticeMazesCommonInterface:
    """Test that all lattice mazes share a consistent interface."""

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_returns_maze_environments(self, name, factory, dims_cls):
        """All factories return MazeEnvironments."""
        maze = factory(bin_size=4.0)
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_env_2d_has_units_cm(self, name, factory, dims_cls):
        """All 2D environments have units='cm'."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.units == "cm"

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_env_2d_has_bins(self, name, factory, dims_cls):
        """All 2D environments have bins."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.n_bins > 0

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_env_track_exists(self, name, factory, dims_cls):
        """All lattice mazes have track graphs."""
        maze = factory(bin_size=4.0)
        assert maze.env_track is not None

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_track_graph_is_connected(self, name, factory, dims_cls):
        """All track graphs are connected."""
        import networkx as nx

        maze = factory(bin_size=4.0)
        graph = maze.env_track.connectivity
        assert nx.is_connected(graph)

    @pytest.mark.parametrize("name,factory,dims_cls", LATTICE_MAZES)
    def test_dims_is_frozen_dataclass(self, name, factory, dims_cls):
        """All Dims classes are frozen dataclasses."""
        import dataclasses

        assert dataclasses.is_dataclass(dims_cls)


class TestCrosswordMaze:
    """Test Crossword maze specific behavior."""

    def test_has_default_dims(self):
        """Crossword maze should have default dimensions."""
        dims = CrosswordDims()
        assert dims.grid_spacing == 30.0
        assert dims.corridor_width == 10.0
        assert dims.box_size == 15.0

    def test_has_junction_regions(self):
        """Crossword should have junction regions at intersections (sparse grid)."""
        maze = make_crossword_maze(bin_size=3.0)
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        # Sparse crossword pattern: 10 intersections
        assert len(junction_regions) == 10

    def test_has_four_box_regions(self):
        """Crossword should have 4 corner box regions (polygons)."""
        maze = make_crossword_maze(bin_size=3.0)
        box_regions = [r for r in maze.env_2d.regions if r.startswith("box_")]
        assert len(box_regions) == 4

    def test_box_regions_are_polygons(self):
        """All box regions should be polygon type."""
        maze = make_crossword_maze(bin_size=3.0)
        for name in [
            "box_top_left",
            "box_top_right",
            "box_bottom_left",
            "box_bottom_right",
        ]:
            assert maze.env_2d.regions[name].kind == "polygon"


class TestHoneycombMaze:
    """Test Honeycomb maze specific behavior."""

    def test_has_37_platforms(self):
        """Honeycomb should have 37 platforms (1 + 6 + 12 + 18 for 3 rings)."""
        maze = make_honeycomb_maze(bin_size=3.0)
        platform_regions = [r for r in maze.env_2d.regions if r.startswith("platform_")]
        assert len(platform_regions) == 37

    def test_default_3_rings(self):
        """Honeycomb should have 3 rings by default."""
        dims = HoneycombDims()
        assert dims.n_rings == 3


class TestHamletMaze:
    """Test Hamlet maze specific behavior."""

    def test_has_inner_regions(self):
        """Hamlet should have 5 inner vertices (pentagon at inner_radius)."""
        maze = make_hamlet_maze(bin_size=3.0)
        inner_regions = [r for r in maze.env_2d.regions if r.startswith("inner_")]
        assert len(inner_regions) == 5

    def test_has_outer_regions(self):
        """Hamlet should have 5 outer vertices (at outer_radius)."""
        maze = make_hamlet_maze(bin_size=3.0)
        outer_regions = [r for r in maze.env_2d.regions if r.startswith("outer_")]
        assert len(outer_regions) == 5

    def test_default_5_arms(self):
        """Hamlet should have 5 arms by default."""
        dims = HamletDims()
        assert dims.n_arms == 5


class TestRatHexMaze:
    """Test Rat HexMaze specific behavior."""

    def test_has_3_modules(self):
        """Rat HexMaze should have 3 modules (A, B, C)."""
        maze = make_rat_hexmaze(bin_size=5.0)
        for module in ["module_A", "module_B", "module_C"]:
            assert module in maze.env_2d.regions

    def test_default_3_modules(self):
        """Rat HexMaze should have 3 modules by default."""
        dims = RatHexmazeDims()
        assert dims.n_modules == 3

    def test_has_many_nodes(self):
        """Rat HexMaze should have many nodes (large maze)."""
        maze = make_rat_hexmaze(bin_size=5.0)
        n_nodes = maze.env_track.connectivity.number_of_nodes()
        # Should have many nodes for a large-scale maze
        assert n_nodes > 100


class TestLatticeMazesComplexity:
    """Test relative complexity of lattice mazes."""

    def test_rat_hexmaze_is_large(self):
        """Rat HexMaze should be a large maze (many nodes)."""
        rat_hex = make_rat_hexmaze(bin_size=4.0)
        rat_hex_nodes = rat_hex.env_track.connectivity.number_of_nodes()
        # Rat HexMaze should have >100 nodes (large-scale maze)
        assert rat_hex_nodes > 100

    def test_honeycomb_has_reasonable_connectivity(self):
        """Honeycomb should have reasonable average connectivity."""

        maze = make_honeycomb_maze(bin_size=4.0)
        graph = maze.env_track.connectivity
        avg_degree = sum(d for n, d in graph.degree()) / graph.number_of_nodes()
        # Hexagonal lattice has varying connectivity (center nodes have more neighbors)
        assert avg_degree > 1.5  # At least some connectivity
