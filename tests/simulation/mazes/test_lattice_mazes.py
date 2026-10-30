"""Integration tests for structured lattice mazes (Panel d from Wijnen et al. 2024).

These tests verify that all lattice mazes (Crossword, Honeycomb, Hamlet) and
the complex Rat HexMaze behave as their sources describe; their shared
interface is tested in test_maze_interface.py.
"""

import numpy as np

from neurospatial.simulation.mazes import (
    CrosswordDims,
    HamletDims,
    HoneycombDims,
    RatHexmazeDims,
    make_crossword_maze,
    make_hamlet_maze,
    make_honeycomb_maze,
    make_rat_hexmaze,
)


class TestCrosswordMaze:
    """Test Crossword maze specific behavior."""

    def test_has_default_dims(self):
        """Crossword maze should have the published 5 cm tracks."""
        dims = CrosswordDims()
        assert dims.grid_spacing == 19.0
        assert dims.corridor_width == 5.0
        assert dims.box_size == 16.0

    def test_has_five_cap_regions(self):
        """Crossword has a plastic cap at each of its five protruding tracks."""
        maze = make_crossword_maze(bin_size=3.0)
        cap_regions = [r for r in maze.env_2d.regions if r.startswith("cap_")]
        assert len(cap_regions) == 5

    def test_has_four_start_box_regions(self):
        """Crossword should have 4 start box regions (polygons)."""
        maze = make_crossword_maze(bin_size=3.0)
        box_regions = [r for r in maze.env_2d.regions if r.startswith("start_box_")]
        assert len(box_regions) == 4

    def test_start_box_regions_are_polygons(self):
        """All start box regions should be polygon type."""
        maze = make_crossword_maze(bin_size=3.0)
        for name in [
            "start_box_top_left",
            "start_box_top_right",
            "start_box_bottom_left",
            "start_box_bottom_right",
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

    def test_has_five_house_regions(self):
        """Hamlet should have 5 houses (goal boxes) and the agora."""
        maze = make_hamlet_maze(bin_size=2.0)
        house_regions = [r for r in maze.env_2d.regions if r.endswith("_house")]
        assert len(house_regions) == 5
        assert "agora" in maze.env_2d.regions

    def test_has_ring_and_split_regions(self):
        """Hamlet marks where each radial alley leaves the ring and splits."""
        maze = make_hamlet_maze(bin_size=2.0)
        ring_regions = [r for r in maze.env_2d.regions if r.startswith("ring_")]
        split_regions = [r for r in maze.env_2d.regions if r.startswith("split_")]
        assert len(ring_regions) == len(split_regions) == 5

    def test_default_5_arms(self):
        """Hamlet should have 5 arms by default."""
        dims = HamletDims()
        assert dims.n_arms == 5


class TestRatHexMaze:
    """Test Rat HexMaze specific behavior."""

    def test_has_4_modules(self):
        """Rat HexMaze has 4 modules (A, B, C, D), as in Wijnen et al. 2024."""
        maze = make_rat_hexmaze(bin_size=5.0)
        for module in ["module_A", "module_B", "module_C", "module_D"]:
            assert module in maze.env_2d.regions

    def test_default_4_modules(self):
        """Rat HexMaze has 4 modules by default."""
        dims = RatHexmazeDims()
        assert dims.n_modules == 4

    def test_has_96_choice_points(self):
        """Every one of the 96 nodes is a track-graph node."""
        maze = make_rat_hexmaze(bin_size=5.0)
        graph = maze.env_track.binning.linearization.graph
        assert graph.number_of_nodes() == 96


class TestLatticeMazesComplexity:
    """Test relative complexity of lattice mazes."""

    def test_rat_hexmaze_is_large(self):
        """Rat HexMaze spans about 9 m."""
        rat_hex = make_rat_hexmaze(bin_size=4.0)
        extent = np.ptp(rat_hex.env_2d.bin_centers, axis=0)
        assert extent[0] > 850.0

    def test_honeycomb_has_reasonable_connectivity(self):
        """Honeycomb should have reasonable average connectivity."""

        maze = make_honeycomb_maze(bin_size=4.0)
        graph = maze.env_track.connectivity
        avg_degree = sum(d for n, d in graph.degree()) / graph.number_of_nodes()
        # Hexagonal lattice has varying connectivity (center nodes have more neighbors)
        assert avg_degree > 1.5  # At least some connectivity
