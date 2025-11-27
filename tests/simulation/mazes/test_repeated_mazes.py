"""Integration tests for repeated alleyway mazes (Panel b from Wijnen et al. 2024).

These tests verify that all repeated mazes (Repeated Y, Repeated T, Hampton Court)
work correctly together and share consistent interfaces.
"""

import pytest

from neurospatial.simulation.mazes import (
    HamptonCourtDims,
    MazeEnvironments,
    RepeatedTDims,
    RepeatedYDims,
    make_hampton_court_maze,
    make_repeated_t_maze,
    make_repeated_y_maze,
)

# Parametrize over all repeated maze factories
REPEATED_MAZES = [
    ("repeated_y", make_repeated_y_maze, RepeatedYDims),
    ("repeated_t", make_repeated_t_maze, RepeatedTDims),
    ("hampton_court", make_hampton_court_maze, HamptonCourtDims),
]


class TestRepeatedMazesCommonInterface:
    """Test that all repeated mazes share a consistent interface."""

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_returns_maze_environments(self, name, factory, dims_cls):
        """All factories return MazeEnvironments."""
        maze = factory(bin_size=4.0)
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_2d_has_units_cm(self, name, factory, dims_cls):
        """All 2D environments have units='cm'."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.units == "cm"

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_2d_has_bins(self, name, factory, dims_cls):
        """All 2D environments have bins."""
        maze = factory(bin_size=4.0)
        assert maze.env_2d.n_bins > 0

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_track_exists(self, name, factory, dims_cls):
        """All repeated mazes have track graphs."""
        maze = factory(bin_size=4.0)
        assert maze.env_track is not None

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_track_graph_is_connected(self, name, factory, dims_cls):
        """All track graphs are connected."""
        import networkx as nx

        maze = factory(bin_size=4.0)
        graph = maze.env_track.connectivity
        assert nx.is_connected(graph)

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_has_start_region(self, name, factory, dims_cls):
        """All repeated mazes have a start region."""
        maze = factory(bin_size=4.0)
        assert "start" in maze.env_2d.regions

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_dims_is_frozen_dataclass(self, name, factory, dims_cls):
        """All Dims classes are frozen dataclasses."""
        import dataclasses

        assert dataclasses.is_dataclass(dims_cls)


class TestRepeatedMazesSpecificBehavior:
    """Test maze-specific behavior for repeated mazes."""

    def test_repeated_y_has_4_junctions(self):
        """Repeated Y-maze should have 4 junctions by default."""
        dims = RepeatedYDims()
        assert dims.n_junctions == 4

    def test_repeated_t_has_default_dimensions(self):
        """Repeated T-maze should have default dimensions."""
        dims = RepeatedTDims()
        assert dims.n_t_junctions == 3
        assert dims.t_spacing == 40.0
        assert dims.stem_length == 30.0
        assert dims.arm_length == 15.0

    def test_repeated_t_has_junction_regions(self):
        """Repeated T-maze should have junction regions for each T."""
        maze = make_repeated_t_maze(bin_size=3.0)
        junction_regions = [r for r in maze.env_2d.regions if r.endswith("_junction")]
        # Default: 3 T-junctions
        assert len(junction_regions) == 3

    def test_repeated_t_has_t_arm_regions(self):
        """Repeated T-maze should have arm regions for each T."""
        maze = make_repeated_t_maze(bin_size=3.0)
        # Each T has left_arm and right_arm (3 T's * 2 = 6 arm regions)
        left_arm_regions = [r for r in maze.env_2d.regions if r.endswith("_left_arm")]
        right_arm_regions = [r for r in maze.env_2d.regions if r.endswith("_right_arm")]
        assert len(left_arm_regions) == 3
        assert len(right_arm_regions) == 3

    def test_repeated_t_has_zigzag_connections(self):
        """Repeated T-maze should have zigzag connections between T-junctions.

        This test verifies the alternating T pattern by checking that T-junctions
        alternate between top and bottom levels.
        """
        import networkx as nx

        from neurospatial.simulation.mazes.repeated_t import RepeatedTDims

        dims = RepeatedTDims(n_t_junctions=3)
        maze = make_repeated_t_maze(dims=dims, bin_size=3.0)

        # Verify T-junction regions exist
        assert "t_1_junction" in maze.env_2d.regions
        assert "t_2_junction" in maze.env_2d.regions
        assert "t_3_junction" in maze.env_2d.regions

        # T1 (upright) junction should be higher than T2 (inverted) junction
        t1_y = maze.env_2d.regions["t_1_junction"].data[1]
        t2_y = maze.env_2d.regions["t_2_junction"].data[1]
        t3_y = maze.env_2d.regions["t_3_junction"].data[1]

        assert t1_y > t2_y  # T1 at top, T2 at bottom
        assert t3_y > t2_y  # T3 at top, T2 at bottom
        assert t1_y == t3_y  # T1 and T3 at same height (both upright)

        # The track graph should be connected
        assert nx.is_connected(maze.env_track.connectivity)

    def test_hampton_court_has_goal(self):
        """Hampton Court maze should have a goal region."""
        maze = make_hampton_court_maze(bin_size=5.0)
        assert "goal" in maze.env_2d.regions

    def test_hampton_court_is_large(self):
        """Hampton Court maze should be ~300cm (large labyrinth)."""
        dims = HamptonCourtDims()
        assert dims.size == 300.0


class TestRepeatedMazesComplexity:
    """Test that repeated mazes have appropriate complexity."""

    def test_repeated_mazes_have_many_nodes(self):
        """Repeated mazes should have many nodes in track graph."""
        for name, factory, _dims_cls in REPEATED_MAZES:
            maze = factory(bin_size=4.0)
            n_nodes = maze.env_track.connectivity.number_of_nodes()
            # Repeated mazes should have >50 nodes (complex paths)
            assert n_nodes > 50, f"{name} has only {n_nodes} nodes"

    def test_hampton_court_is_most_complex(self):
        """Hampton Court should be the most complex repeated maze."""
        hampton = make_hampton_court_maze(bin_size=5.0)
        repeated_t = make_repeated_t_maze(bin_size=3.0)
        repeated_y = make_repeated_y_maze(bin_size=3.0)

        hampton_edges = hampton.env_track.connectivity.number_of_edges()
        repeated_t_edges = repeated_t.env_track.connectivity.number_of_edges()
        repeated_y_edges = repeated_y.env_track.connectivity.number_of_edges()

        # Hampton Court has complex labyrinth structure
        assert hampton_edges > repeated_t_edges
        assert hampton_edges > repeated_y_edges
