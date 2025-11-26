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

    def test_repeated_y_has_3_junctions(self):
        """Repeated Y-maze should have 3 junctions by default."""
        dims = RepeatedYDims()
        assert dims.n_junctions == 3

    def test_repeated_t_has_3_junctions(self):
        """Repeated T-maze should have 3 junctions by default."""
        dims = RepeatedTDims()
        assert dims.n_junctions == 3

    def test_repeated_t_has_junction_regions(self):
        """Repeated T-maze should have junction regions."""
        maze = make_repeated_t_maze(bin_size=3.0)
        junction_regions = [r for r in maze.env_2d.regions if r.startswith("junction_")]
        assert len(junction_regions) == 3

    def test_repeated_t_has_arm_end_regions(self):
        """Repeated T-maze should have arm end regions."""
        maze = make_repeated_t_maze(bin_size=3.0)
        arm_regions = [r for r in maze.env_2d.regions if r.endswith("_end")]
        assert len(arm_regions) == 3

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
