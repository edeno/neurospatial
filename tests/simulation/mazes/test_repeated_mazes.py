"""Integration tests for repeated alleyway mazes (Panel b from Wijnen et al. 2024).

These tests verify that all repeated mazes (Repeated Y, Repeated T, Hampton Court)
work correctly together and share consistent interfaces.
"""

import networkx as nx
import numpy as np
import pytest
import shapely
from shapely.ops import nearest_points

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
        maze = factory()
        assert isinstance(maze, MazeEnvironments)

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_2d_has_units_cm(self, name, factory, dims_cls):
        """All 2D environments have units='cm'."""
        maze = factory()
        assert maze.env_2d.units == "cm"

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_2d_has_bins(self, name, factory, dims_cls):
        """All 2D environments have bins."""
        maze = factory()
        assert maze.env_2d.n_bins > 0

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_env_track_exists(self, name, factory, dims_cls):
        """All repeated mazes have track graphs."""
        maze = factory()
        assert maze.env_track is not None

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_track_graph_is_connected(self, name, factory, dims_cls):
        """All track graphs are connected."""
        maze = factory()
        graph = maze.env_track.connectivity
        assert nx.is_connected(graph)

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_has_start_region(self, name, factory, dims_cls):
        """All repeated mazes have a start region."""
        maze = factory()
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
        # RepeatedTDims has stem_spacing (not t_spacing/n_t_junctions)
        assert dims.stem_spacing == 40.0
        assert dims.stem_length == 60.0
        assert dims.arm_length == 15.0
        assert dims.top_spur_length == 20.0
        assert dims.upper_lower_gap == 30.0
        assert dims.width == 10.0

    def test_repeated_t_has_junction_regions(self):
        """Repeated T-maze should have junction regions for each T."""
        maze = make_repeated_t_maze(bin_size=3.0)
        junction_regions = [r for r in maze.env_2d.regions if r.endswith("_junction")]
        # Has 4 junction regions: upper_junction, lower_left_junction,
        # lower_center_junction, upper_right_junction
        assert len(junction_regions) == 4
        expected = {
            "upper_junction",
            "lower_left_junction",
            "lower_center_junction",
            "upper_right_junction",
        }
        assert set(junction_regions) == expected

    def test_repeated_t_has_t_arm_regions(self):
        """Repeated T-maze should have arm regions extending from stems."""
        maze = make_repeated_t_maze(bin_size=3.0)
        # The maze has lower arm endpoints (lower_left_arm, lower_center_left_arm)
        # Upper arms terminate at start/goal regions, not separate arm regions
        arm_regions = [r for r in maze.env_2d.regions if "_arm" in r]
        assert len(arm_regions) == 2
        expected = {"lower_left_arm", "lower_center_left_arm"}
        assert set(arm_regions) == expected

    def test_repeated_t_has_zigzag_connections(self):
        """Repeated T-maze should have multi-level connections.

        This test verifies the alternating level pattern by checking that
        junctions exist at both upper and lower horizontal levels.
        """
        import networkx as nx

        maze = make_repeated_t_maze(bin_size=3.0)

        # Verify junction regions exist at both levels
        assert "upper_junction" in maze.env_2d.regions
        assert "upper_right_junction" in maze.env_2d.regions
        assert "lower_left_junction" in maze.env_2d.regions
        assert "lower_center_junction" in maze.env_2d.regions

        # Upper junctions should be at higher y than lower junctions
        upper_y = maze.env_2d.regions["upper_junction"].data[1]
        upper_right_y = maze.env_2d.regions["upper_right_junction"].data[1]
        lower_left_y = maze.env_2d.regions["lower_left_junction"].data[1]
        lower_center_y = maze.env_2d.regions["lower_center_junction"].data[1]

        # Upper level junctions should be higher than lower level junctions
        assert upper_y > lower_left_y
        assert upper_y > lower_center_y
        # Both upper junctions at same height
        assert upper_y == upper_right_y
        # Both lower junctions at same height
        assert lower_left_y == lower_center_y

        # The track graph should be connected
        assert nx.is_connected(maze.env_track.connectivity)

    def test_hampton_court_has_goal(self):
        """Hampton Court maze should have a goal region."""
        maze = make_hampton_court_maze()
        assert "goal" in maze.env_2d.regions

    def test_hampton_court_is_large(self):
        """Small's (1901) maze measured 8 by 6 ft."""
        dims = HamptonCourtDims()
        assert (dims.width, dims.height) == pytest.approx((243.84, 182.88))


class TestRepeatedMazesComplexity:
    """Test that repeated mazes have appropriate complexity."""

    def test_repeated_mazes_have_many_nodes(self):
        """Repeated mazes should have many nodes in track graph."""
        for name, factory, _dims_cls in REPEATED_MAZES:
            maze = factory()
            n_nodes = maze.env_track.connectivity.number_of_nodes()
            # Repeated mazes should have >50 nodes (complex paths)
            assert n_nodes > 50, f"{name} has only {n_nodes} nodes"

    def test_hampton_court_is_most_complex(self):
        """Hampton Court should be the most complex repeated maze."""
        hampton = make_hampton_court_maze()
        repeated_t = make_repeated_t_maze(bin_size=3.0)
        repeated_y = make_repeated_y_maze(bin_size=3.0)

        hampton_edges = hampton.env_track.connectivity.number_of_edges()
        repeated_t_edges = repeated_t.env_track.connectivity.number_of_edges()
        repeated_y_edges = repeated_y.env_track.connectivity.number_of_edges()

        # Hampton Court has complex labyrinth structure
        assert hampton_edges > repeated_t_edges
        assert hampton_edges > repeated_y_edges


class TestTrackMatchesTheFloor:
    """The track graph runs along the 2-D maze's corridors, and walls hold."""

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_every_track_edge_lies_on_the_floor(self, name, factory, dims_cls):
        maze = factory()
        floor = maze.env_2d.arena.polygonal_domain()
        fractions = np.linspace(0.0, 1.0, 101)[:, np.newaxis]
        for start, end in np.asarray(maze.env_track.arena.segments):
            points = start + fractions * (end - start)
            assert shapely.covers(floor, shapely.points(points)).all(), (start, end)
            assert (maze.env_2d.bin_at(points) >= 0).all(), (start, end)

    @pytest.mark.parametrize("name,factory,dims_cls", REPEATED_MAZES)
    def test_walls_separate_corridors(self, name, factory, dims_cls):
        """Places close in the plane but far apart along the track are separated
        by a wall, so walking between them in 2-D goes round it."""
        maze = factory()
        env = maze.env_2d
        segments = np.asarray(maze.env_track.arena.segments)
        keys = [tuple(np.round(p, 6)) for p in segments.reshape(-1, 2)]
        graph = nx.Graph()
        for k, (start, end) in enumerate(segments):
            graph.add_edge(
                keys[2 * k], keys[2 * k + 1], weight=float(np.linalg.norm(end - start))
            )
        along = dict(nx.all_pairs_dijkstra_path_length(graph))
        lines = shapely.linestrings(segments)
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                p, q = (
                    np.array(g.coords[0]) for g in nearest_points(lines[i], lines[j])
                )
                gap = float(np.linalg.norm(q - p))
                if gap == 0.0 or gap > 30.0:
                    continue
                on_track = min(
                    np.linalg.norm(p - segments[i, a])
                    + along[keys[2 * i + a]][keys[2 * j + b]]
                    + np.linalg.norm(segments[j, b] - q)
                    for a in range(2)
                    for b in range(2)
                )
                if on_track <= 3.0 * gap:
                    continue  # same corridor, or round a corner
                bin_p, bin_q = env.bin_at(np.array([p, q]))
                walk = nx.shortest_path_length(
                    env.connectivity, bin_p, bin_q, weight="distance"
                )
                assert walk > 1.5 * gap, (name, p, q)
