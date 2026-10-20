"""Tests for the Small Hex Maze (Krausz et al. 2023; Berke lab hexmaze package).

The maze is a 1.30 m equilateral triangle divided into 49 hexagonal units with
reward ports at hexes 1, 2 and 3. Hex numbering, centres and the 63 passages of
the empty maze follow the Berke lab's ``hexmaze`` package; movable barriers
remove hexes from the floor and from the track.
"""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims
from neurospatial.simulation.mazes.hex_small import (
    SmallHexDims,
    _hex_centroids,
    _maze_graph,
    _platform,
    make_small_hex_maze,
)

SPACING = np.sqrt(3) * 7.0  # distance between adjacent hex centres, cm

# Each port has two open half-hexes beside it, named in hexmaze after the hex
# in front of the port: the fourth corner of the rhombus (port, front, other).
_PORT_HALF_HEXES = {4: (1, (6, 5)), 49: (2, (47, 38)), 48: (3, (33, 43))}


@pytest.fixture(scope="module")
def maze():
    return make_small_hex_maze()


@pytest.fixture(scope="module")
def centroids():
    """Hex centres in cm, centred like the built maze."""
    cent = _hex_centroids(14.0)
    middle = np.mean([cent[h] for h in (1, 2, 3)], axis=0)
    return {h: c - middle for h, c in cent.items()}


def _floor_hex_passages(env, centroids, barriers):
    """Pairs of open hexes joined by a bin-to-bin connection of the floor.

    Each bin belongs to the nearest open hex or open half-hex; a half-hex
    counts as the hex in front of its port.
    """
    owners, points = [], []
    for hex_id in sorted(_maze_graph(barriers).nodes):
        owners.append(hex_id)
        points.append(centroids[hex_id])
    for front, (port, others) in _PORT_HALF_HEXES.items():
        for other in others:
            owners.append(front)
            points.append(centroids[port] + centroids[other] - centroids[front])
    owners_arr, points_arr = np.array(owners), np.array(points)
    nearest = np.linalg.norm(
        env.bin_centers[:, None] - points_arr[None], axis=-1
    ).argmin(axis=1)
    owner = owners_arr[nearest]
    return {
        tuple(sorted((int(owner[u]), int(owner[v]))))
        for u, v in env.connectivity.edges
        if owner[u] != owner[v]
    }


def _track_segments(track):
    """Start and end of each track edge, from the bins along it."""
    edge_ids = np.array(
        [track.connectivity.nodes[n]["source_edge_id"] for n in range(track.n_bins)]
    )
    lengths, ends = [], []
    for edge_id in np.unique(edge_ids):
        on_edge = np.flatnonzero(edge_ids == edge_id)
        centers = track.bin_centers[on_edge]
        lengths.append(track.bin_sizes[on_edge].sum())
        direction = centers[-1] - centers[0]
        direction /= np.linalg.norm(direction)
        half = track.bin_sizes[on_edge][[0, -1]] / 2
        ends.append(
            (centers[0] - half[0] * direction, centers[-1] + half[1] * direction)
        )
    return np.array(lengths), ends


class TestSmallHexDims:
    def test_inherits_from_maze_dims(self):
        assert isinstance(SmallHexDims(), MazeDims)

    def test_defaults_are_the_berke_maze(self):
        dims = SmallHexDims()
        assert dims.hex_side == 7.0
        assert len(dims.barriers) == 9

    def test_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            SmallHexDims().hex_side = 8.0  # type: ignore[misc]

    def test_barriers_given_as_a_set_are_stored_frozen(self):
        dims = SmallHexDims(barriers={9, 11, 18, 31, 34, 45})
        assert dims.barriers == frozenset({9, 11, 18, 31, 34, 45})
        assert isinstance(dims.barriers, frozenset)

    @pytest.mark.parametrize("hex_side", [0.0, -7.0, float("nan")])
    def test_non_positive_hex_side_raises(self, hex_side):
        with pytest.raises(ValueError, match="hex_side must be a positive"):
            SmallHexDims(hex_side=hex_side)

    @pytest.mark.parametrize("barrier", [1, 2, 3, 4, 49, 38, 50, 0])
    def test_barrier_on_or_next_to_a_port_or_outside_the_maze_raises(self, barrier):
        with pytest.raises(ValueError, match=rf"barriers \[{barrier}\] are not hexes"):
            SmallHexDims(barriers={barrier, 20})

    def test_non_integer_barrier_raises(self):
        with pytest.raises(TypeError):
            SmallHexDims(barriers={7.5})  # type: ignore[arg-type]

    def test_barriers_that_cut_off_a_hex_raise(self):
        # Hex 44's only open neighbours are 39 and 40.
        with pytest.raises(ValueError, match="cut off open hex 44 "):
            SmallHexDims(barriers={39, 40})

    def test_barriers_that_cut_off_a_region_name_every_hex(self):
        # Hexes 40, 44 and 45 connect to the rest only through 35, 39 and 41.
        with pytest.raises(ValueError, match="cut off open hexes 40, 44, 45 "):
            SmallHexDims(barriers={35, 39, 41})


class TestHexLattice:
    def test_49_hexes_on_the_berke_grid(self):
        cent = _hex_centroids(1.0)
        assert sorted(cent) == list(range(1, 50))
        # 49 distinct centres.
        assert len({tuple(np.round(c, 9)) for c in cent.values()}) == 49

    def test_empty_maze_passages_are_the_geometric_neighbours(self):
        cent = _hex_centroids(1.0)
        neighbours = {
            (a, b)
            for a, b in itertools.combinations(sorted(cent), 2)
            if np.isclose(np.linalg.norm(cent[a] - cent[b]), np.sqrt(3) / 2)
        }
        empty = _maze_graph(frozenset())
        assert {tuple(sorted(e)) for e in empty.edges} == neighbours
        assert empty.number_of_edges() == 63
        assert max(d for _, d in empty.degree()) == 3

    def test_ports_are_126_cm_apart_at_the_triangle_corners(self, maze):
        ports = np.array(
            [maze.env_2d.regions[p].data for p in ("port_A", "port_B", "port_C")]
        )
        sides = np.linalg.norm(ports - np.roll(ports, 1, axis=0), axis=1)
        np.testing.assert_allclose(sides, 18 * 7.0)
        np.testing.assert_allclose(ports.mean(axis=0), 0.0, atol=1e-9)
        # Port A at the top, B bottom left, C bottom right.
        assert ports[0, 1] > 0 > ports[1, 1]
        assert ports[1, 0] < 0 < ports[2, 0]

    def test_platform_sides_are_about_130_cm(self, centroids):
        xy = np.asarray(_platform(centroids, 7.0).exterior.coords)[:-1]
        sides = np.sort(np.linalg.norm(xy - np.roll(xy, 1, axis=0), axis=1))
        # Three long walls about 1.30 m, three corners cut at a hex side.
        np.testing.assert_allclose(sides[:3], 7.0)
        np.testing.assert_allclose(sides[3:], 9.5 * 14.0)
        assert abs(sides[3] - 130.0) < 5.0

    def test_hex_side_scales_the_maze(self):
        small = make_small_hex_maze(SmallHexDims(hex_side=7.0), include_track=False)
        large = make_small_hex_maze(SmallHexDims(hex_side=10.0), include_track=False)
        ratio = np.ptp(large.env_2d.bin_centers, axis=0) / np.ptp(
            small.env_2d.bin_centers, axis=0
        )
        np.testing.assert_allclose(ratio, 10.0 / 7.0, rtol=0.05)


class TestFloor:
    def test_units_and_port_regions(self, maze):
        assert maze.env_2d.units == "cm"
        assert sorted(maze.env_2d.regions) == ["port_A", "port_B", "port_C"]
        assert all(r.kind == "point" for r in maze.env_2d.regions.values())

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 3.0, 4.0])
    def test_floor_is_one_piece(self, bin_size):
        env = make_small_hex_maze(bin_size=bin_size, include_track=False).env_2d
        assert len(env.components()) == 1

    @pytest.mark.parametrize(
        "barriers",
        [SmallHexDims().barriers, frozenset(), frozenset({9, 11, 18, 31, 34, 45})],
    )
    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 4.0])
    def test_floor_connects_exactly_the_open_neighbouring_hexes(
        self, centroids, barriers, bin_size
    ):
        env = make_small_hex_maze(
            SmallHexDims(barriers=barriers), bin_size=bin_size, include_track=False
        ).env_2d
        expected = {tuple(sorted(e)) for e in _maze_graph(barriers).edges}
        assert _floor_hex_passages(env, centroids, barriers) == expected

    def test_barrier_hexes_have_no_floor(self, maze, centroids):
        barrier_centres = np.array([centroids[h] for h in SmallHexDims().barriers])
        assert not maze.env_2d.contains(barrier_centres).any()
        open_centres = np.array(
            [centroids[h] for h in _maze_graph(SmallHexDims().barriers).nodes]
        )
        assert maze.env_2d.contains(open_centres).all()

    def test_finer_bins_give_more_bins(self):
        fine = make_small_hex_maze(bin_size=1.0, include_track=False)
        coarse = make_small_hex_maze(bin_size=5.0, include_track=False)
        assert fine.env_2d.n_bins > coarse.env_2d.n_bins


class TestTrack:
    def test_include_track(self, maze):
        assert maze.env_track is not None
        assert maze.env_track.is_linearized_track
        assert maze.env_track.units == "cm"
        assert make_small_hex_maze(include_track=False).env_track is None

    def test_one_edge_per_passage_each_one_hex_spacing_long(self, maze, centroids):
        lengths, ends = _track_segments(maze.env_track)
        hexes = _maze_graph(SmallHexDims().barriers)
        assert len(lengths) == hexes.number_of_edges() == 39
        np.testing.assert_allclose(lengths, SPACING)
        # Every edge joins the centres of two neighbouring open hexes.
        names = np.array(sorted(hexes.nodes))
        points = np.array([centroids[h] for h in names])
        joined = set()
        for start, end in ends:
            pair = [
                names[np.linalg.norm(points - p, axis=1).argmin()] for p in (start, end)
            ]
            for p, h in zip((start, end), pair, strict=True):
                np.testing.assert_allclose(p, centroids[h], atol=1e-6)
            joined.add(tuple(sorted(int(h) for h in pair)))
        assert joined == {tuple(sorted(e)) for e in hexes.edges}

    def test_at_most_three_passages_per_hex(self, maze):
        _, ends = _track_segments(maze.env_track)
        graph = nx.Graph()
        graph.add_edges_from(
            (tuple(np.round(s, 6)), tuple(np.round(e, 6))) for s, e in ends
        )
        assert max(d for _, d in graph.degree()) <= 3
        assert nx.is_connected(graph)

    def test_track_lies_on_the_floor(self, maze):
        assert maze.env_2d.contains(maze.env_track.bin_centers).all()

    def test_track_is_connected(self, maze):
        assert nx.is_connected(maze.env_track.connectivity)
