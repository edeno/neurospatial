"""Walls fitted to bins separate the bins on their two sides.

The invariant checked throughout: no open neighbor pair has a center link that
crosses a declared wall an odd number of times, so an animal cannot step
through a wall between neighboring bins, and a wall running from one side of
the arena to the other leaves the bins on each side unconnected.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import KnownFaces
from neurospatial.geometry._arena_fit import _wall_sides, clip_wall

SLANTED = Polygon([(0, 0), (20, 3), (17, 20), (-2, 16)])
ROUND = Point(10, 10).buffer(9.3, quad_segs=32)


def _build(outline, wall, **options):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "wall_fit='snap' moved", UserWarning)
        return Environment.from_polygon(
            outline,
            bin_size=1.0,
            arena=OpenArena(outline, walls={"w": wall}),
            **options,
        )


def _leaking_pairs(env, wall):
    """Open pairs whose center link crosses ``wall`` an odd number of times."""
    centers = env.bin_centers
    pairs = env.binning.connections.pairs
    links = shapely.linestrings(
        np.stack((centers[pairs[:, 0]], centers[pairs[:, 1]]), 1)
    )
    hits = shapely.intersection(links, wall)
    crossings = np.where(shapely.is_empty(hits), 0, shapely.get_num_geometries(hits))
    return pairs[crossings % 2 == 1]


def _separated(env, line):
    """Whether no path joins bins whose centers lie on opposite sides of ``line``.

    ``line`` runs across the whole arena: the wall before it was clipped.
    """
    graph = env.to_networkx()
    sides = shapely.ops.split(env.arena.polygonal_domain(), line).geoms
    assert len(sides) == 2
    labels = [
        set(np.flatnonzero(shapely.covers(side, shapely.points(env.bin_centers))))
        for side in sides
    ]
    component = {
        node: number
        for number, nodes in enumerate(nx.connected_components(graph))
        for node in nodes
    }
    return {component[node] for node in labels[0]}.isdisjoint(
        {component[node] for node in labels[1]}
    )


def test_a_snapped_wall_ending_on_a_slanted_outline_separates_its_sides():
    # Bins whose centers are inside a slanted outline stick out past it; the
    # wall snapped onto their edges is clipped at the outline, so it covers
    # only the part of their shared edge inside the arena, which must close
    # the pair.
    line = LineString([(-5, 10.8), (25, 8.6)])
    (wall,) = clip_wall(line, SLANTED)
    env = _build(SLANTED, wall)
    assert len(_leaking_pairs(env, wall)) == 0
    assert _separated(env, line)
    assert nx.number_connected_components(env.to_networkx()) == 2


def test_snapped_walls_across_a_slanted_arena_never_leak():
    rng = np.random.default_rng(1)
    for _ in range(20):
        start, stop = rng.uniform(6, 12, 2)
        line = LineString([(-5, start), (25, stop)])
        (wall,) = clip_wall(line, SLANTED)
        env = _build(SLANTED, wall)
        assert len(_leaking_pairs(env, wall)) == 0, list(wall.coords)
        assert _separated(env, line), list(wall.coords)


def test_a_wall_on_bin_edges_ending_on_a_slanted_outline_separates_its_sides():
    # The wall already runs along bin edges, so nothing is snapped; the end
    # faces stick out past the outline where no wall can be.
    line = LineString([(-5, 9), (25, 9)])
    (wall,) = clip_wall(line, SLANTED)
    env = _build(SLANTED, wall)
    assert len(_leaking_pairs(env, wall)) == 0
    assert _separated(env, line)


def test_a_wall_ending_short_of_a_slanted_outline_leaves_the_gap_open():
    # A wall that stops a bin short of the outline is a doorway, not a wall
    # clipped at the outline: its last face stays open with the open share.
    wall = LineString([(1, 9), (17, 9)])
    env = _build(SLANTED, wall)
    assert not _separated(env, LineString([(-5, 9), (25, 9)]))
    faces = env.binning.build_faces()
    centers = env.bin_centers
    pairs = env.binning.connections.pairs
    end = np.flatnonzero(
        np.all(np.isclose(centers[pairs[:, 0]], (17.5, 8.5)), axis=1)
        & np.all(np.isclose(centers[pairs[:, 1]], (17.5, 9.5)), axis=1)
    )
    assert len(end) == 1
    assert faces.A[end[0]] == pytest.approx(1.0)


def test_points_outside_a_sharp_bend_are_on_the_outer_side():
    # A left turn sharper than a right angle at (1, 0): just past the bend the
    # nearest wall point is the bend itself, and both its segments' lines
    # would give different answers there.
    wall = LineString([(0, 0), (1, 0), (0, 0.2)])
    points = np.array([[1.5, 0.05], [1.5, -0.05], [0.5, 0.05], [0.5, -0.5]])
    # Inside the V is the left of the wall's direction; outside it, the right.
    assert _wall_sides(wall, points).tolist() == [False, False, True, False]
    # Reversing the wall swaps its left and right.
    assert _wall_sides(wall.reverse(), points).tolist() == [True, True, False, True]


def _v_wall(rng):
    apex = rng.uniform([10, 8], [14, 12])
    half = rng.uniform(0.15, 0.35)
    length = rng.uniform(6, 9)
    arm = length * np.array([-np.cos(half), np.sin(half)])
    return LineString([apex + arm, apex, apex + arm * [1, -1]])


def test_snapped_walls_with_sharp_bends_never_leak():
    outline = box(0, 0, 20, 20)
    rng = np.random.default_rng(0)
    for _ in range(40):
        wall = _v_wall(rng)
        env = _build(outline, wall)
        assert len(_leaking_pairs(env, wall)) == 0, list(wall.coords)


def test_split_fits_a_wall_in_a_slanted_arena():
    # wall_fit="split" turns the bins into exact polygon cells, which must lie
    # in the arena, so bins sticking out past the slanted outline are clipped
    # to it; bins inside it keep their shape and center.
    wall = LineString([(8.3, 8.4), (11.7, 9.6)])
    plain = Environment.from_polygon(SLANTED, bin_size=1.0)
    split = _build(SLANTED, wall, wall_fit="split")
    # The wall crosses two bins from edge to edge; each becomes two pieces.
    assert split.n_bins == plain.n_bins + 2
    cells = np.asarray(split.binning.shapes.polygons, dtype=object)
    assert np.all(shapely.area(shapely.difference(cells, SLANTED)) < 1e-12)
    # 1 cm squares inside the outline, away from the wall.
    whole = np.isclose(shapely.area(cells), 1.0) & ~shapely.intersects(cells, wall)
    assert np.count_nonzero(whole) > 250
    plain_index = plain.bin_at(split.bin_centers[whole])
    assert np.all(plain_index >= 0)
    np.testing.assert_allclose(
        split.bin_centers[whole], plain.bin_centers[plain_index], atol=1e-12
    )


@pytest.mark.parametrize("outline", [SLANTED, ROUND], ids=["slanted", "round"])
def test_split_walls_across_an_arena_never_leak(outline):
    rng = np.random.default_rng(1)
    for _ in range(10):
        start, stop = rng.uniform(6, 12, 2)
        line = LineString([(-5, start), (25, stop)])
        (wall,) = clip_wall(line, outline)
        env = _build(outline, wall, wall_fit="split")
        assert len(_leaking_pairs(env, wall)) == 0, list(wall.coords)
        assert _separated(env, line), list(wall.coords)


# Along the bin edge y = 9, then across the in-arena part of the bin centered
# at (18.5, 9.5) to the slanted outline. The whole square is not divided (the
# wall ends inside it), but its part inside the arena is.
BENT_LINE = LineString([(-5, 9), (18, 9), (22, 11)])
(BENT,) = clip_wall(BENT_LINE, SLANTED)


def test_a_wall_dividing_only_the_in_arena_part_of_a_bin_is_refused():
    source = Environment.from_polygon(SLANTED, bin_size=1.0).binning
    source = replace(source, faces=KnownFaces(source.build_faces()))
    with pytest.raises(ValueError, match=r"wall 'w' cuts through bin .*18\.5, 9\.5"):
        Environment.from_bins(source, arena=OpenArena(SLANTED, walls={"w": BENT}))


@pytest.mark.parametrize("wall_fit", ["snap", "split"])
def test_a_wall_dividing_only_the_in_arena_part_of_a_bin_is_fitted(wall_fit):
    env = _build(SLANTED, BENT, wall_fit=wall_fit)
    assert len(_leaking_pairs(env, BENT)) == 0
    assert _separated(env, BENT_LINE)
