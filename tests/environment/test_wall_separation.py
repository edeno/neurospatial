"""Walls fitted to bins separate the bins on their two sides.

The invariant checked throughout: no open neighbor pair has a center link that
crosses a declared wall an odd number of times, so an animal cannot step
through a wall between neighboring bins, and a wall running from one side of
the arena to the other leaves the bins on each side unconnected.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.geometry._arena_fit import clip_wall

SLANTED = Polygon([(0, 0), (20, 3), (17, 20), (-2, 16)])


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
    faces = env.binning.faces.build()
    centers = env.bin_centers
    pairs = env.binning.connections.pairs
    end = np.flatnonzero(
        np.all(np.isclose(centers[pairs[:, 0]], (17.5, 8.5)), axis=1)
        & np.all(np.isclose(centers[pairs[:, 1]], (17.5, 9.5)), axis=1)
    )
    assert len(end) == 1
    assert faces.A[end[0]] == pytest.approx(1.0)
