"""A grid built from samples inside a declared arena tiles the arena.

Walls lie on the arena's lattice, so starting the cell edges at the arena's
edge puts the walls on cell faces without hand-set ``dimension_ranges``.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena

POSITIONS = np.random.default_rng(0).uniform(0, 100, (5000, 2))  # cm


def _arena(wall_x: float) -> OpenArena:
    """A 100 cm box with a wall from the south side to y = 60 cm."""
    return OpenArena(
        box(0, 0, 100, 100), walls=(LineString([(wall_x, 0), (wall_x, 60)]),)
    )


def test_the_grid_tiles_the_arena_bounds():
    env = Environment.from_samples(POSITIONS, bin_size=5.0, arena=_arena(50.0))

    assert env.dimension_ranges == ((0.0, 100.0), (0.0, 100.0))
    np.testing.assert_allclose(env.grid_edges[0], np.arange(0.0, 101.0, 5.0))


def test_a_wall_on_the_lattice_blocks_paths_across_it():
    """Crossing the wall at y = 30 cm means walking around its end at 60 cm."""
    env = Environment.from_samples(POSITIONS, bin_size=5.0, arena=_arena(50.0))

    around = env.distance_between(points=[[47.5, 32.5], [52.5, 32.5]])

    assert around > 2 * (60 - 32.5)


def test_explicit_dimension_ranges_still_win():
    env = Environment.from_samples(
        POSITIONS,
        bin_size=5.0,
        arena=OpenArena(box(0, 0, 100, 100)),
        dimension_ranges=[(-10.0, 110.0), (-10.0, 110.0)],
    )

    assert env.grid_edges[0][0] == -10.0
    assert env.grid_edges[0][-1] == 110.0


def test_a_wall_off_the_lattice_names_the_wall_and_the_bin():
    with pytest.raises(ValueError, match=r"wall cuts through bin \d+") as caught:
        Environment.from_samples(POSITIONS, bin_size=5.0, arena=_arena(33.0))

    message = str(caught.value)
    assert "from (33, 0) to (33, 60)" in message
    assert "bin_size that divides the distance" in message
