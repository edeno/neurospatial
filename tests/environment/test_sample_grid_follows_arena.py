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


def test_a_wall_off_the_lattice_can_split_the_bins_it_crosses():
    """A wall at x = 33 cm cuts 5 cm bins; wall_fit='split' splits them."""
    env = Environment.from_samples(
        POSITIONS, bin_size=5.0, arena=_arena(33.0), wall_fit="split"
    )

    split = env.bin_sizes < 25.0
    assert split.sum() == 2 * 12  # the 12 bins the wall fully crosses
    assert env.bin_sizes.sum() == pytest.approx(100.0 * 100.0)
    # A path across the wall goes around its end at y = 60 cm.
    assert env.distance_between(points=[[31.0, 30.0], [35.0, 30.0]]) > 2 * (60 - 30)


def test_a_wall_off_the_lattice_is_snapped_by_default():
    with pytest.warns(UserWarning, match=r"moved the walls onto bin edges by up to 2"):
        env = Environment.from_samples(POSITIONS, bin_size=5.0, arena=_arena(33.0))

    assert env.n_bins == 400
    assert env.arena.walls[0].coords[0][0] in (30.0, 35.0)
    # Smoothing, the default rate estimator, still works on the whole bins.
    assert env.smooth(np.ones(env.n_bins), bandwidth=5.0, mode="average").shape == (
        env.n_bins,
    )


def test_an_unknown_wall_fit_is_refused():
    with pytest.raises(ValueError, match="wall_fit must be 'split' or 'snap'"):
        Environment.from_samples(
            POSITIONS, bin_size=5.0, arena=_arena(33.0), wall_fit="move"
        )
