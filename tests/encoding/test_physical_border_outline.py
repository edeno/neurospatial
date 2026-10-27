"""Physical border scores on arenas whose bins do not tile the outline exactly.

A curved or diagonal wall leaves slivers inside the arena whose grid cells were
not kept. Positions there belong to the neighboring bin, and so do the wall
pieces there, so every measured wall has an owner and a border cell in a
circular arena gets a score. Wall geometry is prepared once per environment,
so a population scores without rebuilding it for each cell.
"""

import numpy as np
import pytest
import shapely
from shapely.affinity import rotate
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import border_score
from neurospatial.encoding._metrics import batch_border_scores

RADIUS = 50.0


@pytest.fixture(scope="module")
def round_arena():
    circle = shapely.Point(0, 0).buffer(RADIUS, quad_segs=128)
    return Environment.from_polygon(circle, bin_size=4.0, units="cm")


def _wall_field(env, width=6.0):
    distance_to_wall = RADIUS - np.hypot(*env.bin_centers.T)
    return np.where(distance_to_wall < width, 10.0, 0.5)


def _center_field(env):
    return np.where(np.hypot(*env.bin_centers.T) < 15.0, 10.0, 0.5)


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
def test_border_cell_in_a_circular_arena_is_scored(round_arena, metric):
    wall = border_score(round_arena, _wall_field(round_arena), metric=metric)
    center = border_score(round_arena, _center_field(round_arena), metric=metric)
    assert np.isfinite(wall) and np.isfinite(center)
    assert wall > 0.5
    assert center < 0.0


def test_a_ring_field_covers_the_whole_wall(round_arena):
    """Wall pieces in slivers count toward coverage: a ring along the whole
    wall covers its whole length."""
    from neurospatial.encoding._physical_border import _wall_coverage, border_geometry

    geometry = border_geometry(round_arena, "physical")
    # The circle is one ring-shaped wall.
    assert geometry.closed.tolist() == [True]
    assert geometry.wall_lengths[0] == pytest.approx(geometry.walls.length, rel=1e-9)
    rates = _wall_field(round_arena)
    ring = np.flatnonzero(rates > 1)
    assert _wall_coverage(geometry, [ring], rates) == pytest.approx(1.0, rel=1e-9)


def _arc_field(env, degrees, width=6.0):
    """Bins within ``width`` of the wall, within ``degrees / 2`` of west."""
    x, y = env.bin_centers.T
    near = RADIUS - np.hypot(x, y) < width
    toward_west = np.abs(np.angle(-x - 1j * y, deg=True)) <= degrees / 2
    return np.where(near & toward_west, 10.0, 0.5)


@pytest.mark.parametrize(("degrees", "coverage"), [(180, 1.0), (90, 0.5)])
def test_a_ring_offers_each_field_half_of_it(round_arena, degrees, coverage):
    """Høydal et al. (2018): a border cell along half of a circular arena
    covers its whole wall, and one along a quarter covers half of it."""
    from neurospatial.encoding._physical_border import _wall_coverage, border_geometry

    geometry = border_geometry(round_arena, "physical")
    rates = _arc_field(round_arena, degrees)
    field = np.flatnonzero(rates > 1)
    # Bins reach a little past the arc's ends, so allow a bin at each end.
    assert _wall_coverage(geometry, [field], rates) == pytest.approx(
        coverage, abs=2 * 4.0 / (np.pi * RADIUS)
    )


def test_arc_border_cells_score_high_and_a_place_cell_near_the_wall_low():
    """Noise-free maps in a 100 cm circular arena at 2.5 cm bins: a field along
    a 90-degree arc is a border field (0.82 against the half-ring wall, 0.66
    against the whole ring), while a place field 6 cm from the wall is not
    (0.04)."""
    env = Environment.from_polygon(
        shapely.Point(0, 0).buffer(RADIUS, quad_segs=64), bin_size=2.5, units="cm"
    )
    x, y = env.bin_centers.T
    angle = np.linspace(np.pi * 3 / 4, np.pi * 5 / 4, 200)
    arc = shapely.LineString(np.c_[RADIUS * np.cos(angle), RADIUS * np.sin(angle)])
    to_arc = shapely.distance(shapely.points(env.bin_centers), arc)
    arc_cell = 0.2 + 14.8 * np.exp(-0.5 * (to_arc / 5.0) ** 2)
    place = 0.2 + 15 * np.exp(-((x + RADIUS - 6) ** 2 + y**2) / (2 * 8.0**2))
    assert border_score(env, arc_cell) > 0.75
    assert border_score(env, place) < 0.15


def test_rotated_square_scores_like_its_arena_aligned_build():
    square = box(0, 0, 40, 40)
    tilted = rotate(square, 30, origin=(20, 20))
    axis_aligned = Environment.from_polygon(tilted, bin_size=1.0, units="cm")
    aligned = Environment.from_polygon(square, bin_size=1.0, units="cm")

    def wall_strip(env, polygon):
        boundary = polygon.exterior
        near = shapely.distance(shapely.points(env.bin_centers), boundary) < 4.0
        west = shapely.points(env.bin_centers)
        return np.where(near, 10.0, 0.5), west

    rates_tilted, _ = wall_strip(axis_aligned, tilted)
    rates_aligned, _ = wall_strip(aligned, square)
    np.testing.assert_allclose(
        border_score(axis_aligned, rates_tilted, metric="euclidean"),
        border_score(aligned, rates_aligned, metric="euclidean"),
        atol=0.05,
    )


def test_population_scores_match_single_cells(round_arena):
    rates = np.stack(
        [
            _wall_field(round_arena),
            _center_field(round_arena),
            _wall_field(round_arena, 12),
        ]
    )
    batch = batch_border_scores(round_arena, rates).scores
    single = [border_score(round_arena, r) for r in rates]
    np.testing.assert_array_equal(batch, single)


def test_wall_geometry_is_prepared_once_per_environment(monkeypatch):
    from neurospatial.encoding import _physical_border

    calls = []
    original = _physical_border.materialize_cells

    def counting(binning):
        calls.append(1)
        return original(binning)

    monkeypatch.setattr(_physical_border, "materialize_cells", counting)
    circle = shapely.Point(0, 0).buffer(RADIUS, quad_segs=64)
    env = Environment.from_polygon(circle, bin_size=4.0, units="cm")
    rates = np.stack([_wall_field(env, width) for width in np.linspace(4, 20, 20)])
    batch_border_scores(env, rates)
    border_score(env, rates[0])
    assert len(calls) == 1
