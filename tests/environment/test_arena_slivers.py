"""Positions inside a known arena outline always belong to a bin.

Bins are kept where their centers lie inside the outline, so slivers of a
curved or diagonal wall lie inside the arena but in no bin. A position there
belongs to its nearest bin, less than one bin away, instead of being dropped
from occupancy and spike counts.
"""

import numpy as np
import pytest
import shapely

from neurospatial import Environment

RADIUS = 50.0


@pytest.fixture(scope="module")
def circle():
    return shapely.Point(0, 0).buffer(RADIUS, quad_segs=128)


@pytest.fixture(scope="module")
def round_arena(circle):
    return Environment.from_polygon(circle, bin_size=4.0, units="cm")


@pytest.fixture(scope="module")
def sliver_points(circle, round_arena):
    """In-arena points that the whole-bin cells do not cover."""
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 4000)
    radius = RADIUS - rng.uniform(0.0, 3.0, 4000)
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    points = points[shapely.covers(circle, shapely.points(points))]
    cells = round_arena._binning.locator.locate(points)
    return points[cells < 0]


def test_fixture_has_sliver_points(sliver_points):
    assert len(sliver_points) > 50


def test_sliver_positions_belong_to_their_nearest_bin(round_arena, sliver_points):
    bins = round_arena.bin_at(sliver_points)
    np.testing.assert_array_equal(bins, round_arena.nearest_bin(sliver_points))
    assert round_arena.contains(sliver_points).all()
    gaps = np.linalg.norm(round_arena.bin_centers[bins] - sliver_points, axis=1)
    assert gaps.max() < 4.0  # under one bin


def test_positions_outside_the_outline_stay_outside(round_arena):
    outside = np.array([[RADIUS + 0.5, 0.0], [0.0, -RADIUS - 2.0]])
    np.testing.assert_array_equal(round_arena.bin_at(outside), [-1, -1])
    assert not round_arena.contains(outside).any()


def test_wall_hugging_time_is_all_counted(round_arena, sliver_points):
    times = np.arange(len(sliver_points)) / 30.0
    occupancy = round_arena.occupancy(times, sliver_points, max_gap=None)
    assert occupancy.sum() == pytest.approx(times[-1])


def test_samples_without_a_known_outline_are_unchanged():
    """from_samples infers bins from data: a position outside them stays out."""
    rng = np.random.default_rng(1)
    env = Environment.from_samples(rng.uniform(0, 20, (2000, 2)), bin_size=2.0)
    assert env.bin_at(np.array([[30.0, 30.0]]))[0] == -1
