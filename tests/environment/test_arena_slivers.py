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


def _rotation(angle: float, offset: tuple[float, float]) -> np.ndarray:
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[:2, 2] = offset
    return matrix


@pytest.fixture(scope="module")
def in_arena_points(round_arena):
    rng = np.random.default_rng(2)
    points = rng.uniform(-RADIUS, RADIUS, (40_000, 2))
    return points[round_arena.arena.contains(points)]


@pytest.mark.parametrize(
    "matrix",
    [
        _rotation(0.3, (1.5, -2.0)),
        np.diag([-1.0, 1.0, 1.0]),
        np.array([[0.0, 1.0, 3.0], [1.0, 0.0, -4.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["rotation", "reflection", "axis-swap"],
)
def test_transformed_grid_keeps_every_in_arena_position(
    round_arena, in_arena_points, matrix
):
    """A rotated or mirrored grid assigns sliver positions to the same bins."""
    from neurospatial.ops import AffineND

    transform = AffineND(matrix)
    moved = round_arena.apply_transform(transform)
    expected = round_arena.bin_at(in_arena_points)
    assert (expected >= 0).all()
    moved_points = transform(in_arena_points)
    assert moved.arena.contains(moved_points).all()
    np.testing.assert_array_equal(moved.bin_at(moved_points), expected)

    keep = np.ones(round_arena.n_bins, dtype=bool)
    keep[[0, 1, round_arena.n_bins // 2]] = False
    renumbered = np.where(keep, np.cumsum(keep) - 1, -1)
    np.testing.assert_array_equal(
        moved.subset(mask=keep).bin_at(moved_points), renumbered[expected]
    )
    np.testing.assert_array_equal(
        round_arena.subset(mask=keep).bin_at(in_arena_points), renumbered[expected]
    )


def test_rotated_grid_owns_its_whole_physical_wall(round_arena):
    """Wall pieces in outline slivers keep an owner after a rotation."""
    from neurospatial.encoding._physical_border import border_geometry
    from neurospatial.ops import AffineND

    moved = round_arena.apply_transform(AffineND(_rotation(0.3, (1.5, -2.0))))
    original = border_geometry(round_arena, "physical")
    rotated = border_geometry(moved, "physical")
    np.testing.assert_array_equal(rotated.wall_bins, original.wall_bins)


def test_outline_through_cell_centers_gives_every_wall_point_a_bin():
    """A cell whose center lies exactly on the outline is not kept, so it is a
    sliver whose wall points belong to a neighboring bin. These kite edges
    pass through grid cell centers at 2.5 cm bins."""
    from neurospatial.geometry import locate_outline_points

    kite = shapely.Polygon([(0, 0), (100, -10), (120, 90), (10, 100)])
    env = Environment.from_polygon(kite, bin_size=2.5, units="cm")
    outline = shapely.segmentize(kite.exterior, 0.1)
    points = shapely.get_coordinates(outline)[:-1]
    bins = locate_outline_points(env._binning, env.arena, points)
    assert (bins >= 0).all()
    gaps = np.linalg.norm(env.bin_centers[bins] - points, axis=1)
    assert gaps.max() < 2.5 * np.sqrt(2) * 1.5  # within the neighboring cell
