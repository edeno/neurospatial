"""Walls retain source precision while following current clipped support."""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import publish_environment
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.geometry._wall_frame import wall_source_frame
from neurospatial.ops.transforms import Affine2D


def _publish(child, arena):
    return publish_environment(
        name="wall frame",
        layout=child,
        layout_type_used=child.layout_type,
        layout_params_used=child.layout_parameters,
        arena=arena,
    )


def _matrix(offset=2.0):
    return np.array([[0.0, -1.0, offset], [1.0, 0.0, offset + 1], [0.0, 0.0, 1.0]])


def _mapped(polygon, matrix):
    return affine_transform(polygon, [*matrix[0, :2], *matrix[1, :2], *matrix[:2, 2]])


@pytest.mark.parametrize("history", ["clip", "subset", "nested"])
@pytest.mark.parametrize("case", ["removed", "full_face"])
def test_new_walls_validate_current_clipped_support_in_shared_source_frame(
    case, history
):
    base = Environment.from_grid_mask(
        np.ones((2, 1), bool),
        grid_edges=(np.arange(3.0), np.arange(2.0)),
        arena=OpenArena(box(0, 0, 2, 1)),
    )
    if case == "removed":
        region = box(0.1, 0.1, 0.3, 0.9)
        wall = LineString([(0.5, 0), (0.5, 1)])
    else:
        region = box(0.1, 0.25, 1.9, 0.75)
        wall = LineString([(1, 0.25), (1, 0.75)])
    matrix = _matrix()
    child = base.apply_transform(Affine2D(matrix)).clip(_mapped(region, matrix))
    arena = OpenArena(box(0, 0, 2, 1), walls=(wall,)).transformed(matrix)
    if history == "subset":
        child = child.subset(mask=np.ones(child.n_bins, bool))
    elif history == "nested":
        outer = _matrix(5.0)
        child = child.apply_transform(Affine2D(outer))
        arena = arena.transformed(outer)
    published = _publish(child, arena)
    assert published.n_bins == (1 if case == "removed" else 2)
    assert published.connectivity.number_of_edges() == 0


@pytest.mark.parametrize("clipped", [False, True])
def test_source_interior_wall_remains_invalid_when_world_rounding_erases_its_offset(
    clipped,
):
    matrix = _matrix(1e9)
    base = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),)),
    )
    child = base.apply_transform(Affine2D(matrix))
    if clipped:
        child = child.clip(_mapped(box(0.25, 0.25, 3.75, 3.75), matrix))
    invalid = OpenArena(
        box(0, 0, 4, 4),
        walls=(LineString([(2 + 1e-12, 0), (2 + 1e-12, 4)]),),
    ).transformed(matrix)
    np.testing.assert_array_equal(child.arena.walls[0].coords, invalid.walls[0].coords)
    with pytest.raises(ValueError, match=r"wall.*(?:cuts|clearance|unresolved)"):
        _publish(child, invalid)


def _rotated_scaled_clipped_grid(wall, angle=0.3):
    """A 4 x 4 grid rotated, scaled, moved to 1e6 and clipped, with ``wall``.

    ``wall`` is in the grid's source frame, where bins 4 and 8 meet along
    x = 2 for 0.25 <= y <= 1 and bins 5 and 9 along x = 2 for 1 <= y <= 2.
    """
    scale, offset = 1.3, 1e6
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    matrix[:2, 2] = [offset + 0.1, offset - 0.2]
    base = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4)),
    )
    child = base.apply_transform(Affine2D(matrix)).clip(
        _mapped(box(0.25, 0.25, 3.75, 3.75), matrix)
    )
    arena = OpenArena(box(0, 0, 4, 4), walls=(wall,)).transformed(matrix)
    return child, arena


_GAP_WALL_Y_RANGES = [(0.0, 4.0), (0.49, 0.51), (1.49, 1.51)]


@pytest.mark.parametrize("y_range", _GAP_WALL_Y_RANGES)
def test_inverse_projected_clipped_cells_cannot_hide_source_wall_in_artificial_gap(
    y_range,
):
    child, invalid = _rotated_scaled_clipped_grid(
        LineString([(2 + 1e-12, y_range[0]), (2 + 1e-12, y_range[1])])
    )
    with pytest.raises(ValueError, match=r"wall.*(?:cuts|clearance|unresolved)"):
        _publish(child, invalid)


@pytest.mark.parametrize("y_range", _GAP_WALL_Y_RANGES)
@pytest.mark.parametrize("ulps", range(-3, 4))
def test_wall_in_a_rounding_gap_is_refused_whatever_the_last_bits_of_rotation(
    ulps, y_range
):
    """Moving the rotation by a few ULPs never turns the refusal into a pass."""
    angle = 0.3
    for _ in range(abs(ulps)):
        angle = np.nextafter(angle, np.copysign(np.inf, ulps))
    child, invalid = _rotated_scaled_clipped_grid(
        LineString([(2 + 1e-12, y_range[0]), (2 + 1e-12, y_range[1])]), angle
    )
    with pytest.raises(ValueError, match=r"wall.*(?:cuts|clearance|unresolved)"):
        _publish(child, invalid)


@pytest.mark.parametrize("ulps", [-3, 0, 3])
@pytest.mark.parametrize("placement", ["beside the face", "across the face"])
def test_wall_clear_of_a_rounding_gap_leaves_the_face_open(placement, ulps):
    """A wall ten error bounds from the face, or crossing it, blocks no path."""
    angle = 0.3
    for _ in range(abs(ulps)):
        angle = np.nextafter(angle, np.copysign(np.inf, ulps))
    # The bins, and so the error bound, do not depend on the wall.
    child, _ = _rotated_scaled_clipped_grid(LineString([(2, 0), (2, 4)]), angle)
    errors = wall_source_frame(
        child._binning.locator, child.arena, child._binning.bins.centers
    )[5]
    # Bins 4 and 8 face each other across x = 2 to within their two bounds.
    bound = errors[4] + errors[8]
    assert 0 < bound < 1e-8
    if placement == "beside the face":
        wall = LineString([(2 + 10 * bound, 0.49), (2 + 10 * bound, 0.51)])
    else:
        wall = LineString([(1.9, 0.5), (2.1, 0.5)])
    child, arena = _rotated_scaled_clipped_grid(wall, angle)

    published = _publish(child, arena)

    np.testing.assert_array_equal(published.edge_pairs, child.edge_pairs)


def test_genuine_source_inset_does_not_snap_to_a_nearby_parent_boundary():
    matrix = _matrix(0.0)
    base = Environment.from_grid_mask(
        np.ones((2, 1), bool),
        grid_edges=(np.arange(3.0), np.arange(2.0)),
        arena=OpenArena(box(0, 0, 2, 1)),
    )
    child = base.apply_transform(Affine2D(matrix)).clip(
        _mapped(box(0.1, 0.1, 1 - 1e-12, 0.9), matrix)
    )
    wall = LineString([(1 - 5e-13, 0), (1 - 5e-13, 1)])
    arena = OpenArena(box(0, 0, 2, 1), walls=(wall,)).transformed(matrix)
    assert _publish(child, arena).n_bins == 1


def test_distinct_world_inset_is_not_inherited_even_within_coordinate_ulp_bound():
    matrix = _matrix(1e9)
    ulp = np.spacing(1e9)
    base = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([0.0, 1.0]),) * 2,
        arena=OpenArena(box(0, 0, 1, 1)),
    )
    child = base.apply_transform(Affine2D(matrix)).clip(
        _mapped(box(4 * ulp, 0.1, 0.9, 0.9), matrix)
    )
    wall = LineString([(2 * ulp, 0), (2 * ulp, 1)])
    arena = OpenArena(box(0, 0, 1, 1), walls=(wall,)).transformed(matrix)
    assert _publish(child, arena).n_bins == 1


def _rotated_clipped_grid(wall_x, y_range):
    """A clipped, rotated 4 x 4 grid and a short wall inside its source frame."""
    angle = 0.1
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 1e6 + 0.1],
            [np.sin(angle), np.cos(angle), 1e6 - 0.2],
            [0.0, 0.0, 1.0],
        ]
    )
    base = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0),) * 2,
        arena=OpenArena(box(0, 0, 4, 4)),
    )
    child = base.apply_transform(Affine2D(matrix)).clip(
        _mapped(box(0, 0.25, 4, 3.75), matrix)
    )
    arena = OpenArena(
        box(0, 0, 4, 4),
        walls=(LineString([(wall_x, y_range[0]), (wall_x, y_range[1])]),),
    ).transformed(matrix)
    return child, arena


@pytest.mark.parametrize(
    ("wall_x", "y_range"),
    [(1e-12, (1.49, 1.51)), (4 - 1e-12, (0.29, 0.31)), (4 - 1e-12, (1.49, 1.51))],
)
def test_unresolved_exterior_projection_gap_refuses_source_interior_wall(
    wall_x, y_range
):
    child, arena = _rotated_clipped_grid(wall_x, y_range)
    with pytest.raises(ValueError, match=r"wall.*(?:cuts|clearance|unresolved)"):
        _publish(child, arena)


def test_a_resolved_wall_stub_inside_a_bin_leaves_its_paths():
    """A wall that ends inside a bin does not divide it, so no path changes."""
    child, arena = _rotated_clipped_grid(1e-12, (0.29, 0.31))

    published = _publish(child, arena)

    assert published.n_bins == child.n_bins
    np.testing.assert_array_equal(published.edge_pairs, child.edge_pairs)


def test_signed_permutation_does_not_certify_a_rounded_diagonal_parent_edge():
    matrix = _matrix(1e9)
    base = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([0.0, 1.0]),) * 2,
        arena=OpenArena(box(0, 0, 1, 1)),
    ).clip(Polygon([(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]))
    child = base.apply_transform(Affine2D(matrix)).clip(
        _mapped(box(0.15, 0.2, 0.85, 0.7), matrix)
    )
    wall = LineString(
        [
            (0.1 + 0.5 * (0.24 - 0.1) + 1e-12, 0.24),
            (0.1 + 0.5 * (0.26 - 0.1) + 1e-12, 0.26),
        ]
    )
    arena = OpenArena(box(0, 0, 1, 1), walls=(wall,)).transformed(matrix)
    with pytest.raises(ValueError, match=r"wall.*(?:cuts|clearance|unresolved)"):
        _publish(child, arena)
