"""A point on a face shared by two bins keeps its bin when bins and points move.

Rotating, reflecting or translating an environment together with the
positions must not move a position into a different bin, whatever the
rounding of the map: the shared faces between connected bins and the cell
corners are the places where rounding would decide.
"""

import numpy as np
import pytest
import shapely

from neurospatial import Environment
from neurospatial.geometry import materialize_cells
from neurospatial.ops import Affine2D


def rigid(angle, offset, reflect=False):
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if reflect:
        matrix[:2, 0] *= -1
    matrix[:2, 2] = offset
    return matrix


MAPS = [
    rigid(1e-9, (13.7, -2.1)),
    rigid(0.123, (0.0, 0.0)),
    rigid(0.7, (13.7, -2.1)),
    rigid(2.0, (1e4, 3e3)),
    rigid(0.7, (-5.0, 8.0), reflect=True),
]
MAP_IDS = ["tiny-angle", "rotation", "offset", "far", "reflection"]


def face_probes(env):
    """Midpoints of connected bin centers and every cell corner."""
    centers = env.bin_centers
    pairs = env.edge_pairs
    probes = [(centers[pairs[:, 0]] + centers[pairs[:, 1]]) / 2]
    try:
        cells = materialize_cells(env.binning)
    except (ValueError, NotImplementedError):
        cells = ()  # a track's bins are stretches of centerline, not cells
    if len(cells):
        corners = np.concatenate([shapely.get_coordinates(cell) for cell in cells])
        probes.append(np.unique(corners, axis=0))
    return np.concatenate(probes)


@pytest.mark.parametrize(
    "built_environment",
    ["hex", "track", "mesh", "mesh_boundary", "polygon", "mask", "grid8"],
    indirect=True,
)
@pytest.mark.parametrize("matrix", MAPS, ids=MAP_IDS)
def test_faces_keep_their_bin_under_rigid_maps(built_environment, matrix):
    env = built_environment
    probes = face_probes(env)
    expected = env.bin_at(probes)
    assert np.count_nonzero(expected >= 0) > len(probes) // 2
    transform = Affine2D(matrix)
    moved = env.apply_transform(transform)
    np.testing.assert_array_equal(moved.bin_at(transform(probes)), expected)


@pytest.mark.parametrize("matrix", MAPS, ids=MAP_IDS)
def test_clipped_and_combined_faces_keep_their_bin(matrix):
    grid = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0) * 2,) * 2
    )
    clipped = grid.clip(shapely.Point(4.0, 4.0).buffer(3.5))
    joined = Environment.combine(
        {
            "grid": grid,
            "right": Environment.from_grid_mask(
                np.ones((2, 4), dtype=bool),
                grid_edges=(np.arange(8.0, 13.0, 2.0)[:3], np.arange(5.0) * 2),
            ),
        }
    )
    transform = Affine2D(matrix)
    # Cell centers, faces and corners of the grids (a clip's cut cells have
    # corners that rounding moves off the grid's faces, so are not probed).
    steps = np.arange(-1.0, 13.0, 1.0)
    probes = np.stack(np.meshgrid(steps, steps, indexing="ij"), -1).reshape(-1, 2)
    for env in (clipped, joined):
        expected = env.bin_at(probes)
        assert np.count_nonzero(expected >= 0) > 20
        moved = env.apply_transform(transform)
        np.testing.assert_array_equal(moved.bin_at(transform(probes)), expected)


def test_hexagon_sides_belong_to_the_hexagon_above():
    """Ties follow one rule, like a grid's cell above, not rounding."""
    from neurospatial.layout import create_layout

    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )
    centers, pairs = env.bin_centers, env.edge_pairs
    midpoints = (centers[pairs[:, 0]] + centers[pairs[:, 1]]) / 2
    upper = np.where(
        np.isclose(centers[pairs[:, 0], 1], centers[pairs[:, 1], 1]),
        np.where(centers[pairs[:, 0], 0] > centers[pairs[:, 1], 0], 0, 1),
        np.where(centers[pairs[:, 0], 1] > centers[pairs[:, 1], 1], 0, 1),
    )
    expected = pairs[np.arange(len(pairs)), upper]
    np.testing.assert_array_equal(env.bin_at(midpoints), expected)
