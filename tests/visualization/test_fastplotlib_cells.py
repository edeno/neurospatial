"""Cells and colors for GPU drawing; checked without fastplotlib or a GPU."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from shapely.geometry import Polygon, box

from neurospatial import Environment
from neurospatial.environment._fastplotlib import ColorScale, cell_mesh
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import Affine2D

EDGES = np.arange(0.0, 41.0, 4.0)


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return Affine2D(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))


def _grid(mask=None):
    mask = np.ones((10, 10), dtype=bool) if mask is None else mask
    return Environment.from_grid_mask(mask, grid_edges=(EDGES, EDGES), units="cm")


def _cell_areas(cells, n_bins):
    corners = cells.positions[cells.triangles][..., :2].astype(np.float64)
    (ax, ay), (bx, by), (cx, cy) = np.moveaxis(corners, 1, 0).transpose(0, 2, 1)
    areas = 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    owner = cells.vertex_bin[cells.triangles[:, 0]]
    return np.bincount(owner, weights=areas, minlength=n_bins)


def _cell_centroids(cells, n_bins):
    counts = np.bincount(cells.vertex_bin, minlength=n_bins)[:, None]
    sums = np.stack(
        [np.bincount(cells.vertex_bin, weights=cells.positions[:, k]) for k in (0, 1)],
        axis=1,
    )
    return sums / counts


@pytest.mark.parametrize(
    "env",
    [
        pytest.param(_grid(), id="grid"),
        pytest.param(
            _grid(np.eye(10, dtype=bool) | np.eye(10, k=1, dtype=bool)), id="mask"
        ),
        pytest.param(_grid().apply_transform(_rotation(0.4)), id="rotated-grid"),
        pytest.param(
            Environment.from_samples(
                np.random.default_rng(0).uniform(0, 40, (4000, 2)),
                bin_size=4.0,
                layout="hexagonal",
            ),
            id="hexagons",
        ),
        pytest.param(
            Environment.from_layout(
                create_layout(
                    "TriangularMesh",
                    boundary_polygon=box(0, 0, 40, 30),
                    point_spacing=5.0,
                )
            ),
            id="triangles",
        ),
    ],
)
def test_each_bin_is_drawn_as_its_own_cell(env):
    cells = cell_mesh(env)
    assert cells is not None
    assert set(np.unique(cells.vertex_bin)) == set(range(env.n_bins))
    # Convex cells: the mean of a cell's corners is its bin center.
    np.testing.assert_allclose(
        _cell_centroids(cells, env.n_bins), env.bin_centers, atol=1e-5
    )  # float32 vertices


def test_grid_cells_have_the_bins_area():
    mask = np.ones((10, 10), dtype=bool)
    mask[:3, :3] = False
    env = _grid(mask)
    np.testing.assert_allclose(_cell_areas(cell_mesh(env), env.n_bins), 16.0, rtol=1e-6)


def test_clipped_cells_keep_their_exact_area_holes_included():
    env = _grid().clip(
        Polygon([(2, 2), (38, 4), (30, 37), (5, 30)]).difference(box(15, 15, 22, 22))
    )
    cells = cell_mesh(env)
    np.testing.assert_allclose(
        _cell_areas(cells, env.n_bins),
        [polygon.area for polygon in env._binning.shapes.polygons],
        atol=1e-4,  # float32 vertices
    )


def test_tracks_points_and_volumes_are_drawn_as_points():
    track = Environment.linear_track(endpoints=[[0, 0], [100, 0]], bin_size=5.0)
    volume = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (2000, 3)), bin_size=2.0
    )
    assert cell_mesh(track) is None
    assert cell_mesh(volume) is None


def test_missing_values_get_their_own_color_not_the_top_of_the_colormap():
    scale = ColorScale("viridis", 0.0, 10.0, "lightgray")
    values = np.array([0.0, 10.0, np.nan, 20.0])
    colors = scale.rgba(values)
    np.testing.assert_allclose(colors[0], colormaps["viridis"](0.0), atol=1e-6)
    np.testing.assert_allclose(colors[1], colormaps["viridis"](1.0), atol=1e-6)
    np.testing.assert_allclose(colors[2], to_rgba("lightgray"))
    np.testing.assert_allclose(colors[3], colors[1])  # above vmax is clipped
    # The texture coordinate lands in the middle of the same lookup row.
    rows = scale.texture_coordinates(values) * len(scale.lut) - 0.5
    np.testing.assert_allclose(scale.lut[np.rint(rows).astype(int)], colors)
    assert scale.index(values)[2] == 0
    assert ColorScale("viridis", 0.0, 1.0, None).lut[0, 3] == 0.0  # transparent


def test_equal_limits_do_not_divide_by_zero():
    scale = ColorScale("viridis", 2.0, 2.0, "lightgray")
    np.testing.assert_array_equal(scale.index(np.array([2.0, 3.0])), [1, 255])
