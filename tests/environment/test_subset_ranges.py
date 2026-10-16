"""A subset reports the extent of the cells it keeps, with no margin."""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.geometry._polygon import materialize_cells


def _keep_first_quarter(env: Environment) -> np.ndarray:
    keep = np.zeros(env.n_bins, dtype=bool)
    keep[: max(1, env.n_bins // 4)] = True
    return keep


def _cell_bounds(env: Environment) -> np.ndarray:
    bounds = np.array([cell.bounds for cell in materialize_cells(env.binning)])
    return np.column_stack((bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)))


@pytest.mark.parametrize("built_environment", ["hex", "mesh"], indirect=True)
def test_subset_ranges_are_the_kept_cells_extent(built_environment):
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    np.testing.assert_allclose(
        child.dimension_ranges, _cell_bounds(child), rtol=0, atol=1e-12
    )


def test_subset_of_clipped_polygon_cells_bounds_the_kept_polygons():
    grid = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool), grid_edges=(np.linspace(0, 12, 7),) * 2
    )
    clipped = grid.clip(Point(6, 6).buffer(5.0))
    child = clipped.subset(mask=_keep_first_quarter(clipped))
    np.testing.assert_array_equal(child.dimension_ranges, _cell_bounds(child))


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_subset_without_cell_shapes_bounds_the_kept_centers(built_environment):
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    centers = child.bin_centers
    np.testing.assert_array_equal(
        child.dimension_ranges,
        np.column_stack((centers.min(axis=0), centers.max(axis=0))),
    )


@pytest.mark.parametrize("built_environment", ["grid8", "polygon"], indirect=True)
def test_grid_subset_keeps_the_grid_extent(built_environment):
    # A grid subset keeps its full lattice (inactive cells included), so its
    # extent is the lattice's, matching grid_edges.
    child = built_environment.subset(mask=_keep_first_quarter(built_environment))
    np.testing.assert_array_equal(
        child.dimension_ranges,
        [(edge[0], edge[-1]) for edge in child.grid_edges],
    )


@pytest.mark.parametrize("built_environment", ["mesh"], indirect=True)
def test_mesh_subset_shrinks_to_its_kept_triangles(built_environment):
    parent = built_environment
    corner = np.all(parent.bin_centers < 4.0, axis=1)
    assert 0 < corner.sum() < parent.n_bins
    child = parent.subset(mask=corner)
    shapes = child.binning.shapes
    vertices = shapes.points[shapes.triangles].reshape(-1, 2)
    np.testing.assert_array_equal(
        child.dimension_ranges,
        np.column_stack((vertices.min(axis=0), vertices.max(axis=0))),
    )
    # Every mesh point is retained, but the unused ones do not widen the extent.
    assert np.all(np.asarray(child.dimension_ranges)[:, 1] <= 8.0)
    assert np.all(shapes.points.max(axis=0) == 12.0)
