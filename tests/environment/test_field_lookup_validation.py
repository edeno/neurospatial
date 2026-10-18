"""Field interpolation, smoothing bandwidths and nearest-bin inputs are validated.

Interpolation keeps every point the environment contains, whatever the cell
shape; an infinite bandwidth is refused instead of silently emptying a field;
positions with the wrong number of coordinates are refused even when every
row is missing.
"""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture(scope="module")
def stretched_grid():
    """3 x 3 cells, 1 wide and 100 tall: centers far apart along y."""
    edges = (np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 100.0, 200.0, 300.0]))
    return Environment.from_grid_mask(np.ones((3, 3), dtype=bool), grid_edges=edges)


def test_nearest_interpolation_keeps_contained_points_in_tall_cells(stretched_grid):
    point = np.array([[0.5, 1.0]])  # inside cell (0, 0), 49 from its center
    assert stretched_grid.contains(point)[0]
    value = stretched_grid.interpolate(
        np.ones(stretched_grid.n_bins), point, mode="nearest"
    )
    np.testing.assert_allclose(value, [1.0])


def test_interpolation_still_refuses_outside_points(stretched_grid):
    value = stretched_grid.interpolate(
        np.ones(stretched_grid.n_bins), np.array([[5.0, 50.0]]), mode="nearest"
    )
    assert np.isnan(value).all()


@pytest.mark.parametrize("bandwidth", [np.inf, np.nan])
@pytest.mark.parametrize("method", ["smooth", "diffuse"])
def test_non_finite_bandwidth_is_refused(stretched_grid, method, bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        getattr(stretched_grid, method)(
            np.ones(stretched_grid.n_bins), bandwidth=bandwidth, mode="transition"
        )


@pytest.mark.parametrize("fill", [np.nan, 0.0])
def test_nearest_bin_refuses_wrong_coordinate_count_even_when_missing(
    stretched_grid, fill
):
    with pytest.raises(ValueError, match="2-D"):
        stretched_grid.nearest_bin(np.full((2, 3), fill))
