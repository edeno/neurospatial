"""Layout capabilities describe usable cell and track geometry."""

import numpy as np
import pytest
from environment_helpers import layout_of
from shapely.geometry import box

from neurospatial import Environment
from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.layout.engines.graph import GraphLayout
from neurospatial.layout.engines.hexagonal import HexagonalLayout
from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout


@pytest.fixture(params=["samples", "mask", "pixels", "polygon", "polar"])
def grid_layout(request):
    """Built grids whose coordinates match their per-axis cells."""
    return layout_of(lambda: _grid_environment(request.param))


def _grid_environment(kind):
    if kind == "samples":
        env = Environment.from_samples(
            np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=float), bin_size=1.0
        )
    elif kind == "mask":
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0), np.arange(3.0))
        )
    elif kind == "pixels":
        env = Environment.from_pixel_mask(np.ones((2, 2), dtype=bool), pixel_size=1.0)
    elif kind == "polygon":
        env = Environment.from_polygon(box(0, 0, 2, 2), bin_size=1.0)
    else:
        env = Environment.from_polar_egocentric(
            distance_range=(0, 2),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=1.0,
            angle_bin_size=np.pi / 2,
        )
    return env


def test_grid_capability_matches_cell_centers(grid_layout):
    """Descriptor edges and activity reproduce the layout's bin order."""
    grid = grid_layout.binning().rectilinear
    assert grid is not None
    axes = [(edges[:-1] + edges[1:]) / 2 for edges in grid.edges]
    centers = np.stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    np.testing.assert_array_equal(
        centers[grid.active_mask.ravel()], grid_layout.bin_centers
    )
    assert grid.active_mask.shape == grid.shape


def test_unbuilt_grid_refuses_cell_geometry(grid_layout):
    """Having the method does not claim an unbuilt layout has cells yet."""
    unbuilt = type(grid_layout)()
    with pytest.raises(LayoutNotBuiltError, match="rectilinear_grid"):
        unbuilt.rectilinear_grid()


@pytest.mark.parametrize(
    "layout_type", [HexagonalLayout, GraphLayout, TriangularMeshLayout]
)
def test_non_grid_capabilities(layout_type):
    """Only graph layouts provide track conversion; none has rectangular cells."""
    layout = layout_type()
    assert not hasattr(layout, "rectilinear_grid")
    assert layout.is_linearized_track == (layout_type is GraphLayout)


def test_saved_layout_facade_keeps_capabilities_in_the_binning():
    """The facade exposes data components without claiming provider methods."""
    from neurospatial.io import from_dict, to_dict

    original = Environment.linear_track(
        endpoints=[[0.0, 0.0], [2.0, 0.0]], bin_size=1.0
    )
    loaded = from_dict(to_dict(original))
    assert not hasattr(loaded, "rectilinear_grid")
    assert loaded._binning.faces.available
    assert loaded._binning.linearization is not None
    points = np.array([[0.5, 0.0], [1.5, 0.0]])
    np.testing.assert_array_equal(loaded.to_linear(points), original.to_linear(points))


def test_layout_of_says_how_many_layouts_a_build_published():
    """The test helper names the count instead of failing to unpack."""
    positions = np.array([[0.0, 0.0], [4.0, 4.0]])

    def nothing():
        return None

    def two_layouts():
        Environment.from_samples(positions, bin_size=2.0)
        Environment.from_samples(positions, bin_size=1.0)

    for build, count in ((nothing, 0), (two_layouts, 2)):
        with pytest.raises(AssertionError, match=f"but it published {count}"):
            layout_of(build)
    # The suggested call publishes exactly one.
    assert (
        layout_of(lambda: Environment.from_samples(positions, bin_size=2.0)) is not None
    )
