"""Published shape components retain bin order and own their drawing geometry."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
from environment_helpers import native_layout
from shapely.geometry import Point, Polygon

from neurospatial import Environment
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_hexagon_shapes_match_layout(built_environment):
    env = built_environment
    shapes = env._binning.shapes
    assert isinstance(shapes, HexagonShapes)
    source = native_layout(env)
    assert shapes.radius == source.hex_radius_
    assert shapes.orientation == source.hex_orientation_
    np.testing.assert_array_equal(shapes.centers, env.bin_centers)
    assert not shapes.centers.flags.writeable


@pytest.mark.parametrize("built_environment", ["mesh", "mesh_boundary"], indirect=True)
def test_triangle_shapes_cover_active_bins(built_environment):
    env = built_environment
    shapes = env._binning.shapes
    assert isinstance(shapes, TriangleShapes)
    assert shapes.triangles.shape == (env.n_bins, 3)
    source = native_layout(env)
    np.testing.assert_array_equal(shapes.points, source._full_delaunay_tri.points)
    np.testing.assert_array_equal(
        shapes.triangles,
        source._full_delaunay_tri.simplices[source._active_original_simplex_indices],
    )
    np.testing.assert_array_equal(
        shapes.points[shapes.triangles].mean(axis=1), env.bin_centers
    )
    assert shapes.triangles.dtype == np.intp


def test_outline_matches_polygon():
    polygon = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]]
    )
    env = Environment.from_polygon(polygon, bin_size=1.0)
    assert isinstance(env._binning.outline, Outline)
    assert env._binning.outline.geometry.equals(polygon)


def test_shape_components_match_available_geometry(built_environment, request):
    """Rectangular cells and tracks do not allocate polygon shapes for every bin."""
    case = request.node.callspec.params["built_environment"]
    shapes = built_environment._binning.shapes
    outline = built_environment._binning.outline
    if case == "hex":
        assert isinstance(shapes, HexagonShapes)
    elif case in {"mesh", "mesh_boundary"}:
        assert isinstance(shapes, TriangleShapes)
    else:
        assert shapes is None
    if case in {"polygon", "mesh", "mesh_boundary"}:
        assert isinstance(outline, Outline)
    else:
        assert outline is None
    if case == "mesh_boundary":
        assert len(outline.geometry.interiors) == 1
        assert not outline.geometry.covers(Point(5, 4))
        assert not outline.geometry.covers(Point(6, 10))
        assert outline.geometry.covers(Point(3.9, 10))


@pytest.fixture(params=[HexagonShapes, TriangleShapes, Outline])
def component_and_inputs(request):
    if request.param is HexagonShapes:
        inputs = {
            "centers": np.arange(12.0).reshape(3, 4)[:, ::2],
            "radius": 2.0,
            "orientation": 0.0,
        }
    elif request.param is TriangleShapes:
        inputs = {
            "points": np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
            "triangles": np.array([[0, 1, 2], [1, 3, 2]]),
        }
    else:
        inputs = {"geometry": Point(0, 0).buffer(1.0)}
    return request.param(**inputs), inputs


def test_shape_components_own_read_only_arrays(component_and_inputs):
    component, inputs = component_and_inputs
    for name, source in inputs.items():
        if not isinstance(source, np.ndarray):
            continue
        published = getattr(component, name)
        expected = published.copy()
        assert not np.shares_memory(source, published)
        assert source.flags.writeable
        source[:] = 0
        np.testing.assert_array_equal(published, expected)
        assert not published.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            published.flat[0] = 0


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_components_keep_identity_and_ownership_after_copying(
    component_and_inputs, restore
):
    component, _ = component_and_inputs
    equal_values = replace(component)
    assert component != equal_values
    assert len({component, equal_values}) == 2
    with pytest.raises(FrozenInstanceError):
        setattr(component, fields(component)[0].name, None)
    copied = restore(component)
    assert copied != component
    for field in fields(component):
        original = getattr(component, field.name)
        actual = getattr(copied, field.name)
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, original)
            assert not actual.flags.writeable
            assert not np.shares_memory(actual, original)
        else:
            assert actual == original
