"""Owned publication detaches source builders, arrays and graph metadata."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.arena import MaskArena, OpenArena, _TransformedArena
from neurospatial.environment._coordinates import EgocentricPolarSystem
from neurospatial.environment._ownership import own_arena
from neurospatial.layout import RegularGridLayout, create_layout
from neurospatial.layout._construction import binning_from_layout
from neurospatial.layout._locators import SubsetLocator, TransformedLocator
from neurospatial.layout.engines._polar import _PolarGridLayout


def _grid():
    return create_layout(
        "RegularGrid",
        bin_size=1.0,
        dimension_ranges=[(0.0, 3.0), (0.0, 3.0)],
        infer_active_bins=False,
    )


@pytest.mark.parametrize(
    "restore",
    [lambda x: x, copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
)
def test_publication_detaches_source_geometry_and_metadata(restore):
    source = _grid()
    source.connectivity.graph["nested"] = {"values": np.array([1.0, 2.0])}
    source._build_params_used["nested"] = {"values": np.array([4.0])}
    expected = source.bin_centers.copy()
    env = restore(Environment._from_built_layout(source))
    source.bin_centers[:] = -99
    source.grid_edges[0][:] = -99
    source.active_mask[:] = False
    source.connectivity.graph["nested"]["values"][:] = -99
    source._build_params_used["nested"]["values"][:] = -99
    for array in (env.bin_centers, env.bin_sizes, env.grid_edges[0], env.active_mask):
        assert not array.flags.writeable
        try:
            array.setflags(write=True)
        except ValueError:
            continue
        array.flat[0] = -999
    exported = copy.deepcopy(env._binning)
    exported.bins.centers.setflags(write=True)
    exported.bins.centers[:] = 123
    assert not hasattr(exported.faces, "layout")
    np.testing.assert_array_equal(env.bin_centers, expected)
    np.testing.assert_array_equal(env.bin_sizes, np.ones(9))
    np.testing.assert_array_equal(env.connectivity.graph["nested"]["values"], [1, 2])
    np.testing.assert_array_equal(env.layout_parameters["nested"]["values"], [4])


@pytest.mark.parametrize(
    "attribute",
    [
        "bin_centers",
        "connectivity",
        "dimension_ranges",
        "grid_edges",
        "grid_shape",
        "active_mask",
    ],
)
def test_geometry_views_refuse_rebinding(attribute):
    env = Environment._from_built_layout(_grid())
    with pytest.raises(AttributeError, match="read-only"):
        setattr(env, attribute, None)
    with pytest.raises(AttributeError, match="read-only"):
        delattr(env, attribute)


def test_publication_preserves_native_grid_context(built_environment):
    env = built_environment
    assert not hasattr(env, "layout")
    assert not any(isinstance(value, RegularGridLayout) for value in vars(env).values())
    assert env._binning.bins.n_bins == env.n_bins
    assert env.active_mask.sum() == env.n_bins
    np.testing.assert_array_equal(
        env._binning.locator.locate(env.bin_centers), np.arange(env.n_bins)
    )


def test_face_building_stays_lazy_without_native_builder(monkeypatch):
    from neurospatial.layout._native_faces import NativeFaces

    source = _grid()
    original = NativeFaces.build
    calls = []

    def record(recipe):
        calls.append(recipe)
        return original(recipe)

    monkeypatch.setattr(NativeFaces, "build", record)
    env = Environment._from_built_layout(source)
    assert not calls
    monkeypatch.setattr(
        RegularGridLayout,
        "finite_volume_geometry",
        lambda _: pytest.fail("retained builder"),
    )
    source.grid_edges[0][:] = -99
    np.testing.assert_array_equal(env._binning.faces.build().volumes, np.ones(9))
    assert len(calls) == 1
    assert not hasattr(calls[0], "layout")


def test_polar_provenance_cannot_change_published_faces():
    source = _PolarGridLayout()
    source.build(
        active_mask=np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0), np.linspace(-np.pi, np.pi, 3)),
        circular_angle=True,
        connect_diagonal_neighbors=False,
    )
    env = Environment._from_built_layout(
        source, _coordinates=EgocentricPolarSystem(True)
    )
    expected = env._binning.faces.build().A.copy()
    source._build_params_used["circular_angle"] = False
    env._layout_params_used["circular_angle"] = False
    np.testing.assert_array_equal(env._binning.faces.build().A, expected)


def test_custom_layout_refuses_before_opaque_deepcopy():
    class CustomLayout(RegularGridLayout):
        def __deepcopy__(self, memo):
            pytest.fail("opaque custom provider copied")

    with pytest.raises(TypeError, match="snapshot"):
        Environment._from_built_layout(CustomLayout())


@pytest.mark.parametrize(
    "method",
    ["finite_volume_geometry", "bin_sizes", "_axis_widths", "point_to_bin_index"],
)
def test_native_method_overrides_require_snapshot_migration(method):
    source = _grid()
    original = getattr(source, method)
    setattr(source, method, lambda *a, **kw: original(*a, **kw))
    with pytest.raises(TypeError, match=r"owned.*snapshot"):
        Environment._from_built_layout(source)


@pytest.mark.parametrize("wrapper", ["plain", "subset", "transform", "nested"])
def test_wrapped_opaque_locator_requires_snapshot_migration(wrapper):
    source = _grid()
    locator = SimpleNamespace(locate=source.point_to_bin_index)
    if wrapper in {"subset", "nested"}:
        locator = SubsetLocator(locator, np.arange(9, dtype=np.intp))
    if wrapper in {"transform", "nested"}:
        locator = TransformedLocator(locator, np.eye(3))
    binning = replace(binning_from_layout(source, source.connectivity), locator=locator)
    with pytest.raises(TypeError, match="snapshot"):
        Environment._from_binning(
            binning,
            connectivity=source.connectivity,
            dimension_ranges=source.dimension_ranges,
            layout_type_tag="RegularGrid",
            layout_type="grid",
            build_params={},
            is_linearized_track=False,
        )


def test_arena_snapshot_owns_native_arrays_and_recursive_frames():
    source = MaskArena((np.arange(3.0), np.arange(3.0)), np.ones((2, 2), dtype=bool))
    transformed = _TransformedArena(
        source, np.array([[1.0, 0, 2], [0, 1.0, 3], [0, 0, 1.0]])
    )
    snapshot = own_arena(transformed)
    source.active_mask.setflags(write=True)
    source.active_mask[:] = False
    transformed.forward.setflags(write=True)
    transformed.forward[0, -1] = 999
    np.testing.assert_array_equal(
        snapshot.contains([[2.5, 3.5], [9, 9]]), [True, False]
    )


def test_custom_arena_is_refused_even_under_native_transform():
    class CustomArena(OpenArena):
        def __deepcopy__(self, memo):
            pytest.fail("opaque custom arena was copied")

    source = CustomArena(box(0, 0, 2, 2))
    for value in (source, _TransformedArena(source, np.eye(3))):
        with pytest.raises(TypeError, match="snapshot"):
            own_arena(value)
