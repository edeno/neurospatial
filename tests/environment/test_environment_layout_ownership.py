"""Published layout bridges own science and detach their public exports."""

from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from neurospatial.arena import MaskArena, OpenArena, _TransformedArena
from neurospatial.encoding.grid import _detect_grid_method
from neurospatial.environment._ownership import own_arena, own_layout
from neurospatial.layout import RegularGridLayout, create_layout
from neurospatial.layout._binning import LayoutLocator, binning_from_layout
from neurospatial.layout._binning_layout import BinningLayout
from neurospatial.layout._capabilities import (
    RectilinearGridProvider,
    TrackLinearizationProvider,
)
from neurospatial.layout._diffusion import FiniteVolumeGeometryProvider
from neurospatial.layout._locators import SubsetLocator, TransformedLocator
from neurospatial.layout.engines._polar import _PolarGridLayout


def _grid():
    return create_layout(
        "RegularGrid",
        bin_size=1.0,
        dimension_ranges=[(0.0, 3.0), (0.0, 3.0)],
        infer_active_bins=False,
    )


def _restore(value, method):
    if method == "copy":
        return copy.copy(value)
    if method == "deepcopy":
        return copy.deepcopy(value)
    if method == "pickle":
        return pickle.loads(pickle.dumps(value))
    return value


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle"])
def test_bridge_detaches_buffers_headers_graphs_and_provenance(method):
    source = _grid()
    source.connectivity.graph["nested"] = {"values": np.array([1.0, 2.0])}
    source.connectivity.nodes[0]["nested"] = {"values": [1, 2]}
    edge = next(iter(source.connectivity.edges))
    source.connectivity.edges[edge]["nested"] = {"values": np.array([3.0])}
    source._build_params_used["nested"] = {"values": np.array([4.0])}
    expected = source.bin_centers.copy()
    bridge = _restore(own_layout(source), method)

    source.bin_centers[:] = -99
    source.grid_edges[0][:] = -99
    source.active_mask[:] = False
    source.connectivity.graph["nested"]["values"][:] = -99
    source._build_params_used["nested"]["values"][:] = -99

    for array in (
        bridge.bin_centers,
        bridge.bin_sizes(),
        bridge.grid_edges[0],
        bridge.active_mask,
    ):
        array.setflags(write=True)
        array.flat[0] = 123
        array.dtype = np.uint8
        array.shape = (array.size,)

    graph = bridge.connectivity
    assert nx.is_frozen(graph)
    graph.graph["nested"]["values"][:] = 123
    graph.nodes[0]["nested"]["values"][0] = 123
    graph.edges[edge]["nested"]["values"][:] = 123
    bridge._build_params_used["nested"]["values"][:] = 123
    exported = bridge.binning()
    exported.bins.centers.setflags(write=True)
    exported.bins.centers[:] = 123
    exported.faces.layout.grid_edges[0][:] = 123

    np.testing.assert_array_equal(bridge.bin_centers, expected)
    np.testing.assert_array_equal(bridge.bin_sizes(), np.ones(9))
    np.testing.assert_array_equal(bridge.grid_edges[0], np.arange(4.0))
    np.testing.assert_array_equal(bridge.active_mask, np.ones((3, 3), dtype=bool))
    np.testing.assert_array_equal(
        bridge.point_to_bin_index(expected), np.arange(len(expected))
    )
    np.testing.assert_array_equal(bridge.connectivity.graph["nested"]["values"], [1, 2])
    assert bridge.connectivity.nodes[0]["nested"]["values"] == [1, 2]
    np.testing.assert_array_equal(
        bridge.connectivity.edges[edge]["nested"]["values"], [3]
    )
    np.testing.assert_array_equal(bridge._build_params_used["nested"]["values"], [4])


@pytest.mark.parametrize(
    "attribute",
    [
        "bin_centers",
        "connectivity",
        "dimension_ranges",
        "grid_edges",
        "grid_shape",
        "active_mask",
        "_layout_type_tag",
        "_build_params_used",
        "plot",
        "binning",
    ],
)
def test_bridge_refuses_rebinding(attribute):
    bridge = own_layout(_grid())
    with pytest.raises(AttributeError, match="read-only"):
        setattr(bridge, attribute, None)
    with pytest.raises(AttributeError, match="read-only"):
        delattr(bridge, attribute)
    with pytest.raises(TypeError, match="new environment"):
        bridge.build(bin_size=2.0)


def test_bridge_preserves_native_context_and_capabilities(built_environment):
    # Public binning exports detach the native lazy face builder as well.
    source = built_environment.layout.binning().faces.layout
    bridge = own_layout(source)
    assert bridge.layout_type == source.layout_type
    assert bridge._layout_type_tag == source._layout_type_tag
    assert bridge.is_grid_compatible == source.is_grid_compatible
    assert bridge.is_linearized_track == source.is_linearized_track
    assert bridge.dimension_ranges == tuple(source.dimension_ranges)
    assert bridge.grid_shape == source.grid_shape
    np.testing.assert_array_equal(bridge.active_mask, source.active_mask)
    for actual, expected in zip(bridge.grid_edges, source.grid_edges, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bridge.bin_sizes(), source.bin_sizes())
    np.testing.assert_array_equal(
        bridge.point_to_bin_index(source.bin_centers),
        source.point_to_bin_index(source.bin_centers),
    )
    assert not isinstance(bridge, RectilinearGridProvider)
    assert not isinstance(bridge, TrackLinearizationProvider)
    assert not isinstance(bridge, FiniteVolumeGeometryProvider)
    assert own_layout(bridge) is bridge
    assert _detect_grid_method(SimpleNamespace(layout=bridge)) == _detect_grid_method(
        SimpleNamespace(layout=source)
    )


def test_face_building_stays_lazy_and_owns_native_builder(monkeypatch):
    source = _grid()
    original = RegularGridLayout.finite_volume_geometry
    calls = []

    def record(layout):
        calls.append(layout)
        warnings.warn("lazy face construction", UserWarning, stacklevel=2)
        return original(layout)

    monkeypatch.setattr(RegularGridLayout, "finite_volume_geometry", record)
    with warnings.catch_warnings(record=True) as emitted:
        bridge = own_layout(source)
    assert not calls
    assert not emitted
    source.grid_edges[0][:] = -99
    with pytest.warns(UserWarning, match="lazy face construction"):
        faces = bridge._binning.faces.build()
    assert calls[0] is not source
    np.testing.assert_array_equal(faces.volumes, np.ones(9))


def test_polar_build_parameters_cannot_change_published_faces():
    source = _PolarGridLayout()
    source.build(
        active_mask=np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0), np.linspace(-np.pi, np.pi, 3)),
        circular_angle=True,
        connect_diagonal_neighbors=False,
    )
    bridge = own_layout(source)
    expected = bridge._binning.faces.build().A.copy()
    source._build_params_used["circular_angle"] = False
    bridge._build_params_used["circular_angle"] = False
    bridge.binning().faces.layout._build_params_used["circular_angle"] = False
    np.testing.assert_array_equal(bridge._binning.faces.build().A, expected)


def test_custom_layout_is_refused_before_opaque_deepcopy():
    class CustomLayout(RegularGridLayout):
        def __deepcopy__(self, memo):
            pytest.fail("opaque custom provider was copied")

    with pytest.raises(TypeError, match=r"owned.*snapshot|immutable.*snapshot"):
        own_layout(CustomLayout())


@pytest.mark.parametrize("wrapper", ["plain", "subset", "transform", "nested"])
def test_binning_facade_cannot_hide_a_delegated_custom_locator(wrapper):
    source = _grid()
    locator = LayoutLocator(source)
    if wrapper in {"subset", "nested"}:
        locator = SubsetLocator(locator, np.arange(9, dtype=np.intp))
    if wrapper in {"transform", "nested"}:
        locator = TransformedLocator(locator, np.eye(3))
    binning = replace(binning_from_layout(source, source.connectivity), locator=locator)
    facade = BinningLayout(
        binning,
        connectivity=source.connectivity,
        dimension_ranges=source.dimension_ranges,
        layout_type_tag="RegularGrid",
        layout_type="grid",
        build_params={},
        is_linearized_track=False,
    )
    with pytest.raises(TypeError, match=r"owned.*snapshot|immutable.*snapshot"):
        own_layout(facade)


def test_owned_binning_facade_detaches_original_data_and_lazy_faces():
    source = _grid()
    binning = binning_from_layout(source, source.connectivity)
    facade = BinningLayout(
        binning,
        connectivity=source.connectivity,
        dimension_ranges=source.dimension_ranges,
        layout_type_tag="RegularGrid",
        layout_type="grid",
        build_params=source._build_params_used,
        is_linearized_track=False,
    )
    bridge = own_layout(facade)
    binning.bins.centers.setflags(write=True)
    binning.bins.centers[:] = -99
    source.grid_edges[0][:] = -99
    np.testing.assert_array_equal(bridge.bin_centers[0], [0.5, 0.5])
    np.testing.assert_array_equal(bridge._binning.faces.build().volumes, np.ones(9))


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
        with pytest.raises(TypeError, match=r"owned.*snapshot|immutable.*snapshot"):
            own_arena(value)
