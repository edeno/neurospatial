"""Owned publication detaches source builders, arrays and graph metadata."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from environment_helpers import publish_binning, publish_environment
from shapely.geometry import box

from neurospatial.arena import MaskArena, OpenArena
from neurospatial.geometry._arena import _TransformedArena
from neurospatial.geometry._coordinates import EgocentricPolarSystem
from neurospatial.geometry._locators import SubsetLocator, TransformedLocator
from neurospatial.geometry._ownership import own_arena
from neurospatial.layout import (
    GraphLayout,
    HexagonalLayout,
    ImageMaskLayout,
    MaskedGridLayout,
    RegularGridLayout,
    ShapelyPolygonLayout,
    TriangularMeshLayout,
    create_layout,
)
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
    source._build_params_used["choice"] = 4.0
    expected = source.bin_centers.copy()
    env = restore(publish_environment(layout=source))
    source.bin_centers[:] = -99
    source.grid_edges[0][:] = -99
    source.active_mask[:] = False
    source._build_params_used["choice"] = -99.0
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
    assert env.layout_parameters["choice"] == 4.0


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
    env = publish_environment(layout=_grid())
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
    from neurospatial.geometry._native_faces import NativeFaces

    source = _grid()
    original = NativeFaces.build
    calls = []

    def record(recipe):
        calls.append(recipe)
        return original(recipe)

    monkeypatch.setattr(NativeFaces, "build", record)
    env = publish_environment(layout=source)
    assert not calls
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
    env = publish_environment(layout=source, _coordinates=EgocentricPolarSystem(True))
    expected = env._binning.faces.build().A.copy()
    source._build_params_used["circular_angle"] = False
    np.testing.assert_array_equal(env._binning.faces.build().A, expected)


def test_polar_faces_come_from_the_layout_not_its_parameters():
    """Editing recorded parameters before publication cannot change the seam.

    With two angular bins, a circular ring's two sectors share both of their
    radial sides, so that face has twice the radial width.
    """
    source = _PolarGridLayout()
    source.build(
        active_mask=np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(1.0, 4.0), np.linspace(-np.pi, np.pi, 3)),
        circular_angle=True,
        connect_diagonal_neighbors=False,
    )
    source._build_params_used["circular_angle"] = False
    env = publish_environment(layout=source, _coordinates=EgocentricPolarSystem(True))
    faces = env._binning.faces.build()
    radius = env.bin_centers[env._binning.bins.pairs, 0]
    same_ring = radius[:, 0] == radius[:, 1]
    assert np.count_nonzero(same_ring) == 2
    np.testing.assert_array_equal(faces.A[same_ring], [2.0, 2.0])


def test_track_build_record_keeps_choices_and_refuses_data():
    """A track's build record holds no graph; injected data is refused, not dropped."""
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edges_from([(0, 1), (1, 2)], distance=6.0)
    source = GraphLayout()
    source.build(
        graph_definition=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
        bin_size=2.0,
    )
    assert source._build_params_used == {"edge_spacing": 0.0, "bin_size": 2.0}
    env = publish_environment(layout=source)
    assert env.layout_parameters == {"edge_spacing": 0.0, "bin_size": 2.0}

    source._build_params_used["graph_definition"] = graph
    with pytest.raises(TypeError, match="'graph_definition' is not a build choice"):
        publish_environment(layout=source)


def test_custom_engine_publishes_through_its_binning_without_being_copied():
    """A non-native engine uses the same binning() contract as built-ins."""

    class CustomLayout(RegularGridLayout):
        def __deepcopy__(self, memo):
            pytest.fail("the layout itself must not be copied")

    source = CustomLayout()
    source.build(
        bin_size=1.0, dimension_ranges=[(0.0, 3.0), (0.0, 2.0)], infer_active_bins=False
    )
    env = publish_environment(layout=source)
    np.testing.assert_array_equal(env.bin_centers, source.bin_centers)
    np.testing.assert_array_equal(env.bin_sizes, np.ones(6))
    np.testing.assert_array_equal(
        env.bin_at(source.bin_centers + 0.25), np.arange(env.n_bins)
    )
    assert env._binning.faces.available
    assert env.layout_type == "RegularGrid"


def test_layout_without_binning_is_refused_before_copying():
    """An object that cannot supply owned bins gets an actionable refusal."""

    class Opaque:
        def __deepcopy__(self, memo):
            pytest.fail("opaque provider copied")

    with pytest.raises(TypeError, match=r"snapshot adapter.*binning\(\) method"):
        publish_environment(layout=Opaque())


@pytest.mark.parametrize(
    "layout_type",
    [
        RegularGridLayout,
        MaskedGridLayout,
        ImageMaskLayout,
        ShapelyPolygonLayout,
        HexagonalLayout,
        GraphLayout,
        TriangularMeshLayout,
        _PolarGridLayout,
    ],
)
def test_unbuilt_layout_refuses_binning(layout_type):
    from neurospatial._exceptions import LayoutNotBuiltError

    with pytest.raises(LayoutNotBuiltError, match="not built"):
        layout_type().binning()


@pytest.mark.parametrize("wrapper", ["plain", "subset", "transform", "nested"])
def test_wrapped_opaque_locator_requires_snapshot_migration(wrapper):
    source = _grid()
    locator = SimpleNamespace(locate=source.locator().locate)
    if wrapper in {"subset", "nested"}:
        locator = SubsetLocator(locator, np.arange(9, dtype=np.intp))
    if wrapper in {"transform", "nested"}:
        locator = TransformedLocator(locator, np.eye(3))
    binning = replace(source.binning(), locator=locator)
    with pytest.raises(TypeError, match="snapshot"):
        publish_binning(
            binning,
            dimension_ranges=source.dimension_ranges,
            layout_type="RegularGrid",
            category="grid",
            build_params={},
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


def _layout_state(layout):
    """Everything publication reads from a built layout, as comparable values."""
    from neurospatial.geometry._codecs import encode, graph_to_node_link

    metadata, arrays = encode(layout.binning(), "layout")
    return (
        metadata,
        {key: np.asarray(value).tolist() for key, value in arrays.items()},
        graph_to_node_link(layout.connectivity),
        pickle.dumps(layout._build_params_used),
    )


_TRACK = nx.Graph()
_TRACK.add_node(0, pos=(0.0, 0.0))
_TRACK.add_node(1, pos=(6.0, 0.0))
_TRACK.add_node(2, pos=(6.0, 6.0))
_TRACK.add_edge(0, 1, distance=6.0)
_TRACK.add_edge(1, 2, distance=6.0)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("RegularGrid", {"bin_size": 1.0, "dimension_ranges": [(0.0, 4.0)] * 2}),
        ("Hexagonal", {"hexagon_width": 2.0, "dimension_ranges": [(0.0, 8.0)] * 2}),
        (
            "Graph",
            {
                "graph_definition": _TRACK,
                "edge_order": [(0, 1), (1, 2)],
                "edge_spacing": 0.0,
                "bin_size": 1.0,
            },
        ),
        (
            "TriangularMesh",
            {"boundary_polygon": box(0.0, 0.0, 6.0, 6.0), "point_spacing": 2.0},
        ),
    ],
)
def test_publication_leaves_the_built_layout_unchanged(kind, params):
    from neurospatial import Environment
    from neurospatial.environment import _publication

    layout = create_layout(kind, **params)
    before = _layout_state(layout)
    arena = None if kind == "Graph" else OpenArena(box(0.0, 0.0, 3.0, 8.0))
    env = _publication.publish_layout(
        Environment,
        layout,
        metadata=_publication._Metadata("", None, None),
        arena=arena,
        within_arena=True,
    )
    if arena is not None:
        assert env.n_bins < layout.binning().bins.n_bins
    assert _layout_state(layout) == before
