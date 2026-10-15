"""Persistence codecs preserve geometry, ownership and explicit failure reasons."""

from __future__ import annotations

import copy
import json
import pickle
from dataclasses import fields, is_dataclass, replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from neurospatial import Environment
from neurospatial.layout._binning import (
    KnownFaces,
    LayoutLocator,
    MissingFaces,
    TrackLinearization,
)
from neurospatial.layout._codecs import decode, encode
from neurospatial.layout._components import Outline, TriangleShapes
from neurospatial.layout._diffusion import FaceGeometry
from neurospatial.layout._locators import MeshLocator, NearestBinLocator


@pytest.fixture(
    params=[
        "bins",
        "faces",
        "known_faces",
        "missing_faces",
        "grid",
        "track",
        "hexagons",
        "triangles",
        "outline",
        "multipolygon",
        "rectilinear_locator",
        "hex_locator",
        "mesh_locator",
        "nearest_locator",
        "binning",
    ]
)
def codec_part(request):
    """Each persistent kind, including unavailable geometry and polygon holes."""
    env = Environment.from_grid_mask(
        np.array([[True, False], [True, True]]),
        grid_edges=(np.arange(3.0),) * 2,
    )
    binning = env._binning
    faces = binning.faces.build()
    triangle = Polygon([(0, 0), (2, 0), (0, 2)])
    second = Polygon([(4, 0), (6, 0), (4, 2)])
    if request.param == "bins":
        return binning.bins
    if request.param == "faces":
        return faces
    if request.param == "known_faces":
        return KnownFaces(faces)
    if request.param == "missing_faces":
        return MissingFaces("The geometry was projected without valid shared faces.")
    if request.param == "grid":
        return binning.rectilinear
    if request.param == "track":
        return Environment.linear_track(
            endpoints=[[0.0, 0.0], [10.0, 0.0]], bin_size=2.0
        )._binning.linearization
    if request.param in {"hexagons", "hex_locator"}:
        hex_binning = Environment.from_layout(
            "Hexagonal", {"hexagon_width": 2.0, "dimension_ranges": [(0, 5)] * 2}
        )._binning
        return (
            hex_binning.shapes if request.param == "hexagons" else hex_binning.locator
        )
    if request.param == "triangles":
        return TriangleShapes(
            np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
            np.array([[0, 1, 2]]),
        )
    if request.param == "outline":
        return Outline(
            Polygon([(0, 0), (8, 0), (8, 8), (0, 8)], [[(2, 2), (4, 2), (2, 4)]])
        )
    if request.param == "multipolygon":
        return Outline(MultiPolygon([triangle, second]))
    if request.param == "rectilinear_locator":
        return binning.locator
    if request.param == "mesh_locator":
        return MeshLocator(
            np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
            np.array([[1, 2, 0]]),
            np.array([0]),
            triangle,
        )
    if request.param == "nearest_locator":
        return NearestBinLocator(
            np.array([[0.0, 0.0], [2.0, 0.0]]), 3.0, np.array([1, 0])
        )
    return replace(binning, faces=KnownFaces(faces))


def _assert_encoding_equal(expected, actual):
    expected_meta, expected_arrays = expected
    actual_meta, actual_arrays = actual
    assert actual_meta == expected_meta
    assert actual_arrays.keys() == expected_arrays.keys()
    for name, value in expected_arrays.items():
        assert actual_arrays[name].dtype == value.dtype, name
        np.testing.assert_array_equal(actual_arrays[name], value, err_msg=name)


def _geometry_arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for item in fields(value):
            if not item.name.startswith("_"):
                yield from _geometry_arrays(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _geometry_arrays(item)


def test_codec_round_trip(codec_part):
    encoded = encode(codec_part, "part")
    meta, arrays = encoded
    assert json.loads(json.dumps(meta)) == meta
    assert meta["codec_version"] == 1
    assert isinstance(meta["codec"], str)
    assert all(name.startswith("part/") for name in arrays)
    restored = decode(meta, arrays, "part")
    if isinstance(codec_part, FaceGeometry):
        assert isinstance(restored, KnownFaces)
    _assert_encoding_equal(encoded, encode(restored, "part"))
    for array in _geometry_arrays(restored):
        assert not array.flags.writeable
        assert all(not np.shares_memory(array, source) for source in arrays.values())


@pytest.mark.parametrize(
    "restore",
    [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
    ids=["deepcopy", "pickle"],
)
def test_decoded_geometry_stays_read_only_after_serialization(codec_part, restore):
    meta, arrays = encode(codec_part, "part")
    decoded = decode(meta, arrays, "part")
    restored = restore(decoded)
    _assert_encoding_equal(encode(decoded, "part"), encode(restored, "part"))
    for array in _geometry_arrays(restored):
        assert not array.flags.writeable


def test_decode_does_not_freeze_or_retain_payload_arrays(codec_part):
    meta, encoded_arrays = encode(codec_part, "part")
    caller_arrays = {name: value.copy() for name, value in encoded_arrays.items()}
    decoded = decode(meta, caller_arrays, "part")
    expected = encode(decoded, "part")
    for array in caller_arrays.values():
        assert array.flags.writeable
        array[...] = 0
    _assert_encoding_equal(expected, encode(decoded, "part"))


def test_missing_faces_preserves_actionable_reason():
    reason = "A warped mesh has no valid finite-volume faces."
    meta, arrays = encode(MissingFaces(reason), "faces")
    restored = decode(meta, arrays, "faces")
    assert not restored.available
    with pytest.raises(NotImplementedError) as caught:
        restored.build()
    assert reason in str(caught.value)
    assert "Fix:" in str(caught.value)
    assert "smooth" in str(restored.missing_error("smooth"))


def test_unregistered_layout_locator_refuses():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    with pytest.raises(ValueError) as caught:
        encode(LayoutLocator(env.layout), "locator")
    assert type(env.layout).__name__ in str(caught.value)
    assert "Fix:" in str(caught.value)


def test_unregistered_layout_linearization_refuses():
    env = Environment.linear_track(endpoints=[[0.0, 0.0], [2.0, 0.0]], bin_size=1.0)
    with pytest.raises(ValueError) as caught:
        encode(env.layout, "linearization")
    assert type(env.layout).__name__ in str(caught.value)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize("field,value", [("codec", "unknown"), ("codec_version", 999)])
def test_unknown_codec_or_version_refused(field, value):
    meta, arrays = encode(MissingFaces("No faces"), "faces")
    meta[field] = value
    with pytest.raises(ValueError, match=r"codec|version"):
        decode(meta, arrays, "faces")


@pytest.mark.parametrize("field", ["centers", "sizes", "pairs", "pair_distances"])
def test_spatial_bins_decode_validates_geometry(field):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    meta, arrays = encode(env._binning.bins, "bins")
    arrays = {name: value.copy() for name, value in arrays.items()}
    key = f"bins/{field}"
    if field == "pairs":
        arrays[key][0, 0] = env.n_bins
    else:
        arrays[key].flat[0] = np.nan
    with pytest.raises(ValueError, match=field):
        decode(meta, arrays, "bins")


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "complex",
        "fractional_pairs",
        "overflow_pairs",
        "duplicate_map",
        "fractional_map",
        "mask_dtype",
        "nonmonotonic_edges",
        "presence_flag",
        "invalid_wkb",
    ],
)
def test_decode_refuses_malformed_geometry_payload(corruption):
    env = Environment.from_grid_mask(
        np.array([[True, False], [True, True]]), grid_edges=(np.arange(3.0),) * 2
    )
    if corruption == "invalid_wkb":
        part = Outline(Polygon([(0, 0), (1, 0), (0, 1)]))
    elif corruption in {"missing", "complex", "fractional_pairs", "overflow_pairs"}:
        part = env._binning.bins
    else:
        part = env._binning.locator
    meta, arrays = encode(part, "part")
    if corruption == "missing":
        del arrays["part/centers"]
        pattern = "centers"
    elif corruption == "complex":
        arrays["part/centers"] = arrays["part/centers"].astype(complex)
        arrays["part/centers"][0, 0] += 1j
        pattern = "centers"
    elif corruption in {"fractional_pairs", "overflow_pairs"}:
        array = arrays["part/pairs"].astype(float)
        array[0, 0] = 0.5 if corruption == "fractional_pairs" else 2.0**63
        arrays["part/pairs"] = array
        pattern = "pairs"
    elif corruption == "duplicate_map":
        arrays["part/flat_to_active"] = np.zeros(4, dtype=np.intp)
        pattern = "flat_to_active"
    elif corruption == "fractional_map":
        arrays["part/flat_to_active"] = arrays["part/flat_to_active"].astype(float)
        arrays["part/flat_to_active"][0] = 0.5
        pattern = "flat_to_active"
    elif corruption == "mask_dtype":
        arrays["part/active_mask"] = arrays["part/active_mask"].astype(int)
        pattern = "active_mask"
    elif corruption == "nonmonotonic_edges":
        arrays["part/edges_0"][1] = arrays["part/edges_0"][0]
        pattern = "edges_0"
    elif corruption == "presence_flag":
        meta["has_active_mask"] = "false"
        pattern = "has_active_mask"
    else:
        arrays["part/wkb"] = np.arange(7, dtype=np.uint8)
        pattern = "wkb|WKB|geometry"
    with pytest.raises(ValueError, match=pattern):
        decode(meta, arrays, "part")


@pytest.mark.parametrize("field", ["edges", "A", "distance", "volumes"])
def test_face_decode_rejects_invalid_arrays(field):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    meta, arrays = encode(env._binning.faces.build(), "faces")
    arrays[f"faces/{field}"] = arrays[f"faces/{field}"].copy()
    if field == "edges":
        arrays[f"faces/{field}"][0, 0] = env.n_bins
    else:
        arrays[f"faces/{field}"].flat[0] = -1.0
    with pytest.raises(ValueError, match=r"edges|areas|distances|volumes"):
        decode(meta, arrays, "faces")


def test_track_codec_preserves_nested_tuple_node_ids_without_mutating_metadata():
    u, v = ("left", (1, 2)), ("right", (3, 4))
    graph = nx.Graph()
    graph.add_node(u, pos=(0.0, 0.0))
    graph.add_node(v, pos=(2.0, 0.0))
    graph.add_edge(u, v, distance=2.0, edge_id=0)
    component = TrackLinearization(
        graph, [(u, v)], 0.0, (np.arange(3.0),), np.ones(2, dtype=bool)
    )
    meta, arrays = encode(component, "track")
    unchanged = copy.deepcopy(meta)
    decoded = decode(meta, arrays, "track")
    assert meta == unchanged
    assert list(decoded.graph.nodes) == [u, v]
    assert decoded.edge_order == ((u, v),)
    _assert_encoding_equal((meta, arrays), encode(decoded, "track"))
