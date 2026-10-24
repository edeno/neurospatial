"""Saved environments restore exact geometry without rerunning layout builders."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from environment_helpers import publish_binning, publish_environment

from neurospatial import Environment, Regions
from neurospatial.environment._fields import face_geometry
from neurospatial.geometry._binning import KnownFaces, MissingFaces
from neurospatial.geometry._codecs import encode
from neurospatial.io import from_dict, from_file, to_dict, to_file


def _same_value(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same_value(actual[key], expected[key])
    elif isinstance(expected, (tuple, list, np.ndarray)):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def test_file_round_trip_identical(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    assert isinstance(loaded, Environment)
    assert type(loaded) is type(original)
    assert loaded.name == original.name
    assert loaded.units == original.units
    assert loaded.frame == original.frame
    assert loaded.layout_type == original.layout_type
    assert loaded.is_linearized_track == original.is_linearized_track
    np.testing.assert_array_equal(loaded.dimension_ranges, original.dimension_ranges)

    expected = replace(
        original._binning, faces=KnownFaces(original._binning.build_faces())
    )
    expected_meta, expected_arrays = encode(expected, "binning")
    actual_meta, actual_arrays = encode(loaded._binning, "binning")
    assert actual_meta == expected_meta
    assert actual_arrays.keys() == expected_arrays.keys()
    for name, array in expected_arrays.items():
        assert actual_arrays[name].dtype == array.dtype
        np.testing.assert_array_equal(actual_arrays[name], array, err_msg=name)
    for array in (
        loaded.bin_centers,
        loaded.bin_sizes,
        loaded._binning.connections.all_pairs,
        loaded._binning.connections.all_distances,
        loaded._binning.connections.is_open,
        loaded._binning.connections.pairs,
        loaded._binning.connections.pair_distances,
        *face_geometry(loaded._cache, loaded._binning),
    ):
        assert not array.flags.writeable

    _same_value(dict(loaded.connectivity.graph), dict(original.connectivity.graph))
    assert list(loaded.connectivity.nodes) == list(original.connectivity.nodes)
    assert list(loaded.connectivity.edges) == list(original.connectivity.edges)
    for node in original.connectivity:
        _same_value(
            dict(loaded.connectivity.nodes[node]),
            dict(original.connectivity.nodes[node]),
        )
    for edge in original.connectivity.edges:
        _same_value(
            dict(loaded.connectivity.edges[edge]),
            dict(original.connectivity.edges[edge]),
        )
    assert list(loaded.regions) == list(original.regions)
    for name in original.regions:
        _same_value(loaded.regions[name].to_dict(), original.regions[name].to_dict())


def test_loaded_locator_matches_recorded_geometric_queries(
    environment_round_trip, request
):
    original, restore = environment_round_trip
    loaded = restore()
    case = request.node.callspec.params["built_environment"]
    path = Path(__file__).parents[1] / "layout/data/locator_parity.npz"
    with np.load(path) as probes:
        for probe in ("centers", "random", "boundaries", "nonfinite", "empty"):
            points = probes[f"{case}/lookup/{probe}/points"]
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                expected = original._binning.locator.locate(points)
            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                actual = loaded._binning.locator.locate(points)
            np.testing.assert_array_equal(actual, expected)
            assert [(w.category, str(w.message)) for w in actual_warnings] == [
                (w.category, str(w.message)) for w in expected_warnings
            ]


def test_load_never_builds(environment_round_trip, forbid_layout_builds):
    original, restore = environment_round_trip
    loaded = restore()
    np.testing.assert_array_equal(loaded.bin_centers, original.bin_centers)
    np.testing.assert_array_equal(
        loaded._binning.locator.locate(loaded.bin_centers), np.arange(loaded.n_bins)
    )


@pytest.mark.parametrize("mode", ["transition", "density", "average"])
def test_loaded_environment_smooths_like_original(environment_round_trip, mode):
    original, restore = environment_round_trip
    loaded = restore()
    field = np.random.default_rng(14).uniform(0.1, 4.0, original.n_bins)
    np.testing.assert_array_equal(
        loaded.smooth(field, bandwidth=5.0, mode=mode),
        original.smooth(field, bandwidth=5.0, mode=mode),
    )


def test_loaded_finite_volume_operators_match_original(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    np.testing.assert_array_equal(
        loaded.compute_kernel(bandwidth=2.0, mode="density"),
        original.compute_kernel(bandwidth=2.0, mode="density"),
    )
    for method in ("get_differential_operator", "get_divergence_operator"):
        actual = getattr(loaded, method)().toarray()
        expected = getattr(original, method)().toarray()
        np.testing.assert_array_equal(actual, expected, err_msg=method)


def test_loaded_boundary_bins_match_original(environment_round_trip):
    original, restore = environment_round_trip
    np.testing.assert_array_equal(restore().boundary_bins, original.boundary_bins)


def test_loaded_snapshot_pickle_restores_protected_arrays(environment_round_trip):
    original, restore = environment_round_trip
    loaded = pickle.loads(pickle.dumps(restore()))
    np.testing.assert_array_equal(loaded.bin_centers, original.bin_centers)
    np.testing.assert_array_equal(loaded.bin_sizes, original.bin_sizes)
    np.testing.assert_array_equal(
        loaded._binning.locator.locate(loaded.bin_centers), np.arange(loaded.n_bins)
    )
    for array in (
        loaded.bin_centers,
        loaded.bin_sizes,
        loaded._binning.connections.pairs,
        *face_geometry(loaded._cache, loaded._binning),
    ):
        assert not array.flags.writeable


def test_derived_from_loaded_environment(environment_round_trip, forbid_layout_builds):
    original, restore = environment_round_trip
    loaded = restore()
    copied = loaded.with_metadata()
    np.testing.assert_array_equal(copied.bin_centers, original.bin_centers)
    np.testing.assert_array_equal(copied.bin_sizes, original.bin_sizes)
    np.testing.assert_array_equal(
        copied._binning.locator.locate(original.bin_centers), np.arange(original.n_bins)
    )
    field = np.arange(original.n_bins, dtype=float)
    np.testing.assert_array_equal(
        copied.smooth(field, bandwidth=2.0, mode="average"),
        loaded.smooth(field, bandwidth=2.0, mode="average"),
    )
    region_builder = Regions(copied.regions.values())
    region_builder.add("copy_only", point=tuple(original.bin_centers[-1]))
    copied = copied.with_regions(region_builder)
    assert "copy_only" not in loaded.regions


def test_old_environment_file_refused():
    path = Path(__file__).parent / "data/environment_v1"
    with pytest.raises(ValueError) as caught:
        from_file(path)
    text = str(caught.value)
    assert "Environment-v1" in text
    assert "Why:" in text
    assert "Fix:" in text
    assert "save" in text.lower()


def test_old_environment_dictionary_refused():
    path = Path(__file__).parent / "data/environment_v1.json"
    with pytest.raises(ValueError, match="Environment-v1"):
        from_dict(json.loads(path.read_text()))


@pytest.fixture
def simple_saved_grid():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )


@pytest.mark.parametrize("transport", ["file", "dict"])
def test_saved_format_stores_binning(simple_saved_grid, tmp_path, transport):
    if transport == "dict":
        payload = to_dict(simple_saved_grid)
        assert isinstance(payload["arrays"], dict)
        json.dumps(payload)
    else:
        path = tmp_path / "saved"
        to_file(simple_saved_grid, path)
        payload = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as arrays:
            assert "binning/bins/centers" in arrays
            assert "binning/connections/all_pairs" in arrays
            assert "binning/connections/is_open" in arrays
            assert "binning/faces/volumes" in arrays
    assert payload["schema_version"] == "Environment-v3"
    assert payload["layout_type"] == simple_saved_grid.layout_type
    assert payload["derivations"] == list(simple_saved_grid.derivations)
    assert payload["binning"]["codec_version"] == 6


@pytest.mark.parametrize("transport", ["file", "dict"])
def test_nonfinite_saved_centers_name_source(simple_saved_grid, tmp_path, transport):
    if transport == "dict":
        payload = to_dict(simple_saved_grid)
        payload["arrays"]["binning/bins/centers"][0][0] = float("nan")

        def operation():
            return from_dict(payload)

        source = "dictionary"
    else:
        path = tmp_path / "invalid_geometry"
        to_file(simple_saved_grid, path)
        with np.load(path.with_suffix(".npz")) as saved:
            arrays = {name: saved[name].copy() for name in saved}
        arrays["binning/bins/centers"][0, 0] = np.nan
        np.savez_compressed(path.with_suffix(".npz"), **arrays)

        def operation():
            return from_file(path)

        source = "invalid_geometry"
    with pytest.raises(ValueError) as caught:
        operation()
    assert "centers" in str(caught.value)
    assert source in str(caught.value)


def test_conflicting_descriptive_centers_refused(simple_saved_grid, tmp_path):
    path = tmp_path / "conflicting_geometry"
    to_file(simple_saved_grid, path)
    with np.load(path.with_suffix(".npz")) as saved:
        arrays = {name: saved[name].copy() for name in saved}
    arrays["bin_centers"][0][0] += 0.01
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    with pytest.raises(ValueError, match="centers"):
        from_file(path)


@pytest.mark.parametrize("transport", ["file", "dict"])
def test_missing_faces_reason_survives_public_round_trip(
    simple_saved_grid, tmp_path, transport
):
    reason = "These externally supplied bins have no measured cell faces."
    env = simple_saved_grid
    env._geometry = replace(
        env._geometry, binning=replace(env._binning, faces=MissingFaces(reason))
    )
    if transport == "dict":
        loaded = from_dict(to_dict(env))
    else:
        path = tmp_path / "missing_faces"
        to_file(env, path)
        loaded = from_file(path)
    assert not loaded._binning.faces.available
    with pytest.raises(NotImplementedError) as caught:
        loaded.smooth(np.ones(loaded.n_bins), bandwidth=1.0, mode="average")
    assert reason in str(caught.value)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize("built_environment", ["grid8"], indirect=True)
def test_loaded_regular_grid_still_rebins(environment_round_trip):
    original, restore = environment_round_trip
    expected = original.rebin(5)
    actual = restore().rebin(5)
    np.testing.assert_array_equal(actual.bin_centers, expected.bin_centers)
    np.testing.assert_array_equal(actual.bin_sizes, expected.bin_sizes)
    assert list(actual.connectivity.edges) == list(expected.connectivity.edges)


@pytest.mark.parametrize("transport", ["file", "dict"])
@pytest.mark.parametrize(
    "labels",
    [(("left", (1, 2)), ("right", (3, 4))), ((0, (1, 2)), (3, (4, 5)))],
    ids=["mixed", "numeric"],
)
def test_saved_track_preserves_nested_tuple_node_labels(transport, labels, tmp_path):
    """Data-backed tracks preserve node identities in their stored geometry."""
    native = Environment.linear_track(
        endpoints=[[0.0, 0.0], [2.0, 0.0]], bin_size=1.0, track_width=1.0
    )
    component = native._binning.linearization
    old_nodes = list(component.graph)
    graph = nx.relabel_nodes(
        component.graph, dict(zip(old_nodes, labels, strict=True)), copy=True
    )
    component = replace(component, graph=graph, edge_order=[labels])
    binning = replace(
        native._binning,
        linearization=component,
        faces=KnownFaces(face_geometry(native._cache, native._binning)),
    )
    parameters = {"edge_spacing": 0.0, "bin_size": 1.0}
    facade = publish_binning(
        binning,
        dimension_ranges=native.dimension_ranges,
        layout_type="Graph",
        build_params=parameters,
        arena=native.arena,
    )
    original = publish_environment(
        layout=facade, layout_type_used="Graph", arena=native.arena
    )
    if transport == "dict":
        loaded = from_dict(json.loads(json.dumps(to_dict(original))))
    else:
        path = tmp_path / "nested_track"
        to_file(original, path)
        loaded = from_file(path)
    assert list(loaded._binning.linearization.graph) == list(labels)
    assert loaded._binning.linearization.edge_order == (labels,)
    np.testing.assert_array_equal(
        loaded.bin_at(original.bin_centers), np.arange(original.n_bins)
    )
    coordinates = np.array([0.5, 1.5])
    np.testing.assert_array_equal(
        loaded.linear_to_nd(coordinates), original.linear_to_nd(coordinates)
    )


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
@pytest.mark.parametrize("corruption", ["swapped", "missing", "fractional", "bool"])
def test_saved_track_refuses_corrupt_linearization_edge_ids(
    built_environment, corrupted_environment_round_trip, corruption
):
    """Track edge enumeration is shared by forward and inverse projection."""
    env = built_environment
    points = np.array([[1.0, 0.0], [6.0, 1.0], [7.0, 0.0]])
    np.testing.assert_array_equal(env.to_linear(points), [1.0, 7.0, 13.0])

    def corrupt(metadata, arrays):
        links = metadata["binning"]["linearization"]["graph"]["links"]
        if corruption == "swapped":
            links[0]["edge_id"], links[1]["edge_id"] = (
                links[1]["edge_id"],
                links[0]["edge_id"],
            )
        elif corruption == "missing":
            links[0].pop("edge_id")
        elif corruption == "fractional":
            links[0]["edge_id"] = 0.5
        else:
            links[1]["edge_id"] = True

    with pytest.raises(ValueError, match="edge_id"):
        corrupted_environment_round_trip(corrupt)


@pytest.mark.parametrize("built_environment", ["grid8", "polar"], indirect=True)
def test_saved_grid_refuses_corrupt_physical_measures(corrupted_environment_round_trip):
    """Agreement between duplicated measures cannot override cell geometry."""

    def corrupt(metadata, arrays):
        arrays["binning/bins/sizes"] *= 2.0
        arrays["binning/faces/volumes"] *= 2.0

    with pytest.raises(ValueError, match=r"sizes|volumes|measures|area"):
        corrupted_environment_round_trip(corrupt)
