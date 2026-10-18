"""Public binning snapshots protect the environment's owned geometry."""

import copy
import pickle
from dataclasses import fields, is_dataclass, replace

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import KnownFaces
from neurospatial.geometry._native_faces import NativeFaces


def _arrays(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _arrays(getattr(value, field.name), seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item, seen)


@pytest.mark.parametrize("restore", ["direct", "copy", "pickle"])
def test_public_binning_is_read_only_with_independent_array_headers(
    built_environment, restore
):
    env = built_environment
    expected = [(array.shape, array.copy()) for array in _arrays(env._binning)]
    exported = env.binning
    if restore == "copy":
        exported = copy.deepcopy(exported)
    elif restore == "pickle":
        exported = pickle.loads(pickle.dumps(exported))
    arrays = list(_arrays(exported))
    assert len(arrays) == len(expected)
    for exported_array in arrays:
        assert not exported_array.flags.writeable
        if exported_array.size:
            with pytest.raises(ValueError, match="read-only"):
                exported_array.flat[0] = 0
        # A snapshot's headers are its own, so reshaping one cannot reach
        # the environment.
        exported_array.shape = (exported_array.size,)
    for actual, (shape, values) in zip(_arrays(env.binning), expected, strict=True):
        assert actual.shape == shape
        np.testing.assert_array_equal(actual, values)


def test_public_binning_read_does_not_build_native_faces(monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )

    def refuse_build(self):
        raise AssertionError("Reading binning must not materialize native faces")

    monkeypatch.setattr(NativeFaces, "build", refuse_build)
    exported = env.binning
    assert isinstance(exported.faces, NativeFaces)
    np.testing.assert_array_equal(exported.bins.centers, env.bin_centers)


def test_from_bins_builds_and_checks_the_faces_of_an_exported_binning():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    exported = env.binning
    assert isinstance(exported.faces, NativeFaces)
    rebuilt = Environment.from_bins(exported, arena=env.arena)
    assert isinstance(rebuilt._binning.faces, KnownFaces)
    np.testing.assert_array_equal(rebuilt.bin_centers, env.bin_centers)
    # The recipe describes its own bins; faces built for other measures fail.
    resized = replace(
        exported, bins=replace(exported.bins, sizes=exported.bins.sizes * 2)
    )
    with pytest.raises(ValueError, match=r"faces\.volumes"):
        Environment.from_bins(resized, arena=env.arena)
    np.testing.assert_array_equal(
        rebuilt.compute_kernel(bandwidth=0.5, mode="density"),
        env.compute_kernel(bandwidth=0.5, mode="density"),
    )


@pytest.mark.parametrize("restore", ["direct", "copy", "pickle"])
def test_public_binning_detaches_nested_track_metadata(restore: str) -> None:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(4.0, 0.0))
    graph.add_edge(0, 1, distance=4.0)
    graph.graph["recording"] = {"labels": ["original"], "samples": np.array([1.0, 2.0])}
    env = Environment.from_graph(graph, [(0, 1)], edge_spacing=0.0, bin_size=1.0)
    exported = env.binning
    if restore == "copy":
        exported = copy.deepcopy(exported)
    elif restore == "pickle":
        exported = pickle.loads(pickle.dumps(exported))
    details = exported.linearization.graph.graph["recording"]
    details["labels"].append("caller edit")
    details["samples"].shape = (1, 2)
    fresh = env.binning.linearization.graph.graph["recording"]
    assert fresh["labels"] == ["original"]
    np.testing.assert_array_equal(fresh["samples"], [1.0, 2.0])
