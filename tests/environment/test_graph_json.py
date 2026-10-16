"""Read-only connectivity supports ordinary NetworkX JSON interchange."""

from __future__ import annotations

import copy
import json
import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


def _environment():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    payload = env.to_dict()
    payload["graph"]["graph"]["session"] = {"tags": ["first", "second"]}
    return Environment.from_dict(payload)


@pytest.mark.parametrize("restore", ["fresh", "deepcopy", "pickle"])
def test_standard_node_link_export_is_json_serializable(restore):
    env = _environment()
    if restore == "deepcopy":
        env = copy.deepcopy(env)
    elif restore == "pickle":
        env = pickle.loads(pickle.dumps(env))

    payload = json.loads(json.dumps(nx.node_link_data(env.connectivity, edges="links")))
    exported = nx.node_link_graph(payload, edges="links")
    assert exported.graph["session"] == {"tags": ["first", "second"]}
    assert list(exported.edges) == list(env.connectivity.edges)
    assert list(exported.nodes) == list(env.connectivity.nodes)


def test_published_graph_metadata_is_frozen_once_and_read_without_copying():
    env = _environment()
    metadata = env.connectivity.graph
    assert metadata is env.connectivity.graph
    assert metadata["session"]["tags"] == ("first", "second")
    with pytest.raises(TypeError, match="read-only"):
        metadata["caller"] = 1
    with pytest.raises(TypeError, match="read-only"):
        metadata["session"]["tags"] = ["caller"]
    with pytest.raises(AttributeError):
        metadata["session"]["tags"].append("caller")
    assert "caller" not in env.connectivity.graph


def test_exported_graph_metadata_is_editable_and_independent():
    env = _environment()
    exported = env.to_networkx().graph
    copied = env.connectivity.copy().graph
    for metadata in (exported, copied):
        assert metadata == {"session": {"tags": ["first", "second"]}}
        metadata["session"]["tags"].append("caller")
        metadata["caller"] = 1
    assert env.connectivity.graph["session"]["tags"] == ("first", "second")
    assert "caller" not in env.connectivity.graph


@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_graph_metadata_survives_copy_and_pickle_read_only(method):
    env = _environment()
    original = env.connectivity.graph
    restored = {
        "copy": copy.copy,
        "deepcopy": copy.deepcopy,
        "pickle": lambda value: pickle.loads(pickle.dumps(value)),
    }[method](original)
    assert restored == original
    assert json.loads(json.dumps(restored)) == {
        "session": {"tags": ["first", "second"]}
    }
    with pytest.raises(TypeError, match="read-only"):
        restored["session"]["caller"] = 1
