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
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )


@pytest.mark.parametrize("restore", ["fresh", "deepcopy", "pickle"])
def test_standard_node_link_export_is_json_serializable(restore):
    env = _environment()
    if restore == "deepcopy":
        env = copy.deepcopy(env)
    elif restore == "pickle":
        env = pickle.loads(pickle.dumps(env))

    payload = json.loads(json.dumps(nx.node_link_data(env.connectivity, edges="links")))
    exported = nx.node_link_graph(payload, edges="links")
    assert list(exported.edges) == list(env.connectivity.edges)
    assert list(exported.nodes) == list(env.connectivity.nodes)
    assert exported.nodes[0]["pos"] == list(env.connectivity.nodes[0]["pos"])


def test_published_graph_metadata_is_read_only():
    env = _environment()
    metadata = env.connectivity.graph
    assert metadata is env.connectivity.graph
    assert metadata == {}
    with pytest.raises(TypeError, match="read-only"):
        metadata["caller"] = 1
    assert "caller" not in env.connectivity.graph


def test_exported_graph_metadata_is_editable_and_independent():
    env = _environment()
    for metadata in (env.to_networkx().graph, env.connectivity.copy().graph):
        assert type(metadata) is dict
        metadata["caller"] = 1
    assert "caller" not in env.connectivity.graph


@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_graph_metadata_survives_copy_and_pickle_read_only(method):
    original = _environment().connectivity.graph
    restored = {
        "copy": copy.copy,
        "deepcopy": copy.deepcopy,
        "pickle": lambda value: pickle.loads(pickle.dumps(value)),
    }[method](original)
    assert restored == original
    assert json.loads(json.dumps(restored)) == {}
    with pytest.raises(TypeError, match="read-only"):
        restored["caller"] = 1
