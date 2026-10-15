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


@pytest.mark.parametrize("restore", ["fresh", "copy", "deepcopy", "pickle"])
def test_standard_node_link_export_is_json_serializable(restore):
    env = _environment()
    if restore == "copy":
        env = env.copy()
    elif restore == "deepcopy":
        env = copy.deepcopy(env)
    elif restore == "pickle":
        env = pickle.loads(pickle.dumps(env))

    payload = json.loads(json.dumps(nx.node_link_data(env.connectivity, edges="links")))
    exported = nx.node_link_graph(payload, edges="links")
    assert exported.graph["session"] == {"tags": ["first", "second"]}
    assert list(exported.edges) == list(env.connectivity.edges)
    assert list(exported.nodes) == list(env.connectivity.nodes)


def test_json_compatible_metadata_exports_cannot_change_owned_graph_state():
    env = _environment()
    with pytest.raises(TypeError, match="read-only"):
        env.connectivity.graph["caller"] = 1
    metadata = env.connectivity.graph
    # Even base-dict operations affect only the detached export header.
    dict.__setitem__(metadata, "caller", 1)
    metadata["session"]["tags"].append("caller")
    assert "caller" not in env.connectivity.graph
    assert env.connectivity.graph["session"]["tags"] == ["first", "second"]
