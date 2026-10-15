"""Reserved graph attributes have validated, optional node-link sidecars."""

import copy
import json

import networkx as nx
import pytest

from neurospatial.layout._codecs import graph_from_node_link, graph_to_node_link

_SIDECAR = "_neurospatial_reserved_attributes"


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    first, second = ("track", (0, 1)), ("track", (2, 3))
    graph.add_node(first, id="user-node", label="start")
    graph.add_node(second, label="stop")
    if graph.is_multigraph():
        key = ("trial", 7)
        graph.add_edge(first, second, key=key)
        attributes = graph.edges[first, second, key]
        attributes["key"] = "user-key"
    else:
        graph.add_edge(first, second)
        attributes = graph.edges[first, second]
    attributes.update(source="user-source", target="user-target", distance=3.0)
    return graph


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_reserved_graph_attributes_round_trip_with_tuple_identity(graph_type):
    graph = _graph(graph_type)
    metadata = json.loads(json.dumps(graph_to_node_link(graph)))
    original = copy.deepcopy(metadata)
    restored = graph_from_node_link(metadata)
    assert metadata == original
    assert list(restored.nodes(data=True)) == list(graph.nodes(data=True))
    assert list(restored.edges(data=True)) == list(graph.edges(data=True))
    if graph.is_multigraph():
        assert list(restored.edges(keys=True)) == list(graph.edges(keys=True))


def test_graphs_without_reserved_attributes_retain_the_existing_wire_format():
    graph = nx.path_graph(3)
    graph.nodes[0]["label"] = "start"
    metadata = graph_to_node_link(graph)
    assert _SIDECAR not in metadata
    assert metadata == nx.node_link_data(graph, edges="links")
    restored = graph_from_node_link(metadata)
    assert list(restored.nodes(data=True)) == list(graph.nodes(data=True))


def test_legacy_multigraph_records_can_still_omit_automatic_edge_keys():
    metadata = {
        "directed": False,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": 0}, {"id": 1}],
        "links": [{"source": 0, "target": 1}],
    }
    restored = graph_from_node_link(metadata)
    assert list(restored.edges(keys=True)) == [(0, 1, 0)]


def test_legacy_node_link_readers_still_restore_topology_with_the_optional_sidecar():
    graph = nx.path_graph(2)
    graph.nodes[0]["id"] = "user-node"
    graph.edges[0, 1]["source"] = "user-source"
    metadata = json.loads(json.dumps(graph_to_node_link(graph)))
    restored = nx.node_link_graph(metadata, edges="links")
    assert list(restored) == list(graph)
    assert list(restored.edges) == list(graph.edges)


@pytest.mark.parametrize(
    "corruption", ["length", "node_id", "endpoint", "attribute", "group", "row"]
)
def test_malformed_reserved_attribute_sidecars_fail_clearly(corruption):
    metadata = graph_to_node_link(_graph())
    sidecar = metadata[_SIDECAR]
    if corruption == "length":
        sidecar["nodes"].pop()
    elif corruption == "node_id":
        sidecar["nodes"][0]["id"] = "another-node"
    elif corruption == "endpoint":
        sidecar["links"][0]["source"] = "another-node"
    elif corruption == "attribute":
        sidecar["nodes"][0]["attributes"]["pos"] = [0.0, 0.0]
    elif corruption == "group":
        sidecar["unexpected"] = []
    else:
        sidecar["nodes"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="Reserved graph attribute"):
        graph_from_node_link(metadata)
