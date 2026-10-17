"""Node-link storage separates track-graph identity fields from user attributes."""

from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


def _annotate(graph):
    node = next(iter(graph))
    edge = next(iter(graph.edges))
    graph.nodes[node]["id"] = "measured-node-id"
    graph.nodes[node]["source"] = "node-source"
    graph.edges[edge]["source"] = "measured-source"
    graph.edges[edge]["target"] = "measured-target"
    graph.edges[edge]["id"] = "measured-edge-id"


def _environment():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    _annotate(graph)
    return Environment.from_graph(
        graph, list(graph.edges), edge_spacing=0.0, bin_size=2.0, track_width=2.0
    )


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_reserved_user_attributes_survive_environment_storage(tmp_path, storage):
    env = _environment()
    if storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "graph")
        restored = Environment.from_file(tmp_path / "graph")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            "graph", "graph", datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        write_environment(nwbfile, env)
        with pynwb.NWBHDF5IO(str(tmp_path / "graph.nwb"), "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(str(tmp_path / "graph.nwb"), "r") as io:
            restored = read_environment(io.read())

    pairs = [(env._binning.linearization.graph, restored._binning.linearization.graph)]
    for source, loaded in pairs:
        assert list(loaded) == list(source)
        assert list(loaded.edges) == list(source.edges)
        node, edge = next(iter(source)), next(iter(source.edges))
        for name in ("id", "source"):
            assert loaded.nodes[node][name] == source.nodes[node][name]
        for name in ("source", "target", "id"):
            assert loaded.edges[edge][name] == source.edges[edge][name]
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )
