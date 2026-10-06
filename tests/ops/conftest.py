"""Geometry fixtures shared by finite-volume operator and basis tests."""

from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def uniform_grid():
    """Build a square grid with exactly representable bin edges."""

    def build(spacing=1.0, n=20):
        edges = np.arange(n + 1, dtype=float) * spacing
        return Environment.from_grid_mask(np.ones((n, n), bool), (edges, edges))

    return build


@pytest.fixture(params=["grid", "hex", "w-maze"])
def fv_env(request, uniform_grid):
    """Supported geometries, including the track-junction distance correction."""
    if request.param == "grid":
        return uniform_grid(2.0, n=10)
    if request.param == "hex":
        positions = np.random.default_rng(0).uniform(0, 100, (2000, 2))
        return Environment.from_samples(positions, layout="Hexagonal", bin_size=5.0)
    nodes = dict(
        zip(
            ["bl", "bm", "br", "al", "am", "ar"],
            [(0, 0), (50, 0), (100, 0), (0, 50), (50, 50), (100, 50)],
            strict=True,
        )
    )
    return Environment.maze("w", node_positions=nodes, bin_size=5.0)


@pytest.fixture
def reloaded_graph_env(tmp_path):
    """An NWB-reconstructed layout without a finite-volume geometry builder."""
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    graph = nx.Graph()
    for i, pos in enumerate([(0, 0), (0, 100), (-50, 150), (50, 150)]):
        graph.add_node(i, pos=pos)
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        graph.add_edge(
            u,
            v,
            distance=float(
                np.linalg.norm(
                    np.asarray(graph.nodes[v]["pos"]) - graph.nodes[u]["pos"]
                )
            ),
        )
    env = Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2), (1, 3)], edge_spacing=10.0, bin_size=3.0
    )
    nwbfile = pynwb.NWBFile("geometry test", "geometry", datetime.now(timezone.utc))
    write_environment(nwbfile, env)
    path = tmp_path / "environment.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(path, "r") as io:
        return read_environment(io.read())
