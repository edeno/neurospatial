"""Shared fixtures for simulation tests."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def simple_2d_env():
    """Create a simple 2D square environment for testing.

    Returns
    -------
    Environment
        100x100 cm square arena with 2 cm bins.
    """
    # Create a grid of sample points
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    samples = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(samples, bin_size=2.0)
    env.units = "cm"
    env.frame = "test_arena"
    return env


@pytest.fixture
def simple_1d_env():
    """Create a simple 1D linear track for testing.

    Returns
    -------
    Environment
        200 cm linear track with 2 cm bins.
    """
    # Create sample points along a line
    samples = np.linspace(0, 200, 100).reshape(-1, 1)

    # Note: 1D environments require GraphLayout, which we'll implement later
    # For now, this will create a regular 1D grid
    env = Environment.from_samples(samples, bin_size=2.0)
    env.units = "cm"
    env.frame = "linear_track"
    return env


@pytest.fixture
def rng():
    """Create a deterministic random number generator for reproducible tests.

    Returns
    -------
    np.random.Generator
        Seeded random number generator.
    """
    return np.random.default_rng(42)


@pytest.fixture
def sample_positions():
    """Create sample trajectory positions for testing.

    Returns
    -------
    ndarray, shape (1000, 2)
        Random positions in a 100x100 arena.
    """
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, size=(1000, 2))


@pytest.fixture
def sample_times():
    """Create sample time points for testing.

    Returns
    -------
    ndarray, shape (1000,)
        Time points at 100 Hz sampling rate (10 seconds total).
    """
    return np.linspace(0, 10, 1000)


@pytest.fixture(scope="module")
def hairpin_track_env():
    """A hairpin track: two 100 cm arms 1 cm apart, joined at x = 100, 5 cm bins.

    Bin centres on opposite arms are 1 cm apart in space but up to 200 cm
    apart along the track.
    """
    import networkx as nx

    graph = nx.Graph()
    nodes = {"a": (0.0, 0.0), "b": (100.0, 0.0), "c": (100.0, 1.0), "d": (0.0, 1.0)}
    for name, pos in nodes.items():
        graph.add_node(name, pos=pos)
    edge_order = [("a", "b"), ("b", "c"), ("c", "d")]
    for u, v in edge_order:
        graph.add_edge(
            u, v, distance=float(np.linalg.norm(np.subtract(nodes[u], nodes[v])))
        )
    return Environment.from_graph(graph, edge_order, edge_spacing=0.0, bin_size=5.0)
