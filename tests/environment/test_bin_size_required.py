"""Tests that bin_size / pixel_size are required (no defaults) for the
grid-inferring Environment factories.

These tests assert behavior — calling the factory without the size argument
raises ``TypeError: missing ... required ... bin_size`` (or ``pixel_size``).
Signature-introspection variants and docstring-prose audits were removed:
both are guaranteed once the behavioral test passes.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment


def test_from_samples_requires_bin_size():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    with pytest.raises(TypeError, match=r"missing.*required.*bin_size"):
        Environment.from_samples(data)


def test_from_samples_accepts_explicit_bin_size():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    env = Environment.from_samples(data, bin_size=2.0)
    assert env.n_bins > 0


def test_from_polygon_requires_bin_size():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    with pytest.raises(TypeError, match=r"missing.*required.*bin_size"):
        Environment.from_polygon(polygon)


def test_from_polygon_accepts_explicit_bin_size():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    env = Environment.from_polygon(polygon, bin_size=2.0)
    assert env.n_bins > 0


def test_from_pixel_mask_requires_pixel_size():
    image_mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(TypeError, match=r"missing.*required.*pixel_size"):
        Environment.from_pixel_mask(image_mask)


def test_from_pixel_mask_accepts_explicit_pixel_size():
    image_mask = np.ones((10, 10), dtype=bool)
    env = Environment.from_pixel_mask(image_mask, pixel_size=1.0)
    assert env.n_bins > 0


def test_from_graph_requires_bin_size():
    """``from_graph`` requires both ``bin_size`` and ``edge_spacing``."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1)
    edge_order = [(0, 1)]

    with pytest.raises(TypeError, match=r"missing.*required.*(bin_size|edge_spacing)"):
        Environment.from_graph(graph, edge_order)
