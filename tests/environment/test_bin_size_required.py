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
    with pytest.raises(TypeError) as caught:
        Environment.from_samples(data)
    message = str(caught.value)
    assert message.startswith("Environment.from_samples() needs bin_size=.")
    assert "in the units of positions" in message
    assert message.splitlines()[-1] == (
        "Fix: pass bin_size= in the units of positions, as in "
        'Environment.from_samples(positions, bin_size=2.0, units="cm")'
    )


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


def test_bare_environment_raises_beginner_grade_error():
    """Bare ``Environment()`` must raise ValueError with factory guidance.

    The message must mention ``from_samples`` so beginners know the correct
    approach immediately from the error text.
    """
    with pytest.raises(
        ValueError, match=r"Environment cannot be constructed directly"
    ) as exc_info:
        Environment()
    assert "from_samples" in str(exc_info.value)


def test_oversized_bin_size_warning_rounds_and_names_the_units_mistake():
    rng = np.random.default_rng(0)
    positions = rng.uniform(1.0, 98.0, size=(200, 2))
    positions[:2] = [[0.0, 0.0], [99.9987654, 99.9976543]]
    extent = np.ptp(positions, axis=0)
    assert extent.min() > 99.99  # the raw extent would print many digits

    with pytest.warns(UserWarning, match="is at least the per-axis") as caught:
        Environment.from_samples(positions, bin_size=1000)
    message = str(caught[0].message)

    # Extent is shown to 3 significant figures, not as raw floats.
    assert "bin_size=1000 is at least the per-axis data extent [100, 100]" in message
    assert str(extent[0]) not in message
    # The units mistake is named, and the suggestion is a rounded size.
    assert "often a units mistake: positions in cm but bin_size meant in m" in message
    assert message.splitlines()[-1].endswith("such as bin_size=2.0")
    assert "1.999" not in message
