"""Subset lookup reuses the parent's nearest-neighbor memo and bin ownership."""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from neurospatial import Environment
from neurospatial.layout._locators import SubsetLocator


@pytest.fixture
def linearized_parent_and_subset() -> tuple[Environment, Environment, np.ndarray]:
    """A track whose second half is dropped while its parent lookup is retained."""
    g = nx.Graph()
    for i in range(20):
        g.add_node(i, pos=(float(i),))
    edge_order = [(u, v) for u, v in zip(range(19), range(1, 20), strict=False)]
    for u, v in edge_order:
        g.add_edge(u, v, distance=1.0)
    env = Environment.from_graph(
        g, edge_order=edge_order, edge_spacing=0.0, bin_size=1.0
    )
    keep = np.zeros(env.n_bins, dtype=bool)
    keep[: env.n_bins // 2] = True
    return env, env.subset(bins=keep), keep


class TestSubsetKDTreeCaching:
    """Repeated subset queries keep the already-built memo of the parent locator."""

    def test_kdtree_built_at_most_once_over_many_queries(
        self, linearized_parent_and_subset
    ) -> None:
        parent, child, keep = linearized_parent_and_subset
        n_dims = parent.bin_centers.shape[1]
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 19, size=(1000, n_dims))
        locator = child._binning.locator
        assert isinstance(locator, SubsetLocator)
        assert locator.base is parent._binning.locator
        memo = locator.base._tree
        assert memo is not None
        child_of = np.full(parent.n_bins, -1, dtype=np.intp)
        child_of[keep] = np.arange(keep.sum())
        expected = child_of[parent.bin_at(pts)]
        assert np.any(expected == -1)

        with patch("neurospatial.layout._locators.cKDTree", wraps=cKDTree) as mock_ckdt:
            for _ in range(10):
                np.testing.assert_array_equal(child.bin_at(pts), expected)
        assert mock_ckdt.call_count == 0
        assert locator.base._tree is memo
