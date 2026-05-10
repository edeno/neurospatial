"""Tests for SubsetLayout KDTree caching (M5.2)."""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def linearized_subset_env() -> Environment:
    """Build a graph (linearized) env, then take a subset of it.

    Subsetting a graph env routes through ``SubsetLayout`` (the
    Cartesian fast path uses ``from_mask`` instead and never touches
    the layout we are testing).
    """
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
    return env.subset(bins=keep)


class TestSubsetKDTreeCaching:
    """SubsetLayout must build its KDTree at most once across many bin_at calls."""

    def test_kdtree_built_at_most_once_over_many_queries(
        self, linearized_subset_env: Environment
    ) -> None:
        # The graph env is 1D linearized, so bin_centers / query points
        # are shape (n, 1).
        n_dims = linearized_subset_env.bin_centers.shape[1]
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 9, size=(10_000, n_dims))

        layout = linearized_subset_env.layout
        # Sanity: caches start empty.
        assert getattr(layout, "_kdtree", "missing") is None
        assert getattr(layout, "_oof_threshold", "missing") is None

        with patch(
            "scipy.spatial.cKDTree", wraps=__import__("scipy.spatial").spatial.cKDTree
        ) as mock_ckdt:
            for _ in range(10):
                _ = linearized_subset_env.bin_at(pts[:1000])

        # Exactly one KDTree construction across 10 batches × 1000 points.
        assert mock_ckdt.call_count == 1, (
            f"Expected one KDTree build, got {mock_ckdt.call_count}"
        )
        # Cache state is now populated.
        assert layout._kdtree is not None
        assert layout._oof_threshold is None or layout._oof_threshold > 0

    def test_cache_persists_after_layout_attribute_access(
        self, linearized_subset_env: Environment
    ) -> None:
        n_dims = linearized_subset_env.bin_centers.shape[1]
        sample = np.full((1, n_dims), 5.0)

        # First call builds the tree.
        _ = linearized_subset_env.bin_at(sample)
        layout = linearized_subset_env.layout
        first_tree = layout._kdtree
        assert first_tree is not None

        # Subsequent unrelated layout reads must not re-build the tree.
        _ = linearized_subset_env.bin_centers
        _ = linearized_subset_env.dimension_ranges
        _ = linearized_subset_env.bin_at(sample)
        assert layout._kdtree is first_tree
