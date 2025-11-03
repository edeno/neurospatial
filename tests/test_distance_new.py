"""Tests for new distance field functions."""

import networkx as nx
import numpy as np
import pytest

from neurospatial.distance import distance_field, pairwise_distances


class TestDistanceField:
    """Test distance_field function."""

    @pytest.fixture
    def linear_graph(self):
        """Create a simple linear graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_single_source(self, linear_graph):
        """Test distance field from single source."""
        dists = distance_field(linear_graph, sources=[2])

        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        assert np.allclose(dists, expected)

    def test_multiple_sources(self, linear_graph):
        """Test distance field from multiple sources."""
        dists = distance_field(linear_graph, sources=[0, 4])

        # Each node should be closest to either source 0 or 4
        expected = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        assert np.allclose(dists, expected)

    def test_empty_sources_raises_error(self, linear_graph):
        """Test that empty sources list raises error."""
        with pytest.raises(ValueError, match="at least one node"):
            distance_field(linear_graph, sources=[])

    def test_invalid_source_warns(self, linear_graph):
        """Test that invalid source node generates warning."""
        with pytest.warns(UserWarning, match="not in graph"):
            dists = distance_field(linear_graph, sources=[999, 2])

        # Should still compute for valid source
        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        assert np.allclose(dists, expected)

    def test_all_invalid_sources_raises_error(self, linear_graph):
        """Test that all invalid sources raise error."""
        with (
            pytest.raises(ValueError, match="No valid source nodes"),
            pytest.warns(UserWarning),
        ):
            distance_field(linear_graph, sources=[999, 1000])

    def test_empty_graph(self):
        """Test distance field on empty graph."""
        G = nx.Graph()
        dists = distance_field(G, sources=[])

        assert dists.shape == (0,)

    def test_disconnected_graph(self):
        """Test distance field on disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        dists = distance_field(G, sources=[0])

        # Nodes 2 and 3 should be unreachable
        assert dists[0] == 0.0
        assert dists[1] == 1.0
        assert np.isinf(dists[2])
        assert np.isinf(dists[3])


class TestPairwiseDistances:
    """Test pairwise_distances function."""

    @pytest.fixture
    def cycle_graph(self):
        """Create a cycle graph for testing."""
        G = nx.cycle_graph(10)
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_pairwise_basic(self, cycle_graph):
        """Test basic pairwise distance computation."""
        nodes = [0, 5]
        dists = pairwise_distances(cycle_graph, nodes)

        assert dists.shape == (2, 2)
        assert dists[0, 0] == 0.0  # Self-distance
        assert dists[1, 1] == 0.0
        assert dists[0, 1] == 5.0  # Shortest path on cycle
        assert dists[1, 0] == 5.0  # Symmetric

    def test_pairwise_three_nodes(self, cycle_graph):
        """Test pairwise distances with three nodes."""
        nodes = [0, 3, 7]
        dists = pairwise_distances(cycle_graph, nodes)

        assert dists.shape == (3, 3)
        np.fill_diagonal(dists, np.nan)  # Ignore diagonal
        assert np.all(dists[~np.isnan(dists)] > 0)  # All off-diagonal > 0

    def test_pairwise_empty_nodes(self, cycle_graph):
        """Test pairwise with empty node list."""
        dists = pairwise_distances(cycle_graph, [])

        assert dists.shape == (0, 0)

    def test_pairwise_single_node(self, cycle_graph):
        """Test pairwise with single node."""
        dists = pairwise_distances(cycle_graph, [0])

        assert dists.shape == (1, 1)
        assert dists[0, 0] == 0.0

    def test_pairwise_invalid_node(self, cycle_graph):
        """Test that invalid nodes result in inf distances."""
        nodes = [0, 999]  # 999 doesn't exist
        dists = pairwise_distances(cycle_graph, nodes)

        assert dists[0, 0] == 0.0
        assert np.isinf(dists[0, 1])
        assert np.isinf(dists[1, 0])
        assert np.isinf(dists[1, 1])
