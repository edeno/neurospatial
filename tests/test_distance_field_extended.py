"""Tests for extended distance_field() function with Euclidean metric and cutoff.

This module tests the extended functionality of distance_field() including:
- Euclidean metric computation using bin_centers
- Cutoff parameter for both geodesic and Euclidean modes
- Backward compatibility with existing geodesic-only behavior

Tests follow TDD - written BEFORE implementation.
"""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.distance import distance_field


class TestDistanceFieldEuclideanMetric:
    """Test distance_field() with metric='euclidean'."""

    @pytest.fixture
    def grid_graph_with_coords(self):
        """Create a 3x3 grid graph with coordinates."""
        G = nx.Graph()
        # Create 3x3 grid at positions (0,0), (1,0), (2,0), (0,1), ...
        positions = {}
        node_id = 0
        for i in range(3):
            for j in range(3):
                positions[node_id] = (float(i), float(j))
                G.add_node(node_id)
                node_id += 1

        # Add edges (4-connected grid)
        for i in range(3):
            for j in range(3):
                node = i * 3 + j
                if i < 2:  # Right neighbor
                    neighbor = (i + 1) * 3 + j
                    G.add_edge(node, neighbor, distance=1.0, vector=(1.0, 0.0))
                if j < 2:  # Up neighbor
                    neighbor = i * 3 + (j + 1)
                    G.add_edge(node, neighbor, distance=1.0, vector=(0.0, 1.0))

        # Store positions as node attributes
        for node, pos in positions.items():
            G.nodes[node]["pos"] = pos

        # Extract bin_centers array
        n_nodes = G.number_of_nodes()
        bin_centers = np.zeros((n_nodes, 2))
        for node in G.nodes():
            bin_centers[node] = G.nodes[node]["pos"]

        return G, bin_centers

    def test_euclidean_single_source(self, grid_graph_with_coords):
        """Test Euclidean distance from single source."""
        G, bin_centers = grid_graph_with_coords

        # Source at center (node 4 at position (1, 1))
        dists = distance_field(
            G, sources=[4], metric="euclidean", bin_centers=bin_centers
        )

        assert dists.shape == (9,)
        assert dists[4] == 0.0  # Source has zero distance

        # Check some known distances
        # Node 0 at (0, 0), distance to (1, 1) is sqrt(2)
        assert_allclose(dists[0], np.sqrt(2.0))
        # Node 1 at (1, 0), distance to (1, 1) is 1.0
        assert_allclose(dists[1], 1.0)
        # Node 2 at (2, 0), distance to (1, 1) is sqrt(2)
        assert_allclose(dists[2], np.sqrt(2.0))

    def test_euclidean_multiple_sources(self, grid_graph_with_coords):
        """Test Euclidean distance to nearest of multiple sources."""
        G, bin_centers = grid_graph_with_coords

        # Sources at corners: node 0 (0,0) and node 8 (2,2)
        dists = distance_field(
            G, sources=[0, 8], metric="euclidean", bin_centers=bin_centers
        )

        # All bins should have distance to nearest corner
        assert dists[0] == 0.0  # Source
        assert dists[8] == 0.0  # Source

        # Node 4 at (1, 1) equidistant from both corners
        # Distance to (0,0) is sqrt(2), distance to (2,2) is sqrt(2)
        assert_allclose(dists[4], np.sqrt(2.0))

    def test_euclidean_vs_geodesic(self, grid_graph_with_coords):
        """Test that Euclidean <= Geodesic (triangle inequality)."""
        G, bin_centers = grid_graph_with_coords

        sources = [0]

        dist_euc = distance_field(
            G, sources=sources, metric="euclidean", bin_centers=bin_centers
        )
        dist_geo = distance_field(G, sources=sources, metric="geodesic")

        # Euclidean should be <= geodesic for all bins
        assert np.all(dist_euc <= dist_geo + 1e-10)

    def test_euclidean_3d(self):
        """Test Euclidean distance in 3D coordinates."""
        # Create a simple graph with 4 nodes in 3D
        G = nx.Graph()
        bin_centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # Add nodes
        for i in range(4):
            G.add_node(i)
            G.nodes[i]["pos"] = tuple(bin_centers[i])

        # Add edges (all-to-all for simplicity, distances don't matter for Euclidean)
        for i in range(4):
            for j in range(i + 1, 4):
                G.add_edge(i, j, distance=1.0)

        dists = distance_field(
            G, sources=[0], metric="euclidean", bin_centers=bin_centers
        )

        assert dists[0] == 0.0
        assert_allclose(dists[1], 1.0)  # Unit distance along x
        assert_allclose(dists[2], 1.0)  # Unit distance along y
        assert_allclose(dists[3], 1.0)  # Unit distance along z

    def test_euclidean_requires_bin_centers(self, grid_graph_with_coords):
        """Test that Euclidean metric requires bin_centers parameter."""
        G, _ = grid_graph_with_coords

        with pytest.raises(ValueError, match=r"bin_centers.*required.*euclidean"):
            distance_field(G, sources=[0], metric="euclidean")


class TestDistanceFieldCutoff:
    """Test distance_field() with cutoff parameter."""

    @pytest.fixture
    def linear_graph(self):
        """Create a simple linear graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_geodesic_cutoff(self, linear_graph):
        """Test geodesic distance with cutoff."""
        # Source at node 2, cutoff at distance 1.5
        dists = distance_field(linear_graph, sources=[2], cutoff=1.5, metric="geodesic")

        # Within cutoff: nodes 1, 2, 3
        assert dists[1] == 1.0
        assert dists[2] == 0.0
        assert dists[3] == 1.0

        # Beyond cutoff: nodes 0, 4
        assert np.isinf(dists[0])
        assert np.isinf(dists[4])

    def test_geodesic_cutoff_multi_source(self, linear_graph):
        """Test geodesic cutoff with multiple sources."""
        # Sources at ends, cutoff at 1.5
        dists = distance_field(
            linear_graph, sources=[0, 4], cutoff=1.5, metric="geodesic"
        )

        # Node 0: distance 0 (source)
        assert dists[0] == 0.0
        # Node 1: distance 1 from source 0
        assert dists[1] == 1.0
        # Node 2: distance 2 from both sources (beyond cutoff)
        assert np.isinf(dists[2])
        # Node 3: distance 1 from source 4
        assert dists[3] == 1.0
        # Node 4: distance 0 (source)
        assert dists[4] == 0.0

    def test_euclidean_cutoff(self):
        """Test Euclidean distance with cutoff."""
        # Create 2D grid
        G = nx.Graph()
        bin_centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

        for i in range(3):
            G.add_node(i)
            G.nodes[i]["pos"] = tuple(bin_centers[i])

        # Source at (0, 0), cutoff at 3.5
        dists = distance_field(
            G, sources=[0], metric="euclidean", bin_centers=bin_centers, cutoff=3.5
        )

        assert dists[0] == 0.0  # Source
        assert dists[1] == 3.0  # Within cutoff
        assert np.isinf(dists[2])  # Distance 4.0, beyond cutoff

    def test_cutoff_zero(self, linear_graph):
        """Test cutoff=0 returns only sources."""
        dists = distance_field(linear_graph, sources=[2], cutoff=0.0, metric="geodesic")

        assert dists[2] == 0.0  # Source
        assert np.all(np.isinf(dists[np.arange(5) != 2]))  # All others inf

    def test_negative_cutoff_raises_error(self, linear_graph):
        """Test that negative cutoff raises error."""
        with pytest.raises(ValueError, match="cutoff must be non-negative"):
            distance_field(linear_graph, sources=[0], cutoff=-1.0, metric="geodesic")


class TestDistanceFieldBackwardCompatibility:
    """Test backward compatibility with existing behavior."""

    @pytest.fixture
    def linear_graph(self):
        """Create a simple linear graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_default_metric_is_geodesic(self, linear_graph):
        """Test that default metric is 'geodesic'."""
        dists_explicit = distance_field(linear_graph, sources=[2], metric="geodesic")
        dists_default = distance_field(linear_graph, sources=[2])

        assert_allclose(dists_explicit, dists_default)

    def test_default_no_cutoff(self, linear_graph):
        """Test that default behavior has no cutoff."""
        dists = distance_field(linear_graph, sources=[2])

        # All nodes should be reachable (no inf except disconnected)
        assert np.all(np.isfinite(dists))

    def test_old_signature_still_works(self, linear_graph):
        """Test that old function signature still works."""
        # Old signature: distance_field(G, sources, weight="distance")
        dists = distance_field(linear_graph, sources=[2], weight="distance")

        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        assert_allclose(dists, expected)

    def test_custom_weight_attribute(self, linear_graph):
        """Test using custom weight attribute."""
        # Add custom weight
        for u, v in linear_graph.edges:
            linear_graph.edges[u, v]["custom_weight"] = 2.0

        dists = distance_field(linear_graph, sources=[2], weight="custom_weight")

        expected = np.array([4.0, 2.0, 0.0, 2.0, 4.0])
        assert_allclose(dists, expected)


class TestDistanceFieldInputValidation:
    """Test input validation for distance_field()."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)
        return G

    def test_invalid_metric(self, simple_graph):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match=r"metric must be.*geodesic.*euclidean"):
            distance_field(simple_graph, sources=[0], metric="invalid")

    def test_empty_sources(self, simple_graph):
        """Test that empty sources raises ValueError."""
        with pytest.raises(ValueError, match="sources must contain at least one node"):
            distance_field(simple_graph, sources=[])

    def test_euclidean_without_bin_centers(self, simple_graph):
        """Test that Euclidean without bin_centers raises error."""
        with pytest.raises(ValueError, match=r"bin_centers.*required.*euclidean"):
            distance_field(simple_graph, sources=[0], metric="euclidean")

    def test_bin_centers_shape_mismatch(self, simple_graph):
        """Test that bin_centers size must match graph nodes."""
        bin_centers = np.array([[0.0, 0.0]])  # Only 1 row, graph has 2 nodes

        with pytest.raises(
            ValueError, match=r"bin_centers.*must match.*number of nodes"
        ):
            distance_field(
                simple_graph, sources=[0], metric="euclidean", bin_centers=bin_centers
            )

    def test_geodesic_with_bin_centers_ignored(self, simple_graph):
        """Test that bin_centers is ignored for geodesic metric."""
        bin_centers = np.array([[0.0, 0.0], [1.0, 1.0]])

        # Should not raise error, bin_centers just ignored
        dists = distance_field(
            simple_graph, sources=[0], metric="geodesic", bin_centers=bin_centers
        )

        assert dists.shape == (2,)


class TestDistanceFieldEdgeCases:
    """Test edge cases for distance_field()."""

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()

        # Empty graph returns empty array (no nodes to query)
        dists = distance_field(G, sources=[], metric="geodesic")
        assert dists.shape == (0,)

    def test_single_node_graph(self):
        """Test with single node graph."""
        G = nx.Graph()
        G.add_node(0)

        dists = distance_field(G, sources=[0], metric="geodesic")

        assert dists.shape == (1,)
        assert dists[0] == 0.0

    def test_disconnected_graph(self):
        """Test with disconnected graph."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)
        G.add_edge(2, 3, distance=1.0)

        dists = distance_field(G, sources=[0], metric="geodesic")

        assert dists[0] == 0.0
        assert dists[1] == 1.0
        assert np.isinf(dists[2])
        assert np.isinf(dists[3])

    def test_source_not_in_graph_warning(self):
        """Test that invalid source generates warning but continues."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)

        with pytest.warns(UserWarning, match="Source node 999 not in graph"):
            dists = distance_field(G, sources=[999, 0], metric="geodesic")

        # Should still compute for valid source
        assert dists[0] == 0.0
        assert dists[1] == 1.0

    def test_all_sources_invalid(self):
        """Test that all invalid sources raises error."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)

        with (
            pytest.raises(ValueError, match="No valid source nodes found"),
            pytest.warns(UserWarning),
        ):
            distance_field(G, sources=[999, 1000], metric="geodesic")
