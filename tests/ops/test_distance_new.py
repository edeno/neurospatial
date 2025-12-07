"""Tests for new distance field functions."""

import networkx as nx
import numpy as np
import pytest

from neurospatial.ops.distance import (
    distance_field,
    neighbors_within,
    pairwise_distances,
)


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


class TestNeighborsWithinGeodesic:
    """Test neighbors_within function with geodesic metric."""

    @pytest.fixture
    def linear_graph(self):
        """Create a simple linear graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_geodesic_basic(self, linear_graph):
        """Test basic geodesic neighborhood query."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[2], radius=1.5, metric="geodesic"
        )

        assert len(neighborhoods) == 1
        expected = {1, 2, 3}  # Nodes within distance 1.5 from node 2
        assert set(neighborhoods[0]) == expected

    def test_geodesic_multiple_centers(self, linear_graph):
        """Test neighborhoods for multiple centers."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[0, 4], radius=1.0, metric="geodesic"
        )

        assert len(neighborhoods) == 2
        assert set(neighborhoods[0]) == {0, 1}  # Neighbors of 0
        assert set(neighborhoods[1]) == {3, 4}  # Neighbors of 4

    def test_geodesic_exclude_center(self, linear_graph):
        """Test neighborhoods excluding center bin."""
        neighborhoods = neighbors_within(
            linear_graph,
            centers=[2],
            radius=1.0,
            metric="geodesic",
            include_center=False,
        )

        assert len(neighborhoods) == 1
        expected = {1, 3}  # Should exclude 2 itself
        assert set(neighborhoods[0]) == expected

    def test_geodesic_zero_radius(self, linear_graph):
        """Test with zero radius (only center)."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[2], radius=0.0, metric="geodesic"
        )

        assert len(neighborhoods) == 1
        assert set(neighborhoods[0]) == {2}  # Only the center

    def test_geodesic_large_radius(self, linear_graph):
        """Test with radius covering whole graph."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[2], radius=10.0, metric="geodesic"
        )

        assert len(neighborhoods) == 1
        assert set(neighborhoods[0]) == {0, 1, 2, 3, 4}  # All nodes

    def test_geodesic_empty_centers(self, linear_graph):
        """Test with empty centers list."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[], radius=1.0, metric="geodesic"
        )

        assert neighborhoods == []

    def test_geodesic_disconnected_graph(self):
        """Test on disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        neighborhoods = neighbors_within(G, centers=[0], radius=2.0, metric="geodesic")

        assert len(neighborhoods) == 1
        # Should only find nodes connected to 0
        assert set(neighborhoods[0]) == {0, 1}

    def test_geodesic_invalid_center_raises(self, linear_graph):
        """Test that invalid center node raises error."""
        with pytest.raises(nx.NodeNotFound, match="Center node 999 not in graph"):
            neighbors_within(linear_graph, centers=[999], radius=1.0, metric="geodesic")


class TestNeighborsWithinEuclidean:
    """Test neighbors_within function with euclidean metric."""

    @pytest.fixture
    def grid_graph_2d(self):
        """Create a 3x3 grid graph with coordinates."""
        G = nx.grid_2d_graph(3, 3)
        # Relabel nodes to integers
        mapping = {(i, j): i * 3 + j for i, j in G.nodes}
        G = nx.relabel_nodes(G, mapping)

        # Add edge distances and bin centers
        bin_centers = np.array(
            [[i, j] for i in range(3) for j in range(3)], dtype=np.float64
        )

        for u, v in G.edges:
            uc, vc = bin_centers[u], bin_centers[v]
            dist = np.linalg.norm(uc - vc)
            G.edges[u, v]["distance"] = dist

        return G, bin_centers

    def test_euclidean_basic(self, grid_graph_2d):
        """Test basic euclidean neighborhood query."""
        G, bin_centers = grid_graph_2d

        # Center at (1, 1) = node 4, radius 1.1 should get 4 orthogonal neighbors
        neighborhoods = neighbors_within(
            G, centers=[4], radius=1.1, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 1
        # Should include center and 4 orthogonal neighbors (distance 1.0)
        # Diagonal neighbors are at distance sqrt(2) ≈ 1.414 > 1.1
        expected = {1, 3, 4, 5, 7}
        assert set(neighborhoods[0]) == expected

    def test_euclidean_multiple_centers(self, grid_graph_2d):
        """Test neighborhoods for multiple centers."""
        G, bin_centers = grid_graph_2d

        neighborhoods = neighbors_within(
            G, centers=[0, 8], radius=1.1, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 2
        # Corner nodes with radius 1.1
        assert set(neighborhoods[0]) == {0, 1, 3}  # Top-left corner
        assert set(neighborhoods[1]) == {5, 7, 8}  # Bottom-right corner

    def test_euclidean_exclude_center(self, grid_graph_2d):
        """Test neighborhoods excluding center."""
        G, bin_centers = grid_graph_2d

        neighborhoods = neighbors_within(
            G,
            centers=[4],
            radius=1.1,
            metric="euclidean",
            bin_centers=bin_centers,
            include_center=False,
        )

        assert len(neighborhoods) == 1
        expected = {1, 3, 5, 7}  # Exclude 4 itself
        assert set(neighborhoods[0]) == expected

    def test_euclidean_zero_radius(self, grid_graph_2d):
        """Test with zero radius."""
        G, bin_centers = grid_graph_2d

        neighborhoods = neighbors_within(
            G, centers=[4], radius=0.0, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 1
        assert set(neighborhoods[0]) == {4}  # Only the center

    def test_euclidean_large_radius(self, grid_graph_2d):
        """Test with large radius covering all nodes."""
        G, bin_centers = grid_graph_2d

        neighborhoods = neighbors_within(
            G, centers=[4], radius=10.0, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 1
        assert len(neighborhoods[0]) == 9  # All nodes in 3x3 grid


class TestNeighborsWithinValidation:
    """Test parameter validation for neighbors_within."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for validation tests."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        bin_centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        return G, bin_centers

    def test_invalid_metric_raises(self, simple_graph):
        """Test that invalid metric raises error."""
        G, _ = simple_graph

        with pytest.raises(
            ValueError, match="metric must be 'geodesic' or 'euclidean'"
        ):
            neighbors_within(G, centers=[0], radius=1.0, metric="manhattan")

    def test_negative_radius_raises(self, simple_graph):
        """Test that negative radius raises error."""
        G, _ = simple_graph

        with pytest.raises(ValueError, match="radius must be non-negative"):
            neighbors_within(G, centers=[0], radius=-1.0, metric="geodesic")

    def test_euclidean_missing_bin_centers_raises(self, simple_graph):
        """Test that euclidean without bin_centers raises error."""
        G, _ = simple_graph

        with pytest.raises(
            ValueError,
            match="bin_centers parameter is required when metric='euclidean'",
        ):
            neighbors_within(G, centers=[0], radius=1.0, metric="euclidean")

    def test_euclidean_bin_centers_shape_mismatch_raises(self, simple_graph):
        """Test that bin_centers shape mismatch raises error."""
        G, _ = simple_graph
        wrong_centers = np.array([[0.0, 0.0]], dtype=np.float64)  # Only 1 row, need 3

        with pytest.raises(
            ValueError, match=r"bin_centers size .* must match number of nodes"
        ):
            neighbors_within(
                G,
                centers=[0],
                radius=1.0,
                metric="euclidean",
                bin_centers=wrong_centers,
            )


class TestNeighborsWithinEdgeCases:
    """Test edge cases for neighbors_within."""

    def test_empty_graph_geodesic(self):
        """Test geodesic on empty graph."""
        G = nx.Graph()

        neighborhoods = neighbors_within(G, centers=[], radius=1.0, metric="geodesic")

        assert neighborhoods == []

    def test_empty_graph_euclidean(self):
        """Test euclidean on empty graph."""
        G = nx.Graph()
        bin_centers = np.empty((0, 2), dtype=np.float64)

        neighborhoods = neighbors_within(
            G, centers=[], radius=1.0, metric="euclidean", bin_centers=bin_centers
        )

        assert neighborhoods == []

    def test_single_node_graph(self):
        """Test on single node graph."""
        G = nx.Graph()
        G.add_node(0)

        neighborhoods = neighbors_within(G, centers=[0], radius=1.0, metric="geodesic")

        assert len(neighborhoods) == 1
        assert set(neighborhoods[0]) == {0}
