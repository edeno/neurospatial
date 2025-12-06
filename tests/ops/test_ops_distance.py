"""Tests for ops.distance module - verifies new import paths work."""

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from neurospatial.ops.distance import (
    distance_field,
    euclidean_distance_matrix,
    geodesic_distance_between_points,
    geodesic_distance_matrix,
    neighbors_within,
    pairwise_distances,
)


class TestOpsDistanceImports:
    """Test that all functions are importable from neurospatial.ops.distance."""

    def test_distance_field_importable(self):
        """Test distance_field is importable from ops.distance."""
        assert callable(distance_field)

    def test_euclidean_distance_matrix_importable(self):
        """Test euclidean_distance_matrix is importable from ops.distance."""
        assert callable(euclidean_distance_matrix)

    def test_geodesic_distance_matrix_importable(self):
        """Test geodesic_distance_matrix is importable from ops.distance."""
        assert callable(geodesic_distance_matrix)

    def test_geodesic_distance_between_points_importable(self):
        """Test geodesic_distance_between_points is importable from ops.distance."""
        assert callable(geodesic_distance_between_points)

    def test_neighbors_within_importable(self):
        """Test neighbors_within is importable from ops.distance."""
        assert callable(neighbors_within)

    def test_pairwise_distances_importable(self):
        """Test pairwise_distances is importable from ops.distance."""
        assert callable(pairwise_distances)


class TestOpsDistanceBasicFunctionality:
    """Basic functionality tests for ops.distance module."""

    def test_euclidean_distance_matrix_basic(self):
        """Test basic euclidean distance matrix computation."""
        centers = np.array([[0, 0], [3, 4], [6, 8]])
        result = euclidean_distance_matrix(centers)

        # Distance from [0,0] to [3,4] is 5.0
        expected = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]])
        assert_allclose(result, expected)

    def test_geodesic_distance_matrix_basic(self):
        """Test basic geodesic distance matrix computation."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        result = geodesic_distance_matrix(graph, n_states=4)

        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )
        assert_allclose(result, expected)

    def test_geodesic_distance_between_points_basic(self):
        """Test basic geodesic distance between two points."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=2.5)
        graph.add_edge(1, 2, distance=3.0)

        result = geodesic_distance_between_points(graph, 0, 2)
        assert_allclose(result, 5.5)

    def test_distance_field_basic(self):
        """Test basic distance field computation."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        dists = distance_field(G, sources=[2])
        expected = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        assert_allclose(dists, expected)

    def test_pairwise_distances_basic(self):
        """Test basic pairwise distances computation."""
        G = nx.cycle_graph(10)
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        dists = pairwise_distances(G, [0, 3, 7])
        assert dists.shape == (3, 3)
        assert_allclose(dists[0, 1], 3.0)  # Distance from node 0 to node 3

    def test_neighbors_within_basic(self):
        """Test basic neighbors_within computation."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        neighborhoods = neighbors_within(G, centers=[2], radius=1.5, metric="geodesic")
        assert sorted(neighborhoods[0]) == [1, 2, 3]


class TestOpsPackageExports:
    """Test that functions are also accessible via ops package."""

    def test_import_from_ops_package(self):
        """Test that distance functions can be imported from neurospatial.ops."""
        from neurospatial.ops import (
            distance_field,
            euclidean_distance_matrix,
            geodesic_distance_between_points,
            geodesic_distance_matrix,
            neighbors_within,
            pairwise_distances,
        )

        assert callable(distance_field)
        assert callable(euclidean_distance_matrix)
        assert callable(geodesic_distance_between_points)
        assert callable(geodesic_distance_matrix)
        assert callable(neighbors_within)
        assert callable(pairwise_distances)
