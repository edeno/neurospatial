"""Tests for distance module."""

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from neurospatial.distance import (
    euclidean_distance_matrix,
    geodesic_distance_between_points,
    geodesic_distance_matrix,
)


class TestEuclideanDistanceMatrix:
    """Tests for euclidean_distance_matrix function."""

    def test_basic_2d_distances(self):
        """Test basic 2D Euclidean distance calculation."""
        centers = np.array([[0, 0], [3, 4], [6, 8]])

        result = euclidean_distance_matrix(centers)

        # Distance from [0,0] to [3,4] is 5.0
        # Distance from [0,0] to [6,8] is 10.0
        # Distance from [3,4] to [6,8] is 5.0
        expected = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]])

        assert_allclose(result, expected)

    def test_single_point(self):
        """Test distance matrix for single point."""
        centers = np.array([[1, 2]])

        result = euclidean_distance_matrix(centers)
        expected = np.array([[0.0]])

        assert_allclose(result, expected)
        assert result.shape == (1, 1)

    def test_two_points(self):
        """Test distance matrix for two points."""
        centers = np.array([[0, 0], [3, 4]])

        result = euclidean_distance_matrix(centers)

        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0
        assert_allclose(result[0, 1], 5.0)
        assert_allclose(result[1, 0], 5.0)

    def test_3d_distances(self):
        """Test 3D Euclidean distance calculation."""
        centers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = euclidean_distance_matrix(centers)

        # Diagonal should be zero
        assert_allclose(np.diag(result), [0.0, 0.0, 0.0, 0.0])

        # All off-diagonal unit distances should be 1.0
        assert_allclose(result[0, 1], 1.0)
        assert_allclose(result[0, 2], 1.0)
        assert_allclose(result[0, 3], 1.0)

        # Distance between [1,0,0] and [0,1,0] is sqrt(2)
        assert_allclose(result[1, 2], np.sqrt(2.0))

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        centers = np.random.rand(5, 2)

        result = euclidean_distance_matrix(centers)

        assert_allclose(result, result.T)

    def test_empty_points(self):
        """Test with empty points array."""
        centers = np.empty((0, 2))

        result = euclidean_distance_matrix(centers)

        assert result.shape == (0, 0)

    def test_colinear_points(self):
        """Test with co-linear points."""
        centers = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

        result = euclidean_distance_matrix(centers)

        # Distances should be difference in x-coordinates
        expected = np.array(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float
        )

        assert_allclose(result, expected)

    def test_identical_points(self):
        """Test with identical points."""
        centers = np.array([[1, 2], [1, 2], [1, 2]])

        result = euclidean_distance_matrix(centers)

        # All distances should be zero
        expected = np.zeros((3, 3))
        assert_allclose(result, expected)

    def test_large_distances(self):
        """Test with large coordinate values."""
        centers = np.array([[0, 0], [1000, 0], [0, 1000]])

        result = euclidean_distance_matrix(centers)

        assert_allclose(result[0, 1], 1000.0)
        assert_allclose(result[0, 2], 1000.0)
        assert_allclose(result[1, 2], 1000.0 * np.sqrt(2))

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        centers = np.array([[-1, -2], [3, 4]])

        result = euclidean_distance_matrix(centers)

        # Distance is sqrt((3-(-1))^2 + (4-(-2))^2) = sqrt(16 + 36) = sqrt(52)
        expected_dist = np.sqrt(52)
        assert_allclose(result[0, 1], expected_dist)
        assert_allclose(result[1, 0], expected_dist)


class TestGeodesicDistanceMatrix:
    """Tests for geodesic_distance_matrix function."""

    def test_basic_line_graph(self):
        """Test geodesic distances on simple line graph."""
        # Create simple line graph: 0--1--2--3
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

    def test_disconnected_graph(self):
        """Test geodesic distances with disconnected components."""
        # Create two disconnected components: 0--1  2--3
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        result = geodesic_distance_matrix(graph, n_states=4)

        # Disconnected nodes should have inf distance
        assert result[0, 2] == np.inf
        assert result[0, 3] == np.inf
        assert result[1, 2] == np.inf
        assert result[1, 3] == np.inf

        # Connected nodes should have finite distance
        assert result[0, 1] == 1.0
        assert result[2, 3] == 1.0

        # Diagonal should be zero
        assert_allclose(np.diag(result), [0.0, 0.0, 0.0, 0.0])

    def test_weighted_edges(self):
        """Test geodesic distances with different edge weights."""
        # Triangle with different weights
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(0, 2, distance=5.0)  # Longer direct path

        result = geodesic_distance_matrix(graph, n_states=3)

        # Shortest path from 0 to 2 should be via node 1 (distance 2.0, not 5.0)
        assert_allclose(result[0, 2], 2.0)
        assert_allclose(result[2, 0], 2.0)

    def test_single_node_graph(self):
        """Test with single isolated node."""
        graph = nx.Graph()
        graph.add_node(0)

        result = geodesic_distance_matrix(graph, n_states=1)

        expected = np.array([[0.0]])
        assert_allclose(result, expected)

    def test_empty_graph(self):
        """Test with empty graph."""
        graph = nx.Graph()

        result = geodesic_distance_matrix(graph, n_states=0)

        assert result.shape == (0, 0)

    def test_complete_graph(self):
        """Test with complete graph (all nodes connected)."""
        graph = nx.Graph()
        # Complete graph K4
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, distance=1.0)

        result = geodesic_distance_matrix(graph, n_states=4)

        # All off-diagonal elements should be 1.0
        expected = np.ones((4, 4))
        np.fill_diagonal(expected, 0.0)

        assert_allclose(result, expected)

    def test_unweighted_graph(self):
        """Test geodesic distances without edge weights (hop count)."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        result = geodesic_distance_matrix(graph, n_states=4, weight=None)

        # Distance is hop count when unweighted
        expected = np.array(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float
        )

        assert_allclose(result, expected)

    def test_cycle_graph(self):
        """Test geodesic distances on a cycle."""
        # Create cycle: 0--1--2--3--0
        graph = nx.Graph()
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        for u, v, w in edges:
            graph.add_edge(u, v, distance=w)

        result = geodesic_distance_matrix(graph, n_states=4)

        # From 0 to 2: can go 0->1->2 (distance 2) or 0->3->2 (distance 2)
        assert_allclose(result[0, 2], 2.0)

        # From 0 to 3: direct edge (distance 1)
        assert_allclose(result[0, 3], 1.0)

    def test_asymmetric_weights(self):
        """Test with asymmetric edge weights."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=10.0)
        graph.add_edge(2, 3, distance=1.0)
        graph.add_edge(0, 3, distance=2.0)  # Shortcut

        result = geodesic_distance_matrix(graph, n_states=4)

        # From 1 to 3: better to go 1->0->3 (distance 3) than 1->2->3 (distance 11)
        assert_allclose(result[1, 3], 3.0)

    def test_custom_weight_attribute(self):
        """Test using custom weight attribute name."""
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=5.0, distance=1.0)
        graph.add_edge(1, 2, weight=10.0, distance=1.0)

        # Use 'weight' instead of 'distance'
        result = geodesic_distance_matrix(graph, n_states=3, weight="weight")

        assert_allclose(result[0, 2], 15.0)  # Using weight attribute


class TestGeodesicDistanceBetweenPoints:
    """Tests for geodesic_distance_between_points function."""

    def test_basic_distance(self):
        """Test basic distance between two nodes."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=2.5)
        graph.add_edge(1, 2, distance=3.0)

        # Distance from 0 to 2 should be 5.5
        result = geodesic_distance_between_points(graph, 0, 2)
        assert_allclose(result, 5.5)

    def test_direct_connection(self):
        """Test distance between directly connected nodes."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=7.0)

        result = geodesic_distance_between_points(graph, 0, 1)
        assert_allclose(result, 7.0)

    def test_same_node(self):
        """Test distance from node to itself."""
        graph = nx.Graph()
        graph.add_node(0)

        result = geodesic_distance_between_points(graph, 0, 0)
        assert_allclose(result, 0.0)

    def test_no_path_returns_default(self):
        """Test that disconnected nodes return default value."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        # No path from 0 to 2
        result = geodesic_distance_between_points(graph, 0, 2)
        assert result == np.inf

    def test_custom_default(self):
        """Test custom default value for no path."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)

        # Node 2 doesn't exist
        result = geodesic_distance_between_points(graph, 0, 2, default=-1.0)
        assert result == -1.0

    def test_invalid_source_node(self):
        """Test with invalid source node."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)

        result = geodesic_distance_between_points(graph, 999, 0, default=np.inf)
        assert result == np.inf

    def test_invalid_target_node(self):
        """Test with invalid target node."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)

        result = geodesic_distance_between_points(graph, 0, 999, default=np.inf)
        assert result == np.inf

    def test_both_nodes_invalid(self):
        """Test with both nodes invalid."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)

        result = geodesic_distance_between_points(graph, 999, 888, default=-999.0)
        assert result == -999.0

    def test_shortest_path_selection(self):
        """Test that shortest path is selected when multiple paths exist."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(0, 2, distance=5.0)  # Longer direct path

        # Should use path via node 1 (total distance 2)
        result = geodesic_distance_between_points(graph, 0, 2)
        assert_allclose(result, 2.0)

    def test_with_zero_distance_edges(self):
        """Test with zero-distance edges."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=0.0)
        graph.add_edge(1, 2, distance=1.0)

        result = geodesic_distance_between_points(graph, 0, 2)
        assert_allclose(result, 1.0)

    def test_reverse_direction(self):
        """Test that distance is same in both directions (undirected graph)."""
        graph = nx.Graph()
        graph.add_edge(0, 1, distance=2.0)
        graph.add_edge(1, 2, distance=3.0)

        dist_forward = geodesic_distance_between_points(graph, 0, 2)
        dist_backward = geodesic_distance_between_points(graph, 2, 0)

        assert_allclose(dist_forward, dist_backward)
        assert_allclose(dist_forward, 5.0)
