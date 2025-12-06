"""Tests for neighbors_within() function.

This module tests the neighbors_within() function for finding all bins
within a specified radius of center bins, using either geodesic or Euclidean metrics.

Tests follow TDD - written BEFORE implementation.
"""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.ops.distance import neighbors_within


class TestNeighborsWithinGeodesic:
    """Test neighbors_within() with geodesic metric."""

    @pytest.fixture
    def linear_graph(self):
        """Create a simple linear graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_single_center_geodesic(self, linear_graph):
        """Test geodesic neighborhood from single center."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[2], radius=1.5, metric="geodesic"
        )

        assert len(neighborhoods) == 1
        # Node 2 with radius 1.5 should reach nodes 1, 2, 3
        assert set(neighborhoods[0]) == {1, 2, 3}

    def test_multiple_centers_geodesic(self, linear_graph):
        """Test geodesic neighborhoods from multiple centers."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[1, 3], radius=1.0, metric="geodesic"
        )

        assert len(neighborhoods) == 2
        # Center 1 should reach 0, 1, 2
        assert set(neighborhoods[0]) == {0, 1, 2}
        # Center 3 should reach 2, 3, 4
        assert set(neighborhoods[1]) == {2, 3, 4}

    def test_exclude_center(self, linear_graph):
        """Test exclude_center=True."""
        neighborhoods = neighbors_within(
            linear_graph,
            centers=[2],
            radius=1.5,
            metric="geodesic",
            include_center=False,
        )

        assert len(neighborhoods) == 1
        # Should not include center node 2
        assert set(neighborhoods[0]) == {1, 3}

    def test_include_center_default(self, linear_graph):
        """Test that include_center=True is default."""
        neighborhoods_explicit = neighbors_within(
            linear_graph,
            centers=[2],
            radius=1.0,
            metric="geodesic",
            include_center=True,
        )
        neighborhoods_default = neighbors_within(
            linear_graph, centers=[2], radius=1.0, metric="geodesic"
        )

        assert_array_equal(neighborhoods_explicit[0], neighborhoods_default[0])

    def test_radius_zero(self, linear_graph):
        """Test radius=0 returns only center (or empty if excluded)."""
        # With include_center=True
        neighborhoods = neighbors_within(
            linear_graph,
            centers=[2],
            radius=0.0,
            metric="geodesic",
            include_center=True,
        )
        assert_array_equal(neighborhoods[0], [2])

        # With include_center=False
        neighborhoods = neighbors_within(
            linear_graph,
            centers=[2],
            radius=0.0,
            metric="geodesic",
            include_center=False,
        )
        assert len(neighborhoods[0]) == 0

    def test_large_radius(self, linear_graph):
        """Test very large radius reaches all nodes in component."""
        neighborhoods = neighbors_within(
            linear_graph, centers=[0], radius=1000.0, metric="geodesic"
        )

        # Should reach all 5 nodes
        assert set(neighborhoods[0]) == {0, 1, 2, 3, 4}


class TestNeighborsWithinEuclidean:
    """Test neighbors_within() with Euclidean metric."""

    @pytest.fixture
    def grid_graph_with_coords(self):
        """Create a 3x3 grid graph with coordinates."""
        G = nx.Graph()
        positions = {}
        node_id = 0
        for i in range(3):
            for j in range(3):
                positions[node_id] = (float(i), float(j))
                G.add_node(node_id)
                node_id += 1

        # Add edges
        for i in range(3):
            for j in range(3):
                node = i * 3 + j
                if i < 2:
                    neighbor = (i + 1) * 3 + j
                    G.add_edge(node, neighbor, distance=1.0)
                if j < 2:
                    neighbor = i * 3 + (j + 1)
                    G.add_edge(node, neighbor, distance=1.0)

        # Store positions
        for node, pos in positions.items():
            G.nodes[node]["pos"] = pos

        n_nodes = G.number_of_nodes()
        bin_centers = np.zeros((n_nodes, 2))
        for node in G.nodes():
            bin_centers[node] = G.nodes[node]["pos"]

        return G, bin_centers

    def test_single_center_euclidean(self, grid_graph_with_coords):
        """Test Euclidean neighborhood from single center."""
        G, bin_centers = grid_graph_with_coords

        # Center at node 4 (position 1, 1), radius 1.1
        # Should include center and 4-neighbors at distance 1.0, but not diagonal corners
        neighborhoods = neighbors_within(
            G, centers=[4], radius=1.1, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 1
        # Should include center (1,1) and 4-neighbors at distance 1.0 (cross pattern)
        # Diagonal corners at distance sqrt(2) ≈ 1.414 are excluded
        assert set(neighborhoods[0]) == {1, 3, 4, 5, 7}

    def test_corner_euclidean(self, grid_graph_with_coords):
        """Test Euclidean neighborhood from corner."""
        G, bin_centers = grid_graph_with_coords

        # Corner at node 0 (position 0, 0), radius sqrt(2) + 0.1
        neighborhoods = neighbors_within(
            G,
            centers=[0],
            radius=np.sqrt(2.0) + 0.1,
            metric="euclidean",
            bin_centers=bin_centers,
        )

        # Should include 0, 1 (right), 3 (up), and 4 (diagonal at distance sqrt(2))
        assert set(neighborhoods[0]) == {0, 1, 3, 4}

    def test_multiple_centers_euclidean(self, grid_graph_with_coords):
        """Test Euclidean neighborhoods from multiple centers."""
        G, bin_centers = grid_graph_with_coords

        neighborhoods = neighbors_within(
            G, centers=[0, 8], radius=1.1, metric="euclidean", bin_centers=bin_centers
        )

        assert len(neighborhoods) == 2
        # Node 0 at (0, 0): includes itself and right/up neighbors
        assert set(neighborhoods[0]) == {0, 1, 3}
        # Node 8 at (2, 2): includes itself and left/down neighbors
        assert set(neighborhoods[1]) == {5, 7, 8}

    def test_euclidean_3d(self):
        """Test Euclidean neighborhoods in 3D."""
        G = nx.Graph()
        bin_centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        for i in range(4):
            G.add_node(i)

        neighborhoods = neighbors_within(
            G, centers=[0], radius=1.5, metric="euclidean", bin_centers=bin_centers
        )

        # Should include center and 3 unit-distance neighbors
        assert set(neighborhoods[0]) == {0, 1, 2, 3}


class TestNeighborsWithinComparison:
    """Test comparison between geodesic and Euclidean metrics."""

    @pytest.fixture
    def grid_graph_with_coords(self):
        """Create a 5x5 grid graph."""
        G = nx.Graph()
        positions = {}
        node_id = 0
        for i in range(5):
            for j in range(5):
                positions[node_id] = (float(i), float(j))
                G.add_node(node_id)
                node_id += 1

        for i in range(5):
            for j in range(5):
                node = i * 5 + j
                if i < 4:
                    neighbor = (i + 1) * 5 + j
                    G.add_edge(node, neighbor, distance=1.0)
                if j < 4:
                    neighbor = i * 5 + (j + 1)
                    G.add_edge(node, neighbor, distance=1.0)

        for node, pos in positions.items():
            G.nodes[node]["pos"] = pos

        bin_centers = np.array([positions[i] for i in range(25)])

        return G, bin_centers

    def test_euclidean_subset_of_geodesic(self, grid_graph_with_coords):
        """Test that Euclidean neighborhoods can be smaller than geodesic."""
        G, bin_centers = grid_graph_with_coords

        # Center at node 12 (middle of 5x5 grid)
        radius = 2.0

        geo_neighbors = neighbors_within(
            G, centers=[12], radius=radius, metric="geodesic"
        )[0]
        euc_neighbors = neighbors_within(
            G, centers=[12], radius=radius, metric="euclidean", bin_centers=bin_centers
        )[0]

        # Euclidean neighborhood should be subset (or equal) to geodesic
        assert set(euc_neighbors).issubset(set(geo_neighbors))


class TestNeighborsWithinInputValidation:
    """Test input validation for neighbors_within()."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_invalid_metric(self, simple_graph):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match=r"metric must be.*geodesic.*euclidean"):
            neighbors_within(simple_graph, centers=[0], radius=1.0, metric="invalid")

    def test_negative_radius(self, simple_graph):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be non-negative"):
            neighbors_within(simple_graph, centers=[0], radius=-1.0, metric="geodesic")

    def test_empty_centers(self, simple_graph):
        """Test that empty centers returns empty list."""
        neighborhoods = neighbors_within(
            simple_graph, centers=[], radius=1.0, metric="geodesic"
        )

        assert neighborhoods == []

    def test_invalid_center_raises_error(self, simple_graph):
        """Test that invalid center index raises error."""
        with pytest.raises((nx.NodeNotFound, KeyError)):
            neighbors_within(simple_graph, centers=[999], radius=1.0, metric="geodesic")

    def test_euclidean_requires_bin_centers(self, simple_graph):
        """Test that Euclidean metric requires bin_centers."""
        with pytest.raises(ValueError, match=r"bin_centers.*required.*euclidean"):
            neighbors_within(simple_graph, centers=[0], radius=1.0, metric="euclidean")

    def test_bin_centers_shape_mismatch(self, simple_graph):
        """Test that bin_centers size must match graph nodes."""
        bin_centers = np.array([[0.0, 0.0]])  # Only 1 row, graph has 3 nodes

        with pytest.raises(
            ValueError, match=r"bin_centers.*must match.*number of nodes"
        ):
            neighbors_within(
                simple_graph,
                centers=[0],
                radius=1.0,
                metric="euclidean",
                bin_centers=bin_centers,
            )


class TestNeighborsWithinEdgeCases:
    """Test edge cases for neighbors_within()."""

    def test_single_node_graph(self):
        """Test with single node graph."""
        G = nx.Graph()
        G.add_node(0)

        neighborhoods = neighbors_within(G, centers=[0], radius=1.0, metric="geodesic")

        assert len(neighborhoods) == 1
        assert_array_equal(neighborhoods[0], [0])

    def test_disconnected_graph(self):
        """Test with disconnected graph."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)
        G.add_edge(2, 3, distance=1.0)

        # Center in first component
        neighborhoods = neighbors_within(G, centers=[0], radius=10.0, metric="geodesic")

        # Should only reach component containing center
        assert set(neighborhoods[0]) == {0, 1}

    def test_center_in_different_components(self):
        """Test multiple centers in different components."""
        G = nx.Graph()
        G.add_edge(0, 1, distance=1.0)
        G.add_edge(2, 3, distance=1.0)

        neighborhoods = neighbors_within(
            G, centers=[0, 2], radius=1.0, metric="geodesic"
        )

        assert len(neighborhoods) == 2
        assert set(neighborhoods[0]) == {0, 1}
        assert set(neighborhoods[1]) == {2, 3}

    def test_overlapping_neighborhoods(self):
        """Test that overlapping neighborhoods are independent."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        neighborhoods = neighbors_within(
            G, centers=[1, 2], radius=1.0, metric="geodesic"
        )

        # Both neighborhoods should include node 1 and 2
        assert 1 in neighborhoods[0] and 2 in neighborhoods[0]
        assert 1 in neighborhoods[1] and 2 in neighborhoods[1]

    def test_all_nodes_as_centers(self):
        """Test using all nodes as centers."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0

        neighborhoods = neighbors_within(
            G, centers=[0, 1, 2], radius=0.5, metric="geodesic", include_center=True
        )

        assert len(neighborhoods) == 3
        # Each should only contain itself (radius too small for neighbors)
        assert_array_equal(neighborhoods[0], [0])
        assert_array_equal(neighborhoods[1], [1])
        assert_array_equal(neighborhoods[2], [2])


class TestNeighborsWithinReturnTypes:
    """Test return types and data structures."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for u, v in G.edges:
            G.edges[u, v]["distance"] = 1.0
        return G

    def test_returns_list_of_arrays(self, simple_graph):
        """Test that function returns list of numpy arrays."""
        neighborhoods = neighbors_within(
            simple_graph, centers=[1, 2], radius=1.0, metric="geodesic"
        )

        assert isinstance(neighborhoods, list)
        assert len(neighborhoods) == 2
        assert isinstance(neighborhoods[0], np.ndarray)
        assert isinstance(neighborhoods[1], np.ndarray)
        assert neighborhoods[0].dtype == np.int_
        assert neighborhoods[1].dtype == np.int_

    def test_arrays_are_unique(self, simple_graph):
        """Test that returned arrays contain unique indices."""
        neighborhoods = neighbors_within(
            simple_graph, centers=[1], radius=2.0, metric="geodesic"
        )

        neighbors = neighborhoods[0]
        assert len(neighbors) == len(np.unique(neighbors))

    def test_order_unspecified(self, simple_graph):
        """Test that order of neighbors is unspecified (document this)."""
        neighborhoods = neighbors_within(
            simple_graph, centers=[1], radius=1.0, metric="geodesic"
        )

        # Just verify all expected neighbors present (not testing order)
        assert set(neighborhoods[0]) == {0, 1, 2}
