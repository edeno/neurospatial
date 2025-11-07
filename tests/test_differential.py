"""Tests for differential operators on graph-discretized environments.

This module tests the computation of the differential operator D and its
relationship to the graph Laplacian L = D @ D.T.
"""

import networkx as nx
import numpy as np
from scipy import sparse

from neurospatial import Environment
from neurospatial.differential import compute_differential_operator


class TestDifferentialOperatorComputation:
    """Test compute_differential_operator() function."""

    def test_differential_operator_shape(self):
        """Test that differential operator has shape (n_bins, n_edges)."""
        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = compute_differential_operator(env)

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)

        assert D.shape == (n_bins, n_edges)
        assert isinstance(D, sparse.csc_matrix)

    def test_laplacian_from_differential(self):
        """Test that D @ D.T equals the graph Laplacian matrix."""
        # Create simple 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = compute_differential_operator(env)

        # Compute Laplacian from differential operator
        L_from_D = (D @ D.T).toarray()

        # Get networkx Laplacian (unnormalized)
        L_nx = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()

        # They should be equal (within numerical precision)
        np.testing.assert_allclose(L_from_D, L_nx, rtol=1e-10, atol=1e-10)

    def test_differential_operator_sparse(self):
        """Test that differential operator is sparse (CSC format)."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = compute_differential_operator(env)

        assert sparse.issparse(D)
        assert isinstance(D, sparse.csc_matrix)

    def test_differential_operator_edge_weights(self):
        """Test that differential operator uses sqrt of edge distances."""
        # Simple 1D chain with known distances
        data = np.array([[0.0], [1.0], [2.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = compute_differential_operator(env)

        # For a 1D chain, edge distances should be 1.0
        # D should contain +/-sqrt(1) = +/-1
        D_dense = D.toarray()

        # Non-zero elements should be +1 or -1
        nonzero_values = D_dense[D_dense != 0]
        np.testing.assert_allclose(np.abs(nonzero_values), 1.0, atol=1e-10)

    def test_differential_operator_single_node(self):
        """Test edge case: single node (no edges)."""
        data = np.array([[0.0, 0.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = compute_differential_operator(env)

        # Single node should have shape (1, 0) - no edges
        assert D.shape == (1, 0)

    def test_differential_operator_disconnected_graph(self):
        """Test differential operator on disconnected graph."""
        # Create environment with disconnected components
        # Two separate 1D chains
        data = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # This should work even with disconnected components
        D = compute_differential_operator(env)

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

        # Laplacian relationship should still hold
        L_from_D = (D @ D.T).toarray()
        L_nx = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()
        np.testing.assert_allclose(L_from_D, L_nx, rtol=1e-10, atol=1e-10)


class TestEnvironmentCachedProperty:
    """Test differential_operator cached property on Environment."""

    def test_differential_operator_property_exists(self):
        """Test that Environment has differential_operator property."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        assert hasattr(env, "differential_operator")

    def test_differential_operator_caching(self):
        """Test that differential_operator is cached (same object on repeated access)."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Access twice
        D1 = env.differential_operator
        D2 = env.differential_operator

        # Should be the same object (cached)
        assert D1 is D2

    def test_differential_operator_correct_shape(self):
        """Test that cached property returns correct shape."""
        data = np.random.rand(50, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.differential_operator

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

    def test_differential_operator_matches_function(self):
        """Test that property returns same result as direct function call."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        D_property = env.differential_operator
        D_function = compute_differential_operator(env)

        # Should produce identical matrices
        np.testing.assert_array_equal(D_property.toarray(), D_function.toarray())


class TestDifferentialOperatorEdgeCases:
    """Test edge cases and special graph structures."""

    def test_differential_operator_regular_grid(self):
        """Test on regular 2D grid."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.differential_operator

        # Should have correct shape
        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

        # Laplacian relationship should hold
        L_from_D = (D @ D.T).toarray()
        L_nx = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()
        np.testing.assert_allclose(L_from_D, L_nx, rtol=1e-10, atol=1e-10)

    def test_differential_operator_irregular_spacing(self):
        """Test on irregularly spaced points."""
        np.random.seed(42)
        data = np.random.rand(20, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.differential_operator

        # Should still satisfy Laplacian relationship
        L_from_D = (D @ D.T).toarray()
        L_nx = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()
        np.testing.assert_allclose(L_from_D, L_nx, rtol=1e-10, atol=1e-10)

    def test_differential_operator_preserves_symmetry(self):
        """Test that D @ D.T produces symmetric Laplacian."""
        data = np.random.rand(30, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.differential_operator
        L = (D @ D.T).toarray()

        # Laplacian should be symmetric
        np.testing.assert_allclose(L, L.T, rtol=1e-10, atol=1e-10)


class TestGradientOperator:
    """Test gradient() function for computing gradients on graph-discretized fields."""

    def test_gradient_shape(self):
        """Test that gradient output has shape (n_edges,)."""
        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create a test field (one value per bin)
        field = np.random.rand(env.n_bins)

        # Import gradient function (will fail initially - this is TDD RED phase)
        from neurospatial.differential import gradient

        # Compute gradient
        grad_field = gradient(field, env)

        # Output should have shape (n_edges,)
        n_edges = len(env.connectivity.edges)
        assert grad_field.shape == (n_edges,)
        assert isinstance(grad_field, np.ndarray)

    def test_gradient_constant_field(self):
        """Test that gradient of a constant field is zero everywhere."""
        # Create 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create constant field
        field = np.ones(env.n_bins) * 5.0

        from neurospatial.differential import gradient

        # Gradient should be all zeros
        grad_field = gradient(field, env)

        np.testing.assert_allclose(grad_field, 0.0, atol=1e-10)

    def test_gradient_linear_field_regular_grid(self):
        """Test that gradient of linear field is constant on regular grid."""
        # Create 1D chain with uniform spacing
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create linear field: f(x) = 2*x
        # Bins at x = 0, 1, 2, 3, 4
        field = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

        from neurospatial.differential import gradient

        # Compute gradient
        grad_field = gradient(field, env)

        # For uniform grid with spacing 1.0 and slope 2.0,
        # gradient should be constant (approximately 2.0 * sqrt(1.0) = 2.0)
        # The differential operator uses sqrt(distance), so we expect consistent values
        grad_values = np.abs(grad_field)

        # All gradient magnitudes should be similar (constant gradient)
        assert np.allclose(grad_values, grad_values[0], rtol=0.1)

    def test_gradient_validation(self):
        """Test that gradient validates input field shape."""
        # Create environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        from neurospatial.differential import gradient

        # Wrong shape: too few elements
        field_too_small = np.array([1.0, 2.0])

        # Should raise ValueError
        import pytest

        with pytest.raises(ValueError, match=r"field.*shape"):
            gradient(field_too_small, env)

        # Wrong shape: too many elements
        field_too_large = np.random.rand(env.n_bins + 5)

        with pytest.raises(ValueError, match=r"field.*shape"):
            gradient(field_too_large, env)
