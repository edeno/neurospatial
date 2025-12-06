"""Tests for ops/calculus.py - verifying new import path works.

This module tests that the calculus functions (gradient, divergence,
compute_differential_operator) are accessible from the new ops/calculus location.

These tests are designed to FAIL initially (TDD RED phase) until
differential.py is moved to ops/calculus.py.
"""

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from neurospatial import Environment


class TestCalculusImports:
    """Test that functions can be imported from new location."""

    def test_import_from_ops_calculus(self):
        """Test that all functions can be imported from ops.calculus."""
        from neurospatial.ops.calculus import (
            compute_differential_operator,
            divergence,
            gradient,
        )

        # Verify they are callable
        assert callable(compute_differential_operator)
        assert callable(gradient)
        assert callable(divergence)

    def test_import_from_ops_init(self):
        """Test that calculus functions are exported from ops/__init__.py."""
        from neurospatial.ops import (
            compute_differential_operator,
            divergence,
            gradient,
        )

        # Verify they are callable
        assert callable(compute_differential_operator)
        assert callable(gradient)
        assert callable(divergence)


class TestDifferentialOperatorFromOps:
    """Test compute_differential_operator() from new location."""

    def test_differential_operator_shape(self):
        """Test that differential operator has shape (n_bins, n_edges)."""
        from neurospatial.ops.calculus import compute_differential_operator

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
        from neurospatial.ops.calculus import compute_differential_operator

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

    def test_differential_operator_single_node(self):
        """Test edge case: single node (no edges)."""
        from neurospatial.ops.calculus import compute_differential_operator

        data = np.array([[0.0, 0.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = compute_differential_operator(env)

        # Single node should have shape (1, 0) - no edges
        assert D.shape == (1, 0)


class TestGradientFromOps:
    """Test gradient() from new location."""

    def test_gradient_shape(self):
        """Test that gradient output has shape (n_edges,)."""
        from neurospatial.ops.calculus import gradient

        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create a test field (one value per bin)
        rng = np.random.default_rng(42)
        field = rng.random(env.n_bins)

        # Compute gradient
        grad_field = gradient(env, field)

        # Output should have shape (n_edges,)
        n_edges = len(env.connectivity.edges)
        assert grad_field.shape == (n_edges,)
        assert isinstance(grad_field, np.ndarray)

    def test_gradient_constant_field(self):
        """Test that gradient of a constant field is zero everywhere."""
        from neurospatial.ops.calculus import gradient

        # Create 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create constant field
        field = np.ones(env.n_bins) * 5.0

        # Gradient should be all zeros
        grad_field = gradient(env, field)

        np.testing.assert_allclose(grad_field, 0.0, atol=1e-10)

    def test_gradient_validation(self):
        """Test that gradient validates input field shape."""
        from neurospatial.ops.calculus import gradient

        # Create environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Wrong shape: too few elements
        field_too_small = np.array([1.0, 2.0])

        # Should raise ValueError
        with pytest.raises(ValueError, match=r"field.*shape"):
            gradient(env, field_too_small)


class TestDivergenceFromOps:
    """Test divergence() from new location."""

    def test_divergence_shape(self):
        """Test that divergence output has shape (n_bins,)."""
        from neurospatial.ops.calculus import divergence

        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create a test edge field (one value per edge)
        n_edges = len(env.connectivity.edges)
        rng = np.random.default_rng(42)
        edge_field = rng.random(n_edges)

        # Compute divergence
        div_field = divergence(env, edge_field)

        # Output should have shape (n_bins,)
        assert div_field.shape == (env.n_bins,)
        assert isinstance(div_field, np.ndarray)

    def test_divergence_gradient_is_laplacian(self):
        """Test that div(grad(f)) equals Laplacian(f) for scalar field f."""
        from neurospatial.ops.calculus import divergence, gradient

        # Create 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create test field
        field = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

        # Compute div(grad(f))
        grad_field = gradient(env, field)
        div_grad_field = divergence(env, grad_field)

        # Compute Laplacian directly
        L = nx.laplacian_matrix(env.connectivity, weight="distance").toarray()
        laplacian_field = L @ field

        # They should be equal (within numerical precision)
        np.testing.assert_allclose(
            div_grad_field, laplacian_field, rtol=1e-10, atol=1e-10
        )

    def test_divergence_zero_edge_field(self):
        """Test that divergence of zero edge field is zero everywhere."""
        from neurospatial.ops.calculus import divergence

        # Create 2D grid environment
        data = np.array([[i, j] for i in range(3) for j in range(3)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create zero edge field
        n_edges = len(env.connectivity.edges)
        edge_field = np.zeros(n_edges)

        # Divergence should be all zeros
        div_field = divergence(env, edge_field)

        np.testing.assert_allclose(div_field, 0.0, atol=1e-10)

    def test_divergence_validation(self):
        """Test that divergence validates input edge field shape."""
        from neurospatial.ops.calculus import divergence

        # Create environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Wrong shape: too few elements
        edge_field_too_small = np.array([1.0, 2.0])

        # Should raise ValueError
        with pytest.raises(ValueError, match=r"edge_field.*shape"):
            divergence(env, edge_field_too_small)
