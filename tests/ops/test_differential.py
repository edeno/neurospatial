"""Tests for differential operators on graph-discretized environments.

This module tests the computation of the differential operator D and its
relationship to the finite-volume diffusion generator.
"""

import numpy as np
import pytest
from environment_helpers import native_layout
from scipy import sparse

from neurospatial import Environment
from neurospatial.ops.calculus import _compute_divergence_operator


def _diffusion_generator(env):
    """Reference generator assembled from the diffusion geometry."""
    weights, volumes, _, _ = env._diffusion_geometry
    return sparse.diags(1 / volumes) @ (
        sparse.diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
    )


@pytest.mark.parametrize("spacing", [1.0, 2.0, 4.0])
def test_gradient_of_x_is_unit_slope(uniform_grid, spacing):
    from neurospatial.ops.calculus import gradient

    env = uniform_grid(spacing, n=int(20 / spacing))
    edges = np.array(list(env.connectivity.edges()))
    delta = env.bin_centers[edges[:, 1]] - env.bin_centers[edges[:, 0]]
    x_edges = (delta[:, 0] != 0) & (delta[:, 1] == 0)
    np.testing.assert_allclose(
        np.abs(gradient(env, env.bin_centers[:, 0]))[x_edges], 1, atol=1e-12
    )


@pytest.mark.parametrize("spacing", [1.0, 2.0, 4.0])
def test_div_grad_quadratic_is_continuum_laplacian(uniform_grid, spacing):
    from neurospatial.ops.calculus import divergence, gradient

    env = uniform_grid(spacing, n=int(20 / spacing))
    field = np.sum(env.bin_centers**2, axis=1)
    interior = np.array([degree == 8 for _, degree in env.connectivity.degree()])
    np.testing.assert_allclose(
        divergence(env, gradient(env, field))[interior], 4, atol=1e-12
    )


def test_div_grad_equals_negative_diffusion_generator(fv_env):
    from neurospatial.ops.calculus import divergence, gradient

    laplacian = _diffusion_generator(fv_env)
    field = np.random.default_rng(0).normal(size=fv_env.n_bins)
    np.testing.assert_allclose(
        divergence(fv_env, gradient(fv_env, field)),
        -laplacian @ field,
        rtol=1e-10,
        atol=1e-12,
    )


def test_get_divergence_operator_is_cached_and_applied_by_divergence(fv_env):
    from neurospatial.ops.calculus import divergence

    operator = fv_env.get_divergence_operator()
    cached = fv_env._divergence_operator_cached
    exported = fv_env.get_divergence_operator()
    assert exported is not operator
    assert fv_env._divergence_operator_cached is cached
    np.testing.assert_array_equal(exported.toarray(), operator.toarray())
    np.testing.assert_allclose(
        operator.toarray(),
        _compute_divergence_operator(
            fv_env.binning.faces.build(), fv_env.get_differential_operator()
        ).toarray(),
    )
    flux = np.random.default_rng(4).normal(size=fv_env.connectivity.number_of_edges())
    np.testing.assert_allclose(divergence(fv_env, flux), operator @ flux)


def test_divergence_is_negative_adjoint_of_gradient(fv_env):
    from neurospatial.ops.calculus import divergence, gradient

    graph, volumes = native_layout(fv_env).finite_volume_geometry()
    rng = np.random.default_rng(3)
    field = rng.normal(size=fv_env.n_bins)
    flux = rng.normal(size=graph.number_of_edges())
    edge_measure = np.array(
        [data["A"] * data["distance"] for _, _, data in graph.edges(data=True)]
    )
    np.testing.assert_allclose(
        np.sum(edge_measure * gradient(fv_env, field) * flux),
        -np.sum(volumes * field * divergence(fv_env, flux)),
        rtol=1e-10,
    )


@pytest.mark.parametrize("spacing", [1.0, 2.0, 4.0])
def test_goal_is_a_sink(uniform_grid, spacing):
    from neurospatial.ops.calculus import divergence, gradient

    env = uniform_grid(spacing, n=int(20 / spacing))
    goal = np.argmin(np.linalg.norm(env.bin_centers - 10, axis=1))
    distance = np.linalg.norm(env.bin_centers - env.bin_centers[goal], axis=1)
    assert divergence(env, -gradient(env, distance))[goal] < 0


def test_operator_edge_order_matches_connectivity(fv_env):
    from neurospatial.ops.calculus import gradient

    graph, _ = native_layout(fv_env).finite_volume_geometry()
    assert list(graph.edges()) == list(fv_env.connectivity.edges())
    field = np.random.default_rng(5).normal(size=fv_env.n_bins)
    expected = [
        (field[v] - field[u]) / distance
        for u, v, distance in graph.edges(data="distance")
    ]
    np.testing.assert_allclose(gradient(fv_env, field), expected, atol=1e-12)


@pytest.mark.nwb
def test_operators_keep_faces_after_nwb_round_trip(graph_environment_round_trip):
    from neurospatial.ops.calculus import divergence, gradient

    original, loaded = graph_environment_round_trip
    field = np.sin(np.arange(original.n_bins))
    flux = np.cos(np.arange(original.connectivity.number_of_edges()))
    np.testing.assert_array_equal(gradient(loaded, field), gradient(original, field))
    np.testing.assert_array_equal(divergence(loaded, flux), divergence(original, flux))


def test_operators_raise_without_finite_volume_geometry(missing_faces_env):
    from neurospatial.ops.calculus import divergence, gradient

    env = missing_faces_env
    for func, field in [
        (gradient, np.ones(env.n_bins)),
        (divergence, np.ones(env.connectivity.number_of_edges())),
    ]:
        with pytest.raises(NotImplementedError, match=r"gradient/divergence.*") as exc:
            func(env, field)
        assert "\nFix:" in str(exc.value)


class TestDifferentialOperatorComputation:
    """Test get_differential_operator() function."""

    def test_differential_operator_shape(self):
        """Test that differential operator has shape (n_bins, n_edges)."""
        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.get_differential_operator()

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)

        assert D.shape == (n_bins, n_edges)
        assert isinstance(D, sparse.csc_matrix)

    def test_laplacian_from_differential(self):
        """Test that the negative divergence of the gradient equals the diffusion generator."""
        # Create simple 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.get_differential_operator()

        # Compute Laplacian from differential operator
        L_from_D = (
            -_compute_divergence_operator(
                env.binning.faces.build(), env.get_differential_operator()
            )
            @ D.T
        ).toarray()

        # Get finite-volume diffusion generator
        L_fv = _diffusion_generator(env).toarray()

        # They should be equal (within numerical precision)
        np.testing.assert_allclose(L_from_D, L_fv, rtol=1e-10, atol=1e-10)

    def test_differential_operator_sparse(self):
        """Test that differential operator is sparse (CSC format)."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.get_differential_operator()

        assert sparse.issparse(D)
        assert isinstance(D, sparse.csc_matrix)

    def test_differential_operator_edge_weights(self):
        """Test that differential operator uses inverse edge distances."""
        # Simple 1D chain with known distances
        data = np.array([[0.0], [1.0], [2.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.get_differential_operator()

        # For a 1D chain, edge distances should be 1.0
        # D should contain +/-1/d = +/-1 for d=1
        D_dense = D.toarray()

        # Non-zero elements should be +1 or -1
        nonzero_values = D_dense[D_dense != 0]
        np.testing.assert_allclose(np.abs(nonzero_values), 1.0, atol=1e-10)

    def test_differential_operator_single_node(self):
        """Test edge case: single node (no edges)."""
        data = np.array([[0.0, 0.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.get_differential_operator()

        # Single node should have shape (1, 0) - no edges
        assert D.shape == (1, 0)

    def test_differential_operator_disconnected_graph(self):
        """Test differential operator on disconnected graph."""
        # Create environment with disconnected components
        # Two separate 1D chains
        data = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # This should work even with disconnected components
        D = env.get_differential_operator()

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

        # Finite-volume generator relationship should still hold
        L_from_D = (
            -_compute_divergence_operator(
                env.binning.faces.build(), env.get_differential_operator()
            )
            @ D.T
        ).toarray()
        L_fv = _diffusion_generator(env).toarray()
        np.testing.assert_allclose(L_from_D, L_fv, rtol=1e-10, atol=1e-10)


class TestEnvironmentCachedProperty:
    """Test differential_operator cached property on Environment."""

    def test_differential_operator_property_exists(self):
        """Environment exposes get_differential_operator() (method form)."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        assert hasattr(env, "get_differential_operator")
        assert callable(env.get_differential_operator)

    def test_differential_operator_caching(self):
        """The internal operator is cached and public exports are detached."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Access twice
        D1 = env.get_differential_operator()
        D2 = env.get_differential_operator()

        # Should be the same object (cached)
        assert D1 is not D2
        np.testing.assert_array_equal(D1.toarray(), D2.toarray())

    def test_differential_operator_correct_shape(self):
        """Test that cached property returns correct shape."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.get_differential_operator()

        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

    def test_differential_operator_matches_function(self):
        """Test that property returns same result as direct function call."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        D_property = env.get_differential_operator()
        D_function = env.get_differential_operator()

        # Should produce identical matrices
        np.testing.assert_array_equal(D_property.toarray(), D_function.toarray())


class TestDifferentialOperatorEdgeCases:
    """Test edge cases and special graph structures."""

    def test_differential_operator_regular_grid(self):
        """Test on regular 2D grid."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        D = env.get_differential_operator()

        # Should have correct shape
        n_bins = env.n_bins
        n_edges = len(env.connectivity.edges)
        assert D.shape == (n_bins, n_edges)

        # Finite-volume generator relationship should hold
        L_from_D = (
            -_compute_divergence_operator(
                env.binning.faces.build(), env.get_differential_operator()
            )
            @ D.T
        ).toarray()
        L_fv = _diffusion_generator(env).toarray()
        np.testing.assert_allclose(L_from_D, L_fv, rtol=1e-10, atol=1e-10)

    def test_differential_operator_irregular_spacing(self):
        """Test on irregularly spaced points."""
        rng = np.random.default_rng(42)
        data = rng.random((20, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.get_differential_operator()

        # Should still satisfy Finite-volume generator relationship
        L_from_D = (
            -_compute_divergence_operator(
                env.binning.faces.build(), env.get_differential_operator()
            )
            @ D.T
        ).toarray()
        L_fv = _diffusion_generator(env).toarray()
        np.testing.assert_allclose(L_from_D, L_fv, rtol=1e-10, atol=1e-10)

    def test_differential_operator_preserves_symmetry(self):
        """Test that the diffusion generator is symmetric on a uniform-volume grid."""
        rng = np.random.default_rng(42)
        data = rng.random((30, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        D = env.get_differential_operator()
        L = (
            -_compute_divergence_operator(
                env.binning.faces.build(), env.get_differential_operator()
            )
            @ D.T
        ).toarray()

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
        rng = np.random.default_rng(42)
        field = rng.random(env.n_bins)

        # Import gradient function (will fail initially - this is TDD RED phase)
        from neurospatial.ops.calculus import gradient

        # Compute gradient
        grad_field = gradient(env, field)

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

        from neurospatial.ops.calculus import gradient

        # Gradient should be all zeros
        grad_field = gradient(env, field)

        np.testing.assert_allclose(grad_field, 0.0, atol=1e-10)

    def test_gradient_linear_field_regular_grid(self):
        """Test that gradient of linear field is constant on regular grid."""
        # Create 1D chain with uniform spacing
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create linear field: f(x) = 2*x
        # Bins at x = 0, 1, 2, 3, 4
        field = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

        from neurospatial.ops.calculus import gradient

        # Compute gradient
        grad_field = gradient(env, field)

        # For uniform grid with spacing 1.0 and slope 2.0,
        # gradient should be constant (2.0 / 1.0 = 2.0)
        # The derivative divides the field difference by the edge distance
        grad_values = np.abs(grad_field)

        # All gradient magnitudes should be similar (constant gradient)
        assert np.allclose(grad_values, grad_values[0], rtol=0.1)

    def test_gradient_validation(self):
        """Test that gradient validates input field shape."""
        # Create environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        from neurospatial.ops.calculus import gradient

        # Wrong shape: too few elements
        field_too_small = np.array([1.0, 2.0])

        # Should raise ValueError
        import pytest

        with pytest.raises(ValueError, match=r"field.*shape"):
            gradient(env, field_too_small)

        # Wrong shape: too many elements
        rng = np.random.default_rng(42)
        field_too_large = rng.random(env.n_bins + 5)

        with pytest.raises(ValueError, match=r"field.*shape"):
            gradient(env, field_too_large)


class TestDivergenceOperator:
    """Test divergence() function for computing divergence on graph-discretized edge fields."""

    def test_divergence_shape(self):
        """Test that divergence output has shape (n_bins,)."""
        # Create simple 2x2 grid environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create a test edge field (one value per edge)
        n_edges = len(env.connectivity.edges)
        rng = np.random.default_rng(42)
        edge_field = rng.random(n_edges)

        # Import divergence function (will fail initially - this is TDD RED phase)
        from neurospatial.ops.calculus import divergence

        # Compute divergence
        div_field = divergence(env, edge_field)

        # Output should have shape (n_bins,)
        assert div_field.shape == (env.n_bins,)
        assert isinstance(div_field, np.ndarray)

    def test_divergence_gradient_is_laplacian(self):
        """Test that div(grad(f)) equals the negative diffusion generator applied to f for scalar field f."""
        # Create 1D chain environment
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create test field
        field = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

        from neurospatial.ops.calculus import divergence, gradient

        # Compute div(grad(f))
        grad_field = gradient(env, field)
        div_grad_field = divergence(env, grad_field)

        # Compute Laplacian directly
        L = _diffusion_generator(env).toarray()
        laplacian_field = L @ field

        # They should be equal (within numerical precision)
        np.testing.assert_allclose(
            div_grad_field, -laplacian_field, rtol=1e-10, atol=1e-10
        )

    def test_divergence_zero_edge_field(self):
        """Test that divergence of zero edge field is zero everywhere."""
        # Create 2D grid environment
        data = np.array([[i, j] for i in range(3) for j in range(3)])
        env = Environment.from_samples(data, bin_size=1.0)

        # Create zero edge field
        n_edges = len(env.connectivity.edges)
        edge_field = np.zeros(n_edges)

        from neurospatial.ops.calculus import divergence

        # Divergence should be all zeros
        div_field = divergence(env, edge_field)

        np.testing.assert_allclose(div_field, 0.0, atol=1e-10)

    def test_divergence_validation(self):
        """Test that divergence validates input edge field shape."""
        # Create environment
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        env = Environment.from_samples(data, bin_size=1.0)

        from neurospatial.ops.calculus import divergence

        # Wrong shape: too few elements
        edge_field_too_small = np.array([1.0, 2.0])

        # Should raise ValueError
        import pytest

        with pytest.raises(ValueError, match=r"edge_field.*shape"):
            divergence(env, edge_field_too_small)

        # Wrong shape: too many elements
        n_edges = len(env.connectivity.edges)
        rng = np.random.default_rng(42)
        edge_field_too_large = rng.random(n_edges + 5)

        with pytest.raises(ValueError, match=r"edge_field.*shape"):
            divergence(env, edge_field_too_large)
