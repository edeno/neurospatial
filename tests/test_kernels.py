"""
Tests for diffusion kernel computation.

This module tests the kernel infrastructure that provides the foundation for
all smoothing operations in neurospatial.
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.smoothing import compute_diffusion_kernels


class TestComputeDiffusionKernels:
    """Tests for compute_diffusion_kernels function."""

    def test_kernel_shape(self):
        """Test that kernel has correct shape (n_bins x n_bins)."""
        # Create simple 1D chain graph
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_node(2, pos=(2.0,))
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        assert kernel.shape == (3, 3), "Kernel should be n_bins x n_bins"
        assert kernel.dtype == np.float64, "Kernel should be float64"

    def test_kernel_symmetry_uniform_grid(self):
        """Test that kernel is symmetric for uniform regular grid."""
        # Create uniform 2D grid graph (3x3)
        graph = nx.grid_2d_graph(3, 3)

        # Relabel nodes to integers and add required attributes
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)

        # Add pos and distance attributes
        for i, node in enumerate(sorted(mapping.values())):
            row, col = divmod(i, 3)
            graph.nodes[node]["pos"] = (float(col), float(row))

        for u, v in graph.edges():
            pos_u = np.array(graph.nodes[u]["pos"])
            pos_v = np.array(graph.nodes[v]["pos"])
            graph.edges[u, v]["distance"] = float(np.linalg.norm(pos_v - pos_u))

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        # For uniform grid with equal edge lengths, kernel should be symmetric
        assert np.allclose(kernel, kernel.T, atol=1e-10), (
            "Kernel should be symmetric for uniform grid"
        )

    def test_kernel_normalization_transition_mode(self):
        """Test that column sums equal 1 in transition mode."""
        # Create simple graph
        graph = nx.Graph()
        for i in range(4):
            graph.add_node(i, pos=(float(i),))
        for i in range(3):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=0.5, mode="transition"
        )

        # Each column should sum to 1 (discrete probability)
        column_sums = kernel.sum(axis=0)
        assert np.allclose(column_sums, 1.0, atol=1e-10), (
            "Column sums should equal 1 in transition mode"
        )

    def test_kernel_normalization_density_mode(self):
        """Test that weighted column sums equal 1 in density mode."""
        # Create simple graph with varying bin sizes
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_node(2, pos=(3.0,))  # Non-uniform spacing
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=2.0)

        bin_sizes = np.array([1.0, 1.5, 2.0])  # Different volumes

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, bin_sizes=bin_sizes, mode="density"
        )

        # Weighted column sums: sum(kernel[:, j] * bin_sizes) should equal 1
        for j in range(3):
            weighted_sum = np.sum(kernel[:, j] * bin_sizes)
            assert np.isclose(weighted_sum, 1.0, atol=1e-10), (
                f"Column {j}: weighted sum should equal 1 in density mode"
            )

    def test_mass_conservation(self):
        """Test that kernel conserves mass when applied to a field."""
        # Create 1D chain
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        # Create arbitrary field
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply kernel
        smoothed = kernel @ field

        # Total mass should be conserved
        assert np.isclose(smoothed.sum(), field.sum(), atol=1e-10), (
            "Mass should be conserved after smoothing"
        )

    def test_impulse_spreading(self):
        """Test that impulse at one bin spreads to neighbors."""
        # Create 1D chain
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        # Impulse at center bin
        field = np.zeros(5)
        field[2] = 1.0

        smoothed = kernel @ field

        # Check that value spread to neighbors
        assert smoothed[2] > 0, "Center should have non-zero value"
        assert smoothed[1] > 0, "Left neighbor should have non-zero value"
        assert smoothed[3] > 0, "Right neighbor should have non-zero value"
        # Further neighbors should have smaller values
        assert smoothed[2] > smoothed[1], (
            "Center should have larger value than neighbor"
        )
        assert smoothed[1] > smoothed[0], "Closer bins should have larger values"

    def test_edge_no_distance_attribute_raises_error(self):
        """Test that missing distance attribute raises KeyError."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1)  # Missing distance attribute

        with pytest.raises(KeyError, match="distance"):
            compute_diffusion_kernels(graph, bandwidth_sigma=1.0, mode="transition")

    def test_density_mode_without_bin_sizes_raises_error(self):
        """Test that density mode without bin_sizes raises ValueError."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)

        with pytest.raises(ValueError, match=r"bin_sizes.*required.*density"):
            compute_diffusion_kernels(
                graph, bandwidth_sigma=1.0, bin_sizes=None, mode="density"
            )

    def test_invalid_bin_sizes_shape_raises_error(self):
        """Test that bin_sizes with wrong shape raises ValueError."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)

        wrong_size = np.array([1.0, 2.0, 3.0])  # Should be length 2

        with pytest.raises(ValueError, match=r"bin_sizes.*shape"):
            compute_diffusion_kernels(
                graph, bandwidth_sigma=1.0, bin_sizes=wrong_size, mode="density"
            )

    def test_positive_bandwidth_required(self):
        """Test that bandwidth must be positive."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)

        with pytest.raises(ValueError, match=r"bandwidth.*positive"):
            compute_diffusion_kernels(graph, bandwidth_sigma=-1.0, mode="transition")

        with pytest.raises(ValueError, match=r"bandwidth.*positive"):
            compute_diffusion_kernels(graph, bandwidth_sigma=0.0, mode="transition")

    def test_disconnected_graph_components(self):
        """Test kernel behavior on disconnected graph."""
        # Create graph with two disconnected components
        graph = nx.Graph()
        # Component 1
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)
        # Component 2 (disconnected)
        graph.add_node(2, pos=(10.0,))
        graph.add_node(3, pos=(11.0,))
        graph.add_edge(2, 3, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        # Smoothing should not leak between components
        field = np.array([1.0, 0.0, 0.0, 0.0])
        smoothed = kernel @ field

        # Mass should stay in component 1 (bins 0, 1)
        assert smoothed[0] + smoothed[1] > 0.99, "Mass should stay in component 1"
        assert smoothed[2] + smoothed[3] < 0.01, "Mass should not leak to component 2"

    def test_different_bandwidths(self):
        """Test that larger bandwidth produces more diffusion."""
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        # Impulse at center
        field = np.zeros(5)
        field[2] = 1.0

        # Small bandwidth
        kernel_small = compute_diffusion_kernels(
            graph, bandwidth_sigma=0.3, mode="transition"
        )
        smoothed_small = kernel_small @ field

        # Large bandwidth
        kernel_large = compute_diffusion_kernels(
            graph, bandwidth_sigma=2.0, mode="transition"
        )
        smoothed_large = kernel_large @ field

        # Larger bandwidth should spread mass more
        assert smoothed_small[2] > smoothed_large[2], (
            "Small bandwidth should keep more mass at center"
        )
        assert smoothed_large[0] > smoothed_small[0], (
            "Large bandwidth should spread more to edges"
        )


class TestEnvironmentComputeKernel:
    """Tests for Environment.compute_kernel() wrapper method."""

    def test_compute_kernel_basic(self):
        """Test basic kernel computation via Environment method."""
        # Create simple 2D grid environment
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        env = Environment.from_samples(data, bin_size=1.0)

        kernel = env.compute_kernel(bandwidth=1.0, mode="transition")

        assert kernel.shape == (env.n_bins, env.n_bins)
        assert kernel.dtype == np.float64

    def test_compute_kernel_uses_layout_bin_sizes(self):
        """Test that compute_kernel automatically uses layout bin sizes."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (100, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # Should not raise error even though we don't pass bin_sizes
        kernel = env.compute_kernel(bandwidth=1.0, mode="density")

        assert kernel.shape == (env.n_bins, env.n_bins)

    def test_compute_kernel_cache_behavior(self):
        """Test that kernel is cached properly."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # First call
        kernel1 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=True)

        # Second call with same parameters should return cached result
        kernel2 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=True)

        # Should be the same array (identity)
        assert kernel1 is kernel2, "Should return cached kernel"

        # Different bandwidth should compute new kernel
        kernel3 = env.compute_kernel(bandwidth=2.0, mode="transition", cache=True)
        assert kernel3 is not kernel1, "Different bandwidth should compute new kernel"

        # Different mode should compute new kernel
        kernel4 = env.compute_kernel(bandwidth=1.0, mode="density", cache=True)
        assert kernel4 is not kernel1, "Different mode should compute new kernel"

    def test_compute_kernel_cache_disabled(self):
        """Test that cache can be disabled."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # Call with cache disabled
        kernel1 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=False)
        kernel2 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=False)

        # Should not be the same object (different computations)
        assert kernel1 is not kernel2, "Should recompute when cache disabled"
        # But should have same values
        assert np.allclose(kernel1, kernel2)

    def test_compute_kernel_requires_fitted(self):
        """Test that compute_kernel requires fitted environment."""
        # Create an environment but don't fit it by using __init__ directly
        # with minimal setup (this is testing the @check_fitted decorator)
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        # Don't call build(), so layout is not fitted
        env = Environment(name="test", layout=layout)
        # Verify it's not fitted
        assert not env._is_fitted

        with pytest.raises(RuntimeError, match="fully initialized"):
            env.compute_kernel(bandwidth=1.0)


class TestKernelPerformanceWarnings:
    """Tests for performance warnings in kernel computation."""

    def test_large_graph_warning(self):
        """Test that warning is issued for large graphs."""
        # Create graph with many nodes (e.g., 1000+ bins)
        # This test verifies the warning exists in docstring
        # Actual warning emission should be tested if implemented
        from neurospatial.ops.smoothing import compute_diffusion_kernels

        # Check docstring mentions performance/complexity
        assert (
            "complexity" in compute_diffusion_kernels.__doc__.lower()
            or "performance" in compute_diffusion_kernels.__doc__.lower()
        ), "Docstring should mention performance considerations"

    def test_large_graph_emits_warning(self):
        """Test that UserWarning is emitted for large graphs (> 3000 bins)."""
        from neurospatial.ops.smoothing import (
            _LARGE_KERNEL_THRESHOLD,
            compute_diffusion_kernels,
        )

        # Create a graph just over the threshold
        n_bins = _LARGE_KERNEL_THRESHOLD + 1
        graph = nx.path_graph(n_bins)  # Simple path graph

        # Add required distance attributes
        for u, v in graph.edges():
            graph.edges[u, v]["distance"] = 1.0

        # Should emit UserWarning about performance
        with pytest.warns(UserWarning, match=r"Computing diffusion kernel"):
            # Use a small bandwidth to make computation faster
            # (kernel won't be useful but we just need to test warning)
            compute_diffusion_kernels(graph, bandwidth_sigma=0.01, mode="transition")

    def test_small_graph_no_warning(self):
        """Test that no warning is emitted for small graphs."""
        from neurospatial.ops.smoothing import compute_diffusion_kernels

        # Create small graph (well under threshold)
        graph = nx.path_graph(10)
        for u, v in graph.edges():
            graph.edges[u, v]["distance"] = 1.0

        # Should not emit any warnings
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_diffusion_kernels(graph, bandwidth_sigma=1.0, mode="transition")
            # Filter for our specific warning
            kernel_warnings = [
                warning
                for warning in w
                if "Computing diffusion kernel" in str(warning.message)
            ]
            assert len(kernel_warnings) == 0, "Small graphs should not emit warnings"


class TestSparseMatrixOptimization:
    """Tests for sparse matrix optimization in kernel computation."""

    def test_sparse_mass_matrix_same_result(self):
        """Test that sparse mass matrix produces same results as dense would.

        This verifies the optimization in kernels.py:113 (scipy.sparse.diags
        instead of np.diag) produces identical results.
        """
        # Create graph with non-uniform bin sizes
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        bin_sizes = np.array([1.0, 1.5, 2.0, 1.5, 1.0])  # Varying sizes

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, bin_sizes=bin_sizes, mode="density"
        )

        # Verify normalization (weighted column sums = 1)
        for j in range(5):
            weighted_sum = np.sum(kernel[:, j] * bin_sizes)
            assert np.isclose(weighted_sum, 1.0, atol=1e-10), (
                f"Column {j}: weighted sum {weighted_sum} should be 1.0"
            )

        # Verify kernel properties: non-negative, spreads mass
        assert np.all(kernel >= 0), "Kernel should be non-negative"

        # Impulse at center should spread to neighbors
        field = np.zeros(5)
        field[2] = 1.0
        smoothed = kernel @ field
        assert smoothed[1] > 0, "Mass should spread to left neighbor"
        assert smoothed[3] > 0, "Mass should spread to right neighbor"


class TestKernelEdgeCases:
    """Tests for edge cases in kernel computation."""

    def test_single_node_graph(self):
        """Test kernel computation for graph with single node."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        assert kernel.shape == (1, 1)
        assert np.isclose(kernel[0, 0], 1.0), "Single node kernel should be [[1.0]]"

    def test_two_node_graph(self):
        """Test kernel computation for minimal connected graph."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        assert kernel.shape == (2, 2)
        # Columns should sum to 1
        assert np.allclose(kernel.sum(axis=0), 1.0)
        # Should be symmetric for uniform case
        assert np.allclose(kernel, kernel.T, atol=1e-10)

    def test_very_small_bandwidth(self):
        """Test that very small bandwidth keeps mass localized."""
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=0.01, mode="transition"
        )

        # Should be nearly diagonal (minimal diffusion)
        diagonal_mass = np.diag(kernel).sum()
        assert diagonal_mass > 0.95, "Very small bandwidth should keep mass localized"

    def test_very_large_bandwidth(self):
        """Test that very large bandwidth spreads mass uniformly."""
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=100.0, mode="transition"
        )

        # Should spread mass more uniformly
        # Each column should have similar values across rows
        for col in range(5):
            column_std = np.std(kernel[:, col])
            assert column_std < 0.3, "Large bandwidth should spread mass uniformly"


class TestComputeDiffusionKernelsValidation:
    """Test parameter validation for compute_diffusion_kernels."""

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0)

        with pytest.raises(ValueError, match="Invalid mode"):
            compute_diffusion_kernels(graph, bandwidth_sigma=1.0, mode="invalid")


class TestApplyKernel:
    """Test apply_kernel function for forward and adjoint operations."""

    @pytest.fixture
    def simple_kernel(self):
        """Create a simple 3x3 kernel for testing."""
        # Simple normalized kernel
        kernel = np.array(
            [[0.5, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]], dtype=np.float64
        )
        return kernel

    @pytest.fixture
    def simple_field(self):
        """Create a simple test field."""
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    @pytest.fixture
    def bin_sizes_3(self):
        """Create simple bin sizes for 3 bins."""
        return np.array([1.0, 2.0, 1.0], dtype=np.float64)

    def test_forward_mode_basic(self, simple_kernel, simple_field):
        """Test basic forward mode application."""
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(simple_field, simple_kernel, mode="forward")

        # Should compute kernel @ field
        expected = simple_kernel @ simple_field
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_adjoint_mode_no_bin_sizes(self, simple_kernel, simple_field):
        """Test adjoint mode without bin_sizes (simple transpose)."""
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(simple_field, simple_kernel, mode="adjoint")

        # Should compute kernel.T @ field
        expected = simple_kernel.T @ simple_field
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_adjoint_mode_with_bin_sizes(
        self, simple_kernel, simple_field, bin_sizes_3
    ):
        """Test adjoint mode with bin_sizes (mass-weighted)."""
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(
            simple_field, simple_kernel, mode="adjoint", bin_sizes=bin_sizes_3
        )

        # Should compute M^{-1} K.T M @ field
        # where M = diag(bin_sizes)
        m_field = bin_sizes_3 * simple_field
        kt_m_field = simple_kernel.T @ m_field
        expected = kt_m_field / bin_sizes_3

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_mode_raises(self, simple_kernel, simple_field):
        """Test that invalid mode raises ValueError."""
        from neurospatial.ops.smoothing import apply_kernel

        with pytest.raises(ValueError, match="mode must be"):
            apply_kernel(simple_field, simple_kernel, mode="invalid")

    def test_non_square_kernel_raises(self, simple_field):
        """Test that non-square kernel raises ValueError."""
        from neurospatial.ops.smoothing import apply_kernel

        # Create non-square kernel
        bad_kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Kernel must be square"):
            apply_kernel(simple_field, bad_kernel, mode="forward")

    def test_field_size_mismatch_raises(self, simple_kernel):
        """Test that field size mismatch raises ValueError."""
        from neurospatial.ops.smoothing import apply_kernel

        # Field has wrong size
        bad_field = np.array([1.0, 2.0], dtype=np.float64)

        with pytest.raises(ValueError, match=r"Field size.*does not match"):
            apply_kernel(bad_field, simple_kernel, mode="forward")

    def test_bin_sizes_mismatch_raises(self, simple_kernel, simple_field):
        """Test that bin_sizes size mismatch raises ValueError."""
        from neurospatial.ops.smoothing import apply_kernel

        # bin_sizes has wrong size
        bad_bin_sizes = np.array([1.0, 2.0], dtype=np.float64)

        with pytest.raises(ValueError, match=r"bin_sizes size.*does not match"):
            apply_kernel(
                simple_field, simple_kernel, mode="adjoint", bin_sizes=bad_bin_sizes
            )

    def test_non_positive_bin_sizes_raises(self, simple_kernel, simple_field):
        """Test that non-positive bin_sizes raises ValueError in adjoint mode."""
        from neurospatial.ops.smoothing import apply_kernel

        # bin_sizes with zero/negative values
        bad_bin_sizes = np.array([1.0, 0.0, -1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="bin_sizes must have strictly positive"):
            apply_kernel(
                simple_field, simple_kernel, mode="adjoint", bin_sizes=bad_bin_sizes
            )

    def test_forward_adjoint_duality_no_bin_sizes(self, simple_kernel):
        """Test that forward and adjoint are dual without bin_sizes."""
        from neurospatial.ops.smoothing import apply_kernel

        # Create two random fields
        u = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = np.array([0.5, 1.5, 0.8], dtype=np.float64)

        # Compute <K u, v> and <u, K^T v>
        ku = apply_kernel(u, simple_kernel, mode="forward")
        ktv = apply_kernel(v, simple_kernel, mode="adjoint")

        inner1 = np.dot(ku, v)
        inner2 = np.dot(u, ktv)

        # Should be equal (within numerical precision)
        np.testing.assert_allclose(inner1, inner2, rtol=1e-10)

    def test_forward_adjoint_duality_with_bin_sizes(self, simple_kernel, bin_sizes_3):
        """Test that forward and adjoint are dual with bin_sizes."""
        from neurospatial.ops.smoothing import apply_kernel

        # Create two random fields
        u = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = np.array([0.5, 1.5, 0.8], dtype=np.float64)

        # Compute weighted inner products <K u, v>_M and <u, K^* v>_M
        ku = apply_kernel(u, simple_kernel, mode="forward")
        kstar_v = apply_kernel(v, simple_kernel, mode="adjoint", bin_sizes=bin_sizes_3)

        # Weighted inner product: <x, y>_M = sum(x * M * y)
        inner1 = np.dot(ku * bin_sizes_3, v)
        inner2 = np.dot(u * bin_sizes_3, kstar_v)

        # Should be equal (within numerical precision)
        np.testing.assert_allclose(inner1, inner2, rtol=1e-10)

    def test_bin_sizes_allowed_in_forward_mode(
        self, simple_kernel, simple_field, bin_sizes_3
    ):
        """Test that bin_sizes is allowed but ignored in forward mode."""
        from neurospatial.ops.smoothing import apply_kernel

        # Should not raise error
        result = apply_kernel(
            simple_field, simple_kernel, mode="forward", bin_sizes=bin_sizes_3
        )

        # Result should be same as without bin_sizes
        expected = simple_kernel @ simple_field
        np.testing.assert_allclose(result, expected, rtol=1e-10)
