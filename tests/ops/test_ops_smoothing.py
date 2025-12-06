"""Tests for ops/smoothing.py - Diffusion kernel computation and application.

This module tests that smoothing functions are correctly exported from the
new ops.smoothing module location as part of the package reorganization.

The comprehensive tests for the actual kernel functionality remain in
tests/test_kernels.py. This file focuses on:
1. Import paths work correctly from new location
2. Public API is properly exported
3. Private helper remains private
"""

import networkx as nx
import numpy as np

from neurospatial import Environment
from neurospatial.ops.smoothing import apply_kernel, compute_diffusion_kernels


class TestOpsSmoothing:
    """Test that ops.smoothing exports the correct public API."""

    def test_compute_diffusion_kernels_import(self):
        """Test compute_diffusion_kernels is importable from ops.smoothing."""
        import neurospatial.ops.smoothing as smoothing

        assert callable(compute_diffusion_kernels)
        assert hasattr(smoothing, "compute_diffusion_kernels")

    def test_apply_kernel_import(self):
        """Test apply_kernel is importable from ops.smoothing."""
        import neurospatial.ops.smoothing as smoothing

        assert callable(apply_kernel)
        assert hasattr(smoothing, "apply_kernel")

    def test_compute_diffusion_kernels_basic(self):
        """Test basic kernel computation works from new import path."""
        # Create simple 1D chain graph
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(float(i),))
        for i in range(4):
            graph.add_edge(i, i + 1, distance=1.0)

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, mode="transition"
        )

        # Verify kernel shape and normalization
        assert kernel.shape == (5, 5), "Kernel should be n_bins x n_bins"
        assert kernel.dtype == np.float64, "Kernel should be float64"

        # Each column should sum to 1 in transition mode
        column_sums = kernel.sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, atol=1e-10)

    def test_apply_kernel_forward_mode(self):
        """Test apply_kernel forward mode works from new import path."""
        # Create simple kernel
        kernel = np.array(
            [[0.5, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]], dtype=np.float64
        )
        field = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        result = apply_kernel(field, kernel, mode="forward")

        # Should compute kernel @ field
        expected = kernel @ field
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_apply_kernel_adjoint_mode(self):
        """Test apply_kernel adjoint mode works from new import path."""
        kernel = np.array(
            [[0.5, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]], dtype=np.float64
        )
        field = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        result = apply_kernel(field, kernel, mode="adjoint")

        # Should compute kernel.T @ field
        expected = kernel.T @ field
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_apply_kernel_adjoint_with_bin_sizes(self):
        """Test mass-weighted adjoint works from new import path."""
        kernel = np.array(
            [[0.5, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]], dtype=np.float64
        )
        field = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        bin_sizes = np.array([1.0, 2.0, 1.0], dtype=np.float64)

        result = apply_kernel(field, kernel, mode="adjoint", bin_sizes=bin_sizes)

        # Should compute M^{-1} K.T M @ field
        m_field = bin_sizes * field
        kt_m_field = kernel.T @ m_field
        expected = kt_m_field / bin_sizes

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_compute_diffusion_kernels_density_mode(self):
        """Test density mode kernel computation from new import path."""
        graph = nx.Graph()
        for i in range(3):
            graph.add_node(i, pos=(float(i),))
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)

        bin_sizes = np.array([1.0, 1.5, 2.0])

        kernel = compute_diffusion_kernels(
            graph, bandwidth_sigma=1.0, bin_sizes=bin_sizes, mode="density"
        )

        # Weighted column sums should equal 1
        for j in range(3):
            weighted_sum = np.sum(kernel[:, j] * bin_sizes)
            np.testing.assert_allclose(weighted_sum, 1.0, atol=1e-10)


class TestOpsSmoothingAllExports:
    """Test that __all__ exports are correct."""

    def test_all_exports_exist(self):
        """Test that all items in __all__ are actually exported."""
        import neurospatial.ops.smoothing as smoothing

        for name in smoothing.__all__:
            assert hasattr(smoothing, name), f"{name} in __all__ but not exported"

    def test_all_public_functions_in_all(self):
        """Test that public functions are in __all__."""
        import neurospatial.ops.smoothing as smoothing

        expected_public = ["compute_diffusion_kernels", "apply_kernel"]

        for name in expected_public:
            assert name in smoothing.__all__, f"{name} should be in __all__"


class TestOpsSmoothingPrivateHelpers:
    """Test that private helpers remain private."""

    def test_assign_gaussian_weights_is_private(self):
        """Test that _assign_gaussian_weights_from_distance is not in __all__."""
        import neurospatial.ops.smoothing as smoothing

        # Should still be accessible but not in public API
        assert hasattr(smoothing, "_assign_gaussian_weights_from_distance")
        assert "_assign_gaussian_weights_from_distance" not in smoothing.__all__

    def test_large_kernel_threshold_is_private(self):
        """Test that _LARGE_KERNEL_THRESHOLD is accessible but private."""
        import neurospatial.ops.smoothing as smoothing

        # Should be accessible for testing but not in public API
        assert hasattr(smoothing, "_LARGE_KERNEL_THRESHOLD")
        assert smoothing._LARGE_KERNEL_THRESHOLD == 3000


class TestOpsInit:
    """Test that ops/__init__.py exports smoothing functions."""

    def test_smoothing_exports_from_ops(self):
        """Test that smoothing functions are re-exported from ops."""
        from neurospatial import ops

        assert hasattr(ops, "compute_diffusion_kernels")
        assert hasattr(ops, "apply_kernel")

    def test_smoothing_in_ops_all(self):
        """Test that smoothing functions are in ops.__all__."""
        from neurospatial import ops

        assert "compute_diffusion_kernels" in ops.__all__
        assert "apply_kernel" in ops.__all__


class TestEnvironmentIntegration:
    """Test that Environment still works with new smoothing location."""

    def test_environment_compute_kernel(self):
        """Test Environment.compute_kernel still works."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        kernel = env.compute_kernel(bandwidth=1.0, mode="transition")

        assert kernel.shape == (env.n_bins, env.n_bins)
        assert kernel.dtype == np.float64

    def test_environment_smooth(self):
        """Test Environment.smooth still works."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a uniform field
        field = np.ones(env.n_bins) / env.n_bins

        # Use mode='transition' to test mass conservation
        smoothed = env.smooth(field, bandwidth=1.0, mode="transition")

        assert smoothed.shape == (env.n_bins,)
        # For transition mode, total sum should be preserved
        np.testing.assert_allclose(smoothed.sum(), field.sum(), atol=1e-10)
