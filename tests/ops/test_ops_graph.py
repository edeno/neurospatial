"""Tests for ops/graph.py - Graph convolution and neighborhood reduction.

This module tests that graph operations are correctly exported from the
new ops.graph module location as part of the package reorganization.

The comprehensive tests for the actual functionality remain in
tests/test_primitives.py. This file focuses on:
1. Import paths work correctly from new location
2. Public API is properly exported
3. Integration with Environment still works
"""

import numpy as np

from neurospatial import Environment
from neurospatial.ops.graph import convolve, neighbor_reduce


class TestOpsGraph:
    """Test that ops.graph exports the correct public API."""

    def test_neighbor_reduce_import(self):
        """Test neighbor_reduce is importable from ops.graph."""
        import neurospatial.ops.graph as graph

        assert callable(neighbor_reduce)
        assert hasattr(graph, "neighbor_reduce")

    def test_convolve_import(self):
        """Test convolve is importable from ops.graph."""
        import neurospatial.ops.graph as graph

        assert callable(convolve)
        assert hasattr(graph, "convolve")

    def test_neighbor_reduce_basic(self):
        """Test basic neighbor_reduce works from new import path."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create field with known values (bin index as value)
        field = np.arange(env.n_bins, dtype=np.float64)

        # Compute neighbor mean
        result = neighbor_reduce(env, field, op="mean", include_self=False)

        # Center bin (4) has 8 neighbors: [0, 1, 2, 3, 5, 6, 7, 8]
        # Mean = (0 + 1 + 2 + 3 + 5 + 6 + 7 + 8) / 8 = 32/8 = 4.0
        assert result.shape == (env.n_bins,)
        assert np.isclose(result[4], 4.0), f"Expected 4.0, got {result[4]}"

    def test_convolve_basic(self):
        """Test basic convolve works from new import path."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create field with spike at center
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[4] = 1.0  # Center bin

        # Box kernel: uniform weight within distance threshold
        def box_kernel(distances: np.ndarray) -> np.ndarray:
            """Return 1.0 for distances <= 1.5, 0.0 otherwise."""
            return np.where(distances <= 1.5, 1.0, 0.0)

        # Convolve with normalization
        result = convolve(env, field, box_kernel, normalize=True)

        # Result should have correct shape
        assert result.shape == (env.n_bins,)
        # Center bin should have non-zero value
        assert result[4] > 0

    def test_neighbor_reduce_all_operations(self):
        """Test all reduction operations work from new import path."""
        # Create simple grid
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Test all operations
        for op in ["sum", "mean", "max", "min", "std"]:
            result = neighbor_reduce(env, field, op=op)
            assert result.shape == (env.n_bins,)
            assert not np.all(np.isnan(result))


class TestOpsGraphAllExports:
    """Test that __all__ exports are correct."""

    def test_all_exports_exist(self):
        """Test that all items in __all__ are actually exported."""
        import neurospatial.ops.graph as graph

        for name in graph.__all__:
            assert hasattr(graph, name), f"{name} in __all__ but not exported"

    def test_all_public_functions_in_all(self):
        """Test that public functions are in __all__."""
        import neurospatial.ops.graph as graph

        expected_public = ["neighbor_reduce", "convolve"]

        for name in expected_public:
            assert name in graph.__all__, f"{name} should be in __all__"


class TestOpsInit:
    """Test that ops/__init__.py exports graph functions."""

    def test_graph_exports_from_ops(self):
        """Test that graph functions are re-exported from ops."""
        from neurospatial import ops

        assert hasattr(ops, "neighbor_reduce")
        assert hasattr(ops, "convolve")

    def test_graph_in_ops_all(self):
        """Test that graph functions are in ops.__all__."""
        from neurospatial import ops

        assert "neighbor_reduce" in ops.__all__
        assert "convolve" in ops.__all__


class TestEnvironmentIntegration:
    """Test that Environment still works with graph operations."""

    def test_environment_with_neighbor_reduce(self):
        """Test neighbor_reduce works with Environment from new location."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins) / env.n_bins

        # Should work without errors
        result = neighbor_reduce(env, field, op="mean")
        assert result.shape == (env.n_bins,)

    def test_environment_with_convolve(self):
        """Test convolve works with Environment from new location."""
        data = np.array([[i, j] for i in range(5) for j in range(5)])
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins) / env.n_bins

        # Gaussian kernel
        def gaussian_kernel(distances: np.ndarray) -> np.ndarray:
            return np.exp(-(distances**2) / 2.0)

        # Should work without errors
        result = convolve(env, field, gaussian_kernel, normalize=True)
        assert result.shape == (env.n_bins,)
