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
import pytest

from neurospatial import Environment
from neurospatial.ops.graph import graph_convolve, neighbor_reduce

# =============================================================================
# Fixtures for graph operation tests
# =============================================================================


@pytest.fixture(scope="module")
def grid_3x3_env() -> Environment:
    """Create a simple 3x3 grid environment for graph operation tests.

    This is a deterministic 3x3 grid with diagonal connectivity enabled.
    Center bin (index 4) has 8 neighbors for testing neighbor operations.

    Returns
    -------
    Environment
        3x3 grid environment with 9 bins.
    """
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
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture(scope="module")
def grid_1x3_env() -> Environment:
    """Create a simple 1x3 linear grid environment.

    Returns
    -------
    Environment
        Linear grid with 3 bins.
    """
    positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture(scope="module")
def grid_5x5_env() -> Environment:
    """Create a 5x5 grid environment for integration tests.

    Returns
    -------
    Environment
        5x5 grid environment with 25 bins.
    """
    data = np.array([[i, j] for i in range(5) for j in range(5)], dtype=np.float64)
    return Environment.from_samples(data, bin_size=1.0)


# =============================================================================
# Import tests
# =============================================================================


class TestOpsGraph:
    """Test that ops.graph exports the correct public API."""

    def test_neighbor_reduce_import(self) -> None:
        """Test neighbor_reduce is importable from ops.graph."""
        import neurospatial.ops.graph as graph

        assert callable(neighbor_reduce)
        assert hasattr(graph, "neighbor_reduce")

    def test_graph_convolve_import(self) -> None:
        """Test graph_convolve is importable from ops.graph."""
        import neurospatial.ops.graph as graph

        assert callable(graph_convolve)
        assert hasattr(graph, "graph_convolve")

    def test_neighbor_reduce_basic(self, grid_3x3_env: Environment) -> None:
        """Test basic neighbor_reduce works from new import path."""
        env = grid_3x3_env

        # Create field with known values (bin index as value)
        field = np.arange(env.n_bins, dtype=np.float64)

        # Compute neighbor mean
        result = neighbor_reduce(env, field, op="mean", include_self=False)

        # Center bin (4) has 8 neighbors: [0, 1, 2, 3, 5, 6, 7, 8]
        # Mean = (0 + 1 + 2 + 3 + 5 + 6 + 7 + 8) / 8 = 32/8 = 4.0
        assert result.shape == (env.n_bins,), (
            f"Expected shape ({env.n_bins},), got {result.shape}"
        )
        assert np.isclose(result[4], 4.0), f"Expected 4.0, got {result[4]}"

    def test_graph_convolve_basic(self, grid_3x3_env: Environment) -> None:
        """Test basic graph_convolve works from new import path."""
        env = grid_3x3_env

        # Create field with spike at center
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[4] = 1.0  # Center bin

        # Box kernel: uniform weight within distance threshold
        def box_kernel(distances: np.ndarray) -> np.ndarray:
            """Return 1.0 for distances <= 1.5, 0.0 otherwise."""
            return np.where(distances <= 1.5, 1.0, 0.0)

        # Convolve with normalization
        result = graph_convolve(env, field, box_kernel, normalize=True)

        # Result should have correct shape
        assert result.shape == (env.n_bins,), (
            f"Expected shape ({env.n_bins},), got {result.shape}"
        )
        # Center bin should have non-zero value
        assert result[4] > 0, (
            f"Center bin should have positive value after convolution, got {result[4]}"
        )

    def test_neighbor_reduce_all_operations(self, grid_1x3_env: Environment) -> None:
        """Test all reduction operations work from new import path."""
        env = grid_1x3_env
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Test all operations
        for op in ["sum", "mean", "max", "min", "std"]:
            result = neighbor_reduce(env, field, op=op)
            assert result.shape == (env.n_bins,), (
                f"op={op}: Expected shape ({env.n_bins},), got {result.shape}"
            )
            assert not np.all(np.isnan(result)), f"op={op}: All values are NaN"


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

        expected_public = ["neighbor_reduce", "graph_convolve"]

        for name in expected_public:
            assert name in graph.__all__, f"{name} should be in __all__"


class TestOpsInit:
    """Test that ops/__init__.py exports graph functions."""

    def test_graph_exports_from_ops(self):
        """Test that graph functions are re-exported from ops."""
        from neurospatial import ops

        assert hasattr(ops, "neighbor_reduce")
        assert hasattr(ops, "graph_convolve")

    def test_graph_in_ops_all(self):
        """Test that graph functions are in ops.__all__."""
        from neurospatial import ops

        assert "neighbor_reduce" in ops.__all__
        assert "graph_convolve" in ops.__all__


class TestEnvironmentIntegration:
    """Test that Environment still works with graph operations."""

    def test_environment_with_neighbor_reduce(self, grid_5x5_env: Environment) -> None:
        """Test neighbor_reduce works with Environment from new location."""
        env = grid_5x5_env
        field = np.ones(env.n_bins) / env.n_bins

        # Should work without errors
        result = neighbor_reduce(env, field, op="mean")
        assert result.shape == (env.n_bins,), (
            f"Expected shape ({env.n_bins},), got {result.shape}"
        )

    def test_environment_with_graph_convolve(self, grid_5x5_env: Environment) -> None:
        """Test graph_convolve works with Environment from new location."""
        env = grid_5x5_env
        field = np.ones(env.n_bins) / env.n_bins

        # Gaussian kernel
        def gaussian_kernel(distances: np.ndarray) -> np.ndarray:
            return np.exp(-(distances**2) / 2.0)

        # Should work without errors
        result = graph_convolve(env, field, gaussian_kernel, normalize=True)
        assert result.shape == (env.n_bins,), (
            f"Expected shape ({env.n_bins},), got {result.shape}"
        )
