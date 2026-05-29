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
    """Functional smoke tests for ``neurospatial.ops.graph``."""

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


def _reference_neighbor_reduce(env, field, *, op, weights=None, include_self=False):
    """Reference (pure-loop) implementation of neighbor_reduce for equivalence checks.

    Mirrors the original NetworkX neighbor-iteration algorithm so the vectorized
    sum/mean paths can be verified against it.
    """
    result = np.full(env.n_bins, np.nan, dtype=np.float64)
    for bin_id in range(env.n_bins):
        neighbors = list(env.connectivity.neighbors(bin_id))
        if include_self:
            neighbors = [bin_id, *neighbors]
        if len(neighbors) == 0:
            continue
        values = field[neighbors]
        if weights is None:
            if op == "sum":
                result[bin_id] = np.sum(values)
            elif op == "mean":
                result[bin_id] = np.mean(values)
            elif op == "max":
                result[bin_id] = np.max(values)
            elif op == "min":
                result[bin_id] = np.min(values)
            elif op == "std":
                result[bin_id] = np.std(values)
        else:
            w = weights[neighbors]
            if op == "sum":
                result[bin_id] = np.sum(values * w)
            elif op == "mean":
                wsum = np.sum(w)
                result[bin_id] = np.sum(values * w) / wsum if wsum > 0 else np.nan
    return result


class TestNeighborReduceEquivalence:
    """Vectorized neighbor_reduce must match the reference loop for all modes."""

    @pytest.mark.parametrize("op", ["sum", "mean", "max", "min", "std"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_unweighted_matches_reference_loop(
        self, grid_5x5_env: Environment, op: str, include_self: bool
    ) -> None:
        """Unweighted reductions match the pure-loop reference (incl. self toggle)."""
        env = grid_5x5_env
        rng = np.random.default_rng(0)
        field = rng.standard_normal(env.n_bins).astype(np.float64)

        expected = _reference_neighbor_reduce(
            env, field, op=op, include_self=include_self
        )
        actual = neighbor_reduce(env, field, op=op, include_self=include_self)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("op", ["sum", "mean"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_weighted_matches_reference_loop(
        self, grid_5x5_env: Environment, op: str, include_self: bool
    ) -> None:
        """Weighted sum/mean match the pure-loop reference."""
        env = grid_5x5_env
        rng = np.random.default_rng(1)
        field = rng.standard_normal(env.n_bins).astype(np.float64)
        weights = rng.uniform(0.1, 2.0, env.n_bins).astype(np.float64)

        expected = _reference_neighbor_reduce(
            env, field, op=op, weights=weights, include_self=include_self
        )
        actual = neighbor_reduce(
            env, field, op=op, weights=weights, include_self=include_self
        )

        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_isolated_node_returns_nan(self) -> None:
        """A bin with no neighbors yields NaN for sum/mean (vectorized path)."""
        import networkx as nx

        # Build a tiny env-like object with one isolated node via a real env,
        # then drop edges from one node to make it isolated.
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)
        # Isolate node 0 by removing its edges on a copy of connectivity.
        g = env.connectivity
        isolated = next((n for n in g.nodes if g.degree(n) == 0), None)
        if isolated is None:
            # Force isolation of node 0.
            edges = list(g.edges(0))
            g.remove_edges_from(edges)
            isolated = 0
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = neighbor_reduce(env, field, op="sum")
        assert np.isnan(result[isolated])
        # Restore graph (it is module-scoped only if fixture; here it's local).
        assert isinstance(g, nx.Graph)


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
