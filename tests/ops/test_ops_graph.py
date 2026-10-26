"""Vectorized ``neighbor_reduce`` against a per-bin reference loop.

Behavioral tests of ``neighbor_reduce`` and ``graph_convolve`` are in
``tests/test_primitives.py``.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.graph import neighbor_reduce

# =============================================================================
# Fixtures for graph operation tests
# =============================================================================


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

        # An inactive cell between bins 0 and 1 leaves bin 0 with no neighbors.
        env = Environment.from_grid_mask(
            np.array([True, False, True, True]), grid_edges=(np.arange(5.0),)
        )
        isolated = 0
        assert env.connectivity.degree(isolated) == 0
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = neighbor_reduce(env, field, op="sum")
        assert np.isnan(result[isolated])
