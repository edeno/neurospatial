"""Tests for encoding/_core_jax.py - JAX core array operations.

This module tests the JAX implementation of core array operations used
by encoding functions. Tests verify:
1. Functions can import and are available
2. Basic functionality works with JAX arrays
3. Results are numerically equivalent to NumPy implementations
4. Functions handle edge cases properly
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Try to import JAX - skip tests if not available
jax = pytest.importorskip("jax", reason="JAX not installed")

import jax.numpy as jnp  # noqa: E402

from neurospatial.encoding._backend import is_jax_available  # noqa: E402

# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(
    not is_jax_available(),
    reason="JAX is not available on this platform",
)


def is_jax_array(obj):
    """Check if an object is a JAX array."""
    # JAX arrays have a 'devices' method/attribute
    return hasattr(obj, "devices") or type(obj).__module__.startswith("jax")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def spike_counts_1d() -> jnp.ndarray:
    """Single neuron spike counts in JAX array.

    Returns
    -------
    jax.Array, shape (100,)
        Spike counts with a peak in the middle.
    """
    counts = np.zeros(100, dtype=np.float64)
    counts[50] = 10.0  # Peak at center
    counts[40:60] = np.linspace(0, 10, 20)  # Ramp up to peak
    return jnp.array(counts)


@pytest.fixture
def spike_counts_2d() -> jnp.ndarray:
    """Multiple neurons spike counts in JAX array.

    Returns
    -------
    jax.Array, shape (3, 100)
        Spike counts for 3 neurons with different patterns.
    """
    counts = np.zeros((3, 100), dtype=np.float64)
    # Neuron 0: center peak
    counts[0, 50] = 10.0
    # Neuron 1: early peak
    counts[1, 20] = 8.0
    # Neuron 2: late peak
    counts[2, 80] = 6.0
    return jnp.array(counts)


@pytest.fixture
def occupancy() -> jnp.ndarray:
    """Occupancy array in JAX.

    Returns
    -------
    jax.Array, shape (100,)
        Occupancy with uniform values of 1.0 second.
    """
    return jnp.ones(100, dtype=jnp.float64)


@pytest.fixture
def non_uniform_occupancy() -> jnp.ndarray:
    """Non-uniform occupancy array in JAX.

    Returns
    -------
    jax.Array, shape (100,)
        Occupancy with more time in center.
    """
    occ = np.ones(100, dtype=np.float64)
    occ[40:60] = 5.0  # More time in center
    occ[0] = 0.0  # Zero occupancy at start
    return jnp.array(occ)


def _one_neuron(spike_counts, occupancy, **kwargs):
    """One neuron's rate, shape ``(n_bins,)``, as a population of one."""
    from neurospatial.encoding._core_jax import compute_firing_rates_batch

    return compute_firing_rates_batch(spike_counts[None], occupancy, **kwargs)[0]


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Tests for successful imports."""

    def test_import_compute_firing_rates_batch(self):
        """compute_firing_rates_batch should be importable."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        assert callable(compute_firing_rates_batch)


# =============================================================================
# Test compute_firing_rates_batch on one neuron
# =============================================================================


class TestComputeFiringRateOneNeuron:
    """compute_firing_rates_batch on a population of one."""

    def test_returns_jax_array(self, spike_counts_1d, occupancy):
        """Result should be a JAX array."""
        result = _one_neuron(spike_counts_1d, occupancy)
        assert is_jax_array(result)

    def test_output_shape_matches_input(self, spike_counts_1d, occupancy):
        """Output shape should match input shape."""
        result = _one_neuron(spike_counts_1d, occupancy)
        assert result.shape == spike_counts_1d.shape

    def test_correct_rate_computation(self, spike_counts_1d, occupancy):
        """Rate should be spike_counts / occupancy."""
        result = _one_neuron(spike_counts_1d, occupancy)
        expected = spike_counts_1d / occupancy
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_zero_occupancy_gives_nan(self, spike_counts_1d, non_uniform_occupancy):
        """Zero occupancy should result in NaN."""
        result_np = np.asarray(_one_neuron(spike_counts_1d, non_uniform_occupancy))
        # Bin 0 has zero occupancy
        assert np.isnan(result_np[0])

    def test_min_occupancy_threshold(self, spike_counts_1d, non_uniform_occupancy):
        """Bins below min_occupancy should be NaN."""
        result = _one_neuron(spike_counts_1d, non_uniform_occupancy, min_occupancy=2.0)
        result_np = np.asarray(result)
        # Bins with occupancy < 2.0 should be NaN
        occupancy_np = np.asarray(non_uniform_occupancy)
        low_occ_mask = occupancy_np < 2.0
        assert np.all(np.isnan(result_np[low_occ_mask]))

    def test_non_negative_rates(self, spike_counts_1d, occupancy):
        """Firing rates should be non-negative."""
        result_np = np.asarray(_one_neuron(spike_counts_1d, occupancy))
        assert np.all(result_np[~np.isnan(result_np)] >= 0)


# =============================================================================
# Test compute_firing_rates_batch
# =============================================================================


class TestComputeFiringRatesBatch:
    """Tests for compute_firing_rates_batch function."""

    def test_returns_jax_array(self, spike_counts_2d, occupancy):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        result = compute_firing_rates_batch(spike_counts_2d, occupancy)
        assert is_jax_array(result)

    def test_output_shape(self, spike_counts_2d, occupancy):
        """Output shape should be (n_neurons, n_bins)."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        result = compute_firing_rates_batch(spike_counts_2d, occupancy)
        assert result.shape == spike_counts_2d.shape

    def test_correct_rate_computation(self, spike_counts_2d, occupancy):
        """Rate should be spike_counts / occupancy for each neuron."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        result = compute_firing_rates_batch(spike_counts_2d, occupancy)
        expected = spike_counts_2d / occupancy
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_each_row_matches_the_neuron_alone(self, spike_counts_2d, occupancy):
        """A neuron's rate does not depend on the other neurons in the batch."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        batch_result = compute_firing_rates_batch(spike_counts_2d, occupancy)

        for i in range(spike_counts_2d.shape[0]):
            single_result = _one_neuron(spike_counts_2d[i], occupancy)
            assert_allclose(
                np.asarray(batch_result[i]),
                np.asarray(single_result),
                rtol=1e-10,
                equal_nan=True,
            )


# =============================================================================
# Test Numerical Equivalence with NumPy
# =============================================================================


class TestNumericalEquivalenceWithNumPy:
    """Tests verifying JAX and NumPy implementations produce same results."""

    def test_firing_rates_batch_matches(self, spike_counts_2d, occupancy):
        """JAX and NumPy compute_firing_rates_batch should match."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rates_batch as jax_impl,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rates_batch as numpy_impl,
        )

        jax_result = jax_impl(spike_counts_2d, occupancy)
        numpy_result = numpy_impl(np.asarray(spike_counts_2d), np.asarray(occupancy))

        assert_allclose(
            np.asarray(jax_result),
            numpy_result,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_firing_rates_with_min_occupancy_match(
        self, spike_counts_1d, non_uniform_occupancy
    ):
        """JAX and NumPy should match with min_occupancy threshold."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rates_batch as jax_impl,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rates_batch as numpy_impl,
        )

        jax_result = jax_impl(
            spike_counts_1d[None], non_uniform_occupancy, min_occupancy=2.0
        )
        numpy_result = numpy_impl(
            np.asarray(spike_counts_1d)[None],
            np.asarray(non_uniform_occupancy),
            min_occupancy=2.0,
        )

        assert_allclose(
            np.asarray(jax_result),
            numpy_result,
            rtol=1e-10,
            equal_nan=True,
        )


# =============================================================================
# Test JAX-Specific Features
# =============================================================================


class TestJAXSpecificFeatures:
    """Tests for JAX-specific features like JIT compilation."""

    def test_jittable_firing_rates_batch(self, spike_counts_2d, occupancy):
        """compute_firing_rates_batch should be JIT-compilable."""
        import jax

        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        jitted_fn = jax.jit(compute_firing_rates_batch)
        result = jitted_fn(spike_counts_2d, occupancy)

        # Should produce valid result
        assert result.shape == spike_counts_2d.shape
        assert is_jax_array(result)

    def test_vmap_over_neurons(self, spike_counts_2d, occupancy):
        """The rate works with vmap for vectorization."""
        import jax

        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        # vmap over neurons (first axis)
        vmapped_fn = jax.vmap(_one_neuron, in_axes=(0, None))
        result = vmapped_fn(spike_counts_2d, occupancy)

        # Should produce the batch function's result
        assert_allclose(
            np.asarray(result),
            np.asarray(compute_firing_rates_batch(spike_counts_2d, occupancy)),
            rtol=1e-10,
        )


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_zero_spike_counts(self, occupancy):
        """All zero spike counts should return all zeros."""
        zero_counts = jnp.zeros(100, dtype=jnp.float64)

        rate = _one_neuron(zero_counts, occupancy)
        assert_allclose(np.asarray(rate), 0.0, atol=1e-10)

    def test_single_bin(self):
        """Should handle single-bin case."""
        spike_counts = jnp.array([5.0])
        occupancy = jnp.array([2.0])

        result = _one_neuron(spike_counts, occupancy)
        assert_allclose(np.asarray(result), [2.5], rtol=1e-10)

    def test_all_nan_occupancy(self, spike_counts_1d):
        """All zero occupancy should return all NaN."""
        zero_occupancy = jnp.zeros(100, dtype=jnp.float64)

        result = _one_neuron(spike_counts_1d, zero_occupancy)
        assert np.all(np.isnan(np.asarray(result)))
