"""Tests for encoding/_core_jax.py - JAX core array operations.

This module tests the JAX implementation of core array operations used
by encoding functions. Tests verify:
1. Functions can import and are available
2. Basic functionality works with JAX arrays
3. Results are numerically equivalent to NumPy implementations
4. Functions handle edge cases properly

Following TDD approach: tests written before implementation.
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


@pytest.fixture
def adjacency() -> jnp.ndarray:
    """Simple adjacency matrix for smoothing.

    Returns
    -------
    jax.Array, shape (100, 100)
        Tridiagonal adjacency (1D chain graph).
    """
    n_bins = 100
    # Tridiagonal adjacency - each bin connected to neighbors
    adj = np.eye(n_bins, dtype=np.float64)
    for i in range(n_bins - 1):
        adj[i, i + 1] = 0.5
        adj[i + 1, i] = 0.5
    # Normalize rows to sum to 1
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / row_sums
    return jnp.array(adj)


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Tests for successful imports."""

    def test_import_compute_firing_rate_single(self):
        """compute_firing_rate_single should be importable."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        assert callable(compute_firing_rate_single)

    def test_import_compute_firing_rates_batch(self):
        """compute_firing_rates_batch should be importable."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        assert callable(compute_firing_rates_batch)

    def test_import_smooth_rate_map_single(self):
        """smooth_rate_map_single should be importable."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        assert callable(smooth_rate_map_single)

    def test_import_smooth_rate_maps_batch(self):
        """smooth_rate_maps_batch should be importable."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        assert callable(smooth_rate_maps_batch)


# =============================================================================
# Test compute_firing_rate_single
# =============================================================================


class TestComputeFiringRateSingle:
    """Tests for compute_firing_rate_single function."""

    def test_returns_jax_array(self, spike_counts_1d, occupancy):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(spike_counts_1d, occupancy)
        assert is_jax_array(result)

    def test_output_shape_matches_input(self, spike_counts_1d, occupancy):
        """Output shape should match input shape."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(spike_counts_1d, occupancy)
        assert result.shape == spike_counts_1d.shape

    def test_correct_rate_computation(self, spike_counts_1d, occupancy):
        """Rate should be spike_counts / occupancy."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(spike_counts_1d, occupancy)
        expected = spike_counts_1d / occupancy
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_zero_occupancy_gives_nan(self, spike_counts_1d, non_uniform_occupancy):
        """Zero occupancy should result in NaN."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(spike_counts_1d, non_uniform_occupancy)
        result_np = np.asarray(result)
        # Bin 0 has zero occupancy
        assert np.isnan(result_np[0])

    def test_min_occupancy_threshold(self, spike_counts_1d, non_uniform_occupancy):
        """Bins below min_occupancy should be NaN."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(
            spike_counts_1d, non_uniform_occupancy, min_occupancy=2.0
        )
        result_np = np.asarray(result)
        # Bins with occupancy < 2.0 should be NaN
        occupancy_np = np.asarray(non_uniform_occupancy)
        low_occ_mask = occupancy_np < 2.0
        assert np.all(np.isnan(result_np[low_occ_mask]))

    def test_non_negative_rates(self, spike_counts_1d, occupancy):
        """Firing rates should be non-negative."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        result = compute_firing_rate_single(spike_counts_1d, occupancy)
        result_np = np.asarray(result)
        assert np.all(result_np[~np.isnan(result_np)] >= 0)

    def test_matches_numpy_implementation(self, spike_counts_1d, occupancy):
        """JAX result should match NumPy implementation."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single
        from neurospatial.encoding._core_numpy import (
            compute_firing_rate_single as numpy_impl,
        )

        jax_result = compute_firing_rate_single(spike_counts_1d, occupancy)
        numpy_result = numpy_impl(np.asarray(spike_counts_1d), np.asarray(occupancy))
        assert_allclose(np.asarray(jax_result), numpy_result, rtol=1e-10)


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

    def test_matches_looped_single(self, spike_counts_2d, occupancy):
        """Batch result should match looped single calls."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rate_single,
            compute_firing_rates_batch,
        )

        batch_result = compute_firing_rates_batch(spike_counts_2d, occupancy)

        for i in range(spike_counts_2d.shape[0]):
            single_result = compute_firing_rate_single(spike_counts_2d[i], occupancy)
            assert_allclose(
                np.asarray(batch_result[i]),
                np.asarray(single_result),
                rtol=1e-10,
                equal_nan=True,
            )

    def test_matches_numpy_implementation(self, spike_counts_2d, occupancy):
        """JAX result should match NumPy implementation."""
        from neurospatial.encoding._core_jax import compute_firing_rates_batch
        from neurospatial.encoding._core_numpy import (
            compute_firing_rates_batch as numpy_impl,
        )

        jax_result = compute_firing_rates_batch(spike_counts_2d, occupancy)
        numpy_result = numpy_impl(np.asarray(spike_counts_2d), np.asarray(occupancy))
        assert_allclose(np.asarray(jax_result), numpy_result, rtol=1e-10)


# =============================================================================
# Test smooth_rate_map_single - diffusion_kde
# =============================================================================


class TestSmoothRateMapSingleDiffusion:
    """Tests for smooth_rate_map_single with diffusion_kde method.

    Note: These low-level functions apply matrix multiplication with the
    provided adjacency/kernel matrix. The bandwidth parameter is present
    for API compatibility but does not affect the computation - the kernel
    should already encode the bandwidth.
    """

    def test_returns_jax_array(self, spike_counts_1d, adjacency):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        assert is_jax_array(result)

    def test_output_shape_matches_input(self, spike_counts_1d, adjacency):
        """Output shape should match input shape."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        assert result.shape == spike_counts_1d.shape

    def test_applies_kernel_smoothing(self):
        """Smoothing should apply kernel via matrix multiplication."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        # Simple test: apply identity kernel, should return input unchanged
        firing_rate = jnp.array([0.0, 1.0, 5.0, 1.0, 0.0])
        identity_kernel = jnp.eye(5)

        result = smooth_rate_map_single(
            firing_rate, identity_kernel, bandwidth=2.0, method="diffusion_kde"
        )
        assert_allclose(np.asarray(result), np.asarray(firing_rate), rtol=1e-10)

    def test_smoothing_spreads_with_kernel(self):
        """Smoothing should spread values according to kernel weights."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        # Input with a single spike at center
        firing_rate = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0])

        # Simple averaging kernel (each bin averages with neighbors)
        kernel = jnp.array(
            [
                [0.6, 0.4, 0.0, 0.0, 0.0],
                [0.3, 0.4, 0.3, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.3, 0.0],
                [0.0, 0.0, 0.3, 0.4, 0.3],
                [0.0, 0.0, 0.0, 0.4, 0.6],
            ]
        )

        result = smooth_rate_map_single(
            firing_rate, kernel, bandwidth=2.0, method="diffusion_kde"
        )
        result_np = np.asarray(result)

        # Center should have peak (4.0 = 10 * 0.4)
        assert result_np[2] == pytest.approx(4.0)
        # Neighbors should have values (3.0 = 10 * 0.3)
        assert result_np[1] == pytest.approx(3.0)
        assert result_np[3] == pytest.approx(3.0)

    def test_non_negative_output(self, spike_counts_1d, adjacency):
        """Smoothed output should be non-negative for non-negative input."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        result_np = np.asarray(result)
        assert np.all(result_np[~np.isnan(result_np)] >= 0)


# =============================================================================
# Test smooth_rate_map_single - gaussian_kde
# =============================================================================


class TestSmoothRateMapSingleGaussian:
    """Tests for smooth_rate_map_single with gaussian_kde method.

    Note: Like diffusion_kde, these low-level functions apply matrix
    multiplication with the provided adjacency/kernel matrix.
    """

    def test_returns_jax_array(self, spike_counts_1d, adjacency):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="gaussian_kde"
        )
        assert is_jax_array(result)

    def test_output_shape_matches_input(self, spike_counts_1d, adjacency):
        """Output shape should match input shape."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="gaussian_kde"
        )
        assert result.shape == spike_counts_1d.shape

    def test_applies_kernel_smoothing(self):
        """Smoothing should apply kernel via matrix multiplication."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        # Simple test: apply identity kernel, should return input unchanged
        firing_rate = jnp.array([0.0, 1.0, 5.0, 1.0, 0.0])
        identity_kernel = jnp.eye(5)

        result = smooth_rate_map_single(
            firing_rate, identity_kernel, bandwidth=2.0, method="gaussian_kde"
        )
        assert_allclose(np.asarray(result), np.asarray(firing_rate), rtol=1e-10)

    def test_same_result_as_diffusion_kde(self, spike_counts_1d, adjacency):
        """gaussian_kde and diffusion_kde should give same result for same kernel."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result_gaussian = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="gaussian_kde"
        )
        result_diffusion = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )

        # Both methods apply the same kernel multiplication
        assert_allclose(
            np.asarray(result_gaussian), np.asarray(result_diffusion), rtol=1e-10
        )


# =============================================================================
# Test smooth_rate_map_single - binned
# =============================================================================


class TestSmoothRateMapSingleBinned:
    """Tests for smooth_rate_map_single with binned method."""

    def test_returns_jax_array(self, spike_counts_1d, adjacency):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        result = smooth_rate_map_single(
            spike_counts_1d, adjacency, bandwidth=2.0, method="binned"
        )
        assert is_jax_array(result)

    def test_binned_returns_input_unchanged_with_identity(self, spike_counts_1d):
        """With identity adjacency and method=binned, should return input."""
        from neurospatial.encoding._core_jax import smooth_rate_map_single

        identity = jnp.eye(len(spike_counts_1d))
        result = smooth_rate_map_single(
            spike_counts_1d, identity, bandwidth=0.0, method="binned"
        )
        assert_allclose(np.asarray(result), np.asarray(spike_counts_1d), rtol=1e-10)


# =============================================================================
# Test smooth_rate_maps_batch
# =============================================================================


class TestSmoothRateMapsBatch:
    """Tests for smooth_rate_maps_batch function."""

    def test_returns_jax_array(self, spike_counts_2d, adjacency):
        """Result should be a JAX array."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        result = smooth_rate_maps_batch(
            spike_counts_2d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        assert is_jax_array(result)

    def test_output_shape(self, spike_counts_2d, adjacency):
        """Output shape should be (n_neurons, n_bins)."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        result = smooth_rate_maps_batch(
            spike_counts_2d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        assert result.shape == spike_counts_2d.shape

    def test_matches_looped_single(self, spike_counts_2d, adjacency):
        """Batch result should match looped single calls."""
        from neurospatial.encoding._core_jax import (
            smooth_rate_map_single,
            smooth_rate_maps_batch,
        )

        batch_result = smooth_rate_maps_batch(
            spike_counts_2d, adjacency, bandwidth=2.0, method="diffusion_kde"
        )

        for i in range(spike_counts_2d.shape[0]):
            single_result = smooth_rate_map_single(
                spike_counts_2d[i], adjacency, bandwidth=2.0, method="diffusion_kde"
            )
            assert_allclose(
                np.asarray(batch_result[i]),
                np.asarray(single_result),
                rtol=1e-10,
                equal_nan=True,
            )

    def test_all_methods_work(self, spike_counts_2d, adjacency):
        """All smoothing methods should work with batch processing."""
        from neurospatial.encoding._core_jax import smooth_rate_maps_batch

        for method in ["diffusion_kde", "gaussian_kde", "binned"]:
            result = smooth_rate_maps_batch(
                spike_counts_2d, adjacency, bandwidth=2.0, method=method
            )
            assert result.shape == spike_counts_2d.shape


# =============================================================================
# Test Numerical Equivalence with NumPy
# =============================================================================


class TestNumericalEquivalenceWithNumPy:
    """Tests verifying JAX and NumPy implementations produce same results."""

    def test_firing_rate_single_matches(self, spike_counts_1d, occupancy):
        """JAX and NumPy compute_firing_rate_single should match."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rate_single as jax_impl,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rate_single as numpy_impl,
        )

        jax_result = jax_impl(spike_counts_1d, occupancy)
        numpy_result = numpy_impl(np.asarray(spike_counts_1d), np.asarray(occupancy))

        assert_allclose(
            np.asarray(jax_result),
            numpy_result,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_firing_rate_single_with_min_occupancy_matches(
        self, spike_counts_1d, non_uniform_occupancy
    ):
        """JAX and NumPy should match with min_occupancy threshold."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rate_single as jax_impl,
        )
        from neurospatial.encoding._core_numpy import (
            compute_firing_rate_single as numpy_impl,
        )

        jax_result = jax_impl(spike_counts_1d, non_uniform_occupancy, min_occupancy=2.0)
        numpy_result = numpy_impl(
            np.asarray(spike_counts_1d),
            np.asarray(non_uniform_occupancy),
            min_occupancy=2.0,
        )

        assert_allclose(
            np.asarray(jax_result),
            numpy_result,
            rtol=1e-10,
            equal_nan=True,
        )

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


# =============================================================================
# Test JAX-Specific Features
# =============================================================================


class TestJAXSpecificFeatures:
    """Tests for JAX-specific features like JIT compilation."""

    def test_jittable_firing_rate_single(self, spike_counts_1d, occupancy):
        """compute_firing_rate_single should be JIT-compilable."""
        import jax

        from neurospatial.encoding._core_jax import compute_firing_rate_single

        jitted_fn = jax.jit(compute_firing_rate_single)
        result = jitted_fn(spike_counts_1d, occupancy)

        # Should produce valid result
        assert result.shape == spike_counts_1d.shape
        assert is_jax_array(result)

    def test_jittable_firing_rates_batch(self, spike_counts_2d, occupancy):
        """compute_firing_rates_batch should be JIT-compilable."""
        import jax

        from neurospatial.encoding._core_jax import compute_firing_rates_batch

        jitted_fn = jax.jit(compute_firing_rates_batch)
        result = jitted_fn(spike_counts_2d, occupancy)

        # Should produce valid result
        assert result.shape == spike_counts_2d.shape
        assert is_jax_array(result)

    def test_jittable_smooth_rate_map_single(self, spike_counts_1d, adjacency):
        """smooth_rate_map_single should be JIT-compilable."""
        from functools import partial

        import jax

        from neurospatial.encoding._core_jax import smooth_rate_map_single

        # Use partial to fix bandwidth and method (static args)
        jitted_fn = jax.jit(
            partial(smooth_rate_map_single, bandwidth=2.0, method="diffusion_kde")
        )
        result = jitted_fn(spike_counts_1d, adjacency)

        # Should produce valid result
        assert result.shape == spike_counts_1d.shape
        assert is_jax_array(result)

    def test_vmap_over_neurons(self, spike_counts_2d, occupancy):
        """Functions should work with vmap for vectorization."""
        import jax

        from neurospatial.encoding._core_jax import compute_firing_rate_single

        # vmap over neurons (first axis)
        vmapped_fn = jax.vmap(compute_firing_rate_single, in_axes=(0, None))
        result = vmapped_fn(spike_counts_2d, occupancy)

        # Should produce same shape as batch function
        assert result.shape == spike_counts_2d.shape


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_zero_spike_counts(self, occupancy, adjacency):
        """All zero spike counts should return all zeros."""
        from neurospatial.encoding._core_jax import (
            compute_firing_rate_single,
            smooth_rate_map_single,
        )

        zero_counts = jnp.zeros(100, dtype=jnp.float64)

        rate = compute_firing_rate_single(zero_counts, occupancy)
        assert_allclose(np.asarray(rate), 0.0, atol=1e-10)

        smoothed = smooth_rate_map_single(
            zero_counts, adjacency, bandwidth=2.0, method="diffusion_kde"
        )
        assert_allclose(np.asarray(smoothed), 0.0, atol=1e-10)

    def test_single_bin(self):
        """Should handle single-bin case."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        spike_counts = jnp.array([5.0])
        occupancy = jnp.array([2.0])

        result = compute_firing_rate_single(spike_counts, occupancy)
        assert_allclose(np.asarray(result), [2.5], rtol=1e-10)

    def test_all_nan_occupancy(self, spike_counts_1d):
        """All zero occupancy should return all NaN."""
        from neurospatial.encoding._core_jax import compute_firing_rate_single

        zero_occupancy = jnp.zeros(100, dtype=jnp.float64)

        result = compute_firing_rate_single(spike_counts_1d, zero_occupancy)
        assert np.all(np.isnan(np.asarray(result)))
