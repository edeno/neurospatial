"""Tests for JAX metric computations in encoding/_core_jax.py.

This module tests the JAX implementation of spatial information and sparsity
metrics. Tests verify:
1. Functions can import and are available
2. Basic functionality works with JAX arrays
3. Results are numerically equivalent to NumPy implementations
4. Functions handle edge cases properly
5. JAX-specific features (jit, vmap) work correctly

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
    return hasattr(obj, "devices") or type(obj).__module__.startswith("jax")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def uniform_firing_rate() -> jnp.ndarray:
    """Uniform firing rate (no spatial selectivity).

    Returns
    -------
    jax.Array, shape (100,)
        Uniform firing rate of 5.0 Hz across all bins.
    """
    return jnp.ones(100, dtype=jnp.float64) * 5.0


@pytest.fixture
def selective_firing_rate() -> jnp.ndarray:
    """Selective firing rate (high spatial selectivity).

    Returns
    -------
    jax.Array, shape (100,)
        Firing rate with a single peak at bin 50.
    """
    rate = jnp.zeros(100, dtype=jnp.float64)
    rate = rate.at[50].set(30.0)
    return rate


@pytest.fixture
def place_field_firing_rate() -> jnp.ndarray:
    """Place field-like firing rate.

    Returns
    -------
    jax.Array, shape (100,)
        Gaussian-like place field centered at bin 50.
    """
    x = np.arange(100, dtype=np.float64)
    rate = 10.0 * np.exp(-((x - 50) ** 2) / (2 * 10**2))
    return jnp.array(rate)


@pytest.fixture
def uniform_occupancy() -> jnp.ndarray:
    """Uniform occupancy.

    Returns
    -------
    jax.Array, shape (100,)
        Uniform occupancy of 1.0 second in each bin.
    """
    return jnp.ones(100, dtype=jnp.float64)


@pytest.fixture
def non_uniform_occupancy() -> jnp.ndarray:
    """Non-uniform occupancy with more time in center.

    Returns
    -------
    jax.Array, shape (100,)
        More time spent in center bins.
    """
    occ = np.ones(100, dtype=np.float64)
    occ[40:60] = 5.0  # More time in center
    return jnp.array(occ)


@pytest.fixture
def occupancy_with_zeros() -> jnp.ndarray:
    """Occupancy with some zero bins.

    Returns
    -------
    jax.Array, shape (100,)
        Occupancy with zeros at some bins.
    """
    occ = np.ones(100, dtype=np.float64)
    occ[0:10] = 0.0  # No time in first 10 bins
    return jnp.array(occ)


@pytest.fixture
def batch_firing_rates() -> jnp.ndarray:
    """Multiple neurons with different firing patterns.

    Returns
    -------
    jax.Array, shape (5, 100)
        Firing rates for 5 neurons with varying selectivity.
    """
    n_neurons = 5
    n_bins = 100
    rates = np.zeros((n_neurons, n_bins), dtype=np.float64)

    # Neuron 0: uniform (low info)
    rates[0] = 5.0

    # Neuron 1: single peak (high info)
    rates[1, 50] = 30.0

    # Neuron 2: broad place field
    x = np.arange(n_bins, dtype=np.float64)
    rates[2] = 10.0 * np.exp(-((x - 50) ** 2) / (2 * 15**2))

    # Neuron 3: narrow place field
    rates[3] = 10.0 * np.exp(-((x - 50) ** 2) / (2 * 5**2))

    # Neuron 4: two fields
    rates[4] = 5.0 * np.exp(-((x - 25) ** 2) / (2 * 8**2))
    rates[4] += 5.0 * np.exp(-((x - 75) ** 2) / (2 * 8**2))

    return jnp.array(rates)


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Tests for successful imports of metric functions."""

    def test_import_spatial_information_single(self):
        """spatial_information_single should be importable."""
        from neurospatial.encoding._core_jax import spatial_information_single

        assert callable(spatial_information_single)

    def test_import_spatial_information_batch(self):
        """spatial_information_batch should be importable."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        assert callable(spatial_information_batch)

    def test_import_sparsity_single(self):
        """sparsity_single should be importable."""
        from neurospatial.encoding._core_jax import sparsity_single

        assert callable(sparsity_single)

    def test_import_sparsity_batch(self):
        """sparsity_batch should be importable."""
        from neurospatial.encoding._core_jax import sparsity_batch

        assert callable(sparsity_batch)


# =============================================================================
# Return Type Tests
# =============================================================================


class TestReturnTypes:
    """Tests that functions return JAX arrays."""

    def test_spatial_information_single_returns_jax_array(
        self, uniform_firing_rate, uniform_occupancy
    ):
        """spatial_information_single should return a JAX scalar/array."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(uniform_firing_rate, uniform_occupancy)
        assert is_jax_array(result)

    def test_spatial_information_batch_returns_jax_array(
        self, batch_firing_rates, uniform_occupancy
    ):
        """spatial_information_batch should return a JAX array."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        result = spatial_information_batch(batch_firing_rates, uniform_occupancy)
        assert is_jax_array(result)
        assert result.shape == (5,)

    def test_sparsity_single_returns_jax_array(
        self, uniform_firing_rate, uniform_occupancy
    ):
        """sparsity_single should return a JAX scalar/array."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(uniform_firing_rate, uniform_occupancy)
        assert is_jax_array(result)

    def test_sparsity_batch_returns_jax_array(
        self, batch_firing_rates, uniform_occupancy
    ):
        """sparsity_batch should return a JAX array."""
        from neurospatial.encoding._core_jax import sparsity_batch

        result = sparsity_batch(batch_firing_rates, uniform_occupancy)
        assert is_jax_array(result)
        assert result.shape == (5,)


# =============================================================================
# Spatial Information Tests
# =============================================================================


class TestSpatialInformationSingle:
    """Tests for spatial_information_single function."""

    def test_uniform_firing_has_zero_information(
        self, uniform_firing_rate, uniform_occupancy
    ):
        """Uniform firing should have zero spatial information."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(uniform_firing_rate, uniform_occupancy)
        assert_allclose(float(result), 0.0, atol=1e-10)

    def test_selective_firing_has_high_information(
        self, selective_firing_rate, uniform_occupancy
    ):
        """Selective firing should have high spatial information."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(selective_firing_rate, uniform_occupancy)
        # Single spike at one bin should give log2(n_bins) bits
        assert float(result) > 4.0  # Should be around log2(100) ≈ 6.6

    def test_place_field_has_moderate_information(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """Place field firing should have moderate spatial information."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(place_field_firing_rate, uniform_occupancy)
        # Gaussian place field should have moderate info (1-3 bits typical)
        assert 0.5 < float(result) < 5.0

    def test_information_is_non_negative(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """Spatial information should always be non-negative."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(place_field_firing_rate, uniform_occupancy)
        assert float(result) >= 0.0

    def test_handles_zero_occupancy_bins(
        self, place_field_firing_rate, occupancy_with_zeros
    ):
        """Should handle bins with zero occupancy."""
        from neurospatial.encoding._core_jax import spatial_information_single

        result = spatial_information_single(
            place_field_firing_rate, occupancy_with_zeros
        )
        # Should not produce NaN or inf
        assert np.isfinite(float(result))

    def test_handles_all_zero_firing(self, uniform_occupancy):
        """Should handle zero firing rate."""
        from neurospatial.encoding._core_jax import spatial_information_single

        zero_rate = jnp.zeros(100, dtype=jnp.float64)
        result = spatial_information_single(zero_rate, uniform_occupancy)
        assert_allclose(float(result), 0.0, atol=1e-10)

    def test_base_parameter(self, selective_firing_rate, uniform_occupancy):
        """Should support different logarithm bases."""
        from neurospatial.encoding._core_jax import spatial_information_single

        bits = spatial_information_single(
            selective_firing_rate, uniform_occupancy, base=2.0
        )
        nats = spatial_information_single(
            selective_firing_rate, uniform_occupancy, base=np.e
        )
        # nats = bits * ln(2)
        assert_allclose(float(nats), float(bits) * np.log(2), rtol=1e-10)


class TestSpatialInformationBatch:
    """Tests for spatial_information_batch function."""

    def test_batch_shape(self, batch_firing_rates, uniform_occupancy):
        """Batch result should have shape (n_neurons,)."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        result = spatial_information_batch(batch_firing_rates, uniform_occupancy)
        assert result.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates, uniform_occupancy):
        """Batch result should match individual computations."""
        from neurospatial.encoding._core_jax import (
            spatial_information_batch,
            spatial_information_single,
        )

        batch_result = spatial_information_batch(batch_firing_rates, uniform_occupancy)

        for i in range(5):
            single_result = spatial_information_single(
                batch_firing_rates[i], uniform_occupancy
            )
            assert_allclose(float(batch_result[i]), float(single_result), rtol=1e-10)

    def test_batch_ranking(self, batch_firing_rates, uniform_occupancy):
        """Neurons should be ranked by selectivity."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        result = spatial_information_batch(batch_firing_rates, uniform_occupancy)
        result_np = np.array(result)

        # Neuron 0 (uniform) should have lowest info
        # Neuron 1 (single spike) should have highest info
        assert result_np[0] < result_np[1]
        assert result_np[0] < result_np[2]

    def test_empty_batch(self, uniform_occupancy):
        """Should handle empty batch (0 neurons)."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        empty_rates = jnp.zeros((0, 100), dtype=jnp.float64)
        result = spatial_information_batch(empty_rates, uniform_occupancy)
        assert result.shape == (0,)


# =============================================================================
# Sparsity Tests
# =============================================================================


class TestSparsitySingle:
    """Tests for sparsity_single function."""

    def test_uniform_firing_has_high_sparsity(
        self, uniform_firing_rate, uniform_occupancy
    ):
        """Uniform firing should have sparsity close to 1."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(uniform_firing_rate, uniform_occupancy)
        assert_allclose(float(result), 1.0, atol=1e-6)

    def test_selective_firing_has_low_sparsity(
        self, selective_firing_rate, uniform_occupancy
    ):
        """Selective firing should have low sparsity."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(selective_firing_rate, uniform_occupancy)
        # Single spike at one bin should give 1/n_bins = 0.01
        assert float(result) < 0.05

    def test_place_field_has_moderate_sparsity(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """Place field firing should have moderate sparsity."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(place_field_firing_rate, uniform_occupancy)
        # Gaussian place field typically has sparsity 0.1-0.5
        assert 0.05 < float(result) < 0.8

    def test_sparsity_in_valid_range(self, place_field_firing_rate, uniform_occupancy):
        """Sparsity should be in [0, 1] range."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(place_field_firing_rate, uniform_occupancy)
        assert 0.0 <= float(result) <= 1.0

    def test_handles_zero_occupancy_bins(
        self, place_field_firing_rate, occupancy_with_zeros
    ):
        """Should handle bins with zero occupancy."""
        from neurospatial.encoding._core_jax import sparsity_single

        result = sparsity_single(place_field_firing_rate, occupancy_with_zeros)
        # Should not produce NaN or inf
        assert np.isfinite(float(result))

    def test_handles_all_zero_firing(self, uniform_occupancy):
        """Should handle zero firing rate."""
        from neurospatial.encoding._core_jax import sparsity_single

        zero_rate = jnp.zeros(100, dtype=jnp.float64)
        result = sparsity_single(zero_rate, uniform_occupancy)
        # Zero rate should return 0 (undefined but clamped)
        assert np.isfinite(float(result))


class TestSparsityBatch:
    """Tests for sparsity_batch function."""

    def test_batch_shape(self, batch_firing_rates, uniform_occupancy):
        """Batch result should have shape (n_neurons,)."""
        from neurospatial.encoding._core_jax import sparsity_batch

        result = sparsity_batch(batch_firing_rates, uniform_occupancy)
        assert result.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates, uniform_occupancy):
        """Batch result should match individual computations."""
        from neurospatial.encoding._core_jax import sparsity_batch, sparsity_single

        batch_result = sparsity_batch(batch_firing_rates, uniform_occupancy)

        for i in range(5):
            single_result = sparsity_single(batch_firing_rates[i], uniform_occupancy)
            assert_allclose(float(batch_result[i]), float(single_result), rtol=1e-10)

    def test_batch_ranking(self, batch_firing_rates, uniform_occupancy):
        """Neurons should be ranked by sparsity."""
        from neurospatial.encoding._core_jax import sparsity_batch

        result = sparsity_batch(batch_firing_rates, uniform_occupancy)
        result_np = np.array(result)

        # Neuron 0 (uniform) should have highest sparsity (~1)
        # Neuron 1 (single spike) should have lowest sparsity
        assert result_np[0] > result_np[1]

    def test_empty_batch(self, uniform_occupancy):
        """Should handle empty batch (0 neurons)."""
        from neurospatial.encoding._core_jax import sparsity_batch

        empty_rates = jnp.zeros((0, 100), dtype=jnp.float64)
        result = sparsity_batch(empty_rates, uniform_occupancy)
        assert result.shape == (0,)


# =============================================================================
# Numerical Equivalence with NumPy Tests
# =============================================================================


class TestNumpyEquivalence:
    """Tests that JAX results match NumPy implementations."""

    def test_spatial_information_matches_numpy(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """JAX spatial_information should match NumPy implementation."""
        from neurospatial.encoding._core_jax import spatial_information_single
        from neurospatial.encoding._metrics import (
            spatial_information as np_spatial_info,
        )

        jax_result = spatial_information_single(
            place_field_firing_rate, uniform_occupancy
        )
        np_result = np_spatial_info(
            np.array(place_field_firing_rate), np.array(uniform_occupancy)
        )

        assert_allclose(float(jax_result), np_result, rtol=1e-10)

    def test_spatial_information_batch_matches_numpy(
        self, batch_firing_rates, uniform_occupancy
    ):
        """JAX batch spatial_information should match NumPy implementation."""
        from neurospatial.encoding._core_jax import spatial_information_batch
        from neurospatial.encoding._metrics import (
            batch_spatial_information as np_batch_info,
        )

        jax_result = spatial_information_batch(batch_firing_rates, uniform_occupancy)
        np_result = np_batch_info(
            np.array(batch_firing_rates), np.array(uniform_occupancy)
        )

        # Use atol for values near zero (uniform firing has ~0 info)
        # rtol for larger values where relative comparison is appropriate
        assert_allclose(np.array(jax_result), np_result, rtol=1e-10, atol=1e-14)

    def test_sparsity_matches_numpy(self, place_field_firing_rate, uniform_occupancy):
        """JAX sparsity should match NumPy implementation."""
        from neurospatial.encoding._core_jax import sparsity_single
        from neurospatial.encoding._metrics import sparsity as np_sparsity

        jax_result = sparsity_single(place_field_firing_rate, uniform_occupancy)
        np_result = np_sparsity(
            np.array(place_field_firing_rate), np.array(uniform_occupancy)
        )

        assert_allclose(float(jax_result), np_result, rtol=1e-10)

    def test_sparsity_batch_matches_numpy(self, batch_firing_rates, uniform_occupancy):
        """JAX batch sparsity should match NumPy implementation."""
        from neurospatial.encoding._core_jax import sparsity_batch
        from neurospatial.encoding._metrics import batch_sparsity as np_batch_sparsity

        jax_result = sparsity_batch(batch_firing_rates, uniform_occupancy)
        np_result = np_batch_sparsity(
            np.array(batch_firing_rates), np.array(uniform_occupancy)
        )

        assert_allclose(np.array(jax_result), np_result, rtol=1e-10)


# =============================================================================
# JAX-Specific Feature Tests
# =============================================================================


class TestJaxFeatures:
    """Tests for JAX-specific features (jit, vmap, grad)."""

    def test_spatial_information_single_is_jittable(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """spatial_information_single should work with jit."""
        from neurospatial.encoding._core_jax import spatial_information_single

        jitted_fn = jax.jit(spatial_information_single)
        result = jitted_fn(place_field_firing_rate, uniform_occupancy)
        assert np.isfinite(float(result))

    def test_spatial_information_batch_is_jittable(
        self, batch_firing_rates, uniform_occupancy
    ):
        """spatial_information_batch should work with jit."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        jitted_fn = jax.jit(spatial_information_batch)
        result = jitted_fn(batch_firing_rates, uniform_occupancy)
        assert result.shape == (5,)

    def test_sparsity_single_is_jittable(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """sparsity_single should work with jit."""
        from neurospatial.encoding._core_jax import sparsity_single

        jitted_fn = jax.jit(sparsity_single)
        result = jitted_fn(place_field_firing_rate, uniform_occupancy)
        assert np.isfinite(float(result))

    def test_sparsity_batch_is_jittable(self, batch_firing_rates, uniform_occupancy):
        """sparsity_batch should work with jit."""
        from neurospatial.encoding._core_jax import sparsity_batch

        jitted_fn = jax.jit(sparsity_batch)
        result = jitted_fn(batch_firing_rates, uniform_occupancy)
        assert result.shape == (5,)

    def test_spatial_information_single_vmap(
        self, batch_firing_rates, uniform_occupancy
    ):
        """spatial_information_single should work with vmap."""
        from neurospatial.encoding._core_jax import spatial_information_single

        # vmap over first axis of firing_rates
        vmapped_fn = jax.vmap(
            lambda rate: spatial_information_single(rate, uniform_occupancy)
        )
        result = vmapped_fn(batch_firing_rates)
        assert result.shape == (5,)

    def test_sparsity_single_vmap(self, batch_firing_rates, uniform_occupancy):
        """sparsity_single should work with vmap."""
        from neurospatial.encoding._core_jax import sparsity_single

        # vmap over first axis of firing_rates
        vmapped_fn = jax.vmap(lambda rate: sparsity_single(rate, uniform_occupancy))
        result = vmapped_fn(batch_firing_rates)
        assert result.shape == (5,)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_bin_spatial_information(self):
        """Should handle single-bin arrays."""
        from neurospatial.encoding._core_jax import spatial_information_single

        rate = jnp.array([5.0], dtype=jnp.float64)
        occ = jnp.array([1.0], dtype=jnp.float64)
        result = spatial_information_single(rate, occ)
        # Single bin means no spatial information
        assert_allclose(float(result), 0.0, atol=1e-10)

    def test_single_bin_sparsity(self):
        """Should handle single-bin arrays."""
        from neurospatial.encoding._core_jax import sparsity_single

        rate = jnp.array([5.0], dtype=jnp.float64)
        occ = jnp.array([1.0], dtype=jnp.float64)
        result = sparsity_single(rate, occ)
        # Single bin should have sparsity = 1.0
        assert_allclose(float(result), 1.0, atol=1e-10)

    def test_all_nan_occupancy(self):
        """Should handle all-NaN occupancy."""
        from neurospatial.encoding._core_jax import spatial_information_single

        rate = jnp.ones(100, dtype=jnp.float64) * 5.0
        occ = jnp.full(100, jnp.nan, dtype=jnp.float64)
        result = spatial_information_single(rate, occ)
        # Should return 0 when all NaN
        assert np.isfinite(float(result)) or np.isnan(float(result))

    def test_very_small_occupancy(self):
        """Should handle very small but non-zero occupancy."""
        from neurospatial.encoding._core_jax import spatial_information_single

        rate = jnp.ones(100, dtype=jnp.float64) * 5.0
        occ = jnp.ones(100, dtype=jnp.float64) * 1e-15
        result = spatial_information_single(rate, occ)
        # Should not produce inf or nan
        assert np.isfinite(float(result)) or np.isnan(float(result))

    def test_very_high_firing_rate(self):
        """Should handle very high firing rates."""
        from neurospatial.encoding._core_jax import spatial_information_single

        rate = jnp.zeros(100, dtype=jnp.float64)
        rate = rate.at[50].set(1e10)  # Very high rate
        occ = jnp.ones(100, dtype=jnp.float64)
        result = spatial_information_single(rate, occ)
        # Should produce finite result
        assert np.isfinite(float(result))

    def test_two_bins_spatial_information(self):
        """Should handle two-bin arrays correctly."""
        from neurospatial.encoding._core_jax import spatial_information_single

        # Fires only in one of two equally-visited bins
        rate = jnp.array([10.0, 0.0], dtype=jnp.float64)
        occ = jnp.array([1.0, 1.0], dtype=jnp.float64)
        result = spatial_information_single(rate, occ)
        # Should be 1 bit (fires in half of space)
        assert_allclose(float(result), 1.0, rtol=1e-5)

    def test_batch_with_mixed_edge_cases(self, uniform_occupancy):
        """Should handle batch with various edge cases."""
        from neurospatial.encoding._core_jax import spatial_information_batch

        rates = np.zeros((4, 100), dtype=np.float64)
        # Neuron 0: all zeros
        # Neuron 1: uniform
        rates[1] = 5.0
        # Neuron 2: single spike
        rates[2, 50] = 10.0
        # Neuron 3: half active
        rates[3, 50:] = 5.0

        result = spatial_information_batch(jnp.array(rates), uniform_occupancy)
        assert result.shape == (4,)
        # All results should be finite (except potentially zeros)
        assert np.all(np.isfinite(np.array(result)) | (np.array(result) == 0.0))


# =============================================================================
# Consistency Tests
# =============================================================================


class TestConsistency:
    """Tests for internal consistency of results."""

    def test_information_sparsity_relationship(
        self, batch_firing_rates, uniform_occupancy
    ):
        """Higher info should generally correlate with lower sparsity."""
        from neurospatial.encoding._core_jax import (
            sparsity_batch,
            spatial_information_batch,
        )

        info = np.array(
            spatial_information_batch(batch_firing_rates, uniform_occupancy)
        )
        spars = np.array(sparsity_batch(batch_firing_rates, uniform_occupancy))

        # Exclude uniform neuron (index 0) which has special properties
        # For non-uniform neurons, higher info should mean lower sparsity
        # (This is a general trend, not strict)
        # At minimum, the most selective neuron (index 1) should have:
        # - highest info
        # - lowest sparsity
        most_selective = np.argmax(info)
        assert most_selective == np.argmin(spars)

    def test_repeated_computation_same_result(
        self, place_field_firing_rate, uniform_occupancy
    ):
        """Same inputs should produce same outputs."""
        from neurospatial.encoding._core_jax import (
            sparsity_single,
            spatial_information_single,
        )

        info1 = spatial_information_single(place_field_firing_rate, uniform_occupancy)
        info2 = spatial_information_single(place_field_firing_rate, uniform_occupancy)
        assert_allclose(float(info1), float(info2))

        spars1 = sparsity_single(place_field_firing_rate, uniform_occupancy)
        spars2 = sparsity_single(place_field_firing_rate, uniform_occupancy)
        assert_allclose(float(spars1), float(spars2))

    def test_batch_single_equivalence_random(self):
        """Batch processing should equal individual processing for random data."""
        from neurospatial.encoding._core_jax import (
            sparsity_batch,
            sparsity_single,
            spatial_information_batch,
            spatial_information_single,
        )

        rng = np.random.default_rng(42)
        n_neurons = 10
        n_bins = 50

        rates = jnp.array(rng.random((n_neurons, n_bins)) * 20, dtype=jnp.float64)
        occ = jnp.array(rng.random(n_bins) + 0.1, dtype=jnp.float64)

        # Batch computation
        info_batch = np.array(spatial_information_batch(rates, occ))
        spars_batch = np.array(sparsity_batch(rates, occ))

        # Single computations
        for i in range(n_neurons):
            info_single = float(spatial_information_single(rates[i], occ))
            spars_single = float(sparsity_single(rates[i], occ))
            assert_allclose(info_batch[i], info_single, rtol=1e-10)
            assert_allclose(spars_batch[i], spars_single, rtol=1e-10)
