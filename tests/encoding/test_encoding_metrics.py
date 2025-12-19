"""Tests for encoding/_metrics.py - shared metric implementations.

This module tests the shared spatial information and sparsity computations
that will be used by result classes (SpatialRateResult, etc.).

Following TDD approach: tests written before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import will fail until we implement the module
from neurospatial.encoding._metrics import (
    batch_sparsity,
    batch_spatial_information,
    sparsity,
    spatial_information,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def uniform_firing() -> tuple[np.ndarray, np.ndarray]:
    """Uniform firing rate (no spatial information).

    Returns
    -------
    firing_rate : ndarray
        Constant 3 Hz across 100 bins.
    occupancy : ndarray
        Uniform occupancy (equal time in all bins).
    """
    n_bins = 100
    firing_rate = np.ones(n_bins) * 3.0  # 3 Hz everywhere
    occupancy = np.ones(n_bins)  # Equal time in all bins
    return firing_rate, occupancy


@pytest.fixture
def selective_firing() -> tuple[np.ndarray, np.ndarray]:
    """Highly selective firing (one hot bin).

    Returns
    -------
    firing_rate : ndarray
        50 Hz in one bin, 0 Hz elsewhere.
    occupancy : ndarray
        Uniform occupancy.
    """
    n_bins = 100
    firing_rate = np.zeros(n_bins)
    firing_rate[50] = 50.0  # 50 Hz in bin 50 only
    occupancy = np.ones(n_bins)  # Equal time in all bins
    return firing_rate, occupancy


@pytest.fixture
def place_cell_like() -> tuple[np.ndarray, np.ndarray]:
    """Place cell-like firing pattern (Gaussian bump).

    Returns
    -------
    firing_rate : ndarray
        Gaussian-shaped firing field centered at bin 50.
    occupancy : ndarray
        Uniform occupancy.
    """
    n_bins = 100
    x = np.arange(n_bins)
    center = 50
    width = 5
    # Gaussian-shaped place field
    firing_rate = 10.0 * np.exp(-((x - center) ** 2) / (2 * width**2))
    occupancy = np.ones(n_bins)
    return firing_rate, occupancy


@pytest.fixture
def batch_firing_rates() -> tuple[np.ndarray, np.ndarray]:
    """Batch of firing rates for multiple neurons.

    Returns
    -------
    firing_rates : ndarray
        Shape (5, 100): 5 neurons with different patterns.
    occupancy : ndarray
        Uniform occupancy (shared).
    """
    n_neurons = 5
    n_bins = 100
    rng = np.random.default_rng(42)

    firing_rates = np.zeros((n_neurons, n_bins))

    # Neuron 0: uniform (no spatial info)
    firing_rates[0] = np.ones(n_bins) * 5.0

    # Neuron 1: selective (high spatial info)
    firing_rates[1, 20] = 30.0

    # Neuron 2: place cell-like
    x = np.arange(n_bins)
    firing_rates[2] = 15.0 * np.exp(-((x - 60) ** 2) / (2 * 8**2))

    # Neuron 3: multiple fields
    firing_rates[3] = 10.0 * np.exp(-((x - 25) ** 2) / (2 * 5**2))
    firing_rates[3] += 10.0 * np.exp(-((x - 75) ** 2) / (2 * 5**2))

    # Neuron 4: noisy
    firing_rates[4] = rng.uniform(1.0, 5.0, n_bins)

    occupancy = np.ones(n_bins)

    return firing_rates, occupancy


# =============================================================================
# Test spatial_information (single neuron)
# =============================================================================


class TestSpatialInformation:
    """Tests for spatial_information() function."""

    def test_uniform_firing_zero_info(self, uniform_firing):
        """Uniform firing should give zero spatial information."""
        firing_rate, occupancy = uniform_firing
        info = spatial_information(firing_rate, occupancy)
        assert_allclose(info, 0.0, atol=1e-10)

    def test_selective_firing_high_info(self, selective_firing):
        """Highly selective firing should give high spatial information."""
        firing_rate, occupancy = selective_firing
        info = spatial_information(firing_rate, occupancy)
        # One-hot firing gives very high spatial information
        # log2(100) = 6.64 bits would be max
        assert info > 4.0  # Should be close to log2(n_bins)

    def test_place_cell_typical_info(self, place_cell_like):
        """Place cell-like firing should give typical 1-3 bits/spike."""
        firing_rate, occupancy = place_cell_like
        info = spatial_information(firing_rate, occupancy)
        # Typical place cells: 1-3 bits/spike
        assert 0.5 < info < 5.0

    def test_non_negative(self, place_cell_like):
        """Spatial information should always be non-negative."""
        firing_rate, occupancy = place_cell_like
        info = spatial_information(firing_rate, occupancy)
        assert info >= 0.0

    def test_returns_float(self, uniform_firing):
        """Should return a Python float, not array."""
        firing_rate, occupancy = uniform_firing
        info = spatial_information(firing_rate, occupancy)
        assert isinstance(info, float)

    def test_formula_matches_skaggs(self):
        """Test the Skaggs formula is correctly implemented.

        Formula: I = sum_i(p_i * (r_i/r_mean) * log2(r_i/r_mean))
        """
        # Simple case with known values
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)

        # Manual computation:
        # p_i = [0.25, 0.25, 0.25, 0.25]
        # r_mean = (0 + 2 + 4 + 2)/4 = 2
        # bin 0: 0 (skip, r_i = 0)
        # bin 1: 0.25 * (2/2) * log2(2/2) = 0.25 * 1 * 0 = 0
        # bin 2: 0.25 * (4/2) * log2(4/2) = 0.25 * 2 * 1 = 0.5
        # bin 3: 0.25 * (2/2) * log2(2/2) = 0.25 * 1 * 0 = 0
        # Total = 0.5
        assert_allclose(info, 0.5, rtol=0.1)

    def test_base_parameter(self):
        """Test different log bases (bits vs nats)."""
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        info_bits = spatial_information(firing_rate, occupancy, base=2.0)
        info_nats = spatial_information(firing_rate, occupancy, base=np.e)

        # nats = bits * ln(2)
        expected_nats = info_bits * np.log(2)
        assert_allclose(info_nats, expected_nats, rtol=1e-6)

    def test_zero_mean_rate_returns_zero(self):
        """Should return 0.0 when mean firing rate is zero."""
        firing_rate = np.zeros(100)
        occupancy = np.ones(100)
        info = spatial_information(firing_rate, occupancy)
        assert info == 0.0

    def test_nan_values_handled(self):
        """Should handle NaN values in firing rate."""
        firing_rate = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        occupancy = np.ones(5)
        info = spatial_information(firing_rate, occupancy)
        # Should compute info ignoring NaN
        assert not np.isnan(info)
        assert info >= 0.0

    def test_non_uniform_occupancy(self):
        """Should correctly weight by occupancy probability."""
        # More time spent in bin 0
        firing_rate = np.array([5.0, 5.0, 5.0, 5.0])
        occupancy = np.array([10.0, 1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)
        # Still uniform firing rate, so info should still be ~0
        assert_allclose(info, 0.0, atol=1e-10)


# =============================================================================
# Test sparsity (single neuron)
# =============================================================================


class TestSparsity:
    """Tests for sparsity() function."""

    def test_uniform_firing_high_sparsity(self, uniform_firing):
        """Uniform firing should give sparsity close to 1.0."""
        firing_rate, occupancy = uniform_firing
        spars = sparsity(firing_rate, occupancy)
        assert_allclose(spars, 1.0, atol=1e-6)

    def test_selective_firing_low_sparsity(self, selective_firing):
        """Highly selective firing should give low sparsity."""
        firing_rate, occupancy = selective_firing
        spars = sparsity(firing_rate, occupancy)
        # One-hot firing → very sparse
        assert spars < 0.1

    def test_place_cell_typical_sparsity(self, place_cell_like):
        """Place cell-like firing should give typical 0.1-0.3 sparsity."""
        firing_rate, occupancy = place_cell_like
        spars = sparsity(firing_rate, occupancy)
        # Typical place cells: 0.1-0.3
        assert 0.05 < spars < 0.5

    def test_range_zero_to_one(self, place_cell_like):
        """Sparsity should always be in [0, 1]."""
        firing_rate, occupancy = place_cell_like
        spars = sparsity(firing_rate, occupancy)
        assert 0.0 <= spars <= 1.0

    def test_returns_float(self, uniform_firing):
        """Should return a Python float, not array."""
        firing_rate, occupancy = uniform_firing
        spars = sparsity(firing_rate, occupancy)
        assert isinstance(spars, float)

    def test_formula_matches_skaggs(self):
        """Test the Skaggs formula is correctly implemented.

        Formula: S = (sum_i(p_i * r_i))^2 / sum_i(p_i * r_i^2)
        """
        firing_rate = np.array([0.0, 2.0, 4.0, 2.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        spars = sparsity(firing_rate, occupancy)

        # Manual computation:
        # p_i = [0.25, 0.25, 0.25, 0.25]
        # numerator = (0.25*0 + 0.25*2 + 0.25*4 + 0.25*2)^2 = 2^2 = 4
        # denominator = 0.25*0 + 0.25*4 + 0.25*16 + 0.25*4 = 6
        # sparsity = 4/6 = 0.667
        assert_allclose(spars, 0.667, atol=0.01)

    def test_zero_denominator_returns_zero(self):
        """Should return 0.0 when all firing rates are zero."""
        firing_rate = np.zeros(100)
        occupancy = np.ones(100)
        spars = sparsity(firing_rate, occupancy)
        assert spars == 0.0

    def test_nan_values_handled(self):
        """Should handle NaN values in firing rate."""
        firing_rate = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        occupancy = np.ones(5)
        spars = sparsity(firing_rate, occupancy)
        # Should compute sparsity ignoring NaN
        assert not np.isnan(spars)
        assert 0.0 <= spars <= 1.0


# =============================================================================
# Test batch_spatial_information (population)
# =============================================================================


class TestBatchSpatialInformation:
    """Tests for batch_spatial_information() function."""

    def test_batch_shape(self, batch_firing_rates):
        """Should return (n_neurons,) array."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates):
        """Batch computation should match single-neuron results."""
        firing_rates, occupancy = batch_firing_rates
        batch_info = batch_spatial_information(firing_rates, occupancy)

        # Compute single-neuron for each
        single_info = np.array(
            [spatial_information(fr, occupancy) for fr in firing_rates]
        )

        assert_allclose(batch_info, single_info, rtol=1e-10)

    def test_batch_all_non_negative(self, batch_firing_rates):
        """All spatial information values should be non-negative."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert np.all(info >= 0.0)

    def test_batch_dtype(self, batch_firing_rates):
        """Should return float64 array."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.dtype == np.float64

    def test_batch_with_single_neuron(self, place_cell_like):
        """Should work with (1, n_bins) input."""
        firing_rate, occupancy = place_cell_like
        firing_rates = firing_rate[np.newaxis, :]  # (1, n_bins)
        info = batch_spatial_information(firing_rates, occupancy)
        assert info.shape == (1,)
        expected = spatial_information(firing_rate, occupancy)
        assert_allclose(info[0], expected, rtol=1e-10)

    def test_batch_uniform_vs_selective(self, batch_firing_rates):
        """Uniform neuron should have less info than selective neuron."""
        firing_rates, occupancy = batch_firing_rates
        info = batch_spatial_information(firing_rates, occupancy)
        # Neuron 0 is uniform, neuron 1 is selective
        assert info[0] < info[1]


# =============================================================================
# Test batch_sparsity (population)
# =============================================================================


class TestBatchSparsity:
    """Tests for batch_sparsity() function."""

    def test_batch_shape(self, batch_firing_rates):
        """Should return (n_neurons,) array."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.shape == (5,)

    def test_batch_matches_single(self, batch_firing_rates):
        """Batch computation should match single-neuron results."""
        firing_rates, occupancy = batch_firing_rates
        batch_spars = batch_sparsity(firing_rates, occupancy)

        # Compute single-neuron for each
        single_spars = np.array([sparsity(fr, occupancy) for fr in firing_rates])

        assert_allclose(batch_spars, single_spars, rtol=1e-10)

    def test_batch_all_in_range(self, batch_firing_rates):
        """All sparsity values should be in [0, 1]."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert np.all((spars >= 0.0) & (spars <= 1.0))

    def test_batch_dtype(self, batch_firing_rates):
        """Should return float64 array."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.dtype == np.float64

    def test_batch_with_single_neuron(self, place_cell_like):
        """Should work with (1, n_bins) input."""
        firing_rate, occupancy = place_cell_like
        firing_rates = firing_rate[np.newaxis, :]  # (1, n_bins)
        spars = batch_sparsity(firing_rates, occupancy)
        assert spars.shape == (1,)
        expected = sparsity(firing_rate, occupancy)
        assert_allclose(spars[0], expected, rtol=1e-10)

    def test_batch_uniform_vs_selective(self, batch_firing_rates):
        """Uniform neuron should have higher sparsity than selective neuron."""
        firing_rates, occupancy = batch_firing_rates
        spars = batch_sparsity(firing_rates, occupancy)
        # Neuron 0 is uniform, neuron 1 is selective
        assert spars[0] > spars[1]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_bin(self):
        """Should handle single-bin environment."""
        firing_rate = np.array([5.0])
        occupancy = np.array([1.0])

        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        assert info == 0.0  # No spatial information with one bin
        assert spars == 1.0  # All firing in "everywhere"

    def test_all_nan_firing_rate(self):
        """Should handle all-NaN firing rate."""
        firing_rate = np.array([np.nan, np.nan, np.nan])
        occupancy = np.array([1.0, 1.0, 1.0])

        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        assert info == 0.0
        assert spars == 0.0

    def test_inf_handling(self):
        """Should handle inf values gracefully."""
        firing_rate = np.array([1.0, 2.0, np.inf, 3.0])
        occupancy = np.array([1.0, 1.0, 1.0, 1.0])

        # Should not raise, may return inf or handle gracefully
        info = spatial_information(firing_rate, occupancy)
        spars = sparsity(firing_rate, occupancy)

        # Just verify no exceptions were raised
        assert isinstance(info, float)
        assert isinstance(spars, float)

    def test_empty_array_raises(self):
        """Should raise on empty arrays."""
        firing_rate = np.array([])
        occupancy = np.array([])

        with pytest.raises((ValueError, ZeroDivisionError)):
            spatial_information(firing_rate, occupancy)

        with pytest.raises((ValueError, ZeroDivisionError)):
            sparsity(firing_rate, occupancy)

    def test_mismatched_shapes_raises(self):
        """Should raise on mismatched array shapes."""
        firing_rate = np.array([1.0, 2.0, 3.0])
        occupancy = np.array([1.0, 1.0])

        with pytest.raises(ValueError):
            spatial_information(firing_rate, occupancy)

        with pytest.raises(ValueError):
            sparsity(firing_rate, occupancy)

    def test_batch_mismatched_shapes_raises(self):
        """Should raise on mismatched batch shapes."""
        firing_rates = np.ones((3, 100))
        occupancy = np.ones(50)  # Wrong size

        with pytest.raises(ValueError):
            batch_spatial_information(firing_rates, occupancy)

        with pytest.raises(ValueError):
            batch_sparsity(firing_rates, occupancy)


# =============================================================================
# Backwards Compatibility
# =============================================================================


class TestBackwardsCompatibility:
    """Test that _metrics.py matches existing place.py implementations."""

    def test_matches_place_skaggs_information(self, place_cell_like):
        """Should match neurospatial.encoding.place.skaggs_information."""
        from neurospatial.encoding.place import skaggs_information as legacy_skaggs

        firing_rate, occupancy = place_cell_like

        new_info = spatial_information(firing_rate, occupancy)
        legacy_info = legacy_skaggs(firing_rate, occupancy)

        assert_allclose(new_info, legacy_info, rtol=1e-10)

    def test_matches_place_sparsity(self, place_cell_like):
        """Should match neurospatial.encoding.place.sparsity."""
        from neurospatial.encoding.place import sparsity as legacy_sparsity

        firing_rate, occupancy = place_cell_like

        new_spars = sparsity(firing_rate, occupancy)
        legacy_spars = legacy_sparsity(firing_rate, occupancy)

        assert_allclose(new_spars, legacy_spars, rtol=1e-10)
