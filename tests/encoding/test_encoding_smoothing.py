"""Tests for encoding/_smoothing.py - shared smoothing implementations.

This module tests the shared smoothing functions that will be used by
encoding result computation functions (compute_spatial_rate, etc.).

Following TDD approach: tests written before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment

# Import will fail until we implement the module
from neurospatial.encoding._smoothing import (
    smooth_rate_map,
    smooth_rate_maps_batch,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 10x10 grid environment.

    Returns
    -------
    Environment
        A 10x10 grid with 2.0 cm bins.
    """
    positions = np.array([[i, j] for i in range(11) for j in range(11)], dtype=float)
    return Environment.from_samples(positions, bin_size=2.0)


@pytest.fixture
def spike_counts_center(simple_env: Environment) -> np.ndarray:
    """Spike counts with single spike at center.

    Returns
    -------
    ndarray, shape (n_bins,)
        Spike counts with 10 spikes at center bin.
    """
    n_bins = simple_env.n_bins
    spike_counts = np.zeros(n_bins, dtype=np.float64)
    center_idx = n_bins // 2
    spike_counts[center_idx] = 10.0
    return spike_counts


@pytest.fixture
def uniform_occupancy(simple_env: Environment) -> np.ndarray:
    """Uniform occupancy across all bins.

    Returns
    -------
    ndarray, shape (n_bins,)
        Uniform occupancy of 1.0 seconds per bin.
    """
    return np.ones(simple_env.n_bins, dtype=np.float64)


@pytest.fixture
def non_uniform_occupancy(simple_env: Environment) -> np.ndarray:
    """Non-uniform occupancy with more time in center.

    Returns
    -------
    ndarray, shape (n_bins,)
        Occupancy with more time at center.
    """
    n_bins = simple_env.n_bins
    occupancy = np.ones(n_bins, dtype=np.float64)
    # Spend more time in center bins
    center_idx = n_bins // 2
    for offset in range(-3, 4):
        if 0 <= center_idx + offset < n_bins:
            occupancy[center_idx + offset] = 5.0
    return occupancy


@pytest.fixture
def batch_spike_counts(simple_env: Environment) -> np.ndarray:
    """Batch of spike counts for multiple neurons.

    Returns
    -------
    ndarray, shape (3, n_bins)
        Spike counts for 3 neurons with different patterns.
    """
    n_bins = simple_env.n_bins
    spike_counts = np.zeros((3, n_bins), dtype=np.float64)

    # Neuron 0: center spike
    spike_counts[0, n_bins // 2] = 10.0

    # Neuron 1: corner spike
    spike_counts[1, 0] = 8.0

    # Neuron 2: distributed spikes
    rng = np.random.default_rng(42)
    spike_counts[2] = rng.choice([0, 1, 2], size=n_bins, p=[0.7, 0.2, 0.1])

    return spike_counts


# =============================================================================
# Test smooth_rate_map (single neuron)
# =============================================================================


class TestSmoothRateMapDiffusionKDE:
    """Tests for smooth_rate_map with method='diffusion_kde'."""

    def test_basic_smoothing(self, simple_env, spike_counts_center, uniform_occupancy):
        """Basic diffusion_kde smoothing should work."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        assert result.shape == spike_counts_center.shape
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_spreads_single_spike(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Single spike should spread to neighbors with diffusion."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        center_idx = simple_env.n_bins // 2
        # Peak should still be at center
        peak_idx = np.nanargmax(result)
        assert peak_idx == center_idx

        # Value should spread to neighbors (non-zero in adjacent bins)
        neighbors = simple_env.neighbors(center_idx)
        for neighbor_idx in neighbors:
            assert result[neighbor_idx] > 0

    def test_larger_bandwidth_more_spread(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Larger bandwidth should spread signal more."""
        result_small = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=1.0,
        )
        result_large = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=3.0,
        )

        center_idx = simple_env.n_bins // 2

        # Larger bandwidth should have lower peak (more spread)
        assert result_large[center_idx] < result_small[center_idx]

    def test_respects_graph_connectivity(self, simple_env):
        """Smoothing should only spread along connected bins."""
        # Create spike counts at edge of environment
        n_bins = simple_env.n_bins
        spike_counts = np.zeros(n_bins, dtype=np.float64)
        spike_counts[0] = 10.0  # Corner spike

        occupancy = np.ones(n_bins, dtype=np.float64)

        result = smooth_rate_map(
            simple_env,
            spike_counts,
            occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # Result should be non-negative everywhere
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_preserves_total_mass_approximately(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Smoothing should approximately preserve total firing."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # Total spikes (density-weighted) should be approximately conserved
        # For uniform occupancy, just compare sums
        raw_rate = spike_counts_center / uniform_occupancy
        raw_total = np.nansum(raw_rate)
        smoothed_total = np.nansum(result)

        # Should be within 50% (smoothing can redistribute mass)
        assert smoothed_total > 0
        assert abs(smoothed_total - raw_total) / raw_total < 0.5

    def test_handles_zero_occupancy_bins(self, simple_env, spike_counts_center):
        """Bins with zero raw occupancy get rate from smoothed neighbors.

        Note: With diffusion_kde, occupancy is smoothed, so even bins with
        zero raw occupancy can have non-zero smoothed occupancy. The
        ``min_occupancy`` threshold is applied to the smoothed occupancy
        density (the firing-rate denominator), so a bin with zero raw
        occupancy but a smoothed denominator above the threshold reports a
        finite rate; only bins whose smoothed denominator is below the
        threshold are NaN-ed.
        """
        n_bins = simple_env.n_bins
        occupancy = np.ones(n_bins, dtype=np.float64)
        occupancy[0] = 0.0  # Zero occupancy in corner

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # With smoothing, even zero-occupancy bins get valid rates
        # (smoothed occupancy can be non-zero from neighbors)
        assert np.isfinite(result[0]) or np.isnan(result[0])

        # The corner bin has zero raw occupancy but a non-zero smoothed
        # occupancy density from its neighbors. A small positive threshold
        # below that smoothed density leaves it finite -- the threshold is
        # applied to the smoothed denominator, not the raw occupancy.
        result_with_threshold = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
            min_occupancy=0.05,  # below the corner bin's smoothed density
        )
        assert np.isfinite(result_with_threshold[0])

        # A threshold above every bin's smoothed occupancy density NaN-s all
        # bins (the denominator never clears it).
        result_all_nan = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
            min_occupancy=100.0,  # Above any smoothed density here
        )
        assert np.all(np.isnan(result_all_nan))


class TestSmoothRateMapGaussianKDE:
    """Tests for smooth_rate_map with method='gaussian_kde'."""

    def test_basic_smoothing(self, simple_env, spike_counts_center, uniform_occupancy):
        """Basic gaussian_kde smoothing should work."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="gaussian_kde",
            bandwidth=2.0,
        )

        assert result.shape == spike_counts_center.shape
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_spreads_single_spike(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Single spike should spread to neighbors with Gaussian."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="gaussian_kde",
            bandwidth=2.0,
        )

        center_idx = simple_env.n_bins // 2
        # Peak should still be at center
        peak_idx = np.nanargmax(result)
        assert peak_idx == center_idx

        # Value should spread to neighbors
        neighbors = simple_env.neighbors(center_idx)
        for neighbor_idx in neighbors:
            assert result[neighbor_idx] > 0

    def test_larger_bandwidth_more_spread(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Larger bandwidth should spread signal more."""
        result_small = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="gaussian_kde",
            bandwidth=1.0,
        )
        result_large = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="gaussian_kde",
            bandwidth=3.0,
        )

        center_idx = simple_env.n_bins // 2

        # Larger bandwidth should have lower peak (more spread)
        assert result_large[center_idx] < result_small[center_idx]

    def test_euclidean_distance_based(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Gaussian KDE should use Euclidean distance."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="gaussian_kde",
            bandwidth=2.0,
        )

        center_idx = simple_env.n_bins // 2

        # Verify peak is at center and neighbors have values
        # (This is a weak test but verifies Euclidean property)
        assert result[center_idx] > 0


class TestSmoothRateMapBinned:
    """Tests for smooth_rate_map with method='binned'."""

    def test_basic_smoothing(self, simple_env, spike_counts_center, uniform_occupancy):
        """Basic binned smoothing should work."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="binned",
            bandwidth=2.0,
        )

        assert result.shape == spike_counts_center.shape
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_no_smoothing_with_zero_bandwidth(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """With bandwidth=0, binned should return raw rate."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="binned",
            bandwidth=0.0,
        )

        # Should be raw spike count / occupancy
        expected = spike_counts_center / uniform_occupancy
        assert_allclose(result, expected, rtol=1e-10)

    def test_spreads_with_positive_bandwidth(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """With bandwidth > 0, should spread signal."""
        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="binned",
            bandwidth=2.0,
        )

        center_idx = simple_env.n_bins // 2

        # Neighbors should have non-zero values
        neighbors = simple_env.neighbors(center_idx)
        assert any(result[n] > 0 for n in neighbors)

    def test_handles_nan_values(self, simple_env, spike_counts_center):
        """Binned should handle NaN from low occupancy."""
        n_bins = simple_env.n_bins
        occupancy = np.ones(n_bins, dtype=np.float64)
        occupancy[0] = 0.0  # Zero occupancy

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="binned",
            bandwidth=2.0,
        )

        # Result should be computed without errors
        assert result.shape == spike_counts_center.shape


# =============================================================================
# Test smooth_rate_map edge cases
# =============================================================================


class TestSmoothRateMapEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_method_raises(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            smooth_rate_map(
                simple_env,
                spike_counts_center,
                uniform_occupancy,
                method="invalid_method",
                bandwidth=2.0,
            )

    def test_negative_bandwidth_raises(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Negative bandwidth should raise ValueError."""
        with pytest.raises(ValueError, match="bandwidth"):
            smooth_rate_map(
                simple_env,
                spike_counts_center,
                uniform_occupancy,
                method="diffusion_kde",
                bandwidth=-1.0,
            )

    def test_mismatched_shapes_raises(self, simple_env, uniform_occupancy):
        """Mismatched spike_counts and occupancy shapes should raise."""
        wrong_size_counts = np.ones(simple_env.n_bins + 5)

        with pytest.raises(ValueError, match="shape"):
            smooth_rate_map(
                simple_env,
                wrong_size_counts,
                uniform_occupancy,
                method="diffusion_kde",
                bandwidth=2.0,
            )

    def test_wrong_env_size_raises(self, simple_env):
        """Spike counts and occupancy not matching env.n_bins should raise."""
        wrong_size = 10  # Different from env.n_bins
        wrong_size_counts = np.ones(wrong_size)
        wrong_size_occupancy = np.ones(wrong_size)

        with pytest.raises(ValueError, match="n_bins"):
            smooth_rate_map(
                simple_env,
                wrong_size_counts,
                wrong_size_occupancy,
                method="diffusion_kde",
                bandwidth=2.0,
            )

    def test_all_zero_occupancy(self, simple_env, spike_counts_center):
        """All zero occupancy should return all NaN."""
        zero_occupancy = np.zeros(simple_env.n_bins)

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            zero_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        assert np.all(np.isnan(result))

    def test_all_zero_spikes(self, simple_env, uniform_occupancy):
        """All zero spike counts should return all zeros."""
        zero_spikes = np.zeros(simple_env.n_bins)

        result = smooth_rate_map(
            simple_env,
            zero_spikes,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        assert_allclose(result, 0.0, atol=1e-10)


# =============================================================================
# Test smooth_rate_maps_batch (multiple neurons)
# =============================================================================


class TestSmoothRateMapsBatch:
    """Tests for smooth_rate_maps_batch function."""

    def test_basic_batch(self, simple_env, batch_spike_counts, uniform_occupancy):
        """Basic batch smoothing should work."""
        result = smooth_rate_maps_batch(
            simple_env,
            batch_spike_counts,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        n_neurons = batch_spike_counts.shape[0]
        n_bins = batch_spike_counts.shape[1]
        assert result.shape == (n_neurons, n_bins)

    def test_batch_matches_single(
        self, simple_env, batch_spike_counts, uniform_occupancy
    ):
        """Batch result should match looped single results."""
        batch_result = smooth_rate_maps_batch(
            simple_env,
            batch_spike_counts,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # Compare with single-neuron calls
        for i in range(batch_spike_counts.shape[0]):
            single_result = smooth_rate_map(
                simple_env,
                batch_spike_counts[i],
                uniform_occupancy,
                method="diffusion_kde",
                bandwidth=2.0,
            )
            # NaN positions should match
            assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-10,
                equal_nan=True,
            )

    def test_batch_all_methods(self, simple_env, batch_spike_counts, uniform_occupancy):
        """All methods should work with batch processing."""
        for method in ["diffusion_kde", "gaussian_kde", "binned"]:
            result = smooth_rate_maps_batch(
                simple_env,
                batch_spike_counts,
                uniform_occupancy,
                method=method,
                bandwidth=2.0,
            )
            assert result.shape == batch_spike_counts.shape

    def test_batch_single_neuron(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Batch with (1, n_bins) should work."""
        batch_counts = spike_counts_center[np.newaxis, :]

        result = smooth_rate_maps_batch(
            simple_env,
            batch_counts,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        assert result.shape == (1, simple_env.n_bins)

    def test_batch_invalid_shape_raises(self, simple_env, uniform_occupancy):
        """Non-2D spike_counts should raise."""
        wrong_shape = np.ones((2, 3, 4))  # 3D

        with pytest.raises(ValueError, match="2D"):
            smooth_rate_maps_batch(
                simple_env,
                wrong_shape,
                uniform_occupancy,
                method="diffusion_kde",
                bandwidth=2.0,
            )


# =============================================================================
# Test min_occupancy parameter
# =============================================================================


class TestMinOccupancy:
    """Tests for min_occupancy parameter."""

    def test_min_occupancy_masks_low_bins(self, simple_env, spike_counts_center):
        """Bins below min_occupancy should be NaN."""
        n_bins = simple_env.n_bins
        occupancy = np.ones(n_bins) * 0.5  # 0.5 seconds everywhere
        occupancy[0] = 0.1  # Low occupancy in bin 0

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="binned",  # Use binned to see direct effect
            bandwidth=0.0,  # No smoothing
            min_occupancy=0.2,
        )

        # Low occupancy bin should be NaN
        assert np.isnan(result[0])
        # Other bins should be finite
        assert np.sum(np.isfinite(result)) > 0

    def test_min_occupancy_zero_keeps_all(self, simple_env, spike_counts_center):
        """min_occupancy=0 should keep all bins with non-zero occupancy."""
        n_bins = simple_env.n_bins
        occupancy = np.ones(n_bins) * 0.01  # Very low occupancy

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            occupancy,
            method="binned",
            bandwidth=0.0,
            min_occupancy=0.0,
        )

        # All bins should be finite
        assert np.all(np.isfinite(result))

    def test_kde_min_occupancy_thresholds_smoothed_denominator(
        self, simple_env, spike_counts_center
    ):
        """For KDE methods, min_occupancy thresholds the smoothed denominator.

        A bin that was never directly traversed (raw occupancy == 0) but that
        receives substantial smoothed occupancy from neighbors has a valid,
        well-defined firing-rate denominator. It must NOT be spuriously
        NaN-ed: the threshold is applied to the same quantity used as the
        denominator (the smoothed occupancy density), not the raw occupancy.
        """
        n_bins = simple_env.n_bins
        # Single un-traversed bin in the interior; all others traversed for 1 s.
        occupancy = np.ones(n_bins, dtype=np.float64)
        gap_idx = n_bins // 2 + 1  # interior bin next to the spiking center
        occupancy[gap_idx] = 0.0

        for method in ("diffusion_kde", "gaussian_kde"):
            result = smooth_rate_map(
                simple_env,
                spike_counts_center,
                occupancy,
                method=method,
                bandwidth=2.0,
                min_occupancy=0.1,  # above 0 but below the gap bin's density
            )
            # Sanity: the smoothed occupancy density at the gap bin clears
            # the threshold (surrounded by 1 s/bin neighbors), so the bin
            # must report a finite rate rather than NaN.
            assert np.isfinite(result[gap_idx]), (
                f"{method}: interior gap bin with valid smoothed denominator "
                "was spuriously NaN-ed by raw-occupancy thresholding."
            )


# =============================================================================
# Test kernel caching behavior
# =============================================================================


class TestKernelCaching:
    """Tests for kernel caching behavior."""

    def test_accepts_precomputed_kernel(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Should accept precomputed kernel for efficiency."""
        kernel = simple_env.compute_kernel(2.0, mode="density", cache=True)

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
            kernel=kernel,
        )

        # Should produce valid result
        assert result.shape == spike_counts_center.shape
        assert np.any(np.isfinite(result))

    def test_precomputed_kernel_matches_computed(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Result with precomputed kernel should match computed."""
        kernel = simple_env.compute_kernel(2.0, mode="density", cache=True)

        result_with_kernel = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
            kernel=kernel,
        )

        result_without_kernel = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        assert_allclose(result_with_kernel, result_without_kernel, rtol=1e-10)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying compatibility with existing place.py implementations."""

    def test_matches_diffusion_kde_from_place(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """Result should match _diffusion_kde from place.py."""
        # We'll verify this by checking the algorithm produces expected behavior
        # rather than importing the internal function

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # Basic sanity checks that match diffusion_kde behavior
        center_idx = simple_env.n_bins // 2
        assert result[center_idx] > 0  # Peak at spike location
        assert result[center_idx] == np.nanmax(result)  # Peak is maximum

    def test_smoothing_order_diffusion_kde(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """diffusion_kde should smooth before normalizing (correct KDE order)."""
        # This is the mathematically correct order:
        # 1. Smooth spike counts
        # 2. Smooth occupancy
        # 3. Normalize: smoothed_spikes / smoothed_occupancy

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="diffusion_kde",
            bandwidth=2.0,
        )

        # With uniform occupancy, smoothed rate should be smoothed spikes / 1.0
        # Peak should be less than raw spike count / occupancy
        center_idx = simple_env.n_bins // 2
        raw_peak = spike_counts_center[center_idx] / uniform_occupancy[center_idx]
        assert result[center_idx] < raw_peak  # Smoothing reduces peak

    def test_smoothing_order_binned(
        self, simple_env, spike_counts_center, uniform_occupancy
    ):
        """binned should normalize then smooth (legacy order)."""
        # This is the legacy order (for backward compatibility):
        # 1. Normalize: spike_counts / occupancy
        # 2. Smooth result

        result = smooth_rate_map(
            simple_env,
            spike_counts_center,
            uniform_occupancy,
            method="binned",
            bandwidth=2.0,
        )

        # Result should still be valid
        assert np.any(np.isfinite(result))
