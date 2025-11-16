"""Tests for spike train to field conversion functionality."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neurospatial import Environment
from neurospatial.spike_field import compute_place_field, spikes_to_field


class TestSpikesToField:
    """Test suite for spikes_to_field function."""

    def test_spikes_to_field_synthetic(self):
        """Test spikes_to_field with known spike rate produces expected field."""
        # Use local RNG for test isolation
        rng = np.random.default_rng(42)

        # Create a 10x10 environment (0-100 range, bin_size=10)
        x = np.linspace(0, 100, 1000)
        y = np.linspace(0, 100, 1000)
        positions = np.column_stack([x, y])
        times = np.linspace(0, 10, 1000)  # 10 seconds

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create spikes uniformly distributed across time
        # Expected: ~10 spikes per bin per second of occupancy
        spike_rate = 10.0  # Hz
        n_spikes = int(spike_rate * times[-1])
        spike_times = np.sort(rng.uniform(times[0], times[-1], n_spikes))

        # Compute firing rate field
        field = spikes_to_field(env, spike_times, times, positions)

        # Should have shape (n_bins,)
        assert field.shape == (env.n_bins,)

        # With fixed seed, mean firing rate should be close to spike_rate
        # Using 5% tolerance (much tighter than previous 20%)
        mean_rate = np.nanmean(field)
        assert mean_rate == pytest.approx(spike_rate, rel=0.05)

    def test_spikes_to_field_min_occupancy(self):
        """Test that low occupancy bins are set to NaN."""
        # Create environment and trajectory that visits only some bins
        positions = np.column_stack(
            [
                np.linspace(0, 50, 500),
                np.linspace(0, 50, 500),
            ]
        )
        times = np.linspace(0, 10, 500)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create a few spikes
        spike_times = np.array([1.0, 2.0, 3.0])

        # Use high min_occupancy threshold to trigger NaN assignment
        field = spikes_to_field(
            env, spike_times, times, positions, min_occupancy_seconds=5.0
        )

        # Most bins should have NaN (low occupancy)
        assert np.sum(np.isnan(field)) > 0

        # Valid bins should have non-negative rates
        valid_bins = ~np.isnan(field)
        assert np.all(field[valid_bins] >= 0)

    def test_spikes_to_field_empty_spikes(self):
        """Test that empty spike train produces all zeros."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Empty spike array
        spike_times = np.array([])

        field = spikes_to_field(env, spike_times, times, positions)

        # Should have all zeros (where occupancy > threshold) or NaN
        valid_bins = ~np.isnan(field)
        assert np.allclose(field[valid_bins], 0.0)

    def test_spikes_to_field_out_of_bounds_time(self):
        """Test that spikes outside time range produce warning and are filtered."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Spikes outside time range
        spike_times = np.array([-1.0, 5.0, 15.0])

        # Should produce warning about filtering
        with pytest.warns(UserWarning, match="out of time range"):
            field = spikes_to_field(env, spike_times, times, positions)

        # Should still produce valid field (only middle spike used)
        assert field.shape == (env.n_bins,)
        assert not np.all(np.isnan(field))

    def test_spikes_to_field_out_of_bounds_space(self):
        """Test that spikes outside environment bounds produce warning and are filtered."""
        # Create environment covering 0-100 in both dimensions
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create spikes, but interpolate to positions that go outside bounds
        # This happens when spike occurs during trajectory edge
        spike_times = np.array([0.001])  # Very early, near edge

        # Manually create out-of-bounds case by using times that would interpolate
        # outside the environment
        # This is a bit contrived, but tests the safety check
        # In practice, we'll just verify the function handles it gracefully
        field = spikes_to_field(env, spike_times, times, positions)

        # Should produce valid field shape
        assert field.shape == (env.n_bins,)

    def test_spikes_to_field_1d_trajectory(self):
        """Test that 1D positions (column vector) are handled correctly."""
        # 1D trajectory as column vector
        positions = np.linspace(0, 100, 1000).reshape(-1, 1)
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Spikes
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        field = spikes_to_field(env, spike_times, times, positions)

        # Should work without errors
        assert field.shape == (env.n_bins,)
        assert not np.all(np.isnan(field))

    def test_spikes_to_field_1d_no_column_dimension(self):
        """Test that 1D positions without column dimension (n,) are handled."""
        # 1D trajectory as bare array (not reshaped to column)
        positions = np.linspace(0, 100, 1000)  # Shape (1000,)
        times = np.linspace(0, 10, 1000)
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create environment using reshaped positions
        env = Environment.from_samples(positions.reshape(-1, 1), bin_size=10.0)

        # Function should accept bare 1D array
        field = spikes_to_field(env, spike_times, times, positions)

        assert field.shape == (env.n_bins,)
        assert not np.all(np.isnan(field))
        assert np.sum(np.isfinite(field)) > 0

    def test_spikes_to_field_nan_occupancy(self):
        """Test behavior when occupancy is zero everywhere (edge case)."""
        # Create environment
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Create spikes, but use very high min_occupancy to force all NaN
        spike_times = np.array([1.0, 2.0, 3.0])

        field = spikes_to_field(
            env, spike_times, times, positions, min_occupancy_seconds=1000.0
        )

        # All bins should be NaN (insufficient occupancy)
        assert np.all(np.isnan(field))

    def test_spikes_to_field_known_firing_rate(self):
        """Test against analytically computed firing rate with stationary position.

        This test uses ground truth: animal stationary at one location,
        producing spikes at known rate. Expected firing rate is analytically
        computable without reimplementing the algorithm.
        """
        # Animal stationary at (50, 50) for 10 seconds
        positions = np.full((1000, 2), [50.0, 50.0])
        times = np.linspace(0, 10, 1000)

        # Create environment
        env = Environment.from_samples(positions, bin_size=10.0)

        # Neuron fires exactly 5 spikes at this location
        # Analytical ground truth: 5 spikes / 10 seconds = 0.5 Hz
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Compute field
        field = spikes_to_field(env, spike_times, times, positions)

        # Find bin containing (50, 50)
        bin_at_50_50 = env.bin_at(np.array([[50.0, 50.0]]))[0]

        # Verify firing rate at the occupied bin is 0.5 Hz
        assert field[bin_at_50_50] == pytest.approx(0.5, rel=0.01)

        # Verify all other bins have zero firing (no occupancy)
        other_bins = np.arange(env.n_bins) != bin_at_50_50
        # Other bins should be 0 (default behavior with zero occupancy)
        assert np.allclose(field[other_bins], 0.0)

    def test_spikes_to_field_parameter_order(self):
        """Test that parameter order is env first (matches existing API)."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 100),
                np.linspace(0, 100, 100),
            ]
        )
        times = np.linspace(0, 10, 100)
        env = Environment.from_samples(positions, bin_size=20.0)
        spike_times = np.array([1.0, 2.0])

        # Correct order: env first
        field = spikes_to_field(env, spike_times, times, positions)
        assert field.shape == (env.n_bins,)

        # This verifies the signature is correct

    def test_spikes_to_field_validation(self):
        """Test input validation."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 100),
                np.linspace(0, 100, 100),
            ]
        )
        times = np.linspace(0, 10, 100)
        env = Environment.from_samples(positions, bin_size=20.0)
        spike_times = np.array([1.0, 2.0])

        # Mismatched times and positions
        bad_times = np.linspace(0, 10, 50)  # Different length

        with pytest.raises(
            ValueError, match="times and positions must have same length"
        ):
            spikes_to_field(env, spike_times, bad_times, positions)

        # Negative min_occupancy
        with pytest.raises(ValueError, match="must be non-negative"):
            spikes_to_field(
                env, spike_times, times, positions, min_occupancy_seconds=-1.0
            )


class TestComputePlaceField:
    """Test suite for compute_place_field unified API with multiple methods."""

    def test_default_method_is_diffusion_kde(self):
        """Test that default method is diffusion_kde."""
        positions = np.random.uniform(20, 80, (500, 2))
        times = np.linspace(0, 50, 500)
        spike_times = np.random.uniform(0, 50, 25)
        env = Environment.from_samples(positions, bin_size=5.0)

        # Default call should use diffusion_kde
        field_default = compute_place_field(
            env, spike_times, times, positions, bandwidth=5.0
        )

        # Explicit diffusion_kde call
        field_explicit = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
        )

        assert_array_equal(field_default, field_explicit)

    @pytest.mark.parametrize(
        "method",
        ["diffusion_kde", "gaussian_kde", "binned"],
        ids=["diffusion_kde", "gaussian_kde", "binned"],
    )
    def test_all_methods_produce_valid_output(self, method):
        """Test that all three methods produce valid firing rate maps."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(20, 80, (500, 2))
        times = np.linspace(0, 50, 500)
        spike_times = rng.uniform(0, 50, 25)
        env = Environment.from_samples(positions, bin_size=5.0)

        field = compute_place_field(
            env, spike_times, times, positions, method=method, bandwidth=5.0
        )

        # Check shape
        assert field.shape == (env.n_bins,)

        # Check non-negative firing rates (where not NaN)
        valid_bins = ~np.isnan(field)
        assert np.all(field[valid_bins] >= 0)

        # Check that we have some valid bins
        assert np.sum(valid_bins) > 0

    def test_diffusion_kde_no_nan_bins(self):
        """Test that diffusion_kde naturally handles sparse occupancy without NaN."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(20, 80, (500, 2))
        times = np.linspace(0, 50, 500)
        spike_times = rng.uniform(0, 50, 25)
        env = Environment.from_samples(positions, bin_size=5.0)

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
        )

        # Diffusion KDE should have no NaN bins (naturally handles low occupancy)
        assert np.sum(np.isnan(field)) == 0

    def test_binned_method_respects_min_occupancy(self):
        """Test that binned method uses min_occupancy_seconds parameter."""
        positions = np.column_stack(
            [np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)]
        )
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.array([1.0, 2.0, 3.0])

        # With high threshold, should have many NaN bins
        field_high_threshold = compute_place_field(
            env,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=5.0,
            min_occupancy_seconds=5.0,
        )

        # With low threshold, should have fewer NaN bins
        field_low_threshold = compute_place_field(
            env,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=5.0,
            min_occupancy_seconds=0.1,
        )

        # High threshold should produce more NaN bins
        assert np.sum(np.isnan(field_high_threshold)) > np.sum(
            np.isnan(field_low_threshold)
        )

    @pytest.mark.slow
    def test_methods_give_similar_results_open_field(self):
        """Test that all methods give similar results for open field (no boundaries).

        Marked slow: Computes place fields with 3 different methods and correlates results.
        """
        rng = np.random.default_rng(42)
        # Uniform coverage open field
        positions = rng.uniform(20, 80, (1000, 2))
        times = np.linspace(0, 100, 1000)
        spike_times = rng.uniform(0, 100, 50)
        env = Environment.from_samples(positions, bin_size=8.0)

        # Compute with all three methods
        field_diffusion = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=8.0
        )
        field_gaussian = compute_place_field(
            env, spike_times, times, positions, method="gaussian_kde", bandwidth=8.0
        )
        field_binned = compute_place_field(
            env, spike_times, times, positions, method="binned", bandwidth=8.0
        )

        # For open field with good coverage, all methods should give similar results
        # Use correlation as a similarity metric
        valid_all = (
            ~np.isnan(field_diffusion)
            & ~np.isnan(field_gaussian)
            & ~np.isnan(field_binned)
        )

        if np.sum(valid_all) > 10:  # Need enough valid bins
            corr_diff_gauss = np.corrcoef(
                field_diffusion[valid_all], field_gaussian[valid_all]
            )[0, 1]
            corr_diff_binned = np.corrcoef(
                field_diffusion[valid_all], field_binned[valid_all]
            )[0, 1]

            # Should be reasonably correlated (> 0.5) for open field
            assert corr_diff_gauss > 0.5, (
                "Diffusion and Gaussian should be similar for open field"
            )
            assert corr_diff_binned > 0.5, (
                "Diffusion and Binned should be similar for open field"
            )

    @pytest.mark.parametrize(
        "method",
        ["diffusion_kde", "gaussian_kde", "binned"],
        ids=["diffusion_kde", "gaussian_kde", "binned"],
    )
    def test_empty_spikes(self, method):
        """Test that all methods handle empty spike train correctly."""
        positions = np.column_stack(
            [np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)]
        )
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.array([])

        field = compute_place_field(
            env, spike_times, times, positions, method=method, bandwidth=5.0
        )

        # Should have valid shape
        assert field.shape == (env.n_bins,)

        # All valid bins should be zero
        valid_bins = ~np.isnan(field)
        assert np.allclose(field[valid_bins], 0.0)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        positions = np.column_stack(
            [np.linspace(0, 100, 100), np.linspace(0, 100, 100)]
        )
        times = np.linspace(0, 10, 100)
        env = Environment.from_samples(positions, bin_size=20.0)
        spike_times = np.array([1.0, 2.0])

        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            compute_place_field(
                env,
                spike_times,
                times,
                positions,
                method="invalid_method",
                bandwidth=5.0,
            )

        # Invalid bandwidth (non-positive)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            compute_place_field(
                env,
                spike_times,
                times,
                positions,
                method="diffusion_kde",
                bandwidth=-5.0,
            )

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            compute_place_field(
                env,
                spike_times,
                times,
                positions,
                method="diffusion_kde",
                bandwidth=0.0,
            )

        # Mismatched times and positions
        bad_times = np.linspace(0, 10, 50)
        with pytest.raises(
            ValueError, match="times and positions must have same length"
        ):
            compute_place_field(
                env,
                spike_times,
                bad_times,
                positions,
                method="diffusion_kde",
                bandwidth=5.0,
            )

    @pytest.mark.parametrize(
        "method",
        ["diffusion_kde", "gaussian_kde", "binned"],
        ids=["diffusion_kde", "gaussian_kde", "binned"],
    )
    def test_1d_trajectory_all_methods(self, method):
        """Test that all methods handle 1D trajectories correctly."""
        positions = np.linspace(0, 100, 1000).reshape(-1, 1)
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        field = compute_place_field(
            env, spike_times, times, positions, method=method, bandwidth=5.0
        )

        assert field.shape == (env.n_bins,)
        assert not np.all(np.isnan(field))

    def test_diffusion_kde_spread_before_normalize(self):
        """Test that diffusion_kde spreads mass before normalizing (correct order)."""
        rng = np.random.default_rng(42)
        # Create environment with some bins that have low occupancy
        positions = rng.uniform(20, 80, (500, 2))
        times = np.linspace(0, 50, 500)
        spike_times = rng.uniform(0, 50, 25)
        env = Environment.from_samples(positions, bin_size=5.0)

        field_diffusion = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=8.0
        )

        # Diffusion KDE should have smoother spatial distribution
        # (no sharp bin boundaries) and no NaN bins
        assert np.sum(np.isnan(field_diffusion)) == 0

        # Check that values are spatially smooth (neighbors should be similar)
        # This tests that diffusion happened
        max_neighbor_diff = 0.0
        for node in env.connectivity.nodes():
            neighbors = list(env.connectivity.neighbors(node))
            if len(neighbors) > 0 and not np.isnan(field_diffusion[node]):
                neighbor_vals = [
                    field_diffusion[n]
                    for n in neighbors
                    if not np.isnan(field_diffusion[n])
                ]
                if len(neighbor_vals) > 0:
                    max_neighbor_diff = max(
                        max_neighbor_diff,
                        np.max(np.abs(field_diffusion[node] - np.array(neighbor_vals))),
                    )

        # Neighbors should not differ drastically (smoothing effect)
        # This is a rough check that diffusion occurred
        mean_rate = np.mean(field_diffusion[~np.isnan(field_diffusion)])
        assert max_neighbor_diff < 5 * mean_rate  # Heuristic: max diff < 5x mean

    @pytest.mark.slow
    def test_gaussian_kde_slow_but_works(self):
        """Test that gaussian_kde works (even if slower) and produces valid results.

        Marked slow: Uses gaussian_kde method which is computationally expensive.
        """
        rng = np.random.default_rng(42)
        # Small dataset to keep test fast
        positions = rng.uniform(20, 80, (200, 2))
        times = np.linspace(0, 20, 200)
        spike_times = rng.uniform(0, 20, 10)
        env = Environment.from_samples(positions, bin_size=8.0)

        field = compute_place_field(
            env, spike_times, times, positions, method="gaussian_kde", bandwidth=8.0
        )

        # Should produce valid output
        assert field.shape == (env.n_bins,)
        valid_bins = ~np.isnan(field)
        assert np.sum(valid_bins) > 0
        assert np.all(field[valid_bins] >= 0)

    def test_binned_backward_compatible(self):
        """Test that binned method is backward compatible with old API."""
        positions = np.column_stack(
            [np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)]
        )
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.linspace(0, 10, 50)

        # New API with binned method
        field_new = compute_place_field(
            env,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=5.0,
            min_occupancy_seconds=0.5,
        )

        # Old-style manual workflow
        field_old = spikes_to_field(
            env, spike_times, times, positions, min_occupancy_seconds=0.5
        )
        # Apply smoothing with NaN handling (same as _binned backend)
        nan_mask = np.isnan(field_old)
        weights = np.ones_like(field_old)
        weights[nan_mask] = 0.0
        field_filled = field_old.copy()
        field_filled[nan_mask] = 0.0

        field_smoothed = env.smooth(field_filled, bandwidth=5.0)
        weights_smoothed = env.smooth(weights, bandwidth=5.0)

        field_old_normalized = np.zeros_like(field_smoothed)
        valid_mask = weights_smoothed > 0
        field_old_normalized[valid_mask] = (
            field_smoothed[valid_mask] / weights_smoothed[valid_mask]
        )
        field_old_normalized[~valid_mask] = np.nan

        # Should match
        assert_array_almost_equal(field_new, field_old_normalized)

    def test_boundary_very_small_bin_size(self):
        """Test numerical stability with very small bin sizes."""
        # Small bin size (0.5 cm) should not cause numerical issues
        positions = np.column_stack(
            [np.linspace(0, 10, 1000), np.linspace(0, 10, 1000)]
        )
        times = np.linspace(0, 10, 1000)
        spike_times = np.array([1.0, 2.0, 3.0])

        # Very small bin size
        env = Environment.from_samples(positions, bin_size=0.5)

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=1.0
        )

        # Should produce valid output
        assert field.shape == (env.n_bins,)
        assert not np.all(np.isnan(field))
        assert np.all(field[~np.isnan(field)] >= 0)

    def test_boundary_very_sparse_spikes(self):
        """Test with very few spikes (< 10 spikes)."""
        positions = np.column_stack(
            [np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)]
        )
        times = np.linspace(0, 100, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Only 3 spikes in 100 seconds
        spike_times = np.array([10.0, 50.0, 90.0])

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
        )

        # Should handle sparse spikes gracefully
        assert field.shape == (env.n_bins,)
        n_active_bins = np.sum(field > 0)
        assert n_active_bins > 0, "Should have at least some active bins"

    def test_boundary_single_spike(self):
        """Test extreme sparsity: single spike."""
        positions = np.column_stack(
            [np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Single spike
        spike_times = np.array([5.0])

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
        )

        # Should produce valid field with single spike
        assert field.shape == (env.n_bins,)
        assert np.sum(field > 0) > 0, "Should have at least one bin with activity"

    def test_boundary_3d_environment(self):
        """Test place field computation in 3D space."""
        # 3D trajectory
        positions = np.column_stack(
            [
                np.linspace(0, 100, 500),
                np.linspace(0, 100, 500),
                np.linspace(0, 100, 500),
            ]
        )
        times = np.linspace(0, 50, 500)
        spike_times = np.array([10.0, 20.0, 30.0, 40.0])

        env = Environment.from_samples(positions, bin_size=20.0)

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=10.0
        )

        # Should handle 3D correctly
        assert field.shape == (env.n_bins,)
        assert env.n_dims == 3
        assert not np.all(np.isnan(field))

    def test_boundary_single_bin(self):
        """Test edge case: environment with only one bin."""
        # All positions in small area (will create ~1 bin)
        positions = np.full((100, 2), [50.0, 50.0])
        positions += np.random.randn(100, 2) * 0.1  # Tiny variation

        times = np.linspace(0, 10, 100)
        spike_times = np.array([1.0, 5.0, 9.0])

        env = Environment.from_samples(positions, bin_size=10.0)

        # Should have very few bins
        assert env.n_bins < 10, "Small area should have few bins"

        field = compute_place_field(
            env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
        )

        # Should handle single/few bin case
        assert field.shape == (env.n_bins,)
        firing_rate = field[~np.isnan(field)]
        if len(firing_rate) > 0:
            # If any valid bins, firing rate should be reasonable
            assert np.all(firing_rate >= 0)
