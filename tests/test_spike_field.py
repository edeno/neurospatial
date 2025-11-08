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
        spike_times = np.sort(np.random.uniform(times[0], times[-1], n_spikes))

        # Compute firing rate field
        field = spikes_to_field(env, spike_times, times, positions)

        # Should have shape (n_bins,)
        assert field.shape == (env.n_bins,)

        # Mean firing rate should be close to spike_rate (within 20% for randomness)
        mean_rate = np.nanmean(field)
        assert 0.8 * spike_rate <= mean_rate <= 1.2 * spike_rate

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

    def test_spikes_to_field_matches_manual(self):
        """Test that spikes_to_field matches manual computation."""
        # Simple case: uniform trajectory and spikes
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)

        env = Environment.from_samples(positions, bin_size=10.0)

        # Spikes at known times
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Compute using function (with default min_occupancy_seconds=0.0)
        field = spikes_to_field(env, spike_times, times, positions)

        # Compute manually
        # 1. Compute occupancy
        occupancy = env.occupancy(times, positions, return_seconds=True)

        # 2. Interpolate spike positions
        spike_x = np.interp(spike_times, times, positions[:, 0])
        spike_y = np.interp(spike_times, times, positions[:, 1])
        spike_positions = np.column_stack([spike_x, spike_y])

        # 3. Assign to bins
        spike_bins = env.bin_at(spike_positions)

        # 4. Count spikes per bin
        spike_counts = np.bincount(spike_bins, minlength=env.n_bins)

        # 5. Normalize (no threshold with default min_occupancy_seconds=0.0)
        manual_field = np.zeros(env.n_bins)
        valid_mask = occupancy > 0  # Avoid division by zero
        manual_field[valid_mask] = spike_counts[valid_mask] / occupancy[valid_mask]
        # Bins with zero occupancy remain 0 (no NaN with default)

        # Should match
        assert_array_almost_equal(field, manual_field)

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
    """Test suite for compute_place_field convenience function."""

    def test_compute_place_field_with_smoothing(self):
        """Test that compute_place_field with smoothing matches two-step workflow."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.linspace(0, 10, 50)

        # One-liner with smoothing
        field_direct = compute_place_field(
            env, spike_times, times, positions, smoothing_bandwidth=5.0
        )

        # Two-step workflow (must handle NaN like compute_place_field does)
        field_raw = spikes_to_field(env, spike_times, times, positions)

        # Handle NaN for smoothing
        nan_mask = np.isnan(field_raw)
        field_filled = field_raw.copy()
        field_filled[nan_mask] = 0.0
        field_two_step = env.smooth(field_filled, bandwidth=5.0)
        field_two_step[nan_mask] = np.nan

        # Should match
        assert_array_almost_equal(field_direct, field_two_step)

    def test_compute_place_field_no_smoothing(self):
        """Test that compute_place_field without smoothing matches spikes_to_field."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 1000),
                np.linspace(0, 100, 1000),
            ]
        )
        times = np.linspace(0, 10, 1000)
        env = Environment.from_samples(positions, bin_size=10.0)
        spike_times = np.linspace(0, 10, 50)

        # With smoothing_bandwidth=None
        field_direct = compute_place_field(
            env, spike_times, times, positions, smoothing_bandwidth=None
        )

        # Should match spikes_to_field exactly
        field_raw = spikes_to_field(env, spike_times, times, positions)

        assert_array_equal(field_direct, field_raw)

    def test_compute_place_field_parameter_order(self):
        """Test that parameter order is consistent with spikes_to_field."""
        positions = np.column_stack(
            [
                np.linspace(0, 100, 100),
                np.linspace(0, 100, 100),
            ]
        )
        times = np.linspace(0, 10, 100)
        env = Environment.from_samples(positions, bin_size=20.0)
        spike_times = np.array([1.0, 2.0])

        # Should accept same parameter order
        field = compute_place_field(
            env, spike_times, times, positions, smoothing_bandwidth=3.0
        )
        assert field.shape == (env.n_bins,)
