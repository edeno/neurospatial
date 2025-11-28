"""Tests for DecodingResult container.

These tests verify that DecodingResult correctly stores decoding results
and computes derived properties (MAP, mean, entropy) via cached properties.
"""

import numpy as np
import pytest


class TestDecodingResultBasic:
    """Test basic DecodingResult instantiation and properties."""

    def test_decoding_result_stores_posterior(self, small_2d_env):
        """DecodingResult should store the posterior array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.posterior is posterior
        assert result.posterior.shape == (n_time_bins, n_bins)

    def test_decoding_result_stores_env(self, small_2d_env):
        """DecodingResult should store the environment reference."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.env is small_2d_env

    def test_decoding_result_stores_times(self, small_2d_env):
        """DecodingResult should store optional times array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.arange(n_time_bins) * 0.025

        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        assert result.times is times
        np.testing.assert_array_equal(result.times, times)

    def test_decoding_result_times_defaults_to_none(self, small_2d_env):
        """DecodingResult.times should default to None."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.times is None

    def test_n_time_bins_property(self, small_2d_env):
        """n_time_bins property should return correct count."""
        from neurospatial.decoding import DecodingResult

        for n_time_bins in [1, 5, 10, 100]:
            posterior = (
                np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
            )
            result = DecodingResult(posterior=posterior, env=small_2d_env)
            assert result.n_time_bins == n_time_bins


class TestMapEstimate:
    """Test MAP (maximum a posteriori) estimate computation."""

    def test_map_estimate_shape(self, small_2d_env):
        """map_estimate should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_estimate.shape == (n_time_bins,)
        assert result.map_estimate.dtype in (np.int64, np.intp)

    def test_map_estimate_delta_posterior(self, small_2d_env):
        """map_estimate should return correct bin for delta (one-hot) posterior."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, n_bins - 1, 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_equal(result.map_estimate, expected_bins)

    def test_map_estimate_identity_posterior(self, small_2d_env):
        """map_estimate on identity posterior should return diagonal indices."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix (10x10) if env has enough bins
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result.map_estimate, expected)


class TestMapPosition:
    """Test MAP position in environment coordinates."""

    def test_map_position_shape(self, small_2d_env):
        """map_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.map_position.dtype == np.float64

    def test_map_position_matches_bin_centers(self, small_2d_env):
        """map_position should equal bin_centers[map_estimate]."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors at known bins
        posterior = np.zeros((n_time_bins, n_bins))
        known_bins = [0, 1, 2, 3, 4]
        for t, bin_idx in enumerate(known_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected_positions = small_2d_env.bin_centers[known_bins]
        np.testing.assert_array_almost_equal(result.map_position, expected_positions)


class TestMeanPosition:
    """Test posterior mean position computation."""

    def test_mean_position_shape(self, small_2d_env):
        """mean_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.mean_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.mean_position.dtype == np.float64

    def test_mean_position_delta_equals_map(self, small_2d_env):
        """For delta posteriors, mean_position should equal map_position."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(result.mean_position, result.map_position)

    def test_mean_position_uniform_is_centroid(self, small_2d_env):
        """For uniform posterior, mean_position should be centroid of all bins."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: centroid of all bin centers
        expected_centroid = small_2d_env.bin_centers.mean(axis=0)
        for t in range(n_time_bins):
            np.testing.assert_array_almost_equal(
                result.mean_position[t], expected_centroid
            )

    def test_mean_position_two_bin_weighted(self, small_2d_env):
        """Mean position with two bins should be weighted average."""
        from neurospatial.decoding import DecodingResult

        n_bins = small_2d_env.n_bins
        if n_bins < 2:
            pytest.skip("Need at least 2 bins for this test")

        # Put 75% mass on bin 0, 25% on bin 1
        posterior = np.zeros((1, n_bins))
        posterior[0, 0] = 0.75
        posterior[0, 1] = 0.25

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: 0.75 * bin_centers[0] + 0.25 * bin_centers[1]
        expected = (
            0.75 * small_2d_env.bin_centers[0] + 0.25 * small_2d_env.bin_centers[1]
        )
        np.testing.assert_array_almost_equal(result.mean_position[0], expected)


class TestUncertainty:
    """Test posterior entropy (uncertainty) computation."""

    def test_uncertainty_shape(self, small_2d_env):
        """uncertainty should return (n_time_bins,) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.uncertainty.shape == (n_time_bins,)
        assert result.uncertainty.dtype == np.float64

    def test_uncertainty_delta_is_zero(self, small_2d_env):
        """Delta (one-hot) posterior should have zero entropy."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(result.uncertainty, np.zeros(n_time_bins))

    def test_uncertainty_uniform_is_maximum(self, small_2d_env):
        """Uniform posterior should have maximum entropy = log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        max_entropy = np.log2(n_bins)
        np.testing.assert_array_almost_equal(
            result.uncertainty, np.full(n_time_bins, max_entropy)
        )

    def test_uncertainty_bounds(self, small_2d_env):
        """Uncertainty should be bounded: 0 <= entropy <= log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        n_bins = small_2d_env.n_bins

        # Random posterior (normalized)
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert np.all(result.uncertainty >= 0)
        assert np.all(result.uncertainty <= np.log2(n_bins) + 1e-10)

    def test_uncertainty_handles_exact_zeros(self, small_2d_env):
        """Uncertainty should handle exact zeros without NaN (mask-based)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        # Posterior with many exact zeros
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[:, 0] = 0.5
        posterior[:, 1] = 0.5

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Should not be NaN
        assert np.all(np.isfinite(result.uncertainty))
        # Entropy of 50-50 distribution = 1 bit
        np.testing.assert_array_almost_equal(result.uncertainty, np.ones(n_time_bins))


class TestCachedPropertyBehavior:
    """Test that cached properties work correctly."""

    def test_map_estimate_is_cached(self, small_2d_env):
        """map_estimate should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Access twice
        first_call = result.map_estimate
        second_call = result.map_estimate

        # Should be the same object (cached)
        assert first_call is second_call

    def test_map_position_is_cached(self, small_2d_env):
        """map_position should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.map_position
        second_call = result.map_position

        assert first_call is second_call

    def test_mean_position_is_cached(self, small_2d_env):
        """mean_position should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.mean_position
        second_call = result.mean_position

        assert first_call is second_call

    def test_uncertainty_is_cached(self, small_2d_env):
        """uncertainty should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.uncertainty
        second_call = result.uncertainty

        assert first_call is second_call


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self, small_2d_env):
        """DecodingResult should work with a single time bin."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((1, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.n_time_bins == 1
        assert result.map_estimate.shape == (1,)
        assert result.map_position.shape == (1, small_2d_env.n_dims)
        assert result.mean_position.shape == (1, small_2d_env.n_dims)
        assert result.uncertainty.shape == (1,)

    def test_single_bin_environment(self):
        """DecodingResult should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create a minimal single-bin environment
        env = Environment.from_samples(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            bin_size=10.0,  # Large bin size to get single bin
        )

        if env.n_bins == 1:
            n_time_bins = 5
            posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

            result = DecodingResult(posterior=posterior, env=env)

            assert result.n_time_bins == n_time_bins
            np.testing.assert_array_equal(result.map_estimate, np.zeros(n_time_bins))
            # Entropy of single bin is 0 (log2(1) = 0)
            np.testing.assert_array_almost_equal(
                result.uncertainty, np.zeros(n_time_bins)
            )
        else:
            pytest.skip("Could not create single-bin environment")

    def test_identity_posterior_success_criteria(self, small_2d_env):
        """Verify success criteria from TASKS.md with identity posterior."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix as posterior per TASKS.md success criteria
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)
        times = np.arange(n_time_bins) * 0.025

        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        # Success criteria from TASKS.md
        assert result.map_estimate.shape == (n_time_bins,)
        assert result.uncertainty.shape == (n_time_bins,)


class TestPlotAndDataframe:
    """Test visualization and export methods (stubs for now)."""

    def test_plot_method_exists(self, small_2d_env):
        """DecodingResult should have a plot() method."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert hasattr(result, "plot")
        assert callable(result.plot)

    def test_to_dataframe_method_exists(self, small_2d_env):
        """DecodingResult should have a to_dataframe() method."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert hasattr(result, "to_dataframe")
        assert callable(result.to_dataframe)
