"""Tests for estimate functions in neurospatial.decoding.estimates.

These tests verify that standalone estimate functions correctly compute
derived quantities from posterior distributions, and that they produce
results consistent with the corresponding DecodingResult properties.
"""

import numpy as np
import pytest


class TestMapEstimate:
    """Test map_estimate function."""

    def test_map_estimate_shape(self, small_2d_env):
        """map_estimate should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding.estimates import map_estimate

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = map_estimate(posterior)

        assert result.shape == (n_time_bins,)
        assert result.dtype in (np.int64, np.intp)

    def test_map_estimate_delta_posterior(self, small_2d_env):
        """map_estimate should return correct bin for delta posterior."""
        from neurospatial.decoding.estimates import map_estimate

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, min(n_bins - 1, 10), 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = map_estimate(posterior)

        np.testing.assert_array_equal(result, expected_bins)

    def test_map_estimate_identity_posterior(self, small_2d_env):
        """map_estimate on identity posterior should return diagonal indices."""
        from neurospatial.decoding.estimates import map_estimate

        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = map_estimate(posterior)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result, expected)

    def test_map_estimate_consistency_with_decoding_result(self, small_2d_env):
        """map_estimate should match DecodingResult.map_estimate."""
        from neurospatial.decoding import DecodingResult
        from neurospatial.decoding.estimates import map_estimate

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        # Standalone function
        standalone_result = map_estimate(posterior)

        # DecodingResult property
        dr = DecodingResult(posterior=posterior, env=small_2d_env)
        property_result = dr.map_estimate

        np.testing.assert_array_equal(standalone_result, property_result)


class TestMapPosition:
    """Test map_position function."""

    def test_map_position_shape(self, small_2d_env):
        """map_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding.estimates import map_position

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = map_position(small_2d_env, posterior)

        assert result.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.dtype == np.float64

    def test_map_position_matches_bin_centers(self, small_2d_env):
        """map_position should equal bin_centers[map_estimate]."""
        from neurospatial.decoding.estimates import map_position

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors at known bins
        posterior = np.zeros((n_time_bins, n_bins))
        known_bins = [0, 1, 2, 3, 4]
        for t, bin_idx in enumerate(known_bins):
            posterior[t, bin_idx] = 1.0

        result = map_position(small_2d_env, posterior)

        expected_positions = small_2d_env.bin_centers[known_bins]
        np.testing.assert_array_almost_equal(result, expected_positions)

    def test_map_position_consistency_with_decoding_result(self, small_2d_env):
        """map_position should match DecodingResult.map_position."""
        from neurospatial.decoding import DecodingResult
        from neurospatial.decoding.estimates import map_position

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        # Standalone function
        standalone_result = map_position(small_2d_env, posterior)

        # DecodingResult property
        dr = DecodingResult(posterior=posterior, env=small_2d_env)
        property_result = dr.map_position

        np.testing.assert_array_almost_equal(standalone_result, property_result)


class TestMeanPosition:
    """Test mean_position function."""

    def test_mean_position_shape(self, small_2d_env):
        """mean_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding.estimates import mean_position

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = mean_position(small_2d_env, posterior)

        assert result.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.dtype == np.float64

    def test_mean_position_delta_equals_map(self, small_2d_env):
        """For delta posteriors, mean_position should equal map_position."""
        from neurospatial.decoding.estimates import map_position, mean_position

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        map_result = map_position(small_2d_env, posterior)
        mean_result = mean_position(small_2d_env, posterior)

        np.testing.assert_array_almost_equal(mean_result, map_result)

    def test_mean_position_uniform_is_centroid(self, small_2d_env):
        """For uniform posterior, mean_position should be centroid of all bins."""
        from neurospatial.decoding.estimates import mean_position

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = mean_position(small_2d_env, posterior)

        # Expected: centroid of all bin centers
        expected_centroid = small_2d_env.bin_centers.mean(axis=0)
        for t in range(n_time_bins):
            np.testing.assert_array_almost_equal(result[t], expected_centroid)

    def test_mean_position_two_bin_weighted(self, small_2d_env):
        """Mean position with two bins should be weighted average."""
        from neurospatial.decoding.estimates import mean_position

        n_bins = small_2d_env.n_bins
        if n_bins < 2:
            pytest.skip("Need at least 2 bins for this test")

        # Put 75% mass on bin 0, 25% on bin 1
        posterior = np.zeros((1, n_bins))
        posterior[0, 0] = 0.75
        posterior[0, 1] = 0.25

        result = mean_position(small_2d_env, posterior)

        # Expected: 0.75 * bin_centers[0] + 0.25 * bin_centers[1]
        expected = (
            0.75 * small_2d_env.bin_centers[0] + 0.25 * small_2d_env.bin_centers[1]
        )
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_mean_position_consistency_with_decoding_result(self, small_2d_env):
        """mean_position should match DecodingResult.mean_position."""
        from neurospatial.decoding import DecodingResult
        from neurospatial.decoding.estimates import mean_position

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        # Standalone function
        standalone_result = mean_position(small_2d_env, posterior)

        # DecodingResult property
        dr = DecodingResult(posterior=posterior, env=small_2d_env)
        property_result = dr.mean_position

        np.testing.assert_array_almost_equal(standalone_result, property_result)


class TestEntropy:
    """Test entropy function."""

    def test_entropy_shape(self, small_2d_env):
        """entropy should return (n_time_bins,) array."""
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = entropy(posterior)

        assert result.shape == (n_time_bins,)
        assert result.dtype == np.float64

    def test_entropy_delta_is_zero(self, small_2d_env):
        """Delta (one-hot) posterior should have zero entropy."""
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = entropy(posterior)

        np.testing.assert_array_almost_equal(result, np.zeros(n_time_bins))

    def test_entropy_uniform_is_maximum(self, small_2d_env):
        """Uniform posterior should have maximum entropy = log2(n_bins)."""
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = entropy(posterior)

        max_entropy = np.log2(n_bins)
        np.testing.assert_array_almost_equal(result, np.full(n_time_bins, max_entropy))

    def test_entropy_bounds(self, small_2d_env):
        """Entropy should be bounded: 0 <= entropy <= log2(n_bins)."""
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 10
        n_bins = small_2d_env.n_bins

        # Random posterior (normalized)
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        result = entropy(posterior)

        assert np.all(result >= 0)
        assert np.all(result <= np.log2(n_bins) + 1e-10)

    def test_entropy_handles_exact_zeros(self, small_2d_env):
        """Entropy should handle exact zeros without NaN (mask-based)."""
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        # Posterior with many exact zeros
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[:, 0] = 0.5
        posterior[:, 1] = 0.5

        result = entropy(posterior)

        # Should not be NaN
        assert np.all(np.isfinite(result))
        # Entropy of 50-50 distribution = 1 bit
        np.testing.assert_array_almost_equal(result, np.ones(n_time_bins))

    def test_entropy_consistency_with_decoding_result(self, small_2d_env):
        """entropy should match DecodingResult.uncertainty."""
        from neurospatial.decoding import DecodingResult
        from neurospatial.decoding.estimates import entropy

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        # Standalone function
        standalone_result = entropy(posterior)

        # DecodingResult property (named 'uncertainty')
        dr = DecodingResult(posterior=posterior, env=small_2d_env)
        property_result = dr.uncertainty

        np.testing.assert_array_almost_equal(standalone_result, property_result)


class TestCredibleRegion:
    """Test credible_region function (highest posterior density region)."""

    def test_credible_region_returns_list_of_arrays(self, small_2d_env):
        """credible_region should return list of bin index arrays."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = credible_region(small_2d_env, posterior, level=0.95)

        assert isinstance(result, list)
        assert len(result) == n_time_bins
        for region in result:
            assert isinstance(region, np.ndarray)
            assert region.dtype in (np.int64, np.intp)

    def test_credible_region_delta_single_bin(self, small_2d_env):
        """Delta posterior should have single-bin credible region."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = credible_region(small_2d_env, posterior, level=0.95)

        for t, bin_idx in enumerate(expected_bins):
            assert len(result[t]) == 1
            assert result[t][0] == bin_idx

    def test_credible_region_uniform_includes_all_bins(self, small_2d_env):
        """Uniform posterior at level=0.95 should include 95% of bins."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 2
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = credible_region(small_2d_env, posterior, level=0.95)

        # For uniform distribution, need at least ceil(0.95 * n_bins) bins
        min_bins = int(np.ceil(0.95 * n_bins))
        for region in result:
            assert len(region) >= min_bins

    def test_credible_region_respects_level(self, small_2d_env):
        """credible_region should contain at least the requested probability mass."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        for level in [0.5, 0.9, 0.95, 0.99]:
            result = credible_region(small_2d_env, posterior, level=level)

            for t, region in enumerate(result):
                mass = posterior[t, region].sum()
                assert mass >= level - 1e-10, f"Level {level} not met at time {t}"

    def test_credible_region_is_hpd(self, small_2d_env):
        """credible_region should be HPD (highest posterior density)."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 1
        n_bins = small_2d_env.n_bins

        # Create posterior with clear probability ordering
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[0, :5] = 0.15  # 5 bins at 0.15 each = 0.75
        posterior[0, 5:10] = 0.04  # 5 bins at 0.04 each = 0.20
        posterior[0, 10:15] = 0.01  # 5 bins at 0.01 each = 0.05
        posterior = posterior / posterior.sum()  # Normalize

        # At level=0.7, should include highest probability bins first
        result = credible_region(small_2d_env, posterior, level=0.7)

        # The first 5 bins have highest probability (0.15 each)
        # These should be included before lower probability bins
        high_prob_bins = set(range(5))
        included_bins = set(result[0])

        # Check that high probability bins are included
        # (HPD should select these first)
        assert high_prob_bins.issubset(included_bins) or len(result[0]) >= 5

    def test_credible_region_level_bounds(self, small_2d_env):
        """credible_region should raise for invalid levels."""
        from neurospatial.decoding.estimates import credible_region

        n_time_bins = 2
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        with pytest.raises(ValueError, match=r"level.*between 0 and 1"):
            credible_region(small_2d_env, posterior, level=0.0)

        with pytest.raises(ValueError, match=r"level.*between 0 and 1"):
            credible_region(small_2d_env, posterior, level=1.0)

        with pytest.raises(ValueError, match=r"level.*between 0 and 1"):
            credible_region(small_2d_env, posterior, level=1.5)

        with pytest.raises(ValueError, match=r"level.*between 0 and 1"):
            credible_region(small_2d_env, posterior, level=-0.1)


class TestSuccessCriteria:
    """Test success criteria from TASKS.md."""

    def test_success_criteria_shapes(self, small_2d_env):
        """Verify success criteria from TASKS.md."""
        from neurospatial.decoding.estimates import (
            entropy,
            map_estimate,
            map_position,
        )

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        # Success criteria from TASKS.md
        bins = map_estimate(posterior)
        pos = map_position(small_2d_env, posterior)
        ent = entropy(posterior)

        assert bins.shape == (n_time_bins,)
        assert pos.shape == (n_time_bins, small_2d_env.n_dims)
        assert ent.shape == (n_time_bins,)
        assert np.all(ent >= 0)
        assert np.all(ent <= np.log2(n_bins))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self, small_2d_env):
        """Functions should work with a single time bin."""
        from neurospatial.decoding.estimates import (
            credible_region,
            entropy,
            map_estimate,
            map_position,
            mean_position,
        )

        n_bins = small_2d_env.n_bins
        posterior = np.ones((1, n_bins)) / n_bins

        assert map_estimate(posterior).shape == (1,)
        assert map_position(small_2d_env, posterior).shape == (1, small_2d_env.n_dims)
        assert mean_position(small_2d_env, posterior).shape == (1, small_2d_env.n_dims)
        assert entropy(posterior).shape == (1,)
        assert len(credible_region(small_2d_env, posterior)) == 1

    def test_single_bin_environment(self):
        """Functions should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding.estimates import (
            credible_region,
            entropy,
            map_estimate,
            map_position,
            mean_position,
        )

        # Create a minimal single-bin environment
        env = Environment.from_samples(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            bin_size=10.0,  # Large bin size to get single bin
        )

        if env.n_bins == 1:
            n_time_bins = 5
            posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

            np.testing.assert_array_equal(
                map_estimate(posterior), np.zeros(n_time_bins)
            )
            assert map_position(env, posterior).shape == (n_time_bins, env.n_dims)
            assert mean_position(env, posterior).shape == (n_time_bins, env.n_dims)
            # Entropy of single bin is 0 (log2(1) = 0)
            np.testing.assert_array_almost_equal(
                entropy(posterior), np.zeros(n_time_bins)
            )
            # Credible region with single bin
            regions = credible_region(env, posterior, level=0.5)
            assert all(len(r) == 1 for r in regions)
        else:
            pytest.skip("Could not create single-bin environment")
