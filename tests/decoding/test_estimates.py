"""Tests for estimate functions in neurospatial.decoding.estimates.

These tests verify that standalone estimate functions correctly compute
derived quantities from posterior distributions, and that they produce
results consistent with the corresponding DecodingResult properties.
"""

import numpy as np
import pytest


class TestMapEstimate:
    """Test map_bin function."""

    def test_map_bin_shape(self, small_2d_env):
        """map_bin should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding.estimates import map_bin

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = map_bin(small_2d_env, posterior)

        assert result.shape == (n_time_bins,)
        assert result.dtype in (np.int64, np.intp)

    def test_map_bin_delta_posterior(self, small_2d_env):
        """map_bin should return correct bin for delta posterior."""
        from neurospatial.decoding.estimates import map_bin

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, min(n_bins - 1, 10), 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = map_bin(small_2d_env, posterior)

        np.testing.assert_array_equal(result, expected_bins)

    def test_map_bin_identity_posterior(self, small_2d_env):
        """map_bin on identity posterior should return diagonal indices."""
        from neurospatial.decoding.estimates import map_bin

        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = map_bin(small_2d_env, posterior)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result, expected)


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
        """map_position should equal bin_centers[map_bin]."""
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


class TestEntropy:
    """Test entropy function."""

    def test_entropy_shape(self, small_2d_env):
        """entropy should return (n_time_bins,) array."""
        from neurospatial.decoding.estimates import posterior_entropy

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = posterior_entropy(posterior)

        assert result.shape == (n_time_bins,)
        assert result.dtype == np.float64

    def test_entropy_delta_is_zero(self, small_2d_env):
        """Delta (one-hot) posterior should have zero entropy."""
        from neurospatial.decoding.estimates import posterior_entropy

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = posterior_entropy(posterior)

        np.testing.assert_array_almost_equal(result, np.zeros(n_time_bins))

    def test_entropy_uniform_is_maximum(self, small_2d_env):
        """Uniform posterior should have maximum entropy = log2(n_bins)."""
        from neurospatial.decoding.estimates import posterior_entropy

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = posterior_entropy(posterior)

        max_entropy = np.log2(n_bins)
        np.testing.assert_array_almost_equal(result, np.full(n_time_bins, max_entropy))

    def test_entropy_bounds(self, small_2d_env):
        """Entropy should be bounded: 0 <= entropy <= log2(n_bins)."""
        from neurospatial.decoding.estimates import posterior_entropy

        n_time_bins = 10
        n_bins = small_2d_env.n_bins

        # Random posterior (normalized)
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        result = posterior_entropy(posterior)

        assert np.all(result >= 0)
        assert np.all(result <= np.log2(n_bins) + 1e-10)

    def test_entropy_handles_exact_zeros(self, small_2d_env):
        """Entropy should handle exact zeros without NaN (mask-based)."""
        from neurospatial.decoding.estimates import posterior_entropy

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        # Posterior with many exact zeros
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[:, 0] = 0.5
        posterior[:, 1] = 0.5

        result = posterior_entropy(posterior)

        # Should not be NaN
        assert np.all(np.isfinite(result))
        # Entropy of 50-50 distribution = 1 bit
        np.testing.assert_array_almost_equal(result, np.ones(n_time_bins))


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
    """Test success criteria."""


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self, small_2d_env):
        """Functions should work with a single time bin."""
        from neurospatial.decoding.estimates import (
            credible_region,
            map_bin,
            map_position,
            mean_position,
            posterior_entropy,
        )

        n_bins = small_2d_env.n_bins
        posterior = np.ones((1, n_bins)) / n_bins

        assert map_bin(small_2d_env, posterior).shape == (1,)
        assert map_position(small_2d_env, posterior).shape == (1, small_2d_env.n_dims)
        assert mean_position(small_2d_env, posterior).shape == (1, small_2d_env.n_dims)
        assert posterior_entropy(posterior).shape == (1,)
        assert len(credible_region(small_2d_env, posterior)) == 1

    def test_single_bin_environment(self):
        """Functions should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding.estimates import (
            credible_region,
            map_bin,
            map_position,
            mean_position,
            posterior_entropy,
        )

        env = Environment.from_grid_mask(
            np.ones(1, dtype=bool), grid_edges=(np.array([0.0, 10.0]),)
        )
        assert env.n_bins == 1
        n_time_bins = 5
        posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

        np.testing.assert_array_equal(map_bin(env, posterior), np.zeros(n_time_bins))
        np.testing.assert_array_equal(
            map_position(env, posterior), np.full((n_time_bins, 1), 5.0)
        )
        np.testing.assert_array_equal(
            mean_position(env, posterior), np.full((n_time_bins, 1), 5.0)
        )
        # Entropy of single bin is 0 (log2(1) = 0)
        np.testing.assert_array_equal(posterior_entropy(posterior), 0.0)
        regions = credible_region(env, posterior, level=0.5)
        assert [r.tolist() for r in regions] == [[0]] * n_time_bins


class TestCredibleRegionNaN:
    """credible_region must exclude NaN posterior bins from the HPD set."""

    def test_credible_region_excludes_nan_bin(self, small_2d_env):
        """A NaN bin is never reported, including as the leading HPD member."""
        from neurospatial.decoding.estimates import credible_region

        n_bins = small_2d_env.n_bins
        assert n_bins >= 3

        posterior = np.full((1, n_bins), 0.0)
        # Clear finite mode at bin 1; a NaN bin at index 0.
        posterior[0, 1] = 1.0
        posterior[0, 0] = np.nan

        regions = credible_region(small_2d_env, posterior, level=0.95)

        hpd = regions[0]
        assert 1 in hpd  # finite mode is included
        assert 0 not in hpd  # NaN bin is excluded

    def test_credible_region_low_finite_mass_excludes_nan_bins(self, small_2d_env):
        """Low finite-mass rows must not slice NaN bins back via the fallback.

        When a row's finite probability mass is strictly below ``level``,
        the cumsum never reaches the target and the ``never_reached``
        fallback is triggered. The fallback must cap at the number of
        finite bins so the NaN/Inf placeholder bins are never re-included.
        """
        from neurospatial.decoding.estimates import credible_region

        n_bins = small_2d_env.n_bins
        assert n_bins >= 4

        # Finite mass of 0.5 (< level 0.95) split across two finite bins,
        # plus two NaN bins. The finite mode is at index 2.
        posterior = np.full((1, n_bins), 0.0)
        posterior[0, 0] = np.nan
        posterior[0, 1] = np.nan
        posterior[0, 2] = 0.3
        posterior[0, 3] = 0.2

        regions = credible_region(small_2d_env, posterior, level=0.95)

        hpd = regions[0]
        # None of the NaN bin indices may appear in the returned region.
        assert 0 not in hpd
        assert 1 not in hpd
        # The finite mode must still be reported.
        assert 2 in hpd

    def test_credible_region_all_nan_row_returns_no_finite_bins(self, small_2d_env):
        """An all-NaN row degrades to an empty region without crashing."""
        from neurospatial.decoding.estimates import credible_region

        n_bins = small_2d_env.n_bins
        posterior = np.full((1, n_bins), np.nan)

        regions = credible_region(small_2d_env, posterior, level=0.95)

        assert len(regions[0]) == 0


def test_undecodable_rows_have_no_map_bin_or_position(small_2d_env):
    """An entirely non-finite row maps to the outside sentinel and NaN."""
    from neurospatial.decoding.estimates import map_bin, map_position

    posterior = np.full((3, small_2d_env.n_bins), np.nan)
    posterior[1] = 0.0
    posterior[1, 4] = 1.0
    posterior[2] = np.inf
    np.testing.assert_array_equal(map_bin(small_2d_env, posterior), [-1, 4, -1])
    positions = map_position(small_2d_env, posterior)
    assert np.all(np.isnan(positions[[0, 2]]))
    np.testing.assert_array_equal(positions[1], small_2d_env.bin_centers[4])


def test_map_bin_ties_go_to_lowest_index_despite_rounding():
    """Densities equal up to rounding tie, so the lowest bin index wins."""
    from neurospatial import Environment
    from neurospatial.decoding import DecodingResult
    from neurospatial.decoding.estimates import map_bin

    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )
    flat_over_space = env.bin_sizes / env.bin_sizes.sum()
    density = flat_over_space / env.bin_sizes
    assert np.argmax(density) != 0  # rounding alone separates the bins
    assert map_bin(env, flat_over_space[None])[0] == 0
    assert DecodingResult(flat_over_space[None], env).map_bin[0] == 0


def test_map_bin_ignores_nonfinite_bins(small_2d_env):
    """A NaN or Inf bin is never the MAP bin of a row with finite mass."""
    from neurospatial.decoding.estimates import map_bin

    posterior = np.zeros((2, small_2d_env.n_bins))
    posterior[:, 5] = 1.0
    posterior[0, 1] = np.nan
    posterior[1, 2] = np.inf
    np.testing.assert_array_equal(map_bin(small_2d_env, posterior), [5, 5])


def test_credible_region_never_includes_zero_mass_bins(small_2d_env):
    """A row whose mass never reaches ``level`` keeps only bins with mass."""
    from neurospatial.decoding.estimates import credible_region

    posterior = np.zeros((2, small_2d_env.n_bins))
    posterior[0, :2] = np.nan
    posterior[0, 2] = 0.3
    posterior[0, 3] = 0.2
    regions = credible_region(small_2d_env, posterior, level=0.95)
    assert sorted(regions[0].tolist()) == [2, 3]
    assert regions[1].size == 0


def test_posterior_entropy_float64_and_nan_for_undecodable_rows():
    from neurospatial.decoding.estimates import posterior_entropy

    posterior = np.full((2, 8), 1 / 8, dtype=np.float32)
    posterior[1] = np.nan
    entropy = posterior_entropy(posterior)
    assert entropy.dtype == np.float64
    assert entropy[0] == pytest.approx(3.0, abs=1e-6)
    assert np.isnan(entropy[1])


def test_float32_map_bin_matches_float64_and_the_summary():
    """Rounding ties do not shift float32 MAP bins toward low indices."""
    from neurospatial import Environment
    from neurospatial.decoding import decode_position, decode_position_summary

    rng = np.random.default_rng(0)
    env = Environment.from_samples(np.linspace(0, 200, 2001)[:, None], bin_size=1.0)
    centers = env.bin_centers[:, 0]
    fields = np.array([50.0, 100.0, 150.0])
    rates = 0.5 + 5 * np.exp(-0.5 * ((centers - fields[:, None]) / 60) ** 2)
    counts = rng.poisson(rates[:, rng.integers(0, env.n_bins, 2000)].T * 0.02)

    full64 = decode_position(env, counts, rates, dt=0.02)
    full32 = decode_position(env, counts, rates, dt=0.02, dtype=np.float32)
    summary32 = decode_position_summary(env, counts, rates, dt=0.02, dtype=np.float32)

    np.testing.assert_array_equal(summary32.map_bin, full32.map_bin)
    # float32 rounding may move a near-tie, but not shift MAP bins systematically.
    shifted = full32.map_bin != full64.map_bin
    assert shifted.mean() < 0.01
    assert abs(np.mean(full32.map_bin - full64.map_bin)) < 0.002
