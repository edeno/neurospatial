"""Tests for posterior normalization in neurospatial.decoding.posterior.

Tests cover:
- normalize_to_posterior: Log-sum-exp normalization with prior and degenerate handling
- decode_position: Main entry point combining likelihood + posterior
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.decoding import DecodingResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_log_likelihood() -> np.ndarray:
    """Simple log-likelihood matrix: (3 time bins, 4 bins)."""
    # Values chosen to have clear maximum per row
    return np.array(
        [
            [-1.0, -2.0, -3.0, -0.5],  # max at bin 3
            [-2.0, -0.1, -1.5, -3.0],  # max at bin 1
            [-0.2, -0.3, -0.1, -0.4],  # max at bin 2
        ],
        dtype=np.float64,
    )


@pytest.fixture
def uniform_log_likelihood() -> np.ndarray:
    """Uniform log-likelihood (all bins equal)."""
    return np.zeros((2, 5), dtype=np.float64)


@pytest.fixture
def degenerate_log_likelihood() -> np.ndarray:
    """Log-likelihood with degenerate row (all -inf)."""
    ll = np.array(
        [
            [-1.0, -2.0, -3.0],
            [-np.inf, -np.inf, -np.inf],  # Degenerate row
            [-2.0, -1.0, -0.5],
        ],
        dtype=np.float64,
    )
    return ll


@pytest.fixture
def simple_env() -> Environment:
    """Simple 2D environment for testing."""
    positions = np.array(
        [[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [15, 5], [5, 15], [15, 15]],
        dtype=np.float64,
    )
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_spike_counts() -> np.ndarray:
    """Simple spike count matrix: (3 time bins, 2 neurons)."""
    return np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)


@pytest.fixture
def simple_encoding_models(simple_env: Environment) -> np.ndarray:
    """Simple encoding models matching simple_env: (2 neurons, n_bins)."""
    n_bins = simple_env.n_bins
    # Create Gaussian-like place fields
    rng = np.random.default_rng(42)
    models = np.abs(rng.standard_normal((2, n_bins))) * 10 + 0.1
    return models.astype(np.float64)


# =============================================================================
# Tests for normalize_to_posterior
# =============================================================================


class TestNormalizeToPosterior:
    """Tests for normalize_to_posterior function."""

    def test_output_shape(self, simple_log_likelihood: np.ndarray) -> None:
        """Output shape should match input shape."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        assert posterior.shape == simple_log_likelihood.shape

    def test_output_dtype(self, simple_log_likelihood: np.ndarray) -> None:
        """Output dtype should be float64."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        assert posterior.dtype == np.float64

    def test_rows_sum_to_one(self, simple_log_likelihood: np.ndarray) -> None:
        """Each row should sum to 1.0 (probability distribution)."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        row_sums = posterior.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_non_negative(self, simple_log_likelihood: np.ndarray) -> None:
        """All probabilities should be non-negative."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        assert (posterior >= 0).all()

    def test_at_most_one(self, simple_log_likelihood: np.ndarray) -> None:
        """All probabilities should be at most 1.0."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        assert (posterior <= 1.0 + 1e-10).all()

    def test_max_bin_preserved(self, simple_log_likelihood: np.ndarray) -> None:
        """Argmax should be preserved after normalization."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood)
        assert_array_equal(
            simple_log_likelihood.argmax(axis=1), posterior.argmax(axis=1)
        )

    def test_uniform_from_equal_log_likelihood(
        self, uniform_log_likelihood: np.ndarray
    ) -> None:
        """Equal log-likelihoods should give uniform posterior."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(uniform_log_likelihood)
        n_bins = uniform_log_likelihood.shape[1]
        expected = np.ones_like(uniform_log_likelihood) / n_bins
        assert_allclose(posterior, expected, atol=1e-10)

    def test_numerical_stability_extreme_values(self) -> None:
        """Should handle extreme log-likelihood values without overflow."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Very negative values that could cause underflow
        ll = np.array([[-1000.0, -1001.0, -999.0]], dtype=np.float64)
        posterior = normalize_to_posterior(ll)
        assert np.isfinite(posterior).all()
        assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_numerical_stability_large_range(self) -> None:
        """Should handle large range of log-likelihood values."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Large range that could cause numerical issues
        ll = np.array([[0.0, -500.0, -1000.0]], dtype=np.float64)
        posterior = normalize_to_posterior(ll)
        assert np.isfinite(posterior).all()
        # First bin should get almost all probability
        assert posterior[0, 0] > 0.99

    # -------------------------------------------------------------------------
    # Prior handling tests
    # -------------------------------------------------------------------------

    def test_prior_uniform_default(self, simple_log_likelihood: np.ndarray) -> None:
        """Default prior (None) should be uniform."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior_no_prior = normalize_to_posterior(simple_log_likelihood)
        n_bins = simple_log_likelihood.shape[1]
        uniform_prior = np.ones(n_bins) / n_bins
        posterior_uniform = normalize_to_posterior(
            simple_log_likelihood, prior=uniform_prior
        )
        assert_allclose(posterior_no_prior, posterior_uniform, atol=1e-10)

    def test_prior_stationary(self, simple_log_likelihood: np.ndarray) -> None:
        """Stationary prior (1D) should be applied to all time bins."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Non-uniform prior: higher on first bin
        prior = np.array([0.4, 0.3, 0.2, 0.1])
        posterior = normalize_to_posterior(simple_log_likelihood, prior=prior)
        # Should still sum to 1
        assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)
        # Prior should shift probability toward first bin
        posterior_no_prior = normalize_to_posterior(simple_log_likelihood)
        # First bin should have relatively higher probability than without prior
        assert posterior[:, 0].mean() > posterior_no_prior[:, 0].mean()

    def test_prior_time_varying(self, simple_log_likelihood: np.ndarray) -> None:
        """Time-varying prior (2D) should work."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        n_time, n_bins = simple_log_likelihood.shape
        # Different prior for each time bin
        prior = np.ones((n_time, n_bins)) / n_bins
        prior[0, 0] = 0.5  # First time bin has higher prior on first spatial bin
        prior[0, 1:] = 0.5 / (n_bins - 1)
        posterior = normalize_to_posterior(simple_log_likelihood, prior=prior)
        assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_prior_normalization(self, simple_log_likelihood: np.ndarray) -> None:
        """Prior should be normalized internally."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Unnormalized prior
        prior_unnorm = np.array([2.0, 4.0, 6.0, 8.0])
        # Manually normalized
        prior_norm = prior_unnorm / prior_unnorm.sum()

        posterior_unnorm = normalize_to_posterior(
            simple_log_likelihood, prior=prior_unnorm
        )
        posterior_norm = normalize_to_posterior(simple_log_likelihood, prior=prior_norm)
        assert_allclose(posterior_unnorm, posterior_norm, atol=1e-10)

    # -------------------------------------------------------------------------
    # Degenerate handling tests
    # -------------------------------------------------------------------------

    def test_degenerate_uniform_default(
        self, degenerate_log_likelihood: np.ndarray
    ) -> None:
        """Degenerate rows should return uniform by default."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(
            degenerate_log_likelihood, handle_degenerate="uniform"
        )
        n_bins = degenerate_log_likelihood.shape[1]
        # Degenerate row (row 1) should be uniform
        expected_uniform = np.ones(n_bins) / n_bins
        assert_allclose(posterior[1], expected_uniform, atol=1e-10)
        # Other rows should still sum to 1
        assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_degenerate_nan(self, degenerate_log_likelihood: np.ndarray) -> None:
        """Degenerate rows should return NaN when handle_degenerate='nan'."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(
            degenerate_log_likelihood, handle_degenerate="nan"
        )
        # Degenerate row should be all NaN
        assert np.isnan(posterior[1]).all()
        # Other rows should be finite and sum to 1
        assert np.isfinite(posterior[0]).all()
        assert np.isfinite(posterior[2]).all()
        assert_allclose(posterior[0].sum(), 1.0, atol=1e-10)
        assert_allclose(posterior[2].sum(), 1.0, atol=1e-10)

    def test_degenerate_raise(self, degenerate_log_likelihood: np.ndarray) -> None:
        """Degenerate rows should raise ValueError when handle_degenerate='raise'."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        with pytest.raises(ValueError, match="degenerate"):
            normalize_to_posterior(degenerate_log_likelihood, handle_degenerate="raise")

    def test_invalid_handle_degenerate(self, simple_log_likelihood: np.ndarray) -> None:
        """Invalid handle_degenerate value should raise ValueError."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        with pytest.raises(ValueError):
            normalize_to_posterior(simple_log_likelihood, handle_degenerate="invalid")

    # -------------------------------------------------------------------------
    # Axis parameter tests
    # -------------------------------------------------------------------------

    def test_axis_default(self, simple_log_likelihood: np.ndarray) -> None:
        """Default axis (-1) should normalize over last axis."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood, axis=-1)
        assert_allclose(posterior.sum(axis=-1), 1.0, atol=1e-10)

    def test_axis_explicit(self, simple_log_likelihood: np.ndarray) -> None:
        """Explicit axis=1 should normalize over second axis."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood, axis=1)
        assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_axis_not_last_raises(self, simple_log_likelihood: np.ndarray) -> None:
        """axis=0 should raise ValueError (degeneracy handling only supports last axis)."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        with pytest.raises(ValueError, match="axis must be the last dimension"):
            normalize_to_posterior(simple_log_likelihood, axis=0)

    def test_prior_wrong_1d_shape_raises(
        self, simple_log_likelihood: np.ndarray
    ) -> None:
        """1D prior with wrong number of bins should raise ValueError."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # simple_log_likelihood has 4 bins, prior has 3
        prior = np.array([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="1D prior must have shape"):
            normalize_to_posterior(simple_log_likelihood, prior=prior)

    def test_prior_wrong_2d_shape_raises(
        self, simple_log_likelihood: np.ndarray
    ) -> None:
        """2D prior with wrong shape should raise ValueError."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        # simple_log_likelihood is (3, 4), prior is (2, 4)
        prior = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="2D prior must have shape"):
            normalize_to_posterior(simple_log_likelihood, prior=prior)

    def test_prior_3d_raises(self, simple_log_likelihood: np.ndarray) -> None:
        """3D prior should raise ValueError."""
        from neurospatial.decoding.posterior import normalize_to_posterior

        prior = np.ones((3, 4, 2)) / 4
        with pytest.raises(ValueError, match=r"prior must be 1D .* or 2D"):
            normalize_to_posterior(simple_log_likelihood, prior=prior)


# =============================================================================
# Tests for decode_position
# =============================================================================


class TestDecodePosition:
    """Tests for decode_position function."""

    def test_returns_decoding_result(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Should return a DecodingResult instance."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert isinstance(result, DecodingResult)

    def test_posterior_shape(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Posterior shape should be (n_time_bins, n_bins)."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        n_time_bins = simple_spike_counts.shape[0]
        assert result.posterior.shape == (n_time_bins, simple_env.n_bins)

    def test_posterior_sums_to_one(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Each posterior row should sum to 1.0."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        row_sums = result.posterior.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_environment_stored(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Result should store reference to environment."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert result.env is simple_env

    def test_times_stored(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Times should be stored when provided."""
        from neurospatial.decoding.posterior import decode_position

        times = np.array([0.0, 0.025, 0.05])
        result = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            times=times,
        )
        assert_array_equal(result.times, times)

    def test_times_none_default(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Times should be None when not provided."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert result.times is None

    def test_prior_applied(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Prior should affect posterior distribution."""
        from neurospatial.decoding.posterior import decode_position

        # Very strong prior on first bin
        prior = np.zeros(simple_env.n_bins)
        prior[0] = 1.0

        result_prior = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            prior=prior,
        )
        result_no_prior = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )

        # With strong prior on bin 0, that bin should have higher probability
        assert (
            result_prior.posterior[:, 0].mean() > result_no_prior.posterior[:, 0].mean()
        )

    def test_method_poisson_default(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Method should default to 'poisson'."""
        from neurospatial.decoding.posterior import decode_position

        # Should not raise with default method
        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert result.posterior.shape[0] == simple_spike_counts.shape[0]

    def test_method_invalid(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Invalid method should raise ValueError."""
        from neurospatial.decoding.posterior import decode_position

        with pytest.raises(ValueError, match="method"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                method="invalid",
            )

    # -------------------------------------------------------------------------
    # Validation tests
    # -------------------------------------------------------------------------

    def test_validate_false_default(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=False should be default (no extra checks)."""
        from neurospatial.decoding.posterior import decode_position

        # Should complete without extra validation overhead
        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert result is not None

    def test_validate_true_passes_valid_data(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True should pass with valid data."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            validate=True,
        )
        assert result is not None

    def test_validate_catches_nan_input(
        self,
        simple_env: Environment,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True should catch NaN in input."""
        from neurospatial.decoding.posterior import decode_position

        spike_counts_nan = np.array([[0, np.nan], [1, 0], [0, 1]])
        with pytest.raises(ValueError, match=r"NaN|nan|Inf|inf"):
            decode_position(
                simple_env,
                spike_counts_nan,
                simple_encoding_models,
                dt=0.025,
                validate=True,
            )

    def test_validate_catches_inf_encoding(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
    ) -> None:
        """validate=True should catch Inf in encoding models."""
        from neurospatial.decoding.posterior import decode_position

        n_bins = simple_env.n_bins
        encoding_inf = np.ones((2, n_bins))
        encoding_inf[0, 0] = np.inf

        with pytest.raises(ValueError, match=r"NaN|nan|Inf|inf"):
            decode_position(
                simple_env,
                simple_spike_counts,
                encoding_inf,
                dt=0.025,
                validate=True,
            )

    # -------------------------------------------------------------------------
    # Integration tests
    # -------------------------------------------------------------------------

    def test_end_to_end_simple(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """End-to-end test with simple data."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )

        # Verify all derived properties work
        assert result.map_estimate.shape == (simple_spike_counts.shape[0],)
        assert result.map_position.shape == (
            simple_spike_counts.shape[0],
            simple_env.n_dims,
        )
        assert result.mean_position.shape == (
            simple_spike_counts.shape[0],
            simple_env.n_dims,
        )
        assert result.uncertainty.shape == (simple_spike_counts.shape[0],)

    def test_reproducibility(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Multiple calls with same data should give identical results."""
        from neurospatial.decoding.posterior import decode_position

        result1 = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        result2 = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert_allclose(result1.posterior, result2.posterior)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_time_bin(self, simple_env: Environment) -> None:
        """Should handle single time bin."""
        from neurospatial.decoding.posterior import decode_position

        spike_counts = np.array([[1, 2]], dtype=np.int64)
        n_bins = simple_env.n_bins
        encoding = np.abs(np.random.default_rng(42).standard_normal((2, n_bins))) + 0.1

        result = decode_position(simple_env, spike_counts, encoding, dt=0.025)
        assert result.posterior.shape == (1, n_bins)
        assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_single_neuron(self, simple_env: Environment) -> None:
        """Should handle single neuron."""
        from neurospatial.decoding.posterior import decode_position

        spike_counts = np.array([[1], [0], [2]], dtype=np.int64)
        n_bins = simple_env.n_bins
        encoding = np.abs(np.random.default_rng(42).standard_normal((1, n_bins))) + 0.1

        result = decode_position(simple_env, spike_counts, encoding, dt=0.025)
        assert result.posterior.shape == (3, n_bins)
        assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_all_zero_spikes(self, simple_env: Environment) -> None:
        """Should handle all-zero spike counts (valid edge case)."""
        from neurospatial.decoding.posterior import decode_position

        spike_counts = np.zeros((3, 2), dtype=np.int64)
        n_bins = simple_env.n_bins
        encoding = np.abs(np.random.default_rng(42).standard_normal((2, n_bins))) + 0.1

        result = decode_position(simple_env, spike_counts, encoding, dt=0.025)
        assert result.posterior.shape == (3, n_bins)
        # Should still sum to 1 - posterior based on rate penalty only
        assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-10)

    def test_very_small_dt(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Should handle very small dt values."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=1e-6
        )
        assert np.isfinite(result.posterior).all()
        assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)

    def test_large_dt(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """Should handle large dt values (e.g., 1 second bins)."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=1.0
        )
        assert np.isfinite(result.posterior).all()
        assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestConsistency:
    """Tests for consistency with other decoding module components."""

    def test_consistent_with_log_likelihood(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """decode_position should be consistent with manual likelihood + posterior."""
        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import (
            decode_position,
            normalize_to_posterior,
        )

        dt = 0.025

        # Manual computation
        ll = log_poisson_likelihood(simple_spike_counts, simple_encoding_models, dt=dt)
        posterior_manual = normalize_to_posterior(ll)

        # Via decode_position
        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=dt
        )

        assert_allclose(result.posterior, posterior_manual, atol=1e-10)

    def test_map_estimate_matches_posterior_argmax(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """DecodingResult.map_estimate should match argmax of posterior."""
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )

        expected_argmax = np.argmax(result.posterior, axis=1)
        assert_array_equal(result.map_estimate, expected_argmax)
