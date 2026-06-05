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

    def test_raise_distinguishes_nan_from_neg_inf(self) -> None:
        """raise message must distinguish NaN (corruption) from -inf (zero-rate).

        An all -inf row is a legitimate zero-rate degenerate row; a NaN row
        signals upstream corruption. The raise path must call out the NaN
        rows so the user knows the input is corrupted, not merely flat.
        """
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Only a -inf row: message must NOT claim NaN corruption.
        ll_neg_inf = np.array(
            [
                [-1.0, -2.0, -3.0],
                [-np.inf, -np.inf, -np.inf],
            ],
            dtype=np.float64,
        )
        with pytest.raises(ValueError) as exc_neg_inf:
            normalize_to_posterior(ll_neg_inf, handle_degenerate="raise")
        assert "NaN" not in str(exc_neg_inf.value)

        # A NaN row: message MUST mention NaN/corruption.
        ll_nan = np.array(
            [
                [-1.0, -2.0, -3.0],
                [np.nan, -1.0, -0.5],
            ],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="NaN"):
            normalize_to_posterior(ll_nan, handle_degenerate="raise")

    def test_nan_degenerate_posterior_validates_without_spurious_error(self) -> None:
        """A nan-degenerate posterior must pass _validate_output cleanly.

        With handle_degenerate='nan', degenerate rows are intentionally NaN.
        _validate_output should tolerate those rows (not raise about
        NaN/Inf or row sums) while still validating the finite rows.
        """
        from neurospatial.decoding.posterior import (
            _validate_output,
            normalize_to_posterior,
        )

        ll = np.array(
            [
                [-1.0, -2.0, -3.0],
                [-np.inf, -np.inf, -np.inf],
                [-2.0, -1.0, -0.5],
            ],
            dtype=np.float64,
        )
        posterior = normalize_to_posterior(ll, handle_degenerate="nan")
        assert np.isnan(posterior[1]).all()

        # Should not raise despite the NaN row.
        _validate_output(posterior)

    # -------------------------------------------------------------------------
    # Axis parameter tests
    # -------------------------------------------------------------------------

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

    def test_validate_default_is_true(
        self,
        simple_env: Environment,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True should be the default, so a bad input raises by default.

        Regression: previously decode_position defaulted to
        validate=False, so users got nonsense posteriors with no warning
        when their inputs were corrupted. The new default invocation
        runs _validate_inputs and surfaces the problem at the boundary.
        """
        from neurospatial.decoding.posterior import decode_position

        spike_counts_nan = np.array([[0, np.nan], [1, 0], [0, 1]])
        with pytest.raises(ValueError, match=r"NaN|nan|Inf|inf"):
            decode_position(
                simple_env, spike_counts_nan, simple_encoding_models, dt=0.025
            )

    def test_validate_false_opt_out_skips_input_checks(
        self,
        simple_env: Environment,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=False is an explicit opt-out for hot loops.

        Users running decode_position inside a tight loop and confident
        their inputs are clean can skip the per-call validation overhead.
        The contract is that the call doesn't raise on bad input —
        even though ``validate=True`` would catch it. The matching
        positive assertion is that the clean rows still produce a
        well-formed row-stochastic posterior, so we know the call did
        more than "return a stub".
        """
        from neurospatial.decoding.posterior import decode_position

        spike_counts_nan = np.array([[0, np.nan], [1, 0], [0, 1]])
        result = decode_position(
            simple_env,
            spike_counts_nan,
            simple_encoding_models,
            dt=0.025,
            validate=False,
        )
        posterior = np.asarray(result.posterior)
        # No exception was raised (validate=False skipped the check).
        # The clean rows (1 and 2) must still produce well-formed
        # probability distributions — proves the call actually ran the
        # decoder, not just bailed out.
        np.testing.assert_allclose(posterior[1].sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(posterior[2].sum(), 1.0, atol=1e-9)
        assert np.all(np.isfinite(posterior[1]))
        assert np.all(np.isfinite(posterior[2]))

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

    def test_validate_catches_negative_spike_counts(
        self,
        simple_env: Environment,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True (the new default) rejects negative spike counts.

        Regression: a negative spike count is physically
        impossible (a count is "how many spikes per time bin"), but
        nothing was rejecting it before. log_poisson_likelihood would
        produce nonsense (n * log(rate) - rate*dt with negative n flips
        the sign of the log-likelihood term) and a "valid"-looking
        posterior would be returned.
        """
        from neurospatial.decoding.posterior import decode_position

        spike_counts_negative = np.array([[0, 1], [-1, 0], [0, 1]], dtype=np.int64)
        with pytest.raises(ValueError, match=r"negative entr"):
            decode_position(
                simple_env,
                spike_counts_negative,
                simple_encoding_models,
                dt=0.025,
            )

    def test_validate_catches_negative_encoding_rates(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
    ) -> None:
        """validate=True (the new default) rejects negative firing rates.

        Firing rates are non-negative by definition (they're rates of a
        Poisson process). A negative rate is almost always a bug
        upstream (e.g., subtracting a baseline that exceeded the
        observed rate); rejecting at the decoder boundary surfaces it.
        """
        from neurospatial.decoding.posterior import decode_position

        n_bins = simple_env.n_bins
        encoding_negative = np.ones((2, n_bins))
        encoding_negative[0, 0] = -0.5
        with pytest.raises(ValueError, match=r"negative entr"):
            decode_position(
                simple_env,
                simple_spike_counts,
                encoding_negative,
                dt=0.025,
            )

    def test_validate_catches_fractional_spike_counts(
        self,
        simple_env: Environment,
    ) -> None:
        """validate=True rejects fractional float-dtype spike counts.

        Regression for review follow-up: the validator's error message
        promised "non-negative integers" but the implementation only
        checked finite + non-negative, so a float array like
        [[0.5, 1.0]] passed and produced a "valid"-looking posterior
        from a Poisson likelihood that is undefined at non-integer n.
        """
        from neurospatial.decoding.posterior import decode_position

        n_bins = simple_env.n_bins
        encoding_models = np.ones((2, n_bins))
        spike_counts_fractional = np.array([[0.5, 1.0]])
        with pytest.raises(ValueError, match=r"fractional entr"):
            decode_position(
                simple_env, spike_counts_fractional, encoding_models, dt=0.025
            )

    def test_validate_accepts_float_spike_counts_with_integer_values(
        self,
        simple_env: Environment,
    ) -> None:
        """Float arrays with integer-valued entries are still allowed.

        Spike counts are sometimes carried in float64 columns for
        ergonomic reasons (mixed dataframes, JAX backends). Validating
        only the *value* (np.equal(x, floor(x))) -- not the dtype --
        lets those legitimate cases through while still catching truly
        fractional inputs. The float and int paths must produce the
        same posterior up to floating-point tolerance.
        """
        from neurospatial.decoding.posterior import decode_position

        n_bins = simple_env.n_bins
        encoding_models = np.ones((2, n_bins))
        spike_counts_float = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
        spike_counts_int = spike_counts_float.astype(np.int64)

        result_float = decode_position(
            simple_env, spike_counts_float, encoding_models, dt=0.025
        )
        result_int = decode_position(
            simple_env, spike_counts_int, encoding_models, dt=0.025
        )
        # Float path with integer values must match the int path exactly
        # (modulo dtype-promotion rounding) and must yield a well-formed
        # row-stochastic posterior.
        posterior_float = np.asarray(result_float.posterior)
        posterior_int = np.asarray(result_int.posterior)
        np.testing.assert_allclose(posterior_float, posterior_int, atol=1e-10)
        assert np.all(np.isfinite(posterior_float))
        np.testing.assert_allclose(posterior_float.sum(axis=-1), 1.0, atol=1e-9)

    def test_validate_catches_negative_prior(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True rejects priors with negative entries.

        Regression for review follow-up: the prior was checked for
        finite values but not for non-negativity, despite the error
        message claiming "Prior must be finite non-negative values".
        Internal normalization divides by the row sum, so a negative
        entry could yield a finite-looking posterior that no longer
        represents a probability distribution.
        """
        from neurospatial.decoding.posterior import decode_position

        prior = np.full(simple_env.n_bins, 1.0)
        prior[0] = -1.0
        with pytest.raises(ValueError, match=r"negative entr"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                prior=prior,
            )

    def test_validate_catches_zero_mass_prior(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True rejects priors whose total mass is zero.

        Regression for review follow-up: an all-zero prior was silently
        rebuilt as a uniform prior by normalize_to_posterior's 1e-10
        clip, producing a finite-looking posterior that did not reflect
        the user's stated prior. The validator now rejects at the
        boundary so the failure is visible.
        """
        from neurospatial.decoding.posterior import decode_position

        with pytest.raises(ValueError, match=r"zero total mass"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                prior=np.zeros(simple_env.n_bins),
            )

    def test_validate_catches_time_varying_prior_with_zero_row(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True rejects time-varying priors with any zero-mass row."""
        from neurospatial.decoding.posterior import decode_position

        n_time = simple_spike_counts.shape[0]
        prior = np.ones((n_time, simple_env.n_bins))
        prior[1, :] = 0.0  # one zero-mass time bin
        with pytest.raises(ValueError, match=r"zero total mass"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                prior=prior,
            )

    def test_validate_accepts_array_like_prior(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """validate=True accepts list/tuple priors via np.asarray conversion.

        Regression for review follow-up: the validator was running
        ``prior < 0`` directly on the user's input, which raised
        ``TypeError`` for any array-like that wasn't already an ndarray
        (e.g., a Python list). normalize_to_posterior accepts array-like
        priors via np.asarray, so the validator must too.

        The list-prior and ndarray-prior paths must produce identical
        posteriors — verifying that the conversion is faithful and not
        just that the call returned without error.
        """
        from neurospatial.decoding.posterior import decode_position

        prior_list = [1.0] * simple_env.n_bins
        prior_array = np.asarray(prior_list)

        result_list = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            prior=prior_list,  # type: ignore[arg-type]  # runtime-only array-like
        )
        result_array = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            prior=prior_array,
        )
        posterior_list = np.asarray(result_list.posterior)
        posterior_array = np.asarray(result_array.posterior)
        np.testing.assert_allclose(posterior_list, posterior_array, atol=1e-12)
        assert np.all(np.isfinite(posterior_list))
        np.testing.assert_allclose(posterior_list.sum(axis=-1), 1.0, atol=1e-9)

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
        assert result.posterior_entropy.shape == (simple_spike_counts.shape[0],)

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
# Hybrid time_chunk: None = byte-exact full path; int = blockwise likelihood
# =============================================================================


class TestDecodePositionTimeChunkHybrid:
    """``decode_position`` hybrid ``time_chunk`` semantics.

    ``time_chunk=None`` (default) keeps the full-matmul path byte-for-byte
    unchanged. An explicit ``time_chunk=N`` computes the Poisson
    log-likelihood blockwise into the preallocated posterior; this is
    tolerance-equal to the full path (~1e-15; MAP/argmax identical; rows sum
    to 1) due to BLAS shape-dependence.
    """

    @staticmethod
    def _full_path_reference(env, spike_counts, encoding_models, dt, *, prior=None):
        """Recompute the full-matmul posterior independently of decode_position.

        Mirrors the ``time_chunk=None`` code path exactly: full Poisson
        log-likelihood over the whole window -> normalize_to_posterior with
        ``time_chunk=None``.
        """
        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import normalize_to_posterior

        log_ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)
        return normalize_to_posterior(log_ll, prior=prior, time_chunk=None)

    def test_default_is_byte_exact_full_path(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """time_chunk=None (and the plain default) == the full-matmul reference.

        Guards that the default path was NOT routed through blockwise: it must
        reproduce the independently-computed full-matmul posterior
        byte-for-byte (atol=0).
        """
        from neurospatial.decoding.posterior import decode_position

        reference = self._full_path_reference(
            simple_env, simple_spike_counts, simple_encoding_models, 0.025
        )

        result_default = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        result_none = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            time_chunk=None,
        )

        assert_array_equal(np.asarray(result_default.posterior), reference)
        assert_array_equal(np.asarray(result_none.posterior), reference)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_small_env_time_chunk_parity(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
        k: int,
    ) -> None:
        """time_chunk=k matches the full path to tolerance with identical MAP."""
        from neurospatial.decoding.posterior import decode_position

        full = decode_position(
            simple_env, simple_spike_counts, simple_encoding_models, dt=0.025
        )
        chunked = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            time_chunk=k,
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    @staticmethod
    def _bigger_inputs(env, *, n_time=40, n_neurons=8, seed=3):
        rng = np.random.default_rng(seed)
        spike_counts = rng.poisson(1.5, (n_time, n_neurons)).astype(np.int64)
        encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))
        return spike_counts, encoding_models

    @pytest.mark.parametrize("k", [1, 7, 64, 40, 45])
    def test_time_chunk_parity_stationary_prior(
        self, medium_2d_env: Environment, k: int
    ) -> None:
        """Parity for k in {1,7,64,n_time,n_time+5} with a 1-D stationary prior.

        Covers non-dividing block sizes (7, 45) and a stationary prior reused
        across blocks.
        """
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)
        rng = np.random.default_rng(11)
        prior = rng.uniform(0.1, 1.0, env.n_bins)

        full = decode_position(
            env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        chunked = decode_position(
            env, spike_counts, encoding_models, dt=0.025, prior=prior, time_chunk=k
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    @pytest.mark.parametrize("k", [1, 7, 64, 40, 45])
    def test_time_chunk_parity_time_varying_prior(
        self, medium_2d_env: Environment, k: int
    ) -> None:
        """Parity with a 2-D time-varying prior sliced per block."""
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)
        rng = np.random.default_rng(12)
        prior = rng.uniform(0.1, 1.0, (n_time, env.n_bins))

        full = decode_position(
            env, spike_counts, encoding_models, dt=0.025, prior=prior
        )
        chunked = decode_position(
            env, spike_counts, encoding_models, dt=0.025, prior=prior, time_chunk=k
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    @pytest.mark.parametrize("k", [1, 7, 64, 40, 45])
    def test_time_chunk_parity_nan_safe_encoding(
        self, medium_2d_env: Environment, k: int
    ) -> None:
        """Parity for the NaN-safe encoding-model path (some NaN bins).

        validate=False so NaN encoding bins are absorbed as zero-rate; the
        block loop must route through the NaN-safe likelihood identically.
        """
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)
        # Inject NaN into a few (neuron, bin) entries, leaving every bin finite
        # for at least one neuron (so the no-finite-bins guard does not fire).
        encoding_models = encoding_models.copy()
        encoding_models[0, 1] = np.nan
        encoding_models[2, 3] = np.nan
        encoding_models[1, 5] = np.nan

        with pytest.warns(UserWarning, match="non-finite"):
            full = decode_position(
                env,
                spike_counts,
                encoding_models,
                dt=0.025,
                validate=False,
            )
        with pytest.warns(UserWarning, match="non-finite"):
            chunked = decode_position(
                env,
                spike_counts,
                encoding_models,
                dt=0.025,
                validate=False,
                time_chunk=k,
            )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12, equal_nan=True)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    @pytest.mark.parametrize("k", [1, 7, 64, 40, 45])
    def test_time_chunk_parity_float32(
        self, medium_2d_env: Environment, k: int
    ) -> None:
        """Parity at dtype=float32 (looser atol) with identical MAP."""
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)

        full = decode_position(
            env, spike_counts, encoding_models, dt=0.025, dtype=np.float32
        )
        chunked = decode_position(
            env,
            spike_counts,
            encoding_models,
            dt=0.025,
            dtype=np.float32,
            time_chunk=k,
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert post_chunked.dtype == np.float32
        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-5)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("validate", [True, False])
    def test_time_chunk_parity_validate_flag(
        self, medium_2d_env: Environment, validate: bool
    ) -> None:
        """Parity holds with validate=True and validate=False, non-dividing k."""
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)

        full = decode_position(
            env, spike_counts, encoding_models, dt=0.025, validate=validate
        )
        chunked = decode_position(
            env,
            spike_counts,
            encoding_models,
            dt=0.025,
            validate=validate,
            time_chunk=7,
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    def test_time_chunk_degenerate_all_neg_inf_rows(
        self, medium_2d_env: Environment
    ) -> None:
        """All -inf rows (zero spikes + flat model) handled identically.

        Uses a non-dividing block size so a degenerate row falls on a block
        boundary, and checks handle_degenerate='uniform' (the default) gives
        the same uniform rows under chunking as the full path.
        """
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 13
        n_neurons = 4
        # Flat encoding model -> zero-spike rows are all -inf (degenerate).
        encoding_models = np.full((n_neurons, env.n_bins), 3.0)
        rng = np.random.default_rng(7)
        spike_counts = rng.poisson(1.0, (n_time, n_neurons)).astype(np.int64)
        # Force a couple of all-zero (degenerate) rows at non-block-boundaries.
        spike_counts[5] = 0
        spike_counts[10] = 0

        full = decode_position(env, spike_counts, encoding_models, dt=0.025)
        chunked = decode_position(
            env, spike_counts, encoding_models, dt=0.025, time_chunk=7
        )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        # Degenerate rows became uniform on the full path; chunking must match.
        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    def test_time_chunk_degenerate_nan_injected_rows(
        self, medium_2d_env: Environment
    ) -> None:
        """NaN-injected degenerate handling matches across a block boundary.

        handle_degenerate is 'uniform' by default; an all-NaN bin (NaN across
        every neuron) becomes -inf in the likelihood, which combined with a
        zero-spike row can be exercised. We instead inject a fully-degenerate
        encoding (all-NaN bin) and check the chunked/full paths agree.
        """
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 13
        n_neurons = 4
        rng = np.random.default_rng(9)
        encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))
        # Make one bin all-NaN (NaN for every neuron): it becomes -inf in the
        # likelihood (zero posterior mass) on both paths.
        encoding_models[:, 2] = np.nan
        spike_counts = rng.poisson(1.5, (n_time, n_neurons)).astype(np.int64)

        with pytest.warns(UserWarning, match="non-finite"):
            full = decode_position(
                env, spike_counts, encoding_models, dt=0.025, validate=False
            )
        with pytest.warns(UserWarning, match="non-finite"):
            chunked = decode_position(
                env,
                spike_counts,
                encoding_models,
                dt=0.025,
                validate=False,
                time_chunk=7,
            )
        post_full = np.asarray(full.posterior)
        post_chunked = np.asarray(chunked.posterior)

        assert np.allclose(post_chunked, post_full, rtol=0, atol=1e-12, equal_nan=True)
        # The all-NaN bin has zero posterior mass on both paths.
        assert np.allclose(post_chunked[:, 2], 0.0, atol=1e-12)
        assert_array_equal(chunked.map_estimate, full.map_estimate)
        assert_allclose(post_chunked.sum(axis=1), 1.0, atol=1e-9)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_time_chunk_below_one_raises(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
        bad: int,
    ) -> None:
        """time_chunk < 1 is rejected up front with a clear ValueError."""
        from neurospatial.decoding.posterior import decode_position

        with pytest.raises(ValueError, match="time_chunk must be a positive integer"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                time_chunk=bad,
            )

    @pytest.mark.parametrize(
        ("prior_shape", "match"),
        [
            # Wrong 1-D prior length -> caught by the up-front _validate_prior_shape.
            (("n_bins+5",), "1D prior must have shape"),
            # Short 2-D prior: the silent-truncation footgun the up-front check
            # closes. The block loop only slices n_time rows, so without the
            # up-front check a too-short prior would be caught (if at all) only at
            # the final block -- here it must raise before any block is decoded.
            (("n_time-1", "n_bins"), "2D prior must have shape"),
            # Over-long 2-D prior also raises up front.
            (("n_time+3", "n_bins"), "2D prior must have shape"),
        ],
    )
    def test_time_chunk_prior_shape_validation(
        self,
        medium_2d_env: Environment,
        prior_shape: tuple,
        match: str,
    ) -> None:
        """Blockwise time_chunk path validates prior shape up front.

        ``_decode_blockwise`` calls ``_validate_prior_shape`` before the
        time-block loop, so a wrong-length 1-D prior or a wrong-length 2-D prior
        raises a clear ``ValueError`` immediately rather than being silently
        truncated/looped into a per-block slice. Exercises the ``time_chunk=N``
        branch (not ``normalize_to_posterior``'s own check) -- a short 2-D prior
        is the silent-truncation case the up-front check closes.
        """
        from neurospatial.decoding.posterior import decode_position

        env = medium_2d_env
        n_time = 40
        spike_counts, encoding_models = self._bigger_inputs(env, n_time=n_time)

        dims = {"n_bins": env.n_bins, "n_time": n_time}

        def _resolve(spec: str) -> int:
            for name, value in dims.items():
                if spec.startswith(name):
                    offset = spec[len(name) :]
                    return value + (int(offset) if offset else 0)
            raise AssertionError(f"unrecognized dim spec: {spec}")

        shape = tuple(_resolve(s) for s in prior_shape)
        prior = np.random.default_rng(0).uniform(0.1, 1.0, shape)

        with pytest.raises(ValueError, match=match):
            decode_position(
                env,
                spike_counts,
                encoding_models,
                dt=0.025,
                prior=prior,
                time_chunk=7,
            )


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


class TestPosteriorClosedForm:
    """Pin Bayes' rule against a closed-form answer, not against the function itself.

    The existing tests check that rows sum to 1; these check the actual posterior
    values on toy problems where the answer can be hand-computed, so a bug in the
    log-space arithmetic that still produced a normalized (but wrong) posterior
    would be caught.
    """

    def test_two_bin_one_neuron_bayes_closed_form(self) -> None:
        from scipy.stats import poisson

        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import normalize_to_posterior

        # 1 neuron, 2 bins, lambda=[5, 10] Hz, dt=0.1 s, n=2 spikes, uniform prior.
        spike_counts = np.array([[2]], dtype=np.int64)  # (n_time=1, n_neurons=1)
        encoding_models = np.array([[5.0, 10.0]])  # (n_neurons=1, n_bins=2)
        dt = 0.1
        log_likelihood = log_poisson_likelihood(spike_counts, encoding_models, dt)
        posterior = normalize_to_posterior(log_likelihood, prior=np.array([0.5, 0.5]))

        # Closed-form Bayes posterior from the Poisson pmf. The omitted -log(n!)
        # term and the 1/n! factor cancel in normalization, so the analytic pmf
        # ratio is the ground truth.
        pmf = poisson.pmf(2, np.array([5.0, 10.0]) * dt)
        expected = pmf / pmf.sum()
        np.testing.assert_allclose(posterior[0], expected, atol=1e-12)

    def test_uniform_likelihood_returns_prior(self) -> None:
        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import normalize_to_posterior

        # Identical firing rate in every bin -> likelihood is flat across bins,
        # so the posterior must equal the (normalized) prior exactly.
        encoding_models = np.array([[7.0, 7.0, 7.0]])
        log_likelihood = log_poisson_likelihood(
            np.array([[3]], dtype=np.int64), encoding_models, 0.1
        )
        prior = np.array([0.2, 0.3, 0.5])
        posterior = normalize_to_posterior(log_likelihood, prior=prior)
        np.testing.assert_allclose(posterior[0], prior, atol=1e-12)

    def test_zero_prior_bin_is_negligible(self) -> None:
        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import normalize_to_posterior

        # A zero-prior bin should carry essentially no posterior mass. Note:
        # normalize_to_posterior clips priors to 1e-10 before taking the log
        # (see its docstring), so the bin is ~1e-10, not exactly 0.
        encoding_models = np.array([[5.0, 10.0, 8.0]])
        log_likelihood = log_poisson_likelihood(
            np.array([[2]], dtype=np.int64), encoding_models, 0.1
        )
        posterior = normalize_to_posterior(
            log_likelihood, prior=np.array([0.5, 0.5, 0.0])
        )
        assert posterior[0, 2] < 1e-8
        assert np.isclose(posterior[0].sum(), 1.0, atol=1e-12)


class TestLongTrajectoryStability:
    """The log-sum-exp posterior must stay finite and normalized on long inputs.

    Products of per-neuron Poisson likelihoods underflow in linear space over
    many time bins / large populations; the log-space implementation must not.
    """

    def test_long_trajectory_finite_normalized_posterior(self) -> None:
        from neurospatial.decoding.likelihood import log_poisson_likelihood
        from neurospatial.decoding.posterior import normalize_to_posterior

        rng = np.random.default_rng(0)
        n_time, n_neurons, n_bins = 100_000, 50, 80
        encoding_models = rng.uniform(0.5, 30.0, (n_neurons, n_bins))
        dt = 0.02
        spike_counts = rng.poisson(
            encoding_models.mean(axis=1)[None, :] * dt, (n_time, n_neurons)
        ).astype(np.int64)

        log_likelihood = log_poisson_likelihood(spike_counts, encoding_models, dt)
        posterior = normalize_to_posterior(log_likelihood)

        assert np.isfinite(posterior).all()
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)


# =============================================================================
# decode_position input-validation regressions
# =============================================================================


class TestDecodePositionValidation:
    """decode_position must reject mismatched bin counts and times lengths."""

    def test_decode_position_rejects_bin_count_mismatch(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
    ) -> None:
        """encoding_models with wrong n_bins raises, with and without validate."""
        from neurospatial.decoding.posterior import decode_position

        n_neurons = simple_spike_counts.shape[1]
        wrong_n_bins = simple_env.n_bins + 3
        rng = np.random.default_rng(0)
        bad_models = (
            np.abs(rng.standard_normal((n_neurons, wrong_n_bins))) * 10 + 0.1
        ).astype(np.float64)

        with pytest.raises(ValueError, match="bins"):
            decode_position(
                simple_env, simple_spike_counts, bad_models, dt=0.025, validate=True
            )

        with pytest.raises(ValueError, match="bins"):
            decode_position(
                simple_env, simple_spike_counts, bad_models, dt=0.025, validate=False
            )

    def test_decode_position_rejects_times_length_mismatch(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        """times of the wrong length raises a 'Length mismatch' ValueError."""
        from neurospatial.decoding.posterior import decode_position

        n_time_bins = simple_spike_counts.shape[0]
        bad_times = np.arange(n_time_bins + 2, dtype=np.float64)

        with pytest.raises(ValueError, match="Length mismatch"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                times=bad_times,
            )


# =============================================================================
# Tests for centralized time_chunk validation (Fix B)
# =============================================================================


class TestTimeChunkValidation:
    """time_chunk must be a genuine positive int (not bool/float/str)."""

    @pytest.mark.parametrize("bad", [1.5, "2", True, False, 0, -1])
    def test_normalize_to_posterior_rejects_bad_time_chunk(
        self, simple_log_likelihood: np.ndarray, bad: object
    ) -> None:
        from neurospatial.decoding.posterior import normalize_to_posterior

        with pytest.raises(ValueError, match="time_chunk"):
            normalize_to_posterior(simple_log_likelihood, time_chunk=bad)

    @pytest.mark.parametrize("good", [None, 1, 2, np.int64(3)])
    def test_normalize_to_posterior_accepts_good_time_chunk(
        self, simple_log_likelihood: np.ndarray, good: object
    ) -> None:
        from neurospatial.decoding.posterior import normalize_to_posterior

        posterior = normalize_to_posterior(simple_log_likelihood, time_chunk=good)
        assert posterior.shape == simple_log_likelihood.shape

    @pytest.mark.parametrize("bad", [1.5, "2", True, False, 0, -1])
    def test_decode_position_rejects_bad_time_chunk(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
        bad: object,
    ) -> None:
        from neurospatial.decoding.posterior import decode_position

        with pytest.raises(ValueError, match="time_chunk"):
            decode_position(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                time_chunk=bad,
            )

    @pytest.mark.parametrize("good", [None, 1, 2, np.int64(3)])
    def test_decode_position_accepts_good_time_chunk(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
        good: object,
    ) -> None:
        from neurospatial.decoding.posterior import decode_position

        result = decode_position(
            simple_env,
            simple_spike_counts,
            simple_encoding_models,
            dt=0.025,
            time_chunk=good,
        )
        assert result.posterior.shape[0] == simple_spike_counts.shape[0]

    @pytest.mark.parametrize("bad", [1.5, "2", True, False, 0, -1])
    def test_decode_position_summary_rejects_bad_time_chunk(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
        bad: object,
    ) -> None:
        from neurospatial.decoding.posterior import decode_position_summary

        with pytest.raises(ValueError, match="time_chunk"):
            decode_position_summary(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                time_chunk=bad,
            )

    def test_decode_position_summary_none_time_chunk_specific_message(
        self,
        simple_env: Environment,
        simple_spike_counts: np.ndarray,
        simple_encoding_models: np.ndarray,
    ) -> None:
        from neurospatial.decoding.posterior import decode_position_summary

        with pytest.raises(ValueError, match="decode_position"):
            decode_position_summary(
                simple_env,
                simple_spike_counts,
                simple_encoding_models,
                dt=0.025,
                time_chunk=None,
            )
