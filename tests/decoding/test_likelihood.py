"""Tests for likelihood functions in neurospatial.decoding.likelihood.

Tests cover:
- log_poisson_likelihood: numerically stable log-likelihood computation
- poisson_likelihood: thin wrapper with underflow protection
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial.decoding.likelihood import (
    log_poisson_likelihood,
    poisson_likelihood,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_spike_counts() -> np.ndarray:
    """Simple spike count matrix: (2 time bins, 2 neurons)."""
    return np.array([[0, 1], [2, 0]], dtype=np.int64)


@pytest.fixture
def simple_encoding_models() -> np.ndarray:
    """Simple encoding models: (2 neurons, 3 bins)."""
    return np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64)


@pytest.fixture
def uniform_encoding_models() -> np.ndarray:
    """Flat encoding models: all bins have same rate."""
    return np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float64)


@pytest.fixture
def large_spike_counts() -> np.ndarray:
    """Larger spike count matrix for stress tests: (100 time bins, 50 neurons)."""
    rng = np.random.default_rng(42)
    return rng.poisson(2.0, size=(100, 50)).astype(np.int64)


@pytest.fixture
def large_encoding_models() -> np.ndarray:
    """Larger encoding models: (50 neurons, 200 bins)."""
    # Create place-field-like models (peaked at different locations)
    n_neurons, n_bins = 50, 200
    models = np.zeros((n_neurons, n_bins), dtype=np.float64)
    peak_bins = np.linspace(10, 190, n_neurons).astype(int)
    for i, peak in enumerate(peak_bins):
        x = np.arange(n_bins)
        models[i] = 20.0 * np.exp(-0.5 * ((x - peak) / 10.0) ** 2)
    # Add small baseline rate
    models += 0.1
    return models


# =============================================================================
# Tests for log_poisson_likelihood
# =============================================================================


class TestLogPoissonLikelihood:
    """Tests for log_poisson_likelihood function."""

    def test_output_shape(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output shape should be (n_time_bins, n_bins)."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert ll.shape == (2, 3)

    def test_output_dtype(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output dtype should be float64."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert ll.dtype == np.float64

    def test_finite_output(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output should be finite (no NaN or Inf)."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert np.isfinite(ll).all()

    def test_min_rate_prevents_negative_infinity(self) -> None:
        """min_rate parameter should prevent -inf from zero rates."""
        spike_counts = np.array([[1]], dtype=np.int64)
        # Zero rate would cause log(0) = -inf without min_rate
        encoding_models = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        assert np.isfinite(ll).all()

    def test_custom_min_rate(self) -> None:
        """Custom min_rate should be respected."""
        spike_counts = np.array([[1]], dtype=np.int64)
        encoding_models = np.array([[0.0]], dtype=np.float64)

        # With default min_rate (1e-10)
        ll_default = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)

        # With larger min_rate (1e-5)
        ll_larger = log_poisson_likelihood(
            spike_counts, encoding_models, dt=0.025, min_rate=1e-5
        )

        # Larger min_rate should give higher (less negative) log-likelihood
        assert ll_larger[0, 0] > ll_default[0, 0]

    def test_zero_spikes_all_bins(self) -> None:
        """With zero spikes, log-likelihood should be valid (rate-only term)."""
        spike_counts = np.array([[0, 0]], dtype=np.int64)
        encoding_models = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        assert np.isfinite(ll).all()
        # With zero spikes, log-likelihood should be -lambda*dt (negative)
        assert (ll <= 0).all()

    def test_higher_rate_higher_likelihood_when_spikes(self) -> None:
        """Bins with higher rate should have higher likelihood when spikes occur."""
        spike_counts = np.array([[5]], dtype=np.int64)
        encoding_models = np.array([[1.0, 10.0, 20.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        # With 5 spikes, higher rate bins should have higher likelihood
        # (up to a point - Poisson likelihood peaks at rate = count/dt)
        # For dt=0.025 and 5 spikes, expected rate is 200 Hz
        # So 20 Hz is still well below optimal, meaning higher is better
        assert ll[0, 2] > ll[0, 1] > ll[0, 0]

    def test_lower_rate_higher_likelihood_when_no_spikes(self) -> None:
        """Bins with lower rate should have higher likelihood when no spikes occur."""
        spike_counts = np.array([[0]], dtype=np.int64)
        encoding_models = np.array([[1.0, 10.0, 20.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        # With 0 spikes, lower rate bins should have higher (less negative) likelihood
        assert ll[0, 0] > ll[0, 1] > ll[0, 2]

    def test_formula_correctness(self) -> None:
        """Verify formula: sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]."""
        spike_counts = np.array([[1, 2]], dtype=np.int64)
        # One neuron with rate 10 Hz, another with rate 20 Hz
        encoding_models = np.array([[10.0, 20.0], [20.0, 10.0]], dtype=np.float64)
        dt = 0.025

        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)

        # Manual calculation for bin 0 (rates 10, 20 Hz):
        # neuron 0: 1 * log(10 * 0.025) - 10 * 0.025 = 1 * log(0.25) - 0.25
        # neuron 1: 2 * log(20 * 0.025) - 20 * 0.025 = 2 * log(0.5) - 0.5
        # Total = log(0.25) - 0.25 + 2*log(0.5) - 0.5
        expected_bin0 = (
            1 * np.log(10 * 0.025) - 10 * 0.025 + 2 * np.log(20 * 0.025) - 20 * 0.025
        )
        assert_allclose(ll[0, 0], expected_bin0, rtol=1e-10)

    def test_time_bin_independence(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Each time bin's likelihood should be independent of others."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )

        # Compute separately for each time bin
        ll_t0 = log_poisson_likelihood(
            simple_spike_counts[0:1], simple_encoding_models, dt=0.025
        )
        ll_t1 = log_poisson_likelihood(
            simple_spike_counts[1:2], simple_encoding_models, dt=0.025
        )

        assert_allclose(ll[0], ll_t0[0])
        assert_allclose(ll[1], ll_t1[0])

    def test_large_population(
        self, large_spike_counts: np.ndarray, large_encoding_models: np.ndarray
    ) -> None:
        """Should handle large populations without overflow/underflow."""
        ll = log_poisson_likelihood(large_spike_counts, large_encoding_models, dt=0.025)
        assert ll.shape == (100, 200)
        assert np.isfinite(ll).all()

    def test_extreme_rates_warning(self) -> None:
        """Very high rates should still be numerically stable."""
        spike_counts = np.array([[10]], dtype=np.int64)
        # Very high rate: 1000 Hz
        encoding_models = np.array([[1000.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.1)
        # With lambda*dt = 100, this is extreme but should still be finite
        assert np.isfinite(ll).all()

    def test_different_dt_values(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Different dt values should scale appropriately."""
        ll_short = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.001
        )
        ll_long = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.1
        )

        # With longer dt, the rate penalty term (-lambda*dt) is larger
        # This is a basic sanity check - exact relationship depends on spike counts
        assert ll_short.shape == ll_long.shape
        assert np.isfinite(ll_short).all()
        assert np.isfinite(ll_long).all()


# =============================================================================
# Tests for poisson_likelihood
# =============================================================================


class TestPoissonLikelihood:
    """Tests for poisson_likelihood function (thin wrapper)."""

    def test_output_shape(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output shape should be (n_time_bins, n_bins)."""
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert likelihood.shape == (2, 3)

    def test_output_dtype(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output dtype should be float64."""
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert likelihood.dtype == np.float64

    def test_non_negative(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Likelihoods should be non-negative."""
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert (likelihood >= 0).all()

    def test_max_is_one(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Maximum likelihood per row should be 1.0 (normalized to prevent underflow)."""
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        row_maxes = likelihood.max(axis=1)
        assert_allclose(row_maxes, 1.0)

    def test_finite_output(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Output should be finite (no NaN or Inf)."""
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        assert np.isfinite(likelihood).all()

    def test_consistent_with_log_likelihood(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Should be consistent with exp of log_poisson_likelihood (up to normalization)."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )

        # Manually compute what poisson_likelihood should return
        ll_shifted = ll - ll.max(axis=1, keepdims=True)
        expected = np.exp(ll_shifted)

        assert_allclose(likelihood, expected)

    def test_ranking_preserved(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Ranking of bins should match log_poisson_likelihood."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )
        likelihood = poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )

        # Argmax should match
        assert_array_equal(ll.argmax(axis=1), likelihood.argmax(axis=1))

    def test_large_population_no_underflow(
        self, large_spike_counts: np.ndarray, large_encoding_models: np.ndarray
    ) -> None:
        """Should handle large populations without underflow to zero."""
        likelihood = poisson_likelihood(
            large_spike_counts, large_encoding_models, dt=0.025
        )
        assert np.isfinite(likelihood).all()
        # At least one value per row should be nonzero (the max is 1.0)
        assert (likelihood.max(axis=1) == 1.0).all()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_time_bin(self, simple_encoding_models: np.ndarray) -> None:
        """Should handle single time bin."""
        spike_counts = np.array([[1, 2]], dtype=np.int64)
        ll = log_poisson_likelihood(spike_counts, simple_encoding_models, dt=0.025)
        assert ll.shape == (1, 3)
        assert np.isfinite(ll).all()

    def test_single_neuron(self) -> None:
        """Should handle single neuron."""
        spike_counts = np.array([[1], [2]], dtype=np.int64)
        encoding_models = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        assert ll.shape == (2, 3)
        assert np.isfinite(ll).all()

    def test_single_bin(self) -> None:
        """Should handle single spatial bin."""
        spike_counts = np.array([[1, 2]], dtype=np.int64)
        encoding_models = np.array([[5.0], [10.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        assert ll.shape == (1, 1)
        assert np.isfinite(ll).all()

    def test_all_zeros_spike_counts(self) -> None:
        """Should handle all-zero spike counts."""
        spike_counts = np.zeros((3, 2), dtype=np.int64)
        encoding_models = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=0.025)
        assert ll.shape == (3, 2)
        assert np.isfinite(ll).all()

    def test_very_small_dt(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Should handle very small dt values."""
        ll = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=1e-6
        )
        assert np.isfinite(ll).all()

    def test_input_not_modified(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Input arrays should not be modified."""
        spike_counts_copy = simple_spike_counts.copy()
        encoding_models_copy = simple_encoding_models.copy()

        _ = log_poisson_likelihood(
            simple_spike_counts, simple_encoding_models, dt=0.025
        )

        assert_array_equal(simple_spike_counts, spike_counts_copy)
        assert_array_equal(simple_encoding_models, encoding_models_copy)

    def test_dt_zero_raises(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """dt=0 should raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            log_poisson_likelihood(simple_spike_counts, simple_encoding_models, dt=0.0)

    def test_dt_negative_raises(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Negative dt should raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            log_poisson_likelihood(
                simple_spike_counts, simple_encoding_models, dt=-0.025
            )

    def test_min_rate_zero_raises(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """min_rate=0 should raise ValueError."""
        with pytest.raises(ValueError, match="min_rate must be positive"):
            log_poisson_likelihood(
                simple_spike_counts, simple_encoding_models, dt=0.025, min_rate=0.0
            )

    def test_min_rate_negative_raises(
        self, simple_spike_counts: np.ndarray, simple_encoding_models: np.ndarray
    ) -> None:
        """Negative min_rate should raise ValueError."""
        with pytest.raises(ValueError, match="min_rate must be positive"):
            log_poisson_likelihood(
                simple_spike_counts, simple_encoding_models, dt=0.025, min_rate=-1e-10
            )


# =============================================================================
# Reference Implementation Comparison (placeholder for future)
# =============================================================================


class TestReferenceComparison:
    """Tests comparing against reference implementation.

    These tests compare against known correct values from
    replay_trajectory_classification or manual calculations.
    """

    def test_known_values(self) -> None:
        """Test against manually calculated known values."""
        # Simple case: 1 time bin, 1 neuron, 2 bins
        spike_counts = np.array([[2]], dtype=np.int64)
        encoding_models = np.array([[10.0, 20.0]], dtype=np.float64)  # Hz
        dt = 0.05  # 50 ms

        ll = log_poisson_likelihood(spike_counts, encoding_models, dt=dt)

        # Manual calculation:
        # Bin 0: lambda*dt = 10 * 0.05 = 0.5
        # ll = n*log(lambda*dt) - lambda*dt = 2*log(0.5) - 0.5 = -1.886...
        expected_bin0 = 2 * np.log(0.5) - 0.5

        # Bin 1: lambda*dt = 20 * 0.05 = 1.0
        # ll = n*log(lambda*dt) - lambda*dt = 2*log(1.0) - 1.0 = -1.0
        expected_bin1 = 2 * np.log(1.0) - 1.0

        assert_allclose(ll[0, 0], expected_bin0, rtol=1e-10)
        assert_allclose(ll[0, 1], expected_bin1, rtol=1e-10)
