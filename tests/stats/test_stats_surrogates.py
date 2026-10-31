"""Tests for the surrogate spike-train generators in neurospatial.stats.surrogates.

Covers generate_poisson_surrogates, generate_inhomogeneous_poisson_surrogates
and generate_jittered_spikes: their output contract, the distributions they
sample and the inputs they refuse.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


class TestGeneratePoissonSurrogates:
    """Tests for generate_poisson_surrogates (homogeneous Poisson surrogates)."""

    @pytest.fixture
    def spike_counts(self) -> np.ndarray:
        """Sample spike counts: (10 time bins, 5 neurons)."""
        rng = np.random.default_rng(42)
        # Generate realistic spike counts (Poisson-like)
        return rng.poisson(lam=2.0, size=(10, 5)).astype(np.int64)

    def test_yields_correct_number_of_surrogates(
        self, spike_counts: np.ndarray
    ) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        n_surrogates = 10
        surrogates = list(
            generate_poisson_surrogates(spike_counts, n_surrogates=n_surrogates, rng=42)
        )
        assert len(surrogates) == n_surrogates

    def test_yields_non_negative_counts(self, spike_counts: np.ndarray) -> None:
        """Surrogate spike counts should be non-negative."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, n_surrogates=10, rng=42
        ):
            assert (surrogate >= 0).all()

    def test_preserves_mean_rate_approximately(self, spike_counts: np.ndarray) -> None:
        """Mean firing rates should be preserved (statistically).

        Surrogates are generated from mean rates, so average across many
        surrogates should approximate original mean.
        """
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        # Compute original mean rate per neuron
        original_mean_per_neuron = spike_counts.mean(axis=0)

        # Collect mean rates from many surrogates
        surrogate_means = []
        for surrogate in generate_poisson_surrogates(
            spike_counts, n_surrogates=500, rng=42
        ):
            surrogate_means.append(surrogate.mean(axis=0))

        # Average across surrogates
        avg_surrogate_mean = np.mean(surrogate_means, axis=0)

        # Should be close to original (within ~20% for statistical test)
        assert np.allclose(avg_surrogate_mean, original_mean_per_neuron, rtol=0.2), (
            f"Mean rates not preserved: original={original_mean_per_neuron}, "
            f"surrogate avg={avg_surrogate_mean}"
        )

    def test_destroys_temporal_structure(self, spike_counts: np.ndarray) -> None:
        """Surrogates should have destroyed temporal autocorrelation.

        Each time bin is independently sampled, so temporal patterns are lost.
        """
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        # This is a statistical property test - surrogates should not replicate
        # the exact temporal pattern of the original
        for surrogate in generate_poisson_surrogates(
            spike_counts, n_surrogates=5, rng=42
        ):
            # Surrogates should be different from original in most cases
            # (exact match is extremely unlikely)
            assert not np.array_equal(surrogate, spike_counts)

    def test_zero_spike_counts(self) -> None:
        """Should handle all-zero spike counts (silent neurons)."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        zeros = np.zeros((5, 3), dtype=np.int64)
        for surrogate in generate_poisson_surrogates(zeros, n_surrogates=5, rng=42):
            # With zero mean rate, Poisson should produce mostly zeros
            assert surrogate.shape == zeros.shape
            assert (surrogate >= 0).all()

    def test_high_firing_rate(self) -> None:
        """Should handle high firing rates."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        high_rate = np.full((5, 3), 50, dtype=np.int64)  # 50 spikes per bin
        for surrogate in generate_poisson_surrogates(high_rate, n_surrogates=5, rng=42):
            assert surrogate.shape == high_rate.shape
            assert (surrogate >= 0).all()

    def test_yields_correct_shape(self, spike_counts: np.ndarray) -> None:
        """Each surrogate should have same shape as input."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, n_surrogates=5, rng=42
        ):
            assert surrogate.shape == spike_counts.shape

    def test_yields_correct_dtype(self, spike_counts: np.ndarray) -> None:
        """Surrogate arrays should be int64."""
        from neurospatial.stats.surrogates import generate_poisson_surrogates

        for surrogate in generate_poisson_surrogates(
            spike_counts, n_surrogates=5, rng=42
        ):
            assert surrogate.dtype == np.int64


class TestGenerateInhomogeneousPoissonSurrogates:
    """Tests for generate_inhomogeneous_poisson_surrogates (time-varying rate surrogates)."""

    @pytest.fixture
    def spike_counts(self) -> np.ndarray:
        """Sample spike counts with temporal structure: (20 time bins, 5 neurons)."""
        rng = np.random.default_rng(42)
        # Create spike counts with clear temporal modulation
        n_time_bins = 20
        n_neurons = 5
        # Base rates with temporal modulation (ramp up then down)
        time_modulation = np.abs(np.sin(np.linspace(0, np.pi, n_time_bins)))
        base_rates = rng.uniform(1, 5, n_neurons)
        expected_rates = np.outer(time_modulation, base_rates) * 3
        return rng.poisson(expected_rates).astype(np.int64)

    def test_yields_correct_number_of_surrogates(
        self, spike_counts: np.ndarray
    ) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        n_surrogates = 10
        surrogates = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, n_surrogates=n_surrogates, rng=42
            )
        )
        assert len(surrogates) == n_surrogates

    def test_yields_non_negative_counts(self, spike_counts: np.ndarray) -> None:
        """Surrogate spike counts should be non-negative."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, n_surrogates=10, rng=42
        ):
            assert (surrogate >= 0).all()

    def test_preserves_time_varying_rates_approximately(
        self, spike_counts: np.ndarray
    ) -> None:
        """Time-varying firing rates should be preserved (with smoothing).

        Unlike homogeneous Poisson, this preserves slow rate fluctuations.
        """
        # Compute smoothed rates from original (what surrogate should match)
        from scipy.ndimage import uniform_filter1d

        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        smoothed_original = uniform_filter1d(
            spike_counts.astype(np.float64), size=3, axis=0, mode="nearest"
        )

        # Collect rates from many surrogates
        surrogate_rates = []
        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, smoothing_window=3, n_surrogates=500, rng=42
        ):
            surrogate_rates.append(surrogate.mean(axis=1))  # Mean across neurons

        # Average across surrogates
        avg_surrogate_rate = np.mean(surrogate_rates, axis=0)
        avg_original_smoothed = smoothed_original.mean(axis=1)

        # Correlation should be high (preserves temporal modulation)
        corr = np.corrcoef(avg_surrogate_rate, avg_original_smoothed)[0, 1]
        assert corr > 0.7, f"Temporal modulation not preserved: correlation={corr:.3f}"

    def test_smoothing_window_parameter(self, spike_counts: np.ndarray) -> None:
        """Different smoothing windows should produce different rate estimates."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        # Smaller window = more temporal detail preserved
        # Larger window = more smoothed (closer to homogeneous)
        surrogates_small = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, smoothing_window=1, n_surrogates=5, rng=42
            )
        )
        surrogates_large = list(
            generate_inhomogeneous_poisson_surrogates(
                spike_counts, smoothing_window=7, n_surrogates=5, rng=42
            )
        )
        # With different smoothing, at least some surrogates should differ
        # (same seed but different parameters)
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(surrogates_small, surrogates_large, strict=True)
        )
        assert any_different

    def test_zero_spike_counts(self) -> None:
        """Should handle all-zero spike counts (silent neurons)."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        zeros = np.zeros((5, 3), dtype=np.int64)
        for surrogate in generate_inhomogeneous_poisson_surrogates(
            zeros, n_surrogates=5, rng=42
        ):
            assert surrogate.shape == zeros.shape
            assert (surrogate >= 0).all()

    def test_yields_correct_shape(self, spike_counts: np.ndarray) -> None:
        """Each surrogate should have same shape as input."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, n_surrogates=5, rng=42
        ):
            assert surrogate.shape == spike_counts.shape

    def test_yields_correct_dtype(self, spike_counts: np.ndarray) -> None:
        """Surrogate arrays should be int64."""
        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        for surrogate in generate_inhomogeneous_poisson_surrogates(
            spike_counts, n_surrogates=5, rng=42
        ):
            assert surrogate.dtype == np.int64


class TestGenerateJitteredSpikes:
    """Tests for generate_jittered_spikes."""

    @pytest.fixture
    def spike_times(self) -> np.ndarray:
        """Sample spike times (sorted)."""
        return np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7, 0.85, 1.0])

    def test_yields_correct_count(self, spike_times: np.ndarray) -> None:
        """Should yield exactly n_surrogates arrays."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        n_surrogates = 10
        surrogates = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=n_surrogates, rng=42
            )
        )
        assert len(surrogates) == n_surrogates

    def test_yields_same_length(self, spike_times: np.ndarray) -> None:
        """Each surrogate should have same number of spikes as input."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.01, n_surrogates=5, rng=42
        ):
            assert len(surrogate) == len(spike_times)

    def test_spikes_are_sorted(self, spike_times: np.ndarray) -> None:
        """Output spike times should be sorted."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.01, n_surrogates=5, rng=42
        ):
            assert np.all(np.diff(surrogate) >= 0)

    def test_reproducibility_with_seed(self, spike_times: np.ndarray) -> None:
        """Same seed should produce same surrogates."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        surrogates1 = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=5, rng=42
            )
        )
        surrogates2 = list(
            generate_jittered_spikes(
                spike_times, jitter_std=0.01, n_surrogates=5, rng=42
            )
        )
        for s1, s2 in zip(surrogates1, surrogates2, strict=True):
            assert_array_equal(s1, s2)

    def test_small_jitter_stays_close(self, spike_times: np.ndarray) -> None:
        """With small jitter, spikes should stay close to original times."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        jitter_std = 0.001  # 1 ms
        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=jitter_std, n_surrogates=5, rng=42
        ):
            # With 1 ms jitter, max deviation should be small (< 10 ms usually)
            max_deviation = np.max(np.abs(surrogate - spike_times))
            assert max_deviation < 0.05  # 50 ms max

    def test_empty_spikes(self) -> None:
        """Should handle empty spike array."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        empty_spikes = np.array([])
        surrogates = list(
            generate_jittered_spikes(
                empty_spikes, jitter_std=0.01, n_surrogates=3, rng=42
            )
        )
        assert len(surrogates) == 3
        for s in surrogates:
            assert len(s) == 0

    def test_single_spike(self) -> None:
        """Should handle single spike."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        single_spike = np.array([0.5])
        surrogates = list(
            generate_jittered_spikes(
                single_spike, jitter_std=0.01, n_surrogates=3, rng=42
            )
        )
        assert len(surrogates) == 3
        for s in surrogates:
            assert len(s) == 1

    def test_window_constraint(self, spike_times: np.ndarray) -> None:
        """Spikes should stay within window if specified."""
        from neurospatial.stats.surrogates import generate_jittered_spikes

        window = (0.0, 1.5)
        for surrogate in generate_jittered_spikes(
            spike_times, jitter_std=0.05, n_surrogates=5, rng=42, window=window
        ):
            assert np.all(surrogate >= window[0])
            assert np.all(surrogate <= window[1])


class TestInhomogeneousPoissonRatePreservation:
    """Inhomogeneous-Poisson surrogates must preserve the time-varying rate.

    Existing tests check count/shape/dtype/reproducibility; this checks the
    actual contract -- the average surrogate reproduces the (smoothed) input
    rate. Without this, a surrogate that flattened or distorted the rate would
    pass the existing tests.
    """

    def test_average_surrogate_matches_input_rate(self) -> None:
        from scipy.ndimage import gaussian_filter1d

        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        # Ramping rate lambda(t) = 1 + 9 t/T.
        n_time = 500
        t = np.arange(n_time)
        rate = 1.0 + 9.0 * (t / n_time)
        counts = np.random.default_rng(1).poisson(rate).astype(np.int64).reshape(-1, 1)

        accumulator = np.zeros(n_time)
        n_surrogates = 1000
        for surrogate in generate_inhomogeneous_poisson_surrogates(
            counts, smoothing_window=5, n_surrogates=n_surrogates, rng=7
        ):
            accumulator += surrogate[:, 0]
        mean_surrogate = accumulator / n_surrogates

        # Compare heavily-smoothed profiles (the surrogate preserves slow rate
        # structure, not bin-by-bin counts).
        smoothed_surrogate = gaussian_filter1d(mean_surrogate, sigma=5)
        smoothed_input = gaussian_filter1d(counts[:, 0].astype(float), sigma=5)
        np.testing.assert_allclose(
            smoothed_surrogate, smoothed_input, rtol=0.15, atol=0.5
        )

    def test_constant_rate_surrogate_is_flat(self) -> None:
        from scipy.ndimage import gaussian_filter1d

        from neurospatial.stats.surrogates import (
            generate_inhomogeneous_poisson_surrogates,
        )

        n_time = 500
        counts = (
            np.random.default_rng(2)
            .poisson(5.0, n_time)
            .astype(np.int64)
            .reshape(-1, 1)
        )

        accumulator = np.zeros(n_time)
        n_surrogates = 1000
        for surrogate in generate_inhomogeneous_poisson_surrogates(
            counts, smoothing_window=5, n_surrogates=n_surrogates, rng=11
        ):
            accumulator += surrogate[:, 0]
        mean_surrogate = accumulator / n_surrogates

        # A constant input rate -> the averaged surrogate is flat around 5.
        smoothed = gaussian_filter1d(mean_surrogate, sigma=10)
        assert np.allclose(smoothed, 5.0, atol=1.0)


# ---------------------------------------------------------------------------
# Distributions and input checks.
# ---------------------------------------------------------------------------


def test_jitter_without_window_has_requested_spread():
    """Jitter offsets are N(0, jitter_std): checks jitter_std is used."""
    from neurospatial.stats.surrogates import generate_jittered_spikes

    spikes = np.arange(0.0, 100.0, 10.0)  # spaced far beyond the jitter
    offsets = np.concatenate(
        [
            draw - spikes
            for draw in generate_jittered_spikes(
                spikes, jitter_std=0.2, n_surrogates=2000, rng=0
            )
        ]
    )
    assert offsets.mean() == pytest.approx(0.0, abs=0.01)
    assert offsets.std() == pytest.approx(0.2, rel=0.02)


def test_jitter_inside_window_is_truncated_normal_without_edge_pileup():
    """Near an edge the jitter follows the truncated Gaussian; none lands on it."""
    from scipy import stats

    from neurospatial.stats.surrogates import generate_jittered_spikes

    spike, sd, window = 0.02, 0.05, (0.0, 1.0)
    draws = np.array(
        [
            draw[0]
            for draw in generate_jittered_spikes(
                np.array([spike]),
                jitter_std=sd,
                n_surrogates=20_000,
                rng=1,
                window=window,
            )
        ]
    )
    assert np.all((draws >= 0.0) & (draws < 1.0))
    assert np.mean(draws == 0.0) == 0.0
    truncated = stats.truncnorm(
        (window[0] - spike) / sd, (window[1] - spike) / sd, loc=spike, scale=sd
    )
    assert stats.kstest(draws, truncated.cdf).pvalue > 0.01


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"jitter_std": np.nan}, "jitter_std"),
        ({"jitter_std": -0.1}, "jitter_std"),
        ({"jitter_std": 0.1, "window": (1.0, 0.0)}, "window"),
        ({"jitter_std": 0.1, "window": (0.0, 0.3)}, "window"),  # spike outside
        ({"jitter_std": 0.1, "n_surrogates": 0}, "n_surrogates"),
    ],
)
def test_jitter_refuses_invalid_settings(kwargs, match):
    from neurospatial.stats.surrogates import generate_jittered_spikes

    with pytest.raises(ValueError, match=match):
        generate_jittered_spikes(np.array([0.1, 0.5]), **kwargs)


def test_jitter_refuses_non_finite_spikes():
    from neurospatial.stats.surrogates import generate_jittered_spikes

    with pytest.raises(ValueError, match="finite 1-D"):
        generate_jittered_spikes(np.array([0.1, np.nan]), jitter_std=0.1)


def test_poisson_surrogates_are_poisson():
    """Counts per neuron have variance equal to their mean (Fano factor 1)."""
    from neurospatial.stats.surrogates import generate_poisson_surrogates

    counts = np.array([[0, 5], [2, 9], [1, 7], [3, 3]])
    draws = np.stack(
        list(generate_poisson_surrogates(counts, n_surrogates=4000, rng=0))
    )
    per_neuron = draws.reshape(-1, 2)
    assert_allclose(per_neuron.mean(axis=0), counts.mean(axis=0), rtol=0.03)
    assert_allclose(per_neuron.var(axis=0) / per_neuron.mean(axis=0), 1.0, atol=0.06)


@pytest.mark.parametrize(
    "counts",
    [np.array([[0.5, 10.2], [1.5, 30.0]]), np.array([[np.nan, 1.0]]), np.ones(3)],
)
def test_poisson_surrogates_refuse_non_count_input(counts):
    from neurospatial.stats.surrogates import (
        generate_inhomogeneous_poisson_surrogates,
        generate_poisson_surrogates,
    )

    for make in (
        generate_poisson_surrogates,
        generate_inhomogeneous_poisson_surrogates,
    ):
        with pytest.raises(ValueError, match="whole-number counts"):
            make(counts)


def test_inhomogeneous_poisson_refuses_zero_smoothing_window():
    from neurospatial.stats.surrogates import generate_inhomogeneous_poisson_surrogates

    with pytest.raises(ValueError, match="smoothing_window"):
        generate_inhomogeneous_poisson_surrogates(np.ones((3, 2)), smoothing_window=0)
