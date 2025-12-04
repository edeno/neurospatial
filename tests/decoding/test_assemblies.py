"""Tests for neurospatial.decoding.assemblies module.

Tests cover cell assembly detection, activation computation, and reactivation analysis.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.decoding.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    ExplainedVarianceResult,
    assembly_activation,
    detect_assemblies,
    explained_variance_reactivation,
    marchenko_pastur_threshold,
    pairwise_correlations,
    reactivation_strength,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def spike_counts_with_assemblies(rng: np.random.Generator) -> np.ndarray:
    """Spike counts with 3 embedded assemblies.

    Shape: (50 neurons, 2000 time bins)
    Assemblies:
    - Neurons 0-4: co-activate together
    - Neurons 10-14: co-activate together
    - Neurons 20-24: co-activate together
    """
    n_neurons, n_time = 50, 2000

    # Baseline Poisson activity
    spike_counts = rng.poisson(5, (n_neurons, n_time)).astype(np.float64)

    # Add correlated activity for each assembly
    assembly_neurons = [
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
    ]

    for neurons in assembly_neurons:
        # Shared modulation for assembly members
        shared = rng.poisson(5, n_time)
        for n in neurons:
            spike_counts[n] += shared

    return spike_counts


@pytest.fixture
def random_spike_counts(rng: np.random.Generator) -> np.ndarray:
    """Random spike counts with no assembly structure.

    Shape: (30 neurons, 3000 time bins)
    """
    return rng.poisson(5, (30, 3000)).astype(np.float64)


@pytest.fixture
def small_spike_counts(rng: np.random.Generator) -> np.ndarray:
    """Small spike counts for basic tests.

    Shape: (10 neurons, 100 time bins)
    """
    return rng.poisson(3, (10, 100)).astype(np.float64)


# =============================================================================
# Tests for marchenko_pastur_threshold
# =============================================================================


class TestMarchenkoPasturThreshold:
    """Tests for Marchenko-Pastur threshold computation."""

    def test_basic_computation(self) -> None:
        """Should compute threshold correctly for typical dimensions."""
        # For n=100, p=1000: q=0.1, threshold = (1 + sqrt(0.1))^2 ≈ 1.632
        threshold = marchenko_pastur_threshold(100, 1000)
        expected = (1 + np.sqrt(0.1)) ** 2
        assert_allclose(threshold, expected, rtol=1e-10)

    def test_square_matrix(self) -> None:
        """For square matrix, threshold should be 4."""
        threshold = marchenko_pastur_threshold(100, 100)
        assert_allclose(threshold, 4.0, rtol=1e-10)

    def test_warns_when_neurons_exceed_time_bins(self) -> None:
        """Should warn when n_neurons > n_time_bins."""
        with pytest.warns(UserWarning, match="n_neurons.*n_time_bins"):
            marchenko_pastur_threshold(100, 50)

    def test_invalid_n_neurons(self) -> None:
        """Should raise ValueError for non-positive n_neurons."""
        with pytest.raises(ValueError, match="n_neurons must be positive"):
            marchenko_pastur_threshold(0, 100)
        with pytest.raises(ValueError, match="n_neurons must be positive"):
            marchenko_pastur_threshold(-5, 100)

    def test_invalid_n_time_bins(self) -> None:
        """Should raise ValueError for non-positive n_time_bins."""
        with pytest.raises(ValueError, match="n_time_bins must be positive"):
            marchenko_pastur_threshold(100, 0)
        with pytest.raises(ValueError, match="n_time_bins must be positive"):
            marchenko_pastur_threshold(100, -10)

    def test_returns_float(self) -> None:
        """Should return a float, not numpy scalar."""
        threshold = marchenko_pastur_threshold(50, 500)
        assert isinstance(threshold, float)


# =============================================================================
# Tests for detect_assemblies
# =============================================================================


class TestDetectAssemblies:
    """Tests for detect_assemblies function."""

    def test_detects_known_assemblies(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Should detect embedded assemblies."""
        result = detect_assemblies(
            spike_counts_with_assemblies, method="ica", random_state=42
        )

        # Should detect 2-5 assemblies (depends on threshold)
        assert result.n_significant >= 2
        assert result.n_significant <= 5

    def test_random_data_few_assemblies(self, random_spike_counts: np.ndarray) -> None:
        """Random data should yield few significant assemblies."""
        result = detect_assemblies(random_spike_counts, random_state=42)

        # Random data should have 0-2 significant assemblies
        assert result.n_significant <= 2

    def test_pca_method(self, small_spike_counts: np.ndarray) -> None:
        """PCA method should work and return valid results."""
        result = detect_assemblies(small_spike_counts, method="pca")

        assert result.method == "pca"
        assert len(result.patterns) >= 1
        assert result.activations.shape[0] == len(result.patterns)

    def test_ica_method(self, small_spike_counts: np.ndarray) -> None:
        """ICA method should work and return valid results."""
        result = detect_assemblies(small_spike_counts, method="ica", random_state=42)

        assert result.method == "ica"
        assert len(result.patterns) >= 1

    def test_nmf_method(self, small_spike_counts: np.ndarray) -> None:
        """NMF method should work and return valid results."""
        result = detect_assemblies(small_spike_counts, method="nmf", random_state=42)

        assert result.method == "nmf"
        assert len(result.patterns) >= 1

    def test_returns_correct_result_type(self, small_spike_counts: np.ndarray) -> None:
        """Should return AssemblyDetectionResult."""
        result = detect_assemblies(small_spike_counts, random_state=42)

        assert isinstance(result, AssemblyDetectionResult)
        assert isinstance(result.patterns, list)
        assert all(isinstance(p, AssemblyPattern) for p in result.patterns)

    def test_patterns_have_correct_shape(self, small_spike_counts: np.ndarray) -> None:
        """Pattern weights should have n_neurons elements."""
        n_neurons = small_spike_counts.shape[0]
        result = detect_assemblies(small_spike_counts, random_state=42)

        for pattern in result.patterns:
            assert len(pattern.weights) == n_neurons

    def test_activations_have_correct_shape(
        self, small_spike_counts: np.ndarray
    ) -> None:
        """Activations should have shape (n_assemblies, n_time_bins)."""
        n_time = small_spike_counts.shape[1]
        result = detect_assemblies(small_spike_counts, random_state=42)

        assert result.activations.shape == (len(result.patterns), n_time)

    def test_eigenvalues_descending(self, small_spike_counts: np.ndarray) -> None:
        """Eigenvalues should be in descending order."""
        result = detect_assemblies(small_spike_counts, random_state=42)

        # Check monotonically decreasing
        diffs = np.diff(result.eigenvalues)
        assert np.all(diffs <= 0)

    def test_n_components_fixed(self, small_spike_counts: np.ndarray) -> None:
        """Should respect fixed n_components parameter."""
        n_comp = 3
        result = detect_assemblies(
            small_spike_counts, n_components=n_comp, random_state=42
        )

        assert len(result.patterns) == n_comp
        assert result.activations.shape[0] == n_comp

    def test_n_components_auto(self, small_spike_counts: np.ndarray) -> None:
        """Auto n_components should use Marchenko-Pastur threshold."""
        result = detect_assemblies(
            small_spike_counts, n_components="auto", random_state=42
        )

        # n_significant should be based on threshold
        expected_n = max(1, result.n_significant)
        assert len(result.patterns) == expected_n

    def test_z_threshold_affects_membership(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Higher z_threshold should yield fewer assembly members."""
        result_low = detect_assemblies(
            spike_counts_with_assemblies,
            z_threshold=1.5,
            random_state=42,
        )
        result_high = detect_assemblies(
            spike_counts_with_assemblies,
            z_threshold=2.5,
            random_state=42,
        )

        # Lower threshold = more members on average
        avg_low = np.mean([len(p.member_indices) for p in result_low.patterns])
        avg_high = np.mean([len(p.member_indices) for p in result_high.patterns])

        assert avg_low >= avg_high

    def test_reproducibility_with_random_state(
        self, small_spike_counts: np.ndarray
    ) -> None:
        """Same random_state should give same results."""
        result1 = detect_assemblies(small_spike_counts, random_state=42)
        result2 = detect_assemblies(small_spike_counts, random_state=42)

        assert_allclose(result1.activations, result2.activations)
        for p1, p2 in zip(result1.patterns, result2.patterns, strict=True):
            assert_allclose(p1.weights, p2.weights)

    def test_invalid_2d_input(self) -> None:
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="must be 2D"):
            detect_assemblies(np.array([1, 2, 3]))

    def test_too_few_neurons(self, rng: np.random.Generator) -> None:
        """Should raise ValueError for fewer than 3 neurons."""
        spike_counts = rng.poisson(5, (2, 100)).astype(np.float64)
        with pytest.raises(ValueError, match="at least 3 neurons"):
            detect_assemblies(spike_counts)

    def test_invalid_method(self, small_spike_counts: np.ndarray) -> None:
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            detect_assemblies(small_spike_counts, method="unknown")  # type: ignore[arg-type]

    def test_invalid_n_components(self, small_spike_counts: np.ndarray) -> None:
        """Should raise ValueError for invalid n_components."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            detect_assemblies(small_spike_counts, n_components=0)

        with pytest.raises(ValueError, match="cannot exceed n_neurons"):
            detect_assemblies(small_spike_counts, n_components=100)

    def test_warns_short_recording(self, rng: np.random.Generator) -> None:
        """Should warn when n_time_bins < n_neurons."""
        spike_counts = rng.poisson(5, (20, 10)).astype(np.float64)
        with pytest.warns(UserWarning, match="n_time_bins.*n_neurons"):
            detect_assemblies(spike_counts, random_state=42)

    def test_handles_zero_variance_neurons(self, rng: np.random.Generator) -> None:
        """Should handle neurons with constant firing."""
        spike_counts = rng.poisson(5, (10, 100)).astype(np.float64)
        # Make one neuron constant
        spike_counts[0, :] = 5.0

        with pytest.warns(UserWarning, match="zero variance"):
            result = detect_assemblies(spike_counts, random_state=42)

        # Should still return valid results
        assert len(result.patterns) >= 1


# =============================================================================
# Tests for assembly_activation
# =============================================================================


class TestAssemblyActivation:
    """Tests for assembly_activation function."""

    def test_returns_correct_shape(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Activation should have n_time_bins elements."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        activation = assembly_activation(spike_counts_with_assemblies, pattern)

        assert activation.shape == (spike_counts_with_assemblies.shape[1],)

    def test_returns_zscored_activation(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Activation should be z-scored (mean ≈ 0, std ≈ 1)."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        activation = assembly_activation(spike_counts_with_assemblies, pattern)

        # Z-scored values should have mean ≈ 0, std ≈ 1
        assert_allclose(np.mean(activation), 0.0, atol=0.1)
        assert_allclose(np.std(activation), 1.0, atol=0.1)

    def test_mismatched_neurons_raises_error(
        self, spike_counts_with_assemblies: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Should raise ValueError if pattern has wrong number of neurons."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        # Create spike counts with different number of neurons
        wrong_counts = rng.poisson(5, (20, 100)).astype(np.float64)

        with pytest.raises(ValueError, match=r"neurons.*weights.*Must match"):
            assembly_activation(wrong_counts, pattern)

    def test_invalid_2d_input(self, spike_counts_with_assemblies: np.ndarray) -> None:
        """Should raise ValueError for non-2D input."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        with pytest.raises(ValueError, match="must be 2D"):
            assembly_activation(np.array([1, 2, 3]), pattern)

    def test_z_score_input_option(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """z_score_input=False should skip input normalization."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        act_zscore = assembly_activation(
            spike_counts_with_assemblies, pattern, z_score_input=True
        )
        act_raw = assembly_activation(
            spike_counts_with_assemblies, pattern, z_score_input=False
        )

        # Results should differ
        assert not np.allclose(act_zscore, act_raw)


# =============================================================================
# Tests for pairwise_correlations
# =============================================================================


class TestPairwiseCorrelations:
    """Tests for pairwise_correlations function."""

    def test_returns_correct_length(self, small_spike_counts: np.ndarray) -> None:
        """Should return n*(n-1)/2 correlation values."""
        n_neurons = small_spike_counts.shape[0]
        expected_n_pairs = n_neurons * (n_neurons - 1) // 2

        corr = pairwise_correlations(small_spike_counts)

        assert len(corr) == expected_n_pairs

    def test_correlation_bounds(self, small_spike_counts: np.ndarray) -> None:
        """Correlations should be in [-1, 1]."""
        corr = pairwise_correlations(small_spike_counts)

        assert np.all(corr >= -1)
        assert np.all(corr <= 1)

    def test_self_correlation_excluded(self, small_spike_counts: np.ndarray) -> None:
        """Diagonal (self-correlations) should not be included."""
        n_neurons = small_spike_counts.shape[0]
        corr = pairwise_correlations(small_spike_counts)

        # Length should be n*(n-1)/2, not n*n
        assert len(corr) == n_neurons * (n_neurons - 1) // 2

    def test_invalid_2d_input(self) -> None:
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="must be 2D"):
            pairwise_correlations(np.array([1, 2, 3]))

    def test_z_score_input_option(self, small_spike_counts: np.ndarray) -> None:
        """z_score_input option should affect results."""
        corr_zscore = pairwise_correlations(small_spike_counts, z_score_input=True)
        corr_raw = pairwise_correlations(small_spike_counts, z_score_input=False)

        # Results should be very similar since corrcoef also normalizes
        # but process may differ slightly
        assert_allclose(corr_zscore, corr_raw, rtol=0.1)


# =============================================================================
# Tests for reactivation_strength
# =============================================================================


class TestReactivationStrength:
    """Tests for reactivation_strength function."""

    def test_returns_float(
        self, spike_counts_with_assemblies: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Should return a float value."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        # Split data for template and match
        n_time = spike_counts_with_assemblies.shape[1]
        template = spike_counts_with_assemblies[:, : n_time // 2]
        match = spike_counts_with_assemblies[:, n_time // 2 :]

        strength = reactivation_strength(template, match, pattern)

        assert isinstance(strength, float)

    def test_strength_non_negative(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Reactivation strength should be non-negative (ratio of magnitudes)."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        n_time = spike_counts_with_assemblies.shape[1]
        template = spike_counts_with_assemblies[:, : n_time // 2]
        match = spike_counts_with_assemblies[:, n_time // 2 :]

        strength = reactivation_strength(template, match, pattern)

        # Ratio of activation magnitudes should be non-negative
        assert strength >= 0

    def test_strength_similar_data(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Same data should give strength ≈ 1."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        # Use same data for template and match
        strength = reactivation_strength(
            spike_counts_with_assemblies, spike_counts_with_assemblies, pattern
        )

        # Same data should have ratio ≈ 1
        assert_allclose(strength, 1.0, rtol=0.1)


# =============================================================================
# Tests for explained_variance_reactivation
# =============================================================================


class TestExplainedVarianceReactivation:
    """Tests for explained_variance_reactivation function."""

    def test_perfect_reactivation(self) -> None:
        """Identical correlations should give EV = 1."""
        corr = np.array([0.1, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4])

        result = explained_variance_reactivation(corr, corr)

        assert_allclose(result.explained_variance, 1.0, rtol=1e-10)
        assert result.n_pairs == len(corr)

    def test_random_no_reactivation(self, rng: np.random.Generator) -> None:
        """Random correlations should give low EV."""
        corr1 = rng.uniform(-1, 1, 100)
        corr2 = rng.uniform(-1, 1, 100)

        result = explained_variance_reactivation(corr1, corr2)

        # EV should be low for random data
        assert result.explained_variance < 0.1

    def test_returns_correct_type(self) -> None:
        """Should return ExplainedVarianceResult."""
        corr = np.array([0.1, 0.5, -0.3, 0.8, 0.2])

        result = explained_variance_reactivation(corr, corr)

        assert isinstance(result, ExplainedVarianceResult)

    def test_ev_bounds(self, rng: np.random.Generator) -> None:
        """EV should be in [0, 1]."""
        corr1 = rng.uniform(-1, 1, 50)
        corr2 = rng.uniform(-1, 1, 50)

        result = explained_variance_reactivation(corr1, corr2)

        assert 0 <= result.explained_variance <= 1
        assert 0 <= result.reversed_ev <= 1

    def test_mismatched_lengths_raises_error(self) -> None:
        """Should raise ValueError for mismatched correlation lengths."""
        corr1 = np.array([0.1, 0.5, -0.3])
        corr2 = np.array([0.1, 0.5])

        with pytest.raises(ValueError, match="same length"):
            explained_variance_reactivation(corr1, corr2)

    def test_too_few_pairs_raises_error(self) -> None:
        """Should raise ValueError for fewer than 3 pairs."""
        corr = np.array([0.1, 0.5])

        with pytest.raises(ValueError, match="at least 3 pairs"):
            explained_variance_reactivation(corr, corr)

    def test_partial_correlation_with_control(self, rng: np.random.Generator) -> None:
        """Control correlations should affect partial correlation."""
        n_pairs = 50
        corr_template = rng.uniform(-1, 1, n_pairs)
        corr_match = 0.5 * corr_template + 0.5 * rng.uniform(-1, 1, n_pairs)
        corr_control = rng.uniform(-1, 1, n_pairs)

        result_with_control = explained_variance_reactivation(
            corr_template, corr_match, control_correlations=corr_control
        )
        result_no_control = explained_variance_reactivation(corr_template, corr_match)

        # Partial correlation should differ from raw correlation
        # (unless control is uncorrelated with both)
        assert (
            result_with_control.partial_correlation
            != result_no_control.partial_correlation
        )

    def test_control_mismatched_length_raises_error(self) -> None:
        """Should raise ValueError if control has wrong length."""
        corr = np.array([0.1, 0.5, -0.3, 0.8, 0.2])
        control = np.array([0.1, 0.2])

        with pytest.raises(
            ValueError, match="control correlations must have same length"
        ):
            explained_variance_reactivation(corr, corr, control_correlations=control)

    def test_handles_nan_values(self) -> None:
        """Should handle NaN values by excluding them."""
        corr1 = np.array([0.1, np.nan, 0.5, -0.3, 0.8, 0.2])
        corr2 = np.array([0.1, 0.2, np.nan, -0.3, 0.8, 0.2])

        result = explained_variance_reactivation(corr1, corr2)

        # Should exclude NaN pairs (indices 1 and 2)
        assert result.n_pairs == 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestAssemblyWorkflow:
    """Integration tests for complete assembly detection workflow."""

    def test_full_workflow(self, spike_counts_with_assemblies: np.ndarray) -> None:
        """Test complete detection -> activation -> reactivation workflow."""
        # Split data into "behavior" and "rest" periods
        n_time = spike_counts_with_assemblies.shape[1]
        counts_behavior = spike_counts_with_assemblies[:, : n_time // 2]
        counts_rest = spike_counts_with_assemblies[:, n_time // 2 :]

        # Detect assemblies during behavior
        result = detect_assemblies(counts_behavior, random_state=42)
        assert result.n_significant >= 1

        # Compute activation during rest
        pattern = result.patterns[0]
        activation_rest = assembly_activation(counts_rest, pattern)
        assert len(activation_rest) == n_time // 2

        # Compute pairwise correlations
        corr_behavior = pairwise_correlations(counts_behavior)
        corr_rest = pairwise_correlations(counts_rest)

        # Explained variance reactivation
        ev_result = explained_variance_reactivation(corr_behavior, corr_rest)
        assert isinstance(ev_result, ExplainedVarianceResult)

    def test_pattern_immutability(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """AssemblyPattern should be immutable (frozen dataclass)."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            pattern.weights = np.zeros(len(pattern.weights))  # type: ignore[misc]

    def test_result_immutability(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """AssemblyDetectionResult should be immutable (frozen dataclass)."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)

        with pytest.raises(AttributeError):
            result.n_significant = 100  # type: ignore[misc]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_neurons(self, rng: np.random.Generator) -> None:
        """Should work with minimum 3 neurons."""
        spike_counts = rng.poisson(5, (3, 500)).astype(np.float64)

        result = detect_assemblies(spike_counts, random_state=42)

        assert len(result.patterns) >= 1

    def test_single_time_bin_activation(
        self, spike_counts_with_assemblies: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Activation should work even with few time bins."""
        result = detect_assemblies(spike_counts_with_assemblies, random_state=42)
        pattern = result.patterns[0]

        # Create single time bin data
        single_bin = rng.poisson(5, (50, 10)).astype(np.float64)

        activation = assembly_activation(single_bin, pattern)
        assert len(activation) == 10

    def test_all_zero_spike_counts(self) -> None:
        """Should handle all-zero spike counts without NaN."""
        spike_counts = np.zeros((10, 100), dtype=np.float64)

        with pytest.warns(UserWarning, match="zero variance"):
            result = detect_assemblies(spike_counts, random_state=42)

        # Should not have NaN in results
        assert not np.any(np.isnan(result.activations))
        for pattern in result.patterns:
            assert not np.any(np.isnan(pattern.weights))

    def test_explained_variance_identical_correlations(self) -> None:
        """EV should be 1 for identical correlation vectors."""
        corr = np.linspace(-1, 1, 100)

        result = explained_variance_reactivation(corr, corr)

        assert_allclose(result.explained_variance, 1.0)
        assert_allclose(result.reversed_ev, 1.0)
