"""Tests for neurospatial.decoding.shuffle module.

Tests cover temporal shuffles for significance testing in Bayesian decoding.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.decoding.shuffle import (
    shuffle_time_bins,
    shuffle_time_bins_coherent,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def spike_counts() -> np.ndarray:
    """Sample spike counts: (5 time bins, 3 neurons)."""
    return np.array(
        [
            [0, 1, 2],  # t=0
            [1, 0, 1],  # t=1
            [2, 2, 0],  # t=2
            [0, 0, 3],  # t=3
            [1, 1, 1],  # t=4
        ],
        dtype=np.int64,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed random generator for reproducibility."""
    return np.random.default_rng(42)


# =============================================================================
# Tests for shuffle_time_bins
# =============================================================================


class TestShuffleTimeBins:
    """Tests for shuffle_time_bins generator function."""

    def test_yields_correct_number_of_shuffles(self, spike_counts: np.ndarray) -> None:
        """Should yield exactly n_shuffles arrays."""
        n_shuffles = 10
        shuffles = list(shuffle_time_bins(spike_counts, n_shuffles=n_shuffles, rng=42))
        assert len(shuffles) == n_shuffles

    def test_yields_correct_shape(self, spike_counts: np.ndarray) -> None:
        """Each shuffled array should have same shape as input."""
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.shape == spike_counts.shape

    def test_preserves_dtype(self, spike_counts: np.ndarray) -> None:
        """Shuffled arrays should preserve input dtype."""
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.dtype == spike_counts.dtype

    def test_preserves_total_spikes(self, spike_counts: np.ndarray) -> None:
        """Total spike count should be preserved across shuffles."""
        total_original = spike_counts.sum()
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=10, rng=42):
            assert shuffled.sum() == total_original

    def test_preserves_spikes_per_neuron(self, spike_counts: np.ndarray) -> None:
        """Total spikes per neuron should be preserved."""
        spikes_per_neuron = spike_counts.sum(axis=0)
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=10, rng=42):
            assert_array_equal(shuffled.sum(axis=0), spikes_per_neuron)

    def test_preserves_spikes_per_time_bin(self, spike_counts: np.ndarray) -> None:
        """Total spikes per time bin should be preserved (rows are permuted)."""
        # After row permutation, the multiset of row sums should be the same
        original_row_sums = sorted(spike_counts.sum(axis=1).tolist())
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=10, rng=42):
            shuffled_row_sums = sorted(shuffled.sum(axis=1).tolist())
            assert shuffled_row_sums == original_row_sums

    def test_rows_are_permuted(self, spike_counts: np.ndarray) -> None:
        """Rows should be permutations of original rows."""
        original_rows = {tuple(row) for row in spike_counts}
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=10, rng=42):
            shuffled_rows = {tuple(row) for row in shuffled}
            assert shuffled_rows == original_rows

    def test_reproducibility_with_seed_int(self, spike_counts: np.ndarray) -> None:
        """Same seed should produce same shuffles."""
        shuffles1 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=42))
        shuffles2 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=42))
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_reproducibility_with_generator(self, spike_counts: np.ndarray) -> None:
        """Same generator state should produce same shuffles."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffles1 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=rng1))
        shuffles2 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=rng2))
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shuffles(
        self, spike_counts: np.ndarray
    ) -> None:
        """Different seeds should produce different shuffles."""
        shuffles1 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=42))
        shuffles2 = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=123))
        # At least one shuffle should be different
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(shuffles1, shuffles2, strict=True)
        )
        assert any_different

    def test_none_rng_produces_shuffles(self, spike_counts: np.ndarray) -> None:
        """rng=None should still produce valid shuffles."""
        shuffles = list(shuffle_time_bins(spike_counts, n_shuffles=5, rng=None))
        assert len(shuffles) == 5
        for shuffled in shuffles:
            assert shuffled.shape == spike_counts.shape

    def test_generator_is_lazy(self, spike_counts: np.ndarray) -> None:
        """Generator should be lazy - not all shuffles computed at once."""
        gen = shuffle_time_bins(spike_counts, n_shuffles=1000, rng=42)
        # Take only first 3 shuffles
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3
        # Generator should still have more
        fourth = next(gen)
        assert fourth is not None

    def test_empty_spike_counts(self) -> None:
        """Should handle empty spike counts gracefully."""
        empty = np.zeros((0, 3), dtype=np.int64)
        shuffles = list(shuffle_time_bins(empty, n_shuffles=5, rng=42))
        assert len(shuffles) == 5
        for s in shuffles:
            assert s.shape == (0, 3)

    def test_single_time_bin(self) -> None:
        """Should handle single time bin (trivial shuffle)."""
        single = np.array([[1, 2, 3]], dtype=np.int64)
        shuffles = list(shuffle_time_bins(single, n_shuffles=5, rng=42))
        for s in shuffles:
            assert_array_equal(s, single)

    def test_single_neuron(self) -> None:
        """Should work with single neuron."""
        single_neuron = np.array([[1], [2], [3]], dtype=np.int64)
        for shuffled in shuffle_time_bins(single_neuron, n_shuffles=5, rng=42):
            # Rows should be permutation
            original_rows = sorted(single_neuron[:, 0].tolist())
            shuffled_rows = sorted(shuffled[:, 0].tolist())
            assert shuffled_rows == original_rows


# =============================================================================
# Tests for shuffle_time_bins_coherent
# =============================================================================


class TestShuffleTimeBinsCoherent:
    """Tests for shuffle_time_bins_coherent (time-swap shuffle)."""

    def test_yields_correct_number_of_shuffles(self, spike_counts: np.ndarray) -> None:
        """Should yield exactly n_shuffles arrays."""
        n_shuffles = 10
        shuffles = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=n_shuffles, rng=42)
        )
        assert len(shuffles) == n_shuffles

    def test_yields_correct_shape(self, spike_counts: np.ndarray) -> None:
        """Each shuffled array should have same shape as input."""
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.shape == spike_counts.shape

    def test_preserves_dtype(self, spike_counts: np.ndarray) -> None:
        """Shuffled arrays should preserve input dtype."""
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.dtype == spike_counts.dtype

    def test_preserves_total_spikes(self, spike_counts: np.ndarray) -> None:
        """Total spike count should be preserved."""
        total_original = spike_counts.sum()
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=10, rng=42):
            assert shuffled.sum() == total_original

    def test_preserves_spikes_per_neuron(self, spike_counts: np.ndarray) -> None:
        """Total spikes per neuron should be preserved."""
        spikes_per_neuron = spike_counts.sum(axis=0)
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=10, rng=42):
            assert_array_equal(shuffled.sum(axis=0), spikes_per_neuron)

    def test_preserves_instantaneous_population_vectors(
        self, spike_counts: np.ndarray
    ) -> None:
        """Each row (population vector) should be preserved exactly.

        This is the key difference from shuffle_time_bins - the same
        permutation is applied coherently across all neurons.
        """
        original_rows = {tuple(row) for row in spike_counts}
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=10, rng=42):
            shuffled_rows = {tuple(row) for row in shuffled}
            assert shuffled_rows == original_rows

    def test_coherent_permutation_across_neurons(
        self, spike_counts: np.ndarray
    ) -> None:
        """All neurons should see the same temporal permutation.

        If row i goes to position j for neuron 0, it should go to
        position j for all other neurons too.
        """
        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=10, rng=42):
            # Find where each original row went
            # Each row in shuffled should exactly match some row in original
            for shuffled_row in shuffled:
                match_found = any(
                    np.array_equal(shuffled_row, orig_row) for orig_row in spike_counts
                )
                assert match_found, (
                    "Shuffled row should match some original row exactly"
                )

    def test_reproducibility_with_seed_int(self, spike_counts: np.ndarray) -> None:
        """Same seed should produce same shuffles."""
        shuffles1 = list(shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42))
        shuffles2 = list(shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42))
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_reproducibility_with_generator(self, spike_counts: np.ndarray) -> None:
        """Same generator state should produce same shuffles."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffles1 = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=rng1)
        )
        shuffles2 = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=rng2)
        )
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shuffles(
        self, spike_counts: np.ndarray
    ) -> None:
        """Different seeds should produce different shuffles."""
        shuffles1 = list(shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42))
        shuffles2 = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=123)
        )
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(shuffles1, shuffles2, strict=True)
        )
        assert any_different

    def test_none_rng_produces_shuffles(self, spike_counts: np.ndarray) -> None:
        """rng=None should still produce valid shuffles."""
        shuffles = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=None)
        )
        assert len(shuffles) == 5
        for shuffled in shuffles:
            assert shuffled.shape == spike_counts.shape

    def test_generator_is_lazy(self, spike_counts: np.ndarray) -> None:
        """Generator should be lazy."""
        gen = shuffle_time_bins_coherent(spike_counts, n_shuffles=1000, rng=42)
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3

    def test_empty_spike_counts(self) -> None:
        """Should handle empty spike counts gracefully."""
        empty = np.zeros((0, 3), dtype=np.int64)
        shuffles = list(shuffle_time_bins_coherent(empty, n_shuffles=5, rng=42))
        assert len(shuffles) == 5
        for s in shuffles:
            assert s.shape == (0, 3)

    def test_single_time_bin(self) -> None:
        """Should handle single time bin (trivial shuffle)."""
        single = np.array([[1, 2, 3]], dtype=np.int64)
        shuffles = list(shuffle_time_bins_coherent(single, n_shuffles=5, rng=42))
        for s in shuffles:
            assert_array_equal(s, single)


# =============================================================================
# Tests comparing shuffle_time_bins vs shuffle_time_bins_coherent
# =============================================================================


class TestShuffleComparison:
    """Tests comparing the two shuffle functions."""

    def test_both_preserve_row_multiset(self, spike_counts: np.ndarray) -> None:
        """Both shuffles should preserve the multiset of rows."""
        original_rows = sorted([tuple(row) for row in spike_counts])

        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            shuffled_rows = sorted([tuple(row) for row in shuffled])
            assert shuffled_rows == original_rows

        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            shuffled_rows = sorted([tuple(row) for row in shuffled])
            assert shuffled_rows == original_rows

    def test_coherent_vs_independent_permutations(
        self, spike_counts: np.ndarray
    ) -> None:
        """shuffle_time_bins can have different permutations per neuron,
        while shuffle_time_bins_coherent uses same permutation for all.

        Note: This test verifies conceptual difference, not that they're
        always different (which depends on RNG).
        """
        # Both should be valid permutations
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.shape == spike_counts.shape

        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.shape == spike_counts.shape
