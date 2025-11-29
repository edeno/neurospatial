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


# =============================================================================
# Tests for shuffle_cell_identity
# =============================================================================


class TestShuffleCellIdentity:
    """Tests for shuffle_cell_identity (neuron-place field mapping shuffle)."""

    @pytest.fixture
    def encoding_models(self) -> np.ndarray:
        """Sample encoding models: (3 neurons, 4 bins)."""
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Neuron 0 place field
                [4.0, 3.0, 2.0, 1.0],  # Neuron 1 place field
                [2.0, 4.0, 1.0, 3.0],  # Neuron 2 place field
            ],
            dtype=np.float64,
        )

    def test_yields_correct_number_of_shuffles(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Should yield exactly n_shuffles tuples."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        n_shuffles = 10
        shuffles = list(
            shuffle_cell_identity(
                spike_counts, encoding_models, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_yields_tuples(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Should yield (shuffled_counts, encoding_models) tuples."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        for shuffled_counts, models in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert isinstance(shuffled_counts, np.ndarray)
            assert isinstance(models, np.ndarray)

    def test_encoding_models_unchanged(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Encoding models should be returned unchanged (same object)."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        for _, models in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert models is encoding_models

    def test_shuffled_counts_shape(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Shuffled counts should have same shape as input."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled_counts.shape == spike_counts.shape

    def test_shuffled_counts_dtype(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Shuffled counts should preserve dtype."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled_counts.dtype == spike_counts.dtype

    def test_preserves_total_spikes(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Total spike count should be preserved."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        total_original = spike_counts.sum()
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            assert shuffled_counts.sum() == total_original

    def test_preserves_spikes_per_time_bin(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Total spikes per time bin should be preserved (columns permuted)."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        spikes_per_time_bin = spike_counts.sum(axis=1)
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            assert_array_equal(shuffled_counts.sum(axis=1), spikes_per_time_bin)

    def test_columns_are_permuted(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Columns (neurons) should be permutations of original columns."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        original_cols = {
            tuple(spike_counts[:, i]) for i in range(spike_counts.shape[1])
        }
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            shuffled_cols = {
                tuple(shuffled_counts[:, i]) for i in range(shuffled_counts.shape[1])
            }
            assert shuffled_cols == original_cols

    def test_reproducibility_with_seed_int(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Same seed should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        shuffles1 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=42)
        )
        shuffles2 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=42)
        )
        for (s1, _), (s2, _) in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_reproducibility_with_generator(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Same generator state should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffles1 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=rng1)
        )
        shuffles2 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=rng2)
        )
        for (s1, _), (s2, _) in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shuffles(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Different seeds should produce different shuffles."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        shuffles1 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=42)
        )
        shuffles2 = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=123)
        )
        any_different = any(
            not np.array_equal(s1, s2)
            for (s1, _), (s2, _) in zip(shuffles1, shuffles2, strict=True)
        )
        assert any_different

    def test_none_rng_produces_shuffles(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """rng=None should still produce valid shuffles."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        shuffles = list(
            shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=5, rng=None)
        )
        assert len(shuffles) == 5
        for shuffled_counts, _ in shuffles:
            assert shuffled_counts.shape == spike_counts.shape

    def test_generator_is_lazy(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Generator should be lazy."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        gen = shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=1000, rng=42
        )
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3

    def test_single_neuron(self, encoding_models: np.ndarray) -> None:
        """Should handle single neuron (trivial shuffle)."""
        from neurospatial.decoding.shuffle import shuffle_cell_identity

        single_neuron_counts = np.array([[1], [2], [3]], dtype=np.int64)
        single_neuron_models = encoding_models[:1, :]
        for shuffled_counts, _models in shuffle_cell_identity(
            single_neuron_counts, single_neuron_models, n_shuffles=5, rng=42
        ):
            assert_array_equal(shuffled_counts, single_neuron_counts)


# =============================================================================
# Tests for shuffle_place_fields_circular
# =============================================================================


class TestShufflePlaceFieldsCircular:
    """Tests for shuffle_place_fields_circular (1D circular shift)."""

    @pytest.fixture
    def encoding_models(self) -> np.ndarray:
        """Sample encoding models: (3 neurons, 8 bins)."""
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Neuron 0
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # Neuron 1
                [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0],  # Neuron 2
            ],
            dtype=np.float64,
        )

    def test_yields_correct_number_of_shuffles(
        self, encoding_models: np.ndarray
    ) -> None:
        """Should yield exactly n_shuffles arrays."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        n_shuffles = 10
        shuffles = list(
            shuffle_place_fields_circular(
                encoding_models, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_yields_correct_shape(self, encoding_models: np.ndarray) -> None:
        """Each shuffled array should have same shape as input."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled.shape == encoding_models.shape

    def test_preserves_dtype(self, encoding_models: np.ndarray) -> None:
        """Shuffled arrays should preserve input dtype."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled.dtype == encoding_models.dtype

    def test_preserves_row_values(self, encoding_models: np.ndarray) -> None:
        """Each row should contain same values (multiset) as original.

        Circular shift preserves the shape of each place field.
        """
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models.shape[0]):
                original_sorted = sorted(encoding_models[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert shuffled_sorted == original_sorted

    def test_rows_are_circular_shifts(self, encoding_models: np.ndarray) -> None:
        """Each row should be a circular shift of the original.

        A circular shift means the row values appear in the same cyclic order.
        """
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models.shape[0]):
                # Check if shuffled row is a circular shift of original
                orig = encoding_models[i, :]
                shuf = shuffled[i, :]
                n_bins = len(orig)
                is_shift = any(
                    np.allclose(shuf, np.roll(orig, shift)) for shift in range(n_bins)
                )
                assert is_shift, f"Row {i} is not a circular shift"

    def test_each_neuron_shifted_independently(
        self, encoding_models: np.ndarray
    ) -> None:
        """Different neurons should get different shift amounts."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        # With enough shuffles, different neurons should have different shifts
        # at least sometimes
        found_different = False
        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=50, rng=42
        ):
            # Check shift amounts for each neuron
            shifts = []
            for i in range(encoding_models.shape[0]):
                orig = encoding_models[i, :]
                shuf = shuffled[i, :]
                # Find the shift amount
                for shift in range(len(orig)):
                    if np.allclose(shuf, np.roll(orig, shift)):
                        shifts.append(shift)
                        break
            # Check if different neurons got different shifts
            if len(set(shifts)) > 1:
                found_different = True
                break
        assert found_different, "Each neuron should be shifted independently"

    def test_reproducibility_with_seed_int(self, encoding_models: np.ndarray) -> None:
        """Same seed should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        shuffles1 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=42)
        )
        shuffles2 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=42)
        )
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_reproducibility_with_generator(self, encoding_models: np.ndarray) -> None:
        """Same generator state should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffles1 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=rng1)
        )
        shuffles2 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=rng2)
        )
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shuffles(
        self, encoding_models: np.ndarray
    ) -> None:
        """Different seeds should produce different shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        shuffles1 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=42)
        )
        shuffles2 = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=123)
        )
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(shuffles1, shuffles2, strict=True)
        )
        assert any_different

    def test_none_rng_produces_shuffles(self, encoding_models: np.ndarray) -> None:
        """rng=None should still produce valid shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        shuffles = list(
            shuffle_place_fields_circular(encoding_models, n_shuffles=5, rng=None)
        )
        assert len(shuffles) == 5
        for shuffled in shuffles:
            assert shuffled.shape == encoding_models.shape

    def test_generator_is_lazy(self, encoding_models: np.ndarray) -> None:
        """Generator should be lazy."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        gen = shuffle_place_fields_circular(encoding_models, n_shuffles=1000, rng=42)
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3

    def test_single_bin(self) -> None:
        """Should handle single bin (trivial shuffle)."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        single_bin = np.array([[5.0], [3.0]], dtype=np.float64)
        for shuffled in shuffle_place_fields_circular(single_bin, n_shuffles=5, rng=42):
            assert_array_equal(shuffled, single_bin)

    def test_single_neuron(self) -> None:
        """Should work with single neuron."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular

        single_neuron = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        for shuffled in shuffle_place_fields_circular(
            single_neuron, n_shuffles=5, rng=42
        ):
            assert shuffled.shape == single_neuron.shape
            # Should be a circular shift
            is_shift = any(
                np.allclose(shuffled[0], np.roll(single_neuron[0], shift))
                for shift in range(4)
            )
            assert is_shift


# =============================================================================
# Tests for shuffle_place_fields_circular_2d
# =============================================================================


class TestShufflePlaceFieldsCircular2D:
    """Tests for shuffle_place_fields_circular_2d (2D circular shift)."""

    @pytest.fixture
    def env_2d(self):
        """Create a 2D environment with full grid (no inactive bins)."""
        from neurospatial import Environment

        # Create a full grid by using infer_active_bins=False
        positions = np.random.default_rng(42).uniform(0, 10, (1000, 2))
        env = Environment.from_samples(positions, bin_size=2.0, infer_active_bins=False)
        return env

    @pytest.fixture
    def encoding_models_2d(self, env_2d) -> np.ndarray:
        """Sample encoding models for 2D environment: (3 neurons, n_bins)."""
        # For 2D shuffle, encoding models must match full grid size
        grid_shape = env_2d.layout.grid_shape
        n_bins = int(np.prod(grid_shape))
        rng = np.random.default_rng(42)
        # Create simple place fields with some structure
        models = rng.random((3, n_bins))
        return models

    def test_yields_correct_number_of_shuffles(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """Should yield exactly n_shuffles arrays."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        n_shuffles = 10
        shuffles = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_yields_correct_shape(self, encoding_models_2d: np.ndarray, env_2d) -> None:
        """Each shuffled array should have same shape as input."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        for shuffled in shuffle_place_fields_circular_2d(
            encoding_models_2d, env_2d, n_shuffles=5, rng=42
        ):
            assert shuffled.shape == encoding_models_2d.shape

    def test_preserves_dtype(self, encoding_models_2d: np.ndarray, env_2d) -> None:
        """Shuffled arrays should preserve input dtype."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        for shuffled in shuffle_place_fields_circular_2d(
            encoding_models_2d, env_2d, n_shuffles=5, rng=42
        ):
            assert shuffled.dtype == encoding_models_2d.dtype

    def test_preserves_row_values(self, encoding_models_2d: np.ndarray, env_2d) -> None:
        """Each row should contain same values (multiset) as original.

        2D circular shift preserves the shape of each place field.
        """
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        for shuffled in shuffle_place_fields_circular_2d(
            encoding_models_2d, env_2d, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models_2d.shape[0]):
                original_sorted = sorted(encoding_models_2d[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert shuffled_sorted == original_sorted

    def test_reproducibility_with_seed_int(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """Same seed should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        shuffles1 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=42
            )
        )
        shuffles2 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=42
            )
        )
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_reproducibility_with_generator(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """Same generator state should produce same shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shuffles1 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=rng1
            )
        )
        shuffles2 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=rng2
            )
        )
        for s1, s2 in zip(shuffles1, shuffles2, strict=True):
            assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_shuffles(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """Different seeds should produce different shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        shuffles1 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=42
            )
        )
        shuffles2 = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=123
            )
        )
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(shuffles1, shuffles2, strict=True)
        )
        assert any_different

    def test_none_rng_produces_shuffles(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """rng=None should still produce valid shuffles."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        shuffles = list(
            shuffle_place_fields_circular_2d(
                encoding_models_2d, env_2d, n_shuffles=5, rng=None
            )
        )
        assert len(shuffles) == 5
        for shuffled in shuffles:
            assert shuffled.shape == encoding_models_2d.shape

    def test_generator_is_lazy(self, encoding_models_2d: np.ndarray, env_2d) -> None:
        """Generator should be lazy."""
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        gen = shuffle_place_fields_circular_2d(
            encoding_models_2d, env_2d, n_shuffles=1000, rng=42
        )
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3

    def test_requires_2d_environment(self) -> None:
        """Should raise ValueError for non-2D environments."""
        from neurospatial import Environment
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        # Create a 1D environment
        positions = np.linspace(0, 10, 50).reshape(-1, 1)
        env_1d = Environment.from_samples(positions, bin_size=1.0)
        encoding_models = np.random.default_rng(42).random((3, env_1d.n_bins))

        with pytest.raises(ValueError, match="2D"):
            list(
                shuffle_place_fields_circular_2d(
                    encoding_models, env_1d, n_shuffles=5, rng=42
                )
            )

    def test_rejects_masked_grid(self) -> None:
        """Should raise ValueError if environment has inactive bins (masked grid)."""
        from neurospatial import Environment
        from neurospatial.decoding.shuffle import shuffle_place_fields_circular_2d

        # Create 2D environment with inactive bins (default behavior)
        positions = np.random.default_rng(42).uniform(0, 10, (100, 2))
        env = Environment.from_samples(positions, bin_size=2.0)
        # Create encoding models matching n_bins (not grid_shape product)
        encoding_models = np.random.default_rng(42).random((3, env.n_bins))

        # Should fail because n_bins != prod(grid_shape)
        with pytest.raises(ValueError, match="inactive bins"):
            list(
                shuffle_place_fields_circular_2d(
                    encoding_models, env, n_shuffles=5, rng=42
                )
            )
