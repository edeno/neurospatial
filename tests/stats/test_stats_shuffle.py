"""Tests for stats.shuffle module - verifies new import paths work.

This file tests the reorganization of shuffle functions into
the neurospatial.stats.shuffle module per PLAN.md Milestone 4.

Functions moved from decoding/shuffle.py:
- shuffle_time_bins, shuffle_time_bins_coherent
- shuffle_cell_identity
- shuffle_place_fields_circular, shuffle_place_fields_circular_2d
- shuffle_posterior_circular, shuffle_posterior_weighted_circular
- ShuffleTestResult, compute_shuffle_pvalue, compute_shuffle_zscore

New functions per PLAN.md:
- shuffle_trials(): Shuffle trial labels
- shuffle_spikes_isi(): Shuffle inter-spike intervals
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Test imports from new location (stats/shuffle.py)
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    shuffle_spikes_isi,
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_trials,
)


class TestStatsShuffleImports:
    """Test that all functions are importable from neurospatial.stats.shuffle."""

    # --- Temporal order shuffles ---

    def test_shuffle_time_bins_importable(self):
        """Test shuffle_time_bins is importable from stats.shuffle."""
        assert callable(shuffle_time_bins)

    def test_shuffle_time_bins_coherent_importable(self):
        """Test shuffle_time_bins_coherent is importable from stats.shuffle."""
        assert callable(shuffle_time_bins_coherent)

    # --- Cell identity shuffles ---

    def test_shuffle_cell_identity_importable(self):
        """Test shuffle_cell_identity is importable from stats.shuffle."""
        assert callable(shuffle_cell_identity)

    def test_shuffle_place_fields_circular_importable(self):
        """Test shuffle_place_fields_circular is importable from stats.shuffle."""
        assert callable(shuffle_place_fields_circular)

    def test_shuffle_place_fields_circular_2d_importable(self):
        """Test shuffle_place_fields_circular_2d is importable from stats.shuffle."""
        assert callable(shuffle_place_fields_circular_2d)

    # --- Posterior shuffles ---

    def test_shuffle_posterior_circular_importable(self):
        """Test shuffle_posterior_circular is importable from stats.shuffle."""
        assert callable(shuffle_posterior_circular)

    def test_shuffle_posterior_weighted_circular_importable(self):
        """Test shuffle_posterior_weighted_circular is importable."""
        assert callable(shuffle_posterior_weighted_circular)

    # --- Significance testing ---

    def test_shuffle_test_result_importable(self):
        """Test ShuffleTestResult is importable from stats.shuffle."""
        assert ShuffleTestResult is not None

    def test_compute_shuffle_pvalue_importable(self):
        """Test compute_shuffle_pvalue is importable from stats.shuffle."""
        assert callable(compute_shuffle_pvalue)

    def test_compute_shuffle_zscore_importable(self):
        """Test compute_shuffle_zscore is importable from stats.shuffle."""
        assert callable(compute_shuffle_zscore)

    # --- New functions per PLAN.md ---

    def test_shuffle_trials_importable(self):
        """Test shuffle_trials is importable from stats.shuffle."""
        assert callable(shuffle_trials)

    def test_shuffle_spikes_isi_importable(self):
        """Test shuffle_spikes_isi is importable from stats.shuffle."""
        assert callable(shuffle_spikes_isi)


class TestStatsShuffleBasicFunctionality:
    """Basic functionality tests for stats.shuffle module."""

    def test_shuffle_time_bins_preserves_sum(self):
        """Test shuffle_time_bins preserves total spike count."""
        spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
        original_sum = spike_counts.sum()

        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            assert shuffled.sum() == original_sum
            assert shuffled.shape == spike_counts.shape

    def test_shuffle_time_bins_changes_order(self):
        """Test shuffle_time_bins actually changes temporal order."""
        spike_counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int64)

        # With enough shuffles, at least one should differ from original
        any_different = False
        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=10, rng=42):
            if not np.array_equal(shuffled, spike_counts):
                any_different = True
                break
        assert any_different

    def test_shuffle_time_bins_coherent_preserves_rows(self):
        """Test shuffle_time_bins_coherent preserves population vectors."""
        spike_counts = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int64)
        original_rows = {tuple(row) for row in spike_counts}

        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            shuffled_rows = {tuple(row) for row in shuffled}
            assert original_rows == shuffled_rows

    def test_shuffle_cell_identity_preserves_counts(self):
        """Test shuffle_cell_identity preserves spike counts."""
        spike_counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
        encoding_models = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        for shuffled, models in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled.sum() == spike_counts.sum()
            assert models is encoding_models  # Same object

    def test_shuffle_place_fields_circular_preserves_shape(self):
        """Test shuffle_place_fields_circular preserves field shapes."""
        encoding_models = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

        for shuffled in shuffle_place_fields_circular(
            encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled.shape == encoding_models.shape
            # Each row should have same values, just shifted
            for i in range(encoding_models.shape[0]):
                assert set(shuffled[i]) == set(encoding_models[i])

    def test_shuffle_posterior_circular_preserves_normalization(self):
        """Test shuffle_posterior_circular preserves row normalization."""
        raw = np.array(
            [
                [0.1, 0.2, 0.4, 0.2, 0.1],
                [0.5, 0.3, 0.1, 0.05, 0.05],
                [0.05, 0.1, 0.2, 0.4, 0.25],
            ]
        )

        for shuffled in shuffle_posterior_circular(raw, n_shuffles=5, rng=42):
            assert_allclose(shuffled.sum(axis=1), 1.0)

    def test_shuffle_posterior_weighted_circular_preserves_normalization(self):
        """Test shuffle_posterior_weighted_circular preserves row normalization."""
        raw = np.random.default_rng(42).random((3, 10))
        posterior = raw / raw.sum(axis=1, keepdims=True)

        for shuffled in shuffle_posterior_weighted_circular(
            posterior, edge_buffer=2, n_shuffles=5, rng=42
        ):
            assert_allclose(shuffled.sum(axis=1), 1.0)

    def test_compute_shuffle_pvalue_basic(self):
        """Test compute_shuffle_pvalue with basic cases."""
        # Observed higher than all null values
        observed = 10.0
        null = np.array([1.0, 2.0, 3.0, 4.0])
        p = compute_shuffle_pvalue(observed, null, tail="greater")
        assert_allclose(p, 0.2)  # (0 + 1) / (4 + 1)

    def test_compute_shuffle_pvalue_less_tail(self):
        """Test compute_shuffle_pvalue with less tail."""
        observed = 0.5
        null = np.array([1.0, 2.0, 3.0, 4.0])
        p = compute_shuffle_pvalue(observed, null, tail="less")
        assert_allclose(p, 0.2)  # (0 + 1) / (4 + 1)

    def test_compute_shuffle_pvalue_two_sided(self):
        """Test compute_shuffle_pvalue with two-sided test."""
        observed = 10.0
        null = np.array([1.0, 2.0, 3.0, 4.0])
        p = compute_shuffle_pvalue(observed, null, tail="two-sided")
        assert_allclose(p, 0.4)  # 2 * 0.2

    def test_compute_shuffle_zscore_basic(self):
        """Test compute_shuffle_zscore computation."""
        null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = 5.0
        z = compute_shuffle_zscore(observed, null)
        expected_z = (5.0 - 3.0) / np.std(null)
        assert_allclose(z, expected_z)

    def test_compute_shuffle_zscore_zero_variance(self):
        """Test compute_shuffle_zscore returns NaN for zero variance."""
        null = np.array([3.0, 3.0, 3.0])
        z = compute_shuffle_zscore(5.0, null)
        assert np.isnan(z)

    def test_shuffle_test_result_is_significant(self):
        """Test ShuffleTestResult.is_significant property."""
        result_sig = ShuffleTestResult(
            observed_score=5.0,
            null_scores=np.array([1.0, 2.0, 3.0, 4.0]),
            p_value=0.01,
            z_score=3.0,
            shuffle_type="time_bins",
            n_shuffles=4,
        )
        assert result_sig.is_significant

        result_not_sig = ShuffleTestResult(
            observed_score=3.0,
            null_scores=np.array([1.0, 2.0, 3.0, 4.0]),
            p_value=0.5,
            z_score=0.0,
            shuffle_type="time_bins",
            n_shuffles=4,
        )
        assert not result_not_sig.is_significant


class TestNewShuffleFunctions:
    """Tests for new shuffle functions per PLAN.md."""

    def test_shuffle_trials_basic(self):
        """Test shuffle_trials shuffles trial labels."""
        trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Collect shuffled versions
        shuffled_list = list(shuffle_trials(trial_labels, n_shuffles=5, rng=42))

        assert len(shuffled_list) == 5
        for shuffled in shuffled_list:
            assert shuffled.shape == trial_labels.shape
            # Same values, just rearranged
            assert set(shuffled) == set(trial_labels)
            # Same counts per unique value
            for val in np.unique(trial_labels):
                assert np.sum(shuffled == val) == np.sum(trial_labels == val)

    def test_shuffle_trials_changes_order(self):
        """Test shuffle_trials actually changes order."""
        trial_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # With enough shuffles, at least one should differ
        any_different = False
        for shuffled in shuffle_trials(trial_labels, n_shuffles=10, rng=42):
            if not np.array_equal(shuffled, trial_labels):
                any_different = True
                break
        assert any_different

    def test_shuffle_spikes_isi_basic(self):
        """Test shuffle_spikes_isi shuffles inter-spike intervals."""
        # Spike times for a single neuron
        spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])

        shuffled_list = list(shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42))

        assert len(shuffled_list) == 5
        for shuffled in shuffled_list:
            assert shuffled.shape == spike_times.shape
            # Should preserve number of spikes
            assert len(shuffled) == len(spike_times)
            # Should be sorted (times are ordered)
            assert np.all(np.diff(shuffled) >= 0)

    def test_shuffle_spikes_isi_preserves_isi_distribution(self):
        """Test shuffle_spikes_isi preserves ISI values."""
        spike_times = np.array([0.1, 0.15, 0.25, 0.4, 0.45, 0.7])
        original_isis = np.diff(spike_times)

        for shuffled in shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42):
            shuffled_isis = np.diff(shuffled)
            # Same ISI values (just reordered)
            assert_allclose(sorted(shuffled_isis), sorted(original_isis))

    def test_shuffle_spikes_isi_preserves_first_spike(self):
        """Test shuffle_spikes_isi preserves first spike time."""
        spike_times = np.array([0.1, 0.15, 0.25, 0.4])

        for shuffled in shuffle_spikes_isi(spike_times, n_shuffles=5, rng=42):
            assert_allclose(shuffled[0], spike_times[0])


class TestStatsPackageExports:
    """Test that functions are also accessible via stats package."""

    def test_import_from_stats_package(self):
        """Test that shuffle functions can be imported from neurospatial.stats."""
        from neurospatial.stats import (
            ShuffleTestResult,
            compute_shuffle_pvalue,
            compute_shuffle_zscore,
            shuffle_cell_identity,
            shuffle_place_fields_circular,
            shuffle_posterior_circular,
            shuffle_spikes_isi,
            shuffle_time_bins,
            shuffle_time_bins_coherent,
            shuffle_trials,
        )

        assert callable(shuffle_time_bins)
        assert callable(shuffle_time_bins_coherent)
        assert callable(shuffle_cell_identity)
        assert callable(shuffle_place_fields_circular)
        assert callable(shuffle_posterior_circular)
        assert callable(compute_shuffle_pvalue)
        assert callable(compute_shuffle_zscore)
        assert callable(shuffle_trials)
        assert callable(shuffle_spikes_isi)
        assert ShuffleTestResult is not None


class TestShufflePlaceFieldsCircular2D:
    """Tests for shuffle_place_fields_circular_2d with Environment."""

    def test_shuffle_place_fields_circular_2d_basic(self):
        """Test shuffle_place_fields_circular_2d with 2D environment."""
        from neurospatial import Environment

        # Create a 2D environment
        positions = np.random.default_rng(42).uniform(0, 10, (100, 2))
        env = Environment.from_samples(positions, bin_size=2.0)

        # Create encoding models matching grid size
        n_neurons = 3
        encoding_models = np.random.default_rng(42).random((n_neurons, env.n_bins))

        # The function requires full grid (no masked bins)
        # Skip if environment has inactive bins
        if not hasattr(env.layout, "grid_shape") or env.layout.grid_shape is None:
            pytest.skip("Environment doesn't have grid layout")

        expected_bins = int(np.prod(env.layout.grid_shape))
        if env.n_bins != expected_bins:
            pytest.skip("Environment has masked bins")

        for shuffled in shuffle_place_fields_circular_2d(
            encoding_models, env, n_shuffles=3, rng=42
        ):
            assert shuffled.shape == encoding_models.shape

    def test_shuffle_place_fields_circular_2d_raises_for_1d(self):
        """Test shuffle_place_fields_circular_2d raises for 1D environment."""
        from neurospatial import Environment

        # Create a 1D environment
        positions = np.random.default_rng(42).uniform(0, 10, (100, 1))
        env = Environment.from_samples(positions, bin_size=2.0)

        encoding_models = np.random.default_rng(42).random((3, env.n_bins))

        with pytest.raises(ValueError, match="requires a 2D environment"):
            next(shuffle_place_fields_circular_2d(encoding_models, env, n_shuffles=1))
