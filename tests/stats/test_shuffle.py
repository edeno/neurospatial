"""Tests for neurospatial.stats.shuffle module.

Tests cover temporal shuffles for significance testing in Bayesian decoding.
"""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import track_for
from numpy.testing import assert_array_equal

from neurospatial.stats.shuffle import (
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


# =============================================================================
# Shared contract of every shuffle and surrogate generator
# =============================================================================


def _counts(n_time_bins: int, n_neurons: int) -> np.ndarray:
    return (
        np.random.default_rng(42)
        .poisson(2.0, size=(n_time_bins, n_neurons))
        .astype(np.int64)
    )


def _rows(n_rows: int, n_columns: int) -> np.ndarray:
    """Rows of positive values that each sum to 1."""
    raw = np.random.default_rng(42).random((n_rows, n_columns)) + 0.1
    return raw / raw.sum(axis=1, keepdims=True)


def _grid_env():
    """A full 6 x 6 grid of 2 cm bins (no inactive bins)."""
    from neurospatial import Environment

    positions = np.random.default_rng(42).uniform(0, 10, (1000, 2))
    return Environment.from_samples(positions, bin_size=2.0, infer_active_bins=False)


def _draw(name: str, data: np.ndarray, rng, n: int = 5) -> list[np.ndarray]:
    """``n`` shuffles of ``data`` from the generator ``name``."""
    from neurospatial.replay import nulls
    from neurospatial.stats import shuffle, surrogates

    if name == "shuffle_cell_identity":
        models = np.arange(data.shape[1] * 4, dtype=np.float64).reshape(-1, 4)
        draws = nulls.shuffle_cell_identity(data, models, n_shuffles=n, rng=rng)
        return [counts for counts, _ in draws]
    if name.startswith("generate_"):
        return list(getattr(surrogates, name)(data, n_surrogates=n, rng=rng))
    function = getattr(nulls if hasattr(nulls, name) else shuffle, name)
    if name == "shuffle_place_fields_circular_2d":
        return list(function(_grid_env(), data, n_shuffles=n, rng=rng))
    if name == "shuffle_posterior_weighted_circular":
        weights = np.ones(data.shape[1])
        return list(function(track_for(data), data, weights, n_shuffles=n, rng=rng))
    if name in (
        "shuffle_time_bins",
        "shuffle_time_bins_coherent",
        "shuffle_spike_counts_circular",
        "shuffle_spike_counts_uniform",
    ):
        return list(function(data, n_shuffles=n, rng=rng))
    return list(function(track_for(data), data, n_shuffles=n, rng=rng))


# Each generator with an input whose shuffles differ between seeds.
GENERATOR_INPUTS = {
    "shuffle_time_bins": _counts(10, 5),
    "shuffle_time_bins_coherent": _counts(10, 5),
    "shuffle_spike_counts_circular": _counts(10, 5),
    "shuffle_spike_counts_uniform": _counts(10, 5),
    "shuffle_cell_identity": _counts(10, 5),
    "shuffle_place_fields_circular": _rows(3, 8),
    "shuffle_place_fields_circular_2d": _rows(3, 36),
    "shuffle_posterior_circular": _rows(10, 20),
    "shuffle_posterior_weighted_circular": _rows(10, 20),
    "generate_poisson_surrogates": _counts(10, 5),
    "generate_inhomogeneous_poisson_surrogates": _counts(10, 5),
}

# Degenerate inputs and what every shuffle of them must be: "unchanged"
# (nothing to permute), "same_values" (each row keeps its values, as a
# permutation or circular shift does) or "counts" (non-negative integers).
DEGENERATE_INPUTS = [
    ("shuffle_time_bins", "empty", np.zeros((0, 3), dtype=np.int64), "unchanged"),
    ("shuffle_time_bins", "single_time_bin", _counts(1, 3), "unchanged"),
    ("shuffle_time_bins", "single_neuron", _counts(5, 1), "same_values_by_column"),
    ("shuffle_time_bins_coherent", "empty", np.zeros((0, 3), np.int64), "unchanged"),
    ("shuffle_time_bins_coherent", "single_time_bin", _counts(1, 3), "unchanged"),
    ("shuffle_cell_identity", "single_neuron", _counts(5, 1), "unchanged"),
    ("shuffle_spike_counts_uniform", "empty", np.zeros((0, 3), np.int64), "unchanged"),
    ("shuffle_spike_counts_uniform", "single_time_bin", _counts(1, 3), "unchanged"),
    ("shuffle_spike_counts_uniform", "single_neuron", _counts(5, 1), "counts"),
    ("shuffle_place_fields_circular", "single_bin", _rows(2, 1), "unchanged"),
    ("shuffle_place_fields_circular", "single_neuron", _rows(1, 4), "same_values"),
    ("shuffle_posterior_circular", "empty", np.zeros((0, 5)), "unchanged"),
    ("shuffle_posterior_circular", "single_time_bin", _rows(1, 4), "same_values"),
    ("shuffle_posterior_circular", "single_bin", _rows(3, 1), "unchanged"),
    ("shuffle_posterior_weighted_circular", "empty", np.zeros((0, 5)), "unchanged"),
    (
        "shuffle_posterior_weighted_circular",
        "single_time_bin",
        _rows(1, 4),
        "same_values",
    ),
    ("generate_poisson_surrogates", "empty", np.zeros((0, 3), np.int64), "counts"),
    ("generate_poisson_surrogates", "single_time_bin", _counts(1, 3), "counts"),
    ("generate_poisson_surrogates", "single_neuron", _counts(5, 1), "counts"),
    (
        "generate_inhomogeneous_poisson_surrogates",
        "empty",
        np.zeros((0, 3), np.int64),
        "counts",
    ),
    (
        "generate_inhomogeneous_poisson_surrogates",
        "single_time_bin",
        _counts(1, 3),
        "counts",
    ),
    (
        "generate_inhomogeneous_poisson_surrogates",
        "single_neuron",
        _counts(5, 1),
        "counts",
    ),
]


@pytest.mark.parametrize("name", list(GENERATOR_INPUTS))
class TestGeneratorRngContract:
    """Every generator takes an int seed, a Generator or None."""

    def test_int_seed_and_generator_reproduce_the_draws(self, name: str) -> None:
        data = GENERATOR_INPUTS[name]
        first = _draw(name, data, 42)
        for again in (
            _draw(name, data, 42),
            _draw(name, data, np.random.default_rng(42)),
        ):
            for a, b in zip(first, again, strict=True):
                assert_array_equal(a, b)

    def test_different_seeds_give_different_draws(self, name: str) -> None:
        data = GENERATOR_INPUTS[name]
        first, second = _draw(name, data, 42), _draw(name, data, 123)
        assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_rng_none_draws_the_requested_number(self, name: str) -> None:
        data = GENERATOR_INPUTS[name]
        draws = _draw(name, data, None, n=4)
        assert len(draws) == 4
        assert all(draw.shape == data.shape for draw in draws)


@pytest.mark.parametrize(
    ("name", "data", "expect"),
    [
        pytest.param(name, data, expect, id=f"{name}-{case}")
        for name, case, data, expect in DEGENERATE_INPUTS
    ],
)
def test_degenerate_inputs(name: str, data: np.ndarray, expect: str) -> None:
    """Empty, single-bin and single-neuron inputs give valid draws."""
    draws = _draw(name, data, 42)
    assert len(draws) == 5
    for draw in draws:
        assert draw.shape == data.shape
        if expect == "unchanged":
            assert_array_equal(draw, data)
        elif expect == "same_values":
            assert_array_equal(np.sort(draw, axis=1), np.sort(data, axis=1))
        elif expect == "same_values_by_column":
            assert_array_equal(np.sort(draw, axis=0), np.sort(data, axis=0))
        else:
            assert np.issubdtype(draw.dtype, np.integer)
            assert np.all(draw >= 0)


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


# =============================================================================
# Tests comparing shuffle_time_bins vs shuffle_time_bins_coherent
# =============================================================================


class TestShuffleComparison:
    """Tests comparing the two shuffle functions."""

    def test_coherent_matches_time_bins_for_fixed_seed(self):
        """shuffle_time_bins_coherent equals shuffle_time_bins for a fixed seed.

        The coherent variant is documented as the same permutation applied
        to every neuron, which is exactly what shuffle_time_bins does. They
        must produce byte-identical output for the same seed; this guards
        against the two implementations silently diverging.
        """
        rng = np.random.default_rng(7)
        spike_counts = rng.integers(0, 5, size=(8, 4)).astype(np.int64)

        plain = list(shuffle_time_bins(spike_counts, n_shuffles=20, rng=2024))
        coherent = list(
            shuffle_time_bins_coherent(spike_counts, n_shuffles=20, rng=2024)
        )

        assert len(plain) == len(coherent) == 20
        for a, b in zip(plain, coherent, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_both_preserve_row_multiset(self, spike_counts: np.ndarray) -> None:
        """Both shuffles should preserve the multiset of rows."""
        original_rows = sorted([tuple(row) for row in spike_counts])

        for shuffled in shuffle_time_bins(spike_counts, n_shuffles=5, rng=42):
            shuffled_rows = sorted([tuple(row) for row in shuffled])
            assert shuffled_rows == original_rows

        for shuffled in shuffle_time_bins_coherent(spike_counts, n_shuffles=5, rng=42):
            shuffled_rows = sorted([tuple(row) for row in shuffled])
            assert shuffled_rows == original_rows


# =============================================================================
# Tests for generate_inhomogeneous_poisson_surrogates
# =============================================================================


# =============================================================================
# Tests for compute_shuffle_pvalue
# =============================================================================


class TestShufflePvalueCalibration:
    """Under the null, P(p <= alpha) is alpha and p is uniform.

    The observed score and its 199 null scores come from one distribution,
    so the observed rank is uniform among the 200 and the Phipson-Smyth
    p-value (k + 1) / (n + 1) satisfies P(p <= alpha) = alpha for alpha a
    multiple of 1/200. With 2000 trials the Monte Carlo standard error is
    0.005 at alpha = 0.05 and 0.007 at alpha = 0.1.
    """

    @pytest.fixture(scope="class")
    def null_pvalues(self) -> dict[str, np.ndarray]:
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        rng = np.random.default_rng(2026)
        observed = rng.standard_normal(2000)
        null = rng.standard_normal((2000, 199))
        return {
            tail: np.array(
                [
                    compute_shuffle_pvalue(score, scores, tail=tail)
                    for score, scores in zip(observed, null, strict=True)
                ]
            )
            for tail in ("greater", "less", "two-sided")
        }

    @pytest.mark.parametrize("tail", ["greater", "less"])
    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    def test_one_sided_false_positive_rate_is_alpha(
        self, null_pvalues: dict[str, np.ndarray], tail: str, alpha: float
    ) -> None:
        assert np.mean(null_pvalues[tail] <= alpha) == pytest.approx(alpha, abs=0.015)

    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    def test_two_sided_false_positive_rate_is_at_most_alpha(
        self, null_pvalues: dict[str, np.ndarray], alpha: float
    ) -> None:
        assert np.mean(null_pvalues["two-sided"] <= alpha) <= alpha + 0.015

    @pytest.mark.parametrize("tail", ["greater", "less", "two-sided"])
    def test_null_pvalues_are_uniform(
        self, null_pvalues: dict[str, np.ndarray], tail: str
    ) -> None:
        from scipy.stats import kstest

        assert kstest(null_pvalues[tail], "uniform").pvalue > 0.01


class TestComputeShufflePvalue:
    """Tests for compute_shuffle_pvalue function."""

    def test_basic_greater_tail(self) -> None:
        """P-value for observed > all null should be minimal (1/(n+1))."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 10.0
        null_scores = np.array([1.0, 2.0, 3.0, 4.0])  # All less than observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="greater")
        # Monte Carlo correction: (k + 1) / (n + 1) where k=0 (no null >= observed)
        assert p == (0 + 1) / (4 + 1)
        assert p == 0.2

    def test_basic_less_tail(self) -> None:
        """P-value for observed < all null should be minimal."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 0.0
        null_scores = np.array([1.0, 2.0, 3.0, 4.0])  # All greater than observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="less")
        # (k + 1) / (n + 1) where k=0 (no null <= observed)
        assert p == (0 + 1) / (4 + 1)
        assert p == 0.2

    def test_observed_in_null_distribution_greater(self) -> None:
        """P-value when observed is in the middle of null distribution."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([1.0, 2.0, 5.0, 6.0, 10.0])  # 5.0, 6.0, 10.0 >= observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="greater")
        # k=3 (three values >= 5.0), n=5
        assert p == (3 + 1) / (5 + 1)
        assert np.isclose(p, 4 / 6)

    def test_observed_in_null_distribution_less(self) -> None:
        """P-value when observed is in the middle of null distribution (less tail)."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([1.0, 2.0, 5.0, 6.0, 10.0])  # 1.0, 2.0, 5.0 <= observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="less")
        # k=3 (three values <= 5.0), n=5
        assert p == (3 + 1) / (5 + 1)
        assert np.isclose(p, 4 / 6)

    def test_two_sided_tail(self) -> None:
        """Two-sided p-value should be 2 * min(p_greater, p_less)."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 8.0
        null_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Greater: k=0 (no null >= 8.0), p_greater = 1/6
        # Less: k=5 (all null <= 8.0), p_less = 6/6 = 1.0
        # Two-sided: 2 * min(1/6, 1.0) = 2/6 = 1/3
        p = compute_shuffle_pvalue(observed, null_scores, tail="two-sided")
        expected = 2 * (1 / 6)
        assert np.isclose(p, expected)

    def test_two_sided_capped_at_one(self) -> None:
        """Two-sided p-value should not exceed 1.0."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 3.0
        null_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Greater: k=3 (3.0, 4.0, 5.0 >= 3.0), p_greater = 4/6
        # Less: k=3 (1.0, 2.0, 3.0 <= 3.0), p_less = 4/6
        # Two-sided: 2 * min(4/6, 4/6) = 8/6 -> capped to 1.0
        p = compute_shuffle_pvalue(observed, null_scores, tail="two-sided")
        assert p <= 1.0
        assert np.isclose(p, 1.0)

    def test_all_null_equal_observed_greater(self) -> None:
        """When all null values equal observed (greater tail)."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([5.0, 5.0, 5.0])  # All equal observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="greater")
        # k=3 (all >= 5.0), n=3
        assert p == (3 + 1) / (3 + 1)
        assert p == 1.0

    def test_all_null_equal_observed_less(self) -> None:
        """When all null values equal observed (less tail)."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([5.0, 5.0, 5.0])  # All equal observed
        p = compute_shuffle_pvalue(observed, null_scores, tail="less")
        # k=3 (all <= 5.0), n=3
        assert p == (3 + 1) / (3 + 1)
        assert p == 1.0

    def test_single_null_value(self) -> None:
        """Should work with single null value."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([3.0])
        p_greater = compute_shuffle_pvalue(observed, null_scores, tail="greater")
        # k=0 (no null >= 5.0), n=1
        assert p_greater == (0 + 1) / (1 + 1)
        assert p_greater == 0.5

        p_less = compute_shuffle_pvalue(observed, null_scores, tail="less")
        # k=1 (3.0 <= 5.0), n=1
        assert p_less == (1 + 1) / (1 + 1)
        assert p_less == 1.0

    def test_large_null_distribution(self) -> None:
        """Should work with many null values."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        rng = np.random.default_rng(42)
        null_scores = rng.normal(0, 1, 10000)
        observed = 3.0  # Far in the tail

        p = compute_shuffle_pvalue(observed, null_scores, tail="greater")
        # Should be small (around 0.001 for z=3)
        assert 0 < p < 0.01

    def test_pvalue_range(self) -> None:
        """P-value should always be in (0, 1]."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        # Various test cases
        test_cases = [
            (10.0, np.array([1.0, 2.0, 3.0]), "greater"),
            (0.0, np.array([1.0, 2.0, 3.0]), "greater"),
            (2.0, np.array([1.0, 2.0, 3.0]), "greater"),
            (10.0, np.array([1.0, 2.0, 3.0]), "less"),
            (0.0, np.array([1.0, 2.0, 3.0]), "less"),
            (2.0, np.array([1.0, 2.0, 3.0]), "two-sided"),
        ]
        for observed, null, tail in test_cases:
            p = compute_shuffle_pvalue(observed, null, tail=tail)
            assert 0 < p <= 1.0, f"p={p} out of range for {observed}, {null}, {tail}"

    def test_invalid_tail_raises_error(self) -> None:
        """Invalid tail parameter should raise ValueError."""
        from neurospatial.stats.shuffle import compute_shuffle_pvalue

        observed = 5.0
        null_scores = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="tail"):
            compute_shuffle_pvalue(observed, null_scores, tail="invalid")  # type: ignore[arg-type]


# =============================================================================
# Tests for compute_shuffle_zscore
# =============================================================================


class TestComputeShuffleZscore:
    """Tests for compute_shuffle_zscore function."""

    def test_basic_zscore(self) -> None:
        """Z-score should be (observed - mean) / std."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        observed = 5.0
        null_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # mean = 3.0, std = sqrt(2.0)
        expected_z = (5.0 - 3.0) / np.std(null_scores)
        z = compute_shuffle_zscore(observed, null_scores)
        assert np.isclose(z, expected_z)

    def test_observed_equals_mean(self) -> None:
        """Z-score should be 0 when observed equals null mean."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        null_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed = np.mean(null_scores)
        z = compute_shuffle_zscore(observed, null_scores)
        assert np.isclose(z, 0.0)

    def test_positive_zscore(self) -> None:
        """Z-score should be positive when observed > mean."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        observed = 10.0
        null_scores = np.array([1.0, 2.0, 3.0])  # mean = 2.0
        z = compute_shuffle_zscore(observed, null_scores)
        assert z > 0

    def test_negative_zscore(self) -> None:
        """Z-score should be negative when observed < mean."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        observed = 0.0
        null_scores = np.array([1.0, 2.0, 3.0])  # mean = 2.0
        z = compute_shuffle_zscore(observed, null_scores)
        assert z < 0

    def test_zero_variance_returns_nan(self) -> None:
        """Z-score should be NaN when null distribution has zero variance."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        observed = 5.0
        null_scores = np.array([3.0, 3.0, 3.0])  # All equal, std = 0
        z = compute_shuffle_zscore(observed, null_scores)
        assert np.isnan(z)

    def test_non_finite_null_scores_are_left_out(self) -> None:
        """One undefined shuffle in 2000 does not make the z-score NaN."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        rng = np.random.default_rng(0)
        finite = rng.normal(0.0, 1.0, 1999)
        null_scores = np.append(finite, [np.nan])
        z = compute_shuffle_zscore(3.0, null_scores)
        assert z == pytest.approx((3.0 - finite.mean()) / finite.std())
        assert np.isnan(compute_shuffle_zscore(3.0, np.array([np.nan, 1.0])))

    def test_single_null_value_returns_nan(self) -> None:
        """Z-score should be NaN with single null value (undefined std)."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        observed = 5.0
        null_scores = np.array([3.0])
        z = compute_shuffle_zscore(observed, null_scores)
        assert np.isnan(z)

    def test_large_null_distribution(self) -> None:
        """Should work with many null values."""
        from neurospatial.stats.shuffle import compute_shuffle_zscore

        rng = np.random.default_rng(42)
        null_scores = rng.normal(0, 1, 10000)
        observed = 3.0  # 3 standard deviations away

        z = compute_shuffle_zscore(observed, null_scores)
        # Should be close to 3 (within statistical variation)
        assert 2.5 < z < 3.5


# =============================================================================
# Tests for ShuffleTestResult
# =============================================================================


class TestShuffleTestResult:
    """Tests for ShuffleTestResult dataclass."""

    def test_is_frozen(self) -> None:
        """ShuffleTestResult should be a frozen dataclass."""
        from neurospatial.stats.shuffle import ShuffleTestResult

        result = ShuffleTestResult(
            observed_score=5.0,
            null_scores=np.array([1.0, 2.0, 3.0, 4.0]),
            p_value=0.2,
            z_score=1.5,
            shuffle_type="time_bins",
            n_shuffles=4,
        )
        # Try to modify - should raise
        with pytest.raises((AttributeError, TypeError)):
            result.observed_score = 10.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("p_value", "alpha", "expected"),
        [
            (0.01, 0.05, True),
            (0.10, 0.05, False),
            (0.05, 0.05, False),
            (0.04, 0.01, False),
            (0.04, 0.1, True),
        ],
    )
    def test_is_significant(self, p_value: float, alpha: float, expected: bool) -> None:
        """is_significant(alpha) is the strict comparison p_value < alpha."""
        from neurospatial.stats.shuffle import ShuffleTestResult

        result = ShuffleTestResult(
            observed_score=5.0,
            null_scores=np.array([1.0, 2.0, 3.0]),
            p_value=p_value,
            z_score=2.5,
            shuffle_type="time_bins",
            n_shuffles=3,
        )
        assert result.is_significant(alpha=alpha) is expected
