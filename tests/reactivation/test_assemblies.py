"""Tests for neurospatial.reactivation.assemblies module.

Tests cover cell assembly detection, activation computation, and reactivation analysis.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.reactivation.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    ExplainedVarianceResult,
    assembly_activation,
    assembly_reactivation,
    detect_assemblies,
    explained_variance_reactivation,
    marchenko_pastur_threshold,
    pairwise_correlations,
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

    Shape: (2000 time bins, 50 neurons)  -- time-first, matching decode_position.
    Assemblies:
    - Neurons 0-4: co-activate together
    - Neurons 10-14: co-activate together
    - Neurons 20-24: co-activate together
    """
    n_neurons, n_time = 50, 2000

    # Baseline Poisson activity (built neuron-first, returned time-first).
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

    return spike_counts.T  # (n_time_bins, n_neurons)


@pytest.fixture
def random_spike_counts(rng: np.random.Generator) -> np.ndarray:
    """Random spike counts with no assembly structure.

    Shape: (3000 time bins, 30 neurons).
    """
    return rng.poisson(5, (3000, 30)).astype(np.float64)


@pytest.fixture
def small_spike_counts(rng: np.random.Generator) -> np.ndarray:
    """Small spike counts for basic tests.

    Shape: (100 time bins, 10 neurons).
    """
    return rng.poisson(3, (100, 10)).astype(np.float64)


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


class TestDetectAssemblies:
    """Tests for detect_assemblies function."""

    @pytest.mark.parametrize("method", ["ica", "nmf"])
    def test_detects_known_assemblies(
        self, spike_counts_with_assemblies: np.ndarray, method: str
    ) -> None:
        """Each planted assembly is one pattern whose members are exactly its cells.

        Matching neuron ids pins the (n_time_bins, n_neurons) orientation: a
        wrong axis would recover garbage ids.
        """
        result = detect_assemblies(spike_counts_with_assemblies, method=method, rng=42)

        assert result.n_significant == 3
        recovered = sorted(
            pattern.member_indices.tolist() for pattern in result.patterns
        )
        assert recovered == [
            list(range(0, 5)),
            list(range(10, 15)),
            list(range(20, 25)),
        ]

    def test_pca_patterns_span_the_planted_assemblies(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Equally strong assemblies mix in PCA, but their span is recovered.

        The three planted eigenvalues are nearly equal, so each eigenvector
        mixes the assemblies and a pattern can have no core member. The
        subspace they span still holds each assembly's indicator vector, up
        to the sampling perturbation of the eigenvectors, about
        sqrt(n_neurons / n_time_bins) * sqrt(lambda) / (lambda - 1) = 0.13 for
        eigenvalue lambda = 3 with 50 neurons and 2000 bins.
        """
        result = detect_assemblies(spike_counts_with_assemblies, method="pca")
        weights = np.array([pattern.weights for pattern in result.patterns])
        assert weights.shape == (3, 50)
        for cells in (range(0, 5), range(10, 15), range(20, 25)):
            indicator = np.zeros(50)
            indicator[list(cells)] = 1 / np.sqrt(5)
            residual = indicator - weights.T @ (weights @ indicator)
            assert np.linalg.norm(residual) < 0.2

    @pytest.mark.parametrize("shape", [(3000, 40), (3, 8)], ids=["long", "short"])
    def test_nmf_explained_variance_is_variance_along_the_pattern(
        self, rng: np.random.Generator, shape: tuple[int, int]
    ) -> None:
        """w' S w / trace(S): bounded by 1 and free of the W/H scaling."""
        counts = rng.poisson(3, shape).astype(np.float64)
        counts[:, :4] += rng.poisson(3, (shape[0], 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = detect_assemblies(counts, method="nmf", n_components=3, rng=0)
        covariance = np.cov(counts, rowvar=False)
        for pattern in result.patterns:
            weights = pattern.weights
            expected = weights @ covariance @ weights / np.trace(covariance)
            assert_allclose(pattern.explained_variance_ratio, expected, rtol=1e-10)
            assert 0.0 <= pattern.explained_variance_ratio <= 1.0

    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_one_component_explaining_everything_is_at_most_one(
        self, rng: np.random.Generator, method: str
    ) -> None:
        """Three copies of one neuron: one pattern explains all the variance."""
        source = rng.poisson(3, (500, 1)).astype(np.float64)
        counts = np.hstack([source, 2 * source, 3 * source + 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = detect_assemblies(counts, method=method, n_components=1, rng=0)
        (pattern,) = result.patterns
        assert 0.99 < pattern.explained_variance_ratio <= 1.0

    @pytest.fixture
    def assemblies_of_unequal_strength(self) -> np.ndarray:
        """Three 5-cell assemblies whose shared drive is 2, 4 and 6 spikes."""
        rng = np.random.default_rng(3)
        counts = rng.poisson(5, (3000, 40)).astype(np.float64)
        for start, drive in ((0, 2), (10, 4), (20, 6)):
            counts[:, start : start + 5] += rng.poisson(drive, (3000, 1))
        return counts

    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_patterns_are_unit_length_and_ordered_by_explained_variance(
        self, assemblies_of_unequal_strength: np.ndarray, method: str
    ) -> None:
        result = detect_assemblies(assemblies_of_unequal_strength, method=method, rng=0)

        ratios = [pattern.explained_variance_ratio for pattern in result.patterns]
        assert ratios == sorted(ratios, reverse=True)
        assert all(0.0 < ratio < 1.0 for ratio in ratios)
        weights = np.array([pattern.weights for pattern in result.patterns])
        assert_allclose(np.linalg.norm(weights, axis=1), 1.0, atol=1e-12)
        # The strongest assembly (cells 20-24) comes first, then cells 10-14.
        assert result.patterns[0].member_indices.tolist() == list(range(20, 25))
        assert result.patterns[1].member_indices.tolist() == list(range(10, 15))

    @pytest.mark.parametrize("seed", range(5))
    def test_independent_neurons_have_no_significant_dimension(self, seed: int) -> None:
        """Independent Poisson neurons stay below the Marchenko-Pastur bound."""
        counts = np.random.default_rng(seed).poisson(5, (3000, 30)).astype(float)
        with pytest.warns(UserWarning, match="Marchenko-Pastur bound"):
            result = detect_assemblies(counts, rng=42)

        assert result.n_significant == 0
        assert result.patterns == ()

    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_method_is_recorded(
        self, spike_counts_with_assemblies: np.ndarray, method: str
    ) -> None:
        result = detect_assemblies(spike_counts_with_assemblies, method=method, rng=42)

        assert result.method == method
        assert len(result.patterns) == 3
        assert result.activations.shape == (len(spike_counts_with_assemblies), 3)

    def test_patterns_have_correct_shape(self, small_spike_counts: np.ndarray) -> None:
        """Pattern weights should have n_neurons elements."""
        n_neurons = small_spike_counts.shape[1]  # (n_time_bins, n_neurons)
        result = detect_assemblies(small_spike_counts, n_components=2, rng=42)

        for pattern in result.patterns:
            assert len(pattern.weights) == n_neurons

    def test_activations_have_correct_shape(
        self, small_spike_counts: np.ndarray
    ) -> None:
        """Activations are time-first, (n_time_bins, n_assemblies), like the counts."""
        n_time = small_spike_counts.shape[0]  # (n_time_bins, n_neurons)
        result = detect_assemblies(small_spike_counts, n_components=2, rng=42)

        assert result.activations.shape == (n_time, len(result.patterns))

    def test_eigenvalues_descending(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Eigenvalues should be in descending order."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)

        # Check monotonically decreasing
        diffs = np.diff(result.eigenvalues)
        assert np.all(diffs <= 0)

    def test_n_components_fixed(self, small_spike_counts: np.ndarray) -> None:
        """Should respect fixed n_components parameter."""
        n_comp = 3
        result = detect_assemblies(small_spike_counts, n_components=n_comp, rng=42)

        assert len(result.patterns) == n_comp
        assert result.activations.shape[1] == n_comp

    def test_n_components_auto(self, spike_counts_with_assemblies: np.ndarray) -> None:
        """Auto n_components keeps the dimensions above the Marchenko-Pastur bound."""
        result = detect_assemblies(
            spike_counts_with_assemblies, n_components="auto", rng=42
        )

        assert len(result.patterns) == result.n_significant == 3

    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_no_dimension_above_bound_gives_no_patterns(
        self, random_spike_counts: np.ndarray, method: str
    ) -> None:
        """Without correlation structure there is no pattern, not an exploratory one."""
        with pytest.warns(UserWarning, match="Marchenko-Pastur bound"):
            result = detect_assemblies(random_spike_counts, method=method, rng=42)
        assert result.n_significant == 0
        assert len(result.patterns) == 0
        assert result.activations.shape == (random_spike_counts.shape[0], 0)

    def test_z_threshold_affects_membership(self) -> None:
        """Weak participants are members at a low cutoff only; core cells at both."""
        rng = np.random.default_rng(1)
        counts = rng.poisson(1.0, (4000, 60)).astype(np.float64)
        for k, start in enumerate((0, 10, 20)):
            events = rng.choice(4000, 200, replace=False)
            counts[np.ix_(events, range(start, start + 4))] += rng.poisson(
                4.0, (200, 4)
            )
            if k == 0:  # cells 30 and 31 join assembly 0 weakly
                counts[np.ix_(events, [30, 31])] += rng.poisson(1.0, (200, 2))

        def members(z_threshold: float) -> list[list[int]]:
            result = detect_assemblies(counts, z_threshold=z_threshold, rng=0)
            return sorted(p.member_indices.tolist() for p in result.patterns)

        core = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
        assert members(2.5) == core
        assert members(1.0) == [[0, 1, 2, 3, 30, 31], *core[1:]]

    def test_reproducibility_with_random_state(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Same random_state should give same results."""
        result1 = detect_assemblies(spike_counts_with_assemblies, rng=42)
        result2 = detect_assemblies(spike_counts_with_assemblies, rng=42)
        assert len(result1.patterns) == 3

        assert_allclose(result1.activations, result2.activations)
        for p1, p2 in zip(result1.patterns, result2.patterns, strict=True):
            assert_allclose(p1.weights, p2.weights)

    def test_invalid_2d_input(self) -> None:
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="must be 2D"):
            detect_assemblies(np.array([1, 2, 3]))

    def test_too_few_neurons(self, rng: np.random.Generator) -> None:
        """Should raise ValueError for fewer than 3 neurons."""
        spike_counts = rng.poisson(5, (100, 2)).astype(np.float64)  # 2 neurons
        with pytest.raises(ValueError, match="at least 3 neurons"):
            detect_assemblies(spike_counts)

    def test_invalid_method(self, small_spike_counts: np.ndarray) -> None:
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            detect_assemblies(small_spike_counts, method="unknown")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            detect_assemblies(small_spike_counts, algorithm="pca")  # type: ignore[call-arg]

    def test_invalid_n_components(self, small_spike_counts: np.ndarray) -> None:
        """Should raise ValueError for invalid n_components."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            detect_assemblies(small_spike_counts, n_components=0)

        with pytest.raises(ValueError, match="cannot exceed n_neurons"):
            detect_assemblies(small_spike_counts, n_components=100)

    def test_warns_short_recording(self, rng: np.random.Generator) -> None:
        """Warn when n_time_bins < n_neurons, and point at the likely transpose.

        spike_counts is (n_time_bins, n_neurons) -- time-first, matching
        decode_position. Fewer time bins than neurons usually means the matrix is
        transposed, so the warning must name that so users can distinguish it
        from a genuinely short recording.
        """
        spike_counts = rng.poisson(5, (10, 20)).astype(np.float64)  # 10 t, 20 n
        with pytest.warns(UserWarning, match="n_time_bins.*n_neurons") as record:
            detect_assemblies(spike_counts, rng=42)
        msg = str(record[0].message).lower()
        assert "transpos" in msg or "time-first" in msg

    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_constant_neurons_are_excluded(
        self, spike_counts_with_assemblies: np.ndarray, method: str
    ) -> None:
        """A constant neuron gets weight 0 and does not change the detection."""
        counts = spike_counts_with_assemblies.copy()
        counts[:, 7] = 5.0
        keep = np.arange(counts.shape[1]) != 7
        with pytest.warns(
            UserWarning, match=r"same count in every time bin \(units \[7\]\)"
        ):
            result = detect_assemblies(counts, method=method, rng=42)
        without = detect_assemblies(counts[:, keep], method=method, rng=42)

        assert result.n_significant == without.n_significant == 3
        # The threshold uses the 49 varying neurons, not all 50.
        assert result.threshold == marchenko_pastur_threshold(49, counts.shape[0])
        assert_allclose(result.eigenvalues[:49], without.eigenvalues)
        assert result.eigenvalues[49] == 0.0
        for pattern, reference in zip(result.patterns, without.patterns, strict=True):
            assert pattern.weights[7] == 0.0
            assert_allclose(pattern.weights[keep], reference.weights, atol=1e-10)
            assert_allclose(
                pattern.explained_variance_ratio, reference.explained_variance_ratio
            )
            assert 7 not in pattern.member_indices

    def test_too_few_varying_neurons_raise(self, rng: np.random.Generator) -> None:
        spike_counts = rng.poisson(5, (100, 4)).astype(np.float64)
        spike_counts[:, [1, 3]] = 0.0
        with pytest.raises(ValueError, match=r"only 2 of 4 neurons vary") as caught:
            detect_assemblies(spike_counts, rng=42)
        assert "units [1, 3]" in str(caught.value)


class TestPatternSign:
    """Patterns are oriented so an assembly's events give positive activation."""

    @pytest.fixture
    def planted_events(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Three 5-cell assemblies, each co-firing in its own 150 event bins.

        Returns counts ``(n_time_bins, n_neurons)``, the member indices of each
        assembly and a ``(3, n_time_bins)`` mask of each assembly's events.
        Without a sign convention, ICA and PCA patterns may have negative
        largest weights.
        """
        rng = np.random.default_rng(1)
        n_time, n_neurons = 3000, 30
        counts = rng.poisson(1.0, (n_time, n_neurons)).astype(np.float64)
        members = [np.arange(0, 5), np.arange(10, 15), np.arange(20, 25)]
        events = np.zeros((3, n_time), dtype=bool)
        for k, cells in enumerate(members):
            events[k, rng.choice(n_time, 150, replace=False)] = True
            counts[np.ix_(events[k], cells)] += rng.poisson(
                4.0, (int(events[k].sum()), cells.size)
            )
        return counts, members, events

    @pytest.mark.parametrize("method", ["ica", "pca"])
    def test_largest_weight_positive_and_events_activate(
        self,
        planted_events: tuple[np.ndarray, list[np.ndarray], np.ndarray],
        method: str,
    ) -> None:
        counts, members, events = planted_events
        result = detect_assemblies(counts, method=method, rng=0)
        assert result.n_significant == 3

        for k, pattern in enumerate(result.patterns):
            largest = int(np.argmax(np.abs(pattern.weights)))
            assert pattern.weights[largest] > 0
            # The assembly holding the largest weight is active in its events,
            # above the activation-event cutoff of 5 (van de Ven et al., 2016),
            # and near 0 otherwise.
            (assembly,) = [a for a, cells in enumerate(members) if largest in cells]
            in_event = events[assembly]
            activation = assembly_activation(counts, pattern)
            assert activation[in_event].mean() > 5.0
            assert abs(activation[~in_event].mean()) < 1.0
            assert result.activations[in_event, k].mean() > 5.0

    def test_ica_members_have_positive_weights(
        self, planted_events: tuple[np.ndarray, list[np.ndarray], np.ndarray]
    ) -> None:
        counts, members, _ = planted_events
        result = detect_assemblies(counts, method="ica", rng=0)
        recovered = sorted(p.member_indices.tolist() for p in result.patterns)
        assert recovered == [cells.tolist() for cells in members]
        for pattern in result.patterns:
            assert np.all(pattern.weights[pattern.member_indices] > 0)

    @pytest.mark.parametrize("method", ["ica", "pca", "nmf"])
    def test_result_activations_match_assembly_activation(
        self,
        planted_events: tuple[np.ndarray, list[np.ndarray], np.ndarray],
        method: str,
    ) -> None:
        """Every method reports the same activation strength as assembly_activation."""
        counts, _, _ = planted_events
        result = detect_assemblies(counts, method=method, rng=0)
        for k, pattern in enumerate(result.patterns):
            assert_allclose(
                result.activations[:, k],
                assembly_activation(counts, pattern),
                rtol=1e-10,
                atol=1e-9,
            )


# =============================================================================
# Tests for assembly_activation
# =============================================================================


class TestAssemblyActivation:
    """Tests for assembly_activation function."""

    def test_returns_correct_shape(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """Activation should have n_time_bins elements."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        activation = assembly_activation(spike_counts_with_assemblies, pattern)

        assert activation.shape == (spike_counts_with_assemblies.shape[0],)

    @pytest.fixture
    def coactivation_counts(self) -> tuple[np.ndarray, AssemblyPattern]:
        """Five members with weight 1/sqrt(5) among 30 neurons, and planted bins.

        Bins 100-119 hold a burst of member 0 alone; bins 200-219 hold the
        same burst in all five members together.
        """
        rng = np.random.default_rng(4)
        counts = rng.poisson(1.0, (2000, 30)).astype(np.float64)
        counts[100:120, 0] += 6
        counts[200:220, :5] += 6
        weights = np.r_[np.full(5, 1 / np.sqrt(5)), np.zeros(25)]
        return counts, AssemblyPattern(weights, np.arange(5), 0.2)

    def test_lone_member_burst_is_not_coactivation(self, coactivation_counts) -> None:
        """Only members firing together raise the projector activation strength.

        A linear projection w . z(t) scores a burst of one member as a strong
        assembly event; with the diagonal of the projector removed, that bin
        scores near 0 and the joint burst scores far above it.
        """
        counts, pattern = coactivation_counts
        activation = assembly_activation(counts, pattern)
        lone, together = activation[100:120].mean(), activation[200:220].mean()
        baseline = np.delete(activation, np.r_[100:120, 200:220])
        # The lone burst averages 0 up to noise (about 1 for 20 bins).
        assert abs(lone) < 2.5
        assert together > 50.0
        assert together > baseline.mean() + 20 * baseline.std()

    def test_activation_is_higher_in_planted_coactivation_bins(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """A pattern's activation tracks the shared drive of its own members."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        for pattern in result.patterns:
            activation = assembly_activation(spike_counts_with_assemblies, pattern)
            members = pattern.member_indices
            drive = spike_counts_with_assemblies[:, members].mean(axis=1)
            high = drive > np.quantile(drive, 0.9)
            assert activation[high].mean() > 3 * activation[~high].mean() > 0

    def test_sign_flip_gives_identical_activation(self, coactivation_counts) -> None:
        counts, pattern = coactivation_counts
        flipped = AssemblyPattern(-pattern.weights, pattern.member_indices, 0.2)
        assert_allclose(
            assembly_activation(counts, flipped),
            assembly_activation(counts, pattern),
            rtol=0,
            atol=0,
        )

    def test_weights_are_used_at_unit_length(self, coactivation_counts) -> None:
        counts, pattern = coactivation_counts
        scaled = AssemblyPattern(3 * pattern.weights, pattern.member_indices, 0.2)
        assert_allclose(
            assembly_activation(counts, scaled), assembly_activation(counts, pattern)
        )

    def test_independent_neurons_average_zero(self, rng: np.random.Generator) -> None:
        """Mean R(t) is w'Cw - 1, about 0 for independent neurons."""
        counts = rng.poisson(2.0, (5000, 10)).astype(np.float64)
        weights = rng.standard_normal(10)
        pattern = AssemblyPattern(weights, np.array([0]), 0.1)
        activation = assembly_activation(counts, pattern)
        unit = weights / np.linalg.norm(weights)
        expected = unit @ np.corrcoef(counts, rowvar=False) @ unit - 1
        assert_allclose(activation.mean(), expected, atol=1e-12)
        assert abs(activation.mean()) < 4 * np.sqrt(2 / 5000)

    def test_constant_member_warns_and_contributes_nothing(
        self, coactivation_counts
    ) -> None:
        counts, pattern = coactivation_counts
        silent = counts.copy()
        silent[:, 2] = 0.0
        with pytest.warns(UserWarning, match=r"units \[2\] have nonzero weight"):
            activation = assembly_activation(silent, pattern)
        assert np.all(np.isfinite(activation))

    def test_all_zero_weights_raise(self, coactivation_counts) -> None:
        counts, _ = coactivation_counts
        empty = AssemblyPattern(np.zeros(30), np.array([], dtype=np.int64), 0.0)
        with pytest.raises(ValueError, match="weights are all 0"):
            assembly_activation(counts, empty)

    def test_mismatched_neurons_raises_error(
        self, spike_counts_with_assemblies: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Should raise ValueError if pattern has wrong number of neurons."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        # Create spike counts with different number of neurons
        wrong_counts = rng.poisson(5, (20, 100)).astype(np.float64)

        with pytest.raises(ValueError, match=r"neurons.*weights.*Must match"):
            assembly_activation(wrong_counts, pattern)

    def test_invalid_2d_input(self, spike_counts_with_assemblies: np.ndarray) -> None:
        """Should raise ValueError for non-2D input."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        with pytest.raises(ValueError, match="must be 2D"):
            assembly_activation(np.array([1, 2, 3]), pattern)

    def test_z_score_input_option(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """z_score_input=False should skip input normalization."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        from scipy.stats import zscore

        act_zscore = assembly_activation(
            spike_counts_with_assemblies, pattern, z_score_input=True
        )
        act_raw = assembly_activation(
            spike_counts_with_assemblies, pattern, z_score_input=False
        )
        prenormalized = assembly_activation(
            zscore(spike_counts_with_assemblies, axis=0), pattern, z_score_input=False
        )

        assert not np.allclose(act_zscore, act_raw)
        assert_allclose(prenormalized, act_zscore, atol=1e-10)


# =============================================================================
# Tests for pairwise_correlations
# =============================================================================


class TestPairwiseCorrelations:
    """Tests for pairwise_correlations function."""

    def test_returns_correct_length(self, small_spike_counts: np.ndarray) -> None:
        """Should return n*(n-1)/2 correlation values."""
        n_neurons = small_spike_counts.shape[1]  # (n_time_bins, n_neurons)
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
        n_neurons = small_spike_counts.shape[1]  # (n_time_bins, n_neurons)
        corr = pairwise_correlations(small_spike_counts)

        # Length should be n*(n-1)/2, not n*n
        assert len(corr) == n_neurons * (n_neurons - 1) // 2

    def test_invalid_2d_input(self) -> None:
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="must be 2D"):
            pairwise_correlations(np.array([1, 2, 3]))

    def test_silent_neuron_pairs_are_nan(self, small_spike_counts: np.ndarray) -> None:
        """A neuron with one count in every bin has no correlation, not r = 0."""
        counts = small_spike_counts.copy()
        counts[:, 3] = 0.0
        with pytest.warns(UserWarning, match=r"neuron columns \[3\]"):
            corr = pairwise_correlations(counts)
        first, second = np.triu_indices(counts.shape[1], k=1)
        involves = (first == 3) | (second == 3)
        assert np.all(np.isnan(corr[involves]))
        assert np.all(np.isfinite(corr[~involves]))
        keep = np.arange(counts.shape[1]) != 3
        assert_allclose(corr[~involves], pairwise_correlations(counts[:, keep]))


# =============================================================================
# Tests for assembly_reactivation
# =============================================================================


class TestAssemblyReactivationBasics:
    """Mean activation strength after the task minus before it."""

    @staticmethod
    def _period(seed: int, n_time: int, drive: float) -> np.ndarray:
        """Members 0-4 share a drive of ``drive`` spikes on top of 1 Hz noise."""
        rng = np.random.default_rng(seed)
        counts = rng.poisson(1.0, (n_time, 20)).astype(np.float64)
        counts[:, :5] += 2 * rng.poisson(drive, (n_time, 1))
        return counts

    @pytest.fixture
    def pattern(self) -> AssemblyPattern:
        return AssemblyPattern(
            np.r_[np.full(5, 1 / np.sqrt(5)), np.zeros(15)], np.arange(5), 0.2
        )

    def test_identical_periods_give_zero_difference(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        for pattern in result.patterns:
            comparison = assembly_reactivation(
                spike_counts_with_assemblies, spike_counts_with_assemblies, pattern
            )
            assert comparison.mean_activation_difference == 0.0
            # About 1% of a period's bins exceed its own 99th percentile.
            assert comparison.fraction_above_control_threshold <= 0.01

    def test_equals_difference_of_mean_assembly_activation(self, pattern) -> None:
        pre, post = self._period(0, 2000, 0.0), self._period(1, 1500, 0.3)
        expected = (
            assembly_activation(post, pattern).mean()
            - assembly_activation(pre, pattern).mean()
        )
        assert_allclose(
            assembly_reactivation(pre, post, pattern).mean_activation_difference,
            expected,
            rtol=1e-12,
        )

    def test_coactivation_after_but_not_before(self, pattern) -> None:
        """Members co-firing after the task only: positive difference."""
        pre, post = self._period(1, 3000, 0.0), self._period(2, 3000, 0.5)
        result = assembly_reactivation(pre, post, pattern)
        noise = np.sqrt(2 / 3000)
        assert abs(result.control_mean_activation) < 4 * noise
        assert result.mean_activation_difference > 10 * noise
        assert result.fraction_above_control_threshold > 0.1

    def test_neuron_constant_in_one_period_is_excluded(self, pattern) -> None:
        pre, post = self._period(0, 2000, 0.5), self._period(1, 2000, 0.5)
        pre[:, 3] = 0.0  # member 3 silent before the task
        with pytest.warns(
            UserWarning, match=r"units \[3\] .* control_counts.* excluded"
        ):
            result = assembly_reactivation(pre, post, pattern)
        keep = np.arange(20) != 3
        reduced = AssemblyPattern(pattern.weights[keep], np.arange(4), 0.2)
        expected = assembly_reactivation(pre[:, keep], post[:, keep], reduced)
        assert_allclose(
            list(result.summary().values()),
            list(expected.summary().values()),
            rtol=1e-12,
        )

    def test_mismatched_neuron_count_raises(self, pattern) -> None:
        with pytest.raises(ValueError, match="All must match"):
            assembly_reactivation(
                self._period(0, 100, 0.1), self._period(1, 100, 0.1)[:, :19], pattern
            )

    def test_summary_is_flat_scalars(self, pattern) -> None:
        result = assembly_reactivation(
            self._period(0, 500, 0.1), self._period(1, 500, 0.4), pattern
        )
        summary = result.summary()
        assert next(iter(summary)) == "mean_activation_difference"
        assert all(np.isscalar(value) for value in summary.values())


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

    def test_ev_bounds(self, rng: np.random.Generator) -> None:
        """EV should be in [0, 1]."""
        corr1 = rng.uniform(-1, 1, 50)
        corr2 = rng.uniform(-1, 1, 50)

        control = rng.uniform(-1, 1, 50)
        result = explained_variance_reactivation(
            corr1, corr2, control_correlations=control
        )

        assert 0 <= result.explained_variance <= 1
        assert 0 <= result.reversed_explained_variance <= 1

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

    def test_too_few_valid_pairs_returns_nan(self) -> None:
        """Fewer than 3 valid pairs after NaN removal should yield NaN, not 0.0.

        With 0 or 1 valid pair the correlation is undefined; previously
        np.corrcoef returned NaN which was coerced to 0.0, masquerading as a
        confident "no reactivation". The statistics must read as undefined.
        """
        # 5 raw pairs (passes the n_pairs >= 3 guard) but only 1 finite pair
        # survives the joint NaN mask.
        corr1 = np.array([0.1, np.nan, np.nan, np.nan, np.nan])
        corr2 = np.array([0.1, np.nan, np.nan, np.nan, np.nan])

        with pytest.warns(
            UserWarning, match="valid pairs after pair selection and removing NaN"
        ):
            result = explained_variance_reactivation(corr1, corr2)

        assert np.isnan(result.explained_variance)
        assert np.isnan(result.reversed_explained_variance)
        assert np.isnan(result.partial_correlation)
        # n_pairs reflects the actual count of valid pairs.
        assert result.n_pairs == 1


class TestExplainedVarianceUndefined:
    """Undefined correlation terms give NaN, never a confident EV of 0."""

    @pytest.fixture
    def periods(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(8)
        n_time, n_neurons = 3000, 20
        out = {}
        for name, drive in (("template", 0.4), ("match", 0.4)):
            counts = rng.poisson(1.0, (n_time, n_neurons)).astype(np.float64)
            counts[:, :6] += 2 * rng.poisson(drive, (n_time, 1))
            counts[:, 10:14] += 2 * rng.poisson(drive, (n_time, 1))
            out[name] = counts
        return out

    def test_silent_member_drops_its_pairs(self, periods) -> None:
        """A member silent in the match period leaves the other pairs' EV intact."""
        match = periods["match"].copy()
        match[:, 2] = 0.0
        with pytest.warns(UserWarning, match="undefined"):
            match_correlations = pairwise_correlations(match)
        result = explained_variance_reactivation(
            pairwise_correlations(periods["template"]), match_correlations
        )
        keep = np.arange(20) != 2
        without = explained_variance_reactivation(
            pairwise_correlations(periods["template"][:, keep]),
            pairwise_correlations(match[:, keep]),
        )
        assert result.n_pairs == 19 * 18 // 2
        assert_allclose(result.explained_variance, without.explained_variance)
        assert result.explained_variance > 0.8

    def test_constant_correlation_vector_is_nan(self) -> None:
        template = np.linspace(-0.5, 0.5, 10)
        with pytest.warns(UserWarning, match="EV is undefined"):
            result = explained_variance_reactivation(template, np.full(10, 0.2))
        assert np.isnan(result.explained_variance)
        assert np.isnan(result.partial_correlation)

    def test_control_equal_to_template_is_nan(self) -> None:
        rng = np.random.default_rng(2)
        template = rng.uniform(-0.5, 0.5, 30)
        match = template + 0.1 * rng.standard_normal(30)
        with pytest.warns(UserWarning, match="EV is undefined"):
            result = explained_variance_reactivation(
                template, match, control_correlations=template
            )
        assert np.isnan(result.explained_variance)
        assert np.isnan(result.partial_correlation)

    def test_infinite_correlations_raise(self) -> None:
        corr = np.linspace(-0.5, 0.5, 10)
        broken = corr.copy()
        broken[4] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            explained_variance_reactivation(corr, broken)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAssemblyWorkflow:
    """Integration tests for complete assembly detection workflow."""

    def test_full_workflow(self, spike_counts_with_assemblies: np.ndarray) -> None:
        """Test complete detection -> activation -> reactivation workflow."""
        # Split data into "behavior" and "rest" periods (by time = axis 0)
        n_time = spike_counts_with_assemblies.shape[0]  # (n_time_bins, n_neurons)
        counts_behavior = spike_counts_with_assemblies[: n_time // 2, :]
        counts_rest = spike_counts_with_assemblies[n_time // 2 :, :]

        # Detect assemblies during behavior
        result = detect_assemblies(counts_behavior, rng=42)
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
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            pattern.weights = np.zeros(len(pattern.weights))  # type: ignore[misc]

    def test_result_immutability(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        """AssemblyDetectionResult should be immutable (frozen dataclass)."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)

        with pytest.raises(AttributeError):
            result.n_significant = 100  # type: ignore[misc]


class TestResultInvariants:
    """Results refuse field values that would misdescribe the assemblies."""

    def test_patterns_are_an_immutable_tuple(
        self, spike_counts_with_assemblies: np.ndarray
    ) -> None:
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        assert isinstance(result.patterns, tuple)
        with pytest.raises(AttributeError):
            result.patterns.append(result.patterns[0])  # type: ignore[attr-defined]

    @pytest.fixture
    def fields(self) -> dict:
        pattern = AssemblyPattern(np.ones(3) / np.sqrt(3), np.arange(3), 0.5)
        return {
            "patterns": (pattern,),
            "activations": np.zeros((10, 1)),
            "method": "ica",
            "n_significant": 1,
            "eigenvalues": np.array([2.0, 0.5, 0.5]),
            "threshold": 1.5,
        }

    @pytest.mark.parametrize(
        ("override", "error", "match"),
        [
            ({"activations": np.zeros((10, 2))}, ValueError, "n_assemblies"),
            ({"activations": np.zeros((1, 10))}, ValueError, "n_assemblies"),
            ({"method": "kmeans"}, ValueError, "method must be one of"),
            ({"unit_ids": np.array([4, 5, 6])}, ValueError, "carry the result's"),
            ({"patterns": ("pattern",)}, TypeError, "AssemblyPattern objects"),
        ],
        ids=["n-assemblies", "neuron-first", "method", "unit-ids", "type"],
    )
    def test_detection_result_refuses_inconsistent_fields(
        self, fields: dict, override: dict, error: type, match: str
    ) -> None:
        with pytest.raises(error, match=match):
            AssemblyDetectionResult(**{**fields, **override})

    @pytest.mark.parametrize(
        ("members", "match"),
        [
            (np.array([-1, 0]), "in increasing order"),
            (np.array([0, 0]), "in increasing order"),
            (np.array([1, 0]), "in increasing order"),
            (np.array([0, 3]), r"in \[0, 3\)"),
            (np.array([0.0, 1.0]), "integer column positions"),
            (np.array([True, False, True]), "integer column positions"),
        ],
        ids=["negative", "duplicate", "unsorted", "out-of-range", "float", "mask"],
    )
    def test_pattern_refuses_invalid_member_indices(
        self, members: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            AssemblyPattern(np.ones(3), members, 0.5)

    @pytest.mark.parametrize("ratio", [7.0, -0.1, np.nan])
    def test_pattern_refuses_ratio_outside_unit_interval(self, ratio: float) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            AssemblyPattern(np.ones(3), np.array([0]), ratio)

    def test_pattern_accepts_no_members(self) -> None:
        pattern = AssemblyPattern(np.ones(3), np.array([]), 0.0)
        assert pattern.member_indices.dtype == np.int64
        assert pattern.member_unit_ids.size == 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_neurons(self, rng: np.random.Generator) -> None:
        """Should work with minimum 3 neurons."""
        spike_counts = rng.poisson(5, (500, 3)).astype(np.float64)  # 3 neurons
        spike_counts[:, 1] += spike_counts[:, 0]

        result = detect_assemblies(spike_counts, rng=42)

        assert len(result.patterns) == result.n_significant == 1

    def test_single_time_bin_activation(
        self, spike_counts_with_assemblies: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Activation should work even with few time bins."""
        result = detect_assemblies(spike_counts_with_assemblies, rng=42)
        pattern = result.patterns[0]

        # Few time bins, matching the pattern's 50 neurons: (n_time_bins, n_neurons)
        single_bin = rng.poisson(5, (10, 50)).astype(np.float64)

        activation = assembly_activation(single_bin, pattern)
        assert len(activation) == 10

    def test_all_zero_spike_counts(self) -> None:
        """All-silent counts have no assemblies to find: raise, not zero patterns."""
        spike_counts = np.zeros((10, 100), dtype=np.float64)

        with pytest.raises(ValueError, match="only 0 of 100 neurons vary"):
            detect_assemblies(spike_counts, rng=42)

    def test_explained_variance_identical_correlations(self) -> None:
        """EV should be 1 for identical correlation vectors."""
        corr = np.linspace(-1, 1, 100)

        result = explained_variance_reactivation(corr, corr)

        assert_allclose(result.explained_variance, 1.0)
        assert np.isnan(result.reversed_explained_variance)  # no control


# =============================================================================
# Reactivation correctness regressions
# =============================================================================


class TestExplainedVarianceControlAndREV:
    """EV/REV swap pre- and post-behavior periods, retaining the template."""

    def test_reverse_ev_matches_residual_regression_and_time_reversal(self):
        template = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]) / 10
        pre = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0]) / 10
        post = 0.8 * pre + np.array([0.0, 1.0, -1.0, -1.0, 1.0, 0.0]) / 100
        design = np.column_stack([np.ones(post.size), post])
        residuals = np.column_stack([template, pre])
        residuals -= design @ np.linalg.lstsq(design, residuals, rcond=None)[0]
        expected = np.corrcoef(residuals.T)[0, 1] ** 2
        result = explained_variance_reactivation(
            template, post, control_correlations=pre
        )
        reversed_result = explained_variance_reactivation(
            template, pre, control_correlations=post
        )
        assert_allclose(result.reversed_explained_variance, expected, atol=1e-12)
        assert_allclose(
            result.reversed_explained_variance,
            reversed_result.explained_variance,
            atol=1e-12,
        )
        assert_allclose(
            result.explained_variance,
            reversed_result.reversed_explained_variance,
            atol=1e-12,
        )

    def test_explained_variance_rev_differs_from_ev_with_control(
        self,
        corr_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """With a control, REV reverses the two rest periods.

        On a reactivation-positive triplet (template->match correlation
        stronger than template->control after partialling) EV should exceed
        REV, and the two must not be equal.
        """
        template, match, control = corr_triplet

        result = explained_variance_reactivation(
            template, match, control_correlations=control
        )

        assert result.reversed_explained_variance != result.explained_variance
        assert result.explained_variance > result.reversed_explained_variance

    def test_explained_variance_uses_control(
        self,
        corr_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """A control correlated with match must lower EV below r_tm**2."""
        template, match, control = corr_triplet

        with_control = explained_variance_reactivation(
            template, match, control_correlations=control
        )
        no_control = explained_variance_reactivation(template, match)

        # No-control EV is the plain r_tm**2; partialling out a control
        # that is itself correlated with match strictly reduces EV.
        assert with_control.explained_variance < no_control.explained_variance

    def test_explained_variance_without_control_has_no_rev(self) -> None:
        """Without a control, EV is r_tm**2 and REV is undefined (NaN)."""
        rng = np.random.default_rng(11)
        template = rng.standard_normal(80)
        match = 0.7 * template + 0.7 * rng.standard_normal(80)

        result = explained_variance_reactivation(template, match)

        r_tm = np.corrcoef(template, match)[0, 1]
        assert_allclose(result.explained_variance, r_tm**2)
        assert_allclose(result.partial_correlation, r_tm)
        assert np.isnan(result.reversed_explained_variance)


class TestAssemblyReactivationNoCoactivation:
    """A match period without co-activation has a mean activation near 0."""

    def test_match_without_shared_drive_is_near_zero(
        self,
        template_counts: np.ndarray,
        assembly_pattern: AssemblyPattern,
    ) -> None:
        rng = np.random.default_rng(3)
        # Per-neuron baseline over time (axis 0 = time in (n_time_bins, n_neurons)).
        baseline = template_counts.mean(axis=0, keepdims=True)
        match_counts = np.broadcast_to(baseline, template_counts.shape).copy()
        match_counts += 0.05 * rng.standard_normal(template_counts.shape)

        result = assembly_reactivation(template_counts, match_counts, assembly_pattern)

        noise_floor = np.sqrt(2 / template_counts.shape[0])
        assert abs(result.match_mean_activation) < 4 * noise_floor
        assert_allclose(
            result.control_mean_activation,
            assembly_activation(template_counts, assembly_pattern).mean(),
        )


class TestDetectAssembliesShortRecording:
    @pytest.mark.parametrize("method", ["pca", "ica", "nmf"])
    def test_n_time_bins_below_n_components_does_not_crash(
        self, rng: np.random.Generator, method: str
    ) -> None:
        """n_time_bins < n_components <= n_neurons must clamp, not IndexError."""
        # 8 neurons, 3 time bins, request 5 components: (n_time_bins, n_neurons).
        spike_counts = rng.poisson(5, (3, 8)).astype(np.float64)
        with pytest.warns(UserWarning, match="achievable rank"):
            result = detect_assemblies(
                spike_counts,
                method=method,
                n_components=5,
                rng=0,
            )
        # Clamped to min(n_neurons, n_time_bins) == 3 patterns; no crash.
        assert len(result.patterns) <= 3
        assert result.activations.shape[0] == 3  # n_time_bins preserved


@pytest.mark.parametrize("n_components", [2, 3])
def test_ica_whitens_scores_and_recovers_independent_sources(n_components):
    """ICA recovers sources with unequal PCA variances, including a null PC."""
    from scipy.stats import zscore

    rng = np.random.default_rng(42)
    sources = rng.poisson([2.0, 7.0], size=(20000, 2))
    counts = sources @ np.array([[1, 2, 3, 0], [3, 1, 0, 2]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "1 of the 3 requested components")
        result = detect_assemblies(
            counts, method="ica", n_components=n_components, rng=0
        )
    standardized = zscore(counts, axis=0)
    weights = np.array([pattern.weights for pattern in result.patterns])
    # Unit-length patterns (Lopes-dos-Santos et al., 2013), like PCA's.
    assert_allclose(np.linalg.norm(weights, axis=1), 1.0, atol=1e-12)
    # Projected onto the patterns, the counts give uncorrelated sources.
    recovered = weights @ standardized.T
    assert_allclose(np.corrcoef(recovered), np.eye(2), atol=1e-10)
    recovery_correlations = np.abs(np.corrcoef(sources.T, recovered)[:2, 2:])
    assert np.all(recovery_correlations.max(axis=1) > 0.98)

    # Fit the reconstruction independently, then measure the variance each
    # recovered source contributes in neuron space.
    reconstruction_coeffs = np.linalg.lstsq(recovered.T, standardized, rcond=None)[0]
    total_variance = standardized.var(axis=0).sum()
    for i, pattern in enumerate(result.patterns[:2]):
        reconstructed = np.outer(recovered[i], reconstruction_coeffs[i])
        expected = reconstructed.var(axis=0).sum() / total_variance
        assert_allclose(pattern.explained_variance_ratio, expected, atol=1e-12)
    assert_allclose(sum(p.explained_variance_ratio for p in result.patterns), 1.0)
    # The counts span two dimensions; a third requested component has no
    # variance and is dropped.
    assert len(result.patterns) == 2


class TestOneCountConvention:
    """bin_spikes_in_time's default (time, neuron) output feeds both consumers.

    Regression: assemblies used to expect (neuron, time) while decode used
    (time, neuron), so the decode-oriented output silently produced garbage
    assemblies. Both now consume the one time-first convention -- the default
    bin_spikes_in_time output drops straight into detect_assemblies, no
    transpose.
    """

    def test_default_bin_output_feeds_detect_assemblies(self) -> None:
        from neurospatial.decoding import bin_spikes_in_time

        rng = np.random.default_rng(0)
        n_neurons = 6
        spike_trains = [np.sort(rng.uniform(0.0, 100.0, 500)) for _ in range(n_neurons)]
        counts, _centers = bin_spikes_in_time(spike_trains, dt=0.5)  # default orient
        # default orient='time_x_neuron' -> (n_time_bins, n_neurons)
        assert counts.shape[1] == n_neurons

        result = detect_assemblies(counts.astype(np.float64), n_components=2, rng=0)
        # Orientation consumed correctly: weights are sized by neuron count.
        assert len(result.patterns) == 2
        for pattern in result.patterns:
            assert len(pattern.weights) == n_neurons


class TestCountsMustBeObserved:
    """A NaN bin is missing, not silent: every assembly entry point refuses it."""

    @pytest.fixture
    def counts(self) -> np.ndarray:
        rng = np.random.default_rng(5)
        counts = rng.poisson(1.0, (600, 8)).astype(np.float64)
        counts[:, :3] += 3 * rng.poisson(0.3, (600, 1))
        return counts

    @pytest.fixture
    def pattern(self) -> AssemblyPattern:
        weights = np.r_[np.full(3, 1 / np.sqrt(3)), np.zeros(5)]
        return AssemblyPattern(weights, np.array([0, 1, 2]), 0.3)

    def _calls(self, pattern):
        return {
            "detect_assemblies": lambda c: detect_assemblies(c, rng=0),
            "assembly_activation": lambda c: assembly_activation(c, pattern),
            "pairwise_correlations": pairwise_correlations,
            "assembly_reactivation (control)": lambda c: assembly_reactivation(
                c, np.nan_to_num(c), pattern
            ),
            "assembly_reactivation (match)": lambda c: assembly_reactivation(
                np.nan_to_num(c), c, pattern
            ),
        }

    @pytest.mark.parametrize(
        "name",
        [
            "detect_assemblies",
            "assembly_activation",
            "pairwise_correlations",
            "assembly_reactivation (control)",
            "assembly_reactivation (match)",
        ],
    )
    def test_nan_bins_raise_with_columns_and_bin_count(
        self, counts: np.ndarray, pattern: AssemblyPattern, name: str
    ) -> None:
        gapped = counts.copy()
        gapped[[10, 11], 2] = np.nan
        gapped[40, 5] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite") as caught:
            self._calls(pattern)[name](gapped)
        message = str(caught.value)
        assert "in 3 of 600 time bins, in neuron columns [2, 5]" in message
        assert "drop the gap bins" in message

    @pytest.mark.parametrize(
        "name",
        ["detect_assemblies", "assembly_activation", "pairwise_correlations"],
    )
    def test_negative_counts_raise(
        self, counts: np.ndarray, pattern: AssemblyPattern, name: str
    ) -> None:
        negative = counts.copy()
        negative[3, 4] = -1.0
        with pytest.raises(
            ValueError, match=r"negative values in neuron columns \[4\]"
        ):
            self._calls(pattern)[name](negative)

    def test_prenormalized_activation_input_may_be_negative(
        self, counts: np.ndarray, pattern: AssemblyPattern
    ) -> None:
        from scipy.stats import zscore

        activation = assembly_activation(
            zscore(counts, axis=0), pattern, z_score_input=False
        )
        assert np.all(np.isfinite(activation))
