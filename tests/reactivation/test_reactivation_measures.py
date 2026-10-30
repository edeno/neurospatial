"""Published reactivation measures computed exactly as the papers define them.

- ``assembly_reactivation``: mean activation strength after minus before
  (van de Ven et al. 2016; Peyrache et al. 2009) and the share of match bins
  above the control's 99th percentile (Peyrache et al. 2010).
- ``explained_variance_reactivation``: Kudrimoti et al. (1999) partial
  correlation over selected pairs (different tetrodes or regions), several
  control periods and the partial-correlation t test.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from neurospatial.reactivation import (
    AssemblyPattern,
    assembly_reactivation,
    explained_variance_reactivation,
    pairwise_correlations,
)

N_UNITS = 12
MEMBERS = [2, 5, 7]


def _counts(seed: int, n_time: int, drive: float) -> np.ndarray:
    """Poisson counts, shape (n_time, N_UNITS); ``MEMBERS`` share a drive."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, (n_time, N_UNITS)).astype(np.float64)
    counts[:, MEMBERS] += 2 * rng.poisson(drive, (n_time, 1))
    return counts


@pytest.fixture(scope="module")
def pattern() -> AssemblyPattern:
    weights = np.full(N_UNITS, 0.1)
    weights[MEMBERS] = 1.0
    return AssemblyPattern(weights, np.array(MEMBERS), 0.3)


def _hand_activation(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """R(t) = sum_{i != j} w_i w_j z_i(t) z_j(t), counts z-scored per period."""
    z = (counts - counts.mean(axis=0)) / counts.std(axis=0)
    w = weights / np.linalg.norm(weights)
    outer = np.outer(w, w)
    np.fill_diagonal(outer, 0.0)
    return np.einsum("ti,ij,tj->t", z, outer, z)


# =============================================================================
# assembly_reactivation
# =============================================================================


class TestAssemblyReactivation:
    def test_post_minus_pre_and_percentile_share_match_hand_calculation(
        self, pattern
    ) -> None:
        pre, post = _counts(0, 3000, 0.0), _counts(1, 2500, 0.4)
        result = assembly_reactivation(pre, post, pattern)

        activation_pre = _hand_activation(pre, pattern.weights)
        activation_post = _hand_activation(post, pattern.weights)
        threshold = np.percentile(activation_pre, 99)
        assert_allclose(
            result.control_mean_activation, activation_pre.mean(), rtol=1e-10
        )
        assert_allclose(
            result.match_mean_activation, activation_post.mean(), rtol=1e-10
        )
        assert_allclose(
            result.mean_activation_difference,
            activation_post.mean() - activation_pre.mean(),
            rtol=1e-10,
        )
        assert_allclose(result.control_threshold, threshold, rtol=1e-10)
        assert result.fraction_above_control_threshold == np.mean(
            activation_post > threshold
        )
        assert (result.n_control_bins, result.n_match_bins) == (3000, 2500)
        assert result.percentile == 99.0
        # The members co-fire only after the task.
        assert result.mean_activation_difference > 0.5
        assert result.fraction_above_control_threshold > 0.05

    def test_mean_activation_is_gamma_minus_one(self, pattern) -> None:
        """Peyrache et al. (2010): the mean of R(t) over an epoch is γ − 1."""
        pre, post = _counts(2, 2000, 0.1), _counts(3, 2000, 0.5)
        w = pattern.weights / np.linalg.norm(pattern.weights)
        gamma = {
            name: w @ np.corrcoef(counts, rowvar=False) @ w
            for name, counts in (("pre", pre), ("post", post))
        }
        result = assembly_reactivation(pre, post, pattern)
        assert_allclose(result.control_mean_activation, gamma["pre"] - 1, atol=1e-12)
        assert_allclose(result.match_mean_activation, gamma["post"] - 1, atol=1e-12)
        assert_allclose(
            result.mean_activation_difference,
            gamma["post"] - gamma["pre"],
            atol=1e-12,
        )

    def test_same_process_before_and_after_gives_no_reactivation(self, pattern) -> None:
        """Without reactivation the difference is ~0 and ~1% of bins pass the 99th."""
        n_time = 40_000
        pre, post = _counts(4, n_time, 0.3), _counts(5, n_time, 0.3)
        result = assembly_reactivation(pre, post, pattern)
        # Bins are independent, so the difference of means has standard error
        # sqrt(var_pre / n + var_post / n).
        standard_error = np.sqrt(
            sum(_hand_activation(c, pattern.weights).var() for c in (pre, post))
            / n_time
        )
        assert abs(result.mean_activation_difference) < 5 * standard_error
        binomial_se = np.sqrt(0.01 * 0.99 / n_time)
        assert abs(result.fraction_above_control_threshold - 0.01) < 5 * binomial_se

    def test_gain_change_is_not_reactivation(self, pattern) -> None:
        """Each period is z-scored on its own: scaling post counts changes nothing."""
        pre, post = _counts(6, 2000, 0.1), _counts(7, 2000, 0.4)
        base = assembly_reactivation(pre, post, pattern)
        scaled = assembly_reactivation(pre, 3.0 * post, pattern)
        assert_allclose(
            list(scaled.summary().values()), list(base.summary().values()), rtol=1e-12
        )

    def test_percentile_sets_the_threshold(self, pattern) -> None:
        pre, post = _counts(8, 2000, 0.0), _counts(9, 2000, 0.3)
        result = assembly_reactivation(pre, post, pattern, percentile=95.0)
        activation_pre = _hand_activation(pre, pattern.weights)
        assert_allclose(
            result.control_threshold, np.percentile(activation_pre, 95), rtol=1e-10
        )

    @pytest.mark.parametrize("percentile", [0.0, 100.0, 150.0])
    def test_percentile_outside_open_interval_raises(self, pattern, percentile) -> None:
        counts = _counts(0, 200, 0.2)
        with pytest.raises(ValueError, match=r"percentile must be in \(0, 100\)"):
            assembly_reactivation(counts, counts, pattern, percentile=percentile)

    def test_all_weighted_units_constant_is_nan(self) -> None:
        pattern = AssemblyPattern(np.r_[1.0, 1.0, 0.0], np.array([0, 1]), 0.5)
        counts = np.random.default_rng(0).poisson(1.0, (300, 3)).astype(float)
        silent = counts.copy()
        silent[:, :2] = 0.0
        with pytest.warns(UserWarning) as caught:
            result = assembly_reactivation(silent, counts, pattern)
        messages = [str(warning.message) for warning in caught]
        assert any("excluded from both periods" in m for m in messages)
        assert any("comparison is undefined" in m for m in messages)
        assert np.isnan(result.mean_activation_difference)
        assert np.isnan(result.fraction_above_control_threshold)


# =============================================================================
# explained_variance_reactivation: pair selection
# =============================================================================


def _residual_partial(x, y, controls) -> float:
    """Partial correlation by least-squares residuals (independent of the code)."""
    design = np.column_stack([np.ones(x.size), *controls])
    resid_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    resid_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


@pytest.fixture(scope="module")
def tetrode_session() -> dict[str, np.ndarray]:
    """Counts of 16 units on 4 tetrodes in pre rest, running and post rest.

    Each tetrode's units share a spike-sorting-like common component in
    every period, which within-tetrode pairs carry and different-tetrode
    pairs do not; three cross-tetrode assemblies fire in running and post
    rest only.
    """
    rng = np.random.default_rng(12)
    n_units, n_time = 16, 6000
    tetrode = np.repeat(np.arange(4), 4)
    assemblies = [[0, 5, 10], [1, 6, 15], [3, 8, 13]]
    out = {"tetrode": tetrode}
    for period, drive in (("pre", 0.0), ("run", 0.3), ("post", 0.3)):
        counts = rng.poisson(1.0, (n_time, n_units)).astype(float)
        counts += rng.poisson(0.5, (n_time, 4))[:, tetrode]  # same-tetrode sharing
        for members in assemblies:
            counts[:, members] += 2 * rng.poisson(drive, (n_time, 1))
        out[period] = counts
    return out


class TestPairSelection:
    def test_different_tetrode_pairs_match_hand_partial_correlation(
        self, tetrode_session
    ) -> None:
        correlations = {
            period: pairwise_correlations(tetrode_session[period])
            for period in ("pre", "run", "post")
        }
        tetrode = tetrode_session["tetrode"]
        result = explained_variance_reactivation(
            correlations["run"],
            correlations["post"],
            control_correlations=correlations["pre"],
            unit_groups=tetrode,
        )
        first, second = np.triu_indices(tetrode.size, k=1)
        keep = tetrode[first] != tetrode[second]
        partial = _residual_partial(
            correlations["run"][keep],
            correlations["post"][keep],
            [correlations["pre"][keep]],
        )
        reverse = _residual_partial(
            correlations["run"][keep],
            correlations["pre"][keep],
            [correlations["post"][keep]],
        )
        assert result.n_pairs == keep.sum() == 120 - 4 * 6
        assert_allclose(result.partial_correlation, partial, atol=1e-12)
        assert_allclose(result.explained_variance, partial**2, atol=1e-12)
        assert_allclose(result.reversed_explained_variance, reverse**2, atol=1e-12)
        # Within-tetrode pairs share a component in every period, which the
        # pre-task control removes only partly; selecting changes the answer.
        every_pair = explained_variance_reactivation(
            correlations["run"],
            correlations["post"],
            control_correlations=correlations["pre"],
        )
        assert every_pair.n_pairs == 120
        assert not np.isclose(every_pair.explained_variance, result.explained_variance)

    def test_pair_mask_and_groups_combine(self, tetrode_session) -> None:
        correlations = {
            period: pairwise_correlations(tetrode_session[period])
            for period in ("run", "post")
        }
        tetrode = tetrode_session["tetrode"]
        mask = np.random.default_rng(1).random(120) < 0.7
        result = explained_variance_reactivation(
            correlations["run"],
            correlations["post"],
            unit_groups=tetrode,
            pair_mask=mask,
        )
        first, second = np.triu_indices(tetrode.size, k=1)
        keep = mask & (tetrode[first] != tetrode[second])
        r = np.corrcoef(correlations["run"][keep], correlations["post"][keep])[0, 1]
        assert result.n_pairs == keep.sum()
        assert_allclose(result.explained_variance, r**2, atol=1e-12)

    def test_groups_follow_template_labels(self, tetrode_session) -> None:
        """Groups are given in the template's unit order; other periods by label."""
        labels = 100 + np.arange(16)
        order = np.random.default_rng(2).permutation(16)
        tetrode = tetrode_session["tetrode"]
        expected = explained_variance_reactivation(
            pairwise_correlations(tetrode_session["run"]),
            pairwise_correlations(tetrode_session["post"]),
            unit_groups=tetrode,
        )
        result = explained_variance_reactivation(
            pairwise_correlations(tetrode_session["run"]),
            pairwise_correlations(tetrode_session["post"][:, order]),
            unit_groups=tetrode,
            template_unit_ids=labels,
            match_unit_ids=labels[order],
        )
        assert result.n_pairs == expected.n_pairs
        assert_allclose(result.explained_variance, expected.explained_variance)

    def test_integer_pair_mask_is_refused(self, tetrode_session) -> None:
        corr = pairwise_correlations(tetrode_session["run"])
        with pytest.raises(ValueError, match="pair_mask must be a boolean array"):
            explained_variance_reactivation(corr, corr, pair_mask=np.arange(120) % 2)

    def test_groups_need_one_entry_per_unit(self, tetrode_session) -> None:
        corr = pairwise_correlations(tetrode_session["run"])
        with pytest.raises(ValueError, match="one group per unit"):
            explained_variance_reactivation(corr, corr, unit_groups=np.arange(15))

    @pytest.mark.parametrize("missing", [np.nan, None])
    def test_missing_groups_are_refused(self, tetrode_session, missing) -> None:
        corr = pairwise_correlations(tetrode_session["run"])
        groups = np.arange(16, dtype=object if missing is None else float)
        groups[[3, 7]] = missing
        with pytest.raises(
            ValueError, match=r"missing or non-finite group for units \[3, 7\]"
        ):
            explained_variance_reactivation(corr, corr, unit_groups=groups)

    def test_selection_leaving_too_few_pairs_is_nan(self, tetrode_session) -> None:
        corr = pairwise_correlations(tetrode_session["run"])
        with pytest.warns(UserWarning, match="valid pairs after pair selection"):
            result = explained_variance_reactivation(
                corr, corr, pair_mask=np.arange(120) < 2
            )
        assert result.n_pairs == 2
        assert np.isnan(result.explained_variance) and np.isnan(result.p_value)


# =============================================================================
# explained_variance_reactivation: several controls
# =============================================================================


@pytest.fixture(scope="module")
def correlation_vectors() -> dict[str, np.ndarray]:
    """Pair-correlation vectors with shared structure, 300 pairs (25 units)."""
    rng = np.random.default_rng(3)
    n_pairs = 300
    structure = rng.standard_normal(n_pairs)
    familiar = rng.standard_normal(n_pairs)
    return {
        "pre": 0.4 * familiar + rng.standard_normal(n_pairs),
        "familiar": familiar + 0.5 * rng.standard_normal(n_pairs),
        "novel": structure + 0.3 * familiar + 0.5 * rng.standard_normal(n_pairs),
        "post": structure + familiar + rng.standard_normal(n_pairs),
    }


class TestSeveralControls:
    def test_second_order_partial_matches_residual_regression(
        self, correlation_vectors
    ) -> None:
        """Kudrimoti et al. (1999) experiment 2: r(N, POST | PRE, F2)."""
        v = correlation_vectors
        result = explained_variance_reactivation(
            v["novel"], v["post"], control_correlations=[v["pre"], v["familiar"]]
        )
        partial = _residual_partial(v["novel"], v["post"], [v["pre"], v["familiar"]])
        assert_allclose(result.partial_correlation, partial, atol=1e-12)
        assert_allclose(result.explained_variance, partial**2, atol=1e-12)
        assert result.n_controls == 2
        assert np.isnan(result.reversed_explained_variance)  # REV needs one control

    def test_two_dimensional_array_of_controls(self, correlation_vectors) -> None:
        v = correlation_vectors
        listed = explained_variance_reactivation(
            v["novel"], v["post"], control_correlations=[v["pre"], v["familiar"]]
        )
        stacked = explained_variance_reactivation(
            v["novel"],
            v["post"],
            control_correlations=np.vstack([v["pre"], v["familiar"]]),
        )
        assert listed.summary() == stacked.summary()

    def test_one_control_in_a_list_is_the_classic_formula(
        self, correlation_vectors
    ) -> None:
        v = correlation_vectors
        r = np.corrcoef([v["novel"], v["post"], v["pre"]])
        r_tm, r_tc, r_mc = r[0, 1], r[0, 2], r[1, 2]
        classic = (r_tm - r_tc * r_mc) / np.sqrt((1 - r_tc**2) * (1 - r_mc**2))
        reverse = (r_tc - r_tm * r_mc) / np.sqrt((1 - r_tm**2) * (1 - r_mc**2))
        for control in (v["pre"], [v["pre"]]):
            result = explained_variance_reactivation(
                v["novel"], v["post"], control_correlations=control
            )
            assert_allclose(result.partial_correlation, classic, atol=1e-12)
            assert_allclose(result.reversed_explained_variance, reverse**2, atol=1e-12)
            assert result.n_controls == 1

    def test_collinear_controls_are_undefined(self, correlation_vectors) -> None:
        v = correlation_vectors
        with pytest.warns(UserWarning, match="EV is undefined"):
            result = explained_variance_reactivation(
                v["novel"], v["post"], control_correlations=[v["pre"], 2 * v["pre"]]
            )
        assert np.isnan(result.explained_variance) and np.isnan(result.p_value)

    def test_several_controls_need_one_label_array_each(self) -> None:
        rng = np.random.default_rng(4)
        counts = {name: rng.poisson(1.0, (500, 6)).astype(float) for name in "abcd"}
        corr = {name: pairwise_correlations(c) for name, c in counts.items()}
        labels = np.arange(6)
        order = rng.permutation(6)
        reordered = explained_variance_reactivation(
            corr["a"],
            corr["b"],
            control_correlations=[
                corr["c"],
                pairwise_correlations(counts["d"][:, order]),
            ],
            template_unit_ids=labels,
            match_unit_ids=labels,
            control_unit_ids=[labels, labels[order]],
        )
        expected = explained_variance_reactivation(
            corr["a"], corr["b"], control_correlations=[corr["c"], corr["d"]]
        )
        assert_allclose(reordered.partial_correlation, expected.partial_correlation)
        # One control in a one-item list takes its labels the same way.
        listed = explained_variance_reactivation(
            corr["a"],
            corr["b"],
            control_correlations=[pairwise_correlations(counts["d"][:, order])],
            template_unit_ids=labels,
            match_unit_ids=labels,
            control_unit_ids=[labels[order]],
        )
        alone = explained_variance_reactivation(
            corr["a"], corr["b"], control_correlations=corr["d"]
        )
        assert_allclose(listed.partial_correlation, alone.partial_correlation)
        with pytest.raises(ValueError, match="one label array per control"):
            explained_variance_reactivation(
                corr["a"],
                corr["b"],
                control_correlations=[corr["c"], corr["d"]],
                template_unit_ids=labels,
                match_unit_ids=labels,
                control_unit_ids=labels,
            )


# =============================================================================
# explained_variance_reactivation: p-value
# =============================================================================


class TestPartialCorrelationPValue:
    def test_without_control_matches_scipy_pearsonr(self, correlation_vectors) -> None:
        v = correlation_vectors
        result = explained_variance_reactivation(v["pre"], v["familiar"])
        expected = stats.pearsonr(v["pre"], v["familiar"])
        assert_allclose(result.partial_correlation, expected.statistic, atol=1e-12)
        assert_allclose(result.p_value, expected.pvalue, rtol=1e-8)

    @pytest.mark.parametrize("controls", [["pre"], ["pre", "familiar"]])
    def test_matches_regression_coefficient_t_test(
        self, correlation_vectors, controls
    ) -> None:
        """The partial-r t test equals the OLS t test of the template coefficient."""
        v = correlation_vectors
        result = explained_variance_reactivation(
            v["novel"],
            v["post"],
            control_correlations=[v[name] for name in controls],
        )
        n = v["post"].size
        design = np.column_stack(
            [np.ones(n), *(v[name] for name in controls), v["novel"]]
        )
        coefficients, *_ = np.linalg.lstsq(design, v["post"], rcond=None)
        residuals = v["post"] - design @ coefficients
        df = n - design.shape[1]
        assert df == n - 2 - len(controls)
        covariance = residuals @ residuals / df * np.linalg.inv(design.T @ design)
        t_statistic = coefficients[-1] / np.sqrt(covariance[-1, -1])
        assert_allclose(result.p_value, 2 * stats.t.sf(abs(t_statistic), df), rtol=1e-8)
        assert result.p_value < 1e-10

    def test_null_p_values_are_uniform_for_independent_pairs(self) -> None:
        """With independent pairs and no relation the test has its nominal size."""
        rng = np.random.default_rng(5)
        p_values = np.array(
            [
                explained_variance_reactivation(
                    *rng.standard_normal((2, 60)),
                    control_correlations=rng.standard_normal(60),
                ).p_value
                for _ in range(2000)
            ]
        )
        assert stats.kstest(p_values, "uniform").pvalue > 0.01
        assert abs(np.mean(p_values < 0.05) - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 2000)


# =============================================================================
# Kudrimoti's split-half check
# =============================================================================


def test_split_half_control_leaves_rho_over_one_plus_rho() -> None:
    """Split-half control is not 0 in expectation when correlations are noisy.

    Kudrimoti et al. (1999, p. 4094) substituted one half of RUN for PRE and
    found EV ~ 0. If both halves and POST measure the same pair-correlation
    structure with independent noise, each pair of vectors correlates with
    rho and the partial correlation is rho / (1 + rho): about 0 only when the
    correlations are mostly noise. The docstring states this caveat.
    """
    rng = np.random.default_rng(6)
    for noise_variance in (0.25, 1.0, 4.0):
        rho = 1.0 / (1.0 + noise_variance)
        partials = []
        for _ in range(100):
            structure = rng.standard_normal(2000)
            first_half, second_half, post = structure + np.sqrt(
                noise_variance
            ) * rng.standard_normal((3, 2000))
            partials.append(
                explained_variance_reactivation(
                    second_half, post, control_correlations=first_half
                ).partial_correlation
            )
        assert_allclose(np.mean(partials), rho / (1 + rho), atol=0.01)
