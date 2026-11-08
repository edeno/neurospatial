"""Null calibration of the circular correlation and circular-basis Wald p-values.

Each test draws data with no association, computes the p-value many times
with a seeded generator, and checks the share of p < alpha from both ends
with the shared binomial rule in ``tests/calibration_helpers.py``. The two
correlations follow CircStat's ``circ_corrcl`` and ``circ_corrcc``; their
large-sample approximations are conservative at small n, which the tests pin
as documented.
"""

from __future__ import annotations

import numpy as np
import pytest
from calibration_helpers import assert_pvalues_uniform, assert_rejections_at_rate
from scipy import stats

from neurospatial.stats.circular import (
    circular_basis,
    circular_basis_metrics,
    circular_circular_correlation,
    circular_linear_correlation,
    is_modulated,
)

N_NULL = 4000


def _angles(rng: np.random.Generator, n: int, kappa: float, mu: float = 0.0):
    """Uniform angles for ``kappa == 0``, von Mises angles otherwise."""
    if kappa == 0:
        return rng.uniform(-np.pi, np.pi, n)
    return rng.vonmises(mu, kappa, n)


def _circular_linear_null(n: int, kappa: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            circular_linear_correlation(_angles(rng, n, kappa), rng.standard_normal(n))[
                1
            ]
            for _ in range(N_NULL)
        ]
    )


def _circular_circular_null(n: int, kappa: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            circular_circular_correlation(
                _angles(rng, n, kappa), _angles(rng, n, kappa, mu=1.0)
            )[1]
            for _ in range(N_NULL)
        ]
    )


class TestCircularLinearNull:
    """``n r**2`` against chi-squared(2) for independent angle and value."""

    @pytest.mark.parametrize("kappa", [0.0, 2.0])
    @pytest.mark.parametrize("n", [30, 200])
    def test_rejects_at_alpha_from_30_samples(self, n, kappa):
        """Von Mises angles make cos and sin correlated, which stresses df = 2."""
        p_values = _circular_linear_null(n, kappa, seed=n + int(10 * kappa))
        assert_rejections_at_rate(p_values < 0.05, 0.05)

    @pytest.mark.parametrize("kappa", [0.0, 2.0])
    def test_pvalues_are_uniform_at_200_samples(self, kappa):
        assert_pvalues_uniform(_circular_linear_null(200, kappa, seed=7))

    @pytest.mark.parametrize("n", [30, 200])
    def test_two_valued_angles_use_one_degree_of_freedom(self, n):
        """Collinear cos and sin: n r**2 against chi-squared(1), the rank of the fit."""
        rng = np.random.default_rng(n + 1)
        p_values = []
        while len(p_values) < N_NULL:
            angles = rng.choice([0.3, 2.0], n)
            if np.ptp(angles) > 0:
                p_values.append(
                    circular_linear_correlation(angles, rng.standard_normal(n))[1]
                )
        assert_rejections_at_rate(np.array(p_values) < 0.05, 0.05)

    def test_ten_samples_are_conservative(self):
        """At n = 10 the 5% test rejects at most 5% and the 1% test almost never."""
        p_values = _circular_linear_null(10, 0.0, seed=10)
        assert_rejections_at_rate(p_values < 0.05, 0.05, conservative=True)
        assert np.mean(p_values < 0.01) < 0.002

    @pytest.mark.parametrize("n", [3, 5, 9])
    def test_pvalue_floor_is_exp_minus_half_n(self, n):
        """r <= 1, so p >= chi2(2).sf(n) = exp(-n / 2), even for perfect fits."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r, p = circular_linear_correlation(angles, np.cos(angles))
        assert r == pytest.approx(1.0)
        assert p == pytest.approx(np.exp(-n / 2), rel=1e-9)


class TestCircularCircularNull:
    """The Jammalamadaka-SenGupta normal approximation for independent angles."""

    @pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("n", [30, 200])
    def test_rejects_at_alpha_from_30_samples(self, n, kappa):
        p_values = _circular_circular_null(n, kappa, seed=100 + n + int(10 * kappa))
        assert_rejections_at_rate(p_values < 0.05, 0.05)

    @pytest.mark.parametrize("kappa", [0.0, 2.0])
    def test_pvalues_are_uniform_at_200_samples(self, kappa):
        assert_pvalues_uniform(_circular_circular_null(200, kappa, seed=11))

    def test_five_samples_are_conservative(self):
        p_values = _circular_circular_null(5, 0.0, seed=5)
        assert_rejections_at_rate(p_values < 0.05, 0.05, conservative=True)
        assert np.mean(p_values < 0.05) < 0.03


def _fit_poisson_glm(angles, counts, dt):
    """Poisson GLM of counts on [1, sin, cos] with a log(dt) offset (statsmodels)."""
    sm = pytest.importorskip("statsmodels.api")
    design = sm.add_constant(circular_basis(angles).design_matrix)
    model = sm.GLM(
        counts,
        design,
        family=sm.families.Poisson(),
        offset=np.full(len(counts), np.log(dt)),
    ).fit()
    return np.asarray(model.params), np.asarray(model.cov_params())


def _wald_null(n_bins, mean_count, seed, n_cells=2000):
    """Wald p-values and is_modulated verdicts of untuned Poisson cells."""
    rng = np.random.default_rng(seed)
    p_values, modulated = [], []
    while len(p_values) < n_cells:
        angles = rng.uniform(-np.pi, np.pi, n_bins)
        counts = rng.poisson(mean_count, n_bins)
        if counts.sum() == 0:
            continue
        params, cov = _fit_poisson_glm(angles, counts, dt=0.1)
        _, _, p = circular_basis_metrics(params[1], params[2], cov[1:3, 1:3])
        p_values.append(p)
        modulated.append(is_modulated(params[1], params[2], cov[1:3, 1:3]))
    return np.array(p_values), np.array(modulated)


@pytest.mark.slow
class TestCircularBasisWald:
    """Wald test of a Poisson GLM fit with ``circular_basis`` (statsmodels)."""

    @pytest.mark.parametrize(("n_bins", "mean_count"), [(2000, 0.05), (2000, 0.5)])
    def test_null_is_calibrated_from_100_spikes(self, n_bins, mean_count):
        """100 and 1000 expected spikes: rate at alpha and uniform p-values."""
        p_values, modulated = _wald_null(n_bins, mean_count, seed=n_bins + 1)
        assert_rejections_at_rate(p_values < 0.05, 0.05)
        assert_pvalues_uniform(p_values)
        if mean_count == 0.5:
            # With 1000 spikes a null amplitude estimate is rarely as large as
            # the default 0.2 (a 1.49-fold modulation): 1 of 2000 cells here.
            assert modulated.mean() < 0.005

    def test_null_is_conservative_with_25_spikes(self):
        """With few spikes the test is conservative and the gate does nothing."""
        p_values, modulated = _wald_null(500, 0.05, seed=3)
        assert_rejections_at_rate(p_values < 0.05, 0.05, conservative=True)
        assert np.array_equal(modulated, p_values < 0.05)

    def test_wald_region_covers_the_planted_modulation(self):
        """The 95% Wald ellipse holds the true (beta_sin, beta_cos) in 95% of fits."""
        amplitude, phase = 0.3, 1.0
        beta = np.array([amplitude * np.sin(phase), amplitude * np.cos(phase)])
        rng = np.random.default_rng(7)
        outside, significant, phases = [], [], []
        for _ in range(2000):
            angles = rng.uniform(-np.pi, np.pi, 2000)
            counts = rng.poisson(0.2 * np.exp(amplitude * np.cos(angles - phase)))
            params, cov = _fit_poisson_glm(angles, counts, dt=0.1)
            error = params[1:3] - beta
            wald = error @ np.linalg.solve(cov[1:3, 1:3], error)
            outside.append(wald > stats.chi2.ppf(0.95, df=2))
            _, fitted_phase, p = circular_basis_metrics(
                params[1], params[2], cov[1:3, 1:3]
            )
            significant.append(p < 0.05)
            phases.append(fitted_phase)
        assert_rejections_at_rate(outside, 0.05)
        assert np.mean(significant) > 0.9
        assert abs(stats.circmean(phases, high=np.pi, low=-np.pi) - phase) < 0.02
