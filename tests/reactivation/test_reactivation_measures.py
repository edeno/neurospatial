"""Published reactivation measures computed exactly as the papers define them.

- ``assembly_reactivation``: mean activation strength after minus before
  (van de Ven et al. 2016; Peyrache et al. 2009) and the share of match bins
  above the control's 99th percentile (Peyrache et al. 2010).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.reactivation import (
    AssemblyPattern,
    assembly_reactivation,
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
