"""Smoke tests for the ``neurospatial.stats.circular`` API surface.

The canonical correctness tests for these functions live in
``tests/stats/test_circular_metrics.py`` (richer numerical fixtures,
edge-case / NaN / weighted variants, range checks). This file only
keeps the two pieces that file doesn't cover:

- A handful of basic functional smokes for the broader circular API
  (``circular_basis``, ``is_modulated``, ``wrap_angle``,
  ``plot_circular_basis_tuning``) — these aren't in
  ``test_circular_metrics.py``.
- The plot-helper smoke parametrized over polar vs linear projection.

Everything else (the 13 ``test_X_importable`` tautologies and the
duplicated ``test_rayleigh_test_uniform`` / ``test_*_basic`` tests that
overlap with ``test_circular_metrics.py``) was removed.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.stats.circular import (
    circular_basis,
    circular_basis_metrics,
    circular_mean,
    circular_variance,
    is_modulated,
    mean_resultant_length,
    plot_circular_basis_tuning,
    rayleigh_test,
    wrap_angle,
)


def test_circular_basis_basic():
    """``circular_basis`` builds the sin/cos design matrix at the cardinal angles."""
    angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = circular_basis(angles)

    assert result.design_matrix.shape == (4, 2)
    assert_allclose(result.sin_component[0], 0.0, atol=1e-10)
    assert_allclose(result.cos_component[0], 1.0, atol=1e-10)
    assert_allclose(result.sin_component[1], 1.0, atol=1e-10)
    assert_allclose(result.cos_component[1], 0.0, atol=1e-10)


def test_circular_basis_metrics_basic():
    """``circular_basis_metrics`` recovers amplitude / phase from sin/cos coefs."""
    amplitude, phase, pvalue = circular_basis_metrics(1.0, 0.0)
    assert_allclose(amplitude, 1.0)
    assert_allclose(phase, np.pi / 2)
    assert pvalue is None  # No cov_matrix provided


def test_circular_basis_metrics_with_cov():
    """``circular_basis_metrics`` returns a p-value when a covariance is given."""
    cov = np.array([[0.01, 0.001], [0.001, 0.01]])
    amplitude, _phase, pvalue = circular_basis_metrics(0.5, 0.3, cov)
    assert_allclose(amplitude, np.sqrt(0.5**2 + 0.3**2))
    assert pvalue is not None
    assert 0 <= pvalue <= 1


def test_is_modulated_strong_signal():
    """``is_modulated`` returns True for a high-amplitude / low-noise signal."""
    cov = np.array([[0.01, 0], [0, 0.01]])
    assert is_modulated(2.0, 2.0, cov)


def test_is_modulated_weak_signal():
    """``is_modulated`` returns False when the signal is dwarfed by noise."""
    cov = np.array([[1.0, 0], [0, 1.0]])
    assert not is_modulated(0.1, 0.1, cov)


def test_wrap_angle_basic():
    """``wrap_angle`` maps integer multiples of pi to the canonical (-pi, pi] range."""
    angles = np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -np.pi, -2 * np.pi])
    wrapped = wrap_angle(angles)

    assert_allclose(wrapped[0], 0.0, atol=1e-10)
    assert_allclose(np.abs(wrapped[1]), np.pi, atol=1e-10)
    assert_allclose(wrapped[2], 0.0, atol=1e-10)
    assert_allclose(np.abs(wrapped[3]), np.pi, atol=1e-10)


@pytest.mark.parametrize("projection", ["polar", "linear"])
def test_plot_circular_basis_tuning_smoke(projection):
    """``plot_circular_basis_tuning`` renders polar and linear projections."""
    ax = plot_circular_basis_tuning(0.5, 0.5, intercept=1.0, projection=projection)
    assert ax is not None
    plt.close("all")


def test_plot_circular_basis_tuning_with_data():
    """Data overlay branch renders without raising."""
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rates = np.exp(1 + 0.5 * np.cos(angles))
    ax = plot_circular_basis_tuning(
        0.5, 0.0, intercept=1.0, angles=angles, rates=rates, show_data=True
    )
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# Weighted circular statistics: weight validation and NaN co-filtering.
# ---------------------------------------------------------------------------


def test_rayleigh_test_length1_weights_raises(concentrated_angles):
    """A length-1 weight array is a mismatch, never broadcast."""
    assert len(concentrated_angles) == 20
    with pytest.raises(ValueError, match="Length mismatch"):
        rayleigh_test(concentrated_angles, weights=np.array([2.0]))


def test_rayleigh_test_negative_weights_raises(concentrated_angles):
    """A negative weight is rejected (would otherwise allow z < 0)."""
    weights = np.ones(len(concentrated_angles))
    weights[3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        rayleigh_test(concentrated_angles, weights=weights)


def test_rayleigh_test_all_zero_weights_raises(concentrated_angles):
    """All-zero weights raise a ValueError, not a ZeroDivisionError."""
    weights = np.zeros(len(concentrated_angles))
    with pytest.raises(ValueError, match="zero"):
        rayleigh_test(concentrated_angles, weights=weights)


def test_rayleigh_test_nan_cofilters_weights(concentrated_angles):
    """A NaN angle and its weight are dropped together, keeping alignment."""
    angles = concentrated_angles.copy()
    weights = np.linspace(1.0, 2.0, len(angles))

    # Insert a NaN at a known index; the matched weight must drop with it.
    nan_idx = 7
    angles_with_nan = angles.copy()
    angles_with_nan[nan_idx] = np.nan

    keep = np.arange(len(angles)) != nan_idx
    angles_manual = angles[keep]
    weights_manual = weights[keep]

    with pytest.warns(UserWarning, match="Removed 1 NaN"):
        z_cofiltered, p_cofiltered = rayleigh_test(angles_with_nan, weights=weights)
    z_manual, p_manual = rayleigh_test(angles_manual, weights=weights_manual)

    assert_allclose(z_cofiltered, z_manual)
    assert_allclose(p_cofiltered, p_manual)


def test_rayleigh_test_integer_weights_match_replication():
    """Integer weights reproduce physical replication of each angle."""
    angles = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    counts = np.array([1, 2, 1, 3, 1])

    z_weighted, p_weighted = rayleigh_test(angles, weights=counts.astype(float))
    z_repeat, p_repeat = rayleigh_test(np.repeat(angles, counts))

    assert_allclose(z_weighted, z_repeat, rtol=1e-9)
    assert_allclose(p_weighted, p_repeat, rtol=1e-9)


def test_rayleigh_test_concentrated_counts_significant():
    """100 spikes in 2 adjacent bins is strongly tuned, not rejected as NaN.

    The effective sample size for weighted (count) data is sum(weights), not
    the number of occupied angular bins. Two bins carrying 100 spikes must
    therefore yield a small p-value matching physical replication, rather
    than tripping the min-samples gate.
    """
    centers = np.array([0.0, 0.1])
    counts = np.array([60.0, 40.0])  # 100 spikes total in 2 adjacent bins

    z_weighted, p_weighted = rayleigh_test(centers, weights=counts)

    # Significant, not NaN.
    assert np.isfinite(p_weighted)
    assert p_weighted < 1e-10

    # Matches the physical oracle: repeat each angle by its count.
    z_repeat, p_repeat = rayleigh_test(np.repeat(centers, counts.astype(int)))
    assert_allclose(z_weighted, z_repeat, rtol=1e-9)
    assert_allclose(p_weighted, p_repeat, rtol=1e-9)


def test_rayleigh_test_insufficient_total_weight_raises():
    """Too few total counts (sum < 3) is genuinely insufficient and rejected."""
    centers = np.array([0.0, 0.1])
    counts = np.array([1.0, 1.0])  # only 2 spikes total
    with pytest.raises(ValueError, match="3 total weight"):
        rayleigh_test(centers, weights=counts)


def test_mean_resultant_length_length1_weights_raises(concentrated_angles):
    """``mean_resultant_length`` rejects a length-1 weight array."""
    with pytest.raises(ValueError, match="Length mismatch"):
        mean_resultant_length(concentrated_angles, weights=np.array([1.0]))


def test_circular_mean_negative_weights_raises(concentrated_angles):
    """``circular_mean`` rejects negative weights."""
    weights = np.ones(len(concentrated_angles))
    weights[0] = -2.0
    with pytest.raises(ValueError, match="non-negative"):
        circular_mean(concentrated_angles, weights=weights)


def test_circular_variance_weights_validates(concentrated_angles):
    """``circular_variance`` rejects a mismatched-length weight array."""
    weights = np.ones(len(concentrated_angles) - 1)
    with pytest.raises(ValueError, match="Length mismatch"):
        circular_variance(concentrated_angles, weights=weights)
