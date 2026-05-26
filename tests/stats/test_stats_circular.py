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
    is_modulated,
    plot_circular_basis_tuning,
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
