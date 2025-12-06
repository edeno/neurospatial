"""Tests for stats.circular module - verifies new import paths work.

This file tests the reorganization of circular statistics functions into
the neurospatial.stats.circular module per PLAN.md Milestone 4.

Functions moved:
- From metrics/circular.py: rayleigh_test, circular_linear_correlation,
  circular_circular_correlation, phase_position_correlation
- From metrics/circular.py (made public): circular_mean, circular_variance,
  mean_resultant_length
- From metrics/circular_basis.py: circular_basis, circular_basis_metrics,
  is_modulated, plot_circular_basis_tuning, CircularBasisResult
- From metrics/vte.py: wrap_angle
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Test imports from new location (stats/circular.py)
from neurospatial.stats.circular import (
    CircularBasisResult,
    circular_basis,
    circular_basis_metrics,
    circular_circular_correlation,
    circular_linear_correlation,
    circular_mean,
    circular_variance,
    is_modulated,
    mean_resultant_length,
    phase_position_correlation,
    plot_circular_basis_tuning,
    rayleigh_test,
    wrap_angle,
)


class TestStatsCircularImports:
    """Test that all functions are importable from neurospatial.stats.circular."""

    # --- Core circular statistics (from metrics/circular.py) ---

    def test_rayleigh_test_importable(self):
        """Test rayleigh_test is importable from stats.circular."""
        assert callable(rayleigh_test)

    def test_circular_linear_correlation_importable(self):
        """Test circular_linear_correlation is importable from stats.circular."""
        assert callable(circular_linear_correlation)

    def test_circular_circular_correlation_importable(self):
        """Test circular_circular_correlation is importable from stats.circular."""
        assert callable(circular_circular_correlation)

    def test_phase_position_correlation_importable(self):
        """Test phase_position_correlation is importable from stats.circular."""
        assert callable(phase_position_correlation)

    # --- Newly public functions (from metrics/circular.py) ---

    def test_circular_mean_importable(self):
        """Test circular_mean is importable from stats.circular."""
        assert callable(circular_mean)

    def test_circular_variance_importable(self):
        """Test circular_variance is importable from stats.circular."""
        assert callable(circular_variance)

    def test_mean_resultant_length_importable(self):
        """Test mean_resultant_length is importable from stats.circular."""
        assert callable(mean_resultant_length)

    # --- Circular basis functions (from metrics/circular_basis.py) ---

    def test_circular_basis_importable(self):
        """Test circular_basis is importable from stats.circular."""
        assert callable(circular_basis)

    def test_circular_basis_metrics_importable(self):
        """Test circular_basis_metrics is importable from stats.circular."""
        assert callable(circular_basis_metrics)

    def test_is_modulated_importable(self):
        """Test is_modulated is importable from stats.circular."""
        assert callable(is_modulated)

    def test_plot_circular_basis_tuning_importable(self):
        """Test plot_circular_basis_tuning is importable from stats.circular."""
        assert callable(plot_circular_basis_tuning)

    def test_circular_basis_result_importable(self):
        """Test CircularBasisResult is importable from stats.circular."""
        assert CircularBasisResult is not None

    # --- wrap_angle (from metrics/vte.py) ---

    def test_wrap_angle_importable(self):
        """Test wrap_angle is importable from stats.circular."""
        assert callable(wrap_angle)


class TestStatsCircularBasicFunctionality:
    """Basic functionality tests for stats.circular module."""

    def test_rayleigh_test_uniform(self):
        """Test rayleigh_test returns high p-value for uniform distribution."""
        uniform_angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        _z, p = rayleigh_test(uniform_angles)
        assert p > 0.5  # Should not reject null of uniformity

    def test_rayleigh_test_concentrated(self):
        """Test rayleigh_test returns low p-value for concentrated distribution."""
        # Concentrated around 0
        rng = np.random.default_rng(42)
        concentrated_angles = rng.vonmises(mu=0, kappa=10, size=100)
        _z, p = rayleigh_test(concentrated_angles)
        assert p < 0.01  # Should reject null of uniformity

    def test_circular_linear_correlation_basic(self):
        """Test circular_linear_correlation with correlated data."""
        positions = np.linspace(0, 100, 50)
        phases = np.linspace(0, 2 * np.pi, 50)  # Linear relationship
        r, p = circular_linear_correlation(phases, positions)
        assert r > 0.5  # Significant correlation
        assert p < 0.05

    def test_circular_circular_correlation_basic(self):
        """Test circular_circular_correlation with correlated angles."""
        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 100)
        angles2 = angles1 + 0.1 * rng.standard_normal(100)  # Noisy copy
        r, _p = circular_circular_correlation(angles1, angles2)
        assert r > 0.8  # Strong correlation

    def test_phase_position_correlation_is_alias(self):
        """Test phase_position_correlation is an alias for circular_linear_correlation."""
        positions = np.linspace(0, 100, 50)
        phases = np.linspace(0, 2 * np.pi, 50)
        r1, p1 = circular_linear_correlation(phases, positions)
        r2, p2 = phase_position_correlation(phases, positions)
        assert_allclose(r1, r2)
        assert_allclose(p1, p2)

    def test_circular_mean_basic(self):
        """Test circular_mean computation."""
        # Test with angles clustered around pi/4
        angles = np.array([0, np.pi / 4, np.pi / 2])
        result = circular_mean(angles)
        # Mean should be around pi/4
        assert_allclose(result, np.pi / 4, atol=0.01)

    def test_circular_variance_basic(self):
        """Test circular_variance computation."""
        # Concentrated angles should have low variance
        rng = np.random.default_rng(42)
        concentrated = rng.vonmises(mu=0, kappa=50, size=100)
        var = circular_variance(concentrated)
        assert var < 0.1  # Low variance for concentrated distribution

    def test_mean_resultant_length_basic(self):
        """Test mean_resultant_length computation."""
        # Concentrated angles should have high mean resultant length
        rng = np.random.default_rng(42)
        concentrated = rng.vonmises(mu=0, kappa=50, size=100)
        r = mean_resultant_length(concentrated)
        assert r > 0.9  # High R for concentrated distribution

    def test_circular_basis_basic(self):
        """Test circular_basis creates correct design matrix."""
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = circular_basis(angles)

        assert result.design_matrix.shape == (4, 2)
        # sin(0) = 0, cos(0) = 1
        assert_allclose(result.sin_component[0], 0.0, atol=1e-10)
        assert_allclose(result.cos_component[0], 1.0, atol=1e-10)
        # sin(pi/2) = 1, cos(pi/2) = 0
        assert_allclose(result.sin_component[1], 1.0, atol=1e-10)
        assert_allclose(result.cos_component[1], 0.0, atol=1e-10)

    def test_circular_basis_metrics_basic(self):
        """Test circular_basis_metrics computes amplitude and phase correctly."""
        # beta_sin=1, beta_cos=0 -> amplitude=1, phase=pi/2
        amplitude, phase, pvalue = circular_basis_metrics(1.0, 0.0)
        assert_allclose(amplitude, 1.0)
        assert_allclose(phase, np.pi / 2)
        assert pvalue is None  # No cov_matrix provided

    def test_circular_basis_metrics_with_cov(self):
        """Test circular_basis_metrics computes p-value with covariance."""
        cov = np.array([[0.01, 0.001], [0.001, 0.01]])
        amplitude, _phase, pvalue = circular_basis_metrics(0.5, 0.3, cov)
        assert_allclose(amplitude, np.sqrt(0.5**2 + 0.3**2))
        assert pvalue is not None
        assert 0 <= pvalue <= 1

    def test_is_modulated_strong_signal(self):
        """Test is_modulated returns True for strong significant modulation."""
        cov = np.array([[0.01, 0], [0, 0.01]])
        assert is_modulated(2.0, 2.0, cov)

    def test_is_modulated_weak_signal(self):
        """Test is_modulated returns False for weak modulation."""
        cov = np.array([[1.0, 0], [0, 1.0]])
        assert not is_modulated(0.1, 0.1, cov)

    def test_wrap_angle_basic(self):
        """Test wrap_angle wraps angles to (-pi, pi]."""
        angles = np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -np.pi, -2 * np.pi])
        wrapped = wrap_angle(angles)

        # 0 -> 0
        assert_allclose(wrapped[0], 0.0, atol=1e-10)
        # pi -> pi (or -pi)
        assert_allclose(np.abs(wrapped[1]), np.pi, atol=1e-10)
        # 2*pi -> 0
        assert_allclose(wrapped[2], 0.0, atol=1e-10)
        # 3*pi -> pi (or -pi)
        assert_allclose(np.abs(wrapped[3]), np.pi, atol=1e-10)


class TestStatsPackageExports:
    """Test that functions are also accessible via stats package."""

    def test_import_from_stats_package(self):
        """Test that circular functions can be imported from neurospatial.stats."""
        from neurospatial.stats import (
            CircularBasisResult,
            circular_basis,
            circular_basis_metrics,
            circular_circular_correlation,
            circular_linear_correlation,
            circular_mean,
            circular_variance,
            is_modulated,
            mean_resultant_length,
            phase_position_correlation,
            plot_circular_basis_tuning,
            rayleigh_test,
            wrap_angle,
        )

        assert callable(rayleigh_test)
        assert callable(circular_linear_correlation)
        assert callable(circular_circular_correlation)
        assert callable(phase_position_correlation)
        assert callable(circular_mean)
        assert callable(circular_variance)
        assert callable(mean_resultant_length)
        assert callable(circular_basis)
        assert callable(circular_basis_metrics)
        assert callable(is_modulated)
        assert callable(plot_circular_basis_tuning)
        assert callable(wrap_angle)
        assert CircularBasisResult is not None


class TestPlotCircularBasisTuning:
    """Tests for plot_circular_basis_tuning (smoke tests only)."""

    @pytest.mark.parametrize("projection", ["polar", "linear"])
    def test_plot_circular_basis_tuning_smoke(self, projection):
        """Test plot_circular_basis_tuning creates a plot without errors."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        ax = plot_circular_basis_tuning(0.5, 0.5, intercept=1.0, projection=projection)
        assert ax is not None
        plt.close("all")

    def test_plot_circular_basis_tuning_with_data(self):
        """Test plot_circular_basis_tuning with data overlay."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        rates = np.exp(1 + 0.5 * np.cos(angles))
        ax = plot_circular_basis_tuning(
            0.5, 0.0, intercept=1.0, angles=angles, rates=rates, show_data=True
        )
        assert ax is not None
        plt.close("all")
