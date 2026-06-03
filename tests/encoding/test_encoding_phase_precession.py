"""Tests for encoding/phase_precession.py re-export module.

This module verifies that all phase precession functions are correctly
re-exported from the new encoding.phase_precession location.
"""

from __future__ import annotations

import time

import numpy as np
import pytest


class TestPhasePrecessionImports:
    """Test that all phase precession symbols can be imported from encoding.phase_precession."""

    def test_import_phase_precession_result(self) -> None:
        """PhasePrecessionResult should be importable from encoding.phase_precession."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        assert PhasePrecessionResult is not None

    def test_import_phase_precession(self) -> None:
        """phase_precession should be importable from encoding.phase_precession."""
        from neurospatial.encoding.phase_precession import phase_precession

        assert callable(phase_precession)

    def test_import_has_phase_precession(self) -> None:
        """has_phase_precession should be importable from encoding.phase_precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        assert callable(has_phase_precession)

    def test_import_plot_phase_precession(self) -> None:
        """plot_phase_precession should be importable from encoding.phase_precession."""
        from neurospatial.encoding.phase_precession import plot_phase_precession

        assert callable(plot_phase_precession)


class TestPhasePrecessionEncodingPackageImports:
    """Test that all phase precession symbols can be imported from encoding package."""

    def test_import_phase_precession_result_from_encoding(self) -> None:
        """PhasePrecessionResult should be importable from encoding."""
        from neurospatial.encoding import PhasePrecessionResult

        assert PhasePrecessionResult is not None

    def test_import_phase_precession_from_encoding(self) -> None:
        """phase_precession should be importable from encoding."""
        from neurospatial.encoding import phase_precession

        assert callable(phase_precession)

    def test_import_has_phase_precession_from_encoding(self) -> None:
        """has_phase_precession should be importable from encoding."""
        from neurospatial.encoding import has_phase_precession

        assert callable(has_phase_precession)

    def test_import_plot_phase_precession_from_encoding(self) -> None:
        """plot_phase_precession should be importable from encoding."""
        from neurospatial.encoding import plot_phase_precession

        assert callable(plot_phase_precession)


class TestPhasePrecessionModuleStructure:
    """Test module structure and metadata."""

    def test_module_has_all_attribute(self) -> None:
        """Module should have __all__ attribute."""
        import importlib

        pp_module = importlib.import_module("neurospatial.encoding.phase_precession")
        assert hasattr(pp_module, "__all__")

    def test_all_contains_expected_exports(self) -> None:
        """__all__ should contain all expected exports."""
        import importlib

        pp_module = importlib.import_module("neurospatial.encoding.phase_precession")
        expected = {
            "PhasePrecessionResult",
            "phase_precession",
            "has_phase_precession",
            "plot_phase_precession",
            "theta_phase",
        }
        assert set(pp_module.__all__) == expected

    def test_module_docstring_exists(self) -> None:
        """Module should have a docstring."""
        import importlib

        pp_module = importlib.import_module("neurospatial.encoding.phase_precession")
        assert pp_module.__doc__ is not None
        assert len(pp_module.__doc__) > 0


class TestPhasePrecessionReExportIdentity:
    """Test that re-exports are identical to original implementations."""

    def test_phase_precession_result_identity(self) -> None:
        """PhasePrecessionResult should be the same class as original."""
        from neurospatial.encoding.phase_precession import (
            PhasePrecessionResult as EncodingPhasePrecessionResult,
        )
        from neurospatial.encoding.phase_precession import (
            PhasePrecessionResult as MetricsPhasePrecessionResult,
        )

        assert EncodingPhasePrecessionResult is MetricsPhasePrecessionResult

    def test_phase_precession_identity(self) -> None:
        """phase_precession should be the same function as original."""
        from neurospatial.encoding.phase_precession import (
            phase_precession as encoding_phase_precession,
        )
        from neurospatial.encoding.phase_precession import (
            phase_precession as metrics_phase_precession,
        )

        assert encoding_phase_precession is metrics_phase_precession

    def test_has_phase_precession_identity(self) -> None:
        """has_phase_precession should be the same function as original."""
        from neurospatial.encoding.phase_precession import (
            has_phase_precession as encoding_has_phase_precession,
        )
        from neurospatial.encoding.phase_precession import (
            has_phase_precession as metrics_has_phase_precession,
        )

        assert encoding_has_phase_precession is metrics_has_phase_precession

    def test_plot_phase_precession_identity(self) -> None:
        """plot_phase_precession should be the same function as original."""
        from neurospatial.encoding.phase_precession import (
            plot_phase_precession as encoding_plot_phase_precession,
        )
        from neurospatial.encoding.phase_precession import (
            plot_phase_precession as metrics_plot_phase_precession,
        )

        assert encoding_plot_phase_precession is metrics_plot_phase_precession


class TestPhasePrecessionFunctionality:
    """Test that re-exported functions work correctly."""

    def test_phase_precession_returns_result(self) -> None:
        """phase_precession should return a PhasePrecessionResult."""
        from neurospatial.encoding.phase_precession import (
            PhasePrecessionResult,
            phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        phases = rng.uniform(0, 2 * np.pi, 100)

        result = phase_precession(positions, phases)

        assert isinstance(result, PhasePrecessionResult)

    def test_phase_precession_detects_negative_slope(self) -> None:
        """phase_precession should detect negative slope for precession data."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        # Simulate phase precession: phase decreases with position
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = phase_precession(positions, phases)

        # Sign-only check; for magnitude recovery, see
        # TestPhasePrecessionSlopeMagnitude in test_phase_precession_metrics.py.
        assert result.slope < 0

    def test_has_phase_precession_returns_bool(self) -> None:
        """has_phase_precession should return a boolean."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = has_phase_precession(positions, phases)

        assert isinstance(result, bool)

    def test_has_phase_precession_detects_precession(self) -> None:
        """has_phase_precession should detect strong phase precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 200)
        # Strong negative slope
        true_slope = -0.15
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.1, 200)) % (
            2 * np.pi
        )

        result = has_phase_precession(
            positions, phases, alpha=0.05, min_correlation=0.2
        )

        assert result is True

    def test_plot_phase_precession_returns_axes(self) -> None:
        """plot_phase_precession should return matplotlib Axes object."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases)

        try:
            assert ax is not None
            assert hasattr(ax, "plot")  # Check it's an Axes-like object
        finally:
            plt.close("all")

    def test_phase_precession_result_is_significant(self) -> None:
        """PhasePrecessionResult.is_significant should work correctly."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.5,
            pval=0.01,
            mean_resultant_length=0.8,
        )

        assert result.is_significant(alpha=0.05) is True
        assert result.is_significant(alpha=0.001) is False

    def test_phase_precession_result_interpretation(self) -> None:
        """PhasePrecessionResult.interpretation should return a string."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.5,
            pval=0.01,
            mean_resultant_length=0.8,
        )

        interp = result.interpretation()

        assert isinstance(interp, str)
        assert "SIGNIFICANT" in interp
        assert "PRECESSION" in interp


class TestPhasePrecessionShufflePvalue:
    """pval is a shuffle p-value at the fitted slope; correlation is descriptive."""

    def test_phase_precession_pval_tracks_fitted_slope(self, precessing_spikes) -> None:
        from neurospatial.encoding.phase_precession import phase_precession

        d = precessing_spikes
        n_shuffles = 200
        true_fit = phase_precession(
            d["positions"], d["phases"], n_shuffles=n_shuffles, rng=0
        )
        assert true_fit.pval < 0.05
        assert true_fit.slope < 0

        # Pin the SHUFFLE mechanics: a permutation p-value with +1 smoothing
        # (count+1)/(n_shuffles+1) can never fall below 1/(n_shuffles+1). The
        # OLD slope-free circular_linear_correlation p-value returned exactly
        # 0.0 on this strong fixture, violating this floor — so reverting pval
        # to that analytic value would fail here.
        smoothing_floor = 1.0 / (n_shuffles + 1)
        assert true_fit.pval >= smoothing_floor

        # Shuffled phases against the same positions destroy the relationship.
        shuffled = phase_precession(
            d["positions"], d["phase_shuffled"], n_shuffles=n_shuffles, rng=0
        )
        assert shuffled.pval > 0.5

    def test_phase_precession_pval_deterministic(self, precessing_spikes) -> None:
        from neurospatial.encoding.phase_precession import phase_precession

        d = precessing_spikes
        a = phase_precession(d["positions"], d["phases"], n_shuffles=200, rng=7)
        b = phase_precession(d["positions"], d["phases"], n_shuffles=200, rng=7)
        assert a.pval == b.pval

    def test_phase_precession_correlation_is_descriptive(
        self, precessing_spikes
    ) -> None:
        from neurospatial.encoding.phase_precession import phase_precession
        from neurospatial.stats.circular import circular_linear_correlation

        d = precessing_spikes
        result = phase_precession(d["positions"], d["phases"], n_shuffles=200, rng=0)
        assert 0.0 <= result.correlation <= 1.0
        wrapped = d["phases"] % (2 * np.pi)
        expected, _ = circular_linear_correlation(
            angles=wrapped, linear_values=d["positions"]
        )
        assert result.correlation == pytest.approx(expected, rel=1e-9)


class TestHasPhasePrecessionValidation:
    """Genuine input errors raise; insufficient data still maps to False."""

    def test_has_phase_precession_raises_on_length_mismatch(self) -> None:
        from neurospatial.encoding.phase_precession import has_phase_precession

        positions = np.linspace(0, 50, 100)
        phases = np.linspace(0, 2 * np.pi, 90)
        with pytest.raises(ValueError):
            has_phase_precession(positions, phases)

    def test_has_phase_precession_false_on_insufficient_spikes(self) -> None:
        from neurospatial.encoding.phase_precession import has_phase_precession

        positions = np.array([1.0, 2.0, 3.0])
        phases = np.array([0.1, 0.2, 0.3])
        assert has_phase_precession(positions, phases) is False

    def test_has_phase_precession_within_time_budget(self, precessing_spikes) -> None:
        from neurospatial.encoding.phase_precession import has_phase_precession

        d = precessing_spikes
        start = time.perf_counter()
        has_phase_precession(d["positions"], d["phases"], rng=0)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
