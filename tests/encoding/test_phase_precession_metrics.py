"""Tests for phase precession analysis module."""

from __future__ import annotations

import numpy as np
import pytest


class TestPhasePrecessionResult:
    """Tests for PhasePrecessionResult dataclass."""

    def test_is_significant_below_alpha(self) -> None:
        """Should return True when p-value is below alpha."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.5,
            p_value=0.01,
            mean_resultant_length=0.8,
        )
        assert result.is_significant(alpha=0.05) is True

    def test_is_significant_above_alpha(self) -> None:
        """Should return False when p-value is above alpha."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.2,
            p_value=0.10,
            mean_resultant_length=0.5,
        )
        assert result.is_significant(alpha=0.05) is False

    def test_interpretation_significant_precession(self) -> None:
        """Interpretation should indicate significant precession."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.5,
            p_value=0.01,
            mean_resultant_length=0.8,
        )
        interp = result.interpretation()
        assert "SIGNIFICANT" in interp
        assert "PRECESSION" in interp

    def test_interpretation_recession(self) -> None:
        """Interpretation should indicate recession for positive slope."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.5,
            p_value=0.01,
            mean_resultant_length=0.8,
        )
        interp = result.interpretation()
        assert "RECESSION" in interp

    def test_interpretation_claims_precession_only_when_significant(self) -> None:
        """A negative but non-significant fit is not reported as precession."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.05,
            p_value=0.4,
            mean_resultant_length=0.2,
        )
        interp = result.interpretation()
        assert "PRECESSION" not in interp
        assert "not significant" in interp

    def test_result_is_immutable(self) -> None:
        """Fields of a computed result cannot be reassigned."""
        from dataclasses import FrozenInstanceError

        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.5,
            p_value=0.01,
            mean_resultant_length=0.8,
        )
        with pytest.raises(FrozenInstanceError):
            result.slope = 0.3  # type: ignore[misc]

    def test_str_returns_interpretation(self) -> None:
        """str() should return interpretation."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            position_range=(0.0, 40.0),
            offset=1.0,
            correlation=0.5,
            p_value=0.01,
            mean_resultant_length=0.8,
        )
        assert str(result) == result.interpretation()


class TestPhasePrecession:
    """Tests for compute_phase_precession() function."""

    def test_negative_slope_for_precession(self) -> None:
        """Should return negative slope for phase precession data."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        # Simulate phase precession: phase decreases with position
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = compute_phase_precession(positions, phases)

        # Should detect negative slope
        assert result.slope < 0

    def test_recovers_slope_with_narrow_objective_lobe(self) -> None:
        """Recover a known slope when the MRL objective lobe is narrow.

        With a wide position span the main lobe of the circular objective
        (mean resultant length of residuals) becomes narrower than a coarse
        slope grid. A noiseless precessing cell has a unique global optimum
        at the true slope (MRL == 1). The optimizer must locate that lobe
        rather than settling into a neighboring side-lobe minimum.
        """
        from neurospatial.encoding.phase_precession import compute_phase_precession

        n = 400
        positions = np.linspace(0, 200, n)
        # About 17 cycles across the range: a wide search -> narrow MRL lobe.
        true_slope = -0.54211  # offset from coarse-grid sample points
        offset = 1.0
        phases = (offset + true_slope * positions) % (2 * np.pi)

        result = compute_phase_precession(
            positions, phases, slope_bounds=(-40 * np.pi, 40 * np.pi)
        )

        # Noiseless data: the true slope gives a perfect fit (MRL == 1).
        # The optimizer must recover it within a tight tolerance and report
        # near-maximal mean resultant length.
        assert result.slope == pytest.approx(true_slope, abs=1e-3), (
            f"expected slope ~{true_slope}, got {result.slope}"
        )
        assert result.mean_resultant_length > 0.99, (
            "optimizer landed in a side-lobe minimum instead of the global "
            f"optimum (MRL={result.mean_resultant_length})"
        )

    def test_significance_resolves_narrow_precession_lobe(self) -> None:
        """A perfect fit across a physical track must also pass its shuffle test."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        positions = np.sort(np.random.default_rng(0).uniform(0, 200, 300))
        phases = (-2 * np.pi * positions / 200) % (2 * np.pi)
        raw = compute_phase_precession(positions, phases, n_shuffles=99, rng=0)
        normalized = compute_phase_precession(
            positions, phases, position_range=(0, 200), n_shuffles=99, rng=0
        )
        assert raw.mean_resultant_length > 0.99
        assert raw.p_value == normalized.p_value == 0.01

    def test_position_units_do_not_change_the_fit(self) -> None:
        """The same spikes in cm and in m give one fit, reported in each unit."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        stimulus = np.random.default_rng(34)
        positions_cm = stimulus.uniform(0, 200, 80)
        phases = (
            np.pi - 2 * np.pi * positions_cm / 200 + stimulus.vonmises(0, 2, 80)
        ) % (2 * np.pi)
        in_cm = compute_phase_precession(positions_cm, phases, n_shuffles=99, rng=5)
        in_m = compute_phase_precession(
            positions_cm / 100, phases, n_shuffles=99, rng=5
        )
        assert in_m.p_value == in_cm.p_value
        assert in_m.slope == pytest.approx(in_cm.slope * 100, rel=1e-6)
        assert in_m.normalized_slope == pytest.approx(in_cm.normalized_slope, rel=1e-6)
        assert in_m.offset == pytest.approx(in_cm.offset, abs=1e-9)
        assert in_cm.mean_resultant_length == pytest.approx(
            in_m.mean_resultant_length, abs=1e-6
        )

    def test_default_bounds_recover_weakly_coupled_slope(self) -> None:
        """One cycle per 40 cm field is recovered from noisy, sparse spikes.

        Phase coupling is weak (von Mises kappa 0.7) and each cell has 40
        spikes. A search wider than one cycle per field lets noise alias onto
        steep slopes; the default search over one cycle per field recovers the
        true slope in most simulated cells.
        """
        from neurospatial.encoding.phase_precession import compute_phase_precession

        true_slope = -2 * np.pi / 40  # rad/cm
        recovered = []
        for seed in range(40):
            stimulus = np.random.default_rng(seed)
            positions = stimulus.uniform(0, 40, 40)
            phases = (
                np.pi + true_slope * (positions - 20) + stimulus.vonmises(0, 0.7, 40)
            ) % (2 * np.pi)
            result = compute_phase_precession(positions, phases, n_shuffles=0)
            recovered.append(result.slope)
        within = np.mean(np.abs(np.array(recovered) - true_slope) < 0.05)
        assert within >= 0.7, f"slope recovered in only {within:.0%} of cells"

    def test_offset_is_the_phase_at_the_start_of_the_range(self) -> None:
        """offset is the fitted phase at position_range[0]."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        stimulus = np.random.default_rng(11)
        positions = stimulus.uniform(20, 60, 200)
        true_offset, true_slope = 5.0, -2 * np.pi / 40
        phases = (
            true_offset + true_slope * (positions - 20) + stimulus.vonmises(0, 8, 200)
        ) % (2 * np.pi)
        result = compute_phase_precession(
            positions, phases, position_range=(20.0, 60.0), n_shuffles=0
        )
        assert result.position_range == (20.0, 60.0)
        assert result.slope == pytest.approx(true_slope, abs=0.01)
        assert result.normalized_slope == pytest.approx(result.slope * 40.0)
        offset_error = np.angle(np.exp(1j * (result.offset - true_offset)))
        assert abs(offset_error) < 0.1

    def test_constant_positions_raise(self) -> None:
        """A slope cannot be fit when every spike is at the same position."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        positions = np.full(50, 123.0)
        phases = np.random.default_rng(8).uniform(0, 2 * np.pi, len(positions))
        with pytest.raises(ValueError, match="no phase-position slope"):
            compute_phase_precession(positions, phases, n_shuffles=99, rng=1)

    def test_excessive_slope_search_raises(self) -> None:
        """An unresolvable search fails before allocating a huge grid of slopes."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        positions = np.linspace(0, 1e8, 30)
        phases = (-2 * np.pi * positions / 1e8) % (2 * np.pi)
        with pytest.raises(ValueError, match=r"(?s)20,000.*narrow slope_bounds"):
            compute_phase_precession(
                positions, phases, slope_bounds=(-1e5, 1e5), n_shuffles=0
            )
        result = compute_phase_precession(positions, phases, n_shuffles=9, rng=0)
        assert result.mean_resultant_length > 0.99
        assert result.slope == pytest.approx(-2 * np.pi / 1e8, rel=1e-3)
        assert result.p_value == 0.1

    def test_correlation_in_valid_range(self) -> None:
        """Correlation should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = compute_phase_precession(positions, phases)

        assert 0 <= result.correlation <= 1

    def test_pvalue_in_valid_range(self) -> None:
        """P-value should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = compute_phase_precession(positions, phases)

        assert 0 <= result.p_value <= 1

    def test_offset_in_valid_range(self) -> None:
        """Offset should be in [0, 2pi]."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = compute_phase_precession(positions, phases)

        # Offset should be in [0, 2pi] (circular mean of residuals)
        assert 0 <= result.offset <= 2 * np.pi

    def test_mean_resultant_length_in_valid_range(self) -> None:
        """Mean resultant length should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = compute_phase_precession(positions, phases)

        assert 0 <= result.mean_resultant_length <= 1

    def test_insufficient_spikes_raises(self) -> None:
        """Should raise ValueError for insufficient spikes."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        positions = np.array([1.0, 2.0])  # Only 2 spikes
        phases = np.array([0.5, 1.5])

        with pytest.raises(ValueError, match="at least"):
            compute_phase_precession(positions, phases, min_spikes=10)

    def test_mismatched_lengths_raises(self) -> None:
        """Should raise ValueError for mismatched array lengths."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        positions = np.array([1.0, 2.0, 3.0])
        phases = np.array([0.5, 1.5])

        with pytest.raises(ValueError, match="same length"):
            compute_phase_precession(positions, phases)

    def test_degree_input(self) -> None:
        """Should handle phases in degrees."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases_deg = rng.uniform(0, 360, 50)

        result = compute_phase_precession(positions, phases_deg, angle_unit="deg")

        assert 0 <= result.correlation <= 1

    def test_default_position_range_is_the_spike_extent(self) -> None:
        """Without position_range the fit is normalized over the spike positions."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 90, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = compute_phase_precession(positions, phases, n_shuffles=0)

        assert result.position_range == (positions.min(), positions.max())

    def test_invalid_position_range_raises(self) -> None:
        """Invalid position_range should raise ValueError."""
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        with pytest.raises(ValueError, match="end > start"):
            compute_phase_precession(positions, phases, position_range=(100.0, 0.0))


class TestHasPhasePrecession:
    """Tests for has_phase_precession() function."""

    def test_returns_bool(self) -> None:
        """Should return a boolean value."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = has_phase_precession(positions, phases)

        assert isinstance(result, bool)

    def test_detects_precession(self) -> None:
        """Should detect strong phase precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 200)
        # Strong negative slope: about one theta cycle across the field
        true_slope = -0.12
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.1, 200)) % (
            2 * np.pi
        )

        result = has_phase_precession(
            positions, phases, alpha=0.05, min_correlation=0.2, rng=0
        )

        assert result is True

    def test_random_data_usually_false(self) -> None:
        """Random data should usually not show phase precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = has_phase_precession(
            positions, phases, alpha=0.05, min_correlation=0.2, rng=0
        )

        assert result is False

    def test_positive_slope_returns_false(self) -> None:
        """Should return False for positive slope (recession, not precession)."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 200)
        # Strong positive slope (recession)
        true_slope = 0.12
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.1, 200)) % (
            2 * np.pi
        )

        result = has_phase_precession(positions, phases, rng=0)

        # Should return False because slope is positive
        assert result is False

    def test_forwards_slope_bounds(self) -> None:
        """A search restricted to recession cannot report precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        stimulus = np.random.default_rng(5)
        positions = stimulus.uniform(0, 40, 300)
        phases = (np.pi - 2 * np.pi * positions / 40 + stimulus.vonmises(0, 4, 300)) % (
            2 * np.pi
        )

        assert has_phase_precession(positions, phases, rng=0)
        assert not has_phase_precession(
            positions, phases, slope_bounds=(0.0, 2 * np.pi), rng=0
        )

    def test_handles_insufficient_data(self) -> None:
        """Should return False for insufficient data (not raise)."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        positions = np.array([1.0, 2.0])
        phases = np.array([0.5, 1.5])

        # Should return False, not raise
        result = has_phase_precession(positions, phases)

        assert result is False

    def test_too_few_spikes_or_fixed_position_is_false(self) -> None:
        """Data that cannot support a fit screen as False rather than raise."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        phases = np.random.default_rng(3).uniform(0, 2 * np.pi, 30)
        assert has_phase_precession(np.linspace(0, 40, 30), phases, min_spikes=31) is (
            False
        )
        assert has_phase_precession(np.full(30, 12.0), phases) is False

    def test_precessing_cell_on_a_long_track_in_mm(self) -> None:
        """Position units do not change the screen: a 2000 mm track works."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        stimulus = np.random.default_rng(7)
        positions_mm = stimulus.uniform(0, 2000, 200)
        phases = (
            np.pi - 2 * np.pi * positions_mm / 2000 + stimulus.vonmises(0, 4, 200)
        ) % (2 * np.pi)
        assert has_phase_precession(positions_mm, phases, rng=0) is True

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_shuffles": -1}, "n_shuffles"),
            ({"slope_bounds": (-1e5, 1e5)}, "20,000"),
            ({"slope_bounds": (1.0, -1.0)}, "slope_bounds"),
            ({"position_range": (40.0, 0.0)}, "position_range"),
            ({"angle_unit": "grad"}, "angle_unit"),
        ],
    )
    def test_configuration_errors_raise(self, kwargs, match) -> None:
        """A bad argument raises instead of screening the cell as False."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        stimulus = np.random.default_rng(7)
        positions = stimulus.uniform(0, 40, 100)
        phases = stimulus.uniform(0, 2 * np.pi, 100)
        with pytest.raises(ValueError, match=match):
            has_phase_precession(positions, phases, **kwargs)

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should affect result."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        # With strict thresholds, may not pass
        result_strict = has_phase_precession(
            positions, phases, alpha=0.01, min_correlation=0.5
        )
        # With lenient thresholds, more likely to pass
        result_lenient = has_phase_precession(
            positions, phases, alpha=0.1, min_correlation=0.1
        )

        # Lenient should be at least as permissive as strict
        if result_strict:
            assert result_lenient is True


class TestPlotPhasePrecession:
    """Tests for plot_phase_precession() function."""

    def test_returns_axes(self) -> None:
        """Should return matplotlib Axes object."""
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

    def test_accepts_existing_axes(self) -> None:
        """Should plot on provided axes."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        _fig, ax_input = plt.subplots()
        ax_output = plot_phase_precession(positions, phases, ax=ax_input)

        try:
            assert ax_output is ax_input
        finally:
            plt.close("all")

    def test_doubled_phase_axis(self) -> None:
        """Should plot phases doubled (0-4pi) per O'Keefe & Recce convention."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        n_points = 50
        positions = np.linspace(0, 50, n_points)
        phases = rng.uniform(0, 2 * np.pi, n_points)

        ax = plot_phase_precession(positions, phases)

        try:
            # Get y data from scatter plots
            collections = ax.collections
            assert len(collections) >= 1, "Expected at least one scatter collection"

            # Each point should appear twice (at phase and phase + 2pi)
            # So total plotted points should be 2 * n_points
            total_points = sum(len(c.get_offsets()) for c in collections)
            assert total_points == 2 * n_points, (
                f"Expected {2 * n_points} points (doubled), got {total_points}"
            )

            # Y-axis should extend to at least 4pi
            ylim = ax.get_ylim()
            assert ylim[1] >= 4 * np.pi - 0.1, (
                f"Expected y-axis to extend to ~4pi, got upper limit {ylim[1]}"
            )
        finally:
            plt.close("all")

    def test_y_axis_pi_labels(self) -> None:
        """Y-axis should have pi-based labels (0, π, 2π, 3π, 4π)."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases)

        try:
            # Get y-tick labels
            ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
            # Should contain pi symbols or "pi" text
            label_text = " ".join(ytick_labels)
            assert "π" in label_text or "pi" in label_text.lower(), (
                f"Expected pi-based y-axis labels, got: {ytick_labels}"
            )
        finally:
            plt.close("all")

    def test_shows_fit_line_when_result_provided(self) -> None:
        """Should show fitted line when result is provided and show_fit=True."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import (
            compute_phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = compute_phase_precession(positions, phases)
        ax = plot_phase_precession(positions, phases, result=result, show_fit=True)

        try:
            # Should have line plots (fitted lines)
            lines = ax.get_lines()
            assert len(lines) >= 2, (
                f"Expected at least 2 fitted lines (doubled), got {len(lines)}"
            )
        finally:
            plt.close("all")

    def test_no_fit_line_when_show_fit_false(self) -> None:
        """Should not show fitted line when show_fit=False."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import (
            compute_phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        phases = rng.uniform(0, 2 * np.pi, 100)

        result = compute_phase_precession(positions, phases)
        ax = plot_phase_precession(positions, phases, result=result, show_fit=False)

        try:
            # Should have no line plots
            lines = ax.get_lines()
            assert len(lines) == 0, f"Expected no fitted lines, got {len(lines)}"
        finally:
            plt.close("all")

    def test_no_fit_line_when_result_none(self) -> None:
        """Should not show fitted line when result is None."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases, result=None)

        try:
            # Should have no line plots
            lines = ax.get_lines()
            assert len(lines) == 0, f"Expected no fitted lines, got {len(lines)}"
        finally:
            plt.close("all")

    def test_position_label_customization(self) -> None:
        """Should use custom position label on x-axis."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases, position_label="Distance (cm)")

        try:
            xlabel = ax.get_xlabel()
            assert "Distance (cm)" in xlabel, f"Expected custom xlabel, got: {xlabel}"
        finally:
            plt.close("all")

    def test_marker_size_customization(self) -> None:
        """Should respect marker_size parameter."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        custom_size = 50.0
        ax = plot_phase_precession(positions, phases, marker_size=custom_size)

        try:
            collections = ax.collections
            assert len(collections) >= 1
            # Check marker sizes
            sizes = collections[0].get_sizes()
            assert np.allclose(sizes, custom_size), (
                f"Expected marker size {custom_size}, got {sizes[0]}"
            )
        finally:
            plt.close("all")

    def test_marker_alpha_customization(self) -> None:
        """Should respect marker_alpha parameter."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        custom_alpha = 0.3
        ax = plot_phase_precession(positions, phases, marker_alpha=custom_alpha)

        try:
            collections = ax.collections
            assert len(collections) >= 1
            # Alpha is stored in face colors (RGBA)
            face_colors = collections[0].get_facecolors()
            if len(face_colors) > 0:
                alpha = face_colors[0, 3]  # 4th element is alpha
                assert np.isclose(alpha, custom_alpha, atol=0.05), (
                    f"Expected alpha {custom_alpha}, got {alpha}"
                )
        finally:
            plt.close("all")

    def test_scatter_kwargs_passed(self) -> None:
        """scatter_kwargs should be passed to scatter plot."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(
            positions, phases, scatter_kwargs={"marker": "s", "edgecolors": "red"}
        )

        try:
            collections = ax.collections
            assert len(collections) >= 1
            # Check edge colors
            edge_colors = collections[0].get_edgecolors()
            if len(edge_colors) > 0:
                # Red color should have R=1, G=0, B=0
                assert edge_colors[0, 0] > 0.9, "Expected red edge color"
        finally:
            plt.close("all")

    def test_line_kwargs_passed(self) -> None:
        """line_kwargs should be passed to fitted line."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import (
            compute_phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = compute_phase_precession(positions, phases)
        ax = plot_phase_precession(
            positions,
            phases,
            result=result,
            show_fit=True,
            line_kwargs={"linestyle": "--", "linewidth": 3.0},
        )

        try:
            lines = ax.get_lines()
            assert len(lines) >= 1
            # Check line width
            assert lines[0].get_linewidth() == 3.0, (
                f"Expected linewidth 3.0, got {lines[0].get_linewidth()}"
            )
        finally:
            plt.close("all")

    def test_doubled_note_annotation(self) -> None:
        """Should show annotation explaining doubled phase axis when show_doubled_note=True."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases, show_doubled_note=True)

        try:
            # Check for text annotation
            texts = ax.texts
            found_note = False
            for text in texts:
                text_str = text.get_text().lower()
                if "doubled" in text_str or "twice" in text_str or "same" in text_str:
                    found_note = True
                    break
            assert found_note, "Expected annotation explaining doubled phase axis"
        finally:
            plt.close("all")

    def test_no_doubled_note_when_disabled(self) -> None:
        """Should not show doubled note when show_doubled_note=False."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases, show_doubled_note=False)

        try:
            # Should have no text annotations about doubling
            texts = ax.texts
            for text in texts:
                text_str = text.get_text().lower()
                assert "doubled" not in text_str and "twice" not in text_str, (
                    "Found doubled note when disabled"
                )
        finally:
            plt.close("all")

    def test_fit_line_doubled_for_both_phase_copies(self) -> None:
        """Fitted line should appear in both the lower and upper phase regions."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import (
            compute_phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = compute_phase_precession(positions, phases)
        ax = plot_phase_precession(positions, phases, result=result, show_fit=True)

        try:
            lines = ax.get_lines()
            # Should have 2 lines (one for each phase copy)
            assert len(lines) == 2, f"Expected 2 fitted lines, got {len(lines)}"

            # Lines should be offset by 2pi in y
            y_data_0 = lines[0].get_ydata()
            y_data_1 = lines[1].get_ydata()

            # One should be in 0-2pi range, other in 2pi-4pi range
            mean_y_0 = np.mean(y_data_0)
            mean_y_1 = np.mean(y_data_1)
            y_diff = abs(mean_y_1 - mean_y_0)

            assert np.isclose(y_diff, 2 * np.pi, atol=0.5), (
                f"Expected lines offset by 2pi, got difference of {y_diff}"
            )
        finally:
            plt.close("all")

    def test_fit_line_follows_the_data_with_a_field_range(self) -> None:
        """The drawn fit is the fitted phase at each plotted position."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import (
            compute_phase_precession,
            plot_phase_precession,
        )

        positions = np.linspace(20, 60, 100)
        true_phase = (1.0 - 0.15 * (positions - 20)) % (2 * np.pi)
        result = compute_phase_precession(
            positions, true_phase, position_range=(20.0, 60.0), n_shuffles=0
        )
        ax = plot_phase_precession(positions, true_phase, result=result)

        try:
            line_x, line_y = ax.get_lines()[0].get_data()
            expected = (1.0 - 0.15 * (np.asarray(line_x) - 20)) % (2 * np.pi)
            error = np.angle(np.exp(1j * (np.asarray(line_y) - expected)))
            assert np.max(np.abs(error)) < 1e-3
        finally:
            plt.close("all")

    def test_handles_phases_in_radians(self) -> None:
        """Should correctly handle phases in radians."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.phase_precession import plot_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        # Phases in radians [0, 2pi]
        phases = rng.uniform(0, 2 * np.pi, 50)

        ax = plot_phase_precession(positions, phases)

        try:
            collections = ax.collections
            assert len(collections) >= 1
            offsets = collections[0].get_offsets()
            # Y values should be in [0, 4pi] range (doubled)
            y_values = offsets[:, 1]
            assert np.all(y_values >= 0), "Y values should be >= 0"
            assert np.all(y_values <= 4 * np.pi + 0.1), "Y values should be <= 4pi"
        finally:
            plt.close("all")

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched array lengths should raise ValueError with clear message."""
        from neurospatial.encoding.phase_precession import plot_phase_precession

        positions = np.linspace(0, 50, 10)
        phases = np.linspace(0, 2 * np.pi, 5)

        with pytest.raises(ValueError, match="same length"):
            plot_phase_precession(positions, phases)


class TestPhasePrecessionSlopeMagnitude:
    """Recover the precession slope magnitude (not just its sign) and pin the
    position-range normalization contract.

    Existing tests only assert ``slope < 0``; a scale error in the regressor
    would pass those. These simulate a noisy linear phase-position relationship
    with a known slope and assert the recovered slope matches in magnitude.
    """

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def test_slope_magnitude_recovered(self):
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(3)
        positions = np.linspace(0, 100, 200)
        true_slope = -0.1  # rad per position unit
        phases = self._wrap(true_slope * positions + 1.0 + rng.normal(0, 0.2, 200))

        # -0.1 rad/unit over 100 units is 1.6 cycles across the range, more
        # than the default one-cycle search: widen it explicitly.
        result = compute_phase_precession(
            positions, phases, slope_bounds=(-4 * np.pi, 4 * np.pi)
        )
        assert np.isclose(result.slope, true_slope, atol=0.02)

    def test_slope_sign_and_magnitude_for_positive_slope(self):
        from neurospatial.encoding.phase_precession import compute_phase_precession

        rng = np.random.default_rng(3)
        positions = np.linspace(0, 100, 200)
        true_slope = 0.05
        phases = self._wrap(true_slope * positions + 1.0 + rng.normal(0, 0.2, 200))

        result = compute_phase_precession(positions, phases)
        assert np.isclose(result.slope, true_slope, atol=0.02)

    def test_position_range_normalization(self):
        from neurospatial.encoding.phase_precession import compute_phase_precession

        # slope is in radians per position unit whatever range the fit is
        # normalized over; normalized_slope is the phase change across that
        # range. -0.05 rad/unit over 100 units is -5 rad, inside the default
        # one-cycle search.
        rng = np.random.default_rng(3)
        positions = np.linspace(0, 100, 200)
        true_slope = -0.05
        phases = self._wrap(true_slope * positions + 1.0 + rng.normal(0, 0.2, 200))

        default = compute_phase_precession(positions, phases, n_shuffles=0)
        field = compute_phase_precession(
            positions, phases, position_range=(0.0, 100.0), n_shuffles=0
        )

        assert np.isclose(default.slope, true_slope, atol=0.02)
        assert np.isclose(field.slope, true_slope, atol=0.02)
        assert np.isclose(field.normalized_slope, true_slope * 100.0, atol=0.5)


class TestPhasePrecessionShufflePvalue:
    """p_value is a shuffle p-value at the fitted slope; correlation is descriptive."""

    def test_phase_precession_p_value_tracks_fitted_slope(
        self, precessing_spikes
    ) -> None:
        from neurospatial.encoding.phase_precession import compute_phase_precession

        d = precessing_spikes
        n_shuffles = 200
        true_fit = compute_phase_precession(
            d["positions"], d["phases"], n_shuffles=n_shuffles, rng=0
        )
        assert true_fit.p_value < 0.05
        assert true_fit.slope < 0

        # Pin the SHUFFLE mechanics: a permutation p-value with +1 smoothing
        # (count+1)/(n_shuffles+1) can never fall below 1/(n_shuffles+1). The
        # OLD slope-free circular_linear_correlation p-value returned exactly
        # 0.0 on this strong fixture, violating this floor — so reverting p_value
        # to that analytic value would fail here.
        smoothing_floor = 1.0 / (n_shuffles + 1)
        assert true_fit.p_value >= smoothing_floor

        # Shuffled phases against the same positions destroy the relationship.
        shuffled = compute_phase_precession(
            d["positions"], d["phase_shuffled"], n_shuffles=n_shuffles, rng=0
        )
        assert shuffled.p_value > 0.5

    def test_phase_precession_p_value_deterministic(self, precessing_spikes) -> None:
        from neurospatial.encoding.phase_precession import compute_phase_precession

        d = precessing_spikes
        a = compute_phase_precession(d["positions"], d["phases"], n_shuffles=200, rng=7)
        b = compute_phase_precession(d["positions"], d["phases"], n_shuffles=200, rng=7)
        assert a.p_value == b.p_value

    def test_phase_precession_correlation_is_descriptive(
        self, precessing_spikes
    ) -> None:
        from neurospatial.encoding.phase_precession import compute_phase_precession
        from neurospatial.stats.circular import circular_linear_correlation

        d = precessing_spikes
        result = compute_phase_precession(
            d["positions"], d["phases"], n_shuffles=200, rng=0
        )
        assert 0.0 <= result.correlation <= 1.0
        wrapped = d["phases"] % (2 * np.pi)
        expected, _ = circular_linear_correlation(
            angles=wrapped, linear_values=d["positions"]
        )
        assert result.correlation == pytest.approx(expected, rel=1e-9)


class TestHasPhasePrecessionValidation:
    """has_phase_precession input checks and cost."""

    def test_phases_in_degrees_read_as_radians_warn(self, precessing_spikes) -> None:
        from neurospatial.encoding.phase_precession import compute_phase_precession

        d = precessing_spikes
        with pytest.warns(UserWarning, match='angle_unit="deg"'):
            compute_phase_precession(
                d["positions"], np.degrees(d["phases"]), n_shuffles=0
            )

    def test_phases_in_degrees_with_angle_unit_do_not_warn(
        self, precessing_spikes
    ) -> None:
        import warnings

        from neurospatial.encoding.phase_precession import compute_phase_precession

        d = precessing_spikes
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_phase_precession(
                d["positions"],
                np.degrees(d["phases"]),
                angle_unit="deg",
                n_shuffles=0,
            )

    def test_has_phase_precession_raises_on_length_mismatch(self) -> None:
        from neurospatial.encoding.phase_precession import has_phase_precession

        positions = np.linspace(0, 50, 100)
        phases = np.linspace(0, 2 * np.pi, 90)
        with pytest.raises(ValueError):
            has_phase_precession(positions, phases)

    def test_has_phase_precession_default_work_is_bounded(
        self, precessing_spikes
    ) -> None:
        """The default call completes and uses a bounded shuffle count.

        Replaces an earlier wall-clock ``elapsed < 2.0`` assertion that was
        inherently flaky: the suite runs under ``pytest -n auto`` (xdist), which
        saturates every core, so a sub-second op wall-clocks to 2.5-3.7s on CI
        runners and the bound failed across all Linux/Windows jobs. The real
        intent -- guard against a pathologically expensive default (e.g. an
        accidental huge ``n_shuffles``) -- is enforced deterministically here by
        bounding the default shuffle count and confirming the call returns.
        """
        import inspect

        from neurospatial.encoding.phase_precession import has_phase_precession

        default_shuffles = (
            inspect.signature(has_phase_precession).parameters["n_shuffles"].default
        )
        assert isinstance(default_shuffles, int)
        assert default_shuffles <= 1000

        d = precessing_spikes
        result = has_phase_precession(d["positions"], d["phases"], rng=0)
        assert isinstance(result, bool)
