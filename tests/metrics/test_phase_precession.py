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
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.5,
            pval=0.01,
            mean_resultant_length=0.8,
        )
        assert result.is_significant(alpha=0.05) is True

    def test_is_significant_above_alpha(self) -> None:
        """Should return False when p-value is above alpha."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.2,
            pval=0.10,
            mean_resultant_length=0.5,
        )
        assert result.is_significant(alpha=0.05) is False

    def test_interpretation_significant_precession(self) -> None:
        """Interpretation should indicate significant precession."""
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
        assert "SIGNIFICANT" in interp
        assert "PRECESSION" in interp

    def test_interpretation_recession(self) -> None:
        """Interpretation should indicate recession for positive slope."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=0.1,
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.5,
            pval=0.01,
            mean_resultant_length=0.8,
        )
        interp = result.interpretation()
        assert "RECESSION" in interp

    def test_str_returns_interpretation(self) -> None:
        """str() should return interpretation."""
        from neurospatial.encoding.phase_precession import PhasePrecessionResult

        result = PhasePrecessionResult(
            slope=-0.1,
            slope_units="rad/cm",
            offset=1.0,
            correlation=0.5,
            pval=0.01,
            mean_resultant_length=0.8,
        )
        assert str(result) == result.interpretation()


class TestPhasePrecession:
    """Tests for phase_precession() function."""

    def test_negative_slope_for_precession(self) -> None:
        """Should return negative slope for phase precession data."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        # Simulate phase precession: phase decreases with position
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = phase_precession(positions, phases)

        # Should detect negative slope
        assert result.slope < 0

    def test_correlation_in_valid_range(self) -> None:
        """Correlation should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = phase_precession(positions, phases)

        assert 0 <= result.correlation <= 1

    def test_pvalue_in_valid_range(self) -> None:
        """P-value should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = phase_precession(positions, phases)

        assert 0 <= result.pval <= 1

    def test_offset_in_valid_range(self) -> None:
        """Offset should be in [0, 2pi]."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = phase_precession(positions, phases)

        # Offset should be in [0, 2pi] (circular mean of residuals)
        assert 0 <= result.offset <= 2 * np.pi

    def test_mean_resultant_length_in_valid_range(self) -> None:
        """Mean resultant length should be in [0, 1]."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = phase_precession(positions, phases)

        assert 0 <= result.mean_resultant_length <= 1

    def test_insufficient_spikes_raises(self) -> None:
        """Should raise ValueError for insufficient spikes."""
        from neurospatial.encoding.phase_precession import phase_precession

        positions = np.array([1.0, 2.0])  # Only 2 spikes
        phases = np.array([0.5, 1.5])

        with pytest.raises(ValueError, match="at least"):
            phase_precession(positions, phases, min_spikes=10)

    def test_mismatched_lengths_raises(self) -> None:
        """Should raise ValueError for mismatched array lengths."""
        from neurospatial.encoding.phase_precession import phase_precession

        positions = np.array([1.0, 2.0, 3.0])
        phases = np.array([0.5, 1.5])

        with pytest.raises(ValueError, match="same length"):
            phase_precession(positions, phases)

    def test_degree_input(self) -> None:
        """Should handle phases in degrees."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 50)
        phases_deg = rng.uniform(0, 360, 50)

        result = phase_precession(positions, phases_deg, angle_unit="deg")

        assert 0 <= result.correlation <= 1

    def test_position_range_normalization(self) -> None:
        """position_range should normalize positions and change slope units."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        # position_range normalizes to [0, 1] - documented behavior, not a warning
        result = phase_precession(positions, phases, position_range=(0.0, 100.0))

        assert "normalized" in result.slope_units.lower()

    def test_invalid_position_range_raises(self) -> None:
        """Invalid position_range should raise ValueError."""
        from neurospatial.encoding.phase_precession import phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        with pytest.raises(ValueError, match="pos_max > pos_min"):
            phase_precession(positions, phases, position_range=(100.0, 0.0))


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
        # Strong negative slope
        true_slope = -0.15
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.1, 200)) % (
            2 * np.pi
        )

        result = has_phase_precession(
            positions, phases, alpha=0.05, min_correlation=0.2
        )

        assert result is True

    def test_random_data_usually_false(self) -> None:
        """Random data should usually not show phase precession."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, 50)
        phases = rng.uniform(0, 2 * np.pi, 50)

        result = has_phase_precession(
            positions, phases, alpha=0.05, min_correlation=0.2
        )

        assert result is False

    def test_positive_slope_returns_false(self) -> None:
        """Should return False for positive slope (recession, not precession)."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 200)
        # Strong positive slope (recession)
        true_slope = 0.15
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.1, 200)) % (
            2 * np.pi
        )

        result = has_phase_precession(positions, phases)

        # Should return False because slope is positive
        assert result is False

    def test_handles_insufficient_data(self) -> None:
        """Should return False for insufficient data (not raise)."""
        from neurospatial.encoding.phase_precession import has_phase_precession

        positions = np.array([1.0, 2.0])
        phases = np.array([0.5, 1.5])

        # Should return False, not raise
        result = has_phase_precession(positions, phases)

        assert result is False

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
            phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = phase_precession(positions, phases)
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
            phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        phases = rng.uniform(0, 2 * np.pi, 100)

        result = phase_precession(positions, phases)
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
            phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = phase_precession(positions, phases)
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
            phase_precession,
            plot_phase_precession,
        )

        rng = np.random.default_rng(42)
        positions = np.linspace(0, 50, 100)
        true_slope = -0.1
        phases = (np.pi + true_slope * positions + rng.normal(0, 0.2, 100)) % (
            2 * np.pi
        )

        result = phase_precession(positions, phases)
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
