"""
Tests for encoding/head_direction.py module.

These tests verify:
1. All head direction symbols are importable from encoding.head_direction
2. All head direction symbols are importable from encoding/__init__.py
3. Re-exports from stats.circular are available
4. Basic functionality works through the re-exports
"""

from __future__ import annotations

import numpy as np


class TestImportsFromEncodingHeadDirection:
    """Test imports from encoding.head_direction module."""

    def test_import_HeadDirectionMetrics(self) -> None:
        """Test that HeadDirectionMetrics is importable from encoding.head_direction."""
        from neurospatial.encoding.head_direction import HeadDirectionMetrics

        assert HeadDirectionMetrics is not None

    def test_import_head_direction_tuning_curve(self) -> None:
        """Test that head_direction_tuning_curve is importable from encoding.head_direction."""
        from neurospatial.encoding.head_direction import head_direction_tuning_curve

        assert callable(head_direction_tuning_curve)

    def test_import_head_direction_metrics(self) -> None:
        """Test that head_direction_metrics is importable from encoding.head_direction."""
        from neurospatial.encoding.head_direction import head_direction_metrics

        assert callable(head_direction_metrics)

    def test_import_is_head_direction_cell(self) -> None:
        """Test that is_head_direction_cell is importable from encoding.head_direction."""
        from neurospatial.encoding.head_direction import is_head_direction_cell

        assert callable(is_head_direction_cell)

    def test_import_plot_head_direction_tuning(self) -> None:
        """Test that plot_head_direction_tuning is importable from encoding.head_direction."""
        from neurospatial.encoding.head_direction import plot_head_direction_tuning

        assert callable(plot_head_direction_tuning)


class TestReexportsFromStatsCircular:
    """Test re-exports from stats.circular module."""

    def test_import_rayleigh_test(self) -> None:
        """Test that rayleigh_test is re-exported from encoding.head_direction."""
        from neurospatial.encoding.head_direction import rayleigh_test

        assert callable(rayleigh_test)

    def test_import_mean_resultant_length(self) -> None:
        """Test that mean_resultant_length is re-exported from encoding.head_direction."""
        from neurospatial.encoding.head_direction import mean_resultant_length

        assert callable(mean_resultant_length)

    def test_import_circular_mean(self) -> None:
        """Test that circular_mean is re-exported from encoding.head_direction."""
        from neurospatial.encoding.head_direction import circular_mean

        assert callable(circular_mean)


class TestImportsFromEncodingInit:
    """Test imports from encoding/__init__.py."""

    def test_import_HeadDirectionMetrics_from_encoding(self) -> None:
        """Test that HeadDirectionMetrics is importable from encoding."""
        from neurospatial.encoding import HeadDirectionMetrics

        assert HeadDirectionMetrics is not None

    def test_import_head_direction_tuning_curve_from_encoding(self) -> None:
        """Test that head_direction_tuning_curve is importable from encoding."""
        from neurospatial.encoding import head_direction_tuning_curve

        assert callable(head_direction_tuning_curve)

    def test_import_head_direction_metrics_from_encoding(self) -> None:
        """Test that head_direction_metrics is importable from encoding."""
        from neurospatial.encoding import head_direction_metrics

        assert callable(head_direction_metrics)

    def test_import_is_head_direction_cell_from_encoding(self) -> None:
        """Test that is_head_direction_cell is importable from encoding."""
        from neurospatial.encoding import is_head_direction_cell

        assert callable(is_head_direction_cell)

    def test_import_plot_head_direction_tuning_from_encoding(self) -> None:
        """Test that plot_head_direction_tuning is importable from encoding."""
        from neurospatial.encoding import plot_head_direction_tuning

        assert callable(plot_head_direction_tuning)

    def test_import_rayleigh_test_from_encoding(self) -> None:
        """Test that rayleigh_test is importable from encoding."""
        from neurospatial.encoding import rayleigh_test

        assert callable(rayleigh_test)

    def test_import_mean_resultant_length_from_encoding(self) -> None:
        """Test that mean_resultant_length is importable from encoding."""
        from neurospatial.encoding import mean_resultant_length

        assert callable(mean_resultant_length)

    def test_import_circular_mean_from_encoding(self) -> None:
        """Test that circular_mean is importable from encoding."""
        from neurospatial.encoding import circular_mean

        assert callable(circular_mean)


class TestModuleStructure:
    """Test module structure and __all__ exports."""

    def test_module_has_all_attribute(self) -> None:
        """Test that encoding.head_direction has __all__ defined."""
        from neurospatial.encoding import head_direction

        assert hasattr(head_direction, "__all__")
        assert isinstance(head_direction.__all__, list)

    def test_all_contains_expected_symbols(self) -> None:
        """Test that __all__ contains all expected exports."""
        from neurospatial.encoding import head_direction

        expected = {
            # From metrics/head_direction.py
            "HeadDirectionMetrics",
            "head_direction_metrics",
            "head_direction_tuning_curve",
            "is_head_direction_cell",
            "plot_head_direction_tuning",
            # Re-exports from stats/circular.py
            "rayleigh_test",
            "mean_resultant_length",
            "circular_mean",
        }
        assert set(head_direction.__all__) == expected


class TestFunctionality:
    """Test basic functionality works through re-exports."""

    def test_head_direction_tuning_curve_basic(self) -> None:
        """Test head_direction_tuning_curve returns correct shapes."""
        from neurospatial.encoding.head_direction import head_direction_tuning_curve

        # Create sample data: 10 seconds at 30 Hz
        rng = np.random.default_rng(42)
        position_times = np.linspace(0, 10, 300)
        head_directions = rng.uniform(0, 360, 300)
        spike_times = rng.uniform(0, 10, 50)

        bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
        )

        # 360 / 30 = 12 bins
        assert len(bin_centers) == 12
        assert len(firing_rates) == 12
        assert np.all(firing_rates >= 0)

    def test_head_direction_metrics_basic(self) -> None:
        """Test head_direction_metrics returns HeadDirectionMetrics."""
        from neurospatial.encoding.head_direction import (
            HeadDirectionMetrics,
            head_direction_metrics,
        )

        # Create synthetic tuning curve with a peak at 90 degrees
        n_bins = 36
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        peak_direction = np.pi / 2  # 90 degrees
        firing_rates = 10.0 * np.exp(
            -((bin_centers - peak_direction) ** 2) / (2 * 0.5**2)
        )
        firing_rates += 0.1  # Small baseline

        metrics = head_direction_metrics(bin_centers, firing_rates)

        assert isinstance(metrics, HeadDirectionMetrics)
        assert hasattr(metrics, "preferred_direction")
        assert hasattr(metrics, "mean_vector_length")
        assert hasattr(metrics, "is_hd_cell")

    def test_is_head_direction_cell_basic(self) -> None:
        """Test is_head_direction_cell returns bool."""
        from neurospatial.encoding.head_direction import is_head_direction_cell

        # Create non-HD cell data (uniform firing)
        rng = np.random.default_rng(42)
        position_times = np.linspace(0, 10, 300)
        head_directions = rng.uniform(0, 2 * np.pi, 300)
        spike_times = rng.uniform(0, 10, 50)

        result = is_head_direction_cell(
            head_directions, spike_times, position_times, bin_size=0.1
        )

        assert isinstance(result, bool)

    def test_rayleigh_test_reexport(self) -> None:
        """Test rayleigh_test works through re-export."""
        from neurospatial.encoding.head_direction import rayleigh_test

        # Test with uniform angles (should NOT be significant)
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        stat, pval = rayleigh_test(angles)

        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert pval > 0.05  # Uniform distribution should not be significant

    def test_mean_resultant_length_reexport(self) -> None:
        """Test mean_resultant_length works through re-export."""
        from neurospatial.encoding.head_direction import mean_resultant_length

        # Test with clustered angles (should have high MVL)
        angles = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
        mvl = mean_resultant_length(angles)

        assert isinstance(mvl, float)
        assert 0 <= mvl <= 1
        assert mvl > 0.9  # Clustered angles should have high MVL

    def test_circular_mean_reexport(self) -> None:
        """Test circular_mean works through re-export."""
        from neurospatial.encoding.head_direction import circular_mean

        # Test with angles clustered around 0
        angles = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
        mean_angle = circular_mean(angles)

        assert isinstance(mean_angle, float)
        assert -np.pi <= mean_angle <= np.pi
        assert np.abs(mean_angle) < 0.1  # Should be close to 0
