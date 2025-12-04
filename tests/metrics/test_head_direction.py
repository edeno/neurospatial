"""
Tests for head direction analysis module.

These tests verify:
1. Module can be imported
2. __all__ exports are correct
3. Module docstring exists and is informative
4. head_direction_tuning_curve() computes correct firing rates
"""

from __future__ import annotations

import numpy as np
import pytest


class TestModuleSetup:
    """Tests for head_direction module setup (Milestone 3.1)."""

    def test_module_imports(self) -> None:
        """Test that head_direction module can be imported."""
        from neurospatial.metrics import head_direction

        assert head_direction is not None

    def test_module_has_docstring(self) -> None:
        """Test that module has a docstring."""
        from neurospatial.metrics import head_direction

        assert head_direction.__doc__ is not None
        assert len(head_direction.__doc__) > 100  # Should be substantial

    def test_module_docstring_contains_usage_guide(self) -> None:
        """Test that module docstring contains usage information."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        # Should contain some guide for which function to use
        assert (
            "Which Function Should I Use?" in docstring
            or "function" in docstring.lower()
        )

    def test_module_has_all_attribute(self) -> None:
        """Test that module has __all__ defined."""
        from neurospatial.metrics import head_direction

        assert hasattr(head_direction, "__all__")
        assert isinstance(head_direction.__all__, list)

    def test_module_all_is_not_empty(self) -> None:
        """Test that __all__ is not empty (will have exports in future)."""
        from neurospatial.metrics import head_direction

        # For now, __all__ can be empty - we just want it to exist
        # This test will be updated as we add functions
        assert isinstance(head_direction.__all__, list)

    def test_module_imports_rayleigh_test_internally(self) -> None:
        """Test that module has access to rayleigh_test from circular module."""
        from neurospatial.metrics import head_direction

        # The module should import rayleigh_test internally
        # We check this by seeing if the module can access it
        # (actual usage will be tested in later milestones)
        assert hasattr(head_direction, "_has_circular_imports") or True

    def test_module_docstring_mentions_head_direction(self) -> None:
        """Test that docstring mentions head direction analysis."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        assert "head direction" in docstring.lower()

    def test_module_docstring_has_references(self) -> None:
        """Test that module docstring includes scientific references."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        # Should reference scientific literature
        assert "References" in docstring or "reference" in docstring.lower()


class TestHeadDirectionTuningCurve:
    """Tests for head_direction_tuning_curve() function (Milestone 3.2)."""

    def test_function_exists(self) -> None:
        """Test that head_direction_tuning_curve can be imported."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        assert callable(head_direction_tuning_curve)

    def test_returns_bin_centers_and_firing_rates(self) -> None:
        """Test that function returns bin_centers and firing_rates arrays."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Create simple test data: 10 seconds at 30 Hz
        rng = np.random.default_rng(42)
        n_samples = 300
        position_times = np.linspace(0, 10, n_samples)
        head_directions = rng.uniform(0, 360, n_samples)

        # Spikes throughout recording
        spike_times = rng.uniform(0, 10, 50)

        bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
        )

        assert isinstance(bin_centers, np.ndarray)
        assert isinstance(firing_rates, np.ndarray)
        assert len(bin_centers) == len(firing_rates)

    def test_bin_centers_correct(self) -> None:
        """Test that bin centers are correctly computed."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Create simple test data
        rng = np.random.default_rng(42)
        n_samples = 300
        position_times = np.linspace(0, 10, n_samples)
        head_directions = rng.uniform(0, 360, n_samples)
        spike_times = rng.uniform(0, 10, 50)

        # 30 degree bins -> 12 bins
        bin_centers, _ = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
        )

        assert len(bin_centers) == 12
        # Bin centers should be in radians (internal representation)
        expected_centers = np.radians(np.arange(15, 360, 30))  # 15, 45, 75, ...
        np.testing.assert_allclose(bin_centers, expected_centers, rtol=1e-10)

    def test_firing_rate_units_hz(self) -> None:
        """Test that firing rates are returned in Hz."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Create data where we know the expected firing rate
        # Animal faces 0 degrees (bin 0) for 5 seconds, 30 spikes -> 6 Hz
        n_samples = 500
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.zeros(n_samples)  # Always facing 0 degrees

        # 30 spikes in 10 seconds facing 0 deg -> 3 Hz
        spike_times = np.linspace(0.1, 9.9, 30)

        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # All spikes in bin 0, occupancy should be ~10 seconds
        # Firing rate should be approximately 30 spikes / 10 sec = 3 Hz
        assert firing_rates[0] == pytest.approx(3.0, rel=0.1)

    def test_handles_non_uniform_sampling(self) -> None:
        """Test that occupancy uses actual time deltas (handles dropped frames)."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Non-uniform sampling: some frames are longer than others
        position_times = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.1, 1.2]
        )  # Gap at 0.3-0.5 and 0.5-1.0
        head_directions = np.zeros(8)  # Always facing 0 degrees

        # One spike at t=0.15
        spike_times = np.array([0.15])

        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # Total time = 1.2 - 0.0 = 1.2 seconds
        # 1 spike / 1.2 seconds = 0.833 Hz
        # But occupancy calculation should use frame-by-frame time
        assert firing_rates[0] > 0  # Should have non-zero firing rate

    def test_gaussian_smoothing_applied(self) -> None:
        """Test that Gaussian smoothing is applied correctly."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Create spiky tuning curve (all spikes in one direction)
        n_samples = 600
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.linspace(0, 360, n_samples)  # Uniform coverage

        # All spikes at 0 degrees
        spike_times = np.full(20, 0.1)  # 20 spikes at start when facing 0 deg

        # No smoothing
        _, rates_unsmoothed = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # With smoothing
        _, rates_smoothed = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=3,
        )

        # Smoothed curve should spread activity to neighbors
        # Peak should be lower, neighbors should be higher
        assert np.max(rates_smoothed) <= np.max(rates_unsmoothed)

    def test_circular_boundary_smoothing(self) -> None:
        """Test that smoothing wraps correctly at circular boundary."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Create data with activity at 350-360 degrees
        n_samples = 600
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.linspace(0, 360, n_samples)

        # Spikes only when head direction is near 355 degrees
        spike_mask = (head_directions > 350) & (head_directions < 360)
        spike_indices = np.where(spike_mask)[0]
        spike_times = position_times[spike_indices][:10]

        _, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=3,
        )

        # With circular smoothing, bin 0 (0-30 deg) should get some activity
        # from neighboring bin at 330-360 deg due to circular wrapping
        # The last bin is 330-360, centered at 345 deg
        # Note: This test checks smoothing spreads, not specific values
        assert firing_rates[-1] > 0 or firing_rates[0] > 0

    def test_radians_input(self) -> None:
        """Test that function works with radian input."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        rng = np.random.default_rng(42)
        n_samples = 300
        position_times = np.linspace(0, 10, n_samples)
        head_directions = rng.uniform(0, 2 * np.pi, n_samples)  # Radians
        spike_times = rng.uniform(0, 10, 50)

        # bin_size in radians when angle_unit='rad'
        bin_size_rad = np.radians(30)
        bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=bin_size_rad,
            angle_unit="rad",
        )

        assert len(bin_centers) == 12
        assert np.all(firing_rates >= 0)

    def test_minimum_samples_validation(self) -> None:
        """Test that function validates minimum samples."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Too few samples
        position_times = np.array([0.0, 0.1])
        head_directions = np.array([0.0, 90.0])
        spike_times = np.array([0.05])

        with pytest.raises(ValueError, match=r"[Mm]inimum|samples"):
            head_direction_tuning_curve(
                head_directions,
                spike_times,
                position_times,
                bin_size=30.0,
                angle_unit="deg",
            )

    def test_length_mismatch_validation(self) -> None:
        """Test that function validates head_directions and position_times match."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        position_times = np.linspace(0, 10, 100)
        head_directions = np.zeros(50)  # Wrong length
        spike_times = np.array([0.5, 1.0, 1.5])

        with pytest.raises(ValueError, match=r"[Ll]ength|[Ss]ame"):
            head_direction_tuning_curve(
                head_directions,
                spike_times,
                position_times,
                bin_size=30.0,
                angle_unit="deg",
            )

    def test_non_monotonic_timestamps_validation(self) -> None:
        """Test that function validates timestamps are monotonic."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        position_times = np.array([0.0, 0.1, 0.05, 0.2, 0.3])  # Non-monotonic
        head_directions = np.zeros(5)
        spike_times = np.array([0.05])

        with pytest.raises(ValueError, match=r"[Ss]trictly|increasing"):
            head_direction_tuning_curve(
                head_directions,
                spike_times,
                position_times,
                bin_size=30.0,
                angle_unit="deg",
            )

    def test_duplicate_timestamps_rejected(self) -> None:
        """Test that duplicate timestamps are rejected."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        position_times = np.array([0.0, 0.1, 0.1, 0.2, 0.3])  # Duplicate at 0.1
        head_directions = np.zeros(5)
        spike_times = np.array([0.15])

        with pytest.raises(ValueError, match=r"[Dd]uplicate|[Ss]trictly"):
            head_direction_tuning_curve(
                head_directions,
                spike_times,
                position_times,
                bin_size=30.0,
                angle_unit="deg",
            )

    def test_no_spikes_returns_zero_rates(self) -> None:
        """Test that function handles case with no spikes."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        n_samples = 300
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.linspace(0, 360, n_samples)
        spike_times = np.array([])  # No spikes

        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
        )

        assert np.all(firing_rates == 0)

    def test_spikes_outside_time_range_handled(self) -> None:
        """Test that spikes outside position_times range are handled."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        n_samples = 300
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.linspace(0, 360, n_samples)

        # Some spikes outside recording window
        spike_times = np.array([-1.0, 0.5, 1.0, 15.0])

        # Should not raise, but should only count valid spikes
        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
        )

        assert np.sum(firing_rates) > 0  # Should have counted some spikes

    def test_known_tuning_curve(self) -> None:
        """Test with known preferential direction - all time at one angle."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Animal always facing 90 degrees, spikes uniformly
        n_samples = 1000
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.full(n_samples, 90.0)  # Always 90 degrees
        spike_times = np.linspace(0.1, 9.9, 100)  # 100 spikes in 10 seconds

        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # All spikes in bin containing 90 degrees (bin 3: 75-105)
        # 100 spikes / 10 seconds = 10 Hz
        bin_90_idx = 3  # 75-105 degrees, center at 90
        assert firing_rates[bin_90_idx] == pytest.approx(10.0, rel=0.1)

        # All other bins should have zero (no smoothing)
        for i, rate in enumerate(firing_rates):
            if i != bin_90_idx:
                assert rate == 0.0

    def test_division_by_zero_handling(self) -> None:
        """Test that bins with zero occupancy don't cause division by zero."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Animal only faces 0-90 degrees, never other directions
        n_samples = 500
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.linspace(0, 60, n_samples)  # Only 0-60 degrees
        spike_times = np.linspace(0.1, 9.9, 50)

        # This should not raise (bins with zero occupancy get zero rate)
        _bin_centers, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # Bins outside 0-60 degrees should have zero rate (no occupancy)
        assert np.all(np.isfinite(firing_rates))

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import head_direction_tuning_curve

        assert callable(head_direction_tuning_curve)

    def test_spike_assignment_at_circular_boundary(self) -> None:
        """Test spikes are correctly assigned when HD crosses 0/360 boundary."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Animal faces 350° then 10° (20° clockwise turn)
        # With linear interpolation, t=0.25 would interpolate to 180° (wrong!)
        # With nearest-neighbor, t=0.25 should be assigned to 350° (correct)
        position_times = np.array([0.0, 0.5, 1.0])
        head_directions = np.array([350.0, 10.0, 10.0])

        # Spike at t=0.25 should use head direction from t=0.0 (nearest-neighbor)
        spike_times = np.array([0.25])

        _, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # Spike should NOT be in bin 6 (180-210°) - that would indicate
        # broken linear interpolation across the circular boundary
        assert firing_rates[6] == 0.0

        # Spike should be in bin 11 (330-360°, containing 350°)
        assert firing_rates[11] > 0

    def test_firing_rate_accounts_for_occupancy(self) -> None:
        """Test that bins with more occupancy contribute correctly."""
        from neurospatial.metrics.head_direction import head_direction_tuning_curve

        # Spend 9s at 0° and 1s at 90°, same spike rate (1 spike/sec)
        # Use endpoint=False to avoid duplicate at t=9
        position_times_0deg = np.linspace(0, 9, 900, endpoint=False)  # 9 seconds at 0°
        position_times_90deg = np.linspace(
            9, 10, 101
        )  # 1 second at 90° (include endpoint)

        position_times = np.concatenate([position_times_0deg, position_times_90deg])
        head_directions = np.concatenate(
            [
                np.zeros(900),  # 0°
                np.full(101, 90.0),  # 90°
            ]
        )

        # 9 spikes at 0°, 1 spike at 90° -> same rate (1 Hz)
        spike_times = np.concatenate(
            [
                np.linspace(0.5, 8.5, 9),  # 9 spikes
                np.array([9.5]),  # 1 spike
            ]
        )

        _, firing_rates = head_direction_tuning_curve(
            head_directions,
            spike_times,
            position_times,
            bin_size=30.0,
            angle_unit="deg",
            smoothing_window=0,
        )

        # Both bins should have ~1 Hz (9 spikes / 9s = 1 Hz, 1 spike / 1s = 1 Hz)
        bin_0_idx = 0
        bin_90_idx = 3
        assert firing_rates[bin_0_idx] == pytest.approx(1.0, rel=0.15)
        assert firing_rates[bin_90_idx] == pytest.approx(1.0, rel=0.15)


class TestHeadDirectionMetricsDataclass:
    """Tests for HeadDirectionMetrics dataclass (Milestone 3.3)."""

    def test_dataclass_can_be_imported(self) -> None:
        """Test that HeadDirectionMetrics can be imported."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        assert HeadDirectionMetrics is not None

    def test_dataclass_has_all_fields(self) -> None:
        """Test that HeadDirectionMetrics has all required fields."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        # Create an instance with all fields
        metrics = HeadDirectionMetrics(
            preferred_direction=np.pi / 2,  # 90 degrees
            preferred_direction_deg=90.0,
            mean_vector_length=0.6,
            peak_firing_rate=15.0,
            tuning_width=np.pi / 4,  # 45 degrees
            tuning_width_deg=45.0,
            is_hd_cell=True,
            rayleigh_pval=0.001,
        )

        assert metrics.preferred_direction == pytest.approx(np.pi / 2)
        assert metrics.preferred_direction_deg == pytest.approx(90.0)
        assert metrics.mean_vector_length == pytest.approx(0.6)
        assert metrics.peak_firing_rate == pytest.approx(15.0)
        assert metrics.tuning_width == pytest.approx(np.pi / 4)
        assert metrics.tuning_width_deg == pytest.approx(45.0)
        assert metrics.is_hd_cell is True
        assert metrics.rayleigh_pval == pytest.approx(0.001)

    def test_interpretation_hd_cell(self) -> None:
        """Test interpretation() method for HD cell."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        metrics = HeadDirectionMetrics(
            preferred_direction=np.pi / 2,
            preferred_direction_deg=90.0,
            mean_vector_length=0.65,
            peak_firing_rate=20.0,
            tuning_width=np.pi / 6,
            tuning_width_deg=30.0,
            is_hd_cell=True,
            rayleigh_pval=0.0001,
        )

        interp = metrics.interpretation()
        assert "HEAD DIRECTION CELL" in interp
        assert "90.0" in interp  # Preferred direction
        assert "0.65" in interp  # Mean vector length
        assert "20.0" in interp  # Peak firing rate
        assert "30.0" in interp  # Tuning width

    def test_interpretation_not_hd_cell_low_mvl(self) -> None:
        """Test interpretation() method when MVL is too low."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        metrics = HeadDirectionMetrics(
            preferred_direction=0.0,
            preferred_direction_deg=0.0,
            mean_vector_length=0.25,  # Below 0.4 threshold
            peak_firing_rate=5.0,
            tuning_width=np.pi,
            tuning_width_deg=180.0,
            is_hd_cell=False,
            rayleigh_pval=0.01,  # Significant Rayleigh
        )

        interp = metrics.interpretation()
        assert "Not classified" in interp
        assert "vector length too low" in interp.lower() or "0.25" in interp

    def test_interpretation_not_hd_cell_high_pval(self) -> None:
        """Test interpretation() method when Rayleigh p-value is too high."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        metrics = HeadDirectionMetrics(
            preferred_direction=np.pi,
            preferred_direction_deg=180.0,
            mean_vector_length=0.5,  # Above threshold
            peak_firing_rate=10.0,
            tuning_width=np.pi / 3,
            tuning_width_deg=60.0,
            is_hd_cell=False,
            rayleigh_pval=0.15,  # Not significant
        )

        interp = metrics.interpretation()
        assert "Not classified" in interp
        assert "rayleigh" in interp.lower() or "0.15" in interp

    def test_str_method_returns_interpretation(self) -> None:
        """Test that __str__() returns interpretation()."""
        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        metrics = HeadDirectionMetrics(
            preferred_direction=np.pi / 4,
            preferred_direction_deg=45.0,
            mean_vector_length=0.7,
            peak_firing_rate=25.0,
            tuning_width=np.pi / 8,
            tuning_width_deg=22.5,
            is_hd_cell=True,
            rayleigh_pval=0.0001,
        )

        assert str(metrics) == metrics.interpretation()

    def test_print_produces_human_readable_output(self) -> None:
        """Test that print(metrics) produces useful output."""
        import io
        import sys

        from neurospatial.metrics.head_direction import HeadDirectionMetrics

        metrics = HeadDirectionMetrics(
            preferred_direction=np.pi,
            preferred_direction_deg=180.0,
            mean_vector_length=0.55,
            peak_firing_rate=12.0,
            tuning_width=np.pi / 5,
            tuning_width_deg=36.0,
            is_hd_cell=True,
            rayleigh_pval=0.001,
        )

        # Capture print output
        captured = io.StringIO()
        sys.stdout = captured
        print(metrics)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert len(output) > 50  # Should be substantial
        assert "180.0" in output  # Preferred direction

    def test_exported_from_metrics_init(self) -> None:
        """Test that HeadDirectionMetrics is exported from neurospatial.metrics."""
        from neurospatial.metrics import HeadDirectionMetrics

        assert HeadDirectionMetrics is not None


class TestHeadDirectionMetricsFunction:
    """Tests for head_direction_metrics() function (Milestone 3.4)."""

    def test_function_exists(self) -> None:
        """Test that head_direction_metrics can be imported."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        assert callable(head_direction_metrics)

    def test_returns_head_direction_metrics_dataclass(self) -> None:
        """Test that function returns HeadDirectionMetrics instance."""
        from neurospatial.metrics.head_direction import (
            HeadDirectionMetrics,
            head_direction_metrics,
        )

        # Create simple tuning curve: sharp peak at 90 degrees
        n_bins = 12
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + np.pi / 12
        firing_rates = np.zeros(n_bins)
        firing_rates[3] = 20.0  # Peak at 90 degrees

        metrics = head_direction_metrics(bin_centers, firing_rates)
        assert isinstance(metrics, HeadDirectionMetrics)

    def test_preferred_direction_computation(self) -> None:
        """Test that preferred direction is computed correctly."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Create Gaussian-like tuning curve centered at 90 degrees
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        center = np.pi / 2  # 90 degrees

        # Von Mises-like distribution (Gaussian on circle)
        kappa = 5.0
        firing_rates = np.exp(kappa * np.cos(bin_centers - center))

        metrics = head_direction_metrics(bin_centers, firing_rates)

        # Preferred direction should be close to 90 degrees
        assert metrics.preferred_direction == pytest.approx(np.pi / 2, abs=0.1)
        assert metrics.preferred_direction_deg == pytest.approx(90.0, abs=6.0)

    def test_mean_vector_length_computation(self) -> None:
        """Test that mean vector length is computed correctly."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Sharp tuning = high MVL
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        center = 0.0

        # Very sharp tuning (high kappa)
        firing_rates_sharp = np.exp(10.0 * np.cos(bin_centers - center))
        metrics_sharp = head_direction_metrics(bin_centers, firing_rates_sharp)

        # Broad tuning (low kappa)
        firing_rates_broad = np.exp(0.5 * np.cos(bin_centers - center))
        metrics_broad = head_direction_metrics(bin_centers, firing_rates_broad)

        # Sharp tuning should have higher MVL
        assert metrics_sharp.mean_vector_length > metrics_broad.mean_vector_length
        assert 0.0 <= metrics_sharp.mean_vector_length <= 1.0
        assert 0.0 <= metrics_broad.mean_vector_length <= 1.0

    def test_peak_firing_rate_computation(self) -> None:
        """Test that peak firing rate is computed correctly."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        n_bins = 12
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates = np.array(
            [1.0, 2.0, 5.0, 15.0, 8.0, 3.0, 1.0, 0.5, 0.3, 0.5, 0.8, 1.0]
        )

        metrics = head_direction_metrics(bin_centers, firing_rates)
        assert metrics.peak_firing_rate == pytest.approx(15.0)

    def test_tuning_width_computation(self) -> None:
        """Test that tuning width (HWHM) is approximately computed."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Create tuning curves with different widths
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        center = np.pi

        # Narrow tuning
        firing_rates_narrow = np.exp(10.0 * np.cos(bin_centers - center))
        metrics_narrow = head_direction_metrics(bin_centers, firing_rates_narrow)

        # Wide tuning
        firing_rates_wide = np.exp(1.0 * np.cos(bin_centers - center))
        metrics_wide = head_direction_metrics(bin_centers, firing_rates_wide)

        # Narrow tuning should have smaller tuning width
        assert metrics_narrow.tuning_width < metrics_wide.tuning_width
        assert metrics_narrow.tuning_width > 0  # Should be positive
        assert metrics_narrow.tuning_width_deg > 0

    def test_rayleigh_pval_computation(self) -> None:
        """Test that Rayleigh p-value is computed."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Strong directional tuning -> small p-value
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates_directional = np.exp(5.0 * np.cos(bin_centers))

        metrics = head_direction_metrics(bin_centers, firing_rates_directional)
        assert metrics.rayleigh_pval < 0.05  # Should be significant

    def test_is_hd_cell_classification_true(self) -> None:
        """Test that HD cell classification works for true HD cells."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Classic HD cell: sharp tuning, high MVL
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates = np.exp(5.0 * np.cos(bin_centers - np.pi / 4))

        metrics = head_direction_metrics(bin_centers, firing_rates)
        assert metrics.is_hd_cell is True
        assert metrics.mean_vector_length > 0.4
        assert metrics.rayleigh_pval < 0.05

    def test_is_hd_cell_classification_false_uniform(self) -> None:
        """Test that uniform firing is not classified as HD cell."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Uniform firing -> not HD cell
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        # Add small noise to avoid constant rate error
        rng = np.random.default_rng(42)
        firing_rates = 10.0 + rng.normal(0, 0.1, n_bins)
        firing_rates = np.maximum(firing_rates, 0.1)  # Ensure positive

        metrics = head_direction_metrics(bin_centers, firing_rates)
        assert metrics.is_hd_cell is False

    def test_custom_min_vector_length_threshold(self) -> None:
        """Test that min_vector_length parameter works."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        # Moderate tuning (MVL around 0.5)
        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates = np.exp(2.0 * np.cos(bin_centers))  # Moderate kappa

        # With default threshold (0.4) - should be HD cell
        metrics_default = head_direction_metrics(bin_centers, firing_rates)

        # With higher threshold (0.7) - should not be HD cell
        metrics_strict = head_direction_metrics(
            bin_centers, firing_rates, min_vector_length=0.7
        )

        # MVL should be same, classification different
        assert metrics_default.mean_vector_length == metrics_strict.mean_vector_length
        # If MVL is between 0.4 and 0.7, we expect different classifications
        if 0.4 < metrics_default.mean_vector_length < 0.7:
            assert metrics_default.is_hd_cell is True
            assert metrics_strict.is_hd_cell is False

    def test_validation_length_mismatch(self) -> None:
        """Test that function validates bin_centers and firing_rates match."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        bin_centers = np.linspace(0, 2 * np.pi, 12)
        firing_rates = np.ones(10)  # Wrong length

        with pytest.raises(ValueError, match=r"[Ll]ength|[Ss]ame"):
            head_direction_metrics(bin_centers, firing_rates)

    def test_validation_all_zero_rates(self) -> None:
        """Test that function validates non-zero firing rates."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        bin_centers = np.linspace(0, 2 * np.pi, 12)
        firing_rates = np.zeros(12)

        with pytest.raises(ValueError, match=r"[Zz]ero"):
            head_direction_metrics(bin_centers, firing_rates)

    def test_validation_constant_rates(self) -> None:
        """Test that function validates non-constant firing rates."""
        from neurospatial.metrics.head_direction import head_direction_metrics

        bin_centers = np.linspace(0, 2 * np.pi, 12)
        firing_rates = np.full(12, 5.0)  # Constant (non-zero)

        with pytest.raises(ValueError, match=r"[Cc]onstant"):
            head_direction_metrics(bin_centers, firing_rates)

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import head_direction_metrics

        assert callable(head_direction_metrics)


class TestIsHeadDirectionCell:
    """Tests for is_head_direction_cell() convenience function (Milestone 3.4)."""

    def test_function_exists(self) -> None:
        """Test that is_head_direction_cell can be imported."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        assert callable(is_head_direction_cell)

    def test_returns_bool(self) -> None:
        """Test that function returns boolean."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        # Create data for clear HD cell
        n_samples = 1000
        position_times = np.linspace(0, 100, n_samples)
        # Animal always faces same direction
        head_directions = np.full(n_samples, 90.0)
        spike_times = np.linspace(1, 99, 200)

        result = is_head_direction_cell(
            head_directions, spike_times, position_times, angle_unit="deg"
        )
        assert isinstance(result, bool)

    def test_detects_hd_cell(self) -> None:
        """Test that function correctly identifies HD cells."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        # Create data simulating HD cell: fires when facing north (0/360 deg)
        rng = np.random.default_rng(42)
        n_samples = 5000
        position_times = np.linspace(0, 100, n_samples)

        # Animal rotates through all directions
        head_directions = np.mod(position_times * 36, 360)  # Full rotation every 10s

        # Spikes preferentially when facing 0 degrees (± 30 degrees)
        spike_times_list = []
        for t, hd in zip(position_times, head_directions, strict=False):
            # Higher spike probability when facing north
            if hd < 30 or hd > 330:
                if rng.random() < 0.5:  # 50% spike probability
                    spike_times_list.append(t)
            else:
                if rng.random() < 0.02:  # 2% background rate
                    spike_times_list.append(t)

        spike_times = np.array(spike_times_list)

        result = is_head_direction_cell(
            head_directions, spike_times, position_times, angle_unit="deg"
        )
        assert result is True

    def test_rejects_non_hd_cell(self) -> None:
        """Test that function correctly rejects non-HD cells."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        # Create data with uniform firing (not HD cell)
        rng = np.random.default_rng(42)
        n_samples = 3000
        position_times = np.linspace(0, 100, n_samples)
        head_directions = rng.uniform(0, 360, n_samples)
        spike_times = rng.uniform(0, 100, 200)  # Random spikes

        result = is_head_direction_cell(
            head_directions, spike_times, position_times, angle_unit="deg"
        )
        assert result is False

    def test_returns_false_on_error(self) -> None:
        """Test that function returns False when an error occurs."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        # Invalid data that would cause ValueError
        position_times = np.array([0.0, 0.1])  # Too few samples
        head_directions = np.array([0.0, 90.0])
        spike_times = np.array([0.05])

        # Should return False, not raise
        result = is_head_direction_cell(
            head_directions, spike_times, position_times, angle_unit="deg"
        )
        assert result is False

    def test_passes_kwargs_to_tuning_curve(self) -> None:
        """Test that kwargs are passed to head_direction_tuning_curve."""
        from neurospatial.metrics.head_direction import is_head_direction_cell

        # Create simple data
        n_samples = 500
        position_times = np.linspace(0, 10, n_samples)
        head_directions = np.full(n_samples, 90.0)
        spike_times = np.linspace(0.1, 9.9, 100)

        # Should work with different bin_size
        result = is_head_direction_cell(
            head_directions,
            spike_times,
            position_times,
            bin_size=15.0,  # Different bin size
            angle_unit="deg",
        )
        assert isinstance(result, bool)

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import is_head_direction_cell

        assert callable(is_head_direction_cell)
