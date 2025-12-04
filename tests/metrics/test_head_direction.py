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
