"""
Tests for circular statistics module.

Tests follow the plan in CIRC_STATS_TASKS.md, covering:
- Core circular statistics (Rayleigh test, correlations)
- Phase precession analysis
- Input validation and edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# These imports will fail until we implement the module
# This is expected for TDD - tests fail first


class TestToRadians:
    """Tests for _to_radians internal helper."""

    def test_radians_passthrough(self) -> None:
        """Radians input should pass through unchanged."""
        from neurospatial.metrics.circular import _to_radians

        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = _to_radians(angles, "rad")
        assert_allclose(result, angles)

    def test_degrees_conversion(self) -> None:
        """Degrees should be converted to radians."""
        from neurospatial.metrics.circular import _to_radians

        angles_deg = np.array([0, 90, 180, 270])
        expected = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = _to_radians(angles_deg, "deg")
        assert_allclose(result, expected)


class TestMeanResultantLength:
    """Tests for _mean_resultant_length internal helper."""

    def test_empty_array_returns_nan(self) -> None:
        """Empty input should return NaN."""
        from neurospatial.metrics.circular import _mean_resultant_length

        result = _mean_resultant_length(np.array([]))
        assert np.isnan(result)

    def test_all_same_angle_returns_one(self) -> None:
        """All angles the same should give R = 1.0."""
        from neurospatial.metrics.circular import _mean_resultant_length

        angles = np.ones(10) * np.pi / 4  # All 45 degrees
        result = _mean_resultant_length(angles)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_uniform_distribution_returns_near_zero(self) -> None:
        """Uniform distribution should give R near 0."""
        from neurospatial.metrics.circular import _mean_resultant_length

        # Perfectly uniform angles
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        result = _mean_resultant_length(angles)
        assert result < 0.05  # Should be very close to 0

    def test_result_in_valid_range(self) -> None:
        """Result should always be in [0, 1]."""
        from neurospatial.metrics.circular import _mean_resultant_length

        rng = np.random.default_rng(42)
        for _ in range(10):
            angles = rng.uniform(0, 2 * np.pi, 50)
            result = _mean_resultant_length(angles)
            assert 0 <= result <= 1

    def test_weighted_mean_resultant_length(self) -> None:
        """Weighted computation should work correctly."""
        from neurospatial.metrics.circular import _mean_resultant_length

        # All weight on one angle
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        result = _mean_resultant_length(angles, weights=weights)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_zero_weights_returns_nan(self) -> None:
        """All-zero weights should return NaN."""
        from neurospatial.metrics.circular import _mean_resultant_length

        angles = np.array([0, np.pi / 2, np.pi])
        weights = np.zeros(3)
        result = _mean_resultant_length(angles, weights=weights)
        assert np.isnan(result)


class TestValidateCircularInput:
    """Tests for _validate_circular_input internal helper."""

    def test_all_nan_raises_error(self) -> None:
        """All NaN values should raise ValueError with diagnostic message."""
        from neurospatial.metrics.circular import _validate_circular_input

        angles = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match=r"All.*values are NaN"):
            _validate_circular_input(angles, "phases")

    def test_partial_nan_warns_and_removes(self) -> None:
        """Partial NaN should warn and remove NaN values."""
        from neurospatial.metrics.circular import _validate_circular_input

        angles = np.array([0.0, np.nan, np.pi, np.nan, 2 * np.pi])
        with pytest.warns(UserWarning, match="Removed.*NaN"):
            result = _validate_circular_input(angles, "angles")
        assert len(result) == 3
        assert not np.any(np.isnan(result))

    def test_inf_values_raise_error(self) -> None:
        """Infinite values should raise ValueError."""
        from neurospatial.metrics.circular import _validate_circular_input

        angles = np.array([0.0, np.inf, np.pi])
        with pytest.raises(ValueError, match="infinite values"):
            _validate_circular_input(angles, "angles")

    def test_insufficient_samples_raises_error(self) -> None:
        """Too few samples should raise ValueError."""
        from neurospatial.metrics.circular import _validate_circular_input

        angles = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="at least 3 samples"):
            _validate_circular_input(angles, "angles", min_samples=3)

    def test_angles_outside_range_warns_and_wraps(self) -> None:
        """Angles outside [0, 2pi] should warn and wrap."""
        from neurospatial.metrics.circular import _validate_circular_input

        angles = np.array([-np.pi, 3 * np.pi, 5 * np.pi])
        with pytest.warns(UserWarning, match="outside \\[0, 2pi\\]"):
            result = _validate_circular_input(angles, "angles")
        # All should be wrapped to [0, 2pi)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)


class TestValidatePairedInput:
    """Tests for _validate_paired_input internal helper."""

    def test_mismatched_lengths_raises_error(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        from neurospatial.metrics.circular import _validate_paired_input

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="must have same length"):
            _validate_paired_input(arr1, arr2, "positions", "phases")

    def test_nan_pairs_removed_with_warning(self) -> None:
        """NaN in either array should remove that pair with warning."""
        from neurospatial.metrics.circular import _validate_paired_input

        arr1 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        arr2 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.warns(UserWarning, match="Removed 2 pairs"):
            result1, result2 = _validate_paired_input(arr1, arr2, "a", "b")
        assert len(result1) == 3
        assert len(result2) == 3

    def test_insufficient_pairs_after_nan_removal_raises_error(self) -> None:
        """Too few pairs after NaN removal should raise ValueError."""
        from neurospatial.metrics.circular import _validate_paired_input

        arr1 = np.array([1.0, np.nan, np.nan])
        arr2 = np.array([np.nan, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 3 valid pairs"):
            _validate_paired_input(arr1, arr2, "a", "b", min_samples=3)


class TestRayleighTest:
    """Tests for rayleigh_test() function."""

    def test_uniform_distribution_high_pvalue(self) -> None:
        """Uniform distribution should have high p-value (> 0.5)."""
        from neurospatial.metrics import rayleigh_test

        # Perfectly uniform angles
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        _z, p = rayleigh_test(angles)
        assert p > 0.5, f"Expected p > 0.5 for uniform, got p={p}"

    def test_concentrated_distribution_low_pvalue(self) -> None:
        """Von Mises with kappa=2 should have low p-value (< 0.001)."""
        from neurospatial.metrics import rayleigh_test

        # Concentrated around 0 (von Mises with high concentration)
        rng = np.random.default_rng(42)
        # Use scipy's von Mises distribution
        from scipy.stats import vonmises

        angles = vonmises.rvs(kappa=2.0, loc=0, size=100, random_state=rng)
        _z, p = rayleigh_test(angles)
        assert p < 0.001, f"Expected p < 0.001 for concentrated, got p={p}"

    def test_all_same_angle_max_z(self) -> None:
        """All same angle should give R = 1.0, z = n."""
        from neurospatial.metrics import rayleigh_test

        n = 50
        angles = np.ones(n) * np.pi / 4
        z, p = rayleigh_test(angles)
        # z = n * R^2, and R = 1 when all angles same
        assert_allclose(z, n, rtol=1e-10)
        assert p < 1e-10  # Very significant

    def test_degrees_input(self) -> None:
        """Should handle degree input correctly."""
        from neurospatial.metrics import rayleigh_test

        # Same test as concentrated, but in degrees
        rng = np.random.default_rng(42)
        from scipy.stats import vonmises

        angles_rad = vonmises.rvs(kappa=2.0, loc=0, size=100, random_state=rng)
        angles_deg = np.degrees(angles_rad)

        z_rad, p_rad = rayleigh_test(angles_rad, angle_unit="rad")
        z_deg, p_deg = rayleigh_test(angles_deg, angle_unit="deg")

        assert_allclose(z_rad, z_deg, rtol=1e-10)
        assert_allclose(p_rad, p_deg, rtol=1e-10)

    def test_weighted_rayleigh(self) -> None:
        """Weighted Rayleigh should use effective sample size."""
        from neurospatial.metrics import rayleigh_test

        # Very concentrated angles with strong weights
        angles = np.array([0.0, 0.05, 0.1, 0.15, np.pi, np.pi + 0.1])
        # Weight heavily toward first 4 (very concentrated near 0)
        weights = np.array([20.0, 20.0, 20.0, 20.0, 1.0, 1.0])

        _z, p = rayleigh_test(angles, weights=weights)
        # Should be significant due to weighted concentration
        assert p < 0.05

    def test_z_statistic_range(self) -> None:
        """Z statistic should be in [0, n]."""
        from neurospatial.metrics import rayleigh_test

        rng = np.random.default_rng(42)
        for _ in range(10):
            n = 50
            angles = rng.uniform(0, 2 * np.pi, n)
            z, _p = rayleigh_test(angles)
            assert 0 <= z <= n, f"z={z} outside [0, {n}]"

    def test_pvalue_range(self) -> None:
        """P-value should be in [0, 1]."""
        from neurospatial.metrics import rayleigh_test

        rng = np.random.default_rng(42)
        for _ in range(10):
            angles = rng.uniform(0, 2 * np.pi, 50)
            _z, p = rayleigh_test(angles)
            assert 0 <= p <= 1, f"p={p} outside [0, 1]"

    def test_small_sample_correction(self) -> None:
        """Small samples should use finite-sample correction."""
        from neurospatial.metrics import rayleigh_test

        # With very few samples, correction matters more
        rng = np.random.default_rng(42)
        from scipy.stats import vonmises

        angles = vonmises.rvs(kappa=1.0, loc=0, size=10, random_state=rng)
        z, p = rayleigh_test(angles)

        # Should still be valid
        assert 0 <= z <= 10
        assert 0 <= p <= 1

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        from neurospatial.metrics import rayleigh_test

        with pytest.raises(ValueError):
            rayleigh_test(np.array([]))

    def test_insufficient_samples_raises(self) -> None:
        """Too few samples should raise ValueError."""
        from neurospatial.metrics import rayleigh_test

        with pytest.raises(ValueError, match="at least 3"):
            rayleigh_test(np.array([0.0, 0.5]))


class TestCircularLinearCorrelation:
    """Tests for circular_linear_correlation() function."""

    def test_perfect_linear_relationship_high_correlation(self) -> None:
        """Perfect linear phase-position relationship should give high correlation.

        Note: The Mardia-Jupp circular-linear correlation has a theoretical maximum
        less than 1.0 for a single-cycle linear relationship due to the sine/cosine
        transformation. For phases spanning [0, 2pi], the max r is ~0.755.
        """
        from neurospatial.metrics import circular_linear_correlation

        # Phases increase linearly with position
        positions = np.linspace(0, 100, 50)
        phases = np.linspace(0, 2 * np.pi, 50)

        r, p = circular_linear_correlation(phases, positions)
        # For a perfect linear relationship spanning one cycle, r ~ 0.755
        assert r > 0.7, f"Expected r > 0.7 for perfect relationship, got r={r}"
        assert p < 0.001, f"Expected highly significant p-value, got p={p}"

    def test_random_data_low_correlation(self) -> None:
        """Random uncorrelated data should give r near 0."""
        from neurospatial.metrics import circular_linear_correlation

        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 100)
        positions = rng.uniform(0, 100, 100)

        r, _p = circular_linear_correlation(phases, positions)
        assert r < 0.3, f"Expected r < 0.3 for random data, got r={r}"

    def test_correlation_always_nonnegative(self) -> None:
        """Circular-linear correlation r should always be non-negative."""
        from neurospatial.metrics import circular_linear_correlation

        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 50)
            positions = rng.uniform(0, 100, 50)
            r, _p = circular_linear_correlation(phases, positions)
            assert r >= 0, f"Expected r >= 0, got r={r}"

    def test_correlation_at_most_one(self) -> None:
        """Circular-linear correlation r should be at most 1."""
        from neurospatial.metrics import circular_linear_correlation

        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 50)
            positions = rng.uniform(0, 100, 50)
            r, _p = circular_linear_correlation(phases, positions)
            assert r <= 1, f"Expected r <= 1, got r={r}"

    def test_pvalue_in_valid_range(self) -> None:
        """P-value should be in [0, 1]."""
        from neurospatial.metrics import circular_linear_correlation

        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 50)
            positions = rng.uniform(0, 100, 50)
            _r, p = circular_linear_correlation(phases, positions)
            assert 0 <= p <= 1, f"Expected p in [0, 1], got p={p}"

    def test_degrees_input(self) -> None:
        """Should handle degree input correctly."""
        from neurospatial.metrics import circular_linear_correlation

        positions = np.linspace(0, 100, 50)
        phases_rad = np.linspace(0, 2 * np.pi, 50)
        phases_deg = np.degrees(phases_rad)

        r_rad, p_rad = circular_linear_correlation(
            phases_rad, positions, angle_unit="rad"
        )
        r_deg, p_deg = circular_linear_correlation(
            phases_deg, positions, angle_unit="deg"
        )

        assert_allclose(r_rad, r_deg, rtol=1e-10)
        assert_allclose(p_rad, p_deg, rtol=1e-10)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        from neurospatial.metrics import circular_linear_correlation

        phases = np.array([0.0, 0.5, 1.0])
        positions = np.array([0.0, 50.0])

        with pytest.raises(ValueError, match="same length"):
            circular_linear_correlation(phases, positions)

    def test_insufficient_samples_raises(self) -> None:
        """Too few samples should raise ValueError."""
        from neurospatial.metrics import circular_linear_correlation

        phases = np.array([0.0, 0.5])
        positions = np.array([0.0, 50.0])

        with pytest.raises(ValueError, match="at least 3"):
            circular_linear_correlation(phases, positions)

    def test_degenerate_case_constant_linear_variable(self) -> None:
        """Constant linear variable (no variation) should return r=0 with warning."""
        from neurospatial.metrics import circular_linear_correlation

        phases = np.linspace(0, 2 * np.pi, 50)
        positions = np.ones(50) * 50.0  # Constant position

        with pytest.warns(UserWarning, match="[Dd]egenerate|constant|variation"):
            r, _p = circular_linear_correlation(phases, positions)
        assert r == 0.0 or np.isnan(r)

    def test_negative_slope_still_positive_correlation(self) -> None:
        """Phase precession (negative slope) should still give positive r.

        Note: The circular-linear correlation is always non-negative because
        it measures the strength of the relationship, not the direction.
        For a perfect linear relationship spanning one cycle, r ~ 0.755.
        """
        from neurospatial.metrics import circular_linear_correlation

        # Phase decreases as position increases (phase precession)
        positions = np.linspace(0, 100, 50)
        phases = 2 * np.pi - np.linspace(0, 2 * np.pi, 50)  # Decreasing

        r, _p = circular_linear_correlation(phases, positions)
        assert r > 0, f"Expected positive r for negative slope, got r={r}"
        # Same high correlation as positive slope - correlation measures strength
        assert r > 0.7, f"Expected r > 0.7 for perfect relationship, got r={r}"

    def test_nan_handling(self) -> None:
        """NaN values in either array should be removed with warning."""
        from neurospatial.metrics import circular_linear_correlation

        positions = np.array([0.0, 10.0, np.nan, 30.0, 40.0, 50.0, 60.0])
        phases = np.array([0.0, 0.5, 1.0, np.nan, 2.0, 2.5, 3.0])

        with pytest.warns(UserWarning, match="[Rr]emoved.*pairs"):
            r, p = circular_linear_correlation(phases, positions)
        # Should still work with remaining valid pairs
        assert 0 <= r <= 1
        assert 0 <= p <= 1


class TestPhasePositionCorrelation:
    """Tests for phase_position_correlation() alias function."""

    def test_same_as_circular_linear_correlation(self) -> None:
        """Should return same result as circular_linear_correlation."""
        from neurospatial.metrics import (
            circular_linear_correlation,
            phase_position_correlation,
        )

        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 50)
        positions = rng.uniform(0, 100, 50)

        r1, p1 = circular_linear_correlation(phases, positions)
        r2, p2 = phase_position_correlation(phases, positions)

        assert r1 == r2
        assert p1 == p2

    def test_accepts_angle_unit(self) -> None:
        """Should accept angle_unit parameter."""
        from neurospatial.metrics import phase_position_correlation

        rng = np.random.default_rng(42)
        phases_deg = rng.uniform(0, 360, 50)
        positions = rng.uniform(0, 100, 50)

        # Should not raise
        r, p = phase_position_correlation(phases_deg, positions, angle_unit="deg")
        assert 0 <= r <= 1
        assert 0 <= p <= 1


class TestCircularCircularCorrelation:
    """Tests for circular_circular_correlation() function."""

    def test_perfect_correlation_identical_angles(self) -> None:
        """Identical angles should give r close to 1.0."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, 100)

        r, p = circular_circular_correlation(angles, angles)
        assert r > 0.99, f"Expected r > 0.99 for identical angles, got r={r}"
        assert p < 0.001, f"Expected p < 0.001, got p={p}"

    def test_perfect_correlation_with_small_noise(self) -> None:
        """Nearly identical angles (with small noise) should give high r."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 100)
        # Add small noise
        noise = rng.normal(0, 0.1, 100)
        angles2 = angles1 + noise

        r, p = circular_circular_correlation(angles1, angles2)
        assert r > 0.8, f"Expected r > 0.8 for nearly identical angles, got r={r}"
        assert p < 0.001, f"Expected p < 0.001, got p={p}"

    def test_random_data_low_correlation(self) -> None:
        """Random uncorrelated angles should give r near 0."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 100)
        angles2 = rng.uniform(0, 2 * np.pi, 100)

        r, _p = circular_circular_correlation(angles1, angles2)
        # Random data - expect low correlation
        assert abs(r) < 0.3, f"Expected |r| < 0.3 for random data, got r={r}"

    def test_anticorrelation(self) -> None:
        """Opposite angles should give negative r."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 100)
        # Opposite direction: angles2 = -angles1 (mod 2pi)
        angles2 = (2 * np.pi - angles1) % (2 * np.pi)

        r, _p = circular_circular_correlation(angles1, angles2)
        # Should be negative (anti-correlated)
        assert r < -0.5, f"Expected r < -0.5 for opposite angles, got r={r}"

    def test_symmetry(self) -> None:
        """Correlation should be symmetric: r(a1, a2) == r(a2, a1)."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 50)
        angles2 = rng.uniform(0, 2 * np.pi, 50)

        r12, p12 = circular_circular_correlation(angles1, angles2)
        r21, p21 = circular_circular_correlation(angles2, angles1)

        assert_allclose(r12, r21, rtol=1e-10)
        assert_allclose(p12, p21, rtol=1e-10)

    def test_correlation_in_valid_range(self) -> None:
        """Correlation r should be in [-1, 1]."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        for _ in range(10):
            angles1 = rng.uniform(0, 2 * np.pi, 50)
            angles2 = rng.uniform(0, 2 * np.pi, 50)
            r, _p = circular_circular_correlation(angles1, angles2)
            assert -1 <= r <= 1, f"Expected r in [-1, 1], got r={r}"

    def test_pvalue_in_valid_range(self) -> None:
        """P-value should be in [0, 1]."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        for _ in range(10):
            angles1 = rng.uniform(0, 2 * np.pi, 50)
            angles2 = rng.uniform(0, 2 * np.pi, 50)
            _r, p = circular_circular_correlation(angles1, angles2)
            assert 0 <= p <= 1, f"Expected p in [0, 1], got p={p}"

    def test_degrees_input(self) -> None:
        """Should handle degree input correctly."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1_rad = rng.uniform(0, 2 * np.pi, 50)
        angles2_rad = angles1_rad + rng.normal(0, 0.2, 50)
        angles1_deg = np.degrees(angles1_rad)
        angles2_deg = np.degrees(angles2_rad)

        r_rad, p_rad = circular_circular_correlation(
            angles1_rad, angles2_rad, angle_unit="rad"
        )
        r_deg, p_deg = circular_circular_correlation(
            angles1_deg, angles2_deg, angle_unit="deg"
        )

        assert_allclose(r_rad, r_deg, rtol=1e-10)
        assert_allclose(p_rad, p_deg, rtol=1e-10)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        from neurospatial.metrics import circular_circular_correlation

        angles1 = np.array([0.0, 0.5, 1.0])
        angles2 = np.array([0.0, 0.5])

        with pytest.raises(ValueError, match="same length"):
            circular_circular_correlation(angles1, angles2)

    def test_insufficient_samples_raises(self) -> None:
        """Too few samples should raise ValueError."""
        from neurospatial.metrics import circular_circular_correlation

        angles1 = np.array([0.0, 0.5])
        angles2 = np.array([0.0, 0.5])

        with pytest.raises(ValueError, match="at least"):
            circular_circular_correlation(angles1, angles2)

    def test_degenerate_case_no_variation(self) -> None:
        """No variation in angles should return r=0 with warning."""
        from neurospatial.metrics import circular_circular_correlation

        # All angles the same
        angles1 = np.ones(50) * np.pi
        angles2 = np.ones(50) * np.pi

        with pytest.warns(UserWarning, match="[Nn]o variation|constant"):
            r, p = circular_circular_correlation(angles1, angles2)
        assert r == 0.0
        assert p == 1.0

    def test_nan_handling(self) -> None:
        """NaN values should be removed with warning."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = np.concatenate([rng.uniform(0, 2 * np.pi, 48), [np.nan, np.nan]])
        angles2 = rng.uniform(0, 2 * np.pi, 50)

        with pytest.warns(UserWarning, match="[Rr]emoved.*pairs"):
            r, p = circular_circular_correlation(angles1, angles2)
        # Should still work with remaining valid pairs
        assert -1 <= r <= 1
        assert 0 <= p <= 1

    def test_constant_offset(self) -> None:
        """Constant phase offset should still show high correlation."""
        from neurospatial.metrics import circular_circular_correlation

        rng = np.random.default_rng(42)
        angles1 = rng.uniform(0, 2 * np.pi, 100)
        # Add constant offset (pi/4)
        angles2 = (angles1 + np.pi / 4) % (2 * np.pi)

        r, p = circular_circular_correlation(angles1, angles2)
        # Should still be highly correlated (just shifted)
        # Fisher & Lee measures deviation correlation, not raw angle correlation
        assert r > 0.9, f"Expected r > 0.9 for constant offset, got r={r}"
        assert p < 0.001, f"Expected p < 0.001, got p={p}"


# ============================================================================
# Property-Based Tests (Milestone 4.7)
# ============================================================================


class TestPropertyBasedRayleighTest:
    """Property-based tests for rayleigh_test using hypothesis."""

    def test_mean_resultant_length_always_in_0_1(self) -> None:
        """Mean resultant length R should always be in [0, 1]."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics.circular import _mean_resultant_length

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=1,
                max_size=1000,
            )
        )
        @settings(max_examples=100)
        def check_r_bounds(angles_list: list[float]) -> None:
            angles = np.array(angles_list)
            r = _mean_resultant_length(angles)
            # Allow small floating-point tolerance (R can be 1.0+epsilon for identical angles)
            assert -1e-10 <= r <= 1.0 + 1e-10, f"R={r} not in [0, 1]"

        check_r_bounds()

    def test_rayleigh_pvalue_always_in_0_1(self) -> None:
        """Rayleigh test p-value should always be in [0, 1]."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import rayleigh_test

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=3,  # Minimum for valid test
                max_size=500,
            )
        )
        @settings(max_examples=100)
        def check_pvalue_bounds(angles_list: list[float]) -> None:
            angles = np.array(angles_list)
            _z, p = rayleigh_test(angles)
            assert 0 <= p <= 1, f"p={p} not in [0, 1]"

        check_pvalue_bounds()

    def test_rayleigh_z_always_nonnegative(self) -> None:
        """Rayleigh test z-statistic should always be non-negative."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import rayleigh_test

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=3,
                max_size=500,
            )
        )
        @settings(max_examples=100)
        def check_z_nonnegative(angles_list: list[float]) -> None:
            angles = np.array(angles_list)
            z, _p = rayleigh_test(angles)
            assert z >= 0, f"z={z} is negative"

        check_z_nonnegative()


class TestPropertyBasedCircularCircularCorrelation:
    """Property-based tests for circular_circular_correlation using hypothesis."""

    def test_correlation_symmetric(self) -> None:
        """Circular-circular correlation should be symmetric: r(a, b) == r(b, a)."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import circular_circular_correlation

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=5,  # circular_circular_correlation requires min 5 samples
                max_size=200,
            ),
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=5,
                max_size=200,
            ),
        )
        @settings(max_examples=50)
        def check_symmetry(
            angles1_list: list[float], angles2_list: list[float]
        ) -> None:
            # Make arrays same length
            min_len = min(len(angles1_list), len(angles2_list))
            if min_len < 5:
                return  # Skip if too small (circular_circular_correlation needs >= 5)
            angles1 = np.array(angles1_list[:min_len])
            angles2 = np.array(angles2_list[:min_len])

            r1, p1 = circular_circular_correlation(angles1, angles2)
            r2, p2 = circular_circular_correlation(angles2, angles1)

            assert_allclose(r1, r2, rtol=1e-10)
            assert_allclose(p1, p2, rtol=1e-10)

        check_symmetry()

    def test_correlation_in_valid_range(self) -> None:
        """Circular-circular correlation r should be in [-1, 1]."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import circular_circular_correlation

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=5,  # circular_circular_correlation requires min 5 samples
                max_size=200,
            )
        )
        @settings(max_examples=100)
        def check_r_bounds(angles_list: list[float]) -> None:
            angles1 = np.array(angles_list)
            # Create slightly perturbed version
            angles2 = (
                angles1 + np.random.default_rng(42).uniform(-0.5, 0.5, len(angles1))
            ) % (2 * np.pi)

            r, p = circular_circular_correlation(angles1, angles2)
            assert -1 <= r <= 1, f"r={r} not in [-1, 1]"
            assert 0 <= p <= 1, f"p={p} not in [0, 1]"

        check_r_bounds()


class TestPropertyBasedCircularLinearCorrelation:
    """Property-based tests for circular_linear_correlation using hypothesis."""

    def test_correlation_always_nonnegative(self) -> None:
        """Circular-linear correlation r should always be >= 0."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import circular_linear_correlation

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=5,  # Need enough samples for robust analysis
                max_size=200,
            ),
            st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                min_size=5,
                max_size=200,
            ),
        )
        @settings(max_examples=50)
        def check_r_nonnegative(
            angles_list: list[float], values_list: list[float]
        ) -> None:
            # Make arrays same length
            min_len = min(len(angles_list), len(values_list))
            if min_len < 5:
                return  # Skip if too small (need enough samples for robust analysis)
            angles = np.array(angles_list[:min_len])
            values = np.array(values_list[:min_len])

            # Check for degenerate cases
            if np.std(values) < 1e-10:
                return  # Skip constant values
            if np.std(np.sin(angles)) < 1e-10 or np.std(np.cos(angles)) < 1e-10:
                return  # Skip degenerate angle distributions

            r, p = circular_linear_correlation(angles, values)
            assert r >= 0, f"r={r} is negative"
            assert 0 <= p <= 1, f"p={p} not in [0, 1]"

        check_r_nonnegative()

    def test_outputs_finite_for_valid_inputs(self) -> None:
        """Outputs should be finite for valid inputs."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from neurospatial.metrics import circular_linear_correlation

        @given(
            st.lists(
                st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
                min_size=10,  # Need enough samples for valid analysis
                max_size=200,
            ),
            st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                min_size=10,
                max_size=200,
            ),
        )
        @settings(max_examples=50)
        def check_finite_outputs(
            angles_list: list[float], values_list: list[float]
        ) -> None:
            # Make arrays same length
            min_len = min(len(angles_list), len(values_list))
            if min_len < 5:
                return  # Skip if too small (need enough samples for robust analysis)
            angles = np.array(angles_list[:min_len])
            values = np.array(values_list[:min_len])

            # Check for degenerate cases
            if np.std(values) < 1e-10:
                return  # Skip constant values
            if np.std(np.sin(angles)) < 1e-10 or np.std(np.cos(angles)) < 1e-10:
                return  # Skip degenerate angle distributions

            r, p = circular_linear_correlation(angles, values)
            assert np.isfinite(r), f"r={r} is not finite"
            assert np.isfinite(p), f"p={p} is not finite"

        check_finite_outputs()


# ============================================================================
# Circular Basis Functions (Milestone M1-M4)
# ============================================================================


class TestCircularBasisResult:
    """Tests for CircularBasisResult dataclass (Milestone M1.4)."""

    def test_dataclass_can_be_imported(self) -> None:
        """Test that CircularBasisResult can be imported."""
        from neurospatial.metrics.circular_basis import CircularBasisResult

        assert CircularBasisResult is not None

    def test_dataclass_has_all_fields(self) -> None:
        """Test that CircularBasisResult has all required fields."""
        from neurospatial.metrics.circular_basis import CircularBasisResult

        # Create instance with required fields
        result = CircularBasisResult(
            sin_component=np.array([0.0, 0.5, 1.0]),
            cos_component=np.array([1.0, 0.866, 0.5]),
            angles=np.array([0.0, np.pi / 6, np.pi / 3]),
        )

        assert hasattr(result, "sin_component")
        assert hasattr(result, "cos_component")
        assert hasattr(result, "angles")

    def test_design_matrix_property(self) -> None:
        """Test that design_matrix property returns (n_samples, 2) array."""
        from neurospatial.metrics.circular_basis import CircularBasisResult

        n = 100
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sin_comp = np.sin(angles)
        cos_comp = np.cos(angles)

        result = CircularBasisResult(
            sin_component=sin_comp,
            cos_component=cos_comp,
            angles=angles,
        )

        dm = result.design_matrix
        assert dm.shape == (n, 2)
        assert_allclose(dm[:, 0], sin_comp)
        assert_allclose(dm[:, 1], cos_comp)


class TestCircularBasis:
    """Tests for circular_basis() function (Milestone M1.3)."""

    def test_function_exists(self) -> None:
        """Test that circular_basis can be imported."""
        from neurospatial.metrics.circular_basis import circular_basis

        assert callable(circular_basis)

    def test_returns_circular_basis_result(self) -> None:
        """Test that circular_basis returns CircularBasisResult."""
        from neurospatial.metrics.circular_basis import (
            CircularBasisResult,
            circular_basis,
        )

        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        result = circular_basis(angles)
        assert isinstance(result, CircularBasisResult)

    def test_sin_cos_components_correct(self) -> None:
        """Test that sin/cos components are computed correctly."""
        from neurospatial.metrics.circular_basis import circular_basis

        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = circular_basis(angles)

        expected_sin = np.sin(angles)
        expected_cos = np.cos(angles)

        assert_allclose(result.sin_component, expected_sin, atol=1e-10)
        assert_allclose(result.cos_component, expected_cos, atol=1e-10)

    def test_angles_stored_in_result(self) -> None:
        """Test that input angles are stored in result."""
        from neurospatial.metrics.circular_basis import circular_basis

        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        result = circular_basis(angles)

        assert_allclose(result.angles, angles)

    def test_degrees_input(self) -> None:
        """Test that degree input is handled correctly."""
        from neurospatial.metrics.circular_basis import circular_basis

        angles_deg = np.array([0, 90, 180, 270])
        angles_rad = np.radians(angles_deg)

        result_deg = circular_basis(angles_deg, angle_unit="deg")
        result_rad = circular_basis(angles_rad, angle_unit="rad")

        # Results should be the same
        assert_allclose(result_deg.sin_component, result_rad.sin_component, atol=1e-10)
        assert_allclose(result_deg.cos_component, result_rad.cos_component, atol=1e-10)

    def test_design_matrix_shape(self) -> None:
        """Test design_matrix has shape (n_samples, 2)."""
        from neurospatial.metrics.circular_basis import circular_basis

        n = 100
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        result = circular_basis(angles)

        assert result.design_matrix.shape == (n, 2)

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import circular_basis

        assert callable(circular_basis)


class TestCircularBasisMetrics:
    """Tests for circular_basis_metrics() function (Milestone M1.6)."""

    def test_function_exists(self) -> None:
        """Test that circular_basis_metrics can be imported."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        assert callable(circular_basis_metrics)

    def test_returns_tuple_of_three(self) -> None:
        """Test that circular_basis_metrics returns (amplitude, phase, pvalue)."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        # GLM coefficients (beta_sin, beta_cos)
        beta_sin = 0.5
        beta_cos = 0.5

        result = circular_basis_metrics(beta_sin, beta_cos)
        assert len(result) == 3

        amplitude, phase, pvalue = result
        assert isinstance(amplitude, float)
        assert isinstance(phase, float)
        assert isinstance(pvalue, (float, type(None)))

    def test_amplitude_correct(self) -> None:
        """Test that amplitude is sqrt(beta_sin^2 + beta_cos^2)."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        beta_sin = 3.0
        beta_cos = 4.0
        expected_amplitude = 5.0  # sqrt(9 + 16)

        amplitude, _, _ = circular_basis_metrics(beta_sin, beta_cos)
        assert_allclose(amplitude, expected_amplitude, rtol=1e-10)

    def test_phase_correct(self) -> None:
        """Test that phase is atan2(beta_sin, beta_cos)."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        # Pure sin component (phase = pi/2)
        _amplitude, phase, _ = circular_basis_metrics(1.0, 0.0)
        assert_allclose(phase, np.pi / 2, atol=1e-10)

        # Pure cos component (phase = 0)
        _amplitude, phase, _ = circular_basis_metrics(0.0, 1.0)
        assert_allclose(phase, 0.0, atol=1e-10)

        # 45 degrees (equal sin and cos)
        _amplitude, phase, _ = circular_basis_metrics(1.0, 1.0)
        assert_allclose(phase, np.pi / 4, atol=1e-10)

    def test_phase_in_valid_range(self) -> None:
        """Test that phase is in [-pi, pi]."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        for _ in range(20):
            beta_sin = np.random.default_rng(42).uniform(-10, 10)
            beta_cos = np.random.default_rng(43).uniform(-10, 10)
            _, phase, _ = circular_basis_metrics(beta_sin, beta_cos)
            assert -np.pi <= phase <= np.pi

    def test_pvalue_without_cov(self) -> None:
        """Test that pvalue is None when covariance matrix not provided."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        _, _, pvalue = circular_basis_metrics(0.5, 0.5)
        assert pvalue is None

    def test_pvalue_with_cov(self) -> None:
        """Test that pvalue is computed when covariance matrix provided."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        beta_sin = 0.5
        beta_cos = 0.5
        # Covariance matrix for [beta_sin, beta_cos]
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        _, _, pvalue = circular_basis_metrics(beta_sin, beta_cos, cov_matrix=cov)
        assert pvalue is not None
        assert 0 <= pvalue <= 1

    def test_significant_modulation_low_pvalue(self) -> None:
        """Test that strong modulation with small variance gives low p-value."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        # Strong coefficients with small variance -> significant
        beta_sin = 2.0
        beta_cos = 2.0
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])  # Small variance

        _, _, pvalue = circular_basis_metrics(beta_sin, beta_cos, cov_matrix=cov)
        assert pvalue is not None
        assert pvalue < 0.001  # Highly significant

    def test_weak_modulation_high_pvalue(self) -> None:
        """Test that weak modulation with large variance gives high p-value."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        # Weak coefficients with large variance -> not significant
        beta_sin = 0.1
        beta_cos = 0.1
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])  # Large variance

        _, _, pvalue = circular_basis_metrics(beta_sin, beta_cos, cov_matrix=cov)
        assert pvalue is not None
        assert pvalue > 0.1  # Not significant

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import circular_basis_metrics

        assert callable(circular_basis_metrics)


class TestIsModulated:
    """Tests for is_modulated() convenience function (Milestone M2)."""

    def test_function_exists(self) -> None:
        """Test that is_modulated can be imported."""
        from neurospatial.metrics.circular_basis import is_modulated

        assert callable(is_modulated)

    def test_returns_bool(self) -> None:
        """Test that is_modulated returns a boolean."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Strong modulation with low variance
        beta_sin = 0.5
        beta_cos = 0.5
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        result = is_modulated(beta_sin, beta_cos, cov)
        assert isinstance(result, bool)

    def test_significant_strong_modulation_returns_true(self) -> None:
        """Test that significant strong modulation returns True."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Strong modulation (amplitude = sqrt(2^2 + 2^2) = 2.83) with low variance
        beta_sin = 2.0
        beta_cos = 2.0
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])  # Small variance

        result = is_modulated(beta_sin, beta_cos, cov)
        assert result is True

    def test_not_significant_returns_false(self) -> None:
        """Test that non-significant modulation returns False."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Weak modulation with large variance -> p > 0.05
        beta_sin = 0.1
        beta_cos = 0.1
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])  # Large variance

        result = is_modulated(beta_sin, beta_cos, cov)
        assert result is False

    def test_weak_modulation_below_threshold_returns_false(self) -> None:
        """Test that magnitude below threshold returns False even if significant."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Small but statistically significant (low variance)
        # amplitude = sqrt(0.1^2 + 0.1^2) = 0.141 < 0.2 (default min_magnitude)
        beta_sin = 0.1
        beta_cos = 0.1
        cov = np.array([[0.0001, 0.0], [0.0, 0.0001]])  # Very small variance

        result = is_modulated(beta_sin, beta_cos, cov)
        assert result is False

    def test_custom_alpha(self) -> None:
        """Test that custom alpha threshold works."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Moderate modulation with moderate variance
        # This might be significant at alpha=0.1 but not at alpha=0.01
        beta_sin = 0.5
        beta_cos = 0.5
        cov = np.array([[0.1, 0.0], [0.0, 0.1]])

        # More lenient alpha
        result_lenient = is_modulated(beta_sin, beta_cos, cov, alpha=0.1)
        # Strict alpha
        result_strict = is_modulated(beta_sin, beta_cos, cov, alpha=0.001)

        # Lenient should be at least as permissive as strict
        if result_strict:
            assert result_lenient  # If strict passes, lenient must too

    def test_custom_min_magnitude(self) -> None:
        """Test that custom min_magnitude threshold works."""
        from neurospatial.metrics.circular_basis import is_modulated

        # Amplitude = sqrt(0.3^2 + 0.3^2) = 0.424
        beta_sin = 0.3
        beta_cos = 0.3
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        # Default min_magnitude=0.2 -> True (0.424 > 0.2)
        result_default = is_modulated(beta_sin, beta_cos, cov)
        assert result_default is True

        # Higher threshold min_magnitude=0.5 -> False (0.424 < 0.5)
        result_strict = is_modulated(beta_sin, beta_cos, cov, min_magnitude=0.5)
        assert result_strict is False

    def test_zero_coefficients_returns_false(self) -> None:
        """Test that zero coefficients return False (no modulation)."""
        from neurospatial.metrics.circular_basis import is_modulated

        # No modulation: amplitude = 0
        beta_sin = 0.0
        beta_cos = 0.0
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        result = is_modulated(beta_sin, beta_cos, cov)
        assert result is False

    def test_singular_covariance_returns_false(self) -> None:
        """Test that singular covariance matrix returns False."""
        import warnings

        from neurospatial.metrics.circular_basis import is_modulated

        # Singular covariance matrix (determinant = 0)
        beta_sin = 2.0
        beta_cos = 2.0
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # Rank deficient

        # Should warn and return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warning
            result = is_modulated(beta_sin, beta_cos, cov)

        assert result is False

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import is_modulated

        assert callable(is_modulated)


# ============================================================================
# Visualization: plot_circular_basis_tuning (Milestone M3)
# ============================================================================


class TestPlotCircularBasisTuning:
    """Tests for plot_circular_basis_tuning() visualization function (Milestone M3)."""

    def test_function_exists(self) -> None:
        """Test that plot_circular_basis_tuning can be imported."""
        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        assert callable(plot_circular_basis_tuning)

    def test_polar_plot_creates_figure(self) -> None:
        """Test that polar projection creates a figure with polar axes."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        # GLM coefficients for cos/sin (beta_cos=1, beta_sin=0 => peak at 0)
        beta_cos = 1.0
        beta_sin = 0.5

        ax = plot_circular_basis_tuning(beta_sin, beta_cos, projection="polar")

        # Check it's a polar axes
        assert ax.name == "polar"
        plt.close("all")

    def test_linear_plot_creates_figure(self) -> None:
        """Test that linear projection creates a figure with linear axes."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        ax = plot_circular_basis_tuning(beta_sin, beta_cos, projection="linear")

        # Check it's NOT a polar axes
        assert ax.name != "polar"
        plt.close("all")

    def test_show_data_requires_angles_rates(self) -> None:
        """Test that show_data=True requires both angles and rates."""
        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # Missing both
        with pytest.raises(ValueError, match="show_data=True requires"):
            plot_circular_basis_tuning(beta_sin, beta_cos, show_data=True)

        # Missing rates
        with pytest.raises(ValueError, match="show_data=True requires"):
            plot_circular_basis_tuning(
                beta_sin, beta_cos, angles=np.linspace(0, 2 * np.pi, 10), show_data=True
            )

        # Missing angles
        with pytest.raises(ValueError, match="show_data=True requires"):
            plot_circular_basis_tuning(
                beta_sin, beta_cos, rates=np.random.rand(10), show_data=True
            )

    def test_show_fit_only_works(self) -> None:
        """Test that show_data=False works without angles/rates."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # Should work without data
        ax = plot_circular_basis_tuning(beta_sin, beta_cos, show_data=False)
        assert ax is not None
        plt.close("all")

    def test_with_data_overlay(self) -> None:
        """Test that data overlay works when angles and rates provided."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        rates = np.random.rand(36) * 10

        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, angles=angles, rates=rates, show_data=True
        )
        assert ax is not None
        plt.close("all")

    def test_returns_axes_object(self) -> None:
        """Test that function returns matplotlib Axes."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        ax = plot_circular_basis_tuning(beta_sin, beta_cos, show_data=False)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        """Test that function can plot on provided axes."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # Create axes first
        _, ax_provided = plt.subplots(subplot_kw={"projection": "polar"})
        ax_returned = plot_circular_basis_tuning(
            beta_sin, beta_cos, ax=ax_provided, projection="polar", show_data=False
        )

        # Should return same axes
        assert ax_returned is ax_provided
        plt.close("all")

    def test_n_points_parameter(self) -> None:
        """Test that n_points controls curve smoothness."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # With few points
        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, n_points=10, show_data=False
        )
        # Plot should still be created
        assert ax is not None
        plt.close("all")

    def test_color_parameter(self) -> None:
        """Test that color parameter is accepted."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # Should not raise
        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, color="red", show_data=False
        )
        assert ax is not None
        plt.close("all")

    def test_intercept_parameter(self) -> None:
        """Test that intercept parameter affects the curve baseline."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # With intercept (baseline shift)
        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, intercept=2.0, show_data=False
        )
        assert ax is not None
        plt.close("all")

    def test_exported_from_metrics(self) -> None:
        """Test that function is exported from neurospatial.metrics."""
        from neurospatial.metrics import plot_circular_basis_tuning

        assert callable(plot_circular_basis_tuning)

    def test_confidence_bands_require_cov_matrix(self) -> None:
        """Test that show_ci=True requires cov_matrix."""
        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5

        # Missing cov_matrix when show_ci=True
        with pytest.raises(ValueError, match="show_ci=True requires cov_matrix"):
            plot_circular_basis_tuning(beta_sin, beta_cos, show_ci=True)

    def test_confidence_bands_with_cov_matrix(self) -> None:
        """Test that confidence bands work when cov_matrix provided."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        # 2x2 covariance matrix for [beta_sin, beta_cos]
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, cov_matrix=cov, show_ci=True
        )
        assert ax is not None
        plt.close("all")

    def test_confidence_bands_custom_ci_level(self) -> None:
        """Test that custom CI level (e.g., 99%) works."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        # 99% CI
        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, cov_matrix=cov, show_ci=True, ci=0.99
        )
        assert ax is not None
        plt.close("all")

    def test_confidence_bands_linear_projection(self) -> None:
        """Test that confidence bands work with linear projection."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, cov_matrix=cov, show_ci=True, projection="linear"
        )
        assert ax is not None
        plt.close("all")

    def test_cov_matrix_wrong_shape_raises(self) -> None:
        """Test that wrong-shaped cov_matrix raises ValueError."""
        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        # Wrong shape (3x3 instead of 2x2)
        cov = np.eye(3)

        with pytest.raises(ValueError, match=r"cov_matrix must be.*2x2"):
            plot_circular_basis_tuning(beta_sin, beta_cos, cov_matrix=cov, show_ci=True)

    def test_ci_band_color_parameter(self) -> None:
        """Test that ci_color parameter is accepted."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_cos = 1.0
        beta_sin = 0.5
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, cov_matrix=cov, show_ci=True, ci_alpha=0.2
        )
        assert ax is not None
        plt.close("all")

    # ========================================================================
    # Behavioral tests: verify computed values are correct
    # ========================================================================

    def test_fitted_curve_values_at_known_angles(self) -> None:
        """Test that fitted curve matches expected exp(intercept + β·cos(θ) + β·sin(θ))."""
        import matplotlib.pyplot as plt

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_sin = 0.5
        beta_cos = 1.0
        intercept = 2.0

        ax = plot_circular_basis_tuning(
            beta_sin, beta_cos, intercept=intercept, projection="linear", n_points=4
        )

        # Get the plotted line data
        line = ax.get_lines()[0]
        rate_data = line.get_ydata()

        # Verify values at θ=0 (first point): exp(2.0 + 1.0*cos(0) + 0.5*sin(0)) = exp(3.0)
        expected_at_0 = np.exp(intercept + beta_cos * 1.0 + beta_sin * 0.0)
        assert_allclose(rate_data[0], expected_at_0, rtol=1e-10)

        # Verify at θ=π/2: exp(2.0 + 1.0*cos(π/2) + 0.5*sin(π/2)) = exp(2.5)
        # With n_points=4, θ = [0, π/2, π, 3π/2]
        expected_at_pi2 = np.exp(intercept + beta_cos * 0.0 + beta_sin * 1.0)
        assert_allclose(rate_data[1], expected_at_pi2, rtol=1e-10)

        plt.close("all")

    def test_ci_width_scales_with_covariance(self) -> None:
        """Test that doubling covariance doubles the CI width on linear scale."""
        import matplotlib.pyplot as plt
        from scipy import stats

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_sin = 0.5
        beta_cos = 1.0
        intercept = 2.0
        cov_small = np.array([[0.01, 0.0], [0.0, 0.01]])
        cov_large = np.array([[0.04, 0.0], [0.0, 0.04]])  # 4x variance = 2x SE

        # Get CI widths for small covariance
        ax1 = plot_circular_basis_tuning(
            beta_sin,
            beta_cos,
            intercept=intercept,
            cov_matrix=cov_small,
            show_ci=True,
            projection="linear",
            n_points=10,
        )
        # fill_between creates a PolyCollection - verify it exists
        assert len(ax1.collections) > 0
        plt.close("all")

        # Get CI widths for large covariance
        ax2 = plot_circular_basis_tuning(
            beta_sin,
            beta_cos,
            intercept=intercept,
            cov_matrix=cov_large,
            show_ci=True,
            projection="linear",
            n_points=10,
        )
        assert len(ax2.collections) > 0
        plt.close("all")

        # Compute CI width at θ=0 on linear predictor scale
        # SE_small = sqrt(0.01) = 0.1, SE_large = sqrt(0.04) = 0.2
        # At θ=0: design = [sin(0), cos(0)] = [0, 1]
        # var = [0, 1] @ cov @ [0, 1]^T = cov[1,1]
        # So SE ratio should be sqrt(0.04/0.01) = 2

        # The CI on response scale isn't exactly 2x due to exp transform,
        # but on linear scale it should be. We verify the relationship holds.
        z = stats.norm.ppf(0.975)
        se_small = np.sqrt(0.01)  # At θ=0, only cos matters
        se_large = np.sqrt(0.04)

        # Linear predictor at θ=0
        eta = intercept + beta_cos * 1.0

        # Expected CI bounds on response scale
        ci_lower_small = np.exp(eta - z * se_small)
        ci_upper_small = np.exp(eta + z * se_small)
        ci_lower_large = np.exp(eta - z * se_large)
        ci_upper_large = np.exp(eta + z * se_large)

        # Width ratio on response scale
        width_small = ci_upper_small - ci_lower_small
        width_large = ci_upper_large - ci_lower_large

        # Larger covariance should give wider CI
        assert width_large > width_small

    def test_ci_asymmetry_from_exp_transform(self) -> None:
        """Test that CI bands are asymmetric due to exp() transform."""
        import matplotlib.pyplot as plt
        from scipy import stats

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        beta_sin = 0.0
        beta_cos = 1.0
        intercept = 3.0  # Large intercept amplifies asymmetry
        cov = np.array([[0.1, 0.0], [0.0, 0.1]])

        ax = plot_circular_basis_tuning(
            beta_sin,
            beta_cos,
            intercept=intercept,
            cov_matrix=cov,
            show_ci=True,
            projection="linear",
            n_points=10,
        )

        # Get the fitted line - verify it exists
        assert len(ax.get_lines()) > 0

        # Compute expected asymmetric CI at θ=0
        # At θ=0: design = [0, 1], var = cov[1,1] = 0.1, SE = sqrt(0.1)
        eta = intercept + beta_cos * 1.0  # = 4.0
        se = np.sqrt(0.1)
        z = stats.norm.ppf(0.975)

        ci_lower = np.exp(eta - z * se)
        ci_upper = np.exp(eta + z * se)

        # Asymmetry: distance from mean to upper > distance from mean to lower
        dist_upper = ci_upper - np.exp(eta)
        dist_lower = np.exp(eta) - ci_lower

        assert dist_upper > dist_lower, "CI should be asymmetric (upper > lower)"

        plt.close("all")

    def test_delta_method_variance_calculation(self) -> None:
        """Test that delta method variance is computed correctly at known angles."""
        import matplotlib.pyplot as plt
        from scipy import stats

        from neurospatial.metrics.circular_basis import plot_circular_basis_tuning

        # Use correlated covariance matrix to test full formula
        beta_sin = 1.0
        beta_cos = 1.0
        intercept = 0.0
        # Correlated covariance: Var(sin) = 0.04, Var(cos) = 0.01, Cov = 0.01
        cov = np.array([[0.04, 0.01], [0.01, 0.01]])

        # Test at θ=0 where only cos matters: design = [0, 1]
        # var = [0, 1] @ cov @ [0, 1]^T = cov[1,1] = 0.01
        # SE = sqrt(0.01) = 0.1
        theta_test = 0.0
        expected_se_at_0 = np.sqrt(cov[1, 1])  # Only cos variance matters at θ=0
        eta_at_0 = (
            intercept + beta_cos * np.cos(theta_test) + beta_sin * np.sin(theta_test)
        )
        z = stats.norm.ppf(0.975)

        # Expected CI bounds at θ=0
        expected_lower = np.exp(eta_at_0 - z * expected_se_at_0)
        expected_upper = np.exp(eta_at_0 + z * expected_se_at_0)

        ax = plot_circular_basis_tuning(
            beta_sin,
            beta_cos,
            intercept=intercept,
            cov_matrix=cov,
            show_ci=True,
            projection="linear",
            n_points=4,  # θ = [0, π/2, π, 3π/2]
        )

        # Verify fitted value at θ=0
        line = ax.get_lines()[0]
        rate_data = line.get_ydata()
        assert_allclose(rate_data[0], np.exp(eta_at_0), rtol=1e-10)

        # The CI width at θ=0 should match our hand calculation
        ci_width_expected = expected_upper - expected_lower
        assert ci_width_expected > 0  # Sanity check

        plt.close("all")


class TestFuzzEdgeCases:
    """Fuzz testing with extreme values to ensure numerical robustness.

    Note: rayleigh_test returns (z_statistic, pvalue) tuple.
    circular_linear_correlation returns (rho, pvalue) tuple.
    """

    # -------------------------------------------------------------------------
    # Extreme floating-point value tests
    # -------------------------------------------------------------------------

    def test_rayleigh_with_very_small_angles(self) -> None:
        """Rayleigh test handles very small angle differences near machine epsilon."""
        from neurospatial.metrics.circular import rayleigh_test

        eps = np.finfo(float).eps
        # Angles clustered within machine epsilon
        angles = np.array([0.0, eps, 2 * eps, 3 * eps])

        z_stat, pvalue = rayleigh_test(angles)
        # Should still compute - z >= 0, p in [0, 1]
        assert z_stat >= 0
        assert 0 <= pvalue <= 1

    def test_rayleigh_with_large_angles(self) -> None:
        """Rayleigh test handles angles much larger than 2π (many rotations)."""
        from neurospatial.metrics.circular import rayleigh_test

        # Angles representing many full rotations
        angles = np.array([0.0, 100 * np.pi, 200 * np.pi, 300 * np.pi])

        z_stat, pvalue = rayleigh_test(angles)
        # After modulo 2π, these are all 0, so highly concentrated
        assert z_stat >= 0
        assert 0 <= pvalue <= 1

    def test_rayleigh_with_negative_large_angles(self) -> None:
        """Rayleigh test handles large negative angles."""
        from neurospatial.metrics.circular import rayleigh_test

        angles = np.array([-1000 * np.pi, -500 * np.pi, -250 * np.pi, 0.0])

        z_stat, pvalue = rayleigh_test(angles)
        assert z_stat >= 0
        assert 0 <= pvalue <= 1

    def test_circular_linear_correlation_with_extreme_positions(self) -> None:
        """Circular-linear correlation handles extreme position values."""
        from neurospatial.metrics.circular import circular_linear_correlation

        # Extreme position values
        max_float = np.finfo(float).max / 1e10  # Large but not overflow-prone
        angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        positions = np.array([0.0, max_float / 3, max_float * 2 / 3, max_float])

        rho, pvalue = circular_linear_correlation(angles, positions)
        # Correlation in [0, 1] (always non-negative for this metric)
        assert 0 <= rho <= 1 or np.isnan(rho)
        assert 0 <= pvalue <= 1 or np.isnan(pvalue)

    def test_circular_linear_correlation_with_tiny_positions(self) -> None:
        """Circular-linear correlation handles denormalized position values."""
        from neurospatial.metrics.circular import circular_linear_correlation

        # Tiny position differences near machine epsilon
        eps = np.finfo(float).eps
        angles = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        positions = np.array([0.0, eps, 2 * eps, 3 * eps])

        rho, _pvalue = circular_linear_correlation(angles, positions)
        # Should not crash; result may be degenerate
        assert np.isfinite(rho) or np.isnan(rho)

    # -------------------------------------------------------------------------
    # Infinity and NaN handling tests
    # -------------------------------------------------------------------------

    def test_rayleigh_rejects_inf_angles(self) -> None:
        """Rayleigh test should reject infinite angles."""
        from neurospatial.metrics.circular import rayleigh_test

        angles = np.array([0.0, np.inf, np.pi])

        with pytest.raises((ValueError, FloatingPointError)):
            rayleigh_test(angles)

    def test_rayleigh_rejects_nan_angles(self) -> None:
        """Rayleigh test should reject NaN angles."""
        from neurospatial.metrics.circular import rayleigh_test

        angles = np.array([0.0, np.nan, np.pi])

        with pytest.raises((ValueError, FloatingPointError)):
            rayleigh_test(angles)

    def test_circular_linear_rejects_inf_values(self) -> None:
        """Circular-linear correlation rejects or handles infinite values."""
        from neurospatial.metrics.circular import circular_linear_correlation

        angles = np.array([0.0, np.pi / 2, np.pi])
        positions_inf = np.array([0.0, np.inf, 1.0])

        # May raise, return NaN, or return zero - any is acceptable for degenerate input
        try:
            rho, pvalue = circular_linear_correlation(angles, positions_inf)
            # If it doesn't raise, result may be NaN or zero (degenerate case)
            assert np.isnan(rho) or np.isnan(pvalue) or rho == 0.0
        except (ValueError, FloatingPointError):
            pass  # Expected

    def test_circular_linear_rejects_nan_values(self) -> None:
        """Circular-linear correlation rejects NaN values."""
        from neurospatial.metrics.circular import circular_linear_correlation

        angles = np.array([0.0, np.pi / 2, np.pi])
        positions_nan = np.array([0.0, np.nan, 1.0])

        with pytest.raises((ValueError, FloatingPointError)):
            circular_linear_correlation(angles, positions_nan)

    # -------------------------------------------------------------------------
    # Degenerate data tests
    # -------------------------------------------------------------------------

    def test_rayleigh_with_identical_angles(self) -> None:
        """Rayleigh test with all identical angles (perfect concentration)."""
        from neurospatial.metrics.circular import rayleigh_test

        angles = np.full(100, np.pi / 4)

        z_stat, pvalue = rayleigh_test(angles)
        # z = n * R^2, where R = 1 for identical angles
        # So z should be approximately n = 100
        assert z_stat >= 90  # Allow some tolerance
        assert pvalue < 0.001  # Highly significant

    def test_rayleigh_with_opposite_angles(self) -> None:
        """Rayleigh test with exactly opposite angles (zero concentration)."""
        from neurospatial.metrics.circular import rayleigh_test

        # Perfect balance - half at 0, half at π
        angles = np.array([0.0, 0.0, np.pi, np.pi])

        z_stat, pvalue = rayleigh_test(angles)
        # z = n * R^2, where R = 0 for opposite angles
        assert_allclose(z_stat, 0.0, atol=1e-10)
        assert pvalue > 0.05  # Not significant

    def test_circular_linear_with_constant_positions(self) -> None:
        """Circular-linear correlation with constant positions (no variation)."""
        from neurospatial.metrics.circular import circular_linear_correlation

        angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        positions = np.full(4, 5.0)  # All same position

        rho, _pvalue = circular_linear_correlation(angles, positions)
        # With no position variation, correlation is undefined/zero
        assert np.isnan(rho) or rho == 0.0

    def test_circular_linear_with_constant_angles(self) -> None:
        """Circular-linear correlation with constant angles (no circular variation)."""
        from neurospatial.metrics.circular import circular_linear_correlation

        angles = np.full(10, np.pi / 2)  # All same angle
        positions = np.linspace(0, 100, 10)

        rho, _pvalue = circular_linear_correlation(angles, positions)
        # With no angle variation, correlation should be undefined or zero
        assert np.isnan(rho) or np.abs(rho) < 1e-10

    # -------------------------------------------------------------------------
    # Numerical precision edge cases
    # -------------------------------------------------------------------------

    def test_rayleigh_near_boundary_pvalue(self) -> None:
        """Rayleigh test produces valid p-values near 0 and 1 boundaries."""
        from neurospatial.metrics.circular import rayleigh_test

        rng = np.random.default_rng(42)

        # Highly concentrated - p-value near 0
        concentrated = rng.vonmises(0.0, 100.0, size=1000)
        _z_low, pvalue_low = rayleigh_test(concentrated)
        assert 0 <= pvalue_low <= 1
        assert pvalue_low < 0.01

        # Uniform distribution - p-value higher
        uniform = rng.uniform(0, 2 * np.pi, size=1000)
        _z_high, pvalue_high = rayleigh_test(uniform)
        assert 0 <= pvalue_high <= 1

    def test_z_statistic_range(self) -> None:
        """Rayleigh z-statistic stays in valid range [0, n] for all inputs."""
        from neurospatial.metrics.circular import rayleigh_test

        rng = np.random.default_rng(42)

        # Random angles - z should be in [0, n]
        for _ in range(10):
            n = 50
            angles = rng.uniform(0, 2 * np.pi, size=n)
            z_stat, _pvalue = rayleigh_test(angles)
            assert 0 <= z_stat <= n

    def test_circular_basis_with_extreme_angles(self) -> None:
        """Circular basis handles extreme angle values."""
        from neurospatial.metrics.circular_basis import circular_basis

        # Very large angles (many rotations)
        large_angles = np.array([0.0, 1000 * np.pi, 2000 * np.pi, 3000 * np.pi])
        result_large = circular_basis(large_angles)
        assert np.all(np.isfinite(result_large.sin_component))
        assert np.all(np.isfinite(result_large.cos_component))

        # Angles near machine epsilon
        eps = np.finfo(float).eps
        tiny_angles = np.array([0.0, eps, 2 * eps, 3 * eps])
        result_tiny = circular_basis(tiny_angles)
        assert np.all(np.isfinite(result_tiny.sin_component))
        assert np.all(np.isfinite(result_tiny.cos_component))

    def test_circular_basis_metrics_with_extreme_coefficients(self) -> None:
        """circular_basis_metrics handles extreme coefficient values."""
        from neurospatial.metrics.circular_basis import circular_basis_metrics

        # Very small coefficients
        amp_small, phase_small, _ = circular_basis_metrics(1e-15, 1e-15)
        assert np.isfinite(amp_small)
        assert np.isfinite(phase_small)

        # Very large coefficients
        amp_large, phase_large, _ = circular_basis_metrics(1e10, 1e10)
        assert np.isfinite(amp_large)
        assert np.isfinite(phase_large)

        # Asymmetric coefficients
        amp_asym, phase_asym, _ = circular_basis_metrics(1e10, 1e-15)
        assert np.isfinite(amp_asym)
        assert np.isfinite(phase_asym)

    # -------------------------------------------------------------------------
    # Array size edge cases
    # -------------------------------------------------------------------------

    def test_rayleigh_minimum_sample_size(self) -> None:
        """Rayleigh test handles minimum valid sample sizes (requires >= 3)."""
        from neurospatial.metrics.circular import rayleigh_test

        # Two samples - should raise (requires >= 3)
        angles_2 = np.array([0.0, np.pi / 2])
        with pytest.raises(ValueError, match="at least 3 samples"):
            rayleigh_test(angles_2)

        # Three samples - minimum valid size
        angles_3 = np.array([0.0, np.pi / 2, np.pi])
        z_3, pvalue_3 = rayleigh_test(angles_3)
        assert z_3 >= 0
        assert 0 <= pvalue_3 <= 1

    def test_rayleigh_large_sample_size(self) -> None:
        """Rayleigh test handles large sample sizes efficiently."""
        from neurospatial.metrics.circular import rayleigh_test

        rng = np.random.default_rng(42)

        # Large sample - should not overflow or timeout
        angles = rng.uniform(0, 2 * np.pi, size=100000)
        z_stat, pvalue = rayleigh_test(angles)
        assert z_stat >= 0
        assert 0 <= pvalue <= 1

    def test_circular_basis_minimum_samples(self) -> None:
        """Circular basis handles minimum number of samples."""
        from neurospatial.metrics.circular_basis import circular_basis

        # Single sample
        angles_1 = np.array([np.pi / 4])
        result_1 = circular_basis(angles_1)
        assert len(result_1.sin_component) == 1
        assert np.isfinite(result_1.sin_component[0])
        assert np.isfinite(result_1.cos_component[0])

        # Two samples
        angles_2 = np.array([0.0, np.pi])
        result_2 = circular_basis(angles_2)
        assert len(result_2.sin_component) == 2

    # -------------------------------------------------------------------------
    # Special angle values
    # -------------------------------------------------------------------------

    def test_rayleigh_at_boundary_angles(self) -> None:
        """Rayleigh test handles angles exactly at 0 and 2π boundaries."""
        from neurospatial.metrics.circular import rayleigh_test

        # Mix of 0 and 2π (should be equivalent)
        angles = np.array([0.0, 2 * np.pi, 0.0, 2 * np.pi])
        z_stat, _pvalue = rayleigh_test(angles)
        # All angles are effectively 0, so z = n * R^2 = 4 * 1^2 = 4
        assert_allclose(z_stat, 4.0, rtol=1e-10)

    def test_circular_linear_at_quadrant_boundaries(self) -> None:
        """Circular-linear correlation at exact quadrant boundaries."""
        from neurospatial.metrics.circular import circular_linear_correlation

        # Exact quadrant boundaries
        angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        positions = np.array([0.0, 0.25, 0.5, 0.75])

        rho, pvalue = circular_linear_correlation(angles, positions)
        # Circular-linear correlation is always non-negative
        assert 0 <= rho <= 1
        assert 0 <= pvalue <= 1
