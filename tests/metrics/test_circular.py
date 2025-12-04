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
