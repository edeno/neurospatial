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
