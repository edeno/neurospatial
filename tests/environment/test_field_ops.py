"""
Tests for field math utility functions.

These functions operate on bin-valued fields (NDArray[np.float64], shape (n_bins,))
and provide common operations for spatial analysis.
"""

import numpy as np
import pytest

# Import the functions we'll be testing
from neurospatial.ops.normalize import (
    clamp,
    combine_fields,
    normalize_field,
)


class TestNormalizeField:
    """Test suite for normalize_field function."""

    def test_normalize_sums_to_one(self):
        """Normalized field sums to 1.0."""
        field = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_field(field)
        assert np.isclose(normalized.sum(), 1.0)

    def test_normalize_preserves_shape(self):
        """Normalization preserves field shape."""
        rng = np.random.default_rng(42)
        field = rng.random(100)
        normalized = normalize_field(field)
        assert normalized.shape == field.shape

    def test_normalize_preserves_proportions(self):
        """Normalization preserves relative proportions."""
        field = np.array([1.0, 2.0, 3.0])
        normalized = normalize_field(field)
        # Ratios should be preserved
        assert np.isclose(normalized[1] / normalized[0], 2.0)
        assert np.isclose(normalized[2] / normalized[0], 3.0)

    def test_normalize_uniform_field(self):
        """Uniform field normalizes to uniform distribution."""
        field = np.ones(50)
        normalized = normalize_field(field)
        expected = np.full(50, 1.0 / 50)
        assert np.allclose(normalized, expected)

    def test_normalize_zero_field_raises(self):
        """Zero field raises ValueError."""
        field = np.zeros(10)
        with pytest.raises(ValueError, match=r"Cannot normalize.*all zeros"):
            normalize_field(field)

    def test_normalize_negative_values(self):
        """Negative values raise ValueError."""
        field = np.array([1.0, -1.0, 2.0])
        with pytest.raises(ValueError, match=r"Field contains negative values"):
            normalize_field(field)

    def test_normalize_with_eps(self):
        """eps parameter prevents division by near-zero."""
        field = np.array([1e-15, 1e-15, 1e-15])
        # Should not raise, uses eps
        normalized = normalize_field(field, eps=1e-12)
        assert np.isclose(normalized.sum(), 1.0)

    def test_normalize_nan_values_raise(self):
        """NaN values raise ValueError."""
        field = np.array([1.0, np.nan, 2.0])
        with pytest.raises(ValueError, match=r"Field contains NaN"):
            normalize_field(field)

    def test_normalize_inf_values_raise(self):
        """Inf values raise ValueError."""
        field = np.array([1.0, np.inf, 2.0])
        with pytest.raises(ValueError, match=r"Field contains Inf"):
            normalize_field(field)


class TestClamp:
    """Test suite for clamp function."""

    def test_clamp_basic(self):
        """Basic clamping to [0, 1] range."""
        field = np.array([-1.0, 0.5, 2.0, 0.0, 1.0])
        clamped = clamp(field, lo=0.0, hi=1.0)
        expected = np.array([0.0, 0.5, 1.0, 0.0, 1.0])
        assert np.allclose(clamped, expected)

    def test_clamp_preserves_shape(self):
        """Clamping preserves field shape."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(100)
        clamped = clamp(field, lo=0.0, hi=1.0)
        assert clamped.shape == field.shape

    def test_clamp_default_lower_bound(self):
        """Default lower bound is 0.0."""
        field = np.array([-5.0, -1.0, 0.0, 1.0, 2.0])
        clamped = clamp(field)
        assert np.all(clamped >= 0.0)
        assert clamped[0] == 0.0  # -5 clamped to 0

    def test_clamp_default_upper_bound(self):
        """Default upper bound is inf (no upper limit)."""
        field = np.array([0.0, 100.0, 1e6, 1e10])
        clamped = clamp(field, lo=0.0)
        assert np.allclose(clamped, field)  # No upper clamping

    def test_clamp_no_modification_in_range(self):
        """Values within range are unchanged."""
        field = np.array([0.1, 0.5, 0.9])
        clamped = clamp(field, lo=0.0, hi=1.0)
        assert np.allclose(clamped, field)

    def test_clamp_lo_greater_than_hi_raises(self):
        """lo > hi raises ValueError."""
        field = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=r"lo.*must be less than or equal to.*hi"):
            clamp(field, lo=5.0, hi=1.0)

    def test_clamp_with_nan_propagates(self):
        """NaN values are preserved (propagated)."""
        field = np.array([1.0, np.nan, 2.0])
        clamped = clamp(field, lo=0.0, hi=1.0)
        assert np.isnan(clamped[1])

    def test_clamp_with_inf(self):
        """Inf values are clamped to hi."""
        field = np.array([1.0, np.inf, 2.0])
        clamped = clamp(field, lo=0.0, hi=10.0)
        assert clamped[1] == 10.0


class TestCombineFields:
    """Test suite for combine_fields function."""

    def test_combine_mean_uniform_weights(self):
        """Mean combination with uniform weights."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        combined = combine_fields([f1, f2], mode="mean")
        expected = np.array([2.0, 3.0, 4.0])
        assert np.allclose(combined, expected)

    def test_combine_mean_weighted(self):
        """Mean combination with custom weights."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        weights = [0.25, 0.75]
        combined = combine_fields([f1, f2], weights=weights, mode="mean")
        expected = 0.25 * f1 + 0.75 * f2
        assert np.allclose(combined, expected)

    def test_combine_max(self):
        """Max combination."""
        f1 = np.array([1.0, 5.0, 3.0])
        f2 = np.array([3.0, 2.0, 4.0])
        combined = combine_fields([f1, f2], mode="max")
        expected = np.array([3.0, 5.0, 4.0])
        assert np.allclose(combined, expected)

    def test_combine_min(self):
        """Min combination."""
        f1 = np.array([1.0, 5.0, 3.0])
        f2 = np.array([3.0, 2.0, 4.0])
        combined = combine_fields([f1, f2], mode="min")
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(combined, expected)

    def test_combine_preserves_shape(self):
        """Combination preserves field shape."""
        rng = np.random.default_rng(42)
        f1 = rng.random(100)
        f2 = rng.random(100)
        f3 = rng.random(100)
        combined = combine_fields([f1, f2, f3], mode="mean")
        assert combined.shape == f1.shape

    def test_combine_single_field(self):
        """Single field returns copy."""
        field = np.array([1.0, 2.0, 3.0])
        combined = combine_fields([field], mode="mean")
        assert np.allclose(combined, field)
        assert combined is not field  # Should be a copy

    def test_combine_three_fields_mean(self):
        """Mean of three fields."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([2.0, 3.0, 4.0])
        f3 = np.array([3.0, 4.0, 5.0])
        combined = combine_fields([f1, f2, f3], mode="mean")
        expected = np.array([2.0, 3.0, 4.0])
        assert np.allclose(combined, expected)

    def test_combine_mismatched_shapes_raises(self):
        """Mismatched field shapes raise ValueError."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"All fields must have the same shape"):
            combine_fields([f1, f2], mode="mean")

    def test_combine_weights_wrong_length_raises(self):
        """Wrong number of weights raises ValueError."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        weights = [0.5]  # Should be 2 weights
        with pytest.raises(ValueError, match=r"Number of weights.*must match"):
            combine_fields([f1, f2], weights=weights, mode="mean")

    def test_combine_weights_dont_sum_to_one_raises(self):
        """Weights not summing to 1 raise ValueError."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        weights = [0.3, 0.5]  # Sum to 0.8
        with pytest.raises(ValueError, match=r"Weights must sum to 1"):
            combine_fields([f1, f2], weights=weights, mode="mean")

    def test_combine_weights_with_max_mode_raises(self):
        """Weights parameter with max mode raises ValueError."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match=r"Weights only valid for mode='mean'"):
            combine_fields([f1, f2], weights=[0.5, 0.5], mode="max")

    def test_combine_empty_fields_raises(self):
        """Empty field list raises ValueError."""
        with pytest.raises(ValueError, match=r"At least one field required"):
            combine_fields([], mode="mean")


class TestFieldOpsIntegration:
    """Integration tests combining multiple field operations."""

    def test_normalize_then_clamp(self):
        """Normalize then clamp workflow."""
        field = np.array([1.0, 2.0, 3.0, 100.0])
        normalized = normalize_field(field)
        clamped = clamp(normalized, lo=0.0, hi=0.5)
        # The large value (100) dominates after normalization
        # After clamping, sum will be < 1
        assert np.all(clamped <= 0.5)
        assert np.all(clamped >= 0.0)

    def test_combine_then_normalize(self):
        """Combine fields then normalize."""
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([2.0, 3.0, 4.0])
        combined = combine_fields([f1, f2], mode="mean")
        normalized = normalize_field(combined)
        assert np.isclose(normalized.sum(), 1.0)
