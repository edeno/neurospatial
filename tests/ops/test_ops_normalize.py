"""
Tests for neurospatial.ops.normalize module.

Tests that field operations are importable from the new ops.normalize location.
"""

import numpy as np
import pytest


class TestNormalizeField:
    """Basic smoke tests for normalize_field function."""

    def test_normalizes_to_sum_one(self):
        """normalize_field should produce values summing to 1."""
        from neurospatial.ops.normalize import normalize_field

        field = np.array([1.0, 2.0, 3.0, 4.0])
        result = normalize_field(field)

        assert np.isclose(result.sum(), 1.0)

    def test_preserves_proportions(self):
        """normalize_field should preserve relative proportions."""
        from neurospatial.ops.normalize import normalize_field

        field = np.array([2.0, 4.0, 6.0])
        result = normalize_field(field)

        # Check ratios are preserved
        assert np.isclose(result[1] / result[0], 2.0)
        assert np.isclose(result[2] / result[0], 3.0)

    def test_rejects_negative_values(self):
        """normalize_field should reject fields with negative values."""
        from neurospatial.ops.normalize import normalize_field

        field = np.array([1.0, -2.0, 3.0])
        with pytest.raises(ValueError, match="negative"):
            normalize_field(field)


class TestClamp:
    """Basic smoke tests for clamp function."""

    def test_clamps_to_range(self):
        """clamp should restrict values to [lo, hi]."""
        from neurospatial.ops.normalize import clamp

        field = np.array([-1.0, 0.5, 2.0])
        result = clamp(field, lo=0.0, hi=1.0)

        np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])

    def test_default_lo_is_zero(self):
        """clamp default lo should be 0."""
        from neurospatial.ops.normalize import clamp

        field = np.array([-1.0, 0.5, 2.0])
        result = clamp(field)  # hi defaults to inf

        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 2.0

    def test_rejects_invalid_bounds(self):
        """clamp should reject lo > hi."""
        from neurospatial.ops.normalize import clamp

        field = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="less than or equal"):
            clamp(field, lo=5.0, hi=1.0)


class TestCombineFields:
    """Basic smoke tests for combine_fields function."""

    def test_mean_mode(self):
        """combine_fields with mode='mean' should compute average."""
        from neurospatial.ops.normalize import combine_fields

        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        result = combine_fields([f1, f2], mode="mean")

        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])

    def test_max_mode(self):
        """combine_fields with mode='max' should compute element-wise max."""
        from neurospatial.ops.normalize import combine_fields

        f1 = np.array([1.0, 5.0, 3.0])
        f2 = np.array([3.0, 2.0, 4.0])
        result = combine_fields([f1, f2], mode="max")

        np.testing.assert_array_equal(result, [3.0, 5.0, 4.0])

    def test_min_mode(self):
        """combine_fields with mode='min' should compute element-wise min."""
        from neurospatial.ops.normalize import combine_fields

        f1 = np.array([1.0, 5.0, 3.0])
        f2 = np.array([3.0, 2.0, 4.0])
        result = combine_fields([f1, f2], mode="min")

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_weighted_mean(self):
        """combine_fields with weights should compute weighted average."""
        from neurospatial.ops.normalize import combine_fields

        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([3.0, 4.0, 5.0])
        result = combine_fields([f1, f2], weights=[0.25, 0.75], mode="mean")

        np.testing.assert_array_equal(result, [2.5, 3.5, 4.5])

    def test_rejects_empty_fields(self):
        """combine_fields should reject empty fields list."""
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="At least one field"):
            combine_fields([])


class TestNormalizeFieldValidation:
    def test_nan_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="NaN"):
            normalize_field(np.array([1.0, np.nan, 3.0]))

    def test_inf_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="Inf"):
            normalize_field(np.array([1.0, np.inf, 3.0]))

    def test_all_zeros_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="all zeros"):
            normalize_field(np.zeros(4))

    def test_non_positive_eps_rejected(self):
        from neurospatial.ops.normalize import normalize_field

        with pytest.raises(ValueError, match="eps must be positive"):
            normalize_field(np.array([1.0, 2.0]), eps=0.0)


class TestCombineFieldsValidation:
    def test_mismatched_shapes_rejected(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="same shape"):
            combine_fields([np.zeros(3), np.zeros(4)])

    def test_weights_only_valid_for_mean(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="mode='mean'"):
            combine_fields([np.ones(3), np.ones(3)], weights=[0.5, 0.5], mode="max")

    def test_weights_length_must_match_fields(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="must match"):
            combine_fields([np.ones(3), np.ones(3)], weights=[1.0])

    def test_weights_must_sum_to_one(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="sum to 1"):
            combine_fields([np.ones(3), np.ones(3)], weights=[0.3, 0.3])

    def test_unknown_mode_rejected(self):
        from neurospatial.ops.normalize import combine_fields

        with pytest.raises(ValueError, match="Unknown mode"):
            combine_fields([np.ones(3), np.ones(3)], mode="median")  # type: ignore[arg-type]
