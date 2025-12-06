"""
Tests for neurospatial.ops.normalize module.

Tests that field operations are importable from the new ops.normalize location.
"""

import numpy as np
import pytest


class TestNormalizeModuleImports:
    """Test that normalize operations are importable from new location."""

    def test_import_normalize_field(self):
        """normalize_field should be importable from ops.normalize."""
        from neurospatial.ops.normalize import normalize_field

        assert callable(normalize_field)

    def test_import_clamp(self):
        """clamp should be importable from ops.normalize."""
        from neurospatial.ops.normalize import clamp

        assert callable(clamp)

    def test_import_combine_fields(self):
        """combine_fields should be importable from ops.normalize."""
        from neurospatial.ops.normalize import combine_fields

        assert callable(combine_fields)

    def test_import_from_ops_package(self):
        """Functions should be importable from ops package."""
        from neurospatial.ops import clamp, combine_fields, normalize_field

        assert callable(normalize_field)
        assert callable(clamp)
        assert callable(combine_fields)


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
