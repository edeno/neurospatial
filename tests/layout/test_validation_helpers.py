"""Behavioral tests for the public layout validators."""

import numpy as np
import pytest

from neurospatial.layout.validation import (
    validate_bin_size,
    validate_dimension_ranges,
)


class TestValidateBinSize:
    def test_scalar_promoted_to_array(self):
        np.testing.assert_array_equal(validate_bin_size(2.0), np.array([2.0]))

    def test_per_dimension_array_passthrough(self):
        np.testing.assert_array_equal(
            validate_bin_size(np.array([2.0, 3.0])), np.array([2.0, 3.0])
        )

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(0.0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(-1.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_bin_size(np.nan)

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="infinite"):
            validate_bin_size(np.inf)

    def test_one_bad_entry_in_array_rejected(self):
        # Mixed valid/invalid: the negative entry must still trip the guard.
        with pytest.raises(ValueError, match="must be positive"):
            validate_bin_size(np.array([2.0, -1.0]))


class TestValidateDimensionRanges:
    def test_valid_ranges_returned_as_float_tuples(self):
        result = validate_dimension_ranges([(0, 100), (0, 200)])
        assert result == [(0.0, 100.0), (0.0, 200.0)]

    def test_n_dims_match_ok(self):
        result = validate_dimension_ranges([(0, 100), (0, 200)], n_dims=2)
        assert result == [(0.0, 100.0), (0.0, 200.0)]

    def test_n_dims_mismatch_rejected(self):
        with pytest.raises(ValueError, match="expected 2"):
            validate_dimension_ranges([(0, 100)], n_dims=2)

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="min >= max"):
            validate_dimension_ranges([(100, 0)])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dimension_ranges([])

    def test_non_sequence_rejected(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            validate_dimension_ranges("not-a-list")  # type: ignore[arg-type]

    def test_non_finite_bound_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_dimension_ranges([(0.0, np.inf)])

    def test_wrong_tuple_length_rejected(self):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_dimension_ranges([(0.0, 1.0, 2.0)])  # type: ignore[list-item]
