"""Unit tests for the shared input-validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial._validation import validate_finite, validate_lengths


class TestValidateFinite:
    """Tests for ``validate_finite``."""

    def test_clean_finite_input_returns_float64_preserving_values(self) -> None:
        result = validate_finite([1.0, 2.5, -3.0], name="x")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.5, -3.0]))

    def test_accepts_list_input_and_returns_float64(self) -> None:
        result = validate_finite([0, 1, 2], name="counts")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))

    def test_raises_on_positive_inf(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.inf, 3.0], name="x")

    def test_raises_on_negative_inf(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, -np.inf, 3.0], name="x")

    def test_raises_on_nan_by_default(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.nan, 3.0], name="x")

    def test_allows_nan_when_requested(self) -> None:
        result = validate_finite([1.0, np.nan, 3.0], name="x", allow_nan=True)
        assert result.dtype == np.float64
        assert np.isnan(result[1])
        assert result[0] == 1.0
        assert result[2] == 3.0

    def test_still_raises_on_inf_even_when_allow_nan(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([1.0, np.inf, np.nan], name="x", allow_nan=True)

    def test_still_raises_on_neg_inf_even_when_allow_nan(self) -> None:
        with pytest.raises(ValueError):
            validate_finite([np.nan, -np.inf], name="x", allow_nan=True)

    def test_error_message_contains_name_count_and_first_index(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_finite([1.0, 2.0, np.inf, np.inf], name="velocity")
        message = str(exc_info.value)
        assert "velocity" in message
        # Two offending values.
        assert "2 non-finite value(s)" in message
        # First offending value is at index 2.
        assert "index 2" in message

    def test_error_message_reports_first_nan_index(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_finite([0.0, 0.0, np.nan], name="signal")
        message = str(exc_info.value)
        assert "signal" in message
        assert "1 non-finite value(s)" in message
        assert "index 2" in message

    def test_works_on_2d_array_first_offending_index_via_flat(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, np.inf]])
        with pytest.raises(ValueError) as exc_info:
            validate_finite(arr, name="grid")
        message = str(exc_info.value)
        assert "grid" in message
        # Flattened index of the inf (row 1, col 1) is 3.
        assert "index 3" in message

    def test_2d_clean_input_preserved(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_finite(arr, name="grid")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_raises_valueerror_exact_type(self) -> None:
        # Ensure the raised exception is exactly ValueError, not a subclass.
        with pytest.raises(ValueError):
            validate_finite([np.nan], name="x")


class TestValidateLengths:
    """Tests for ``validate_lengths``."""

    def test_single_entry_passes(self) -> None:
        # A single-entry mapping can never mismatch; should not raise.
        validate_lengths({"only": np.array([1.0, 2.0, 3.0])})

    def test_all_equal_multi_entry_passes(self) -> None:
        s = np.array([0.1, 0.2, 0.3])
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # positions is 2-D but has len 3, matching s and t.
        validate_lengths({"spike_times": s, "times": t, "positions": p})

    def test_empty_mapping_passes(self) -> None:
        # No arrays -> no mismatch possible.
        validate_lengths({})

    def test_raises_on_length_mismatch_with_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "spike_times": np.array([0.1, 0.2, 0.3]),
                    "times": np.array([0.0, 1.0]),
                }
            )
        message = str(exc_info.value)
        assert "Length mismatch" in message
        assert "spike_times=3" in message
        assert "times=2" in message

    def test_length_one_among_longer_is_mismatch_not_broadcastable(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "scalar_like": np.array([5.0]),
                    "times": np.array([0.0, 1.0, 2.0]),
                }
            )
        message = str(exc_info.value)
        assert "Length mismatch" in message
        assert "scalar_like=1" in message
        assert "times=3" in message

    def test_message_lists_each_name_and_length(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_lengths(
                {
                    "a": np.array([0.0, 1.0]),
                    "b": np.array([0.0, 1.0, 2.0]),
                    "c": np.array([0.0]),
                }
            )
        message = str(exc_info.value)
        assert "a=2" in message
        assert "b=3" in message
        assert "c=1" in message

    def test_raises_valueerror_exact_type(self) -> None:
        with pytest.raises(ValueError):
            validate_lengths({"a": np.array([1.0]), "b": np.array([1.0, 2.0])})
