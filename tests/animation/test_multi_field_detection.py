"""Tests for hardened multi-field detection (Phase 3.4).

This module tests the robustness of multi-field vs single-field detection
in the animation pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the detection function directly for unit testing
from neurospatial.animation.backends.napari_backend import _is_multi_field_input

# Check for optional dependencies
try:
    import napari  # noqa: F401

    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False


class TestMultiFieldDetectionEdgeCases:
    """Test edge cases for multi-field detection."""

    def test_single_field_list_of_arrays(self):
        """Single-field: list of numpy arrays should return False."""
        fields = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        assert _is_multi_field_input(fields) is False

    def test_multi_field_list_of_lists(self):
        """Multi-field: list of lists should return True."""
        fields = [[np.array([1, 2]), np.array([3, 4])], [np.array([5, 6])]]
        assert _is_multi_field_input(fields) is True

    def test_multi_field_list_of_tuples(self):
        """Multi-field: list of tuples should return True."""
        fields = [(np.array([1, 2]),), (np.array([3, 4]),)]
        assert _is_multi_field_input(fields) is True

    def test_empty_list(self):
        """Empty list should return False (single-field)."""
        assert _is_multi_field_input([]) is False

    def test_single_array_element(self):
        """Single array in list should return False."""
        fields = [np.array([1, 2, 3])]
        assert _is_multi_field_input(fields) is False

    def test_single_list_element_is_multi(self):
        """Single list in list should return True (multi-field with 1 sequence)."""
        fields = [[np.array([1, 2]), np.array([3, 4])]]
        assert _is_multi_field_input(fields) is True


@pytest.mark.skipif(not HAS_NAPARI, reason="napari not installed")
class TestMultiFieldValidationRobustness:
    """Test validation catches invalid inputs."""

    @pytest.fixture
    def simple_env(self):
        """Create simple 2D environment for testing."""
        from neurospatial import Environment

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        return Environment.from_samples(positions, bin_size=10.0)

    def test_mixed_types_raises_error(self, simple_env):
        """Mixed types (some lists, some arrays) should raise ValueError.

        This is the key robustness test - the old implementation only checked
        fields[0], which would incorrectly classify this as multi-field.
        """
        from neurospatial.animation.backends.napari_backend import render_napari

        rng = np.random.default_rng(42)

        # First element is a list, second is an array - invalid mixed input
        fields = [
            [rng.random(simple_env.n_bins) for _ in range(5)],  # list
            rng.random(simple_env.n_bins),  # array - wrong!
        ]

        with pytest.raises(ValueError, match=r"inconsistent|mixed|type"):
            render_napari(simple_env, fields, layout="horizontal")

    def test_all_elements_validated_not_just_first(self, simple_env):
        """Validation should check all elements, not just fields[0].

        If only the first element is checked, [array, list, list] would be
        incorrectly classified as single-field.
        """
        from neurospatial.animation.backends.napari_backend import render_napari

        rng = np.random.default_rng(42)

        # First element is array, but subsequent elements are lists - invalid
        fields = [
            rng.random(simple_env.n_bins),  # array
            [rng.random(simple_env.n_bins) for _ in range(5)],  # list - wrong!
        ]

        # This should fail because types are mixed
        with pytest.raises(ValueError, match=r"inconsistent|mixed|type"):
            render_napari(simple_env, fields)


class TestMismatchedShapeErrors:
    """Test that mismatched shapes raise correct errors."""

    @pytest.fixture
    def simple_env(self):
        """Create simple 2D environment for testing."""
        from neurospatial import Environment

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        return Environment.from_samples(positions, bin_size=10.0)

    def test_single_field_wrong_bins_error(self, simple_env):
        """Single-field with wrong number of bins should raise clear error."""
        from neurospatial.animation.core import animate_fields

        rng = np.random.default_rng(42)
        wrong_n_bins = simple_env.n_bins + 10
        fields = [rng.random(wrong_n_bins) for _ in range(5)]

        with pytest.raises(ValueError, match=r"values but environment has"):
            animate_fields(simple_env, fields, backend="napari")

    def test_multi_field_sequences_different_lengths(self, simple_env):
        """Multi-field sequences with different lengths should raise error."""
        from neurospatial.animation.backends.napari_backend import render_napari

        rng = np.random.default_rng(42)
        seq1 = [rng.random(simple_env.n_bins) for _ in range(10)]
        seq2 = [rng.random(simple_env.n_bins) for _ in range(5)]  # different!
        fields = [seq1, seq2]

        with pytest.raises(ValueError, match="same length"):
            render_napari(simple_env, fields, layout="horizontal")

    def test_multi_field_inner_array_wrong_bins(self, simple_env):
        """Multi-field inner arrays with wrong bins should raise error."""
        from neurospatial.animation.backends.napari_backend import render_napari

        rng = np.random.default_rng(42)
        wrong_n_bins = simple_env.n_bins + 10
        seq1 = [rng.random(simple_env.n_bins) for _ in range(5)]
        seq2 = [rng.random(wrong_n_bins) for _ in range(5)]  # wrong bins!
        fields = [seq1, seq2]

        # This should raise an error about bin mismatch
        with pytest.raises(ValueError, match=r"bins|shape"):
            render_napari(simple_env, fields, layout="horizontal")


class TestNdarrayIsNotSequence:
    """Test that numpy arrays are correctly identified as arrays, not sequences.

    ndarray is technically iterable and has __len__, but should be treated
    as a single field element, not as a multi-field sequence.
    """

    def test_ndarray_not_detected_as_sequence(self):
        """2D ndarray should not trigger multi-field detection."""
        # A 2D array where rows could be mistaken for sequence elements
        fields = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # This should be treated as invalid input (expecting list of arrays)
        # OR detected as single-field if we treat the whole array as one field
        # The key is: it should NOT be detected as multi-field
        assert _is_multi_field_input([fields]) is False

    def test_list_of_1d_arrays_is_single_field(self):
        """List of 1D arrays should be single-field."""
        fields = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        assert _is_multi_field_input(fields) is False

    def test_list_of_2d_arrays_is_single_field(self):
        """List of 2D arrays should still be single-field (each array is a frame)."""
        fields = [np.array([[1], [2]]), np.array([[3], [4]])]
        assert _is_multi_field_input(fields) is False
