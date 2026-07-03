"""Tests for neurospatial.encoding._spikes module.

This module tests spike format normalization for encoding functions:
- as_spike_trains: Converts various input formats to canonical list-of-arrays

TDD approach: Tests written first, implementation follows.

Input formats supported:
- 1D array (single neuron) → wrapped in list
- 2D array (n_neurons, max_spikes) with NaN padding → split, NaNs removed
- list/tuple of 1D arrays (canonical) → each element converted to array
- Ragged object arrays → rejected with clear error

Output: list[NDArray[np.float64]] - list of 1D spike time arrays
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def single_neuron_spikes() -> NDArray[np.float64]:
    """Single neuron spike times as 1D array."""
    return np.array([0.1, 0.5, 1.2, 2.3, 3.0])


@pytest.fixture
def multi_neuron_2d() -> NDArray[np.float64]:
    """Multiple neurons as 2D array with NaN padding.

    Shape: (3, 5) where:
    - Neuron 0 has 3 spikes: [0.1, 0.5, 1.2]
    - Neuron 1 has 5 spikes: [0.2, 0.4, 0.8, 1.5, 2.0]
    - Neuron 2 has 2 spikes: [0.3, 0.7]
    """
    return np.array(
        [
            [0.1, 0.5, 1.2, np.nan, np.nan],
            [0.2, 0.4, 0.8, 1.5, 2.0],
            [0.3, 0.7, np.nan, np.nan, np.nan],
        ]
    )


@pytest.fixture
def multi_neuron_list() -> list[NDArray[np.float64]]:
    """Multiple neurons as list of 1D arrays (canonical format)."""
    return [
        np.array([0.1, 0.5, 1.2]),
        np.array([0.2, 0.4, 0.8, 1.5, 2.0]),
        np.array([0.3, 0.7]),
    ]


# ==============================================================================
# Test as_spike_trains: 1D array input (single neuron)
# ==============================================================================


class TestNormalizeSpikeTimesSingleNeuron:
    """Tests for 1D array input (single neuron)."""

    def test_1d_array_returns_list_of_length_1(
        self, single_neuron_spikes: NDArray[np.float64]
    ) -> None:
        """1D array should return list with one element."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(single_neuron_spikes)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_1d_array_element_is_correct_array(
        self, single_neuron_spikes: NDArray[np.float64]
    ) -> None:
        """The element should be a 1D float64 array with same values."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(single_neuron_spikes)
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float64
        assert result[0].ndim == 1
        np.testing.assert_array_equal(result[0], single_neuron_spikes)

    def test_empty_1d_array(self) -> None:
        """Empty 1D array should return list with one empty array."""
        from neurospatial.encoding._spikes import as_spike_trains

        empty = np.array([], dtype=np.float64)
        result = as_spike_trains(empty)
        assert len(result) == 1
        assert len(result[0]) == 0
        assert result[0].dtype == np.float64

    def test_single_spike(self) -> None:
        """Single spike should work correctly."""
        from neurospatial.encoding._spikes import as_spike_trains

        single = np.array([1.5])
        result = as_spike_trains(single)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.5])

    def test_1d_integer_array_converted_to_float(self) -> None:
        """Integer arrays should be converted to float64."""
        from neurospatial.encoding._spikes import as_spike_trains

        int_spikes = np.array([1, 2, 3])
        result = as_spike_trains(int_spikes)
        assert result[0].dtype == np.float64
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])


# ==============================================================================
# Test as_spike_trains: 2D array input (NaN-padded)
# ==============================================================================


class TestNormalizeSpikeTimesNaNPadded:
    """Tests for 2D NaN-padded array input."""

    def test_2d_array_returns_list_of_correct_length(
        self, multi_neuron_2d: NDArray[np.float64]
    ) -> None:
        """2D array should return list with one element per row."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_2d)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_2d_array_nan_removed(self, multi_neuron_2d: NDArray[np.float64]) -> None:
        """NaN values should be removed from each row."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_2d)
        # Neuron 0: 3 spikes
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.2])
        # Neuron 1: 5 spikes (no NaNs)
        np.testing.assert_array_equal(result[1], [0.2, 0.4, 0.8, 1.5, 2.0])
        # Neuron 2: 2 spikes
        np.testing.assert_array_equal(result[2], [0.3, 0.7])

    def test_2d_array_all_nan_row(self) -> None:
        """Row with all NaNs should return empty array."""
        from neurospatial.encoding._spikes import as_spike_trains

        arr = np.array(
            [
                [0.1, 0.5, np.nan],
                [np.nan, np.nan, np.nan],  # All NaN
                [0.3, np.nan, np.nan],
            ]
        )
        result = as_spike_trains(arr)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0.1, 0.5])
        assert len(result[1]) == 0  # Empty array for all-NaN row
        np.testing.assert_array_equal(result[2], [0.3])

    def test_2d_array_no_nan(self) -> None:
        """2D array without NaNs should split rows correctly."""
        from neurospatial.encoding._spikes import as_spike_trains

        arr = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        result = as_spike_trains(arr)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(result[1], [0.4, 0.5, 0.6])

    def test_2d_array_single_row(self) -> None:
        """2D array with single row should return list of length 1."""
        from neurospatial.encoding._spikes import as_spike_trains

        arr = np.array([[0.1, 0.5, np.nan]])
        result = as_spike_trains(arr)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5])

    def test_2d_array_dtype_float64(self, multi_neuron_2d: NDArray[np.float64]) -> None:
        """All output arrays should be float64."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_2d)
        for arr in result:
            assert arr.dtype == np.float64


# ==============================================================================
# Test as_spike_trains: list/tuple of arrays (canonical format)
# ==============================================================================


class TestNormalizeSpikeTimesListInput:
    """Tests for list of arrays input (canonical format)."""

    def test_list_of_arrays_passthrough(
        self, multi_neuron_list: list[NDArray[np.float64]]
    ) -> None:
        """List of 1D arrays should return equivalent list."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_list)
        assert isinstance(result, list)
        assert len(result) == len(multi_neuron_list)
        for i, arr in enumerate(result):
            np.testing.assert_array_equal(arr, multi_neuron_list[i])

    def test_tuple_of_arrays(self) -> None:
        """Tuple of 1D arrays should return list."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = (
            np.array([0.1, 0.5]),
            np.array([0.2, 0.3, 0.8]),
        )
        result = as_spike_trains(spikes)
        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0.1, 0.5])
        np.testing.assert_array_equal(result[1], [0.2, 0.3, 0.8])

    def test_list_of_lists(self) -> None:
        """List of Python lists should be converted to arrays."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            [0.1, 0.5, 1.2],
            [0.2, 0.3],
        ]
        result = as_spike_trains(spikes)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.2])
        np.testing.assert_array_equal(result[1], [0.2, 0.3])

    def test_list_with_empty_arrays(self) -> None:
        """List containing empty arrays should preserve them."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            np.array([]),  # Empty
            np.array([0.3]),
        ]
        result = as_spike_trains(spikes)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0.1, 0.5])
        assert len(result[1]) == 0
        np.testing.assert_array_equal(result[2], [0.3])

    def test_list_dtype_conversion(self) -> None:
        """List elements should be converted to float64."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        result = as_spike_trains(spikes)
        assert result[0].dtype == np.float64
        assert result[1].dtype == np.float64


# ==============================================================================
# Test as_spike_trains: Error handling
# ==============================================================================


class TestNormalizeSpikeTimesErrors:
    """Tests for error handling."""

    def test_ragged_object_array_raises(self) -> None:
        """Ragged array (dtype=object) should raise ValueError."""
        from neurospatial.encoding._spikes import as_spike_trains

        # Create a ragged array by casting
        ragged = np.array(
            [np.array([0.1, 0.5]), np.array([0.2])],
            dtype=object,
        )
        with pytest.raises(ValueError, match="ragged"):
            as_spike_trains(ragged)

    def test_3d_array_raises(self) -> None:
        """3D array should raise ValueError."""
        from neurospatial.encoding._spikes import as_spike_trains

        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="shape"):
            as_spike_trains(arr)

    def test_list_with_2d_element_raises(self) -> None:
        """List containing 2D array should raise ValueError."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            np.array([[0.2, 0.3]]),  # 2D - invalid
        ]
        with pytest.raises(ValueError, match="1D"):
            as_spike_trains(spikes)

    def test_list_with_0d_element_raises(self) -> None:
        """List containing 0D array (scalar) should raise ValueError."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            np.array(0.2),  # 0D scalar - invalid
        ]
        with pytest.raises(ValueError, match="1D"):
            as_spike_trains(spikes)


# ==============================================================================
# Test as_spike_trains: Edge cases
# ==============================================================================


class TestNormalizeSpikeTimesEdgeCases:
    """Tests for edge cases."""

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([])
        assert result == []

    def test_single_element_list(self) -> None:
        """List with single element should work."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [np.array([0.1, 0.5, 1.2])]
        result = as_spike_trains(spikes)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.2])

    def test_preserves_spike_order(self) -> None:
        """Spike times should maintain their order."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = np.array([3.0, 1.0, 2.0, 0.5])  # Unsorted
        result = as_spike_trains(spikes)
        np.testing.assert_array_equal(result[0], [3.0, 1.0, 2.0, 0.5])

    def test_negative_spike_times(self) -> None:
        """Negative spike times should be preserved (e.g., relative to event)."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = as_spike_trains(spikes)
        np.testing.assert_array_equal(result[0], [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_very_large_number_of_neurons(self) -> None:
        """Should handle large number of neurons efficiently."""
        from neurospatial.encoding._spikes import as_spike_trains

        n_neurons = 1000
        rng = np.random.default_rng(42)
        spikes = [rng.random(rng.integers(0, 100)) for _ in range(n_neurons)]
        result = as_spike_trains(spikes)
        assert len(result) == n_neurons

    def test_mixed_types_in_list(self) -> None:
        """List with mixed types (arrays and lists) should work."""
        from neurospatial.encoding._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            [0.2, 0.3, 0.4],  # Python list
            (0.6, 0.7),  # Tuple
        ]
        result = as_spike_trains(spikes)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0.1, 0.5])
        np.testing.assert_array_equal(result[1], [0.2, 0.3, 0.4])
        np.testing.assert_array_equal(result[2], [0.6, 0.7])


# ==============================================================================
# Test imports
# ==============================================================================


class TestNormalizeSpikeTimesListOfScalars:
    """Tests for list of scalars input (single neuron).

    This is a common user input pattern: [0.1, 0.5, 1.0] representing
    spike times for a single neuron.
    """

    def test_list_of_floats_returns_single_neuron(self) -> None:
        """List of floats should be treated as single neuron."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([0.1, 0.5, 1.0])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.0])

    def test_list_of_ints_returns_single_neuron(self) -> None:
        """List of ints should be treated as single neuron, converted to float64."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([1, 2, 3])
        assert len(result) == 1
        assert result[0].dtype == np.float64
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_tuple_of_floats_returns_single_neuron(self) -> None:
        """Tuple of floats should be treated as single neuron."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains((0.1, 0.5, 1.0))
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.0])

    def test_single_float_in_list_returns_single_neuron(self) -> None:
        """List with single float should return one neuron with one spike."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([0.5])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.5])

    def test_list_of_numpy_scalars_returns_single_neuron(self) -> None:
        """List of numpy scalar types should be treated as single neuron."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([np.float64(0.1), np.float64(0.5)])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5])

    def test_list_of_numpy_ints_returns_single_neuron(self) -> None:
        """List of numpy integer types should be treated as single neuron."""
        from neurospatial.encoding._spikes import as_spike_trains

        result = as_spike_trains([np.int64(1), np.int64(2), np.int64(3)])
        assert len(result) == 1
        assert result[0].dtype == np.float64
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])


class TestSpikesImports:
    """Test that expected items are importable from _spikes module."""

    def test_as_spike_trains_importable(self) -> None:
        """as_spike_trains should be importable from encoding._spikes."""
        from neurospatial.encoding._spikes import as_spike_trains

        assert callable(as_spike_trains)
