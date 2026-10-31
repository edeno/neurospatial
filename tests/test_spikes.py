"""Tests for neurospatial._spikes module.

This module tests spike format normalization for encoding, decoding and
peri-event functions:
- as_spike_trains: Converts various input formats to canonical list-of-arrays

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
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains(single_neuron_spikes)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_1d_array_element_is_correct_array(
        self, single_neuron_spikes: NDArray[np.float64]
    ) -> None:
        """The element should be a 1D float64 array with same values."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains(single_neuron_spikes)
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float64
        assert result[0].ndim == 1
        np.testing.assert_array_equal(result[0], single_neuron_spikes)

    def test_empty_1d_array(self) -> None:
        """Empty 1D array should return list with one empty array."""
        from neurospatial._spikes import as_spike_trains

        empty = np.array([], dtype=np.float64)
        result = as_spike_trains(empty)
        assert len(result) == 1
        assert len(result[0]) == 0
        assert result[0].dtype == np.float64

    def test_single_spike(self) -> None:
        """Single spike should work correctly."""
        from neurospatial._spikes import as_spike_trains

        single = np.array([1.5])
        result = as_spike_trains(single)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.5])

    def test_1d_integer_array_converted_to_float(self) -> None:
        """Integer arrays should be converted to float64."""
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_2d)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_2d_array_nan_removed(self, multi_neuron_2d: NDArray[np.float64]) -> None:
        """NaN values should be removed from each row."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_2d)
        # Neuron 0: 3 spikes
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.2])
        # Neuron 1: 5 spikes (no NaNs)
        np.testing.assert_array_equal(result[1], [0.2, 0.4, 0.8, 1.5, 2.0])
        # Neuron 2: 2 spikes
        np.testing.assert_array_equal(result[2], [0.3, 0.7])

    def test_2d_array_all_nan_row(self) -> None:
        """Row with all NaNs should return empty array."""
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

        arr = np.array([[0.1, 0.5, np.nan]])
        result = as_spike_trains(arr)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5])

    def test_2d_array_dtype_float64(self, multi_neuron_2d: NDArray[np.float64]) -> None:
        """All output arrays should be float64."""
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains(multi_neuron_list)
        assert isinstance(result, list)
        assert len(result) == len(multi_neuron_list)
        for i, arr in enumerate(result):
            np.testing.assert_array_equal(arr, multi_neuron_list[i])

    def test_tuple_of_arrays(self) -> None:
        """Tuple of 1D arrays should return list."""
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

        # Create a ragged array by casting
        ragged = np.array(
            [np.array([0.1, 0.5]), np.array([0.2])],
            dtype=object,
        )
        with pytest.raises(ValueError, match="ragged"):
            as_spike_trains(ragged)

    def test_3d_array_raises(self) -> None:
        """3D array should raise ValueError."""
        from neurospatial._spikes import as_spike_trains

        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="shape"):
            as_spike_trains(arr)

    def test_list_with_2d_element_raises(self) -> None:
        """List containing 2D array should raise ValueError."""
        from neurospatial._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            np.array([[0.2, 0.3]]),  # 2D - invalid
        ]
        with pytest.raises(ValueError, match="1-D"):
            as_spike_trains(spikes)

    def test_list_with_0d_element_raises(self) -> None:
        """List containing 0D array (scalar) should raise ValueError."""
        from neurospatial._spikes import as_spike_trains

        spikes = [
            np.array([0.1, 0.5]),
            np.array(0.2),  # 0D scalar - invalid
        ]
        with pytest.raises(ValueError, match="1-D"):
            as_spike_trains(spikes)


# ==============================================================================
# Test as_spike_trains: Edge cases
# ==============================================================================


class TestNormalizeSpikeTimesEdgeCases:
    """Tests for edge cases."""

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([])
        assert result == []

    def test_single_element_list(self) -> None:
        """List with single element should work."""
        from neurospatial._spikes import as_spike_trains

        spikes = [np.array([0.1, 0.5, 1.2])]
        result = as_spike_trains(spikes)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.2])

    def test_sorts_a_train_given_out_of_order(self) -> None:
        """A train merged from several sources comes back in time order."""
        from neurospatial._spikes import as_spike_trains

        spikes = np.array([3.0, 1.0, 2.0, 0.5])
        result = as_spike_trains(spikes)
        np.testing.assert_array_equal(result[0], [0.5, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(spikes, [3.0, 1.0, 2.0, 0.5])  # untouched

    def test_sorted_train_is_not_copied(self) -> None:
        from neurospatial._spikes import as_spike_trains

        spikes = np.array([0.5, 1.0, 1.0, 2.0])
        assert as_spike_trains(spikes)[0] is spikes

    def test_negative_spike_times(self) -> None:
        """Negative spike times should be preserved (e.g., relative to event)."""
        from neurospatial._spikes import as_spike_trains

        spikes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = as_spike_trains(spikes)
        np.testing.assert_array_equal(result[0], [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_very_large_number_of_neurons(self) -> None:
        """Should handle large number of neurons efficiently."""
        from neurospatial._spikes import as_spike_trains

        n_neurons = 1000
        rng = np.random.default_rng(42)
        spikes = [rng.random(rng.integers(0, 100)) for _ in range(n_neurons)]
        result = as_spike_trains(spikes)
        assert len(result) == n_neurons

    def test_mixed_types_in_list(self) -> None:
        """List with mixed types (arrays and lists) should work."""
        from neurospatial._spikes import as_spike_trains

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
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([0.1, 0.5, 1.0])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.0])

    def test_list_of_ints_returns_single_neuron(self) -> None:
        """List of ints should be treated as single neuron, converted to float64."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([1, 2, 3])
        assert len(result) == 1
        assert result[0].dtype == np.float64
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_tuple_of_floats_returns_single_neuron(self) -> None:
        """Tuple of floats should be treated as single neuron."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains((0.1, 0.5, 1.0))
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5, 1.0])

    def test_single_float_in_list_returns_single_neuron(self) -> None:
        """List with single float should return one neuron with one spike."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([0.5])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.5])

    def test_list_of_numpy_scalars_returns_single_neuron(self) -> None:
        """List of numpy scalar types should be treated as single neuron."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([np.float64(0.1), np.float64(0.5)])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.1, 0.5])

    def test_list_of_numpy_ints_returns_single_neuron(self) -> None:
        """List of numpy integer types should be treated as single neuron."""
        from neurospatial._spikes import as_spike_trains

        result = as_spike_trains([np.int64(1), np.int64(2), np.int64(3)])
        assert len(result) == 1
        assert result[0].dtype == np.float64
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])


class TestSpikeInputBoundaries:
    """Inputs whose layout or values would silently change the units counted."""

    def test_column_vector_is_refused_not_read_as_many_units(self) -> None:
        from neurospatial._spikes import as_spike_trains

        with pytest.raises(ValueError, match=r"spike_times\[:, 0\]"):
            as_spike_trains(np.arange(5.0)[:, None])

    def test_one_by_one_array_is_one_unit(self) -> None:
        from neurospatial._spikes import as_spike_trains

        (train,) = as_spike_trains(np.array([[0.5]]))
        np.testing.assert_array_equal(train, [0.5])

    def test_units_in_columns_is_refused(self) -> None:
        from neurospatial._spikes import as_spike_trains

        units_in_columns = np.column_stack(
            [np.linspace(0, 10, 50), np.linspace(0.1, 9, 50), np.linspace(1, 5, 50)]
        )
        with pytest.raises(ValueError, match=r"spike_times\.T"):
            as_spike_trains(units_in_columns)

    def test_nan_before_a_spike_is_not_padding(self) -> None:
        from neurospatial._spikes import as_spike_trains

        with pytest.raises(ValueError, match="NaN before a spike"):
            as_spike_trains(np.array([[0.1, np.nan, 0.5], [0.2, 0.3, 0.4]]))

    @pytest.mark.parametrize(
        "spikes",
        [
            np.array([0.1, np.nan, 0.5]),
            [0.1, np.nan],
            [np.array([0.1]), np.array([0.2, np.inf])],
            np.array([[0.1, np.inf], [0.2, 0.3]]),
            {"a": np.array([0.1, np.nan])},
        ],
        ids=["1-D", "scalars", "list", "2-D inf", "dict"],
    )
    def test_non_finite_spike_times_are_refused(self, spikes: object) -> None:
        from neurospatial._spikes import as_spike_trains

        with pytest.raises(ValueError, match="NaN or infinite"):
            as_spike_trains(spikes)

    def test_scalar_input_is_refused(self) -> None:
        from neurospatial._spikes import as_spike_trains

        with pytest.raises(ValueError, match=r"got shape \(\)"):
            as_spike_trains(np.float64(0.5))

    def test_pandas_series_is_one_unit(self) -> None:
        pd = pytest.importorskip("pandas")
        from neurospatial._spikes import as_labelled_spike_trains

        trains, ids = as_labelled_spike_trains(
            pd.Series([0.1, 0.5, 1.2], index=[10, 11, 12]), None, context="demo"
        )
        assert len(trains) == 1
        np.testing.assert_array_equal(trains[0], [0.1, 0.5, 1.2])
        assert ids is None

    def test_pynapple_ts_is_one_unit(self) -> None:
        nap = pytest.importorskip("pynapple")
        from neurospatial._spikes import as_labelled_spike_trains, as_spike_trains

        ts = nap.Ts(t=np.array([0.1, 0.5, 1.2]))
        trains, ids = as_labelled_spike_trains(ts, None, context="demo")
        np.testing.assert_array_equal(trains[0], [0.1, 0.5, 1.2])
        assert ids is None
        np.testing.assert_array_equal(as_spike_trains(ts)[0], [0.1, 0.5, 1.2])

    def test_pynapple_trains_read_by_timestamps_on_every_route(self) -> None:
        """A Tsd carries data in its values; every route reads its timestamps."""
        nap = pytest.importorskip("pynapple")
        from neurospatial._spikes import as_spike_trains

        a = nap.Tsd(t=np.array([1.0, 2.0, 3.0]), d=np.array([55.0, 60.0, 70.0]))
        b = nap.Tsd(t=np.array([1.5, 2.5]), d=np.array([80.0, 90.0]))
        for spikes in ([a, b], (a, b), {0: a, 1: b}):
            trains = as_spike_trains(spikes)
            np.testing.assert_array_equal(trains[0], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(trains[1], [1.5, 2.5])

    def test_t_attribute_of_other_objects_is_not_read(self) -> None:
        """Only pynapple objects are read by ``.t``: an xarray DataArray with
        a ``t`` dimension exposes its index there, not the spike times."""
        xr = pytest.importorskip("xarray")
        from neurospatial._spikes import as_spike_trains

        train = xr.DataArray([0.5, 0.7], dims="t")
        np.testing.assert_array_equal(as_spike_trains({0: train})[0], [0.5, 0.7])

    def test_dict_of_trains_keeps_unit_labels(self) -> None:
        from neurospatial._spikes import as_labelled_spike_trains

        trains, ids = as_labelled_spike_trains(
            {"ca1_3": [0.1, 0.2], "ca1_7": np.array([0.5])}, None, context="demo"
        )
        assert ids.tolist() == ["ca1_3", "ca1_7"]
        np.testing.assert_array_equal(trains[1], [0.5])

    def test_dataframe_is_refused_with_a_conversion(self) -> None:
        pd = pytest.importorskip("pandas")
        from neurospatial._spikes import as_spike_trains

        frame = pd.DataFrame({"u0": [0.1, 0.2], "u1": [0.3, np.nan]})
        with pytest.raises(ValueError, match="DataFrame"):
            as_spike_trains(frame)

    def test_decoding_counts_refuse_nan_spikes(self) -> None:
        from neurospatial.decoding import bin_spikes_in_time

        with pytest.raises(ValueError, match="NaN or infinite"):
            bin_spikes_in_time(
                [np.array([0.05, np.nan, 0.3])], dt=0.25, t_start=0.0, t_stop=1.0
            )

    def test_decoding_counts_refuse_infinite_bounds(self) -> None:
        from neurospatial.decoding import bin_spikes_in_time

        with pytest.raises(ValueError, match="finite t_start and t_stop"):
            bin_spikes_in_time([np.array([0.05])], dt=0.25, t_stop=np.inf)


class TestRastersAndZeroPadding:
    """Raster values and zero padding must never be read as spike times."""

    @pytest.mark.parametrize(
        "spikes",
        [
            np.array([False, False, True, False, True]),
            [True, False, True],
            np.array([[False, True, False], [True, False, False]]),
            {"u1": np.array([False, True, True])},
            np.array([0, 0, 1, 0, 1]),
            [0, 1, 1, 0],
            [np.array([0.1, 0.5]), np.array([0, 1, 0])],
            np.array([[0, 1, 0], [1, 0, 0]]),
            np.array([0.0, 1.0, 1.0, 0.0]),
        ],
        ids=[
            "bool 1-D",
            "list of bools",
            "bool 2-D",
            "bool dict",
            "int 1-D",
            "list of ints",
            "int in list of trains",
            "int 2-D",
            "float 0/1",
        ],
    )
    def test_spike_raster_is_refused(self, spikes: object) -> None:
        from neurospatial._spikes import as_spike_trains

        with pytest.raises(
            ValueError, match=r"(?s)spike raster.*times\[raster\.astype\(bool\)\]"
        ):
            as_spike_trains(spikes)

    @pytest.mark.parametrize(
        ("spikes", "expected"),
        [
            (np.array([0, 1]), [0.0, 1.0]),
            (np.array([1]), [1.0]),
            (np.array([0, 1, 2, 5]), [0.0, 1.0, 2.0, 5.0]),
            (np.array([0.0, 0.0, 0.5]), [0.0, 0.0, 0.5]),
        ],
        ids=["distinct 0 and 1", "one spike", "integer seconds", "repeat at 0"],
    )
    def test_spike_times_near_zero_and_one_are_kept(
        self, spikes: NDArray[np.float64], expected: list[float]
    ) -> None:
        """Only a train of 0s and 1s that repeats a value reads as a raster."""
        from neurospatial._spikes import as_spike_trains

        np.testing.assert_array_equal(as_spike_trains(spikes)[0], expected)

    def test_zero_padded_rows_are_refused(self) -> None:
        from neurospatial._spikes import as_spike_trains

        padded = np.array([[10.0, 20.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]])
        with pytest.raises(ValueError, match=r"(?s)row\(s\) \[0\] decrease.*zeros"):
            as_spike_trains(padded)

    def test_unsorted_trains_are_sorted_with_their_labels(self) -> None:
        """Each train is sorted in place of itself; labels stay with it."""
        from neurospatial._spikes import as_labelled_spike_trains

        trains, ids = as_labelled_spike_trains(
            {"b": np.array([0.5, 0.1, 0.3]), "a": np.array([-2.0, -3.0])},
            None,
            context="demo",
        )
        assert ids.tolist() == ["b", "a"]
        np.testing.assert_array_equal(trains[0], [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(trains[1], [-3.0, -2.0])

    def test_spike_trains_container_stores_sorted_read_only_copies(self) -> None:
        from neurospatial.encoding import SpikeTrains

        given = np.array([0.5, 0.1, 0.3])
        stored = SpikeTrains([given], unit_ids=["u"]).trains[0]
        np.testing.assert_array_equal(stored, [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(given, [0.5, 0.1, 0.3])
        assert not stored.flags.writeable

    def test_single_train_is_sorted(self) -> None:
        from neurospatial._spikes import as_spike_train

        np.testing.assert_array_equal(as_spike_train([0.4, -0.1]), [-0.1, 0.4])
        with pytest.raises(ValueError, match="1-D"):
            as_spike_train(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_decoding_counts_refuse_raster_and_zero_padding(self) -> None:
        from neurospatial.decoding import bin_spikes_in_time

        raster = np.zeros(10, dtype=bool)
        raster[[3, 7]] = True
        with pytest.raises(ValueError, match="spike raster"):
            bin_spikes_in_time([raster], dt=0.5, t_start=0.0, t_stop=5.0)
        padded = np.array([[10.0, 20.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]])
        with pytest.raises(ValueError, match="decrease"):
            bin_spikes_in_time(padded, dt=1.0, t_start=0.0, t_stop=30.0)


class TestLabelledSpikeContainers:
    """Containers that carry unit ids keep them, whatever the id type."""

    def test_series_of_trains_is_a_group_keyed_by_index(self) -> None:
        pd = pytest.importorskip("pandas")
        from neurospatial._spikes import as_labelled_spike_trains

        # As from units.to_dataframe()["spike_times"] or
        # df.groupby("unit")["time"].apply(np.array).
        series = pd.Series(
            [np.array([0.1, 0.2]), np.array([0.3])], index=[7, 9], name="spike_times"
        )
        trains, ids = as_labelled_spike_trains(series, None, context="demo")
        assert ids.tolist() == [7, 9]
        np.testing.assert_array_equal(trains[0], [0.1, 0.2])
        np.testing.assert_array_equal(trains[1], [0.3])

    def test_groupby_long_table_round_trips(self) -> None:
        pd = pytest.importorskip("pandas")
        from neurospatial._spikes import as_labelled_spike_trains

        table = pd.DataFrame(
            {"unit": ["b", "a", "b", "a", "a"], "time": [0.4, 0.1, 0.5, 0.2, 0.3]}
        )
        series = table.groupby("unit")["time"].apply(np.array)
        trains, ids = as_labelled_spike_trains(series, None, context="demo")
        assert ids.tolist() == ["a", "b"]
        np.testing.assert_array_equal(trains[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(trains[1], [0.4, 0.5])

    def test_tuple_unit_ids_stay_one_label_per_unit(self) -> None:
        from neurospatial._spikes import as_labelled_spike_trains

        trains, ids = as_labelled_spike_trains(
            {(1, 3): [0.1], (2, 5): [0.2, 0.3]}, None, context="demo"
        )
        assert ids.shape == (2,)
        assert ids.tolist() == [(1, 3), (2, 5)]
        np.testing.assert_array_equal(trains[1], [0.2, 0.3])

    def test_tuple_unit_ids_can_be_passed_explicitly(self) -> None:
        from neurospatial._spikes import as_labelled_spike_trains

        _, ids = as_labelled_spike_trains(
            [[0.1], [0.2, 0.3]], [(1, 3), (2, 5)], context="demo"
        )
        assert ids.tolist() == [(1, 3), (2, 5)]
