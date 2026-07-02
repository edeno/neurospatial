"""Tests for the ``SpikeTrains`` ragged-spike-train container.

``SpikeTrains`` is the one justified new container in Phase 3: ragged per-unit
spike times genuinely do not fit a rectangular array. It must duck-type as a
``SpikeTrainsLike`` group so it flows through the Phase 3.1 spike-input adapter
(:func:`as_spike_trains_with_ids`) via the **iterate-yields-trains** branch:

- it is NOT a ``collections.abc.Mapping`` (so the adapter iterates it),
- it exposes a non-callable ``.index`` property (the unit ids), and
- ``__iter__`` yields the per-unit train arrays (not the ids).

Getting this wrong re-introduces the garbage-trains footgun that 3.1 fixed for a
real pynapple ``TsGroup`` (a ``UserDict`` whose iteration yields KEYS). The
duck-typing tests below are the footgun guard.
"""

from __future__ import annotations

import collections.abc
import dataclasses

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, SpikeTrains
from neurospatial.encoding import SpikeTrains as SpikeTrainsFromEncoding
from neurospatial.encoding import compute_spatial_rates
from neurospatial.encoding._spikes import (
    _looks_like_spike_group,
    as_spike_trains_with_ids,
)


@pytest.fixture
def trains() -> list[np.ndarray]:
    """Three ragged per-unit spike trains (distinct, NaN-free)."""
    return [
        np.array([0.10, 0.55, 1.20], dtype=np.float64),
        np.array([0.20, 0.30, 0.80, 1.90], dtype=np.float64),
        np.array([0.05], dtype=np.float64),
    ]


@pytest.fixture
def unit_table() -> pd.DataFrame:
    """Per-unit metadata table aligned to three units."""
    return pd.DataFrame(
        {
            "region": ["CA1", "CA3", "CA1"],
            "quality": [0.9, 0.4, 0.8],
        }
    )


# ---------------------------------------------------------------------------
# 1. Construction + defaults
# ---------------------------------------------------------------------------


def test_default_unit_ids_are_arange(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains)
    np.testing.assert_array_equal(st.unit_ids, np.arange(len(trains)))


def test_explicit_unit_ids_are_kept(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([10, 20, 30]))
    np.testing.assert_array_equal(st.unit_ids, np.array([10, 20, 30]))


def test_trains_coerced_to_1d_float64() -> None:
    # Integer-valued Python lists get coerced to 1-D float64 arrays.
    st = SpikeTrains([[1, 2, 3], [4, 5]])
    assert all(t.dtype == np.float64 for t in st.trains)
    assert all(t.ndim == 1 for t in st.trains)
    np.testing.assert_array_equal(st.trains[0], np.array([1.0, 2.0, 3.0]))


def test_non_1d_train_raises() -> None:
    with pytest.raises(ValueError, match=r"1-D|1D|one dimension"):
        SpikeTrains([np.array([[0.1, 0.2], [0.3, 0.4]])])


def test_unit_ids_length_mismatch_raises(trains: list[np.ndarray]) -> None:
    with pytest.raises(ValueError, match="length"):
        SpikeTrains(trains, unit_ids=np.array([1, 2]))


def test_unit_table_length_mismatch_raises(trains: list[np.ndarray]) -> None:
    bad_table = pd.DataFrame({"region": ["CA1", "CA3"]})  # 2 rows for 3 units
    with pytest.raises(ValueError, match="unit_table"):
        SpikeTrains(trains, unit_table=bad_table)


def test_duplicate_unit_ids_raise(trains: list[np.ndarray]) -> None:
    with pytest.raises(ValueError, match="unique"):
        SpikeTrains(trains, unit_ids=np.array([5, 5, 7]))


# ---------------------------------------------------------------------------
# 2. Label access (by unit id, NOT position)
# ---------------------------------------------------------------------------


def test_getitem_is_label_access_not_position(trains: list[np.ndarray]) -> None:
    # A permutation of ids proves label lookup: id 0 lives at position 1.
    st = SpikeTrains(trains, unit_ids=np.array([2, 0, 1]))
    np.testing.assert_array_equal(st[0], trains[1])
    np.testing.assert_array_equal(st[2], trains[0])
    np.testing.assert_array_equal(st[1], trains[2])


def test_getitem_string_labels(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains, unit_ids=np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(st["b"], trains[1])


def test_getitem_absent_id_raises_keyerror(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([10, 20, 30]))
    with pytest.raises(KeyError, match="999"):
        st[999]


# ---------------------------------------------------------------------------
# 3. Iteration + len
# ---------------------------------------------------------------------------


def test_iteration_yields_trains_in_order(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([2, 0, 1]))
    yielded = list(st)
    assert len(yielded) == len(trains)
    for got, ref in zip(yielded, trains, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_len_is_n_units(trains: list[np.ndarray]) -> None:
    assert len(SpikeTrains(trains)) == 3


# ---------------------------------------------------------------------------
# 4. filter
# ---------------------------------------------------------------------------


def test_filter_selects_matching_units(
    trains: list[np.ndarray], unit_table: pd.DataFrame
) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([10, 20, 30]), unit_table=unit_table)
    filtered = st.filter("region == 'CA1'")

    # Units 0 and 2 are the CA1 units.
    assert len(filtered) == 2
    np.testing.assert_array_equal(filtered.unit_ids, np.array([10, 30]))
    np.testing.assert_array_equal(filtered.trains[0], trains[0])
    np.testing.assert_array_equal(filtered.trains[1], trains[2])
    assert list(filtered.unit_table["region"]) == ["CA1", "CA1"]


def test_filter_without_unit_table_raises(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains)
    with pytest.raises(ValueError, match="unit_table"):
        st.filter("region == 'CA1'")


def test_filter_returns_new_object_original_unchanged(
    trains: list[np.ndarray], unit_table: pd.DataFrame
) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([10, 20, 30]), unit_table=unit_table)
    filtered = st.filter("region == 'CA1'")

    assert filtered is not st
    # Original is untouched.
    assert len(st) == 3
    np.testing.assert_array_equal(st.unit_ids, np.array([10, 20, 30]))
    assert list(st.unit_table["region"]) == ["CA1", "CA3", "CA1"]


# ---------------------------------------------------------------------------
# 5. Immutability
# ---------------------------------------------------------------------------


def test_reassigning_field_raises(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.unit_ids = np.array([1, 2, 3])  # type: ignore[misc]


def test_trains_stored_as_immutable_tuple(trains: list[np.ndarray]) -> None:
    # trains is stored as a tuple, so in-place mutation raises AttributeError
    # (a tuple has no ``append``) -- this keeps len(unit_ids) == len(trains).
    st = SpikeTrains(trains)
    assert isinstance(st.trains, tuple)
    with pytest.raises(AttributeError):
        st.trains.append(np.array([9.9]))  # type: ignore[attr-defined]


def test_list_input_still_accepted(trains: list[np.ndarray]) -> None:
    # Users pass a plain list; it is accepted and coerced to a tuple.
    st = SpikeTrains(list(trains))
    assert isinstance(st.trains, tuple)
    assert len(st) == len(trains)


def test_top_level_and_encoding_export_same_class() -> None:
    assert SpikeTrains is SpikeTrainsFromEncoding


# ---------------------------------------------------------------------------
# 6. Duck-typing as SpikeTrainsLike (the footgun guard)
# ---------------------------------------------------------------------------


def test_looks_like_spike_group_true_and_not_mapping(
    trains: list[np.ndarray],
) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([7, 8, 9]))
    # Detected as a group (non-callable .index)...
    assert _looks_like_spike_group(st) is True
    # ...but NOT a Mapping, so the adapter takes the ITERATE branch.
    assert not isinstance(st, collections.abc.Mapping)


def test_index_property_is_non_callable_unit_ids(trains: list[np.ndarray]) -> None:
    st = SpikeTrains(trains, unit_ids=np.array([7, 8, 9]))
    assert not callable(st.index)
    np.testing.assert_array_equal(st.index, np.array([7, 8, 9]))


def test_adapter_extracts_actual_trains_and_ids(trains: list[np.ndarray]) -> None:
    unit_ids = np.array([7, 8, 9])
    st = SpikeTrains(trains, unit_ids=unit_ids)

    extracted_trains, extracted_ids = as_spike_trains_with_ids(st)

    np.testing.assert_array_equal(extracted_ids, unit_ids)
    assert len(extracted_trains) == len(trains)
    # NaN-aware equality per train: these are the ACTUAL trains, not the ids.
    for got, ref in zip(extracted_trains, trains, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_adapter_handles_nan_bearing_trains() -> None:
    # A train containing NaN must survive the adapter's iterate branch intact
    # (NaN-aware comparison), proving iteration yields the trains, not ids.
    trains = [np.array([0.1, np.nan, 0.3]), np.array([0.2])]
    st = SpikeTrains(trains, unit_ids=np.array([4, 5]))

    extracted_trains, extracted_ids = as_spike_trains_with_ids(st)

    np.testing.assert_array_equal(extracted_ids, np.array([4, 5]))
    np.testing.assert_array_equal(extracted_trains[0], trains[0])
    np.testing.assert_array_equal(extracted_trains[1], trains[1])


def test_flows_through_compute_spatial_rates_with_ids_preserved() -> None:
    rng = np.random.default_rng(1)
    positions = np.column_stack([np.linspace(0.0, 100.0, 400), rng.uniform(0, 40, 400)])
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0.0, 40.0, 400)
    trains = [
        np.sort(rng.uniform(0.0, 40.0, 50)),
        np.sort(rng.uniform(0.0, 40.0, 30)),
        np.sort(rng.uniform(0.0, 40.0, 40)),
    ]
    unit_ids = np.array([11, 22, 33])
    st = SpikeTrains(trains, unit_ids=unit_ids)

    from_container = compute_spatial_rates(env, st, times, positions, bandwidth=5.0)
    from_arrays = compute_spatial_rates(
        env, list(trains), times, positions, bandwidth=5.0, unit_ids=unit_ids
    )

    # Unit ids surface from the container.
    np.testing.assert_array_equal(from_container.unit_ids, unit_ids)
    # Rates are byte-for-byte identical to the plain-array path.
    np.testing.assert_array_equal(
        np.asarray(from_container.firing_rates),
        np.asarray(from_arrays.firing_rates),
    )
