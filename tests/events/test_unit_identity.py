"""Tests for durable unit identity on peri-event population/single results.

Covers the v0.6 "Unit identity" api-contract rule for the events module:

- ``population_peri_event_histogram`` accepts a keyword-only ``unit_ids=``
  that threads onto ``PopulationPeriEventResult.unit_ids``;
- omitting ``unit_ids`` defaults to ``np.arange(n_units)``;
- a length mismatch raises a clear ``ValueError``;
- string labels round-trip;
- ``PeriEventResult`` gains a singular ``unit_id`` field;
- the new array fields are ``compare=False`` so they do not alter the
  population result's pre-existing equality/hash behavior.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurospatial.events._core import PeriEventResult, PopulationPeriEventResult
from neurospatial.events.alignment import population_peri_event_histogram


@pytest.fixture
def spike_trains() -> list[np.ndarray]:
    """Three units' spike times near three events."""
    return [
        np.array([9.8, 10.1, 19.9, 20.2, 30.0]),
        np.array([10.5, 20.5, 30.5, 30.7]),
        np.array([9.9, 10.0, 20.1, 29.8]),
    ]


@pytest.fixture
def event_times() -> np.ndarray:
    return np.array([10.0, 20.0, 30.0])


STR_IDS = ["mua_a", "mua_b", "mua_c"]


def test_unit_ids_round_trip(spike_trains, event_times):
    ids = np.array([7, 8, 9])
    result = population_peri_event_histogram(
        spike_trains, event_times, window=(-0.5, 1.0), bin_size=0.1, unit_ids=ids
    )
    np.testing.assert_array_equal(result.unit_ids, ids)


def test_unit_ids_default_arange(spike_trains, event_times):
    result = population_peri_event_histogram(
        spike_trains, event_times, window=(-0.5, 1.0), bin_size=0.1
    )
    np.testing.assert_array_equal(result.unit_ids, np.arange(len(spike_trains)))


def test_length_mismatch_raises(spike_trains, event_times):
    with pytest.raises(ValueError, match="unit_ids length mismatch"):
        population_peri_event_histogram(
            spike_trains,
            event_times,
            window=(-0.5, 1.0),
            bin_size=0.1,
            unit_ids=["only_one"],
        )


def test_string_ids_round_trip(spike_trains, event_times):
    result = population_peri_event_histogram(
        spike_trains,
        event_times,
        window=(-0.5, 1.0),
        bin_size=0.1,
        unit_ids=STR_IDS,
    )
    assert list(result.unit_ids) == STR_IDS


def test_population_unit_id_fields_are_compare_false():
    fields = {f.name: f for f in dataclasses.fields(PopulationPeriEventResult)}
    assert fields["unit_ids"].compare is False
    assert fields["unit_table"].compare is False


def test_peri_event_result_has_unit_id():
    """Single-unit PeriEventResult carries an optional unit_id (default None)."""
    bin_centers = np.array([-0.25, 0.25, 0.75])
    histogram = np.array([1.0, 2.0, 3.0])
    sem = np.array([0.1, 0.2, 0.3])
    default = PeriEventResult(
        bin_centers=bin_centers,
        histogram=histogram,
        sem=sem,
        n_events=3,
        window=(-0.5, 1.0),
        bin_size=0.5,
    )
    assert default.unit_id is None

    labeled = PeriEventResult(
        bin_centers=bin_centers,
        histogram=histogram,
        sem=sem,
        n_events=3,
        window=(-0.5, 1.0),
        bin_size=0.5,
        unit_id="ca1_07",
    )
    assert labeled.unit_id == "ca1_07"
    # The cached firing_rate field still computes correctly.
    np.testing.assert_allclose(labeled.firing_rate, histogram / 0.5)


def test_population_getitem_stamps_unit_id(spike_trains, event_times):
    """rates[i] returns a single-unit PeriEventResult with unit_id stamped (I5)."""
    result = population_peri_event_histogram(
        spike_trains,
        event_times,
        window=(-0.5, 1.0),
        bin_size=0.1,
        unit_ids=STR_IDS,
    )
    for i in range(len(spike_trains)):
        child = result[i]
        assert isinstance(child, PeriEventResult)
        assert child.unit_id == result.unit_ids[i]
        # The child's histogram is the i-th row of the population histograms.
        np.testing.assert_array_equal(child.histogram, np.asarray(result.histograms)[i])
        np.testing.assert_array_equal(child.sem, np.asarray(result.sem)[i])
        np.testing.assert_array_equal(child.bin_centers, result.bin_centers)


def test_population_iteration_preserves_order_and_labels(spike_trains, event_times):
    """Iterating yields children whose unit_id matches unit_ids, in order (I5)."""
    result = population_peri_event_histogram(
        spike_trains,
        event_times,
        window=(-0.5, 1.0),
        bin_size=0.1,
        unit_ids=STR_IDS,
    )
    assert [child.unit_id for child in result] == list(result.unit_ids)


def test_population_unit_table_wrong_length_raises(spike_trains, event_times):
    """A wrong-length unit_table raises a clear ValueError (C2)."""
    import pandas as pd

    n_units = len(spike_trains)
    histograms = np.ones((n_units, 3))
    bad_table = pd.DataFrame({"region": ["ca1"] * (n_units + 1)})
    with pytest.raises(ValueError, match="unit_table length mismatch"):
        PopulationPeriEventResult(
            bin_centers=np.array([-0.25, 0.25, 0.75]),
            histograms=histograms,
            sem=np.zeros_like(histograms),
            mean_histogram=histograms.mean(axis=0),
            n_events=3,
            n_units=n_units,
            window=(-0.5, 1.0),
            bin_size=0.5,
            unit_table=bad_table,
        )
