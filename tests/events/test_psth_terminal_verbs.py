"""Contract tests for PSTH terminal verbs (Task 1.3).

``PeriEventResult`` and ``PopulationPeriEventResult`` gained ``ResultMixin`` and
the terminal verbs. These tests assert:

- the frozen-dataclass construction (cached ``firing_rate(s)`` fields) still
  works after inheriting ``ResultMixin``;
- ``to_dataframe()`` is dense and carries ``unit_id``;
- ``summary()`` / ``summary_table()`` exist and round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurospatial._results import ResultMixin
from neurospatial.events._core import PeriEventResult, PopulationPeriEventResult


def _single() -> PeriEventResult:
    bc = np.array([-0.5, 0.0, 0.5])
    return PeriEventResult(
        bin_centers=bc,
        histogram=np.array([0.1, 0.4, 0.2]),
        sem=np.array([0.01, 0.02, 0.01]),
        n_events=10,
        window=(-0.75, 0.75),
        bin_size=0.5,
        unit_id="u0",
    )


def _population() -> PopulationPeriEventResult:
    bc = np.array([-0.5, 0.0, 0.5])
    hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
    return PopulationPeriEventResult(
        bin_centers=bc,
        histograms=hist,
        sem=np.zeros_like(hist),
        mean_histogram=hist.mean(axis=0),
        n_events=10,
        n_units=2,
        window=(-0.75, 0.75),
        bin_size=0.5,
    )


def test_psth_results_inherit_canonical_result_mixin():
    assert issubclass(PeriEventResult, ResultMixin)
    assert issubclass(PopulationPeriEventResult, ResultMixin)


def test_frozen_dataclass_construction_and_cached_fields():
    """Inheriting ResultMixin must not break the cached __post_init__ fields."""
    single = _single()
    np.testing.assert_allclose(single.firing_rate, single.histogram / single.bin_size)

    pop = _population()
    np.testing.assert_allclose(pop.firing_rates, pop.histograms / pop.bin_size)
    np.testing.assert_allclose(pop.mean_firing_rate, pop.mean_histogram / pop.bin_size)
    # unit_ids resolved to defaults.
    assert list(pop.unit_ids) == [0, 1]


def test_single_to_dataframe_one_row_per_time_bin_carries_unit_id():
    single = _single()
    df = single.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == single.bin_centers.shape[0]
    assert "unit_id" in df.columns
    assert (df["unit_id"] == "u0").all()
    assert {"bin_center", "firing_rate"} <= set(df.columns)


def test_single_summary_round_trips():
    single = _single()
    s = single.summary()
    assert s["unit_id"] == "u0"
    assert s["n_events"] == 10
    assert s["peak_rate"] == max(single.firing_rate)
    assert s["peak_latency"] == 0.0


def test_population_to_dataframe_is_dense_and_carries_unit_id():
    pop = _population()
    df = pop.to_dataframe()
    n_units, n_bins = pop.histograms.shape
    assert len(df) == n_units * n_bins
    assert "unit_id" in df.columns
    assert list(df["unit_id"].unique()) == list(pop.unit_ids)
    assert {"bin_center", "firing_rate"} <= set(df.columns)


def test_population_summary_table_one_row_per_unit():
    pop = _population()
    st = pop.summary_table()
    assert len(st) == pop.n_units
    assert st.index.name == "unit_id"
    assert {"peak_rate", "peak_latency", "baseline_rate"} == set(st.columns)


def test_population_summary_flat_dict():
    pop = _population()
    s = pop.summary()
    assert set(s) == {
        "n_units",
        "n_events",
        "mean_peak_rate",
        "population_peak_latency",
    }
    assert s["n_units"] == 2
