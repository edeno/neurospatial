"""Contract tests for PSTH terminal verbs.

``PeriEventResult`` and ``PopulationPeriEventResult`` gained ``ResultMixin`` and
the terminal verbs. These tests assert:

- the frozen-dataclass construction and derived ``firing_rate(s)`` still
  work after inheriting ``ResultMixin``;
- ``to_dataframe()`` is dense and carries ``unit_id``;
- ``summary()`` / ``summary_table()`` exist and round-trip, with identical
  ``summary_table`` columns on the single-unit and population results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurospatial._results import ResultMixin
from neurospatial.events._core import PeriEventResult, PopulationPeriEventResult


def _single() -> PeriEventResult:
    return PeriEventResult(
        histogram=np.array([0.1, 0.4, 0.2]),
        sem=np.array([0.01, 0.02, 0.01]),
        n_events=10,
        window=(-0.75, 0.75),
        bin_size=0.5,
        unit_id="u0",
    )


def _population() -> PopulationPeriEventResult:
    hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
    return PopulationPeriEventResult(
        histograms=hist,
        sem=np.zeros_like(hist),
        mean_sem=np.zeros(3),
        n_events=10,
        window=(-0.75, 0.75),
        bin_size=0.5,
    )


def test_psth_results_inherit_canonical_result_mixin():
    assert issubclass(PeriEventResult, ResultMixin)
    assert issubclass(PopulationPeriEventResult, ResultMixin)


def test_frozen_dataclass_construction_and_derived_rates():
    """Inheriting ResultMixin must not break construction or derived rates."""
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
        "n_events_dropped",
        "mean_peak_rate",
        "population_peak_latency",
    }
    assert s["n_units"] == 2


def test_single_and_population_summary_tables_share_columns_and_attrs():
    single, pop = _single(), _population()
    single_table, pop_table = single.summary_table(), pop.summary_table()

    assert list(single_table.columns) == list(pop_table.columns)
    assert single_table.index.tolist() == ["u0"]
    assert single_table.index.name == "unit_id"
    for table in (single_table, pop_table, single.to_dataframe(), pop.to_dataframe()):
        assert table.attrs == {
            "window": (-0.75, 0.75),
            "bin_size": 0.5,
            "n_events": 10,
            "n_events_dropped": 0,
        }
    np.testing.assert_allclose(
        pop[1].summary_table().to_numpy(), pop_table.iloc[[1]].to_numpy()
    )


def test_rate_sem_column_is_in_hz():
    df = _single().to_dataframe()
    np.testing.assert_allclose(df["firing_rate_sem"], df["sem"] / 0.5)


def test_baseline_excludes_the_bin_that_straddles_the_event():
    """Bins [-0.35, -0.15), [-0.15, 0.05), [0.05, 0.25): only the first is baseline.

    The middle bin's center is negative but it holds 50 ms after the event.
    """
    result = PeriEventResult(
        histogram=np.array([0.2, 3.0, 1.0]),
        sem=np.zeros(3),
        n_events=4,
        window=(-0.35, 0.35),
        bin_size=0.2,
    )
    np.testing.assert_allclose(result.bin_centers, [-0.25, -0.05, 0.15])
    assert result.summary()["baseline_rate"] == 1.0
    assert result.summary_table()["baseline_rate"].iloc[0] == 1.0


def test_baseline_keeps_a_bin_that_ends_exactly_at_the_event():
    """-0.3 + 3 * 0.1 is 5.6e-17, not 0; that bin still ends at the event."""
    result = PeriEventResult(
        histogram=np.array([0.1, 0.2, 0.3, 0.9, 0.9, 0.9]),
        sem=np.zeros(6),
        n_events=4,
        window=(-0.3, 0.3),
        bin_size=0.1,
    )
    np.testing.assert_allclose(result.summary()["baseline_rate"], 2.0)


def test_baseline_is_nan_without_pre_event_bins():
    result = PeriEventResult(
        histogram=np.ones(2),
        sem=np.zeros(2),
        n_events=3,
        window=(0.0, 1.0),
        bin_size=0.5,
    )
    assert np.isnan(result.summary()["baseline_rate"])
