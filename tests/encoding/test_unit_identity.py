"""Tests for durable unit identity on encoding population/single results.

Covers the v0.6 "Unit identity" api-contract rule:

- batch compute functions accept ``unit_ids=`` (keyword-only, optional) and
  thread it onto the returned population result's ``unit_ids`` field;
- omitting ``unit_ids`` defaults to ``np.arange(n_units)``;
- indexing/iterating a batch result stamps the per-unit ``unit_id`` onto the
  child single-unit result, in order;
- a length mismatch raises a clear ``ValueError``;
- string labels round-trip;
- the new array fields are ``compare=False`` so they do not alter the
  classes' pre-existing equality/hash behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._results import resolve_unit_ids
from neurospatial.encoding.directional import (
    DirectionalRatesResult,
    compute_directional_rates,
)
from neurospatial.encoding.egocentric import compute_egocentric_rates
from neurospatial.encoding.spatial import SpatialRatesResult, compute_spatial_rates
from neurospatial.encoding.view import compute_view_rates


@pytest.fixture
def trajectory() -> tuple[Environment, np.ndarray, np.ndarray, np.ndarray]:
    """Return (env, times, positions, headings) for a small seeded session."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    headings = rng.uniform(-np.pi, np.pi, 500)
    return env, times, positions, headings


@pytest.fixture
def spike_trains() -> list[np.ndarray]:
    """Return spike trains for three units."""
    rng = np.random.default_rng(1)
    return [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]


STR_IDS = ["ca1_01", "ca1_02", "ca1_03"]


def _compute(name, env, times, positions, headings, trains, **kwargs):
    """Dispatch to a batch compute function by short name."""
    if name == "spatial":
        return compute_spatial_rates(env, trains, times, positions, **kwargs)
    if name == "directional":
        return compute_directional_rates(trains, times, headings, **kwargs)
    if name == "view":
        return compute_view_rates(
            env, trains, times, positions, headings, view_distance=10.0, **kwargs
        )
    if name == "egocentric":
        obj = np.array([[25.0, 25.0]])
        return compute_egocentric_rates(
            None, trains, times, positions, headings, obj, **kwargs
        )
    raise ValueError(name)


ALL = ["spatial", "directional", "view", "egocentric"]


@pytest.mark.parametrize("name", ALL)
def test_unit_ids_round_trip(name, trajectory, spike_trains):
    """Provided integer unit_ids appear verbatim on the result."""
    env, times, positions, headings = trajectory
    ids = np.array([10, 20, 30])
    result = _compute(name, env, times, positions, headings, spike_trains, unit_ids=ids)
    np.testing.assert_array_equal(result.unit_ids, ids)


@pytest.mark.parametrize("name", ALL)
def test_unit_ids_default_arange(name, trajectory, spike_trains):
    """Omitting unit_ids defaults to np.arange(n_units)."""
    env, times, positions, headings = trajectory
    result = _compute(name, env, times, positions, headings, spike_trains)
    np.testing.assert_array_equal(result.unit_ids, np.arange(len(spike_trains)))


@pytest.mark.parametrize("name", ALL)
def test_indexing_carries_label(name, trajectory, spike_trains):
    """rates[i].unit_id == rates.unit_ids[i] for every i."""
    env, times, positions, headings = trajectory
    result = _compute(
        name, env, times, positions, headings, spike_trains, unit_ids=STR_IDS
    )
    for i in range(len(spike_trains)):
        assert result[i].unit_id == result.unit_ids[i]


@pytest.mark.parametrize("name", ALL)
def test_iteration_preserves_order_and_labels(name, trajectory, spike_trains):
    """Iterating yields children whose unit_id matches unit_ids, in order."""
    env, times, positions, headings = trajectory
    result = _compute(
        name, env, times, positions, headings, spike_trains, unit_ids=STR_IDS
    )
    assert [child.unit_id for child in result] == list(result.unit_ids)


@pytest.mark.parametrize("name", ALL)
def test_length_mismatch_raises(name, trajectory, spike_trains):
    """Wrong-length unit_ids raises a clear ValueError naming the mismatch."""
    env, times, positions, headings = trajectory
    with pytest.raises(ValueError, match="unit_ids length mismatch"):
        _compute(
            name,
            env,
            times,
            positions,
            headings,
            spike_trains,
            unit_ids=["only_one"],
        )


def test_resolve_unit_ids_rejects_2d():
    """A 2-D unit_ids array raises a clear ValueError naming the shape."""
    bad = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match=r"unit_ids must be 1-D.*\(2, 2\)"):
        resolve_unit_ids(bad, 2)


@pytest.mark.parametrize("name", ALL)
def test_string_ids_round_trip_and_index(name, trajectory, spike_trains):
    """String unit_ids round-trip and index onto children."""
    env, times, positions, headings = trajectory
    result = _compute(
        name, env, times, positions, headings, spike_trains, unit_ids=STR_IDS
    )
    assert list(result.unit_ids) == STR_IDS
    assert result[1].unit_id == "ca1_02"


@pytest.mark.parametrize("name", ALL)
def test_indexed_int_unit_id_is_python_int(name, trajectory, spike_trains):
    """Integer unit_ids stamp a plain Python ``int`` (not np.int64) onto children."""
    env, times, positions, headings = trajectory
    ids = np.array([10, 20, 30])
    result = _compute(name, env, times, positions, headings, spike_trains, unit_ids=ids)
    for i in range(len(spike_trains)):
        assert type(result[i].unit_id) is int


@pytest.mark.parametrize("name", ALL)
def test_indexed_str_unit_id_is_python_str(name, trajectory, spike_trains):
    """String unit_ids stamp a plain Python ``str`` (not np.str_) onto children."""
    env, times, positions, headings = trajectory
    result = _compute(
        name, env, times, positions, headings, spike_trains, unit_ids=STR_IDS
    )
    for i in range(len(spike_trains)):
        assert type(result[i].unit_id) is str


@pytest.mark.parametrize("cls", [SpatialRatesResult, DirectionalRatesResult])
def test_unit_ids_field_is_compare_false(cls):
    """unit_ids/unit_table must not participate in __eq__ (declared compare=False).

    These batch result classes already compare their (NumPy) ``firing_rates``
    elementwise, so ``==`` was never a plain bool before this change. The
    contract here is narrower: adding ``unit_ids``/``unit_table`` must not
    introduce a *new* array field into ``__eq__``. We assert that by
    confirming the dataclass field metadata marks both new fields as
    ``compare=False``.
    """
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(cls)}
    assert fields["unit_ids"].compare is False
    assert fields["unit_table"].compare is False


def test_default_unit_ids_when_constructed_directly(trajectory, spike_trains):
    """Constructing a result without unit_ids defaults to arange via __post_init__."""
    env, times, positions, _ = trajectory
    result = compute_spatial_rates(env, spike_trains, times, positions)
    # Re-build a result object directly (no unit_ids kwarg).
    rebuilt = SpatialRatesResult(
        firing_rates=np.asarray(result.firing_rates),
        occupancy=np.asarray(result.occupancy),
        env=env,
        method=result.method,
        bandwidth=result.bandwidth,
    )
    np.testing.assert_array_equal(rebuilt.unit_ids, np.arange(len(spike_trains)))
    assert isinstance(rebuilt.unit_ids, np.ndarray)


def test_unit_table_wrong_length_raises(trajectory, spike_trains):
    """A unit_table whose row count != n_units raises a clear ValueError."""
    import pandas as pd

    env, times, positions, _ = trajectory
    result = compute_spatial_rates(env, spike_trains, times, positions)
    n_units = len(spike_trains)
    bad_table = pd.DataFrame({"region": ["ca1"] * (n_units + 1)})
    with pytest.raises(ValueError, match="unit_table length mismatch"):
        SpatialRatesResult(
            firing_rates=np.asarray(result.firing_rates),
            occupancy=np.asarray(result.occupancy),
            env=env,
            method=result.method,
            bandwidth=result.bandwidth,
            unit_table=bad_table,
        )


def test_unit_table_correct_length_accepted(trajectory, spike_trains):
    """A unit_table with exactly one row per unit is accepted."""
    import pandas as pd

    env, times, positions, _ = trajectory
    result = compute_spatial_rates(env, spike_trains, times, positions)
    n_units = len(spike_trains)
    good_table = pd.DataFrame({"region": ["ca1"] * n_units})
    rebuilt = SpatialRatesResult(
        firing_rates=np.asarray(result.firing_rates),
        occupancy=np.asarray(result.occupancy),
        env=env,
        method=result.method,
        bandwidth=result.bandwidth,
        unit_table=good_table,
    )
    assert rebuilt.unit_table is good_table


def test_all_nan_unit_summary_table_and_peaks_no_crash(trajectory, spike_trains):
    """A population with one all-NaN unit yields NaN peaks, no crash (I1)."""
    env, times, positions, _ = trajectory
    result = compute_spatial_rates(env, spike_trains, times, positions)
    rates = np.asarray(result.firing_rates).copy()
    rates[0, :] = np.nan  # dead unit
    dead = SpatialRatesResult(
        firing_rates=rates,
        occupancy=np.asarray(result.occupancy),
        env=env,
        method=result.method,
        bandwidth=result.bandwidth,
    )
    peaks = dead.peak_locations()
    assert np.all(np.isnan(peaks[0]))
    assert np.all(np.isfinite(peaks[1:]))

    table = dead.summary_table()
    peak_cols = [c for c in table.columns if c.startswith("peak_")]
    # The dead unit's peak coordinates must be NaN, others finite.
    assert table.iloc[0][[c for c in peak_cols if c != "peak_rate"]].isna().all()
