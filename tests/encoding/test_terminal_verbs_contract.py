"""Contract tests for the split terminal verbs (Task 1.3).

The v0.6 api-contract requires that the analysis-ending verbs mean ONE thing on
every result class:

- ``to_dataframe()`` -> dense tidy: one row per ``(unit, bin)`` (single-unit:
  one row per ``bin``), ALWAYS carrying a ``unit_id`` column.
- ``summary_table()`` -> one row per unit, ``unit_id``-indexed, scalar columns.

These tests assert the contract holds, and that the column vocabulary is
consistent across the four batch encoding result classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.directional import DirectionalRatesResult
from neurospatial.encoding.egocentric import compute_egocentric_rates
from neurospatial.encoding.spatial import compute_spatial_rate, compute_spatial_rates
from neurospatial.encoding.view import compute_view_rates


@pytest.fixture
def spatial_batch():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    return compute_spatial_rates(env, spike_times, times, positions, bandwidth=10.0)


@pytest.fixture
def directional_batch():
    rng = np.random.default_rng(0)
    n = 60
    bin_centers = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return DirectionalRatesResult(
        firing_rates=rng.random((3, n)) * 10,
        occupancy=np.ones(n) * 0.5,
        bin_centers=bin_centers,
        bin_size=np.pi / 30,
        bandwidth=None,
    )


@pytest.fixture
def view_batch():
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.random((200, 2)) * 50, bin_size=5.0)
    times = np.linspace(0, 10, 200)
    trajectory = rng.random((200, 2)) * 50
    headings = rng.uniform(-np.pi, np.pi, 200)
    spike_times = [np.sort(rng.uniform(0, 10, 15)) for _ in range(3)]
    return compute_view_rates(
        env, spike_times, times, trajectory, headings, view_distance=10.0
    )


@pytest.fixture
def egocentric_batch():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 100, 1000)
    positions = rng.uniform(10, 90, (1000, 2))
    headings = rng.uniform(-np.pi, np.pi, 1000)
    object_positions = np.array([[50.0, 50.0]])
    spike_times = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_egocentric_rates(
        None, spike_times, times, positions, headings, object_positions
    )


BATCH_FIXTURES = [
    "spatial_batch",
    "directional_batch",
    "view_batch",
    "egocentric_batch",
]


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_to_dataframe_is_dense_and_carries_unit_id(fixture_name, request):
    """to_dataframe() is one row per (unit, bin), carrying unit_id."""
    result = request.getfixturevalue(fixture_name)
    df = result.to_dataframe()

    n_units = len(result)
    n_bins = np.asarray(result.firing_rates).shape[1]
    assert len(df) == n_units * n_bins

    # Always carries unit_id and bin columns plus firing_rate/occupancy.
    assert {"unit_id", "bin", "firing_rate", "occupancy"} <= set(df.columns)
    # The unit_id column carries the real per-unit labels.
    assert list(df["unit_id"].unique()) == list(result.unit_ids)
    # Each unit contributes exactly n_bins rows.
    assert (df.groupby("unit_id").size() == n_bins).all()


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_summary_table_is_one_row_per_unit(fixture_name, request):
    """summary_table() is one row per unit, unit_id-indexed."""
    result = request.getfixturevalue(fixture_name)
    summary = result.summary_table()

    assert len(summary) == len(result)
    assert summary.index.name == "unit_id"
    assert list(summary.index) == list(result.unit_ids)
    # peak_rate is the shared scalar metric on every summary table.
    assert "peak_rate" in summary.columns


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_summary_table_custom_unit_ids(fixture_name, request):
    """summary_table(unit_ids=...) re-labels the index."""
    result = request.getfixturevalue(fixture_name)
    custom = [f"u{i}" for i in range(len(result))]
    summary = result.summary_table(unit_ids=custom)
    assert list(summary.index) == custom


def test_to_dataframe_bin_center_vocabulary_is_consistent(
    spatial_batch, directional_batch, view_batch, egocentric_batch
):
    """Bin-center columns use the shared vocabulary per coordinate space."""
    # Cartesian results -> bin_center_x / bin_center_y.
    for result in (spatial_batch, view_batch):
        cols = set(result.to_dataframe().columns)
        assert {"bin_center_x", "bin_center_y"} <= cols

    # Directional result (no env, angular) -> bin_center_angle.
    assert "bin_center_angle" in set(directional_batch.to_dataframe().columns)

    # Egocentric polar result -> bin_center_distance / bin_center_angle.
    ego_cols = set(egocentric_batch.to_dataframe().columns)
    assert {"bin_center_distance", "bin_center_angle"} <= ego_cols


def test_single_unit_to_dataframe_carries_unit_id():
    """Single-unit to_dataframe() is one row per bin, carrying unit_id."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    spike_times = np.sort(rng.uniform(0, 50, 30))
    result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)

    df = result.to_dataframe()
    assert len(df) == env.n_bins
    assert {"unit_id", "bin", "firing_rate"} <= set(df.columns)
    # Standalone single-unit computation has no identity -> the unit_id column
    # is a single absent value (pd.NA), not a fabricated 0. Every row shares
    # that same absent identity.
    assert df["unit_id"].isna().all()
    assert df["unit_id"].nunique(dropna=False) == 1


def test_single_unit_to_dataframe_uses_unit_id_when_set(spatial_batch):
    """Indexing a batch carries the unit's real label into the dense frame."""
    single = spatial_batch[1]
    df = single.to_dataframe()
    assert (df["unit_id"] == spatial_batch.unit_ids[1]).all()
