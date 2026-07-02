"""Adapter wiring tests for the spatial encoding entry points.

Verifies, in the default env (no pynapple installed):

- ``as_spike_trains_with_ids`` normalizes arrays (ids ``None``) and a duck-typed
  ``TsGroup``-like (surfacing ids).
- ``compute_spatial_rate`` / ``compute_spatial_rates`` accept a duck-typed
  ``PositionLike`` and produce results byte-for-byte identical to the array path.
- Unit ids from a ``TsGroup``-like flow into ``SpatialRatesResult.unit_ids``,
  while the firing rates stay identical to the plain-array input.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import (
    as_spike_trains,
    as_spike_trains_with_ids,
    compute_spatial_rate,
    compute_spatial_rates,
)


class _FakeTsdFrame:
    """Duck-typed ``PositionLike`` (pynapple ``TsdFrame`` stand-in)."""

    def __init__(self, t: np.ndarray, values: np.ndarray) -> None:
        self._t = t
        self._values = values

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def values(self) -> np.ndarray:
        return self._values


class _FakeTsGroup:
    """Duck-typed ``SpikeTrainsLike``: iterates trains, carries ``.index`` ids."""

    def __init__(self, trains: list[np.ndarray], index: np.ndarray) -> None:
        self._trains = trains
        self.index = index

    def __iter__(self):
        return iter(self._trains)


@pytest.fixture
def session():
    rng = np.random.default_rng(1)
    positions = np.column_stack([np.linspace(0.0, 100.0, 400), rng.uniform(0, 40, 400)])
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0.0, 40.0, 400)
    spikes = [
        np.sort(rng.uniform(0.0, 40.0, 50)),
        np.sort(rng.uniform(0.0, 40.0, 30)),
        np.sort(rng.uniform(0.0, 40.0, 40)),
    ]
    return env, times, positions, spikes


# ---------------------------------------------------------------------------
# as_spike_trains_with_ids
# ---------------------------------------------------------------------------


def test_as_spike_trains_with_ids_array_returns_none_ids() -> None:
    spikes = [np.array([0.1, 0.5]), np.array([0.2, 0.3, 0.8])]
    trains, ids = as_spike_trains_with_ids(spikes)

    assert ids is None
    canonical = as_spike_trains(spikes)
    assert len(trains) == len(canonical)
    for a, b in zip(trains, canonical, strict=True):
        np.testing.assert_array_equal(a, b)


def test_as_spike_trains_with_ids_2d_array_returns_none_ids() -> None:
    spikes = np.array([[0.1, 0.5, np.nan], [0.2, 0.3, 0.8]])
    trains, ids = as_spike_trains_with_ids(spikes)

    assert ids is None
    np.testing.assert_array_equal(trains[0], np.array([0.1, 0.5]))
    np.testing.assert_array_equal(trains[1], np.array([0.2, 0.3, 0.8]))


def test_as_spike_trains_with_ids_group_surfaces_ids() -> None:
    trains_in = [np.array([0.1, 0.5]), np.array([0.2])]
    group = _FakeTsGroup(trains_in, index=np.array([7, 9]))

    trains, ids = as_spike_trains_with_ids(group)

    np.testing.assert_array_equal(ids, np.array([7, 9]))
    np.testing.assert_array_equal(trains[0], trains_in[0])
    np.testing.assert_array_equal(trains[1], trains_in[1])


# ---------------------------------------------------------------------------
# compute_spatial_rate(s): PositionLike + array-path parity
# ---------------------------------------------------------------------------


def test_compute_spatial_rate_positionlike_matches_arrays(session) -> None:
    env, times, positions, spikes = session

    from_arrays = compute_spatial_rate(env, spikes[0], times, positions, bandwidth=5.0)
    from_poslike = compute_spatial_rate(
        env, spikes[0], _FakeTsdFrame(times, positions), bandwidth=5.0
    )

    np.testing.assert_array_equal(from_arrays.firing_rate, from_poslike.firing_rate)


def test_compute_spatial_rate_missing_positions_raises(session) -> None:
    env, times, _positions, spikes = session
    with pytest.raises(ValueError, match="no `positions`"):
        compute_spatial_rate(env, spikes[0], times)


def test_compute_spatial_rates_positionlike_matches_arrays(session) -> None:
    env, times, positions, spikes = session

    from_arrays = compute_spatial_rates(env, spikes, times, positions, bandwidth=5.0)
    from_poslike = compute_spatial_rates(
        env, spikes, _FakeTsdFrame(times, positions), bandwidth=5.0
    )

    np.testing.assert_array_equal(from_arrays.firing_rates, from_poslike.firing_rates)
    np.testing.assert_array_equal(from_arrays.unit_ids, from_poslike.unit_ids)


# ---------------------------------------------------------------------------
# compute_spatial_rates: TsGroup-like unit id threading
# ---------------------------------------------------------------------------


def test_compute_spatial_rates_group_ids_flow_into_result(session) -> None:
    env, times, positions, spikes = session
    group = _FakeTsGroup(spikes, index=np.array([11, 22, 33]))

    from_arrays = compute_spatial_rates(env, spikes, times, positions, bandwidth=5.0)
    from_group = compute_spatial_rates(env, group, times, positions, bandwidth=5.0)

    # unit ids surfaced from the group; rates identical to the plain-array path.
    np.testing.assert_array_equal(from_group.unit_ids, np.array([11, 22, 33]))
    np.testing.assert_array_equal(from_arrays.firing_rates, from_group.firing_rates)


def test_compute_spatial_rates_explicit_unit_ids_win_over_group(session) -> None:
    env, times, positions, spikes = session
    group = _FakeTsGroup(spikes, index=np.array([11, 22, 33]))

    result = compute_spatial_rates(
        env, group, times, positions, bandwidth=5.0, unit_ids=np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(result.unit_ids, np.array([1, 2, 3]))


def test_compute_spatial_rates_array_default_ids_unchanged(session) -> None:
    env, times, positions, spikes = session
    result = compute_spatial_rates(env, spikes, times, positions, bandwidth=5.0)
    np.testing.assert_array_equal(result.unit_ids, np.arange(len(spikes)))
