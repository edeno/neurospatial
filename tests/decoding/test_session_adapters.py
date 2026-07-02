"""Adapter wiring tests for the decode-session entry points.

Verifies, in the default env (no pynapple installed), that ``decode_session``
and ``decode_session_summary`` accept a duck-typed ``PositionLike`` and a
``SpikeTrainsLike`` group — both an iterate-yields-trains double (the future
``SpikeTrains``) and a ``UserDict``-based double (a real pynapple ``TsGroup``,
whose iteration yields KEYS, not trains) — and produce results byte-for-byte
identical to the plain-array path (the scientific core is array-only).
"""

from __future__ import annotations

from collections import UserDict

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_session, decode_session_summary


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
    """Iterate-yields-trains double (models the future ``SpikeTrains``).

    Iterating yields the per-unit trains directly; carries ``.index`` ids. This
    is NOT how a real pynapple ``TsGroup`` behaves (see ``_FakeTsGroupMapping``).
    """

    def __init__(self, trains: list[np.ndarray], index: np.ndarray) -> None:
        self._trains = trains
        self.index = index

    def __iter__(self):
        return iter(self._trains)


class _FakeTs:
    """Minimal pynapple ``Ts`` stand-in: exposes ``.t`` (spike timestamps)."""

    def __init__(self, t: np.ndarray) -> None:
        self.t = t


class _FakeTsGroupMapping(UserDict):
    """``UserDict``-based ``TsGroup`` double: iterating yields KEYS, not trains.

    Models a real pynapple ``TsGroup`` (subclasses ``collections.UserDict``):
    iterating yields the unit-id keys, ``group[uid]`` returns a ``Ts``-like
    object with ``.t``, and ``.index`` returns the keys.
    """

    def __init__(self, trains: list[np.ndarray], index: np.ndarray) -> None:
        super().__init__(
            {int(uid): _FakeTs(t) for uid, t in zip(index, trains, strict=True)}
        )
        self._index = np.asarray(index)

    @property
    def index(self) -> np.ndarray:
        return self._index


@pytest.fixture
def session():
    rng = np.random.default_rng(2)
    positions = np.column_stack([np.linspace(0.0, 100.0, 400), rng.uniform(0, 40, 400)])
    env = Environment.from_samples(positions, bin_size=10.0)
    times = np.linspace(0.0, 40.0, 400)
    spikes = [
        np.sort(rng.uniform(0.0, 40.0, 60)),
        np.sort(rng.uniform(0.0, 40.0, 45)),
        np.sort(rng.uniform(0.0, 40.0, 50)),
    ]
    return env, times, positions, spikes


def test_decode_session_positionlike_matches_arrays(session) -> None:
    env, times, positions, spikes = session

    from_arrays = decode_session(env, spikes, times, positions, dt=0.5)
    from_poslike = decode_session(env, spikes, _FakeTsdFrame(times, positions), dt=0.5)

    np.testing.assert_array_equal(from_arrays.posterior, from_poslike.posterior)


def test_decode_session_group_spikes_matches_arrays(session) -> None:
    env, times, positions, spikes = session
    group = _FakeTsGroup(spikes, index=np.array([11, 22, 33]))

    from_arrays = decode_session(env, spikes, times, positions, dt=0.5)
    from_group = decode_session(env, group, times, positions, dt=0.5)

    np.testing.assert_array_equal(from_arrays.posterior, from_group.posterior)


def test_decode_session_mapping_group_matches_arrays(session) -> None:
    """A ``UserDict``-based ``TsGroup`` double (iteration yields keys) must
    yield a posterior byte-for-byte identical to the plain-array path. Catches
    C1 in the default env, with no pynapple installed."""
    env, times, positions, spikes = session
    group = _FakeTsGroupMapping(spikes, index=np.array([11, 22, 33]))

    from_arrays = decode_session(env, spikes, times, positions, dt=0.5)
    from_group = decode_session(env, group, times, positions, dt=0.5)

    np.testing.assert_array_equal(from_arrays.posterior, from_group.posterior)


def test_decode_session_summary_positionlike_matches_arrays(session) -> None:
    env, times, positions, spikes = session

    from_arrays = decode_session_summary(env, spikes, times, positions, dt=0.5)
    from_poslike = decode_session_summary(
        env, spikes, _FakeTsdFrame(times, positions), dt=0.5
    )

    np.testing.assert_array_equal(from_arrays.map_position, from_poslike.map_position)
    np.testing.assert_array_equal(
        from_arrays.posterior_entropy, from_poslike.posterior_entropy
    )


def test_decode_session_missing_positions_raises(session) -> None:
    env, times, _positions, spikes = session
    with pytest.raises(ValueError, match="no `positions`"):
        decode_session(env, spikes, times, dt=0.5)
