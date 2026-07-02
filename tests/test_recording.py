"""Tests for the frozen ``Session`` bundle and ``load_session`` dispatch.

``Session`` (``neurospatial.recording``) is a **discoverability bundle**, NOT a
god-object: it holds ``env`` / ``position`` / ``spikes`` / ``epochs`` /
``metadata`` and exposes only the raw arrays; heavy compute stays as functions
taking the bundle's fields (verified by the ``compute_spatial_rates`` compose
test). It is frozen: ``with_environment`` / ``restrict`` return NEW sessions and
never mutate the original.

Tiers:

- **Array path** (always run): ``from_arrays``, immutability, ``restrict``
  (identity-preserving spikes), ``__post_init__`` validation, ``load_session``
  dispatch, and the import-safety assertion that ``recording`` never pulls in
  pynwb / pynapple.
- **NWB path** (``@pytest.mark.nwb`` + ``skipif`` when pynwb is absent):
  ``from_nwb`` on a small in-memory ``NWBFile`` equals ``from_arrays`` on the
  same underlying arrays.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from collections.abc import Mapping

import numpy as np
import pytest

from neurospatial import Environment, SpikeTrains, load_session
from neurospatial.encoding import compute_spatial_rates
from neurospatial.recording import Position, Session

HAS_PYNWB = importlib.util.find_spec("pynwb") is not None


# ---------------------------------------------------------------------------
# Shared array fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arrays():
    """Small deterministic (times, positions, trains, unit_ids) bundle."""
    rng = np.random.default_rng(0)
    n = 200
    times = np.arange(n) / 20.0  # 20 Hz, spans [0, 9.95] s
    positions = np.clip(np.cumsum(rng.normal(0, 3, (n, 2)), axis=0) + 50.0, 0, 100)
    trains = [
        np.sort(rng.uniform(0.0, times[-1], 40)),
        np.sort(rng.uniform(0.0, times[-1], 25)),
    ]
    unit_ids = np.array([10, 20])
    return times, positions, trains, unit_ids


@pytest.fixture
def env(arrays):
    """A small fitted environment over the fixture positions (fast KDE)."""
    _, positions, _, _ = arrays
    return Environment.from_samples(positions, bin_size=10.0)


# A duck-typed pynapple-``TsGroup``-like fake: a Mapping exposing a non-callable
# ``.index`` (unit ids) whose values carry per-unit ``.t`` timestamps. This
# exercises the same ``as_spike_trains_with_ids`` Mapping branch a real TsGroup
# (a UserDict) hits, without importing pynapple.
class _FakeTs:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=np.float64)


class _FakeTsGroup(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    @property
    def index(self):
        return np.asarray(list(self._data.keys()))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


# ---------------------------------------------------------------------------
# 1. from_arrays
# ---------------------------------------------------------------------------


def test_from_arrays_builds_session(arrays, env):
    times, positions, trains, unit_ids = arrays
    sess = Session.from_arrays(
        env=env,
        times=times,
        positions=positions,
        spike_times=trains,
        unit_ids=unit_ids,
    )

    assert sess.env is env
    np.testing.assert_array_equal(sess.times, times)
    np.testing.assert_array_equal(sess.positions, positions)
    # position holder exposes .t / .values uniformly
    np.testing.assert_array_equal(sess.position.t, times)
    np.testing.assert_array_equal(sess.position.values, positions)
    assert isinstance(sess.position, Position)

    assert isinstance(sess.spikes, SpikeTrains)
    np.testing.assert_array_equal(sess.spikes.unit_ids, unit_ids)
    assert len(sess.spikes) == 2
    np.testing.assert_array_equal(sess.spikes[10], trains[0])


def test_from_arrays_accepts_spiketrains_and_preserves_ids(arrays, env):
    times, positions, trains, unit_ids = arrays
    st = SpikeTrains(trains, unit_ids=unit_ids)
    sess = Session.from_arrays(
        env=env, times=times, positions=positions, spike_times=st
    )
    assert isinstance(sess.spikes, SpikeTrains)
    np.testing.assert_array_equal(sess.spikes.unit_ids, unit_ids)


def test_from_arrays_accepts_tsgroup_like_and_surfaces_ids(arrays, env):
    times, positions, trains, _ = arrays
    group = _FakeTsGroup({7: _FakeTs(trains[0]), 9: _FakeTs(trains[1])})
    sess = Session.from_arrays(
        env=env, times=times, positions=positions, spike_times=group
    )
    assert isinstance(sess.spikes, SpikeTrains)
    np.testing.assert_array_equal(sess.spikes.unit_ids, np.array([7, 9]))
    np.testing.assert_array_equal(sess.spikes[7], trains[0])


def test_from_arrays_env_optional(arrays):
    times, positions, trains, _ = arrays
    sess = Session.from_arrays(times=times, positions=positions, spike_times=trains)
    assert sess.env is None
    assert isinstance(sess.spikes, SpikeTrains)


# ---------------------------------------------------------------------------
# 2. Immutability
# ---------------------------------------------------------------------------


def test_with_environment_returns_new_and_leaves_original(arrays, env):
    times, positions, trains, unit_ids = arrays
    sess = Session.from_arrays(
        times=times, positions=positions, spike_times=trains, unit_ids=unit_ids
    )
    assert sess.env is None

    new_sess = sess.with_environment(env)
    assert new_sess is not sess
    assert new_sess.env is env
    # Original unchanged.
    assert sess.env is None
    # Spikes / position ride along.
    np.testing.assert_array_equal(new_sess.spikes.unit_ids, unit_ids)
    np.testing.assert_array_equal(new_sess.times, times)


def test_session_is_frozen(arrays, env):
    times, positions, trains, _ = arrays
    sess = Session.from_arrays(
        env=env, times=times, positions=positions, spike_times=trains
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        sess.env = None  # type: ignore[misc]


def test_with_environment_rejects_non_environment(arrays, env):
    times, positions, trains, _ = arrays
    sess = Session.from_arrays(
        env=env, times=times, positions=positions, spike_times=trains
    )
    with pytest.raises((ValueError, TypeError)):
        sess.with_environment(object())


# ---------------------------------------------------------------------------
# 3. restrict
# ---------------------------------------------------------------------------


def test_restrict_slices_position_and_spikes_preserving_identity(arrays, env):
    times, positions, trains, unit_ids = arrays
    unit_table_cols = {"region": ["CA1", "CA3"]}
    import pandas as pd

    sess = Session.from_arrays(
        env=env,
        times=times,
        positions=positions,
        spike_times=trains,
        unit_ids=unit_ids,
        unit_table=pd.DataFrame(unit_table_cols),
    )

    t0, t1 = 2.0, 6.0
    sess_r = sess.restrict((t0, t1))
    assert sess_r is not sess

    # Position sliced to the epoch.
    mask = (times >= t0) & (times <= t1)
    np.testing.assert_array_equal(sess_r.times, times[mask])
    np.testing.assert_array_equal(sess_r.positions, positions[mask])

    # Spikes sliced per-train by their own timestamps.
    for uid, tr in zip(unit_ids, trains, strict=True):
        expected = tr[(tr >= t0) & (tr <= t1)]
        np.testing.assert_array_equal(sess_r.spikes[uid], expected)

    # Identity preserved: ids AND unit_table ride along unchanged.
    assert isinstance(sess_r.spikes, SpikeTrains)
    np.testing.assert_array_equal(sess_r.spikes.unit_ids, unit_ids)
    assert sess_r.spikes.unit_table is not None
    assert list(sess_r.spikes.unit_table["region"]) == ["CA1", "CA3"]

    # The new session records the restriction.
    assert sess_r.epochs == (t0, t1)

    # Original unchanged.
    np.testing.assert_array_equal(sess.times, times)
    assert sess.epochs is None
    assert len(sess.spikes[unit_ids[0]]) == len(trains[0])


def test_restrict_composes_with_compute_spatial_rates(arrays, env):
    times, positions, trains, unit_ids = arrays
    sess = Session.from_arrays(
        env=env,
        times=times,
        positions=positions,
        spike_times=trains,
        unit_ids=unit_ids,
    )
    epochs = (2.0, 6.0)
    sess_r = sess.restrict(epochs)

    result = compute_spatial_rates(
        sess_r.env, sess_r.spikes, sess_r.times, sess_r.positions
    )

    # Hand-masked reference on the restricted data only.
    mask = (times >= epochs[0]) & (times <= epochs[1])
    ref_trains = [tr[(tr >= epochs[0]) & (tr <= epochs[1])] for tr in trains]
    ref = compute_spatial_rates(
        env, ref_trains, times[mask], positions[mask], unit_ids=unit_ids
    )

    np.testing.assert_allclose(
        np.asarray(result.firing_rates), np.asarray(ref.firing_rates)
    )
    np.testing.assert_array_equal(result.unit_ids, ref.unit_ids)


# ---------------------------------------------------------------------------
# 4. __post_init__ validation
# ---------------------------------------------------------------------------


def test_mismatched_position_lengths_raise(arrays, env):
    times, positions, trains, _ = arrays
    bad_position = Position(t=times[:-1], values=positions)  # length mismatch
    with pytest.raises(ValueError, match=r"(?i)length|len|position"):
        Session(env=env, position=bad_position, spikes=SpikeTrains(trains))


def test_non_environment_env_raises(arrays):
    times, positions, trains, _ = arrays
    with pytest.raises((ValueError, TypeError)):
        Session(
            env=object(),  # not EnvironmentLike
            position=Position(t=times, values=positions),
            spikes=SpikeTrains(trains),
        )


def test_session_coerces_plain_list_spikes(arrays, env):
    times, positions, trains, _ = arrays
    # Constructing Session directly with a plain list coerces to SpikeTrains.
    sess = Session(env=env, position=Position(t=times, values=positions), spikes=trains)
    assert isinstance(sess.spikes, SpikeTrains)
    assert len(sess.spikes) == 2


# ---------------------------------------------------------------------------
# 5. load_session dispatch
# ---------------------------------------------------------------------------


def test_load_session_rejects_non_path_source():
    with pytest.raises(TypeError, match=r"from_arrays"):
        load_session(np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# 6. Import safety: recording never imports pynwb / pynapple
# ---------------------------------------------------------------------------


def test_recording_import_does_not_load_pynwb_or_pynapple():
    code = (
        "import sys\n"
        "import neurospatial\n"
        "import neurospatial.recording\n"
        "assert 'pynwb' not in sys.modules, 'pynwb imported at import'\n"
        "assert 'pynapple' not in sys.modules, 'pynapple imported at import'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 7. from_nwb (NWB-gated)
# ---------------------------------------------------------------------------


def _make_nwbfile(with_env: bool):
    """Build a small in-memory NWBFile: units + position (+ optional env)."""
    from datetime import datetime, timezone

    from pynwb import NWBFile
    from pynwb.behavior import Position as NWBPosition
    from pynwb.behavior import SpatialSeries

    nwbfile = NWBFile(
        session_description="recording test",
        identifier="recording-test",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    trains = [np.array([0.1, 0.5, 1.2, 2.4]), np.array([0.3, 0.9, 3.1])]
    unit_ids = [7, 11]
    for tr, uid in zip(trains, unit_ids, strict=True):
        nwbfile.add_unit(spike_times=tr, id=uid)

    n = 120
    timestamps = np.arange(n) / 30.0
    rng = np.random.default_rng(1)
    positions = np.clip(np.cumsum(rng.normal(0, 2, (n, 2)), axis=0) + 50.0, 0, 100)
    pos_container = NWBPosition(name="Position")
    pos_container.add_spatial_series(
        SpatialSeries(
            name="position",
            description="animal position",
            data=positions,
            timestamps=timestamps,
            reference_frame="corner",
            unit="cm",
        )
    )
    module = nwbfile.create_processing_module(name="behavior", description="behavior")
    module.add(pos_container)

    env = None
    if with_env:
        from neurospatial.io.nwb import write_environment

        env = Environment.from_samples(positions, bin_size=10.0)
        write_environment(nwbfile, env)

    return nwbfile, timestamps, positions, trains, np.asarray(unit_ids)


@pytest.mark.nwb
@pytest.mark.skipif(not HAS_PYNWB, reason="pynwb not installed")
def test_from_nwb_builds_session_with_env():
    nwbfile, timestamps, positions, trains, unit_ids = _make_nwbfile(with_env=True)

    sess = Session.from_nwb(nwbfile)

    # Environment present.
    assert sess.env is not None
    assert sess.env.n_bins > 0

    # Position populated and matches the written arrays.
    np.testing.assert_allclose(sess.times, timestamps)
    np.testing.assert_allclose(sess.positions, positions)

    # Spikes carry the NWB unit ids.
    assert isinstance(sess.spikes, SpikeTrains)
    np.testing.assert_array_equal(sess.spikes.unit_ids, unit_ids)
    for uid, tr in zip(unit_ids, trains, strict=True):
        np.testing.assert_allclose(sess.spikes[uid], np.sort(tr))

    # Equivalent to from_arrays on the same underlying data.
    ref = Session.from_arrays(
        env=sess.env,
        times=timestamps,
        positions=positions,
        spike_times=trains,
        unit_ids=unit_ids,
    )
    np.testing.assert_allclose(sess.times, ref.times)
    np.testing.assert_allclose(sess.positions, ref.positions)
    np.testing.assert_array_equal(sess.spikes.unit_ids, ref.spikes.unit_ids)


@pytest.mark.nwb
@pytest.mark.skipif(not HAS_PYNWB, reason="pynwb not installed")
def test_from_nwb_without_env_is_none():
    nwbfile, _, _, _, unit_ids = _make_nwbfile(with_env=False)
    sess = Session.from_nwb(nwbfile)
    assert sess.env is None
    np.testing.assert_array_equal(sess.spikes.unit_ids, unit_ids)


@pytest.mark.nwb
@pytest.mark.skipif(not HAS_PYNWB, reason="pynwb not installed")
def test_load_session_dispatches_open_nwbfile():
    nwbfile, timestamps, _positions, _, unit_ids = _make_nwbfile(with_env=False)
    sess = load_session(nwbfile)
    assert isinstance(sess, Session)
    np.testing.assert_array_equal(sess.spikes.unit_ids, unit_ids)
    np.testing.assert_allclose(sess.times, timestamps)
