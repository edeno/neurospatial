"""Shuffles that cannot test overlapping windows refuse them.

Overlapping windows share spikes, so neighboring rows of an event are
correlated even when the spikes carry no sequence. These nulls lose that
correlation, so the observed event looks smoother than its shuffles and a 5%
test rejects far too often. The two nulls that re-decode the same rows keep it.
"""

import dataclasses

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    compare_replay_methods,
    decode_intervals,
    replay_significance,
)
from neurospatial.replay.methods import WU_2017
from neurospatial.replay.trajectory import fit_linear_trajectory

REFUSED = (
    "column_cycle",
    "column_cycle_weighted",
    "time_bin_permutation",
    "pseudo_event",
    "spike_train_circular",
)
KEPT = ("unit_identity", "place_field_circular")
N_CELLS = 10


@pytest.fixture(scope="module")
def track():
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, N_CELLS)[:, None]) / 8.0) ** 2
    )
    return env, rates


@pytest.fixture(scope="module")
def spikes():
    rng = np.random.default_rng(0)
    return [np.sort(rng.uniform(0.0, 3.0, 60)) for _ in range(N_CELLS)]


WINDOWS = np.array([[0.5, 0.7], [1.5, 1.7]])


def _method(null):
    return dataclasses.replace(
        WU_2017, name=null, nulls=(null,), n_shuffles=5, empty_time_bins="keep"
    )


def _significance(env, rates, event, counts, null, dt):
    return replay_significance(
        env,
        event.posterior,
        event.times,
        score=fit_linear_trajectory,
        nulls=(null,),
        n_shuffles=5,
        spike_counts=counts,
        encoding_models=rates,
        dt=dt,
        pseudo_event_pool=event.posterior,
        rng=0,
    )


@pytest.mark.parametrize("null", REFUSED)
def test_refused_on_overlapping_windows(track, spikes, null):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    with pytest.raises(ValueError, match=f"cannot test overlapping.*{null}"):
        _significance(env, rates, events[0], counts[0], null, dt=0.02)


@pytest.mark.parametrize("null", REFUSED)
def test_runs_on_tiled_windows(track, spikes, null):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02)
    result = _significance(env, rates, events[0], counts[0], null, dt=0.02)
    assert np.isfinite(result.p_value)


@pytest.mark.parametrize("null", KEPT)
def test_nulls_that_keep_the_rows_run_on_overlapping_windows(track, spikes, null):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    result = _significance(env, rates, events[0], counts[0], null, dt=0.02)
    assert np.isfinite(result.p_value)


@pytest.mark.parametrize("null", ["column_cycle", "time_bin_permutation"])
def test_compare_refuses_overlap_without_dt(track, spikes, null):
    """Posterior-only nulls need no ``dt``, so the event's own record must tell."""
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    method = dataclasses.replace(_method(null), criteria=())
    with pytest.raises(ValueError, match="cannot test overlapping"):
        compare_replay_methods(env, events, replay_methods=[method], rng=0)


def test_compare_runs_tiled_events_without_dt(track, spikes):
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02)
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    table = compare_replay_methods(env, events, replay_methods=[method], rng=0)
    assert len(table) == 2


def test_refusal_names_a_null_that_works(track, spikes):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.01)
    with pytest.raises(ValueError, match="unit_identity or place_field_circular"):
        _significance(env, rates, events[0], counts[0], "column_cycle", dt=0.02)
