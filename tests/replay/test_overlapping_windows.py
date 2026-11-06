"""Shuffles that cannot test overlapping windows refuse them.

Overlapping windows share spikes, so neighboring rows of an event are
correlated even when the spikes carry no sequence. These nulls lose that
correlation, so the observed event looks smoother than its shuffles and a 5%
test rejects far too often. The two nulls that re-decode the same rows keep it.
A recipe whose paper published such windows still runs, and warns.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult, decode_jointly
from neurospatial.replay import (
    compare_replay_methods,
    dataset_values,
    decode_intervals,
    detect_replay,
    replay_significance,
    shuffle_events,
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
    """A method that does not publish overlapping windows, so they are refused."""
    return dataclasses.replace(
        WU_2017,
        name=null,
        nulls=(null,),
        n_shuffles=5,
        empty_time_bins="keep",
        decoding={"time_bin": 0.02},
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
    with pytest.raises(ValueError, match=rf"cannot test overlapping.*{null}"):
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


UNIX_CLOCK = 1.7e9


@pytest.mark.parametrize("null", ["column_cycle", "time_bin_permutation"])
def test_detect_replay_refuses_posterior_only_nulls(track, spikes, null):
    """These nulls need no ``dt`` to run, but ``dt`` still says the rows overlap."""
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    with pytest.raises(ValueError, match="cannot test overlapping"):
        detect_replay(
            env,
            events[0].posterior,
            events[0].times,
            replay_method=_method(null),
            spike_counts=counts[0],
            dt=0.02,
            rng=0,
        )


def test_detect_replay_warns_on_windows_the_method_publishes(track, spikes):
    """``WU_2017`` used 20 ms stepped by 10 ms, so it runs as published and warns."""
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.01)
    with pytest.warns(UserWarning, match="too small") as record:
        result = detect_replay(
            env,
            events[0].posterior,
            events[0].times,
            replay_method=dataclasses.replace(WU_2017, n_shuffles=5),
            spike_counts=counts[0],
            dt=0.02,
            rng=0,
        )
    assert len(record) == 1
    assert "time_bin_permutation" in str(record[0].message)
    assert np.isfinite(result.p_value)


def test_compare_warns_once_for_a_method_that_publishes_the_windows(track, spikes):
    """One warning per method, not one per event."""
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.01)
    method = dataclasses.replace(WU_2017, n_shuffles=5)
    with pytest.warns(UserWarning, match="too small") as record:
        table = compare_replay_methods(
            env, events, replay_methods=[method], spike_counts=counts, rng=0
        )
    assert sum("too small" in str(item.message) for item in record) == 1
    assert "compare_replay_methods()" in str(record[0].message)
    assert np.isfinite(table["p_value"]).all()


def test_a_tiled_event_does_not_warn_for_a_method_that_publishes_overlap(track, spikes):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02)
    method = dataclasses.replace(WU_2017, n_shuffles=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_replay_methods(
            env, events, replay_methods=[method], spike_counts=counts, rng=0
        )


def test_overlap_the_method_does_not_publish_is_still_refused(track, spikes):
    """The warning is for reproducing a paper, not for adding overlap to one."""
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.01)
    method = dataclasses.replace(WU_2017, n_shuffles=5, decoding={"time_bin": 0.02})
    with pytest.raises(ValueError, match="cannot test overlapping"):
        detect_replay(
            env,
            events[0].posterior,
            events[0].times,
            replay_method=method,
            spike_counts=counts[0],
            dt=0.02,
            rng=0,
        )


def test_detect_replay_refuses_a_joint_decode_by_its_record(track):
    env, rates = track
    counts = np.ones((8, N_CELLS), dtype=np.int64)
    joint = decode_jointly(
        env,
        counts,
        {"out": rates, "in": rates[::-1]},
        dt=0.02,
        times=5.0 + 0.005 * np.arange(8),
    )
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    with pytest.raises(ValueError, match="cannot test overlapping"):
        detect_replay(
            {"out": env, "in": env},
            joint,
            joint.conditional["out"].times,
            replay_method=method,
            template="out",
            rng=0,
        )


@pytest.mark.parametrize("dt", [0.005, 0.015, 0.02, 0.025])
def test_tiled_events_on_a_unix_clock_are_not_refused(track, dt):
    """Times near 1.7e9 s round by about 2e-7 s, which is not an overlap."""
    env, rates = track
    rng = np.random.default_rng(1)
    trains = [np.sort(UNIX_CLOCK + rng.uniform(0.0, 3.0, 60)) for _ in range(N_CELLS)]
    windows = UNIX_CLOCK + WINDOWS
    events, counts = decode_intervals(env, trains, windows, rates, dt=dt)
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    table = compare_replay_methods(
        env, events, replay_methods=[method], spike_counts=counts, dt=dt, rng=0
    )
    assert table["error"].eq("").all()


def test_tiled_times_stored_as_float32_are_not_refused(track):
    """Single precision moves each center by up to about 4 microseconds at 100 s."""
    env, _ = track
    n_bins = 10
    times = np.float64(np.float32(100.0 + 0.02 * np.arange(n_bins)))
    posterior = np.full((n_bins, env.n_bins), 1.0 / env.n_bins)
    posterior[np.arange(n_bins), np.arange(n_bins)] += 0.5
    posterior /= posterior.sum(axis=1, keepdims=True)
    result = replay_significance(
        env,
        posterior,
        times,
        score=fit_linear_trajectory,
        nulls=("column_cycle",),
        n_shuffles=5,
        rng=0,
    )
    assert np.isfinite(result.p_value)
    event = DecodingResult(posterior=posterior, env=env, times=times, dt=0.02)
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    table = compare_replay_methods(env, [event], replay_methods=[method], rng=0)
    assert table["error"].eq("").all()


def test_the_recorded_width_wins_over_a_mistaken_dt(track, spikes):
    """``dt=step`` is the mistake the decode_intervals docstring warns about."""
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    with pytest.raises(ValueError, match="cannot test overlapping"):
        compare_replay_methods(
            env, events, replay_methods=[method], spike_counts=counts, dt=0.005, rng=0
        )


def test_an_overlapping_event_with_a_missing_row_is_refused(track, spikes):
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    keep = np.delete(np.arange(len(events[0].times)), 10)
    gapped = DecodingResult(
        posterior=events[0].posterior[keep],
        env=env,
        times=events[0].times[keep],
        dt=events[0].dt,
    )
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    with pytest.raises(ValueError, match="cannot test overlapping"):
        compare_replay_methods(env, [gapped], replay_methods=[method], rng=0)


def test_a_trimmed_tiled_event_is_not_refused(track, spikes):
    """Its recorded support is wider than its rows, which is not an overlap."""
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02)
    trimmed = dataclasses.replace(
        events[0], posterior=events[0].posterior[:6], times=events[0].times[:6]
    )
    method = dataclasses.replace(_method("column_cycle"), criteria=())
    table = compare_replay_methods(env, [trimmed], replay_methods=[method], rng=0)
    assert table["error"].eq("").all()


@pytest.mark.parametrize("null", ["time_bin_permutation", "column_cycle"])
def test_dataset_routes_refuse_overlapping_windows(track, spikes, null):
    env, rates = track
    events, counts = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02, step=0.005)
    with pytest.raises(ValueError, match=r"dataset_values\(\) cannot test overlapping"):
        dataset_values(
            env,
            events,
            replay_method=dataclasses.replace(_method(null), criteria=()),
            null=null,
            spike_counts=counts,
            rng=0,
        )
    with pytest.raises(ValueError, match=r"shuffle_events\(\) cannot test overlapping"):
        shuffle_events(env, events, null=null, rng=0)


def test_shuffled_events_keep_their_record(track, spikes):
    """A shuffled copy is still that event, so a later check can read it."""
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=0.02)
    (shuffled,) = shuffle_events(env, events, null="time_bin_permutation", rng=0)
    for original, copy in zip(events, shuffled.events, strict=True):
        assert copy.dt == original.dt
        assert copy.interval_id == original.interval_id
        np.testing.assert_array_equal(copy.time_support, original.time_support)
