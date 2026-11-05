"""Replay tables say when each event happened and which interval it came from.

``event`` is a position in the list passed in, so it stops naming the same
event after the list is filtered. Each decoded event therefore carries its own
interval identity, and both tables report it with the event's clock time.
"""

import dataclasses

import numpy as np
import pytest

from neurospatial import Environment, Intervals
from neurospatial.decoding import DecodingResult, decode_jointly, decode_position
from neurospatial.replay import (
    compare_replay_methods,
    decode_intervals,
    event_properties,
)
from neurospatial.replay.methods import DAVIDSON_2009, SILVA_2015

DT = 0.02
METHOD = dataclasses.replace(SILVA_2015, n_shuffles=5)


@pytest.fixture(scope="module")
def track():
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 8)[:, None]) / 8.0) ** 2
    )
    return env, rates


@pytest.fixture(scope="module")
def spikes():
    rng = np.random.default_rng(0)
    return [np.sort(rng.uniform(0.0, 20.0, 400)) for _ in range(8)]


# The ids are not row numbers, and the first interval leaves a 5 ms remainder.
WINDOWS = np.array([[1.0, 1.105], [4.0, 4.1], [9.0, 9.14]])
IDS = [7, 3, 11]


@pytest.mark.parametrize("overlapping", [False, True], ids=["batched", "one_by_one"])
def test_decoded_events_keep_their_interval(track, spikes, overlapping):
    """Both decoding paths record the interval and the time they covered."""
    env, rates = track
    # A row overlapping the first sends decode_intervals down its other path.
    windows = (
        np.array([[1.0, 1.105], [1.05, 1.15], [4.0, 4.1], [9.0, 9.14]])
        if overlapping
        else WINDOWS
    )
    ids = [7, 20, 3, 11] if overlapping else IDS
    intervals = Intervals.from_arrays(windows[:, 0], windows[:, 1], interval_ids=ids)

    events, _ = decode_intervals(env, spikes, intervals, rates, dt=DT)

    for event, (start, stop), interval_id in zip(events, windows, ids, strict=True):
        assert event.interval_id == interval_id
        n_bins = int((stop - start + 1e-9) // DT)
        assert event.time_support == pytest.approx(
            np.array([[start, start + n_bins * DT]])
        )


def test_array_intervals_are_identified_by_row(track, spikes):
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=DT)
    assert [event.interval_id for event in events] == [0, 1, 2]


def test_tables_report_clock_time_and_interval(track, spikes):
    env, rates = track
    intervals = Intervals.from_arrays(WINDOWS[:, 0], WINDOWS[:, 1], interval_ids=IDS)
    events, _ = decode_intervals(env, spikes, intervals, rates, dt=DT)

    properties = event_properties(env, events)
    table = compare_replay_methods(env, events, replay_methods=[METHOD], rng=0)

    # 1.105 s holds five whole 20 ms bins: the 5 ms remainder was not decoded.
    expected_stop = [1.1, 4.1, 9.14]
    assert properties["start_time"].to_numpy() == pytest.approx(WINDOWS[:, 0])
    assert properties["stop_time"].to_numpy() == pytest.approx(expected_stop)
    assert properties["interval_id"].tolist() == IDS
    assert table["start_time"].to_numpy() == pytest.approx(WINDOWS[:, 0])
    assert table["stop_time"].to_numpy() == pytest.approx(expected_stop)
    assert table["interval_id"].tolist() == IDS


def test_identity_survives_filtering_the_events(track, spikes):
    """The reason ``event`` is not enough: after a filter it is a new position."""
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=DT)

    kept = compare_replay_methods(env, events[1:], replay_methods=[METHOD], rng=0)

    assert kept["event"].tolist() == [0, 1]
    assert kept["interval_id"].tolist() == [1, 2]
    assert kept["start_time"].to_numpy() == pytest.approx(WINDOWS[1:, 0])


def test_joint_decode_has_time_but_no_interval(track):
    env, rates = track
    counts = np.ones((4, 8), dtype=np.int64)
    times = 5.0 + DT * (np.arange(4) + 0.5)
    joint = decode_jointly(
        env, counts, {"out": rates, "in": rates[::-1]}, dt=DT, times=times
    )

    row = event_properties(env, [joint], template="out").iloc[0]

    assert row["start_time"] == pytest.approx(5.0)
    assert row["stop_time"] == pytest.approx(5.0 + 4 * DT)
    assert np.isnan(row["interval_id"])


def test_separately_decoded_templates_keep_time_and_interval(track, spikes):
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=DT)
    mapping = {"out": events[1], "in": events[1]}

    method = dataclasses.replace(
        DAVIDSON_2009,
        n_shuffles=5,
        nulls=("column_cycle",),
        template_choice="smallest_p",
    )
    row = compare_replay_methods(
        {"out": env, "in": env}, [mapping], replay_methods=[method], rng=0
    ).iloc[0]

    assert row["start_time"] == pytest.approx(4.0)
    assert row["stop_time"] == pytest.approx(4.1)
    assert row["interval_id"] == 1


def test_span_comes_from_dt_when_the_event_has_no_support(track):
    """A hand-built event: ``dt`` gives the span, and without it the span is unknown."""
    env, _ = track
    posterior = np.full((3, env.n_bins), 1.0 / env.n_bins)
    event = DecodingResult(posterior=posterior, env=env, times=[2.01, 2.03, 2.05])

    with_dt = event_properties(env, [event], dt=DT).iloc[0]
    without = event_properties(env, [event]).iloc[0]

    assert with_dt["start_time"] == pytest.approx(2.0)
    assert with_dt["stop_time"] == pytest.approx(2.06)
    assert np.isnan(without["start_time"])
    assert np.isnan(without["stop_time"])


def test_decode_position_events_report_their_span(track):
    env, rates = track
    counts = np.ones((3, 8), dtype=np.int64)
    event = decode_position(
        env, counts, rates, dt=DT, times=np.array([2.01, 2.03, 2.05])
    )

    row = event_properties(env, [event]).iloc[0]

    assert row["start_time"] == pytest.approx(2.0)
    assert row["stop_time"] == pytest.approx(2.06)


def test_interval_id_survives_replace(track, spikes):
    env, rates = track
    events, _ = decode_intervals(env, spikes, WINDOWS, rates, dt=DT)
    assert (
        dataclasses.replace(events[2], posterior=events[2].posterior).interval_id == 2
    )
