"""Replay functions take an Intervals of events as they take an (n, 2) array.

Each event stays one row: rows are neither merged nor dropped, so results
follow the Intervals' rows and map back to their interval_ids.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, Intervals
from neurospatial.replay import (
    dataset_values,
    decode_intervals,
    mean_pairwise_bias,
    null_events,
    relative_rank_template,
    template_match_dataset_test,
)
from neurospatial.replay.methods import DIBA_BUZSAKI_2007, LEE_WILSON_2002

STEP = 0.06  # seconds between cells in an event
N_CELLS = 8


@pytest.fixture(scope="module")
def track():
    """A 160 cm track with 8 cells whose fields peak 20 cm apart."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [160.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 20.0 * np.exp(
        -0.5 * ((x - (10.0 + 20.0 * np.arange(N_CELLS))[:, None]) / 6.0) ** 2
    )
    return env, rates


@pytest.fixture(scope="module")
def session():
    """Five events 2 s apart, cells firing forward or backward in each.

    Returns the spike trains, the events as a time-sorted (n, 2) array, and
    the same events as Intervals whose ids are not their positions.
    """
    orders = [range(N_CELLS), reversed(range(N_CELLS))] * 2 + [range(N_CELLS)]
    trains = [[] for _ in range(N_CELLS)]
    rows = []
    for index, order in enumerate(orders):
        onset = 1.0 + 2.0 * index
        for k, cell in enumerate(order):
            trains[cell].extend([onset + STEP * k, onset + STEP * k + 0.01])
        rows.append((onset, onset + STEP * N_CELLS))
    rows = np.array(rows)
    ids = np.array([40, 10, 30, 0, 20])
    # Given out of time order; construction sorts the rows and keeps the ids.
    shuffled = np.array([3, 0, 4, 1, 2])
    events = Intervals.from_arrays(
        rows[shuffled, 0], rows[shuffled, 1], interval_ids=ids[shuffled]
    )
    return [np.sort(np.array(t)) for t in trains], rows, events


def test_null_events_source_events_follow_the_intervals_rows(session):
    trains, rows, events = session
    from_array = null_events(trains, rows, kind="spike_shift", n_copies=2, rng=3)
    from_intervals = null_events(trains, events, kind="spike_shift", n_copies=2, rng=3)

    np.testing.assert_array_equal(from_intervals.intervals, from_array.intervals)
    np.testing.assert_array_equal(
        from_intervals.source_events, from_array.source_events
    )
    for got, expected in zip(
        from_intervals.spike_times, from_array.spike_times, strict=True
    ):
        np.testing.assert_array_equal(got, expected)
    # Each copy maps back to its event's id.
    np.testing.assert_array_equal(
        events.interval_ids[from_intervals.source_events],
        np.repeat([40, 10, 30, 0, 20], 2),
    )


def test_decode_intervals_decodes_each_row_in_order(track, session):
    env, rates = track
    trains, rows, events = session
    expected, expected_counts = decode_intervals(env, trains, rows, rates, dt=0.02)
    decoded, counts = decode_intervals(env, trains, events, rates, dt=0.02)

    assert len(decoded) == len(rows)
    for got, want in zip(decoded, expected, strict=True):
        np.testing.assert_array_equal(got.times, want.times)
        np.testing.assert_array_equal(got.posterior, want.posterior)
    for got, want in zip(counts, expected_counts, strict=True):
        np.testing.assert_array_equal(got, want)


def test_relative_rank_template_reads_intervals(session):
    trains, rows, events = session
    np.testing.assert_array_equal(
        relative_rank_template(trains, events), relative_rank_template(trains, rows)
    )


def test_rank_order_dataset_values_read_intervals(track, session):
    env, rates = track
    trains, rows, events = session
    method = dataclasses.replace(DIBA_BUZSAKI_2007, template_choice="most_active")

    def values(given):
        return dataset_values(
            env,
            given,
            replay_method=method,
            encoding_models=rates,
            spike_times=trains,
            rng=0,
        )

    expected = values(rows)
    assert len(expected) == len(rows)
    pd.testing.assert_frame_equal(values(events), expected)


def test_template_match_dataset_test_reads_intervals(track, session):
    env, rates = track
    trains, rows, events = session
    method = dataclasses.replace(LEE_WILSON_2002, n_shuffles=5)

    def groups(given):
        return template_match_dataset_test(
            env, trains, given, rates, template_match_method=method, rng=0
        ).groups_table()

    pd.testing.assert_frame_equal(groups(events), groups(rows))


def test_mean_pairwise_bias_reads_intervals(session):
    trains, rows, events = session
    np.testing.assert_array_equal(
        mean_pairwise_bias(trains, events), mean_pairwise_bias(trains, rows)
    )


def test_a_zero_duration_event_is_refused_not_dropped(session):
    """Dropping it would shift every later event's results by one row."""
    trains = session[0]
    events = Intervals.from_arrays([1.0, 3.0], [1.0, 3.5])
    with pytest.raises(ValueError, match="no duration"):
        relative_rank_template(trains, events)
