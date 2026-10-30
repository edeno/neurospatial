"""Inputs of data-set-level replay tests added to existing replay functions.

Null data set copies, rates against chance, batched decoding, the criterion
values and z-scores of comparison tables, and matched alphas of the
pairwise-order test.
"""

from __future__ import annotations

import dataclasses
import warnings
import zlib

import numpy as np
import pandas as pd
import pytest
from replay_helpers import sweeps_and_jumps
from scipy import stats

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    apply_alpha,
    compare_replay_methods,
    decode_intervals,
    detect_replay,
    detection_rates,
    matched_alpha,
    null_events,
    replay_significance,
)
from neurospatial.replay.methods import GUPTA_2010, SILVA_2015
from neurospatial.simulation import simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )


@pytest.fixture(scope="module")
def session(track):
    """Twelve simulated events, sweeps and random jumps, decoded in 20 ms bins."""
    return sweeps_and_jumps(track, dt=DT)


def test_null_events_name_each_copy():
    trains = [np.array([1.01, 1.05, 3.02]), np.array([1.02, 3.05])]
    null = null_events(
        trains, np.array([[1.0, 1.1], [3.0, 3.1]]), kind="cell_identity", n_copies=3
    )
    np.testing.assert_array_equal(null.copy_index, np.tile([0, 1, 2], 2))
    np.testing.assert_array_equal(null.source_events, np.repeat([0, 1], 3))


def test_detection_rates_test_against_a_chance_rate():
    table = pd.DataFrame(
        {
            "event": np.tile(np.arange(60), 2),
            "method": np.repeat(["a", "b"], 60),
            "is_replay": np.r_[np.arange(60) < 9, np.arange(60) < 1],
        }
    )
    rates = detection_rates(table, expected_rate=0.05, alternative="greater")
    for row, n_replay in zip(rates.itertuples(), (9, 1), strict=True):
        assert row.p_value == pytest.approx(
            stats.binomtest(n_replay, 60, 0.05, alternative="greater").pvalue
        )
    assert "p_value" not in detection_rates(table)


def test_batched_decoding_equals_decoding_each_interval(track, session):
    rates, trains, intervals, _, _ = session
    start, stop = intervals[0]
    # Unsorted intervals, one touching another, take the batched path;
    # overlapping ones are decoded one at a time.
    for windows in (
        np.vstack([intervals[::-1], [[stop, stop + 0.1]]]),
        np.array([[start, stop], [start + 0.05, stop]]),
    ):
        events, counts = decode_intervals(track, trains, windows, rates, dt=DT)
        for (a, b), event, event_counts in zip(windows, events, counts, strict=True):
            expected_counts, grid = bin_spikes_in_time(
                trains, dt=DT, t_start=a, t_stop=b
            )
            expected = decode_position(track, expected_counts, rates, dt=DT, times=grid)
            np.testing.assert_array_equal(event_counts, expected_counts)
            np.testing.assert_array_equal(event.times, grid)
            np.testing.assert_allclose(
                event.posterior, expected.posterior, rtol=0, atol=1e-12
            )


def test_compare_table_reports_values_and_z_scores(track, session):
    _, _, _, events, counts = session
    method = dataclasses.replace(
        SILVA_2015, nulls=("column_cycle",), n_shuffles=50, use_absolute=True
    )
    table = compare_replay_methods(
        track, events[:3], replay_methods=[method], spike_counts=counts[:3], rng=0
    )
    for index, event in enumerate(events[:3]):
        stream = np.random.default_rng(
            np.random.SeedSequence(
                table.attrs["rng_entropy"],
                spawn_key=(index, zlib.crc32(b"silva_2015")),
            )
        )
        result = detect_replay(
            track,
            event.posterior,
            event.times,
            replay_method=method,
            spike_counts=counts[index],
            rng=stream,
        )
        row = table.loc[index]
        assert row["z_score[column_cycle]"] == result.z_scores["column_cycle"]
        assert row["value[|score| > 0.6]"] == result.score
    tested = replay_significance(
        track,
        events[1].posterior,
        events[1].times,
        score=method.score,
        nulls=("column_cycle",),
        n_shuffles=50,
        use_absolute=True,
        rng=0,
    )
    z = tested.shuffles["column_cycle"]
    assert z.z_score == pytest.approx(
        (abs(tested.observed.score) - np.mean(z.null_scores)) / np.std(z.null_scores)
    )


def test_matched_alpha_takes_a_pairwise_order_method(track):
    """matched_alpha and apply_alpha handle Gupta et al.'s pairwise-order test."""
    x = track.bin_centers[:, 0]
    rates = 0.2 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 12)[:, None]) / 6.0) ** 2
    )
    trains = [[] for _ in range(12)]
    intervals = []
    for index in range(10):
        event = simulate_replay_event(
            track, rates, kind="sweep", speed=400.0, rate_gain=4.0, seed=index
        )
        for cell, train in enumerate(event.spike_times):
            trains[cell].append(np.asarray(train) + 2.0 * index)
        intervals.append((2.0 * index, 2.0 * index + event.duration))
    trains = [np.concatenate(parts) for parts in trains]
    null = null_events(
        trains, np.asarray(intervals), kind="cell_identity", n_copies=2, rng=0
    )
    method = dataclasses.replace(GUPTA_2010, n_shuffles=60)

    def table_of(spikes, windows, methods):
        events, _ = decode_intervals(track, spikes, windows, rates, dt=DT)
        return compare_replay_methods(
            track,
            events,
            replay_methods=methods,
            spike_times=spikes,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )

    null_table = table_of(null.spike_times, null.intervals, [method])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        matched = matched_alpha(null_table, replay_methods=[method], target=0.1)
    row = matched.iloc[0]
    assert row["n_events"] == 20 and row["rate"] <= 0.1
    passing = null_table["passes[n_active_cells >= 3]"].to_numpy(dtype=bool)
    assert row["rate"] == np.mean(passing & (null_table["p_value"] < row["alpha"]))
    # Rethresholding the table equals running the method at the new alpha.
    alpha = 0.3
    applied = apply_alpha(
        null_table, alpha={method.name: alpha}, replay_methods=[method]
    )
    rerun = table_of(
        null.spike_times, null.intervals, [dataclasses.replace(method, alpha=alpha)]
    )
    np.testing.assert_array_equal(applied["is_replay"], rerun["is_replay"])
    np.testing.assert_array_equal(applied["p_value"], rerun["p_value"])
    assert (applied["is_replay"] != null_table["is_replay"]).any()
