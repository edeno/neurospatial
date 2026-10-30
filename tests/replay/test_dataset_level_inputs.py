"""Inputs of data-set-level replay tests added to existing replay functions.

Null data set copies, rates against chance, batched decoding, the criterion
values and z-scores of comparison tables, and matched alphas of the
pairwise-order test.
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np
import pandas as pd
import pytest
from replay_helpers import sweeps_and_jumps
from scipy import stats

from neurospatial import Environment
from neurospatial.replay import (
    compare_replay_methods,
    detect_replay,
    detection_rates,
    null_events,
    replay_significance,
)
from neurospatial.replay.methods import SILVA_2015

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
