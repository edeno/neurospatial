"""Inputs of data-set-level replay tests added to existing replay functions.

Null data set copies, rates against chance, batched decoding, the criterion
values and z-scores of comparison tables, and matched alphas of the
pairwise-order test.
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np
import pytest
from replay_helpers import sweeps_and_jumps

from neurospatial import Environment
from neurospatial.replay import (
    compare_replay_methods,
    detect_replay,
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
