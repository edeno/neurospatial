"""How null_events and simulated_null_events randomize each cell and event."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import null_events, simulated_null_events

DURATION = 0.2


@pytest.fixture(scope="module")
def recording():
    """Three 200 ms events of 6 cells; cell c fires c + 1 spikes in each."""
    rng = np.random.default_rng(2)
    intervals = np.array([[1.0, 1.2], [3.0, 3.2], [7.0, 7.2]])
    trains = []
    for cell in range(6):
        inside = [start + rng.uniform(0, DURATION, cell + 1) for start, _ in intervals]
        outside = rng.uniform(0, 10, 5)
        trains.append(np.sort(np.concatenate([*inside, outside])))
    return trains, intervals


def _in_events(trains, intervals):
    """Each event's spikes relative to its start, per cell, to 1 ns."""
    return [
        [
            np.round(np.sort(train[(train >= a) & (train < b)] - a), 9)
            for train in trains
        ]
        for a, b in intervals
    ]


def _circular_shift(original, shifted):
    """The shift s in [0, DURATION) with shifted = (original + s) mod DURATION."""
    for candidate in np.mod(shifted[0] - original, DURATION):
        if np.allclose(np.sort(np.mod(original + candidate, DURATION)), shifted):
            return candidate
    raise AssertionError("the copy is not a circular shift of the original")


def test_spike_shift_draws_its_own_shift_for_each_cell(recording):
    """One shift for all cells would keep the cells' relative timing."""
    trains, intervals = recording
    null = null_events(trains, intervals, kind="spike_shift", n_copies=2, rng=0)
    real = _in_events(trains, intervals)
    copies = _in_events(null.spike_times, null.intervals)
    for copy, source in zip(copies, null.source_events, strict=True):
        shifts = [_circular_shift(real[source][cell], copy[cell]) for cell in range(6)]
        assert np.unique(np.round(shifts, 9)).size == 6


@pytest.mark.parametrize("kind", ["cell_identity", "spike_shift", "poisson"])
def test_only_the_masked_cells_are_randomized(recording, kind):
    trains, intervals = recording
    cells = np.array([True, False, True, False, True, True])
    null = null_events(trains, intervals, kind=kind, n_copies=4, cells=cells, rng=0)
    real = _in_events(trains, intervals)
    copies = _in_events(null.spike_times, null.intervals)
    changed = 0
    for copy, source in zip(copies, null.source_events, strict=True):
        original = real[source]
        for cell in np.flatnonzero(~cells):
            np.testing.assert_array_equal(copy[cell], original[cell])
        inside = np.flatnonzero(cells)
        if kind == "cell_identity":
            # Trains are swapped among the masked cells only.
            assert sorted(tuple(copy[c]) for c in inside) == sorted(
                tuple(original[c]) for c in inside
            )
        else:
            assert [copy[c].size for c in inside] == [original[c].size for c in inside]
        changed += sum(
            copy[c].size != original[c].size or not np.allclose(copy[c], original[c])
            for c in inside
        )
    assert changed > 0


@pytest.fixture(scope="module")
def track_rates():
    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )
    x = track.bin_centers[:, 0]
    rates = 1.0 + 30.0 * np.exp(
        -0.5 * ((x - np.linspace(5, 95, 8)[:, None]) / 8.0) ** 2
    )
    return track, rates


def _per_event(null):
    """Each event's spike times relative to its start, per cell."""
    return [
        tuple(
            np.round(train[(train >= a) & (train < b)] - a, 12)
            for train in null.spike_times
        )
        for a, b in null.intervals
    ]


def _same(first, second):
    return all(
        a.size == b.size and np.array_equal(a, b)
        for a, b in zip(first, second, strict=True)
    )


@pytest.mark.parametrize("kind", ["random", "stationary"])
def test_simulated_events_follow_the_rng(track_rates, kind):
    track, rates = track_rates

    def simulate(rng):
        return _per_event(
            simulated_null_events(
                track, rates, n_events=4, kind=kind, rate_gain=4.0, rng=rng
            )
        )

    first, again, other = simulate(0), simulate(0), simulate(1)
    assert all(_same(a, b) for a, b in zip(first, again, strict=True))
    assert not any(_same(a, b) for a, b in zip(first, other, strict=True))


def test_simulated_random_events_differ_from_each_other(track_rates):
    track, rates = track_rates
    events = _per_event(
        simulated_null_events(track, rates, n_events=4, rate_gain=4.0, rng=0)
    )
    for previous, event in itertools.pairwise(events):
        assert not _same(previous, event)
