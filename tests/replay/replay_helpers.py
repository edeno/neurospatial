"""Posteriors shared by the replay tests."""

from __future__ import annotations

import numpy as np


def wrapped_bump(env, centers, sigma):
    """Posterior rows of a Gaussian bump wrapped around the track's ends."""
    position = env.bin_centers[:, 0]
    length = float(env.dimension_ranges[0][1] - env.dimension_ranges[0][0])
    distance = (position[None, :] - np.asarray(centers)[:, None] + length / 2) % length
    rows = np.exp(-0.5 * ((distance - length / 2) / sigma) ** 2)
    return rows / rows.sum(axis=1, keepdims=True)


def sweeps_and_jumps(track, dt=0.02):
    """Twelve simulated events, sweeps and random jumps, decoded in ``dt`` bins.

    Returns the rate maps (20 cells), the spike trains, the event intervals,
    and the decoded events and their spike counts.
    """
    from neurospatial.replay import decode_intervals
    from neurospatial.simulation import simulate_replay_event

    x = track.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 20)[:, None]) / 8.0) ** 2
    )
    trains = [[] for _ in range(20)]
    intervals = []
    for index in range(12):
        kind = "sweep" if index % 2 else "random"
        extra = {"speed": 400.0} if kind == "sweep" else {}
        event = simulate_replay_event(
            track, rates, kind=kind, rate_gain=3.0, seed=index, **extra
        )
        start = 2.0 * index
        for cell, train in enumerate(event.spike_times):
            trains[cell].append(np.asarray(train) + start)
        intervals.append((start, start + event.duration))
    trains = [np.concatenate(parts) for parts in trains]
    intervals = np.asarray(intervals)
    events, counts = decode_intervals(track, trains, intervals, rates, dt=dt)
    return rates, trains, intervals, events, counts


def template_stream(seed, name):
    """The random stream one template draws from: the seed and its name."""
    import zlib

    entropy = np.random.SeedSequence(seed).entropy
    return np.random.default_rng(
        np.random.SeedSequence(entropy, spawn_key=(zlib.crc32(name.encode()),))
    )
