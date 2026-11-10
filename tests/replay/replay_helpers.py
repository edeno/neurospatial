"""Posteriors shared by the replay tests."""

from __future__ import annotations

import dataclasses

import numpy as np
from calibration_helpers import monte_carlo_rejection_rate, null_rejection_interval


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
            track, rates, kind=kind, rate_gain=3.0, rng=index, **extra
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
    """The random stream one template draws from: the seed and its name.

    A call's streams derive from ``np.random.SeedSequence(entropy)``, where
    ``entropy`` is the first draw of ``np.random.default_rng(seed)``.
    """
    import zlib

    entropy = int(np.random.default_rng(seed).integers(2**63))
    return np.random.default_rng(
        np.random.SeedSequence(entropy, spawn_key=(zlib.crc32(name.encode()),))
    )


def assert_null_verdicts_at_alpha(
    p_values, is_replay, eligible, *, alpha, n_shuffles, conservative=False
):
    """Check a recipe's verdicts on null events against its alpha.

    ``eligible`` marks the events that pass every criterion; only those can
    be replay, and each is replay with probability
    ``monte_carlo_rejection_rate(n_shuffles, alpha)`` if the shuffle test is
    exact. The count of replay verdicts must fall inside the central 99%
    binomial interval. ``P(p < 0.5)`` over all null events must match too,
    so a null that never rejects fails even when a small alpha puts the
    interval's low end at 0.
    A ``conservative`` test, such as the largest of several p-values, is
    checked against the high end only.
    """
    p_values = np.asarray(p_values, dtype=float)
    is_replay = np.asarray(is_replay, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    assert not np.any(is_replay & ~eligible), "replay verdict failing a criterion"
    rate = monte_carlo_rejection_rate(n_shuffles, alpha)
    low, high = null_rejection_interval(int(eligible.sum()), rate)
    n_replay = int(is_replay.sum())
    message = f"{n_replay} of {int(eligible.sum())} null events are replay"
    assert n_replay <= high, f"{message}; at most {high} expected at rate {rate}"
    if conservative:
        return
    assert n_replay >= low, f"{message}; at least {low} expected at rate {rate}"
    defined = p_values[np.isfinite(p_values)]
    half_low, half_high = null_rejection_interval(
        defined.size, monte_carlo_rejection_rate(n_shuffles, 0.5)
    )
    n_half = int(np.sum(defined < 0.5))
    assert half_low <= n_half <= half_high, (
        f"{n_half} of {defined.size} null p-values below 0.5; "
        f"{half_low} to {half_high} expected"
    )


@dataclasses.dataclass(frozen=True)
class NullVerdicts:
    """A recipe's verdicts and p-values on events without replay.

    ``replay`` holds the verdict of each event that passes every criterion
    and ``p_values`` the defined p-values. ``below(level)`` marks the
    p-values under ``level``; an exact shuffle test puts a share
    ``exact_rate(level)`` of them there.
    """

    replay: np.ndarray
    p_values: np.ndarray
    n_shuffles: int

    def below(self, level):
        return self.p_values < level

    def exact_rate(self, level):
        return monte_carlo_rejection_rate(self.n_shuffles, level)


def null_verdicts(p_values, is_replay, eligible, *, n_shuffles):
    """Collect a recipe's p-values and verdicts on null events.

    ``eligible`` marks the events that pass every criterion. A replay
    verdict for an event that fails one raises ``ValueError``. It is not an
    ``AssertionError``, so it does not count as the failed rate assertion of
    a test marked ``xfail(raises=AssertionError)``.
    """
    p_values = np.asarray(p_values, dtype=float)
    is_replay = np.asarray(is_replay, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if np.any(is_replay & ~eligible):
        msg = "replay verdict for an event that fails a criterion"
        raise ValueError(msg)
    return NullVerdicts(
        replay=is_replay[eligible],
        p_values=p_values[np.isfinite(p_values)],
        n_shuffles=n_shuffles,
    )
