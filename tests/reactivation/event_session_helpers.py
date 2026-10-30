"""A small simulated session with sharp-wave ripples (SWRs) and theta cycles.

Units fire as independent Poisson processes. During the task (template)
each assembly fires together in brief events. Rest before and after hold
SWRs of varying length, many shorter than a 100 ms bin, in which every unit
fires faster; in a share ``reactivation`` of the SWRs after the task, one
assembly's members also fire together. The task is tiled by 125 ms theta
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from reactivation_helpers import ASSEMBLIES, N_UNITS


@dataclass(frozen=True)
class EventSession:
    spike_times: list[np.ndarray]
    template: np.ndarray
    match: np.ndarray
    control: np.ndarray
    theta_cycles: np.ndarray
    control_events: np.ndarray
    match_events: np.ndarray


def _swrs(rng: np.random.Generator, start: float, stop: float) -> np.ndarray:
    """Non-overlapping SWRs 5-150 ms long, about one every 2 s."""
    onsets = np.sort(rng.uniform(start, stop - 0.2, rng.poisson(0.5 * (stop - start))))
    onsets = onsets[np.r_[True, np.diff(onsets) > 0.2]]
    return np.column_stack([onsets, onsets + rng.uniform(0.005, 0.15, onsets.size)])


def event_session(
    seed: int,
    *,
    reactivation: float = 0.5,
    rate_change: float = 0.0,
    duration: float = 600.0,
    rate: float = 2.0,
) -> EventSession:
    """Rest, task and rest of ``duration`` seconds each.

    ``rate_change`` raises the assembly members' background rate in rest
    after the task without any co-firing.
    """
    rng = np.random.default_rng(seed)
    control = np.array([[0.0, duration]])
    template = np.array([[duration, 2 * duration]])
    match = np.array([[2 * duration, 3 * duration]])
    total = 3 * duration
    members = np.unique(np.concatenate(ASSEMBLIES))
    trains = [
        list(rng.uniform(0.0, total, rng.poisson(rate * total))) for _ in range(N_UNITS)
    ]
    for unit in members:
        trains[unit].extend(
            rng.uniform(
                match[0, 0], match[0, 1], rng.poisson(rate_change * rate * duration)
            )
        )
    for assembly in ASSEMBLIES:
        events = rng.uniform(duration, 2 * duration, rng.poisson(duration))
        for unit in assembly:
            n_spikes = rng.poisson(2.0, events.size)
            trains[unit].extend(
                np.repeat(events, n_spikes) + rng.normal(0.0, 0.01, n_spikes.sum())
            )
    swrs = {}
    for name, (start, stop) in (("control", control[0]), ("match", match[0])):
        swrs[name] = _swrs(rng, start, stop)
        lengths = swrs[name][:, 1] - swrs[name][:, 0]
        for unit in range(N_UNITS):
            n_spikes = rng.poisson(20.0 * lengths)
            trains[unit].extend(
                np.repeat(swrs[name][:, 0], n_spikes)
                + rng.uniform(0, 1, n_spikes.sum()) * np.repeat(lengths, n_spikes)
            )
    after = swrs["match"]
    reactivated = rng.uniform(size=len(after)) < reactivation
    which = rng.integers(len(ASSEMBLIES), size=len(after))
    for event in np.flatnonzero(reactivated):
        start, stop = after[event]
        for unit in ASSEMBLIES[which[event]]:
            trains[unit].extend(rng.uniform(start, stop, rng.poisson(2.0)))
    edges = np.arange(duration, 2 * duration + 1e-9, 0.125)
    return EventSession(
        spike_times=[np.sort(np.clip(train, 0.0, total - 1e-6)) for train in trains],
        template=template,
        match=match,
        control=control,
        theta_cycles=np.column_stack([edges[:-1], edges[1:]]),
        control_events=swrs["control"],
        match_events=after,
    )
