"""A small simulated sleep-run-sleep session for the reactivation recipes.

Units fire as independent Poisson processes. Groups of units (assemblies)
also fire together in brief events: during running (the template period)
at ``event_rate`` events per second, during rest after running (the match
period) at ``reactivation * event_rate``, and never during rest before
running (the control period).
"""

from __future__ import annotations

import numpy as np

# Each assembly spans both regions, two shanks per region and four tetrodes.
ASSEMBLIES = ((0, 5, 12, 17), (2, 7, 14, 21))
N_UNITS = 24
# Two units per tetrode, two tetrodes per shank, twelve units per region.
UNIT_GROUPS = {
    "tetrode": np.arange(N_UNITS) // 2,
    "shank": np.arange(N_UNITS) // 4,
    "region": np.arange(N_UNITS) // 12,
}


def session(
    seed: int,
    *,
    reactivation: float = 1.0,
    duration: float = 1200.0,
    rate: float = 2.0,
    event_rate: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Spike times and the (template, match, control) epochs of one session.

    Returns
    -------
    spike_times : list of ndarray
        One sorted train per unit, seconds.
    template_epochs, match_epochs, control_epochs : ndarray, shape (1, 2)
        Running, rest after and rest before.
    """
    rng = np.random.default_rng(seed)
    control = np.array([[0.0, duration]])
    template = np.array([[duration, 2 * duration]])
    match = np.array([[2 * duration, 3 * duration]])
    total = 3 * duration
    trains = [
        list(rng.uniform(0.0, total, rng.poisson(rate * total))) for _ in range(N_UNITS)
    ]
    for members in ASSEMBLIES:
        for (start, stop), events_per_second in (
            (template[0], event_rate),
            (match[0], reactivation * event_rate),
        ):
            events = rng.uniform(
                start, stop, rng.poisson(events_per_second * (stop - start))
            )
            for unit in members:
                n_spikes = rng.poisson(2.0, events.size)
                spikes = np.repeat(events, n_spikes) + rng.normal(
                    0.0, 0.01, n_spikes.sum()
                )
                trains[unit].extend(spikes)
    spike_times = [np.sort(np.clip(train, 0.0, total - 1e-6)) for train in trains]
    return spike_times, template, match, control
