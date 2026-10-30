"""Null data, simulated sessions and detection rates for reactivation methods.

:func:`~neurospatial.reactivation.compare_reactivation_methods` returns one
row per (method, match window, component). To learn how often a method
calls reactivation when there is none, and what drives a false call:

- :func:`null_reactivation_data` replaces the spikes of the match period
  (rest after the task) with spikes that keep some of their properties and
  lose their co-firing with the template: cell identities permuted, spike
  trains rotated in time, or Poisson spikes at each unit's rate.
- :func:`simulated_reactivation_session` simulates rest, running and rest
  with assemblies that fire together during running and, by a chosen
  amount, during rest after it; ``rate_change`` raises the assembly
  members' rates after running without any co-firing, the confound
  Tingley & Peyrache (2020) warn about.
- :func:`reactivation_detection_rates` gives each method's share of rows
  called reactivated, with a confidence interval: on null data its
  false-positive rate, on simulated reactivation its sensitivity.

Tingley & Peyrache (2020) list three limits of explained variance: it
"does not account for the possible change in firing rates (independent of
correlations) that inevitably leads to changes in correlation", "does not
allow for the resolution of reactivation in time", and considers "all
cell-pair correlation values …, not controlling for the fact that data of
limited duration inevitably generates non-zero correlations". The null
data and the rate-change simulation test the first and the last.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._spikes import as_spike_trains

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SimulatedReactivationSession",
    "null_reactivation_data",
    "reactivation_detection_rates",
    "simulated_reactivation_session",
]

_KINDS = ("cell_identity", "circular_shift", "rate_poisson")


def null_reactivation_data(
    spike_times: Any,
    match_epochs: Any,
    *,
    kind: Literal["cell_identity", "circular_shift", "rate_poisson"],
    rng: np.random.Generator | int | None = None,
) -> list[NDArray[np.float64]]:
    """Spike times whose match-period spikes carry no reactivation.

    Spikes outside ``match_epochs`` are kept; inside them each unit's spikes
    are replaced, so a method applied to the result with the same template,
    control and match epochs measures its false-positive rate.

    Parameters
    ----------
    spike_times : sequence of arrays or spike group
        One spike train per unit, in seconds.
    match_epochs : array-like, shape (n, 2), or IntervalSet
        Epochs of the match period, for example rest after the task.
    kind : {"cell_identity", "circular_shift", "rate_poisson"}
        - ``"cell_identity"``: unit ``i`` gets the match-period spikes of
          unit ``π(i)`` for a random permutation ``π``. The match period
          keeps its own co-firing, eigenvalues and rates, but which cells
          fire together no longer matches the template.
        - ``"circular_shift"``: each unit's spikes in each match epoch are
          rotated in time by its own offset, drawn uniformly over the
          epoch, wrapping at the epoch's end. Each unit keeps its spike
          count and, but for the wrap, its inter-spike intervals; co-firing
          between units is lost (the spike-time version of the circular
          shift of Lopes-dos-Santos et al. 2013).
        - ``"rate_poisson"``: each unit fires as a homogeneous Poisson
          process at its mean rate over the match epochs. Keeps only the
          rates.
    rng : numpy.random.Generator, int, or None
        Seed or generator.

    Returns
    -------
    list of ndarray
        One sorted spike train per unit.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the kinds above.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import null_reactivation_data
    >>> spike_times = [np.array([1.0, 5.0, 12.0]), np.array([2.0, 15.0])]
    >>> null = null_reactivation_data(
    ...     spike_times, [[10.0, 20.0]], kind="cell_identity", rng=3
    ... )
    >>> [train.tolist() for train in null]  # spikes before 10 s are kept
    [[1.0, 5.0, 15.0], [2.0, 12.0]]
    """
    if kind not in _KINDS:
        raise ValueError(
            f"null_reactivation_data(): kind must be one of {_KINDS}, got {kind!r}."
        )
    generator = np.random.default_rng(rng)
    trains = as_spike_trains(spike_times)
    epochs = as_intervals(match_epochs, name="match_epochs")
    assert epochs is not None
    inside = [_in_epochs(train, epochs) for train in trains]
    outside = [train[~mask] for train, mask in zip(trains, inside, strict=True)]
    within = [train[mask] for train, mask in zip(trains, inside, strict=True)]
    if kind == "cell_identity":
        order = generator.permutation(len(trains))
        replaced = [within[source] for source in order]
    elif kind == "circular_shift":
        replaced = [_rotated(train, epochs, generator) for train in within]
    else:
        duration = float(np.sum(epochs[:, 1] - epochs[:, 0]))
        replaced = [
            _poisson(train.size / duration, epochs, generator) for train in within
        ]
    return [
        np.sort(np.concatenate([kept, new]))
        for kept, new in zip(outside, replaced, strict=True)
    ]


def _in_epochs(
    train: NDArray[np.float64], epochs: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Spikes inside any half-open ``[start, stop)`` epoch."""
    index = np.searchsorted(epochs[:, 0], train, side="right") - 1
    inside: NDArray[np.bool_] = (index >= 0) & (
        train < epochs[np.clip(index, 0, None), 1]
    )
    return inside


def _rotated(
    train: NDArray[np.float64],
    epochs: NDArray[np.float64],
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    """Rotate the spikes of each epoch by a uniform offset within it."""
    pieces = []
    for start, stop in epochs:
        spikes = train[(train >= start) & (train < stop)]
        length = stop - start
        offset = generator.uniform(0.0, length)
        pieces.append(start + np.mod(spikes - start + offset, length))
    return np.concatenate(pieces) if pieces else train


def _poisson(
    rate: float, epochs: NDArray[np.float64], generator: np.random.Generator
) -> NDArray[np.float64]:
    """A homogeneous Poisson train at ``rate`` over the epochs."""
    pieces = [
        generator.uniform(start, stop, generator.poisson(rate * (stop - start)))
        for start, stop in epochs
    ]
    return np.concatenate(pieces)


@dataclass(frozen=True)
class SimulatedReactivationSession:
    """A simulated rest - task - rest session.

    Attributes
    ----------
    spike_times : tuple of ndarray
        One sorted spike train per unit, seconds.
    template_epochs, match_epochs, control_epochs : ndarray, shape (1, 2)
        The task (running), rest after it and rest before it.
    assemblies : tuple of tuple of int
        Units of each assembly.
    reactivation, rate_change : float
        The simulation's settings (see :func:`simulated_reactivation_session`).
    control_events, match_events : ndarray, shape (n_events, 2)
        Start and stop of each population burst of the two rest periods,
        overlapping bursts merged into one event: stand-ins for the
        sharp-wave ripples of sleep, for the recipes that count spikes per
        event or follow R(t) around events.
    """

    spike_times: tuple[NDArray[np.float64], ...]
    template_epochs: NDArray[np.float64]
    match_epochs: NDArray[np.float64]
    control_epochs: NDArray[np.float64]
    assemblies: tuple[tuple[int, ...], ...]
    reactivation: float
    rate_change: float
    control_events: NDArray[np.float64]
    match_events: NDArray[np.float64]


def simulated_reactivation_session(
    *,
    n_units: int = 30,
    n_assemblies: int = 3,
    assembly_size: int = 5,
    duration: float = 1200.0,
    rate: float = 2.0,
    event_rate: float = 1.0,
    reactivation: float = 1.0,
    rate_change: float = 0.0,
    burst_rate: float = 0.5,
    burst_gain: float = 5.0,
    rng: np.random.Generator | int | None = None,
) -> SimulatedReactivationSession:
    """Simulate rest, a task and rest, with assemblies that may reactivate.

    Every unit fires as a Poisson process at ``rate``. Each assembly, a
    group of ``assembly_size`` units, also fires together in brief events:
    during the task at ``event_rate`` events per second, during rest after
    it at ``reactivation * event_rate``, and never during rest before it. In
    an event each member fires a Poisson number of spikes (mean 2) within
    about 10 ms. During both rest periods every unit's rate is multiplied by
    ``burst_gain`` in 100 ms population bursts that occur ``burst_rate``
    times per second, like the population events of sleep; they make pair
    correlations in rest grow with firing rates. The bursts are returned as
    ``control_events`` and ``match_events``, stand-ins for sharp-wave
    ripples; assembly events in rest fall in them only by chance, as often
    as the bursts cover the period (5% with the defaults).

    Parameters
    ----------
    n_units : int, default=30
        Number of units.
    n_assemblies, assembly_size : int, default=3 and 5
        Number of assemblies and units per assembly; assemblies do not
        share units and take every ``n_units // (n_assemblies *
        assembly_size)``-th unit, so they span the recording's tetrodes.
    duration : float, default=1200.0
        Seconds of each period: rest before, the task, rest after.
    rate : float, default=2.0
        Background rate of every unit, Hz.
    event_rate : float, default=1.0
        Assembly events per second during the task.
    reactivation : float, default=1.0
        Rate of assembly events during rest after the task, as a fraction of
        ``event_rate``; 0 is a session without reactivation.
    rate_change : float, default=0.0
        Fractional increase of the assembly members' rates during rest after
        the task, without co-firing: 1.0 doubles them. With
        ``reactivation=0`` this is a session without reactivation whose
        members fire more after the task, a confound for every
        correlation-based measure (Tingley & Peyrache 2020).
    burst_rate : float, default=0.5
        Population bursts per second during rest.
    burst_gain : float, default=5.0
        Rate multiplier of every unit during a burst.
    rng : numpy.random.Generator, int, or None
        Seed or generator.

    Returns
    -------
    SimulatedReactivationSession
        Spike times, the three periods' epochs, the assemblies and the
        bursts of the two rest periods as events.

    Raises
    ------
    ValueError
        If the assemblies need more units than ``n_units``, or a rate,
        duration or fraction is negative.

    Examples
    --------
    >>> from neurospatial.reactivation import simulated_reactivation_session
    >>> session = simulated_reactivation_session(reactivation=0.0, rng=0)
    >>> len(session.spike_times), session.assemblies[0]
    (30, (0, 6, 12, 18, 24))
    >>> session.control_epochs.tolist(), session.match_epochs.tolist()
    ([[0.0, 1200.0]], [[2400.0, 3600.0]])
    """
    call = "simulated_reactivation_session()"
    n_members = n_assemblies * assembly_size
    if n_assemblies < 0 or assembly_size < 1 or n_members > n_units:
        raise ValueError(
            _format_error(
                f"{call}: {n_assemblies} assemblies of {assembly_size} units "
                f"need {n_members} units, more than n_units={n_units}.",
                fix="raise n_units or lower n_assemblies or assembly_size",
            )
        )
    for name, value in (
        ("duration", duration),
        ("rate", rate),
        ("event_rate", event_rate),
        ("reactivation", reactivation),
        ("rate_change", rate_change),
        ("burst_rate", burst_rate),
        ("burst_gain", burst_gain),
    ):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"{call}: {name} must be finite and >= 0, got {value!r}.")
    generator = np.random.default_rng(rng)
    control = np.array([[0.0, duration]])
    template = np.array([[duration, 2 * duration]])
    match = np.array([[2 * duration, 3 * duration]])
    spacing = max(1, n_units // max(1, n_members))
    members = np.arange(n_members) * spacing
    assemblies = tuple(
        tuple(int(unit) for unit in members[index::n_assemblies])
        for index in range(n_assemblies)
    )
    member = np.zeros(n_units, dtype=bool)
    member[members] = True
    trains: list[list[NDArray[np.float64]]] = [[] for _ in range(n_units)]
    burst_times: dict[bool, NDArray[np.float64]] = {}
    for start, stop in (control[0], template[0], match[0]):
        rest = start != template[0, 0]
        after = start == match[0, 0]
        gains = np.where(member & after, 1.0 + rate_change, 1.0)
        for unit in range(n_units):
            unit_rate = rate * gains[unit]
            trains[unit].append(
                generator.uniform(start, stop, generator.poisson(unit_rate * duration))
            )
        if rest and burst_rate > 0 and burst_gain > 1:
            bursts = generator.uniform(
                start, stop - 0.1, generator.poisson(burst_rate * duration)
            )
            burst_times[bool(after)] = bursts
            for unit in range(n_units):
                extra = (burst_gain - 1) * rate * gains[unit] * 0.1
                counts = generator.poisson(extra, bursts.size)
                trains[unit].append(
                    np.repeat(bursts, counts) + generator.uniform(0, 0.1, counts.sum())
                )
    for units in assemblies:
        for (start, stop), events_per_second in (
            (template[0], event_rate),
            (match[0], reactivation * event_rate),
        ):
            events = generator.uniform(
                start, stop, generator.poisson(events_per_second * duration)
            )
            for unit in units:
                counts = generator.poisson(2.0, events.size)
                trains[unit].append(
                    np.repeat(events, counts) + generator.normal(0, 0.01, counts.sum())
                )
    total = 3 * duration
    spike_times = tuple(
        np.sort(np.clip(np.concatenate(pieces), 0.0, np.nextafter(total, 0.0)))
        for pieces in trains
    )
    return SimulatedReactivationSession(
        spike_times=spike_times,
        template_epochs=template,
        match_epochs=match,
        control_epochs=control,
        control_events=_merged_events(burst_times.get(False)),
        match_events=_merged_events(burst_times.get(True)),
        assemblies=assemblies,
        reactivation=float(reactivation),
        rate_change=float(rate_change),
    )


def _merged_events(starts: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """100 ms events starting at ``starts``, overlapping ones merged, (n, 2)."""
    if starts is None or not starts.size:
        return np.zeros((0, 2))
    starts = np.sort(starts)
    rows = as_intervals(np.column_stack([starts, starts + 0.1]), name="events")
    assert rows is not None
    return rows


def reactivation_detection_rates(
    table: pd.DataFrame,
    *,
    groups: Sequence[str] = (),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Each method's share of rows called reactivated, with an interval.

    On null data the rate is the method's false-positive rate; on simulated
    reactivation, its sensitivity.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of
        :func:`~neurospatial.reactivation.compare_reactivation_methods`, or
        several concatenated with a column per ``groups`` (such as the
        session or the kind of null data).
    groups : sequence of str, default=()
        Columns to break rates down by.
    confidence : float, default=0.95
        Level of the Clopper-Pearson (exact binomial) interval, as in
        :func:`neurospatial.replay.detection_rates`.

    Returns
    -------
    pandas.DataFrame
        One row per method and group: ``n_rows``; ``n_tested``, the rows
        with a verdict; ``n_reactivated``; ``rate`` and ``ci_low``,
        ``ci_high`` over the tested rows (NaN for a method without a
        within-session test); ``n_errors``; and ``mean_value``, the mean of
        the method's statistic over rows where it is defined, which is the
        quantity to calibrate for methods without a verdict.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``method``, ``value``, ``is_reactivated`` or
        ``error`` columns or a ``groups`` column, or ``confidence`` is not
        in (0, 1).

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.reactivation import reactivation_detection_rates
    >>> table = pd.DataFrame(
    ...     {
    ...         "method": ["a", "a", "a", "b"],
    ...         "value": [0.2, 0.0, 0.1, 0.4],
    ...         "is_reactivated": pd.array([True, False, True, pd.NA], "boolean"),
    ...         "error": ["", "", "", ""],
    ...     }
    ... )
    >>> rates = reactivation_detection_rates(table)
    >>> rates[["method", "n_tested", "n_reactivated"]].to_dict("records")
    [{'method': 'a', 'n_tested': 3, 'n_reactivated': 2}, {'method': 'b', 'n_tested': 0, 'n_reactivated': 0}]
    """
    import pandas as pd
    from scipy.stats import binomtest

    call = "reactivation_detection_rates()"
    if not 0 < confidence < 1:
        raise ValueError(f"{call}: confidence must be in (0, 1), got {confidence}.")
    needed = ["method", "value", "is_reactivated", "error", *groups]
    missing = [name for name in needed if name not in table]
    if missing:
        raise ValueError(
            _format_error(
                f"{call}: table has no columns {missing}.",
                fix="pass the table of compare_reactivation_methods(), with "
                "the groups columns added",
            )
        )
    verdicts = table["is_reactivated"].astype("boolean")
    data = table.assign(
        _tested=verdicts.notna().to_numpy(),
        _called=verdicts.fillna(False).to_numpy(dtype=bool),
        _failed=table["error"].astype(str).ne("").to_numpy(),
    )
    rows = []
    for key, frame in data.groupby(["method", *groups], sort=False, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        n_tested = int(frame["_tested"].sum())
        n_called = int(frame["_called"].sum())
        if n_tested:
            interval = binomtest(n_called, n_tested).proportion_ci(
                confidence_level=confidence, method="exact"
            )
            rate, low, high = n_called / n_tested, interval.low, interval.high
        else:
            rate = low = high = np.nan
        values = frame["value"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                **dict(zip(["method", *groups], keys, strict=True)),
                "n_rows": len(frame),
                "n_tested": n_tested,
                "n_reactivated": n_called,
                "rate": rate,
                "ci_low": low,
                "ci_high": high,
                "n_errors": int(frame["_failed"].sum()),
                "mean_value": float(finite.mean()) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)
