"""Calibrate and compare replay methods on tables of verdicts.

:func:`~neurospatial.replay.compare_replay_methods` returns one row per
(event, method). The functions here summarize such a table:

- :func:`event_properties` measures each event (active cells, spikes,
  duration, sharpness, coverage), so rates can be broken down by them.
- :func:`detection_rates` gives each method's share of events called
  replay, with a confidence interval, overall or per stratum of event
  properties. On events without replay (null data) this is the method's
  false-positive rate; on simulated replays, its sensitivity.
- :func:`method_agreement` gives Cohen's kappa between every pair of
  methods.

A single overall false-positive rate hides how it depends on the event:
short events and events with few cells pass by chance more often
(Tingley & Peyrache 2020), so report rates per stratum, and on several
kinds of null data rather than one.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult, JointDecodingResult
    from neurospatial.environment import Environment

__all__ = ["detection_rates", "event_properties", "method_agreement"]


def event_properties(
    env: Environment,
    events: Sequence[DecodingResult | JointDecodingResult],
    *,
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
) -> pd.DataFrame:
    """Measure each decoded event, to break detection rates down by them.

    Parameters
    ----------
    env : Environment
        The environment the events were decoded on.
    events : sequence of DecodingResult or JointDecodingResult
        The decoded events, each with ``times``. A joint decode is measured
        on its posterior summed over templates, which must share ``env``'s
        bins.
    spike_counts : sequence of arrays or None
        Each event's spike counts, ``(n_time_bins, n_cells)``.

    Returns
    -------
    pandas.DataFrame
        One row per event, indexed by ``event``: ``n_time_bins``,
        ``duration`` (last minus first time-bin center, seconds),
        ``mean_peak_probability`` (mean over time bins of the largest
        posterior mass, a measure of sharpness), ``coverage`` (range of the
        MAP position over the track length; NaN off a track or on a
        branched track) and, with ``spike_counts``, ``n_active_cells`` and
        ``n_spikes``.

    Raises
    ------
    ValueError
        If an event has no ``times``, if ``spike_counts`` does not have one
        entry per event with one row per time bin, or if a joint decode's
        templates are not on ``env``'s bins.

    See Also
    --------
    detection_rates : Rates per stratum of these properties.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import DecodingResult
    >>> from neurospatial.replay import event_properties
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> event = DecodingResult(
    ...     posterior=np.eye(10)[[1, 2, 3, 4]], env=env, times=np.arange(4) * 0.02
    ... )
    >>> event_properties(env, [event]).round(3).to_dict("records")
    [{'n_time_bins': 4, 'duration': 0.06, 'mean_peak_probability': 1.0, 'coverage': 0.3}]
    """
    import pandas as pd

    from neurospatial.decoding import JointDecodingResult
    from neurospatial.replay.criteria import trajectory_metrics

    operation = "event_properties()"
    if spike_counts is not None and len(spike_counts) != len(events):
        raise ValueError(
            f"{operation}: spike_counts has {len(spike_counts)} entries but "
            f"there are {len(events)} events."
        )
    rows = []
    for index, event in enumerate(events):
        if isinstance(event, JointDecodingResult):
            shares = [event.joint(name) for name in event.template_names]
            if any(share.shape[1] != env.n_bins for share in shares):
                raise ValueError(
                    _format_error(
                        f"{operation}: event {index} is a joint decode whose "
                        f"templates are not on env's {env.n_bins} bins.",
                        why="its posterior is measured summed over templates",
                        fix="measure each template's share as a DecodingResult",
                    )
                )
            posterior = np.sum(shares, axis=0)
            times = event.conditional[event.template_names[0]].times
        else:
            posterior, times = event.posterior, event.times
        if times is None:
            raise ValueError(
                f"{operation}: event {index} has no times; decode with times=."
            )
        counts = None
        if spike_counts is not None:
            counts = np.asarray(spike_counts[index])
            if counts.ndim == 1:
                counts = counts[:, np.newaxis]
            if len(counts) != len(posterior):
                raise ValueError(
                    f"{operation}: spike_counts[{index}] has {len(counts)} rows "
                    f"but event {index} has {len(posterior)} time bins."
                )
        metrics = trajectory_metrics(env, posterior, np.asarray(times))
        coverage = (
            metrics.map_range / metrics.track_length
            if np.isfinite(metrics.track_length)
            else np.nan
        )
        row: dict[str, Any] = {
            "event": index,
            "n_time_bins": len(posterior),
            "duration": float(metrics.duration),
            "mean_peak_probability": float(metrics.mean_peak_probability),
            "coverage": float(coverage),
        }
        if counts is not None:
            row["n_active_cells"] = int(np.sum(counts.sum(axis=0) > 0))
            row["n_spikes"] = int(counts.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def detection_rates(
    table: pd.DataFrame,
    *,
    properties: pd.DataFrame | None = None,
    by: Mapping[str, Sequence[float]] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Each method's share of events called replay, with a confidence interval.

    On events without replay (null data) the rate is the method's
    false-positive rate; on simulated replays, its sensitivity.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`.
        Rows with an ``error`` are counted in ``n_errors`` and left out of
        the rate.
    properties : pandas.DataFrame or None
        Event properties indexed by event, such as from
        :func:`event_properties`, for ``by``.
    by : Mapping[str, Sequence[float]] or None
        Break the rates down by event properties: each key is a column of
        ``properties`` and each value its bin edges, such as
        ``{"n_active_cells": [5, 10, 20, np.inf]}``. Bins are closed on the
        left; events outside every bin are left out.
    confidence : float, default=0.95
        Level of the Clopper-Pearson (exact binomial) interval.

    Returns
    -------
    pandas.DataFrame
        One row per method (and stratum): ``method``, one column per key of
        ``by`` (the bin, as an interval), ``n_events``, ``n_replay``,
        ``rate``, ``ci_low``, ``ci_high`` and ``n_errors``. ``rate`` and the
        interval are NaN for a stratum without events.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``event``, ``method`` or ``is_replay``
        columns, if ``by`` is given without ``properties`` or names a column
        ``properties`` lacks, or if ``confidence`` is not in (0, 1).

    See Also
    --------
    event_properties : Properties to break rates down by.
    method_agreement : How often two methods agree.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.replay import detection_rates
    >>> table = pd.DataFrame(
    ...     {
    ...         "event": [0, 0, 1, 1, 2, 2],
    ...         "method": ["a", "b"] * 3,
    ...         "is_replay": [True, False, False, False, True, True],
    ...         "error": [""] * 6,
    ...     }
    ... )
    >>> rates = detection_rates(table)
    >>> rates[["method", "n_events", "n_replay"]].to_dict("records")
    [{'method': 'a', 'n_events': 3, 'n_replay': 2}, {'method': 'b', 'n_events': 3, 'n_replay': 1}]
    """
    import pandas as pd
    from scipy.stats import binomtest

    operation = "detection_rates()"
    missing = [c for c in ("event", "method", "is_replay") if c not in table]
    if missing:
        raise ValueError(
            f"{operation}: table lacks columns {missing}; pass the table of "
            "compare_replay_methods()."
        )
    if not 0 < confidence < 1:
        raise ValueError(
            f"{operation}: confidence must be in (0, 1), got {confidence}."
        )
    data = table.copy()
    errors = (
        data["error"].fillna("").astype(str) != ""
        if "error" in data
        else pd.Series(False, index=data.index)
    )
    keys = ["method"]
    if by:
        if properties is None:
            raise ValueError(
                _format_error(
                    f"{operation}: by= needs properties=.",
                    why="strata are bins of event properties",
                    fix="pass properties=event_properties(env, events, ...)",
                )
            )
        absent = [name for name in by if name not in properties]
        if absent:
            raise ValueError(
                f"{operation}: properties lacks columns {absent}; it has "
                f"{list(properties.columns)}."
            )
        for name, edges in by.items():
            values = data["event"].map(properties[name])
            data[name] = pd.cut(values, bins=list(edges), right=False)
            keys.append(name)
    data["_error"] = errors
    # Every method and bin, including strata without events.
    levels = [list(dict.fromkeys(data["method"]))] + [
        list(data[name].cat.categories) for name in keys[1:]
    ]
    groups = dict(iter(data.groupby(keys, observed=True, sort=False)))
    rows = []
    for group in itertools.product(*levels):
        frame = groups.get(group, data.iloc[:0])
        kept = frame[~frame["_error"]]
        n_events = len(kept)
        n_replay = int(kept["is_replay"].astype(bool).sum())
        if n_events:
            interval = binomtest(n_replay, n_events).proportion_ci(
                confidence_level=confidence, method="exact"
            )
            rate, low, high = n_replay / n_events, interval.low, interval.high
        else:
            rate = low = high = np.nan
        row = dict(zip(keys, group, strict=True))
        row.update(
            n_events=n_events,
            n_replay=n_replay,
            rate=rate,
            ci_low=low,
            ci_high=high,
            n_errors=int(frame["_error"].sum()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def method_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa between the verdicts of every pair of methods.

    Kappa is the agreement beyond chance: 1 when two methods call the same
    events replay, 0 when they agree only as often as their rates predict.
    Each pair uses the events both methods scored without an ``error``.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`.

    Returns
    -------
    pandas.DataFrame
        Square, indexed and columned by method; NaN when a pair shares no
        events or both methods give one verdict to every event.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``event``, ``method`` or ``is_replay``
        columns.

    See Also
    --------
    detection_rates : Each method's rate.

    Notes
    -----
    Tingley & Peyrache (2020) found little agreement between rank-order
    and weighted-correlation verdicts on the same events.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.replay import method_agreement
    >>> table = pd.DataFrame(
    ...     {
    ...         "event": [0, 0, 1, 1, 2, 2, 3, 3],
    ...         "method": ["a", "b"] * 4,
    ...         "is_replay": [True, True, False, False, True, False, False, False],
    ...     }
    ... )
    >>> method_agreement(table).round(2).to_dict()
    {'a': {'a': 1.0, 'b': 0.5}, 'b': {'a': 0.5, 'b': 1.0}}
    """
    import pandas as pd

    missing = [c for c in ("event", "method", "is_replay") if c not in table]
    if missing:
        raise ValueError(
            f"method_agreement(): table lacks columns {missing}; pass the table "
            "of compare_replay_methods()."
        )
    kept = table
    if "error" in table:
        kept = table[table["error"].fillna("").astype(str) == ""]
    verdicts = kept.pivot(index="event", columns="method", values="is_replay")
    methods = list(dict.fromkeys(table["method"]))
    kappa = pd.DataFrame(np.nan, index=methods, columns=methods)
    for first in methods:
        for second in methods:
            if first not in verdicts or second not in verdicts:
                continue
            pair = (
                verdicts[[first, second]].dropna()
                if first != second
                else (verdicts[[first]].dropna())
            )
            if pair.empty:
                continue
            a = pair[first].astype(bool).to_numpy()
            b = pair[second].astype(bool).to_numpy()
            observed = float(np.mean(a == b))
            chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
            kappa.loc[first, second] = (
                (observed - chance) / (1 - chance) if chance < 1 else np.nan
            )
    return kappa
