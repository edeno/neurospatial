"""Calibrate and compare replay methods on tables of verdicts.

:func:`~neurospatial.replay.compare_replay_methods` returns one row per
(event, method). The functions here summarize such a table:

- :func:`event_properties` measures each event (cells, spikes, duration,
  sharpness, spread, path), so rates can be broken down by them.
- :func:`detection_rates` gives each method's share of events called
  replay, with a confidence interval, overall or per stratum of event
  properties and of other table columns (such as the kind of null data
  or the session). On events without replay (null data) this is the
  method's false-positive rate; on simulated replays, its sensitivity.
- :func:`method_agreement` counts how often every pair of methods agree,
  with Cohen's kappa.
- :func:`matched_alpha` finds each method's alpha that holds its
  false-positive rate on null data to a target (Takigawa et al. 2024), and
  :func:`apply_alpha` gives a table's verdicts at those alphas.

Null data for false-positive rates: :func:`null_events` makes events
without replay from real candidate events (cell identities reassigned,
spikes shifted, or Poisson spikes), :func:`simulated_null_events` simulates
random or stationary events, and :func:`decode_intervals` decodes either.

A single overall false-positive rate hides how it depends on the event:
Tingley & Peyrache (2020) found the chance rate of rank-order replay
depends strongly on the number of cells and their firing rates. Report
rates per stratum, and on several kinds of null data rather than one.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult, JointDecodingResult
    from neurospatial.environment import Environment
    from neurospatial.replay.arms import ArmReplayMethod
    from neurospatial.replay.methods import RankOrderMethod, ReplayMethod

__all__ = [
    "NullSpikeData",
    "apply_alpha",
    "decode_intervals",
    "detection_rates",
    "event_properties",
    "matched_alpha",
    "method_agreement",
    "null_events",
    "simulated_null_events",
]

_ERRORS = ("not_replay", "exclude")


def event_properties(
    env: Environment | Mapping[str, Environment],
    events: Sequence[DecodingResult | JointDecodingResult],
    *,
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    template: str | None = None,
    dt: float | None = None,
) -> pd.DataFrame:
    """Measure each decoded event, to break detection rates down by them.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the events were decoded on, or the mapping of
        environments of a joint decode over separate tracks.
    events : sequence of DecodingResult or JointDecodingResult
        The decoded events, each with ``times``. A joint decode is measured
        on its posterior summed over templates, which must all be on
        ``env``; pass ``template=`` to measure one template's share.
    spike_counts : sequence of arrays or None
        Each event's spike counts, ``(n_time_bins, n_cells)``.
    template : str or None
        For joint decodes, the template whose share is measured; needed
        when the templates are on separate tracks.
    dt : float or None
        Width of each decoding window, in seconds, for ``window_duration``.

    Returns
    -------
    pandas.DataFrame
        One row per event, indexed by ``event``:

        - ``n_time_bins``.
        - ``duration``: last minus first time-bin center, in seconds;
          with ``dt``, ``window_duration`` adds one window, the span
          :func:`~neurospatial.replay.compare_replay_methods` scores
          rank-order methods over.
        - ``mean_peak_probability``: mean over time bins of the largest
          posterior mass. It grows with the bin size; compare it only
          across events decoded on the same bins.
        - ``mean_spread``: mean posterior spread (square root of the
          second central moment), in environment units, which does not
          depend on the bin size.
        - ``map_path_length``: summed distance between consecutive MAP
          positions, in environment units.
        - ``map_range_fraction``: range of the MAP position along the
          track over the track's length; at most ``1 - bin_size / length``
          for a sweep from end to end; NaN off a track and on a branched
          track.
        - With ``spike_counts``: ``n_active_cells`` (columns with a
          spike), ``n_spikes``, ``spikes_per_active_cell`` and
          ``empty_fraction`` (time bins without spikes).

    Raises
    ------
    ValueError
        If ``events`` is empty or an event has no ``times``; if
        ``spike_counts`` does not have one entry per event of shape
        (n_time_bins, n_cells), or has negative or non-integer counts; or if a joint
        decode's templates are on different environments without
        ``template=``, or ``template`` is not one of its templates.

    See Also
    --------
    detection_rates : Rates per stratum of these properties.
    neurospatial.replay.trajectory_metrics : The per-event metrics.

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
    >>> row = event_properties(env, [event], dt=0.02).iloc[0]
    >>> print(f"{row['duration']:.2f} s, {row['window_duration']:.2f} s with windows")
    0.06 s, 0.08 s with windows
    >>> print(
    ...     f"{row['map_range_fraction']:.1f} of the track, {row['map_path_length']:.0f} cm"
    ... )
    0.3 of the track, 30 cm
    """
    import pandas as pd

    from neurospatial.replay.criteria import trajectory_metrics

    operation = "event_properties()"
    if len(events) == 0:
        raise ValueError(f"{operation}: events is empty; pass at least one event.")
    if spike_counts is not None and len(spike_counts) != len(events):
        raise ValueError(
            f"{operation}: spike_counts has {len(spike_counts)} entries but "
            f"there are {len(events)} events."
        )
    rows = []
    for index, event in enumerate(events):
        event_env, posterior, times = _measured(env, event, template, index, operation)
        if times is None:
            raise ValueError(
                f"{operation}: event {index} has no times; decode with times=."
            )
        counts = None
        if spike_counts is not None:
            counts = np.asarray(spike_counts[index])
            if counts.ndim != 2:
                raise ValueError(
                    _format_error(
                        f"{operation}: spike_counts[{index}] has shape "
                        f"{counts.shape}, but the active cells are counted per "
                        "cell.",
                        why="spike counts summed over cells cannot count active "
                        "cells: every spike would count as one cell",
                        fix="pass (n_time_bins, n_cells) counts, one column per cell",
                    )
                )
            if len(counts) != len(posterior):
                raise ValueError(
                    f"{operation}: spike_counts[{index}] has {len(counts)} rows "
                    f"but event {index} has {len(posterior)} time bins."
                )
            if np.any(counts < 0) or not np.all(np.equal(np.mod(counts, 1), 0)):
                raise ValueError(
                    f"{operation}: spike_counts[{index}] has negative or "
                    "non-integer counts; pass spike counts per time bin and cell."
                )
        metrics = trajectory_metrics(
            event_env, posterior, np.asarray(times), spike_counts=counts
        )
        row: dict[str, Any] = {
            "event": index,
            "n_time_bins": len(posterior),
            "duration": float(metrics.duration),
            "mean_peak_probability": float(metrics.mean_peak_probability),
            "mean_spread": float(np.mean(metrics.spread)),
            "map_path_length": float(metrics.map_path_length),
            "map_range_fraction": float(metrics.map_range / metrics.track_length)
            if np.isfinite(metrics.track_length)
            else np.nan,
        }
        if dt is not None:
            row["window_duration"] = float(metrics.duration) + float(dt)
        if counts is not None:
            active = int(np.sum(counts.sum(axis=0) > 0))
            row["n_active_cells"] = active
            row["n_spikes"] = int(counts.sum())
            row["spikes_per_active_cell"] = (
                row["n_spikes"] / active if active else np.nan
            )
            row["empty_fraction"] = float(metrics.empty_fraction)
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def _measured(
    env: Environment | Mapping[str, Environment],
    event: Any,
    template: str | None,
    index: int,
    operation: str,
) -> tuple[Environment, NDArray[np.float64], Any]:
    """The environment, posterior and times an event is measured on."""
    from neurospatial.decoding import JointDecodingResult

    if not isinstance(event, JointDecodingResult):
        if isinstance(env, Mapping):
            if template is None:
                raise ValueError(
                    f"{operation}: a mapping of environments needs template=."
                )
            return env[template], event.posterior, event.times
        return env, event.posterior, event.times
    names = event.template_names
    if template is not None:
        if template not in names:
            raise ValueError(
                f"{operation}: template {template!r} is not one of {list(names)} "
                f"in event {index}."
            )
        chosen = env[template] if isinstance(env, Mapping) else env
        return chosen, event.joint(template), event.conditional[template].times
    envs = [event.conditional[name].env for name in names]
    first = envs[0]
    shared = all(
        other is first
        or (
            other.n_bins == first.n_bins
            and np.allclose(other.bin_centers, first.bin_centers)
        )
        for other in envs[1:]
    )
    if not shared or isinstance(env, Mapping):
        raise ValueError(
            _format_error(
                f"{operation}: event {index} is a joint decode over separate "
                "environments.",
                why="its templates' posteriors cannot be summed bin by bin",
                fix=f"pass template= one of {list(names)} to measure that share",
            )
        )
    posterior = np.sum([event.joint(name) for name in names], axis=0)
    return env, posterior, event.conditional[names[0]].times


def detection_rates(
    table: pd.DataFrame,
    *,
    properties: pd.DataFrame | None = None,
    by: Mapping[str, Sequence[float]] | None = None,
    groups: Sequence[str] = (),
    errors: Literal["not_replay", "exclude"] = "not_replay",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Each method's share of events called replay, with a confidence interval.

    On events without replay (null data) the rate is the method's
    false-positive rate; on simulated replays, its sensitivity.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`,
        or several such tables concatenated with a column per ``groups``
        (such as the kind of null data or the session) that tells their
        events apart.
    properties : pandas.DataFrame or None
        Event properties indexed by event, such as from
        :func:`event_properties`, for ``by`` keys that are not table
        columns. Joined on ``event``, so use it with one call's table.
    by : Mapping[str, Sequence[float]] or None
        Break rates down by numeric properties: each key a column of
        ``table`` or ``properties``, each value at least two bin edges,
        such as ``{"n_active_cells": [5, 10, 20, np.inf]}``. Bins are
        closed on the left. Every event must fall in a bin.
    groups : sequence of str, default=()
        Columns of ``table`` to break rates down by their values, such as
        ``("null_kind",)``.
    errors : {"not_replay", "exclude"}, default="not_replay"
        Events a method could not score (rows with an ``error``):
        ``"not_replay"`` counts them as not replay, so every method's rate
        is over the same events (a rank-order method with too few active
        cells gives an error row where a decoded method with a cell
        criterion gives a plain non-replay); ``"exclude"`` leaves them out
        of ``n_events``, giving the rate among the events the method
        scored. ``n_errors`` counts them either way.
    confidence : float, default=0.95
        Level of the Clopper-Pearson (exact binomial) interval.

    Returns
    -------
    pandas.DataFrame
        One row per method and stratum, in every combination: ``method``,
        one column per ``groups`` and ``by`` key (the bin, as an interval),
        ``n_events``, ``n_replay``, ``rate``, ``ci_low``, ``ci_high`` and
        ``n_errors``. ``rate`` and the interval are NaN without events.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``event``, ``method`` or ``is_replay``
        columns, has a non-boolean ``is_replay`` or repeats an (event,
        method) pair within its ``groups``; if a ``by`` key is in neither
        ``table`` nor ``properties``, has fewer than two edges, or leaves
        events outside its bins (NaN, out of range, or absent from
        ``properties``); or if ``errors`` or ``confidence`` is invalid.

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
    if errors not in _ERRORS:
        raise ValueError(
            f"{operation}: errors must be one of {_ERRORS}, got {errors!r}."
        )
    if not 0 < confidence < 1:
        raise ValueError(
            f"{operation}: confidence must be in (0, 1), got {confidence}."
        )
    data = _checked_table(table, groups, operation)
    keys = ["method", *groups]
    for name, edges in (by or {}).items():
        if len(edges) < 2:
            raise ValueError(
                f"{operation}: by[{name!r}] needs at least two bin edges, got "
                f"{list(edges)}."
            )
        if name in data:
            values = data[name]
        elif properties is not None and name in properties:
            values = data["event"].map(properties[name])
        else:
            raise ValueError(
                _format_error(
                    f"{operation}: {name!r} is neither a column of table nor of "
                    "properties.",
                    why="strata are bins of event properties",
                    fix="pass properties=event_properties(env, events, ...) or "
                    "add the column to the table",
                )
            )
        binned = pd.cut(values, bins=list(edges), right=False)
        outside = data.loc[binned.isna(), "event"].unique()
        if outside.size:
            raise ValueError(
                _format_error(
                    f"{operation}: events {sorted(outside.tolist())[:10]} have no "
                    f"bin for {name!r} (NaN, outside {list(edges)}, or missing "
                    "from properties).",
                    why="they would silently drop out of every stratum",
                    fix="widen the edges (e.g. end with np.inf), or leave those "
                    "events out of the table first",
                )
            )
        data[name] = binned
        keys.append(name)
    levels = [list(dict.fromkeys(data[key])) for key in ["method", *groups]] + [
        list(data[name].cat.categories) for name in (by or {})
    ]
    frames = dict(iter(data.groupby(keys, observed=True, sort=False)))
    rows = []
    for group in itertools.product(*levels):
        frame = frames.get(group, data.iloc[:0])
        failed = frame["_error"]
        calls = frame["is_replay"] & ~failed
        n_events = int(len(frame) if errors == "not_replay" else (~failed).sum())
        n_replay = int(calls.sum())
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
            n_errors=int(failed.sum()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def method_agreement(
    table: pd.DataFrame,
    *,
    groups: Sequence[str] = (),
    errors: Literal["not_replay", "exclude"] = "not_replay",
) -> pd.DataFrame:
    """How often every pair of methods agree, with Cohen's kappa.

    Kappa is the agreement beyond chance: 1 when two methods call the same
    events replay, 0 when they agree only as often as their rates predict.
    On null data with low rates, kappa is small and uncertain even when
    raw agreement is high, so read it with the counts.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`,
        or several concatenated with ``groups`` columns that tell their
        events apart.
    groups : sequence of str, default=()
        Columns of ``table`` that, with ``event``, identify an event.
    errors : {"not_replay", "exclude"}, default="not_replay"
        Events a method could not score count as not replay, or leave the
        pair's events (see :func:`detection_rates`).

    Returns
    -------
    pandas.DataFrame
        One row per pair of methods, in the table's method order:
        ``first``, ``second``, ``n_events`` (scored by both), ``n_both``,
        ``n_first_only``, ``n_second_only``, ``n_neither``, ``agreement``
        (share of events with the same verdict) and ``kappa``. ``kappa`` is
        NaN when both methods give the same verdict to every event, or
        without shared events.

    Raises
    ------
    ValueError
        As :func:`detection_rates` for the table, or for ``errors``.

    See Also
    --------
    detection_rates : Each method's rate.

    Notes
    -----
    Tingley & Peyrache (2020) found that rank-order and weighted-correlation
    scores of the same events correlate weakly (R = 0.13), and weighted
    correlation and line-fit scores moderately (R = 0.43).

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
    >>> row = method_agreement(table).iloc[0]
    >>> print(row["n_both"], row["n_first_only"], row["agreement"], row["kappa"])
    1 1 0.75 0.5
    """
    import pandas as pd

    operation = "method_agreement()"
    if errors not in _ERRORS:
        raise ValueError(
            f"{operation}: errors must be one of {_ERRORS}, got {errors!r}."
        )
    data = _checked_table(table, groups, operation)
    if errors == "exclude":
        data = data[~data["_error"]]
    else:
        data = data.assign(is_replay=data["is_replay"] & ~data["_error"])
    index = [*groups, "event"]
    verdicts = data.pivot(index=index, columns="method", values="is_replay")
    methods = list(dict.fromkeys(table["method"]))
    rows = []
    for first, second in itertools.combinations(methods, 2):
        if first in verdicts and second in verdicts:
            pair = verdicts[[first, second]].dropna()
        else:
            pair = pd.DataFrame(columns=[first, second])
        a = pair[first].astype(bool).to_numpy()
        b = pair[second].astype(bool).to_numpy()
        n = len(a)
        agreement = float(np.mean(a == b)) if n else np.nan
        chance = (
            float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
            if n
            else np.nan
        )
        kappa = (agreement - chance) / (1 - chance) if n and chance < 1 else np.nan
        rows.append(
            {
                "first": first,
                "second": second,
                "n_events": n,
                "n_both": int(np.sum(a & b)),
                "n_first_only": int(np.sum(a & ~b)),
                "n_second_only": int(np.sum(~a & b)),
                "n_neither": int(np.sum(~a & ~b)),
                "agreement": agreement,
                "kappa": kappa,
            }
        )
    return pd.DataFrame(rows)


def matched_alpha(
    null_table: pd.DataFrame,
    *,
    replay_methods: Sequence[ReplayMethod | RankOrderMethod | ArmReplayMethod],
    target: float = 0.05,
    groups: Sequence[str] = (),
) -> pd.DataFrame:
    """Each method's alpha that holds its false-positive rate to a target.

    Methods with the same nominal alpha call different shares of events
    without replay significant. Takigawa et al. (2024) compared methods at
    the alpha that gives each the same false-positive rate on null data.
    This finds, for each method, the largest alpha at which at most
    ``target`` of the null events are called replay, from the p-values
    and criteria already in the table: the shuffles are not run again.

    Parameters
    ----------
    null_table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`
        on events without replay, such as from :func:`null_events` or
        :func:`simulated_null_events`, or several concatenated with
        ``groups`` columns that tell their events apart.
    replay_methods : sequence of ReplayMethod or RankOrderMethod
        The methods the table was made with. Each must test a p-value
        against its ``alpha``: a :class:`ReplayMethod` with ``nulls``, or
        a :class:`RankOrderMethod` with the ``"t_distribution"`` or
        ``"cell_order"`` test that tests one template.
    target : float, default=0.05
        The false-positive rate to hold to, in (0, 1).
    groups : sequence of str, default=()
        Columns of ``null_table`` to match alpha in separately, such as
        ``("null_kind",)`` or ``("session",)``.

    Returns
    -------
    pandas.DataFrame
        One row per method (and group): ``method``, the ``groups``
        columns, ``n_events``, ``published_alpha`` and
        ``published_rate`` (the method's own alpha and its rate on these
        events), ``alpha`` and ``rate`` (the matched alpha and the rate at
        it, never above ``target``). ``alpha`` is 1 when the criteria
        alone hold the rate to ``target``, and NaN (with ``rate``) when no
        alpha does. Events a method could not score count as not replay,
        as in :func:`detection_rates`. A missing ``groups`` value forms its
        own group.

    Raises
    ------
    ValueError
        If ``target`` is not in (0, 1); if the table lacks the ``event``,
        ``method``, ``is_replay``, ``p_value`` or a method's ``passes[...]``
        columns, or repeats an (event, method) pair; if a method in the
        table is not in ``replay_methods``, does not test a p-value
        against alpha, or is a rank-order method that chose among several
        templates; or if the table's verdicts do not follow from its
        p-values and criteria at the method's own alpha (it was made with
        other settings).

    Warns
    -----
    UserWarning
        If no alpha holds the rate: more than ``target`` of the null events
        pass with p = 0, as the t-distribution test gives when the
        correlation is 1 (alpha is then NaN). If the matched alpha is at
        the smallest p-value the shuffles can give, so real events cannot
        reach it either: run more shuffles. If there are too few null
        events for ``target`` to allow any to pass.

    See Also
    --------
    apply_alpha : Verdicts of a table at the matched alphas.
    null_events, simulated_null_events : Events without replay.
    detection_rates : Rates at each method's own alpha.

    Notes
    -----
    Takigawa et al. (2024) reassigned cell identities in copies of each
    session's events. For each method they chose the p-value threshold, on
    a log-spaced grid from 0.001 to 0.2, whose false-positive rate (a mean
    over bootstrap resamples of the pooled events) was closest to 0.05, so
    it could lie above it. Their rate counts each track's verdicts and
    averages the two, tests half events at alpha / 2, and calls an event
    significant at p <= alpha. Here each event gets one verdict across
    templates, so on two-track data this rate can be up to about twice
    theirs; alpha is exact, the rate never exceeds ``target``, and, as in
    each method, p < alpha.

    Matching holds the false-positive rate on one kind of null data.
    Each kind keeps some structure of real events and breaks the rest, so
    rates on other kinds still differ (see :func:`null_events`); match on
    several kinds with ``groups`` and report each. The matched alpha is
    chosen on these null events, so the rate on new null data is likely a
    little higher, and with few null events it is poorly determined: with
    ``target * n_events`` below about 10 it rests on a handful of events.
    A method that is conservative at its own alpha gets a larger matched
    alpha.

    Methods without a shuffle test, arm methods and the pooled-SSI test
    have no p-value to threshold; leave them out of the table. A
    rank-order method with ``template_choice`` picks the most active
    template among those that are replay at its alpha, so its rows change
    with alpha; rerun it at each alpha instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.replay import matched_alpha
    >>> from neurospatial.replay.methods import KARLSSON_FRANK_2009
    >>> p_values = [0.002, 0.01, 0.03, 0.04] + [0.5] * 16
    >>> null_table = pd.DataFrame(
    ...     {
    ...         "event": range(20),
    ...         "method": "karlsson_frank_2009",
    ...         "is_replay": [p < 0.05 for p in p_values],
    ...         "p_value": p_values,
    ...         "passes[n_active_cells >= 5]": True,
    ...     }
    ... )
    >>> row = matched_alpha(null_table, replay_methods=[KARLSSON_FRANK_2009]).iloc[0]
    >>> print(row["published_rate"], row["alpha"], row["rate"])
    0.2 0.01 0.05
    """
    import pandas as pd

    operation = "matched_alpha()"
    if not 0 < target < 1:
        raise ValueError(f"{operation}: target must be in (0, 1), got {target!r}.")
    data = _checked_table(null_table, groups, operation)
    methods = _alpha_methods(
        data, replay_methods, list(dict.fromkeys(data["method"])), operation
    )
    rows = []
    for key, frame in data.groupby(["method", *groups], sort=False, dropna=False):
        method = methods[key[0]]
        criteria_met = _verdicts(frame, method, np.inf, operation)
        _check_verdicts(frame, method, operation)
        n_events = len(frame)
        p_values = np.sort(frame["p_value"].to_numpy(dtype=float)[criteria_met])
        p_values = p_values[np.isfinite(p_values)]
        # Most null events allowed at the target, with a tolerance for
        # round-off in target * n_events.
        allowed = int(np.floor(target * n_events + 1e-9))
        alpha = float(p_values[allowed]) if p_values.size > allowed else 1.0
        problem = None
        if alpha == 0.0:
            alpha = np.nan
            problem = (
                f"{int(np.sum(p_values == 0))} of {n_events} null events pass "
                "with p = 0 (the t-distribution test gives p = 0 when the "
                f"correlation is 1), more than {target:g}, so no alpha holds "
                "the rate; require more active cells or use the cell_order test"
            )
        elif alpha <= _smallest_p_value(method) * (1 + 1e-9):
            problem = (
                f"the matched alpha, {alpha:g}, is the smallest p-value its "
                f"{method.n_shuffles} shuffles can give, so real events cannot "
                "reach it either; run more shuffles (n_shuffles)"
            )
        elif allowed == 0 and p_values.size:
            problem = (
                f"{n_events} null events are too few for a rate of {target:g} "
                "to allow any to pass, so the matched alpha is the smallest "
                "null p-value; use more null events (n_copies)"
            )
        if problem is not None:
            warnings.warn(
                f"{operation}: {method.name!r}: {problem}.", UserWarning, stacklevel=2
            )
        rows.append(
            {
                **dict(zip(["method", *groups], key, strict=True)),
                "n_events": n_events,
                "published_alpha": method.alpha,
                "published_rate": float(frame["is_replay"].sum()) / n_events,
                "alpha": alpha,
                "rate": float(np.sum(p_values < alpha)) / n_events
                if np.isfinite(alpha)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def apply_alpha(
    table: pd.DataFrame,
    *,
    alpha: Mapping[str, float] | pd.Series,
    replay_methods: Sequence[ReplayMethod | RankOrderMethod | ArmReplayMethod],
    groups: Sequence[str] = (),
) -> pd.DataFrame:
    """A comparison table's verdicts at other alphas, such as matched ones.

    Each event is called replay at the new alpha when it passes the
    method's criteria and its p-value is below the alpha. The p-values and
    criteria come from the table, so the shuffles are not run again; this
    gives the verdicts :func:`~neurospatial.replay.compare_replay_methods`
    would give with ``dataclasses.replace(method, alpha=...)`` and the
    same ``rng`` (methods take alpha below 1).

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`,
        such as on real candidate events.
    alpha : Mapping[str, float] or pandas.Series
        The new alpha in (0, 1] of each method by name, such as
        ``matched_alpha(...).set_index("method")["alpha"]``. Methods not
        named keep their verdicts. After ``matched_alpha`` with
        ``groups``, choose one group's alphas first.
    replay_methods : sequence of ReplayMethod or RankOrderMethod
        The methods the table was made with; those named in ``alpha`` must
        test a p-value against alpha (see :func:`matched_alpha`).
    groups : sequence of str, default=()
        Columns of ``table`` that, with ``event``, identify an event, for
        several concatenated tables.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` with ``is_replay`` at the new alphas and an
        ``alpha`` column: the new alpha, or the method's own (which a
        method without nulls does not use; NaN for an arm method, which
        has none).

    Raises
    ------
    ValueError
        If an alpha is not in (0, 1], a method is named twice, or a name
        is not a method in the table; otherwise as :func:`matched_alpha`.

    See Also
    --------
    matched_alpha : The alpha that holds each method's false-positive rate.
    detection_rates, method_agreement : Summaries of the new table.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.replay import apply_alpha
    >>> from neurospatial.replay.methods import KARLSSON_FRANK_2009
    >>> table = pd.DataFrame(
    ...     {
    ...         "event": [0, 1, 2],
    ...         "method": "karlsson_frank_2009",
    ...         "is_replay": [True, True, False],
    ...         "p_value": [0.002, 0.03, 0.03],
    ...         "passes[n_active_cells >= 5]": [True, True, False],
    ...     }
    ... )
    >>> matched = apply_alpha(
    ...     table,
    ...     alpha={"karlsson_frank_2009": 0.01},
    ...     replay_methods=[KARLSSON_FRANK_2009],
    ... )
    >>> print(matched["is_replay"].tolist())
    [True, False, False]
    """
    import pandas as pd

    operation = "apply_alpha()"
    if isinstance(alpha, pd.Series) and alpha.index.has_duplicates:
        repeated = sorted(set(alpha.index[alpha.index.duplicated()]))
        raise ValueError(
            _format_error(
                f"{operation}: alpha names methods {repeated} more than once.",
                why="matched_alpha with groups gives one alpha per group, and "
                "a table's verdicts take one alpha per method",
                fix="choose one group's alphas, e.g. matched[matched['null_kind'] "
                "== 'cell_identity'].set_index('method')['alpha']",
            )
        )
    alphas = {str(name): float(value) for name, value in dict(alpha).items()}
    bad = {name: value for name, value in alphas.items() if not 0 < value <= 1}
    if bad:
        raise ValueError(
            f"{operation}: alphas must be in (0, 1], got {bad}; a NaN from "
            "matched_alpha means no alpha holds that method's rate."
        )
    data = _checked_table(table, groups, operation)
    absent = sorted(set(alphas) - set(data["method"]))
    if absent:
        raise ValueError(
            f"{operation}: alpha names methods {absent} that are not in the "
            f"table, whose methods are {list(dict.fromkeys(data['method']))}."
        )
    methods = _alpha_methods(data, replay_methods, list(alphas), operation)
    result = table.copy()
    own = {m.name: getattr(m, "alpha", np.nan) for m in replay_methods}
    result["alpha"] = data["method"].map(own).astype(float)
    for name, new_alpha in alphas.items():
        rows = data["method"] == name
        frame = data[rows]
        _check_verdicts(frame, methods[name], operation)
        result.loc[rows, "is_replay"] = _verdicts(
            frame, methods[name], new_alpha, operation
        )
        result.loc[rows, "alpha"] = new_alpha
    return result


def _alpha_methods(
    data: pd.DataFrame,
    replay_methods: Sequence[ReplayMethod | RankOrderMethod | ArmReplayMethod],
    names: list[str],
    operation: str,
) -> dict[str, ReplayMethod | RankOrderMethod]:
    """The named methods, checked to test a p-value against alpha."""
    from neurospatial.replay.methods import RankOrderMethod, ReplayMethod

    by_name = {method.name: method for method in replay_methods}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            _format_error(
                f"{operation}: the table has methods {unknown} that are not in "
                "replay_methods.",
                why="each method's criteria and alpha are needed to call events "
                "replay at another alpha",
                fix="pass the replay_methods the table was made with",
            )
        )
    if "p_value" not in data:
        raise ValueError(
            f"{operation}: table lacks a p_value column; pass the table of "
            "compare_replay_methods()."
        )
    tested: dict[str, ReplayMethod | RankOrderMethod] = {}
    untested, choosing = [], []
    for name in names:
        method = by_name[name]
        if isinstance(method, RankOrderMethod) and method.template_choice:
            rows = data["method"] == name
            if "template" in data and data.loc[rows, "template"].notna().any():
                choosing.append(name)
                continue
        if (isinstance(method, ReplayMethod) and method.nulls) or (
            isinstance(method, RankOrderMethod) and method.test != "pooled_ssi"
        ):
            tested[name] = method
        else:
            untested.append(name)
    if choosing:
        raise ValueError(
            _format_error(
                f"{operation}: methods {choosing} chose among several templates.",
                why="they pick the most active template among those that are "
                "replay at their alpha, so the tested template and its p-value "
                "change with alpha",
                fix="rerun compare_replay_methods with dataclasses.replace("
                "method, alpha=...) at each alpha, or pass one template's rate "
                "maps",
            )
        )
    if untested:
        raise ValueError(
            _format_error(
                f"{operation}: methods {untested} do not test a p-value against alpha.",
                why="methods without nulls and arm methods apply criteria only, "
                "and the pooled-SSI test thresholds the score against a null "
                "pooled over the epoch",
                fix=f"leave them out, e.g. table[~table['method'].isin({untested})]",
            )
        )
    return tested


def _smallest_p_value(method: ReplayMethod | RankOrderMethod) -> float:
    """The smallest p-value a method's shuffles can give, (0 + 1) / (n + 1)."""
    if getattr(method, "test", None) == "t_distribution":
        return 0.0
    return 1.0 / (method.n_shuffles + 1)


def _verdicts(
    frame: pd.DataFrame,
    method: ReplayMethod | RankOrderMethod,
    alpha: float,
    operation: str,
) -> NDArray[np.bool_]:
    """Rows scored without error that pass the criteria with p < alpha."""
    columns = [f"passes[{criterion.label}]" for criterion in method.criteria]
    absent = [column for column in columns if column not in frame]
    if absent:
        raise ValueError(
            f"{operation}: table lacks columns {absent} of method "
            f"{method.name!r}; pass the table of compare_replay_methods() made "
            "with this method."
        )
    verdicts: NDArray[np.bool_] = ~frame["_error"].to_numpy(dtype=bool)
    for column in columns:
        verdicts &= frame[column].eq(True).to_numpy()
    if np.isfinite(alpha):
        verdicts &= frame["p_value"].to_numpy(dtype=float) < alpha
    return verdicts


def _check_verdicts(
    frame: pd.DataFrame, method: ReplayMethod | RankOrderMethod, operation: str
) -> None:
    """Refuse a table whose verdicts do not follow from the method."""
    expected = _verdicts(frame, method, method.alpha, operation)
    differ = int(np.sum(expected != frame["is_replay"].to_numpy(dtype=bool)))
    if differ:
        raise ValueError(
            _format_error(
                f"{operation}: {differ} verdicts of method {method.name!r} do not "
                f"follow from its p-values and criteria at alpha = {method.alpha:g}.",
                why="the table was made with other settings under the same name",
                fix="pass the replay_methods the table was made with",
            )
        )


def _checked_table(
    table: pd.DataFrame, groups: Sequence[str], operation: str
) -> pd.DataFrame:
    """A copy with an ``_error`` column, checked to be a comparison table."""
    import pandas as pd

    missing = [c for c in ("event", "method", "is_replay", *groups) if c not in table]
    if missing:
        raise ValueError(
            f"{operation}: table lacks columns {missing}; pass the table of "
            "compare_replay_methods(), with any groups columns added."
        )
    if not pd.api.types.is_bool_dtype(table["is_replay"]):
        raise ValueError(
            f"{operation}: table's is_replay must be boolean, got "
            f"{table['is_replay'].dtype}."
        )
    keys = [*groups, "event", "method"]
    repeated = table.duplicated(subset=keys)
    if repeated.any():
        raise ValueError(
            _format_error(
                f"{operation}: table repeats {int(repeated.sum())} (event, "
                "method) pairs.",
                why="events are numbered within each compare_replay_methods "
                "call, so concatenated tables reuse the numbers",
                fix="add a column per table (e.g. null_kind or session) and "
                "pass it as groups=",
            )
        )
    data = table.copy()
    data["_error"] = (
        data["error"].fillna("").astype(str) != ""
        if "error" in data
        else pd.Series(False, index=data.index)
    )
    return data


# --------------------------------------------------------------------------
# Null data
# --------------------------------------------------------------------------

_NULL_KINDS = ("cell_identity", "spike_shift", "poisson")
_SIMULATED_KINDS = ("random", "stationary")
# Seconds between consecutive null events in their session layout.
_GAP = 1.0


@dataclass(frozen=True, repr=False, eq=False)
class NullSpikeData(ReadOnlyArraysMixin, ResultMixin):
    """Spike trains of events without replay, laid out like a session.

    Each null event occupies its own interval, separated by a second of
    silence, so the data can be decoded and scored exactly as the real
    events are: decode each interval (for example with
    :func:`decode_intervals`) and pass the decoded events, their spike
    counts and ``spike_times`` to
    :func:`~neurospatial.replay.compare_replay_methods`. Score null events
    in their own call, not together with real events: the pseudo-event null
    draws from every event of a call. Concatenate the tables with a column
    telling them apart and pass it to ``detection_rates(groups=...)``. The
    table's ``event`` indexes ``intervals``, so ``source_events[event]``
    joins a null event to its real event's properties.

    Attributes
    ----------
    spike_times : tuple of NDArray[np.float64]
        Spike times of each cell, in seconds, in the null layout.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Start and stop of each null event, in seconds.
    source_events : NDArray[np.intp], shape (n_events,)
        Index of the real event each null event was made from; -1 for a
        simulated event.
    kind : str
        How the null events were made.

    See Also
    --------
    null_events : Null events made from real events.
    simulated_null_events : Simulated events without replay.
    """

    spike_times: tuple[NDArray[np.float64], ...]
    intervals: NDArray[np.float64]
    source_events: NDArray[np.intp]
    kind: str

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays, including each spike train."""
        trains = []
        for train in self.spike_times:
            owned = np.array(train, dtype=np.float64)
            owned.flags.writeable = False
            trains.append(owned)
        object.__setattr__(self, "spike_times", tuple(trains))
        self._own_arrays()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping every array read-only."""
        super().__setstate__(state)
        for train in self.spike_times:
            train.flags.writeable = False

    @property
    def n_events(self) -> int:
        """Number of null events."""
        return len(self.intervals)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "kind": self.kind,
            "n_events": self.n_events,
            "n_cells": len(self.spike_times),
            "n_spikes": int(sum(train.size for train in self.spike_times)),
        }


def null_events(
    spike_times: Any,
    intervals: NDArray[np.float64],
    *,
    kind: Literal["cell_identity", "spike_shift", "poisson"],
    n_copies: int = 1,
    cells: NDArray[np.bool_] | None = None,
    rng: np.random.Generator | int | None = None,
) -> NullSpikeData:
    """Make events without replay from real candidate events.

    Each null event keeps a real event's duration and, depending on
    ``kind``, its cells' spike counts or spike trains, but breaks the
    order of the cells' firing relative to their fields:

    - ``"cell_identity"``: the event's spike trains are reassigned to
      cells at random, so each cell fires with another cell's spikes
      (Takigawa et al. 2024 made three such copies of each event).
    - ``"spike_shift"``: each cell's spikes are shifted circularly within
      the event by an independent random amount.
    - ``"poisson"``: each cell keeps its number of spikes, placed uniformly
      at random in the event (a Poisson process given its count).

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        The session's spike times, one train per cell, in seconds.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Start and stop of each real candidate event, in seconds.
    kind : {"cell_identity", "spike_shift", "poisson"}
        How each null event is made.
    n_copies : int, default=1
        Null events made from each real event.
    cells : NDArray[np.bool_] or None, shape (n_cells,)
        The cells to randomize, such as the place cells of the decoder
        (Takigawa et al. 2024 reassigned only "good" place cells); the
        others keep their spikes. None randomizes every cell.
    rng : np.random.Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    NullSpikeData
        ``n_events * n_copies`` null events, the copies of each event
        together, with ``source_events`` naming the real event.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the options, ``n_copies`` is not a
        positive integer, ``cells`` is not one boolean per cell, or
        ``intervals`` is not a list of finite ``(start, stop)`` with
        ``start < stop``.

    See Also
    --------
    simulated_null_events : Simulated events without replay.
    decode_intervals : Decode the null events.

    Notes
    -----
    What each kind keeps and breaks:

    ================  ======================  ==========================
    kind              keeps                   breaks
    ================  ======================  ==========================
    cell_identity     every spike train;      which field each train
                      the population's        belongs to, and with it
                      timing                  any link between a cell's
                                              count and its field
    spike_shift       each cell's count and   the timing between cells
                      its spike intervals
                      (circularly)
    poisson           each cell's count       all spike timing
    ================  ======================  ==========================

    A null along the same dimension as a method's own shuffle
    underestimates its false-positive rate (Takigawa et al. 2024):
    ``cell_identity`` repeats the unit-identity null of ``DAVIDSON_2009``
    and the cell-order test of ``DIBA_BUZSAKI_2007``, and ``spike_shift``
    the spike-train shift of ``TIROLE_2022``. Rank-order methods can pass
    ``spike_shift`` and ``poisson`` events more often than ``alpha`` when
    an event covers part of the track, because which cells spike more then
    still follows their fields; report several kinds side by side.

    Reassigning spike trains is the same as Takigawa et al.'s
    reassignment of rate maps for one set of rate maps. For a joint decode
    over several templates (tracks or directions) it differs: they
    permuted each template's maps separately, and one permutation of the
    spikes keeps each cell's maps together.

    Copies of one real event share its spikes and are not independent:
    the intervals of :func:`detection_rates` treat every event as
    independent and are too narrow with ``n_copies > 1``.

    References
    ----------
    Takigawa, M., Huelin Gorriz, M., Tirole, M., & Bendor, D. (2024).
    Evaluating hippocampal replay without a ground truth. *eLife*, 13,
    e85635.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import null_events
    >>> spikes = [np.array([1.01, 1.05]), np.array([1.02]), np.array([5.0])]
    >>> null = null_events(
    ...     spikes, np.array([[1.0, 1.1]]), kind="poisson", n_copies=3, rng=0
    ... )
    >>> print(null.n_events, [train.size for train in null.spike_times])
    3 [6, 3, 0]
    """
    from neurospatial._spikes import as_spike_trains

    operation = "null_events()"
    if kind not in _NULL_KINDS:
        raise ValueError(
            f"{operation}: kind must be one of {_NULL_KINDS}, got {kind!r}."
        )
    _positive_count(n_copies, "n_copies", operation)
    windows = _checked_intervals(intervals, operation)
    trains = as_spike_trains(spike_times)
    chosen = np.ones(len(trains), dtype=bool) if cells is None else np.asarray(cells)
    if chosen.dtype != np.bool_ or chosen.shape != (len(trains),):
        raise ValueError(
            f"{operation}: cells must be one boolean per cell, shape "
            f"({len(trains)},), got dtype {chosen.dtype} and shape {chosen.shape}."
        )
    randomized = np.flatnonzero(chosen)
    generator = np.random.default_rng(rng)
    durations = windows[:, 1] - windows[:, 0]
    spacing = float(durations.max()) + _GAP
    placed: list[list[NDArray[np.float64]]] = [[] for _ in trains]
    out_intervals = []
    sources = []
    slot = 0
    for index, (start, stop) in enumerate(windows):
        duration = stop - start
        in_event = [
            np.asarray(train[(train >= start) & (train < stop)]) - start
            for train in trains
        ]
        for _ in range(n_copies):
            offset = slot * spacing
            copies = list(in_event)
            if kind == "cell_identity":
                order = generator.permutation(randomized)
                for cell, source in zip(randomized, order, strict=True):
                    copies[cell] = in_event[source]
            elif kind == "spike_shift":
                for cell in randomized:
                    shift = generator.uniform(0.0, duration)
                    copies[cell] = np.mod(in_event[cell] + shift, duration)
            else:
                for cell in randomized:
                    copies[cell] = generator.uniform(
                        0.0, duration, size=in_event[cell].size
                    )
            for cell, spikes in enumerate(copies):
                placed[cell].append(np.sort(spikes) + offset)
            out_intervals.append((offset, offset + duration))
            sources.append(index)
            slot += 1
    return NullSpikeData(
        spike_times=tuple(
            np.concatenate(parts) if parts else np.empty(0) for parts in placed
        ),
        intervals=np.asarray(out_intervals, dtype=np.float64),
        source_events=np.asarray(sources, dtype=np.intp),
        kind=kind,
    )


def simulated_null_events(
    env: Environment,
    encoding_models: Any,
    *,
    n_events: int,
    kind: Literal["random", "stationary"] = "random",
    duration: float = 0.2,
    rate_gain: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> NullSpikeData:
    """Simulate events without replay from the cells' rate maps.

    Each event is simulated with
    :func:`~neurospatial.simulation.simulate_replay_event`: cells fire as
    Poisson processes at ``rate_gain`` times their rate at a position that
    jumps at random every 20 ms (``"random"``) or stays put, at a position
    drawn uniformly along the track (``"stationary"``, which line fits
    score well but correlation scores do not). Match ``rate_gain`` to the
    real events' spike counts, or stratify rates by active cells: with
    running rates, many events fail cell criteria and their rates are not
    comparable.

    Parameters
    ----------
    env : Environment
        A 1-D environment or a linearized track.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz.
    n_events : int
        Number of events.
    kind : {"random", "stationary"}, default="random"
        The simulated content.
    duration : float, default=0.2
        Duration of each event, in seconds.
    rate_gain : float, default=1.0
        Multiplier of the rate maps during events.
    rng : np.random.Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    NullSpikeData
        The simulated events in a session layout, with ``source_events``
        of -1.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the options, ``n_events`` is not a
        positive integer, or for inputs ``simulate_replay_event`` refuses.

    See Also
    --------
    null_events : Null events made from real events.
    decode_intervals : Decode the null events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import simulated_null_events
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> centers = np.linspace(5, 95, 8)[:, None]
    >>> rates = 20.0 * np.exp(-0.5 * ((x - centers) / 8.0) ** 2)
    >>> null = simulated_null_events(env, rates, n_events=4, rng=0)
    >>> null.intervals.round(2).tolist()[:2]
    [[0.0, 0.2], [1.2, 1.4]]
    """
    from neurospatial.environment._queries import _require_track_extents
    from neurospatial.simulation import simulate_replay_event

    operation = "simulated_null_events()"
    if kind not in _SIMULATED_KINDS:
        raise ValueError(
            f"{operation}: kind must be one of {_SIMULATED_KINDS}, got {kind!r}."
        )
    _positive_count(n_events, "n_events", operation)
    generator = np.random.default_rng(rng)
    seeds = generator.integers(2**63, size=n_events)
    starts_, stops_ = _require_track_extents(
        env, operation, alternative="simulate events on a 1-D track"
    )
    lengths = stops_ - starts_
    spacing = float(duration) + _GAP
    placed: list[list[NDArray[np.float64]]] = []
    for slot, seed in enumerate(seeds):
        start_position = None
        if kind == "stationary":
            # Uniform along the track: a bin by its length, then inside it.
            k = generator.choice(len(lengths), p=lengths / lengths.sum())
            start_position = float(generator.uniform(starts_[k], stops_[k]))
        event = simulate_replay_event(
            env,
            encoding_models,
            kind=kind,
            duration=duration,
            rate_gain=rate_gain,
            start_position=start_position,
            seed=int(seed),
        )
        if not placed:
            placed = [[] for _ in event.spike_times]
        for cell, train in enumerate(event.spike_times):
            placed[cell].append(np.asarray(train) + slot * spacing)
    starts = np.arange(n_events) * spacing
    return NullSpikeData(
        spike_times=tuple(np.concatenate(parts) for parts in placed),
        intervals=np.column_stack([starts, starts + float(duration)]),
        source_events=np.full(n_events, -1, dtype=np.intp),
        kind=f"simulated_{kind}",
    )


def decode_intervals(
    env: Environment,
    spike_times: Any,
    intervals: NDArray[np.float64],
    encoding_models: Any,
    *,
    dt: float,
    prior: NDArray[np.float64] | None = None,
) -> tuple[list[DecodingResult], list[NDArray[np.int64]]]:
    """Decode each interval in non-overlapping time bins of width ``dt``.

    A convenience for candidate events and null events alike. Decode real
    and null events with the same function and settings: this one only
    makes non-overlapping bins on one set of rate maps, while many
    published methods decode overlapping windows (their ``decoding``
    settings) or jointly over templates.

    Parameters
    ----------
    env : Environment
        The environment of the rate maps.
    spike_times : array, sequence of arrays, or spike group
        Spike times of each cell, in seconds.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Start and stop of each event, in seconds; a trailing part shorter
        than ``dt`` is dropped.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz.
    dt : float
        Time-bin width, in seconds.
    prior : NDArray[np.float64] or None
        Prior over bins, as for :func:`~neurospatial.decoding.decode_position`.

    Returns
    -------
    events : list of DecodingResult
        Each interval's decoded posterior, with the time-bin centers.
    spike_counts : list of NDArray[np.int64]
        Each interval's spike counts, ``(n_time_bins, n_cells)``.

    Raises
    ------
    ValueError
        If an interval holds no time bin of width ``dt``, or for inputs
        :func:`~neurospatial.decoding.bin_spikes_in_time` or
        :func:`~neurospatial.decoding.decode_position` refuse.

    See Also
    --------
    null_events : Null events to decode.
    neurospatial.replay.compare_replay_methods : Score the decoded events.
    """
    from neurospatial.decoding import bin_spikes_in_time, decode_position

    operation = "decode_intervals()"
    windows = _checked_intervals(intervals, operation)
    events = []
    counts = []
    for start, stop in windows:
        event_counts, grid = bin_spikes_in_time(
            spike_times, dt=dt, t_start=float(start), t_stop=float(stop)
        )
        events.append(
            decode_position(
                env, event_counts, encoding_models, dt=dt, prior=prior, times=grid
            )
        )
        counts.append(np.asarray(event_counts))
    return events, counts


def _positive_count(value: Any, name: str, operation: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(
            f"{operation}: {name} must be a positive integer, got {value!r}."
        )


def _checked_intervals(intervals: Any, operation: str) -> NDArray[np.float64]:
    windows = np.asarray(intervals, dtype=np.float64)
    if (
        windows.ndim != 2
        or windows.shape[1] != 2
        or len(windows) == 0
        or not np.all(np.isfinite(windows))
        or np.any(windows[:, 1] <= windows[:, 0])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: intervals must be finite (start, stop) rows with "
                f"start < stop, got shape {windows.shape}.",
                fix="pass an array of shape (n_events, 2) in seconds",
            )
        )
    return windows
