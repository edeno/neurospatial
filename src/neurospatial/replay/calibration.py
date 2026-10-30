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
  with Cohen's kappa, and :func:`score_agreement` correlates their scores
  across events.
- :func:`matched_alpha` finds each method's alpha that holds its
  false-positive rate on null data to a target (Takigawa et al. 2024), and
  :func:`apply_alpha` gives a table's verdicts at those alphas.

Null data for false-positive rates: :func:`null_events` makes events
without replay from real candidate events (cell identities reassigned,
spikes shifted, spike times redrawn, or Poisson spikes at each cell's mean
rate), :func:`simulated_null_events` simulates
random or stationary events, and :func:`decode_intervals` decodes either.

A single overall false-positive rate hides how it depends on the event.
Tingley & Peyrache (2020) simulated random firing orders tested with a
parametric rank-correlation p-value: sets of 100 events passed p < 0.05
5-12% of the time with 10 cells and up to 20% with 5 (the expected rates
of that test are 5.4% and 8.3%, and 33% with 3 cells). They simulated only
the number of cells; that the rate also depends on firing rates and on the
method, they argued without simulating it. Report rates per stratum, and
on several kinds of null data rather than one.
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
from neurospatial.replay._compare import _tolerance

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult, JointDecodingResult
    from neurospatial.environment import Environment
    from neurospatial.replay.arms import ArmReplayMethod
    from neurospatial.replay.methods import RankOrderMethod, ReplayMethod
    from neurospatial.replay.pairwise_order import PairwiseOrderMethod
    from neurospatial.replay.sequence_bias import BiasSequenceMethod
    from neurospatial.replay.template_match import TemplateMatchMethod

    # Methods whose verdict tests a p-value against alpha.
    _TestedMethod = (
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    )

__all__ = [
    "NullSpikeData",
    "apply_alpha",
    "decode_intervals",
    "detection_rates",
    "event_properties",
    "matched_alpha",
    "method_agreement",
    "null_events",
    "score_agreement",
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
    from neurospatial.replay.methods import _environment_mismatch

    def checked(chosen: Environment, decoded: Environment) -> Environment:
        mismatch = _environment_mismatch(chosen, decoded)
        if mismatch is not None:
            raise ValueError(
                _format_error(
                    f"{operation}: env differs from the environment event "
                    f"{index} was decoded on in {mismatch}.",
                    why="positions and distances would be read on other bins",
                    fix="pass the environment the events were decoded on",
                )
            )
        return chosen

    if not isinstance(event, JointDecodingResult):
        if isinstance(env, Mapping):
            if template is None:
                raise ValueError(
                    f"{operation}: a mapping of environments needs template=."
                )
            return checked(env[template], event.env), event.posterior, event.times
        return checked(env, event.env), event.posterior, event.times
    names = event.template_names
    if template is not None:
        if template not in names:
            raise ValueError(
                f"{operation}: template {template!r} is not one of {list(names)} "
                f"in event {index}."
            )
        chosen = env[template] if isinstance(env, Mapping) else env
        return (
            checked(chosen, event.conditional[template].env),
            event.joint(template),
            event.conditional[template].times,
        )
    envs = [event.conditional[name].env for name in names]
    first = envs[0]
    shared = all(_environment_mismatch(other, first) is None for other in envs[1:])
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
    return checked(env, first), posterior, event.conditional[names[0]].times


def detection_rates(
    table: pd.DataFrame,
    *,
    properties: pd.DataFrame | None = None,
    by: Mapping[str, Sequence[float]] | None = None,
    groups: Sequence[str] = (),
    errors: Literal["not_replay", "exclude"] = "not_replay",
    confidence: float = 0.95,
    expected_rate: float | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> pd.DataFrame:
    """Each method's share of events called replay, with a confidence interval.

    On events without replay (null data) the rate is the method's
    false-positive rate; on simulated replays, its sensitivity. With
    ``expected_rate``, each rate is also tested against a rate expected by
    chance, such as the share of events with p < 0.05 against 0.05.

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
        columns. Joined on ``event``, or on the ``groups`` and ``event``
        when it has the ``groups`` as columns or index levels. It must have
        them for several concatenated tables, which each number their
        events from 0.
    by : Mapping[str, Sequence[float]] or None
        Break rates down by numeric properties: each key a column of
        ``table`` or ``properties``, each value at least two bin edges,
        such as ``{"n_active_cells": [5, 10, 20, np.inf]}``. Bins are
        closed on the left. Every event must fall in a bin.
    groups : sequence of str, default=()
        Columns of ``table`` to break rates down by their values, such as
        ``("null_kind",)``. A missing value forms its own group.
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
    expected_rate : float or None, default=None
        A rate in (0, 1) to test each stratum's rate against with an exact
        binomial test, ``scipy.stats.binomtest(n_replay, n_events,
        expected_rate)``. Grosmark & Buzsáki (2016, Figs. S5C and S7B)
        compared the share of events with p < 0.05 with 5% by a chi-square
        test of proportions; the exact binomial test is its exact form, and
        the p-values differ slightly.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Side of the binomial test: ``"greater"`` asks whether more events
        are called replay than ``expected_rate`` predicts.

    Returns
    -------
    pandas.DataFrame
        One row per method and stratum, in every combination: ``method``,
        one column per ``groups`` and ``by`` key (the bin, as an interval),
        ``n_events``, ``n_replay``, ``rate``, ``ci_low``, ``ci_high`` and
        ``n_errors``; with ``expected_rate``, the binomial ``p_value``.
        ``rate``, the interval and ``p_value`` are NaN without events.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``event``, ``method`` or ``is_replay``
        columns, has a non-boolean ``is_replay`` or repeats an (event,
        method) pair within its ``groups``; if a ``by`` key is in neither
        ``table`` nor ``properties``, has fewer than two edges, or leaves
        events outside its bins (NaN, out of range, or absent from
        ``properties``); if ``properties`` lacks the ``groups`` of a
        table that numbers its events again in each group, or repeats a
        (``groups``, ``event``) key; or if ``errors``, ``confidence``,
        ``expected_rate`` or ``alternative`` is invalid.

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

    Against a chance rate: 9 of 100 events with p < 0.05 is more than the
    5% expected, but not significantly (two-sided exact binomial test).

    >>> table = pd.DataFrame(
    ...     {"event": range(100), "method": "a", "is_replay": [True] * 9 + [False] * 91}
    ... )
    >>> row = detection_rates(table, expected_rate=0.05).iloc[0]
    >>> print(f"{row['rate']:.2f}, p = {row['p_value']:.3f}")
    0.09, p = 0.100
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
    if expected_rate is not None and not (
        isinstance(expected_rate, (int, float, np.integer, np.floating))
        and not isinstance(expected_rate, bool)
        and 0 < expected_rate < 1
    ):
        raise ValueError(
            f"{operation}: expected_rate must be a rate in (0, 1) or None, got "
            f"{expected_rate!r}; for the share of p < 0.05 events by chance, "
            "pass 0.05."
        )
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"{operation}: alternative must be 'two-sided', 'greater' or "
            f"'less', got {alternative!r}."
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
            values = _property_values(data, properties, name, groups, operation)
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
    # Group on integer codes, so a missing groups value (NaN or None) forms
    # its own group, as in matched_alpha, and is found again below.
    levels = {}
    for key in ["method", *groups]:
        codes, uniques = pd.factorize(data[key], use_na_sentinel=False)
        data[f"_code[{key}]"] = codes
        levels[key] = list(uniques)
    for name in by or {}:
        data[f"_code[{name}]"] = data[name].cat.codes
        levels[name] = list(data[name].cat.categories)
    coded = [f"_code[{key}]" for key in keys]
    frames = dict(iter(data.groupby(coded, sort=False)))
    rows = []
    for group in itertools.product(*(range(len(levels[key])) for key in keys)):
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
        row = {key: levels[key][code] for key, code in zip(keys, group, strict=True)}
        row.update(
            n_events=n_events,
            n_replay=n_replay,
            rate=rate,
            ci_low=low,
            ci_high=high,
            n_errors=int(failed.sum()),
        )
        if expected_rate is not None:
            row["p_value"] = (
                float(
                    binomtest(
                        n_replay,
                        n_events,
                        float(expected_rate),
                        alternative=alternative,
                    ).pvalue
                )
                if n_events
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _property_values(
    data: pd.DataFrame,
    properties: pd.DataFrame,
    name: str,
    groups: Sequence[str],
    operation: str,
) -> pd.Series:
    """Each table row's event property.

    Joined on ``[*groups, "event"]`` when ``properties`` has the ``groups``
    (an index without the name ``event`` is the event), and otherwise on
    the index, as the event, when each event number names one event of the
    table.
    """
    import pandas as pd

    available = {*properties.columns, *properties.index.names}
    missing = [key for key in groups if key not in available]
    if not groups or missing:
        if groups and data.duplicated(subset=["event", "method"]).any():
            raise ValueError(
                _format_error(
                    f"{operation}: properties lacks {missing} as columns or "
                    "index levels, and the table numbers its events again in "
                    f"each group of {list(groups)}.",
                    why="events are numbered within each compare_replay_methods "
                    "call, so the event number alone does not say which call's "
                    "event a property belongs to",
                    fix="add the groups columns to each call's properties before "
                    "concatenating them, e.g. event_properties(env, events)"
                    ".assign(session='s1')",
                )
            )
        return data["event"].map(properties[name])
    join = [*groups, "event"]
    named = [level for level in properties.index.names if level is not None]
    flat = properties.reset_index(level=named) if named else properties
    if "event" not in flat:
        flat = flat.rename_axis("event").reset_index()
    flat = flat[[*join, name]]
    repeated = flat.duplicated(subset=join)
    if repeated.any():
        raise ValueError(
            _format_error(
                f"{operation}: properties has {int(repeated.sum())} repeated "
                f"{join} keys.",
                why="each event needs one value of each property",
                fix="pass one row per event and group",
            )
        )
    joined = data[join].merge(flat, on=join, how="left")
    return pd.Series(joined[name].to_numpy(), index=data.index, name=name)


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
        Columns of ``table`` that, with ``event``, identify an event. A
        missing value forms its own group.
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
    score_agreement : How closely the methods' scores correlate.

    Notes
    -----
    This compares verdicts. Tingley & Peyrache (2020, Fig. 3) compared
    scores instead (:func:`score_agreement`): the correlation, across one
    session's events, of two methods' scores, averaged over sessions and
    templates. The weighted correlation correlated with the rank-order
    score at R = 0.126, with the Radon line-fit integral (the line-fit
    score) at R = 0.194 and with the Radon line's slope at R = 0.43; the
    rank-order score correlated with neither the integral (R = 0.003) nor
    the slope (R = 0.07). The weighted correlation is signed and the
    integral is not, which alone keeps their correlation near 0 when
    forward and reverse events are equally common (see
    :func:`score_agreement`). Their rank-order score correlates the two
    orders' sorting indices (MATLAB's ``sort`` outputs, which list cell
    numbers) rather than ranks, so it depends on how the cells are
    numbered.

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


def score_agreement(
    table: pd.DataFrame,
    *,
    groups: Sequence[str] = (),
    absolute: bool = False,
) -> pd.DataFrame:
    """How closely every pair of methods' scores correlate across events.

    The Pearson correlation, across events, between two methods' scores:
    the comparison of Tingley & Peyrache (2020, Fig. 3). Unlike
    :func:`method_agreement`, it compares the scores, not the verdicts, so
    it does not depend on the methods' shuffles or thresholds.

    Parameters
    ----------
    table : pandas.DataFrame
        The table of :func:`~neurospatial.replay.compare_replay_methods`,
        or several concatenated with ``groups`` columns that tell their
        events apart.
    groups : sequence of str, default=()
        Columns of ``table`` to compute the correlations in separately,
        such as ``("session",)``. A missing value forms its own group.
    absolute : bool, default=False
        Correlate the absolute values of the scores. Use it when a signed
        score (a weighted or rank-order correlation, negative for reverse
        order) is compared with an unsigned one (line fit, R²).

    Returns
    -------
    pandas.DataFrame
        One row per pair of methods, in the table's method order, and per
        group: ``first``, ``second``, the ``groups`` columns, ``n_events``
        (events with a finite score from both) and ``correlation``.
        ``correlation`` is NaN with fewer than 3 such events or when either
        score is constant over them.

    Raises
    ------
    ValueError
        If ``table`` lacks the ``event``, ``method``, ``score`` or
        ``groups`` columns, or repeats an (event, method) pair within its
        ``groups``.

    See Also
    --------
    method_agreement : How often the methods' verdicts agree.

    Notes
    -----
    Tingley & Peyrache computed the correlation in each session and
    template and reported its mean: pass ``groups=("session",)`` and
    average ``correlation``.

    Scores enter as the table reports them. Correlating a signed score with
    an unsigned one measures whether the sign follows the unsigned score,
    not whether the methods agree: if forward and reverse events are equally
    common and the direction is unrelated to the unsigned score, the
    correlation is near 0 even when both methods rank every sequence above
    every other event (the covariance of ``s * |x|`` with ``y``, for a sign
    ``s`` of mean 0 independent of ``|x|`` and ``y``, is 0). Pass
    ``absolute=True`` for such pairs, and keep the signs when both scores
    are signed and you ask whether the methods agree on direction.

    A score's scale differs between methods (line-fit mass, R², rho), and a
    Pearson correlation measures only a linear relation; read it with a
    scatter plot of the two scores. Events a method could not score (NaN
    scores) are left out of that method's pairs; events with an ``error``
    but a score, such as an undefined p-value, are kept.

    Examples
    --------
    A signed score (``"wc"``) that agrees with an unsigned one
    (``"line_fit"``) on which events are strong, with half the events
    reversed:

    >>> import pandas as pd
    >>> from neurospatial.replay import score_agreement
    >>> table = pd.DataFrame(
    ...     {
    ...         "event": [0, 0, 1, 1, 2, 2, 3, 3],
    ...         "method": ["line_fit", "wc"] * 4,
    ...         "score": [0.9, 0.8, 0.9, -0.8, 0.3, 0.1, 0.3, -0.1],
    ...     }
    ... )
    >>> print(round(score_agreement(table)["correlation"].item(), 6))
    0.0
    >>> print(round(score_agreement(table, absolute=True)["correlation"].item(), 6))
    1.0
    """
    import pandas as pd

    operation = "score_agreement()"
    data = _checked_table(table, groups, operation, values="score")
    data["score"] = data["score"].astype(float)
    if absolute:
        data["score"] = data["score"].abs()
    methods = list(dict.fromkeys(data["method"]))
    grouped = (
        data.groupby(list(groups), sort=False, dropna=False) if groups else [((), data)]
    )
    rows = []
    for key, frame in grouped:
        scores = frame.pivot(index="event", columns="method", values="score")
        scores = scores.reindex(columns=methods)
        for first, second in itertools.combinations(methods, 2):
            a = scores[first].to_numpy()
            b = scores[second].to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            rows.append(
                {
                    "first": first,
                    "second": second,
                    **dict(zip(groups, key, strict=True)),
                    "n_events": int(both.sum()),
                    "correlation": _pearson(a[both], b[both]),
                }
            )
    return pd.DataFrame(rows)


def _pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation; NaN with fewer than 3 values or a constant one."""
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))


def matched_alpha(
    null_table: pd.DataFrame,
    *,
    replay_methods: Sequence[
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    ],
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
    replay_methods : sequence of ReplayMethod, RankOrderMethod, PairwiseOrderMethod or TemplateMatchMethod
        The methods the table was made with. Each must test a p-value
        against its ``alpha``: a :class:`ReplayMethod` with ``nulls`` (but
        not ``tested_quantities`` with a segmentation that has
        ``choose="any"``, such as ``WIDLOSKI_FOSTER_2025``), a
        :class:`RankOrderMethod` with the ``"t_distribution"`` or
        ``"cell_order"`` test that tests one template, a
        :class:`~neurospatial.replay.PairwiseOrderMethod` (whose reported
        test, the smallest p-value over templates and directions, and
        active-cell rule, over every cell, do not depend on alpha), or a
        :class:`TemplateMatchMethod` (whose p-values are exact and take few
        values, so the matched alpha moves in steps).
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
        against alpha, is a rank-order method that chose among several
        templates, or is a method that chose among segments by their
        tested quantities; or if the table's verdicts do not follow from its
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
    each method, p < alpha (p <= alpha for a template-matching method with
    ``p_comparison="<="``, whose matched alpha then lies just below the
    first null p-value it excludes).

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
    template among those that are replay at its alpha, and a
    :class:`ReplayMethod` with ``tested_quantities`` and a segmentation
    with ``choose="any"`` (``WIDLOSKI_FOSTER_2025``) reports the longest
    segment that is replay at its alpha, so their rows change with alpha
    and the table's one p-value cannot give the verdict at another; rerun
    them at each alpha instead.

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
        if getattr(method, "p_comparison", "<") == "<=" and p_values.size > allowed:
            # p <= alpha (equal within round-off) would pass this event too:
            # the largest alpha that does not lies just below it.
            alpha = max(alpha - 2 * float(_tolerance(alpha)), 0.0)
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
                "rate": float(np.sum(_p_passes(p_values, alpha, method))) / n_events
                if np.isfinite(alpha)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def apply_alpha(
    table: pd.DataFrame,
    *,
    alpha: Mapping[str, float] | pd.Series,
    replay_methods: Sequence[
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    ],
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
    replay_methods : sequence of ReplayMethod, RankOrderMethod, PairwiseOrderMethod or TemplateMatchMethod
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
        method without nulls does not use). ``alpha`` is NaN for an arm
        method, which has none, and for a table method not in
        ``replay_methods``, whose alpha is unknown; its verdicts are kept.

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
    replay_methods: Sequence[
        ReplayMethod
        | RankOrderMethod
        | PairwiseOrderMethod
        | ArmReplayMethod
        | TemplateMatchMethod
        | BiasSequenceMethod
    ],
    names: list[str],
    operation: str,
) -> dict[str, _TestedMethod]:
    """The named methods, checked to test a p-value against alpha."""
    from neurospatial.replay.methods import RankOrderMethod, ReplayMethod
    from neurospatial.replay.pairwise_order import PairwiseOrderMethod
    from neurospatial.replay.sequence_bias import BiasSequenceMethod
    from neurospatial.replay.template_match import TemplateMatchMethod

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
    tested: dict[str, _TestedMethod] = {}
    untested, choosing, segment_choosing = [], [], []
    for name in names:
        method = by_name[name]
        if isinstance(method, RankOrderMethod) and method.template_choice:
            rows = data["method"] == name
            if "template" in data and data.loc[rows, "template"].notna().any():
                choosing.append(name)
                continue
        if (
            isinstance(method, ReplayMethod)
            and method.tested_quantities
            and method.segmentation is not None
            and method.segmentation.choose == "any"
        ):
            segment_choosing.append(name)
            continue
        # A template-matching method's choice of template (the largest
        # score) and a pairwise-order method's (the smallest p-value, with an
        # active-cell rule over every cell) do not depend on alpha.
        if (
            (isinstance(method, ReplayMethod) and method.nulls)
            or (isinstance(method, RankOrderMethod) and method.test != "pooled_ssi")
            or isinstance(
                method, (TemplateMatchMethod, PairwiseOrderMethod, BiasSequenceMethod)
            )
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
    if segment_choosing:
        raise ValueError(
            _format_error(
                f"{operation}: methods {segment_choosing} chose among segments.",
                why="they test the segments from the longest down and report the "
                "first that is replay at their alpha, so the reported segment "
                "and its p-value change with alpha",
                fix="rerun compare_replay_methods with dataclasses.replace("
                "method, alpha=...) at each alpha",
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


def _smallest_p_value(
    method: _TestedMethod,
) -> float:
    """The smallest p-value a method's shuffles can give.

    ``1 / (n + 1)`` for a one-sided test and twice that for a two-sided
    test, which doubles the smaller tail (``compute_shuffle_pvalue``). It
    uses the method's full ``n_shuffles``, not the shuffles a null event
    kept: the warning asks whether real events, tested with every shuffle,
    can reach the matched alpha. Analytic tests have no shuffle floor: the
    t-distribution, and the template-matching probabilities (at least
    ``1 / n!`` over an event's ``n`` letters or cells).
    """
    from neurospatial.replay.template_match import TemplateMatchMethod

    if (
        isinstance(method, TemplateMatchMethod)
        or getattr(method, "test", None) == "t_distribution"
    ):
        return 0.0
    smallest = 1.0 / (method.n_shuffles + 1)
    # A pairwise-order test is one-sided: the score against larger shuffles.
    two_sided = getattr(method, "alternative", "greater") == "two-sided"
    return min(2.0 * smallest, 1.0) if two_sided else smallest


def _verdicts(
    frame: pd.DataFrame,
    method: _TestedMethod,
    alpha: float,
    operation: str,
) -> NDArray[np.bool_]:
    """Rows scored without error that pass the criteria with p < alpha.

    A template-matching method compares the p-value as its verdicts do
    (``<`` or ``<=``, equal within round-off).
    """
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
        verdicts &= _p_passes(frame["p_value"].to_numpy(dtype=float), alpha, method)
    return verdicts


def _p_passes(
    p_values: NDArray[np.float64],
    alpha: float,
    method: _TestedMethod,
) -> NDArray[np.bool_]:
    """Whether each p-value passes alpha by the method's comparison."""
    from neurospatial.replay.sequence_match import _passes
    from neurospatial.replay.template_match import TemplateMatchMethod

    if isinstance(method, TemplateMatchMethod):
        return np.asarray(_passes(p_values, alpha, method.p_comparison), dtype=bool)
    return p_values < alpha


def _check_verdicts(
    frame: pd.DataFrame,
    method: _TestedMethod,
    operation: str,
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
    table: pd.DataFrame,
    groups: Sequence[str],
    operation: str,
    *,
    values: str = "is_replay",
) -> pd.DataFrame:
    """A copy with an ``_error`` column, checked to be a comparison table.

    ``values`` is the column the caller summarizes, ``is_replay`` (which
    must be boolean) or ``score``.
    """
    import pandas as pd

    missing = [c for c in ("event", "method", values, *groups) if c not in table]
    if missing:
        raise ValueError(
            f"{operation}: table lacks columns {missing}; pass the table of "
            "compare_replay_methods(), with any groups columns added."
        )
    if values == "is_replay" and not pd.api.types.is_bool_dtype(table["is_replay"]):
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

_NULL_KINDS = ("cell_identity", "spike_shift", "spike_time_uniform", "rate_poisson")
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
    joins a null event to its real event's properties, and
    ``copy_index[event]`` names the null data set it belongs to: copy ``k``
    of every real event forms the ``k``-th null data set of a data-set-level
    test (see :mod:`neurospatial.replay.dataset`).

    Attributes
    ----------
    spike_times : tuple of NDArray[np.float64]
        Spike times of each cell, in seconds, in the null layout.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Start and stop of each null event, in seconds.
    source_events : NDArray[np.intp], shape (n_events,)
        Index of the real event each null event was made from; -1 for a
        simulated event.
    copy_index : NDArray[np.intp], shape (n_events,)
        Which copy of its real event each null event is, from 0 to
        ``n_copies - 1``; 0 for a simulated event. The events with
        ``copy_index == k`` form one null data set the size of the real one.
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
    copy_index: NDArray[np.intp]
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
    kind: Literal["cell_identity", "spike_shift", "spike_time_uniform", "rate_poisson"],
    n_copies: int = 1,
    cells: NDArray[np.bool_] | None = None,
    rng: np.random.Generator | int | None = None,
) -> NullSpikeData:
    """Make events without replay from real candidate events.

    Each null event keeps a real event's duration and, depending on
    ``kind``, its cells' spike counts or spike trains, or only their mean
    rates, but breaks the order of the cells' firing relative to their
    fields:

    - ``"cell_identity"``: the event's spike trains are reassigned to
      cells at random, so each cell fires with another cell's spikes
      (Takigawa et al. 2024 made three such copies of each event).
    - ``"spike_shift"``: each cell's spikes are shifted circularly within
      the event by an independent random amount.
    - ``"spike_time_uniform"``: each cell keeps its number of spikes, placed
      uniformly at random in the event (a Poisson process given its count;
      the "spike jittering" of Stella et al. 2019).
    - ``"rate_poisson"``: each cell fires as a Poisson process at its mean
      rate over all ``intervals``: its number of spikes in a null event is
      Poisson(rate x the event's duration), placed uniformly in the event.
      These are the Poisson surrogate data of Silva et al. (2015), Grosmark
      & Buzsáki (2016), Maboudi et al. (2018) and Farooq et al. (2019).

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        The session's spike times, one train per cell, in seconds.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Start and stop of each real candidate event, in seconds.
    kind : {"cell_identity", "spike_shift", "spike_time_uniform", "rate_poisson"}
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
        together, with ``source_events`` naming the real event and
        ``copy_index`` the copy.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the options, ``n_copies`` is not a
        positive integer, ``cells`` is not one boolean per cell, or
        ``intervals`` is not a list of finite ``(start, stop)`` with
        ``start < stop`` that do not overlap.

    See Also
    --------
    simulated_null_events : Simulated events without replay.
    decode_intervals : Decode the null events.

    Notes
    -----
    What each kind keeps and breaks:

    ==================  ======================  ==========================
    kind                keeps                   breaks
    ==================  ======================  ==========================
    cell_identity       every spike train;      which field each train
                        the population's        belongs to, and with it
                        timing                  any link between a cell's
                                                count and its field
    spike_shift         each cell's count and   the timing between cells
                        its spike intervals
                        (circularly)
    spike_time_uniform  each cell's count       all spike timing
    rate_poisson        each cell's mean rate   all spike timing, the
                        over ``intervals``      event's own spike counts
                                                and its population burst
    ==================  ======================  ==========================

    A null along the same dimension as a method's own shuffle
    underestimates its false-positive rate (Takigawa et al. 2024):
    ``cell_identity`` repeats the unit-identity null of ``DAVIDSON_2009``
    and the cell-order test of ``DIBA_BUZSAKI_2007``, and ``spike_shift``
    the spike-train shift of ``TIROLE_2022``. Rank-order methods can pass
    ``spike_shift`` and ``spike_time_uniform`` events more often than
    ``alpha`` when
    an event covers part of the track, because which cells spike more then
    still follows their fields; report several kinds side by side.

    Reassigning spike trains is the same as Takigawa et al.'s
    reassignment of rate maps for one set of rate maps. For a joint decode
    over several templates (tracks or directions) it differs: they
    permuted each template's maps separately, and one permutation of the
    spikes keeps each cell's maps together.

    ``rate_poisson`` does not keep an event's spike count: each cell's
    count is drawn again from its mean rate, so every null event carries
    the session's mean activity, not the event's own burst. Its rate is the
    cell's spikes in all ``intervals`` divided by their summed duration
    (the pooled rate, as in the nelpy code of Maboudi et al.); Silva et
    al.'s "mean firing rate across all candidate events" may also be read
    as the mean of each event's rate. Grosmark & Buzsáki estimated the
    rates separately for each behavioral epoch and running direction: call
    this function once per group of intervals. The papers compared whole
    surrogate data sets with the real one; Grosmark & Buzsáki found that
    Poisson surrogates decode more cleanly than real events, whose
    within-ripple firing is burstier than Poisson.

    Copies of one real event share its spikes and are not independent:
    the intervals of :func:`detection_rates` treat every event as
    independent and are too narrow with ``n_copies > 1``.

    References
    ----------
    Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across
    hippocampal place cells require previous experience. *Nature
    Neuroscience*, 18(12), 1772-1779.

    Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing
    dynamics supports both rigid and learned hippocampal sequences.
    *Science*, 351(6280), 1440-1443.

    Maboudi, K., Ackermann, E., de Jong, L. W., Pfeiffer, B. E., Foster, D.,
    Diba, K., & Kemere, C. (2018). Uncovering temporal structure in
    hippocampal output patterns. *eLife*, 7, e34467.

    Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened
    temporal coordination within pre-existing sequential cell assemblies
    supports trajectory replay. *Neuron*, 103(4), 719-733.

    Stella, F., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019).
    Hippocampal reactivation of random trajectories resembling Brownian
    diffusion. *Neuron*, 102(2), 450-461.

    Takigawa, M., Huelin Gorriz, M., Tirole, M., & Bendor, D. (2024).
    Evaluating hippocampal replay without a ground truth. *eLife*, 13,
    e85635.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import null_events
    >>> spikes = [np.array([1.01, 1.05]), np.array([1.02]), np.array([5.0])]
    >>> null = null_events(
    ...     spikes, np.array([[1.0, 1.1]]), kind="spike_time_uniform", n_copies=3, rng=0
    ... )
    >>> print(null.n_events, [train.size for train in null.spike_times])
    3 [6, 3, 0]

    With ``"rate_poisson"`` each copy draws its counts from the cells'
    mean rates, here over one 100 ms interval: 20 Hz, 10 Hz and 0 Hz.

    >>> null = null_events(
    ...     spikes, np.array([[1.0, 1.1]]), kind="rate_poisson", n_copies=3, rng=0
    ... )
    >>> print([train.size for train in null.spike_times])
    [6, 5, 0]
    """
    from neurospatial._spikes import as_spike_trains

    operation = "null_events()"
    if kind not in _NULL_KINDS:
        raise ValueError(
            f"{operation}: kind must be one of {_NULL_KINDS}, got {kind!r}."
        )
    _positive_count(n_copies, "n_copies", operation)
    windows = _checked_intervals(intervals, operation)
    ordered = windows[np.argsort(windows[:, 0], kind="stable")]
    overlaps = np.flatnonzero(ordered[1:, 0] < ordered[:-1, 1])
    if overlaps.size:
        first = ordered[overlaps[0] : overlaps[0] + 2].tolist()
        raise ValueError(
            _format_error(
                f"{operation}: {overlaps.size} intervals overlap the one before "
                f"them, such as {first[0]} and {first[1]}.",
                why="spikes in the overlap would be copied into two null "
                "events, which are then not independent",
                fix="merge overlapping candidate events, or pass one interval "
                "per event without overlap",
            )
        )
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
    if kind == "rate_poisson":
        # Each cell's pooled rate: its spikes in every interval over their
        # summed duration.
        rates = (
            np.array(
                [
                    sum(
                        np.count_nonzero((train >= start) & (train < stop))
                        for start, stop in windows
                    )
                    for train in trains
                ],
                dtype=np.float64,
            )
            / durations.sum()
        )
    spacing = float(durations.max()) + _GAP
    placed: list[list[NDArray[np.float64]]] = [[] for _ in trains]
    out_intervals = []
    sources = []
    copy_numbers = []
    slot = 0
    for index, (start, stop) in enumerate(windows):
        duration = stop - start
        in_event = [
            np.asarray(train[(train >= start) & (train < stop)]) - start
            for train in trains
        ]
        for copy_number in range(n_copies):
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
            elif kind == "spike_time_uniform":
                for cell in randomized:
                    copies[cell] = generator.uniform(
                        0.0, duration, size=in_event[cell].size
                    )
            else:
                for cell in randomized:
                    copies[cell] = generator.uniform(
                        0.0, duration, size=generator.poisson(rates[cell] * duration)
                    )
            for cell, spikes in enumerate(copies):
                placed[cell].append(np.sort(spikes) + offset)
            out_intervals.append((offset, offset + duration))
            sources.append(index)
            copy_numbers.append(copy_number)
            slot += 1
    return NullSpikeData(
        spike_times=tuple(
            np.concatenate(parts) if parts else np.empty(0) for parts in placed
        ),
        intervals=np.asarray(out_intervals, dtype=np.float64),
        source_events=np.asarray(sources, dtype=np.intp),
        copy_index=np.asarray(copy_numbers, dtype=np.intp),
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
    drawn uniformly along the track (``"stationary"``, which the line
    search of ``detect_trajectory_radon`` scores well but correlation scores
    and the R² of ``fit_linear_trajectory`` do not). Match ``rate_gain`` to the
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
        copy_index=np.zeros(n_events, dtype=np.intp),
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

    Notes
    -----
    Intervals that do not overlap (touching is fine) and a prior with one
    weight per bin are binned in one pass and decoded together, in blocks
    of 50,000 time bins: the time bins are decoded independently, so this
    gives each interval's posterior up to round-off (about 1e-15). On a
    laptop, 5,000 events of 10 time bins, 100 cells and 72 bins took 0.16 s
    this way against 11 s one interval at a time, which matters for the
    many surrogate events of a data-set-level test.
    Overlapping intervals, or a prior per time bin, are decoded one interval
    at a time.

    Every time bin's counts and posterior are returned, and held at once:
    about ``n_time_bins * (n_cells + n_bins) * 8`` bytes. For hundreds of
    surrogate copies of a session's events, decode and measure one copy at
    a time (one :func:`null_events` call per copy, each with its own seed)
    rather than all copies in one call.
    """
    from neurospatial.decoding import bin_spikes_in_time, decode_position

    operation = "decode_intervals()"
    windows = _checked_intervals(intervals, operation)
    ordered = np.argsort(windows[:, 0], kind="stable")
    disjoint = bool(np.all(windows[ordered][1:, 0] >= windows[ordered][:-1, 1]))
    if disjoint and (prior is None or np.ndim(prior) == 1):
        return _decode_intervals_together(
            env, spike_times, windows, encoding_models, dt=dt, prior=prior
        )
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


# Time bins decoded at once by the batched path of decode_intervals.
_DECODE_BLOCK = 50_000


def _decode_intervals_together(
    env: Environment,
    spike_times: Any,
    windows: NDArray[np.float64],
    encoding_models: Any,
    *,
    dt: float,
    prior: NDArray[np.float64] | None,
) -> tuple[list[DecodingResult], list[NDArray[np.int64]]]:
    """:func:`decode_intervals` for disjoint intervals, in one decode.

    The bins are those :func:`~neurospatial.decoding.bin_spikes_in_time`
    makes for each interval alone (each window tiled from its own start),
    counted in one pass, and every bin is decoded in one call to
    :func:`~neurospatial.decoding.decode_position`.
    """
    from neurospatial._spikes import as_spike_trains
    from neurospatial._time_bins import (
        _count_spikes_in_time_bins,
        _time_bins_in_windows,
    )
    from neurospatial.decoding import DecodingResult, decode_position
    from neurospatial.decoding._time_bins import validate_dt

    dt = validate_dt(dt)
    order = np.argsort(windows[:, 0], kind="stable")
    sorted_windows = windows[order]
    left, right = _time_bins_in_windows(sorted_windows, dt)
    per_window = np.bincount(
        np.searchsorted(sorted_windows[:, 0], left, side="right") - 1,
        minlength=len(sorted_windows),
    )
    if np.any(per_window == 0):
        empty = sorted_windows[per_window == 0][0]
        span = float(empty[1] - empty[0])
        raise ValueError(
            _format_error(
                f"decode_intervals(): the interval {empty.tolist()} spans {span} "
                f"s, smaller than one bin dt ({dt}); no whole time bin fits.",
                why="each interval is decoded in whole time bins of width dt",
                fix=f"use dt <= {span:g} or leave out intervals shorter than dt",
            )
        )
    all_counts = _count_spikes_in_time_bins(as_spike_trains(spike_times), left, right)
    decoded = decode_position(
        env,
        all_counts,
        encoding_models,
        dt=dt,
        prior=prior,
        time_chunk=_DECODE_BLOCK,
    ).posterior
    stops = np.cumsum(per_window)
    starts = stops - per_window
    centers = left + dt / 2.0
    events: list[DecodingResult] = [None] * len(windows)  # type: ignore[list-item]
    counts: list[NDArray[np.int64]] = [None] * len(windows)  # type: ignore[list-item]
    for slot, (first, last) in zip(order, zip(starts, stops, strict=True), strict=True):
        events[slot] = DecodingResult(
            posterior=decoded[first:last], env=env, times=centers[first:last]
        )
        counts[slot] = all_counts[first:last].copy()
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
