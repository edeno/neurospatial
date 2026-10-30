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

__all__ = [
    "NullSpikeData",
    "decode_intervals",
    "detection_rates",
    "event_properties",
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
        ``spike_counts`` does not have one entry per event with one row per
        time bin, or has negative or non-integer counts; or if a joint
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
            if counts.ndim == 1:
                counts = counts[:, np.newaxis]
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
    :func:`~neurospatial.replay.compare_replay_methods`.

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
        positive integer, or ``intervals`` is not a list of finite
        ``(start, stop)`` with ``start < stop``.

    See Also
    --------
    simulated_null_events : Simulated events without replay.
    decode_intervals : Decode the null events.

    Notes
    -----
    Reassigning spike trains among cells is the same as reassigning rate
    maps, which Takigawa et al. (2024) did; it keeps each event's
    population activity. Pass the cells the method's decoder uses: a
    reassignment among all recorded cells also moves spikes of cells
    without a field onto place cells.

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
            if kind == "cell_identity":
                order = generator.permutation(len(in_event))
                copies = [in_event[source] for source in order]
            elif kind == "spike_shift":
                copies = [
                    np.mod(spikes + generator.uniform(0.0, duration), duration)
                    for spikes in in_event
                ]
            else:
                copies = [
                    generator.uniform(0.0, duration, size=spikes.size)
                    for spikes in in_event
                ]
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
    jumps at random every 20 ms (``"random"``) or stays put
    (``"stationary"``, which line fits score well but correlation scores
    do not).

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
    from neurospatial.simulation import simulate_replay_event

    operation = "simulated_null_events()"
    if kind not in _SIMULATED_KINDS:
        raise ValueError(
            f"{operation}: kind must be one of {_SIMULATED_KINDS}, got {kind!r}."
        )
    _positive_count(n_events, "n_events", operation)
    generator = np.random.default_rng(rng)
    seeds = generator.integers(2**31, size=n_events)
    spacing = float(duration) + _GAP
    placed: list[list[NDArray[np.float64]]] = []
    for slot, seed in enumerate(seeds):
        event = simulate_replay_event(
            env,
            encoding_models,
            kind=kind,
            duration=duration,
            rate_gain=rate_gain,
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

    A convenience for candidate events and null events alike. Methods that
    decode in overlapping windows need the same decode as the real events;
    decode the null events with it instead.

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
        If an interval is shorter than ``dt``, or for inputs
        :func:`~neurospatial.decoding.decode_position` refuses.

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
    for index, (start, stop) in enumerate(windows):
        if stop - start < dt:
            raise ValueError(
                f"{operation}: interval {index} lasts {stop - start:g} s, shorter "
                f"than dt = {dt:g} s; it has no time bin."
            )
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
