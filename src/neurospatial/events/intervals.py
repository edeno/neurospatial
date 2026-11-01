"""
Interval conversion utilities for neurospatial events.

This module provides functions to convert between point events and intervals:
- intervals_to_events: Convert intervals to point events (start/stop times)
- events_to_intervals: Pair start and stop events into intervals
- filter_by_intervals: Filter events to those within (or outside) intervals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._intervals import _format_rows, _merge, times_in_intervals
from neurospatial._pynapple import intervals_from_interval_set, is_pynapple_type
from neurospatial._validation import validate_finite_scalar
from neurospatial.events._core import _event_times

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._interval_table import Interval, Intervals


def _as_intervals(intervals: Any, *, call: str) -> Intervals[Any]:
    """``intervals`` as :class:`~neurospatial.Intervals`; a pynapple set converted."""
    from neurospatial._interval_table import Intervals

    if isinstance(intervals, Intervals):
        return intervals
    if is_pynapple_type(intervals, "IntervalSet"):
        converted: Intervals[Any] = intervals_from_interval_set(intervals)
        return converted
    hint = (
        "wrap the table with Intervals.from_dataframe(table) (columns start_time, "
        "stop_time, then any fields)"
        if hasattr(intervals, "columns")
        else "pass an Intervals, for example from Intervals.from_dataframe or a "
        "detector"
    )
    raise TypeError(
        _format_error(
            f"{call}: intervals must be an Intervals or a pynapple IntervalSet, "
            f"got {type(intervals).__name__}.",
            why="intervals are half-open [start, stop) rows with named fields and "
            "stable ids, whatever their source.",
            fix=hint + ".",
        )
    )


def intervals_to_events(
    intervals: Intervals[Any] | Any,
    *,
    which: Literal["start", "stop", "both"] = "start",
    preserve_fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert intervals to point events at their starts and/or stops.

    For peri-event analysis aligned to trial starts, arrivals (trial stops)
    or both.

    Parameters
    ----------
    intervals : Intervals or IntervalSet
        Trials or other intervals, for example from ``detect_trials``,
        ``read_trials`` or ``Intervals.from_dataframe``. A pynapple
        ``IntervalSet`` is converted with its metadata as fields; its
        inclusive ``end`` becomes the half-open stop, the next float above
        ``end``.
    which : {"start", "stop", "both"}, default="start"
        Which interval boundaries to extract as events:
        - "start": Extract only start times
        - "stop": Extract only stop times
        - "both": Extract both, with a 'boundary' column indicating type
    preserve_fields : list[str], optional
        Fields of the intervals to copy onto their events (duplicated for
        both boundaries when ``which="both"``).

    Returns
    -------
    pd.DataFrame
        Events with a ``timestamp`` column, the ``interval_id`` each event
        came from, a ``boundary`` column (``"start"`` or ``"stop"``) when
        ``which="both"``, and the preserved fields. Sorted by timestamp with a
        reset index; at equal times a stop comes before a start, so abutting
        intervals read stop, start.

    Raises
    ------
    TypeError
        If intervals is not an Intervals or IntervalSet (a DataFrame: wrap it
        with ``Intervals.from_dataframe``).
    ValueError
        If which is invalid or preserve_fields names a field the intervals
        do not have.

    See Also
    --------
    events_to_intervals : Pair start/stop events into intervals.
    filter_by_intervals : Filter events by time intervals.

    Examples
    --------
    >>> from neurospatial import Intervals
    >>> from neurospatial.events import intervals_to_events

    Extract trial start times for PSTH alignment:

    >>> trials = Intervals.from_arrays(
    ...     [0.0, 10.0, 20.0], [8.0, 18.0, 28.0], trial_type=["go", "nogo", "go"]
    ... )
    >>> trial_starts = intervals_to_events(
    ...     trials, which="start", preserve_fields=["trial_type"]
    ... )
    >>> trial_starts["timestamp"].tolist(), trial_starts["trial_type"].tolist()
    ([0.0, 10.0, 20.0], ['go', 'nogo', 'go'])

    Extract both boundaries with labels:

    >>> events = intervals_to_events(trials, which="both")
    >>> events["boundary"].tolist()[:4]
    ['start', 'stop', 'start', 'stop']
    """
    import pandas as pd

    call = "intervals_to_events"
    rows = _as_intervals(intervals, call=call)
    valid_which = ("start", "stop", "both")
    if which not in valid_which:
        msg = (
            f"'which' must be one of {valid_which}, got '{which}'.\n"
            "  WHY: Only 'start', 'stop', or 'both' are valid options.\n"
            "  Fix: Use which='start' to extract start times only."
        )
        raise ValueError(msg)
    preserved = list(preserve_fields or [])
    missing = [name for name in preserved if name not in rows.fields]
    if missing:
        msg = (
            f"{call}: fields to preserve not found: {missing}.\n"
            "  WHY: only fields of the intervals can be copied onto events.\n"
            f"  Fix: choose from {list(rows.fields)}."
        )
        raise ValueError(msg)

    used = ["start", "stop"] if which == "both" else [which]

    def boundary_events(boundary: str) -> pd.DataFrame:
        times = rows.start if boundary == "start" else rows.stop
        frame = pd.DataFrame(
            {"timestamp": np.array(times), "interval_id": np.array(rows.interval_ids)}
        )
        if which == "both":
            frame["boundary"] = boundary
        for name in preserved:
            frame[name] = np.array(rows._field_array(name))
        return frame

    result = pd.concat([boundary_events(b) for b in used], ignore_index=True)
    # A stop sorts before a start at the same time, so abutting intervals read
    # stop, start; equal times otherwise keep their input order.
    is_start = (
        result["boundary"].to_numpy() == "start"
        if which == "both"
        else np.zeros(len(result), dtype=bool)
    )
    order = np.lexsort((is_start, result["timestamp"].to_numpy()))
    return result.iloc[order].reset_index(drop=True)


def _refuse_inverted(
    starts: np.ndarray, stops: np.ndarray, call: str, *, rows: np.ndarray | None = None
) -> None:
    """Raise if any interval stops before it starts."""
    inverted = np.flatnonzero(stops < starts)
    if inverted.size:
        labels = inverted if rows is None else rows[inverted]
        k = inverted[0]
        msg = (
            f"{call}: interval row(s) {_format_rows(labels)} stop before they "
            f"start (e.g. start {starts[k]:g} s, stop {stops[k]:g} s).\n"
            "  WHY: an interval [start, stop) needs stop >= start; a negative "
            "duration means a start and stop were mismatched.\n"
            "  Fix: repair those rows, or pair starts and stops by an id with "
            "match_by=."
        )
        raise ValueError(msg)


def events_to_intervals(
    start_events: pd.DataFrame,
    stop_events: pd.DataFrame,
    *,
    match_by: str | None = None,
    max_duration: float | None = None,
) -> Intervals[Interval]:
    """
    Pair start and stop events into intervals.

    This function pairs start events with corresponding stop events to
    create interval data. Events can be paired sequentially (first start
    with first stop) or by matching a column value (e.g., trial_id).

    Parameters
    ----------
    start_events : pd.DataFrame
        Events marking interval starts. Must have 'timestamp' column.
    stop_events : pd.DataFrame
        Events marking interval stops. Must have 'timestamp' column.
    match_by : str, optional
        Column to match start/stop events by (e.g., "trial_id").
        If None, pairs sequentially (first start with first stop, etc.).
    max_duration : float, optional
        Maximum interval duration in seconds. Intervals longer than this are
        excluded.

    Returns
    -------
    Intervals of Interval
        One row per pair, ``[start, stop)``, sorted by start, ids ``0`` to
        ``n - 1`` in that order. The other columns of ``start_events``
        (including ``match_by``) become fields; ``durations`` gives
        ``stop - start``.

    Raises
    ------
    TypeError
        If start_events or stop_events is not a DataFrame.
    ValueError
        If a 'timestamp' column is missing, not numeric, or holds NaN or inf;
        if ``max_duration`` is negative or not finite.
        Sequential pairing (``match_by=None``): if event counts don't match,
        or if the sorted starts and stops do not alternate
        (``start[k] <= stop[k] <= start[k + 1]``), which means an event is
        missing or extra.
        Matching (``match_by=...``): if the column is missing or has missing
        values, if a value is unmatched or duplicated (one-to-one pairing is
        required), or if a pair stops before it starts.

    See Also
    --------
    intervals_to_events : Convert intervals to point events.
    filter_by_intervals : Filter events by time intervals.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.events import events_to_intervals

    Sequential pairing of zone entries and exits:

    >>> entries = pd.DataFrame({"timestamp": [10.0, 30.0, 50.0]})
    >>> exits = pd.DataFrame({"timestamp": [15.0, 35.0, 58.0]})
    >>> dwell = events_to_intervals(entries, exits)
    >>> dwell.durations.tolist()
    [5.0, 5.0, 8.0]

    Match by trial ID (handles out-of-order events):

    >>> starts = pd.DataFrame(
    ...     {
    ...         "timestamp": [1.0, 2.0, 3.0],
    ...         "trial_id": [1, 2, 3],
    ...     }
    ... )
    >>> stops = pd.DataFrame(
    ...     {
    ...         "timestamp": [5.0, 4.0, 6.0],
    ...         "trial_id": [2, 1, 3],
    ...     }
    ... )
    >>> intervals = events_to_intervals(starts, stops, match_by="trial_id")
    >>> intervals.stop.tolist(), intervals.trial_id.tolist()
    ([4.0, 5.0, 6.0], [1, 2, 3])
    """
    import pandas as pd

    call = "events_to_intervals"
    start_times = _event_times(
        start_events, "timestamp", name="start_events", call=call
    )
    stop_times = _event_times(stop_events, "timestamp", name="stop_events", call=call)
    validate_finite_scalar(
        max_duration, name="max_duration", positive=False, none_allowed=True
    )
    n_start, n_stop = start_times.size, stop_times.size

    if match_by is None:
        # Sequential pairing: require matching counts
        if n_start != n_stop:
            msg = (
                f"Event counts don't match: {n_start} starts, {n_stop} stops.\n"
                "  WHY: Sequential pairing requires equal numbers of start/stop events.\n"
                "  Fix: Use match_by='column_name' for non-sequential pairing,\n"
                "       or ensure equal numbers of start and stop events."
            )
            raise ValueError(msg)
        start_order = np.argsort(start_times, kind="stable")
        starts = start_times[start_order]
        stops = np.sort(stop_times, kind="stable")
        _refuse_unpaired_sequence(starts, stops)
        result = pd.DataFrame({"start_time": starts, "stop_time": stops})
        extra = start_events.iloc[start_order].drop(columns="timestamp")
    else:
        for frame, name in (
            (start_events, "start_events"),
            (stop_events, "stop_events"),
        ):
            if match_by not in frame.columns:
                msg = (
                    f"Column '{match_by}' not found in {name}.\n"
                    f"  WHY: Cannot match by a column that doesn't exist.\n"
                    f"  Available columns: {list(frame.columns)}"
                )
                raise ValueError(msg)
            if frame[match_by].isna().any():
                msg = (
                    f"{call}: {name}[{match_by!r}] has missing values.\n"
                    "  WHY: a row without an id cannot be paired.\n"
                    f"  Fix: drop or label those rows before pairing by {match_by!r}."
                )
                raise ValueError(msg)

        # Check for unmatched values
        start_values = set(start_events[match_by])
        stop_values = set(stop_events[match_by])
        unmatched_start = start_values - stop_values
        unmatched_stop = stop_values - start_values

        if unmatched_start or unmatched_stop:
            msg = (
                f"Found unmatched values in '{match_by}' column.\n"
                f"  Start events with no matching stop: {unmatched_start}\n"
                f"  Stop events with no matching start: {unmatched_stop}\n"
                "  WHY: Each start event needs a matching stop event.\n"
                f"  Fix: Ensure all values in '{match_by}' column have both start and stop."
            )
            raise ValueError(msg)

        # Each match_by value must appear at most once on each side. Without
        # this, DataFrame.merge(how="inner") forms a Cartesian product for any
        # duplicated key, silently inflating the interval count (e.g. two
        # starts and two stops sharing a key yield four intervals, not two).
        start_dups = start_events[match_by][
            start_events[match_by].duplicated(keep=False)
        ].unique()
        stop_dups = stop_events[match_by][
            stop_events[match_by].duplicated(keep=False)
        ].unique()
        if len(start_dups) > 0 or len(stop_dups) > 0:
            msg = (
                f"Duplicate values in '{match_by}' column prevent one-to-one "
                f"start/stop pairing.\n"
                f"  Duplicated in start_events: {start_dups.tolist()}\n"
                f"  Duplicated in stop_events: {stop_dups.tolist()}\n"
                "  WHY: Matching by a non-unique key would cross-join "
                "(Cartesian product) every repeated value, producing more "
                "intervals than real start/stop pairs.\n"
                "  Fix: De-duplicate so each value of "
                f"'{match_by}' appears once per side, or use match_by=None "
                "for sequential pairing of equal-length start/stop events."
            )
            raise ValueError(msg)

        stop_df = stop_events[["timestamp", match_by]].rename(
            columns={"timestamp": "stop_time"}
        )
        merged = start_events.merge(stop_df, on=match_by, how="inner")
        merged_starts = merged["timestamp"].to_numpy(dtype=np.float64)
        merged_stops = merged["stop_time"].to_numpy(dtype=np.float64)
        _refuse_inverted(merged_starts, merged_stops, call, rows=np.arange(len(merged)))
        result = pd.DataFrame({"start_time": merged_starts, "stop_time": merged_stops})
        extra = merged.drop(columns=["timestamp", "stop_time"])

    from neurospatial._interval_table import Intervals

    # Preserve additional columns from start_events as fields.
    for col in extra.columns:
        if col not in result.columns:
            result[col] = extra[col].to_numpy()
    if max_duration is not None:
        result = result[result["stop_time"] - result["start_time"] <= max_duration]
    result = result.sort_values("start_time", kind="stable").reset_index(drop=True)
    return Intervals.from_dataframe(result)


def _refuse_unpaired_sequence(starts: np.ndarray, stops: np.ndarray) -> None:
    """Raise unless sorted starts and stops alternate, start <= stop <= next start.

    Pairing the k-th start with the k-th stop is right only when they
    alternate; a missed and an extra event elsewhere keep the counts equal and
    otherwise pair every event in between with the wrong partner.
    """
    before = np.flatnonzero(stops < starts)
    after = np.flatnonzero(stops[:-1] > starts[1:])
    if not (before.size or after.size):
        return
    k = int(min(before[:1].tolist() + after[:1].tolist()))
    if before.size and before[0] == k:
        problem = f"stop {stops[k]:g} s comes before its start {starts[k]:g} s"
    else:
        problem = (
            f"start {starts[k + 1]:g} s comes before stop {stops[k]:g} s of the "
            f"start at {starts[k]:g} s"
        )
    msg = (
        f"events_to_intervals: start and stop events do not alternate in time: "
        f"{problem} (pair {k} in time order).\n"
        "  WHY: sequential pairing takes the k-th start with the k-th stop, so a "
        "missing or extra event pairs every later event with the wrong partner.\n"
        "  Fix: repair the missing or extra event near that time, or pair by an "
        "id column with match_by=."
    )
    raise ValueError(msg)


def filter_by_intervals(
    events: pd.DataFrame,
    intervals: Intervals[Any] | Any,
    *,
    include: bool = True,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Filter events to those within (or outside) intervals.

    This function filters an events DataFrame based on whether each event's
    timestamp falls within any of the specified intervals.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame to filter. Must have timestamp column.
    intervals : Intervals or IntervalSet
        Trials or other intervals, for example from ``detect_trials`` or
        ``Intervals.from_dataframe``. A pynapple ``IntervalSet`` is converted;
        its inclusive ends keep an event at ``end`` inside, as pynapple's
        ``restrict`` does.
    include : bool, default=True
        If True, keep events within intervals, half-open
        ``[start, stop)``: an event exactly at a stop is outside.
        If False, keep events outside intervals (exclusive filtering).
    timestamp_column : str, default="timestamp"
        Name of timestamp column in events.

    Returns
    -------
    pd.DataFrame
        Filtered events DataFrame with original index preserved.

    Raises
    ------
    TypeError
        If events is not a DataFrame, or intervals is not an Intervals or
        IntervalSet (a DataFrame: wrap it with ``Intervals.from_dataframe``).
    ValueError
        If the timestamp column is missing, not numeric, or holds NaN or inf.
        A missing event time is neither inside nor outside an interval, so it
        is refused in both modes.

    See Also
    --------
    intervals_to_events : Convert intervals to point events.
    events_to_intervals : Pair start/stop events into intervals.
    neurospatial.behavior.restrict : Restrict time arrays and aligned arrays
        (positions, speed) to epochs.
    neurospatial.behavior.in_epochs : Boolean mask of the timestamps inside
        epochs.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial import Intervals
    >>> from neurospatial.events import filter_by_intervals

    Keep only events during trial periods:

    >>> events = pd.DataFrame({"timestamp": [5.0, 15.0, 25.0, 35.0]})
    >>> trials = Intervals.from_arrays([0.0, 20.0], [10.0, 30.0])
    >>> filter_by_intervals(events, trials)["timestamp"].tolist()
    [5.0, 25.0]

    Remove events during artifact periods:

    >>> artifacts = Intervals.from_arrays([10.0], [20.0])
    >>> filter_by_intervals(events, artifacts, include=False)["timestamp"].tolist()
    [5.0, 25.0, 35.0]
    """
    call = "filter_by_intervals"
    rows = _as_intervals(intervals, call=call)
    timestamps = _event_times(events, timestamp_column, name="events", call=call)
    windows = rows.windows
    if windows.shape[0] == 0:
        in_any_interval = np.zeros(timestamps.size, dtype=bool)
    else:
        # Zero-length rows hold no time; merging the rest makes membership one
        # binary search per event.
        in_any_interval = times_in_intervals(_merge(windows), timestamps)

    mask = in_any_interval if include else ~in_any_interval
    return events[mask].copy()
