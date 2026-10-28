"""
Interval conversion utilities for neurospatial events.

This module provides functions to convert between point events and intervals:
- intervals_to_events: Convert intervals to point events (start/stop times)
- events_to_intervals: Pair start and stop events into intervals
- filter_by_intervals: Filter events to those within (or outside) intervals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from neurospatial._intervals import _format_rows, _merge, times_in_intervals
from neurospatial._validation import validate_finite_scalar
from neurospatial.events._core import _event_times

if TYPE_CHECKING:
    import pandas as pd


def intervals_to_events(
    intervals: pd.DataFrame,
    *,
    which: Literal["start", "stop", "both"] = "start",
    start_column: str = "start_time",
    stop_column: str = "stop_time",
    preserve_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert intervals to point events (start and/or stop times).

    This function extracts timestamps from interval boundaries, creating
    a point event DataFrame suitable for peri-event analysis or other
    event-based workflows.

    Parameters
    ----------
    intervals : pd.DataFrame
        Intervals DataFrame with start and stop time columns.
    which : {"start", "stop", "both"}, default="start"
        Which interval boundaries to extract as events:
        - "start": Extract only start times
        - "stop": Extract only stop times
        - "both": Extract both, with a 'boundary' column indicating type
    start_column : str, default="start_time"
        Name of the start time column in intervals.
    stop_column : str, default="stop_time"
        Name of the stop time column in intervals.
    preserve_columns : list[str], optional
        Additional columns from intervals to preserve in output.
        These columns are duplicated for both boundaries when which="both".

    Returns
    -------
    pd.DataFrame
        Events DataFrame with 'timestamp' column. When which="both",
        includes a 'boundary' column with values "start" or "stop".
        Output is sorted by timestamp with reset index; at equal times a stop
        comes before a start, so abutting intervals read stop, start.

    Raises
    ------
    TypeError
        If intervals is not a DataFrame.
    ValueError
        If required columns are missing, not numeric, or hold NaN or inf, if
        which is invalid, if preserve_columns contains non-existent columns,
        or (``which="both"``) if an interval stops before it starts.

    See Also
    --------
    events_to_intervals : Pair start/stop events into intervals.
    filter_by_intervals : Filter events by time intervals.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.events import intervals_to_events

    Extract trial start times for PSTH alignment:

    >>> trials = pd.DataFrame(
    ...     {
    ...         "start_time": [0.0, 10.0, 20.0],
    ...         "stop_time": [8.0, 18.0, 28.0],
    ...         "trial_type": ["go", "nogo", "go"],
    ...     }
    ... )
    >>> trial_starts = intervals_to_events(
    ...     trials, which="start", preserve_columns=["trial_type"]
    ... )
    >>> print(trial_starts["timestamp"].values)
    [ 0. 10. 20.]

    Extract both boundaries with labels:

    >>> events = intervals_to_events(trials, which="both")
    >>> print(events[["timestamp", "boundary"]].head())
       timestamp boundary
    0        0.0    start
    1        8.0     stop
    2       10.0    start
    3       18.0     stop
    4       20.0    start
    """
    import pandas as pd

    # Validate input type
    if not isinstance(intervals, pd.DataFrame):
        raise TypeError(
            f"Expected pd.DataFrame, got {type(intervals).__name__}.\n"
            "  WHY: Intervals must be a pandas DataFrame.\n"
            "  Fix: Convert using pd.DataFrame({'start_time': starts, 'stop_time': stops})"
        )

    # Validate which parameter
    valid_which = {"start", "stop", "both"}
    if which not in valid_which:
        raise ValueError(
            f"'which' must be one of {valid_which}, got '{which}'.\n"
            "  WHY: Only 'start', 'stop', or 'both' are valid options.\n"
            "  Fix: Use which='start' to extract start times only."
        )

    # Validate required columns based on 'which'
    required_cols = []
    if which in ("start", "both"):
        required_cols.append(start_column)
    if which in ("stop", "both"):
        required_cols.append(stop_column)

    missing = [col for col in required_cols if col not in intervals.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.\n"
            f"  WHY: These columns are needed to extract '{which}' events.\n"
            f"  Fix: Ensure intervals DataFrame has columns: {required_cols}.\n"
            f"  Available columns: {list(intervals.columns)}"
        )

    # Validate preserve_columns
    if preserve_columns:
        missing_preserve = [
            col for col in preserve_columns if col not in intervals.columns
        ]
        if missing_preserve:
            raise ValueError(
                f"Columns to preserve not found: {missing_preserve}.\n"
                "  WHY: Cannot preserve columns that don't exist in intervals.\n"
                f"  Fix: Check column names. Available: {list(intervals.columns)}"
            )

    used = [start_column] if which == "start" else [stop_column]
    if which == "both":
        used = [start_column, stop_column]
    times = {
        col: _event_times(intervals, col, name="intervals", call="intervals_to_events")
        for col in used
    }
    if which == "both":
        _refuse_inverted(times[start_column], times[stop_column], "intervals_to_events")

    def boundary_events(column: str) -> pd.DataFrame:
        frame = pd.DataFrame({"timestamp": times[column]})
        if which == "both":
            frame["boundary"] = "start" if column == start_column else "stop"
        for col in preserve_columns or []:
            frame[col] = intervals[col].to_numpy()
        return frame

    result = pd.concat([boundary_events(col) for col in used], ignore_index=True)
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
        raise ValueError(
            f"{call}: interval row(s) {_format_rows(labels)} stop before they "
            f"start (e.g. start {starts[k]:g} s, stop {stops[k]:g} s).\n"
            "  WHY: an interval [start, stop) needs stop >= start; a negative "
            "duration means a start and stop were mismatched.\n"
            "  Fix: repair those rows, or pair starts and stops by an id with "
            "match_by=."
        )


def events_to_intervals(
    start_events: pd.DataFrame,
    stop_events: pd.DataFrame,
    *,
    match_by: str | None = None,
    max_duration: float | None = None,
) -> pd.DataFrame:
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
    pd.DataFrame
        Intervals with columns:
        - 'start_time': Start timestamp
        - 'stop_time': Stop timestamp
        - 'duration': Interval duration (stop - start)
        - Additional columns from start_events are preserved

        Output is sorted by start_time.

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
    >>> print(dwell["duration"].values)
    [5. 5. 8.]

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
    >>> print(intervals[["start_time", "stop_time", "duration"]].values)
    [[1. 4. 3.]
     [2. 5. 3.]
     [3. 6. 3.]]
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
            raise ValueError(
                f"Event counts don't match: {n_start} starts, {n_stop} stops.\n"
                "  WHY: Sequential pairing requires equal numbers of start/stop events.\n"
                "  Fix: Use match_by='column_name' for non-sequential pairing,\n"
                "       or ensure equal numbers of start and stop events."
            )
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
                raise ValueError(
                    f"Column '{match_by}' not found in {name}.\n"
                    f"  WHY: Cannot match by a column that doesn't exist.\n"
                    f"  Available columns: {list(frame.columns)}"
                )
            if frame[match_by].isna().any():
                raise ValueError(
                    f"{call}: {name}[{match_by!r}] has missing values.\n"
                    "  WHY: a row without an id cannot be paired.\n"
                    f"  Fix: drop or label those rows before pairing by {match_by!r}."
                )

        # Check for unmatched values
        start_values = set(start_events[match_by])
        stop_values = set(stop_events[match_by])
        unmatched_start = start_values - stop_values
        unmatched_stop = stop_values - start_values

        if unmatched_start or unmatched_stop:
            raise ValueError(
                f"Found unmatched values in '{match_by}' column.\n"
                f"  Start events with no matching stop: {unmatched_start}\n"
                f"  Stop events with no matching start: {unmatched_stop}\n"
                "  WHY: Each start event needs a matching stop event.\n"
                f"  Fix: Ensure all values in '{match_by}' column have both start and stop."
            )

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
            raise ValueError(
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

        stop_df = stop_events[["timestamp", match_by]].rename(
            columns={"timestamp": "stop_time"}
        )
        merged = start_events.merge(stop_df, on=match_by, how="inner")
        merged_starts = merged["timestamp"].to_numpy(dtype=np.float64)
        merged_stops = merged["stop_time"].to_numpy(dtype=np.float64)
        _refuse_inverted(merged_starts, merged_stops, call, rows=np.arange(len(merged)))
        result = pd.DataFrame({"start_time": merged_starts, "stop_time": merged_stops})
        extra = merged.drop(columns=["timestamp", "stop_time"])

    # Preserve additional columns from start_events
    for col in extra.columns:
        if col not in result.columns:
            result[col] = extra[col].to_numpy()
    result.insert(2, "duration", result["stop_time"] - result["start_time"])

    if max_duration is not None:
        result = result[result["duration"] <= max_duration]

    return result.sort_values("start_time", kind="stable").reset_index(drop=True)


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
    raise ValueError(
        f"events_to_intervals: start and stop events do not alternate in time: "
        f"{problem} (pair {k} in time order).\n"
        "  WHY: sequential pairing takes the k-th start with the k-th stop, so a "
        "missing or extra event pairs every later event with the wrong partner.\n"
        "  Fix: repair the missing or extra event near that time, or pair by an "
        "id column with match_by=."
    )


def filter_by_intervals(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    include: bool = True,
    timestamp_column: str = "timestamp",
    start_column: str = "start_time",
    stop_column: str = "stop_time",
) -> pd.DataFrame:
    """
    Filter events to those within (or outside) intervals.

    This function filters an events DataFrame based on whether each event's
    timestamp falls within any of the specified intervals.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame to filter. Must have timestamp column.
    intervals : pd.DataFrame
        Intervals DataFrame with start and stop time columns.
    include : bool, default=True
        If True, keep events within intervals, half-open
        ``[start, stop)``: an event exactly at a stop is outside.
        If False, keep events outside intervals (exclusive filtering).
    timestamp_column : str, default="timestamp"
        Name of timestamp column in events.
    start_column : str, default="start_time"
        Name of start time column in intervals.
    stop_column : str, default="stop_time"
        Name of stop time column in intervals.

    Returns
    -------
    pd.DataFrame
        Filtered events DataFrame with original index preserved.

    Raises
    ------
    TypeError
        If events or intervals is not a DataFrame.
    ValueError
        If a time column is missing, not numeric, or holds NaN or inf, or if
        an interval stops before it starts. A missing event time is neither
        inside nor outside an interval, so it is refused in both modes.

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
    >>> from neurospatial.events import filter_by_intervals

    Keep only events during trial periods:

    >>> events = pd.DataFrame({"timestamp": [5.0, 15.0, 25.0, 35.0]})
    >>> trials = pd.DataFrame(
    ...     {
    ...         "start_time": [0.0, 20.0],
    ...         "stop_time": [10.0, 30.0],
    ...     }
    ... )
    >>> trial_events = filter_by_intervals(events, trials, include=True)
    >>> print(trial_events["timestamp"].values)
    [ 5. 25.]

    Remove events during artifact periods:

    >>> artifacts = pd.DataFrame(
    ...     {
    ...         "start_time": [10.0],
    ...         "stop_time": [20.0],
    ...     }
    ... )
    >>> clean_events = filter_by_intervals(events, artifacts, include=False)
    >>> print(clean_events["timestamp"].values)
    [ 5. 25. 35.]
    """

    call = "filter_by_intervals"
    timestamps = _event_times(events, timestamp_column, name="events", call=call)
    starts = _event_times(intervals, start_column, name="intervals", call=call)
    stops = _event_times(intervals, stop_column, name="intervals", call=call)
    _refuse_inverted(starts, stops, call)

    if not (stops > starts).any():
        in_any_interval = np.zeros(timestamps.size, dtype=bool)
    else:
        # Zero-length rows hold no time; merging the rest makes membership one
        # binary search per event.
        rows = np.column_stack([starts, stops])[stops > starts]
        in_any_interval = times_in_intervals(_merge(rows), timestamps)

    mask = in_any_interval if include else ~in_any_interval
    return events[mask].copy()
