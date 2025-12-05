"""
Interval conversion utilities for neurospatial events.

This module provides functions to convert between point events and intervals:
- intervals_to_events: Convert intervals to point events (start/stop times)
- events_to_intervals: Pair start and stop events into intervals
- filter_by_intervals: Filter events to those within (or outside) intervals
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def intervals_to_events(
    intervals: pd.DataFrame,
    which: Literal["start", "stop", "both"] = "start",
    *,
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
        Output is sorted by timestamp with reset index.

    Raises
    ------
    TypeError
        If intervals is not a DataFrame.
    ValueError
        If required columns are missing, if which is invalid,
        or if preserve_columns contains non-existent columns.

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
            "  HOW: Convert using pd.DataFrame({'start_time': starts, 'stop_time': stops})"
        )

    # Validate which parameter
    valid_which = {"start", "stop", "both"}
    if which not in valid_which:
        raise ValueError(
            f"'which' must be one of {valid_which}, got '{which}'.\n"
            "  WHY: Only 'start', 'stop', or 'both' are valid options.\n"
            "  HOW: Use which='start' to extract start times only."
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
            f"  HOW: Ensure intervals DataFrame has columns: {required_cols}.\n"
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
                f"  HOW: Check column names. Available: {list(intervals.columns)}"
            )

    # Handle empty intervals
    if len(intervals) == 0:
        columns = ["timestamp"]
        if which == "both":
            columns.append("boundary")
        if preserve_columns:
            columns.extend(preserve_columns)
        return pd.DataFrame(columns=columns)

    # Extract events based on 'which'
    if which == "start":
        result = pd.DataFrame({"timestamp": intervals[start_column].values})
        if preserve_columns:
            for col in preserve_columns:
                result[col] = intervals[col].values

    elif which == "stop":
        result = pd.DataFrame({"timestamp": intervals[stop_column].values})
        if preserve_columns:
            for col in preserve_columns:
                result[col] = intervals[col].values

    else:  # which == "both"
        # Create start events
        start_df = pd.DataFrame(
            {
                "timestamp": intervals[start_column].values,
                "boundary": "start",
            }
        )
        # Create stop events
        stop_df = pd.DataFrame(
            {
                "timestamp": intervals[stop_column].values,
                "boundary": "stop",
            }
        )

        # Add preserved columns to both
        if preserve_columns:
            for col in preserve_columns:
                start_df[col] = intervals[col].values
                stop_df[col] = intervals[col].values

        # Combine and sort
        result = pd.concat([start_df, stop_df], ignore_index=True)

    # Sort by timestamp and reset index
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


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
        Maximum interval duration. Intervals longer than this are excluded.

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
    ValueError
        If event counts don't match (when match_by is None),
        if match_by column is missing from either DataFrame,
        or if there are unmatched values in the match_by column.

    Warns
    -----
    UserWarning
        If any interval has negative duration (stop before start).

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

    # Get number of events
    n_start = len(start_events)
    n_stop = len(stop_events)

    # Handle empty case
    if n_start == 0 and n_stop == 0:
        return pd.DataFrame(columns=["start_time", "stop_time", "duration"])

    if match_by is None:
        # Sequential pairing: require matching counts
        if n_start != n_stop:
            raise ValueError(
                f"Event counts don't match: {n_start} starts, {n_stop} stops.\n"
                "  WHY: Sequential pairing requires equal numbers of start/stop events.\n"
                "  HOW: Use match_by='column_name' for non-sequential pairing,\n"
                "       or ensure equal numbers of start and stop events."
            )

        # Sort both by timestamp
        start_sorted = start_events.sort_values("timestamp").reset_index(drop=True)
        stop_sorted = stop_events.sort_values("timestamp").reset_index(drop=True)

        # Create intervals
        result = pd.DataFrame(
            {
                "start_time": start_sorted["timestamp"].values,
                "stop_time": stop_sorted["timestamp"].values,
            }
        )

        # Preserve additional columns from start_events
        for col in start_sorted.columns:
            if col != "timestamp" and col not in result.columns:
                result[col] = start_sorted[col].values

    else:
        # Match by column value
        if match_by not in start_events.columns:
            raise ValueError(
                f"Column '{match_by}' not found in start_events.\n"
                f"  WHY: Cannot match by a column that doesn't exist.\n"
                f"  Available columns: {list(start_events.columns)}"
            )
        if match_by not in stop_events.columns:
            raise ValueError(
                f"Column '{match_by}' not found in stop_events.\n"
                f"  WHY: Cannot match by a column that doesn't exist.\n"
                f"  Available columns: {list(stop_events.columns)}"
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
                "  HOW: Ensure all values in '{match_by}' column have both start and stop."
            )

        # Merge on match_by column
        start_df = start_events.copy()
        stop_df = stop_events[["timestamp", match_by]].copy()
        stop_df = stop_df.rename(columns={"timestamp": "stop_time"})

        merged = start_df.merge(stop_df, on=match_by, how="inner")

        result = pd.DataFrame(
            {
                "start_time": merged["timestamp"].values,
                "stop_time": merged["stop_time"].values,
            }
        )

        # Preserve additional columns from start_events
        for col in merged.columns:
            if col not in ["timestamp", "stop_time"] and col not in result.columns:
                result[col] = merged[col].values

    # Calculate duration
    result["duration"] = result["stop_time"] - result["start_time"]

    # Warn about negative durations
    if (result["duration"] < 0).any():
        n_negative = (result["duration"] < 0).sum()
        warnings.warn(
            f"Found {n_negative} interval(s) with negative duration "
            "(stop time before start time).",
            UserWarning,
            stacklevel=2,
        )

    # Filter by max_duration if specified
    if max_duration is not None:
        result = result[result["duration"] <= max_duration].copy()

    # Sort by start_time and reset index
    result = result.sort_values("start_time").reset_index(drop=True)

    return result


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
        If True, keep events within intervals (inclusive filtering).
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
    ValueError
        If required columns are missing from events or intervals.

    See Also
    --------
    intervals_to_events : Convert intervals to point events.
    events_to_intervals : Pair start/stop events into intervals.

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

    # Validate events has timestamp column
    if timestamp_column not in events.columns:
        raise ValueError(
            f"Events DataFrame missing '{timestamp_column}' column.\n"
            "  WHY: Need timestamps to check interval membership.\n"
            f"  HOW: Ensure events has '{timestamp_column}' column.\n"
            f"  Available columns: {list(events.columns)}"
        )

    # Validate intervals has required columns
    for col, name in [(start_column, "start"), (stop_column, "stop")]:
        if col not in intervals.columns:
            raise ValueError(
                f"Intervals DataFrame missing '{col}' column.\n"
                f"  WHY: Need {name} times to define intervals.\n"
                f"  HOW: Ensure intervals has '{col}' column.\n"
                f"  Available columns: {list(intervals.columns)}"
            )

    # Handle empty cases
    if len(events) == 0:
        return events.copy()

    if len(intervals) == 0:
        if include:
            # No intervals to include from → return empty
            return events.iloc[0:0].copy()
        else:
            # No intervals to exclude → return all
            return events.copy()

    # Get event timestamps
    timestamps = events[timestamp_column].values

    # Create mask for events within any interval
    # Using vectorized comparison with broadcasting
    starts = intervals[start_column].values
    stops = intervals[stop_column].values

    # For each event, check if it's within any interval
    # Shape: (n_events, n_intervals)
    within_intervals = (timestamps[:, np.newaxis] >= starts) & (
        timestamps[:, np.newaxis] <= stops
    )

    # Event is "in an interval" if it's within any interval
    in_any_interval = within_intervals.any(axis=1)

    # Apply filter based on include flag
    mask = in_any_interval if include else ~in_any_interval

    return events[mask].copy()
