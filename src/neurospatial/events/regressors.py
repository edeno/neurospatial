"""
GLM regressor generation for neurospatial events.

This module provides functions to generate regressors for GLM design matrices:

Temporal regressors:
- time_to_nearest_event: Signed time to nearest event (peri-event time)
- event_count_in_window: Count events in time window
- event_indicator: Binary indicator of event presence
- exponential_kernel: Convolve events with exponential kernel

Spatial regressors:
- distance_to_event_at_time: Distance to last/next event location
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_to_nearest_event(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    signed: bool = True,
    max_time: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute time to nearest event for each sample.

    Returns signed time relative to nearest event: negative values indicate
    time before the event, positive values indicate time after. This matches
    the convention used in peri-event time histograms (PSTH).

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute regressor.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    signed : bool, default=True
        If True, return signed time (negative before event, positive after).
        If False, return absolute distance to nearest event.
    max_time : float, optional
        Maximum absolute time to return. Values beyond this are clipped.
        For signed=True, clips to [-max_time, +max_time].
        For signed=False, clips to [0, max_time].

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time to nearest event (seconds).
        - signed=True: Negative before event, positive after, zero at event.
        - signed=False: Absolute distance to nearest event.
        - NaN when no events are provided.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If max_time is negative.

    Notes
    -----
    This function finds the nearest event for each sample and computes the
    signed time difference. For peri-event analysis:

    - Use `signed=True` (default) for PSTH-like time axis
    - Use `signed=False` for distance-based filtering
    - Use `max_time` to define a peri-event window

    Common use cases:
    - PSTH time axis: samples within window of events
    - GLM regressor: continuous time-to-event predictor
    - Event-triggered filtering: select samples near events

    At exact midpoints between two events, the earlier event is used
    (tie-breaking is consistent but arbitrary).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import time_to_nearest_event

    Basic usage - PSTH-like time axis:

    >>> sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> event_times = np.array([2.0])
    >>> time_to_nearest_event(sample_times, event_times)
    array([-2., -1.,  0.,  1.,  2.])

    Filter to peri-event window:

    >>> times = time_to_nearest_event(sample_times, event_times)
    >>> peri_event_mask = np.abs(times) <= 1.5  # +/- 1.5s window
    >>> sample_times[peri_event_mask]
    array([1., 2., 3.])

    Clip to maximum time (for GLM design matrix):

    >>> time_to_nearest_event(sample_times, event_times, max_time=1.0)
    array([-1., -1.,  0.,  1.,  1.])
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    if max_time is not None and max_time < 0:
        raise ValueError(
            f"max_time must be non-negative, got {max_time}.\n"
            "  WHY: max_time defines a symmetric window around events.\n"
            "  HOW: Use max_time >= 0 or None for no clipping."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.float64)

    # Handle empty events
    if len(event_times) == 0:
        return np.full(len(sample_times), np.nan, dtype=np.float64)

    # Sort events (handles unsorted input)
    sorted_events = np.sort(event_times)

    # Find indices of surrounding events using searchsorted
    # idx_right: index of first event >= sample_time (or len if none)
    idx_right = np.searchsorted(sorted_events, sample_times, side="left")

    # idx_left: index of last event < sample_time (or -1 if none)
    idx_left = idx_right - 1

    # Compute distances to left and right events
    n_events = len(sorted_events)
    n_samples = len(sample_times)

    # Initialize with large values
    dist_left = np.full(n_samples, np.inf, dtype=np.float64)
    dist_right = np.full(n_samples, np.inf, dtype=np.float64)

    # Distance to left event (if exists)
    has_left = idx_left >= 0
    dist_left[has_left] = sample_times[has_left] - sorted_events[idx_left[has_left]]

    # Distance to right event (if exists)
    has_right = idx_right < n_events
    dist_right[has_right] = (
        sorted_events[idx_right[has_right]] - sample_times[has_right]
    )

    # Select nearest event
    # For ties (dist_left == dist_right), prefer left (earlier) event
    use_left = dist_left <= dist_right

    # Compute signed time to nearest event
    result = np.empty(n_samples, dtype=np.float64)
    result[use_left] = dist_left[use_left]  # Positive (after left event)
    result[~use_left] = -dist_right[~use_left]  # Negative (before right event)

    # Convert to unsigned if requested
    if not signed:
        result = np.abs(result)

    # Convert -0.0 to 0.0 for cleaner output
    result = result + 0.0  # Adding 0.0 converts -0.0 to 0.0

    # Apply max_time clipping
    if max_time is not None:
        result = np.clip(result, -max_time if signed else 0.0, max_time)

    return result


def event_count_in_window(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
) -> NDArray[np.int64]:
    """
    Count events within time window around each sample.

    For each sample time, counts the number of events that fall within
    the specified window. Useful for creating GLM regressors that capture
    recent event history (e.g., "rewards in last 5 seconds").

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute count.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, end) relative to sample_time.
        E.g., (-1.0, 0.0) counts events in previous 1 second.
        E.g., (0.0, 1.0) counts events in next 1 second.
        E.g., (-0.5, 0.5) counts events within +/- 0.5 seconds.
        Window boundaries are inclusive.

    Returns
    -------
    NDArray[np.int64], shape (n_samples,)
        Number of events within window of each sample.
        Always non-negative integers.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If window start > window end.

    Notes
    -----
    Common use cases:

    - Count rewards in last N seconds: ``window=(-N, 0.0)``
    - Count licks before reward: ``window=(-1.0, 0.0)``
    - Count spikes around event: ``window=(-0.5, 0.5)``

    Window boundaries are inclusive on both ends, so an event exactly
    at ``sample_time + window[0]`` or ``sample_time + window[1]`` is counted.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_count_in_window

    Count rewards in previous 5 seconds:

    >>> sample_times = np.array([0.0, 3.0, 6.0, 10.0])
    >>> reward_times = np.array([1.0, 2.0, 5.0])
    >>> event_count_in_window(sample_times, reward_times, window=(-5.0, 0.0))
    array([0, 2, 3, 1])

    Count events in symmetric window:

    >>> event_count_in_window(sample_times, reward_times, window=(-1.5, 1.5))
    array([1, 1, 1, 0])
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    # Validate window
    window_start, window_end = window
    if window_start > window_end:
        raise ValueError(
            f"window start ({window_start}) must be <= window end ({window_end}).\n"
            "  WHY: Window defines a time range [start, end] relative to sample.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.int64)

    # Handle empty events
    if len(event_times) == 0:
        return np.zeros(len(sample_times), dtype=np.int64)

    # Sort events for efficient searchsorted
    sorted_events = np.sort(event_times)

    # For each sample, compute window bounds in absolute time
    # and count events within those bounds
    window_starts = sample_times + window_start
    window_ends = sample_times + window_end

    # Use searchsorted to find event indices at window boundaries
    # left_indices: first event >= window_start
    # right_indices: first event > window_end (so count = right - left)
    left_indices = np.searchsorted(sorted_events, window_starts, side="left")
    right_indices = np.searchsorted(sorted_events, window_ends, side="right")

    # Count = number of events in [window_start, window_end]
    counts = right_indices - left_indices

    return counts.astype(np.int64)


def event_indicator(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    window: float = 0.0,
) -> NDArray[np.bool_]:
    """
    Binary indicator of whether an event occurs near each sample time.

    For each sample, returns True if any event falls within the specified
    window around that sample. Useful for creating indicator variables in
    GLM design matrices.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps.
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps.
    window : float, default=0.0
        Symmetric half-width of temporal window (seconds).
        Creates a window [sample_time - window, sample_time + window].
        If 0, only exact matches count (sample_time == event_time).
        If > 0, events within the symmetric window are considered a match.

    Returns
    -------
    NDArray[np.bool_], shape (n_samples,)
        True if event occurs within window of sample, False otherwise.

    Raises
    ------
    ValueError
        If sample_times or event_times contain NaN or Inf values.
        If window is negative.

    Notes
    -----
    Common use cases:

    - Binary event regressor: ``window=0.0`` for impulse at event time
    - Smoothed indicator: ``window=bin_size/2`` for time-binned data
    - Event presence detection: ``window=0.1`` for events within 100ms

    Window boundaries are inclusive, so events exactly at ``sample_time ± window``
    are included (returns True).

    This function is semantically clearer than ``event_count_in_window() > 0``
    for creating binary indicators and returns bool dtype directly.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_indicator

    Check for exact event occurrence:

    >>> sample_times = np.array([0.0, 1.0, 2.0, 3.0])
    >>> event_times = np.array([1.0, 3.0])
    >>> event_indicator(sample_times, event_times)
    array([False,  True, False,  True])

    Check for events within window:

    >>> event_indicator(sample_times, event_times, window=0.5)
    array([False,  True, False,  True])

    Create GLM design matrix column:

    >>> indicator = event_indicator(sample_times, event_times, window=0.1)
    >>> X = indicator.astype(float)  # Convert to float for design matrix
    """
    # Convert to numpy arrays and validate
    sample_times = np.asarray(sample_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Input validation
    if np.any(np.isnan(sample_times)):
        raise ValueError(
            "sample_times contains NaN values.\n"
            "  WHY: Times must be valid numeric values.\n"
            "  HOW: Remove or interpolate NaN values before calling."
        )

    if np.any(np.isinf(sample_times)):
        raise ValueError(
            "sample_times contains inf values.\n"
            "  WHY: Times must be finite.\n"
            "  HOW: Remove or clip infinite values before calling."
        )

    if len(event_times) > 0:
        if np.any(np.isnan(event_times)):
            raise ValueError(
                "event_times contains NaN values.\n"
                "  WHY: Event times must be valid numeric values.\n"
                "  HOW: Remove NaN values from event times."
            )

        if np.any(np.isinf(event_times)):
            raise ValueError(
                "event_times contains inf values.\n"
                "  WHY: Event times must be finite.\n"
                "  HOW: Remove infinite values from event times."
            )

    if window < 0:
        raise ValueError(
            f"window must be non-negative, got {window}.\n"
            "  WHY: window defines a symmetric half-width around each sample.\n"
            "  HOW: Use window >= 0."
        )

    # Handle empty sample_times
    if len(sample_times) == 0:
        return np.array([], dtype=np.bool_)

    # Handle empty events
    if len(event_times) == 0:
        return np.zeros(len(sample_times), dtype=np.bool_)

    # Sort events for efficient searchsorted
    sorted_events = np.sort(event_times)

    # For each sample, check if any event falls within [sample - window, sample + window]
    # Use searchsorted to find events at window boundaries
    window_starts = sample_times - window
    window_ends = sample_times + window

    # left_indices: first event >= window_start
    # right_indices: first event > window_end
    left_indices = np.searchsorted(sorted_events, window_starts, side="left")
    right_indices = np.searchsorted(sorted_events, window_ends, side="right")

    # If right > left, there's at least one event in the window
    result = right_indices > left_indices

    return result
