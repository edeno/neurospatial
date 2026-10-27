"""
GLM regressor generation for neurospatial events.

This module provides functions to generate time-based regressors for GLM
design matrices:

- time_to_nearest_event: Signed time to nearest event (peri-event time)
- event_count_in_window: Count events in time window
- event_indicator: Binary indicator of event presence

Per-sample spatial covariates (distance to a reward, a boundary or a region)
are in :mod:`neurospatial.behavior`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._time_bins import _time_bin_rounding
from neurospatial.events._core import _validate_window


def _as_times(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``values`` as a 1-D float64 array of finite seconds."""
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of times in seconds, got shape "
            f"{times.shape}.\n"
            "  WHY: Each entry is one time on the recording clock.\n"
            "  Fix: Pass one time per sample or event."
        )
    for is_bad, what in ((np.isnan, "NaN"), (np.isinf, "inf")):
        if is_bad(times).any():
            raise ValueError(
                f"{name} contains {what} values.\n"
                "  WHY: Times must be finite seconds.\n"
                f"  Fix: Remove {what} values before calling."
            )
    return times


def _share_coincident_edges(
    starts: NDArray[np.float64], stops: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Give window edges that agree up to float rounding one shared value.

    Adjacent windows of samples ``bin_size`` apart meet at an edge computed
    twice, as ``s[k] + stop`` and ``s[k + 1] + start``. Rounding can leave a
    gap or an overlap of a few ulp between the two, which counts an event on
    that edge at no sample or at two. Edges within a few ulp of each other
    are the same instant at float precision, so each group of them takes its
    smallest value; then the half-open windows partition time exactly.
    """
    edges = np.concatenate([starts, stops])
    order = np.argsort(edges, kind="stable")
    ordered = edges[order]
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(ordered[:-1]), np.abs(ordered[1:])))
    new_group = np.r_[True, np.diff(ordered) > tolerance]
    first = np.flatnonzero(new_group)
    shared = np.empty_like(edges)
    shared[order] = ordered[first][np.cumsum(new_group) - 1]
    return shared[: starts.size], shared[starts.size :]


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
        Times at which to compute regressor (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    signed : bool, default=True
        If True, return signed time (negative before event, positive after).
        If False, return absolute distance to nearest event.
    max_time : float, optional
        Maximum absolute time to return, in seconds. Values beyond this are
        clipped to it; no sample is removed.
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
        If sample_times or event_times are not 1-D or contain NaN or Inf
        values. If max_time is negative or NaN.

    Notes
    -----
    This function finds the nearest event for each sample and computes the
    signed time difference. For peri-event analysis:

    - Use `signed=True` (default) for PSTH-like time axis
    - Use `signed=False` for distance-based filtering
    - Use `max_time` to bound the regressor; select samples in a window
      with a mask on the result (see Examples)

    Common use cases:
    - PSTH time axis: samples within window of events
    - GLM regressor: continuous time-to-event predictor
    - Event-triggered filtering: select samples near events

    At exact midpoints between two events, the earlier event is used
    (tie-breaking is consistent but arbitrary).

    ``max_time`` is a symmetric clip on the returned value, not a
    ``(start, end)`` window: :func:`event_count_in_window` and
    :func:`event_indicator` take windows because they select events.

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
    sample_times = _as_times(sample_times, "sample_times")
    event_times = _as_times(event_times, "event_times")

    if max_time is not None and not max_time >= 0:
        raise ValueError(
            f"max_time must be a non-negative number of seconds, got {max_time}.\n"
            "  WHY: max_time is the largest time-to-event magnitude returned.\n"
            "  Fix: Use max_time >= 0 or None for no clipping."
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
    *,
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
        Times at which to compute count (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, end) relative to sample_time, in seconds.
        E.g., (-1.0, 0.0) counts events in previous 1 second.
        E.g., (0.0, 1.0) counts events in next 1 second.
        E.g., (-0.5, 0.5) counts events within +/- 0.5 seconds.
        Half-open: ``[sample_time + start, sample_time + stop)``.

    Returns
    -------
    NDArray[np.int64], shape (n_samples,)
        Number of events within window of each sample.
        Always non-negative integers.

    Raises
    ------
    ValueError
        If sample_times or event_times are not 1-D or contain NaN or Inf
        values. If a window bound is not finite, or window start >= window
        stop. If the window is too narrow to resolve in float64 at these
        timestamps.

    Notes
    -----
    Common use cases:

    - Count rewards in last N seconds: ``window=(-N, 0.0)``
    - Count licks before reward: ``window=(-1.0, 0.0)``
    - Count spikes in a +/- 0.5 s window around each sample:
      ``window=(-0.5, 0.5)``

    Windows are half-open, ``[sample_time + start, sample_time + stop)``: an
    event exactly at the start is counted and one exactly at the stop is not.
    Window edges that agree up to float rounding are treated as one instant,
    so windows of samples ``bin_size`` apart with ``window=(-bin_size / 2,
    bin_size / 2)`` tile time: each event inside the samples' windows counts
    at exactly one sample, even when events and samples share a clock grid
    and fall on bin edges. Events outside every window are not counted.

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
    sample_times = _as_times(sample_times, "sample_times")
    event_times = _as_times(event_times, "event_times")
    window_start, window_end = _validate_window(
        window, context="event_count_in_window", warn_wide=False
    )

    if len(sample_times) == 0:
        return np.array([], dtype=np.int64)
    if len(event_times) == 0:
        return np.zeros(len(sample_times), dtype=np.int64)

    starts = sample_times + window_start
    stops = sample_times + window_end
    # Refuse windows too narrow for float64 at these timestamps; below that,
    # sharing coincident edges could also merge a window's own two edges.
    _time_bin_rounding(
        np.array([[starts.min(), stops.max()]]),
        window_end - window_start,
        context="event_count_in_window windows",
        width_name="window width",
    )
    starts, stops = _share_coincident_edges(starts, stops)

    # Count = number of events in [start, stop)
    sorted_events = np.sort(event_times)
    counts = np.searchsorted(sorted_events, stops, side="left") - np.searchsorted(
        sorted_events, starts, side="left"
    )
    return counts.astype(np.int64)


def event_indicator(
    sample_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    window: tuple[float, float],
) -> NDArray[np.bool_]:
    """
    Binary indicator of whether an event occurs near each sample time.

    For each sample, returns True if any event falls within the specified
    window around that sample. Useful for creating indicator variables in
    GLM design matrices.

    Parameters
    ----------
    sample_times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, stop) relative to sample_time (seconds), half-open
        ``[sample_time + start, sample_time + stop)``. Required: no window is
        right for every sampling rate. For time-binned samples ``bin_size``
        apart, ``(-bin_size / 2, bin_size / 2)`` marks each event at exactly
        one sample. E.g., (-1.0, 0.0) matches events in the previous 1 second.

    Returns
    -------
    NDArray[np.bool_], shape (n_samples,)
        True if event occurs within window of sample, False otherwise.

    Raises
    ------
    ValueError
        As for :func:`event_count_in_window`.

    Notes
    -----
    Common use cases:

    - Impulse regressor for time-binned data: ``window=(-bin_size/2, bin_size/2)``
      marks each event at the one sample whose bin holds it
    - Event presence detection: ``window=(-0.1, 0.1)`` for events within 100ms

    Windows are half-open, so an event exactly at ``sample_time + start`` is
    included and one exactly at ``sample_time + stop`` is not.

    This function is semantically clearer than ``event_count_in_window() > 0``
    for creating binary indicators and returns bool dtype directly. It shares
    the same keyword-only ``window=(start, end)`` contract as
    ``event_count_in_window``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_indicator

    Mark each event at the sample whose 1 s bin holds it:

    >>> sample_times = np.array([0.0, 1.0, 2.0, 3.0])
    >>> event_times = np.array([1.0, 3.0])
    >>> event_indicator(sample_times, event_times, window=(-0.5, 0.5))
    array([False,  True, False,  True])

    Create GLM design matrix column:

    >>> indicator = event_indicator(sample_times, event_times, window=(-0.1, 0.1))
    >>> X = indicator.astype(float)  # Convert to float for design matrix
    """
    counts = event_count_in_window(sample_times, event_times, window=window)
    indicator: NDArray[np.bool_] = counts > 0
    return indicator
