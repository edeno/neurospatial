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

from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import as_intervals, intervals_contain
from neurospatial._time_bins import _time_bin_rounding
from neurospatial.events._core import (
    _finite_times,
    _inner_bounds,
    _is_real_number,
    _validate_window,
)


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
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    signed: bool = True,
    max_time: float | None = None,
    epochs: Any = None,
) -> NDArray[np.float64]:
    """
    Compute time to nearest event for each sample.

    Returns signed time relative to nearest event: negative values indicate
    time before the event, positive values indicate time after. This matches
    the convention used in peri-event time histograms (PSTH).

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
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
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        When events were recorded: half-open [start, stop) windows in seconds
        on the same clock as ``times``. A value ``d`` is known only if no
        unseen event could be closer, so it is NaN unless ``(t - d, t + d)``
        lies inside one window. ``None`` (default) treats all time as
        recorded.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time to nearest event (seconds).
        - signed=True: Negative before event, positive after, zero at event.
        - signed=False: Absolute distance to nearest event.
        - NaN without any event (unless ``signed=False`` with ``max_time``:
          no event is nearer than ``max_time``), and with ``epochs`` where
          an unrecorded event could be nearer or, for ``signed=True``, on the
          other side.

    Raises
    ------
    ValueError
        If times or event_times are not 1-D or contain NaN or Inf
        values. If max_time is negative or NaN, or epochs is invalid.

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

    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> event_times = np.array([2.0])
    >>> time_to_nearest_event(times, event_times)
    array([-2., -1.,  0.,  1.,  2.])

    Filter to peri-event window:

    >>> offsets = time_to_nearest_event(times, event_times)
    >>> peri_event_mask = np.abs(offsets) <= 1.5  # +/- 1.5s window
    >>> times[peri_event_mask]
    array([1., 2., 3.])

    Clip to maximum time (for GLM design matrix):

    >>> time_to_nearest_event(times, event_times, max_time=1.0)
    array([-1., -1.,  0.,  1.,  1.])

    With recording from 1 s on, an unseen event before 1 s could be nearer
    than the one at 2 s to the samples at 0 s and 1 s:

    >>> time_to_nearest_event(times, event_times, epochs=(1.0, 10.0))
    array([nan, nan,  0.,  1.,  2.])
    """
    times = _finite_times(times, "times")
    event_times = _finite_times(event_times, "event_times")

    if max_time is not None and not (
        _is_real_number(max_time) and np.isfinite(max_time) and max_time >= 0
    ):
        raise ValueError(
            f"max_time must be a finite non-negative number of seconds, got "
            f"{max_time!r}.\n"
            "  WHY: max_time is the largest time-to-event magnitude returned.\n"
            "  Fix: Use max_time >= 0, or None for no clipping."
        )

    recorded = as_intervals(epochs, name="epochs")

    # Handle empty times
    if len(times) == 0:
        return np.array([], dtype=np.float64)

    # Sort events (handles unsorted input)
    sorted_events = np.sort(event_times)

    # Find indices of surrounding events using searchsorted
    # idx_right: index of first event >= sample time (or len if none)
    idx_right = np.searchsorted(sorted_events, times, side="left")

    # idx_left: index of last event < sample time (or -1 if none)
    idx_left = idx_right - 1

    # Compute distances to left and right events
    n_events = len(sorted_events)
    n_samples = len(times)

    # Initialize with large values
    dist_left = np.full(n_samples, np.inf, dtype=np.float64)
    dist_right = np.full(n_samples, np.inf, dtype=np.float64)

    # Distance to left event (if exists)
    has_left = idx_left >= 0
    dist_left[has_left] = times[has_left] - sorted_events[idx_left[has_left]]

    # Distance to right event (if exists)
    has_right = idx_right < n_events
    dist_right[has_right] = sorted_events[idx_right[has_right]] - times[has_right]

    # Infinite when there are no events at all.
    distance = np.minimum(dist_left, dist_right)
    if signed:
        # Ties prefer the earlier event: positive after it, negative before
        # the next. Without any event there is no side.
        result = np.where(dist_left <= dist_right, dist_left, -dist_right)
        result[np.isinf(distance)] = np.nan
    else:
        result = distance.copy()
    result = result + 0.0  # -0.0 -> 0.0

    if max_time is not None:
        result = np.clip(result, -max_time if signed else 0.0, max_time)
    result[np.isinf(result)] = np.nan

    if recorded is not None:
        # The value is known when no unseen event could change it: no event
        # in the unrecorded time is nearer than the distance that decides it.
        # A clipped unsigned value is decided within max_time; a signed value
        # reports the side of the true nearest event, however far it is.
        reach = distance if signed else result
        starts, stops = _inner_bounds(times - reach, times + reach)
        known = (reach == 0) | intervals_contain(recorded, starts, stops)
        result[~known] = np.nan
    return result


@overload
def event_count_in_window(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: None = None,
) -> NDArray[np.int64]: ...
@overload
def event_count_in_window(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: ArrayLike,
) -> NDArray[np.float64]: ...


def event_count_in_window(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: Any = None,
) -> NDArray[np.int64] | NDArray[np.float64]:
    """
    Count events within time window around each sample.

    For each sample time, counts the number of events that fall within
    the specified window. Useful for creating GLM regressors that capture
    recent event history (e.g., "rewards in last 5 seconds").

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Times at which to compute count (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, end) relative to each sample time, in seconds.
        E.g., (-1.0, 0.0) counts events in previous 1 second.
        E.g., (0.0, 1.0) counts events in next 1 second.
        E.g., (-0.5, 0.5) counts events within +/- 0.5 seconds.
        Half-open: ``[time + start, time + stop)``.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        When events were recorded: half-open [start, stop) windows in seconds
        on the same clock as ``times``. A sample whose window is not inside
        one of them is NaN, because an event there would have gone unseen.
        ``None`` (default) treats all time as recorded.

    Returns
    -------
    NDArray[np.int64] or NDArray[np.float64], shape (n_samples,)
        Number of events within window of each sample, non-negative. Integers
        without ``epochs``; with ``epochs``, floats that are NaN where the
        window is not inside one recorded window.

    Raises
    ------
    ValueError
        If times or event_times are not 1-D or contain NaN or Inf
        values. If a window bound is not finite, or window start >= window
        stop. If the window is too narrow to resolve in float64 at these
        timestamps. If epochs is invalid.

    Notes
    -----
    Common use cases:

    - Count rewards in last N seconds: ``window=(-N, 0.0)``
    - Count licks before reward: ``window=(-1.0, 0.0)``
    - Count spikes in a +/- 0.5 s window around each sample:
      ``window=(-0.5, 0.5)``

    Windows are half-open, ``[time + start, time + stop)``: an
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

    >>> times = np.array([0.0, 3.0, 6.0, 10.0])
    >>> reward_times = np.array([1.0, 2.0, 5.0])
    >>> event_count_in_window(times, reward_times, window=(-5.0, 0.0))
    array([0, 2, 3, 1])

    Count events in symmetric window:

    >>> event_count_in_window(times, reward_times, window=(-1.5, 1.5))
    array([1, 1, 1, 0])

    Recording started at 0 s, so the 5 s history of the first two samples
    reaches unrecorded time:

    >>> event_count_in_window(times, reward_times, (-5.0, 0.0), epochs=(0.0, 20.0))
    array([nan, nan,  3.,  1.])
    """
    times = _finite_times(times, "times")
    event_times = _finite_times(event_times, "event_times")
    window_start, window_end = _validate_window(
        window, context="event_count_in_window", warn_wide=False
    )

    recorded = as_intervals(epochs, name="epochs")

    counts = np.zeros(len(times), dtype=np.int64)
    starts, stops = times + window_start, times + window_end
    if len(times) and len(event_times):
        # Refuse windows too narrow for float64 at these timestamps; below
        # that, sharing coincident edges could also merge a window's own edges.
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
    if recorded is None:
        return counts.astype(np.int64)
    observed = counts.astype(np.float64)
    observed[~intervals_contain(recorded, *_inner_bounds(starts, stops))] = np.nan
    return observed


@overload
def event_indicator(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: None = None,
) -> NDArray[np.bool_]: ...
@overload
def event_indicator(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: ArrayLike,
) -> NDArray[np.float64]: ...


def event_indicator(
    times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    epochs: Any = None,
) -> NDArray[np.bool_] | NDArray[np.float64]:
    """
    Binary indicator of whether an event occurs near each sample time.

    For each sample, returns True if any event falls within the specified
    window around that sample. Useful for creating indicator variables in
    GLM design matrices.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps (seconds).
    event_times : NDArray[np.float64], shape (n_events,)
        Event timestamps (seconds).
    window : tuple[float, float]
        Time window (start, stop) relative to each sample time (seconds), half-open
        ``[time + start, time + stop)``. Required: no window is
        right for every sampling rate. For time-binned samples ``bin_size``
        apart, ``(-bin_size / 2, bin_size / 2)`` marks each event at exactly
        one sample. E.g., (-1.0, 0.0) matches events in the previous 1 second.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        When events were recorded: half-open [start, stop) windows in seconds
        on the same clock as ``times``. A sample whose window is not inside
        one of them is NaN, because an event there would have gone unseen.
        ``None`` (default) treats all time as recorded.

    Returns
    -------
    NDArray[np.bool_] or NDArray[np.float64], shape (n_samples,)
        True if event occurs within window of sample, False otherwise. With
        ``epochs``, floats 1.0 / 0.0, NaN where the window is not inside one
        recorded window.

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

    Windows are half-open, so an event exactly at ``time + start`` is
    included and one exactly at ``time + stop`` is not.

    This function is semantically clearer than ``event_count_in_window() > 0``
    for creating binary indicators and returns bool dtype directly. It takes
    the same ``window=(start, end)`` as ``event_count_in_window``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import event_indicator

    Mark each event at the sample whose 1 s bin holds it:

    >>> times = np.array([0.0, 1.0, 2.0, 3.0])
    >>> event_times = np.array([1.0, 3.0])
    >>> event_indicator(times, event_times, window=(-0.5, 0.5))
    array([False,  True, False,  True])

    Create GLM design matrix column:

    >>> indicator = event_indicator(times, event_times, window=(-0.1, 0.1))
    >>> X = indicator.astype(float)  # Convert to float for design matrix
    """
    counts = event_count_in_window(times, event_times, window, epochs=epochs)
    if epochs is None:
        indicator: NDArray[np.bool_] = counts > 0
        return indicator
    return np.where(np.isnan(counts), np.nan, (counts > 0).astype(np.float64))
