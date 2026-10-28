"""Pre-decision windows and their speed, shared by decision and VTE analyses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import run_sample_bounds
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING, smoothed_speed


def first_entry(
    position_bins: NDArray[np.int64],
    region_bins: NDArray[np.int_],
    observed: NDArray[np.bool_],
) -> int | None:
    """Index of the first observed sample inside the region; None if none."""
    hits = np.flatnonzero(observed & np.isin(position_bins, region_bins))
    return int(hits[0]) if hits.size else None


def validated_window_duration(window_duration: float, *, name: str, call: str) -> float:
    """A finite, positive pre-decision window length in seconds."""
    duration = float(window_duration)
    if not (np.isfinite(duration) and duration > 0):
        raise ValueError(
            f"{call}() needs a finite, positive {name} in seconds; got "
            f"{window_duration!r}.\n"
            "Why: the pre-decision window is the stretch of time before the "
            "decision-region entry; an empty or reversed window measures "
            "nothing.\n"
            f"Fix: pass {name} as the window length in seconds, e.g. {name}=1.0."
        )
    return duration


def validated_entry_time(
    times: NDArray[np.float64], entry_time: float, *, call: str
) -> float:
    """A finite entry time inside the recording, ``[times[0], times[-1]]``."""
    entry = float(entry_time)
    if not (np.isfinite(entry) and times[0] <= entry <= times[-1]):
        raise ValueError(
            f"{call}() got entry_time={entry_time!r}, outside the recording "
            f"[{times[0]:g}, {times[-1]:g}] s.\n"
            "Why: the pre-decision window ends at the entry time, so an entry "
            "the recording does not cover has no window.\n"
            "Fix: pass the decision-region entry in seconds on the same clock "
            "as times (a millisecond entry_time with times in seconds lands "
            "past the recording), e.g. from decision_region_entry_time()."
        )
    return entry


def pre_decision_window(
    times: NDArray[np.float64],
    interval_gates: NDArray[np.bool_],
    entry_time: float,
    window_duration: float,
) -> slice:
    """Sample slice of the pre-decision window; empty if entry is in no segment.

    Segments are runs of ``interval_gates`` (``max_gap``, ``epochs`` and
    declared breaks), whatever the validity of their samples.
    """
    for first, last in run_sample_bounds(interval_gates):
        if times[first] <= entry_time <= times[last]:
            window_start = max(entry_time - window_duration, times[first])
            segment = times[first : last + 1]
            start = int(first + np.searchsorted(segment, window_start, side="left"))
            stop = int(first + np.searchsorted(segment, entry_time, side="left"))
            return slice(start, stop)
    return slice(0, 0)


def speed_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Mean and minimum smoothed speed over usable intervals; NaN without any."""
    if not interval_mask.any():
        return np.nan, np.nan
    speeds = smoothed_speed(
        times, positions, interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
    )
    return float(np.nanmean(speeds)), float(np.nanmin(speeds))
