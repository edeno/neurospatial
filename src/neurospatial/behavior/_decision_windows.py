"""Decision-point windows and their speed, shared by decision and VTE analyses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import run_sample_bounds


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
        msg = (
            f"{call}() needs a finite, positive {name} in seconds; got "
            f"{window_duration!r}.\n"
            "Why: the pre-decision window is the stretch of time before the "
            "decision-region entry; an empty or reversed window measures "
            "nothing.\n"
            f"Fix: pass {name} as the window length in seconds, e.g. {name}=1.0."
        )
        raise ValueError(msg)
    return duration


def validated_entry_time(
    times: NDArray[np.float64], entry_time: float, *, call: str
) -> float:
    """A finite entry time inside the recording, ``[times[0], times[-1]]``."""
    entry = float(entry_time)
    if not (np.isfinite(entry) and times[0] <= entry <= times[-1]):
        msg = (
            f"{call}() got entry_time={entry_time!r}, outside the recording "
            f"[{times[0]:g}, {times[-1]:g}] s.\n"
            "Why: the pre-decision window ends at the entry time, so an entry "
            "the recording does not cover has no window.\n"
            "Fix: pass the decision-region entry in seconds on the same clock "
            "as times (a millisecond entry_time with times in seconds lands "
            "past the recording), e.g. from decision_region_entry_time()."
        )
        raise ValueError(msg)
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


def validated_pass_bounds(
    times: NDArray[np.float64], start_time: float, stop_time: float, *, call: str
) -> tuple[float, float]:
    """Finite pass bounds ``start_time < stop_time``, the start in the recording."""
    start, stop = float(start_time), float(stop_time)
    if not (np.isfinite(start) and np.isfinite(stop) and start < stop):
        msg = (
            f"{call}() needs finite start_time < stop_time in seconds; got "
            f"start_time={start_time!r}, stop_time={stop_time!r}.\n"
            "Why: the window is the half-open span [start_time, stop_time) of "
            "the pass through the choice point; an empty or reversed span "
            "holds no sample.\n"
            "Fix: pass the time the animal entered the choice-point zone as "
            "start_time and the time it left as stop_time."
        )
        raise ValueError(msg)
    if not times[0] <= start <= times[-1]:
        msg = (
            f"{call}() got start_time={start_time!r}, outside the recording "
            f"[{times[0]:g}, {times[-1]:g}] s.\n"
            "Why: the window starts at start_time, so a start the recording "
            "does not cover has no samples.\n"
            "Fix: pass start_time in seconds on the same clock as times (a "
            "millisecond start_time with times in seconds lands past the "
            "recording)."
        )
        raise ValueError(msg)
    return start, stop


def span_window(
    times: NDArray[np.float64],
    interval_gates: NDArray[np.bool_],
    start_time: float,
    stop_time: float,
) -> slice:
    """Samples in ``[start_time, stop_time)`` of the segment holding ``start_time``.

    Segments are runs of ``interval_gates`` (``max_gap``, ``epochs`` and
    declared breaks), whatever the validity of their samples. Empty when
    ``start_time`` lies in no segment.
    """
    for first, last in run_sample_bounds(interval_gates):
        if times[first] <= start_time <= times[last]:
            segment = times[first : last + 1]
            start = int(first + np.searchsorted(segment, start_time, side="left"))
            stop = int(first + np.searchsorted(segment, stop_time, side="left"))
            return slice(start, stop)
    return slice(0, 0)


def first_pass_stop(
    entry: int, leaves: NDArray[np.bool_], interval_gates: NDArray[np.bool_]
) -> int:
    """Exclusive stop index of the pass through a region that starts at ``entry``.

    ``leaves`` marks each sample that ends a pass: an observed sample outside
    the region, or a sample outside ``epochs``. A missing sample marks
    nothing, so a bridged stretch of missing samples does not end the pass.
    The pass ends before the first marked sample after ``entry`` or after the
    last sample of the gate segment holding ``entry``, whichever comes first.
    """
    breaks = np.flatnonzero(~interval_gates[entry:])
    segment_stop = entry + int(breaks[0]) + 1 if breaks.size else leaves.size
    exits = np.flatnonzero(leaves[entry + 1 : segment_stop])
    return entry + 1 + int(exits[0]) if exits.size else segment_stop


def speed_stats(speeds: NDArray[np.float64]) -> tuple[float, float]:
    """Mean and minimum of the observed speeds; NaN when none is observed."""
    observed = speeds[~np.isnan(speeds)]
    if observed.size == 0:
        return np.nan, np.nan
    return float(np.mean(observed)), float(np.min(observed))
