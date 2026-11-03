"""The plain-array record every loader returns, its validation and expectations.

``Recording`` holds a session in the units and conventions neurospatial expects:
seconds, radians with 0 = East and +pi/2 = North, NaN for missing samples. It
is internal to the survey harness, not a proposal for a neurospatial object.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

POSITION_UNITS = {"cm", "m", "pixel", "arbitrary"}


@dataclass(frozen=True)
class Recording:
    """One session as plain arrays.

    Attributes
    ----------
    dataset, session, source, license : str
        Registry key, session identifier, DOI or URL of the exact files loaded,
        and the data license ("unstated" if none).
    times : ndarray, shape (n_time,), optional
        Tracking sample times in seconds, strictly increasing.
    positions : ndarray, shape (n_time, n_dims), optional
        Tracked position in ``position_units``; NaN where tracking was lost.
    position_units : str, optional
        "cm", "m", "pixel" or "arbitrary".
    headings : ndarray, shape (n_time,), optional
        Allocentric head direction in radians, wrapped to [-pi, pi].
    spike_times : tuple of ndarray, optional
        One sorted array of spike times (s) per unit.
    unit_ids : ndarray, shape (n_units,), optional
        Unique unit identifiers, one per spike train.
    activity, activity_times, activity_kind : optional
        Imaging activity (n_frames, n_cells), its frame times (s) and kind.
    epochs : mapping of str to ndarray, shape (n, 2)
        Named [start, stop) intervals in seconds.
    events : mapping of str to ndarray, shape (n,) or (n, 2)
        Named event times or intervals in seconds.
    geometry : mapping
        Arena facts stated by the source; inferred values are prefixed ``inferred_``.
    params : mapping
        Analysis choices the driver uses (``bin_size`` in ``position_units``,
        ``dt`` in s, and optional ``min_speed``, ``analysis_epochs``,
        ``train_epochs``, ``test_epochs``, ``replay_dt``, ``nwb_source``).
    conversions : tuple of str
        Every transformation the loader applied, one line each.
    """

    dataset: str
    session: str
    source: str
    license: str
    times: NDArray[np.float64] | None = None
    positions: NDArray[np.float64] | None = None
    position_units: str | None = None
    headings: NDArray[np.float64] | None = None
    spike_times: tuple[NDArray[np.float64], ...] | None = None
    unit_ids: NDArray[Any] | None = None
    activity: NDArray[np.float64] | None = None
    activity_times: NDArray[np.float64] | None = None
    activity_kind: str | None = None
    epochs: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    events: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    conversions: tuple[str, ...] = ()


def _is_interval_table(table: NDArray[np.float64]) -> bool:
    """Whether ``table`` is finite (n, 2) [start, stop) rows with stop >= start."""
    return bool(
        table.ndim == 2
        and table.shape[1] == 2
        and np.isfinite(table).all()
        and np.all(table[:, 1] >= table[:, 0])
    )


def validate_recording(rec: Recording) -> None:
    """Raise ValueError listing every contract violation in ``rec``."""
    problems: list[str] = []
    tracked = rec.positions is not None or rec.headings is not None
    if (rec.times is None) == tracked:
        problems.append("times must be set exactly when positions or headings are")
    if rec.times is not None and (
        rec.times.ndim != 1 or not np.all(np.diff(rec.times) > 0)
    ):
        problems.append("times must be 1-D and strictly increasing")
    if rec.positions is not None:
        n_time = None if rec.times is None else rec.times.shape[0]
        if rec.positions.ndim != 2 or rec.positions.shape[0] != n_time:
            problems.append(
                f"positions shape {rec.positions.shape} does not match {n_time} times"
            )
        if rec.position_units not in POSITION_UNITS:
            problems.append(
                f"position_units {rec.position_units!r} not in {sorted(POSITION_UNITS)}"
            )
        required = {"bin_size"} | ({"dt"} if rec.spike_times is not None else set())
        if missing := sorted(required - set(rec.params)):
            problems.append(f"params missing {missing}")
    if rec.headings is not None:
        if rec.times is None or rec.headings.shape != rec.times.shape:
            problems.append("headings must have one value per time")
        finite = rec.headings[np.isfinite(rec.headings)]
        if finite.size and (finite.min() < -np.pi or finite.max() > np.pi):
            problems.append("headings must be radians in [-pi, pi]")
    if rec.spike_times is not None:
        if rec.unit_ids is None or len(rec.unit_ids) != len(rec.spike_times):
            problems.append("unit_ids must have one id per spike train")
        elif len(np.unique(rec.unit_ids)) != len(rec.unit_ids):
            problems.append("unit_ids must be unique")
        for index, train in enumerate(rec.spike_times):
            if not np.all(np.isfinite(train)) or np.any(np.diff(train) < 0):
                problems.append(f"spike train {index} must be finite and sorted")
    if (rec.activity is None) != (rec.activity_times is None):
        problems.append("activity and activity_times must be set together")
    elif (
        rec.activity is not None
        and rec.activity_times is not None
        and rec.activity.shape[0] != rec.activity_times.shape[0]
    ):
        problems.append("activity must have one row per activity time")
    for name, table in rec.epochs.items():
        if not _is_interval_table(table):
            problems.append(
                f"epochs[{name!r}] must be finite (n, 2) rows with stop >= start"
            )
    problems.extend(
        f"params[{name!r}] must be finite (n, 2) rows with stop >= start"
        for name in ("train_epochs", "test_epochs", "analysis_epochs")
        if name in rec.params
        and not _is_interval_table(np.asarray(rec.params[name], dtype=float))
    )
    for name, table in rec.events.items():
        table = np.asarray(table, dtype=float)
        if not (
            (table.ndim == 1 and np.isfinite(table).all()) or _is_interval_table(table)
        ):
            problems.append(
                f"events[{name!r}] must be finite times (n,) or (n, 2) intervals"
                " with stop >= start"
            )
    if problems:
        msg = (
            f"{rec.dataset}/{rec.session} breaks the Recording contract:\n- "
            + "\n- ".join(problems)
        )
        raise ValueError(msg)


def _optional(attr: str, measure: Callable[[Any], Any]) -> Callable[[Recording], Any]:
    def run(rec: Recording) -> Any:
        value = getattr(rec, attr)
        return None if value is None else measure(value)

    return run


MEASURES: dict[str, Callable[[Recording], Any]] = {
    "n_units": _optional("spike_times", len),
    "n_cells": _optional("activity", lambda a: a.shape[1]),
    "n_dims": _optional("positions", lambda p: p.shape[1]),
    "sampling_rate_hz": _optional(
        "times", lambda t: float(1.0 / np.median(np.diff(t)))
    ),
    "duration_s": _optional("times", lambda t: float(t[-1] - t[0])),
    "position_min": _optional("positions", lambda p: np.nanmin(p, axis=0)),
    "position_max": _optional("positions", lambda p: np.nanmax(p, axis=0)),
    "position_nan_fraction": _optional(
        "positions", lambda p: float(np.isnan(p).any(axis=1).mean())
    ),
    "epoch_names": lambda r: sorted(r.epochs),
    "event_names": lambda r: sorted(r.events),
}


@dataclass(frozen=True)
class Range:
    """Inclusive expectation range, compared elementwise for array measures."""

    low: Any
    high: Any

    def __post_init__(self) -> None:
        if np.any(
            np.asarray(self.low, dtype=float) > np.asarray(self.high, dtype=float)
        ):
            msg = f"Range low {self.low!r} is above high {self.high!r}"
            raise ValueError(msg)

    def contains(self, actual: Any) -> bool:
        values = np.asarray(actual, dtype=float)
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)
        return bool(np.all((values >= low) & (values <= high)))


def _check(rec: Recording, key: str, expected: Any) -> tuple[bool, str]:
    if callable(expected):
        return bool(expected(rec)), "predicate"
    actual = MEASURES[key](rec)
    detail = f"expected {expected!r}, got {actual!r}"
    if actual is None:
        return False, detail
    if isinstance(expected, Range):
        return expected.contains(actual), detail
    return bool(np.array_equal(np.asarray(actual), np.asarray(expected))), detail


def check_expectations(
    rec: Recording, expectations: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate pre-registered expectations; a failed or erroring check never raises.

    Each value is an exact value, a ``Range``, or a predicate on the
    ``Recording``. An unknown measure name is a mistake in the expectations and
    raises ``KeyError``.
    """
    unknown = [
        k for k, v in expectations.items() if not callable(v) and k not in MEASURES
    ]
    if unknown:
        msg = f"Unknown expectation {unknown[0]!r}; use one of {sorted(MEASURES)} or a predicate."
        raise KeyError(msg)
    rows = []
    for key, expected in expectations.items():
        try:
            passed, detail = _check(rec, key, expected)
        except Exception as exc:  # a check that cannot run has failed
            passed, detail = False, f"check raised {type(exc).__name__}: {exc}"
        rows.append({"key": key, "passed": passed, "detail": detail})
    return rows


def alternating_blocks(
    start: float, stop: float, length: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split [start, stop) into alternating train and test blocks of ``length`` s.

    The last block ends at ``stop`` and is shorter than ``length`` unless the
    span is a whole number of blocks; an edge within a millionth of a block
    of ``stop`` (float rounding) is not a block of its own.

    Returns
    -------
    train, test : ndarray, shape (n, 2)
        Even-numbered and odd-numbered blocks as [start, stop) rows.
    """
    n_blocks = max(int(np.ceil((stop - start) / length - 1e-6)), 1)
    edges = np.append(start + length * np.arange(n_blocks), stop)
    blocks = np.column_stack([edges[:-1], edges[1:]])
    return blocks[::2], blocks[1::2]


def heading_vs_movement(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    min_speed: float,
    lag: int = 1,
) -> tuple[float, float]:
    """Compare a heading series with the direction of movement while running.

    A heading in the same allocentric frame as the positions points along the
    path while the animal runs, so the circular mean of heading minus movement
    direction is near 0 and the spread is tens of degrees. A mean near ±π/2 or
    π says the frame is rotated or mirrored; a spread near 90° says the
    series is not a heading of this path.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    positions : ndarray, shape (n_time, 2)
        Same length unit as ``min_speed``.
    headings : ndarray, shape (n_time,)
        Radians.
    min_speed : float
        Steps faster than this (length per second) count as running.
    lag : int, default 1
        The direction of movement at sample i is that of the step to sample
        i + ``lag``. On positions in whole pixels, one-sample steps of a few
        pixels take only a few directions (multiples of 45° for one-pixel
        steps), which widens the spread and biases the mean; a longer lag
        averages that out.

    Returns
    -------
    mean, spread : float
        Circular mean of heading minus movement direction, and the median
        absolute deviation from that mean, in radians.
    """
    if lag < 1:
        msg = f"lag must be a positive number of samples, got {lag}"
        raise ValueError(msg)
    step = positions[lag:] - positions[:-lag]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(step, axis=1) / (times[lag:] - times[:-lag])
    moving = np.arctan2(step[:, 1], step[:, 0])
    start = headings[:-lag]
    running = (speed > min_speed) & np.isfinite(start)
    diff = np.angle(np.exp(1j * (start[running] - moving[running])))
    mean = float(np.angle(np.mean(np.exp(1j * diff))))
    spread = float(np.median(np.abs(np.angle(np.exp(1j * (diff - mean))))))
    return mean, spread


def checked_span(
    positions: NDArray[np.float64], low: float, high: float, *, units: str = "cm"
) -> str:
    """Refuse positions whose largest per-axis span is outside ``[low, high]``.

    A loader states the span it expects of the apparatus in ``units``; a span
    outside it means a unit or scale error. Returns the line for
    ``conversions``.
    """
    tracked = positions[np.isfinite(positions).all(axis=1)]
    lo, hi = tracked.min(axis=0), tracked.max(axis=0)
    span = hi - lo
    if not low <= span.max() <= high:
        msg = (
            f"largest position span {span.max():.1f} {units} is outside the expected"
            f" {low:g}-{high:g} {units}; recheck the unit and scale"
        )
        raise ValueError(msg)
    axes = " by ".join(f"{a:.1f} to {b:.1f}" for a, b in zip(lo, hi, strict=True))
    return f"range checked: {axes} {units} (largest span {span.max():.1f}, expected {low:g}-{high:g})"


def intersect_intervals(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise intersections of two sets of [start, stop) rows, shape (n, 2)."""
    start = np.maximum(a[:, None, 0], b[None, :, 0])
    stop = np.minimum(a[:, None, 1], b[None, :, 1])
    keep = stop > start
    return np.column_stack([start[keep], stop[keep]])


def mask_lost_tracking(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_frozen: int,
    max_speed: float,
    units: str = "cm",
) -> tuple[NDArray[np.float64], str]:
    """Set held and bridged tracking to NaN, by a stated rule.

    A tracker that loses the animal often keeps reporting the last fix
    (identical samples) and then joins it to the next fix with a straight
    line or a jump. On positions with sub-pixel values, an animal at rest
    still moves the estimate, so identical samples are a held value:

    - frozen: a sample identical in every coordinate to the one before it,
      in a run of at least ``min_frozen`` such repeats; the run's first
      sample (the last real fix) is kept;
    - fast: one sequential pass over the remaining finite samples. The
      anchor is the last kept sample; each later sample is kept, and becomes
      the anchor, when its distance from the anchor over the time since it
      is at most ``max_speed`` (``units`` per second, a speed the animal
      cannot exceed), and is set to NaN otherwise. Samples already NaN (in
      the input or held) are skipped and do not move the anchor. A one-frame
      glitch removes only the glitch; an excursion removes all its samples
      and keeps the return; the step from a held run to the next fix is
      checked against the run's kept first sample. After a real jump of
      distance ``d`` the pass re-anchors on its own once ``d / max_speed``
      seconds have passed.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    positions : ndarray, shape (n_time, n_dims)
    min_frozen : int
        Repeats of one value, in frames, from which a run counts as held.
    max_speed : float
        Fastest believable step, ``units`` per second.
    units : str
        Length unit of ``positions``, for the ``conversions`` line.

    Returns
    -------
    positions : ndarray, shape (n_time, n_dims)
        A copy with the frozen and fast samples set to NaN.
    line : str
        The rule and its counts, for ``conversions``.
    """
    positions = np.array(positions, dtype=float)
    repeat = np.r_[False, np.all(np.diff(positions, axis=0) == 0, axis=1)]
    edges = np.diff(np.r_[0, repeat.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_runs = stops - starts >= min_frozen
    frozen = np.zeros(len(positions), dtype=bool)
    for start, stop in zip(starts[long_runs], stops[long_runs], strict=True):
        frozen[start:stop] = True
    longest = (
        float(np.max(times[stops[long_runs] - 1] - times[starts[long_runs] - 1]))
        if long_runs.any()
        else 0.0
    )
    positions[frozen] = np.nan
    fast = np.zeros(len(positions), dtype=bool)
    finite = np.flatnonzero(np.isfinite(positions).all(axis=1))
    points, clock = positions.tolist(), np.asarray(times, dtype=float).tolist()
    if len(finite):
        anchor = int(finite[0])
        for k in finite[1:].tolist():  # sequential: each test needs the last kept
            if math.dist(points[k], points[anchor]) <= max_speed * (
                clock[k] - clock[anchor]
            ):
                anchor = k
            else:
                fast[k] = True
    positions[fast] = np.nan
    line = (
        f"tracking: {int(frozen.sum())} samples repeating the one before in runs of"
        f" {min_frozen}+ frames ({int(long_runs.sum())} runs, longest held"
        f" {longest:.1f} s) and {int(fast.sum())} samples reached faster than"
        f" {max_speed:g} {units}/s from the last kept sample set to NaN (held, then"
        " bridged, tracking)"
    )
    return positions, line
