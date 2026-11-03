"""The plain-array record every loader returns, its validation and expectations.

``Recording`` holds a session in the units and conventions neurospatial expects:
seconds, radians with 0 = East and +pi/2 = North, NaN for missing samples. It
is internal to the survey harness, not a proposal for a neurospatial object.
"""

from __future__ import annotations

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
        if table.ndim != 2 or table.shape[1] != 2 or np.any(table[:, 1] < table[:, 0]):
            problems.append(f"epochs[{name!r}] must be (n, 2) with stop >= start")
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

    Returns
    -------
    train, test : ndarray, shape (n, 2)
        Even-numbered and odd-numbered blocks as [start, stop) rows.
    """
    edges = np.append(np.arange(start, stop, length), stop)
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
