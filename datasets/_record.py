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
        ``train_epochs``, ``test_epochs``, ``replay_dt``, ``nwb_path``).
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
        if "bin_size" not in rec.params:
            problems.append("params missing ['bin_size']")
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


def check_expectations(
    rec: Recording, expectations: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate pre-registered expectations; never raises on a failed check.

    Each value is an exact value, a ``(low, high)`` range (inclusive, compared
    elementwise for array measures), or a predicate on the ``Recording``.
    """
    rows = []
    for key, expected in expectations.items():
        if callable(expected):
            rows.append(
                {"key": key, "passed": bool(expected(rec)), "detail": "predicate"}
            )
            continue
        if key not in MEASURES:
            msg = f"Unknown expectation {key!r}; use one of {sorted(MEASURES)} or a predicate."
            raise KeyError(msg)
        actual = MEASURES[key](rec)
        if isinstance(expected, tuple) and len(expected) == 2 and actual is not None:
            low, high = expected
            values = np.asarray(actual, dtype=float)
            passed = bool(
                np.all((values >= np.asarray(low)) & (values <= np.asarray(high)))
            )
        else:
            passed = actual is not None and np.array_equal(
                np.asarray(actual), np.asarray(expected)
            )
        rows.append(
            {
                "key": key,
                "passed": passed,
                "detail": f"expected {expected!r}, got {actual!r}",
            }
        )
    return rows
