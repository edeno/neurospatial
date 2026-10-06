"""Time-window normalization and containment tests.

Every analysis that combines data streams restricts itself to time inside the
optional ``epochs`` and ``spike_window`` arguments. This module is the single
place those arguments are parsed, merged and tested, so every analysis family
applies identical semantics: each row is a half-open ``[start, stop)`` window in
seconds, on the same clock as ``times``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_ACCEPTED_FORMS = (
    "None, a (start, stop) pair, an (n, 2) array-like of [start, stop) rows, "
    "or an object with 1-D .start and .end arrays (for example a pynapple "
    "IntervalSet)"
)


def _format_rows(idx: NDArray[np.intp]) -> str:
    """Format row indices as ``"0, 3, 7"``, truncated after five entries."""
    shown = ", ".join(str(int(i)) for i in idx[:5])
    return shown + (f" (+{idx.size - 5} more)" if idx.size > 5 else "")


def _parse_intervals(
    value: Any, name: str
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Convert ``value`` to an unsorted ``(n, 2)`` array and list every problem.

    Returns ``(rows, problems)``. ``rows`` is ``None`` when numbers or shape
    are unusable; ``problems`` is empty when ``rows`` is valid.
    """
    if hasattr(value, "start") and hasattr(value, "end"):
        try:
            starts = np.asarray(value.start, dtype=np.float64)
            stops = np.asarray(value.end, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return None, [
                f"{name}.start and {name}.end could not be read as numbers ({exc})"
            ]
        if starts.ndim != 1 or stops.shape != starts.shape:
            return None, [
                f"{name}.start and {name}.end must be 1-D arrays of equal "
                f"length, got shapes {starts.shape} and {stops.shape}"
            ]
        rows = np.column_stack([starts, stops])
    else:
        try:
            rows = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None, [
                f"{name} could not be read as numbers "
                f"(got {type(value).__name__}: {value!r:.80})"
            ]
        if rows.shape == (2,):
            rows = rows.reshape(1, 2)
        if rows.ndim != 2 or rows.shape[1] != 2:
            return None, [f"{name} has shape {rows.shape}; expected (2,) or (n, 2)"]
    if rows.shape[0] == 0:
        return None, [f"{name} has no rows, so it would exclude all data"]

    problems: list[str] = []
    finite = np.isfinite(rows).all(axis=1)
    nonfinite = np.flatnonzero(~finite)
    if nonfinite.size:
        problems.append(
            f"{name} row(s) {_format_rows(nonfinite)} contain NaN or inf "
            f"(e.g. {rows[nonfinite[0]].tolist()})"
        )
    inverted = np.flatnonzero(finite & (rows[:, 1] <= rows[:, 0]))
    if inverted.size:
        problems.append(
            f"{name} row(s) {_format_rows(inverted)} have stop <= start "
            f"(e.g. {rows[inverted[0]].tolist()})"
        )
    return rows, problems


def _merge(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sort rows by start and merge overlapping or touching rows (vectorized)."""
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    running_stop = np.maximum.accumulate(rows[:, 1])
    opens_new = np.ones(rows.shape[0], dtype=bool)
    opens_new[1:] = rows[1:, 0] > running_stop[:-1]
    first = np.flatnonzero(opens_new)
    return np.column_stack([rows[first, 0], np.maximum.reduceat(rows[:, 1], first)])


def _raise_invalid(problems: list[str]) -> None:
    raise ValueError(
        "Invalid time window: "
        + "; ".join(problems)
        + ".\nWhy: epochs and spike_window are half-open [start, stop) windows "
        "in seconds on the same clock as `times`; every row needs a finite "
        f"start < stop. Accepted forms: {_ACCEPTED_FORMS}.\n"
        "Fix: pass e.g. epochs=[(0.0, 100.0), (1100.0, 1200.0)] or "
        "spike_window=(0.0, 1200.0), or None for no restriction."
    )


def as_intervals(value: Any, *, name: str) -> NDArray[np.float64] | None:
    """Normalize a time-window argument to sorted, merged ``(n, 2)`` rows.

    Parameters
    ----------
    value : None, (start, stop), array-like of shape (n, 2), or IntervalSet-like
        The window(s). An object exposing 1-D ``.start`` and ``.end`` arrays
        (for example a pynapple ``IntervalSet``) is accepted by duck typing;
        pynapple is never imported.
    name : str
        Argument name used in error messages (``"epochs"``, ``"spike_window"``).

    Returns
    -------
    ndarray of float64, shape (n_windows, 2), or None
        Rows sorted by start, with overlapping or touching rows merged.
        ``None`` when ``value`` is ``None``.

    Raises
    ------
    ValueError
        On a wrong shape, zero rows, non-finite values or ``stop <= start``.
        Every problem is listed in one message.
    """
    if value is None:
        return None
    rows, problems = _parse_intervals(value, name)
    if problems:
        _raise_invalid(problems)
    assert rows is not None  # _parse_intervals returns rows when no problems
    return _merge(rows)


def resolve_time_windows(
    epochs: Any, spike_window: Any
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Normalize ``epochs`` and ``spike_window`` together.

    Problems in both arguments are reported in a single ``ValueError``.
    """
    resolved: list[NDArray[np.float64] | None] = []
    problems: list[str] = []
    for name, value in (("epochs", epochs), ("spike_window", spike_window)):
        if value is None:
            resolved.append(None)
            continue
        rows, found = _parse_intervals(value, name)
        problems.extend(found)
        resolved.append(None if found or rows is None else _merge(rows))
    if problems:
        _raise_invalid(problems)
    return resolved[0], resolved[1]


def intervals_contain(
    windows: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Test whether each ``[starts[i], stops[i])`` lies inside one window row.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, merged rows (the output of :func:`as_intervals`).
    starts, stops : ndarray, shape (n,)
        Query intervals.

    Returns
    -------
    ndarray of bool, shape (n,)
        ``True`` iff ``windows[j, 0] <= starts[i]`` and
        ``stops[i] <= windows[j, 1]`` for a single row ``j``. A query with a
        NaN start or stop is ``False``.
    """
    starts = np.asarray(starts, dtype=np.float64)
    stops = np.asarray(stops, dtype=np.float64)
    if windows.shape[0] == 0:
        return np.zeros(starts.shape, dtype=bool)
    idx = np.searchsorted(windows[:, 0], starts, side="right") - 1
    j = np.maximum(idx, 0)
    # searchsorted places a NaN start after every row, so idx alone cannot
    # reject it; the explicit start comparison is False for NaN.
    keep: NDArray[np.bool_] = (
        (idx >= 0) & (starts >= windows[j, 0]) & (stops <= windows[j, 1])
    )
    return keep


def intersect_intervals(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Intersect two sorted, merged interval sets (vectorized).

    Returns
    -------
    ndarray, shape (n, 2)
        Sorted, disjoint rows; shape ``(0, 2)`` when the sets do not overlap.
    """
    lo = np.searchsorted(
        b[:, 1], a[:, 0], side="right"
    )  # first b row ending after a_start
    hi = np.searchsorted(b[:, 0], a[:, 1], side="left")  # b rows starting before a_stop
    counts = np.maximum(hi - lo, 0)
    a_idx = np.repeat(np.arange(a.shape[0]), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    b_idx = np.repeat(lo, counts) + offsets
    starts = np.maximum(a[a_idx, 0], b[b_idx, 0])
    stops = np.minimum(a[a_idx, 1], b[b_idx, 1])
    keep = stops > starts
    return np.column_stack([starts[keep], stops[keep]])


def run_sample_bounds(interval_mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Return the first and last sample of every maximal run of valid intervals.

    Parameters
    ----------
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Per-interval validity (``interval_valid_mask`` output).

    Returns
    -------
    ndarray of intp, shape (n_runs, 2)
        Row ``r`` is ``(first_sample, last_sample)``, inclusive, so the run's
        samples are ``slice(first_sample, last_sample + 1)``. A sample whose
        two neighbouring intervals are both invalid belongs to no run.
    """
    padded = np.concatenate([[False], np.asarray(interval_mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return np.column_stack([edges[0::2], edges[1::2]]).astype(np.intp)


def run_time_bounds(
    times: NDArray[np.float64], interval_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Return ``[times[first], times[last]]`` for every maximal valid run.

    Returns
    -------
    ndarray, shape (n_runs, 2)
        Half-open ``[start, stop)`` time windows covering exactly the valid
        intervals.
    """
    bounds = run_sample_bounds(interval_mask)
    return np.column_stack([times[bounds[:, 0]], times[bounds[:, 1]]])
