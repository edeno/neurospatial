"""Time-window normalization, shared interval gates and observed runs.

Every analysis that combines data streams restricts itself to time inside the
optional ``epochs`` and ``spike_window`` arguments. This module is the single
place those arguments are parsed, merged and tested, so every analysis family
applies identical semantics: each row is a half-open ``[start, stop)`` window in
seconds, on the same clock as ``times``.

The interval mask also accepts precomputed speed and start-bin assignments,
keeping spike counts and occupancy on the same observed time support.
"""

from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, overload

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial._stacklevel import _external_stacklevel


class _PositionBinLookup(Protocol):
    """Map position samples to bin IDs for an optional interval-start gate."""

    def bin_at(self, positions: NDArray[np.float64]) -> NDArray[np.intp]: ...


_ACCEPTED_FORMS = (
    "None, a (start, stop) pair, an (n, 2) array-like of [start, stop) rows, "
    "or an object with 1-D .start and .end arrays (for example a pynapple "
    "IntervalSet)"
)


@dataclass(frozen=True, eq=False)
class TimeSupport(ReadOnlyArraysMixin):
    """Resolved windows and the exact intervals retained by an operation.

    ``mask`` gates both spike counts and dwell time. ``runs`` records its
    contiguous half-open time bounds without retaining the recording arrays;
    under linear time allocation, they are the fractional pieces retained
    inside those bounds. Recording support and training support are separate
    instances: training may additionally exclude slow movement or invalid
    coordinates. ``break_times`` holds the time of the first sample after
    each declared recording break (``times[k]`` where ``break_before[k]``,
    ``k >= 1``), or ``None`` when no ``break_before`` was given. The arrays
    are read-only copies.
    """

    mask: NDArray[np.bool_]
    runs: NDArray[np.float64]
    epochs: NDArray[np.float64] | None
    spike_window: NDArray[np.float64] | None
    max_gap: float | None
    break_times: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self._own_arrays()

    @classmethod
    def from_mask(
        cls,
        times: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        epochs: NDArray[np.float64] | None,
        spike_window: NDArray[np.float64] | None,
        max_gap: float | None,
        break_before: NDArray[np.bool_] | None = None,
        runs: NDArray[np.float64] | None = None,
    ) -> TimeSupport:
        """Snapshot an interval mask and already normalized time windows.

        ``break_before`` (already validated, shape ``(n_samples,)``) is
        recorded as the times of the samples that start a new recording.
        ``runs`` (shape ``(n_runs, 2)``) replaces the mask's run bounds when
        the retained time is a fraction of each run, as under linear time
        allocation.
        """
        return cls(
            mask,
            run_time_bounds(times, mask) if runs is None else runs,
            epochs,
            spike_window,
            max_gap,
            break_sample_times(times, break_before),
        )

    def analysis_settings(self) -> dict[str, Any]:
        """Export independent, JSON-compatible temporal provenance."""
        return {
            "epochs": None if self.epochs is None else self.epochs.tolist(),
            "spike_window": (
                None if self.spike_window is None else self.spike_window.tolist()
            ),
            "observed_intervals": self.runs.tolist(),
            "max_gap": None if self.max_gap is None else float(self.max_gap),
            "break_times": (
                None if self.break_times is None else self.break_times.tolist()
            ),
        }


def validate_max_gap(max_gap: Any) -> float | None:
    """Return ``max_gap`` as a float, or None; refuse any other value."""
    if max_gap is None:
        return None
    if (
        isinstance(max_gap, bool)
        or not isinstance(max_gap, numbers.Real)
        or not math.isfinite(max_gap)
        or max_gap <= 0
    ):
        raise ValueError(
            _format_error(
                "max_gap must be a positive finite number of seconds or None "
                f"(None keeps every interval, however long); got {max_gap!r}.",
                why="Why: sampling intervals longer than max_gap are excluded "
                "as recording gaps, so a zero, negative or NaN max_gap would "
                "exclude every interval.",
                fix="pass a number of seconds above the sampling interval, "
                "e.g. max_gap=0.5, or max_gap=None.",
            )
        )
    return float(max_gap)


def recording_index(break_before: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Index of the recording each sample belongs to, counting declared breaks.

    Parameters
    ----------
    break_before : ndarray of bool, shape (n_samples,)
        Already validated declared breaks; entry 0 is ignored.

    Returns
    -------
    ndarray of intp, shape (n_samples,)
        0 for the first recording, incremented at every break.
    """
    index: NDArray[np.intp] = np.cumsum(np.r_[False, break_before[1:]], dtype=np.intp)
    return index


@overload
def break_sample_times(
    times: NDArray[np.float64], break_before: NDArray[np.bool_]
) -> NDArray[np.float64]: ...
@overload
def break_sample_times(times: NDArray[np.float64], break_before: None) -> None: ...
def break_sample_times(
    times: NDArray[np.float64], break_before: NDArray[np.bool_] | None
) -> NDArray[np.float64] | None:
    """Times of the samples that start a new recording, or None.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times in seconds.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared breaks; entry 0 is ignored.

    Returns
    -------
    ndarray of float64, shape (n_breaks,), or None
        ``times[k]`` for every ``k >= 1`` with ``break_before[k]``; ``None``
        when ``break_before`` is ``None``.
    """
    if break_before is None:
        return None
    times = np.asarray(times, dtype=np.float64)
    flags = np.asarray(break_before, dtype=bool)
    return np.asarray(times[1:][flags[1:]], dtype=np.float64)


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
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, np.ndarray) and v.ndim == 1 for v in value)
    ):
        # The retired parallel (starts, stops) form. With two length-2 arrays
        # it also has shape (2, 2), so reading it as rows would be silent.
        return None, [
            f"{name} is a pair of 1-D arrays, the retired parallel "
            f"(starts, stops) form; pass np.column_stack([starts, stops]) for "
            f"those windows, or np.asarray(rows) if each array is one "
            f"[start, stop) row"
        ]
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


def interval_valid_mask(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    env: _PositionBinLookup | None = None,
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    start_bin: NDArray[np.intp] | None = None,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Compute the per-interval validity mask shared by spikes and occupancy.

    This is the **single source of truth** for which trajectory intervals are
    counted toward a firing-rate map. A rate map is
    ``spike_counts (numerator) / occupancy (denominator)`` per bin, and BOTH
    sides MUST drop the IDENTICAL set of intervals or the rate is biased.
    ``env.occupancy`` calls this helper to build its interval mask, and the
    spike-binning kernel gates each spike by the validity of the interval it
    falls in using this SAME helper — so numerator and denominator stay aligned
    by construction.

    Interval ``k`` spans ``[t_k, t_{k+1})`` (``time_allocation="start"``
    semantics) and is assigned to the bin at sample ``k``. It is valid iff:

    .. math::
        (\\text{max\\_gap is None or } \\Delta t_k \\le \\text{max\\_gap})
        \\;\\wedge\\;
        (\\text{min\\_speed is None or } \\text{speed}_k \\ge \\text{min\\_speed})
        \\;\\wedge\\;
        (\\text{start\\_bin}_k \\ge 0)
        \\;\\wedge\\;
        ([t_k, t_{k+1}) \\subseteq \\text{epochs})
        \\;\\wedge\\;
        ([t_k, t_{k+1}) \\subseteq \\text{spike\\_window})
        \\;\\wedge\\;
        \\neg\\,\\text{break\\_before}_{k+1}

    where ``start_bin = env.bin_at(positions)`` and ``\\Delta t = diff(times)``.
    Bounds are checked only when ``start_bin`` or ``env`` is supplied. Window
    gates apply when given.

    ``max_gap`` must sit above the sampling interval. Timestamps such as
    ``np.arange(n) / 30`` are not exact, so with ``max_gap`` equal to the
    sampling interval about a quarter to a half of the intervals exceed it by
    float rounding and are dropped. :func:`warn_excluded_intervals` names this
    case; use e.g. ``max_gap = 1.5 / sampling_rate`` or the default 0.5 s.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Trajectory timestamps in seconds (assumed sorted/finite; validated
        upstream).
    positions : ndarray, shape (n_samples, n_dims), or None
        Trajectory coordinates. Required with ``env`` unless ``start_bin``
        has already been computed.
    env : Environment or None
        The spatial environment; ``env.bin_at`` maps each start sample to its
        bin (``-1`` outside the active mask).
    speed : ndarray, shape (n_samples,), or None
        Per-sample speed, required with ``min_speed``; the gate uses
        ``speed[:-1] >= min_speed`` (the speed of each interval's start
        sample). Consumed as-is; never re-derived here.
    min_speed : float or None
        Minimum speed threshold. When ``None`` (default) no speed gating.
    max_gap : float or None
        Maximum time gap in seconds, positive and finite. Intervals with
        ``dt > max_gap`` are dropped. Default 0.5; ``None`` disables gap
        gating on BOTH sides (spikes and occupancy) so they stay aligned.
    start_bin : ndarray of intp, shape (n_samples,), or None
        Precomputed ``env.bin_at(positions)`` for the out-of-bounds-start
        gate, so callers that already binned ``positions`` skip a second
        ``bin_at``. It must come from the same ``positions``. When ``None``,
        it is computed from ``env`` if given.
    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Already normalized, sorted and merged half-open windows in seconds.
        Each kept interval must lie wholly inside one row of each supplied
        window set.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks: True at sample ``k + 1``
        drops interval ``k``. Entry 0 is ignored.

    Returns
    -------
    ndarray of bool, shape (n_samples - 1,)
        ``True`` for each valid interval. Empty (length 0) when fewer than
        two samples are provided.

    Raises
    ------
    ValueError
        If ``max_gap`` is not None or a positive finite number, or if
        ``min_speed`` is given without ``speed``.
    """
    validate_max_gap(max_gap)
    if min_speed is not None and speed is None:
        raise ValueError(
            _format_error(
                f"min_speed={min_speed} was given without speed.",
                why="Why: the speed gate compares each interval's start-sample "
                "speed with min_speed.",
                fix="pass speed= (one value per sample), or min_speed=None.",
            )
        )
    times = np.asarray(times, dtype=np.float64)

    n_samples = len(times)
    if n_samples < 2:
        return np.zeros(0, dtype=bool)

    dt = np.diff(times)
    valid_mask = np.ones(len(dt), dtype=bool)

    if max_gap is not None:
        valid_mask &= dt <= max_gap

    # Filter by min_speed (speed of each interval's starting sample).
    if min_speed is not None:
        speed = np.asarray(speed, dtype=np.float64)
        valid_mask &= speed[:-1] >= min_speed

    # Filter out intervals whose START sample is outside the active
    # environment (bin_at returns -1 for points outside any active bin).
    # Reuse a caller-supplied bin_at result when available (e.g. env.occupancy
    # already computes bin_at(positions)); otherwise compute it here.
    if start_bin is None and env is not None:
        pos = np.asarray(positions, dtype=np.float64)
        start_bin = env.bin_at(pos.reshape(-1, 1) if pos.ndim == 1 else pos)
    if start_bin is not None:
        valid_mask &= np.asarray(start_bin)[:-1] >= 0
    for windows in (epochs, spike_window):
        if windows is not None:
            valid_mask &= intervals_contain(windows, times[:-1], times[1:])
    if break_before is not None:
        valid_mask &= ~np.asarray(break_before, dtype=bool)[1:]

    return valid_mask


def start_allocated_occupancy(
    start_bin: NDArray[np.intp],
    dt: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    n_bins: int,
    *,
    return_seconds: bool = True,
) -> NDArray[np.float64]:
    """Sum each valid interval's duration into the bin of its start sample.

    Parameters
    ----------
    start_bin : ndarray of intp, shape (n_samples,)
        Bin of every sample (``-1`` = invalid; such intervals must already be
        excluded by ``interval_mask``).
    dt : ndarray, shape (n_samples - 1,)
        ``np.diff(times)``.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        The shared validity mask.
    n_bins : int
        Number of bins.
    return_seconds : bool, default=True
        Weight by ``dt`` (seconds) or count intervals.

    Returns
    -------
    ndarray, shape (n_bins,)

    Raises
    ------
    ValueError
        If a retained interval starts in a bin ``>= n_bins``.
    """
    bins = start_bin[:-1][interval_mask]
    if bins.size and int(bins.max()) >= n_bins:
        raise ValueError(
            f"A retained interval starts in bin {int(bins.max())}, but there "
            f"are only {n_bins} bins: start_bin does not come from this "
            "environment. Fix: compute start_bin with env.bin_at on the same "
            "environment."
        )
    weights = dt[interval_mask] if return_seconds else None
    return np.bincount(bins, weights=weights, minlength=n_bins).astype(np.float64)


def observed_interval_mask(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None = None,
    result: str = "the result is empty or NaN rather than a measurement",
) -> NDArray[np.bool_]:
    """Per-interval validity for position-only analyses (gap, epochs, breaks).

    ``epochs`` is the raw user argument; it is normalized here with
    ``as_intervals(epochs, name="epochs")``. ``break_before`` is already
    validated (``validate_break_before``). ``result`` names what an empty
    selection produces in the warning, as in `warn_excluded_intervals`.

    Returns
    -------
    ndarray of bool, shape (n_samples - 1,)
    """
    times = np.asarray(times, dtype=np.float64)
    resolved_epochs = as_intervals(epochs, name="epochs")
    mask = interval_valid_mask(
        times, max_gap=max_gap, epochs=resolved_epochs, break_before=break_before
    )
    warn_excluded_intervals(
        times,
        mask,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        result=result,
    )
    return mask


def warn_excluded_intervals(
    times: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    start_bin_gate: bool = False,
    break_before: NDArray[np.bool_] | None = None,
    result: str = "the result is empty or NaN rather than a measurement",
    suppress_hint: str = "",
) -> None:
    """Warn when the interval gates leave no data or ``times`` look wrong.

    Analyses call this with their final interval mask, so an empty or
    gap-dominated selection warns the same way everywhere. It emits at most
    one ``UserWarning``, and nothing for fewer than two samples:

    - When every interval is excluded, the message names the active gates and
      how to fix each one, so an empty result is not mistaken for a
      measurement.
    - Otherwise, when most sampling intervals exceed ``max_gap``, they count
      as recording gaps. That usually means ``times`` are not in seconds (for
      example milliseconds), and the message says so.
    - Otherwise, when some intervals exceed ``max_gap`` only by float
      rounding, ``max_gap`` equals the sampling interval, and the message
      says to set it above the sampling interval.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample timestamps, expected in seconds.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        The analysis's final interval validity.
    max_gap : float or None
        The gap threshold in seconds, or None when gap gating is off.
    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Active normalized time windows, named when supplied.
    min_speed : float or None
        Active speed threshold, named when set.
    start_bin_gate : bool, default=False
        Whether the mask also drops intervals at samples outside the
        environment or at NaN positions.
    break_before : ndarray of bool, shape (n_samples,), or None
        Declared recording breaks, named when any is set.
    result : str
        What an all-excluded mask leaves, e.g. ``"the rate map is empty"``.
    suppress_hint : str, optional
        Closing sentence naming how to silence the warning.
    """
    if interval_mask.size == 0:
        return
    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times)
    median_dt = float(np.median(dt))
    n_long = n_rounded = 0
    slow_clock = False
    if max_gap is not None:
        # Each difference is exact to a few ulp of the timestamps' magnitude;
        # an excess within that is rounding, not a gap.
        rounding = 8.0 * np.spacing(np.maximum(np.abs(times[:-1]), np.abs(times[1:])))
        excess = dt - max_gap
        n_long = int(np.count_nonzero(excess > rounding))
        n_rounded = int(np.count_nonzero((excess > 0) & (excess <= rounding)))
        slow_clock = float(np.median(excess - rounding)) > 0
    clock = (
        f"the median sampling interval is {median_dt:.4g} s, longer than "
        f"max_gap={max_gap} s, so `times` are probably not in seconds"
    )
    rounded = (
        f"{n_rounded} of {dt.size} sampling intervals exceed max_gap={max_gap} s "
        "only by float rounding of the timestamps, so max_gap equals the "
        "sampling interval"
    )
    rounded_fix = (
        f"set max_gap above the sampling interval, e.g. max_gap={1.5 * median_dt:.3g}"
    )
    if interval_mask.any():
        if slow_clock:
            message = (
                f"{n_long} of {dt.size} sampling intervals are longer than "
                f"max_gap={max_gap} s and are excluded as recording gaps: {clock}. "
                f"Fix: convert times to seconds (divide milliseconds by 1000); "
                f"if the sampling really is this coarse, raise max_gap above "
                f"{median_dt:.4g} or pass max_gap=None to disable gap gating. "
                f"{suppress_hint}"
            )
        elif n_rounded:
            message = (
                f"{rounded}; they are excluded as recording gaps. "
                f"Fix: {rounded_fix}. {suppress_hint}"
            )
        else:
            return
        warnings.warn(message.rstrip(), UserWarning, stacklevel=_external_stacklevel())
        return

    gates = [f"max_gap={max_gap}"] if max_gap is not None else []
    fixes = []
    if slow_clock:
        fixes.append(
            f"{clock}: convert times to seconds, raise max_gap above "
            f"{median_dt:.4g}, or pass max_gap=None to disable gap gating"
        )
    elif n_rounded:
        fixes.append(f"{rounded}: {rounded_fix}")
    elif n_long:
        fixes.append("pass max_gap=None to disable gap gating")
    if min_speed is not None:
        gates.append(f"min_speed={min_speed}")
        fixes.append(
            f"check that min_speed={min_speed} is in the positions' length "
            "units per second"
        )
    windows = [
        name
        for name, value in (("epochs", epochs), ("spike_window", spike_window))
        if value is not None
    ]
    if windows:
        gates.extend(windows)
        fixes.append(
            f"check that {'/'.join(windows)} overlap `times` (same clock, seconds)"
        )
    if break_before is not None and np.any(np.asarray(break_before)[1:]):
        gates.append("break_before")
        fixes.append(
            "check that break_before is True only at the first sample of each "
            "new recording"
        )
    if start_bin_gate:
        gates.append("samples out of bounds or at NaN positions")
        fixes.append("check that positions use the environment's coordinates")
    if not fixes:
        fixes.append("check that the samples are finite, observed values")
    active = f" (active gate(s) {', '.join(gates)})" if gates else ""
    message = (
        f"Interval filtering excluded ALL trajectory intervals{active}, so "
        f"{result}. Fix: {'; '.join(fixes)}. {suppress_hint}"
    )
    warnings.warn(message.rstrip(), UserWarning, stacklevel=_external_stacklevel())


def observed_runs(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None = None,
) -> list[slice]:
    """One sample slice per maximal run of valid intervals, in time order."""
    mask = observed_interval_mask(
        times, max_gap=max_gap, epochs=epochs, break_before=break_before
    )
    return [slice(int(a), int(b) + 1) for a, b in run_sample_bounds(mask)]
