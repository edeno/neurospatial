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
from typing import Any, Literal, Protocol, overload

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

        def listed(array: NDArray[np.float64] | None) -> list[Any] | None:
            return None if array is None else array.tolist()

        return {
            "epochs": listed(self.epochs),
            "spike_window": listed(self.spike_window),
            "observed_intervals": self.runs.tolist(),
            "max_gap": None if self.max_gap is None else float(self.max_gap),
            "break_times": listed(self.break_times),
        }


#: Default longest stretch of missing tracking, in seconds, that is bridged.
#: Calibrated by simulation; see the Notes of :func:`validate_max_gap`.
DEFAULT_MAX_GAP = 1.0


def validate_max_gap(max_gap: Any) -> float | None:
    """Return ``max_gap`` as a float, or None; refuse any other value.

    ``max_gap`` is the longest stretch of missing tracking that is bridged:
    the time between two consecutive valid samples (a sample is missing when
    its row was dropped or its coordinates are NaN). A gap no longer than
    ``max_gap`` is treated as observed, with its time held at the last valid
    sample; a longer gap is excluded as a tracking loss. ``None`` bridges
    every gap.

    Notes
    -----
    The default, ``DEFAULT_MAX_GAP = 1.0`` s, is calibrated by simulation.
    Scenarios: 15 min of running back and forth on a 200 cm linear track
    (30-50 cm/s runs, 2-6 s pauses at the ends), tracked at 30, 60 and
    120 Hz, with 1 % or 5 % isolated dropped frames, DeepLabCut-style
    low-confidence NaN bursts (1-15 frames, about 3 % of frames, four times
    as frequent during the pauses near the walls), three 2-5 s tracking
    losses while running and one ~10 s pickup during a pause; 12 replicates
    per scenario. ``max_gap`` was swept from 0.03 s to 10 s on a log grid,
    and each value was scored on occupancy error against the true trajectory
    (total variation and L1, 2 cm and 5 cm bins), the binned rate-map error
    of a place cell, running-bout count and complete track traversals.

    Bout and traversal counts and the occupancy L1 error need ``max_gap``
    longer than the longest routine low-confidence burst (15 frames at 30 Hz,
    0.5 s), and every metric degrades once ``max_gap`` reaches the shortest
    real loss (2 s). From 0.55 s to 2.0 s, 38 of the 42 scenario-metric pairs
    stay within two Monte-Carlo standard errors of their best value; the
    other four, occupancy total-variation errors, are within 26 % of their
    best, because a bridged burst holds the last position while the animal
    keeps running. 1.0 s is the round value nearest the geometric middle of
    that plateau (1.05 s). 0.5 s lies just below it: at 30 Hz it splits
    running bouts at the longest bursts.
    """
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
                f"(None bridges every gap, however long); got {max_gap!r}.",
                why="Why: max_gap is the longest stretch of missing tracking "
                "that is bridged, so a zero, negative or NaN max_gap would "
                "exclude every interval.",
                fix="pass a number of seconds longer than routine dropped or "
                "low-confidence frames and shorter than real tracking losses, "
                f"e.g. the default max_gap={DEFAULT_MAX_GAP}, or max_gap=None.",
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


def _resolve_intervals(
    value: Any, name: str
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Merged rows (None for None or on problems) and every problem found."""
    if value is None:
        return None, []
    rows, problems = _parse_intervals(value, name)
    if problems or rows is None:
        return None, problems
    return _merge(rows), []


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
        Non-finite and inverted rows are all listed in one message; a value
        that is unreadable or wrongly shaped is reported on its own.
    """
    rows, problems = _resolve_intervals(value, name)
    if problems:
        _raise_invalid(problems)
    return rows


def resolve_time_windows(
    epochs: Any, spike_window: Any
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Normalize ``epochs`` and ``spike_window`` together.

    Problems in both arguments are reported in a single ``ValueError``.
    """
    epoch_rows, epoch_problems = _resolve_intervals(epochs, "epochs")
    window_rows, window_problems = _resolve_intervals(spike_window, "spike_window")
    if epoch_problems or window_problems:
        _raise_invalid(epoch_problems + window_problems)
    return epoch_rows, window_rows


def ragged_arange(counts: NDArray[np.integer[Any]]) -> NDArray[np.intp]:
    """Concatenate ``arange(counts[i])`` for every ``i`` (vectorized).

    Examples
    --------
    >>> ragged_arange(np.array([2, 0, 3])).tolist()
    [0, 1, 0, 1, 2]
    """
    offsets: NDArray[np.intp] = np.arange(counts.sum()) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return offsets


def times_in_intervals(
    windows: NDArray[np.float64], times: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether each instant lies inside a window row, tested as ``[t, t + ulp)``.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, merged rows (the output of :func:`as_intervals`).
    times : ndarray, shape (n,)
        Instants in seconds.

    Returns
    -------
    ndarray of bool, shape (n,)
    """
    times = np.asarray(times, dtype=np.float64)
    return intervals_contain(windows, times, np.nextafter(times, np.inf))


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

    Parameters
    ----------
    a, b : ndarray, shape (n_a, 2) and (n_b, 2)
        Sorted, disjoint half-open rows in seconds (:func:`as_intervals`
        output).

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
    b_idx = np.repeat(lo, counts) + ragged_arange(counts)
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

    Notes
    -----
    Read as indices into ``interval_mask`` itself, row ``r`` is the half-open
    ``[start, stop)`` of its ``r``-th maximal stretch of True, so this also
    finds the stretches of any Boolean array, such as samples inside a region.
    """
    padded = np.concatenate([[False], np.asarray(interval_mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return np.column_stack([edges[0::2], edges[1::2]]).astype(np.intp)


def run_time_bounds(
    times: NDArray[np.float64], interval_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Return ``(times[first], times[last])`` for every maximal valid run.

    Returns
    -------
    ndarray, shape (n_runs, 2)
        Half-open ``[start, stop)`` time windows covering exactly the valid
        intervals.
    """
    bounds = run_sample_bounds(interval_mask)
    return np.column_stack([times[bounds[:, 0]], times[bounds[:, 1]]])


def gap_endpoints(
    sample_valid: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """Valid samples bounding each sampling interval, across missing samples.

    Missing samples (``sample_valid`` False) are treated exactly like deleted
    rows: the gap containing interval ``k`` runs from the last valid sample at
    or before ``k`` to the first valid sample at or after ``k + 1``. Gates are
    then evaluated on that gap, so a NaN row and a dropped row give the same
    result.

    Parameters
    ----------
    sample_valid : ndarray of bool, shape (n_samples,)

    Returns
    -------
    start, end : ndarray of intp, shape (max(n_samples - 1, 0),)
        Bounding valid samples of the gap containing each interval. An
        interval before the first or after the last valid sample has no such
        gap; its own samples ``(k, k + 1)`` are returned instead.
    linked : ndarray of bool, shape (max(n_samples - 1, 0),)
        True where a valid sample exists on both sides.

    Examples
    --------
    >>> start, end, linked = gap_endpoints(np.array([True, False, False, True, False]))
    >>> start.tolist(), end.tolist(), linked.tolist()
    ([0, 0, 0, 3], [3, 3, 3, 4], [True, True, True, False])
    """
    valid = np.asarray(sample_valid, dtype=bool)
    n = valid.size
    if n < 2:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, np.zeros(0, dtype=bool)
    index = np.arange(n, dtype=np.intp)
    previous = np.maximum.accumulate(np.where(valid, index, -1))[:-1]
    following = np.minimum.accumulate(np.where(valid, index, n)[::-1])[::-1][1:]
    linked = (previous >= 0) & (following < n)
    start = np.where(linked, previous, index[:-1]).astype(np.intp)
    end = np.where(linked, following, index[1:]).astype(np.intp)
    return start, end, linked


def interval_valid_mask(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    env: _PositionBinLookup | None = None,
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    start_bin: NDArray[np.intp] | None = None,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
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

    Missing samples (``sample_valid`` False) are treated exactly like deleted
    rows. Interval ``k`` spans ``[t_k, t_{k+1})`` and belongs to the gap
    ``[t_a, t_b)`` between the last valid sample ``a <= k`` and the first
    valid sample ``b >= k + 1`` (:func:`gap_endpoints`); without missing
    samples, ``a = k`` and ``b = k + 1``. Interval ``k`` is valid iff a valid
    sample exists on both sides and:

    .. math::
        (\\text{max\\_gap is None or } t_b - t_a \\le \\text{max\\_gap})
        \\;\\wedge\\;
        (\\text{min\\_speed is None or } \\text{speed}_a \\ge \\text{min\\_speed})
        \\;\\wedge\\;
        (\\text{start\\_bin}_a \\ge 0)
        \\;\\wedge\\;
        ([t_a, t_b) \\subseteq \\text{epochs})
        \\;\\wedge\\;
        ([t_a, t_b) \\subseteq \\text{spike\\_window})
        \\;\\wedge\\;
        \\neg\\,\\text{break\\_before}_{a+1..b}

    where ``start_bin = env.bin_at(positions)``. A valid interval inside a
    bridged gap belongs to the bin of sample ``a`` (:func:`hold_last_valid`),
    so occupancy and spike counts equal those of the recording with the
    missing rows deleted. Bounds are checked only when ``start_bin`` or
    ``env`` is supplied. Window gates apply when given.

    ``max_gap`` must sit above the sampling interval. Timestamps such as
    ``np.arange(n) / 30`` are not exact, so with ``max_gap`` equal to the
    sampling interval about a quarter to a half of the intervals exceed it by
    float rounding and are dropped. :func:`warn_excluded_intervals` names this
    case.

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
        Per-sample speed, required with ``min_speed``; the gate uses the
        speed of each gap's start sample. Consumed as-is; never re-derived
        here.
    min_speed : float or None
        Minimum speed threshold. When ``None`` (default) no speed gating.
    max_gap : float or None, default=1.0
        Longest gap in seconds between consecutive valid samples that is
        bridged (see :func:`validate_max_gap`); longer gaps are dropped.
        ``None`` bridges every gap, on BOTH sides (spikes and occupancy) so
        they stay aligned.
    start_bin : ndarray of intp, shape (n_samples,), or None
        Precomputed ``env.bin_at(positions)`` for the out-of-bounds-start
        gate, so callers that already binned ``positions`` skip a second
        ``bin_at``. It must come from the same ``positions``. When ``None``,
        it is computed from ``env`` if given.
    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Already normalized, sorted and merged half-open windows in seconds.
        Each kept gap must lie wholly inside one row of each supplied
        window set.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks: True at any sample
        ``a + 1 .. b`` drops the gap ``[t_a, t_b)``. Entry 0 is ignored.
    sample_valid : ndarray of bool, shape (n_samples,), or None
        False at missing samples (for example NaN coordinates). ``None``
        treats every sample as valid.

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

    if sample_valid is None:
        start = np.arange(n_samples - 1, dtype=np.intp)
        end = start + 1
        valid_mask = np.ones(n_samples - 1, dtype=bool)
    else:
        start, end, valid_mask = gap_endpoints(sample_valid)
    t_start, t_end = times[start], times[end]

    if max_gap is not None:
        valid_mask &= t_end - t_start <= max_gap

    # Filter by min_speed (speed of each gap's starting sample).
    if min_speed is not None:
        speed = np.asarray(speed, dtype=np.float64)
        valid_mask &= speed[start] >= min_speed

    # Filter out intervals whose START sample is outside the active
    # environment (bin_at returns -1 for points outside any active bin).
    # Reuse a caller-supplied bin_at result when available (e.g. env.occupancy
    # already computes bin_at(positions)); otherwise compute it here.
    if start_bin is None and env is not None:
        pos = np.asarray(positions, dtype=np.float64)
        start_bin = env.bin_at(pos.reshape(-1, 1) if pos.ndim == 1 else pos)
    if start_bin is not None:
        valid_mask &= np.asarray(start_bin)[start] >= 0
    for windows in (epochs, spike_window):
        if windows is not None:
            valid_mask &= intervals_contain(windows, t_start, t_end)
    if break_before is not None:
        n_breaks = np.cumsum(np.r_[False, np.asarray(break_before, dtype=bool)[1:]])
        valid_mask &= n_breaks[end] == n_breaks[start]

    return valid_mask


def hold_last_valid(
    values: NDArray[Any], sample_valid: NDArray[np.bool_] | None, *, fill: Any
) -> NDArray[Any]:
    """Each sample's value, with a missing sample holding the last valid one.

    Parameters
    ----------
    values : ndarray, shape (n_samples, ...)
        Per-sample values, for example frame bins.
    sample_valid : ndarray of bool, shape (n_samples,), or None
        False at missing samples; ``None`` returns ``values`` unchanged.
    fill : scalar
        Value for missing samples before the first valid sample.

    Returns
    -------
    ndarray, shape (n_samples, ...)
        Under :func:`interval_valid_mask` with the same ``sample_valid``,
        entry ``k`` of a valid interval is the value of its gap's start
        sample.

    Examples
    --------
    >>> hold_last_valid(
    ...     np.array([4, -1, -1, 7]), np.array([True, False, False, True]), fill=-1
    ... ).tolist()
    [4, 4, 4, 7]
    """
    if sample_valid is None:
        return values
    valid = np.asarray(sample_valid, dtype=bool)
    index = np.arange(valid.size)
    last = np.maximum.accumulate(np.where(valid, index, -1))
    held: NDArray[Any] = values[np.maximum(last, 0)].copy()
    held[last < 0] = fill
    return held


def drop_missing_samples(
    sample_valid: NDArray[np.bool_],
    break_before: NDArray[np.bool_] | None,
) -> tuple[NDArray[np.intp], NDArray[np.bool_] | None]:
    """Indices of the valid samples, and the declared breaks between them.

    Deleting the missing rows is the definition every analysis follows for a
    missing sample; analyses that work on the remaining rows directly use
    these indices. A break declared at a deleted row moves to the next kept
    row, so it still separates the recordings.

    Parameters
    ----------
    sample_valid : ndarray of bool, shape (n_samples,)
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared breaks.

    Returns
    -------
    index : ndarray of intp, shape (n_valid,)
        The kept rows.
    break_before : ndarray of bool, shape (n_valid,), or None
        True at kept row ``i >= 1`` when a break is declared at any row in
        ``(index[i - 1], index[i]]``.
    """
    index = np.flatnonzero(sample_valid).astype(np.intp)
    if break_before is None:
        return index, None
    n_breaks = np.cumsum(np.r_[False, np.asarray(break_before, dtype=bool)[1:]])
    kept = np.zeros(index.size, dtype=bool)
    kept[1:] = n_breaks[index[1:]] != n_breaks[index[:-1]]
    return index, kept


def fill_missing_positions(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    time_allocation: Literal["start", "linear"],
) -> NDArray[np.float64]:
    """Fill missing rows from the valid rows that bound their gap.

    Used after the interval mask has been resolved with ``sample_valid``
    (missing rows are those with a non-finite coordinate), so the filled rows
    only place the time and spikes of bridged gaps. Under ``"start"``
    allocation a missing row holds the last valid position, so the whole gap
    belongs to its start bin; under ``"linear"`` allocation it lies on the
    straight path between the bounding valid rows at its time, so the path
    is split exactly as the gap's single straight segment would be. Either
    way the result equals that of the recording with the missing rows
    deleted. Rows before the first or after the last valid row stay NaN.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
    time_allocation : {"start", "linear"}

    Returns
    -------
    ndarray, shape (n_samples, n_dims)
        A copy with bounded missing rows filled.
    """
    positions = np.array(positions, dtype=np.float64, copy=True)
    valid = np.asarray(np.isfinite(positions).all(axis=1), dtype=bool)
    if valid.all() or not valid.any():
        return positions
    start, end, linked = gap_endpoints(valid)
    missing = np.flatnonzero(~valid[1:-1]) + 1
    missing = missing[linked[missing]]
    a, b = start[missing], end[missing]
    if time_allocation == "start":
        positions[missing] = positions[a]
    else:
        weight = ((times[missing] - times[a]) / (times[b] - times[a]))[:, None]
        positions[missing] = (1 - weight) * positions[a] + weight * positions[b]
    return positions


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
    sample_valid: NDArray[np.bool_] | None = None,
    result: str = "the result is empty or NaN rather than a measurement",
) -> NDArray[np.bool_]:
    """Per-interval validity for position-only analyses (gap, epochs, breaks).

    ``epochs`` is the raw user argument; it is normalized here with
    ``as_intervals(epochs, name="epochs")``. ``break_before`` is already
    validated (``validate_break_before``). ``sample_valid`` marks missing
    samples, which are bridged like deleted rows (:func:`interval_valid_mask`).
    ``result`` names what an empty selection produces in the warning, as in
    :func:`warn_excluded_intervals`.

    Returns
    -------
    ndarray of bool, shape (n_samples - 1,)
    """
    times = np.asarray(times, dtype=np.float64)
    resolved_epochs = as_intervals(epochs, name="epochs")
    mask = interval_valid_mask(
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        break_before=break_before,
        sample_valid=sample_valid,
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
    samples: Literal["positions", "headings"] = "positions",
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
        environment or at NaN positions (NaN headings for ``"headings"``).
    samples : {"positions", "headings"}, default="positions"
        What the start-bin gate bins, so the message names the right input.
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
    if start_bin_gate and samples == "headings":
        gates.append("samples at NaN headings")
        fixes.append(
            "check that headings are finite while the animal was tracked "
            "(NaN marks a missing sample; all-NaN headings usually mean the "
            "heading was never computed for these times)"
        )
    elif start_bin_gate:
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
    sample_valid: NDArray[np.bool_] | None = None,
) -> list[NDArray[np.intp]]:
    """Indices of the valid samples of each maximal run, in time order.

    A run is a maximal sequence of valid samples joined by bridged gaps
    (:func:`interval_valid_mask`), so missing samples inside it are skipped
    exactly as if their rows were deleted.
    """
    mask = observed_interval_mask(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        sample_valid=sample_valid,
    )
    runs = [np.arange(a, b + 1, dtype=np.intp) for a, b in run_sample_bounds(mask)]
    if sample_valid is None:
        return runs
    valid = np.asarray(sample_valid, dtype=bool)
    return [run[valid[run]] for run in runs]
