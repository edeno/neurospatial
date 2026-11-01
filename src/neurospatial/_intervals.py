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

import warnings
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import EMPTY_SELECTION, ReadOnlyArraysMixin
from neurospatial._stacklevel import _external_stacklevel

_ACCEPTED_FORMS = (
    "None, a (start, stop) pair, an (n, 2) array-like of [start, stop) rows, "
    "an object with an (n, 2) .windows array (for example "
    "neurospatial.Intervals), or an object with 1-D .start and .end arrays "
    "(for example a pynapple IntervalSet)"
)


def pynapple_end_to_stop(end: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert pynapple's inclusive interval ends to half-open stops.

    A pynapple interval ``[start, end]`` holds exactly the floats of
    ``[start, nextafter(end, inf))``, so a sample at ``end`` stays inside.
    """
    return np.nextafter(end, np.inf)


def stop_to_pynapple_end(stop: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert half-open stops to pynapple's inclusive ends (the inverse)."""
    return np.nextafter(stop, -np.inf)


def _is_pynapple_object(value: Any) -> bool:
    """Return whether ``value`` is a pynapple object, without importing pynapple."""
    return type(value).__module__.partition(".")[0] == "pynapple"


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

    def temporal_settings(self) -> dict[str, Any]:
        """The temporal part of a result's ``analysis_settings``, JSON-compatible."""

        def listed(array: NDArray[np.float64] | None) -> list[Any] | None:
            return None if array is None else array.tolist()

        return {
            "epochs": listed(self.epochs),
            "spike_window": listed(self.spike_window),
            "observed_intervals": self.runs.tolist(),
            "max_gap": None if self.max_gap is None else float(self.max_gap),
            "break_times": listed(self.break_times),
        }


#: What an all-excluded selection leaves, named by the excluded-interval
#: warning when the analysis does not say more precisely.
EMPTY_RESULT = "the result is empty or NaN rather than a measurement"

#: Default longest stretch of missing tracking, in seconds, that is bridged.
#: Calibrated by simulation; see the Notes of :func:`validate_max_gap`.
DEFAULT_MAX_GAP = 1.0


def validate_max_gap(max_gap: Any) -> float | None:
    """Return ``max_gap`` as a float, or None; refuse any other value.

    ``max_gap`` is the longest stretch of missing tracking that is bridged:
    the time between two consecutive valid samples (a sample is missing when
    its row was dropped or its coordinates are NaN). A gap no longer than
    ``max_gap`` is treated as observed: its time goes to the last valid
    sample, or, where an analysis allocates time along the path
    (``time_allocation="linear"``), along the straight path to the next
    valid sample. A longer gap is excluded as a tracking loss. ``None``
    bridges every gap.

    Notes
    -----
    The default, ``DEFAULT_MAX_GAP = 1.0`` s, is calibrated by simulation
    (``scripts/calibrate_max_gap.py``, about two minutes). Scenarios: about
    15 min of running back and forth on a 200 cm linear track (30-50 cm/s
    runs, 2-6 s pauses at the ends), tracked at 30, 60 and 120 Hz, with 1 %
    or 5 % isolated dropped frames, DeepLabCut-style low-confidence NaN
    bursts (1-15 frames, about 3 % of frames, four times as frequent during
    the pauses at the ends), three 2-5 s tracking losses while running and a
    10 s pickup during a pause; 12 replicates per scenario. ``max_gap`` is
    swept from 0.03 s to 13 s on a log grid, and each value is scored
    against the true trajectory on occupancy (total variation and L1, 2 cm
    and 5 cm bins), the binned rate-map error of a place cell, and the
    running-bout and complete-traversal counts.

    Occupancy and bout counts need ``max_gap`` longer than the longest
    routine low-confidence burst (15 frames at 30 Hz, 0.53 s): at 0.5 s the
    bout-count error of 30 Hz recordings nearly doubles (1.0-1.2 against
    0.6 bouts per session). Occupancy degrades once ``max_gap`` bridges the
    real losses (2 s and longer), because a bridged gap's time goes to its
    last valid sample while the animal keeps running. From 0.56 s to 2.0 s,
    32 of the 42 scenario-metric pairs stay within two Monte-Carlo standard
    errors of their best value, more than in any other range; the rate-map
    error barely depends on ``max_gap``. The other ten are the six traversal
    counts, three bout counts and one occupancy total variation (60 Hz, 2 cm
    bins, 15 % above its best). Counts are best above 4 s, where the 2-5 s
    losses are bridged and the runs they interrupt are whole again; at 1.0 s
    each loss during a run costs about one traversal. 1.0 s is the round
    value nearest the geometric middle of the 0.56-2.0 s range (1.06 s).
    """
    # Imported here: _validation imports this module.
    from neurospatial._validation import is_finite_real

    if max_gap is None:
        return None
    if not is_finite_real(max_gap, positive=True):
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
    if value is EMPTY_SELECTION:
        return EMPTY_SELECTION, []
    kind = type(value).__name__
    if _is_pynapple_object(value) and kind != "IntervalSet":
        # A Ts/Tsd/TsdFrame has scalar .start/.end (its first and last
        # timestamps), which would read as one window over everything.
        return None, [
            f"{name} is a pynapple {kind}, a time series rather than a set of "
            f"intervals; pass the windows it was recorded in, "
            f"{name}={kind.lower()}.time_support"
        ]
    if not _is_pynapple_object(value) and hasattr(value, "windows"):
        # An interval container (neurospatial.Intervals, or any object with
        # an (n, 2) ``windows`` array of [start, stop) rows with duration).
        # Rows without duration are not in ``windows``; no rows is an empty
        # selection, not an error.
        try:
            rows = np.asarray(value.windows, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return None, [f"{name}.windows could not be read as numbers ({exc})"]
        if rows.ndim != 2 or rows.shape[1] != 2:
            return None, [f"{name}.windows has shape {rows.shape}; expected (n, 2)"]
        if rows.shape[0] == 0:
            return EMPTY_SELECTION, []
    elif hasattr(value, "start") and hasattr(value, "end"):
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
        if _is_pynapple_object(value):
            # pynapple intervals include their end; neurospatial windows are
            # half-open, so move each end to the next float.
            stops = pynapple_end_to_stop(stops)
        rows = np.column_stack([starts, stops])
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(not isinstance(v, (list, tuple)) and np.ndim(v) == 1 for v in value)
    ):
        # A pair of 1-D arrays (or Series, ...) is ambiguous: two
        # [start, stop) rows, or parallel starts and stops. With two windows
        # both readings have shape (2, 2), so either guess could be silently
        # wrong. Plain lists stay rows, the documented form.
        return None, [
            f"{name} is a pair of 1-D arrays, which could be two [start, stop) "
            f"rows or parallel starts and stops; pass np.column_stack([starts, "
            f"stops]) for starts and stops, or np.asarray(rows) if each array "
            f"is one row"
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
            hint = (
                "; if its rows are starts and stops, pass "
                "np.column_stack([starts, stops])"
                if rows.ndim == 2 and rows.shape[0] == 2
                else ""
            )
            return None, [
                f"{name} has shape {rows.shape}; expected (2,) or (n, 2){hint}"
            ]
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
    if rows.shape[0] == 0:
        return EMPTY_SELECTION
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
    value : None, (start, stop), array-like of shape (n, 2), Intervals, or \
IntervalSet-like
        The window(s). An object with a ``windows`` attribute, an ``(n, 2)``
        array of ``[start, stop)`` rows (for example
        :class:`~neurospatial.Intervals`), is read through it; its rows
        without duration are not windows, and no windows is a valid, empty
        selection. An object exposing 1-D ``.start`` and ``.end`` arrays
        (for example a pynapple ``IntervalSet``) is accepted by duck typing;
        pynapple is never imported. A pynapple interval includes its end, so
        its stop becomes the next float above ``end``: a sample exactly at
        ``end`` stays inside, as in pynapple.
    name : str
        Argument name used in error messages (``"epochs"``, ``"spike_window"``).

    Returns
    -------
    ndarray of float64, shape (n_windows, 2), or None
        Rows sorted by start, with overlapping or touching rows merged.
        ``None`` when ``value`` is ``None``; zero rows only for an object
        with empty ``windows``.

    Raises
    ------
    ValueError
        On a wrong shape, zero rows (except empty ``windows``), non-finite
        values or ``stop <= start``.
        Non-finite and inverted rows are all listed in one message; a value
        that is unreadable or wrongly shaped is reported on its own. A
        pynapple time series (``Ts``, ``Tsd``, ``TsdFrame``) is refused with
        a pointer to its ``time_support``.
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

    ``[t, t + ulp)`` lies in ``[start, stop)`` exactly when
    ``start <= t < stop``, so an instant at a window's stop is outside.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, 2)
        Sorted, merged rows (the output of :func:`as_intervals`).
    times : ndarray, shape (n,)
        Instants in seconds.

    Returns
    -------
    ndarray of bool, shape (n,)
        False for a NaN instant.
    """
    times = np.asarray(times, dtype=np.float64)
    if windows.shape[0] == 0:
        return np.zeros(times.shape, dtype=bool)
    row = np.searchsorted(windows[:, 0], times, side="right") - 1
    j = np.maximum(row, 0)
    inside: NDArray[np.bool_] = (
        (row >= 0) & (times >= windows[j, 0]) & (times < windows[j, 1])
    )
    return inside


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


def valid_run_samples(
    interval_mask: NDArray[np.bool_], sample_valid: NDArray[np.bool_]
) -> list[NDArray[np.intp]]:
    """Indices of the valid samples of every maximal run of valid intervals.

    Parameters
    ----------
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Per-interval validity, with missing samples bridged
        (``interval_valid_mask`` output with ``sample_valid``).
    sample_valid : ndarray of bool, shape (n_samples,)
        False at missing samples.

    Returns
    -------
    list of ndarray of intp
        One array per run of :func:`run_sample_bounds`, in time order: the
        run's samples with the missing ones skipped, as if their rows were
        deleted.

    Examples
    --------
    >>> runs = valid_run_samples(
    ...     np.array([True, True, False, True]),
    ...     np.array([True, False, True, True, True]),
    ... )
    >>> [run.tolist() for run in runs]
    [[0, 2], [3, 4]]
    """
    return [
        first + np.flatnonzero(sample_valid[first : last + 1])
        for first, last in run_sample_bounds(interval_mask)
    ]


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
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    start_bin: NDArray[np.intp] | None = None,
    require_end_bin: bool = False,
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
        (\\neg\\,\\text{require\\_end\\_bin} \\text{ or } \\text{start\\_bin}_b \\ge 0)
        \\;\\wedge\\;
        ([t_a, t_b) \\subseteq \\text{epochs})
        \\;\\wedge\\;
        ([t_a, t_b) \\subseteq \\text{spike\\_window})
        \\;\\wedge\\;
        \\neg\\,\\text{break\\_before}_{a+1..b}

    where ``start_bin`` is the bin of each sample (for example
    ``env.bin_at(positions)``). Under the default ``time_allocation="start"``
    a valid interval inside a bridged gap belongs to the bin of sample ``a``
    (:func:`hold_last_valid`); under ``"linear"`` the gap's time is spread
    along the straight path from ``a`` to ``b``
    (:func:`fill_missing_positions`). Either way occupancy and spike counts
    equal those of the recording with the missing rows deleted.
    Bounds are checked only when ``start_bin`` is supplied. Window gates
    apply when given.

    ``max_gap`` must sit above the sampling interval. Timestamps such as
    ``np.arange(n) / 30`` are not exact, so with ``max_gap`` equal to the
    sampling interval a large fraction of the intervals (about a sixth to
    a half, depending on frame rate and recording length) exceed it by float
    rounding and are dropped. :func:`warn_excluded_intervals` names this
    case.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Trajectory timestamps in seconds (assumed sorted/finite; validated
        upstream).
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
        Bin of every sample (``-1`` outside the active bins), for the
        out-of-bounds-start gate. ``None`` applies no bounds gate.
    require_end_bin : bool, default=False
        Also require each gap's end sample ``b`` to be in a bin: the
        complete-case rule of spatial rate maps and ``env.occupancy``, where
        the animal's place during an interval is known only when both of its
        samples are inside the environment; one that ends outside is left
        out rather than imputing where the animal was. Needs ``start_bin``.
        Frame-binned rate maps leave it False: in head-direction maps every
        observed sample has a heading bin, so the gate would change nothing;
        view maps pass no ``start_bin``; and in egocentric (object-vector)
        maps ``-1`` also marks an object closer than ``distance_range[0]``,
        a normal state rather than lost tracking, so requiring it would drop
        observed time.
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
        If ``max_gap`` is not None or a positive finite number, if ``speed``
        and ``min_speed`` are not given together with one speed per sample
        (:func:`validate_speed_gate`), or if ``require_end_bin`` is set
        without ``start_bin``.
    """
    validate_max_gap(max_gap)
    if require_end_bin and start_bin is None:
        msg = "require_end_bin=True needs start_bin, the bin of every sample."
        raise ValueError(msg)
    times = np.asarray(times, dtype=np.float64)
    speed = validate_speed_gate(speed, min_speed, times.size)
    if len(times) < 2:
        return np.zeros(0, dtype=bool)
    return gated_intervals(
        times,
        None if sample_valid is None else gap_endpoints(sample_valid),
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        start_bin=start_bin,
        require_end_bin=require_end_bin,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    )


def gated_intervals(
    times: NDArray[np.float64],
    endpoints: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]] | None,
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    start_bin: NDArray[np.intp] | None = None,
    require_end_bin: bool = False,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """The gates of :func:`interval_valid_mask` on already validated inputs.

    Parameters
    ----------
    times : ndarray of float64, shape (n_samples,), n_samples >= 2
    endpoints : tuple of ndarray, or None
        ``gap_endpoints(sample_valid)``, for callers that reuse it; None
        when every sample is valid.
    speed, min_speed, max_gap, start_bin, require_end_bin, epochs, spike_window, break_before
        As for :func:`interval_valid_mask`, already validated.

    Returns
    -------
    ndarray of bool, shape (n_samples - 1,)
    """
    start: NDArray[np.intp] | slice
    end: NDArray[np.intp] | slice
    if endpoints is None:
        # Each interval's own samples, as views rather than index arrays.
        start, end = slice(None, -1), slice(1, None)
        valid_mask = np.ones(times.size - 1, dtype=bool)
    else:
        start, end, linked = endpoints
        valid_mask = linked.copy()
    t_start, t_end = times[start], times[end]

    if max_gap is not None:
        valid_mask &= t_end - t_start <= max_gap

    # Filter by min_speed (speed of each gap's starting sample).
    if min_speed is not None and speed is not None:
        valid_mask &= np.asarray(speed, dtype=np.float64)[start] >= min_speed

    # Filter out intervals whose START sample is outside the active
    # environment (bin_at returns -1 for points outside any active bin).
    if start_bin is not None:
        bins = np.asarray(start_bin)
        valid_mask &= bins[start] >= 0
        if require_end_bin:
            valid_mask &= bins[end] >= 0
    for windows in (epochs, spike_window):
        if windows is not None:
            valid_mask &= intervals_contain(windows, t_start, t_end)
    if break_before is not None:
        recording = recording_index(np.asarray(break_before, dtype=bool))
        valid_mask &= recording[end] == recording[start]

    return valid_mask


@dataclass(frozen=True, eq=False)
class IntervalGates:
    """An interval mask and the gate inputs it was evaluated with.

    Built by :func:`interval_gates`, so :func:`warn_excluded_intervals`
    names exactly the gates that built the mask.

    Attributes
    ----------
    times : ndarray, shape (n_samples,)
        Sample times in seconds.
    mask : ndarray of bool, shape (max(n_samples - 1, 0),)
        True for each kept interval.
    max_gap : float or None
    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Normalized windows.
    min_speed : float or None
    start_bin_gate : bool
        Whether intervals were also dropped at samples outside the
        environment, at NaN positions or at samples without a bin.
    break_before : ndarray of bool, shape (n_samples,), or None
    sample_valid : ndarray of bool, shape (n_samples,), or None
        False at the missing samples the mask bridged like deleted rows.
    """

    times: NDArray[np.float64]
    mask: NDArray[np.bool_]
    max_gap: float | None
    epochs: NDArray[np.float64] | None = None
    spike_window: NDArray[np.float64] | None = None
    min_speed: float | None = None
    start_bin_gate: bool = False
    break_before: NDArray[np.bool_] | None = None
    sample_valid: NDArray[np.bool_] | None = None

    def narrowed(self, keep: NDArray[np.bool_]) -> IntervalGates:
        """These gates, with a bin check also dropping the intervals not in ``keep``.

        Parameters
        ----------
        keep : ndarray of bool, shape (n_samples - 1,)
            Intervals the bin or geometry check keeps.

        Returns
        -------
        IntervalGates
            ``mask & keep``; the bin check is named when it could drop an
            interval the other gates kept.
        """
        return replace(
            self,
            mask=self.mask & keep,
            start_bin_gate=self.start_bin_gate or bool(self.mask.any()),
        )


def interval_gates(
    times: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    start_bin: NDArray[np.intp] | None = None,
    require_end_bin: bool = False,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
) -> IntervalGates:
    """:func:`interval_valid_mask`, recorded with the inputs it was evaluated with.

    Parameters are those of :func:`interval_valid_mask`.

    Returns
    -------
    IntervalGates
    """
    times = np.asarray(times, dtype=np.float64)
    mask = interval_valid_mask(
        times,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        start_bin=start_bin,
        require_end_bin=require_end_bin,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
        sample_valid=sample_valid,
    )
    return IntervalGates(
        times,
        mask,
        max_gap,
        epochs=epochs,
        spike_window=spike_window,
        min_speed=min_speed,
        start_bin_gate=start_bin is not None,
        break_before=break_before,
        sample_valid=sample_valid,
    )


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
    recording = recording_index(np.asarray(break_before, dtype=bool))
    kept = np.zeros(index.size, dtype=bool)
    kept[1:] = recording[index[1:]] != recording[index[:-1]]
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


def validate_speed_gate(
    speed: Any, min_speed: float | None, n_samples: int
) -> NDArray[np.float64] | None:
    """Return ``speed`` as float64 when the speed gate is complete, else None.

    The speed gate needs both a threshold and one speed per sample; either
    one alone would be silently ignored or could not be applied.

    Parameters
    ----------
    speed : array-like, shape (n_samples,), or None
    min_speed : float or None
    n_samples : int

    Returns
    -------
    ndarray of float64, shape (n_samples,), or None
        None when neither is given.

    Raises
    ------
    ValueError
        If only one of ``speed`` and ``min_speed`` is given, or ``speed``
        does not have one value per sample.
    """
    if speed is None and min_speed is None:
        return None
    if speed is None:
        raise ValueError(
            _format_error(
                f"min_speed={min_speed} was given without speed.",
                why="the speed gate compares each interval's start-sample "
                "speed with min_speed.",
                fix="pass speed= (one value per sample, for example "
                "compute_speed(times, positions)), or min_speed=None.",
            )
        )
    if min_speed is None:
        raise ValueError(
            _format_error(
                "speed was given without min_speed, so it would be ignored.",
                why="speed only gates intervals against a min_speed threshold.",
                fix="pass min_speed=<threshold> to gate by speed, or omit speed.",
            )
        )
    values = np.asarray(speed, dtype=np.float64)
    if values.shape != (n_samples,):
        raise ValueError(
            _format_error(
                f"speed has shape {values.shape}, but times has {n_samples} samples.",
                why="the speed gate reads the speed of each interval's start sample.",
                fix="pass one speed per sample, e.g. compute_speed(times, positions).",
            )
        )
    return values


@dataclass(frozen=True, eq=False)
class StartAllocation:
    """Spatial support of start-allocated occupancy, shared with spike counts.

    Attributes
    ----------
    bins : ndarray of intp, shape (n_samples,)
        Bin of each sample; -1 outside the environment or at a missing one.
    held_bins : ndarray of intp, shape (n_samples,)
        ``bins`` with each missing sample holding the last valid sample's
        bin (:func:`hold_last_valid`), so a bridged gap's time and spikes go
        to the bin where it started.
    gates : IntervalGates
        The interval mask: every gate of :func:`interval_gates`, and the
        complete-case rule that both samples bounding an interval's gap are
        in a bin.
    """

    bins: NDArray[np.intp]
    held_bins: NDArray[np.intp]
    gates: IntervalGates

    def occupancy(
        self, n_bins: int, *, return_seconds: bool = True
    ) -> NDArray[np.float64]:
        """Seconds (or intervals) in each bin, shape (n_bins,)."""
        return start_allocated_occupancy(
            self.held_bins,
            np.diff(self.gates.times),
            self.gates.mask,
            n_bins,
            return_seconds=return_seconds,
        )


def start_allocation(
    times: NDArray[np.float64],
    bins: NDArray[np.intp],
    sample_valid: NDArray[np.bool_] | None,
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> StartAllocation:
    """Resolve start-allocated spatial support for occupancy and spike counts.

    One definition for ``env.occupancy``, spatial rate maps and the
    behavior analyses that bin dwell: an interval counts when its gap passes
    the gates of :func:`interval_valid_mask` and both samples bounding the
    gap are in a bin (the complete-case rule: where the animal was during an
    interval is known only when both of its samples are inside), and its
    time goes to the bin of the gap's first sample.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing, in seconds.
    bins : ndarray of intp, shape (n_samples,)
        Bin of each sample (for example ``env.bin_at(positions)``); -1
        outside the environment or at a missing sample.
    sample_valid : ndarray of bool, shape (n_samples,), or None
        False at missing samples (such as NaN positions), which are bridged
        like deleted rows. None makes every sample with bin -1 unobserved
        rather than missing.
    speed, min_speed, max_gap, epochs, spike_window, break_before
        The gates of :func:`interval_valid_mask`; ``speed`` and
        ``min_speed`` go together (:func:`validate_speed_gate`).

    Returns
    -------
    StartAllocation
    """
    bins = np.asarray(bins, dtype=np.intp)
    gates = interval_gates(
        times,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        start_bin=bins,
        require_end_bin=True,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
        sample_valid=sample_valid,
    )
    return StartAllocation(bins, hold_last_valid(bins, sample_valid, fill=-1), gates)


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
        msg = (
            f"A retained interval starts in bin {int(bins.max())}, but there "
            f"are only {n_bins} bins: start_bin does not come from this "
            "environment. Fix: compute start_bin with env.bin_at on the same "
            "environment."
        )
        raise ValueError(msg)
    weights = dt[interval_mask] if return_seconds else None
    return np.bincount(bins, weights=weights, minlength=n_bins).astype(np.float64)


def observed_interval_mask(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
    result: str = EMPTY_RESULT,
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
    gates = interval_gates(
        times,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=break_before,
        sample_valid=sample_valid,
    )
    warn_excluded_intervals(gates, result=result)
    return gates.mask


def warn_excluded_intervals(
    gates: IntervalGates,
    *,
    samples: Literal["positions", "headings"] = "positions",
    result: str = EMPTY_RESULT,
    suppress_hint: str = "",
) -> None:
    """Warn when the interval gates leave no data or ``times`` look wrong.

    Analyses call this with their final interval mask, so an empty or
    gap-dominated selection warns the same way everywhere. It emits at most
    one ``UserWarning``, and nothing for fewer than two samples:

    - When every interval is excluded, the message names the active gates and
      how to fix each one, so an empty result is not mistaken for a
      measurement. When fewer than two samples are observed at all, it names
      the missing samples instead, since no gate is to blame.
    - Otherwise, when most sampling intervals exceed ``max_gap``, they count
      as recording gaps. That usually means ``times`` are not in seconds (for
      example milliseconds), and the message says so.
    - Otherwise, when some intervals exceed ``max_gap`` only by float
      rounding, ``max_gap`` equals the sampling interval, and the message
      says to set it above the sampling interval.

    Parameters
    ----------
    gates : IntervalGates
        The analysis's final interval mask, with the gates that built it
        (:func:`interval_gates`). Times are expected in seconds. Active
        windows, ``min_speed``, declared breaks and the start-bin gate (NaN
        headings for ``"headings"``) are named; when every interval is
        excluded only because the gaps across missing samples
        (``sample_valid``) exceed ``max_gap``, the message says so.
    samples : {"positions", "headings"}, default="positions"
        What the start-bin gate bins, so the message names the right input.
    result : str
        What an all-excluded mask leaves, e.g. ``"the rate map is empty"``.
    suppress_hint : str, optional
        Closing sentence naming how to silence the warning.
    """
    if gates.mask.size == 0:
        return
    times, max_gap = gates.times, gates.max_gap
    dt = np.diff(times)
    if gates.mask.any():
        # Data are kept, so only intervals longer than max_gap can warn.
        if max_gap is None or not np.any(dt > max_gap):
            return
        message = _kept_intervals_message(times, dt, max_gap)
        if message is None:
            return
    else:
        message = _all_excluded_message(gates, dt, samples=samples, result=result)
    warnings.warn(
        f"{message} {suppress_hint}".rstrip(),
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _timestamp_rounding(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rounding allowance, in seconds, of the difference of timestamps ``a``, ``b``.

    Each difference is exact to a few ulp of the timestamps' magnitude, so an
    excess within this allowance is rounding, not a gap.
    """
    rounding: NDArray[np.float64] = 8.0 * np.spacing(np.maximum(np.abs(a), np.abs(b)))
    return rounding


def _gap_excess(
    times: NDArray[np.float64], dt: NDArray[np.float64], max_gap: float | None
) -> tuple[int, int, bool]:
    """Intervals longer than ``max_gap``: beyond rounding, by rounding only, median.

    Returns the number of intervals longer than ``max_gap`` by more than
    the timestamps' rounding, the number longer only by rounding, and
    whether the median interval is longer beyond rounding (a clock that is
    probably not in seconds). All zero or False when ``max_gap`` is None.
    """
    if max_gap is None:
        return 0, 0, False
    rounding = _timestamp_rounding(times[:-1], times[1:])
    excess = dt - max_gap
    n_long = int(np.count_nonzero(excess > rounding))
    n_rounded = int(np.count_nonzero((excess > 0) & (excess <= rounding)))
    return n_long, n_rounded, float(np.median(excess - rounding)) > 0


def _clock_problem(median_dt: float, max_gap: float | None) -> str:
    return (
        f"the median sampling interval is {median_dt:.4g} s, longer than "
        f"max_gap={max_gap} s, so `times` are probably not in seconds"
    )


def _rounded_problem(n_rounded: int, n_intervals: int, max_gap: float | None) -> str:
    return (
        f"{n_rounded} of {n_intervals} sampling intervals exceed max_gap={max_gap} s "
        "only by float rounding of the timestamps, so max_gap equals the "
        "sampling interval"
    )


def _rounded_fix(median_dt: float) -> str:
    return (
        f"set max_gap above the sampling interval, e.g. max_gap={1.5 * median_dt:.3g}"
    )


def _kept_intervals_message(
    times: NDArray[np.float64], dt: NDArray[np.float64], max_gap: float
) -> str | None:
    """The warning for a selection that keeps data, or None when ``times`` look right."""
    n_long, n_rounded, slow_clock = _gap_excess(times, dt, max_gap)
    median_dt = float(np.median(dt))
    if slow_clock:
        return (
            f"{n_long} of {dt.size} sampling intervals are longer than "
            f"max_gap={max_gap} s and are excluded as recording gaps: "
            f"{_clock_problem(median_dt, max_gap)}. "
            f"Fix: convert times to seconds (divide milliseconds by 1000); "
            f"if the sampling really is this coarse, raise max_gap above "
            f"{median_dt:.4g} or pass max_gap=None to disable gap gating."
        )
    if n_rounded:
        return (
            f"{_rounded_problem(n_rounded, dt.size, max_gap)}; they are excluded "
            f"as recording gaps. Fix: {_rounded_fix(median_dt)}."
        )
    return None


_SAMPLES_FIX = (
    "check that the samples are finite, observed values (NaN marks a missing sample)"
)
_HEADINGS_FIX = (
    "check that headings are finite while the animal was tracked (NaN marks a "
    "missing sample; all-NaN headings usually mean the heading was never "
    "computed for these times)"
)


def _all_excluded_message(
    gates: IntervalGates,
    dt: NDArray[np.float64],
    *,
    samples: Literal["positions", "headings"],
    result: str,
) -> str:
    """The warning for a selection that excludes every interval."""
    empty = [
        name
        for name, value in (
            ("epochs", gates.epochs),
            ("spike_window", gates.spike_window),
        )
        if value is not None and value.shape[0] == 0
    ]
    if empty:
        # A deliberate empty selection, such as a condition with no trials:
        # nothing is wrong with the clock, so say what happened.
        verb = "select" if len(empty) > 1 else "selects"
        return (
            f"{' and '.join(empty)} {verb} no time (an Intervals with no windows, "
            f"such as a condition with no trials), so {result}. Fix: select "
            "intervals with duration, or leave this condition out."
        )
    times, max_gap, sample_valid = gates.times, gates.max_gap, gates.sample_valid
    min_speed, break_before = gates.min_speed, gates.break_before
    if sample_valid is not None and not gap_endpoints(sample_valid)[2].any():
        # No interval has an observed sample on each side: the missing samples
        # leave nothing to gate, whatever the gates are.
        n_observed = int(np.count_nonzero(sample_valid))
        return (
            "Interval filtering excluded ALL trajectory intervals: only "
            f"{n_observed} of {times.size} samples are observed, and an interval "
            f"needs an observed sample on each side, so {result}. Fix: "
            f"{_HEADINGS_FIX if samples == 'headings' else _SAMPLES_FIX}."
        )

    n_long, n_rounded, slow_clock = _gap_excess(times, dt, max_gap)
    median_dt = float(np.median(dt))
    named = [f"max_gap={max_gap}"] if max_gap is not None else []
    fixes = []
    if slow_clock:
        fixes.append(
            f"{_clock_problem(median_dt, max_gap)}: convert times to seconds, "
            f"raise max_gap above {median_dt:.4g}, or pass max_gap=None to "
            "disable gap gating"
        )
    elif n_rounded:
        fixes.append(
            f"{_rounded_problem(n_rounded, dt.size, max_gap)}: "
            f"{_rounded_fix(median_dt)}"
        )
    elif n_long:
        fixes.append("pass max_gap=None to disable gap gating")
    elif max_gap is not None and sample_valid is not None:
        start, end, linked = gap_endpoints(sample_valid)
        bridged = times[end] - times[start]
        rounding = _timestamp_rounding(times[start], times[end])
        n_bridged_long = int(np.count_nonzero(linked & (bridged - max_gap > rounding)))
        if n_bridged_long:
            fixes.append(
                f"{n_bridged_long} of {dt.size} sampling intervals lie in gaps "
                f"across missing samples (non-finite rows) longer than "
                f"max_gap={max_gap} s; raise max_gap above the longest gap that "
                "should be bridged (the longest is "
                f"{float(np.max(bridged[linked])):.4g} s), or pass max_gap=None "
                "to disable gap gating"
            )
    if min_speed is not None:
        named.append(f"min_speed={min_speed}")
        fixes.append(
            f"check that min_speed={min_speed} is in the positions' length "
            "units per second"
        )
    windows = [
        name
        for name, value in (
            ("epochs", gates.epochs),
            ("spike_window", gates.spike_window),
        )
        if value is not None
    ]
    if windows:
        named.extend(windows)
        fixes.append(
            f"check that {'/'.join(windows)} overlap `times` (same clock, seconds)"
        )
    if break_before is not None and np.any(np.asarray(break_before)[1:]):
        named.append("break_before")
        fixes.append(
            "check that break_before is True only at the first sample of each "
            "new recording"
        )
    if gates.start_bin_gate and samples == "headings":
        named.append("samples at NaN headings")
        fixes.append(_HEADINGS_FIX)
    elif gates.start_bin_gate:
        named.append("samples out of bounds or at NaN positions")
        fixes.append("check that positions use the environment's coordinates")
    if not fixes:
        fixes.append("check that the samples are finite, observed values")
    active = f" (active gate(s) {', '.join(named)})" if named else ""
    return (
        f"Interval filtering excluded ALL trajectory intervals{active}, so "
        f"{result}. Fix: {'; '.join(fixes)}."
    )
