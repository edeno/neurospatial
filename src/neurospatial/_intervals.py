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
from typing import Any, Literal, NamedTuple, overload

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import DISCARD_WARN_FRACTION
from neurospatial._exceptions import InputError, _format_error
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial._stacklevel import _external_stacklevel

_ACCEPTED_FORMS = (
    "None, a (start, stop) pair, an (n, 2) array-like of [start, stop) rows "
    "in seconds, a table with start_time and stop_time columns, "
    "an object with an (n, 2) .windows array (for example "
    "neurospatial.Intervals), one interval record with .start_time and "
    ".stop_time (for example a Trial), or an object with 1-D .start and .end "
    "arrays (for example a pynapple IntervalSet)"
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

#: Shortest run of removed samples, in seconds, that counts as a sign of
#: boolean masking. A speed mask (speed > 5 cm/s) on a simulated 100 Hz open
#: field removes pauses with a median of 24 samples (0.24 s); its runs of at
#: least this length make up about 9% of the analysed time.
#: Dropped tracking frames can last longer: low-likelihood frames removed
#: during an occlusion (``df[df.likelihood > 0.9]``) span 0.3-0.8 s, so the
#: warning names them as the other cause of such runs.
MASKED_RUN_MIN_DURATION = 0.25

#: Largest distance, as a fraction of the sampling interval, from a sampling
#: step to a whole multiple of the interval for the step to count as a
#: whole number of samples.
MASKED_MULTIPLE_TOLERANCE = 0.25

#: Least share of sampling steps within ``MASKED_MULTIPLE_TOLERANCE`` of a
#: whole multiple of the sampling interval for the boolean-masking check to
#: read the sampling as regular. Measured on simulated 30 Hz recordings
#: (``test_masking_gate_calibration``): a speed mask keeps 1.0, 0.97 with
#: timestamps jittered within a 10 ms window and 0.86 with Gaussian jitter of
#: 4 ms SD; a fragmented mask (kept runs of 1-4 frames between removed runs
#: of 10-40) keeps 1.0, though only 0.6 of its steps are within half of the
#: median step. Uniform or exponential irregular steps, and positions
#: sampled at spike times (0.17 of steps within half of the median step, as
#: in DANDI 001754), keep about 0.5, the share of random steps that land
#: near a multiple.
MASKED_REGULAR_SHARE = 0.75

#: Largest probability that random steps put at least as many steps near a
#: multiple, for the boolean-masking check to read the sampling as regular.
#: Without it, 41 in 40,000 short runs (11-54 steps) of uniform irregular
#: steps reached ``MASKED_REGULAR_SHARE`` by chance and warned.
MASKED_REGULAR_CHANCE = 1e-4


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


_STARTS_STOPS_HINT = (
    "; if its rows are starts and stops, pass np.column_stack([starts, stops])"
)


def _is_table(value: Any) -> bool:
    """Return whether ``value`` is a pandas DataFrame, without importing pandas."""
    return (
        not _is_pynapple_object(value)
        and hasattr(value, "columns")
        and hasattr(value, "iloc")
        and np.ndim(value) == 2
    )


def _is_time_intervals_table(value: Any) -> bool:
    """Return whether ``value`` is an NWB ``TimeIntervals``-like table."""
    columns = getattr(value, "colnames", None)
    return (
        not _is_table(value)
        and callable(getattr(value, "to_dataframe", None))
        and columns is not None
        and {"start_time", "stop_time"} <= set(columns)
    )


def _non_numeric_problem(raw: NDArray[Any], name: str) -> str | None:
    """The problem with window bounds that are not numbers, or None.

    Timedeltas, datetimes, Booleans and text would otherwise be coerced to
    numbers that are not seconds (a timedelta becomes nanoseconds).
    """
    kind = raw.dtype.kind
    if kind == "m":
        return (
            f"{name} holds timedelta values; convert to float seconds, for "
            f"example {name} / np.timedelta64(1, 's'), or "
            "series.dt.total_seconds() for a pandas column"
        )
    if kind == "M":
        return (
            f"{name} holds datetime values; convert to float seconds on the "
            f"clock of `times`, for example ({name} - t0) / np.timedelta64(1, 's')"
        )
    if kind == "b":
        return (
            f"{name} holds bool values, which are not times; convert to float "
            "seconds (a Boolean mask over samples is not a set of windows)"
        )
    if kind in "US" or (
        kind == "O" and any(isinstance(v, (str, bytes)) for v in raw.ravel())
    ):
        return (
            f"{name} holds text; convert to float seconds, for example "
            f"np.asarray({name}, dtype=float)"
        )
    return None


def _parse_intervals(
    value: Any, name: str
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Convert ``value`` to an unsorted ``(n, 2)`` array and list every problem.

    Returns ``(rows, problems)``. ``rows`` is ``None`` when numbers or shape
    are unusable; ``problems`` is empty when ``rows`` is valid.
    """
    kind = type(value).__name__
    if _is_pynapple_object(value) and kind != "IntervalSet":
        # A Ts/Tsd/TsdFrame has scalar .start/.end (its first and last
        # timestamps), which would read as one window over everything.
        return None, [
            f"{name} is a pynapple {kind}, a time series rather than a set of "
            f"intervals; pass the windows it was recorded in, "
            f"{name}={kind.lower()}.time_support"
        ]
    if _is_time_intervals_table(value):
        # An NWB TimeIntervals table: its columns are VectorData, which
        # would read as a 3-D array or fail to read as numbers.
        return None, [
            f"{name} is an NWB {kind} table; read it as Intervals with "
            "neurospatial.io.read_intervals(nwbfile, interval_name=...) or "
            f"Intervals.from_dataframe({name}.to_dataframe()), and pass those"
        ]
    table = _is_table(value)
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
            return rows, []
    elif table and {"start_time", "stop_time"} <= set(value.columns):
        # A table (a task log, an NWB table's DataFrame): its bounds by name,
        # whatever its other columns or their order.
        bounds = [np.asarray(value[column]) for column in ("start_time", "stop_time")]
        for bound in bounds:
            problem = _non_numeric_problem(bound, name)
            if problem is not None:
                return None, [problem]
        try:
            rows = np.column_stack(bounds).astype(np.float64)
        except (TypeError, ValueError) as exc:
            return None, [
                f"{name}['start_time'] and {name}['stop_time'] could not be "
                f"read as numbers ({exc})"
            ]
    elif (
        hasattr(value, "start_time")
        and hasattr(value, "stop_time")
        and np.ndim(value.start_time) == 0
        and np.ndim(value.stop_time) == 0
    ):
        # One interval record (Interval, Trial, RegionVisit): its own window,
        # or an empty selection when it has no duration, as in ``windows``.
        # Tables with start_time/stop_time columns are read as rows below.
        try:
            start, stop = float(value.start_time), float(value.stop_time)
        except (TypeError, ValueError) as exc:
            return None, [
                f"{name}.start_time and {name}.stop_time could not be read as "
                f"numbers ({exc})"
            ]
        if np.isfinite(start) and start == stop:
            return np.empty((0, 2)), []
        rows = np.array([[start, stop]])
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
            raw = np.asarray(value)
            problem = _non_numeric_problem(raw, name)
            rows = raw.astype(np.float64) if problem is None else raw
        except (TypeError, ValueError):
            return None, [
                f"{name} could not be read as numbers "
                f"(got {type(value).__name__}: {value!r:.80})"
            ]
        if problem is not None:
            return None, [problem]
        if rows.shape == (2,):
            rows = rows.reshape(1, 2)
        if rows.ndim != 2 or rows.shape[1] != 2:
            if table:
                hint = (
                    "; name the table's bound columns start_time and stop_time "
                    "(one row per window)"
                )
            else:
                hint = (
                    _STARTS_STOPS_HINT if rows.ndim == 2 and rows.shape[0] == 2 else ""
                )
            return None, [
                f"{name} has shape {rows.shape}; expected (2,) or (n, 2){hint}"
            ]
        if (
            rows.shape == (2, 2)
            and not table
            and not isinstance(value, (list, tuple))
            and rows[0, 0] < rows[1, 0] < rows[0, 1] < rows[1, 1]
        ):
            # Overlapping rows whose columns are two ordered windows: most
            # likely np.vstack([starts, stops]), which read as rows would
            # silently merge into one window.
            (a, b), (c, d) = rows.tolist()
            return None, [
                f"{name} is a (2, 2) array whose rows [{a:g}, {b:g}) and "
                f"[{c:g}, {d:g}) overlap, while its columns are two ordered "
                f"windows [{a:g}, {c:g}) and [{b:g}, {d:g}), so it is "
                f"ambiguous{_STARTS_STOPS_HINT}; if they are two overlapping "
                f"rows, pass the merged row [[{a:g}, {d:g}]]"
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
        return rows
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
        selection. One interval record (``start_time`` and ``stop_time``,
        such as a :class:`~neurospatial.behavior.Trial`) is its own window,
        or an empty selection without duration. A pandas DataFrame with
        ``start_time`` and ``stop_time`` columns is read by those columns,
        whatever its other columns; an NWB ``TimeIntervals`` table is
        refused with a pointer to ``read_intervals``. Bounds must be numbers
        in seconds: timedeltas, datetimes, Booleans and text are refused. An
        ``np.ndarray`` of shape ``(2, 2)`` whose rows overlap while its
        columns are two ordered windows (``np.vstack([starts, stops])``) is
        refused as ambiguous. An object exposing 1-D ``.start`` and ``.end``
        arrays (for example a pynapple ``IntervalSet``) is accepted by duck typing;
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
    if isinstance(value, SampleSupports):
        rows, _, problems = _resolve_epochs(value)
    else:
        rows, problems = _resolve_intervals(value, name)
    if problems:
        _raise_invalid(problems)
    return rows


def recorded_windows(value: Any, *, name: str) -> NDArray[np.float64] | None:
    """Windows as a result records them: resolved rows, possibly none.

    A result keeps the windows its analysis resolved, so an array or list
    without rows is the empty selection that analysis was given (an
    ``Intervals`` with no windows), not a probable bug as it would be in an
    analysis argument. Any other value is read by :func:`as_intervals`.

    Parameters
    ----------
    value : None, array-like of shape (n, 2), or any form :func:`as_intervals` reads
    name : str
        Field name used in error messages.

    Returns
    -------
    ndarray of float64, shape (n_windows, 2), or None
    """
    if isinstance(value, (np.ndarray, list)) and np.shape(value) in ((0,), (0, 2)):
        return np.empty((0, 2))
    return as_intervals(value, name=name)


@dataclass(frozen=True)
class SampleSupports:
    """``epochs`` as passed, with the time supports of the pynapple samples.

    :func:`~neurospatial._pynapple.reads_pynapple_samples` passes this as
    ``epochs`` when pynapple ``positions`` or ``headings`` carry a
    ``time_support`` (and ``compute_tuning_curve(s)`` for pynapple
    ``values``); :func:`resolve_time_windows` and :func:`as_intervals`
    resolve it.

    Attributes
    ----------
    epochs : Any
        The caller's ``epochs`` (None or any form :func:`as_intervals` reads).
    supports : dict of str to ndarray, each shape (n_intervals, 2)
        Each sample's ``time_support`` as half-open rows, by argument name.
    """

    epochs: Any
    supports: dict[str, NDArray[np.float64]]


class TimeWindows(NamedTuple):
    """Resolved ``epochs`` and ``spike_window`` of one analysis call."""

    #: Sorted, merged ``[start, stop)`` rows, or None for no restriction.
    epochs: NDArray[np.float64] | None
    #: Sorted, merged ``[start, stop)`` rows, or None for no restriction.
    spike_window: NDArray[np.float64] | None
    #: ``interval_ids`` of the rows of an ``Intervals`` given as ``epochs``
    #: that remain after the pynapple supports; None for any other ``epochs``.
    epoch_interval_ids: list[int] | None


def resolve_time_windows(epochs: Any, spike_window: Any) -> TimeWindows:
    """Resolve ``epochs`` and ``spike_window`` of one analysis call.

    ``epochs`` may carry pynapple sample supports (:class:`SampleSupports`):
    with no ``epochs`` given, the samples' shared support is the epochs (two
    samples with different supports are refused); explicit epochs are
    intersected with every support, so time outside a sample's support is
    never analyzed. Epochs that miss every support are kept unchanged, so the
    analysis reports them as it does epochs that miss plain samples (often
    another clock), not as a deliberately empty selection.

    Problems in both arguments are reported in a single ``ValueError``.

    Returns
    -------
    TimeWindows
        The resolved rows, and the ids of an :class:`~neurospatial.Intervals`
        given as ``epochs``: those of its rows that overlap every sample
        support (a row without duration: whose instant lies inside it), or
        all of them when there is no support or the epochs miss it.
    """
    epoch_rows, epoch_ids, epoch_problems = _resolve_epochs(epochs)
    window_rows, window_problems = _resolve_intervals(spike_window, "spike_window")
    if epoch_problems or window_problems:
        _raise_invalid(epoch_problems + window_problems)
    return TimeWindows(epoch_rows, window_rows, epoch_ids)


def _resolve_epochs(
    value: Any,
) -> tuple[NDArray[np.float64] | None, list[int] | None, list[str]]:
    """Resolved ``epochs`` rows, the ids of an ``Intervals``, and any problems."""
    from neurospatial._interval_table import Intervals

    supports: dict[str, NDArray[np.float64]] = {}
    if isinstance(value, SampleSupports):
        value, supports = value.epochs, value.supports
    if value is None and supports:
        # The samples' support is the epochs, read as epochs are.
        value, supports = _shared_support(supports), {}
    intervals = value if isinstance(value, Intervals) else None
    ids = None if intervals is None else intervals.interval_ids.tolist()
    rows, problems = _resolve_intervals(value, "epochs")
    if problems or rows is None or not supports:
        return rows, ids, problems
    kept, observed = rows, None
    for support in supports.values():
        if kept.shape[0] == 0:
            # No sample lies in these epochs: report them as given.
            return rows, ids, []
        merged = as_intervals(support, name="time_support")
        assert merged is not None
        kept = intersect_intervals(kept, merged)
        observed = merged if observed is None else intersect_intervals(observed, merged)
    if kept.shape[0] == 0:
        return rows, ids, []
    if intervals is not None and observed is not None:
        overlaps = _rows_overlap(intervals.start, intervals.stop, observed)
        ids = intervals.interval_ids[overlaps].tolist()
    return kept, ids, []


def _rows_overlap(
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    windows: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Whether each ``[starts[i], stops[i])`` overlaps a sorted, merged window.

    A row without duration overlaps the window that holds its instant.
    """
    first = np.searchsorted(windows[:, 1], starts, side="right")
    inside = first < windows.shape[0]
    # A row without duration reaches just past its instant.
    reach = np.maximum(stops, np.nextafter(starts, np.inf))
    starts_before = windows[np.minimum(first, windows.shape[0] - 1), 0] < reach
    overlaps: NDArray[np.bool_] = inside & starts_before
    return overlaps


def _shared_support(supports: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """The one time support all pynapple samples share, else a refusal."""
    (first, rows), *others = supports.items()
    for name, other in others:
        if not np.array_equal(rows, other):
            raise ValueError(
                _format_error(
                    f"{first} and {name} have different pynapple time supports, "
                    "so the epochs to analyze are ambiguous.",
                    why="with epochs=None each pynapple sample's time_support is "
                    "used as the epochs, and these disagree.",
                    fix="pass epochs= explicitly, e.g. "
                    f"epochs={first}.time_support.intersect({name}.time_support).",
                )
            )
    return rows


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
        Whether the bin check dropped an interval the other gates kept: at
        samples outside the environment, at NaN positions or at samples
        without a bin. Messages name the bin check only then, so ``times``
        on the wrong clock are not blamed on the positions.
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
            ``mask & keep``; the bin check is named when it drops an
            interval the other gates kept.
        """
        return replace(
            self,
            mask=self.mask & keep,
            start_bin_gate=self.start_bin_gate or bool(np.any(self.mask & ~keep)),
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

    def mask_for(bins: NDArray[np.intp] | None) -> NDArray[np.bool_]:
        return interval_valid_mask(
            times,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            start_bin=bins,
            require_end_bin=require_end_bin and bins is not None,
            epochs=epochs,
            spike_window=spike_window,
            break_before=break_before,
            sample_valid=sample_valid,
        )

    mask = mask_for(start_bin)
    # The bin check is named only when it drops an interval the other gates
    # keep, so a wrong clock is not blamed on the positions. The flag is read
    # only when every interval is excluded, so only then is the mask without
    # the bin check evaluated.
    bin_drops = start_bin is not None and (
        bool(mask.any()) or bool(np.any(mask_for(None)))
    )
    return IntervalGates(
        times,
        mask,
        max_gap,
        epochs=epochs,
        spike_window=spike_window,
        min_speed=min_speed,
        start_bin_gate=bin_drops,
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


def linear_allocated_occupancy(
    values: NDArray[np.float64],
    dt: NDArray[np.float64],
    edges: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Spread each valid interval's duration over the 1-D bins it crosses.

    Between two samples the value is taken to change at constant speed, so
    an interval's time is split over the bins in proportion to the part of
    the range between its two end values that lies in each bin. An interval
    whose ends are equal puts all its time in the bin of that value. Bins are
    half-open ``[left, right)``: time beyond the edges, at ``edges[-1]`` or
    at a non-finite end is not counted.

    Parameters
    ----------
    values : ndarray, shape (n_samples,)
        The variable at each sample (for example distance from an event).
    dt : ndarray, shape (n_samples - 1,)
        ``np.diff(times)``.
    edges : ndarray, shape (n_bins + 1,)
        Strictly increasing bin edges, in the units of ``values``.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Intervals to count.

    Returns
    -------
    ndarray, shape (n_bins,)
        Seconds in each bin.
    """
    values = np.asarray(values, dtype=np.float64)
    first = values[:-1][interval_mask]
    second = values[1:][interval_mask]
    dt = np.asarray(dt, dtype=np.float64)[interval_mask]
    low, high = np.minimum(first, second), np.maximum(first, second)
    # inf - inf is NaN (not moving, then left out like any non-finite value).
    with np.errstate(invalid="ignore"):
        covered = high - low
    n_bins = edges.size - 1
    moving = covered > 0
    # A moving interval overlaps bins j with edges[j] < high and edges[j+1] > low.
    first_bin = np.maximum(np.searchsorted(edges, low[moving], side="right") - 1, 0)
    last_bin = np.minimum(
        np.searchsorted(edges, high[moving], side="left") - 1, n_bins - 1
    )
    n_crossed = np.maximum(last_bin - first_bin + 1, 0)
    crossing = np.repeat(np.flatnonzero(moving), n_crossed)
    crossed_bin = np.repeat(first_bin, n_crossed) + ragged_arange(n_crossed)
    overlap = np.minimum(high[crossing], edges[crossed_bin + 1]) - np.maximum(
        low[crossing], edges[crossed_bin]
    )
    seconds = np.bincount(
        crossed_bin,
        weights=dt[crossing] * overlap / covered[crossing],
        minlength=n_bins,
    ).astype(np.float64)
    still_bin = value_frame_bins(low[~moving], edges, closed_last=False)
    inside = still_bin >= 0
    seconds += np.bincount(
        still_bin[inside], weights=dt[~moving][inside], minlength=n_bins
    )
    return seconds


def value_frame_bins(
    values: NDArray[np.float64], edges: NDArray[np.float64], *, closed_last: bool
) -> NDArray[np.intp]:
    """Bin index of each value, ``-1`` outside the edges or not finite.

    Parameters
    ----------
    values : ndarray, shape (n,)
    edges : ndarray, shape (n_bins + 1,)
        Strictly increasing.
    closed_last : bool
        Include ``edges[-1]`` in the last bin (as ``numpy.histogram`` does);
        otherwise every bin is half-open.

    Returns
    -------
    ndarray of intp, shape (n,)
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n_bins = len(edges) - 1
    with np.errstate(invalid="ignore"):
        upper = values <= edges[-1] if closed_last else values < edges[-1]
        inside = np.isfinite(values) & (values >= edges[0]) & upper
    bins = np.full(values.shape, -1, dtype=np.intp)
    found = np.searchsorted(edges, values[inside], side="right") - 1
    if closed_last:
        found[found == n_bins] = n_bins - 1
    bins[inside] = found
    return bins


def observed_interval_mask(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None = None,
    sample_valid: NDArray[np.bool_] | None = None,
    result: str = EMPTY_RESULT,
    raise_if_empty: bool = True,
) -> NDArray[np.bool_]:
    """Per-interval validity for position-only analyses (gap, epochs, breaks).

    ``epochs`` is the raw user argument; it is normalized here with
    ``as_intervals(epochs, name="epochs")``. ``break_before`` is already
    validated (``validate_break_before``). ``sample_valid`` marks missing
    samples, which are bridged like deleted rows (:func:`interval_valid_mask`).
    ``result`` names what an empty selection produces in the warning, and
    ``raise_if_empty`` whether an all-excluded mask raises, as in
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
    warn_excluded_intervals(gates, result=result, raise_if_empty=raise_if_empty)
    return gates.mask


def warn_excluded_intervals(
    gates: IntervalGates,
    *,
    samples: Literal["positions", "headings", "values"] = "positions",
    result: str = EMPTY_RESULT,
    suppress_hint: str = "",
    raise_if_empty: bool = True,
) -> None:
    """Warn with :func:`excluded_intervals_message`, if it has one.

    Parameters
    ----------
    gates, samples, result, raise_if_empty
        As for :func:`excluded_intervals_message`.
    suppress_hint : str, optional
        Closing sentence naming how to silence the warning.

    Raises
    ------
    InputError
        As for :func:`excluded_intervals_message`.
    """
    message = excluded_intervals_message(
        gates, samples=samples, result=result, raise_if_empty=raise_if_empty
    )
    if message is not None:
        warnings.warn(
            f"{message} {suppress_hint}".rstrip(),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def excluded_intervals_message(
    gates: IntervalGates,
    *,
    samples: Literal["positions", "headings", "values"] = "positions",
    result: str = EMPTY_RESULT,
    raise_if_empty: bool = True,
) -> str | None:
    """The warning for interval gates that leave no data or ``times`` that look wrong.

    Analyses check their final interval mask with this (through
    :func:`warn_excluded_intervals`, or joined with their own spike-drop
    message), so an empty or gap-dominated selection warns the same way
    everywhere. There is at most one message, and none for fewer than two
    samples:

    - When every interval is excluded, nothing usable is left, so this
      raises :class:`~neurospatial.InputError` (or, with
      ``raise_if_empty=False``, returns the message). The message names the
      active gates and how to fix each one, so an empty result is not
      mistaken for a measurement. When fewer than two samples are observed at
      all, it names the missing samples instead, since no gate is to blame.
      An empty selection (``epochs`` or ``spike_window`` with no windows,
      such as a condition with no trials) is deliberate, so its empty result
      is the answer: it only returns a message saying so.
    - Otherwise, when the kept intervals bridge runs of removed samples
      lasting from ``MASKED_RUN_MIN_DURATION`` to ``max_gap`` that add up to
      more than ``DISCARD_WARN_FRACTION`` of the analysed time, ``times``
      were boolean-masked (for example ``times[speed > 5]``) or tracking
      frames were dropped, and the time and spikes inside the removed runs
      still count. With
      ``max_gap=None`` every gap is bridged on purpose, so this check is
      skipped.
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
    samples : {"positions", "headings", "values"}, default="positions"
        What the start-bin gate bins, so the message names the right input:
        ``"values"`` for the variable of a tuning curve.
    result : str
        What an all-excluded mask leaves, e.g. ``"the rate map is empty"``.
    raise_if_empty : bool, default=True
        Raise when every interval is excluded. False only for callers whose
        result is still meaningful with no usable interval, such as a display
        that falls back to the exact samples.

    Returns
    -------
    str or None
        The message, without a closing hint on how to silence it.

    Raises
    ------
    InputError
        If every interval is excluded and ``raise_if_empty`` is True, unless
        the selection itself is empty.
    """
    if gates.mask.size == 0:
        return None
    times, max_gap = gates.times, gates.max_gap
    dt = np.diff(times)
    if not gates.mask.any():
        empty = empty_selection_names(gates.epochs, gates.spike_window)
        if empty:
            # A deliberate empty selection, such as a condition with no trials:
            # nothing is wrong with the clock, and the empty result is the
            # answer, so say what happened without raising.
            return empty_selection_message(empty, result)
        refusal = _all_excluded_message(gates, dt, samples=samples, result=result)
        if raise_if_empty:
            raise InputError(refusal)
        return refusal
    message = _masked_samples_message(times, gates.mask, max_gap)
    if message is not None:
        return message
    # Only intervals longer than max_gap can still warn.
    if max_gap is None or not np.any(dt > max_gap):
        return None
    return _kept_intervals_message(times, dt, max_gap)


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


def _sampling_interval(dt: NDArray[np.float64]) -> float:
    """The sampling interval of steps that may skip removed samples.

    The median step is a gap when most steps skip samples (a fragmented
    mask keeps runs of one or two frames), so the interval is read from the
    shortest steps: the median of the steps within half of the
    shortest-steps median, which are the steps of one sample.

    Parameters
    ----------
    dt : ndarray, shape (n_steps,)
        Positive sampling steps in seconds.

    Returns
    -------
    float
        The interval in seconds.

    Examples
    --------
    Most steps skip samples, but the interval is still the 0.1 s step:

    >>> import numpy as np
    >>> dt = np.r_[np.full(3, 0.1), np.full(7, 1.2)]
    >>> float(np.median(dt)), round(_sampling_interval(dt), 6)
    (1.2, 0.1)
    """
    low = float(np.quantile(dt, 0.1))
    # An element of dt, so the steps near it below include it.
    shortest = float(np.quantile(dt[dt <= 1.5 * low], 0.5, method="lower"))
    one_sample = dt[(dt > 0.5 * shortest) & (dt < 1.5 * shortest)]
    return float(np.median(one_sample))


def _masked_sample_fraction(
    times: NDArray[np.float64], kept: NDArray[np.bool_], max_gap: float
) -> float:
    """Fraction of the analysed time spent in long, bridged runs of removed samples.

    Boolean masking (``times[speed > 5]``) removes whole samples, so the
    steps left are whole multiples of the sampling interval
    (:func:`_sampling_interval`). Only kept steps no longer than ``max_gap``
    whose removed run lasts at least ``MASKED_RUN_MIN_DURATION`` count:
    shorter runs are dropped frames, longer steps are recording gaps, and
    steps outside the kept intervals (between epochs, for example) are not
    bridged. The sampling counts as regular only when at least
    ``MASKED_REGULAR_SHARE`` of the steps lie within
    ``MASKED_MULTIPLE_TOLERANCE`` of a whole multiple of the interval, and
    random steps would put that many there with probability at most
    ``MASKED_REGULAR_CHANCE``; irregular sampling, whose steps say nothing
    about removed samples, counts nothing.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times in seconds.
    kept : ndarray of bool, shape (n_samples - 1,)
        The intervals the analysis keeps.
    max_gap : float

    Returns
    -------
    float
        Skipped seconds divided by the kept intervals' duration, the time the
        analysis uses; 0 for fewer than 11 samples or nothing kept.

    Examples
    --------
    Three removed samples at 10 Hz (0.3 s) count; a single one does not:

    >>> import numpy as np
    >>> times = np.delete(np.arange(100) * 0.1, [10, 11, 12, 50])
    >>> round(_masked_sample_fraction(times, np.ones(times.size - 1, bool), 1.0), 4)
    0.0303

    Only the kept intervals count, in the numerator and the denominator:

    >>> kept = times[:-1] < 5.0
    >>> round(_masked_sample_fraction(times, kept, 1.0), 4)
    0.0588

    Irregular sampling counts nothing, even where two steps are three times
    the median step:

    >>> steps = [0.02, 0.05, 0.08, 0.11, 0.125, 0.125, 0.14, 0.17, 0.2, 0.375, 0.38]
    >>> irregular = np.cumsum(np.r_[0.0, steps])
    >>> _masked_sample_fraction(irregular, np.ones(len(steps), bool), 0.5)
    0.0
    """
    dt = np.diff(times)
    if dt.size < 10:
        return 0.0
    step = _sampling_interval(dt)
    if not np.isfinite(step) or step <= 0:
        return 0.0
    ratio = dt / step
    n_near = int(
        np.count_nonzero(
            (ratio >= 0.5)
            & (np.abs(ratio - np.round(ratio)) < MASKED_MULTIPLE_TOLERANCE)
        )
    )
    if n_near < MASKED_REGULAR_SHARE * dt.size:
        return 0.0
    # Random steps land near a multiple with probability 2 * tolerance, so a
    # few steps can reach the share by chance: also require at least n_near
    # of them to be improbable for random steps.
    from scipy.special import bdtrc

    chance = float(bdtrc(n_near - 1, dt.size, 2 * MASKED_MULTIPLE_TOLERANCE))
    if chance > MASKED_REGULAR_CHANCE:
        return 0.0
    skipped_time = (ratio - 1.0) * step
    skipped = (
        kept
        & (ratio >= 1.5)
        & (np.abs(ratio - np.round(ratio)) < 0.1)
        & (skipped_time >= MASKED_RUN_MIN_DURATION)
        & (dt <= max_gap)
    )
    analysed = float(dt[kept].sum())
    if analysed <= 0:
        return 0.0
    return float(skipped_time[skipped].sum() / analysed)


def _masked_samples_message(
    times: NDArray[np.float64], kept: NDArray[np.bool_], max_gap: float | None
) -> str | None:
    """The warning for masked samples or dropped tracking frames, or None."""
    if max_gap is None:
        return None
    fraction = _masked_sample_fraction(times, kept, max_gap)
    if fraction <= DISCARD_WARN_FRACTION:
        return None
    return _format_error(
        f"About {100 * fraction:.0f}% of the analysed time is in runs of "
        f"removed samples lasting {MASKED_RUN_MIN_DURATION:g} s to "
        f"max_gap={max_gap} s. They are either boolean-masked samples (for "
        "example times[speed > 5]) or dropped tracking frames (for example "
        "df[df.likelihood > 0.9] during occlusions).",
        why="runs shorter than max_gap are bridged as continuous occupancy, so "
        "the time and spikes inside the removed runs still count.",
        fix="for boolean-masked samples, pass the full arrays and select with "
        "the function's min_speed= or epochs= where it has them. For dropped "
        "tracking frames, bridging them is intended up to max_gap; to exclude "
        "them instead, pass a max_gap below the shortest removed run.",
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


def empty_selection_names(
    epochs: NDArray[np.float64] | None, spike_window: NDArray[np.float64] | None
) -> list[str]:
    """The resolved windows that are an empty selection (no rows)."""
    return [
        name
        for name, windows in (("epochs", epochs), ("spike_window", spike_window))
        if windows is not None and windows.shape[0] == 0
    ]


def empty_selection_message(names: list[str], consequence: str) -> str:
    """Say that the ``names`` windows select no time, so ``consequence``."""
    return (
        f"{' and '.join(names)} {'select' if len(names) > 1 else 'selects'} no "
        "time (an Intervals with no windows, such as a condition with no "
        f"trials), so {consequence}. Fix: select intervals with duration, or "
        "leave this condition out."
    )


def _all_excluded_message(
    gates: IntervalGates,
    dt: NDArray[np.float64],
    *,
    samples: Literal["positions", "headings", "values"],
    result: str,
) -> str:
    """The message for a selection that excludes every interval."""
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
    elif gates.start_bin_gate and samples == "values":
        named.append("samples at NaN values or outside the bin edges")
        fixes.append(
            "check that values are defined (not NaN) in the analyzed time and "
            "that the bin edges cover them"
        )
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
