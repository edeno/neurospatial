"""
Core dataclasses and validation helpers for the events module.

This module provides:
- Result dataclasses: PeriEventResult, PopulationPeriEventResult
- Validation helpers for windows, bins and event tables
- Visualization: plot_peri_event_histogram
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    UnitSelectionMixin,
    _copy_unit_table,
    align_unit_table,
    label_at,
    resolve_unit_ids,
)
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._time_bins import (
    _edge_rounding,
    _time_bin_rounding,
    _time_bins_in_windows,
)
from neurospatial._validation import timestamps_of

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def _validate_event_indices(
    event_indices: NDArray[np.intp] | None, n_events: int, n_events_dropped: int
) -> None:
    """Validate optional cohort provenance without retaining raw event times."""
    if event_indices is None:
        return
    indices = np.asarray(event_indices)
    if (
        indices.shape != (n_events,)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= n_events + n_events_dropped)
        or np.any(np.diff(indices.astype(np.intp)) <= 0)
    ):
        msg = (
            "event_indices must contain one increasing original input position per "
            "retained event, within n_events + n_events_dropped. "
            "Use None when the original event cohort is unknown."
        )
        raise ValueError(msg)


def empty_selection_names(
    epochs: NDArray[np.float64] | None, spike_window: NDArray[np.float64] | None
) -> list[str]:
    """The resolved windows that are an empty selection (no rows)."""
    return [
        name
        for name, windows in (("epochs", epochs), ("spike_window", spike_window))
        if windows is not None and windows.shape[0] == 0
    ]


def warn_empty_selection(names: list[str], *, n_events: int, result: str) -> None:
    """Warn that an empty selection leaves no event to average."""
    warnings.warn(
        f"{' and '.join(names)} {'select' if len(names) > 1 else 'selects'} no "
        "time (an Intervals with no windows, "
        f"such as a condition with no trials), so none of the {n_events} "
        f"events is analyzed and {result} is NaN with n_events == 0. Fix: "
        "select intervals with duration, or leave this condition out.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _is_real_number(value: Any) -> bool:
    """True for one real number; False for bools, arrays, strings and None."""
    return isinstance(value, Real) and not isinstance(value, bool | np.bool_)


def _finite_times(
    values: Any, name: str, *, non_empty: bool = False
) -> NDArray[np.float64]:
    """Return ``values`` as a 1-D float64 array of finite seconds.

    A pynapple ``Ts``/``Tsd``/``TsdFrame`` is read by its timestamps, not its
    values. ``non_empty`` also refuses an empty array, for analyses that need
    at least one event to align to.
    """
    values = timestamps_of(values, name=name)
    try:
        times = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        msg = (
            f"{name} must be numbers in seconds, got {type(values).__name__}.\n"
            "  WHY: Each entry is one time on the recording clock.\n"
            "  Fix: Pass a flat numeric array with one time per spike, sample or "
            "event."
        )
        raise ValueError(msg) from None
    if times.ndim != 1:
        msg = (
            f"{name} must be a 1-D array of times in seconds, got shape "
            f"{times.shape}.\n"
            "  WHY: Each entry is one time on the recording clock.\n"
            "  Fix: Pass a flat array with one time per spike, sample or event."
        )
        raise ValueError(msg)
    if non_empty and times.size == 0:
        msg = (
            f"{name} is empty.\n"
            "  WHY: The analysis needs at least one event to align to.\n"
            "  Fix: Provide at least one event time."
        )
        raise ValueError(msg)
    for is_bad, what in ((np.isnan, "NaN"), (np.isinf, "infinite (Inf)")):
        bad = np.flatnonzero(is_bad(times))
        if bad.size:
            msg = (
                f"{name} contains {bad.size} {what} value(s), first at index "
                f"{bad[0]}.\n"
                "  WHY: Times must be finite seconds.\n"
                f"  Fix: Remove them before calling, e.g. "
                f"{name}[np.isfinite({name})]."
            )
            raise ValueError(msg)
    return times


def _inner_bounds(
    starts: NDArray[np.float64], stops: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shrink computed ``[starts, stops)`` bounds by their float rounding.

    Bounds computed as ``time + offset`` are off by a few ulp, so a window
    that ends exactly at an epoch's stop (``0.1 + 0.2`` against ``0.3``)
    would read as leaving it. Test containment with these bounds instead.
    """
    slack = _edge_rounding(starts, stops)
    return starts + slack, stops - slack


ALIGNS = ("start", "stop")


def interval_events(
    event_times: Any, align: str, *, call: str
) -> tuple[
    Any,
    NDArray[np.int64] | None,
    tuple[NDArray[np.float64], NDArray[np.float64]] | None,
]:
    """Alignment times, row ids and window limits for events or intervals.

    Plain event times pass through unchanged (the caller validates them).
    For an :class:`~neurospatial.Intervals`, each row is aligned at its start
    or stop, and its window is limited on the other side: aligned at the
    start, nothing at or after the row's stop; aligned at the stop, nothing
    before the row's start.
    Rows with ``outcome == "censored"`` are refused at their stop.

    Returns
    -------
    events : array-like
        ``event_times`` as given, or each row's start or stop, shape
        ``(n_rows,)``.
    interval_ids : ndarray of int64, shape (n_rows,), or None
        The rows' ``interval_id``; None for plain times.
    limits : tuple of ndarray of float64, each shape (n_rows,), or None
        ``(-inf, stop)`` per row aligned at the start, ``(start, inf)``
        aligned at the stop; None for plain times.
    """
    from neurospatial._interval_table import Intervals

    if align not in ALIGNS:
        raise ValueError(
            _format_error(
                f"{call}: align={align!r} is not supported.",
                why="An interval is aligned at its start or at its stop.",
                fix="pass align='start' or align='stop'",
            )
        )
    if not isinstance(event_times, Intervals):
        if align != "start":
            raise ValueError(
                _format_error(
                    f"{call}: align={align!r} was given with plain event times.",
                    why="align chooses the start or stop of each interval; "
                    "plain times are already the alignment points.",
                    fix="pass an Intervals (trials, visits) as event_times, or "
                    "drop align",
                )
            )
        return event_times, None, None
    if len(event_times) == 0:
        raise ValueError(
            _format_error(
                f"{call}: event_times is an Intervals with no rows.",
                why="There is no interval to align.",
                fix="check the selection that produced it",
            )
        )
    if align == "stop" and "outcome" in event_times.fields:
        n_censored = int(np.count_nonzero(event_times.outcome.to_numpy() == "censored"))
        if n_censored:
            raise ValueError(
                _format_error(
                    f"{call}: align='stop' was given with {n_censored} censored "
                    "trial(s) (outcome == 'censored').",
                    why="A censored trial's stop is where observation ended, "
                    "not an arrival, so aligning at it would mix the end of "
                    "observation with arrivals.",
                    fix="leave the censored trials out: "
                    'trials[trials.outcome != "censored"]',
                )
            )
    starts = np.asarray(event_times.start, dtype=np.float64)
    stops = np.asarray(event_times.stop, dtype=np.float64)
    unbounded = np.full(starts.shape, np.inf)
    ids = np.asarray(event_times.interval_ids)
    if align == "start":
        return starts, ids, (-unbounded, stops)
    return stops, ids, (starts, unbounded)


def _snap_to(
    times: NDArray[np.float64], limit: NDArray[np.float64], scale: Any
) -> NDArray[np.float64]:
    """``times`` with those within float rounding of ``limit`` set to it.

    Bounds computed as ``event + offset`` are off by a few ulp of the larger
    of the two, ``scale`` (``2.0 - 1.9`` is ``0.09999999999999987``), so a
    window edge meant to be a trial's start or stop can land just inside or
    outside it. Snapping treats both sides alike.
    """
    near = np.abs(times - limit) <= _edge_rounding(times, np.asarray(scale))
    return np.where(near, limit, times)


def _offset_scale(
    event_times: NDArray[np.float64], window: tuple[float, float]
) -> NDArray[np.float64]:
    """Magnitude bounding ``event + offset`` for offsets inside ``window``."""
    return np.abs(event_times) + max(abs(window[0]), abs(window[1]))


def _cut_to_limits(
    times: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip absolute ``times`` to ``[lower, upper]``, snapping rounded bounds."""
    snapped = _snap_to(_snap_to(times, lower, scale), upper, scale)
    cut: NDArray[np.float64] = np.clip(snapped, lower, upper)
    return cut


def _window_bounds(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each event's absolute window, cut to its own ``limits`` when given.

    A bound within float rounding of its limit is the limit. A window
    entirely outside its limits becomes empty (``stop <= start``).
    """
    starts = event_times + window[0]
    stops = event_times + window[1]
    if limits is not None:
        scale = _offset_scale(event_times, window)
        starts = np.maximum(_snap_to(starts, limits[0], scale), limits[0])
        stops = np.minimum(_snap_to(stops, limits[1], scale), limits[1])
    return starts, stops


def _validate_window(
    window: tuple[float, float], *, context: str, warn_wide: bool = True
) -> tuple[float, float]:
    """Validate a window in seconds and return its bounds as floats.

    ``warn_wide`` warns when the window spans more than 60 s, which for a
    peri-event window usually means milliseconds were passed. History
    regressors (rewards in the last two minutes) turn it off.
    """
    try:
        if np.asarray(window).dtype.kind not in "biuf":
            msg = "window bounds must be real numbers"
            raise TypeError(msg)
        bounds = np.asarray(window, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            _format_error(
                f"{context}: window must contain two numeric bounds, got {window!r}.",
                why="Why: peri-event windows are expressed in seconds relative to each event.",
                fix="pass window=(-0.5, 1.0)",
            )
        ) from exc
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            _format_error(
                f"{context}: window must contain two finite bounds, got {window!r} with shape {bounds.shape}.",
                why="Why: peri-event windows need a finite start and stop in seconds.",
                fix="pass window=(-0.5, 1.0)",
            )
        )
    if window[0] >= window[1]:
        raise ValueError(
            _format_error(
                f"{context}: Window start ({window[0]}) must be before its stop ({window[1]}).",
                why="Why: a peri-event window is the half-open range [start, stop) "
                "relative to each event, so a window with stop <= start holds no time.",
                fix="pass window=(-0.5, 1.0), with start < stop, in seconds",
            )
        )
    if warn_wide and bounds[1] - bounds[0] > 60:
        warnings.warn(
            _format_error(
                f"{context}: window={window!r} spans {bounds[1] - bounds[0]:g} seconds, more than 60 seconds.",
                why="Why: an unusually wide peri-event window may be specified in milliseconds instead of seconds.",
                fix="use window=(-0.5, 1.0) if you meant milliseconds",
            ),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return float(bounds[0]), float(bounds[1])


def _make_bin_edges(
    window: tuple[float, float], bin_size: float
) -> NDArray[np.float64]:
    """Build PSTH bin edges that stay within the window.

    When ``window_duration`` is not an integer multiple of ``bin_size``,
    the trailing partial bin is dropped rather than appended. This keeps
    every edge (and therefore every center) inside ``[window[0],
    window[1]]``; otherwise the final bin would extend past ``window[1]``
    and silently absorb spikes that fall outside the requested window.

    Every bin is half-open: a spike at the final whole-bin edge or at the
    window stop is excluded. Edges are clamped to the window stop using
    timestamp-relative rounding rather than an absolute epsilon.

    When ``window[0]`` is a whole number of bins from 0 (up to float
    rounding), each edge is the grid value ``k * bin_size`` rather than
    ``window[0] + j * bin_size``, so the edge at the event is exactly 0
    (``-0.3 + 3 * 0.1`` is ``5.6e-17``, which would put a spike at the event
    in the bin before it). The outer edges are the window bounds.

    Parameters
    ----------
    window : tuple[float, float]
        Time window ``(start, end)`` relative to each event, in seconds.
    bin_size : float
        Width of each time bin in seconds.

    Returns
    -------
    bin_edges : NDArray[np.float64], shape (n_bins + 1,)
        Bin edges from ``window[0]`` up to ``window[0] + n_bins * bin_size``,
        which never exceeds ``window[1]``.
    """
    left, _ = _time_bins_in_windows(np.asarray([window], dtype=np.float64), bin_size)
    steps = np.arange(left.size + 1, dtype=np.float64)
    first = window[0] / bin_size
    on_grid = abs(first - round(first)) <= 4.0 * np.finfo(np.float64).eps * max(
        1.0, abs(first)
    )
    edges = (
        (round(first) + steps) * bin_size if on_grid else window[0] + bin_size * steps
    )
    clipped: NDArray[np.float64] = np.clip(edges, window[0], window[1])
    return clipped


def _psth_bins(
    window: tuple[float, float], bin_size: Any, *, context: str
) -> NDArray[np.float64]:
    """Validate ``bin_size`` for a valid window; return the bin edges.

    Raises
    ------
    ValueError
        If ``bin_size`` is not a finite positive number of seconds, or no
        complete bin fits in the window.
    """
    if not (_is_real_number(bin_size) and np.isfinite(bin_size) and bin_size > 0):
        raise ValueError(
            _format_error(
                f"{context}: bin_size must be positive and finite, got {bin_size!r}.",
                why="bin_size is the width of each peri-event time bin in seconds.",
                fix="pass bin_size=0.025 (25 ms)",
            )
        )
    edges = _make_bin_edges(window, float(bin_size))
    if edges.size < 2:
        raise ValueError(
            _format_error(
                f"{context}: bin_size={bin_size:g} s is longer than the "
                f"{window[1] - window[0]:g} s window, so no complete bin fits.",
                why="a trailing partial bin is dropped so every bin has the same "
                "width.",
                fix="choose a bin_size no larger than the window, e.g. bin_size=0.025",
            )
        )
    return edges


def _validated_psth_fields(
    result: Any, arrays: dict[str, tuple[int, ...]], *, context: str
) -> None:
    """Check and normalize the fields shared by the PSTH results in place.

    ``arrays`` maps each array field to its expected shape, with ``-1``
    standing for the number of bins implied by ``window`` and ``bin_size``.
    """
    window = _validate_window(result.window, context=context, warn_wide=False)
    n_bins = _psth_bins(window, result.bin_size, context=context).size - 1
    object.__setattr__(result, "window", window)
    object.__setattr__(result, "bin_size", float(result.bin_size))
    for name in ("n_events", "n_events_dropped"):
        value = getattr(result, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            msg = f"{context}: {name} must be an integer, got {value!r}."
            raise ValueError(msg)
        object.__setattr__(result, name, int(value))
    # n_events == 0 only for an empty selection, with all-NaN arrays.
    if result.n_events < 0 or result.n_events_dropped < 0:
        msg = (
            f"{context}: n_events and n_events_dropped must be at least 0, got "
            f"{result.n_events} and {result.n_events_dropped}."
        )
        raise ValueError(msg)
    for name, shape in arrays.items():
        array = np.asarray(getattr(result, name), dtype=np.float64)
        expected = tuple(n_bins if n == -1 else n for n in shape)
        if array.shape != expected:
            msg = (
                f"{context}: {name} has shape {array.shape}; window={window} "
                f"with bin_size={result.bin_size:g} s gives {n_bins} bins, "
                f"so it needs shape {expected}."
            )
            raise ValueError(msg)
        if np.any(array < 0):
            msg = (
                f"{context}: {name} has negative values; spike counts and their "
                "standard errors are never negative."
            )
            raise ValueError(msg)
        if result.n_events == 0 and not np.isnan(array).all():
            msg = (
                f"{context}: n_events is 0 (an empty selection), but {name} has "
                "values; with no event analyzed every value is NaN."
            )
            raise ValueError(msg)
        object.__setattr__(result, name, array)
    _validate_event_indices(
        result.event_indices, result.n_events, result.n_events_dropped
    )
    if result.align not in (None, *ALIGNS):
        msg = f"{context}: align must be None, 'start' or 'stop', got {result.align!r}."
        raise ValueError(msg)
    if (result.events_per_bin is None) != (result.align is None):
        msg = (
            f"{context}: events_per_bin and align go together: both are set for "
            "intervals and both None for plain event times."
        )
        raise ValueError(msg)
    if result.events_per_bin is not None:
        per_bin = np.asarray(result.events_per_bin)
        if (
            per_bin.shape != (n_bins,)
            or per_bin.dtype.kind not in "iu"
            or np.any(per_bin < 0)
            or np.any(per_bin > result.n_events)
        ):
            msg = (
                f"{context}: events_per_bin must hold one integer per bin "
                f"({n_bins}), each between 0 and n_events ({result.n_events})."
            )
            raise ValueError(msg)
        object.__setattr__(result, "events_per_bin", per_bin.astype(np.int64))


def _pre_event_bins(window: tuple[float, float], bin_size: float) -> NDArray[np.bool_]:
    """Bins that end at or before the event, up to float rounding of the edges."""
    edges = _make_bin_edges(window, bin_size)
    rounding = _time_bin_rounding(np.asarray([window]), bin_size)[0]
    pre: NDArray[np.bool_] = edges[1:] <= rounding
    return pre


def _peak_latencies(
    rates: NDArray[np.float64], bin_centers: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Peak rate and the time of the peak for each row of ``rates``.

    The first bin wins a tie. A row with no bin above zero (a unit that never
    fired in the window) has no peak, so its latency is NaN rather than the
    first bin's center; an all-NaN row has NaN rate and latency.

    Parameters
    ----------
    rates : ndarray, shape (n_rows, n_bins)
        Non-negative rates, NaN where undefined.
    bin_centers : ndarray, shape (n_bins,)
        Bin centers relative to the event (s).

    Returns
    -------
    peak_rate, peak_latency : ndarray, shape (n_rows,)
    """
    filled = np.where(np.isnan(rates), -np.inf, rates)
    peak_idx = np.argmax(filled, axis=1)
    rows = np.arange(rates.shape[0])
    latency = np.where(filled[rows, peak_idx] > 0, bin_centers[peak_idx], np.nan)
    return rates[rows, peak_idx], latency


def _response_metrics(
    rates: NDArray[np.float64], bin_centers: NDArray[np.float64], pre: NDArray[np.bool_]
) -> dict[str, NDArray[np.float64]]:
    """Peak rate, peak latency and baseline rate per row of ``rates`` (Hz)."""
    n_units = rates.shape[0]
    peak_rate, peak_latency = _peak_latencies(rates, bin_centers)
    # The mean of each unit's observed pre-event bins; NaN when it has none.
    pre_rates = rates[:, pre]
    n_observed = np.sum(~np.isnan(pre_rates), axis=1)
    baseline = np.full(n_units, np.nan)
    observed = n_observed > 0
    baseline[observed] = np.nanmean(pre_rates[observed], axis=1)
    return {
        "peak_rate": peak_rate,
        "peak_latency": peak_latency,
        "baseline_rate": baseline,
    }


class _PeriEventExports:
    """Tables shared by the single-unit and population PSTH results."""

    window: tuple[float, float]
    bin_size: float
    n_events: int
    n_events_dropped: int
    events_per_bin: NDArray[np.int64] | None
    align: str | None

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Bin centers relative to the event (s), shape ``(n_bins,)``."""
        return _make_bin_edges(self.window, self.bin_size)[:-1] + self.bin_size / 2

    def _table_attrs(self) -> dict[str, Any]:
        """Analysis settings stored in ``df.attrs`` of every table."""
        return {
            "window": self.window,
            "bin_size": self.bin_size,
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
            "align": self.align,
        }

    def _metrics_table(self, rates: NDArray[np.float64], unit_ids: Any) -> pd.DataFrame:
        import pandas as pd

        metrics = _response_metrics(
            rates, self.bin_centers, _pre_event_bins(self.window, self.bin_size)
        )
        df = pd.DataFrame(metrics, index=pd.Index(unit_ids, name="unit_id"))
        df.attrs = self._table_attrs()
        return df

    def _bins_table(
        self,
        unit_ids: NDArray[Any],
        histograms: NDArray[np.float64],
        sem: NDArray[np.float64],
    ) -> pd.DataFrame:
        import pandas as pd

        n_units, n_bins = histograms.shape
        per_bin = (
            np.full(n_bins, self.n_events)
            if self.events_per_bin is None
            else self.events_per_bin
        )
        df = pd.DataFrame(
            {
                "unit_id": np.repeat(unit_ids, n_bins),
                "bin_center": np.tile(self.bin_centers, n_units),
                "firing_rate": (histograms / self.bin_size).reshape(-1),
                "firing_rate_sem": (sem / self.bin_size).reshape(-1),
                "histogram": histograms.reshape(-1),
                "sem": sem.reshape(-1),
                "n_events": np.tile(per_bin, n_units),
            }
        )
        df.attrs = self._table_attrs()
        return df


# --- Result Dataclasses ---


@dataclass(frozen=True, repr=False, eq=False)
class PeriEventResult(_PeriEventExports, ResultMixin, ReadOnlyArraysMixin):
    """
    Result from peri-event histogram analysis.

    This dataclass stores the output of `peri_event_histogram()`, including
    the histogram bins, spike counts/rates, and statistics across events.

    Attributes
    ----------
    histogram : NDArray[np.float64], shape (n_bins,)
        Mean spike count per time bin across events.
    sem : NDArray[np.float64], shape (n_bins,)
        Standard error of the mean across events, in **count units**
        (matching ``histogram``); ``firing_rate_sem`` is the same in Hz.
        NaN with a single event (SEM is undefined).
    n_events : int
        Number of events used in analysis, at least 1.
    window : tuple[float, float]
        Time window (start, end) relative to event in seconds.
    bin_size : float
        Width of time bins (seconds). Together with ``window`` it fixes the
        bins: ``histogram`` and ``sem`` need one entry per bin.
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        :class:`PopulationPeriEventResult` (``rates[i].unit_id ==
        rates.unit_ids[i]``); ``None`` for a standalone single-unit
        computation unless supplied explicitly.
    n_events_dropped : int
        Number of events excluded because their windows leave the recording.
    event_indices : NDArray[np.intp], shape (n_events,), or None
        Original input positions of retained events, in input order. Use
        ``event_times[result.event_indices]`` to align a raster to the same
        cohort as this histogram. Read-only; None for manually constructed
        results whose original event cohort is unknown. Raw event times are
        not retained.
    events_per_bin : NDArray[np.int64], shape (n_bins,), or None
        For intervals (trials), the trials under way for the whole of each
        bin, which the bin's mean is over; None for plain event times, where
        every retained event covers every bin.
    align : {"start", "stop"} or None
        For intervals, the bound each trial was aligned at; None for plain
        event times.
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds), derived from ``window`` and
        ``bin_size``. Negative = before, positive = after.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate (Hz), ``histogram / bin_size``.
    firing_rate_sem : NDArray[np.float64], shape (n_bins,)
        Standard error of ``firing_rate`` across events (Hz),
        ``sem / bin_size``.

    Raises
    ------
    ValueError
        On construction, if ``window`` or ``bin_size`` is invalid, no complete
        bin fits, an array has the wrong shape or negative values, or the
        event counts are not non-negative integers with ``n_events >= 1``.

    See Also
    --------
    peri_event_histogram : Compute PSTH from spikes and events.
    plot_peri_event_histogram : Visualize PSTH results.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import peri_event_histogram
    >>> spikes = np.array([0.8, 1.0, 1.2, 2.8, 3.0, 3.2])
    >>> events = np.array([1.0, 3.0])
    >>> result = peri_event_histogram(spikes, events, window=(-0.5, 0.5), bin_size=0.1)
    >>> rate = result.firing_rate
    >>> rate.shape == result.bin_centers.shape
    True
    >>> peak_time = result.bin_centers[np.argmax(rate)]

    """

    histogram: NDArray[np.float64]
    sem: NDArray[np.float64]
    n_events: int
    window: tuple[float, float]
    bin_size: float
    unit_id: int | str | None = None
    n_events_dropped: int = 0
    event_indices: NDArray[np.intp] | None = None
    events_per_bin: NDArray[np.int64] | None = None
    align: str | None = None

    def __post_init__(self) -> None:
        _validated_psth_fields(
            self, {"histogram": (-1,), "sem": (-1,)}, context="PeriEventResult"
        )
        self._own_arrays()

    @property
    def firing_rate(self) -> NDArray[np.float64]:
        """Firing rate (Hz), shape ``(n_bins,)``: ``histogram / bin_size``."""
        rate: NDArray[np.float64] = self.histogram / self.bin_size
        return rate

    @property
    def firing_rate_sem(self) -> NDArray[np.float64]:
        """SEM of ``firing_rate`` across events (Hz), shape ``(n_bins,)``."""
        sem: NDArray[np.float64] = self.sem / self.bin_size
        return sem

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table of the PSTH: one row per time bin.

        One row per time bin, carrying a ``unit_id`` column (the unit this PSTH
        was computed for; ``None`` for a standalone single-unit computation).
        Use this for plotting / detailed inspection.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin_center`` (s, time relative to event),
            ``firing_rate`` (Hz), ``firing_rate_sem`` (Hz), ``histogram``
            (mean spike count per bin), ``sem`` (count units) and
            ``n_events`` (events averaged in the bin). ``df.attrs`` holds
            ``window``, ``bin_size``, ``n_events``, ``n_events_dropped`` and
            ``align``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PeriEventResult
        >>> result = PeriEventResult(
        ...     histogram=np.array([0.1, 0.4, 0.2]),
        ...     sem=np.array([0.01, 0.02, 0.01]),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ...     unit_id="u0",
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['unit_id', 'bin_center', 'firing_rate', 'firing_rate_sem', 'histogram', 'sem', 'n_events']
        >>> df["bin_center"].tolist()
        [-0.5, 0.0, 0.5]
        """
        return self._bins_table(
            np.array([self.unit_id], dtype=object),
            self.histogram[np.newaxis],
            self.sem[np.newaxis],
        )

    def summary_table(self) -> pd.DataFrame:
        """Scalar response metrics in one row, indexed by ``unit_id``.

        Same columns as :meth:`PopulationPeriEventResult.summary_table`.

        Returns
        -------
        pandas.DataFrame
            One row indexed by ``unit_id`` with ``peak_rate`` (Hz),
            ``peak_latency`` (s, time of the peak relative to the event; NaN
            when no bin is above zero, as for a unit with no spikes) and
            ``baseline_rate`` (Hz, mean rate over the bins that end at or
            before the event; NaN when the window has none). ``df.attrs``
            holds ``window``, ``bin_size``, ``n_events``,
            ``n_events_dropped`` and ``align``.
        """
        return self._metrics_table(
            self.firing_rate[np.newaxis], np.array([self.unit_id], dtype=object)
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of headline scalars for this PSTH.

        Returns
        -------
        dict
            Mapping with ``unit_id``, ``n_events`` and ``n_events_dropped``
            (int), ``peak_rate`` (Hz), ``peak_latency`` (s, time of the peak
            relative to the event; NaN when no bin is above zero, as for a
            unit with no spikes), and ``baseline_rate`` (Hz, mean rate over
            the bins that end at or before the event; a bin that straddles
            the event is not baseline).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PeriEventResult
        >>> result = PeriEventResult(
        ...     histogram=np.array([0.1, 0.4, 0.2]),
        ...     sem=np.array([0.01, 0.02, 0.01]),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> s = result.summary()
        >>> sorted(s)
        ['baseline_rate', 'n_events', 'n_events_dropped', 'peak_latency', 'peak_rate', 'unit_id']
        >>> round(s["peak_latency"], 3)
        0.0
        """
        metrics = _response_metrics(
            self.firing_rate[np.newaxis],
            self.bin_centers,
            _pre_event_bins(self.window, self.bin_size),
        )
        return {
            "unit_id": self.unit_id,
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
            **{name: float(values[0]) for name, values in metrics.items()},
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the PSTH, returning the axis for composition.

        Delegates to :func:`plot_peri_event_histogram`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure/axes is created.
        **kwargs
            Forwarded to :func:`plot_peri_event_histogram`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the PSTH was drawn on.
        """
        return plot_peri_event_histogram(self, ax=ax, **kwargs)


@_copy_unit_table
@dataclass(frozen=True, repr=False, eq=False)
class PopulationPeriEventResult(_PeriEventExports, ResultMixin, UnitSelectionMixin):
    """
    Result from population peri-event histogram analysis.

    This dataclass stores the output of `population_peri_event_histogram()`,
    including per-unit histograms and population-level statistics.

    Attributes
    ----------
    histograms : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit mean spike count per time bin across events.
    sem : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit standard error of the mean across events, in **count
        units** (matching ``histograms``). Divide by ``bin_size`` to
        convert to Hz.
    mean_sem : NDArray[np.float64], shape (n_bins,)
        Standard error across events of the population-mean histogram, in
        count units: the SEM of each event's mean count over units. This is
        the uncertainty of ``mean_histogram`` and the band ``plot`` draws.
        For the spread across units use
        ``histograms.std(axis=0, ddof=1) / np.sqrt(n_units)``.
    n_events : int
        Number of events used in analysis, at least 1.
    window : tuple[float, float]
        Time window (start, end) relative to event in seconds.
    bin_size : float
        Width of time bins (seconds).
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. Labelled tables align by unit identity; an unnamed default
        RangeIndex uses unit order. Mismatched or duplicate labels raise ValueError.
    n_events_dropped : int
        Number of events excluded because their windows leave the recording.
    event_indices : NDArray[np.intp], shape (n_events,), or None
        Original input positions of retained events, in input order. Use
        ``event_times[result.event_indices]`` to align a raster to the same
        cohort as this histogram. Read-only; None for manually constructed
        results whose original event cohort is unknown. Raw event times are
        not retained.
    events_per_bin : NDArray[np.int64], shape (n_bins,), or None
        For intervals (trials), the trials under way for the whole of each
        bin, which the bin's mean is over; None for plain event times, where
        every retained event covers every bin.
    align : {"start", "stop"} or None
        For intervals, the bound each trial was aligned at; None for plain
        event times.
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds), derived from ``window`` and
        ``bin_size``.
    n_units : int
        Number of units, ``len(histograms)``.
    mean_histogram : NDArray[np.float64], shape (n_bins,)
        Population average histogram, ``histograms.mean(axis=0)``.
    firing_rates : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit firing rates (Hz), ``histograms / bin_size``.
    mean_firing_rate : NDArray[np.float64], shape (n_bins,)
        Population-average firing rate (Hz), ``mean_histogram / bin_size``;
        the population-level analog of ``PeriEventResult.firing_rate``.

    Raises
    ------
    ValueError
        On construction, as for :class:`PeriEventResult`, and if
        ``unit_ids`` has the wrong length or repeats a value.

    See Also
    --------
    population_peri_event_histogram : Compute population PSTH.
    peri_event_histogram : Single-unit PSTH.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import population_peri_event_histogram
    >>> spikes = [np.array([0.8, 1.0, 3.2]), np.array([1.2, 2.8, 3.0])]
    >>> events = np.array([1.0, 3.0])
    >>> result = population_peri_event_histogram(
    ...     spikes, events, window=(-0.5, 0.5), bin_size=0.1
    ... )
    >>> rates = result.firing_rates
    >>> rates.shape[0]
    2
    >>> pop_rate = result.mean_firing_rate

    """

    histograms: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_sem: NDArray[np.float64]
    n_events: int
    window: tuple[float, float]
    bin_size: float
    unit_ids: NDArray[Any] | Sequence[Any] | None = None
    unit_table: pd.DataFrame | None = None
    n_events_dropped: int = 0
    event_indices: NDArray[np.intp] | None = None
    events_per_bin: NDArray[np.int64] | None = None
    align: str | None = None

    def __post_init__(self) -> None:
        context = "PopulationPeriEventResult"
        histograms = np.asarray(self.histograms)
        if histograms.ndim != 2 or histograms.shape[0] == 0:
            msg = (
                f"{context}: histograms must have shape (n_units, n_bins) with at "
                f"least one unit, got {histograms.shape}."
            )
            raise ValueError(msg)
        n_units = histograms.shape[0]
        _validated_psth_fields(
            self,
            {"histograms": (n_units, -1), "sem": (n_units, -1), "mean_sem": (-1,)},
            context=context,
        )
        object.__setattr__(self, "unit_ids", resolve_unit_ids(self.unit_ids, n_units))
        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(self.unit_table, self.unit_ids, context=context),
        )
        self._own_arrays()

    @property
    def n_units(self) -> int:
        """Number of units, ``len(histograms)``."""
        return len(self)

    @property
    def mean_histogram(self) -> NDArray[np.float64]:
        """Population average histogram (counts), shape ``(n_bins,)``."""
        mean: NDArray[np.float64] = self.histograms.mean(axis=0)
        return mean

    @property
    def firing_rates(self) -> NDArray[np.float64]:
        """Per-unit firing rates (Hz), shape ``(n_units, n_bins)``."""
        rates: NDArray[np.float64] = self.histograms / self.bin_size
        return rates

    @property
    def mean_firing_rate(self) -> NDArray[np.float64]:
        """Population-average firing rate (Hz), shape ``(n_bins,)``."""
        return self.mean_histogram / self.bin_size

    def __len__(self) -> int:
        """Return the number of units in the population.

        Returns
        -------
        int
            Number of units (first dimension of ``histograms``).
        """
        return int(self.histograms.shape[0])

    def __getitem__(self, idx: int) -> PeriEventResult:
        """Return the single-unit :class:`PeriEventResult` for unit ``idx``.

        Slices the per-unit ``histograms``/``sem`` rows, carries the shared
        bin metadata, and stamps ``unit_id`` from :attr:`unit_ids` so that
        ``self[i].unit_id == self.unit_ids[i]``.

        Parameters
        ----------
        idx : int
            Unit position (0-based), the row of :attr:`histograms`. Select by
            label with ``sel(unit_id=...)``.

        Returns
        -------
        PeriEventResult
            The PSTH for the selected unit.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_sem=np.zeros(3),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ...     unit_ids=["a", "b"],
        ... )
        >>> result[1].unit_id
        'b'
        """
        idx = self._unit_position(idx)
        unit_id = label_at(self.unit_ids, idx)
        return PeriEventResult(
            histogram=self.histograms[idx],
            sem=self.sem[idx],
            n_events=self.n_events,
            window=self.window,
            bin_size=self.bin_size,
            unit_id=unit_id,
            n_events_dropped=self.n_events_dropped,
            event_indices=self.event_indices,
            events_per_bin=self.events_per_bin,
            align=self.align,
        )

    def __iter__(self) -> Iterator[PeriEventResult]:
        """Iterate over single-unit :class:`PeriEventResult` objects.

        Yields
        ------
        PeriEventResult
            One per unit, in ``unit_ids`` order, each with ``unit_id`` stamped.
        """
        for idx in range(len(self)):
            yield self[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table of the population PSTH: one row per (unit, time-bin).

        One row per ``(unit, time-bin)``, carrying a ``unit_id`` column (the
        real per-unit identity labels, :attr:`unit_ids`). Use this for plotting
        / detailed inspection; for the per-unit scalar summary use
        :meth:`summary_table`.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin_center`` (s), ``firing_rate`` (Hz),
            ``firing_rate_sem`` (Hz), ``histogram`` (mean spike count per
            bin), ``sem`` (count units) and ``n_events`` (events averaged in
            the bin). ``df.attrs`` holds ``window``, ``bin_size``,
            ``n_events``, ``n_events_dropped`` and ``align``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_sem=np.zeros(3),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['unit_id', 'bin_center', 'firing_rate', 'firing_rate_sem', 'histogram', 'sem', 'n_events']
        >>> len(df) == 2 * 3
        True
        """
        return self._bins_table(np.asarray(self.unit_ids), self.histograms, self.sem)

    def summary_table(self) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        One row per unit with scalar response metrics. For the dense per-bin
        frame (one row per ``(unit, time-bin)``) use :meth:`to_dataframe`.

        Returns
        -------
        pandas.DataFrame
            One row per unit, indexed by ``unit_id``, with columns
            ``peak_rate`` (Hz), ``peak_latency`` (s, time of the per-unit peak
            relative to the event; NaN for a unit with no bin above zero), and
            ``baseline_rate`` (Hz, mean rate over
            the bins that end at or before the event). ``df.attrs`` holds
            ``window``, ``bin_size``, ``n_events``, ``n_events_dropped`` and
            ``align``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_sem=np.zeros(3),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> df = result.summary_table()
        >>> list(df.columns)
        ['peak_rate', 'peak_latency', 'baseline_rate']
        >>> df.index.name
        'unit_id'
        >>> len(df)
        2
        """
        return self._metrics_table(self.firing_rates, np.asarray(self.unit_ids))

    def summary(self) -> dict[str, Any]:
        """Flat dict of population headline scalars.

        Returns
        -------
        dict
            Mapping with ``n_units``, ``n_events`` and ``n_events_dropped``
            (int), ``mean_peak_rate`` (Hz, population-average peak firing
            rate), and ``population_peak_latency`` (s, time of the mean
            histogram peak; NaN when no unit fired in the window).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_sem=np.zeros(3),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> sorted(result.summary())
        ['mean_peak_rate', 'n_events', 'n_events_dropped', 'n_units', 'population_peak_latency']
        """
        _, latency = _peak_latencies(
            self.mean_firing_rate[np.newaxis], self.bin_centers
        )
        # Units without a defined rate (no event analyzed) have no peak.
        peaks, _ = _peak_latencies(self.firing_rates, self.bin_centers)
        defined = peaks[~np.isnan(peaks)]
        return {
            "n_units": self.n_units,
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
            "mean_peak_rate": float(defined.mean()) if defined.size else np.nan,
            "population_peak_latency": float(latency[0]),
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the population-average PSTH, returning the axis.

        The band is ``mean_sem``, the standard error across events of the
        population-mean histogram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure/axes is created.
        **kwargs
            Forwarded to :func:`plot_peri_event_histogram`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the population-average PSTH was drawn on.
        """
        mean_result = PeriEventResult(
            histogram=self.mean_histogram,
            sem=self.mean_sem,
            n_events=self.n_events,
            window=self.window,
            bin_size=self.bin_size,
            n_events_dropped=self.n_events_dropped,
            event_indices=self.event_indices,
            events_per_bin=self.events_per_bin,
            align=self.align,
        )
        return plot_peri_event_histogram(mean_result, ax=ax, **kwargs)


# --- Validation Helpers ---


def _validate_events_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    timestamp_column: str = "timestamp",
    context: str = "",
) -> None:
    """
    Validate events DataFrame structure with diagnostic error messages.

    This function checks that the input is a pandas DataFrame with the
    required columns and proper data types. It provides detailed error
    messages following the WHAT/WHY/HOW pattern used throughout neurospatial.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame to validate.
    required_columns : list[str], optional
        Additional required columns beyond timestamp.
    timestamp_column : str, default="timestamp"
        Name of timestamp column to check.
    context : str, optional
        Additional context for error messages (e.g., function name).

    Returns
    -------
    None
        Returns normally when the DataFrame is valid; otherwise raises.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        If required columns are missing or timestamp is not numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0]})
    >>> _validate_events_dataframe(df)  # No error

    >>> df_bad = pd.DataFrame({"time": [1.0, 2.0]})
    >>> _validate_events_dataframe(df_bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Missing required columns: ['timestamp']...
    """
    import pandas as pd

    # Check type
    if not isinstance(df, pd.DataFrame):
        msg = (
            f"Expected pd.DataFrame, got {type(df).__name__}.\n"
            "  WHY: Event tables are pandas DataFrames with one row per event.\n"
            "  Fix: Convert using pd.DataFrame({'timestamp': times})"
        )
        raise TypeError(msg)

    # Check required columns
    all_required = [timestamp_column] + (required_columns or [])
    missing = [col for col in all_required if col not in df.columns]
    if missing:
        context_str = f" for {context}" if context else ""
        msg = (
            f"Missing required columns: {missing}.\n"
            f"  WHY: These columns are needed{context_str}.\n"
            f"  Fix: Add missing columns to DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )
        raise ValueError(msg)

    # Check timestamp is numeric
    if not np.issubdtype(df[timestamp_column].dtype, np.number):
        msg = (
            f"Timestamp column '{timestamp_column}' contains non-numeric values.\n"
            "  WHY: Timestamps must be numeric (seconds from session start).\n"
            f"  Fix: Convert timestamps: df['{timestamp_column}'] = "
            f"df['{timestamp_column}'].astype(float)"
        )
        raise ValueError(msg)


def _event_times(
    df: pd.DataFrame, column: str, *, name: str, call: str
) -> NDArray[np.float64]:
    """Validate an event table and return its ``column`` as finite seconds.

    Parameters
    ----------
    df : pd.DataFrame
        Event or interval table.
    column : str
        Numeric time column in seconds.
    name, call : str
        Argument and function names for the error messages.

    Returns
    -------
    NDArray[np.float64], shape (n_rows,)

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame.
    ValueError
        If ``column`` is missing or not numeric, or holds NaN or inf.
    """
    _validate_events_dataframe(df, timestamp_column=column, context=f"{call} {name}")
    times: NDArray[np.float64] = df[column].to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(times))
    if bad.size:
        msg = (
            f"{call}: {name}[{column!r}] has {bad.size} NaN or inf time(s), "
            f"first at row {int(bad[0])}.\n"
            "  WHY: a row without a time cannot be placed on the clock, and "
            "dropping or keeping it silently would change the result.\n"
            f"  Fix: drop or repair those rows first, e.g. "
            f"{name}.dropna(subset=[{column!r}])."
        )
        raise ValueError(msg)
    return times


# --- Visualization ---


def plot_peri_event_histogram(
    result: PeriEventResult,
    *,
    ax: Axes | None = None,
    show_sem: bool = True,
    color: str = "C0",
    as_rate: bool = True,
    title: str | None = None,
    xlabel: str = "Time from event (s)",
    ylabel: str | None = None,
) -> Axes:
    """
    Plot peri-event time histogram (PSTH).

    Creates a standard PSTH visualization with the mean response and
    optional shaded standard error region.

    Parameters
    ----------
    result : PeriEventResult
        Result from `peri_event_histogram()`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_sem : bool, default=True
        Show shaded SEM region around mean.
    color : str, default="C0"
        Line color (matplotlib color specification).
    as_rate : bool, default=True
        Plot as firing rate (Hz) rather than spike counts.
    title : str, optional
        Plot title. Default: "PSTH (n={n_events} events)".
    xlabel : str, default="Time from event (s)"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Default: "Firing rate (Hz)" or "Spike count".

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    See Also
    --------
    peri_event_histogram : Compute PSTH from spikes and events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import peri_event_histogram
    >>> spikes = np.array([0.8, 1.0, 1.2, 2.8, 3.0, 3.2])
    >>> result = peri_event_histogram(
    ...     spikes, np.array([1.0, 3.0]), window=(-0.5, 0.5), bin_size=0.1
    ... )
    >>> ax = plot_peri_event_histogram(result, title="Reward response")
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # Get data to plot
    x = result.bin_centers
    if as_rate:
        y = result.firing_rate
        sem = result.firing_rate_sem
    else:
        y = result.histogram
        sem = result.sem

    # Plot mean
    ax.plot(x, y, color=color, linewidth=1.5)

    # Plot SEM shading
    if show_sem:
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.3)

    # Labels
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = "Firing rate (Hz)" if as_rate else "Spike count"
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"PSTH (n={result.n_events} events)"
    ax.set_title(title)

    # Mark event time
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    return ax
