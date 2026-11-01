"""
Peri-event analysis functions for neurospatial.

This module provides functions for aligning neural data to events:
- align_spikes_to_events: Get per-trial spike times relative to events
- peri_event_histogram: Compute peri-event time histogram (PSTH)
- population_peri_event_histogram: Compute PSTH for multiple units
- align_events: Align events to reference events
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import (
    intervals_contain,
    ragged_arange,
    resolve_time_windows,
)
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import resolve_unit_ids
from neurospatial._spikes import _as_spike_trains_with_ids, spike_unit_table
from neurospatial._time_bins import _time_bin_rounding
from neurospatial.events._core import (
    PeriEventResult,
    PopulationPeriEventResult,
    _event_times,
    _finite_times,
    _inner_bounds,
    _psth_bins,
    _validate_window,
    _window_bounds,
    empty_selection_names,
    interval_events,
    warn_empty_selection,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._interval_table import Intervals
    from neurospatial._typing import SpikeTrainsLike


@reads_pynapple_samples
def align_spikes_to_events(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64] | Intervals[Any],
    window: tuple[float, float],
    *,
    align: str = "start",
) -> list[NDArray[np.float64]]:
    """
    Get spike times relative to each event for raster plots.

    This is a low-level function that aligns spike times to multiple events,
    returning relative spike times for each event. It's useful for creating
    raster plots or for custom peri-event analyses.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Spike times in seconds. Does not need to be sorted.
    event_times : NDArray[np.float64], shape (n_events,), pynapple Ts/Tsd, or Intervals
        Event times (e.g., stimulus onset) in seconds. A pynapple ``Ts`` or ``Tsd`` is read by its timestamps.
        An :class:`~neurospatial.Intervals` (trials) gives one event per row,
        aligned by ``align``; see Notes.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, half-open
        ``[start, stop)``: a spike exactly at ``event + stop`` is left out.
        For example, ``(-0.5, 1.0)`` captures 500 ms before to 1 s after each
        event.
    align : {"start", "stop"}, default="start"
        For an ``Intervals``, align each row at its start or at its stop.
        Plain event times are the alignment points; only the default is
        allowed with them.

    Returns
    -------
    list[NDArray[np.float64]]
        List of arrays, one per event (one per row of an ``Intervals``, in
        row order, matching ``interval_ids``), containing spike times relative
        to that event (spike_time - event_time). Arrays are sorted by relative
        time. Empty arrays are returned for events with no spikes in window.

    Raises
    ------
    ValueError
        If window start >= stop, spike/event times contain NaN or Inf, or
        ``align`` is invalid or given with plain event times.

    Notes
    -----
    For intervals, each row's window ends where the row does on the side away
    from the alignment point: aligned at the start, no spike at or after the
    trial's stop (it belongs to the next trial); aligned at the stop, no spike
    before the trial's start.

    See Also
    --------
    peri_event_histogram : Compute binned PSTH from aligned spikes.
    align_events : Align events (not spikes) to reference events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events.alignment import align_spikes_to_events

    Align spikes to stimulus events for raster plot:

    >>> spike_times = np.array([9.5, 10.2, 10.8, 19.3, 20.1, 20.5])
    >>> stim_times = np.array([10.0, 20.0])
    >>> aligned = align_spikes_to_events(spike_times, stim_times, window=(-1.0, 1.0))
    >>> len(aligned)
    2
    >>> aligned[0]  # Spikes around first stimulus
    array([-0.5,  0.2,  0.8])
    >>> aligned[1]  # Spikes around second stimulus
    array([-0.7,  0.1,  0.5])

    Create raster plot data:

    >>> for trial_idx, trial_spikes in enumerate(aligned):
    ...     # Plot each trial's spikes at y=trial_idx
    ...     pass  # plt.scatter(trial_spikes, [trial_idx]*len(trial_spikes))
    """
    call = "align_spikes_to_events"
    _validate_window(window, context=call)
    spike_times = _finite_times(spike_times, "spike_times")
    event_times, _, limits = interval_events(event_times, align, call=call)
    event_times = _finite_times(event_times, "event_times")

    spikes = _sorted_times(spike_times)
    # Spikes in each half-open window [event + start, event + stop), cut to
    # each interval's limits.
    starts, stops = _window_bounds(event_times, window, limits)
    first = np.searchsorted(spikes, starts, side="left")
    stop = np.maximum(first, np.searchsorted(spikes, stops, side="left"))
    return [
        spikes[i:j] - event
        for i, j, event in zip(first, stop, event_times, strict=True)
    ]


def _sorted_times(times: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``times`` in increasing order, sorting only when needed."""
    return times if bool(np.all(times[1:] >= times[:-1])) else np.sort(times)


def _keep_observed_events(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    limits: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Keep events with observed windows and return their original input indices.

    An empty selection (``epochs`` or ``spike_window`` without rows) keeps no
    event; the caller warns. Non-empty windows that keep no event raise:
    that usually means another clock. With ``limits`` (intervals), each
    window is first cut to its row's limits.
    """
    if empty_selection_names(epochs, spike_window):
        return event_times[:0], np.empty(0, dtype=np.intp)
    keep = np.ones(event_times.shape, dtype=bool)
    starts, stops = _inner_bounds(*_window_bounds(event_times, window, limits))
    for windows in (epochs, spike_window):
        if windows is not None:
            keep &= intervals_contain(windows, starts, stops)
    if not keep.any():
        msg = (
            f"All {event_times.size} events were dropped from the peri-event "
            f"histogram: no event's window [event{window[0]:+g} s, "
            f"event{window[1]:+g} s) lies entirely inside epochs ∩ spike_window. "
            f"\nWhy: a window that reaches unrecorded time would be averaged as "
            f"if no spikes occurred there.\nFix: check that event_times, epochs "
            f"and spike_window share one clock (seconds), or narrow `window`."
        )
        raise ValueError(msg)
    return event_times[keep], np.flatnonzero(keep)


def _peri_event_bin_edges(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    bin_edges: NDArray[np.float64],
    bin_size: float,
) -> NDArray[np.float64]:
    """Absolute bin edges shared by histogram counting and rasters.

    Returns
    -------
    NDArray[np.float64], shape (n_events, n_bins + 1)
        Each event's bin edges on the recording clock, clamped to its window
        stop, so every row is non-decreasing.
    """
    windows = np.c_[event_times + window[0], event_times + window[1]]
    _time_bin_rounding(
        windows,
        bin_size,
        context="peri-event bins",
        width_name="bin_size",
        clock_fix=(
            "subtract a time origin from spike_times and event_times first "
            "(keep the relative window unchanged), or use a larger bin_size."
        ),
    )
    edges: NDArray[np.float64] = np.minimum(
        event_times[:, None] + bin_edges, windows[:, 1:]
    )
    return edges


class _PeriEventCounts(NamedTuple):
    """What :func:`_peri_event_histograms` returns."""

    histograms: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_sem: NDArray[np.float64]
    event_indices: NDArray[np.intp]
    n_dropped: int
    events_per_bin: NDArray[np.int64] | None


def _peri_event_histograms(
    trains: Sequence[NDArray[np.float64]],
    event_times: Any,
    window: tuple[float, float],
    bin_size: float,
    epochs: Any,
    spike_window: Any,
    *,
    context: str,
    align: str = "start",
) -> _PeriEventCounts:
    """Per-unit PSTHs of finite 1-D spike trains, one unit in memory at a time.

    ``window`` is already validated, so its wide-window warning points at the
    public caller. For an ``Intervals``, a bin counts for a row only when it
    lies inside the row's cut window (:func:`interval_events`), so each bin
    is averaged over the rows still under way.

    Returns
    -------
    histograms, sem : NDArray[np.float64], shape (n_units, n_bins)
        Each unit's mean count per bin across retained events and its SEM.
    mean_sem : NDArray[np.float64], shape (n_bins,)
        SEM across events of each event's mean over units.
    event_indices : NDArray[np.intp], shape (n_events,)
        Input positions of the retained events.
    n_dropped : int
        Events whose windows leave ``epochs`` or ``spike_window``, or, for
        intervals, hold no whole bin.
    events_per_bin : NDArray[np.int64], shape (n_bins,), or None
        Retained rows covering each bin, for intervals; None for times.
    """
    bin_edges = _psth_bins(window, bin_size, context=context)
    event_times, _, limits = interval_events(event_times, align, call=context)
    event_times = _finite_times(event_times, "event_times", non_empty=True)
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    n_input_events = event_times.size
    event_times, event_indices = _keep_observed_events(
        event_times, window, resolved_epochs, resolved_spike_window, limits
    )
    n_bins = bin_edges.size - 1
    # Count on the absolute clock, not on relative times: subtraction can move
    # an exact boundary spike across a bin edge.
    edges = _peri_event_bin_edges(event_times, window, bin_edges, bin_size)
    covered = np.ones((event_times.size, n_bins), dtype=bool)
    if limits is not None:
        lefts, rights = _inner_bounds(edges[:, :-1], edges[:, 1:])
        covered = (lefts >= limits[0][event_indices, None]) & (
            rights <= limits[1][event_indices, None]
        )
        has_bin = covered.any(axis=1)
        event_times, event_indices = event_times[has_bin], event_indices[has_bin]
        edges, covered = edges[has_bin], covered[has_bin]
    n_events = event_times.size
    events_per_bin = None if limits is None else covered.sum(axis=0)
    if n_events == 0:
        names = empty_selection_names(resolved_epochs, resolved_spike_window)
        if not names:
            msg = (
                f"{context}: no interval holds a whole bin of the window "
                f"[{window[0]:g}, {window[1]:g}) s aligned at its {align}.\n"
                "Why: each trial's window is cut where the trial ends, on the "
                "side away from the alignment point.\n"
                "Fix: use a window on the trial's side of the alignment point "
                "(window=(0, 2) with align='start'), or a smaller bin_size."
            )
            raise ValueError(msg)
        warn_empty_selection(names, n_events=n_input_events, result="the histogram")
        missing = np.full((len(trains), n_bins), np.nan, dtype=np.float64)
        return _PeriEventCounts(
            missing,
            missing.copy(),
            np.full(n_bins, np.nan, dtype=np.float64),
            event_indices,
            n_input_events,
            events_per_bin,
        )

    histograms = np.empty((len(trains), n_bins), dtype=np.float64)
    sem = np.empty((len(trains), n_bins), dtype=np.float64)
    count_sum = np.zeros((n_events, n_bins), dtype=np.float64)
    for unit, train in enumerate(trains):
        counts = np.diff(np.searchsorted(_sorted_times(train), edges), axis=1).astype(
            np.float64
        )
        histograms[unit], sem[unit] = _mean_and_sem(counts, covered)
        count_sum += counts

    # The population mean's SEM is across events of each event's mean over
    # units, the same estimand as each unit's SEM.
    _, mean_sem = _mean_and_sem(count_sum / len(trains), covered)
    return _PeriEventCounts(
        histograms,
        sem,
        mean_sem,
        event_indices,
        n_input_events - n_events,
        events_per_bin,
    )


def _mean_and_sem(
    counts: NDArray[np.float64], covered: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean across events of each bin's counts and its SEM, covered cells only.

    Parameters
    ----------
    counts : ndarray of float64, shape (n_events, n_bins)
    covered : ndarray of bool, shape (n_events, n_bins)
        Cells that count; at least one per event.

    Returns
    -------
    mean, sem : ndarray of float64, shape (n_bins,)
        NaN where no event covers the bin; the SEM is NaN with fewer than two.
    """
    n = covered.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(covered, counts, 0.0).sum(axis=0) / n
        squares = np.where(covered, (counts - mean) ** 2, 0.0).sum(axis=0)
        sem = np.sqrt(squares / (n - 1)) / np.sqrt(n)
    mean[n == 0] = np.nan
    sem[n < 2] = np.nan
    return mean, sem


@reads_pynapple_samples
def peri_event_histogram(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64] | Intervals[Any],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
    align: str = "start",
    epochs: Any = None,
    spike_window: Any = None,
) -> PeriEventResult:
    """
    Compute peri-event time histogram (PSTH).

    This function computes the average spike count per time bin around
    events, commonly called a peri-stimulus time histogram (PSTH) or
    peri-event time histogram (PETH).

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Spike times in seconds.
    event_times : NDArray[np.float64], shape (n_events,), pynapple Ts/Tsd, or Intervals
        Event times (e.g., stimulus onset) in seconds. A pynapple ``Ts`` or ``Tsd`` is read by its timestamps.
        An :class:`~neurospatial.Intervals` (trials) gives one event per row,
        aligned by ``align``; see Notes.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, half-open
        ``[start, stop)``: a spike exactly at ``event + stop`` is left out.
        For example, ``(-0.5, 1.0)`` captures 500 ms before to 1 s after each
        event.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).
    align : {"start", "stop"}, default="start"
        For an ``Intervals``, align each row at its start or at its stop.
        Plain event times are the alignment points; only the default is
        allowed with them.
    epochs : (start, stop), array-like of shape (n, 2), Intervals, IntervalSet, \
or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``event_times``). An event is kept only if its entire
        analysis window lies inside one recording window. ``None`` (default)
        means unrestricted. An :class:`~neurospatial.Intervals` with no windows
        (a condition with no trials) is an empty selection: the result is NaN
        with ``n_events == 0``, and a warning says so.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording, on the same clock as
        ``event_times``. Events are kept only if their entire analysis window
        lies inside both ``epochs`` and ``spike_window``. ``None`` adds no
        recording constraint; this spike-only analysis has no position stream
        from which to assume coverage. Pass ``spike_window`` (or ``epochs``)
        when the recording has gaps or edges inside your event windows.

    Returns
    -------
    PeriEventResult
        Dataclass containing:
        - bin_centers: Time relative to event (seconds)
        - histogram: Mean spike count per bin across events
        - sem: Standard error of the mean across events
        - n_events: Number of retained events
        - n_events_dropped: Events whose windows leave the recording
        - event_indices: Positions of retained events in the original input;
          use ``event_times[result.event_indices]`` for a matching raster cohort
        - window: The input window
        - bin_size: The input bin_size

    Raises
    ------
    ValueError
        If window is invalid or inverted, bin_size is not a finite positive
        number or leaves no complete bin in the window, spike_times or
        event_times is not 1-D or holds NaN or Inf, event_times is empty,
        time windows are invalid, or windows with duration keep no event
        (usually event times on another clock).

    Warns
    -----
    UserWarning
        When the window spans more than 60 s (it may be in milliseconds), or
        when ``epochs`` or ``spike_window`` is an empty selection.

    Notes
    -----
    With a single retained event the SEM is undefined and is NaN.

    For intervals, each row's window ends where the row does on the side away
    from the alignment point: aligned at the start, no bin at or after the
    trial's stop counts (it would be the next trial); aligned at the stop, no
    bin before the trial's start. The other side counts as for any event, so
    the time at the well before a departure or after an arrival stays in. A
    bin counts for a trial only when it lies wholly inside the trial's
    window, so each bin is the mean over the trials under way for all of it
    (``events_per_bin``), NaN where there are none. ``n_events_dropped`` also
    counts trials holding no whole bin. To keep only events whose whole
    window lies inside one trial, pass plain times with ``epochs=trials``.

    Every bin is half-open ``[left, right)`` and stays inside ``window``.
    A trailing partial bin is dropped; spikes at the last whole-bin edge or
    the window stop are not counted.

    See Also
    --------
    align_spikes_to_events : Get raw per-trial spike times.
    population_peri_event_histogram : PSTH for multiple units.
    plot_peri_event_histogram : Visualize PSTH results.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events.alignment import peri_event_histogram

    Compute PSTH around reward events:

    >>> spike_times = np.array([9.8, 10.1, 10.3, 19.9, 20.2])
    >>> reward_times = np.array([10.0, 20.0])
    >>> result = peri_event_histogram(
    ...     spike_times, reward_times, window=(-0.5, 1.0), bin_size=0.1
    ... )
    >>> print(f"Peak at {result.bin_centers[result.histogram.argmax()]:.2f}s")
    Peak at -0.15s

    Convert to firing rate:

    >>> rate = result.firing_rate  # spikes/second
    """
    window = _validate_window(window, context="peri_event_histogram")
    spike_times = _finite_times(spike_times, "spike_times")
    psth = _peri_event_histograms(
        [spike_times],
        event_times,
        window,
        bin_size,
        epochs,
        spike_window,
        context="peri_event_histogram",
        align=align,
    )
    return PeriEventResult(
        histogram=psth.histograms[0],
        sem=psth.sem[0],
        n_events=psth.event_indices.size,
        window=window,
        bin_size=bin_size,
        n_events_dropped=psth.n_dropped,
        event_indices=psth.event_indices,
        events_per_bin=psth.events_per_bin,
        align=None if psth.events_per_bin is None else align,
    )


@reads_pynapple_samples
def population_peri_event_histogram(
    spike_times: SpikeTrainsLike,
    event_times: NDArray[np.float64] | Intervals[Any],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
    align: str = "start",
    epochs: Any = None,
    spike_window: Any = None,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> PopulationPeriEventResult:
    """
    Compute peri-event histogram for a population of units.

    This function computes PSTH for multiple units simultaneously,
    providing both per-unit histograms and population-level statistics.

    Parameters
    ----------
    spike_times : sequence of NDArray[np.float64], or pynapple TsGroup
        Spike times in seconds, one 1-D array of shape (n_spikes_for_unit,)
        per unit, or a labelled spike group such as a pynapple ``TsGroup``.
        A group's index becomes the result's ``unit_ids``.
    event_times : NDArray[np.float64], shape (n_events,), pynapple Ts/Tsd, or Intervals
        Event times (e.g., stimulus onset) in seconds. A pynapple ``Ts`` or ``Tsd`` is read by its timestamps.
        An :class:`~neurospatial.Intervals` (trials) gives one event per row,
        aligned by ``align``; see Notes.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, in seconds,
        half-open ``[start, stop)``: a spike exactly at ``event + stop`` is
        left out.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).
    align : {"start", "stop"}, default="start"
        For an ``Intervals``, align each row at its start or at its stop.
        Plain event times are the alignment points; only the default is
        allowed with them.
    epochs : (start, stop), array-like of shape (n, 2), Intervals, IntervalSet, \
or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``event_times``). An event is kept only if its entire
        analysis window lies inside one recording window. ``None`` (default)
        means unrestricted. An :class:`~neurospatial.Intervals` with no windows
        (a condition with no trials) is an empty selection: the result is NaN
        with ``n_events == 0``, and a warning says so.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording, on the same clock as
        ``event_times``. Events are kept only if their entire analysis window
        lies inside both ``epochs`` and ``spike_window``. ``None`` adds no
        recording constraint; this spike-only analysis has no position stream
        from which to assume coverage. Pass ``spike_window`` (or ``epochs``)
        when the recording has gaps or edges inside your event windows.
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per unit in the
        same order as ``spike_times``. Stored on the result's ``unit_ids``
        field. Defaults to ``np.arange(n_units)``, or to the group's index for
        a labelled group. A wrong-length or repeated value raises
        ``ValueError``, and so does a value that differs from a labelled
        group's index (relabel the group itself instead).

    Returns
    -------
    PopulationPeriEventResult
        Dataclass containing:
        - bin_centers: Time relative to event (seconds), shape (n_bins,)
        - histograms: Per-unit mean spike counts, shape (n_units, n_bins)
        - sem: Per-unit SEM across events, shape (n_units, n_bins)
        - mean_histogram: Population average, shape (n_bins,)
        - mean_sem: SEM across events of the population average, shape
          (n_bins,)
        - n_events: Number of retained events
        - n_events_dropped: Events whose windows leave the recording
        - event_indices: Positions of retained events in the original input;
          use ``event_times[result.event_indices]`` for a matching raster cohort
        - n_units: Number of units
        - unit_table: Per-unit metadata of a ``SpikeTrains`` or a pynapple
          ``TsGroup`` input (its ``rate`` column dropped), else None
        - window: The input window
        - bin_size: The input bin_size

    Raises
    ------
    ValueError
        If spike_times is empty, a spike train or event_times is not 1-D or
        holds NaN or Inf, event_times is empty, window is invalid or
        inverted, bin_size is not a finite positive number or leaves no
        complete bin, time windows are invalid, or windows with duration keep
        no event (usually event times on another clock).

    Warns
    -----
    UserWarning
        When the window spans more than 60 s (it may be in milliseconds), or
        when ``epochs`` or ``spike_window`` is an empty selection.

    Notes
    -----
    With a single retained event the SEM is undefined and is NaN.

    For intervals, each row's window ends where the row does on the side away
    from the alignment point: aligned at the start, no bin at or after the
    trial's stop counts (it would be the next trial); aligned at the stop, no
    bin before the trial's start. The other side counts as for any event, so
    the time at the well before a departure or after an arrival stays in. A
    bin counts for a trial only when it lies wholly inside the trial's
    window, so each bin is the mean over the trials under way for all of it
    (``events_per_bin``), NaN where there are none. ``n_events_dropped`` also
    counts trials holding no whole bin. To keep only events whose whole
    window lies inside one trial, pass plain times with ``epochs=trials``.

    Every bin is half-open ``[left, right)`` and stays inside ``window``.
    A trailing partial bin is dropped; spikes at the last whole-bin edge or
    the window stop are not counted.

    See Also
    --------
    peri_event_histogram : PSTH for single unit.
    align_spikes_to_events : Get raw per-trial spike times.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events.alignment import population_peri_event_histogram

    Compute population PSTH for multi-unit recording:

    >>> spike_times = [
    ...     np.array([9.8, 10.1, 10.3]),  # Unit 1
    ...     np.array([9.9, 10.0, 10.5, 10.8]),  # Unit 2
    ... ]
    >>> stim_times = np.array([10.0, 20.0, 30.0])
    >>> result = population_peri_event_histogram(
    ...     spike_times, stim_times, window=(-0.5, 1.0), bin_size=0.1
    ... )
    >>> print(f"{result.n_units} units, {result.n_events} events")
    2 units, 3 events

    Get firing rates for all units:

    >>> rates = result.firing_rates  # shape: (n_units, n_bins)
    """
    # Extract trains by unit label from a spike group (iterating a TsGroup
    # yields its keys, not its trains).
    trains, extracted_ids = _as_spike_trains_with_ids(spike_times)

    if len(trains) == 0:
        msg = (
            "spike_times is empty.\n"
            "  WHY: Cannot compute population PSTH without any units.\n"
            "  Fix: Provide at least one spike train."
        )
        raise ValueError(msg)
    context = "population_peri_event_histogram"
    window = _validate_window(window, context=context)
    psth = _peri_event_histograms(
        trains,
        event_times,
        window,
        bin_size,
        epochs,
        spike_window,
        context=context,
        align=align,
    )
    # A labelled input keeps its own labels; a differing unit_ids= raises.
    resolved_unit_ids = resolve_unit_ids(
        unit_ids, len(trains), context=context, input_ids=extracted_ids
    )
    return PopulationPeriEventResult(
        histograms=psth.histograms,
        sem=psth.sem,
        mean_sem=psth.mean_sem,
        n_events=psth.event_indices.size,
        window=window,
        bin_size=bin_size,
        unit_ids=resolved_unit_ids,
        unit_table=spike_unit_table(spike_times),
        n_events_dropped=psth.n_dropped,
        event_indices=psth.event_indices,
        events_per_bin=psth.events_per_bin,
        align=None if psth.events_per_bin is None else align,
    )


def align_events(
    events: pd.DataFrame,
    reference_events: pd.DataFrame,
    window: tuple[float, float],
    *,
    event_column: str = "timestamp",
    reference_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Align events to reference events within a time window.

    This function extracts events that occur within a time window of each
    reference event, adding columns for the relative time and which reference
    event each is aligned to.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame to align. Must have timestamp column.
    reference_events : pd.DataFrame
        Reference events to align to. Must have timestamp column.
    window : tuple[float, float]
        Time window (start, stop) relative to each reference event, half-open
        ``[start, stop)``.
    event_column : str, default="timestamp"
        Name of timestamp column in events.
    reference_column : str, default="timestamp"
        Name of timestamp column in reference_events.

    Returns
    -------
    pd.DataFrame
        Events with added columns:
        - 'relative_time': Time relative to reference event (event - reference),
          in seconds
        - 'reference_index': Position (0-based) of the reference event in
          ``reference_events``, not its index label

        All original columns and dtypes from events are preserved, and the
        index holds each row's label in ``events``. An event inside several
        reference windows appears once per window, repeating its label. Rows
        are ordered by reference event, then by time.

    Raises
    ------
    TypeError
        If events or reference_events is not a DataFrame.
    ValueError
        If the window is invalid, a time column is missing or not numeric, or
        a time is NaN or infinite.

    See Also
    --------
    align_spikes_to_events : Align spike times (arrays) to events.
    peri_event_histogram : Compute PSTH from aligned spikes.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial.events.alignment import align_events

    Align lick events to reward times:

    >>> licks = pd.DataFrame(
    ...     {"timestamp": [9.8, 10.2, 20.1], "lick_strength": [0.5, 0.8, 0.7]}
    ... )
    >>> rewards = pd.DataFrame({"timestamp": [10.0, 20.0]})
    >>> aligned = align_events(licks, rewards, window=(-0.5, 1.0))
    >>> print(aligned[["timestamp", "relative_time", "reference_index"]])
       timestamp  relative_time  reference_index
    0        9.8           -0.2                0
    1       10.2            0.2                0
    2       20.1            0.1                1
    """
    _validate_window(window, context="align_events")
    event_times = _event_times(events, event_column, name="events", call="align_events")
    ref_times = _event_times(
        reference_events, reference_column, name="reference_events", call="align_events"
    )

    # Events in each reference's half-open window, grouped by reference and
    # ordered by time within each group.
    order = np.argsort(event_times, kind="stable")
    sorted_times = event_times[order]
    first = np.searchsorted(sorted_times, ref_times + window[0], side="left")
    stop = np.searchsorted(sorted_times, ref_times + window[1], side="left")
    n_per_reference = stop - first
    reference_index = np.repeat(np.arange(ref_times.size), n_per_reference)
    rows = order[np.repeat(first, n_per_reference) + ragged_arange(n_per_reference)]
    return events.iloc[rows].assign(
        relative_time=event_times[rows] - ref_times[reference_index],
        reference_index=reference_index.astype(np.int64),
    )
