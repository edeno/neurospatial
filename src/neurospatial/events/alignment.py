"""
Peri-event analysis functions for neurospatial.

This module provides functions for aligning neural data to events:
- align_spikes_to_events: Get per-trial spike times relative to events
- peri_event_histogram: Compute peri-event time histogram (PSTH)
- population_peri_event_histogram: Compute PSTH for multiple units
- align_events: Align events to reference events
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import (
    intervals_contain,
    ragged_arange,
    resolve_time_windows,
)
from neurospatial._spikes import _as_spike_trains_with_ids
from neurospatial._time_bins import (
    _count_spikes_in_time_bins,
    _time_bin_rounding,
)
from neurospatial.events._core import (
    PeriEventResult,
    PopulationPeriEventResult,
    _event_times,
    _finite_times,
    _psth_bins,
    _validate_window,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._typing import SpikeTrainsLike


def align_spikes_to_events(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
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
    event_times : NDArray[np.float64], shape (n_events,)
        Event times (e.g., stimulus onset) in seconds.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, half-open
        ``[start, stop)``: a spike exactly at ``event + stop`` is left out.
        For example, ``(-0.5, 1.0)`` captures 500 ms before to 1 s after each
        event.

    Returns
    -------
    list[NDArray[np.float64]]
        List of arrays, one per event, containing spike times relative to
        that event (spike_time - event_time). Arrays are sorted by relative
        time. Empty arrays are returned for events with no spikes in window.

    Raises
    ------
    ValueError
        If window start >= stop, or if spike/event times contain NaN or Inf.

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
    _validate_window(window, context="align_spikes_to_events")
    spike_times = _finite_times(spike_times, "spike_times")
    event_times = _finite_times(event_times, "event_times")

    # Handle empty cases
    if event_times.size == 0:
        return []

    # Sort spike times for efficient searching
    sorted_spikes = np.sort(spike_times)

    # Process each event
    result: list[NDArray[np.float64]] = []

    for event_t in event_times:
        # Compute window bounds in absolute time
        t_start = event_t + window[0]
        t_end = event_t + window[1]

        # Spikes in the half-open window [t_start, t_end).
        idx_start = np.searchsorted(sorted_spikes, t_start, side="left")
        idx_end = np.searchsorted(sorted_spikes, t_end, side="left")

        # Extract spikes in window and compute relative times
        spikes_in_window = sorted_spikes[idx_start:idx_end]
        relative_times = spikes_in_window - event_t

        result.append(relative_times)

    return result


def _keep_observed_events(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Keep events with observed windows and return their original input indices."""
    keep = np.ones(event_times.shape, dtype=bool)
    for windows in (epochs, spike_window):
        if windows is not None:
            keep &= intervals_contain(
                windows, event_times + window[0], event_times + window[1]
            )
    n_dropped = int(event_times.size - keep.sum())
    if n_dropped == event_times.size:
        raise ValueError(
            f"All {event_times.size} events were dropped from the peri-event "
            f"histogram: no event's window [event{window[0]:+g} s, "
            f"event{window[1]:+g} s) lies entirely inside epochs ∩ spike_window. "
            f"\nWhy: a window that reaches unrecorded time would be averaged as "
            f"if no spikes occurred there.\nFix: check that event_times, epochs "
            f"and spike_window share one clock (seconds), or narrow `window`."
        )
    return event_times[keep], np.flatnonzero(keep)


def _iter_peri_event_bin_edges(
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    bin_edges: NDArray[np.float64],
    bin_size: float,
) -> Iterator[NDArray[np.float64]]:
    """Yield the absolute bin edges shared by histogram counting and rasters."""
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
    for event, bounds in zip(event_times, windows, strict=True):
        yield np.minimum(event + bin_edges, bounds[1])


def _count_peri_event_bins(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    bin_edges: NDArray[np.float64],
    bin_size: float,
) -> NDArray[np.float64]:
    """Count on the absolute clock without rounding spikes across boundaries."""
    # Count on the absolute clock, not on relative times: subtraction can move
    # an exact boundary spike across a bin edge.
    spikes = np.sort(_finite_times(spike_times, "spike_times"))
    histograms = np.zeros((event_times.size, bin_edges.size - 1), dtype=np.float64)
    for i, edges in enumerate(
        _iter_peri_event_bin_edges(event_times, window, bin_edges, bin_size)
    ):
        start = np.searchsorted(spikes, edges[0], side="left")
        stop = np.searchsorted(spikes, edges[-1], side="left")
        histograms[i] = _count_spikes_in_time_bins(
            [spikes[start:stop]], edges[:-1], edges[1:]
        )[:, 0]
    return histograms


def peri_event_histogram(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
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
    event_times : NDArray[np.float64], shape (n_events,)
        Event times (e.g., stimulus onset) in seconds.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, half-open
        ``[start, stop)``: a spike exactly at ``event + stop`` is left out.
        For example, ``(-0.5, 1.0)`` captures 500 ms before to 1 s after each
        event.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``event_times``). An event is kept only if its entire
        analysis window lies inside one recording window. ``None`` (default)
        means unrestricted.
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
        time windows are invalid, or all events are dropped.

    Warns
    -----
    UserWarning
        When computing PSTH with a single event (SEM is undefined), and when
        the window spans more than 60 s (it may be in milliseconds).

    Notes
    -----
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
    bin_edges, _ = _psth_bins(window, bin_size, context="peri_event_histogram")
    spike_times = _finite_times(spike_times, "spike_times")
    event_times = _finite_times(event_times, "event_times")

    # Validate event_times not empty
    if event_times.size == 0:
        raise ValueError(
            "event_times is empty.\n"
            "  WHY: Cannot compute PSTH without events to align to.\n"
            "  Fix: Provide at least one event time."
        )

    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    n_input_events = event_times.size
    event_times, event_indices = _keep_observed_events(
        event_times, window, resolved_epochs, resolved_spike_window
    )
    n_dropped = int(n_input_events - event_indices.size)
    n_events = len(event_times)

    # Warn about single event
    if n_events == 1:
        warnings.warn(
            "Computing PSTH with single event - SEM is undefined (will be NaN).",
            UserWarning,
            stacklevel=2,
        )

    n_bins = bin_edges.size - 1

    # Compute histogram for each event
    histograms = _count_peri_event_bins(
        spike_times, event_times, window, bin_edges, bin_size
    )

    # Compute mean and SEM
    mean_histogram = histograms.mean(axis=0)

    if n_events > 1:
        # SEM = std / sqrt(n)
        sem = histograms.std(axis=0, ddof=1) / np.sqrt(n_events)
    else:
        sem = np.full(n_bins, np.nan, dtype=np.float64)

    return PeriEventResult(
        histogram=mean_histogram,
        sem=sem,
        n_events=n_events,
        window=window,
        bin_size=bin_size,
        n_events_dropped=n_dropped,
        event_indices=event_indices,
    )


def population_peri_event_histogram(
    spike_times: SpikeTrainsLike,
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
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
    event_times : NDArray[np.float64], shape (n_events,)
        Event times (e.g., stimulus onset) in seconds.
    window : tuple[float, float]
        Time window (start, stop) relative to each event, in seconds,
        half-open ``[start, stop)``: a spike exactly at ``event + stop`` is
        left out.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``event_times``). An event is kept only if its entire
        analysis window lies inside one recording window. ``None`` (default)
        means unrestricted.
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
        - window: The input window
        - bin_size: The input bin_size

    Raises
    ------
    ValueError
        If spike_times is empty, a spike train or event_times is not 1-D or
        holds NaN or Inf, event_times is empty, window is invalid or
        inverted, bin_size is not a finite positive number or leaves no
        complete bin, time windows are invalid, or all events are dropped.

    Warns
    -----
    UserWarning
        With a single event (SEM is undefined), and when the window spans
        more than 60 s (it may be in milliseconds).

    Notes
    -----
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
        raise ValueError(
            "spike_times is empty.\n"
            "  WHY: Cannot compute population PSTH without any units.\n"
            "  Fix: Provide at least one spike train."
        )
    context = "population_peri_event_histogram"
    window = _validate_window(window, context=context)
    bin_edges, _ = _psth_bins(window, bin_size, context=context)
    trains = [_finite_times(train, "spike_times") for train in trains]
    event_times = _finite_times(event_times, "event_times")

    # Validate event_times not empty
    if event_times.size == 0:
        raise ValueError(
            "event_times is empty.\n"
            "  WHY: Cannot compute PSTH without events to align to.\n"
            "  Fix: Provide at least one event time."
        )

    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    n_input_events = event_times.size
    event_times, event_indices = _keep_observed_events(
        event_times, window, resolved_epochs, resolved_spike_window
    )
    n_dropped = int(n_input_events - event_indices.size)
    n_units = len(trains)
    n_events = len(event_times)

    # Resolve and validate per-unit identity labels (defaults to arange). A
    # labelled input keeps its own labels; a differing unit_ids= raises.
    from neurospatial._results import resolve_unit_ids

    resolved_unit_ids = resolve_unit_ids(
        unit_ids,
        n_units,
        context="population_peri_event_histogram",
        input_ids=extracted_ids,
    )

    # Warn about single event
    if n_events == 1:
        warnings.warn(
            "Computing PSTH with single event - SEM is undefined (will be NaN).",
            UserWarning,
            stacklevel=2,
        )

    n_bins = bin_edges.size - 1

    # Compute per-unit histograms
    # Shape: (n_units, n_events, n_bins)
    all_histograms = np.zeros((n_units, n_events, n_bins), dtype=np.float64)

    for unit_idx, train in enumerate(trains):
        all_histograms[unit_idx] = _count_peri_event_bins(
            train, event_times, window, bin_edges, bin_size
        )

    # Compute per-unit mean and SEM across events
    # histograms shape: (n_units, n_bins)
    histograms = all_histograms.mean(axis=1)

    # The population mean's SEM is across events of each event's mean over
    # units, the same estimand as each unit's SEM.
    per_event_mean = all_histograms.mean(axis=0)
    if n_events > 1:
        sem = all_histograms.std(axis=1, ddof=1) / np.sqrt(n_events)
        mean_sem = per_event_mean.std(axis=0, ddof=1) / np.sqrt(n_events)
    else:
        sem = np.full((n_units, n_bins), np.nan, dtype=np.float64)
        mean_sem = np.full(n_bins, np.nan, dtype=np.float64)

    return PopulationPeriEventResult(
        histograms=histograms,
        sem=sem,
        mean_sem=mean_sem,
        n_events=n_events,
        window=window,
        bin_size=bin_size,
        unit_ids=resolved_unit_ids,
        n_events_dropped=n_dropped,
        event_indices=event_indices,
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
