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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.events._core import PeriEventResult, PopulationPeriEventResult

if TYPE_CHECKING:
    import pandas as pd


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
        Time window (start, end) relative to each event. For example,
        ``(-0.5, 1.0)`` captures 500ms before to 1s after each event.

    Returns
    -------
    list[NDArray[np.float64]]
        List of arrays, one per event, containing spike times relative to
        that event (spike_time - event_time). Arrays are sorted by relative
        time. Empty arrays are returned for events with no spikes in window.

    Raises
    ------
    ValueError
        If window start > end, or if spike/event times contain NaN or Inf.

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
    # Validate window
    if window[0] > window[1]:
        raise ValueError(
            f"Window start ({window[0]}) must be <= end ({window[1]}).\n"
            "  WHY: Window defines valid time range relative to events.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Convert to arrays if needed
    spike_times = np.asarray(spike_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)

    # Validate no NaN/Inf
    if spike_times.size > 0:
        if np.any(np.isnan(spike_times)):
            raise ValueError(
                "spike_times contains NaN values.\n"
                "  WHY: Cannot compute relative times with undefined values.\n"
                "  HOW: Remove or interpolate NaN values before alignment."
            )
        if np.any(np.isinf(spike_times)):
            raise ValueError(
                "spike_times contains Inf values.\n"
                "  WHY: Cannot compute relative times with infinite values.\n"
                "  HOW: Remove Inf values before alignment."
            )

    if event_times.size > 0 and np.any(np.isnan(event_times)):
        raise ValueError(
            "event_times contains NaN values.\n"
            "  WHY: Cannot align spikes to undefined event times.\n"
            "  HOW: Remove NaN event times before alignment."
        )

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

        # Find spikes within window using binary search
        # searchsorted with side='left' gives first index where value could be inserted
        # searchsorted with side='right' gives last index + 1
        idx_start = np.searchsorted(sorted_spikes, t_start, side="left")
        idx_end = np.searchsorted(sorted_spikes, t_end, side="right")

        # Extract spikes in window and compute relative times
        spikes_in_window = sorted_spikes[idx_start:idx_end]
        relative_times = spikes_in_window - event_t

        result.append(relative_times)

    return result


def peri_event_histogram(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
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
        Time window (start, end) relative to each event. For example,
        ``(-0.5, 1.0)`` captures 500ms before to 1s after each event.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).

    Returns
    -------
    PeriEventResult
        Dataclass containing:
        - bin_centers: Time relative to event (seconds)
        - histogram: Mean spike count per bin across events
        - sem: Standard error of the mean across events
        - n_events: Number of events
        - window: The input window
        - bin_size: The input bin_size

    Raises
    ------
    ValueError
        If window is inverted, bin_size is non-positive, or event_times is empty.

    Warns
    -----
    UserWarning
        When computing PSTH with a single event (SEM is undefined).

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

    >>> rate = result.firing_rate()  # spikes/second
    """
    # Validate bin_size
    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            "  WHY: bin_size defines histogram bin width.\n"
            "  HOW: Use a positive value like bin_size=0.025 (25ms)."
        )

    # Validate window
    if window[0] > window[1]:
        raise ValueError(
            f"Window start ({window[0]}) must be <= end ({window[1]}).\n"
            "  WHY: Window defines valid time range relative to events.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Convert to arrays
    event_times = np.asarray(event_times, dtype=np.float64)

    # Validate event_times not empty
    if event_times.size == 0:
        raise ValueError(
            "event_times is empty.\n"
            "  WHY: Cannot compute PSTH without events to align to.\n"
            "  HOW: Provide at least one event time."
        )

    n_events = len(event_times)

    # Warn about single event
    if n_events == 1:
        warnings.warn(
            "Computing PSTH with single event - SEM is undefined (will be NaN).",
            UserWarning,
            stacklevel=2,
        )

    # Get aligned spikes for each event
    aligned = align_spikes_to_events(spike_times, event_times, window)

    # Create bin edges
    window_duration = window[1] - window[0]
    n_bins = int(np.ceil(window_duration / bin_size))
    bin_edges = np.linspace(window[0], window[0] + n_bins * bin_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram for each event
    histograms = np.zeros((n_events, n_bins), dtype=np.float64)
    for i, trial_spikes in enumerate(aligned):
        if len(trial_spikes) > 0:
            counts, _ = np.histogram(trial_spikes, bins=bin_edges)
            histograms[i] = counts

    # Compute mean and SEM
    mean_histogram = histograms.mean(axis=0)

    if n_events > 1:
        # SEM = std / sqrt(n)
        sem = histograms.std(axis=0, ddof=1) / np.sqrt(n_events)
    else:
        sem = np.full(n_bins, np.nan, dtype=np.float64)

    return PeriEventResult(
        bin_centers=bin_centers,
        histogram=mean_histogram,
        sem=sem,
        n_events=n_events,
        window=window,
        bin_size=bin_size,
    )


def population_peri_event_histogram(
    spike_trains: list[NDArray[np.float64]],
    event_times: NDArray[np.float64],
    window: tuple[float, float],
    *,
    bin_size: float = 0.025,
) -> PopulationPeriEventResult:
    """
    Compute peri-event histogram for a population of units.

    This function computes PSTH for multiple units simultaneously,
    providing both per-unit histograms and population-level statistics.

    Parameters
    ----------
    spike_trains : list[NDArray[np.float64]]
        List of spike time arrays, one per unit. Each array has shape
        (n_spikes_for_unit,) in seconds.
    event_times : NDArray[np.float64], shape (n_events,)
        Event times (e.g., stimulus onset) in seconds.
    window : tuple[float, float]
        Time window (start, end) relative to each event.
    bin_size : float, default=0.025
        Width of time bins in seconds (default 25ms).

    Returns
    -------
    PopulationPeriEventResult
        Dataclass containing:
        - bin_centers: Time relative to event (seconds), shape (n_bins,)
        - histograms: Per-unit mean spike counts, shape (n_units, n_bins)
        - sem: Per-unit SEM across events, shape (n_units, n_bins)
        - mean_histogram: Population average, shape (n_bins,)
        - n_events: Number of events
        - n_units: Number of units
        - window: The input window
        - bin_size: The input bin_size

    Raises
    ------
    ValueError
        If spike_trains is empty, event_times is empty, window is inverted,
        or bin_size is non-positive.

    See Also
    --------
    peri_event_histogram : PSTH for single unit.
    align_spikes_to_events : Get raw per-trial spike times.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events.alignment import population_peri_event_histogram

    Compute population PSTH for multi-unit recording:

    >>> spike_trains = [
    ...     np.array([9.8, 10.1, 10.3]),  # Unit 1
    ...     np.array([9.9, 10.0, 10.5, 10.8]),  # Unit 2
    ... ]
    >>> stim_times = np.array([10.0, 20.0, 30.0])
    >>> result = population_peri_event_histogram(
    ...     spike_trains, stim_times, window=(-0.5, 1.0), bin_size=0.1
    ... )
    >>> print(f"{result.n_units} units, {result.n_events} events")
    2 units, 3 events

    Get firing rates for all units:

    >>> rates = result.firing_rates()  # shape: (n_units, n_bins)
    """
    # Validate spike_trains
    if len(spike_trains) == 0:
        raise ValueError(
            "spike_trains is empty.\n"
            "  WHY: Cannot compute population PSTH without any units.\n"
            "  HOW: Provide at least one spike train."
        )

    # Validate bin_size
    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            "  WHY: bin_size defines histogram bin width.\n"
            "  HOW: Use a positive value like bin_size=0.025 (25ms)."
        )

    # Validate window
    if window[0] > window[1]:
        raise ValueError(
            f"Window start ({window[0]}) must be <= end ({window[1]}).\n"
            "  WHY: Window defines valid time range relative to events.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Convert event_times to array
    event_times = np.asarray(event_times, dtype=np.float64)

    # Validate event_times not empty
    if event_times.size == 0:
        raise ValueError(
            "event_times is empty.\n"
            "  WHY: Cannot compute PSTH without events to align to.\n"
            "  HOW: Provide at least one event time."
        )

    n_units = len(spike_trains)
    n_events = len(event_times)

    # Warn about single event
    if n_events == 1:
        warnings.warn(
            "Computing PSTH with single event - SEM is undefined (will be NaN).",
            UserWarning,
            stacklevel=2,
        )

    # Create bin edges
    window_duration = window[1] - window[0]
    n_bins = int(np.ceil(window_duration / bin_size))
    bin_edges = np.linspace(window[0], window[0] + n_bins * bin_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute per-unit histograms
    # Shape: (n_units, n_events, n_bins)
    all_histograms = np.zeros((n_units, n_events, n_bins), dtype=np.float64)

    for unit_idx, spike_times in enumerate(spike_trains):
        # Get aligned spikes for this unit
        aligned = align_spikes_to_events(spike_times, event_times, window)

        # Compute histogram for each event
        for event_idx, trial_spikes in enumerate(aligned):
            if len(trial_spikes) > 0:
                counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                all_histograms[unit_idx, event_idx] = counts

    # Compute per-unit mean and SEM across events
    # histograms shape: (n_units, n_bins)
    histograms = all_histograms.mean(axis=1)

    if n_events > 1:
        # SEM = std / sqrt(n) for each unit
        sem = all_histograms.std(axis=1, ddof=1) / np.sqrt(n_events)
    else:
        sem = np.full((n_units, n_bins), np.nan, dtype=np.float64)

    # Population mean is average across units
    mean_histogram = histograms.mean(axis=0)

    return PopulationPeriEventResult(
        bin_centers=bin_centers,
        histograms=histograms,
        sem=sem,
        mean_histogram=mean_histogram,
        n_events=n_events,
        n_units=n_units,
        window=window,
        bin_size=bin_size,
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
        Time window (start, end) relative to each reference event.
    event_column : str, default="timestamp"
        Name of timestamp column in events.
    reference_column : str, default="timestamp"
        Name of timestamp column in reference_events.

    Returns
    -------
    pd.DataFrame
        Events with added columns:
        - 'relative_time': Time relative to reference event (event - reference)
        - 'reference_index': Index of the reference event (0-indexed)

        All original columns from events are preserved. If an event falls
        within multiple reference windows, it appears multiple times.

    Raises
    ------
    ValueError
        If window is inverted or required columns are missing.

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
    import pandas as pd

    # Validate window
    if window[0] > window[1]:
        raise ValueError(
            f"Window start ({window[0]}) must be <= end ({window[1]}).\n"
            "  WHY: Window defines valid time range relative to events.\n"
            "  HOW: Use window=(start, end) where start <= end."
        )

    # Validate event_column exists
    if event_column not in events.columns:
        raise ValueError(
            f"Events DataFrame missing '{event_column}' column.\n"
            "  WHY: Need timestamps to compute relative times.\n"
            f"  HOW: Ensure events has '{event_column}' column.\n"
            f"  Available columns: {list(events.columns)}"
        )

    # Validate reference_column exists
    if reference_column not in reference_events.columns:
        raise ValueError(
            f"Reference DataFrame missing '{reference_column}' column.\n"
            "  WHY: Need reference timestamps to align events to.\n"
            f"  HOW: Ensure reference_events has '{reference_column}' column.\n"
            f"  Available columns: {list(reference_events.columns)}"
        )

    # Handle empty cases
    if len(events) == 0 or len(reference_events) == 0:
        result = events.iloc[0:0].copy()
        result["relative_time"] = pd.Series([], dtype=np.float64)
        result["reference_index"] = pd.Series([], dtype=np.int64)
        return result

    # Get timestamps as arrays
    event_times = events[event_column].values
    ref_times = reference_events[reference_column].values

    # Sort event times for efficient searching
    sorted_idx = np.argsort(event_times)
    sorted_event_times = event_times[sorted_idx]

    # Collect aligned events
    aligned_rows = []

    for ref_idx, ref_t in enumerate(ref_times):
        # Compute window bounds
        t_start = ref_t + window[0]
        t_end = ref_t + window[1]

        # Find events within window using binary search
        idx_start = np.searchsorted(sorted_event_times, t_start, side="left")
        idx_end = np.searchsorted(sorted_event_times, t_end, side="right")

        # Get original indices of events in window
        event_indices = sorted_idx[idx_start:idx_end]

        for orig_idx in event_indices:
            event_t = event_times[orig_idx]
            relative_time = event_t - ref_t

            # Copy event row with new columns
            row_data = events.iloc[orig_idx].to_dict()
            row_data["relative_time"] = relative_time
            row_data["reference_index"] = ref_idx
            aligned_rows.append(row_data)

    # Create result DataFrame
    if aligned_rows:
        result = pd.DataFrame(aligned_rows)
    else:
        # Empty result with correct columns
        result = events.iloc[0:0].copy()
        result["relative_time"] = pd.Series([], dtype=np.float64)
        result["reference_index"] = pd.Series([], dtype=np.int64)

    return result
