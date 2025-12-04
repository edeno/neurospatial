"""
Core dataclasses and validation helpers for the events module.

This module provides:
- Result dataclasses: PeriEventResult, PopulationPeriEventResult
- Validation helpers: validate_events_dataframe, validate_spatial_columns
- Visualization: plot_peri_event_histogram
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


# --- Result Dataclasses ---


@dataclass(frozen=True)
class PeriEventResult:
    """
    Result from peri-event histogram analysis.

    This dataclass stores the output of `peri_event_histogram()`, including
    the histogram bins, spike counts/rates, and statistics across events.

    Attributes
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds). Negative = before, positive = after.
    histogram : NDArray[np.float64], shape (n_bins,)
        Mean spike count per time bin across events.
    sem : NDArray[np.float64], shape (n_bins,)
        Standard error of the mean across events.
    n_events : int
        Number of events used in analysis.
    window : tuple[float, float]
        Time window (start, end) relative to event in seconds.
    bin_size : float
        Width of time bins (seconds).

    Methods
    -------
    firing_rate()
        Convert histogram counts to firing rate (Hz).

    See Also
    --------
    peri_event_histogram : Compute PSTH from spikes and events.
    plot_peri_event_histogram : Visualize PSTH results.

    Examples
    --------
    >>> result = peri_event_histogram(spikes, events, window=(-1, 2))  # doctest: +SKIP
    >>> # Access firing rate
    >>> rate = result.firing_rate()
    >>> # Find peak response time
    >>> peak_time = result.bin_centers[np.argmax(rate)]
    """

    bin_centers: NDArray[np.float64]
    histogram: NDArray[np.float64]
    sem: NDArray[np.float64]
    n_events: int
    window: tuple[float, float]
    bin_size: float

    def firing_rate(self) -> NDArray[np.float64]:
        """
        Convert spike counts to firing rate (Hz).

        Returns
        -------
        NDArray[np.float64], shape (n_bins,)
            Firing rate in spikes per second (Hz).
        """
        return self.histogram / self.bin_size


@dataclass(frozen=True)
class PopulationPeriEventResult:
    """
    Result from population peri-event histogram analysis.

    This dataclass stores the output of `population_peri_event_histogram()`,
    including per-unit histograms and population-level statistics.

    Attributes
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Time relative to event (seconds).
    histograms : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit spike count per time bin.
    sem : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit standard error of the mean across events.
    mean_histogram : NDArray[np.float64], shape (n_bins,)
        Population average histogram across all units.
    n_events : int
        Number of events used in analysis.
    n_units : int
        Number of units in population.
    window : tuple[float, float]
        Time window (start, end) relative to event in seconds.
    bin_size : float
        Width of time bins (seconds).

    Methods
    -------
    firing_rates()
        Convert histogram counts to firing rates (Hz) for all units.

    See Also
    --------
    population_peri_event_histogram : Compute population PSTH.
    peri_event_histogram : Single-unit PSTH.

    Examples
    --------
    >>> result = population_peri_event_histogram(  # doctest: +SKIP
    ...     spike_trains, events, window=(-1, 2)
    ... )
    >>> # Get firing rates for all units
    >>> rates = result.firing_rates()  # shape: (n_units, n_bins)
    >>> # Get population average
    >>> pop_rate = result.mean_histogram / result.bin_size
    """

    bin_centers: NDArray[np.float64]
    histograms: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_histogram: NDArray[np.float64]
    n_events: int
    n_units: int
    window: tuple[float, float]
    bin_size: float

    def firing_rates(self) -> NDArray[np.float64]:
        """
        Convert spike counts to firing rates (Hz) for all units.

        Returns
        -------
        NDArray[np.float64], shape (n_units, n_bins)
            Firing rates in spikes per second (Hz).
        """
        return self.histograms / self.bin_size


# --- Validation Helpers ---


def validate_events_dataframe(
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
    >>> validate_events_dataframe(df)  # No error

    >>> df_bad = pd.DataFrame({"time": [1.0, 2.0]})
    >>> validate_events_dataframe(df_bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Missing required columns: ['timestamp']...
    """
    import pandas as pd

    # Check type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pd.DataFrame, got {type(df).__name__}.\n"
            "  WHY: Events must be a pandas DataFrame for NWB compatibility.\n"
            "  HOW: Convert using pd.DataFrame({'timestamp': times})"
        )

    # Check required columns
    all_required = [timestamp_column] + (required_columns or [])
    missing = [col for col in all_required if col not in df.columns]
    if missing:
        context_str = f" for {context}" if context else ""
        raise ValueError(
            f"Missing required columns: {missing}.\n"
            f"  WHY: These columns are needed{context_str}.\n"
            f"  HOW: Add missing columns to DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )

    # Check timestamp is numeric
    if not np.issubdtype(df[timestamp_column].dtype, np.number):
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains non-numeric values.\n"
            "  WHY: Timestamps must be numeric (seconds from session start).\n"
            f"  HOW: Convert timestamps: df['{timestamp_column}'] = "
            f"df['{timestamp_column}'].astype(float)"
        )


def validate_spatial_columns(
    df: pd.DataFrame,
    *,
    require_positions: bool = False,
    context: str = "",
) -> bool:
    """
    Check if DataFrame has spatial columns (x, y); optionally require them.

    This function checks for the presence of 'x' and 'y' columns in an
    events DataFrame. Spatial columns are needed for functions that
    compute spatial event rates or filter events by region.

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame to check.
    require_positions : bool, default=False
        If True, raise ValueError when spatial columns are missing.
    context : str, optional
        Context for error messages (e.g., function name).

    Returns
    -------
    bool
        True if spatial columns ('x', 'y') are present, False otherwise.

    Raises
    ------
    ValueError
        If require_positions=True and spatial columns are missing.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"timestamp": [1.0], "x": [10.0], "y": [20.0]})
    >>> validate_spatial_columns(df)
    True

    >>> df_no_spatial = pd.DataFrame({"timestamp": [1.0]})
    >>> validate_spatial_columns(df_no_spatial)
    False

    >>> validate_spatial_columns(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     df_no_spatial, require_positions=True
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Events DataFrame missing spatial columns...
    """
    has_x = "x" in df.columns
    has_y = "y" in df.columns
    has_positions = has_x and has_y

    if require_positions and not has_positions:
        context_str = context if context else "This function"
        raise ValueError(
            "Events DataFrame missing spatial columns ('x', 'y').\n"
            f"  WHY: {context_str} requires event positions.\n"
            "  HOW: Use add_positions(events, positions, times)"
        )

    return has_positions


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
    >>> result = peri_event_histogram(spikes, events)  # doctest: +SKIP
    >>> ax = plot_peri_event_histogram(result, title="Reward response")
    >>> ax.axvline(0, color="k", linestyle="--", alpha=0.5)  # Mark event
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # Get data to plot
    x = result.bin_centers
    if as_rate:
        y = result.firing_rate()
        sem = result.sem / result.bin_size
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
