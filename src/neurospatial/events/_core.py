"""
Core dataclasses and validation helpers for the events module.

This module provides:
- Result dataclasses: PeriEventResult, PopulationPeriEventResult
- Validation helpers: validate_events_dataframe, validate_spatial_columns
- Visualization: plot_peri_event_histogram
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._results import ResultMixin

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


# --- Result Dataclasses ---


@dataclass(frozen=True)
class PeriEventResult(ResultMixin):
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
        Standard error of the mean across events, in **count units**
        (matching ``histogram``). Divide by ``bin_size`` to convert to
        Hz when plotting alongside ``firing_rate``. NaN with a single
        event (SEM is undefined).
    n_events : int
        Number of events used in analysis.
    window : tuple[float, float]
        Time window (start, end) relative to event in seconds.
    bin_size : float
        Width of time bins (seconds).
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation. Currently set only if
        provided explicitly: PSTH auto-population is forward-compat scaffolding
        pending the PSTH ResultMixin work in Phase 1.3.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate (Hz). Cached on construction as ``histogram /
        bin_size``; treat as a read-only attribute.

    See Also
    --------
    peri_event_histogram : Compute PSTH from spikes and events.
    plot_peri_event_histogram : Visualize PSTH results.

    Examples
    --------
    >>> result = peri_event_histogram(spikes, events, window=(-1, 2))  # doctest: +SKIP
    >>> # Access firing rate
    >>> rate = result.firing_rate  # doctest: +SKIP
    >>> # Find peak response time
    >>> peak_time = result.bin_centers[np.argmax(rate)]  # doctest: +SKIP
    """

    bin_centers: NDArray[np.float64]
    histogram: NDArray[np.float64]
    sem: NDArray[np.float64]
    n_events: int
    window: tuple[float, float]
    bin_size: float
    unit_id: int | str | None = None
    firing_rate: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass setattr to populate the cached field.
        object.__setattr__(self, "firing_rate", self.histogram / self.bin_size)

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table of the PSTH: one row per time bin.

        One row per time bin, carrying a ``unit_id`` column (the unit this PSTH
        was computed for; ``None`` for a standalone single-unit computation).
        Use this for plotting / detailed inspection.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin_center`` (s, time relative to event),
            ``firing_rate`` (Hz), ``histogram`` (mean spike count per bin), and
            ``sem`` (count units).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PeriEventResult
        >>> bc = np.array([-0.5, 0.0, 0.5])
        >>> result = PeriEventResult(
        ...     bin_centers=bc,
        ...     histogram=np.array([0.1, 0.4, 0.2]),
        ...     sem=np.array([0.01, 0.02, 0.01]),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ...     unit_id="u0",
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['unit_id', 'bin_center', 'firing_rate', 'histogram', 'sem']
        >>> len(df)
        3
        """
        import pandas as pd

        bin_centers = np.asarray(self.bin_centers)
        n_bins = bin_centers.shape[0]
        return pd.DataFrame(
            {
                "unit_id": [self.unit_id] * n_bins,
                "bin_center": bin_centers,
                "firing_rate": np.asarray(self.firing_rate),
                "histogram": np.asarray(self.histogram),
                "sem": np.asarray(self.sem),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of headline scalars for this PSTH.

        Returns
        -------
        dict
            Mapping with ``unit_id``, ``n_events`` (int), ``peak_rate`` (Hz),
            ``peak_latency`` (s, time of the peak relative to the event), and
            ``baseline_rate`` (Hz, mean rate over the pre-event window).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PeriEventResult
        >>> bc = np.array([-0.5, 0.0, 0.5])
        >>> result = PeriEventResult(
        ...     bin_centers=bc,
        ...     histogram=np.array([0.1, 0.4, 0.2]),
        ...     sem=np.array([0.01, 0.02, 0.01]),
        ...     n_events=10,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> s = result.summary()
        >>> sorted(s)
        ['baseline_rate', 'n_events', 'peak_latency', 'peak_rate', 'unit_id']
        >>> round(s["peak_latency"], 3)
        0.0
        """
        rate = np.asarray(self.firing_rate)
        bin_centers = np.asarray(self.bin_centers)
        if rate.size == 0:
            peak_rate = float("nan")
            peak_latency = float("nan")
        else:
            peak_idx = int(np.nanargmax(rate))
            peak_rate = float(rate[peak_idx])
            peak_latency = float(bin_centers[peak_idx])

        pre = bin_centers < 0
        baseline_rate = float(np.nanmean(rate[pre])) if np.any(pre) else float("nan")
        return {
            "unit_id": self.unit_id,
            "n_events": int(self.n_events),
            "peak_rate": peak_rate,
            "peak_latency": peak_latency,
            "baseline_rate": baseline_rate,
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


@dataclass(frozen=True)
class PopulationPeriEventResult(ResultMixin):
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
        Per-unit standard error of the mean across events, in **count
        units** (matching ``histograms``). Divide by ``bin_size`` to
        convert to Hz.
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
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. Rides alongside the rates for downstream filtering/grouping.
    firing_rates : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit firing rates (Hz). Cached on construction as
        ``histograms / bin_size``; treat as a read-only attribute.
    mean_firing_rate : NDArray[np.float64], shape (n_bins,)
        Population-average firing rate (Hz). Cached on construction as
        ``mean_histogram / bin_size``; the population-level analog of
        ``PeriEventResult.firing_rate``. Treat as a read-only attribute.

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
    >>> rates = result.firing_rates  # shape: (n_units, n_bins)  # doctest: +SKIP
    >>> # Get population average firing rate (Hz)
    >>> pop_rate = result.mean_firing_rate  # shape: (n_bins,)  # doctest: +SKIP
    """

    bin_centers: NDArray[np.float64]
    histograms: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_histogram: NDArray[np.float64]
    n_events: int
    n_units: int
    window: tuple[float, float]
    bin_size: float
    unit_ids: NDArray[Any] = field(default=None, compare=False)  # type: ignore[arg-type]
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    firing_rates: NDArray[np.float64] = field(init=False)
    mean_firing_rate: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        from neurospatial._results import resolve_unit_ids

        # Frozen dataclass: bypass setattr to populate the cached fields.
        object.__setattr__(self, "firing_rates", self.histograms / self.bin_size)
        object.__setattr__(
            self, "mean_firing_rate", self.mean_histogram / self.bin_size
        )
        n_units = int(np.asarray(self.histograms).shape[0])
        object.__setattr__(
            self,
            "unit_ids",
            resolve_unit_ids(self.unit_ids, n_units),
        )

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
            ``histogram`` (mean spike count per bin), and ``sem`` (count units).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> bc = np.array([-0.5, 0.0, 0.5])
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     bin_centers=bc,
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_histogram=hist.mean(axis=0),
        ...     n_events=10,
        ...     n_units=2,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['unit_id', 'bin_center', 'firing_rate', 'histogram', 'sem']
        >>> len(df) == 2 * 3
        True
        """
        import pandas as pd

        bin_centers = np.asarray(self.bin_centers)
        n_units, n_bins = np.asarray(self.histograms).shape
        firing_rates = np.asarray(self.firing_rates)
        histograms = np.asarray(self.histograms)
        sem = np.asarray(self.sem)
        return pd.DataFrame(
            {
                "unit_id": np.repeat(np.asarray(self.unit_ids), n_bins),
                "bin_center": np.tile(bin_centers, n_units),
                "firing_rate": firing_rates.reshape(-1),
                "histogram": histograms.reshape(-1),
                "sem": sem.reshape(-1),
            }
        )

    def summary_table(self) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        One row per unit with scalar response metrics. For the dense per-bin
        frame (one row per ``(unit, time-bin)``) use :meth:`to_dataframe`.

        Returns
        -------
        pandas.DataFrame
            One row per unit, indexed by ``unit_id``, with columns
            ``peak_rate`` (Hz), ``peak_latency`` (s, time of the per-unit peak
            relative to the event), and ``baseline_rate`` (Hz, mean rate over
            the pre-event window).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> bc = np.array([-0.5, 0.0, 0.5])
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     bin_centers=bc,
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_histogram=hist.mean(axis=0),
        ...     n_events=10,
        ...     n_units=2,
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
        import pandas as pd

        rates = np.asarray(self.firing_rates)
        bin_centers = np.asarray(self.bin_centers)
        n_units = rates.shape[0]

        if rates.size == 0:
            peak_rates = np.full(n_units, np.nan)
            peak_latencies = np.full(n_units, np.nan)
        else:
            peak_idx = np.nanargmax(rates, axis=1)
            peak_rates = rates[np.arange(n_units), peak_idx]
            peak_latencies = bin_centers[peak_idx]

        pre = bin_centers < 0
        if np.any(pre):
            baseline_rates = np.nanmean(rates[:, pre], axis=1)
        else:
            baseline_rates = np.full(n_units, np.nan)

        return pd.DataFrame(
            {
                "peak_rate": peak_rates,
                "peak_latency": peak_latencies,
                "baseline_rate": baseline_rates,
            },
            index=pd.Index(np.asarray(self.unit_ids), name="unit_id"),
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of population headline scalars.

        Returns
        -------
        dict
            Mapping with ``n_units`` (int), ``n_events`` (int),
            ``mean_peak_rate`` (Hz, population-average peak firing rate), and
            ``population_peak_latency`` (s, time of the mean histogram peak).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events._core import PopulationPeriEventResult
        >>> bc = np.array([-0.5, 0.0, 0.5])
        >>> hist = np.array([[0.1, 0.4, 0.2], [0.0, 0.1, 0.5]])
        >>> result = PopulationPeriEventResult(
        ...     bin_centers=bc,
        ...     histograms=hist,
        ...     sem=np.zeros_like(hist),
        ...     mean_histogram=hist.mean(axis=0),
        ...     n_events=10,
        ...     n_units=2,
        ...     window=(-0.75, 0.75),
        ...     bin_size=0.5,
        ... )
        >>> sorted(result.summary())
        ['mean_peak_rate', 'n_events', 'n_units', 'population_peak_latency']
        """
        mean_rate = np.asarray(self.mean_firing_rate)
        bin_centers = np.asarray(self.bin_centers)
        if mean_rate.size == 0:
            pop_peak_latency = float("nan")
        else:
            pop_peak_latency = float(bin_centers[int(np.nanargmax(mean_rate))])

        rates = np.asarray(self.firing_rates)
        if rates.size == 0:
            mean_peak_rate = float("nan")
        else:
            mean_peak_rate = float(np.nanmean(np.nanmax(rates, axis=1)))

        return {
            "n_units": int(self.n_units),
            "n_events": int(self.n_events),
            "mean_peak_rate": mean_peak_rate,
            "population_peak_latency": pop_peak_latency,
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the population-average PSTH, returning the axis.

        Builds a :class:`PeriEventResult` from the population-average histogram
        and delegates to :func:`plot_peri_event_histogram`.

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
        mean_sem = np.nanmean(np.asarray(self.sem), axis=0)
        mean_result = PeriEventResult(
            bin_centers=np.asarray(self.bin_centers),
            histogram=np.asarray(self.mean_histogram),
            sem=mean_sem,
            n_events=self.n_events,
            window=self.window,
            bin_size=self.bin_size,
        )
        return plot_peri_event_histogram(mean_result, ax=ax, **kwargs)


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
    >>> ax = plot_peri_event_histogram(
    ...     result, title="Reward response"
    ... )  # doctest: +SKIP
    >>> ax.axvline(
    ...     0, color="k", linestyle="--", alpha=0.5
    ... )  # Mark event  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # Get data to plot
    x = result.bin_centers
    if as_rate:
        y = result.firing_rate
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
