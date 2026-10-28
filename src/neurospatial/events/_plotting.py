"""Plots that combine a peri-event result with caller-owned raw inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.events._core import PeriEventResult, _make_bin_edges
from neurospatial.events.alignment import _peri_event_bin_edges, _sorted_times

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_peri_event_response(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    result: PeriEventResult,
    *,
    axes: tuple[Axes, Axes] | None = None,
    show_sem: bool = True,
    as_rate: bool = True,
    color: str = "C0",
    raster_color: str = "black",
) -> tuple[Axes, Axes]:
    """Plot a matching raster and histogram for one unit's peri-event result.

    The raster reuses ``result.event_indices`` and the histogram's actual
    half-open bin range. No events are selected again and no histogram is
    recomputed. Raw inputs are supplied explicitly and are not stored on the
    result. The lower panel delegates to :meth:`PeriEventResult.plot`.

    Parameters
    ----------
    spike_times : ndarray of shape (n_spikes,)
        Spike times in seconds for the unit represented by ``result``. Unsorted
        times and an empty train are supported. A plain array has no unit
        identity metadata: the caller must supply the matching train.
    event_times : ndarray of shape (n_input_events,)
        The complete original event array, in its original order and on the
        same clock as ``spike_times``. Include the events dropped by the
        histogram's exposure filter. Duplicate timestamps remain distinct rows.
    result : PeriEventResult
        A single-unit result with known ``event_indices``, returned by
        ``peri_event_histogram`` or selected from a population with
        ``population.sel(unit_id=...)``. Select the corresponding spike train
        by the same unit label before calling this function.
    axes : pair of matplotlib.axes.Axes, optional
        Raster and histogram axes, in that order. If None, create two axes with
        a shared x-axis. Supplied axes are reused and given matching x-limits.
    show_sem : bool, default=True
        Show the histogram's standard error across retained events. With a
        single event it is NaN, so there is no finite uncertainty band.
    as_rate : bool, default=True
        Plot the histogram and SEM in Hz; False uses spike counts per bin.
    color : str, default="C0"
        Histogram line and SEM color.
    raster_color : str, default="black"
        Raster spike color.

    Returns
    -------
    raster_ax, histogram_ax : tuple of matplotlib.axes.Axes
        Axes for further composition; access the figure via ``raster_ax.figure``.
        Raster row ``i`` corresponds to original event
        ``event_times[result.event_indices[i]]``, including rows with no spikes.

    Raises
    ------
    TypeError
        If result is not a single-unit PeriEventResult.
    ValueError
        If cohort provenance is unknown, event-array length is incompatible,
        times are not finite 1-D arrays, no complete histogram bin exists, bin
        metadata is inconsistent, or axes is not a pair of distinct axes.

    Notes
    -----
    For a window not divisible by ``result.bin_size``, the histogram drops the
    trailing partial bin. This plot excludes that tail from the raster and both
    x-limits as well. Event inclusion still follows the original result's
    full-window exposure selection.

    Spike inclusion is determined on the absolute clock before subtracting each
    event time. This preserves exact boundary membership at large timestamps.
    Display offsets are clipped to the plotted range if subtraction rounding
    would otherwise put an included boundary spike outside the axes.
    The raster and the histogram use the same half-open ``[start, stop)``
    window as ``align_spikes_to_events``.

    Shape checks cannot prove that caller-supplied arrays contain the original
    values. Pass the original event array and the spikes used for this unit's
    histogram, without changing their clock or reordering events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import (
    ...     peri_event_histogram,
    ...     plot_peri_event_response,
    ... )
    >>> spike_times = np.array([0.9, 1.1, 2.9, 3.1])
    >>> event_times = np.array([1.0, 3.0])
    >>> result = peri_event_histogram(
    ...     spike_times,
    ...     event_times,
    ...     (-0.5, 0.5),
    ...     bin_size=0.1,
    ... )
    >>> axes = plot_peri_event_response(spike_times, event_times, result)
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if not isinstance(result, PeriEventResult):
        raise TypeError(
            "result must be a single-unit PeriEventResult. "
            "Select a population unit with .sel(unit_id=...) and pass that unit's spikes."
        )
    if result.event_indices is None:
        raise ValueError(
            "result.event_indices is unknown; a matching raster cohort cannot be inferred. "
            "Compute the result with peri_event_histogram or population_peri_event_histogram."
        )
    spike_times = np.asarray(spike_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    for name, values in (("spike_times", spike_times), ("event_times", event_times)):
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must be a finite 1-D array of times in seconds. "
                "Pass one unit's spike train and the complete original event array."
            )
    n_input_events = result.n_events + result.n_events_dropped
    if event_times.size != n_input_events:
        raise ValueError(
            f"event_times has {event_times.size} events; expected {n_input_events} "
            "(n_events + n_events_dropped). Pass the complete original event array, "
            "including excluded events, in its original order."
        )
    bin_edges = _make_bin_edges(result.window, result.bin_size)
    if axes is not None:
        try:
            raster_ax, histogram_ax = axes
        except (TypeError, ValueError) as error:
            raise ValueError(
                "axes must contain two distinct Matplotlib Axes."
            ) from error
        if (
            not isinstance(raster_ax, Axes)
            or not isinstance(histogram_ax, Axes)
            or raster_ax is histogram_ax
        ):
            raise ValueError("axes must contain two distinct Matplotlib Axes.")

    selected_events = event_times[result.event_indices]
    spikes = _sorted_times(spike_times)
    edges = _peri_event_bin_edges(
        selected_events, result.window, bin_edges, result.bin_size
    )
    first = np.searchsorted(spikes, edges[:, 0], side="left")
    stop = np.searchsorted(spikes, edges[:, -1], side="left")
    below_stop = np.nextafter(bin_edges[-1], bin_edges[0])
    aligned = [
        np.clip(spikes[i:j] - event, bin_edges[0], below_stop)
        for i, j, event in zip(first, stop, selected_events, strict=True)
    ]

    if axes is None:
        _, (raster_ax, histogram_ax) = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
            constrained_layout=True,
        )
    raster_ax.eventplot(aligned, colors=raster_color, linelengths=0.8, linewidths=0.8)
    raster_ax.set_ylabel("Retained event row")
    raster_ax.set_ylim(-0.5, result.n_events - 0.5)
    unit_label = "" if result.unit_id is None else f"Unit {result.unit_id}: "
    raster_ax.set_title(f"{unit_label}{result.n_events} retained events")
    raster_ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    result.plot(ax=histogram_ax, show_sem=show_sem, as_rate=as_rate, color=color)
    for ax in (raster_ax, histogram_ax):
        ax.set_xlim(bin_edges[0], bin_edges[-1])
    return raster_ax, histogram_ax
