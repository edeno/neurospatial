"""Firing aligned to the distance travelled from an event.

A peri-event time histogram asks how firing changes in the seconds around an
event. Many spatial questions are about distance instead: how firing changes
over the centimetres after the animal passes a cue, enters a reward zone or
leaves the start box. :func:`peri_event_distance_histogram` bins spikes by
the path length travelled from each event and divides by the time spent in
each distance bin, so the result is a firing rate in Hz, like a rate map on
the coordinate "distance from the event", and does not depend on how fast
the animal ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    intersect_intervals,
    linear_allocated_occupancy,
    value_frame_bins,
)
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.events._core import (
    _finite_times,
    _make_bin_edges,
    empty_selection_names,
    warn_empty_selection,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

__all__ = ["PeriEventDistanceResult", "peri_event_distance_histogram"]


@dataclass(frozen=True, repr=False, eq=False)
class PeriEventDistanceResult(ResultMixin, ReadOnlyArraysMixin):
    """Firing rate by distance travelled from an event.

    Built by :func:`peri_event_distance_histogram`.

    Attributes
    ----------
    spike_counts : NDArray[np.float64], shape (n_events, n_bins)
        Spikes of each event's path in each distance bin.
    occupancy : NDArray[np.float64], shape (n_events, n_bins)
        Seconds each event's path spent in each distance bin; 0 where the
        path did not reach the bin.
    window : tuple[float, float]
        Distance range (start, stop) relative to each event, in the units of
        the positions; negative before the event.
    bin_size : float
        Width of the distance bins, in the units of the positions.
    unit_id : int, str or None, default=None
        The unit the spikes came from.
    n_events_dropped : int, default=0
        Events at a time with no tracked path (in a tracking gap, outside
        ``epochs`` or ``spike_window``), which have no distance origin.
    event_indices : NDArray[np.intp] or None, default=None
        Positions of the kept events in the input ``event_times``.
    bin_centers : NDArray[np.float64], shape (n_bins,)
        Distance from the event at each bin's center (derived).
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spikes over seconds in each bin, pooled over events, in Hz (derived).
        NaN where no event's path reached the bin: no time, no rate.
    """

    spike_counts: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    window: tuple[float, float]
    bin_size: float
    unit_id: int | str | None = None
    n_events_dropped: int = 0
    event_indices: NDArray[np.intp] | None = None

    def __post_init__(self) -> None:
        counts = np.asarray(self.spike_counts, dtype=np.float64)
        occupancy = np.asarray(self.occupancy, dtype=np.float64)
        n_bins = len(_make_bin_edges(self.window, self.bin_size)) - 1
        if (
            counts.ndim != 2
            or counts.shape != occupancy.shape
            or counts.shape[1] != n_bins
        ):
            raise ValueError(
                _format_error(
                    f"PeriEventDistanceResult: spike_counts {counts.shape} and "
                    f"occupancy {occupancy.shape} must both be (n_events, {n_bins}).",
                    why="each event has one count and one dwell time per distance bin.",
                    fix="build the result with peri_event_distance_histogram().",
                )
            )
        object.__setattr__(self, "spike_counts", counts)
        object.__setattr__(self, "occupancy", occupancy)
        self._own_arrays()

    @property
    def n_events(self) -> int:
        """Number of events aligned."""
        return int(self.spike_counts.shape[0])

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Distance from the event at each bin's center, shape ``(n_bins,)``."""
        edges = _make_bin_edges(self.window, self.bin_size)
        return (edges[:-1] + edges[1:]) / 2

    @property
    def firing_rate(self) -> NDArray[np.float64]:
        """Pooled spikes over seconds in each bin (Hz), shape ``(n_bins,)``."""
        return _pooled_rate(self.spike_counts.sum(axis=0), self.occupancy.sum(axis=0))

    @property
    def n_events_covering(self) -> NDArray[np.intp]:
        """Events whose path spent time in each bin, shape ``(n_bins,)``."""
        return np.asarray((self.occupancy > 0).sum(axis=0), dtype=np.intp)

    def _attrs(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "bin_size": self.bin_size,
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table: one row per distance bin.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin_center`` (distance from the event),
            ``firing_rate`` (Hz), ``spike_count`` and ``occupancy`` (s) summed
            over events, and ``n_events_covering``. ``df.attrs`` holds
            ``window``, ``bin_size``, ``n_events`` and ``n_events_dropped``.
        """
        import pandas as pd

        df = pd.DataFrame(
            {
                "unit_id": np.full(self.bin_centers.size, self.unit_id, dtype=object),
                "bin_center": self.bin_centers,
                "firing_rate": self.firing_rate,
                "spike_count": self.spike_counts.sum(axis=0),
                "occupancy": self.occupancy.sum(axis=0),
                "n_events_covering": self.n_events_covering,
            }
        )
        df.attrs = self._attrs()
        return df

    def summary_table(self) -> pd.DataFrame:
        """Scalar response metrics in one row, indexed by ``unit_id``.

        Returns
        -------
        pandas.DataFrame
            ``peak_rate`` (Hz), ``peak_distance`` (center of the bin with the
            highest rate; NaN when no bin has a rate) and ``baseline_rate``
            (Hz, spikes over seconds pooled over the bins that end at or
            before the event; NaN when there are none or they hold no time).
        """
        import pandas as pd

        rate = self.firing_rate
        defined = np.isfinite(rate)
        if defined.any():
            peak = int(np.nanargmax(rate))
            peak_rate, peak_distance = float(rate[peak]), float(self.bin_centers[peak])
        else:
            peak_rate = peak_distance = np.nan
        edges = _make_bin_edges(self.window, self.bin_size)
        before = edges[1:] <= 0
        baseline = float(
            _pooled_rate(
                self.spike_counts[:, before].sum(), self.occupancy[:, before].sum()
            )
        )
        df = pd.DataFrame(
            {
                "peak_rate": [peak_rate],
                "peak_distance": [peak_distance],
                "baseline_rate": [baseline],
            },
            index=pd.Index([self.unit_id], name="unit_id"),
        )
        df.attrs = self._attrs()
        return df

    def summary(self) -> dict[str, Any]:
        """Flat dict of headline scalars.

        Returns
        -------
        dict
            ``unit_id``, ``n_events``, ``n_events_dropped``, ``peak_rate``,
            ``peak_distance``, ``baseline_rate`` and ``total_occupancy`` (s).
        """
        row = self.summary_table().iloc[0]
        return {
            "unit_id": self.unit_id,
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
            "peak_rate": float(row["peak_rate"]),
            "peak_distance": float(row["peak_distance"]),
            "baseline_rate": float(row["baseline_rate"]),
            "total_occupancy": float(self.occupancy.sum()),
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot firing rate against distance from the event.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.bin_centers, self.firing_rate, **kwargs)
        ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Distance from event")
        ax.set_ylabel("Firing rate (Hz)")
        return ax


def _pooled_rate(spikes: Any, seconds: Any) -> NDArray[np.float64]:
    """Spikes over seconds; NaN where no time was spent (no rate, not zero)."""
    spikes = np.asarray(spikes, dtype=np.float64)
    seconds = np.asarray(seconds, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            seconds > 0, spikes / np.where(seconds > 0, seconds, 1.0), np.nan
        )


def _validated_distance_window(window: Any, bin_size: float) -> tuple[float, float]:
    """A finite ``(start, stop)`` distance window holding at least one bin."""
    fix = (
        "pass window=(-50.0, 50.0) and bin_size=5.0 for 50 cm either side in 5 cm bins"
    )
    try:
        bounds = np.asarray(window, dtype=np.float64)
    except (TypeError, ValueError):
        bounds = np.array([np.nan])
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
    ):
        raise ValueError(
            _format_error(
                f"peri_event_distance_histogram: window must be two finite "
                f"distances (start, stop) with start < stop, got {window!r}.",
                why="the window is the half-open distance range [start, stop) "
                "travelled from each event, negative before it.",
                fix=fix,
            )
        )
    if not (np.isfinite(bin_size) and 0 < bin_size <= bounds[1] - bounds[0]):
        raise ValueError(
            _format_error(
                f"peri_event_distance_histogram: bin_size={bin_size!r} must be a "
                f"positive distance no wider than the window.",
                why="the window is split into bins of this width.",
                fix=fix,
            )
        )
    return float(bounds[0]), float(bounds[1])


@reads_pynapple_samples
def peri_event_distance_histogram(
    spike_times: NDArray[np.float64],
    event_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    window: tuple[float, float],
    *,
    bin_size: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: Any = None,
    spike_window: Any = None,
    min_speed: float | None = None,
) -> PeriEventDistanceResult:
    """Firing rate by distance travelled from each event.

    For each event, distance is the path length travelled from where the
    animal was at the event, along its tracked path: negative before the
    event, positive after. Spikes are binned by that distance, and each bin's
    count is divided by the time the path spent in the bin, pooled over
    events, giving a firing rate in Hz. A slow pass and a fast pass over the
    same distance therefore give the same rate, and a pause adds time at one
    distance.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        One unit's spike times, in seconds.
    event_times : NDArray[np.float64], shape (n_events,)
        Event times, in seconds (for example cue passes or zone entries).
        For trials, pass ``trials.start`` or ``trials.stop``.
    times : NDArray[np.float64], shape (n_samples,)
        Position sample times, in seconds, strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Positions at ``times``; NaN rows are missing samples.
    window : tuple[float, float]
        Distance range ``[start, stop)`` from each event, in the units of
        ``positions``; for example ``(-50.0, 50.0)`` for 50 cm before and
        after.
    bin_size : float
        Width of the distance bins, in the units of ``positions``. A trailing
        partial bin is dropped.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge. A longer
        gap ends the path: distance is not measured across it, so an event's
        distance axis stops there.
    epochs : (start, stop), array-like of shape (n, 2), Intervals, IntervalSet, \
or None
        Analyze only these half-open time windows (seconds). When
        ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``. An :class:`~neurospatial.Intervals` with no windows
        (a condition with no trials) is an empty selection: the result has no
        events and a NaN rate, and a warning says so.
    break_before : array-like of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break before it (for example a new
        recording); distance does not continue across it.
    spike_window : same forms as ``epochs``, or None
        When the spikes were recorded. Time outside it is not counted, so it
        is never read as silence.
    min_speed : float or None, default=None
        Speed, in position units per second, below which a step between
        samples adds no distance, so a stop stays at one distance. Use it on
        smoothed positions: raw tracking jitter alone moves faster than a
        few cm/s at video rates. None keeps every step.

    Returns
    -------
    PeriEventDistanceResult
        Spike counts and seconds per event and distance bin, and the pooled
        ``firing_rate`` (Hz); NaN in a bin no event's path reached.

    Raises
    ------
    ValueError
        If ``window`` or ``bin_size`` is invalid, ``min_speed`` is negative
        or not finite, ``event_times`` is empty or an ``Intervals``, or no
        event falls on tracked time inside windows with duration.

    Warns
    -----
    UserWarning
        When ``epochs`` or ``spike_window`` is an empty selection.

    Notes
    -----
    Between two tracked samples the animal is taken to move at constant
    speed, so an interval's time is spread over the distance bins it crosses
    in proportion to the distance covered in each; an interval without
    movement puts all its time at one distance. Distance is the length of the
    straight segments between the positions as given, so tracking jitter
    adds distance: with 0.5 cm of jitter, a 5 s stop at a reward adds about
    130 cm at 30 Hz (580 cm at 125 Hz). Smooth raw tracking first with
    :func:`~neurospatial.behavior.smooth_positions`, which brings the stop
    to a few cm.

    A spike counts for every event whose path it falls on, so with events
    closer together than the window, a spike can count for two events, as
    in a peri-event time histogram.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.events import peri_event_distance_histogram
    >>> times = np.arange(0.0, 10.0, 0.1)
    >>> positions = (times * 10.0)[:, None]  # 10 cm/s along a line
    >>> result = peri_event_distance_histogram(
    ...     np.array([5.2, 5.4]),
    ...     np.array([5.0]),
    ...     times,
    ...     positions,
    ...     (-20.0, 20.0),
    ...     bin_size=10.0,
    ... )
    >>> result.firing_rate.round(2).tolist()  # two spikes 2 and 4 cm after it
    [0.0, 0.0, 2.0, 0.0]
    """
    from neurospatial._interval_table import Intervals
    from neurospatial.behavior.trajectory import compute_cumulative_distance

    window = _validated_distance_window(window, bin_size)
    if min_speed is not None and not (np.isfinite(min_speed) and min_speed >= 0):
        raise ValueError(
            _format_error(
                f"peri_event_distance_histogram: min_speed={min_speed!r} must be "
                "a finite speed >= 0.",
                why="min_speed is the speed, in position units per second, below "
                "which a step adds no distance.",
                fix="pass min_speed=None, or a speed such as min_speed=2.0 for 2 cm/s.",
            )
        )
    if isinstance(cast("object", event_times), Intervals):
        raise ValueError(
            _format_error(
                "peri_event_distance_histogram: event_times is an Intervals; it "
                "takes event times in seconds.",
                why="distance is measured from one time per event, and an "
                "interval has two (its start and its stop).",
                fix="pass trials.start to measure from each start, or "
                "trials.stop to measure from each stop",
            )
        )
    edges = _make_bin_edges(window, bin_size)
    spikes = np.sort(_finite_times(spike_times, "spike_times"))
    events = _finite_times(event_times, "event_times", non_empty=True)
    analyzed = as_intervals(epochs, name="epochs")
    recorded = as_intervals(spike_window, name="spike_window")
    empty = empty_selection_names(analyzed, recorded)
    if empty:
        # Validate the recording as any call does, so the result never
        # depends on the selection being empty.
        compute_cumulative_distance(
            times, positions, max_gap=max_gap, break_before=break_before
        )
        warn_empty_selection(empty, n_events=events.size, result="the rate")
        no_events = np.zeros((0, edges.size - 1), dtype=np.float64)
        return PeriEventDistanceResult(
            spike_counts=no_events,
            occupancy=no_events,
            window=window,
            bin_size=float(bin_size),
            n_events_dropped=events.size,
            event_indices=np.empty(0, dtype=np.intp),
        )
    if analyzed is None:
        analyzed = recorded
    elif recorded is not None:
        analyzed = intersect_intervals(analyzed, recorded)
    path = compute_cumulative_distance(
        times,
        positions,
        max_gap=max_gap,
        epochs=analyzed,
        break_before=break_before,
    )
    times = np.asarray(times, dtype=np.float64)
    n_bins = edges.size - 1
    paths: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    counts, occupancy, kept = [], [], []
    for index, event in enumerate(events):
        run = _run_at(path.run_time_bounds, event)
        if run < 0:
            continue
        if run not in paths:
            samples = np.flatnonzero(path.run_ids == run)
            paths[run] = _run_path_length(
                times[samples], path.run_distance[samples], min_speed
            )
        t, s = paths[run]
        distance = s - np.interp(event, t, s)
        occupancy.append(
            linear_allocated_occupancy(
                distance, np.diff(t), edges, np.ones(t.size - 1, dtype=bool)
            )
        )
        # Half-open [first, last) sample times: the last sample ends the path.
        in_run = spikes[(spikes >= t[0]) & (spikes < t[-1])]
        # Half-open bins [left, right): a spike at the window stop is left out.
        spike_bin = value_frame_bins(
            np.interp(in_run, t, distance), edges, closed_last=False
        )
        counts.append(np.bincount(spike_bin[spike_bin >= 0], minlength=n_bins))
        kept.append(index)
    if not kept:
        raise ValueError(
            _format_error(
                f"peri_event_distance_histogram: none of the {events.size} events "
                "falls on tracked time.",
                why="an event's distance axis starts where the animal was at the "
                "event, so the event must lie within a tracked run of positions "
                "(and inside epochs and spike_window).",
                fix="check that event_times and times share one clock in seconds, "
                "or widen max_gap if the events fall in short tracking gaps.",
            )
        )
    return PeriEventDistanceResult(
        spike_counts=np.asarray(counts, dtype=np.float64),
        occupancy=np.asarray(occupancy, dtype=np.float64),
        window=window,
        bin_size=float(bin_size),
        n_events_dropped=events.size - len(kept),
        event_indices=np.asarray(kept, dtype=np.intp),
    )


def _run_at(run_time_bounds: NDArray[np.float64], time: float) -> int:
    """The run whose ``[start, end]`` holds ``time``, or -1."""
    inside = np.flatnonzero(
        (run_time_bounds[:, 0] <= time) & (time <= run_time_bounds[:, 1])
    )
    return int(inside[0]) if inside.size else -1


def _run_path_length(
    t: NDArray[np.float64],
    run_distance: NDArray[np.float64],
    min_speed: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Times and path length from the start of one tracked run.

    ``run_distance`` is already measured on the smoothed path; a step slower
    than ``min_speed`` adds nothing.
    """
    if min_speed is None:
        return t, run_distance
    steps = np.diff(run_distance)
    steps = np.where(steps / np.diff(t) < min_speed, 0.0, steps)
    return t, np.concatenate([[0.0], np.cumsum(steps)])
