"""Build a ``ripple_detection`` recording from a session and its fetched LFP.

The dataset pages' worked results detect candidate events with
``ripple_detection.literature_methods``, whose ``Recording.from_arrays`` wants
every signal on one time grid at the LFP's rate. :func:`ripple_recording`
bins the spikes and interpolates the speed onto that grid. It chooses no
scientific input: place cells, sleep, baseline, example and external ripples
are arguments, and a method's ``behavior_intervals`` go to ``run_method``.

Loaders never call this: they stay data-only. ``ripple_detection`` is
unreleased, so it is not in the ``datasets`` extra and is imported only here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._lfp import GAP_PERIODS

if TYPE_CHECKING:
    from neurospatial.datasets._recording import Recording

__all__ = ["RIPPLE_DETECTION_COMMIT", "event_rows", "ripple_recording", "spike_counts"]

#: The ``ripple_detection`` commit the worked results were run with; the same
#: one pyproject.toml pins for development.
RIPPLE_DETECTION_COMMIT = "0eddae64a5ebf7b810fb1ec693643b1673650142"
_REPOSITORY = "https://github.com/Eden-Kramer-Lab/ripple_detection"
#: Centimetres per position unit; ``ripple_detection`` takes speed in cm/s.
_CM_PER_UNIT = {"cm": 1.0, "m": 100.0}


def spike_counts(
    spike_times: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    sampling_rate: float,
) -> NDArray[np.unsignedinteger[Any]]:
    """Count each unit's spikes per sample of a time grid.

    A count belongs to its own timestamp, as ``ripple_detection`` reads it:
    sample ``k`` holds the spikes from ``times[k]`` up to the next sample. A
    spike before the first sample, or in a gap (a step over 1.5 sampling
    periods), past the one period that follows the sample, is on no sample and
    is not counted.

    Parameters
    ----------
    spike_times : sequence of ndarray, each shape (n_spikes,)
        Sorted spike times of each unit, seconds.
    times : ndarray, shape (n_time,)
        The grid, seconds, strictly increasing.
    sampling_rate : float
        The grid's rate, Hz.

    Returns
    -------
    ndarray, shape (n_time, n_units)
        Counts in the smallest unsigned integer type that holds them: one byte
        per sample per unit at any ordinary rate.

    Examples
    --------
    >>> import numpy as np
    >>> grid = np.array([0.0, 0.5, 1.0, 5.0, 5.5])
    >>> spike_counts([np.array([0.1, 0.2, 1.2, 3.0, 5.6])], grid, sampling_rate=2.0)[
    ...     :, 0
    ... ].tolist()
    [2, 0, 1, 0, 1]
    """
    period = 1.0 / sampling_rate
    steps = np.diff(times, append=np.inf)
    ends = times + np.where(steps <= GAP_PERIODS * period, steps, period)
    samples, tallies = [], []
    for train in spike_times:
        index = np.searchsorted(times, train, side="right") - 1
        held = index >= 0
        held[held] = train[held] < ends[index[held]]
        sample, tally = np.unique(index[held], return_counts=True)
        samples.append(sample)
        tallies.append(tally)
    largest = max((int(tally.max(initial=0)) for tally in tallies), default=0)
    counts = np.zeros((times.size, len(tallies)), dtype=np.min_scalar_type(largest))
    for unit, (sample, tally) in enumerate(zip(samples, tallies, strict=True)):
        counts[sample, unit] = tally
    return counts


def event_rows(
    spike_times: Sequence[NDArray[np.float64]],
    start: float,
    stop: float,
    *,
    time_bin: float,
    step: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Count an event's spikes in overlapping time bins that reach its end.

    Rows are ``time_bin`` wide and start every ``step`` from ``start``. The last
    row is the first to reach ``stop`` and is cut there, so that every spike in
    ``[start, stop)`` is in a row and none after ``stop`` is. Whole steps alone,
    as ``bin_spikes_in_time`` tiles them, would lose a spike in the last partial
    step, which for an event that ends at its last spike is that spike.

    The cut row is between ``time_bin - step`` and ``time_bin`` wide. It keeps
    its place on the grid and is decoded as a whole row, so a window of ``n``
    full steps gets the same rows as ``bin_spikes_in_time`` summed by hand.

    Parameters
    ----------
    spike_times : sequence of ndarray, each shape (n_spikes,)
        Sorted spike times of each unit, seconds.
    start, stop : float
        The event's half-open ``[start, stop)``, seconds. An event bound that is a
        sample's time holds that sample's spikes only up to the next sample, so
        pass one sample past it.
    time_bin, step : float
        Row width and spacing, seconds.

    Returns
    -------
    counts : ndarray, shape (n_rows, n_units)
        Spikes per row, each spike in as many rows as overlap it.
    centers : ndarray, shape (n_rows,)
        Each row's nominal centre, ``time_bin / 2`` after its left edge, evenly
        spaced even for a cut row.

    Examples
    --------
    >>> import numpy as np
    >>> train = np.array([0.0, 0.035, 0.0449, 0.06])
    >>> counts, centers = event_rows([train], 0.0, 0.045, time_bin=0.02, step=0.01)
    >>> counts[:, 0].tolist()
    [1, 0, 1, 2]
    >>> centers.round(3).tolist()
    [0.01, 0.02, 0.03, 0.04]
    """
    # The tolerance keeps a width that is a whole number of steps from rounding up.
    n_steps = int(np.ceil((stop - start - time_bin) / step - 1e-9))
    left = start + step * np.arange(max(n_steps, 0) + 1)
    right = np.minimum(left + time_bin, stop)
    counts = np.array(
        [
            np.searchsorted(train, right) - np.searchsorted(train, left)
            for train in spike_times
        ],
        dtype=np.int64,
    ).T
    return counts, left + time_bin / 2


def _speed_on_grid(
    rec: Recording, times: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """The session's speed in cm/s at ``times``; NaN where it is unknown.

    Linear between tracking samples, as the session's speed is defined at
    them. NaN outside the tracked span, across a gap over ``rec``'s
    ``max_gap``, and next to a sample without a speed. None when the session
    has no speed in a metric unit.
    """
    speed = rec._speed
    scale = _CM_PER_UNIT.get(str(rec.position_units))
    if speed is None or scale is None:
        return None
    right = np.clip(np.searchsorted(rec.times, times, side="right"), 1, speed.size - 1)
    left = right - 1
    span = rec.times[right] - rec.times[left]
    weight = (times - rec.times[left]) / span
    # A time exactly on a sample takes that sample, whatever its neighbour holds.
    on_grid = np.select(
        [weight == 0.0, weight == 1.0],
        [speed[left], speed[right]],
        (1.0 - weight) * speed[left] + weight * speed[right],
    )
    outside = (times < rec.times[0]) | (times > rec.times[-1])
    if rec._max_gap is not None:
        outside |= span > rec._max_gap
    return np.where(outside, np.nan, on_grid) * scale


def _unit_mask(rec: Recording, unit_ids: ArrayLike | None, name: str) -> Any:
    """``unit_ids`` as a mask over ``rec.units``, in spike-train order."""
    if unit_ids is None:
        return None
    known = rec.units.index
    chosen = np.asarray(unit_ids)
    if missing := [unit for unit in chosen.tolist() if unit not in known]:
        raise InputError(
            _format_error(
                f"{name}: {missing} are not units of {rec.dataset}/{rec.session}.",
                why="a selection is by unit id, so it cannot shift with the "
                "order of the spike trains.",
                fix="pick from rec.units.index.",
            )
        )
    return known.isin(chosen.tolist())


def ripple_recording(
    rec: Recording,
    samples: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    place_cells: ArrayLike | None = None,
    pyramidal: ArrayLike | None = None,
    sleep_intervals: ArrayLike | None = None,
    baseline_intervals: ArrayLike | None = None,
    artifact_intervals: ArrayLike | None = None,
    example_ripples: ArrayLike | None = None,
    external_ripples: ArrayLike | None = None,
    sharp_wave_lfp: ArrayLike | None = None,
    reference_lfp: ArrayLike | None = None,
) -> Any:
    """Build ``ripple_detection``'s recording from a session and its fetched LFP.

    Every unit of ``rec`` is binned onto the LFP's grid (:func:`spike_counts`),
    and ``rec``'s speed is interpolated onto it, in cm/s. Spikes off the grid
    are not counted, so passing a slice of the LFP analyses that slice.

    ``ripple_detection`` keeps one float64 copy of every signal and refuses
    sparse counts: 8 bytes per sample per unit, 5.76 GB for 2 kHz x 1 h x 100
    units, plus 0.72 GB of byte counts while it is built. Pass the slice of
    the LFP the analysis needs when that is too much. Its methods read per-unit
    activity (``require_active_units``), so units cannot be pooled instead.

    Parameters
    ----------
    rec : Recording
        The session; ``rec.lfp`` gives the sampling rate.
    samples : ndarray, shape (n_time, n_channels)
        LFP from ``rec.lfp.fetch``, channels in the order the method wants
        (a single-channel method uses the first).
    times : ndarray, shape (n_time,)
        The LFP's times, seconds on the recording's clock.
    place_cells, pyramidal : array_like of unit ids, optional
        Selections from ``rec.units.index``, chosen by the caller; a method
        that needs one and has none refuses.
    sleep_intervals, baseline_intervals, artifact_intervals : array_like, optional
        Sorted, disjoint [start, end] rows, seconds.
    example_ripples : array_like, optional
        Hand-selected [start, end] rows, for Carey's spectral template.
    external_ripples : array_like, optional
        Independently detected [start, end] or [start, end, peak] rows.
    sharp_wave_lfp, reference_lfp : array_like, shape (n_time,), optional
        The radiatum and reference LFP, on ``times``.

    Returns
    -------
    ripple_detection.literature_methods.Recording
        Give it to ``run_method(name, recording, behavior_intervals=...)``.
        Speed is None without positions or in pixel or arbitrary units; a
        method that needs speed then refuses.

    Raises
    ------
    ImportError
        If ``ripple_detection`` is not installed.
    InputError
        If ``rec`` has no LFP source, or a selection names an unknown unit.
    ValueError
        If ``ripple_detection`` refuses an input.

    Examples
    --------
    >>> from neurospatial.datasets import load
    >>> from ripple_detection.literature_methods import (  # doctest: +SKIP
    ...     run_method,
    ... )
    >>> rec = load("carey_2019")  # doctest: +SKIP
    >>> samples, times = rec.lfp.fetch(rec.lfp.ripple_channels)  # doctest: +SKIP
    >>> recording = ripple_recording(  # doctest: +SKIP
    ...     rec, samples, times, example_ripples=rec.events["hand_labelled"]
    ... )
    >>> events = run_method("carey_2019", recording)  # doctest: +SKIP
    """
    try:
        from ripple_detection.literature_methods import Recording as Signals
    except ModuleNotFoundError as exc:
        if exc.name != "ripple_detection":
            raise
        raise ImportError(
            _format_error(
                "ripple_detection is required to detect candidate events but "
                "is not installed.",
                why="it is unreleased, so the datasets extra cannot include it.",
                fix=f"pip install 'ripple_detection @ git+{_REPOSITORY}"
                f"@{RIPPLE_DETECTION_COMMIT}'",
            )
        ) from exc
    if rec.lfp is None:
        raise InputError(
            _format_error(
                f"{rec.dataset}/{rec.session} has no LFP source.",
                why="the spikes and speed are put on the LFP's time grid.",
                fix="use a session whose rec.lfp is not None.",
            )
        )
    rate = rec.lfp.sampling_rate
    return Signals.from_arrays(
        times,
        rate,
        lfps=samples,
        multiunit=spike_counts(rec.spike_times, times, sampling_rate=rate),
        speed=_speed_on_grid(rec, times),
        place_cells=_unit_mask(rec, place_cells, "place_cells"),
        pyramidal=_unit_mask(rec, pyramidal, "pyramidal"),
        sleep_intervals=sleep_intervals,
        baseline_intervals=baseline_intervals,
        artifact_intervals=artifact_intervals,
        example_ripples=example_ripples,
        external_ripples=external_ripples,
        sharp_wave_lfp=sharp_wave_lfp,
        reference_lfp=reference_lfp,
    )
