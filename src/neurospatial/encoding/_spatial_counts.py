"""Binning layer for spatial encoding.

This module converts spike trains and trajectories into discrete spike counts
and occupancy arrays that can be processed by the smoothing layer.

The functions in this module handle:
1. Assigning each spike to the bin that receives its sampling interval's
   dwell time
2. Occupancy computation from continuous trajectory to time-per-bin
3. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons

The binning layer is intentionally separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing position bins for efficiency
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import DISCARD_WARN_FRACTION
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    IntervalGates,
    TimeSupport,
    excluded_intervals_message,
    fill_missing_positions,
    hold_last_valid,
    intersect_intervals,
    interval_gates,
    interval_valid_mask,
    resolve_time_windows,
    run_time_bounds,
    start_allocated_occupancy,
    start_allocation,
    times_in_intervals,
)
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._velocity import resolve_speed
from neurospatial.geometry import require_cartesian, require_position_columns
from neurospatial.ops import (
    _polygon_linear_occupancy,
    _RetainedLinearInterval,
    _uses_polygon_allocation,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "bin_spike_trains",
    "compute_occupancy",
]


@dataclass(frozen=True)
class LinearSpatialIntervals:
    """Owned fractional support prepared once for a spatial encoding call.

    ``gates`` is the observation mask with the gates that built it, for the
    excluded-interval warning; None when it was prepared from a supplied
    ``interval_mask``, which its resolver already reported.
    """

    support: TimeSupport
    pieces: tuple[_RetainedLinearInterval, ...]
    occupancy: NDArray[np.float64]
    gates: IntervalGates | None


def _validate_time_allocation(time_allocation: str) -> None:
    if time_allocation not in ("start", "linear"):
        msg = "time_allocation must be 'start' or 'linear'."
        raise ValueError(msg)


def prepare_linear_spatial_intervals(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    interval_mask: NDArray[np.bool_] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> LinearSpatialIntervals:
    """Resolve exact planar dwell and half-open spike support once.

    Complete observation intervals pass the gap, speed and recording-window
    gates before their straight paths are subdivided by cells and Arena
    support. Finite endpoints may lie outside the environment. Retained
    fractions preserve the original clock and are never renormalized.

    Parameters
    ----------
    env : Environment
        Environment publishing exact planar Cartesian area cells.
    times : ndarray, shape (n_samples,)
        Observation timestamps in seconds, sorted and finite.
    positions : ndarray, shape (n_samples, 2)
        Straight segment endpoints, which may lie outside the environment.
    speed : ndarray, shape (n_samples,), or None
        Resolved speed used with ``min_speed`` to gate complete intervals.
    min_speed, max_gap : float or None
        Speed and duration gates applied before geometric subdivision.
    epochs, spike_window : ndarray of shape (n_windows, 2), or None
        Resolved windows (:func:`~neurospatial._intervals.resolve_time_windows`)
        that must contain each complete observation interval.
    interval_mask : ndarray of bool, shape (n_samples - 1,), or None
        Optional precomputed observation validity, before geometric bounds.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks; recorded in the support
        and, without ``interval_mask``, applied as a gate.

    Returns
    -------
    LinearSpatialIntervals
        Immutable pieces, owned read-only occupancy, and temporal support
        whose runs describe the exact retained fractional intervals.

    Raises
    ------
    CartesianOnlyError
        If the environment is polar.
    NotImplementedError
        If the environment cannot publish exact planar cell and Arena support.
    """
    require_linear_allocation(env)
    binning, arena = env.binning, env.arena
    if env.n_dims != 2 or not binning.bins.has_measure_dim(2):
        msg = (
            "time_allocation='linear' spatial encoding requires exact planar "
            "Cartesian area cells. Use time_allocation='start' for this environment."
        )
        raise NotImplementedError(msg)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(times), 2):
        msg = "linear spatial encoding needs (n_samples, 2) positions."
        raise ValueError(msg)
    # This checks known nonmaterializable Arena support before any cell work.
    _uses_polygon_allocation(binning, arena)
    gates: IntervalGates | None = None
    if interval_mask is None:
        gates = interval_gates(
            times,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            spike_window=spike_window,
            break_before=break_before,
            sample_valid=np.asarray(np.isfinite(positions).all(axis=1), dtype=bool),
        )
        observed = gates.mask
    else:
        observed = np.asarray(interval_mask, dtype=bool)
        if observed.shape != (max(len(times) - 1, 0),):
            msg = "interval_mask must contain one value per observation interval."
            raise ValueError(msg)
    # A bridged gap is one straight segment between its valid samples.
    positions = fill_missing_positions(times, positions, time_allocation="linear")
    finite = np.all(np.isfinite(positions), axis=1)
    keep = finite[:-1] & finite[1:]
    mask = observed & keep
    if gates is not None:
        gates = gates.narrowed(keep)
    pieces, occupancy = _polygon_linear_occupancy(
        binning, arena, times, positions, mask, return_seconds=True
    )
    runs: list[list[float]] = []
    for piece in pieces:
        index = piece.sample_index
        start, stop = times[index], times[index + 1]
        duration = stop - start
        left = (
            float(start)
            if piece.start_fraction == 0
            else float(start + duration * piece.start_fraction)
        )
        right = (
            float(stop)
            if piece.stop_fraction == 1
            else float(start + duration * piece.stop_fraction)
        )
        if right <= left:
            msg = (
                "Linear spatial support is shorter than this timestamp clock "
                "can represent. Use elapsed timestamps (subtract the recording "
                "origin from both trajectory and spike times) or coarser "
                "spatial binning."
            )
            raise ValueError(msg)
        if runs and left <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], right)
        else:
            runs.append([left, right])
    occupancy.setflags(write=False)
    support = TimeSupport.from_mask(
        times,
        mask,
        epochs=epochs,
        spike_window=spike_window,
        max_gap=max_gap,
        break_before=break_before,
        runs=np.asarray(runs, dtype=np.float64).reshape(-1, 2),
    )
    return LinearSpatialIntervals(support, pieces, occupancy, gates)


def require_linear_allocation(env: Environment) -> None:
    """Refuse a polar environment for linear time allocation.

    Parameters
    ----------
    env : Environment
        The environment the paths are split in.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    """
    require_cartesian(
        env.coordinate_system,
        "time_allocation='linear'",
        why="It splits each straight Cartesian path between the cells it crosses.",
        fix="compute rates on the allocentric environment the positions were "
        "recorded in",
    )


def require_allocentric(env: Environment, operation: str) -> None:
    """Refuse a polar environment for rates estimated from the animal's positions.

    Parameters
    ----------
    env : Environment
        The environment the rates are binned in.
    operation : str
        The refused call, e.g. ``"compute_spatial_rate()"``.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    """
    require_cartesian(
        env.coordinate_system,
        operation,
        why="It reads positions as the animal's allocentric positions: spike "
        "positions are interpolated between samples along each axis, which "
        "does not follow an angle across its seam.",
        fix="estimate rates on the allocentric environment the positions were "
        "recorded in; for egocentric (distance, angle) maps use "
        "neurospatial.encoding.compute_egocentric_rate",
    )


class _SpikeTrack(NamedTuple):
    """Trajectory lookups that place every unit's spikes, built once per batch."""

    times: NDArray[np.float64]  # (n_samples,) seconds
    t_min: float  # first trajectory timestamp
    t_max: float  # last trajectory timestamp
    interval_mask: NDArray[np.bool_]  # (n_samples - 1,) shared counts/dwell gate
    frame_bins: NDArray[np.intp] | None  # (n_samples,) "start" allocation
    linear_intervals: LinearSpatialIntervals | None  # "linear" allocation
    observed_times: NDArray[np.float64] | None  # (n_observed,) "linear"
    observed_coordinates: NDArray[np.float64] | None  # (n_dims, n_observed)


def _start_frame_bins(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Bin of each frame under ``"start"`` allocation, shape (n_samples,).

    A missing frame holds the last valid frame's bin
    (:func:`~neurospatial._intervals.hold_last_valid`), so a bridged gap's
    dwell time and spikes go to the bin where it started.
    """
    finite = np.asarray(np.isfinite(positions).all(axis=1), dtype=bool)
    return hold_last_valid(env.bin_at(positions), finite, fill=-1)


def _spike_track(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    interval_mask: NDArray[np.bool_],
    linear_intervals: LinearSpatialIntervals | None,
) -> _SpikeTrack:
    """Prepare the trajectory lookups shared by every unit's spike counts.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : ndarray, shape (n_samples,)
        Validated trajectory timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Trajectory coordinates, already reshaped to two dimensions.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Shared gap, speed, bounds and time-window validity.
    linear_intervals : LinearSpatialIntervals or None
        Prepared ``time_allocation="linear"`` intervals. ``None`` means
        ``"start"`` allocation.
    """
    if linear_intervals is None:
        return _SpikeTrack(
            times,
            times.min(),
            times.max(),
            interval_mask,
            _start_frame_bins(env, positions),
            None,
            None,
            None,
        )
    observed = np.isfinite(positions).all(axis=1)
    return _SpikeTrack(
        times,
        times.min(),
        times.max(),
        interval_mask,
        None,
        linear_intervals,
        times[observed],
        np.ascontiguousarray(positions[observed].T),
    )


def _track_occupancy(track: _SpikeTrack, n_bins: int) -> NDArray[np.float64]:
    """Seconds in each bin on the track's shared support, shape (n_bins,)."""
    if track.linear_intervals is not None:
        return np.array(track.linear_intervals.occupancy, dtype=np.float64, copy=True)
    assert track.frame_bins is not None
    return start_allocated_occupancy(
        track.frame_bins, np.diff(track.times), track.interval_mask, n_bins
    )


def _bin_spike_train_with_stats(
    env: Environment,
    spike_times: NDArray[np.float64],
    track: _SpikeTrack,
) -> tuple[
    NDArray[np.float64],  # spike_counts, shape (n_bins,)
    int,  # n_time_dropped
    int,  # n_bin_dropped
    int,  # n_total spikes
    int,  # n_after_time  (spikes surviving the time-window filter)
]:
    """Bin spikes using the caller's shared occupancy-interval mask.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in seconds, already cast to float64.
    track : _SpikeTrack
        Trajectory lookups from :func:`_spike_track`. Its ``interval_mask``
        gates occupancy and every unit's spike counts alike. A spike belongs
        to interval ``searchsorted(times, t, side="right") - 1``; a spike at
        ``times[-1]`` belongs to no interval. Exclusions by this mask are
        intentional and do not count toward inactive-bin drop warnings.

    Returns
    -------
    spike_counts : ndarray, shape (n_bins,)
        Counts after time and interval gates. Under ``"start"`` allocation a
        spike takes the bin of its interval's start frame, the same bin that
        receives that interval's dwell time. Under ``"linear"`` allocation it
        takes the bin of its interpolated position, matching the dwell time
        split along the segment.
    n_time_dropped : int
        Spikes outside ``[times[0], times[-1])``.
    n_bin_dropped : int
        Interval-valid spikes whose position is outside every bin.
    n_total : int
        Input spike count.
    n_after_time : int
        Spikes surviving the time and interval filters.
    """
    n_bins = env.n_bins
    spike_counts = np.zeros(n_bins, dtype=np.float64)
    n_total = len(spike_times)

    if n_total == 0:
        return spike_counts, 0, 0, 0, 0

    valid_time_mask = (spike_times >= track.t_min) & (spike_times < track.t_max)
    spike_times_valid = spike_times[valid_time_mask]
    n_time_dropped = n_total - len(spike_times_valid)

    spike_interval = np.searchsorted(track.times, spike_times_valid, side="right") - 1
    if len(spike_times_valid) > 0:
        keep = track.interval_mask[spike_interval]
        spike_times_valid = spike_times_valid[keep]
        spike_interval = spike_interval[keep]

    linear_intervals = track.linear_intervals
    if linear_intervals is None:
        # Start allocation: the interval's whole dwell time goes to its start
        # frame's bin, so its spikes must go there too. Interpolating the spike
        # position instead would put spikes in the next bin while the time
        # stays in this one, biasing the rate along the direction of travel.
        n_after_time = len(spike_times_valid)
        if n_after_time == 0:
            return spike_counts, n_time_dropped, 0, n_total, 0
        assert track.frame_bins is not None
        spike_bins = track.frame_bins[spike_interval]
    else:
        runs = linear_intervals.support.runs
        spike_times_valid = spike_times_valid[
            times_in_intervals(runs, spike_times_valid)
        ]
        n_after_time = len(spike_times_valid)
        if n_after_time == 0:
            return spike_counts, n_time_dropped, 0, n_total, 0
        # Linear allocation splits each interval's dwell time along the
        # straight segment between its samples, so spikes take the bin of
        # their interpolated position.
        assert track.observed_times is not None
        assert track.observed_coordinates is not None
        spike_positions = np.column_stack(
            [
                np.interp(spike_times_valid, track.observed_times, coordinate)
                for coordinate in track.observed_coordinates
            ]
        )
        spike_bins = env.bin_at(spike_positions)

    valid_bins = spike_bins[spike_bins >= 0]
    n_bin_dropped = n_after_time - len(valid_bins)

    if len(valid_bins) > 0:
        spike_counts = np.bincount(valid_bins, minlength=n_bins).astype(np.float64)
        if linear_intervals is not None and np.any(
            (spike_counts > 0) & (linear_intervals.occupancy == 0)
        ):
            msg = (
                "Linear spatial encoding assigns spikes by boundary ownership "
                "to a bin with zero dwell time, so its firing rate is undefined. "
                "Adjust the spatial binning or tracking/time resolution to "
                "provide positive exposure for that bin."
            )
            raise ValueError(msg)

    return spike_counts, n_time_dropped, n_bin_dropped, n_total, n_after_time


def count_spikes_by_frame(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    frame_bins: NDArray[np.intp],
    interval_mask: NDArray[np.bool_],
    n_bins: int,
) -> NDArray[np.float64]:
    """Count spikes per bin from the most recent frame, gated by interval validity.

    Interval ``k`` is the half-open ``[times[k], times[k+1])``. A spike at ``t``
    lies in interval ``frame = searchsorted(times, t, "right") - 1``, takes that
    frame's bin, and is kept iff the interval is valid and
    ``frame_bins[frame] >= 0``. A spike before ``times[0]`` or at or after
    ``times[-1]`` lies in no interval and is not counted. Interval validity is
    the SAME mask used for the occupancy denominator
    (``start_allocated_occupancy``), so numerator and denominator drop identical
    intervals.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    spikes = spike_times[(spike_times >= times[0]) & (spike_times < times[-1])]
    frame = np.searchsorted(times, spikes, side="right") - 1
    bins = frame_bins[frame]
    keep = interval_mask[frame] & (bins >= 0)
    return np.bincount(bins[keep], minlength=n_bins).astype(np.float64)


_WARN_ON_DROP_HINT = "Set warn_on_drop=False to suppress this warning."


@dataclass(frozen=True)
class FrameBinning:
    """Prepared frame geometry and time support shared by counts and occupancy.

    ``frame_bins`` holds each missing sample at the bin of the last valid
    one; ``gates`` is the interval mask with the gates that built it, and
    its ``sample_valid`` (None when every sample is valid) records which
    samples were observed.
    """

    times: NDArray[np.float64]
    frame_bins: NDArray[np.intp]
    n_bins: int
    support: TimeSupport
    gates: IntervalGates
    samples: Literal["positions", "headings", "values"] = "positions"

    def count(
        self,
        spike_times: Sequence[NDArray[np.float64]],
        *,
        n_jobs: int = 1,
        warn_on_drop: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Count a population and its shared occupancy on the prepared support."""
        return count_frames_and_occupancy(
            spike_times, self, n_jobs=n_jobs, warn_on_drop=warn_on_drop
        )


def prepare_frame_binning(
    gates: IntervalGates,
    frame_bins: NDArray[np.intp],
    n_bins: int,
    *,
    samples: Literal["positions", "headings", "values"] = "positions",
) -> FrameBinning:
    """Resolve frame support once for counting and result provenance.

    ``gates`` holds the final interval mask, with temporal and bin validity
    (including declared breaks and missing samples), and the gates that
    built it (:func:`~neurospatial._intervals.interval_gates`), so geometry
    consumers can share one gate. ``samples`` names what ``frame_bins``
    bins, for the empty-support warning. Missing samples
    (``gates.sample_valid`` False, such as NaN coordinates or headings) are
    bridged like deleted rows: each missing frame holds the bin of the last
    valid frame, so a bridged gap's time and spikes go to the bin where it
    started.
    """
    support = TimeSupport.from_mask(
        gates.times,
        gates.mask,
        epochs=gates.epochs,
        spike_window=gates.spike_window,
        max_gap=gates.max_gap,
        break_before=gates.break_before,
    )
    held = hold_last_valid(frame_bins, gates.sample_valid, fill=-1)
    return FrameBinning(gates.times, held, n_bins, support, gates, samples)


def count_frames_and_occupancy(
    spike_times_list: Sequence[NDArray[np.float64]],
    prepared: FrameBinning,
    *,
    n_jobs: int,
    warn_on_drop: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Count spikes and dwell time using exactly the recorded interval mask.

    Warns once when the interval gates or the spike times discard data
    (:func:`warn_discarded`), unless ``warn_on_drop`` is False. A selection
    that excludes every interval raises whatever ``warn_on_drop``
    (:func:`~neurospatial._intervals.excluded_intervals_message`).
    """
    times, frame_bins, n_bins = prepared.times, prepared.frame_bins, prepared.n_bins
    support = prepared.support
    mask = support.mask
    messages = [
        excluded_intervals_message(
            prepared.gates,
            samples=prepared.samples,
            result="the tuning curve is empty"
            if prepared.samples == "values"
            else "the rate map is empty",
        )
    ]
    if len(times):
        all_spikes = (
            np.concatenate(spike_times_list)
            if spike_times_list
            else np.empty(0, dtype=np.float64)
        )
        outside = ~((all_spikes >= times[0]) & (all_spikes < times[-1]))
        n_dropped, n_considered, within = time_window_drops(
            all_spikes,
            outside,
            float(times[0]),
            float(times[-1]),
            epochs=prepared.gates.epochs,
            spike_window=prepared.gates.spike_window,
        )
        messages.append(
            _time_window_message(
                n_dropped,
                n_considered,
                float(times[0]),
                float(times[-1]),
                all_spikes,
                scope=within
                + ("across all neurons " if len(spike_times_list) > 1 else ""),
            )
        )
    if warn_on_drop:
        warn_discarded(messages, suppress_hint="")
    occupancy = start_allocated_occupancy(frame_bins, np.diff(times), mask, n_bins)
    spike_counts = np.zeros((len(spike_times_list), n_bins), dtype=np.float64)
    if len(spike_times_list) and n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(count_spikes_by_frame)(spikes, times, frame_bins, mask, n_bins)
            for spikes in spike_times_list
        )
        spike_counts = np.asarray(results, dtype=np.float64)
    else:
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = count_spikes_by_frame(
                spikes, times, frame_bins, mask, n_bins
            )
    return spike_counts, occupancy


def _spatial_interval_gates(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> IntervalGates:
    """Compute the per-interval validity mask once for a whole trajectory.

    The interval-valid mask depends only on ``(times, positions, env, speed,
    min_speed, max_gap)`` — none of which vary per neuron — so the batch path
    computes it ONCE here and reuses it across every per-neuron spike-binning
    call, instead of re-running ``env.bin_at(positions)`` over the full
    trajectory inside each neuron's kernel call. The single-neuron path uses it
    too, keeping one code path.

    The bounds gate always applies, even when gap and speed checks are off,
    and requires both of an interval's samples to be inside the environment.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    times : ndarray, shape (n_samples,)
        Trajectory timestamps (already cast/finite).
    positions : ndarray, shape (n_samples, n_dims)
        Trajectory positions (already reshaped to 2-D).
    speed : ndarray, shape (n_samples,), or None
        Resolved speed array.
    min_speed : float or None
        Speed threshold.
    max_gap : float or None
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.

    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Normalized analysis and acquisition windows applied to each complete
        trajectory interval.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks; the interval ending at
        each break is dropped.

    Returns
    -------
    IntervalGates
        Shared validity, shape (n_samples - 1,) (empty when fewer than two
        samples form an interval), with the gates that built it.
    """
    return start_allocation(
        times,
        env.bin_at(positions),
        np.asarray(np.isfinite(positions).all(axis=1), dtype=bool),
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    ).gates


def _resolve_spatial_time_support(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    warn_on_drop: bool,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    break_before: NDArray[np.bool_] | None = None,
    raise_if_empty: bool = True,
) -> tuple[TimeSupport, str | None]:
    """Resolve training support shared by counting, occupancy and provenance.

    ``epochs`` and ``spike_window`` are resolved windows; ``break_before``
    is already validated and drops the interval ending at each declared
    recording break. A selection with no usable interval raises unless
    ``raise_if_empty`` is False
    (:func:`~neurospatial._intervals.excluded_intervals_message`).

    Returns
    -------
    support : TimeSupport
    drop_message : str or None
        The excluded-interval message when ``warn_on_drop``, for spike
        counting to emit with its own in one warning (:func:`warn_discarded`).
    """
    _validate_time_allocation(time_allocation)
    if time_allocation == "linear":
        if linear_intervals is None:
            linear_intervals = prepare_linear_spatial_intervals(
                env,
                times,
                positions,
                speed=speed,
                min_speed=min_speed,
                max_gap=max_gap,
                epochs=epochs,
                spike_window=spike_window,
                break_before=break_before,
            )
        support = linear_intervals.support
        message = None
        if support.mask.any() and not len(support.runs):
            message = (
                "Linear spatial encoding retained no spatial support: "
                "the observation-valid paths do not enter the active cells "
                "and Arena domain. Check trajectory coordinates and env.arena."
            )
        elif linear_intervals.gates is not None:
            # Prepared from a supplied mask, already reported by its resolver.
            # Raises when nothing usable is left, whatever warn_on_drop.
            message = excluded_intervals_message(
                linear_intervals.gates,
                result="the rate map is empty",
                raise_if_empty=raise_if_empty,
            )
        return support, message if warn_on_drop else None
    gates = _spatial_interval_gates(
        env,
        times,
        positions.reshape(-1, 1) if positions.ndim == 1 else positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    )
    # Raises when nothing usable is left, whatever warn_on_drop.
    message = excluded_intervals_message(
        gates, result="the rate map is empty", raise_if_empty=raise_if_empty
    )
    if not warn_on_drop:
        message = None
    support = TimeSupport.from_mask(
        times,
        gates.mask,
        epochs=epochs,
        spike_window=spike_window,
        max_gap=max_gap,
        break_before=break_before,
    )
    return support, message


_SILENCE_MIN_UNITS = 5
_SILENCE_MIN_SECONDS = 60.0


def tracked_samples(
    n_samples: int, *signals: NDArray[np.float64] | None
) -> NDArray[np.bool_]:
    """True at samples where every given signal is finite, shape (n_samples,).

    Parameters
    ----------
    n_samples : int
    *signals : ndarray, shape (n_samples, ...), or None
        Per-sample positions, headings or similar; None is skipped.
    """
    tracked = np.ones(n_samples, dtype=bool)
    for signal in signals:
        if signal is not None:
            rows = np.asarray(signal, dtype=np.float64).reshape(n_samples, -1)
            tracked &= np.isfinite(rows).all(axis=1)
    return tracked


def _warn_if_population_silent(
    spike_trains: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    tracked: NDArray[np.bool_],
    break_before: NDArray[np.bool_] | None = None,
) -> None:
    """Warn once if every unit is silent for >= 60 s of tracked time.

    Skipped when the caller passed ``spike_window`` (recording coverage is then
    explicit), for populations below ``_SILENCE_MIN_UNITS`` and for recordings
    shorter than ``_SILENCE_MIN_SECONDS``. Otherwise silence is measured over
    the maximal runs of intervals passing the max_gap, epochs and
    break_before gates, with samples that are not ``tracked`` (shape
    ``(n_samples,)``) bridged like deleted rows
    (``run_time_bounds``), independently of speed or frame-bin validity: inside
    a single run, from the run start to the first spike, between consecutive
    spikes of the merged train, and from the last spike to the run end, so a
    stretch never spans an untracked pause.
    """
    n_units = len(spike_trains)
    if (
        spike_window is not None
        or n_units < _SILENCE_MIN_UNITS
        or times[-1] - times[0] < _SILENCE_MIN_SECONDS
    ):
        return
    observed_runs = run_time_bounds(
        times,
        interval_valid_mask(
            times,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            sample_valid=tracked,
        ),
    )
    if observed_runs.shape[0] == 0:
        return
    nonempty = [np.asarray(s, dtype=np.float64) for s in spike_trains if len(s)]
    spikes = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)
    if not _has_empty_observed_bin(spikes, observed_runs, _SILENCE_MIN_SECONDS / 3):
        return  # no silence of _SILENCE_MIN_SECONDS: skip sorting every spike
    run_idx = np.searchsorted(observed_runs[:, 0], spikes, side="right") - 1
    inside = run_idx >= 0
    inside[inside] = spikes[inside] < observed_runs[run_idx[inside], 1]
    n_runs = observed_runs.shape[0]
    marks = np.concatenate([observed_runs[:, 0], observed_runs[:, 1], spikes[inside]])
    owner = np.concatenate([np.arange(n_runs), np.arange(n_runs), run_idx[inside]])
    order = np.lexsort((marks, owner))  # by run, then time
    marks, owner = marks[order], owner[order]
    silence = np.where(owner[1:] == owner[:-1], np.diff(marks), -np.inf)
    k = int(np.argmax(silence))
    if silence[k] < _SILENCE_MIN_SECONDS:
        return
    a, b = float(marks[k]), float(marks[k + 1])
    warnings.warn(
        f"All {n_units} units are silent from {a:.1f} s to {b:.1f} s "
        f"({b - a:.0f} s) while position is tracked. If the electrophysiology "
        f"was not recording then, pass spike_window=(start, stop) so that time "
        f"is excluded from occupancy.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _has_empty_observed_bin(
    spikes: NDArray[np.float64], observed_runs: NDArray[np.float64], width: float
) -> bool:
    """Whether a spike-free bin of ``width`` seconds lies wholly inside a run.

    Bins are aligned at the first run's start. A silence of at least
    ``3 * width`` inside a run always contains such a bin, so when there is
    none no silence that long exists; the converse does not hold.

    Proof: let the silence be the open stretch ``(a, b)`` between
    consecutive marks of one run ``[r0, r1)`` (run start, spikes, run end),
    with ``r0 <= a < b <= r1`` and ``b - a >= 3 * width``; no spike lies in
    ``(a, b)``.
    The first bin edge at or after ``a`` is ``a + d`` with
    ``0 <= d < width``. If ``d >= width / 2``, the bin starting there ends
    at ``a + d + width <= b - width``; otherwise the next bin, from
    ``a + d + width`` to ``a + d + 2 * width <= b - width / 2``, is used.
    Either bin lies inside ``(a, b)``, hence inside the run and free of
    spikes, at least ``width / 2`` from ``a``, ``b`` and the run's edges.
    Locating spikes and run edges in bins errs by a few ulps of the
    timestamps (under 0.02 s below 10**14 s), far less than ``width / 2``,
    so the bin is found empty.

    Parameters
    ----------
    spikes : ndarray, shape (n_spikes,)
        Spike times of every unit, in seconds, in any order.
    observed_runs : ndarray, shape (n_runs, 2)
        Sorted, disjoint ``[start, stop)`` runs, at least one.
    width : float
        Bin width in seconds.

    Returns
    -------
    bool
    """
    origin = observed_runs[0, 0]
    first = np.ceil((observed_runs[:, 0] - origin) / width)
    stop = np.floor((observed_runs[:, 1] - origin) / width)
    whole = stop > first  # runs holding at least one whole bin
    if not whole.any():
        return False
    first, stop = first[whole].astype(np.intp), stop[whole].astype(np.intp)
    n_bins = int(stop.max())
    index = np.floor((spikes - origin) / width)
    index = index[(index >= 0) & (index < n_bins)].astype(np.intp)
    occupied = np.bincount(index, minlength=n_bins) > 0
    edges = np.zeros(n_bins + 1, dtype=np.intp)
    np.add.at(edges, first, 1)
    np.add.at(edges, stop, -1)
    inside = np.cumsum(edges[:-1]) > 0
    return bool(np.any(inside & ~occupied))


def _clock_ratio_hint(
    spikes: NDArray[np.float64], t_start: float, t_stop: float
) -> str:
    """Name a milliseconds/seconds mix-up when spans and magnitudes differ ~1000x.

    A short decode window inside a long recording also has a much shorter span
    than its spikes, but the spike times there are no larger in magnitude than
    the window's, so both ratios must be about 1000 (within ~3x).
    """
    spike_span = float(spikes.max() - spikes.min())
    times_span = t_stop - t_start
    spike_scale = float(np.max(np.abs(spikes)))
    times_scale = max(abs(t_start), abs(t_stop))
    if min(spike_span, times_span, spike_scale, times_scale) <= 0:
        return ""
    log_span = np.log10(spike_span / times_span)
    log_scale = np.log10(spike_scale / times_scale)
    if abs(log_span - 3) < 0.5 and abs(log_scale - 3) < 0.5:
        return (
            f"The spike times span {spike_span:.6g} and reach {spike_scale:.6g}, "
            f"about 1000 times the {times_span:.6g} s of times: if spike_times "
            "are in milliseconds, divide them by 1000. "
        )
    if abs(log_span + 3) < 0.5 and abs(log_scale + 3) < 0.5:
        return (
            f"The spike times span {spike_span:.6g} and reach {spike_scale:.6g}, "
            f"about 1/1000 of the {times_span:.6g} span of times: if times are "
            "in milliseconds, divide them by 1000. "
        )
    return ""


#: Spike times that span more than this many times the tracked time, and
#: reach more than this many times the largest time and window bound, are
#: taken to be on another clock (milliseconds against seconds) by
#: :func:`time_window_drops`. It is the geometric midpoint of the same clock
#: (1) and milliseconds (1000). On the 21 recordings with spikes in the
#: real-data survey (``datasets/``), spikes in seconds spanned at most 17
#: times the tracked time and reached at most 2 times its bounds; in
#: milliseconds both ratios were at least 300, also where the tracking
#: outlasts the spike recording (DANDI 000165). Magnitudes cannot tell
#: milliseconds from a spike recording over 32 times longer than tracking
#: that starts with it (10 min at the start of 6 h); such a recording warns.
_OTHER_CLOCK_FACTOR = 10**1.5


def time_window_drops(
    spikes: NDArray[np.float64],
    outside: NDArray[np.bool_],
    t_min: float,
    t_max: float,
    *,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
) -> tuple[int, int, str]:
    """Count spikes lost because they fall outside the tracked time.

    A spike outside ``epochs`` or ``spike_window`` is excluded by the
    caller's own selection, so only spikes inside both (the analysed
    windows) count: tracking only during wake, with ``epochs=wake``, loses
    no spike. When the spike times look like another clock than ``times``,
    the windows are on the clock of ``times`` and say nothing about the
    spikes, so every spike counts. Another clock is: spans and magnitudes
    about 1000 times apart (:func:`_clock_ratio_hint`), spike times that
    span more than ``_OTHER_CLOCK_FACTOR`` times the tracked time and reach
    more than that times every time and window bound, or no spike inside the
    tracked time. A unit with tracked spikes but none inside the analysed
    windows is silent there and drops no spike.
    A deliberately empty selection (a window set with no rows) counts no
    spike.

    Parameters
    ----------
    spikes : ndarray, shape (n_spikes,)
        Every unit's spike times, concatenated, in seconds.
    outside : ndarray of bool, shape (n_spikes,)
        True for spikes outside the tracked time ``[t_min, t_max]``, in the
        caller's convention for the closing edge.
    t_min, t_max : float
        The tracked time.
    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Resolved windows (:func:`~neurospatial._intervals.as_intervals`).

    Returns
    -------
    n_dropped : int
        Spikes inside the analysed windows but outside the tracked time.
    n_considered : int
        Spikes inside the analysed windows.
    scope : str
        Phrase naming the windows for the message, ``""`` when every spike
        counts.
    """
    windows = [w for w in (epochs, spike_window) if w is not None]
    if not windows or not spikes.size:
        return int(np.count_nonzero(outside)), int(spikes.size), ""
    # Windows are on the clock of ``times``. Spike times far longer and larger
    # than ``times`` are on another clock (milliseconds), and the windows say
    # nothing about which of them were meant: the milliseconds spikes of the
    # recording's first instants can still land inside them. Count every one.
    largest = max(
        abs(t_min), abs(t_max), *(float(np.abs(w).max()) for w in windows if w.size)
    )
    if _clock_ratio_hint(spikes, t_min, t_max) or (
        float(spikes.max() - spikes.min()) > _OTHER_CLOCK_FACTOR * (t_max - t_min)
        and float(np.abs(spikes).max()) > _OTHER_CLOCK_FACTOR * largest
    ):
        return int(np.count_nonzero(outside)), int(spikes.size), ""
    if any(w.shape[0] == 0 for w in windows):
        # A deliberately empty selection drops every spike by choice; its own
        # message says so.
        return 0, 0, ""
    if outside.all():
        # No spike in the tracked time at all: the spikes are on another clock
        # (an offset, as from a misaligned NWB file), not outside a selection.
        # Count every one.
        return int(np.count_nonzero(outside)), int(spikes.size), ""
    # Some spikes are tracked, so none in the analysed windows is a unit
    # silent there (a cell active only in sleep), which drops nothing.
    analysed = windows[0] if len(windows) == 1 else intersect_intervals(*windows)
    inside = times_in_intervals(analysed, spikes)
    names = " and ".join(
        name
        for name, value in (("epochs", epochs), ("spike_window", spike_window))
        if value is not None
    )
    return (
        int(np.count_nonzero(outside & inside)),
        int(np.count_nonzero(inside)),
        f"inside {names} ",
    )


def warn_discarded(messages: Sequence[str | None], *, suppress_hint: str) -> None:
    """Emit the discarded-data messages of one call as one ``UserWarning``.

    A call warns at most once, but no problem is hidden: every message that
    applies is in the warning, followed by ``suppress_hint``.
    """
    present = "\n\n".join(message for message in messages if message)
    if present:
        warnings.warn(
            f"{present} {suppress_hint}".rstrip(),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def _emit_time_window_warning(
    n_time_dropped: int,
    n_total: int,
    t_min: float,
    t_max: float,
    all_spike_times: NDArray[np.float64] | None,
    *,
    window: str = "position",
    scope: str = "",
    suppress_hint: str = _WARN_ON_DROP_HINT,
) -> None:
    """Warn with :func:`_time_window_message` alone."""
    warn_discarded(
        [
            _time_window_message(
                n_time_dropped,
                n_total,
                t_min,
                t_max,
                all_spike_times,
                scope=scope,
                window=window,
            )
        ],
        suppress_hint=suppress_hint,
    )


def _time_window_message(
    n_time_dropped: int,
    n_total: int,
    t_min: float,
    t_max: float,
    all_spike_times: NDArray[np.float64] | None,
    *,
    scope: str = "",
    window: str = "position",
) -> str | None:
    """The time-window spike-drop message, or None at or below the threshold.

    Parameters
    ----------
    n_time_dropped : int
        Number of spikes dropped due to time-window exclusion.
    n_total : int
        Total number of spikes.
    t_min, t_max : float
        Position time window bounds.
    all_spike_times : ndarray or None
        Concatenated spike times (for min/max display).  If None the values
        are omitted from the message.
    scope : str, optional
        Extra phrase inserted into the message (e.g. "across all neurons ").
    window : str, optional
        Which time window the message names: ``"position"`` or ``"decode"``.
    """
    if n_total == 0 or n_time_dropped == 0:
        return None
    frac = n_time_dropped / n_total
    if frac <= DISCARD_WARN_FRACTION:
        return None
    if all_spike_times is not None and len(all_spike_times) > 0:
        range_part = (
            f"spike_times.min()={all_spike_times.min():.6g} "
            f"spike_times.max()={all_spike_times.max():.6g}. "
            f"{_clock_ratio_hint(all_spike_times, t_min, t_max)}"
        )
    else:
        range_part = ""
    return (
        f"{n_time_dropped}/{n_total} spike_times "
        f"({100 * frac:.0f}%) {scope}fell outside the {window} time "
        f"window [{t_min:.6g}, {t_max:.6g}]; "
        f"{range_part}"
        f"Check that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute."
    )


def _inactive_bin_message(
    n_bin_dropped: int,
    n_after_time: int,
    *,
    scope: str = "",
) -> str | None:
    """The inactive-bin spike-drop message, or None at or below the threshold.

    Parameters
    ----------
    n_bin_dropped : int
        Spikes that mapped to inactive/out-of-environment bins.
    n_after_time : int
        Spikes that survived the time-window filter (denominator).
    scope : str, optional
        Extra phrase inserted into the message (e.g. "across all neurons ").
    """
    if n_after_time == 0 or n_bin_dropped == 0:
        return None
    frac = n_bin_dropped / n_after_time
    if frac <= DISCARD_WARN_FRACTION:
        return None
    return (
        f"{n_bin_dropped}/{n_after_time} spikes "
        f"({100 * frac:.0f}%) {scope}mapped to positions outside "
        f"the active environment bins (bin index -1). "
        f"Check that positions are in the same coordinate frame as the "
        f"environment and that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute."
    )


def _resolve_spatial_mask(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    interval_mask: NDArray[np.bool_] | None,
    time_allocation: Literal["start", "linear"],
    linear_intervals: LinearSpatialIntervals | None,
) -> tuple[NDArray[np.bool_], LinearSpatialIntervals | None]:
    """Resolve the interval gate shared by spike counts and occupancy.

    Parameters are those of :func:`bin_spike_trains`; ``positions`` is already
    two-dimensional.

    Returns
    -------
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        The supplied mask, the linear support's mask, or the mask computed
        from the gates and windows.
    linear_intervals : LinearSpatialIntervals or None
        The supplied or prepared linear support; ``None`` for ``"start"``.
    """
    if time_allocation == "linear":
        if linear_intervals is None:
            epochs, spike_window, _ = resolve_time_windows(epochs, spike_window)
            linear_intervals = prepare_linear_spatial_intervals(
                env,
                times,
                positions,
                speed=speed,
                min_speed=min_speed,
                max_gap=max_gap,
                epochs=epochs,
                spike_window=spike_window,
                interval_mask=interval_mask,
            )
        return linear_intervals.support.mask, linear_intervals
    if interval_mask is None:
        resolved_epochs, resolved_spike_window, _ = resolve_time_windows(
            epochs, spike_window
        )
        interval_mask = _spatial_interval_gates(
            env,
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
        ).mask
    return interval_mask, None


def compute_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    spike_window: Any = None,
    interval_mask: NDArray[np.bool_] | None = None,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    context: str = "compute_occupancy",
) -> NDArray[np.float64]:
    """Compute occupancy (time spent in each bin).

    Computes the total time spent in each spatial bin by accumulating
    time intervals from the trajectory.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    speed : ndarray, shape (n_samples,), or None
        Concrete, already-resolved speed array (typically from
        :func:`resolve_speed`). Used by the shared mask; when provided
        with ``min_speed``, interval ``k`` is gated by ``speed[k] >= min_speed``.
    min_speed : float or None
        Minimum speed threshold. When ``None`` (default), no speed filtering
        is applied.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.

    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict complete sampling intervals to these half-open windows in
        seconds on the same clock as ``times``. None means unrestricted.
    spike_window : same forms as epochs, or None
        When electrophysiology was recording. Sampling intervals outside
        these windows contribute neither spikes nor occupancy.
    interval_mask : ndarray of bool, shape (n_samples - 1,), or None
        Optional precomputed validity shared by counts and occupancy. When
        supplied, it already includes all requested gates and windows, so
        these helpers do not recalculate or renormalize them.
    time_allocation : {"start", "linear"}, default="start"
        Assign complete intervals to their start bins or split their straight
        paths by exact planar support, using the same clock as spike counts.
    linear_intervals : LinearSpatialIntervals or None
        Prepared linear support shared with spike counting.

    Returns
    -------
    ndarray, shape (n_bins,)
        Time in seconds spent in each spatial bin.

    Raises
    ------
    ValueError
        If times and positions have different lengths.
        If positions have wrong number of dimensions.

    Notes
    -----
    Counts and dwell share the same temporal support. Linear allocation keeps
    original fractional durations without renormalizing excluded support.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._spatial_counts import compute_occupancy

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100

    >>> # Compute occupancy
    >>> occupancy = compute_occupancy(env, times, trajectory)
    >>> occupancy.shape[0] == env.n_bins
    True
    """
    _validate_time_allocation(time_allocation)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    # Validate input shapes
    if len(times) != len(positions):
        msg = (
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )
        raise ValueError(msg)

    # Handle 1D positions
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    require_position_columns(positions, env.n_dims, call=context)

    interval_mask, linear_intervals = _resolve_spatial_mask(
        env,
        times,
        positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        interval_mask=interval_mask,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )
    if linear_intervals is not None:
        return np.array(linear_intervals.occupancy, dtype=np.float64, copy=True)
    return start_allocated_occupancy(
        _start_frame_bins(env, positions),
        np.diff(times),
        interval_mask,
        env.n_bins,
    )


def bin_spike_trains(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    spike_window: Any = None,
    interval_mask: NDArray[np.bool_] | None = None,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    n_jobs: int = 1,
    warn_on_drop: bool = True,
    drop_scope: str = "across all neurons ",
    drop_message: str | None = None,
    check_time_window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains into spatial bins.

    Precomputes shared quantities (frame bins, occupancy) once for every
    neuron and optionally parallelizes spike counting with joblib. Each spike
    goes to the bin that receives its sampling interval's dwell time: the bin
    of the interval's start sample under ``"start"`` allocation, the bin of
    its interpolated position under ``"linear"``.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Can be:
        - List/tuple of 1D arrays (one per neuron)
        - 2D array shape (n_neurons, max_spikes) with NaN padding
        Input is coerced to per-neuron spike trains via as_spike_trains().
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    speed : ndarray, shape (n_samples,), or None
        Caller-supplied speed array. When ``min_speed`` is set, the speed
        array is resolved ONCE (auto-derived via forward difference if this is
        ``None``) and the SAME concrete array gates BOTH the shared occupancy
        denominator and every per-neuron spike numerator, guaranteeing
        numerator/denominator alignment by construction. See
        :func:`resolve_speed`.
    min_speed : float or None
        Minimum speed threshold (physical units / second). When ``None``
        (default) no speed filtering happens. When set, low-speed intervals are excluded from BOTH
        occupancy and spike counts.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a single ``UserWarning`` (per drop
        cause) when a large fraction of spikes are silently dropped across
        all neurons.  The warning is computed in the main process from
        aggregate statistics, so it fires exactly once even when
        ``n_jobs != 1`` (joblib worker warnings are commonly swallowed).
        Set to ``False`` to suppress all drop-related warnings.
    drop_scope : str, default="across all neurons "
        Phrase naming the spikes a drop warning counts; ``""`` for one unit
        analyzed on its own.
    drop_message : str or None, optional
        The caller's own discarded-data message (an excluded-interval
        message), emitted with the spike-drop messages in one warning.
    check_time_window : bool, default=True
        Whether to report spikes outside the tracked time. False when the
        caller checks the whole recording once, for example per-label maps.

    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict complete sampling intervals to these half-open windows in
        seconds on the same clock as ``times``. None means unrestricted.
    spike_window : same forms as epochs, or None
        When electrophysiology was recording. Sampling intervals outside
        these windows contribute neither spikes nor occupancy.
    interval_mask : ndarray of bool, shape (n_samples - 1,), or None
        Optional precomputed validity shared by counts and occupancy. When
        supplied, it already includes all requested gates and windows, so
        these helpers do not recalculate or renormalize them.
    time_allocation : {"start", "linear"}, default="start"
        Start allocation or exact planar segment allocation, applied equally
        to all units and their shared dwell time.
    linear_intervals : LinearSpatialIntervals or None
        Prepared linear support. If absent, linear geometry is resolved once
        for this batch, including an empty batch.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent in each spatial bin (shared across neurons).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._spatial_counts import bin_spike_trains

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory and spikes for 3 neurons
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0]),  # Neuron 2
    ... ]

    >>> # Bin spikes
    >>> spike_counts, occupancy = bin_spike_trains(
    ...     env, spike_times, times, trajectory, n_jobs=2
    ... )
    >>> spike_counts.shape[0] == 3  # 3 neurons
    True
    >>> spike_counts.shape[1] == env.n_bins
    True
    >>> occupancy.shape[0] == env.n_bins
    True

    See Also
    --------
    compute_occupancy : Compute occupancy only
    as_spike_trains : Coerce input to canonical per-neuron spike trains
    """
    from neurospatial._spikes import as_spike_trains

    _validate_time_allocation(time_allocation)
    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = as_spike_trains(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    # Resolve the speed gate ONCE here so the SAME concrete array feeds both
    # the (shared) occupancy denominator and every per-neuron spike numerator.
    # This is what guarantees numerator/denominator alignment by construction.
    resolved_speed = resolve_speed(times, positions, speed, min_speed, max_gap=max_gap)
    interval_mask, linear_intervals = _resolve_spatial_mask(
        env,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        interval_mask=interval_mask,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )
    require_position_columns(positions, env.n_dims, call="bin_spike_trains")
    # The frame bins (or linear lookups) are built once and shared by the
    # occupancy and every unit's spike counts.
    track = _spike_track(
        env,
        times,
        positions,
        interval_mask=interval_mask,
        linear_intervals=linear_intervals,
    )
    occupancy = _track_occupancy(track, env.n_bins)

    # Each unit's counts come back with its drop statistics, so the warnings
    # below are aggregated in the main process: warnings.warn() from a joblib
    # worker is commonly swallowed.
    spike_arrays = [np.asarray(spikes, dtype=np.float64) for spikes in spike_times_list]
    if n_jobs == 1:
        results = [
            _bin_spike_train_with_stats(env, spikes, track) for spikes in spike_arrays
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_bin_spike_train_with_stats)(env, spikes, track)
            for spikes in spike_arrays
        )
    spike_counts = np.array([counts for counts, *_ in results], dtype=np.float64)
    spike_counts = spike_counts.reshape(n_neurons, env.n_bins)
    _, n_bin_dropped, _, n_after_time = (
        np.array([stats for _, *stats in results], dtype=np.int64)
        .reshape(n_neurons, 4)
        .sum(axis=0)
        .tolist()
    )

    # Warn ONCE in the main process from the aggregated statistics.
    if warn_on_drop:
        messages = [drop_message]
        if n_neurons > 0 and check_time_window:
            all_spikes = np.concatenate(spike_arrays)
            # Resolved windows (an (n, 2) ndarray, possibly with no rows) pass
            # as is; any other form is resolved here.
            analysed_epochs, analysed_window = (
                (epochs, spike_window)
                if all(
                    w is None or (isinstance(w, np.ndarray) and w.ndim == 2)
                    for w in (epochs, spike_window)
                )
                else resolve_time_windows(epochs, spike_window)[:2]
            )
            n_time_dropped, n_considered, within = time_window_drops(
                all_spikes,
                (all_spikes < track.t_min) | (all_spikes >= track.t_max),
                track.t_min,
                track.t_max,
                epochs=analysed_epochs,
                spike_window=analysed_window,
            )
            messages.append(
                _time_window_message(
                    n_time_dropped,
                    n_considered,
                    track.t_min,
                    track.t_max,
                    all_spikes,
                    scope=within + drop_scope,
                )
            )
        if n_neurons > 0:
            messages.append(
                _inactive_bin_message(n_bin_dropped, n_after_time, scope=drop_scope)
            )
        warn_discarded(messages, suppress_hint=_WARN_ON_DROP_HINT)

    return spike_counts, occupancy
