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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import (
    TimeSupport,
    interval_valid_mask,
    resolve_time_windows,
    run_time_bounds,
    start_allocated_occupancy,
    warn_excluded_intervals,
)
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.geometry import require_cartesian, require_position_columns
from neurospatial.ops import (
    _polygon_linear_occupancy,
    _RetainedLinearInterval,
    _uses_polygon_allocation,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "bin_spike_train",
    "bin_spike_trains",
    "compute_occupancy",
    "resolve_speed",
]


# _DROP_WARN_THRESHOLD is always < 1.0, so 100%-dropped (fraction == 1.0)
# always satisfies `frac > threshold` and never needs a separate == check.
_DROP_WARN_THRESHOLD = 0.5  # warn when fraction dropped exceeds this


@dataclass(frozen=True)
class LinearSpatialIntervals:
    """Owned fractional support prepared once for a spatial encoding call."""

    support: TimeSupport
    pieces: tuple[_RetainedLinearInterval, ...]
    occupancy: NDArray[np.float64]


def _validate_time_allocation(time_allocation: str) -> None:
    if time_allocation not in ("start", "linear"):
        raise ValueError("time_allocation must be 'start' or 'linear'.")


def prepare_linear_spatial_intervals(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
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
    epochs, spike_window : array-like of shape (n_windows, 2), or None
        Windows that must contain each complete observation interval.
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
        raise NotImplementedError(
            "time_allocation='linear' spatial encoding requires exact planar "
            "Cartesian area cells. Use time_allocation='start' for this environment."
        )
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(times), 2):
        raise ValueError("linear spatial encoding needs (n_samples, 2) positions.")
    # This checks known nonmaterializable Arena support before any cell work.
    _uses_polygon_allocation(binning, arena)
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    if interval_mask is None:
        mask = interval_valid_mask(
            times,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
            break_before=break_before,
        )
    else:
        mask = np.array(interval_mask, dtype=bool, copy=True)
        if mask.shape != (max(len(times) - 1, 0),):
            raise ValueError(
                "interval_mask must contain one value per observation interval."
            )
    finite = np.all(np.isfinite(positions), axis=1)
    mask &= finite[:-1] & finite[1:] & (np.diff(times) > 0)
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
            raise ValueError(
                "Linear spatial support is shorter than this timestamp clock "
                "can represent. Use elapsed timestamps (subtract the recording "
                "origin from both trajectory and spike times) or coarser "
                "spatial binning."
            )
        if runs and left <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], right)
        else:
            runs.append([left, right])
    occupancy.setflags(write=False)
    support = TimeSupport.from_mask(
        times,
        mask,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        max_gap=max_gap,
        break_before=break_before,
        runs=np.asarray(runs, dtype=np.float64).reshape(-1, 2),
    )
    return LinearSpatialIntervals(support, pieces, occupancy)


def resolve_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    *,
    max_gap: float | None,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64] | None:
    """Resolve the speed array used for the shared speed gate.

    This is the single source of truth for the speed array that gates BOTH
    the spike numerator and the occupancy denominator. Both the single and
    batch encode paths call this ONCE at the top and pass the resulting
    concrete array down to both the spike kernel and ``env.occupancy`` — so
    the retained-sample set is identical on both sides by construction, and a
    firing rate (spikes / occupancy) cannot become biased by gating only one
    side.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Trajectory timestamps in seconds (assumed sorted/finite; validated
        upstream by ``validate_trajectory``). Division by ``Δt`` is guarded
        defensively against non-positive gaps.
    positions : ndarray, shape (n_samples, n_dims)
        Trajectory position coordinates.
    speed : ndarray, shape (n_samples,), or None
        Caller-supplied speed array. When provided, it is validated and
        returned unchanged (no re-derivation downstream).
    min_speed : float or None
        Speed threshold. When ``None``, NO speed filtering happens: this
        returns ``None`` and the caller passes nothing speed-related to
        ``env.occupancy``. It is an error to pass ``speed``
        without ``min_speed`` (see Raises): a bare ``speed`` would otherwise be
        silently ignored, asymmetric with ``env.occupancy`` (which raises on
        ``min_speed`` without ``speed``).

    Returns
    -------
    ndarray, shape (n_samples,), or None
        - ``None`` when ``min_speed is None`` (no filtering).
        - The validated ``speed`` array when one was passed.
        - Otherwise :func:`neurospatial.behavior.compute_speed` (smoothed).

    Raises
    ------
    ValueError
        If ``speed`` is provided without ``min_speed`` (the message names both
        ``speed`` and ``min_speed``). This mirrors ``env.occupancy``, which
        raises on ``min_speed`` without ``speed``.
        If a ``speed`` array is passed whose length does not match
        ``times`` (the message names ``speed``).

    max_gap : float or None
        Longest sampling interval treated as continuous; the derived speed is
        smoothed separately within each run between longer gaps.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks; the derived speed is
        never smoothed across one.

    Notes
    -----
    When ``speed is None`` and ``min_speed`` is set, speed is
    :func:`neurospatial.behavior.compute_speed`: the magnitude of the
    Gaussian-smoothed (0.1 s) velocity, so tracking jitter at rest does not
    pass the gate. Pass ``speed=`` to use another definition, for example
    speed along a linearized track.
    """
    if min_speed is None:
        if speed is not None:
            # Asymmetric footgun: a caller passing `speed` without `min_speed`
            # would otherwise have it silently ignored (no filtering applied).
            # env.occupancy raises the mirror case (min_speed without speed),
            # so raise here too rather than swallow `speed`.
            raise ValueError(
                "speed was provided without min_speed; pass "
                "min_speed=<threshold> to enable speed filtering, or omit "
                "speed."
            )
        # No filtering requested: callers pass no speed gate.
        return None

    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    if speed is not None:
        speed_arr = np.asarray(speed, dtype=np.float64)
        if len(speed_arr) != len(times):
            raise ValueError(
                f"speed length ({len(speed_arr)}) must match times length "
                f"({len(times)})."
            )
        return speed_arr

    from neurospatial._intervals import interval_valid_mask
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING, smoothed_speed

    if len(times) <= 1:
        # No intervals: speed is undefined; return zeros so a min_speed gate
        # consistently excludes the (non-existent) intervals on both sides.
        return np.zeros(len(times), dtype=np.float64)
    mask = interval_valid_mask(
        times, max_gap=max_gap, epochs=None, break_before=break_before
    )
    return smoothed_speed(times, positions, mask, smooth_window=DEFAULT_SPEED_SMOOTHING)


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


def _bin_spike_train_with_stats(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    interval_mask: NDArray[np.bool_],
    linear_intervals: LinearSpatialIntervals | None = None,
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
    times : ndarray, shape (n_samples,)
        Validated trajectory timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Trajectory coordinates, already reshaped to two dimensions.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Precomputed gap, speed, bounds and time-window validity. The same
        array gates occupancy and every unit's spike counts. A spike belongs
        to interval ``searchsorted(times, t, side="right") - 1``; a spike at
        ``times[-1]`` belongs to no interval. Exclusions by this mask are
        intentional and do not count toward inactive-bin drop warnings.

    linear_intervals : LinearSpatialIntervals or None
        Prepared ``time_allocation="linear"`` intervals. ``None`` means
        ``"start"`` allocation.

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

    t_min, t_max = times.min(), times.max()
    valid_time_mask = (spike_times >= t_min) & (spike_times < t_max)
    spike_times_valid = spike_times[valid_time_mask]
    n_time_dropped = n_total - len(spike_times_valid)

    spike_interval = np.searchsorted(times, spike_times_valid, side="right") - 1
    if len(spike_times_valid) > 0:
        keep = interval_mask[spike_interval]
        spike_times_valid = spike_times_valid[keep]
        spike_interval = spike_interval[keep]

    if linear_intervals is None:
        # Start allocation: the interval's whole dwell time goes to its start
        # frame's bin, so its spikes must go there too. Interpolating the spike
        # position instead would put spikes in the next bin while the time
        # stays in this one, biasing the rate along the direction of travel.
        n_after_time = len(spike_times_valid)
        if n_after_time == 0:
            return spike_counts, n_time_dropped, 0, n_total, 0
        spike_bins = env.bin_at(positions[spike_interval])
    else:
        runs = linear_intervals.support.runs
        if len(runs) == 0:
            spike_times_valid = spike_times_valid[:0]
        elif len(spike_times_valid) > 0:
            run = np.searchsorted(runs[:, 0], spike_times_valid, side="right") - 1
            keep = (run >= 0) & (spike_times_valid < runs[np.maximum(run, 0), 1])
            spike_times_valid = spike_times_valid[keep]
        n_after_time = len(spike_times_valid)
        if n_after_time == 0:
            return spike_counts, n_time_dropped, 0, n_total, 0
        # Linear allocation splits each interval's dwell time along the
        # straight segment between its samples, so spikes take the bin of
        # their interpolated position.
        n_dims = positions.shape[1]
        spike_positions = np.empty((n_after_time, n_dims), dtype=np.float64)
        for d in range(n_dims):
            spike_positions[:, d] = np.interp(spike_times_valid, times, positions[:, d])
        spike_bins = env.bin_at(spike_positions)

    valid_bins = spike_bins[spike_bins >= 0]
    n_bin_dropped = n_after_time - len(valid_bins)

    if len(valid_bins) > 0:
        spike_counts = np.bincount(valid_bins, minlength=n_bins).astype(np.float64)
        if linear_intervals is not None and np.any(
            (spike_counts > 0) & (linear_intervals.occupancy == 0)
        ):
            raise ValueError(
                "Linear spatial encoding assigns spikes by boundary ownership "
                "to a bin with zero dwell time, so its firing rate is undefined. "
                "Adjust the spatial binning or tracking/time resolution to "
                "provide positive exposure for that bin."
            )

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
    """Prepared frame geometry and time support shared by counts and occupancy."""

    times: NDArray[np.float64]
    frame_bins: NDArray[np.intp]
    n_bins: int
    support: TimeSupport
    break_before: NDArray[np.bool_] | None = None

    def count(
        self, spike_times: Sequence[NDArray[np.float64]], *, n_jobs: int = 1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Count a population and its shared occupancy on the prepared support."""
        return count_frames_and_occupancy(spike_times, self, n_jobs=n_jobs)


def prepare_frame_binning(
    times: NDArray[np.float64],
    frame_bins: NDArray[np.intp],
    n_bins: int,
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    interval_mask: NDArray[np.bool_] | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> FrameBinning:
    """Resolve frame support once for counting and result provenance.

    A supplied ``interval_mask`` already includes temporal and bin validity
    (including ``break_before``); geometry consumers can share that gate
    without evaluating windows again. ``break_before`` is already validated.
    """
    if interval_mask is None:
        mask = interval_valid_mask(
            times,
            start_bin=frame_bins,
            max_gap=max_gap,
            epochs=epochs,
            spike_window=spike_window,
            break_before=break_before,
        )
    else:
        # Geometry consumers may already have resolved the shared gate.
        mask = interval_mask
    support = TimeSupport.from_mask(
        times,
        mask,
        epochs=epochs,
        spike_window=spike_window,
        max_gap=max_gap,
        break_before=break_before,
    )
    return FrameBinning(times, frame_bins, n_bins, support, break_before)


def count_frames_and_occupancy(
    spike_times_list: Sequence[NDArray[np.float64]],
    prepared: FrameBinning,
    *,
    n_jobs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Count spikes and dwell time using exactly the recorded interval mask."""
    times, frame_bins, n_bins = prepared.times, prepared.frame_bins, prepared.n_bins
    support = prepared.support
    mask = support.mask
    warn_excluded_intervals(
        times,
        mask,
        max_gap=support.max_gap,
        epochs=support.epochs,
        spike_window=support.spike_window,
        start_bin_gate=True,
        break_before=prepared.break_before,
        result="the rate map is empty",
    )
    if len(times):
        all_spikes = (
            np.concatenate(spike_times_list)
            if spike_times_list
            else np.empty(0, dtype=np.float64)
        )
        outside = ~((all_spikes >= times[0]) & (all_spikes < times[-1]))
        _emit_time_window_warning(
            int(outside.sum()),
            all_spikes.size,
            float(times[0]),
            float(times[-1]),
            all_spikes,
            scope="across all neurons " if len(spike_times_list) > 1 else "",
            suppress_hint="",
        )
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


def _spatial_interval_mask(
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
) -> NDArray[np.bool_]:
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
        Maximum interval gap in seconds.

    epochs, spike_window : ndarray, shape (n_windows, 2), or None
        Normalized analysis and acquisition windows applied to each complete
        trajectory interval.
    break_before : ndarray of bool, shape (n_samples,), or None
        Already validated declared recording breaks; the interval ending at
        each break is dropped.

    Returns
    -------
    ndarray of bool, shape (n_samples - 1,)
        Shared validity; empty when fewer than two samples form an interval.
    """
    from neurospatial._intervals import interval_valid_mask

    sample_bins = env.bin_at(positions)
    mask = interval_valid_mask(
        times,
        positions,
        env,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        start_bin=sample_bins,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    )
    # Complete-case rule: an interval counts only when both of its samples are
    # tracked inside the environment. One whose end sample is a dropout (NaN)
    # or outside is left out of the spike counts and the occupancy alike,
    # rather than imputing where the animal was.
    return mask & (sample_bins[1:] >= 0)


def _resolve_spatial_time_support(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    warn_on_drop: bool,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> TimeSupport:
    """Resolve training support shared by counting, occupancy and provenance.

    ``break_before`` is already validated and drops the interval ending at
    each declared recording break.
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
        if warn_on_drop:
            if support.mask.any() and not len(support.runs):
                warnings.warn(
                    "Linear spatial encoding retained no spatial support: "
                    "the observation-valid paths do not enter the active cells "
                    "and Arena domain. Check trajectory coordinates and env.arena. "
                    + _WARN_ON_DROP_HINT,
                    UserWarning,
                    stacklevel=_external_stacklevel(),
                )
            else:
                warn_excluded_intervals(
                    times,
                    support.mask,
                    max_gap=max_gap,
                    epochs=support.epochs,
                    spike_window=support.spike_window,
                    min_speed=min_speed,
                    start_bin_gate=True,
                    break_before=break_before,
                    result="the rate map is empty",
                    suppress_hint=_WARN_ON_DROP_HINT,
                )
        return support
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    interval_mask = _spatial_interval_mask(
        env,
        times,
        positions.reshape(-1, 1) if positions.ndim == 1 else positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=break_before,
    )
    if warn_on_drop:
        warn_excluded_intervals(
            times,
            interval_mask,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
            min_speed=min_speed,
            start_bin_gate=True,
            break_before=break_before,
            result="the rate map is empty",
            suppress_hint=_WARN_ON_DROP_HINT,
        )
    return TimeSupport.from_mask(
        times,
        interval_mask,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        max_gap=max_gap,
        break_before=break_before,
    )


_SILENCE_MIN_UNITS = 5
_SILENCE_MIN_SECONDS = 60.0


def _warn_if_population_silent(
    spike_trains: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    break_before: NDArray[np.bool_] | None = None,
) -> None:
    """Warn once if every unit is silent for >= 60 s of tracked time.

    Skipped when the caller passed ``spike_window`` (recording coverage is then
    explicit), for populations below ``_SILENCE_MIN_UNITS`` and for recordings
    shorter than ``_SILENCE_MIN_SECONDS``. Otherwise silence is measured over
    the maximal runs of intervals passing the max_gap, epochs and
    break_before gates
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
            times, max_gap=max_gap, epochs=epochs, break_before=break_before
        ),
    )
    if observed_runs.shape[0] == 0:
        return
    nonempty = [np.asarray(s, dtype=np.float64) for s in spike_trains if len(s)]
    spikes = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)
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


def _emit_time_window_warning(
    n_time_dropped: int,
    n_total: int,
    t_min: float,
    t_max: float,
    all_spike_times: NDArray[np.float64] | None,
    *,
    scope: str = "",
    suppress_hint: str = _WARN_ON_DROP_HINT,
) -> None:
    """Emit a UserWarning for time-window spike drops if the fraction exceeds threshold.

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
    suppress_hint : str, optional
        Closing sentence naming how to silence the warning; empty for callers
        without a ``warn_on_drop`` parameter.
    """
    if n_total == 0 or n_time_dropped == 0:
        return
    frac = n_time_dropped / n_total
    if frac <= _DROP_WARN_THRESHOLD:
        return
    if all_spike_times is not None and len(all_spike_times) > 0:
        range_part = (
            f"spike_times.min()={all_spike_times.min():.6g} "
            f"spike_times.max()={all_spike_times.max():.6g}. "
        )
    else:
        range_part = ""
    warnings.warn(
        (
            f"{n_time_dropped}/{n_total} spike_times "
            f"({100 * frac:.0f}%) {scope}fell outside the position time "
            f"window [{t_min:.6g}, {t_max:.6g}]; "
            f"{range_part}"
            f"Check that spike_times and times share units (both seconds). "
            f"Dropped spikes do not contribute. {suppress_hint}"
        ).rstrip(),
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _emit_inactive_bin_warning(
    n_bin_dropped: int,
    n_after_time: int,
    *,
    scope: str = "",
) -> None:
    """Emit a UserWarning for inactive-bin spike drops if the fraction exceeds threshold.

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
        return
    frac = n_bin_dropped / n_after_time
    if frac <= _DROP_WARN_THRESHOLD:
        return
    warnings.warn(
        f"{n_bin_dropped}/{n_after_time} spikes "
        f"({100 * frac:.0f}%) {scope}mapped to positions outside "
        f"the active environment bins (bin index -1). "
        f"Check that positions are in the same coordinate frame as the "
        f"environment and that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def bin_spike_train(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    interval_mask: NDArray[np.bool_] | None = None,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    context: str = "bin_spike_train",
    warn_on_drop: bool = True,
) -> NDArray[np.float64]:
    """Bin spike train into spatial bins.

    Converts continuous spike times to spike counts per spatial bin. Each
    spike goes to the bin that receives its sampling interval's dwell time
    (see Notes).

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    speed : ndarray, shape (n_samples,), or None
        Concrete, already-resolved speed array (typically from
        :func:`resolve_speed`). When provided with ``min_speed``, spikes are
        gated by the same per-interval speed criterion ``env.occupancy`` uses
        (see ``min_speed``). Consumed as-is; never re-derived here.
    min_speed : float or None
        Minimum speed threshold. When provided together with ``speed``, each
        time-window-valid spike at time ``t`` is kept iff
        ``speed[searchsorted(times, t, "right") - 1] >= min_speed`` — the same
        per-interval gate ``env.occupancy`` applies to the denominator. When
        ``None`` (default) no speed filtering happens.
    max_gap : float or None
        Maximum time gap in seconds. Spikes inside intervals with
        ``dt > max_gap`` are excluded from the count, matching
        ``env.occupancy`` excluding that interval's time from the denominator
        (default 0.5). ``None`` disables gap gating on both sides.
    context : str, optional
        Label used in error messages to identify the calling function.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).
        Set to ``False`` to suppress all drop-related warnings (e.g.
        when the calling batch function will issue its own aggregate
        warning).

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
        Start allocation keeps complete intervals with both endpoints inside
        and assigns each interval, with its spikes, to its start sample's bin.
        Linear allocation uses exact retained planar segment support and
        assigns spikes to their interpolated position.
    linear_intervals : LinearSpatialIntervals or None
        Prepared linear support shared with occupancy and other units.

    Returns
    -------
    ndarray, shape (n_bins,)
        Number of spikes in each spatial bin (float64 for compatibility
        with smoothing operations).

    Raises
    ------
    ValueError
        If times and positions have different lengths.

    Notes
    -----
    A spike in sampling interval ``[times[k], times[k + 1])`` is counted in
    the same bin as that interval's dwell time: the bin of ``positions[k]``
    under ``"start"`` allocation, and the bin of the spike's linearly
    interpolated position under ``"linear"`` allocation (which splits the
    interval's time along the segment). Matching numerator and denominator
    keeps the rate of an untuned neuron flat. Spikes outside the trajectory
    time range are excluded, and spikes whose position is outside the
    environment bins are excluded from counts.

    The default ``warn_on_drop=True`` guards against common unit
    mismatches (e.g. spike_times in milliseconds while times is in
    seconds), which would silently produce a near-empty firing field.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._spatial_counts import bin_spike_train

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory and spikes
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5])

    >>> # Bin spikes
    >>> spike_counts = bin_spike_train(env, spike_times, times, trajectory)
    >>> spike_counts.shape[0] == env.n_bins
    True
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    _validate_time_allocation(time_allocation)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if len(times) != len(positions):
        raise ValueError(
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

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
                interval_mask=interval_mask,
            )
        interval_mask = linear_intervals.support.mask
    elif interval_mask is None:
        resolved_epochs, resolved_spike_window = resolve_time_windows(
            epochs, spike_window
        )
        interval_mask = _spatial_interval_mask(
            env,
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
        )

    spike_counts, n_time_dropped, n_bin_dropped, n_total, n_after_time = (
        _bin_spike_train_with_stats(
            env,
            spike_times,
            times,
            positions,
            interval_mask=interval_mask,
            linear_intervals=linear_intervals,
        )
    )

    if warn_on_drop:
        t_min, t_max = times.min(), times.max()
        _emit_time_window_warning(
            n_time_dropped,
            n_total,
            t_min,
            t_max,
            spike_times,
        )
        _emit_inactive_bin_warning(
            n_bin_dropped,
            n_after_time,
        )

    return spike_counts


def compute_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
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
    max_gap : float or None
        Maximum time gap in seconds (default
        0.5, matching ``env.occupancy``'s own default). ``None`` disables gap
        gating.

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
        raise ValueError(
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )

    # Handle 1D positions
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    require_position_columns(positions, env.n_dims, call=context)

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
                interval_mask=interval_mask,
            )
        occupancy: NDArray[np.float64] = np.array(
            linear_intervals.occupancy, dtype=np.float64, copy=True
        )
        return occupancy
    if interval_mask is None:
        resolved_epochs, resolved_spike_window = resolve_time_windows(
            epochs, spike_window
        )
        interval_mask = _spatial_interval_mask(
            env,
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
        )
    return start_allocated_occupancy(
        env.bin_at(positions),
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
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    interval_mask: NDArray[np.bool_] | None = None,
    time_allocation: Literal["start", "linear"] = "start",
    linear_intervals: LinearSpatialIntervals | None = None,
    n_jobs: int = 1,
    warn_on_drop: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains into spatial bins.

    Batch version of bin_spike_train that efficiently processes multiple
    neurons. Precomputes shared quantities (position bins, occupancy) and
    optionally parallelizes spike counting with joblib.

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
    max_gap : float or None
        Maximum time gap in seconds. Intervals with ``dt > max_gap`` are
        excluded from BOTH occupancy and every per-neuron spike count
        (default 0.5). ``None`` disables gap gating on both sides.
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
    bin_spike_train : Single-neuron version
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

    if time_allocation == "linear":
        if linear_intervals is None:
            linear_intervals = prepare_linear_spatial_intervals(
                env,
                times,
                positions,
                speed=resolved_speed,
                min_speed=min_speed,
                max_gap=max_gap,
                epochs=epochs,
                spike_window=spike_window,
                interval_mask=interval_mask,
            )
        interval_mask = linear_intervals.support.mask
    elif interval_mask is None:
        resolved_epochs, resolved_spike_window = resolve_time_windows(
            epochs, spike_window
        )
        interval_mask = _spatial_interval_mask(
            env,
            times,
            positions,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
        )
    occupancy = compute_occupancy(
        env,
        times,
        positions,
        interval_mask=interval_mask,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )

    # Spike-counting pass.  We use the private kernel _bin_spike_train_with_stats
    # which returns (counts, n_time_dropped, n_bin_dropped, n_total, n_after_time)
    # so that drop statistics are accumulated FOR FREE during the single counting
    # pass — no separate re-interpolation loop.  Workers are allowed to return
    # stats as data; only emitting warnings.warn() from a worker is forbidden
    # (they are commonly swallowed by joblib).

    # Pre-cast all spike arrays once (avoids repeated asarray inside kernel)
    spike_arrays = [np.asarray(spikes, dtype=np.float64) for spikes in spike_times_list]

    if n_jobs == 1:
        spike_counts = np.zeros((n_neurons, env.n_bins), dtype=np.float64)
        total_spikes = 0
        total_time_dropped = 0
        total_after_time = 0
        total_bin_dropped = 0

        for i, spikes in enumerate(spike_arrays):
            counts, n_td, n_bd, n_tot, n_at = _bin_spike_train_with_stats(
                env,
                spikes,
                times,
                positions,
                interval_mask=interval_mask,
                linear_intervals=linear_intervals,
            )
            spike_counts[i] = counts
            total_spikes += n_tot
            total_time_dropped += n_td
            total_after_time += n_at
            total_bin_dropped += n_bd
    else:
        from joblib import Parallel, delayed

        def _process_neuron(
            spikes: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], int, int, int, int]:
            # Return stats as data — do NOT call warnings.warn() from here.
            return _bin_spike_train_with_stats(
                env,
                spikes,
                times,
                positions,
                interval_mask=interval_mask,
                linear_intervals=linear_intervals,
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_neuron)(spikes) for spikes in spike_arrays
        )
        spike_counts = np.zeros((n_neurons, env.n_bins), dtype=np.float64)
        total_spikes = 0
        total_time_dropped = 0
        total_after_time = 0
        total_bin_dropped = 0
        for i, (counts, n_td, n_bd, n_tot, n_at) in enumerate(results):
            spike_counts[i] = counts
            total_spikes += n_tot
            total_time_dropped += n_td
            total_after_time += n_at
            total_bin_dropped += n_bd

    # Warn ONCE in the main process from the aggregated statistics.
    if warn_on_drop and n_neurons > 0:
        t_min, t_max = times.min(), times.max()

        if total_spikes > 0 and total_time_dropped > 0:
            all_spikes_cat = (
                np.concatenate([s for s in spike_arrays if len(s) > 0])
                if any(len(s) > 0 for s in spike_arrays)
                else None
            )
            _emit_time_window_warning(
                total_time_dropped,
                total_spikes,
                t_min,
                t_max,
                all_spikes_cat,
                scope="across all neurons ",
            )

        if total_after_time > 0 and total_bin_dropped > 0:
            _emit_inactive_bin_warning(
                total_bin_dropped,
                total_after_time,
                scope="across all neurons ",
            )

    return spike_counts, occupancy
