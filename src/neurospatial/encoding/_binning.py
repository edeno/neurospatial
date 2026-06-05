"""Binning layer for spatial encoding.

This module converts spike trains and trajectories into discrete spike counts
and occupancy arrays that can be processed by the smoothing layer.

The functions in this module handle:
1. Spike interpolation from continuous spike times to bin positions
2. Occupancy computation from continuous trajectory to time-per-bin
3. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons

The binning layer is intentionally separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing position bins for efficiency
- Future JAX implementations to leverage different parallelization strategies
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "_emit_all_excluded_speed_warning",
    "bin_spike_train",
    "bin_spike_trains",
    "compute_occupancy",
    "resolve_speed",
]


# _DROP_WARN_THRESHOLD is always < 1.0, so 100%-dropped (fraction == 1.0)
# always satisfies `frac > threshold` and never needs a separate == check.
_DROP_WARN_THRESHOLD = 0.5  # warn when fraction dropped exceeds this


def resolve_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
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
        ``env.occupancy`` (so ``max_gap`` etc. stay exactly as before — the
        output is byte-for-byte unchanged). It is an error to pass ``speed``
        without ``min_speed`` (see Raises): a bare ``speed`` would otherwise be
        silently ignored, asymmetric with ``env.occupancy`` (which raises on
        ``min_speed`` without ``speed``).

    Returns
    -------
    ndarray, shape (n_samples,), or None
        - ``None`` when ``min_speed is None`` (no filtering).
        - The validated ``speed`` array when one was passed.
        - Otherwise an auto-derived forward-difference speed array.

    Raises
    ------
    ValueError
        If ``speed`` is provided without ``min_speed`` (the message names both
        ``speed`` and ``min_speed``). This mirrors ``env.occupancy``, which
        raises on ``min_speed`` without ``speed``.
        If a ``speed`` array is passed whose length does not match
        ``times`` (the message names ``speed``).

    Notes
    -----
    **Auto-speed convention (forward difference).** When ``speed is None`` and
    ``min_speed is not None``, speed is derived from the trajectory with a
    FORWARD difference, matching ``env.occupancy``'s ``time_allocation="start"``
    interval semantics (interval ``k`` spans ``[t_k, t_{k+1})`` and is assigned
    to the bin at sample ``k``):

    .. math::
        \\text{speed}[k] = \\frac{\\lVert p_{k+1} - p_k \\rVert_2}
                                  {t_{k+1} - t_k}, \\quad k = 0 \\ldots n-2

    The final sample starts no occupancy interval, so its speed is defined by
    repeating the last finite-difference value: ``speed[n-1] = speed[n-2]``.

    This Euclidean default is intentionally simple. For geodesic / linearized
    track environments where straight-line distance is not the relevant speed,
    pass your own precomputed ``speed`` array instead of relying on the
    auto-derivation.
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
        # No filtering requested: return None so callers leave env.occupancy
        # exactly as before (byte-for-byte unchanged output).
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

    n_samples = len(times)
    if n_samples <= 1:
        # No intervals: speed is undefined; return zeros so a min_speed gate
        # consistently excludes the (non-existent) intervals on both sides.
        return np.zeros(n_samples, dtype=np.float64)

    dt = np.diff(times)
    # Forward differences in position (Euclidean) over each interval.
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    # Guard division against non-positive / non-finite dt (duplicate
    # timestamps). validate_trajectory enforces sorted/finite upstream, but
    # equal consecutive timestamps (dt == 0) are still possible.
    interval_speed = np.zeros(n_samples - 1, dtype=np.float64)
    valid_dt = dt > 0
    interval_speed[valid_dt] = step[valid_dt] / dt[valid_dt]

    speed_full = np.empty(n_samples, dtype=np.float64)
    speed_full[: n_samples - 1] = interval_speed
    # The final sample starts no occupancy interval; repeat the last value so
    # a spike landing exactly on the final sample is gated consistently.
    speed_full[n_samples - 1] = interval_speed[-1]
    return speed_full


def _bin_spike_train_with_stats(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
) -> tuple[
    NDArray[np.float64],  # spike_counts, shape (n_bins,)
    int,  # n_time_dropped
    int,  # n_bin_dropped
    int,  # n_total spikes
    int,  # n_after_time  (spikes surviving the time-window filter)
]:
    """Core spike-binning kernel: interp + bin_at, done exactly once.

    Private helper used by both the single-neuron public function
    (``bin_spike_train``) and the batch function (``bin_spike_trains``).
    Doing the interpolation and bin-mapping here – and returning the drop
    counts alongside the spike-count array – means the batch path can
    accumulate drop statistics for free during the single counting pass
    instead of repeating the O(spikes) work in a separate aggregation loop.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    spike_times : ndarray, shape (n_spikes,)
        Already cast to float64.
    times : ndarray, shape (n_samples,)
        Already cast to float64.
    positions : ndarray, shape (n_samples, n_dims)
        Already reshaped to 2-D and cast to float64.
    speed : ndarray, shape (n_samples,), or None
        Concrete, already-resolved speed array (from :func:`resolve_speed`).
        Consumed as-is; this kernel never re-derives speed.
    min_speed : float or None
        Speed threshold. When both ``speed`` and ``min_speed`` are provided,
        each time-window-valid spike is gated by the speed of the interval it
        falls in: for a spike at time ``t``, its interval index is
        ``k = searchsorted(times, t, side="right") - 1`` (clamped to
        ``[0, n_samples-1]``), and the spike is KEPT iff
        ``speed[k] >= min_speed``. This is the SAME per-interval gate that
        ``env.occupancy`` applies to occupancy (``speed[:-1] >= min_speed``),
        so the numerator (spikes) and denominator (occupancy) drop exactly the
        same intervals. Speed-excluded spikes are INTENTIONAL exclusions and
        are NOT counted as ``n_bin_dropped`` (the drop stats stay about
        time-window and inactive-bin drops only).

    Returns
    -------
    spike_counts : ndarray, shape (n_bins,)
    n_time_dropped : int
        Spikes outside the position time window.
    n_bin_dropped : int
        Time-valid (and speed-valid) spikes that mapped to
        inactive/out-of-environment bins.
    n_total : int
        Total number of spikes (= len(spike_times)).
    n_after_time : int
        Spikes surviving the time-window filter (and, when speed filtering is
        active, the speed gate) — i.e. the denominator for the inactive-bin
        drop fraction.
    """
    n_bins = env.n_bins
    spike_counts = np.zeros(n_bins, dtype=np.float64)
    n_total = len(spike_times)

    if n_total == 0:
        return spike_counts, 0, 0, 0, 0

    t_min, t_max = times.min(), times.max()
    valid_time_mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_valid = spike_times[valid_time_mask]
    n_time_dropped = n_total - len(spike_times_valid)

    # Speed gate: applied AFTER the time-window filter and BEFORE bin_at.
    # For each surviving spike at time t, find the interval it falls in
    #   k = searchsorted(times, t, side="right") - 1   (clamped to [0, n-2])
    # and keep it iff speed[k] >= min_speed. This is the SAME per-interval
    # gate env.occupancy uses (speed[:-1] >= min_speed), so spikes and
    # occupancy drop exactly the same intervals — numerator/denominator stay
    # aligned. Speed-excluded spikes are INTENTIONAL: they are NOT counted as
    # dropped (they do not inflate n_bin_dropped or its warning). The full
    # trajectory is still used for position interpolation below; only which
    # spikes survive changes.
    #
    # The upper clip is n-2 (the index of the LAST occupancy interval), not
    # n-1: a spike landing exactly on times[-1] would otherwise be gated by
    # speed[n-1], which occupancy never consults (it gates the last interval by
    # speed[n-2] via speed[:-1]). For auto-derived speed this is harmless
    # (speed[n-1] == speed[n-2]), but a caller-supplied speed with a differing
    # final element could drop a t_max spike while occupancy keeps the matching
    # last interval. Clipping to n-2 gates the t_max spike by speed[n-2],
    # matching occupancy's last interval exactly. (Guarded for n >= 2; for
    # n == 1 there are no intervals and this branch's speed gate never excludes
    # — n-2 == -1 only when len(times) == 1, which is handled by resolve_speed
    # returning all-zeros for that degenerate case.)
    if speed is not None and min_speed is not None and len(spike_times_valid) > 0:
        spike_interval = np.searchsorted(times, spike_times_valid, side="right") - 1
        upper = max(len(times) - 2, 0)
        spike_interval = np.clip(spike_interval, 0, upper)
        speed_keep = speed[spike_interval] >= min_speed
        spike_times_valid = spike_times_valid[speed_keep]

    n_after_time = len(spike_times_valid)

    if n_after_time == 0:
        return spike_counts, n_time_dropped, 0, n_total, 0

    # Linearly interpolate spike positions from the trajectory in each
    # dimension, then map to a bin.  Using the most recent trajectory frame
    # (snapshot) instead of interp would shift spikes that fall between
    # samples to the previous bin under fast movement / sparse sampling.
    n_dims = positions.shape[1]
    spike_positions = np.empty((n_after_time, n_dims), dtype=np.float64)
    for d in range(n_dims):
        spike_positions[:, d] = np.interp(spike_times_valid, times, positions[:, d])

    spike_bins = env.bin_at(spike_positions)
    valid_bins = spike_bins[spike_bins >= 0]
    n_bin_dropped = n_after_time - len(valid_bins)

    if len(valid_bins) > 0:
        spike_counts = np.bincount(valid_bins, minlength=n_bins).astype(np.float64)

    return spike_counts, n_time_dropped, n_bin_dropped, n_total, n_after_time


def _emit_all_excluded_speed_warning(
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a UserWarning when ``min_speed`` excludes (almost) all intervals.

    A ``min_speed`` set too high — or in the wrong units (e.g. an m/s threshold
    against a cm/s trajectory) — can exclude EVERY movement interval, leaving an
    all-zero occupancy and an all-NaN/zero rate map with no other signal. This
    mirrors the unit-mismatch tone of the time-window / inactive-bin warnings.

    Detection uses the resolved per-interval speed gate
    (``speed[:-1] >= min_speed``) — the exact same mask ``env.occupancy`` and the
    spike kernel apply — so it fires iff the rate map is genuinely empty.

    Parameters
    ----------
    speed : ndarray, shape (n_samples,), or None
        Resolved speed array (from :func:`resolve_speed`). ``None`` means no
        filtering was requested; nothing is emitted.
    min_speed : float or None
        Speed threshold. ``None`` means no filtering; nothing is emitted.
    stacklevel : int, optional
        ``warnings.warn`` stacklevel.
    """
    if speed is None or min_speed is None:
        return
    # Per-interval gate: interval k spans [t_k, t_{k+1}) and uses speed[k],
    # so only the first n-1 entries gate occupancy (matching env.occupancy).
    interval_speed = speed[:-1]
    if interval_speed.size == 0:
        return
    if np.any(interval_speed >= min_speed):
        return
    finite = interval_speed[np.isfinite(interval_speed)]
    if finite.size > 0:
        s_min = float(finite.min())
        s_max = float(finite.max())
        range_part = f"speed in [{s_min:.3g}, {s_max:.3g}] units/s"
    else:
        range_part = "no finite speeds"
    warnings.warn(
        f"min_speed={min_speed} excluded all trajectory intervals "
        f"({range_part}); the rate map is empty. Check that min_speed and "
        f"the trajectory share units. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _emit_time_window_warning(
    n_time_dropped: int,
    n_total: int,
    t_min: float,
    t_max: float,
    all_spike_times: NDArray[np.float64] | None,
    *,
    scope: str = "",
    stacklevel: int = 2,
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
    stacklevel : int, optional
        ``warnings.warn`` stacklevel.
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
        f"{n_time_dropped}/{n_total} spike_times "
        f"({100 * frac:.0f}%) {scope}fell outside the position time "
        f"window [{t_min:.6g}, {t_max:.6g}]; "
        f"{range_part}"
        f"Check that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _emit_inactive_bin_warning(
    n_bin_dropped: int,
    n_after_time: int,
    *,
    scope: str = "",
    stacklevel: int = 2,
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
    stacklevel : int, optional
        ``warnings.warn`` stacklevel.
    """
    if n_after_time == 0 or n_bin_dropped == 0:
        return
    frac = n_bin_dropped / n_after_time
    if frac <= _DROP_WARN_THRESHOLD:
        return
    warnings.warn(
        f"{n_bin_dropped}/{n_after_time} spikes "
        f"({100 * frac:.0f}%) {scope}interpolated to positions outside "
        f"the active environment bins (bin index -1). "
        f"Check that positions are in the same coordinate frame as the "
        f"environment and that spike_times and times share units (both seconds). "
        f"Dropped spikes do not contribute. "
        f"Set warn_on_drop=False to suppress this warning.",
        UserWarning,
        stacklevel=stacklevel,
    )


def bin_spike_train(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    context: str = "bin_spike_train",
    warn_on_drop: bool = True,
) -> NDArray[np.float64]:
    """Bin spike train into spatial bins.

    Converts continuous spike times to spike counts per spatial bin by
    interpolating spike positions from the trajectory and counting spikes
    in each bin.

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
    Spikes are interpolated to positions using linear interpolation of
    spike times onto the trajectory. Spikes outside the trajectory time
    range are excluded. Spikes that fall in invalid bins (outside the
    environment) are excluded from counts.

    The default ``warn_on_drop=True`` guards against common unit
    mismatches (e.g. spike_times in milliseconds while times is in
    seconds), which would silently produce a near-empty firing field.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._binning import bin_spike_train

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
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if len(times) != len(positions):
        raise ValueError(
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    spike_counts, n_time_dropped, n_bin_dropped, n_total, n_after_time = (
        _bin_spike_train_with_stats(
            env, spike_times, times, positions, speed=speed, min_speed=min_speed
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
            stacklevel=2,
        )
        _emit_inactive_bin_warning(
            n_bin_dropped,
            n_after_time,
            stacklevel=2,
        )

    return spike_counts


def compute_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
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
        :func:`resolve_speed`). Forwarded to ``env.occupancy``; when provided
        with ``min_speed``, interval ``k`` is gated by ``speed[k] >= min_speed``.
    min_speed : float or None
        Minimum speed threshold. When ``None`` (default), no speed filtering
        is applied and nothing speed-related is passed to ``env.occupancy``, so
        the result is byte-for-byte identical to before.

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
    Delegates to Environment.occupancy() which handles:
    - Time interval allocation to bins
    - Speed filtering (if configured)
    - Gap handling
    - Kernel smoothing (if configured)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._binning import compute_occupancy

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

    # Check position dimensions match environment
    if positions.shape[1] != env.n_dims:
        raise ValueError(
            f"in {context}: positions have {positions.shape[1]} dimensions "
            f"but environment has {env.n_dims} dimensions"
        )

    # Delegate to Environment.occupancy() which handles all the complexity.
    # When min_speed is None we pass nothing speed-related so the call is
    # byte-for-byte identical to before (max_gap defaults etc. untouched).
    if min_speed is None:
        occupancy = cast("EnvironmentProtocol", env).occupancy(
            times, positions, return_seconds=True
        )
    else:
        occupancy = cast("EnvironmentProtocol", env).occupancy(
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            return_seconds=True,
        )

    return occupancy.astype(np.float64)


def bin_spike_trains(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
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
        (default) no speed filtering happens and the output is byte-for-byte
        unchanged. When set, low-speed intervals are excluded from BOTH
        occupancy and spike counts.
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
    >>> from neurospatial.encoding._binning import bin_spike_trains

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
    from neurospatial.encoding._spikes import as_spike_trains

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
    resolved_speed = resolve_speed(times, positions, speed, min_speed)

    # Occupancy is independent of which neuron we're binning, so compute once.
    # Spike binning itself depends on per-neuron spike_times (interpolated to
    # spike positions), so it stays inside the per-neuron loop.
    occupancy = compute_occupancy(
        env, times, positions, speed=resolved_speed, min_speed=min_speed
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
                speed=resolved_speed,
                min_speed=min_speed,
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
                speed=resolved_speed,
                min_speed=min_speed,
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
                stacklevel=2,
            )

        if total_after_time > 0 and total_bin_dropped > 0:
            _emit_inactive_bin_warning(
                total_bin_dropped,
                total_after_time,
                scope="across all neurons ",
                stacklevel=2,
            )

    return spike_counts, occupancy
