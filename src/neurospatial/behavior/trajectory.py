"""Trajectory characterization metrics.

This module provides functions for analyzing spatial trajectories, including
turn angles, step lengths, home range estimation, and mean square displacement
analysis. These metrics are commonly used in behavioral ecology and neuroscience
to characterize movement patterns and spatial exploration.

References
----------
.. [1] Traja: https://traja.readthedocs.io/ - Trajectory analysis toolkit
.. [2] Fiaschi, L., Nair, R., Koethe, U., & Hamprecht, F. A. (2012).
       "Learning to count with regression forest and structured labels."
       ICPR 2012.
.. [3] Kays, R., Crofoot, M. C., Jetz, W., & Wikelski, M. (2015).
       "Terrestrial animal tracking as an eye on life and planet."
       Science, 348(6240), aaa2478.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._observation import ObservationSupport, validated_observations
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial.environment import Environment


@dataclass(frozen=True, repr=False)
class MSDResult(ReadOnlyArraysMixin, ResultMixin):
    """Mean square displacement curve returned by ``mean_square_displacement``.

    Attributes
    ----------
    lags : NDArray[np.float64], shape (n_lags,)
        Lag times τ (in the same units as the input ``times`` array) at
        which MSD was computed. Always strictly increasing and starts at
        the median sample interval.
    msd : NDArray[np.float64], shape (n_lags,)
        Mean square displacement values, one per lag. NaN when no valid
        displacement pair supports that lag.
    """

    lags: NDArray[np.float64]
    msd: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of lags and displacement values."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_lags": int(self.lags.size),
            "max_lag": float(self.lags.max()) if self.lags.size else float("nan"),
            "final_msd": float(self.msd[-1]) if self.msd.size else float("nan"),
        }


@dataclass(frozen=True, repr=False)
class CumulativeDistanceResult(ReadOnlyArraysMixin, ResultMixin):
    """Observed distance travelled, returned by ``compute_cumulative_distance``.

    Distances are Euclidean arclength along the observed polyline, in the
    units of the positions. Intervals that are not usable (a missing sample, a
    gap longer than ``max_gap``, outside ``epochs``, or a declared break)
    add nothing, so distance never crosses a break.

    Attributes
    ----------
    step_lengths : NDArray[np.float64], shape (n_samples - 1,)
        Length of each interval; NaN on unusable intervals.
    run_distance : NDArray[np.float64], shape (n_samples,)
        Distance since the start of the sample's run (0 at each run start);
        NaN where ``run_ids == -1``.
    cumulative_distance : NDArray[np.float64], shape (n_samples,)
        Observed distance since the first sample, summed over all runs so far;
        NaN where ``run_ids == -1``.
    run_ids : NDArray[np.intp], shape (n_samples,)
        Run index in time order; -1 for samples that touch no usable interval.
    sample_valid : NDArray[np.bool_], shape (n_samples,)
        True where every coordinate is finite.
    run_sample_bounds : NDArray[np.intp], shape (n_runs, 2)
        Inclusive ``(first, last)`` sample indices of each run.
    run_time_bounds : NDArray[np.float64], shape (n_runs, 2)
        Start and end time of each run, in seconds.
    run_path_lengths : NDArray[np.float64], shape (n_runs,)
        Distance travelled in each run.
    max_gap : float or None
        The gap limit that was applied, in seconds.
    epochs : NDArray[np.float64] or None, shape (n_epochs, 2)
        Normalized half-open ``[start, stop)`` windows that were applied.
    """

    step_lengths: NDArray[np.float64]
    run_distance: NDArray[np.float64]
    cumulative_distance: NDArray[np.float64]
    run_ids: NDArray[np.intp]
    sample_valid: NDArray[np.bool_]
    run_sample_bounds: NDArray[np.intp]
    run_time_bounds: NDArray[np.float64]
    run_path_lengths: NDArray[np.float64]
    max_gap: float | None
    epochs: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    @property
    def total_distance(self) -> float:
        """Distance over all runs; NaN when no interval was usable."""
        if self.run_path_lengths.size == 0:
            return float("nan")
        return float(self.run_path_lengths.sum())

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        ``n_samples`` counts the input samples, one per entry of ``run_ids``.
        """
        widths = self.run_time_bounds[:, 1] - self.run_time_bounds[:, 0]
        return {
            "n_samples": int(self.run_ids.size),
            "n_runs": int(self.run_path_lengths.size),
            "n_isolated_samples": int(
                np.count_nonzero(self.sample_valid & (self.run_ids == -1))
            ),
            "total_distance": self.total_distance,
            "observed_duration": float(widths.sum()),
        }


def compute_speed(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Running speed at each tracking sample, robust to tracking jitter.

    The velocity vector is estimated by centered differences and smoothed
    with a Gaussian of ``smooth_window`` seconds before its magnitude is
    taken. Differencing raw positions instead turns camera jitter into
    speed: an animal sitting still with 0.5 cm of jitter reads about
    20 cm/s at 30 Hz and 200 cm/s at 500 Hz, so a ``min_speed`` filter would
    keep nearly all of its rest. Every ``min_speed`` gate in the package
    (rate maps, decoding, ``segment_by_velocity``, decision and goal
    metrics) uses this speed, so passing ``speed=compute_speed(...)``
    explicitly gives the same result as leaving it out, provided the gate
    gets the same ``break_before``.

    Speed is never smoothed across a declared break, a missing sample, a gap
    longer than ``max_gap`` or the edge of an ``epochs`` window.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Positions in environment units (e.g. cm). A 1-D array is one
        coordinate (for example a linearized track). Rows with a non-finite
        coordinate are missing.
    smooth_window : float, default=0.1
        Standard deviation, in seconds, of the Gaussian smoothing applied to
        each velocity component (the default of spyglass's position
        pipeline). 0 gives the unsmoothed centered-difference speed.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Each run is smoothed separately. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows
        (seconds, same clock as ``times``). ``None`` (default) means
        unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    speed : NDArray[np.float64], shape (n_samples,)
        Speed in position units per second (e.g. cm/s). NaN at missing
        samples and at samples in no run (isolated by a gap, a declared break,
        a missing neighbour or ``epochs``).

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` are empty, swapped or misaligned; if
        ``times`` repeats or is not increasing; if ``max_gap`` or
        ``break_before`` is invalid; or if ``smooth_window`` is negative.

    See Also
    --------
    segment_by_velocity : Running bouts from this speed.
    heading_from_velocity : Movement direction from the same smoothed
        velocity.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import compute_speed
    >>> times = np.arange(0.0, 10.0, 1 / 30)
    >>> rng = np.random.default_rng(0)
    >>> still = 50.0 + rng.normal(0.0, 0.5, (len(times), 2))  # 0.5 cm jitter
    >>> bool(np.median(compute_speed(times, still)) < 2.0)
    True
    >>> bool(np.median(compute_speed(times, still, smooth_window=0.0)) > 10.0)
    True

    The jump to where the animal was put back reads as 98 cm/s unless the
    break is declared.

    >>> times = np.arange(6) / 4  # seconds
    >>> x = np.array([0.0, 1.0, 2.0, 50.0, 51.0, 52.0])  # cm; new recording at sample 3
    >>> compute_speed(times, x, smooth_window=0.0).tolist()
    [4.0, 4.0, 98.0, 98.0, 4.0, 4.0]
    >>> new_recording = np.array([False, False, False, True, False, False])
    >>> compute_speed(times, x, smooth_window=0.0, break_before=new_recording).tolist()
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    """
    from neurospatial._velocity import smoothed_speed

    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call="compute_speed",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every speed is NaN",
    )
    return smoothed_speed(
        times_arr, positions_arr, support.interval_mask, smooth_window=smooth_window
    )


def heading_from_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    min_speed: float = 0.0,
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
    allow_all_nan: bool = False,
) -> NDArray[np.float64]:
    """Compute heading from position timeseries using velocity direction.

    Parameters
    ----------
    times : array-like, shape (n_time,)
        Finite, strictly increasing timestamps in seconds, one per position.
    positions : NDArray, shape (n_time, 2)
        Animal positions over time, in environment units (e.g. cm). A row with
        a non-finite coordinate (NaN, for a lost frame) is a missing sample:
        it has no heading, and no velocity is computed across it.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted. Windows that
        touch merge into one, so to keep recordings joined on one clock apart
        use ``break_before``.
    break_before : ArrayLike of bool, shape (n_time,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No velocity or smoothing spans a declared
        break.
    min_speed : float, default 0.0
        Minimum speed threshold in **the same units per second as
        ``positions``** (e.g. cm/s if positions are in cm). Samples slower
        than this have no heading (NaN). Samples with zero speed have no
        heading at any threshold. The velocity is smoothed, so at the default
        of 0 most of a short stop still has a heading; set ``min_speed`` (for
        example 2 cm/s) to mark stops as having no heading.
    smooth_window : float, default 0.1
        Standard deviation, in seconds, of the Gaussian that smooths the
        velocity before its direction and speed are taken, the same
        smoothing as :func:`neurospatial.behavior.compute_speed`; without
        it, tracking jitter at rest gives random headings that pass
        ``min_speed``. Set to 0 to disable smoothing.
    allow_all_nan : bool, default False
        Controls the degenerate case where **every observed** sample is below
        ``min_speed`` or still (heading undefined everywhere). ``False`` (the
        default) raises ``ValueError`` so the failure is loud; ``True`` returns
        an all-NaN array with a ``UserWarning`` instead, for batch pipelines
        that handle NaN explicitly.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint, in the **allocentric
        world-frame convention** (0 = East, π/2 = North, π = West,
        -π/2 = South), wrapped to ``[-π, π]`` per ``numpy.arctan2``
        (so westward motion returns +π, not -π). NaN at missing samples, at
        samples in no run (an isolated sample, or a single-sample input), and
        at samples slower than ``min_speed`` or with zero speed.

    Raises
    ------
    ValueError
        If there are no samples, if positions is not 2-D with at least x/y
        coordinates, if times is not 1-D, finite, strictly increasing (no
        repeated timestamp) and aligned with positions, if ``break_before`` is
        not a Boolean array with one entry per sample, or if every observed
        sample is below ``min_speed`` or still and ``allow_all_nan`` is
        ``False`` (the default).

    Warns
    -----
    UserWarning
        If every observed sample is below ``min_speed`` or still and
        ``allow_all_nan=True`` (an all-NaN heading array is returned).

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``
    and not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause or a declared break.

    Velocity is the centered secant of position over each sample's actual
    elapsed time (one-sided at a run's ends), smoothed separately within each
    run. Samples touching no valid interval have NaN headings. If no samples
    belong to any run, the result is all NaN.

    Movement heading is the direction of the velocity vector, so it is not
    observed while the animal is slower than ``min_speed`` or still, and the
    function returns NaN there rather than an interpolated direction the
    animal may never have faced. For a heading while the animal is still, use
    :func:`heading_from_body_orientation` with pose keypoints.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import heading_from_velocity

    Trajectory moving East:

    >>> t = np.linspace(0, 10, 100)
    >>> positions = np.column_stack([t * 10, np.zeros_like(t)])
    >>> headings = heading_from_velocity(t, positions)
    >>> np.allclose(headings[10:-10], 0.0, atol=0.1)
    True

    Trajectory moving North:

    >>> positions = np.column_stack([np.zeros_like(t), t * 10])
    >>> headings = heading_from_velocity(t, positions)
    >>> np.allclose(headings[10:-10], np.pi / 2, atol=0.1)
    True

    Two recordings joined on one clock: the animal runs East, then is put
    back 40 cm to the west and runs North. Undeclared, the jump reads as
    running West; declared with ``break_before``, it does not.

    >>> times = np.arange(8) / 4  # s
    >>> x = np.array([0.0, 1.0, 2.0, 3.0, -37.0, -37.0, -37.0, -37.0])  # cm
    >>> y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])  # cm
    >>> positions = np.column_stack([x, y])
    >>> new_recording = times == 1.0
    >>> np.round(heading_from_velocity(times, positions, smooth_window=0.0), 2).tolist()
    [0.0, 0.0, 0.0, 3.14, 3.12, 1.57, 1.57, 1.57]
    >>> np.round(
    ...     heading_from_velocity(
    ...         times, positions, smooth_window=0.0, break_before=new_recording
    ...     ),
    ...     2,
    ... ).tolist()
    [0.0, 0.0, 0.0, 0.0, 1.57, 1.57, 1.57, 1.57]

    A lost frame is a NaN row. It has no heading, and no velocity is
    computed across it:

    >>> times = np.arange(8) * 0.1  # s
    >>> positions = np.column_stack([np.arange(8.0), np.zeros(8)])  # cm
    >>> positions[3] = np.nan
    >>> heading_from_velocity(times, positions, smooth_window=0.0).tolist()
    [0.0, 0.0, 0.0, nan, 0.0, 0.0, 0.0, 0.0]

    The animal runs East, stops for 0.3 s and runs back West. While it is
    slower than ``min_speed`` its direction of movement is not observed:

    >>> times = np.arange(7) * 0.1  # s
    >>> positions = np.column_stack(
    ...     [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], np.zeros(7)]
    ... )  # cm
    >>> np.round(heading_from_velocity(times, positions, min_speed=2.0), 2).tolist()
    [0.0, 0.0, nan, nan, 3.14, 3.14, 3.14]
    """
    from neurospatial._observation import validated_observations
    from neurospatial.behavior._kinematics import (
        _movement_heading_and_speed,
        _validate_velocity_positions,
    )

    # Validate first so a swapped (positions, times) call gets that message.
    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        raw_positions,
        call="heading_from_velocity",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every heading is NaN",
    )
    _validate_velocity_positions(raw_positions, call="heading_from_velocity")
    heading, speed = _movement_heading_and_speed(
        times, positions, support.interval_mask, smooth_window=smooth_window
    )
    # The velocity direction is observed only where the animal moves.
    moving = (speed > 0) & (speed >= min_speed)
    heading[~moving] = np.nan
    in_run = support.run_ids >= 0
    if in_run.any() and not moving.any():
        fastest = float(np.max(speed[in_run]))
        if not allow_all_nan:
            fix = (
                "lower min_speed in position-units per second, or pass "
                "allow_all_nan=True for a batch pipeline that handles NaN."
                if min_speed > 0
                else "the animal never moves, so pass allow_all_nan=True for a "
                "batch pipeline that handles NaN, or use "
                "heading_from_body_orientation() with pose keypoints."
            )
            raise ValueError(
                f"Cannot compute heading: every observed sample's speed is "
                f"below min_speed={min_speed} or zero (the fastest is "
                f"{fastest:.4g}).\n"
                "Why: velocity direction is undefined for an all-stationary "
                f"or too-slow trajectory.\nFix: {fix}"
            )
        warnings.warn(
            f"Every observed sample's speed (max {fastest:.4g}) is below "
            f"min_speed={min_speed} or zero; returning an all-NaN heading "
            "array because allow_all_nan=True.",
            UserWarning,
            stacklevel=2,
        )
    return heading


def _interpolate_heading_circular(
    heading: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Interpolate masked headings along the shorter arc, linearly in angle.

    Unwrap consecutive finite anchors, interpolate angles, then wrap to
    (-pi, pi]. For an exactly antipodal pair, the sign of the stored
    difference chooses the turn. Samples beyond the anchors keep the nearest
    valid heading. Unmasked non-finite values are not interpolation anchors.

    Parameters
    ----------
    heading : NDArray, shape (n_time,)
        Heading values in radians.
    mask : NDArray, shape (n_time,)
        Boolean mask where True indicates values to interpolate.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading with masked values interpolated.
    """
    if not np.any(mask):
        return heading

    valid = ~mask & np.isfinite(heading)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return heading
    unwrapped = np.unwrap(heading[valid_idx])
    filled = np.interp(np.flatnonzero(mask), valid_idx, unwrapped)
    result: NDArray[np.float64] = heading.copy()
    result[mask] = np.pi - np.mod(np.pi - filled, 2.0 * np.pi)
    return result


def heading_from_body_orientation(
    nose: NDArray[np.float64],
    tail: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute heading from nose and tail keypoints.

    Parameters
    ----------
    nose : NDArray, shape (n_time, 2)
        Nose keypoint positions. May contain NaN values.
    tail : NDArray, shape (n_time, 2)
        Tail keypoint positions. May contain NaN values.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint. NaN keypoints are
        interpolated along the shorter arc, linearly in angle.

    Raises
    ------
    ValueError
        If all keypoints are NaN.

    Notes
    -----
    NaN keypoints are interpolated by sample index across any gap, including a
    gap that spans a recording break; this function has no ``times`` argument
    and cannot see breaks. When several recordings are concatenated, call it
    once per recording segment, splitting where a recording-id array (or a
    ``break_before`` array) changes, so that no fill bridges two recordings::

        starts = np.flatnonzero(break_before)  # break_before[0] is ignored
        segments = np.split(np.arange(len(nose)), starts[starts > 0])
        headings = np.concatenate(
            [heading_from_body_orientation(nose[s], tail[s]) for s in segments]
        )

    Do not split at every missing frame (for example at the runs of
    ``compute_cumulative_distance``): that would leave the NaN frames unfilled.
    If a gap within a recording is too long for the filled heading to be
    trusted, set the samples inside it back to NaN afterwards.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import heading_from_body_orientation

    Heading from nose/tail pointing East:

    >>> n = 50
    >>> nose = np.tile([10.0, 0.0], (n, 1))
    >>> tail = np.tile([0.0, 0.0], (n, 1))
    >>> headings = heading_from_body_orientation(nose, tail)
    >>> np.allclose(headings, 0.0)
    True
    """
    nose = np.asarray(nose, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)

    # Compute body vector: nose - tail
    body_vector = nose - tail

    # Identify NaN samples
    nan_mask = np.any(np.isnan(body_vector), axis=1)

    if np.all(nan_mask):
        raise ValueError(
            "Cannot compute heading: all keypoints are NaN.\n\n"
            "WHAT: Both nose and tail positions are NaN at all timepoints\n"
            "WHY: Need at least one valid (nose, tail) pair for body orientation\n\n"
            "Fix:\n"
            "1. Check pose estimation output for tracking failures\n"
            "2. Verify keypoint extraction completed successfully\n"
            "3. Consider using heading_from_velocity() if pose data unavailable"
        )

    # Compute heading where valid
    heading = np.arctan2(body_vector[:, 1], body_vector[:, 0])

    # Interpolate NaN samples using circular interpolation
    if np.any(nan_mask):
        heading = _interpolate_heading_circular(heading, nan_mask)

    return heading


def compute_turn_angles(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute turn angles between consecutive movement vectors from continuous positions.

    Turn angles quantify changes in movement direction at each position along
    the trajectory. A turn angle of 0 indicates straight movement, positive
    angles indicate left turns, and negative angles indicate right turns.

    This function operates on continuous position data, preserving sub-bin spatial
    precision for accurate directional analysis. This is the ecology-standard
    approach and is more accurate than bin-based computation.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space. Each row is a position vector.
        For 2D trajectories, shape is (n_samples, 2). For 1D, shape is (n_samples, 1).

    Returns
    -------
    NDArray[np.float64], shape (n_angles,)
        Turn angles in radians, in the range [-π, π]. Consecutive duplicate
        positions (stationary periods) are dropped first, so the length is
        ``n_unique - 2``, where ``n_unique`` is the number of positions left,
        not ``n_samples - 2``. The output is not aligned with the input samples
        and carries no times; use ``compute_planar_turn_angles`` for a
        sample-aligned result.

    Notes
    -----
    Turn angles are computed from the angle between consecutive movement vectors:

    .. math::

        \\theta_i = \\text{atan2}(v_i \\times v_{i+1}, v_i \\cdot v_{i+1})

    where :math:`v_i` is the movement vector from position i-1 to position i.

    With ``v1`` and ``v2`` the incoming and outgoing movement vectors, the
    angle is ``atan2(v1_x * v2_y - v1_y * v2_x, sum(v1 * v2))``. The cross
    product uses the first two columns (x, y); the dot product sums over all
    columns. For positions with more than two columns this mixes a projected
    cross product with a full-dimensional dot product, so the result is neither
    the angle in the x-y plane nor the angle in 3-D space. The positions
    ``[[0, 0, 0], [1, 0, 1], [1, 1, 2]]`` give ``π/4`` here, whereas the x-y
    projection ``positions[:, :2]`` gives ``π/2``. For a horizontal-plane
    analysis pass ``positions[:, :2]`` explicitly.

    Stationary periods (consecutive duplicate positions) are removed before
    computing turn angles to avoid undefined angles from zero-length vectors.

    **Why continuous positions?**

    Using continuous positions instead of discretized bins preserves directional
    precision. For example, a gradual turn from [0,0]→[1,0.1]→[2,0.3] would all
    map to the same bins on a coarse grid, giving zero turn angle (incorrect),
    whereas continuous computation correctly captures the subtle turning.

    See Also
    --------
    compute_planar_turn_angles : Sample-aligned planar turning angles that
        respect time gaps, missing samples and recording breaks.
    compute_planar_curvature : Signed geometric curvature (inverse length).
    compute_movement_angular_velocity : Turning rate of the movement direction.
    compute_step_lengths : Compute distances between consecutive positions
    mean_square_displacement : Analyze diffusive properties of trajectory

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_turn_angles
    >>>
    >>> # Straight line trajectory
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> angles = compute_turn_angles(positions)
    >>> np.allclose(angles, 0.0, atol=0.01)  # Straight = no turning
    True
    >>>
    >>> # Right-angle turn: [0,0] → [10,0] → [10,10]
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    >>> angles = compute_turn_angles(positions)
    >>> float(angles[0])  # π/2: a 90° left turn
    1.5707963267948966
    >>>
    >>> # With more than two columns the cross product uses x, y only but the
    >>> # dot product uses every column
    >>> positions_3d = np.array([[0.0, 0, 0], [1, 0, 1], [1, 1, 2]])
    >>> float(compute_turn_angles(positions_3d)[0])  # π/4
    0.7853981633974483
    >>> float(compute_turn_angles(positions_3d[:, :2])[0])  # x-y projection, π/2
    1.5707963267948966

    References
    ----------
    .. [1] Batschelet, E. (1981). Circular Statistics in Biology.
           Academic Press.
    .. [2] Muller, M., & Wehner, R. (1988). "Path integration in desert ants."
           PNAS, 85(14), 5287-5290.
    .. [3] Traja documentation: https://traja.readthedocs.io/
    """
    # Coerce array-likes (e.g. Python lists) before any .ndim access.
    try:
        positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as e:
        actual_type = type(positions).__name__
        raise TypeError(
            f"positions must be a numeric array-like object (e.g., numpy array, "
            f"list of lists, pandas DataFrame). Got {actual_type}: {positions!r}"
        ) from e

    # Input validation
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if len(positions) < 3:
        # Need at least 3 positions to compute turn angles
        return np.array([], dtype=np.float64)

    # Remove consecutive duplicates (stationary periods)
    # Compare all dimensions for exact equality
    is_different = np.any(positions[1:] != positions[:-1], axis=1)
    # Keep first position and all positions that differ from previous
    mask = np.concatenate([[True], is_different])
    unique_positions = positions[mask]

    # Need at least 3 unique positions to compute turn angles
    if len(unique_positions) < 3:
        return np.array([], dtype=np.float64)

    # Compute movement vectors (differences between consecutive positions)
    vectors = np.diff(unique_positions, axis=0)  # shape (n_unique-1, n_dims)

    # Need at least 2 vectors to compute turn angles
    if len(vectors) < 2:
        return np.array([], dtype=np.float64)

    # Vectorized turn angle computation
    v1 = vectors[:-1]  # shape (n_angles, n_dims)
    v2 = vectors[1:]  # shape (n_angles, n_dims)

    # Compute vector norms
    norms1 = np.linalg.norm(v1, axis=1)
    norms2 = np.linalg.norm(v2, axis=1)

    # Handle zero-length vectors
    zero_mask = (norms1 == 0) | (norms2 == 0)

    n_dims = vectors.shape[1]
    angles: NDArray[np.float64]
    if n_dims >= 2:
        # 2D cross product (scalar): v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        # Dot product
        dot = np.sum(v1 * v2, axis=1)
        angles = np.arctan2(cross, dot)
    else:
        # 1D: angle is 0 if same direction, π if opposite
        same_direction = v1[:, 0] * v2[:, 0] > 0
        angles = np.asarray(np.where(same_direction, 0.0, np.pi))

    # Set zero-length vector pairs to 0
    angles[zero_mask] = 0.0

    return angles


def _planar_turns(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    call: str,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Incoming and outgoing displacement at each sample of a planar path.

    Returns
    -------
    times : NDArray[np.float64], shape (n_samples,)
        Validated sample times in seconds.
    a, b : NDArray[np.float64], shape (n_samples, 2)
        ``positions[i] - positions[i - 1]`` and ``positions[i + 1] - positions[i]``;
        NaN rows where ``supported`` is False.
    supported : NDArray[np.bool_], shape (n_samples,)
        True at interior samples whose two neighbouring intervals are both
        usable and whose two displacements are both nonzero.
    """
    # Validate first so a swapped (positions, times) call gets that message.
    t, p, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no turning was observed",
    )
    shape = np.shape(positions)
    if len(shape) != 2 or shape[1] != 2:
        raise ValueError(
            _format_error(
                f"{call}() needs positions with exactly two columns (x, y); "
                f"got shape {shape}.",
                why=(
                    "a signed turn, planar curvature and the angular velocity of "
                    "the movement direction are defined for two coordinates, and "
                    "the legacy N-D turn formulas are neither a projection nor "
                    "3-D curvature."
                ),
                fix="pass positions[:, :2] to analyze movement in the horizontal plane",
            )
        )
    n_samples = t.size
    a = np.full((n_samples, 2), np.nan)
    b = np.full((n_samples, 2), np.nan)
    supported = np.zeros(n_samples, dtype=bool)
    if n_samples >= 3:
        with np.errstate(invalid="ignore"):  # inf - inf on missing rows
            a[1:-1] = p[1:-1] - p[:-2]
            b[1:-1] = p[2:] - p[1:-1]
        supported[1:-1] = support.interval_mask[:-1] & support.interval_mask[1:]
        supported &= np.any(a != 0, axis=1) & np.any(b != 0, axis=1)
    a[~supported] = np.nan
    b[~supported] = np.nan
    return t, a, b, supported


def _cross_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z component of the cross product ``a x b``; ``a``, ``b`` are ``(n, 2)``."""
    cross: NDArray[np.float64] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return cross


def _signed_turns(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Turn from ``a`` to ``b`` in ``(-pi, pi]``; ``a``, ``b`` are ``(n, 2)``."""
    angles: NDArray[np.float64] = np.arctan2(_cross_2d(a, b), np.sum(a * b, axis=1))
    angles[angles == -np.pi] = np.pi  # a reversal is pi, not -pi
    return angles


def compute_planar_turn_angles(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute the signed turn in movement direction at each tracking sample.

    At sample ``i`` the turn is the angle from the incoming displacement
    ``a = positions[i] - positions[i - 1]`` to the outgoing displacement
    ``b = positions[i + 1] - positions[i]``, in the horizontal (x, y) plane.
    Positive turns are counterclockwise (to the animal's left, seen from above).
    Only observed movement is used: a turn is never computed across a missing
    sample, a time gap or a declared recording break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Turn in radians, in ``(-pi, pi]``, aligned with ``times``. An exact
        reversal is ``pi``. NaN at the first and last sample, at samples next to
        a missing sample, time gap or break, and where the incoming or outgoing
        displacement is zero (the animal did not move, so there is no
        direction).

    Raises
    ------
    ValueError
        If ``positions`` does not have exactly two columns, ``times`` and
        ``positions`` are empty, swapped or misaligned, ``times`` is not
        strictly increasing, ``max_gap`` is not positive and finite or
        ``None``, or ``break_before`` is not a Boolean array with one entry per
        sample.

    Notes
    -----
    The turn is ``atan2(a_x * b_y - a_y * b_x, a . b)``. It is a turn per
    sample, so it depends on the frame rate: sampling the same smooth path
    four times faster makes each turn about four times smaller. For a measure
    of path shape that does not depend on the frame rate, use
    ``compute_planar_curvature``.

    Tracking jitter makes small displacements point in random directions, so
    turns at slow speeds are noisy. Resample the path at a spacing you choose
    with ``resample_by_distance`` first, or smooth the positions explicitly;
    this function does no smoothing.

    For 3-D tracking, pass ``positions[:, :2]`` to analyze turns in the
    horizontal plane.

    See Also
    --------
    compute_planar_curvature : Signed geometric curvature, independent of frame rate.
    compute_movement_angular_velocity : Turn per second, independent of frame rate.
    resample_by_distance : Equal-distance samples, a distance-scale regularization.
    compute_turn_angles : Turns between successive distinct positions, without
        times, observation gaps or sample alignment.

    Examples
    --------
    A right-angle left turn:

    >>> import numpy as np
    >>> from neurospatial.behavior import compute_planar_turn_angles
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    >>> turns = compute_planar_turn_angles(times, positions)
    >>> np.round(turns, 4).tolist()
    [nan, 1.5708, nan]

    Running counterclockwise around a circle at 0.8 rad/s, sampled at 30 Hz,
    turns by ``0.8 / 30`` radians per sample:

    >>> times = np.arange(0, 5, 1 / 30)
    >>> positions = 7.0 * np.c_[np.cos(0.8 * times), np.sin(0.8 * times)]
    >>> turns = compute_planar_turn_angles(times, positions)
    >>> bool(np.allclose(turns[1:-1], 0.8 / 30))
    True
    """
    _, a, b, _ = _planar_turns(
        times,
        positions,
        call="compute_planar_turn_angles",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
    )
    return _signed_turns(a, b)


def compute_planar_curvature(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute the signed curvature of the observed path at each tracking sample.

    Curvature at sample ``i`` is one over the radius of the circle through
    positions ``i - 1``, ``i`` and ``i + 1`` in the horizontal (x, y) plane:
    0 on a straight segment, ``1 / r`` on a circle of radius ``r``. It is
    positive when the path bends counterclockwise (to the animal's left, seen
    from above) and negative when it bends clockwise. Only observed movement is
    used: curvature is never computed across a missing sample, a time gap or a
    declared recording break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Signed curvature in inverse position units (for example 1/cm), aligned
        with ``times``. NaN at the first and last sample, at samples next to a
        missing sample, time gap or break, where the incoming or outgoing
        displacement is zero, and at a reversal of direction (no circle passes
        through the three points).

    Raises
    ------
    ValueError
        If ``positions`` does not have exactly two columns, ``times`` and
        ``positions`` are empty, swapped or misaligned, ``times`` is not
        strictly increasing, ``max_gap`` is not positive and finite or
        ``None``, or ``break_before`` is not a Boolean array with one entry per
        sample.

    Notes
    -----
    With ``a = positions[i] - positions[i - 1]`` and
    ``b = positions[i + 1] - positions[i]``, this is the discrete Menger
    curvature ``2 (a_x b_y - a_y b_x) / (|a| |b| |a + b|)``. It is exact for any
    three points on a circle, however unevenly they are spaced. It is NaN at
    any reversal (``a`` and ``b`` point in opposite directions, a turn of
    ``pi``), whatever the two step lengths, because no circle passes through
    three collinear points that double back. For a path of
    constant curvature it therefore does not depend on frame rate or running
    speed; on a general path it estimates curvature at the scale of the step
    length (speed times the sampling interval), and noise amplification grows
    as steps shrink. Resample by distance first to fix that scale. Scaling
    positions by ``s`` divides it by ``s``; time units do not enter.

    Tracking jitter dominates the curvature of very short steps. Resample the
    path at a spacing you choose with ``resample_by_distance`` first (curvature
    is then measured at that distance scale), or smooth the positions
    explicitly; this function does no smoothing.

    For 3-D tracking, pass ``positions[:, :2]`` to analyze curvature in the
    horizontal plane.

    See Also
    --------
    compute_planar_turn_angles : Signed turn per sample, in radians.
    compute_movement_angular_velocity : Turn per second (rad/s), not per metre.
    resample_by_distance : Equal-distance samples, a distance-scale regularization.
    compute_trajectory_curvature : Smoothed turn angle per sample (radians, not
        geometric curvature), without observation gaps.

    Examples
    --------
    A right-angle left turn through ``(10, 0)`` lies on a circle of radius
    ``5 * sqrt(2)``:

    >>> import numpy as np
    >>> from neurospatial.behavior import compute_planar_curvature
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    >>> curvature = compute_planar_curvature(times, positions)
    >>> np.round(curvature * 5 * np.sqrt(2), 6).tolist()
    [nan, 1.0, nan]

    A counterclockwise circle of radius 7 cm has curvature 1/7 per cm at every
    interior sample, and the reversed path has -1/7:

    >>> times = np.arange(0, 5, 1 / 30)
    >>> positions = 7.0 * np.c_[np.cos(0.8 * times), np.sin(0.8 * times)]
    >>> curvature = compute_planar_curvature(times, positions)
    >>> bool(np.allclose(curvature[1:-1], 1 / 7))
    True
    >>> reversed_path = compute_planar_curvature(times, positions[::-1])
    >>> bool(np.allclose(reversed_path[1:-1], -1 / 7))
    True
    """
    _, a, b, _ = _planar_turns(
        times,
        positions,
        call="compute_planar_curvature",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
    )
    cross = _cross_2d(a, b)
    chord = np.linalg.norm(a + b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * chord
    with np.errstate(invalid="ignore"):  # 0 / 0 at an equal-length reversal
        curvature: NDArray[np.float64] = 2.0 * cross / denominator
    # Opposite directions (turn of +-pi, any step lengths): no circle fits.
    curvature[(cross == 0) & (np.sum(a * b, axis=1) < 0)] = np.nan
    return curvature


def compute_movement_angular_velocity(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute how fast the movement direction turns, in radians per second.

    The movement direction is the direction of each observed displacement
    (allocentric: 0 = East, +pi/2 = North). At sample ``i`` the turn between the
    incoming and outgoing displacement is divided by the time it took, the mean
    of the two neighbouring intervals ``(times[i + 1] - times[i - 1]) / 2``.
    Positive values are counterclockwise (to the animal's left, seen from
    above). Only observed movement is used: nothing is computed across a missing
    sample, a time gap or a declared recording break.

    This is not head angular velocity. For head direction, unwrap the measured
    headings and difference them over time yourself.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Angular velocity of the movement direction in rad/s, aligned with
        ``times``. NaN wherever ``compute_planar_turn_angles`` is NaN: at the
        first and last sample, next to a missing sample, time gap or break, and
        where the animal did not move (a stationary sample has no direction and
        none is interpolated).

    Raises
    ------
    ValueError
        If ``positions`` does not have exactly two columns, ``times`` and
        ``positions`` are empty, swapped or misaligned, ``times`` is not
        strictly increasing, ``max_gap`` is not positive and finite or
        ``None``, or ``break_before`` is not a Boolean array with one entry per
        sample.

    Notes
    -----
    The turn is ``atan2(a_x * b_y - a_y * b_x, a . b)`` with
    ``a = positions[i] - positions[i - 1]`` and
    ``b = positions[i + 1] - positions[i]``. Dividing by the actual elapsed time
    rather than by an assumed sampling rate keeps irregular timestamps correct:
    for constant rotation on a circle the result equals the rotation rate at
    any sampling rate, even with uneven spacing. Scaling ``times`` by ``c``
    divides the result by ``c``.

    Tracking jitter makes the direction of slow, small displacements noisy.
    Resample the path at a spacing you choose with ``resample_by_distance``
    first, or smooth the positions explicitly; this function does no smoothing.

    For 3-D tracking, pass ``positions[:, :2]`` to analyze turning in the
    horizontal plane.

    See Also
    --------
    compute_planar_turn_angles : Signed turn per sample, in radians.
    compute_planar_curvature : Signed turn per unit distance, independent of speed.
    resample_by_distance : Equal-distance samples, a distance-scale regularization.

    Examples
    --------
    Running counterclockwise around a circle at 0.8 rad/s recovers 0.8 rad/s at
    every interior sample, whatever the sampling rate:

    >>> import numpy as np
    >>> from neurospatial.behavior import compute_movement_angular_velocity
    >>> times = np.arange(0, 5, 1 / 30)
    >>> positions = 7.0 * np.c_[np.cos(0.8 * times), np.sin(0.8 * times)]
    >>> omega = compute_movement_angular_velocity(times, positions)
    >>> bool(np.isnan(omega[[0, -1]]).all())
    True
    >>> bool(np.allclose(omega[1:-1], 0.8))
    True

    The same circle run clockwise gives -0.8 rad/s:

    >>> clockwise = compute_movement_angular_velocity(times, positions[::-1])
    >>> bool(np.allclose(clockwise[1:-1], -0.8))
    True
    """
    t, a, b, _ = _planar_turns(
        times,
        positions,
        call="compute_movement_angular_velocity",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
    )
    angles = _signed_turns(a, b)
    elapsed = np.full(t.size, np.nan)
    elapsed[1:-1] = (t[2:] - t[:-2]) / 2
    omega: NDArray[np.float64] = angles / elapsed
    return omega


def compute_step_lengths(
    positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """
    Compute step lengths (distances) between consecutive positions.

    Step lengths quantify the spatial displacement at each step of the trajectory.
    This function supports both Euclidean (straight-line) and geodesic (graph
    shortest-path) distance metrics for different use cases.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space. Each row is a position vector.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric to use:
        - "euclidean": Straight-line distance in physical space (ecology standard).
          Fast and appropriate for open environments without obstacles.
        - "geodesic": Graph shortest-path distance respecting environment topology.
          Requires `env` parameter. Appropriate for constrained navigation
          (walls, obstacles, tracks).
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        metric="geodesic", ignored otherwise.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Step lengths in the same units as the position coordinates.
        Geodesic steps touching a missing or out-of-environment position are
        NaN. Steps between disconnected valid bins are infinite.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.
        If positions is not a 2D array.
        If metric is not "euclidean" or "geodesic".

    Notes
    -----
    **Euclidean distance** (default):

    .. math::

        d_i = ||r_{i+1} - r_i||_2 = \\sqrt{\\sum_j (r_{i+1,j} - r_{i,j})^2}

    This is the straight-line distance in physical space, matching ecology
    literature standards (Kays et al. 2015, Traja).

    **Geodesic distance** (graph-based):

    Uses NetworkX shortest_path_length on env.connectivity graph with edge
    weights. This respects environmental constraints (walls, obstacles) and is
    appropriate for neuroscience applications with constrained environments.

    **When to use each**:

    - **Euclidean**: Open-field recordings, ecology studies, comparisons to
      movement ecology literature, maximum precision
    - **Geodesic**: Linear tracks, T-mazes, environments with barriers,
      neuroscience-specific analyses

    See Also
    --------
    compute_turn_angles : Compute turn angles between movements
    mean_square_displacement : Analyze diffusive properties

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_step_lengths
    >>>
    >>> # Straight line trajectory with Euclidean distance
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> step_lengths = compute_step_lengths(positions, metric="euclidean")
    >>> len(step_lengths)
    19
    >>> np.allclose(step_lengths, step_lengths[0], rtol=0.01)  # Uniform steps
    True
    >>>
    >>> # Geodesic distance on graph (requires env)
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> # Map positions to bins for geodesic
    >>> bins = env.bin_at(positions)
    >>> # Use bin centers as proxy positions for geodesic
    >>> bin_positions = env.bin_centers[bins]
    >>> step_lengths_geo = compute_step_lengths(
    ...     bin_positions, metric="geodesic", env=env
    ... )  # doctest: +SKIP

    References
    ----------
    .. [1] Kays, R., et al. (2015). "Terrestrial animal tracking as an eye
           on life and planet." Science, 348(6240), aaa2478.
    .. [2] Traja documentation: https://traja.readthedocs.io/
    """
    # Coerce array-likes (e.g. Python lists) before any .ndim access.
    try:
        positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as e:
        actual_type = type(positions).__name__
        raise TypeError(
            f"positions must be a numeric array-like object (e.g., numpy array, "
            f"list of lists, pandas DataFrame). Got {actual_type}: {positions!r}"
        ) from e

    # Input validation
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")

    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter. "
            "Use metric='euclidean' if env is not available."
        )

    n_steps = len(positions) - 1
    step_lengths = np.zeros(n_steps, dtype=np.float64)

    if metric == "euclidean":
        # Vectorized Euclidean distance computation
        displacements = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(displacements, axis=1)

    else:  # metric == "geodesic"
        # Map positions to bins
        assert env is not None  # Already checked above, but satisfy type checker
        require_cartesian(
            env.coordinate_system,
            "compute_step_lengths(metric='geodesic')",
            why="It reads positions as the animal's allocentric positions, located "
            "in the environment's bins.",
            fix="use the allocentric environment the positions were recorded in, or "
            "metric='euclidean'",
        )
        position_bins = env.bin_at(positions)

        # Optimization: Precompute geodesic distance matrix
        # This is O(V*E*log(V)) once vs O(n_steps * V*log(V)) per-step Dijkstra
        from neurospatial.ops.distance import geodesic_distance_matrix

        # Get full distance matrix (uses scipy.sparse.csgraph.shortest_path)
        dist_matrix = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

        # Outside/missing positions map to -1, which is not a graph node.
        # Never let NumPy interpret it as the last bin.
        valid = (position_bins[:-1] >= 0) & (position_bins[1:] >= 0)
        step_lengths = np.full(n_steps, np.nan, dtype=np.float64)
        step_lengths[valid] = dist_matrix[
            position_bins[:-1][valid], position_bins[1:][valid]
        ]

    return step_lengths


def _observed_cumulative(
    positions: NDArray[np.float64], support: ObservationSupport
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Step lengths and observed cumulative distance over usable intervals.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
    support : ObservationSupport
        Resolved from the same samples.

    Returns
    -------
    steps : NDArray[np.float64], shape (n_samples - 1,)
        Interval lengths; NaN on unusable intervals.
    cumulative : NDArray[np.float64], shape (n_samples,)
        Distance accumulated over usable intervals only, so it is flat across
        every break. Defined for every sample, including those outside runs.
    """
    with np.errstate(invalid="ignore"):  # inf - inf on consecutive missing rows
        raw = compute_step_lengths(positions, metric="euclidean")
    steps = np.where(support.interval_mask, raw, np.nan)
    cumulative = np.r_[0.0, np.cumsum(np.where(support.interval_mask, raw, 0.0))]
    return steps, cumulative


def compute_cumulative_distance(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> CumulativeDistanceResult:
    """
    Compute observed distance travelled, never crossing a recording break.

    Distance is added only over usable intervals: both samples have finite
    coordinates, the time between them is at most ``max_gap``, they fall inside
    one of ``epochs``, and no break is declared between them. A missing sample
    or a break therefore splits the path into runs instead of adding the jump
    across it or turning the total into NaN.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Positions in physical units. A 1-D array is one coordinate (for example
        a linearized track). Rows with a non-finite coordinate are missing.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    CumulativeDistanceResult
        ``run_distance`` restarts at 0 at the start of each run;
        ``cumulative_distance`` keeps adding across runs but never adds the
        displacement across a break. Both are NaN for isolated samples (valid
        positions with no usable interval). ``total_distance`` is NaN when no
        interval was usable, which is different from ``0.0`` (observed and
        stationary).

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` are empty, swapped or misaligned,
        ``times`` is not strictly increasing, ``max_gap`` is not positive and
        finite or ``None``, or ``break_before`` is not a Boolean array with one
        entry per sample.

    Notes
    -----
    Distance is Euclidean arclength over all coordinate columns, in the
    positions' units. It is not geodesic distance; use
    ``compute_step_lengths(metric="geodesic")`` for graph distance.

    See Also
    --------
    compute_step_lengths : Per-step lengths without observation gating.

    Examples
    --------
    Two recordings joined in one array; the 49 cm jump between them is not
    travelled distance.

    >>> import numpy as np
    >>> from neurospatial.behavior import compute_cumulative_distance
    >>> times = np.array([0.0, 0.1, 0.2, 0.3])
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    >>> new_recording = np.array([False, False, True, False])
    >>> result = compute_cumulative_distance(
    ...     times, positions, break_before=new_recording
    ... )
    >>> result.run_distance.tolist()
    [0.0, 1.0, 0.0, 1.0]
    >>> result.cumulative_distance.tolist()
    [0.0, 1.0, 1.0, 2.0]
    >>> result.total_distance
    2.0
    """
    _, positions_arr, support = validated_observations(
        times,
        positions,
        call="compute_cumulative_distance",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no distance was observed",
    )
    steps, cumulative = _observed_cumulative(positions_arr, support)
    in_run = support.run_ids >= 0
    bounds = support.run_sample_bounds
    run_distance = np.full(cumulative.size, np.nan)
    run_distance[in_run] = (
        cumulative[in_run] - cumulative[bounds[support.run_ids[in_run], 0]]
    )
    return CumulativeDistanceResult(
        step_lengths=steps,
        run_distance=run_distance,
        cumulative_distance=np.where(in_run, cumulative, np.nan),
        run_ids=support.run_ids,
        sample_valid=support.sample_valid,
        run_sample_bounds=bounds,
        run_time_bounds=support.run_time_bounds,
        run_path_lengths=cumulative[bounds[:, 1]] - cumulative[bounds[:, 0]],
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
    )


def _sample_dwell_times(
    times: NDArray[np.float64], interval_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Sample dwell from valid intervals, padding each run's final sample."""
    from neurospatial._intervals import run_sample_bounds

    dt = np.diff(times)
    weights: NDArray[np.float64] = np.zeros(len(times), dtype=np.float64)
    weights[:-1] = np.where(interval_mask, dt, 0.0)
    for first, last in run_sample_bounds(interval_mask):
        weights[last] = np.median(dt[first:last])
    return weights


def compute_home_range(
    position_bins: NDArray[np.int_],
    *,
    times: NDArray[np.float64] | None = None,
    percentile: float = 95.0,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.int_]:
    """
    Compute home range as bins containing a percentile of time spent.

    The home range is defined as the set of bins that collectively contain
    a specified percentile of the total time spent in the environment. This
    is analogous to the minimum convex polygon (MCP) method in ecology, but
    adapted for discretized spatial environments.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory. ``-1`` marks a
        sample outside the environment or missing; it is in no bin, so it
        adds nothing to the home range or to the total time.
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps (seconds) for each sample. When provided, occupancy is
        time-weighted: each sample contributes its observed interval dwell rather than a unit count. This is more accurate for
        non-uniformly sampled trajectories. Each run's final sample is assigned
        that run's median interval (it has no successor within the run). When ``None``
        (default), occupancy is the visit count per bin, which is correct only
        when sampling is uniform.
    percentile : float, default=95.0
        Percentile of time to include in home range (0 to 100). Common values:
        - 50%: core area (most frequently used)
        - 95%: standard home range
        - 100%: all visited bins
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. Requires ``times``.

    Returns
    -------
    NDArray[np.int_], shape (n_bins_in_range,)
        Bin indices in the home range, sorted by occupancy (most visited first).

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause.

    When ``times`` is given, dwell comes only from valid intervals. Each
    run's last sample receives that run's median interval duration; samples
    in no run receive zero dwell. Bins with zero observed dwell are omitted.
    No observed dwell gives an empty home
    range. With ``times=None``, counts are used, time gates are unused and
    ``break_before`` raises.

    The home range is computed by:
    1. Computing occupancy per bin (visit counts, or time-weighted dwell time
       when ``times`` is provided)
    2. Sorting bins by occupancy (descending)
    3. Selecting bins until cumulative occupancy reaches the percentile threshold

    This is related to the utilization distribution (UD) in movement ecology,
    where the UD is estimated from occupancy and the home range is defined
    as the isopleth containing a specified probability mass.

    See Also
    --------
    neurospatial.Environment.occupancy : Compute time-weighted occupancy

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import compute_home_range
    >>> # Trajectory with known distribution
    >>> position_bins = np.concatenate(
    ...     [
    ...         np.repeat(0, 50),  # Bin 0: 50 visits
    ...         np.repeat(1, 30),  # Bin 1: 30 visits
    ...         np.repeat(2, 15),  # Bin 2: 15 visits
    ...         np.repeat(3, 5),  # Bin 3: 5 visits
    ...     ]
    ... )
    >>> home_range = compute_home_range(position_bins, percentile=95.0)
    >>> set(home_range) == {0, 1, 2}  # 95% includes bins 0, 1, 2
    True

    References
    ----------
    .. [1] Worton, B. J. (1989). "Kernel methods for estimating the utilization
           distribution in home-range studies." Ecology, 70(1), 164-168.
    .. [2] Börger, L., et al. (2006). "Effects of sampling regime on the mean
           and variance of home range size estimates." Journal of Animal
           Ecology, 75(6), 1393-1405.
    """
    position_bins = np.asarray(position_bins)
    if (position_bins < -1).any():
        raise ValueError(
            _format_error(
                f"position_bins contains {int(position_bins.min())}, which is "
                "not a bin.",
                why="Bin indices are 0 or greater, and -1 marks a sample "
                "outside the environment.",
                fix="pass bins from env.bin_at(positions)",
            )
        )
    in_environment = position_bins >= 0

    if times is None:
        if break_before is not None:
            raise ValueError(
                _format_error(
                    "compute_home_range() got break_before without times.",
                    why="Without times the home range counts samples per bin, "
                    "and a recording break does not change a count.",
                    fix="pass times= to weight each bin by the dwell time "
                    "observed within each recording, or drop break_before.",
                )
            )
        # Count-based occupancy (assumes uniform sampling).
        unique_bins, occupancy = np.unique(
            position_bins[in_environment], return_counts=True
        )
        occupancy = occupancy.astype(np.float64)
    else:
        # Time-weighted occupancy: each sample contributes its dwell time.
        times = np.asarray(times, dtype=np.float64)
        if len(times) != len(position_bins):
            raise ValueError(
                f"times and position_bins must have same length. "
                f"Got {len(times)} and {len(position_bins)}."
            )
        from neurospatial._intervals import observed_interval_mask
        from neurospatial._observation import validate_break_before

        breaks = validate_break_before(
            break_before, len(times), call="compute_home_range"
        )
        interval_mask = observed_interval_mask(
            times, max_gap=max_gap, epochs=epochs, break_before=breaks
        )
        if len(position_bins) == 0:
            return np.array([], dtype=np.intp)
        sample_weights = _sample_dwell_times(times, interval_mask)[in_environment]
        # Sum dwell time per bin.
        unique_bins, inverse = np.unique(
            position_bins[in_environment], return_inverse=True
        )
        occupancy = np.zeros(len(unique_bins), dtype=np.float64)
        np.add.at(occupancy, inverse, sample_weights)
        observed = occupancy != 0
        unique_bins, occupancy = unique_bins[observed], occupancy[observed]

    # Sort bins by occupancy (descending)
    sort_idx = np.argsort(occupancy)[::-1]
    sorted_bins = unique_bins[sort_idx]
    sorted_counts = occupancy[sort_idx]

    # Compute cumulative percentage
    total_counts = np.sum(sorted_counts)
    if total_counts == 0:
        return np.array([], dtype=np.intp)
    cumulative_pct = np.cumsum(sorted_counts) / total_counts * 100.0

    # Find bins to include (cumulative percentage >= threshold)
    # searchsorted with side='left' gives first index where cumulative_pct >= percentile
    # We want bins 0 through that index (inclusive), so add 1 for Python slicing
    n_bins_include = np.searchsorted(cumulative_pct, percentile, side="left") + 1
    n_bins_include = min(n_bins_include, len(sorted_bins))

    # Return home range bins (sorted by occupancy)
    home_range = sorted_bins[:n_bins_include]

    return home_range.astype(np.intp)


def mean_square_displacement(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
    max_tau: float | None = None,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> MSDResult:
    """
    Compute mean square displacement (MSD) from continuous trajectory positions.

    Mean square displacement quantifies how spatial displacement grows with time
    lag, which is critical for characterizing diffusive motion and classifying
    movement patterns (ballistic, diffusive, subdiffusive, confined). This function
    operates on continuous position data for maximum accuracy.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Strictly increasing timestamps in seconds, one per sample.
    positions : array-like, shape (n_samples, n_dims) or (n_samples,)
        Trajectory positions in continuous space. A 1-D array is one
        coordinate. Rows with a non-finite coordinate are missing.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing displacements:
        - "euclidean": Straight-line distance (ecology standard, most accurate).
          Use for diffusion analysis, comparing to ecology literature, and
          classification of movement types (ballistic vs. diffusive).
        - "geodesic": Graph shortest-path distance (neuroscience-specific).
          Requires `env` parameter. Use for constrained navigation analysis.
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        metric="geodesic", ignored otherwise.
    max_tau : float, optional
        Maximum lag time to compute. If None, uses half the total duration.
        Recommended: use max_tau ≤ T/4 where T is total duration, to ensure
        sufficient samples for reliable averaging.
    epochs : array-like, shape (n_epochs, 2), optional
        ``(start, stop)`` windows in seconds. A displacement pair is used only
        when both samples lie in the same window; touching windows merge.
        ``None`` uses the whole recording.
    break_before : array of bool, shape (n_samples,), optional
        True at sample ``k`` declares a recording break between samples
        ``k - 1`` and ``k``. No pair spans a break. Use it when the
        positions of two recordings are joined in one array.

    Returns
    -------
    MSDResult
        Frozen dataclass with ``lags`` (lag times) and ``msd`` (MSD values)
        fields, each of shape ``(n_lags,)``. Lags without observed, finite
        displacement pairs have NaN MSD rather than zero.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.
        If there are no samples, or positions and times have different lengths.
        If times are not finite and strictly increasing (a repeated timestamp
        raises).
        If ``break_before`` is not a Boolean array with one entry per sample.

    Warns
    -----
    UserWarning
        If ``epochs`` leaves no displacement pair observed.

    Notes
    -----
    The mean square displacement is defined as:

    .. math::

        MSD(\\tau) = \\langle |r(t + \\tau) - r(t)|^2 \\rangle_t

    where :math:`r(t)` is the position at time t, and the angle brackets denote
    averaging over all time points t.

    Each target time is matched to the nearest recorded sample within half
    the median sample interval. Targets beyond the recording are excluded.
    Pairs with missing positions are excluded; geodesic pairs additionally
    require both positions to be inside the environment and connected. A pair
    is also excluded when a declared ``break_before`` or an ``epochs`` edge
    lies between its two samples. A single sample returns an empty result.

    There is no ``max_gap``. MSD compares two observed positions a lag apart
    and does not integrate along the path, so samples missing between them do
    not change the displacement. A gap matters only when the two positions are
    not in the same coordinate frame, which is what ``break_before`` and
    ``epochs`` declare.

    **Movement classification from scaling exponent**:

    MSD scales with time lag τ as :math:`MSD \\sim \\tau^\\alpha`:

    - **α ≈ 2**: Ballistic motion (directed, constant velocity)
    - **α ≈ 1**: Diffusive motion (random walk, Brownian motion)
    - **α < 1**: Subdiffusive motion (obstructed, confined)
    - **α > 2**: Superdiffusive motion (Lévy flight)
    - **α ≈ 0**: Confined motion (bounded, plateau)

    The exponent α can be estimated from log-log regression: log(MSD) ~ α·log(τ).

    **Why continuous positions?**

    Using continuous positions (not bins) is critical for MSD accuracy because:

    1. **Precision**: Bin discretization introduces a "floor" on displacements
       equal to the bin size, artificially inflating MSD for small displacements.
    2. **Scaling exponent**: Discretization can bias the diffusion exponent α,
       making diffusive motion appear more subdiffusive.
    3. **Ecology standard**: All movement ecology literature uses continuous MSD
       (Saxton 1997, Ferrari 2001, Kepten 2015).

    Example: Movement [0,0]→[0.1,0.1]→[0.2,0.2] maps to the same bin, giving
    MSD=0 (incorrect), whereas continuous gives MSD ∝ t (correct diffusive).

    See Also
    --------
    compute_turn_angles : Analyze directional changes
    compute_step_lengths : Compute step distances

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.trajectory import mean_square_displacement
    >>>
    >>> # Random walk trajectory (diffusive motion, α ≈ 1)
    >>> rng = np.random.default_rng(42)
    >>> n_steps = 100
    >>> steps = rng.standard_normal((n_steps, 2)) * 5
    >>> positions = np.cumsum(steps, axis=0)
    >>> times = np.arange(n_steps) * 0.1
    >>>
    >>> # Compute MSD with Euclidean distance (default)
    >>> result = mean_square_displacement(
    ...     times, positions, metric="euclidean", max_tau=5.0
    ... )
    >>> len(result.lags) > 0
    True
    >>> bool(result.msd[-1] > result.msd[0])  # MSD increases with lag
    True
    >>>
    >>> # Estimate diffusion exponent from log-log fit
    >>> log_tau = np.log(result.lags)
    >>> log_msd = np.log(result.msd + 1e-10)  # Add small constant to avoid log(0)
    >>> alpha = np.polyfit(log_tau, log_msd, 1)[0]  # doctest: +SKIP
    >>> print(f"Diffusion exponent: {alpha:.2f}")  # doctest: +SKIP
    Diffusion exponent: 1.02  # ≈ 1.0 (diffusive motion)
    >>>
    >>> # Two recordings joined in one array: declare the join so no
    >>> # displacement pair spans it.
    >>> times = np.arange(10.0)
    >>> positions = np.c_[[0, 1, 2, 3, 4, 100, 101, 102, 103, 104], np.zeros(10)]
    >>> new_recording = np.arange(10) == 5
    >>> mean_square_displacement(
    ...     times, positions, max_tau=3.0, break_before=new_recording
    ... ).msd
    array([1., 4., 9.])

    References
    ----------
    .. [1] Saxton, M. J. (1997). "Single-particle tracking: the distribution
           of diffusion coefficients." Biophysical Journal, 72(4), 1744-1753.
    .. [2] Ferrari, R., et al. (2001). "Tracking single particles in biological
           systems." Physical Review E, 63(4), 041904.
    .. [3] Kepten, E., et al. (2015). "Improved estimation of anomalous
           diffusion exponents in single-particle tracking experiments."
           Physical Review E, 87(5), 052713.
    .. [4] Traja documentation: https://traja.readthedocs.io/
    """
    times, positions, support = validated_observations(
        times,
        positions,
        call="mean_square_displacement",
        max_gap=None,
        epochs=epochs,
        break_before=break_before,
        result="no displacement pair was observed",
    )
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")

    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter. "
            "Use metric='euclidean' if env is not available."
        )

    if metric == "geodesic":
        assert env is not None  # Already checked above
        require_cartesian(
            env.coordinate_system,
            "mean_square_displacement(metric='geodesic')",
            why="It reads positions as the animal's allocentric positions, located "
            "in the environment's bins.",
            fix="use the allocentric environment the positions were recorded in, or "
            "metric='euclidean'",
        )

    n_samples = len(positions)
    if n_samples < 2:
        return MSDResult(lags=np.empty(0), msd=np.empty(0))

    # Determine lag times to compute
    if max_tau is None:
        max_tau = (times[-1] - times[0]) / 2.0

    # Find all unique time differences up to max_tau
    time_diffs = np.diff(times)
    # The resolver guarantees strictly increasing times, so dt > 0.
    dt = float(np.median(time_diffs))  # Typical time step
    n_lags = int(max_tau / dt)
    n_lags = max(1, min(n_lags, n_samples // 2))  # At least 1, at most n_samples/2

    tau_values = np.linspace(dt, max_tau, n_lags)
    msd_values = np.full(n_lags, np.nan, dtype=np.float64)

    if metric == "geodesic":
        # Map positions to bins for geodesic distance
        assert env is not None  # Already checked above
        position_bins = env.bin_at(positions)
        valid_bins = (position_bins >= 0) & (position_bins < env.n_bins)

        # Optimization: Precompute geodesic distance matrix ONCE
        # This is O(V*E*log(V)) once vs O(n_lags * n_samples * V*log(V)) per lookup
        from neurospatial.ops.distance import geodesic_distance_matrix

        dist_matrix = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

    sample_indices = np.arange(n_samples)
    finite_positions = support.sample_valid
    # Segments are separated by declared breaks and epoch edges; a pair is
    # valid only inside one segment.
    segment = np.r_[0, np.cumsum(~support.interval_gates)]
    # Adding a lag can round an observed endpoint one timestamp ULP upward.
    # Allow that roundoff without extending support by the matching tolerance.
    recording_stop = np.nextafter(times[-1], np.inf)
    for i, tau in enumerate(tau_values):
        target_times = times + tau
        # searchsorted gives the successor, which need not be the nearest
        # sample. Compare both neighbors before applying the lag tolerance.
        right_idx = np.clip(np.searchsorted(times, target_times), 0, n_samples - 1)
        left_idx = np.maximum(right_idx - 1, 0)
        right_error = np.abs(times[right_idx] - target_times)
        left_error = np.abs(times[left_idx] - target_times)
        future_idx = np.where(left_error <= right_error, left_idx, right_idx)
        time_errors = np.abs(times[future_idx] - target_times)
        valid_pairs = (
            (future_idx > sample_indices)
            & (target_times <= recording_stop)
            & (time_errors < dt / 2)
            & finite_positions
            & finite_positions[future_idx]
            & (segment == segment[future_idx])
        )

        if metric == "euclidean":
            displacements = positions[future_idx[valid_pairs]] - positions[valid_pairs]
            sq_dists = np.sum(displacements**2, axis=1)
        else:
            # bin_at returns -1 for missing/outside positions; never index the
            # distance matrix with it, where NumPy would select the last bin.
            valid_pairs &= valid_bins & valid_bins[future_idx]
            distances = dist_matrix[
                position_bins[valid_pairs], position_bins[future_idx[valid_pairs]]
            ]
            sq_dists = distances**2

        finite_distances = np.isfinite(sq_dists)
        if np.any(finite_distances):
            msd_values[i] = np.mean(sq_dists[finite_distances])

    return MSDResult(lags=tau_values, msd=msd_values)


def compute_trajectory_curvature(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    smooth_window: float | None = 0.2,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the smoothed turn angle at each tracking sample.

    Despite its name, this returns turning angles in radians (the signed change
    of movement direction at each sample), not geometric curvature. For
    curvature in inverse length units use ``compute_planar_curvature``.
    Works for any number of columns (1D, 2D, 3D, N-D), but see Notes for how
    extra columns enter. Only observed movement is used: no turn is computed
    or smoothed across a missing sample, a time gap or a declared recording
    break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Position coordinates over time in any dimensional space. Rows with a
        non-finite coordinate are missing.
    smooth_window : float or None, default=0.2
        Standard deviation, in seconds, of the Gaussian that smooths the turn
        angles within each run (0.2 s suits 30-60 Hz tracking). ``None``
        disables smoothing.

        **Important**: For high-speed tracking (120+ Hz) or fast-moving animals,
        use shorter windows (0.05-0.1s) or disable smoothing (smooth_window=None)
        to preserve rapid turns.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Heading change at each timepoint in radians:
        - Positive values: counterclockwise turn (left in 2D top-down view)
        - Negative values: clockwise turn (right in 2D top-down view)
        - Zero: straight movement
        - NaN: missing samples and samples in no run (isolated by a gap, a
          declared break, a missing neighbour or ``epochs``)

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` are empty, swapped or misaligned,
        ``times`` is not strictly increasing, ``max_gap`` is not positive and
        finite or ``None``, or ``break_before`` is not a Boolean array with
        one entry per sample.

    Notes
    -----
    A run is a stretch of observed samples with no missing row, no gap longer
    than ``max_gap``, no ``epochs`` edge and no declared break. Each run is
    analyzed as a separate recording: turn angles are computed and smoothed
    within it, so a missing row or a break never spreads into its neighbours.

    Within a run this function wraps `compute_turn_angles()` and adds:
    - Mapping to the original sample positions
    - Optional temporal smoothing

    `compute_turn_angles()` uses atan2(cross, dot) for proper signed angles in [-π, π],
    filters stationary periods automatically, and returns at most (n_samples - 2)
    angles. Each angle is assigned to the original index of its middle position
    after consecutive duplicates are removed (keeping their first sample).
    Stationary duplicates and the first and last retained positions of a run
    have zero turn before smoothing, preserving the timestamps of actual turns.

    For positions with more than two columns the angle is
    ``atan2(v1_x * v2_y - v1_y * v2_x, sum(v1 * v2))``: the cross product uses
    the first two columns (x, y) and the dot product sums over all columns
    (as in `compute_turn_angles()`). This is neither the horizontal-plane angle
    nor the 3-D angle; pass ``positions[:, :2]`` for a horizontal-plane
    analysis.

    The turn is per sample, so it depends on the frame rate and the running
    speed. To compare path shape across sessions, use
    ``compute_planar_curvature``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import compute_trajectory_curvature

    A straight run has no turns:

    >>> times = np.arange(20) / 30  # 30 Hz, seconds
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])
    >>> bool(np.allclose(compute_trajectory_curvature(times, positions), 0.0))
    True

    A right-angle left turn, without smoothing:

    >>> times = np.arange(4) * 0.1
    >>> positions = np.array([[0.0, 0], [1, 0], [2, 0], [2, 1]])
    >>> turns = compute_trajectory_curvature(times, positions, smooth_window=None)
    >>> np.round(turns, 4).tolist()
    [0.0, 0.0, 1.5708, 0.0]

    A missing row (lost tracking) is NaN and splits the path into two runs;
    the turn is not computed across it:

    >>> times = np.arange(7) * 0.1
    >>> positions = np.array(
    ...     [[0.0, 0], [1, 0], [2, 0], [2, 1], [np.nan, np.nan], [3, 1], [3, 2]]
    ... )
    >>> turns = compute_trajectory_curvature(times, positions, smooth_window=None)
    >>> np.round(turns, 4).tolist()
    [0.0, 0.0, 1.5708, 0.0, nan, 0.0, 0.0]

    With three columns the cross product uses x, y and the dot product uses
    all three: the turn at the middle sample is π/4, not the π/2 of the x-y
    projection:

    >>> times = np.array([0.0, 0.1, 0.2])
    >>> x = np.array([[0.0, 0, 0], [1, 0, 1], [1, 1, 2]])
    >>> float(compute_trajectory_curvature(times, x, smooth_window=None)[1])
    0.7853981633974483
    >>> float(compute_trajectory_curvature(times, x[:, :2], smooth_window=None)[1])
    1.5707963267948966

    See Also
    --------
    compute_turn_angles : Raw turn angles without padding
    compute_planar_turn_angles : Sample-aligned planar turning angles
    compute_planar_curvature : Signed geometric curvature (inverse length)
    compute_movement_angular_velocity : Turning rate of the movement direction
    """
    t, p, support = validated_observations(
        times,
        positions,
        call="compute_trajectory_curvature",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every curvature value is NaN",
    )
    curvature: NDArray[np.float64] = np.full(t.size, np.nan)
    for first, last in support.run_sample_bounds:
        run = slice(first, last + 1)
        curvature[run] = _compute_trajectory_curvature_contiguous(
            p[run], t[run], smooth_window=smooth_window
        )
    return curvature


def _compute_trajectory_curvature_contiguous(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    smooth_window: float | None,
) -> NDArray[np.float64]:
    """Map angles to their original samples and smooth within one recording."""
    angles = compute_turn_angles(positions)
    n_samples = len(positions)
    curvature = np.zeros(n_samples, dtype=np.float64)
    if len(angles):
        changed = np.any(positions[1:] != positions[:-1], axis=1)
        retained_indices = np.flatnonzero(np.r_[True, changed])
        curvature[retained_indices[1:-1]] = angles

    # Optional temporal smoothing
    if smooth_window is not None:
        from scipy.ndimage import gaussian_filter1d

        # Compute sigma from time resolution
        dt_median = np.median(np.diff(times))
        sigma = smooth_window / dt_median

        # Apply Gaussian smoothing
        curvature = gaussian_filter1d(curvature, sigma=sigma)

    return curvature
