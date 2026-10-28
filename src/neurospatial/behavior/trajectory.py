"""Trajectory characterization metrics.

This module provides functions for analyzing spatial trajectories: running
speed, movement and body heading, turn angles, planar curvature and the
angular velocity of the movement direction, step lengths, cumulative
distance travelled, home range estimation, and mean square displacement.
These metrics are commonly used in behavioral ecology and neuroscience to
characterize movement patterns and spatial exploration.

References
----------
.. [1] Traja: https://traja.readthedocs.io/ - Trajectory analysis toolkit
.. [2] Kays, R., Crofoot, M. C., Jetz, W., & Wikelski, M. (2015).
       "Terrestrial animal tracking as an eye on life and planet."
       Science, 348(6240), aaa2478.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    gap_endpoints,
    observed_interval_mask,
    validate_max_gap,
)
from neurospatial._observation import (
    ObservationSupport,
    empty_samples_error,
    validate_break_before,
    validated_observations,
)
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._validation import (
    validate_strictly_increasing,
    validate_times_positions,
)
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.behavior._metric import metric_paths, require_metric
from neurospatial.behavior.epochs import _mask_in_intervals

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.ops._types import DistanceMetric


@dataclass(frozen=True, repr=False, eq=False)
class MSDResult(ReadOnlyArraysMixin, ResultMixin):
    """Mean square displacement curve returned by ``mean_square_displacement``.

    Attributes
    ----------
    lags : NDArray[np.float64], shape (n_lags,)
        Lag times τ in seconds, strictly increasing. Lag ``k`` is the mean
        time separation of the sample pairs averaged there: ``k`` times the
        median sample interval for uniform sampling.
    msd : NDArray[np.float64], shape (n_lags,)
        Mean square displacement values, one per lag, in squared position
        units. NaN when no valid displacement pair supports that lag.
    metric : {"euclidean", "geodesic"}
        Distance the displacements were measured with.
    epochs : NDArray[np.float64] or None, shape (n_epochs, 2)
        Normalized half-open ``[start, stop)`` windows that were applied.
    """

    lags: NDArray[np.float64]
    msd: NDArray[np.float64]
    metric: DistanceMetric
    epochs: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Check the curve and own read-only copies of its arrays."""
        lags = np.asarray(self.lags, dtype=np.float64)
        msd = np.asarray(self.msd, dtype=np.float64)
        if lags.ndim != 1 or msd.shape != lags.shape:
            raise ValueError(
                _format_error(
                    f"MSDResult needs 1-D lags and msd of the same length; got "
                    f"shapes {lags.shape} and {msd.shape}.",
                    why="each MSD value is the mean square displacement at one lag.",
                    fix="pass one msd value per lag.",
                )
            )
        if np.any(np.diff(lags) <= 0):
            raise ValueError(
                _format_error(
                    "MSDResult needs strictly increasing lags.",
                    why="each lag is a distinct time separation.",
                    fix="sort the lags and their msd values together.",
                )
            )
        require_metric(self.metric, where="MSDResult")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "msd", msd)
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_lags": int(self.lags.size),
            "max_lag": float(self.lags.max()) if self.lags.size else float("nan"),
            "final_msd": float(self.msd[-1]) if self.msd.size else float("nan"),
        }


def _invalid_result(owner: str, problem: str) -> ValueError:
    """The error for a result built from fields that disagree."""
    return ValueError(
        _format_error(
            f"{owner} got inconsistent fields: {problem}.",
            why=f"{owner} describes runs of one path, so its arrays must agree "
            "in length and its run bounds must be ordered and in range.",
            fix="build it with the function that returns it rather than by hand.",
        )
    )


class _RunPathMixin:
    """Run structure shared by the distance results along an observed path.

    Subclasses are frozen dataclasses with ``run_distance``, ``run_ids``,
    ``run_sample_bounds``, ``run_time_bounds`` and ``run_path_lengths``.
    """

    run_distance: NDArray[np.float64]
    run_ids: NDArray[np.intp]
    run_sample_bounds: NDArray[np.intp]
    run_time_bounds: NDArray[np.float64]
    run_path_lengths: NDArray[np.float64]

    @property
    def cumulative_distance(self) -> NDArray[np.float64]:
        """Path length of all earlier runs plus ``run_distance``, read-only.

        NaN where ``run_ids == -1``.
        """
        offsets = np.r_[0.0, np.cumsum(self.run_path_lengths)[:-1]]
        in_run = self.run_ids >= 0
        value = np.full(self.run_ids.shape, np.nan)
        value[in_run] = offsets[self.run_ids[in_run]] + self.run_distance[in_run]
        value.setflags(write=False)
        return value

    def _check_runs(self, owner: str, *, n_samples: int, n_source: int) -> None:
        """Check the run arrays against each other and the source samples.

        ``n_samples`` is the length of ``run_ids``; ``n_source`` the number
        of original samples the run bounds index.
        """
        bounds = np.asarray(self.run_sample_bounds)
        time_bounds = np.asarray(self.run_time_bounds)
        run_ids = np.asarray(self.run_ids)
        n_runs = len(self.run_path_lengths)
        if bounds.shape != (n_runs, 2) or time_bounds.shape != (n_runs, 2):
            raise _invalid_result(
                owner,
                f"run_sample_bounds {bounds.shape} and run_time_bounds "
                f"{time_bounds.shape} must both be ({n_runs}, 2), one row per "
                "run_path_lengths entry",
            )
        if np.shape(self.run_distance) != (n_samples,):
            raise _invalid_result(
                owner,
                f"run_distance has shape {np.shape(self.run_distance)}, but "
                f"run_ids has {n_samples} entries",
            )
        if n_runs and (
            np.any(bounds[:, 0] > bounds[:, 1])
            or bounds.min() < 0
            or bounds.max() >= n_source
            or np.any(time_bounds[:, 0] > time_bounds[:, 1])
        ):
            raise _invalid_result(
                owner,
                "each run needs first <= last sample, inside the "
                f"{n_source} source samples, and start <= stop time",
            )
        if n_samples and (run_ids.min() < -1 or run_ids.max() >= n_runs):
            raise _invalid_result(
                owner, f"run_ids must be -1 or a run index below {n_runs}"
            )


@dataclass(frozen=True, repr=False, eq=False)
class CumulativeDistanceResult(_RunPathMixin, ReadOnlyArraysMixin, ResultMixin):
    """Observed distance travelled, returned by ``compute_cumulative_distance``.

    Distances are Euclidean arclength along the observed polyline, in the
    units of the positions. A missing sample is bridged like a deleted row
    when the valid samples around it are within ``max_gap``: the straight
    displacement between them is counted once. Only a longer gap, an
    ``epochs`` edge or a declared break splits a run, and those add nothing,
    so distance never crosses a break.

    Attributes
    ----------
    step_lengths : NDArray[np.float64], shape (n_samples - 1,)
        Length of each interval; NaN on unusable intervals. A bridged gap's
        displacement is stored on its last interval (the one ending at the
        next valid sample) and its other intervals are NaN.
    run_distance : NDArray[np.float64], shape (n_samples,)
        Distance since the start of the sample's run (0 at each run start);
        NaN where ``run_ids == -1``.
    cumulative_distance : NDArray[np.float64], shape (n_samples,)
        Observed distance since the first sample, summed over all runs so far:
        the path length of earlier runs plus ``run_distance``. NaN where
        ``run_ids == -1``. Derived from those fields, read-only.
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
    run_ids: NDArray[np.intp]
    sample_valid: NDArray[np.bool_]
    run_sample_bounds: NDArray[np.intp]
    run_time_bounds: NDArray[np.float64]
    run_path_lengths: NDArray[np.float64]
    max_gap: float | None
    epochs: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Check that the fields describe one path, then own the arrays."""
        owner = "CumulativeDistanceResult"
        n_samples = len(self.run_ids)
        if np.shape(self.sample_valid) != (n_samples,) or np.shape(
            self.step_lengths
        ) != (max(n_samples - 1, 0),):
            raise _invalid_result(
                owner,
                f"sample_valid {np.shape(self.sample_valid)} and step_lengths "
                f"{np.shape(self.step_lengths)} must match {n_samples} samples",
            )
        self._check_runs(owner, n_samples=n_samples, n_source=n_samples)
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
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Running speed at each tracking sample, robust to tracking jitter.

    The velocity vector is estimated by centered differences and smoothed
    with a Gaussian of ``smooth_window`` seconds before its magnitude is
    taken. Differencing raw positions instead turns camera jitter into
    speed: an animal sitting still with 0.5 cm of jitter reads about
    20 cm/s at 30 Hz and 200 cm/s at 500 Hz, so a ``min_speed`` filter would
    keep nearly all of its rest. The per-sample ``min_speed`` gates of rate
    maps, decoding, ``segment_by_velocity`` and the decision and goal
    metrics use this speed, so passing ``speed=compute_speed(...)``
    explicitly gives the same result as leaving it out, provided the gate
    gets the same ``break_before``. ``detect_runs_between_regions`` is
    different: its ``min_speed`` drops whole runs by their mean speed
    between bin centres.

    A missing sample is bridged like a deleted row when the valid samples
    around it are within ``max_gap``. Speed is never smoothed across a longer
    gap, the edge of an ``epochs`` window or a declared break.

    All coordinate columns enter the speed: for 3-D tracking it is the 3-D
    speed. ``heading_from_velocity`` uses x and y only; pass
    ``positions[:, :2]`` for the horizontal speed.

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
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
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
        samples and at samples in no run (isolated by a gap longer than
        ``max_gap``, a declared break or ``epochs``).

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
    max_gap: float | None = DEFAULT_MAX_GAP,
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
    positions : NDArray, shape (n_time, n_dims)
        Animal positions over time, in environment units (e.g. cm), with
        ``n_dims >= 2``; only x and y are used, so the heading and the
        ``min_speed`` gate are horizontal. A row with a non-finite coordinate
        (NaN, for a lost frame) is a missing sample: it has no heading, and
        it is bridged like a deleted row when the valid samples around it are
        within ``max_gap``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted. Windows that
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
        not a Boolean array with one entry per sample, if ``min_speed`` is not
        finite and non-negative, or if every observed sample is below
        ``min_speed`` or still and ``allow_all_nan`` is ``False`` (the
        default).

    Warns
    -----
    UserWarning
        If every observed sample is below ``min_speed`` or still and
        ``allow_all_nan=True`` (an all-NaN heading array is returned).

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``
    and not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a longer gap, an ``epochs`` edge or a
    declared break.

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

    A lost frame is a NaN row. It has no heading, and its neighbours are
    bridged as if the row were deleted:

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
        validated_min_speed,
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
    min_speed = validated_min_speed(min_speed, call="heading_from_velocity")
    heading, speed = _movement_heading_and_speed(
        times,
        positions,
        support.interval_mask,
        smooth_window=smooth_window,
        min_speed=min_speed,
    )
    # The velocity direction is observed only where the animal moves.
    moving = np.isfinite(heading)
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
    times: NDArray[np.float64],
    heading: NDArray[np.float64],
    fill: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Fill headings linearly in time along the shorter arc.

    The anchors are the finite headings outside ``fill``. They are unwrapped,
    interpolated at ``times[fill]`` and wrapped back to (-pi, pi]. For an
    exactly antipodal pair, the sign of the stored difference chooses the turn.
    The caller passes only samples that lie between two anchors, so nothing
    is extrapolated.

    Parameters
    ----------
    times : NDArray, shape (n_time,)
        Strictly increasing timestamps in seconds.
    heading : NDArray, shape (n_time,)
        Heading values in radians; NaN where not observed.
    fill : NDArray of bool, shape (n_time,)
        True at samples to fill, each between two anchors.

    Returns
    -------
    NDArray, shape (n_time,)
        A copy of ``heading`` with the ``fill`` samples interpolated.
    """
    result: NDArray[np.float64] = heading.copy()
    if not np.any(fill):
        return result
    anchors = np.isfinite(heading) & ~fill
    unwrapped = np.unwrap(heading[anchors])
    filled = np.interp(times[fill], times[anchors], unwrapped)
    result[fill] = np.pi - np.mod(np.pi - filled, 2.0 * np.pi)
    return result


def heading_from_body_orientation(
    times: ArrayLike,
    nose: ArrayLike,
    tail: ArrayLike,
    *,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute heading from nose and tail keypoints.

    The heading of a frame is the direction of the tail-to-nose vector. A frame
    without that vector (a missing keypoint, or nose and tail at the same
    point) is filled from the frames around it only across a short gap inside
    one recording.

    Parameters
    ----------
    times : array-like, shape (n_time,)
        Finite, strictly increasing timestamps in seconds, one per frame.
    nose : array-like, shape (n_time, n_dims)
        Nose keypoint positions in position units (e.g. cm), ``n_dims >= 2``;
        only x and y are used. A row with a non-finite coordinate (NaN, for
        a lost or low-likelihood keypoint) is missing.
    tail : array-like, shape (n_time, n_dims)
        Tail-base keypoint positions, in the same units and shape as ``nose``.
    max_gap : float or None, default=1.0
        Longest stretch of missing frames, in seconds, to fill: the time
        between the observed frames on either side. Longer gaps stay NaN;
        ``None`` fills gaps of any length inside one recording and epoch.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). Frames outside every window are NaN, and no
        fill spans a window edge. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_time,), optional
        ``True`` at frame ``k`` declares a break between frames ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No fill spans a declared break.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians in the allocentric convention (0 = East,
        π/2 = North), in ``[-π, π]``. Filled frames are interpolated linearly
        in time along the shorter arc and wrapped to ``(-π, π]``. NaN where a
        frame is missing and not filled, and outside ``epochs``.

    Raises
    ------
    ValueError
        If there are no frames, if ``times`` is not 1-D, finite, strictly
        increasing and aligned with the keypoints, if ``nose`` and ``tail``
        differ in shape or have no x and y columns, if ``break_before`` is not
        a Boolean array with one entry per frame, or if no frame has both
        keypoints at distinct points.

    Warns
    -----
    UserWarning
        If ``max_gap``, ``epochs`` and ``break_before`` together exclude every
        interval, so no frame is filled; every heading is NaN when no frame
        lies inside ``epochs``.

    See Also
    --------
    heading_from_velocity : Direction of movement, from positions alone.

    Notes
    -----
    A gap is filled only when it lasts at most ``max_gap`` and lies inside one
    recording and one epoch; longer gaps and samples outside ``epochs`` are
    NaN. Frames before the first or after the last observed frame are not
    extrapolated. A frame whose nose and tail coincide has no direction
    (``arctan2(0, 0)`` would read as East), so it is treated as missing.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import heading_from_body_orientation

    The animal faces East, loses one frame, turns North and loses another:

    >>> times = np.arange(8) * 0.1  # s
    >>> tail = np.zeros((8, 2))
    >>> nose = np.column_stack([np.ones(8), np.zeros(8)])  # facing East
    >>> nose[2] = np.nan  # one lost frame: filled
    >>> nose[5:7] = [[0.0, 1.0], [np.nan, np.nan]]  # turns North, then a lost frame
    >>> np.round(heading_from_body_orientation(times, nose, tail), 3).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.571, 0.785, 0.0]

    When a new recording starts at 0.6 s, its first frame is the lost one,
    and it is not filled across the break:

    >>> breaks = np.arange(8) == 6  # a new recording starts at frame 6 (0.6 s)
    >>> np.round(
    ...     heading_from_body_orientation(times, nose, tail, break_before=breaks), 3
    ... ).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.571, nan, 0.0]
    """
    call = "heading_from_body_orientation"
    n_times = np.shape(times)[0] if np.ndim(times) else None
    for name, keypoint in (("nose", nose), ("tail", tail)):
        n_rows = np.shape(keypoint)[0] if np.ndim(keypoint) else None
        if n_rows is not None and n_times is not None and n_rows != n_times:
            raise ValueError(
                _format_error(
                    f"{call}() needs one {name} row per timestamp; times has "
                    f"{n_times} samples but {name} has {n_rows}.",
                    why=(
                        "Why: the heading at each frame is the direction of that "
                        "frame's tail-to-nose vector, so both keypoints need a "
                        "row for every timestamp."
                    ),
                    fix=(
                        f"pass {name} with shape (n_samples, 2), the same "
                        "n_samples as times, or trim all three to the same frames."
                    ),
                )
            )
    t, nose_arr = validate_times_positions(times, nose, call=call)
    _, tail_arr = validate_times_positions(t, tail, call=call)
    if t.size == 0:
        raise empty_samples_error(call, ("times", "nose", "tail"))
    if nose_arr.ndim != 2 or nose_arr.shape[1] < 2 or tail_arr.shape != nose_arr.shape:
        raise ValueError(
            _format_error(
                f"{call}() needs nose and tail with the same shape "
                f"(n_samples, 2) or wider; got nose {nose_arr.shape} and tail "
                f"{tail_arr.shape}.",
                why=(
                    "Why: the heading is the direction of the tail-to-nose "
                    "vector in the x-y plane, one vector per frame."
                ),
                fix=(
                    "pass the x and y of each keypoint, for example "
                    "np.column_stack([nose_x, nose_y]), with one row per timestamp."
                ),
            )
        )
    validate_strictly_increasing(t, call=call)
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, t.size, call=call)

    dx = nose_arr[:, 0] - tail_arr[:, 0]
    dy = nose_arr[:, 1] - tail_arr[:, 1]
    observed = (
        np.isfinite(nose_arr).all(axis=1)
        & np.isfinite(tail_arr).all(axis=1)
        & (np.hypot(dx, dy) > 0)
    )
    if not observed.any():
        raise ValueError(
            _format_error(
                f"{call}() found no frame with both keypoints at distinct "
                "points: every frame has a non-finite nose or tail coordinate, "
                "or the two coincide.",
                why="Why: the body heading needs a tail-to-nose vector.",
                fix=(
                    "check the pose likelihood filtering (it may have set every "
                    "keypoint to NaN), or use heading_from_velocity() for the "
                    "direction of movement from positions."
                ),
            )
        )
    observed &= _mask_in_intervals(
        t, as_intervals(epochs, name="epochs"), closed="left"
    )
    # A missing frame is filled when it lies in a bridged gap between two
    # observed frames, the same rule every analysis applies to missing samples.
    bridged = observed_interval_mask(
        t,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=observed,
        result="no missing frame is filled",
    )
    heading = np.full(t.size, np.nan)
    heading[observed] = np.arctan2(dy[observed], dx[observed])
    fill = np.zeros(t.size, dtype=bool)
    fill[:-1] = ~observed[:-1] & bridged
    return _interpolate_heading_circular(t, heading, fill)


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
        Turn angles in radians, in the range [-π, π] (unsigned, in [0, π],
        for three or more columns; see Notes). Consecutive duplicate
        positions (stationary periods) are dropped first, so the length is
        ``n_unique - 2``, where ``n_unique`` is the number of positions left,
        not ``n_samples - 2``. A turn whose incoming or outgoing step touches a
        row with a non-finite coordinate is NaN in any number of dimensions:
        missing samples are not bridged here. The output is not aligned with
        the input samples and carries no times; use
        ``compute_planar_turn_angles`` for a sample-aligned result that
        bridges missing samples.

    Notes
    -----
    Turn angles are computed from the angle between consecutive movement vectors:

    .. math::

        \\theta_i = \\text{atan2}(v_i \\times v_{i+1}, v_i \\cdot v_{i+1})

    where :math:`v_i` is the movement vector from position i-1 to position i.

    With ``v1`` and ``v2`` the incoming and outgoing movement vectors, the
    planar angle is ``atan2(v1_x * v2_y - v1_y * v2_x, v1 . v2)``. One column
    gives 0 (same direction) or π (reversal). With three or more columns
    a turn has no sign without a reference axis, so the result is the
    unsigned angle between ``v1`` and ``v2`` in the full space, in
    ``[0, π]``: the positions ``[[0, 0, 0], [1, 0, 1], [1, 1, 2]]`` give
    ``π/3``, whereas the x-y projection ``positions[:, :2]`` gives the signed
    ``π/2``. For a horizontal-plane analysis pass ``positions[:, :2]``
    explicitly.

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
    >>> # With three columns the turn is the unsigned 3-D angle
    >>> positions_3d = np.array([[0.0, 0, 0], [1, 0, 1], [1, 1, 2]])
    >>> round(float(compute_turn_angles(positions_3d)[0]), 12)  # π/3
    1.047197551197
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

    # Vectorized turn angle computation. Consecutive retained positions
    # differ, so no step has zero length (NaN rows give NaN steps).
    v1 = vectors[:-1]  # shape (n_angles, n_dims)
    v2 = vectors[1:]  # shape (n_angles, n_dims)

    n_dims = vectors.shape[1]
    angles: NDArray[np.float64]
    if n_dims == 2:
        # 2D cross product (scalar): v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        # Dot product
        dot = np.sum(v1 * v2, axis=1)
        angles = np.arctan2(cross, dot)
    elif n_dims > 2:
        # Unsigned angle between the full vectors (no turn sign exists in 3D
        # without a reference axis), by Kahan's formula, accurate near 0 and
        # pi: 2 atan2(|u - w|, |u + w|) for unit vectors u, w.
        u = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        w = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        angles = 2.0 * np.arctan2(
            np.linalg.norm(u - w, axis=1), np.linalg.norm(u + w, axis=1)
        )
    else:
        # 1D: 0 if same direction, π if opposite, NaN next to a missing
        # sample (as in 2D, where NaN propagates through atan2).
        product = v1[:, 0] * v2[:, 0]
        angles = np.where(product > 0, 0.0, np.pi)
        angles[np.isnan(product)] = np.nan

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
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Incoming and outgoing displacement at each sample of a planar path.

    Missing samples are skipped as if deleted: the neighbours of a valid
    sample are the valid samples before and after it, across bridged gaps.

    Returns
    -------
    times : NDArray[np.float64], shape (n_samples,)
        Validated sample times in seconds.
    a, b : NDArray[np.float64], shape (n_samples, 2)
        ``positions[i] - positions[h]`` and ``positions[j] - positions[i]``,
        where ``h`` and ``j`` are the previous and next valid samples; NaN
        rows where ``supported`` is False.
    supported : NDArray[np.bool_], shape (n_samples,)
        True at valid samples whose two neighbouring gaps are both usable and
        whose two displacements are both nonzero.
    span : NDArray[np.float64], shape (n_samples,)
        ``times[j] - times[h]``; NaN where ``supported`` is False.
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
                    "the movement direction are defined in a plane; in 3-D a turn "
                    "has no sign without a reference axis."
                ),
                fix="pass positions[:, :2] to analyze movement in the horizontal plane",
            )
        )
    n_samples = t.size
    a = np.full((n_samples, 2), np.nan)
    b = np.full((n_samples, 2), np.nan)
    span = np.full(n_samples, np.nan)
    supported = np.zeros(n_samples, dtype=bool)
    if n_samples >= 3:
        start, end, _ = gap_endpoints(support.sample_valid)
        previous, following = start[:-1], end[1:]
        with np.errstate(invalid="ignore"):  # inf - inf on missing rows
            a[1:-1] = p[1:-1] - p[previous]
            b[1:-1] = p[following] - p[1:-1]
        span[1:-1] = t[following] - t[previous]
        supported[1:-1] = (
            support.sample_valid[1:-1]
            & support.interval_mask[:-1]
            & support.interval_mask[1:]
        )
        supported &= np.any(a != 0, axis=1) & np.any(b != 0, axis=1)
    a[~supported] = np.nan
    b[~supported] = np.nan
    span[~supported] = np.nan
    return t, a, b, supported, span


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
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute the signed turn in movement direction at each tracking sample.

    At sample ``i`` the turn is the angle from the incoming displacement
    ``a = positions[i] - positions[i - 1]`` to the outgoing displacement
    ``b = positions[i + 1] - positions[i]``, in the horizontal (x, y) plane.
    Positive turns are counterclockwise (to the animal's left, seen from above).
    A missing sample is skipped as if deleted when the valid samples around
    it are within ``max_gap``: ``i - 1`` and ``i + 1`` are then the previous
    and next valid samples. A turn is never computed across a longer gap, an
    ``epochs`` edge or a declared break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
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
        reversal is ``pi``. NaN at missing samples, at the first and last
        sample of each run (next to a gap longer than ``max_gap``, an
        ``epochs`` edge or a break), and where the incoming or outgoing
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
    _, a, b, _, _ = _planar_turns(
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
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute the signed curvature of the observed path at each tracking sample.

    Curvature at sample ``i`` is one over the radius of the circle through
    positions ``i - 1``, ``i`` and ``i + 1`` in the horizontal (x, y) plane:
    0 on a straight segment, ``1 / r`` on a circle of radius ``r``. It is
    positive when the path bends counterclockwise (to the animal's left, seen
    from above) and negative when it bends clockwise. A missing sample is
    skipped as if deleted when the valid samples around it are within
    ``max_gap``; curvature is never computed across a longer gap, an
    ``epochs`` edge or a declared break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
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
        with ``times``. NaN at missing samples, at the first and last sample
        of each run (next to a gap longer than ``max_gap``, an ``epochs`` edge
        or a break), where the incoming or outgoing displacement is zero, and
        at a reversal of direction (no circle passes through the three
        points).

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
    _, a, b, _, _ = _planar_turns(
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
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute how fast the movement direction turns, in radians per second.

    The movement direction is the direction of each observed displacement
    (allocentric: 0 = East, +pi/2 = North). At sample ``i`` the turn between the
    incoming and outgoing displacement is divided by the time it took, half
    the time between the previous and next valid samples,
    ``(times[next] - times[previous]) / 2``. Positive values are
    counterclockwise (to the animal's left, seen from above). A missing sample
    is skipped as if deleted when the valid samples around it are within
    ``max_gap``; nothing is computed across a longer gap, an ``epochs`` edge
    or a declared break.

    This is not head angular velocity. For head direction, unwrap the measured
    headings and difference them over time yourself.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, 2)
        x, y positions in physical units. Rows with a non-finite coordinate are
        missing. Exactly two columns are required.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
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
        ``times``. NaN wherever ``compute_planar_turn_angles`` is NaN: at
        missing samples, at the first and last sample of each run, and where
        the animal did not move (a stationary sample has no direction and none
        is interpolated).

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
    _, a, b, _, span = _planar_turns(
        times,
        positions,
        call="compute_movement_angular_velocity",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
    )
    angles = _signed_turns(a, b)
    omega: NDArray[np.float64] = angles / (span / 2)
    return omega


def compute_step_lengths(
    positions: NDArray[np.float64],
    *,
    metric: DistanceMetric = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """
    Compute step lengths (distances) between consecutive positions.

    Step lengths quantify the spatial displacement at each step of the trajectory.
    This function supports both Euclidean (straight-line) and geodesic
    (shortest path inside the environment) distance metrics.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions in continuous space. Each row is a position vector.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric to use:
        - "euclidean": Straight-line distance in physical space (ecology standard).
          Fast and appropriate for open environments without obstacles.
        - "geodesic": the length of the shortest path between two
          positions that stays inside the environment: it goes around walls
          and never leaves the environment, and on a track it runs along the
          track's centerline. Requires ``env``.
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        metric="geodesic", ignored otherwise.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Step lengths in the same units as the position coordinates.
        Geodesic steps touching a missing or out-of-environment position are
        NaN. Steps between disconnected parts of the environment are infinite.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.
        If positions is not a 2D array or has no rows.
        If metric is not "euclidean" or "geodesic".

    Notes
    -----
    **Euclidean distance** (default):

    .. math::

        d_i = ||r_{i+1} - r_i||_2 = \\sqrt{\\sum_j (r_{i+1,j} - r_{i,j})^2}

    This is the straight-line distance in physical space, matching ecology
    literature standards (Kays et al. 2015, Traja).

    **Geodesic distance** (shortest path inside the environment):

    The animal moves continuously inside the environment, so a step's length
    is the shortest path between its two positions that stays inside. Two
    positions in the same or neighboring bins, or whose bin centers see each
    other, are the straight step apart, so on an open field and between
    nearby samples the geodesic and Euclidean steps agree. Otherwise (the
    straight line crosses a wall, as across a bridged tracking gap) the step
    is the route around the walls; its waypoints are bin centers, so it
    keeps about half a bin from each corner. On a track the step runs along
    the centerline between the two positions' projections.

    The length is not measured bin center to bin center along the graph:
    that staircase of grid steps made a straight oblique run read up to a
    third longer than it is, by an amount that changed with the bin size and
    the sampling rate.

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
    >>> # In an open arena the geodesic steps are the straight steps
    >>> from neurospatial import Environment
    >>> x, y = np.meshgrid(np.arange(0.0, 101.0), np.arange(-10.0, 11.0))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)
    >>> geodesic = compute_step_lengths(positions, metric="geodesic", env=env)
    >>> bool(np.allclose(geodesic, step_lengths))
    True

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

    if len(positions) == 0:
        raise empty_samples_error("compute_step_lengths", ("positions",))

    paths = metric_paths(metric, env, call="compute_step_lengths")
    if paths is None:
        step_lengths: NDArray[np.float64] = np.linalg.norm(
            np.diff(positions, axis=0), axis=1
        )
        return step_lengths
    return paths.between(positions[:-1], positions[1:])


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
        Interval lengths; NaN on unusable intervals. A bridged gap's straight
        displacement is the length of its last interval (the one ending at
        the next valid sample); its other intervals are NaN.
    cumulative : NDArray[np.float64], shape (n_samples,)
        Distance accumulated over usable intervals only, so it is flat across
        every break. Defined for every sample, including those outside runs.
    """
    start, end, _ = gap_endpoints(support.sample_valid)
    with np.errstate(invalid="ignore"):  # inf - inf on consecutive missing rows
        raw = np.linalg.norm(positions[end] - positions[start], axis=1)
    counted = support.interval_mask & support.sample_valid[1:]
    steps = np.where(counted, raw, np.nan)
    cumulative = np.r_[0.0, np.cumsum(np.where(counted, raw, 0.0))]
    return steps, cumulative


def compute_cumulative_distance(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> CumulativeDistanceResult:
    """
    Compute observed distance travelled, never crossing a recording break.

    Distance is added between consecutive valid samples (finite
    coordinates) when the time between them is at most ``max_gap``, they fall
    inside one of ``epochs``, and no break is declared between them; missing
    samples in between are skipped as if their rows were deleted. A longer
    gap or a break therefore splits the path into runs instead of adding the
    jump across it or turning the total into NaN.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Positions in physical units. A 1-D array is one coordinate (for example
        a linearized track). Rows with a non-finite coordinate are missing.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
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
    ``compute_step_lengths(metric="geodesic")`` for steps along the shortest
    paths inside an environment.

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
        run_ids=support.run_ids,
        sample_valid=support.sample_valid,
        run_sample_bounds=bounds,
        run_time_bounds=support.run_time_bounds,
        run_path_lengths=cumulative[bounds[:, 1]] - cumulative[bounds[:, 0]],
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
    )


def _sample_dwell_times(
    times: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    sample_valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Sample dwell from valid intervals, padding each run's final sample.

    A bridged gap's time goes to the valid sample that starts it, and each
    run's last sample receives the median gap of the run, as if the missing
    rows were deleted.
    """
    from neurospatial._intervals import hold_last_valid, run_sample_bounds

    dt = np.diff(times)
    held = hold_last_valid(np.arange(len(times)), sample_valid, fill=-1)[:-1]
    used = interval_mask & (held >= 0)
    weights: NDArray[np.float64] = np.bincount(
        held[used], weights=dt[used], minlength=len(times)
    ).astype(np.float64)
    for first, last in run_sample_bounds(interval_mask):
        valid_times = times[first : last + 1][sample_valid[first : last + 1]]
        weights[last] = np.median(np.diff(valid_times))
    return weights


def compute_home_range(
    position_bins: NDArray[np.int_],
    *,
    times: NDArray[np.float64] | None = None,
    percentile: float = 95.0,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.int_]:
    """
    Compute home range as the most-occupied bins holding a percent of time.

    The home range is the set of most-occupied bins that together hold at
    least ``percentile`` percent of the observed time (or samples): the
    isopleth of the binned utilization distribution. It is not a minimum
    convex polygon; it follows the shape of the occupied area, holes included.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Sequence of bin indices representing the trajectory. ``-1`` marks a
        sample outside the environment or missing; it is in no bin, so it
        adds nothing to the home range or to the total time.
    times : NDArray[np.float64], shape (n_samples,), optional
        Finite, strictly increasing timestamps (seconds), one per sample.
        When provided, occupancy is time-weighted: each sample contributes its
        observed interval dwell rather than a unit count. This is more
        accurate for non-uniformly sampled trajectories. Each run's final sample is assigned
        that run's median interval (it has no successor within the run). When ``None``
        (default), occupancy is the visit count per bin, which is correct only
        when sampling is uniform.
    percentile : float, default=95.0
        Percent of observed time the home range holds, in ``(1, 100]``
        (percent, not a fraction). Common values:
        - 50%: core area (most frequently used)
        - 95%: standard home range
        - 100%: all visited bins
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
        A value other than the default requires ``times``.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted. Requires ``times``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. Requires ``times``.

    Returns
    -------
    NDArray[np.int_], shape (n_bins_in_range,)
        Bin indices in the home range, sorted by occupancy (most visited first).

    Raises
    ------
    TypeError
        If ``position_bins`` is not an integer array.
    ValueError
        If ``position_bins`` holds a value below -1; if ``percentile`` is not
        in ``(1, 100]``; if ``times`` is not finite, strictly increasing and
        one per sample; if ``break_before`` is not a Boolean array with one
        entry per sample; or if ``max_gap``, ``epochs`` or ``break_before`` is
        given without ``times``.

    Notes
    -----
    When ``times`` is given, dwell comes only from valid intervals. A sample
    outside the environment (bin ``-1``) is missing and is bridged like a
    deleted row when the located samples around it are within ``max_gap``:
    the gap's time goes to the located sample before it. Only a longer gap,
    an ``epochs`` edge or a declared break splits a run. Each run's last
    sample receives that run's median interval duration; samples in no run
    receive zero dwell. Bins with zero observed dwell are omitted. No
    observed dwell gives an empty home range. With ``times=None``, counts are used, and ``max_gap``, ``epochs``
    or ``break_before`` raise because they are judged on the clock.

    The home range is computed by:

    1. Computing occupancy per bin (visit counts, or time-weighted dwell time
       when ``times`` is provided).
    2. Sorting bins by occupancy, most occupied first.
    3. Finding the cutoff bin, where the cumulative occupancy first reaches
       ``percentile`` percent of the total.
    4. Keeping every bin with at least the cutoff bin's occupancy.

    Bins tied with the cutoff bin (equal occupancy share to 12 decimals, so
    roundoff in summed dwell times does not break a tie) are all kept, so the
    result is a level set of occupancy and never depends on the order of
    tied bins; it can then hold more than ``percentile`` percent. Under
    uniform use, for example, every visited bin is tied and the 50 % home
    range is all of them. Tied bins are listed in ascending bin order.

    In movement ecology the utilization distribution (UD) is the probability
    of finding the animal at each location, and the home range is the UD
    isopleth holding a given probability mass (Worton 1989). Here the UD is
    the binned occupancy, without kernel smoothing.

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
    call = "compute_home_range"
    position_bins = np.asarray(position_bins)
    if position_bins.size and position_bins.dtype.kind not in "iu":
        raise TypeError(
            _format_error(
                f"{call}() needs integer bin indices; got position_bins of dtype "
                f"{position_bins.dtype}.",
                why="a fractional bin index is not a bin, and truncating it "
                "would silently move samples to another bin.",
                fix="pass bins from env.bin_at(positions).",
            )
        )
    if not (np.isfinite(percentile) and 1.0 < percentile <= 100.0):
        hint = (
            f" Did you mean percentile={percentile * 100:g}? percentile is in "
            "percent, not a fraction."
            if 0.0 < percentile <= 1.0
            else ""
        )
        raise ValueError(
            _format_error(
                f"{call}() needs percentile in (1, 100]; got {percentile!r}.{hint}",
                why="percentile is the percent of observed time the home range "
                "holds: 50 for a core area, 95 for the usual home range, 100 "
                "for every visited bin.",
                fix="pass percentile=95.0 or another percent above 1.",
            )
        )
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
        time_gates = [
            name
            for name, given in (
                ("max_gap", max_gap != DEFAULT_MAX_GAP),
                ("epochs", epochs is not None),
                ("break_before", break_before is not None),
            )
            if given
        ]
        if time_gates:
            raise ValueError(
                _format_error(
                    f"{call}() got {', '.join(time_gates)} without times.",
                    why="Without times the home range counts samples per bin; "
                    "gaps, epochs and recording breaks are judged on the "
                    "clock, so they would be silently ignored.",
                    fix=f"pass times= to weight each bin by the dwell time "
                    f"observed inside them, or drop {', '.join(time_gates)}.",
                )
            )
        # Count-based occupancy (assumes uniform sampling).
        unique_bins, occupancy = np.unique(
            position_bins[in_environment], return_counts=True
        )
        occupancy = occupancy.astype(np.float64)
    else:
        # Time-weighted occupancy: each sample contributes its dwell time.
        from neurospatial._observation import _validated_signal_times

        times = _validated_signal_times(times, call=call)
        if len(times) != len(position_bins):
            raise ValueError(
                f"times and position_bins must have same length. "
                f"Got {len(times)} and {len(position_bins)}."
            )
        breaks = validate_break_before(break_before, len(times), call=call)
        interval_mask = observed_interval_mask(
            times,
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
            sample_valid=in_environment,
        )
        sample_weights = _sample_dwell_times(times, interval_mask, in_environment)[
            in_environment
        ]
        # Sum dwell time per bin.
        unique_bins, inverse = np.unique(
            position_bins[in_environment], return_inverse=True
        )
        occupancy = np.zeros(len(unique_bins), dtype=np.float64)
        np.add.at(occupancy, inverse, sample_weights)
        observed = occupancy != 0
        unique_bins, occupancy = unique_bins[observed], occupancy[observed]

    total = float(np.sum(occupancy))
    if total == 0:
        return np.array([], dtype=np.intp)
    # Occupancy shares equal to 12 decimals are ties: summed dwell times
    # differ by roundoff even when the time spent is the same.
    share = np.round(occupancy / total, 12)
    # Most occupied first; ties in ascending bin order (np.unique sorted them).
    order = np.lexsort((unique_bins, -share))
    sorted_bins, sorted_share = unique_bins[order], share[order]
    cumulative_pct = np.cumsum(occupancy[order]) / total * 100.0
    # First bin at which the cumulative share reaches the percentile, with
    # the same roundoff allowance.
    cutoff = min(
        int(np.searchsorted(cumulative_pct, percentile * (1 - 1e-9), side="left")),
        len(sorted_bins) - 1,
    )
    # The home range is a level set of occupancy: every bin tied with the
    # cutoff bin is in it, so the result never depends on tie order.
    home_range: NDArray[np.intp] = sorted_bins[
        sorted_share >= sorted_share[cutoff]
    ].astype(np.intp)
    return home_range


def mean_square_displacement(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    metric: DistanceMetric = "euclidean",
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
        - "euclidean": Straight-line distance. Use for diffusion analysis,
          comparison with the single-particle-tracking literature, and
          classification of movement types (ballistic vs. diffusive).
        - "geodesic": the length of the shortest path between two
          positions that stays inside the environment: it goes around walls
          and never leaves the environment, and on a track it runs along the
          track's centerline. Requires ``env``. Use for constrained
          navigation analysis.
    env : Environment, optional
        Environment instance for computing geodesic distances. Required if
        metric="geodesic", ignored otherwise.
    max_tau : float, optional
        Longest lag, in seconds. Lags are the whole multiples of the median
        sample interval up to ``max_tau``, at most ``n_samples // 2`` of them.
        ``None`` uses half the recording's duration. Use at most a quarter of
        the duration to average enough pairs at every lag.
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
        If ``max_tau`` is not positive and finite, is longer than the
        recording or is shorter than one sample interval.
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

    The ``k``-th lag is ``k`` times the median sample interval ``dt``. Each
    target time ``t + k * dt`` is matched to the nearest recorded sample
    within ``dt / 2``, and the reported lag is the mean time separation of
    the pairs averaged there, so it is exactly ``k * dt`` for uniform sampling
    and stays the lag actually measured for irregular sampling. A lag with no
    pair reports ``k * dt`` and NaN MSD. Targets beyond the recording are
    excluded.
    Pairs with missing positions are excluded; geodesic pairs additionally
    require both positions to be inside the environment and connected. The
    geodesic metric measures each displacement from the positions
    themselves (not their bin centers) and grows one shortest-path field per
    visited bin, so its memory grows with the square of the number of
    visited bins. A pair
    is also excluded when a declared ``break_before`` or an ``epochs`` edge
    lies between its two samples. A single sample returns an empty result.

    There is no ``max_gap``. MSD compares two observed positions a lag apart
    and does not integrate along the path, so samples missing between them do
    not change the displacement. A gap matters only when the two positions are
    not in the same coordinate frame, which is what ``break_before`` and
    ``epochs`` declare.

    **Movement classification from scaling exponent**:

    MSD scales with time lag τ as :math:`MSD \\sim \\tau^\\alpha`:

    - **α ≈ 0**: Confined motion (MSD plateaus at the size of the enclosure)
    - **0 < α < 1**: Subdiffusive motion (obstructed or partly confined)
    - **α ≈ 1**: Diffusive motion (random walk, Brownian motion)
    - **1 < α < 2**: Superdiffusive motion (persistent, partly directed)
    - **α ≈ 2**: Ballistic motion (straight, constant velocity), the upper
      bound for an animal moving at finite speed

    The exponent α can be estimated from log-log regression: log(MSD) ~ α·log(τ).
    In an arena, every path is confined at lags long enough to cross it, so
    estimate α over lags shorter than that.

    **Why continuous positions?**

    Bin discretization rounds each position to its bin center, so a
    displacement smaller than a bin reads as zero or as a whole bin. Short-lag
    MSD is then wrong in either direction, and the fitted exponent α is
    biased. The single-particle-tracking methods cited below (Saxton 1997;
    Kepten et al. 2013) likewise compute MSD from continuous positions.

    Example: steady movement [0,0]→[0.1,0.1]→[0.2,0.2] stays in one bin and
    gives MSD = 0, whereas continuous positions give MSD ∝ τ² (ballistic).

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
    .. [2] Kepten, E., Bronshtein, I., & Garini, Y. (2013). "Improved
           estimation of anomalous diffusion exponents in single-particle
           tracking experiments." Physical Review E, 87(5), 052713.
    .. [3] Traja documentation: https://traja.readthedocs.io/
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
    paths = metric_paths(metric, env, call="mean_square_displacement")

    if max_tau is not None and not (np.isfinite(max_tau) and max_tau > 0):
        raise ValueError(
            _format_error(
                f"mean_square_displacement() needs max_tau to be a positive, "
                f"finite number of seconds; got {max_tau!r}.",
                why="max_tau is the longest time lag at which displacement is "
                "measured.",
                fix="pass max_tau in seconds, for example a quarter of the "
                "recording, or None for half of it.",
            )
        )

    n_samples = len(positions)
    if n_samples < 2:
        return MSDResult(
            lags=np.empty(0), msd=np.empty(0), metric=metric, epochs=support.epochs
        )

    duration = float(times[-1] - times[0])
    if max_tau is None:
        max_tau = duration / 2.0
    elif max_tau > duration:
        raise ValueError(
            _format_error(
                f"mean_square_displacement() got max_tau={max_tau!r} s, which is "
                f"longer than the recording ({duration:.6g} s).",
                why="no two samples are farther apart in time than the "
                "recording lasts, so no displacement is observed at that lag.",
                fix="pass a shorter max_tau (at most a quarter of the recording "
                "averages enough pairs per lag), and check that times are in "
                "seconds.",
            )
        )

    # The resolver guarantees strictly increasing times, so dt > 0. Lags are
    # whole multiples of the typical interval, so each target time matches a
    # recorded sample; the tolerance absorbs roundoff in max_tau / dt.
    dt = float(np.median(np.diff(times)))
    n_lags = min(int(np.floor(max_tau / dt + 1e-9)), n_samples // 2)
    if n_lags < 1:
        raise ValueError(
            _format_error(
                f"mean_square_displacement() got max_tau={max_tau!r} s, which is "
                f"shorter than one sample interval (median {dt:.6g} s).",
                why="the shortest lag that can be measured is the time between "
                "two consecutive samples.",
                fix=f"pass max_tau of at least {dt:.6g} s.",
            )
        )

    tau_values = dt * np.arange(1, n_lags + 1)
    msd_values = np.full(n_lags, np.nan, dtype=np.float64)
    # Reported lag: the mean time separation of the pairs averaged at each
    # lag (k * dt exactly for uniform sampling); nominal where none exist.
    lag_values = tau_values.copy()

    # Locate the positions and grow the field of every bin they visit once;
    # each geodesic lag is then a lookup.
    pairs = None if paths is None else paths.pairwise(positions)

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

        if pairs is None:
            displacements = positions[future_idx[valid_pairs]] - positions[valid_pairs]
            sq_dists = np.sum(displacements**2, axis=1)
        else:
            # NaN for a position outside the environment, inf between
            # disconnected parts: neither is a displacement.
            distances = pairs.between(
                sample_indices[valid_pairs], future_idx[valid_pairs]
            )
            sq_dists = distances**2

        finite_distances = np.isfinite(sq_dists)
        if np.any(finite_distances):
            msd_values[i] = np.mean(sq_dists[finite_distances])
            separations = times[future_idx[valid_pairs]] - times[valid_pairs]
            lag_values[i] = np.mean(separations[finite_distances])

    return MSDResult(
        lags=lag_values, msd=msd_values, metric=metric, epochs=support.epochs
    )


def compute_trajectory_curvature(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    smooth_window: float = 0.2,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute the smoothed turn angle at each tracking sample.

    Despite its name, this returns turning angles in radians (the signed change
    of movement direction at each sample), not geometric curvature. For
    curvature in inverse length units use ``compute_planar_curvature``.
    Works for any number of columns (1D, 2D, 3D, N-D); with three or more
    columns the turn is unsigned (see Notes). A missing sample is skipped as
    if deleted when the valid samples around it are within ``max_gap``; no
    turn is computed or smoothed across a longer gap, an ``epochs`` edge or a
    declared break.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples,) or (n_samples, n_dims)
        Position coordinates over time in any dimensional space. Rows with a
        non-finite coordinate are missing.
    smooth_window : float, default=0.2
        Standard deviation, in seconds, of the Gaussian that smooths the turn
        angles within each run (0.2 s suits 30-60 Hz tracking). 0 disables
        smoothing.

        **Important**: For high-speed tracking (120+ Hz) or fast-moving animals,
        use shorter windows (0.05-0.1s) or disable smoothing (smooth_window=0)
        to preserve rapid turns.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
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
        - NaN: where no turn is defined: missing samples, the first and last
          sample of each run, repeated (stationary) positions after the
          first, and samples in no run (isolated by a long gap, a declared
          break or ``epochs``)

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` are empty, swapped or misaligned,
        ``times`` is not strictly increasing, ``max_gap`` is not positive and
        finite or ``None``, ``smooth_window`` is not a finite number >= 0, or
        ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    A missing row is bridged like a deleted row when the valid samples around
    it are within ``max_gap``; only a longer gap, an ``epochs`` edge or a
    declared break splits a run. Each run is analyzed as a separate
    recording: turn angles are computed and smoothed within it, so a break
    never spreads into its neighbours.

    Within a run this function wraps `compute_turn_angles()` and adds:
    - Mapping to the original sample positions
    - Optional temporal smoothing

    `compute_turn_angles()` uses atan2(cross, dot) for signed angles in
    [-π, π], drops stationary repeats, and returns at most (n_samples - 2)
    angles. Each angle is assigned to the original index of its middle position
    after consecutive duplicates are removed (keeping their first sample).
    Stationary repeats and the first and last retained positions of a run
    have no turn, so they are NaN rather than zero. Smoothing averages only
    the defined turns (a normalized Gaussian convolution), so an undefined
    turn neither pulls real turns toward zero nor receives a value.

    For positions with three or more columns the turn is the unsigned angle
    between consecutive steps in the full space, in ``[0, π]`` (as in
    `compute_turn_angles()`): a 3-D turn has no sign without a reference
    axis. Pass ``positions[:, :2]`` for a signed horizontal-plane analysis.

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
    >>> turns = compute_trajectory_curvature(times, positions)
    >>> bool(np.isnan(turns[[0, -1]]).all() and np.allclose(turns[1:-1], 0.0))
    True

    A right-angle left turn, without smoothing. The first and last samples
    have no turn:

    >>> times = np.arange(4) * 0.1
    >>> positions = np.array([[0.0, 0], [1, 0], [2, 0], [2, 1]])
    >>> turns = compute_trajectory_curvature(times, positions, smooth_window=0.0)
    >>> np.round(turns, 4).tolist()
    [nan, 0.0, 1.5708, nan]

    A missing row (lost tracking) is NaN and is skipped as if deleted, so
    the turns around it span the bridged gap:

    >>> times = np.arange(7) * 0.1
    >>> positions = np.array(
    ...     [[0.0, 0], [1, 0], [2, 0], [2, 1], [np.nan, np.nan], [3, 1], [3, 2]]
    ... )
    >>> turns = compute_trajectory_curvature(times, positions, smooth_window=0.0)
    >>> np.round(turns, 4).tolist()
    [nan, 0.0, 1.5708, -1.5708, nan, 1.5708, nan]

    A gap longer than ``max_gap`` splits the path into two runs, and no turn
    is computed across it; the two-sample second run has no turn at all:

    >>> turns = compute_trajectory_curvature(
    ...     times, positions, smooth_window=0.0, max_gap=0.15
    ... )
    >>> np.round(turns, 4).tolist()
    [nan, 0.0, 1.5708, nan, nan, nan, nan]

    Smoothing averages only defined turns, so running a circle reads the
    true turn per sample (0.8 rad/s at 30 Hz) from the first defined sample:

    >>> times = np.arange(0, 5, 1 / 30)
    >>> circle = 7.0 * np.c_[np.cos(0.8 * times), np.sin(0.8 * times)]
    >>> turns = compute_trajectory_curvature(times, circle)
    >>> bool(np.allclose(turns[1:-1], 0.8 / 30))
    True

    With three columns the turn is the unsigned 3-D angle between the two
    steps, π/3 here, not the signed π/2 of the x-y projection:

    >>> times = np.array([0.0, 0.1, 0.2])
    >>> x = np.array([[0.0, 0, 0], [1, 0, 1], [1, 1, 2]])
    >>> round(float(compute_trajectory_curvature(times, x, smooth_window=0)[1]), 12)
    1.047197551197
    >>> float(compute_trajectory_curvature(times, x[:, :2], smooth_window=0)[1])
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
    if not (
        isinstance(smooth_window, (int, float, np.number))
        and np.isfinite(smooth_window)
        and smooth_window >= 0
    ):
        raise ValueError(
            _format_error(
                f"compute_trajectory_curvature() needs smooth_window to be a "
                f"finite number of seconds >= 0; got {smooth_window!r}.",
                why="smooth_window is the standard deviation of the Gaussian "
                "that smooths the turns over time.",
                fix="pass smooth_window=0.0 for no smoothing, or a positive "
                "number of seconds such as 0.2.",
            )
        )
    curvature: NDArray[np.float64] = np.full(t.size, np.nan)
    for run in support.run_samples():
        curvature[run] = _compute_trajectory_curvature_contiguous(
            p[run], t[run], smooth_window=smooth_window
        )
    return curvature


def _compute_trajectory_curvature_contiguous(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    smooth_window: float,
) -> NDArray[np.float64]:
    """Map angles to their original samples and smooth within one recording.

    Samples without a turn (the run's ends and stationary repeats) are NaN.
    Smoothing is a normalized convolution over the defined turns only, so an
    undefined turn neither counts as zero nor gets a value.
    """
    angles = compute_turn_angles(positions)
    curvature = np.full(len(positions), np.nan, dtype=np.float64)
    if len(angles):
        changed = np.any(positions[1:] != positions[:-1], axis=1)
        retained_indices = np.flatnonzero(np.r_[True, changed])
        curvature[retained_indices[1:-1]] = angles

    defined = np.isfinite(curvature)
    if smooth_window > 0 and defined.any():
        from scipy.ndimage import gaussian_filter1d

        sigma = smooth_window / float(np.median(np.diff(times)))
        weight = gaussian_filter1d(defined.astype(np.float64), sigma, mode="constant")
        total = gaussian_filter1d(
            np.where(defined, curvature, 0.0), sigma, mode="constant"
        )
        curvature[defined] = total[defined] / weight[defined]
    return curvature
