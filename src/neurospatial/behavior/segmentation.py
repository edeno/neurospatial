"""Behavioral segmentation and trajectory analysis.

This module segments trajectories by regions, laps, speed and approaches to
a goal. Trials between regions are in :mod:`neurospatial.behavior.trials`.

Imports
-------
All segmentation functions are accessible from this module:

>>> from neurospatial.behavior.segmentation import (
...     Crossing,
...     Lap,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
...     detect_goal_directed_runs,
...     segment_by_velocity,
...     trajectory_similarity,
... )

Or import from the behavior package:

>>> from neurospatial.behavior import (
...     Crossing,
...     Lap,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
... )

Functions
---------
detect_region_crossings
    Detect entry and exit events for a spatial region
segment_by_velocity
    Movement epochs from a speed threshold
detect_laps
    Detect laps on circular tracks
trajectory_similarity
    Compare similarity between two trajectories
detect_goal_directed_runs
    Efficient approaches to a goal

Classes
-------
Trial
    One trial: start_time, stop_time, start_region, end_region (None unless
    reached) and outcome ('reached', 'returned' or 'censored')
Crossing
    One region entry or exit
Lap
    One lap on a circular track
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from neurospatial._exceptions import _format_error
from neurospatial._interval_table import Interval, Intervals
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    run_sample_bounds,
    start_allocation,
    times_in_intervals,
    warn_excluded_intervals,
)
from neurospatial._observation import validate_break_before
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._validation import validate_choice, validate_finite_scalar
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.behavior._position_bins import bridge_missing_positions, sample_bins
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from neurospatial.behavior._environment_paths import EnvironmentPaths
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem
    from neurospatial.ops._types import DistanceMetric


def _region_names(
    regions: str | Sequence[str],
    *,
    argument: str,
    why: str = "each attempt must finish at a named region.",
) -> list[str]:
    """One region name or a sequence of them, as a list.

    A plain string is one name; iterating it would give its characters.
    ``why`` explains, in the error for an empty sequence, what the regions
    are for.
    """
    names = [regions] if isinstance(regions, str) else list(regions)
    if not names:
        raise ValueError(
            _format_error(
                f"{argument} cannot be empty",
                fix=f"pass {argument}=['goal']",
                why=why,
            )
        )
    return names


DEFAULT_MIN_PROGRESS_CM = 20.0
"""Default net progress toward the goal of a goal-directed run, in cm.

About one rat body length, so a run must bring the animal at least one body
length closer to the goal; expressed in ``env.units`` when they are metric.
"""


def _event_runs(
    times: NDArray[np.float64],
    position_bins: NDArray[np.int64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None,
) -> list[NDArray[np.intp]]:
    """Observed runs holding only samples inside the half-open ``epochs``.

    Takes the samples left after NaN rows were dropped
    (:func:`~neurospatial.behavior._position_bins.bridge_missing_positions`,
    which also warns when nothing is observed). As in ``env.occupancy``, an
    interval is observed when its gap passes ``max_gap``, ``epochs`` and
    ``break_before`` and both of its samples are in a bin: a sample at bin
    -1 (outside the environment) ends the run.

    An observed run keeps the sample at an epoch's ``stop``, because the
    interval that ends there lies inside the epoch. That instant is outside
    ``[start, stop)``, so an event detector must not report an entry, exit
    or trial end there: such a run ends at its last sample before ``stop``,
    as :func:`detect_region_visits` does.
    """
    gates = start_allocation(
        times,
        position_bins,
        None,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=break_before,
    ).gates
    runs = [
        np.arange(first, last + 1, dtype=np.intp)
        for first, last in run_sample_bounds(gates.mask)
    ]
    windows = as_intervals(epochs, name="epochs")
    if windows is None:
        return runs
    inside = times_in_intervals(windows, times)
    trimmed = [run[inside[run]] for run in runs]
    return [run for run in trimmed if run.size]


# =============================================================================
# Event records
# =============================================================================

Outcome = Literal["reached", "returned", "censored"]
"""How a trial ended.

``"reached"``: the animal arrived at a different region. ``"returned"``: it
came back to the region it left. ``"censored"``: observation stopped first,
at a tracking gap longer than ``max_gap``, an ``epochs`` edge, a declared
``break_before`` or the end of the recording, so the trial's result is
unknown.
"""

_OUTCOMES: tuple[str, ...] = ("reached", "returned", "censored")


def _checked_interval(record: str, start: object, end: object) -> tuple[float, float]:
    """Return ``(start, end)`` as finite Python floats with ``start <= end``."""
    start_f, end_f = float(start), float(end)  # type: ignore[arg-type]
    if not (np.isfinite(start_f) and np.isfinite(end_f)) or start_f > end_f:
        raise ValueError(
            _format_error(
                f"{record} needs finite start_time <= stop_time; got "
                f"start_time={start_f!r}, stop_time={end_f!r}.",
                why="an event interval runs forward in time on one clock.",
                fix="pass the interval's start and end times in seconds, start first.",
            )
        )
    return start_f, end_f


@dataclass(frozen=True)
class Crossing:
    """Region crossing event.

    Attributes
    ----------
    time : float
        Time of crossing event (seconds): the first sample inside the region
        for an entry, the first sample outside it for an exit.
    direction : {'entry', 'exit'}
        Direction of crossing.
    bin_index : int
        Bin index of the sample at ``time``.
    region : str
        Name of the region crossed.
    """

    time: float
    direction: Literal["entry", "exit"]
    bin_index: int
    region: str

    def __post_init__(self) -> None:
        """Check the direction and store plain Python scalars."""
        validate_choice(
            self.direction, ("entry", "exit"), name="direction", owner="Crossing"
        )
        time = float(self.time)
        if not np.isfinite(time):
            msg = f"Crossing.time must be finite; got {time!r}."
            raise ValueError(msg)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "bin_index", int(self.bin_index))


@dataclass(frozen=True)
class Lap:
    """Detected lap on circular track.

    Attributes
    ----------
    start_time : float
        Start time of lap (seconds).
    stop_time : float
        End time of lap (seconds); the lap is the half-open window
        ``[start_time, stop_time)``.
    direction : str
        Direction of lap: 'clockwise', 'counter-clockwise', or 'unknown'.
    overlap_score : float
        Jaccard overlap of the lap's bins with the template lap's bins, both
        widened by one ring of neighbor bins, in [0, 1], for the
        ``'auto'`` and ``'reference'`` methods; NaN for the ``'region'``
        method, which uses no template.
    """

    start_time: float
    stop_time: float
    direction: Literal["clockwise", "counter-clockwise", "unknown"]
    overlap_score: float

    def __post_init__(self) -> None:
        """Check the record and store plain Python floats."""
        start, stop = _checked_interval("Lap", self.start_time, self.stop_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "stop_time", stop)
        validate_choice(
            self.direction,
            ("clockwise", "counter-clockwise", "unknown"),
            name="direction",
            owner="Lap",
        )
        score = float(self.overlap_score)
        if not (np.isnan(score) or 0.0 <= score <= 1.0):
            msg = f"Lap.overlap_score must be in [0, 1] or NaN; got {score!r}."
            raise ValueError(msg)
        object.__setattr__(self, "overlap_score", score)


@dataclass(frozen=True)
class Trial:
    """One trial: the run from leaving one region to arriving at the next.

    Trials come from :func:`~neurospatial.behavior.detect_trials` (tracked
    positions) or :func:`~neurospatial.behavior.trials_between_visits` (visits
    you bring, such as port events).

    Attributes
    ----------
    start_time : float
        Departure: the first sample outside the start region (seconds).
    stop_time : float
        Time when the trial ended (seconds); see ``outcome``. The trial is
        the half-open window ``[start_time, stop_time)``, so the sample at
        ``stop_time`` (for example the arrival) is outside it.
    start_region : str
        Name of the region the trial started from (non-empty).
    end_region : str or None
        Name of the region reached, different from ``start_region``; None
        unless ``outcome == 'reached'``.
    outcome : {'reached', 'returned', 'censored'}
        How the trial ended:

        - ``'reached'``: arrived at a different region; ``stop_time`` is the
          first sample inside it.
        - ``'returned'``: came back to the start region first; ``stop_time``
          is the first sample back inside it.
        - ``'censored'``: observation stopped first (a tracking gap longer
          than ``max_gap``, an ``epochs`` edge, a ``break_before`` or the end
          of the recording); ``stop_time`` is the last observed time, and
          the trial's result is unknown. Leave censored trials out of
          success rates rather than counting them as errors.
    success : bool
        Read-only property, ``outcome == 'reached'``.
    """

    start_time: float
    stop_time: float
    start_region: str
    end_region: str | None
    outcome: Outcome

    def __post_init__(self) -> None:
        """Check the record and store plain Python floats."""
        start, stop = _checked_interval("Trial", self.start_time, self.stop_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "stop_time", stop)
        validate_choice(self.outcome, _OUTCOMES, name="outcome", owner="Trial")
        for name in ("start_region", "end_region"):
            value = getattr(self, name)
            if name == "end_region" and value is None:
                continue
            if not isinstance(value, str) or value == "":
                error = ValueError if isinstance(value, str) else TypeError
                raise error(
                    _format_error(
                        f"Trial.{name} must be a region name (a non-empty str); "
                        f"got {value!r}.",
                        why="a trial runs between named regions of the environment.",
                        fix="map port or well numbers to the environment's region "
                        "names first"
                        + (
                            ", and pass end_region=None for a trial that reached "
                            "no region."
                            if name == "end_region"
                            else "."
                        ),
                    )
                )
        if (self.outcome == "reached") != (self.end_region is not None):
            raise ValueError(
                _format_error(
                    f"Trial.end_region={self.end_region!r} does not match "
                    f"outcome={self.outcome!r}.",
                    why="end_region names the end region reached, so it is set "
                    "exactly when outcome == 'reached'.",
                    fix="pass end_region='<name>' with outcome='reached', and "
                    "end_region=None with any other outcome.",
                )
            )
        if self.outcome == "reached" and self.end_region == self.start_region:
            raise ValueError(
                _format_error(
                    f"Trial reached its own start region {self.start_region!r}.",
                    why="'reached' means arriving at a different region; coming "
                    "back to the start region is 'returned'.",
                    fix="pass outcome='returned' and end_region=None for a trial "
                    "that came back, or the different region it reached.",
                )
            )

    @property
    def success(self) -> bool:
        """Whether the trial reached an end region (``outcome == 'reached'``)."""
        return self.outcome == "reached"


@reads_pynapple_samples
def detect_region_crossings(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    region_name: str,
    direction: Literal["both", "entry", "exit"] = "both",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> list[Crossing]:
    """Detect region entry and exit events in a trajectory.

    Identifies time points where the trajectory enters or exits a named region.
    Useful for analyzing exploration patterns, region preference, and behavioral
    transitions.

    Parameters
    ----------
    env : Environment
        Environment containing the named region.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where tracking was
        lost, bridged like a deleted row (see ``max_gap``). Each is located
        in its bin inside the function.
    region_name : str
        Name of region to detect crossings for. Must exist in env.regions.
        Passed as a keyword-only argument.
    direction : {'both', 'entry', 'exit'}, optional
        Which crossings to detect:
        - 'both': detect entries and exits (default)
        - 'entry': only detect entries
        - 'exit': only detect exits
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
        counts only if it lies entirely inside one window, and a sample at a
        window's ``stop`` is outside it, so no event is reported there.
        ``None`` (default) means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    list[Crossing]
        List of crossing events, each with time, direction, bin_index and
        region name. Sorted by time in ascending order.

    Raises
    ------
    ValueError
        If region_name not in env.regions, or direction is not one of
        'both', 'entry', 'exit'.
    ValueError
        If ``times`` are not finite and strictly increasing, or
        ``positions`` do not have one row per time with the environment's
        number of coordinates.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_trials : Trials between visits to named regions.
    segment_by_velocity : Segment trajectory by velocity threshold

    Notes
    -----
    A NaN position (lost tracking, a low-confidence frame) is a missing
    sample, the same as a deleted row: the samples around it are joined
    when no more than ``max_gap`` apart, and a longer gap splits the
    recording. A position outside the environment is observed but not
    located, so an interval with a sample there is not observed and splits
    the recording too, like an ``epochs`` edge or a ``break_before``. Each
    resulting stretch is analyzed on its own, so no crossing is reported
    across a split.

    A crossing is detected when the trajectory transitions from outside to inside
    the region (entry) or inside to outside (exit). The crossing time is assigned
    to the first sample inside (for entry) or outside (for exit) the region.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> # Create 2D environment and add region
    >>> x, y = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("goal", polygon=Point(50.0, 50.0).buffer(10.0))
    >>> env = env.with_regions(regions)
    >>> # Create trajectory that crosses region
    >>> traj_x = np.array([10.0, 30.0, 50.0, 70.0, 50.0, 30.0])
    >>> traj_y = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    >>> trajectory = np.column_stack([traj_x, traj_y])
    >>> times = np.arange(len(trajectory), dtype=float)
    >>> # Treat these 1 Hz samples as continuous with max_gap=None
    >>> crossings = detect_region_crossings(
    ...     env,
    ...     times,
    ...     trajectory,
    ...     region_name="goal",
    ...     direction="both",
    ...     max_gap=None,
    ... )
    >>> len(crossings) > 0  # Should detect entries and exits
    True
    """
    region_mask = _require_region_mask(
        env, region_name, operation="detect_region_crossings()", argument="region_name"
    )
    validate_choice(
        direction,
        ("both", "entry", "exit"),
        name="direction",
        owner="detect_region_crossings()",
    )

    times, position_bins = sample_bins(
        env, times, positions, call="detect_region_crossings"
    )
    times, position_bins, breaks, _ = bridge_missing_positions(
        times,
        position_bins,
        positions,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=validate_break_before(
            break_before, len(times), call="detect_region_crossings"
        ),
    )
    results: list[Crossing] = []
    for run in _event_runs(
        times,
        position_bins,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
    ):
        results.extend(
            _detect_region_crossings_contiguous(
                position_bins[run],
                times[run],
                region_mask,
                region_name=region_name,
                direction=direction,
            )
        )
    return results


def _detect_region_crossings_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    region_mask: NDArray[np.bool_],
    *,
    region_name: str,
    direction: Literal["both", "entry", "exit"] = "both",
) -> list[Crossing]:
    """Detect crossings within one recording, after public input validation.

    ``region_mask`` (shape ``(n_bins,)``) marks the region's bins.
    """
    in_region = region_mask[position_bins]
    # Samples whose region membership differs from the sample before, in time
    # order: an entry where the sample is inside, an exit where it is outside.
    changes = np.flatnonzero(in_region[1:] != in_region[:-1]) + 1
    kinds = np.where(in_region[changes], "entry", "exit")
    keep = kinds == direction if direction != "both" else slice(None)
    return [
        Crossing(
            time=times[idx],
            direction=kind,
            bin_index=position_bins[idx],
            region=region_name,
        )
        for idx, kind in zip(changes[keep].tolist(), kinds[keep].tolist(), strict=True)
    ]


@reads_pynapple_samples
def segment_by_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    speed: ArrayLike | None = None,
    min_duration: float = 0.5,
    hysteresis: float = 2.0,
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> Intervals[Interval]:
    """Find movement epochs: stretches where speed stays above a threshold.

    Uses hysteresis thresholding on the smoothed speed to find movement
    epochs and returns them as half-open ``[start, stop)`` intervals; time
    outside every interval is rest or unobserved. Epochs shorter than ``min_duration`` are
    dropped. Useful for running-bout analysis and for restricting place-field
    and decoding analyses to movement with ``epochs=``.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to positions (seconds).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Continuous position samples (e.g., in cm).
    min_speed : float
        Velocity threshold for movement classification (units/second).
        Samples with velocity > ``min_speed`` are considered movement.
    min_duration : float, optional
        Minimum segment duration in seconds. Default: 0.5.
        Segments shorter than this are excluded.
    hysteresis : float, optional
        Hysteresis factor on ``min_speed``. Default: 2.0.
        - Movement starts when velocity > ``min_speed``
        - Movement ends when velocity < ``min_speed / hysteresis``
        Hysteresis prevents rapid switching near the threshold.
    smooth_window : float, default=0.1
        Standard deviation, in seconds, of the Gaussian that smooths the
        velocity vector before its magnitude is thresholded, as in
        :func:`compute_speed`, so tracking jitter at rest is not movement.
        0 thresholds the unsmoothed central-difference speed. Not used when
        ``speed`` is given.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed`.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
        For counting, the trade-off differs: a loss longer than ``max_gap``
        in the middle of a run splits it in two, so recordings with
        multi-second losses during runs count runs better with a larger
        value (``scripts/calibrate_max_gap.py``).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    Intervals of Interval
        Movement epochs in seconds, in time order, ids ``0`` to ``n - 1``,
        with no fields. Pass them as ``epochs=`` to any analysis to restrict
        it to movement; ``.windows`` is the ``(n_epochs, 2)`` array.
        ``analysis_settings`` records the thresholds, ``max_gap`` and
        ``epochs``.

    Raises
    ------
    ValueError
        If positions and times have different lengths, or are empty.
    ValueError
        If ``min_speed`` is not finite and positive, ``min_duration`` or
        ``smooth_window`` is negative or not finite, or ``hysteresis`` is not
        a finite number greater than 1.
    ValueError
        If times contains non-finite values, or is not strictly increasing
        (duplicate or out-of-order timestamps would yield ``inf`` velocities).
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_trials : Trials between visits to named regions.
    detect_region_crossings : Detect region entry/exit events

    Notes
    -----
    A missing sample (a position row with a NaN coordinate) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

    Smoothing and hysteresis are applied separately within each recording.
    A movement epoch in progress at its end is truncated there.

    Speed is the speed of :func:`compute_speed`: the magnitude of the slope
    of a straight line fitted to the positions around each sample, weighted
    by a Gaussian of standard deviation ``smooth_window`` seconds. Near the
    end of a recording the fit uses the samples on one side, so movement that
    reaches the end is not suppressed.

    Hysteresis thresholding prevents "flickering" near the threshold:
    - Movement starts at the first sample whose speed exceeds ``min_speed``
    - Movement continues until speed drops below ``min_speed / hysteresis``;
      the epoch's ``stop`` is that first slower sample
    - This creates a "buffer zone" for stable state transitions

    Common use cases:
    - Preprocessing for place field analysis (exclude stationary periods)
    - Identifying exploration vs. exploitation epochs
    - Detecting theta sequences (movement) vs. sharp-wave ripples (rest)
    - Computing speed-filtered occupancy maps

    Examples
    --------
    >>> import numpy as np
    >>> # Create trajectory with clear movement and rest
    >>> rest1 = np.zeros((50, 1))
    >>> movement = np.linspace(0, 100, 100)[:, None]
    >>> rest2 = np.ones((50, 1)) * 100
    >>> trajectory = np.vstack([rest1, movement, rest2])
    >>> times = np.linspace(0, 20, len(trajectory))
    >>> # Segment by velocity
    >>> segments = segment_by_velocity(
    ...     times, trajectory, min_speed=2.0, min_duration=0.5
    ... )
    >>> len(segments)  # one movement epoch, [start, stop)
    1
    >>> bool(np.all(segments.durations >= 0.5))  # min_duration
    True
    """
    from neurospatial._observation import validated_observations

    times, positions, support = validated_observations(
        times,
        positions,
        call="segment_by_velocity",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no movement epoch was found",
    )
    validate_finite_scalar(min_speed, name="min_speed", positive=True)
    validate_finite_scalar(min_duration, name="min_duration", positive=False)
    validate_finite_scalar(smooth_window, name="smooth_window", positive=False)
    validate_finite_scalar(hysteresis, name="hysteresis", positive=True)
    if hysteresis <= 1.0:
        msg = f"hysteresis must be > 1.0 for stability. Got {hysteresis}"
        raise ValueError(msg)

    from neurospatial._velocity import smoothed_speed
    from neurospatial.behavior._kinematics import observed_run_samples, validated_motion

    speed = validated_motion(
        speed, name="speed", n_samples=len(times), call="segment_by_velocity"
    )
    speed_given = speed is not None
    if speed is None:
        speed = smoothed_speed(
            times, positions, support.interval_mask, smooth_window=smooth_window
        )
    else:
        speed = np.where(
            observed_run_samples(positions, support.interval_mask), speed, np.nan
        )
    segments = np.concatenate(
        [
            np.empty((0, 2)),
            *(
                _segment_by_velocity_contiguous(
                    speed[run],
                    times[run],
                    min_speed=min_speed,
                    min_duration=min_duration,
                    hysteresis=hysteresis,
                )
                for run in support.run_samples()
            ),
        ]
    )
    windows = as_intervals(epochs, name="epochs")
    return _windows_as_intervals(
        segments,
        fields={},
        settings={
            "min_speed": float(min_speed),
            "min_duration": float(min_duration),
            "hysteresis": float(hysteresis),
            "smooth_window": None if speed_given else float(smooth_window),
            "max_gap": None if max_gap is None else float(max_gap),
            "epochs": None if windows is None else windows.tolist(),
        },
    )


def _segment_by_velocity_contiguous(
    speed: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float,
    min_duration: float,
    hysteresis: float,
) -> NDArray[np.float64]:
    """Movement epochs of one recording, shape (n_epochs, 2).

    ``speed`` and ``times`` are those of the valid samples of one observed
    run, already validated by :func:`segment_by_velocity`.
    """
    # Speed of each transition i -> i+1, read at its end sample: a state
    # change at transition i starts or stops an epoch at times[i + 1].
    speed = speed[1:]

    # Hysteresis: movement switches on when speed exceeds min_speed and off
    # when it drops below min_speed / hysteresis; in between (or NaN) the
    # state holds. Since min_speed / hysteresis < min_speed the two events are
    # exclusive, so the state at i is the last event at or before i.
    switch_on = speed > min_speed
    is_event = switch_on | (speed < min_speed / hysteresis)
    last_event = np.maximum.accumulate(np.where(is_event, np.arange(speed.size), -1))
    is_moving = (last_event >= 0) & switch_on[last_event]

    edges = np.diff(np.concatenate([[False], is_moving, [False]]).astype(np.int8))
    starts = times[np.flatnonzero(edges == 1) + 1]
    # An epoch still in progress at the end of the run is truncated there.
    stops = times[np.minimum(np.flatnonzero(edges == -1) + 1, times.size - 1)]
    keep = (stops - starts) >= min_duration
    return np.column_stack([starts[keep], stops[keep]])


def _windows_as_intervals(
    rows: NDArray[np.float64], *, fields: dict[str, Any], settings: dict[str, Any]
) -> Intervals[Interval]:
    """``[start, stop)`` rows, shape (n, 2), as Intervals with constant fields."""
    return Intervals(
        [
            Interval(float(start), float(stop), tuple(fields.items()))
            for start, stop in rows
        ],
        record_type=Interval,
        fields=tuple(fields),
        analysis_settings=settings,
    )


# =============================================================================
# Laps
# =============================================================================


def _widen(bins: NDArray[np.int64], adjacency: Any) -> NDArray[np.bool_]:
    """Bins visited by a path plus their neighbors, as a mask over bins.

    Two traversals of the same route pass through slightly different bins:
    a sample every few centimeters steps over some bins, and a path near a
    bin edge visits one row or the other. Widening both paths by one ring
    of neighbors makes the overlap measure the route, not the sampling.
    """
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    visited[bins] = True
    widened: NDArray[np.bool_] = visited | (adjacency @ visited > 0)
    return widened


def _detect_lap_direction(
    bin_centers: NDArray[np.float64], bins: NDArray[np.int64]
) -> Literal["clockwise", "counter-clockwise", "unknown"]:
    """Detect lap direction from the signed area of the closed lap path.

    The path is closed back to its first point and centered on its centroid
    before the shoelace sum, so the sign does not depend on where the
    coordinate origin is or at which phase of the circuit the lap starts.
    An open path's shoelace sum is not translation invariant: an 85%
    counter-clockwise arc far from the origin can read as clockwise.

    Parameters
    ----------
    bin_centers : NDArray[np.float64], shape (n_bins, n_dims)
        Bin center coordinates.
    bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices forming the lap.

    Returns
    -------
    str
        'clockwise', 'counter-clockwise', or 'unknown' (fewer than three
        samples, a non-2-D environment, or zero enclosed area).
    """
    if len(bins) < 3 or bin_centers.shape[1] != 2:
        return "unknown"

    # (n_samples, 2), centered so the sum is well conditioned far from 0.
    positions = bin_centers[bins]
    positions = positions - positions.mean(axis=0)
    x, y = positions[:, 0], positions[:, 1]
    # Shoelace over the closed polygon (last point joined to the first).
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    if area > 0:
        return "counter-clockwise"
    if area < 0:
        return "clockwise"
    return "unknown"


@reads_pynapple_samples
def detect_laps(
    env: Environment[CartesianSystem],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["auto", "reference", "region"] = "auto",
    min_overlap: float = 0.8,
    direction: Literal["both", "clockwise", "counter-clockwise"] = "both",
    reference_lap: NDArray[np.int64] | None = None,
    start_region: str | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> Intervals[Lap]:
    """Detect laps in a circular track trajectory.

    Identifies repeating traversals of a circular track using spatial overlap
    with a template lap. Useful for analyzing learning, spatial coding, and
    behavioral consistency across laps.

    Parameters
    ----------
    env : Environment
        Environment containing spatial structure.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where tracking was
        lost, bridged like a deleted row (see ``max_gap``). Each is located
        in its bin inside the function.
    method : {'auto', 'reference', 'region'}, optional
        Method for detecting laps:
        - 'auto': the first complete circuit is the template (default)
        - 'reference': Use user-provided reference_lap
        - 'region': Detect laps as segments between start_region entries
    min_overlap : float, optional
        Minimum Jaccard overlap, in (0, 1], between the bins a lap visits and
        the template's bins (each widened by one ring of neighbors). Default
        is 0.8. Not used for 'region' method.
    direction : {'both', 'clockwise', 'counter-clockwise'}, optional
        Which lap directions to include:
        - 'both': all laps (default)
        - 'clockwise': only clockwise laps
        - 'counter-clockwise': only counter-clockwise laps
    reference_lap : NDArray[np.int64] | None, optional
        Bin sequence of one complete lap, for the 'reference' method, for
        example ``env.bin_at(positions[lap_start:lap_stop])``; its first bin
        marks where laps start. Required when method='reference'.
    start_region : str | None, optional
        Name of start region for 'region' method, drawn across the loop
        track. Required when method='region'.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
        For counting, the trade-off differs: a loss longer than ``max_gap``
        in the middle of a run splits it in two, so recordings with
        multi-second losses during runs count runs better with a larger
        value (``scripts/calibrate_max_gap.py``).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window, and a sample at a
        window's ``stop`` is outside it, so no event is reported there.
        ``None`` (default) means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    Intervals of Lap
        One row per lap in time order, ids ``0`` to ``n - 1``, with the
        fields ``direction`` and ``overlap_score``; each lap is the
        half-open window ``[start_time, stop_time)``. ``analysis_settings``
        records the method, its options, ``max_gap`` and ``epochs``.

    Raises
    ------
    ValueError
        If method is not one of 'auto', 'reference', 'region', direction is
        not one of 'both', 'clockwise', 'counter-clockwise', or min_overlap
        is not in (0, 1].
    ValueError
        If method='reference' but reference_lap is not provided, is empty,
        or holds bins outside the environment.
    ValueError
        If method='region' but start_region not provided.
    ValueError
        If start_region not in env.regions, or (method='region') if it does
        not cut a loop of connected bins into two sides: in an open arena,
        at a dead end, or where the environment's bins do not connect.
    CartesianOnlyError
        If ``env`` is polar: each lap's clockwise or counter-clockwise
        direction is the signed area of Cartesian bin centers.
    ValueError
        If ``times`` are not finite and strictly increasing, or
        ``positions`` do not have one row per time with the environment's
        number of coordinates.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_region_crossings : Detect region entry/exit events
    detect_trials : Trials between visits to named regions.

    Notes
    -----
    A NaN position (lost tracking, a low-confidence frame) is a missing
    sample, the same as a deleted row: the samples around it are joined
    when no more than ``max_gap`` apart, and a longer gap splits the
    recording. A position outside the environment is observed but not
    located, so an interval with a sample there is not observed and splits
    the recording too, like an ``epochs`` edge or a ``break_before``. Each
    resulting stretch is analyzed on its own, so no lap spans a split or a
    recording boundary.

    **Lap Detection Methods:**

    - **Reference**: The first bin of ``reference_lap`` is the start line.
      Each pass through it (the anchor bin and its neighbors, so a fast
      animal cannot step over it) marks a candidate boundary, placed at the
      pass's sample nearest the anchor bin; a pass after a stretch that
      stays within one ring of the start line (jitter, a pause there) does
      not count as a new one. A lap ends at the first boundary at which the
      bins visited since some earlier boundary overlap the template's bins
      with Jaccard index >= ``min_overlap``, and starts at the latest such
      earlier boundary. Both bin sets are widened by one ring of neighbors
      first, so the overlap compares routes rather than which bins a sparse
      sampling happened to hit (``overlap_score`` reports this widened
      index). Boundaries come from places, not sample counts, so laps run
      at any speed are found whole. A turn back through the start line (an
      aborted excursion, however far) belongs to no lap: the next lap
      starts where it returned. A route that crosses its start line midway
      (a figure-eight) is still one lap, since half of it alone does not
      match the template. Each lap covers ``[boundary, boundary)``; laps
      tile the recording when there are no turn backs.
    - **Auto**: As 'reference', with the first complete circuit as the
      template. The start line is the first observed bin. On a loop track
      the circuit is the first trip from the start line that goes round the
      loop, judged as for the 'region' method with the start line as the
      region, so a turn back at the start of the recording is skipped. On a
      track run back and forth or in an open arena, where the start line
      does not cut a loop, it runs from the first sample to the first return
      to the start line after a trip to the far side of the track (a bin at
      least half the greatest path distance away). This first circuit is
      itself the first lap. Useful when no canonical lap is at hand; use
      'reference' when the first circuit is atypical.
    - **Region**: A lap is a trip out of a start region and round the loop
      back into it, useful for well-controlled tasks with clear start zones.
      The start region cuts the loop track into a stretch on each side of
      it. A trip goes round the loop when, staying away from the region, it
      moves from the stretch on one side to the stretch on the other; the
      two stretches meet only at the far side of the loop, so every lap
      passes it. A trip that comes back the way it went (an out-and-back,
      however far) is a turn back and not a lap, and a trip that stays
      within the few bins around the region, such as tracking jitter
      across the region's edge, does not interrupt a lap. Each lap covers
      ``[entry, entry that ends its trip)``, starting at the entry that
      precedes the trip, so time spent on a turn back belongs to no lap.
      When the recording starts inside the start region, the first lap
      starts at the first sample. ``overlap_score`` is NaN.

      "Stay away from the region" is measured in connectivity-graph steps:
      the bins ``k`` and ``k + 1`` steps from the region form a band, and
      the sides are the separate pieces of the first band that falls apart
      (``k = 1`` for a region drawn across the track; a larger ``k`` when
      the region leaves a gap beside it).

    **Direction Detection:**

    For 2D trajectories, direction is the sign of the area enclosed by the
    lap path closed back to its start (shoelace formula on bin centers):
    positive is counter-clockwise, negative clockwise. Closing the path makes
    the answer independent of the coordinate origin and of where on the
    circuit the lap starts.

    **Applications:**

    - Lap-by-lap learning curves
    - Trajectory stereotypy quantification
    - Performance variability across trials
    - Spatial strategy consistency

    Examples
    --------
    Detect laps on a circular track. The first complete circuit is the
    automatic template, and each of the ten circuits is one lap:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.segmentation import detect_laps
    >>> theta = np.linspace(0, 20 * np.pi, 1801)
    >>> x = 50 + 30 * np.cos(theta)
    >>> y = 50 + 30 * np.sin(theta)
    >>> positions = np.column_stack([x, y])
    >>> env = Environment.from_samples(positions, bin_size=3.0)
    >>> times = np.linspace(0, 60, 1801)
    >>> laps = detect_laps(env, times, positions, method="auto")
    >>> len(laps)
    10

    Detect laps with user-provided reference:

    >>> reference = env.bin_at(positions[:181])
    >>> laps = detect_laps(
    ...     env,
    ...     times,
    ...     positions,
    ...     method="reference",
    ...     reference_lap=reference,
    ... )
    >>> all(lap.overlap_score >= 0.8 for lap in laps)
    True

    Filter laps by direction:

    >>> laps_cw = detect_laps(env, times, positions, direction="clockwise")
    >>> laps_ccw = detect_laps(
    ...     env,
    ...     times,
    ...     positions,
    ...     direction="counter-clockwise",
    ... )
    >>> len(laps_ccw) > 0
    True
    >>> len(laps_cw)
    0

    Feed detected laps into directional place fields: each sample takes its
    lap's direction (``None`` outside every lap and in laps of unknown
    direction):

    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.segmentation import detect_laps
    >>> from neurospatial.encoding.spatial import (
    ...     compute_directional_place_fields,
    ... )
    >>> theta = np.linspace(0, 6 * np.pi, 300)
    >>> x = 50 + 30 * np.cos(theta)
    >>> y = 50 + 30 * np.sin(theta)
    >>> positions = np.column_stack([x, y])
    >>> env = Environment.from_samples(positions, bin_size=3.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("start", polygon=Point(80.0, 50.0).buffer(5.0))
    >>> env = env.with_regions(regions)
    >>> times = np.linspace(0, 60, 300)
    >>> laps = detect_laps(env, times, positions, method="region", start_region="start")
    >>> known = laps[laps.direction != "unknown"]
    >>> direction_labels = known.sample_values(times, "direction")
    >>> rng = np.random.default_rng(0)
    >>> spike_times = np.sort(rng.uniform(0, 60, 40))
    >>> fields = compute_directional_place_fields(
    ...     env, spike_times, times, positions, direction_labels, bandwidth=6.0
    ... )
    >>> "counter-clockwise" in fields.firing_rates
    True
    """
    require_cartesian(
        env.coordinate_system,
        "detect_laps()",
        why="Lap direction comes from Cartesian steps between bin centers.",
        fix=(
            "detect laps on an allocentric environment built from the animal's "
            "positions, for example Environment.from_samples(positions, bin_size=...)"
        ),
    )
    validate_choice(
        method, ("auto", "reference", "region"), name="method", owner="detect_laps()"
    )
    if direction not in ("both", "clockwise", "counter-clockwise"):
        raise ValueError(
            _format_error(
                f"direction must be one of 'both', 'clockwise', "
                f"'counter-clockwise'; got {direction!r}.",
                why="an unknown direction would silently match no lap.",
                fix="pass direction='both', 'clockwise' or 'counter-clockwise'.",
            )
        )
    if method != "region" and not (
        np.isfinite(min_overlap) and 0.0 < min_overlap <= 1.0
    ):
        raise ValueError(
            _format_error(
                f"min_overlap must be in (0, 1]; got {min_overlap!r}.",
                why="it is the smallest Jaccard index between a lap's bins and "
                "the template's bins; 0 would accept any stretch as a lap.",
                fix="pass min_overlap=0.8 (the default) or another value in (0, 1].",
            )
        )

    # Validate method-specific requirements
    if method == "reference" and reference_lap is None:
        raise ValueError(
            _format_error(
                "reference_lap is required when method='reference'",
                fix="pass reference_lap=env.bin_at(positions_of_one_complete_lap) when "
                "method='reference'",
                why="Why: reference-based detection compares each traversal to that lap.",
            )
        )

    if method == "region":
        if start_region is None:
            raise ValueError(
                _format_error(
                    "start_region is required when method='region'",
                    fix="pass start_region='home' when method='region'",
                    why="Why: a region-based lap starts and ends at a named region.",
                )
            )
        start_mask = _require_region_mask(
            env, start_region, operation="detect_laps()", argument="start_region"
        )

    times, position_bins = sample_bins(env, times, positions, call="detect_laps")
    times, position_bins, breaks, _ = bridge_missing_positions(
        times,
        position_bins,
        positions,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=validate_break_before(
            break_before, len(times), call="detect_laps"
        ),
    )
    runs = _event_runs(
        times,
        position_bins,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
    )
    windows = as_intervals(epochs, name="epochs")
    settings = {
        "method": method,
        "min_overlap": None if method == "region" else float(min_overlap),
        "direction": direction,
        "start_region": start_region,
        "max_gap": None if max_gap is None else float(max_gap),
        "epochs": None if windows is None else windows.tolist(),
    }
    laps: list[Lap] = []
    if method == "region":
        sides = _region_sides(env, start_mask)
        if sides is None:
            raise ValueError(
                _format_error(
                    f"start_region {start_region!r} does not cut a loop: the "
                    "track around it does not split into one stretch on each side.",
                    why="a region lap is a trip that leaves the start region on "
                    "one side and comes back on the other after going round the "
                    "loop; in an open arena, at a dead end, or where the "
                    "environment's bins do not connect, every trip out is a turn "
                    "back.",
                    fix="draw start_region across the full width of a loop track "
                    "whose bins connect (for example Environment.from_samples on "
                    "densely sampled positions). In an open arena use "
                    "method='auto' or 'reference'; on a track the animal runs "
                    "back and forth use detect_trials.",
                )
            )
        for run in runs:
            laps.extend(
                _detect_laps_region_contiguous(
                    position_bins[run],
                    times[run],
                    env,
                    sides=sides,
                    direction=direction,
                )
            )
        return Intervals(laps, record_type=Lap, analysis_settings=settings)
    if method == "reference":
        assert reference_lap is not None
        from neurospatial.environment import _validate_target_bins

        template = _validate_target_bins(
            reference_lap,
            n_bins=env.n_bins,
            coordinates=env.coordinate_system,
            operation="detect_laps(method='reference')",
            argument="reference_lap",
        )
        if template.size == 0:
            msg = (
                "reference_lap is empty; pass the bin sequence of one complete "
                "lap, for example env.bin_at(positions[lap_start:lap_stop])."
            )
            raise ValueError(msg)
    else:
        found = _first_circuit(position_bins, runs, env)
        if found is None:
            return Intervals(laps, record_type=Lap, analysis_settings=settings)
        template = found
    anchor = int(template[0])
    zone = _start_zone(env, anchor)
    adjacency = nx.to_scipy_sparse_array(
        env.connectivity, nodelist=range(env.n_bins), weight=None, format="csr"
    )
    for run in runs:
        laps.extend(
            _detect_laps_search_contiguous(
                position_bins[run],
                times[run],
                env,
                template=_widen(template, adjacency),
                adjacency=adjacency,
                anchor=anchor,
                zone=zone,
                min_overlap=min_overlap,
                direction=direction,
            )
        )
    return Intervals(laps, record_type=Lap, analysis_settings=settings)


def _detect_laps_region_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    sides: _RegionSides,
    direction: str,
) -> list[Lap]:
    """Find the trips round the loop within one recording.

    A lap runs from the entry that precedes its trip round the loop (the
    recording's first sample if the animal starts in the region) to the
    entry that ends it (see :func:`_trips`), so time spent on a turn back
    belongs to no lap.
    """
    lap_start: int | None = 0 if sides.in_region[position_bins[0]] else None
    laps: list[Lap] = []
    for leave, back, kind in _trips(position_bins, sides):
        if leave == 0:  # the recording starts outside: wait for an entry
            lap_start = back
            continue
        if kind == "near":
            continue  # the animal never got away from the region
        if kind == "round" and lap_start is not None:
            lap_bins = position_bins[lap_start:back]
            lap_direction = _detect_lap_direction(env.bin_centers, lap_bins)
            if direction == "both" or lap_direction == direction:
                laps.append(
                    Lap(
                        start_time=float(times[lap_start]),
                        stop_time=float(times[back]),
                        direction=lap_direction,
                        overlap_score=np.nan,  # no template to overlap
                    )
                )
        lap_start = back
    return laps


def _trips(
    position_bins: NDArray[np.int64], sides: _RegionSides
) -> list[tuple[int, int, Literal["round", "back", "near"]]]:
    """Stretches outside the region that end by entering it, and their kind.

    Each is ``(first sample out, first sample back in, kind)``. A trip goes
    ``"round"`` the loop when, while staying at least ``sides.cut`` steps
    from the region, it moves from the stretch of track on one side of the
    region to the stretch on the other: that change happens only at the far
    side of the loop. A trip that never gets more than ``sides.cut`` steps
    from the region, such as tracking jitter across the region's edge, is
    ``"near"``; any other trip turns ``"back"``. A recording that starts
    outside the region yields its first stretch with first sample 0.
    """
    inside = sides.in_region[position_bins]
    steps = sides.steps[position_bins]
    side = sides.side[position_bins]
    trips: list[tuple[int, int, Literal["round", "back", "near"]]] = []
    for leave, back in run_sample_bounds(~inside).tolist():
        if back == len(inside):  # still outside when the recording stops
            break
        if steps[leave:back].max() <= sides.cut:
            trips.append((leave, back, "near"))
        elif _goes_round(steps[leave:back], side[leave:back], sides.cut):
            trips.append((leave, back, "round"))
        else:
            trips.append((leave, back, "back"))
    return trips


def _goes_round(steps: NDArray[np.float64], side: NDArray[np.intp], cut: int) -> bool:
    """Whether a trip changes side while staying ``cut`` or more steps out."""
    stretches = run_sample_bounds(steps >= cut)
    first, last = side[stretches[:, 0]], side[stretches[:, 1] - 1]
    return bool(np.any((first != last) & (first >= 0) & (last >= 0)))


@dataclass(frozen=True, eq=False)
class _RegionSides:
    """Where each bin lies relative to a start region on a loop track.

    Attributes
    ----------
    in_region : NDArray[np.bool_], shape (n_bins,)
        Bins in the region.
    steps : NDArray[np.float64], shape (n_bins,)
        Fewest connectivity-graph steps from each bin to a region bin; inf
        when none is reachable.
    cut : int
        The step count at which the track splits into separate stretches,
        one on each side of the region.
    side : NDArray[np.intp], shape (n_bins,)
        Which stretch each bin ``cut`` or more steps out belongs to; -1 for
        nearer bins.
    """

    in_region: NDArray[np.bool_]
    steps: NDArray[np.float64]
    cut: int
    side: NDArray[np.intp]


def _region_sides(
    env: Environment, in_region: NDArray[np.bool_]
) -> _RegionSides | None:
    """Split the track around a region into the stretches on each side.

    The bins ``k`` and ``k + 1`` steps from the region form a band; ``cut``
    is the smallest ``k`` at which the band falls into two or more connected
    pieces. A region drawn across a loop track cuts it at once (``k = 1``);
    a region that leaves a gap beside it at the track's edge cuts it once
    ``k`` passes the gap. Each piece is one side, and every bin farther out
    takes the side of the band bin it reaches in the fewest steps without
    coming nearer the region, so the sides meet only at the far side of the
    loop. None when no band splits: the region is not on a loop of connected
    bins (an open arena, a dead end, or an environment whose bins do not
    connect).
    """
    from scipy.sparse import diags_array
    from scipy.sparse.csgraph import connected_components, dijkstra

    from neurospatial.ops.distance import _bin_adjacency

    adjacency = _bin_adjacency(env.connectivity, weight=None, operation="detect_laps()")
    steps = dijkstra(
        adjacency,
        directed=False,
        indices=np.flatnonzero(in_region),
        unweighted=True,
        min_only=True,
    )
    reachable = steps[np.isfinite(steps)]
    farthest = int(reachable.max()) if reachable.size else 0
    for cut in range(1, farthest):
        band = np.flatnonzero((steps == cut) | (steps == cut + 1))
        n_pieces, piece = connected_components(adjacency[band][:, band], directed=False)
        if n_pieces >= 2:
            break
    else:
        return None
    piece_of = np.full(env.n_bins, -1, dtype=np.intp)
    piece_of[band] = piece
    # Spread each band piece outward without stepping nearer the region.
    away = diags_array((steps >= cut).astype(np.float64))
    _, _, nearest = dijkstra(
        away @ adjacency @ away,
        directed=False,
        indices=np.flatnonzero(steps == cut),
        unweighted=True,
        min_only=True,
        return_predecessors=True,
    )
    side = np.where(nearest >= 0, piece_of[np.maximum(nearest, 0)], -1)
    side[steps < cut] = -1
    return _RegionSides(in_region, steps, cut, side.astype(np.intp))


def _distances_from(
    env: Environment, sources: NDArray[np.int_]
) -> tuple[NDArray[np.float64], float]:
    """Geodesic distance from ``sources`` to every bin, and half its finite maximum.

    The second value is the "far side" threshold of the first circuit: on a
    closed loop of length L the farthest point from any place is L/2 away,
    and every full circuit passes it.
    """
    from neurospatial.behavior._environment_paths import environment_paths

    distances = environment_paths(env, call="detect_laps").from_bins(
        np.unique(np.asarray(sources, dtype=np.intp))
    )
    finite = distances[np.isfinite(distances)]
    return distances, 0.5 * float(finite.max()) if finite.size else 0.0


def _start_zone(env: Environment, anchor: int) -> NDArray[np.bool_]:
    """The anchor bin and its graph neighbors, as a mask over bins."""
    zone = np.zeros(env.n_bins, dtype=bool)
    zone[anchor] = True
    zone[list(env.connectivity.neighbors(anchor))] = True
    return zone


def _pass_samples(
    position_bins: NDArray[np.int64],
    env: Environment,
    anchor: int,
    zone: NDArray[np.bool_],
) -> list[int]:
    """One sample per pass through the start zone: the one nearest the anchor.

    A pass is a maximal stretch of consecutive samples inside ``zone``. Ties
    go to the earliest sample.
    """
    passes = run_sample_bounds(zone[position_bins])
    # (n_samples,) distance from each sample's bin center to the anchor's
    offsets = np.linalg.norm(
        env.bin_centers[position_bins] - env.bin_centers[anchor], axis=1
    )
    return [int(a + np.argmin(offsets[a:b])) for a, b in passes.tolist()]


def _first_circuit(
    position_bins: NDArray[np.int64],
    runs: list[NDArray[np.intp]],
    env: Environment,
) -> NDArray[np.int64] | None:
    """Bins of the first complete circuit in the first recording that has one.

    The start zone is the recording's first bin and its neighbors. When the
    zone cuts a loop track into two sides (:func:`_region_sides`), the
    circuit is the first trip out of the zone that goes round the loop, from
    the pass through the zone it leaves to the pass it returns to; a turn
    back before it is skipped. Otherwise (an open arena or a track run back
    and forth) the circuit starts at the first sample and ends at the first
    return to the zone after a trip to the far side of the track. Each pass
    is represented by its sample nearest the first bin. None when no
    recording completes a circuit.
    """
    for run in runs:
        bins = position_bins[run]
        anchor = int(bins[0])
        zone = _start_zone(env, anchor)
        passes = np.asarray(_pass_samples(bins, env, anchor, zone))
        sides = _region_sides(env, zone)
        if sides is not None:
            for leave, back, kind in _trips(bins, sides):
                if kind == "round":
                    start = passes[np.searchsorted(passes, leave) - 1]
                    found: NDArray[np.int64] = bins[
                        start : passes[np.searchsorted(passes, back)]
                    ]
                    return found
            continue
        from_anchor, far_distance = _distances_from(env, np.array([anchor]))
        reached_far = np.maximum.accumulate(from_anchor[bins] >= far_distance)
        for stop in passes[1:]:
            if reached_far[stop - 1]:
                return bins[:stop]
    return None


def _detect_laps_search_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    template: NDArray[np.bool_],
    adjacency: Any,
    anchor: int,
    zone: NDArray[np.bool_],
    min_overlap: float,
    direction: str,
) -> list[Lap]:
    """Split one recording into laps at passes through the template's start.

    A lap ends at the first pass ``p`` for which some stretch from an
    earlier pass to ``p`` overlaps the template; it starts at the latest
    such pass. A turn back through the start line (an aborted excursion)
    therefore starts the next lap afresh instead of being merged into it,
    while a lap whose route crosses the start line midway (a figure-eight)
    is still found whole, because its second half alone does not match.
    """
    laps: list[Lap] = []
    passes = _pass_samples(position_bins, env, anchor, zone)
    if not passes:
        return laps
    # A pass after a stretch that never leaves one ring around the start
    # zone (jitter across its edge, a pause at the start line) continues
    # the previous pass.
    near_zone = zone | (adjacency @ zone > 0)
    boundaries = [passes[0]]
    for candidate in passes[1:]:
        if not near_zone[position_bins[boundaries[-1] : candidate]].all():
            boundaries.append(candidate)
    n_template = np.count_nonzero(template)
    starts = [boundaries[0]]  # candidate lap starts since the last lap
    pieces: list[NDArray[np.bool_]] = []  # widened bins between them
    reach = np.zeros_like(template)  # widened bins since the last lap
    for candidate in boundaries[1:]:
        pieces.append(_widen(position_bins[starts[-1] : candidate], adjacency))
        starts.append(candidate)
        reach |= pieces[-1]
        # Jaccard <= |template & bins| / |template|, and the bins of a stretch
        # from a later start are a subset of reach: no start can match.
        if np.count_nonzero(template & reach) < min_overlap * n_template:
            continue
        visited = np.zeros_like(template)
        for k in range(len(pieces) - 1, -1, -1):  # latest start first
            visited |= pieces[k]
            overlap = np.count_nonzero(template & visited) / np.count_nonzero(
                template | visited
            )
            if overlap >= min_overlap:
                break
        else:
            continue
        lap_start = starts[k]
        lap_direction = _detect_lap_direction(
            env.bin_centers, position_bins[lap_start:candidate]
        )
        if direction == "both" or lap_direction == direction:
            laps.append(
                Lap(
                    start_time=float(times[lap_start]),
                    stop_time=float(times[candidate]),
                    direction=lap_direction,
                    overlap_score=float(overlap),
                )
            )
        starts, pieces = [candidate], []
        reach = np.zeros_like(template)
    return laps


# =============================================================================
# Trajectory similarity and goal-directed runs
# =============================================================================


def trajectory_similarity(
    position_bins1: NDArray[np.int64],
    position_bins2: NDArray[np.int64],
    env: Environment,
    *,
    method: Literal["jaccard", "correlation", "hausdorff", "dtw"] = "jaccard",
    metric: DistanceMetric | None = None,
) -> float:
    """Compute similarity between two trajectories.

    Compares spatial paths using various metrics. Useful for identifying
    stereotyped behaviors and comparing navigation strategies across trials
    or sessions.

    Parameters
    ----------
    position_bins1 : NDArray[np.int64], shape (n_samples1,)
        First trajectory as sequence of bin indices. ``-1`` marks a sample
        outside the environment or missing; such samples are left out.
    position_bins2 : NDArray[np.int64], shape (n_samples2,)
        Second trajectory as sequence of bin indices, with the same ``-1``
        convention.
    env : Environment
        Spatial environment containing bin information.
    method : {'jaccard', 'correlation', 'hausdorff', 'dtw'}, optional
        Similarity metric to use:
        - 'jaccard': Area (or length, or volume) of the visited bins that
          both trajectories share, over the area either visits. Range [0, 1].
        - 'correlation': Pearson correlation of the positions along the
          track, with negative correlations reported as 0. Needs a 1-D
          environment or a linearized track. Range [0, 1], or NaN when
          undefined (see Returns).
        - 'hausdorff': 1 - Hausdorff distance / environment diameter.
          Range [0, 1].
        - 'dtw': 1 - dynamic time warping distance / its largest possible
          value in this environment. Range [0, 1].
        Default is 'jaccard'.
    metric : {'geodesic', 'euclidean'} or None, default=None
        Distance between two places for ``'hausdorff'`` and ``'dtw'``:
        ``'geodesic'`` (what None means) is the shortest path through the
        environment's connectivity graph, so it goes around walls and along
        a maze's arms; ``'euclidean'`` is the straight line between bin
        centers. Runs up two neighboring arms of a maze are close in a
        straight line but far apart along the maze. ``'jaccard'`` and
        ``'correlation'`` compare bins, not distances, and refuse a metric.

    Returns
    -------
    float
        Similarity score in [0, 1] where 1.0 indicates identical trajectories
        and 0.0 indicates completely different trajectories. For
        ``method='correlation'``, NaN when the correlation is undefined:
        fewer than 3 aligned pairs have both samples inside the environment
        (two points always correlate exactly +-1), or either trajectory stays
        in one bin over those pairs. NaN is not 0 ("unrelated"); skip such
        pairs with ``np.nanmean`` or ``np.isfinite`` when averaging.

    Raises
    ------
    ValueError
        If a trajectory is empty or has no sample inside the environment.
    ValueError
        If a bin index is not ``-1`` or an active bin of ``env``.
    ValueError
        If method is not one of the supported options, or ``metric`` is
        given with ``'jaccard'`` or ``'correlation'``.
    ValueError
        If ``method`` is ``"correlation"`` and ``env`` is neither 1-D nor a
        linearized track.
    CartesianOnlyError
        If ``method`` is ``"hausdorff"`` or ``"dtw"`` and ``env`` is polar:
        both measure Cartesian distances between bin centers.

    See Also
    --------
    detect_goal_directed_runs : Detect efficient paths toward goal regions.

    Notes
    -----
    **Jaccard similarity** measures spatial overlap:

    .. math::

        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    where A and B are the regions covered by the visited bins and
    :math:`|\\cdot|` is their area (``env.bin_sizes``: length in 1-D,
    volume in 3-D), so the score does not change when a bin is split into
    pieces. Useful for comparing spatial coverage independent of temporal
    order.

    **Correlation** measures sequential similarity by computing the Pearson
    correlation between the two sequences of positions along the track: the
    linear position of each visited bin on a linearized track, or its
    coordinate in a 1-D environment. Sensitive to order and timing. A
    negative correlation (for example a run in the opposite direction) is
    reported as 0, so the score cannot tell anti-correlated from unrelated
    paths. A 2-D or 3-D path has no single coordinate to correlate; use
    ``'dtw'`` or ``'hausdorff'`` there.

    **Hausdorff distance** measures maximum deviation between trajectories:

    .. math::

        d_H(A, B) = \\max\\{\\sup_{a \\in A} \\inf_{b \\in B} d(a, b),
                            \\sup_{b \\in B} \\inf_{a \\in A} d(a, b)\\}

    Converted to similarity as :math:`1 - d_H / d_{max}` where :math:`d_{max}`
    is the environment's diameter in the chosen metric: the longest shortest
    path between two connected bins for ``'geodesic'`` (computed once per
    environment), or the diagonal of the bin centers' bounding box for
    ``'euclidean'``.

    **Dynamic time warping (DTW)** finds optimal alignment between trajectories
    allowing temporal shifts. Useful for comparing paths with different speeds.
    The summed cost of the optimal alignment is divided by
    :math:`\\max(n_1, n_2)\\, d_{max}`: a diagonal alignment path has
    :math:`\\max(n_1, n_2)` steps and no step costs more than :math:`d_{max}`,
    so the optimum can never exceed it.

    Both Hausdorff and DTW scores are relative to the environment's size:
    the same two paths, 5 cm apart, score higher in a 10 m arena than on a 1
    m track. Compare these scores only between paths in one environment.
    Geodesic distances run along the bin graph, so on an open grid they can
    exceed the straight line by up to about 8% (8-connected neighbors).
    Two routes in parts of the environment not connected to each other have
    no path between them and score 0.

    Samples with bin ``-1`` have no place in the environment. Jaccard,
    Hausdorff and DTW use only each trajectory's samples inside the
    environment; correlation drops the aligned sample pairs in which either
    sample is ``-1``.

    Examples
    --------
    Compare two trajectories using Jaccard overlap:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.segmentation import trajectory_similarity
    >>> positions = np.random.uniform(0, 100, (200, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> traj1 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    >>> traj2 = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    >>> similarity = trajectory_similarity(traj1, traj2, env, method="jaccard")
    >>> print(f"Jaccard similarity: {similarity:.3f}")  # doctest: +SKIP
    Jaccard similarity: 0.429

    Compare two runs along a linear track using correlation (sensitive to
    sequence order):

    >>> track = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]],
    ...     bin_size=5.0,
    ...     track_width=10.0,
    ...     units="cm",
    ... )
    >>> outbound = track.bin_at(np.c_[np.linspace(0, 99, 20), np.zeros(20)])
    >>> inbound = outbound[::-1]
    >>> same = trajectory_similarity(outbound, outbound, track, method="correlation")
    >>> reverse = trajectory_similarity(outbound, inbound, track, method="correlation")
    >>> print(f"same run: {same:.2f}, reverse run: {reverse:.2f}")
    same run: 1.00, reverse run: 0.00
    """
    # Input validation
    if len(position_bins1) == 0 or len(position_bins2) == 0:
        msg = "Trajectories cannot be empty"
        raise ValueError(msg)

    validate_choice(
        method,
        ("jaccard", "correlation", "hausdorff", "dtw"),
        name="method",
        owner="trajectory_similarity()",
    )
    if metric is not None:
        if method in ("jaccard", "correlation"):
            raise ValueError(
                _format_error(
                    f"metric={metric!r} has no effect with method={method!r}.",
                    why=f"method={method!r} compares bins, not distances between them.",
                    fix="remove metric, or use method='hausdorff' or method='dtw'.",
                )
            )
        validate_choice(
            metric,
            ("geodesic", "euclidean"),
            name="metric",
            owner="trajectory_similarity()",
        )
    if method == "correlation":
        from neurospatial.environment import _require_track_extents

        starts, stops = _require_track_extents(
            env,
            "trajectory_similarity(method='correlation')",
            alternative="for a 2-D or 3-D path use method='dtw' or "
            "method='hausdorff', which compare positions in space",
        )
        track_position = (starts + stops) / 2
    if method in ("hausdorff", "dtw"):
        require_cartesian(
            env.coordinate_system,
            f"trajectory_similarity(method={method!r})",
            why=(
                "Hausdorff and DTW distances measure straight lines between bin "
                "centers."
            ),
            fix="use method='jaccard' or method='correlation', which compare bins",
        )

    from neurospatial.environment import _validate_target_bins

    validated = []
    for bins, argument in (
        (position_bins1, "position_bins1"),
        (position_bins2, "position_bins2"),
    ):
        bins = _validate_target_bins(
            bins,
            n_bins=env.n_bins,
            coordinates=env.coordinate_system,
            operation="trajectory_similarity()",
            argument=argument,
            allow_missing=True,
        )
        if not (bins >= 0).any():
            raise ValueError(
                _format_error(
                    f"{argument} has no sample inside the environment.",
                    why="Every bin is -1, so the trajectory has no place to compare.",
                    fix="pass bins from env.bin_at(positions) for positions "
                    "inside the environment",
                )
            )
        validated.append(bins)
    position_bins1, position_bins2 = validated

    if method != "correlation":
        # Set and path comparisons use each trajectory's places only.
        position_bins1 = position_bins1[position_bins1 >= 0]
        position_bins2 = position_bins2[position_bins2 >= 0]

    if method == "jaccard":
        # Spatial overlap: the area of the shared bins over the area of all
        # visited bins (both sets hold at least one bin, checked above).
        sizes = env.bin_sizes
        shared = np.intersect1d(position_bins1, position_bins2)
        either = np.union1d(position_bins1, position_bins2)
        return float(sizes[shared].sum() / sizes[either].sum())

    if method == "correlation":
        # Sequential correlation
        # Align to same length using indices modulo shorter length
        if len(position_bins1) < len(position_bins2):
            shorter, longer = position_bins1, position_bins2
        else:
            shorter, longer = position_bins2, position_bins1

        # Sample longer trajectory at same rate as shorter
        indices = np.linspace(0, len(longer) - 1, len(shorter), dtype=int)
        longer_sampled = longer[indices]
        # Compare only aligned pairs in which both samples have a place.
        observed = (shorter >= 0) & (longer_sampled >= 0)
        # Correlate positions along the track, not bin numbers.
        first = track_position[shorter[observed]]
        second = track_position[longer_sampled[observed]]
        # Fewer than 3 aligned pairs (two points always correlate +-1, fewer
        # not at all) or a sequence that stays in one bin has no defined
        # correlation, which is not "no similarity".
        if first.size < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
            return np.nan

        corr, _ = pearsonr(first, second)
        # Map [-1, 1] to [0, 1], clamp negative correlations to 0
        return float(max(0.0, corr))

    # Hausdorff and DTW: distances between the two routes' places.
    first, row = np.unique(position_bins1, return_inverse=True)
    second, column = np.unique(position_bins2, return_inverse=True)
    if metric == "euclidean":
        centers = env.bin_centers
        table = cdist(centers[first], centers[second])
        diameter = float(np.linalg.norm(np.ptp(centers, axis=0)))
    else:
        from neurospatial.ops.distance import _geodesic_pair_distances

        table = _geodesic_pair_distances(
            env.connectivity,
            np.repeat(first, len(second)),
            np.tile(second, len(first)),
            operation="trajectory_similarity()",
        ).reshape(len(first), len(second))
        diameter = env._geodesic_diameter()
    if method == "hausdorff":
        distance = max(table.min(axis=1).max(), table.min(axis=0).max())
        largest = diameter
    else:
        distance = _dtw_distance(table[row][:, column])
        # The diagonal alignment path has max(len1, len2) steps, each at most
        # the diameter, so the optimal cost can never exceed it.
        largest = max(len(position_bins1), len(position_bins2)) * diameter
    if largest == 0:
        return 1.0  # Degenerate environment
    return float(1.0 - min(distance / largest, 1.0))


def _dtw_distance(cost: NDArray[np.float64]) -> float:
    """Compute Dynamic Time Warping distance between two sequences.

    The cumulative cost ``D[i, j] = cost(i, j) + min(D[i-1, j], D[i, j-1],
    D[i-1, j-1])`` is filled one anti-diagonal ``i + j = k`` at a time: every
    cell on diagonal ``k`` depends only on diagonals ``k - 1`` and ``k - 2``,
    so each diagonal is one vectorized step and only three diagonals are kept.

    Parameters
    ----------
    cost : NDArray[np.float64], shape (n1, n2)
        Distance between each place of the first sequence and each place of
        the second; non-empty, ``inf`` where no path joins them.

    Returns
    -------
    float
        DTW distance: the smallest summed cost over monotone alignments of
        the two sequences.
    """
    n1, n2 = cost.shape
    # Diagonals indexed by row i of D (shape (n1 + 1, n2 + 1)); D[0, 0] = 0 and
    # the rest of row 0 and column 0 is inf.
    before_previous = np.full(n1 + 1, np.inf)
    before_previous[0] = 0.0
    previous = np.full(n1 + 1, np.inf)
    for k in range(2, n1 + n2 + 1):
        rows = np.arange(max(1, k - n2), min(n1, k - 1) + 1)
        current = np.full(n1 + 1, np.inf)
        current[rows] = cost[rows - 1, k - rows - 1] + np.minimum(
            np.minimum(previous[rows - 1], previous[rows]), before_previous[rows - 1]
        )
        before_previous, previous = previous, current
    return float(previous[n1])


@reads_pynapple_samples
def detect_goal_directed_runs(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    goal_region: str,
    directedness_threshold: float = 0.7,
    min_progress: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> Intervals[Interval]:
    """Detect goal-directed runs: approaches to a goal that are efficient.

    Splits the trajectory into approaches (stretches where the distance to
    the goal falls, from the departure to the closest point reached) and
    keeps those that move efficiently toward the goal region. Useful for
    analyzing goal-directed replay, learning dynamics, and navigation
    strategies.

    Parameters
    ----------
    env : Environment
        Environment containing the goal region definition.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where tracking was
        lost. Each is located in its bin inside the function.
    goal_region : str
        Name of goal region in env.regions. Runs are detected toward this region.
    directedness_threshold : float, optional
        Minimum directedness score of an approach to classify it as
        goal-directed. Range [0, 1]. Higher values require more efficient
        paths. Default is 0.7.
    min_progress : float or None, optional
        Minimum net progress of an approach toward the goal, in the
        environment's units (path distance at the departure minus at the
        arrival). Filters out short approaches. None (default) means 20 cm, about one rat
        body length, expressed in ``env.units``; pass a value when the units
        are not a metric length.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive samples (a dropped frame; a NaN row is not
        bridged here, see Notes). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
        For counting, the trade-off differs: a loss longer than ``max_gap``
        in the middle of a run splits it in two, so recordings with
        multi-second losses during runs count runs better with a larger
        value (``scripts/calibrate_max_gap.py``).
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    Intervals of Interval
        Goal-directed approaches in seconds, in time order, ids ``0`` to
        ``n - 1``, with the field ``goal_region``: ``start`` is the
        departure sample and ``stop`` the arrival sample, so a row covers
        the travel between them (interval ``k`` is ``[times[k], times[k +
        1])``, as in :func:`segment_by_velocity`). A row means the
        directedness and net-progress criteria were met; the arrival need
        not be in the goal region. Pass them as ``epochs=`` to restrict
        another analysis to them; ``.windows`` is the ``(n_runs, 2)`` array.
        ``analysis_settings`` records the criteria, ``max_gap`` and
        ``epochs``.

    Raises
    ------
    ValueError
        If goal_region not in env.regions, or if directedness_threshold
        not in [0, 1].
    ValueError
        If min_progress is negative or not finite, or is None while
        ``env.units`` is not a metric length.
    ValueError
        If ``times`` are not finite and strictly increasing, or
        ``positions`` do not have one row per time with the environment's
        number of coordinates.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    trajectory_similarity : Compare similarity between trajectories.
    detect_trials : Trials between visits to named regions.

    Notes
    -----
    A NaN position (lost tracking) or one outside the environment is not
    located (bin -1), so an interval with either sample unlocated is not
    observed and splits the recording, like a gap longer than ``max_gap``,
    an ``epochs`` edge or a ``break_before``; each resulting stretch is
    analyzed on its own, so no approach spans a pause. A NaN row is not
    bridged here (``env.occupancy`` does bridge it): to bridge missing
    samples, delete their rows, and a deleted row is bridged when the
    samples around it are no more than ``max_gap`` apart. A sample in a part of the
    environment that is not connected to the goal has no distance to it and
    splits the recording the same way: no approach includes it, and the
    approaches before and after it are scored on their own.

    **Approaches.** Within each recording, the distance from each sample's
    bin to the goal is followed through time. An approach starts at the
    departure, the last sample at a peak of that distance (so rest before
    leaving is excluded), and ends at the arrival, the first sample at the
    following trough (so rest at the goal is excluded). Rises of up to one
    edge length between neighboring bins do not end an approach: moving to
    a neighboring bin changes the distance by at most that edge (triangle
    inequality), so such a rise can be tracking jitter across a bin edge.
    Pauses on the way stay inside the approach. A larger step back ends it,
    and the next approach starts at the farthest point of the retreat. This
    assumes tracking jitter smaller than a bin; smooth noisier positions
    first, or one approach may be split in two. A walk away from the goal
    is never an approach. To segment completed trips between regions, use
    ``detect_trials``.

    **Directedness score** measures each approach's efficiency:

    .. math::

        D = \\frac{d_{start} - d_{end}}{L_{path}}

    where :math:`d_{start}` and :math:`d_{end}` are distances to the goal at
    the departure and the arrival, and :math:`L_{path}` is the path length
    between them.

    A straight path toward the goal has :math:`D = 1`; a detour lowers it.
    Distances are path distances between bin centers on the environment's
    connectivity graph.

    Examples
    --------
    A straight approach to a goal well at the end of a 100 cm corridor:

    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.regions import Regions
    >>> from neurospatial.behavior.segmentation import detect_goal_directed_runs
    >>> track = np.c_[np.linspace(0, 100, 401), np.zeros(401)]
    >>> env = Environment.from_samples(track, bin_size=2.0).with_metadata(units="cm")
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("goal", polygon=box(95, -5, 105, 5))
    >>> env = env.with_regions(regions)
    >>> x = np.linspace(10, 90, 41)  # 80 cm toward the goal in 4 s
    >>> positions = np.c_[x, np.zeros_like(x)]
    >>> times = np.arange(41) / 10.0
    >>> runs = detect_goal_directed_runs(env, times, positions, goal_region="goal")
    >>> runs.windows.tolist(), runs.goal_region.tolist()
    ([[0.0, 4.0]], ['goal'])

    Out to the goal, back, and out again: each approach is its own row, and
    the walk back is not goal-directed:

    >>> trips = np.r_[x, x[::-1][1:], x[1:]]
    >>> detect_goal_directed_runs(
    ...     env,
    ...     np.arange(trips.size) / 10.0,
    ...     np.c_[trips, np.zeros_like(trips)],
    ...     goal_region="goal",
    ... ).windows.tolist()
    [[0.0, 4.0], [8.0, 12.0]]

    Walking the same path away from the goal is not goal-directed:

    >>> len(detect_goal_directed_runs(env, times, positions[::-1], goal_region="goal"))
    0

    """
    # Input validation
    _require_region_mask(
        env,
        goal_region,
        operation="detect_goal_directed_runs()",
        argument="goal_region",
    )

    if not 0.0 <= directedness_threshold <= 1.0:
        msg = f"directedness_threshold must be in [0, 1], got {directedness_threshold}"
        raise ValueError(msg)

    if min_progress is None:
        from neurospatial._units import default_length

        min_progress = default_length(
            env,
            DEFAULT_MIN_PROGRESS_CM,
            name="min_progress",
            call="detect_goal_directed_runs()",
        )
    validate_finite_scalar(min_progress, name="min_progress", positive=False)

    times, position_bins = sample_bins(
        env, times, positions, call="detect_goal_directed_runs"
    )
    breaks = validate_break_before(
        break_before, len(times), call="detect_goal_directed_runs"
    )

    from neurospatial.behavior._environment_paths import environment_paths
    from neurospatial.ops.distance import _bin_adjacency
    from neurospatial.ops.mapping import regions_to_mask

    # Path distances between bin centers on the connectivity graph; inf for
    # bins that cannot reach the goal.
    paths = environment_paths(env, call="detect_goal_directed_runs")
    distance_to_goal = paths.from_bins(
        np.flatnonzero(regions_to_mask(env, [goal_region]))
    )
    edges = _bin_adjacency(
        env.connectivity, weight="distance", operation="detect_goal_directed_runs()"
    )
    # Complete-case rule, as in env.occupancy: an interval is observed only
    # when both of its samples are in the environment.
    gates = start_allocation(
        times,
        position_bins,
        None,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=breaks,
    ).gates
    warn_excluded_intervals(gates)
    results: list[tuple[float, float]] = []
    for first, last in run_sample_bounds(gates.mask):
        run = slice(first, last + 1)
        results.extend(
            _detect_goal_directed_runs_contiguous(
                position_bins[run],
                times[run],
                paths=paths,
                edges=edges,
                distance_to_goal=distance_to_goal,
                directedness_threshold=directedness_threshold,
                min_progress=min_progress,
            )
        )
    windows = as_intervals(epochs, name="epochs")
    return _windows_as_intervals(
        np.array(results, dtype=np.float64).reshape(-1, 2),
        fields={"goal_region": goal_region},
        settings={
            "goal_region": goal_region,
            "directedness_threshold": float(directedness_threshold),
            "min_progress": float(min_progress),
            "max_gap": None if max_gap is None else float(max_gap),
            "epochs": None if windows is None else windows.tolist(),
        },
    )


def _approach_bouts(
    distances: NDArray[np.float64], tolerance: float
) -> list[tuple[int, int]]:
    """Approaches in a distance-to-goal series, as (departure, arrival) indices.

    An approach starts at the last sample at a local maximum of the distance
    and ends at the first sample at the following local minimum. Changes of
    no more than ``tolerance`` against the current direction (jitter, pauses)
    do not end an approach or a retreat.
    """
    bouts: list[tuple[int, int]] = []
    peak, trough, approaching = 0, 0, False
    for k in range(1, len(distances)):
        d = distances[k]
        if approaching:
            if d < distances[trough]:
                trough = k  # first sample at the new minimum
            elif d > distances[trough] + tolerance:
                bouts.append((peak, trough))
                peak, approaching = k, False
        elif d >= distances[peak]:
            peak = k  # last sample at the maximum: the departure
        elif d < distances[peak] - tolerance:
            trough, approaching = k, True
    if approaching:
        bouts.append((peak, trough))
    return bouts


def _detect_goal_directed_runs_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    *,
    paths: EnvironmentPaths,
    edges: Any,
    distance_to_goal: NDArray[np.float64],
    directedness_threshold: float,
    min_progress: float,
) -> list[tuple[float, float]]:
    """Score each approach to the goal within one recording."""
    distances = distance_to_goal[position_bins]
    # A move between neighboring bins changes the distance by at most the
    # edge between them (triangle inequality), so a change of one edge can
    # be jitter across a bin edge; anything larger is movement. The factor
    # allows for rounding in the summed edge lengths.
    tolerance = float(edges.max()) * (1 + 1e-9) if edges.nnz else 0.0
    rows: list[tuple[float, float]] = []
    # Samples that cannot reach the goal split the recording.
    finite = np.r_[False, np.isfinite(distances), False]
    starts = np.flatnonzero(~finite[:-1] & finite[1:])
    stops = np.flatnonzero(finite[:-1] & ~finite[1:])
    for first, stop in zip(starts, stops, strict=True):
        # Path length accumulates within the stretch only: both bins of each
        # step reach the goal, so they lie in its connected component and the
        # step is finite. A step to an unreachable bin would add inf.
        stretch = position_bins[first:stop]
        path = np.r_[0.0, np.cumsum(paths.between_bins(stretch[:-1], stretch[1:]))]
        for departure, arrival in _approach_bouts(distances[first:stop], tolerance):
            progress = distances[first + departure] - distances[first + arrival]
            directedness = progress / (path[arrival] - path[departure])
            if directedness >= directedness_threshold and progress >= min_progress:
                rows.append(
                    (float(times[first + departure]), float(times[first + arrival]))
                )
    return rows


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Dataclasses
    "Crossing",
    "Lap",
    "Trial",
    # Functions
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "segment_by_velocity",
    "trajectory_similarity",
]
