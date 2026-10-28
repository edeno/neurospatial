"""Behavioral segmentation and trajectory analysis.

This module provides functions for segmenting trajectories based on regions,
velocity, and behavioral epochs.

Imports
-------
All segmentation functions are accessible from this module:

>>> from neurospatial.behavior.segmentation import (
...     Crossing,
...     Lap,
...     Run,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
...     detect_runs_between_regions,
...     detect_goal_directed_runs,
...     segment_by_velocity,
...     segment_trials,
...     trajectory_similarity,
... )

Or import from the behavior package:

>>> from neurospatial.behavior import (
...     Crossing,
...     Lap,
...     Run,
...     Trial,
...     detect_laps,
...     detect_region_crossings,
...     segment_trials,
... )

Functions
---------
detect_region_crossings
    Detect entry and exit events for a spatial region
detect_runs_between_regions
    Detect runs from source region to target region
segment_by_velocity
    Movement epochs from a speed threshold, as an (n, 2) array
detect_laps
    Detect laps on circular tracks
segment_trials
    Segment trajectory into behavioral trials (T-maze, Y-maze, etc.)
trajectory_similarity
    Compare similarity between two trajectories
detect_goal_directed_runs
    Efficient approaches to a goal, as an (n, 2) array
laps_to_direction_labels
    Per-sample direction labels from detected laps
runs_to_direction_labels
    Per-sample labels from detected runs
running_direction_labels
    Per-sample inbound/outbound labels on a linear or W-track

Classes
-------
Trial
    One trial: start_time, end_time, start_region, end_region (None unless
    reached) and outcome ('reached', 'timeout', 'returned' or 'censored')
Crossing
    One region entry or exit
Lap
    One lap on a circular track
Run
    One attempt to travel from a source region to a target region
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.stats import pearsonr

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    observed_runs,
    times_in_intervals,
)
from neurospatial._observation import validate_break_before
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial._validation import validate_finite_scalar
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from neurospatial.behavior._environment_paths import EnvironmentPaths
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem


def _safe_gather(
    values: NDArray,
    position_bins: NDArray[np.int64],
    *,
    fill: object,
) -> NDArray:
    """Gather ``values[position_bins]`` with out-of-bounds bins mapped to ``fill``.

    ``position_bins`` may contain ``-1`` (sample off the environment / unmapped)
    or indices ``>= len(values)``. Plain fancy indexing would wrap ``-1`` to the
    last element and silently mis-attribute an out-of-environment sample to a
    real bin. This helper instead returns ``fill`` for every out-of-bounds
    position, so callers can treat those samples as "outside" rather than
    accidentally "inside the last region".

    Parameters
    ----------
    values : NDArray, shape (n_bins,) or (n_bins, ...)
        Per-bin array to gather from (e.g. a boolean region mask or an integer
        Voronoi-label array).
    position_bins : NDArray[np.int64], shape (n_samples,)
        Bin index per trajectory sample. May contain ``-1`` or out-of-range
        indices.
    fill : object
        Value substituted wherever ``position_bins`` is out of bounds. Use
        ``False`` for boolean region masks and ``-1`` for integer label arrays
        (matching the ``-1`` "unreachable" sentinel those arrays already use).

    Returns
    -------
    NDArray
        Array of length ``len(position_bins)`` (leading axis) with out-of-bounds
        rows set to ``fill``.
    """
    position_bins = np.asarray(position_bins, dtype=np.int64)
    n_bins = values.shape[0]
    in_bounds = (position_bins >= 0) & (position_bins < n_bins)
    safe_idx = np.where(in_bounds, position_bins, 0)
    gathered: NDArray = values[safe_idx]
    gathered[~in_bounds] = fill
    return gathered


def _validated_samples(
    position_bins: ArrayLike, times: ArrayLike, env: Environment, *, call: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Check a bin trajectory and its timestamps; return them as int64 and float64.

    ``position_bins`` must be integer bin indices of ``env`` or ``-1`` (outside
    the environment, as :meth:`Environment.bin_at` returns); a float array
    would be truncated to unrelated bins. ``times`` must be finite and
    strictly increasing, in seconds.
    """
    from neurospatial._observation import _validated_signal_times
    from neurospatial.behavior._position_bins import validated_position_bins

    bins = validated_position_bins(position_bins, env, call=call, times=times)
    if bins.size == 0:
        return bins, np.asarray(times, dtype=np.float64)
    return bins, _validated_signal_times(times, call=call)


def _validate_durations(min_duration: float, max_duration: float, *, call: str) -> None:
    """Require finite ``0 <= min_duration < max_duration`` (seconds)."""
    validate_finite_scalar(min_duration, name="min_duration", positive=False)
    validate_finite_scalar(max_duration, name="max_duration", positive=True)
    if max_duration <= min_duration:
        raise ValueError(
            _format_error(
                f"{call}() needs max_duration > min_duration; got "
                f"max_duration={max_duration}, min_duration={min_duration}.",
                why="no attempt could be both long enough and short enough, so "
                "every one would be dropped.",
                fix="pass a max_duration longer than min_duration (seconds).",
            )
        )


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


def _located(position_bins: NDArray[np.int64], env: Environment) -> NDArray[np.bool_]:
    """True for samples in a bin of ``env``; any other sample is missing.

    Missing samples are skipped as if their rows were deleted, and the gap
    around them is bridged when it is no longer than ``max_gap``.
    """
    bins = np.asarray(position_bins)
    located: NDArray[np.bool_] = (bins >= 0) & (bins < env.n_bins)
    return located


def _event_runs(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None,
    sample_valid: NDArray[np.bool_],
) -> list[NDArray[np.intp]]:
    """Observed runs holding only samples inside the half-open ``epochs``.

    :func:`observed_runs` keeps the sample at an epoch's ``stop``, because the
    interval that ends there lies inside the epoch. That instant is outside
    ``[start, stop)``, so an event detector must not report an entry, exit
    or trial end there: such a run ends at its last sample before ``stop``,
    as :func:`detect_region_visits` does.
    """
    runs = observed_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        sample_valid=sample_valid,
    )
    windows = as_intervals(epochs, name="epochs")
    if windows is None:
        return runs
    inside = times_in_intervals(windows, times)
    trimmed = [run[inside[run]] for run in runs]
    return [run for run in trimmed if run.size]


def _interval_labels(
    times: NDArray[np.float64],
    intervals: Iterable[tuple[float, float, object]],
) -> NDArray[np.object_]:
    """Build a per-timepoint object label array from labeled time intervals.

    Allocates an object array of length ``len(times)`` filled with ``None``
    and, for each ``(start_time, end_time, label)`` interval, assigns ``label``
    to every sample with ``start_time <= times < end_time``. This is the shared
    primitive behind the trial/lap/run direction-label helpers (so the
    half-open, last-writer-wins semantics live in one place). A sample's label
    describes the interval that starts at it, so the last sample of a lap or
    run, whose interval leaves it, is not given its label. Later intervals
    overwrite earlier ones on any overlap.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for the whole session (seconds).
    intervals : iterable of (float, float, object)
        ``(start_time, end_time, label)`` tuples, half-open
        ``[start_time, end_time)``.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Per-timepoint labels, dtype ``object``; ``None`` outside every
        interval.
    """
    times = np.asarray(times, dtype=np.float64)
    labels: NDArray[np.object_] = np.full(len(times), None, dtype=object)

    for start_time, end_time, label in intervals:
        mask = (times >= start_time) & (times < end_time)
        labels[mask] = label

    return labels


# =============================================================================
# Event records
# =============================================================================

Outcome = Literal["reached", "timeout", "returned", "censored"]
"""How a trial or a run between regions ended.

``"reached"``: the end (target) region was entered within ``max_duration``.
``"timeout"``: ``max_duration`` elapsed first. ``"returned"``: the animal
re-entered the start (source) region first. ``"censored"``: observation stopped
first, at a tracking gap longer than ``max_gap``, an ``epochs`` edge, a declared
``break_before`` or the end of the recording, so the attempt's result is
unknown.
"""

_OUTCOMES: tuple[str, ...] = ("reached", "timeout", "returned", "censored")


def _checked_interval(record: str, start: object, end: object) -> tuple[float, float]:
    """Return ``(start, end)`` as finite Python floats with ``start <= end``."""
    start_f, end_f = float(start), float(end)  # type: ignore[arg-type]
    if not (np.isfinite(start_f) and np.isfinite(end_f)) or start_f > end_f:
        raise ValueError(
            _format_error(
                f"{record} needs finite start_time <= end_time; got "
                f"start_time={start_f!r}, end_time={end_f!r}.",
                why="an event interval runs forward in time on one clock.",
                fix="pass the interval's start and end times in seconds, start first.",
            )
        )
    return start_f, end_f


def _check_choice(
    record: str, name: str, value: object, choices: Sequence[str]
) -> None:
    """Raise a ValueError naming ``choices`` when ``value`` is not one of them."""
    if value not in choices:
        raise ValueError(
            _format_error(
                f"{record}.{name} must be one of {list(choices)}; got {value!r}.",
                why=f"each {name} has one documented meaning.",
                fix=f"pass one of {list(choices)}.",
            )
        )


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
        _check_choice("Crossing", "direction", self.direction, ("entry", "exit"))
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError(f"Crossing.time must be finite; got {time!r}.")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "bin_index", int(self.bin_index))


@dataclass(frozen=True, eq=False)
class Run(ReadOnlyArraysMixin):
    """One attempt to travel from a source region to a target region.

    A run starts at the first sample outside ``source_region`` after the
    animal leaves it and ends at the first of: entry to ``target_region``,
    re-entry to ``source_region``, the deadline ``max_duration``, or the end
    of the observed stretch of recording. Runs from one detection never
    overlap.

    Attributes
    ----------
    start_time : float
        Time of the first sample outside the source region (seconds).
    end_time : float
        Time at which the run ended (seconds); see ``outcome``.
    source_region : str
        Name of the region the run left.
    target_region : str
        Name of the region the run was heading for.
    outcome : {'reached', 'timeout', 'returned', 'censored'}
        How the run ended:

        - ``'reached'``: entered ``target_region`` within ``max_duration``;
          ``end_time`` is the first sample inside it.
        - ``'timeout'``: ``max_duration`` elapsed first; ``end_time`` is the
          last sample within it.
        - ``'returned'``: re-entered ``source_region`` first (an abandoned
          excursion); ``end_time`` is the first sample back inside it.
        - ``'censored'``: observation stopped first (a tracking gap longer
          than ``max_gap``, an ``epochs`` edge, a ``break_before`` or the end
          of the recording); ``end_time`` is the last observed sample, and
          whether the run would have reached the target is unknown.
    bins : NDArray[np.int64], shape (n_run_samples,)
        Bin index of each observed sample from ``start_time`` to ``end_time``,
        inclusive. Read-only.
    success : bool
        Read-only property, ``outcome == 'reached'``.
    """

    start_time: float
    end_time: float
    source_region: str
    target_region: str
    outcome: Outcome
    bins: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Check the record and own a read-only copy of the bin sequence."""
        start, end = _checked_interval("Run", self.start_time, self.end_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        _check_choice("Run", "outcome", self.outcome, _OUTCOMES)
        bins = np.asarray(self.bins)
        if bins.ndim != 1 or (bins.size and bins.dtype.kind not in "iu"):
            raise ValueError(
                "Run.bins must be a 1-D integer array of bin indices; got "
                f"dtype {bins.dtype} with shape {bins.shape}."
            )
        object.__setattr__(self, "bins", bins.astype(np.int64))
        self._own_arrays()

    @property
    def success(self) -> bool:
        """Whether the run reached its target region (``outcome == 'reached'``)."""
        return self.outcome == "reached"


@dataclass(frozen=True)
class Lap:
    """Detected lap on circular track.

    Attributes
    ----------
    start_time : float
        Start time of lap (seconds).
    end_time : float
        End time of lap (seconds).
    direction : str
        Direction of lap: 'clockwise', 'counter-clockwise', or 'unknown'.
    overlap_score : float
        Jaccard overlap of the lap's bins with the template lap's bins, both
        widened by one ring of neighbor bins, in [0, 1], for the
        ``'auto'`` and ``'reference'`` methods; NaN for the ``'region'``
        method, which uses no template.
    """

    start_time: float
    end_time: float
    direction: Literal["clockwise", "counter-clockwise", "unknown"]
    overlap_score: float

    def __post_init__(self) -> None:
        """Check the record and store plain Python floats."""
        start, end = _checked_interval("Lap", self.start_time, self.end_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        _check_choice(
            "Lap",
            "direction",
            self.direction,
            ("clockwise", "counter-clockwise", "unknown"),
        )
        score = float(self.overlap_score)
        if not (np.isnan(score) or 0.0 <= score <= 1.0):
            raise ValueError(
                f"Lap.overlap_score must be in [0, 1] or NaN; got {score!r}."
            )
        object.__setattr__(self, "overlap_score", score)


@dataclass(frozen=True)
class Trial:
    """One behavioral trial: a departure from the start region and its result.

    Attributes
    ----------
    start_time : float
        Time when trial started: the last sample in the start region before
        the animal left it (seconds).
    end_time : float
        Time when the trial ended (seconds); see ``outcome``.
    start_region : str
        Name of the region where trial started.
    end_region : str or None
        Name of the end region reached; None unless ``outcome == 'reached'``.
    outcome : {'reached', 'timeout', 'returned', 'censored'}
        How the trial ended:

        - ``'reached'``: entered an end region within ``max_duration``;
          ``end_time`` is the first sample inside it.
        - ``'timeout'``: ``max_duration`` elapsed first; ``end_time`` is the
          last sample at or before ``start_time + max_duration``.
        - ``'returned'``: re-entered the start region first (an aborted
          trial); ``end_time`` is the first sample back inside it.
        - ``'censored'``: observation stopped first (a tracking gap longer
          than ``max_gap``, an ``epochs`` edge, a ``break_before`` or the end
          of the recording); ``end_time`` is the last observed sample, and
          the trial's result is unknown. Leave censored trials out of
          success rates rather than counting them as errors.
    success : bool
        Read-only property, ``outcome == 'reached'``.
    """

    start_time: float
    end_time: float
    start_region: str
    end_region: str | None
    outcome: Outcome

    def __post_init__(self) -> None:
        """Check the record and store plain Python floats."""
        start, end = _checked_interval("Trial", self.start_time, self.end_time)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        _check_choice("Trial", "outcome", self.outcome, _OUTCOMES)
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

    @property
    def success(self) -> bool:
        """Whether the trial reached an end region (``outcome == 'reached'``)."""
        return self.outcome == "reached"


def detect_region_crossings(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
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
    position_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to position bins (seconds).
    env : Environment
        Environment containing the named region.
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
        If position_bins and times are not 1-D with the same length, if
        times are not finite and strictly increasing, or if a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices (a float array
        would be truncated to unrelated bins).
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_runs_between_regions : Detect runs from source to target regions
    segment_by_velocity : Segment trajectory by velocity threshold

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

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
    >>> position_bins = env.bin_at(trajectory)
    >>> times = np.arange(len(trajectory), dtype=float)
    >>> # Treat these 1 Hz samples as continuous with max_gap=None
    >>> crossings = detect_region_crossings(
    ...     position_bins,
    ...     times,
    ...     env,
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
    _check_choice(
        "detect_region_crossings()", "direction", direction, ("both", "entry", "exit")
    )

    position_bins, times = _validated_samples(
        position_bins, times, env, call="detect_region_crossings"
    )
    breaks = validate_break_before(
        break_before, len(times), call="detect_region_crossings"
    )
    results: list[Crossing] = []
    for run in _event_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
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
    # Check which trajectory samples are in region
    position_bins = np.asarray(position_bins, dtype=np.int64)
    in_region = _safe_gather(region_mask, position_bins, fill=False)

    # Detect transitions using vectorized diff operation
    # Convert bool to int: in_region[i] - in_region[i-1]
    # +1 = entry (False -> True), -1 = exit (True -> False)
    transitions = np.diff(in_region.astype(np.int8))

    # Find entry and exit indices
    crossings = []

    if direction in ["both", "entry"]:
        entry_indices = np.where(transitions == 1)[0] + 1  # +1 because diff shifts by 1
        for idx in entry_indices:
            crossings.append(
                Crossing(
                    time=times[idx],
                    direction="entry",
                    bin_index=position_bins[idx],
                    region=region_name,
                )
            )

    if direction in ["both", "exit"]:
        exit_indices = np.where(transitions == -1)[0] + 1
        for idx in exit_indices:
            crossings.append(
                Crossing(
                    time=times[idx],
                    direction="exit",
                    bin_index=position_bins[idx],
                    region=region_name,
                )
            )

    # Sort by time if we collected both entry and exit
    if direction == "both" and len(crossings) > 1:
        crossings.sort(key=lambda c: c.time)

    return crossings


def detect_runs_between_regions(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    source: str,
    target: str,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> list[Run]:
    """Detect runs from source region to target region.

    Identifies behavioral epochs where the animal exits the source region,
    travels through the environment, and either reaches the target region,
    times out, returns to the source, or stops being observed. Useful for
    analyzing goal-directed navigation, learning trajectories, and spatial
    strategies.

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Discretized position indices (one per sample). Use
        ``env.bin_at(positions)`` to obtain these from continuous
        coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to ``position_bins`` (seconds).
    env : Environment
        Environment containing source and target region definitions.
    source : str
        Name of source region (start point).
    target : str
        Name of target region (goal).
    min_duration : float, optional
        Minimum run duration in seconds. Default: 0.5.
        Runs shorter than this are excluded.
    max_duration : float, optional
        Maximum run duration in seconds. Default: 10.0.
        Runs exceeding this are marked as failed (timeout).
    min_speed : float or None, optional
        Drop runs whose mean speed is below this value (position units per
        second). A run's mean speed is the mean, over consecutive sample
        pairs, of the distance between their bin centers divided by the
        elapsed time, so it is bin-quantized and does not gate individual
        samples; to restrict runs to movement on continuous positions, pass
        movement epochs from :func:`segment_by_velocity` as ``epochs``.
        If None (default), no speed filtering is applied.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    list[Run]
        Detected runs in time order. Each run records ``start_time`` (first
        sample outside the source), ``end_time``, ``source_region``,
        ``target_region``, ``outcome`` (``'reached'``, ``'timeout'``,
        ``'returned'`` or ``'censored'``, see :class:`Run`), the observed
        ``bins`` and the derived ``success`` (``outcome == 'reached'``).

    Raises
    ------
    ValueError
        If source or target regions not in env.regions.
    ValueError
        If min_duration is negative or not finite, max_duration is not
        greater than min_duration, or min_speed is negative or not finite.
    ValueError
        If position_bins and times are not 1-D with the same length, if
        times are not finite and strictly increasing, or if a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices (a float array
        would be truncated to unrelated bins).
    CartesianOnlyError
        If ``min_speed`` is given and ``env`` is polar: run speed comes from
        Cartesian distances between bin centers.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_region_crossings : Detect individual region crossings
    segment_by_velocity : Segment by velocity threshold
    runs_to_direction_labels : Convert these runs into per-timepoint direction
        labels for ``compute_directional_place_fields``.
    running_direction_labels : First-class inbound/outbound labeler for linear
        tracks built on this detector.

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

    A run in progress when a recording ends is emitted there with
    ``outcome='censored'``: it neither reached the target nor timed out
    while observed.

    A run is defined as:
    1. Exit from source region (the first sample outside it)
    2. Trajectory through environment
    3. Whichever comes first: entry to the target (``'reached'``), re-entry
       to the source (``'returned'``), ``max_duration`` elapsing
       (``'timeout'``, ending at the last sample within it) or the end of
       observation (``'censored'``)

    Only the last exit before an arrival can start a reached run: an
    excursion that comes back to the source is its own ``'returned'`` run,
    so runs never overlap and a walk back home is never part of a run away
    from it.

    Runs shorter than ``min_duration`` are dropped, whatever their outcome;
    no run lasts longer than ``max_duration``. With ``min_speed``, runs
    whose mean bin-center speed is below it are dropped too.

    This function is commonly used in:
    - Spatial alternation tasks (T-maze, Y-maze)
    - Goal-directed navigation experiments
    - Trajectory analysis for learning curves
    - Replay detection (theta sequences, sharp-wave ripples)

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> x = np.linspace(0, 100, 200)
    >>> y = np.full(200, 50.0)
    >>> positions = np.column_stack([x, y])
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("start", polygon=Point(10.0, 50.0).buffer(5.0))
    >>> _ = regions.add("goal", polygon=Point(90.0, 50.0).buffer(5.0))
    >>> env = env.with_regions(regions)
    >>> traj_x = np.linspace(10.0, 90.0, 100)
    >>> traj_y = np.ones(100) * 50.0
    >>> trajectory = np.column_stack([traj_x, traj_y])
    >>> times = np.linspace(0, 5.0, 100)
    >>> position_bins = env.bin_at(trajectory)
    >>> runs = detect_runs_between_regions(
    ...     position_bins,
    ...     times,
    ...     env,
    ...     source="start",
    ...     target="goal",
    ...     min_duration=0.5,
    ...     max_duration=10.0,
    ... )
    >>> len(runs) > 0
    True
    """
    if min_speed is not None:
        _require_run_speed_frame(env, call="detect_runs_between_regions")
    masks = {
        name: _require_region_mask(
            env, name, operation="detect_runs_between_regions()", argument=argument
        )
        for name, argument in ((source, "source"), (target, "target"))
    }
    _validate_durations(min_duration, max_duration, call="detect_runs_between_regions")
    validate_finite_scalar(
        min_speed, name="min_speed", positive=False, none_allowed=True
    )

    position_bins, times = _validated_samples(
        position_bins, times, env, call="detect_runs_between_regions"
    )
    breaks = validate_break_before(
        break_before, len(times), call="detect_runs_between_regions"
    )
    results: list[Run] = []
    for run in _event_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
    ):
        results.extend(
            _detect_runs_between_regions_contiguous(
                position_bins[run],
                times[run],
                env,
                source=source,
                target=target,
                source_mask=masks[source],
                target_mask=masks[target],
                min_duration=min_duration,
                max_duration=max_duration,
                min_speed=min_speed,
            )
        )
    return results


def _require_run_speed_frame(env: Environment, *, call: str) -> None:
    """Refuse ``min_speed`` on a polar environment: run speed is Cartesian."""
    require_cartesian(
        env.coordinate_system,
        f"{call}(min_speed=...)",
        why=(
            "Run speed comes from the Cartesian distance between successive bin "
            "centers."
        ),
        fix="pass min_speed=None to detect runs from region entries alone",
    )


def _detect_runs_between_regions_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    source: str,
    target: str,
    source_mask: NDArray[np.bool_],
    target_mask: NDArray[np.bool_],
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    min_speed: float | None = None,
) -> list[Run]:
    """Analyze one recording after the public input validation.

    ``source_mask`` and ``target_mask`` (shape ``(n_bins,)``) mark the bins
    of ``source`` and ``target``.
    """
    # Check which samples are in each region
    in_source = _safe_gather(source_mask, position_bins, fill=False)
    in_target = _safe_gather(target_mask, position_bins, fill=False)

    # Detect source exits
    source_exits = []
    for i in range(1, len(in_source)):
        if in_source[i - 1] and not in_source[i]:  # Exit source
            source_exits.append(i)

    # For each exit, track trajectory until target entry or timeout
    runs = []

    for exit_idx in source_exits:
        start_time = times[exit_idx]

        # Track trajectory: advance end_idx through samples until the target is
        # reached or the run times out. end_idx always points at the last
        # sample belonging to the run, so the run path is the single contiguous
        # slice position_bins[exit_idx : end_idx + 1]. Both the stored path and
        # the optional velocity filter are derived from this one slice so they
        # can never diverge (e.g. by an off-by-one at the timeout boundary).
        # Only the last exit before the target counts: re-entering the source
        # ends the attempt ("returned"), so runs never overlap. A run that
        # reaches the end of the recording is censored: its result is unknown.
        outcome: Outcome = "censored"
        end_idx = exit_idx

        for j in range(exit_idx, len(position_bins)):
            elapsed = times[j] - start_time

            # Check timeout (do not include the timing-out sample in the run)
            if elapsed > max_duration:
                outcome = "timeout"
                break

            end_idx = j

            if in_target[j]:
                outcome = "reached"
                break
            if in_source[j]:
                outcome = "returned"
                break
            if elapsed >= max_duration:
                # The deadline passed at this sample without an arrival.
                outcome = "timeout"
                break

        run_slice = slice(exit_idx, end_idx + 1)
        run_bin_idx = position_bins[run_slice]
        run_times = times[run_slice]

        end_time = times[end_idx]
        duration = end_time - start_time

        # Filter by duration
        if duration < min_duration:
            continue

        # Optional velocity filter (bin-quantized), using the same slice as the
        # stored path. Off-environment samples (``-1``) or out-of-range bins must
        # not be treated as real positions: plain ``bin_centers[run_bin_idx]``
        # would wrap ``-1`` to the last bin (inflating speed) or raise on indices
        # ``>= n_bins``. Compute speed only over consecutive sample pairs where
        # both endpoints map to a valid bin.
        if min_speed is not None:
            # A displacement is usable only when both consecutive samples are
            # valid (and the underlying dt is positive). Off-environment samples
            # (``-1``) or out-of-range bins are not real positions.
            if len(run_times) > 1:
                valid = (run_bin_idx >= 0) & (run_bin_idx < env.n_bins)
                pair_valid = valid[:-1] & valid[1:]
            else:
                pair_valid = np.zeros(0, dtype=bool)

            if not pair_valid.any():
                # No consecutive pair has both endpoints on the environment, so
                # the run's speed cannot be validated. A run must not silently
                # pass a speed gate it never satisfied: drop it (treat as
                # failing the filter) rather than keeping a spurious
                # (e.g. -1-inflated) velocity.
                continue

            safe_idx = np.where(valid, run_bin_idx, 0)
            run_positions = env.bin_centers[safe_idx]
            displacements = np.diff(run_positions, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            dt = np.diff(run_times)  # > 0: times are strictly increasing
            velocities = (distances / dt)[pair_valid]
            mean_velocity = float(np.mean(velocities))

            if mean_velocity < min_speed:
                continue

        # Create run
        runs.append(
            Run(
                start_time=start_time,
                end_time=end_time,
                source_region=source,
                target_region=target,
                outcome=outcome,
                bins=run_bin_idx,
            )
        )

    return runs


def segment_by_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    min_duration: float = 0.5,
    hysteresis: float = 2.0,
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Find movement epochs: stretches where speed stays above a threshold.

    Uses hysteresis thresholding on the smoothed speed to find movement
    epochs and returns them as ``[start, stop)`` rows; time outside every
    row is rest or unobserved. Epochs shorter than ``min_duration`` are
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
        0 thresholds the unsmoothed central-difference speed.
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
    NDArray[np.float64], shape (n_epochs, 2)
        Movement epochs as ``[start, stop)`` rows in seconds, in time order.
        Pass them as ``epochs=`` to any analysis to restrict it to movement.

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
    detect_runs_between_regions : Detect runs between spatial regions
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

    Speed is the speed of :func:`compute_speed`: the centered-secant velocity
    ``(p[i+1] - p[i-1]) / (t[i+1] - t[i-1])`` (one-sided at the ends of a
    recording), each component smoothed with a Gaussian of standard
    deviation ``smooth_window`` seconds (edges extended with the nearest
    value, ``mode="nearest"``, so movement that reaches the end of a
    recording is not suppressed), then its magnitude.

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
    >>> segments.shape  # one movement epoch, as a [start, stop) row
    (1, 2)
    >>> bool(np.all(segments[:, 1] - segments[:, 0] >= 0.5))  # min_duration
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
        raise ValueError(f"hysteresis must be > 1.0 for stability. Got {hysteresis}")

    segments = [
        _segment_by_velocity_contiguous(
            positions[run],
            times[run],
            min_speed=min_speed,
            min_duration=min_duration,
            hysteresis=hysteresis,
            smooth_window=smooth_window,
        )
        for run in support.run_samples()
    ]
    return np.concatenate([np.empty((0, 2)), *segments])


def _segment_by_velocity_contiguous(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float,
    min_duration: float,
    hysteresis: float,
    smooth_window: float,
) -> NDArray[np.float64]:
    """Movement epochs of one recording, shape (n_epochs, 2).

    ``times`` and ``positions`` are the valid samples of one observed run,
    already validated by :func:`segment_by_velocity`.
    """
    from neurospatial._velocity import smoothed_speed

    # Speed of each transition i -> i+1, read at its end sample: a state
    # change at transition i starts or stops an epoch at times[i + 1]. One
    # run, so every interval is observed.
    speed = smoothed_speed(
        times,
        positions,
        np.ones(len(times) - 1, dtype=bool),
        smooth_window=smooth_window,
    )[1:]

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


def detect_laps(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment[CartesianSystem],
    *,
    method: Literal["auto", "reference", "region"] = "auto",
    min_overlap: float = 0.8,
    direction: Literal["both", "clockwise", "counter-clockwise"] = "both",
    reference_lap: NDArray[np.int64] | None = None,
    start_region: str | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> list[Lap]:
    """Detect laps in a circular track trajectory.

    Identifies repeating traversals of a circular track using spatial overlap
    with a template lap. Useful for analyzing learning, spatial coding, and
    behavioral consistency across laps.

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to position bins (seconds).
    env : Environment
        Environment containing spatial structure.
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
        Bin sequence of one complete lap, for the 'reference' method; its
        first bin marks where laps start. Required when method='reference'.
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
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window, and a sample at a
        window's ``stop`` is outside it, so no event is reported there.
        ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    list[Lap]
        List of detected laps, each with start_time, end_time, direction,
        and overlap_score. Sorted by start_time in ascending order.

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
        If position_bins and times are not 1-D with the same length, if
        times are not finite and strictly increasing, or if a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices (a float array
        would be truncated to unrelated bins).
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_region_crossings : Detect region entry/exit events
    detect_runs_between_regions : Detect runs between regions
    laps_to_direction_labels : Convert these laps into per-timepoint direction
        labels for ``compute_directional_place_fields``.

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

    Laps never span recordings. Samples outside the environment are left
    out as if their rows were deleted.

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
    >>> position_bins = env.bin_at(positions)
    >>> times = np.linspace(0, 60, 1801)
    >>> laps = detect_laps(position_bins, times, env, method="auto")
    >>> len(laps)
    10

    Detect laps with user-provided reference:

    >>> reference = position_bins[:181]
    >>> laps = detect_laps(
    ...     position_bins,
    ...     times,
    ...     env,
    ...     method="reference",
    ...     reference_lap=reference,
    ... )
    >>> all(lap.overlap_score >= 0.8 for lap in laps)
    True

    Filter laps by direction:

    >>> laps_cw = detect_laps(position_bins, times, env, direction="clockwise")
    >>> laps_ccw = detect_laps(
    ...     position_bins,
    ...     times,
    ...     env,
    ...     direction="counter-clockwise",
    ... )
    >>> len(laps_ccw) > 0
    True
    >>> len(laps_cw)
    0

    Feed detected laps into directional place fields via
    :func:`laps_to_direction_labels`:

    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.segmentation import (
    ...     detect_laps,
    ...     laps_to_direction_labels,
    ... )
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
    >>> position_bins = env.bin_at(positions)
    >>> times = np.linspace(0, 60, 300)
    >>> laps = detect_laps(
    ...     position_bins, times, env, method="region", start_region="start"
    ... )
    >>> direction_labels = laps_to_direction_labels(laps, times)
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
    # Validate method
    if method not in ("auto", "reference", "region"):
        raise ValueError(
            f"method must be one of 'auto', 'reference', 'region', got '{method}'"
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
                fix="pass reference_lap=position_bins_of_one_complete_lap when method='reference'",
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

    position_bins, times = _validated_samples(
        position_bins, times, env, call="detect_laps"
    )
    breaks = validate_break_before(break_before, len(times), call="detect_laps")
    runs = _event_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
    )
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
                    "back and forth use segment_trials or "
                    "detect_runs_between_regions.",
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
        return laps
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
            raise ValueError(
                "reference_lap is empty; pass the bin sequence of one complete "
                "lap, for example position_bins[lap_start:lap_stop]."
            )
    else:
        found = _first_circuit(position_bins, runs, env)
        if found is None:
            return []
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
    return laps


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
                        end_time=float(times[back]),
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
    edges = np.diff(np.r_[1, inside.astype(np.int8), 1])
    exits, returns = np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)
    trips: list[tuple[int, int, Literal["round", "back", "near"]]] = []
    for leave, back in zip(exits.tolist(), returns.tolist(), strict=True):
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
    away = np.r_[False, steps >= cut, False]
    starts = np.flatnonzero(~away[:-1] & away[1:])
    stops = np.flatnonzero(away[:-1] & ~away[1:]) - 1
    first, last = side[starts], side[stops]
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
    inside = zone[position_bins].astype(np.int8)
    edges = np.diff(np.r_[0, inside, 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # (n_samples,) distance from each sample's bin center to the anchor's
    offsets = np.linalg.norm(
        env.bin_centers[position_bins] - env.bin_centers[anchor], axis=1
    )
    return [
        int(a + np.argmin(offsets[a:b])) for a, b in zip(starts, stops, strict=True)
    ]


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
                found = bins[:stop]
                return found
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
                    end_time=float(times[candidate]),
                    direction=lap_direction,
                    overlap_score=float(overlap),
                )
            )
        starts, pieces = [candidate], []
        reach = np.zeros_like(template)
    return laps


# =============================================================================
# Direction-label bridges (lap/run -> per-timepoint labels)
# =============================================================================


def laps_to_direction_labels(
    laps: list[Lap],
    times: NDArray[np.float64],
) -> NDArray[np.object_]:
    """Convert detected laps into per-timepoint direction labels.

    Bridges :func:`detect_laps` to :func:`compute_directional_place_fields`
    and :func:`labels_to_epochs`: every sample falling inside a lap is labeled
    with that lap's ``direction`` (``"clockwise"`` or ``"counter-clockwise"``).
    Samples outside every lap, and inside a lap whose direction is
    ``"unknown"``, are ``None``, which both functions leave out. The returned
    array is drop-in for their labels argument.

    Parameters
    ----------
    laps : list[Lap]
        Laps from :func:`detect_laps`. Each contributes its ``[start_time,
        end_time)`` interval (half-open) and ``direction``.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for the whole session (seconds). The output has this length.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Per-timepoint direction label, dtype ``object``. Samples outside every
        lap, or in a lap of unknown direction (not resolved, e.g. in non-2D
        environments), are ``None``.

    See Also
    --------
    detect_laps : Detect laps in a circular-track trajectory.
    runs_to_direction_labels : Direction labels from detected runs.
    goal_pair_direction_labels : Direction labels from trialized tasks.

    Notes
    -----
    Laps later in the list overwrite earlier ones wherever their time intervals
    overlap (last-writer-wins). ``detect_laps`` emits non-overlapping laps, so
    this matters only for hand-constructed or merged lap lists.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.segmentation import (
    ...     Lap,
    ...     laps_to_direction_labels,
    ... )
    >>> times = np.linspace(0.0, 10.0, 11)
    >>> laps = [
    ...     Lap(0.0, 4.0, "clockwise", 1.0),
    ...     Lap(6.0, 10.0, "counter-clockwise", 1.0),
    ... ]
    >>> labels = laps_to_direction_labels(laps, times)
    >>> labels[0], labels[5], labels[9], labels[-1]
    ('clockwise', None, 'counter-clockwise', None)
    """
    intervals = [
        (
            lap.start_time,
            lap.end_time,
            None if lap.direction == "unknown" else lap.direction,
        )
        for lap in laps
    ]
    return _interval_labels(times, intervals)


def runs_to_direction_labels(
    runs: list[Run],
    times: NDArray[np.float64],
    *,
    labels: object | Sequence[object] = "run",
    success_only: bool = False,
) -> NDArray[np.object_]:
    """Convert detected runs into per-timepoint direction labels.

    Bridges :func:`detect_runs_between_regions` to
    :func:`compute_directional_place_fields` and :func:`labels_to_epochs`.
    The caller supplies the label(s): either a single label for every run, or
    one label per run.
    Samples inside no run are ``None``, which both functions leave out. The
    returned array is drop-in for their labels argument.

    Parameters
    ----------
    runs : list[Run]
        Runs from :func:`detect_runs_between_regions`. Each contributes its
        ``[start_time, end_time)`` interval (half-open).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for the whole session (seconds). The output has this length.
    labels : object or sequence of object, optional
        Label to assign. A scalar (e.g. ``"run"``, the default) labels every
        run identically. A sequence must have one entry per run (matched by
        position) and is applied before any ``success_only`` filtering, so its
        length must equal ``len(runs)``.
    success_only : bool, optional
        If True, only runs that reached their target (``outcome ==
        'reached'``) contribute labels; timed-out, returned and censored runs
        are left as ``None``. Default False.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Per-timepoint label, dtype ``object``. Samples outside every (selected)
        run are ``None``.

    Raises
    ------
    ValueError
        If ``labels`` is a sequence whose length differs from ``len(runs)``.

    See Also
    --------
    detect_runs_between_regions : Detect runs from a source to a target region.
    running_direction_labels : First-class inbound/outbound labeler.
    laps_to_direction_labels : Direction labels from detected laps.

    Notes
    -----
    Runs later in the list overwrite earlier ones wherever their time intervals
    overlap (last-writer-wins).

    A plain ``str`` is treated as a single scalar label, not as a per-run
    sequence of characters.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.segmentation import (
    ...     Run,
    ...     runs_to_direction_labels,
    ... )
    >>> times = np.linspace(0.0, 10.0, 11)
    >>> bins = np.array([0, 1, 2], dtype=np.int64)
    >>> runs = [
    ...     Run(0.0, 4.0, "home", "reward", "reached", bins),
    ...     Run(6.0, 10.0, "reward", "home", "reached", bins[::-1]),
    ... ]
    >>> labels = runs_to_direction_labels(runs, times, labels=["outbound", "inbound"])
    >>> labels[0], labels[5], labels[9], labels[-1]
    ('outbound', None, 'inbound', None)
    """
    if isinstance(labels, (Sequence, np.ndarray)) and not isinstance(
        labels, (str, bytes)
    ):
        per_run_labels: list[object] = list(labels)
        if len(per_run_labels) != len(runs):
            raise ValueError(
                f"labels sequence must have one entry per run. "
                f"Got {len(per_run_labels)} labels for {len(runs)} runs."
            )
    else:
        per_run_labels = [labels] * len(runs)

    intervals = [
        (run.start_time, run.end_time, label)
        for run, label in zip(runs, per_run_labels, strict=True)
        if not (success_only and not run.success)
    ]
    return _interval_labels(times, intervals)


def running_direction_labels(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    start_region: str,
    end_regions: str | Sequence[str],
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    min_speed: float | None = None,
    successful_only: bool = True,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.object_]:
    """Label per-timepoint inbound/outbound running direction on a linear track.

    Inbound/outbound labeler for linear-track and W-track tasks, built on
    :func:`detect_runs_between_regions`. Outbound runs go from
    ``start_region`` to an end region; inbound runs come back from an end region
    to ``start_region``. Every sample inside an outbound run is labeled
    ``"outbound"``, every sample inside an inbound run ``"inbound"``, and all
    other samples ``None``, which :func:`compute_directional_place_fields` and
    :func:`labels_to_epochs` leave out. The returned array is drop-in for
    their labels argument.

    Follows the segmentation argument-order convention
    (``position_bins, times, env, *, ...``).

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Discretized position indices (one per sample). Use
        ``env.bin_at(positions)`` to obtain these from continuous coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to ``position_bins`` (seconds). The output has
        this length.
    env : Environment
        Environment containing ``start_region`` and every end region.
    start_region : str
        Name of the home / start region. Outbound runs leave it; inbound runs
        return to it.
    end_regions : str or sequence of str
        End region name(s) (e.g. ``"reward"`` or ``["reward_left",
        "reward_right"]`` for a W-track). A single string is treated as one
        region.
    min_duration : float, optional
        Minimum run duration in seconds, forwarded to
        :func:`detect_runs_between_regions`. Default 0.5.
    max_duration : float, optional
        Maximum run duration in seconds, forwarded to
        :func:`detect_runs_between_regions`. Default 10.0.
    min_speed : float or None, optional
        Optional minimum (bin-quantized) speed, forwarded to
        :func:`detect_runs_between_regions`. Default None.
    successful_only : bool, optional
        If True (default), only runs that reached their target contribute
        labels. If False, timed-out, returned and censored runs are labeled
        by their attempted direction too, unless they overlap a run that
        reached its target (an attempt toward one end region that reached
        another is not a failed attempt).
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Per-timepoint label drawn from ``{"inbound", "outbound", None}``,
        dtype ``object``.

    Raises
    ------
    ValueError
        If ``start_region`` or any end region is not in ``env.regions``,
        or if ``position_bins`` and ``times`` are not 1-D with the same
        length, times are not finite and strictly increasing, or a bin index
        is neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.
    CartesianOnlyError
        If ``min_speed`` is given and ``env`` is polar: run speed comes from
        Cartesian distances between bin centers.

    See Also
    --------
    detect_runs_between_regions : Underlying source-to-target run detector.
    runs_to_direction_labels : Generic run-to-label bridge.
    goal_pair_direction_labels : Arrow-notation labels for trialized tasks.

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

    Outbound and inbound runs are detected per end region (start -> end and
    end -> start) with :func:`detect_runs_between_regions`, so each run ends
    when the animal re-enters the region it left: on a W-track, center ->
    left -> center -> right gives outbound, inbound, outbound runs, never one
    run spanning the trip home. Where intervals still overlap, inbound labels
    are applied after outbound (last-writer-wins); on an out-and-back
    trajectory the two never overlap, yielding contiguous ``"outbound"`` then
    ``"inbound"`` stretches separated by ``None`` at the turnaround.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior.segmentation import running_direction_labels
    >>> # Thin 2D corridor along x (a linear track).
    >>> x = np.linspace(0.0, 100.0, 200)
    >>> samples = np.column_stack([x, np.zeros_like(x)])
    >>> env = Environment.from_samples(samples, bin_size=5.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("home", polygon=Point(0.0, 0.0).buffer(8.0))
    >>> _ = regions.add("reward", polygon=Point(100.0, 0.0).buffer(8.0))
    >>> env = env.with_regions(regions)
    >>> # Out-and-back trajectory: home -> reward -> home.
    >>> xt = np.concatenate([np.linspace(0.0, 100.0, 50), np.linspace(100.0, 0.0, 50)])
    >>> trajectory = np.column_stack([xt, np.zeros_like(xt)])
    >>> position_bins = env.bin_at(trajectory)
    >>> ts = np.linspace(0.0, 10.0, len(trajectory))
    >>> labels = running_direction_labels(
    ...     position_bins, ts, env, start_region="home", end_regions="reward"
    ... )
    >>> "outbound" in labels and "inbound" in labels
    True
    """
    position_bins, times = _validated_samples(
        position_bins, times, env, call="running_direction_labels"
    )

    end_region_names = _region_names(end_regions, argument="end_regions")
    _validate_durations(min_duration, max_duration, call="running_direction_labels")
    masks = {
        start_region: _require_region_mask(
            env,
            start_region,
            operation="running_direction_labels()",
            argument="start_region",
        )
    }
    for end_region in end_region_names:
        masks[end_region] = _require_region_mask(
            env,
            end_region,
            operation="running_direction_labels()",
            argument="end_regions",
            in_list=not isinstance(end_regions, str),
        )
    breaks = validate_break_before(
        break_before, len(times), call="running_direction_labels"
    )
    if min_speed is not None:
        _require_run_speed_frame(env, call="running_direction_labels")
    validate_finite_scalar(
        min_speed, name="min_speed", positive=False, none_allowed=True
    )
    recordings = _event_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
    )

    runs: list[tuple[Run, str]] = []
    for end_region in end_region_names:
        for source, target, label in (
            (start_region, end_region, "outbound"),
            (end_region, start_region, "inbound"),
        ):
            for recording in recordings:
                runs.extend(
                    (run, label)
                    for run in _detect_runs_between_regions_contiguous(
                        position_bins[recording],
                        times[recording],
                        env,
                        source=source,
                        target=target,
                        source_mask=masks[source],
                        target_mask=masks[target],
                        min_duration=min_duration,
                        max_duration=max_duration,
                        min_speed=min_speed,
                    )
                )

    reached = [run for run, _ in runs if run.success]
    intervals: list[tuple[float, float, object]] = []
    for run, label in runs:
        if not run.success and (
            successful_only
            # An attempt toward one end region that overlaps a run that
            # reached another (or came home) was not a failed attempt.
            or any(
                run.start_time < other.end_time and other.start_time < run.end_time
                for other in reached
            )
        ):
            continue
        intervals.append((run.start_time, run.end_time, label))
    # Inbound after outbound, as documented (last-writer-wins on overlap).
    intervals.sort(key=lambda interval: interval[2] == "inbound")
    return _interval_labels(times, intervals)


# =============================================================================
# Trials
# =============================================================================


def segment_trials(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    start_region: str,
    end_regions: str | Sequence[str],
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> list[Trial]:
    """Segment trajectory into behavioral trials.

    Identifies discrete behavioral trials defined by departures from a start
    region followed by navigation to one of several end regions. Commonly used in
    spatial navigation tasks like T-mazes, Y-mazes, and radial arm mazes to
    analyze choice behavior, learning curves, and spatial strategies.

    A trial is defined as:
    1. Departure from the start region (trial start, at the last sample
       inside it, so waiting at the start is not part of the trial)
    2. Trajectory through environment
    3. Whichever happens first:
       - Entry to one of the end regions (``outcome='reached'``)
       - ``max_duration`` elapses without reaching an end region
         (``outcome='timeout'``)
       - Return to the start region before reaching an end region (an
         aborted trial, ``outcome='returned'``)
       - Observation stops: a tracking gap longer than ``max_gap``, an
         ``epochs`` edge, a ``break_before`` or the end of the recording
         (``outcome='censored'``; the result is unknown)

    Samples outside every bin (``-1``, such as lost tracking while the animal
    waits at a well) are missing: a dropout no longer than ``max_gap`` is
    bridged, so it is not a departure or a return, and a longer one ends the
    recording as any tracking loss does.

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to position bins (seconds).
    env : Environment
        Environment containing region definitions.
    start_region : str
        Name of start region. Must exist in env.regions.
    end_regions : str or sequence of str
        End region name(s) (e.g., ``['left', 'right']`` for a T-maze). A
        single string is one region. Must all exist in env.regions.
    min_duration : float, optional
        Minimum trial duration in seconds. Default: 1.0.
        Trials shorter than this are excluded.
    max_duration : float, optional
        Maximum trial duration in seconds. Default: 15.0.
        A trial that has not reached an end region by ``start_time +
        max_duration`` times out at the last sample within that deadline,
        even when the next sample is inside an end region.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    list[Trial]
        Detected trials in time order. Each records ``start_time`` (the last
        sample in the start region before departure), ``end_time``,
        ``start_region``, ``end_region`` (the end region reached, or None),
        ``outcome`` (``'reached'``, ``'timeout'``, ``'returned'`` or
        ``'censored'``, see :class:`Trial`) and the derived ``success``
        (``outcome == 'reached'``).

    Raises
    ------
    ValueError
        If start_region is None or not in env.regions.
    ValueError
        If end_regions is None, empty, or any region not in env.regions.
    ValueError
        If position_bins and times are not 1-D with the same length, if
        times are not finite and strictly increasing, or if a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices (a float array
        would be truncated to unrelated bins).
    ValueError
        If min_duration is negative or not finite, or max_duration is not
        greater than min_duration.
    ValueError
        If ``start_region`` is also one of ``end_regions``.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_region_crossings : Detect individual region crossings
    detect_runs_between_regions : Detect runs between two specific regions

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

    A trial in progress when a recording ends is emitted there with
    ``outcome='censored'``, as at the end of a complete recording. Leave
    censored trials out of fraction-correct measures: the animal was not seen
    to fail.

    Trial segmentation is fundamental for analyzing spatial navigation tasks:

    **T-maze**: start_region='start', end_regions=['left', 'right']
    - Analyzes spatial alternation behavior
    - Learning curves: fraction correct over trials
    - Strategy changes: random → alternation → place-based

    **Y-maze**: start_region='center', end_regions=['arm1', 'arm2', 'arm3']
    - Spontaneous alternation (working memory)
    - Arm entry sequences and patterns
    - Exploration vs exploitation

    **Radial arm maze**: start_region='center', end_regions=['arm1', ..., 'arm8']
    - Working and reference memory tasks
    - Optimal foraging strategies
    - Error analysis (revisits, skipped arms)

    Trials are filtered by duration to exclude:
    - Very brief entries/exits (min_duration filter)
    - Stuck or disengaged periods (max_duration timeout)

    If an animal re-enters the start region before reaching an end region,
    the previous (aborted) trial is emitted with ``outcome='returned'`` and
    ``end_region=None`` provided it lasted at least ``min_duration``; the next
    trial begins when the animal leaves the start region again. Aborted,
    timed-out and censored trials are reported rather than silently dropped.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>> # Create T-maze environment
    >>> x = np.linspace(0, 100, 100)
    >>> y = np.linspace(0, 100, 100)
    >>> xx, yy = np.meshgrid(x, y)
    >>> positions = np.column_stack([xx.ravel(), yy.ravel()])
    >>> env = Environment.from_samples(positions, bin_size=3.0)
    >>> # Define regions: start at bottom, left/right at top
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("start", polygon=Point(50.0, 10.0).buffer(8.0))
    >>> _ = regions.add("left", polygon=Point(20.0, 80.0).buffer(8.0))
    >>> _ = regions.add("right", polygon=Point(80.0, 80.0).buffer(8.0))
    >>> env = env.with_regions(regions)
    >>> # Create trajectory with 2 trials: start→left, start→right
    >>> x_traj = np.concatenate(
    ...     [
    ...         np.linspace(50, 50, 10),  # In start
    ...         np.linspace(50, 20, 15),  # Go to left
    ...         np.linspace(20, 50, 15),  # Return to start
    ...         np.linspace(50, 80, 15),  # Go to right
    ...     ]
    ... )
    >>> y_traj = np.concatenate(
    ...     [
    ...         np.linspace(10, 10, 10),
    ...         np.linspace(10, 80, 15),
    ...         np.linspace(80, 10, 15),
    ...         np.linspace(10, 80, 15),
    ...     ]
    ... )
    >>> trajectory = np.column_stack([x_traj, y_traj])
    >>> position_bins = env.bin_at(trajectory)
    >>> times = np.arange(len(trajectory), dtype=float)
    >>> # Treat these 1 Hz samples as continuous with max_gap=None
    >>> trials = segment_trials(
    ...     position_bins,
    ...     times,
    ...     env,
    ...     start_region="start",
    ...     end_regions=["left", "right"],
    ...     min_duration=5.0,
    ...     max_duration=50.0,
    ...     max_gap=None,
    ... )
    >>> len(trials) >= 1  # Should detect at least one trial
    True
    >>> [t.outcome for t in trials]  # Both reached an end region
    ['reached', 'reached']
    >>> end_regions = [t.end_region for t in trials]
    >>> "left" in end_regions and "right" in end_regions  # Both choices present
    True

    References
    ----------
    .. [1] Olton, D. S., & Samuelson, R. J. (1976). Remembrance of places
           passed: Spatial memory in rats. Journal of Experimental Psychology:
           Animal Behavior Processes, 2(2), 97-116.
    .. [2] Wood, E. R., Dudchenko, P. A., Robitsek, R. J., & Eichenbaum, H.
           (2000). Hippocampal neurons encode information about different
           types of memory episodes occurring in the same location. Neuron,
           27(3), 623-633.
    """
    start_mask = _require_region_mask(
        env, start_region, operation="segment_trials()", argument="start_region"
    )

    end_regions = _region_names(end_regions, argument="end_regions")
    end_masks = {
        region: _require_region_mask(
            env,
            region,
            operation="segment_trials()",
            argument="end_regions",
            in_list=True,
        )
        for region in end_regions
    }

    # Prevent start_region from being in end_regions (typical neuroscience practice)
    if start_region in end_regions:
        raise ValueError(
            f"start_region '{start_region}' cannot be in end_regions. "
            f"Start and end regions must be spatially distinct for trial segmentation."
        )

    position_bins, times = _validated_samples(
        position_bins, times, env, call="segment_trials"
    )

    _validate_durations(min_duration, max_duration, call="segment_trials")

    breaks = validate_break_before(break_before, len(times), call="segment_trials")
    results: list[Trial] = []
    for run in _event_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
    ):
        results.extend(
            _segment_trials_contiguous(
                position_bins[run],
                times[run],
                start_region=start_region,
                start_mask=start_mask,
                end_masks=end_masks,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        )
    return results


def _segment_trials_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    *,
    start_region: str,
    start_mask: NDArray[np.bool_],
    end_masks: dict[str, NDArray[np.bool_]],
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[Trial]:
    """Analyze one recording after the public input validation.

    ``start_mask`` and each of ``end_masks`` (shape ``(n_bins,)``) mark a
    region's bins; ``end_masks`` is in ``end_regions`` order.
    """
    # Check which trajectory samples are in which regions. Unlocated samples
    # were removed by the caller and the gaps around them bridged.
    position_bins = np.asarray(position_bins, dtype=np.int64)
    in_start = _safe_gather(start_mask, position_bins, fill=False)
    in_end_regions = {
        region: _safe_gather(mask, position_bins, fill=False)
        for region, mask in end_masks.items()
    }

    # Segment into trials. A trial starts when the animal leaves the start
    # region, at the last sample inside it, so time spent waiting at the start
    # does not count toward the trial. It ends at whichever comes first: an
    # end-region entry, a return to the start region, the max_duration
    # deadline, or the end of this recording.
    trials: list[Trial] = []
    trial_start_time: float | None = None

    def _emit(end_time: float, outcome: Outcome, end_region: str | None) -> None:
        assert trial_start_time is not None
        if end_time - trial_start_time >= min_duration:
            trials.append(
                Trial(
                    start_time=trial_start_time,
                    end_time=end_time,
                    start_region=start_region,
                    end_region=end_region,
                    outcome=outcome,
                )
            )

    for i in range(len(position_bins)):
        if trial_start_time is None:
            if i > 0 and in_start[i - 1] and not in_start[i]:
                trial_start_time = float(times[i - 1])
            else:
                continue

        elapsed = times[i] - trial_start_time
        if elapsed > max_duration:
            # The deadline fell between samples: the trial timed out at the
            # last sample within it, whatever this sample shows.
            _emit(float(times[i - 1]), "timeout", None)
            trial_start_time = None
            continue
        reached = next(
            (region for region, inside in in_end_regions.items() if inside[i]), None
        )
        if reached is not None:
            _emit(float(times[i]), "reached", reached)
        elif in_start[i]:
            _emit(float(times[i]), "returned", None)
        elif elapsed >= max_duration:
            _emit(float(times[i]), "timeout", None)
        else:
            continue
        trial_start_time = None

    # A trial still under way when the recording ends was not seen to finish.
    if trial_start_time is not None:
        _emit(float(times[-1]), "censored", None)

    return trials


# =============================================================================
# Trajectory similarity and goal-directed runs
# =============================================================================


def trajectory_similarity(
    position_bins1: NDArray[np.int64],
    position_bins2: NDArray[np.int64],
    env: Environment,
    *,
    method: Literal["jaccard", "correlation", "hausdorff", "dtw"] = "jaccard",
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
          environment or a linearized track. Range [0, 1].
        - 'hausdorff': 1 - Hausdorff distance / environment extent. Range [0, 1].
        - 'dtw': 1 - dynamic time warping distance / its largest possible
          value in this environment. Range [0, 1].
        Default is 'jaccard'.

    Returns
    -------
    float
        Similarity score in [0, 1] where 1.0 indicates identical trajectories
        and 0.0 indicates completely different trajectories.

    Raises
    ------
    ValueError
        If a trajectory is empty or has no sample inside the environment.
    ValueError
        If a bin index is not ``-1`` or an active bin of ``env``.
    ValueError
        If method is not one of the supported options.
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
    is the environment extent (the diagonal of the bin centers' bounding box).

    **Dynamic time warping (DTW)** finds optimal alignment between trajectories
    allowing temporal shifts. Useful for comparing paths with different speeds.
    The summed cost of the optimal alignment is divided by
    :math:`\\max(n_1, n_2)\\, d_{max}`: a diagonal alignment path has
    :math:`\\max(n_1, n_2)` steps and no step costs more than :math:`d_{max}`,
    so the optimum can never exceed it.

    Both Hausdorff and DTW scores are relative to the environment's size:
    the same two paths, 5 cm apart, score higher in a 10 m arena than on a 1
    m track. Compare these scores only between paths in one environment.

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
        raise ValueError("Trajectories cannot be empty")

    valid_methods = ["jaccard", "correlation", "hausdorff", "dtw"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    if method == "correlation":
        from neurospatial.environment._queries import require_track_extents

        starts, stops = require_track_extents(
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

    elif method == "correlation":
        # Sequential correlation
        # Align to same length using indices modulo shorter length
        if len(position_bins1) < len(position_bins2):
            shorter, longer = position_bins1, position_bins2
        else:
            shorter, longer = position_bins2, position_bins1

        # Sample longer trajectory at same rate as shorter
        if len(shorter) < 2:
            # Can't compute correlation with < 2 points
            return 1.0 if np.array_equal(position_bins1, position_bins2) else 0.0

        indices = np.linspace(0, len(longer) - 1, len(shorter), dtype=int)
        longer_sampled = longer[indices]
        # Compare only aligned pairs in which both samples have a place.
        observed = (shorter >= 0) & (longer_sampled >= 0)
        shorter, longer_sampled = shorter[observed], longer_sampled[observed]
        if len(shorter) < 2:
            return 1.0 if np.array_equal(shorter, longer_sampled) else 0.0

        # Correlate positions along the track, not bin numbers.
        first = track_position[shorter]
        second = track_position[longer_sampled]
        if np.ptp(first) == 0 or np.ptp(second) == 0:
            # Constant sequence - return 1 if identical, 0 otherwise
            return 1.0 if np.array_equal(first, second) else 0.0

        corr, _ = pearsonr(first, second)
        # Map [-1, 1] to [0, 1], clamp negative correlations to 0
        return float(max(0.0, corr))

    elif method == "hausdorff":
        # Hausdorff distance between paths
        # Get positions from bin indices
        pos1 = env.bin_centers[position_bins1]
        pos2 = env.bin_centers[position_bins2]

        # Compute Hausdorff distance
        d_hausdorff = max(
            directed_hausdorff(pos1, pos2)[0],
            directed_hausdorff(pos2, pos1)[0],
        )

        # Normalize by environment extent (maximum possible distance)
        extent = np.ptp(env.bin_centers, axis=0)  # range per dimension
        max_distance = np.linalg.norm(extent)

        if max_distance == 0:
            return 1.0  # Degenerate environment

        # Convert distance to similarity [0, 1]
        similarity = 1.0 - min(d_hausdorff / max_distance, 1.0)
        return float(similarity)

    elif method == "dtw":
        # Dynamic time warping
        pos1 = env.bin_centers[position_bins1]
        pos2 = env.bin_centers[position_bins2]

        # Compute DTW distance using dynamic programming
        dtw_distance = _dtw_distance(pos1, pos2)

        # Normalize by the largest possible optimal cost: the diagonal path
        # has max(len1, len2) steps, each at most the environment's extent.
        extent = np.ptp(env.bin_centers, axis=0)
        max_bin_distance = np.linalg.norm(extent)
        max_dtw = max(len(pos1), len(pos2)) * max_bin_distance

        if max_dtw == 0:
            return 1.0

        # Convert to similarity [0, 1]
        similarity = 1.0 - min(dtw_distance / max_dtw, 1.0)
        return float(similarity)

    # All method cases covered above
    raise ValueError(f"Unreachable: method validation failed for '{method}'")


def _dtw_distance(seq1: NDArray[np.float64], seq2: NDArray[np.float64]) -> float:
    """Compute Dynamic Time Warping distance between two sequences.

    The cumulative cost ``D[i, j] = cost(i, j) + min(D[i-1, j], D[i, j-1],
    D[i-1, j-1])`` is filled one anti-diagonal ``i + j = k`` at a time: every
    cell on diagonal ``k`` depends only on diagonals ``k - 1`` and ``k - 2``,
    so each diagonal is one vectorized step and only three diagonals are kept.

    Parameters
    ----------
    seq1 : NDArray[np.float64], shape (n1, n_dims)
        First sequence of positions; non-empty.
    seq2 : NDArray[np.float64], shape (n2, n_dims)
        Second sequence of positions; non-empty.

    Returns
    -------
    float
        DTW distance: the smallest summed Euclidean distance over monotone
        alignments of the two sequences.
    """
    cost = cdist(seq1, seq2)
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


def detect_goal_directed_runs(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    goal_region: str,
    directedness_threshold: float = 0.7,
    min_progress: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Detect goal-directed runs: approaches to a goal that are efficient.

    Splits the trajectory into approaches (stretches where the distance to
    the goal falls, from the departure to the closest point reached) and
    keeps those that move efficiently toward the goal region. Useful for
    analyzing goal-directed replay, learning dynamics, and navigation
    strategies.

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory. ``-1`` marks a
        sample outside the environment or missing.
    times : NDArray[np.float64], shape (n_samples,)
        Time stamps corresponding to position bins (seconds).
    env : Environment
        Environment containing the goal region definition.
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
    NDArray[np.float64], shape (n_runs, 2)
        Goal-directed approaches as ``[start, stop)`` rows in seconds, in
        time order: ``start`` is the departure sample and ``stop`` the
        arrival sample, so a row covers the travel between them (interval
        ``k`` is ``[times[k], times[k + 1])``, as in
        :func:`segment_by_velocity`). A row means the directedness and
        net-progress criteria were met; the arrival need not be in the goal
        region. Pass the rows as ``epochs=`` to restrict another analysis to
        them.

    Raises
    ------
    ValueError
        If goal_region not in env.regions, or if directedness_threshold
        not in [0, 1].
    ValueError
        If min_progress is negative or not finite, or is None while
        ``env.units`` is not a metric length.
    ValueError
        If position_bins and times are not 1-D with the same length, if
        times are not finite and strictly increasing, or if a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices (a float array
        would be truncated to unrelated bins).
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    trajectory_similarity : Compare similarity between trajectories.
    detect_runs_between_regions : Detect runs from source to target regions.

    Notes
    -----
    A missing sample (bin ``-1``, outside the environment) is skipped as if
    its row were deleted, and the gap around it is bridged when the valid
    samples on either side are no more than ``max_gap`` apart. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording; each resulting stretch is analyzed
    on its own, so no segment spans a pause.

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
    is never an approach. To segment completed source-to-goal traversals,
    use ``detect_runs_between_regions``.

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
    >>> position_bins = env.bin_at(np.c_[x, np.zeros_like(x)])
    >>> times = np.arange(41) / 10.0
    >>> detect_goal_directed_runs(position_bins, times, env, goal_region="goal")
    array([[0., 4.]])

    Out to the goal, back, and out again: each approach is its own row, and
    the walk back is not goal-directed:

    >>> trips = np.r_[x, x[::-1][1:], x[1:]]
    >>> detect_goal_directed_runs(
    ...     env.bin_at(np.c_[trips, np.zeros_like(trips)]),
    ...     np.arange(trips.size) / 10.0,
    ...     env,
    ...     goal_region="goal",
    ... )
    array([[ 0.,  4.],
           [ 8., 12.]])

    Walking the same path away from the goal is not goal-directed:

    >>> detect_goal_directed_runs(
    ...     position_bins[::-1], times, env, goal_region="goal"
    ... ).shape
    (0, 2)

    """
    # Input validation
    _require_region_mask(
        env,
        goal_region,
        operation="detect_goal_directed_runs()",
        argument="goal_region",
    )

    if not 0.0 <= directedness_threshold <= 1.0:
        raise ValueError(
            f"directedness_threshold must be in [0, 1], got {directedness_threshold}"
        )

    if min_progress is None:
        from neurospatial._units import default_length

        min_progress = default_length(
            env,
            DEFAULT_MIN_PROGRESS_CM,
            name="min_progress",
            call="detect_goal_directed_runs()",
        )
    validate_finite_scalar(min_progress, name="min_progress", positive=False)

    position_bins, times = _validated_samples(
        position_bins, times, env, call="detect_goal_directed_runs"
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
    results: list[tuple[float, float]] = []
    for run in observed_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=_located(position_bins, env),
    ):
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
    return np.array(results, dtype=np.float64).reshape(-1, 2)


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
    # Both bins of every step summed below lie in the goal's connected
    # component, so every summed step is finite.
    path = np.r_[
        0.0, np.cumsum(paths.between_bins(position_bins[:-1], position_bins[1:]))
    ]
    rows: list[tuple[float, float]] = []
    # Samples that cannot reach the goal split the recording.
    finite = np.r_[False, np.isfinite(distances), False]
    starts = np.flatnonzero(~finite[:-1] & finite[1:])
    stops = np.flatnonzero(finite[:-1] & ~finite[1:])
    for first, stop in zip(starts, stops, strict=True):
        for departure, arrival in _approach_bouts(distances[first:stop], tolerance):
            departure, arrival = first + departure, first + arrival
            progress = distances[departure] - distances[arrival]
            directedness = progress / (path[arrival] - path[departure])
            if directedness >= directedness_threshold and progress >= min_progress:
                rows.append((float(times[departure]), float(times[arrival])))
    return rows


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Dataclasses
    "Crossing",
    "Lap",
    "Run",
    "Trial",
    # Functions
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "detect_runs_between_regions",
    "laps_to_direction_labels",
    "running_direction_labels",
    "runs_to_direction_labels",
    "segment_by_velocity",
    "segment_trials",
    "trajectory_similarity",
]
