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
    Goal-directed stretches of recording, as an (n, 2) array

Classes
-------
Trial
    Dataclass representing a behavioral trial with fields:
    - start_time: Trial onset time
    - end_time: Trial offset time
    - start_region: Region where trial started
    - end_region: Region reached (None unless outcome is 'reached')
    - outcome: 'reached', 'timeout', 'returned' or 'censored'
Crossing
    Dataclass for region entry/exit events
Lap
    Dataclass for lap detection results
Run
    Dataclass for runs between regions
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import directed_hausdorff, euclidean
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
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
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
    """Check a bin trajectory and its timestamps; return int64 and float64 copies.

    ``position_bins`` must be integer bin indices of ``env`` or ``-1`` (outside
    the environment, as :meth:`Environment.bin_at` returns); a float array
    would be truncated to unrelated bins. ``times`` must be finite and
    strictly increasing, in seconds.
    """
    from neurospatial._observation import _validated_signal_times
    from neurospatial.environment import _validate_target_bins

    n_bins, n_times = np.shape(position_bins), np.shape(times)
    if len(n_bins) != 1 or len(n_times) != 1 or n_bins != n_times:
        raise ValueError(
            _format_error(
                f"position_bins and times must be 1-D with the same length; got "
                f"shapes {n_bins} and {n_times}.",
                why="each trajectory sample needs its corresponding timestamp.",
                fix="pass one timestamp per position bin on the same clock.",
            )
        )
    bins = _validate_target_bins(
        position_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation=f"{call}()",
        argument="position_bins",
        allow_missing=True,
    ).astype(np.int64)
    if n_times[0] == 0:
        return bins, np.asarray(times, dtype=np.float64)
    return bins, _validated_signal_times(times, call=call)


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


def _positive_dt(
    times: NDArray[np.float64], *, name: str = "times"
) -> NDArray[np.float64]:
    """Return strictly-positive consecutive time deltas, or raise.

    Computes ``np.diff(times)`` and rejects any delta that is not strictly
    positive (duplicate or out-of-order timestamps) with an actionable message.
    Use this before dividing distances/displacements by a per-sample ``dt``.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Monotonically increasing timestamps in seconds.
    name : str, optional
        Argument name, used in the error message. Default ``"times"``.

    Returns
    -------
    NDArray[np.float64], shape (n_samples - 1,)
        Consecutive time deltas, all ``> 0``.

    Raises
    ------
    ValueError
        If any ``np.diff(times) <= 0`` (duplicate or non-increasing timestamps).
    """
    dt = np.diff(np.asarray(times, dtype=np.float64))
    bad = dt <= 0
    if bad.any():
        n = int(bad.sum())
        first = int(np.argmax(bad))
        raise ValueError(
            f"{name} must be strictly increasing for a per-sample time step, "
            f"but {n} consecutive delta(s) are <= 0 "
            f"(first between index {first} and {first + 1}: "
            f"dt={dt[first]!r}). Sort and de-duplicate timestamps before calling."
        )
    return dt


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
        Time of crossing event (seconds).
    direction : str
        Direction of crossing: 'entry' or 'exit'.
    bin_index : int
        Bin index at crossing.
    """

    time: float
    direction: Literal["entry", "exit"]
    bin_index: int


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

    The :class:`~neurospatial.Environment` holding the region definition is
    passed positionally as the third argument in the new (0.6+) call form.

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
        List of crossing events, each with time, direction, and bin_index.
        Sorted by time in ascending order.

    Raises
    ------
    ValueError
        If region_name not in env.regions.
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
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

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
    # Validate inputs
    _require_region_mask(
        env, region_name, operation="detect_region_crossings()", argument="region_name"
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
                env,
                region_name=region_name,
                direction=direction,
            )
        )
    return results


def _detect_region_crossings_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    region_name: str,
    direction: Literal["both", "entry", "exit"] = "both",
) -> list[Crossing]:
    """Detect crossings within one recording, after public input validation."""
    # Get bins in region using existing regions_to_mask functionality
    from neurospatial.ops.mapping import regions_to_mask

    region_mask = regions_to_mask(env, [region_name])

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
                    time=times[idx], direction="entry", bin_index=position_bins[idx]
                )
            )

    if direction in ["both", "exit"]:
        exit_indices = np.where(transitions == -1)[0] + 1
        for idx in exit_indices:
            crossings.append(
                Crossing(
                    time=times[idx], direction="exit", bin_index=position_bins[idx]
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
        Minimum velocity (bin-center units/second) to count as
        movement. Velocity is computed from consecutive
        ``env.bin_centers[position_bins[i]]`` rather than continuous
        positions, so the value is bin-quantized; pre-filter on
        continuous positions if you need true sub-bin precision.
        If None, no velocity filtering is applied. Default: None.
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
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

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
        require_cartesian(
            env.coordinate_system,
            "detect_runs_between_regions(min_speed=...)",
            why=(
                "Run speed comes from the Cartesian distance between successive bin "
                "centers."
            ),
            fix="pass min_speed=None to detect runs from region entries alone",
        )
    # Validate inputs
    for name, argument in ((source, "source"), (target, "target")):
        _require_region_mask(
            env, name, operation="detect_runs_between_regions()", argument=argument
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
                min_duration=min_duration,
                max_duration=max_duration,
                min_speed=min_speed,
            )
        )
    return results


def _detect_runs_between_regions_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    source: str,
    target: str,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    min_speed: float | None = None,
) -> list[Run]:
    """Analyze one recording after the public input validation."""
    # Get region masks
    from neurospatial.ops.mapping import regions_to_mask

    source_mask = regions_to_mask(env, [source])
    target_mask = regions_to_mask(env, [target])

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
    """Segment trajectory into movement and rest periods based on velocity.

    Uses hysteresis thresholding to classify trajectory epochs as movement
    (velocity above ``min_speed``) or rest (velocity below it). Filters out
    brief segments shorter than ``min_duration``. Useful for identifying
    behavioral states, analyzing exploration patterns, and preprocessing
    for place field analysis.

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
        If positions and times have different lengths.
    ValueError
        If ``min_speed <= 0`` or ``hysteresis <= 1``.
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
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

    Smoothing and hysteresis are applied separately within each recording.
    A movement epoch in progress at its end is truncated there.

    Velocity is computed as Euclidean distance between consecutive samples
    divided by time difference. Velocities are smoothed using a moving average
    to reduce noise from measurement errors. The moving average uses
    edge-value (boundary-reflecting) padding so that velocity is not
    artificially suppressed at the start and end of the recording; this
    preserves movement epochs that extend to the recording boundary.

    Hysteresis thresholding prevents "flickering" near the threshold:
    - Movement starts when velocity exceeds ``min_speed``
    - Movement continues until velocity drops below ``min_speed / hysteresis``
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
    # Validate inputs
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="segment_by_velocity"
    )
    if min_speed <= 0:
        raise ValueError(f"min_speed must be positive. Got {min_speed}")

    if hysteresis <= 1.0:
        raise ValueError(f"hysteresis must be > 1.0 for stability. Got {hysteresis}")
    breaks = validate_break_before(break_before, len(times), call="segment_by_velocity")

    results: list[tuple[float, float]] = []
    for run in observed_runs(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        sample_valid=np.asarray(np.isfinite(positions).all(axis=1), dtype=bool),
    ):
        results.extend(
            _segment_by_velocity_contiguous(
                positions[run],
                times[run],
                min_speed,
                min_duration=min_duration,
                hysteresis=hysteresis,
                smooth_window=smooth_window,
            )
        )
    return np.array(results, dtype=np.float64).reshape(-1, 2)


def _segment_by_velocity_contiguous(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    min_speed: float,
    *,
    min_duration: float = 0.5,
    hysteresis: float = 2.0,
    smooth_window: float = 0.2,
) -> list[tuple[float, float]]:
    """Analyze one recording after the public input validation."""
    from neurospatial._velocity import smoothed_speed

    _positive_dt(times, name="times")
    # Speed of each transition i -> i+1, read at its end sample (see the
    # segment loop below); one run, so every interval is observed.
    speed = smoothed_speed(
        times,
        positions,
        np.ones(len(times) - 1, dtype=bool),
        smooth_window=smooth_window,
    )
    velocities = speed[1:]

    # Hysteresis thresholding
    lower_threshold = min_speed / hysteresis
    is_moving = np.zeros(len(velocities), dtype=bool)
    currently_moving = False

    for i in range(len(velocities)):
        if currently_moving:
            # Continue moving until velocity drops below lower threshold
            if velocities[i] < lower_threshold:
                currently_moving = False
            else:
                is_moving[i] = True
        else:
            # Start moving when velocity exceeds min_speed
            if velocities[i] > min_speed:
                currently_moving = True
                is_moving[i] = True

    # Detect segments (note: velocities correspond to transitions, so index times[1:])
    epochs: list[tuple[float, float]] = []
    in_segment = False
    segment_start: float | None = None

    def _emit(start: float, end: float) -> None:
        if (end - start) >= min_duration:
            epochs.append((float(start), float(end)))

    for i in range(len(is_moving)):
        time_idx = i + 1  # Velocity at i corresponds to transition from i to i+1

        if is_moving[i] and not in_segment:
            in_segment = True
            segment_start = times[time_idx]

        elif not is_moving[i] and in_segment:
            assert segment_start is not None  # Type narrowing for mypy
            _emit(segment_start, times[time_idx])
            in_segment = False

    # Handle case where trajectory ends while moving
    if in_segment:
        assert segment_start is not None  # Type narrowing for mypy
        _emit(segment_start, times[-1])

    return epochs


# =============================================================================
# Functions from laps.py
# =============================================================================


def _widened_overlap(
    template: NDArray[np.bool_],
    bins: NDArray[np.int64],
    adjacency: Any,
) -> float:
    """Jaccard index of two paths, each widened by one ring of neighbor bins.

    Parameters
    ----------
    template : NDArray[np.bool_], shape (n_bins,)
        The template's widened bins, from :func:`_widen`.
    bins : NDArray[np.int64], shape (n_samples,)
        Bin sequence of a candidate lap.
    adjacency : scipy.sparse array, shape (n_bins, n_bins)
        Bin connectivity.

    Returns
    -------
    float
        |A & B| / |A | B| of the widened bin sets, in [0, 1].
    """
    lap = _widen(bins, adjacency)
    union = np.count_nonzero(template | lap)
    return float(np.count_nonzero(template & lap) / union) if union else 0.0


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
        Name of start region for 'region' method.
        Required when method='region'.
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
        If start_region not in env.regions.
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
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

    Laps never span recordings. Samples outside the environment are left
    out as if their rows were deleted.

    **Lap Detection Methods:**

    - **Reference**: The first bin of ``reference_lap`` is the start line.
      Each pass through it (the anchor bin and its neighbors, so a fast
      animal cannot step over it) marks a candidate boundary, placed at the
      pass's sample nearest the anchor bin. A lap runs from one boundary to
      the next boundary at which the bins visited since the lap began
      overlap the template's bins with Jaccard index >= ``min_overlap``.
      Both bin sets are widened by one ring of neighbors first, so the
      overlap compares routes rather than which bins a sparse sampling
      happened to hit (``overlap_score`` reports this widened index).
      Boundaries come from places, not sample counts, so laps run at any
      speed are found whole; a pass without enough overlap (a turn back)
      stays inside the lap. Each lap covers ``[boundary, next boundary)``.
    - **Auto**: As 'reference', with the first complete circuit as the
      template: from the first observed sample to the first return to its
      bin after a trip to the far side of the track (a bin at least half the
      greatest path distance away). This first circuit is itself the first
      lap. Useful when no canonical lap is at hand; use 'reference' when the
      first circuit is atypical.
    - **Region**: Defines laps as segments between entries to a start
      region, useful for well-controlled tasks with clear start zones. A lap
      must reach the far side of the track, a bin at least half the greatest
      path distance from the start region (on a closed loop of length L the
      farthest point is L/2 away, and every full circuit passes it), so
      tracking jitter across the region edge does not split a lap. Each lap
      covers ``[entry, next lap's entry)``: ``end_time`` is the entry that
      starts the next lap. ``overlap_score`` is NaN.

    **Direction Detection:**

    For 2D trajectories, direction is the sign of the area enclosed by the
    lap path closed back to its start (shoelace formula on bin centers):
    positive is counter-clockwise, negative clockwise. Closing the path makes
    the answer independent of the coordinate origin and of where on the
    circuit the lap starts.

    **Applications:**

    - Lap-by-lap learning curves (Barnes et al., 1997)
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

    References
    ----------
    .. [1] Barnes, C. A., Suster, M. S., Shen, J., & McNaughton, B. L. (1997).
           Multistability of cognitive maps in the hippocampus of old rats.
           Nature, 388(6639), 272-275.
    .. [2] Dupret, D., O'Neill, J., Pleydell-Bouverie, B., & Csicsvari, J. (2010).
           The reorganization and reactivation of hippocampal maps predict
           spatial memory performance. Nature Neuroscience, 13(8), 995-1002.
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
        _require_region_mask(
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
        assert start_region is not None
        from_start, far_distance = _distances_from(
            env, env.bins_in_region(start_region)
        )
        for run in runs:
            laps.extend(
                _detect_laps_region_contiguous(
                    position_bins[run],
                    times[run],
                    env,
                    start_region=start_region,
                    direction=direction,
                    from_start=from_start,
                    far_distance=far_distance,
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
    start_region: str,
    direction: str,
    from_start: NDArray[np.float64],
    far_distance: float,
) -> list[Lap]:
    """Pair start-region entries within one recording into laps.

    A lap runs from one entry to the next entry that follows a trip to the
    far side of the track: a bin at least ``far_distance`` (half the
    greatest path distance from the start region) away. Re-entries without
    such a trip, such as tracking jitter across the region edge, stay inside
    the lap instead of splitting it.
    """
    crossings = _detect_region_crossings_contiguous(
        position_bins, times, env, region_name=start_region, direction="entry"
    )
    entry_indices = [int(np.searchsorted(times, c.time)) for c in crossings]
    if len(entry_indices) < 2:
        return []

    # Lap boundaries: the first entry, then each entry that ends a far trip.
    boundaries = [entry_indices[0]]
    for previous, entry in itertools.pairwise(entry_indices):
        if from_start[position_bins[previous:entry]].max() >= far_distance:
            boundaries.append(entry)

    laps: list[Lap] = []
    for start_idx, end_idx in itertools.pairwise(boundaries):
        lap_direction = _detect_lap_direction(
            env.bin_centers, position_bins[start_idx:end_idx]
        )
        if direction != "both" and lap_direction != direction:
            continue
        laps.append(
            Lap(
                start_time=float(times[start_idx]),
                end_time=float(times[end_idx]),
                direction=lap_direction,
                overlap_score=np.nan,  # no template to overlap
            )
        )
    return laps


def _distances_from(
    env: Environment, sources: NDArray[np.int_]
) -> tuple[NDArray[np.float64], float]:
    """Path distance from ``sources`` to every bin, and half its finite maximum.

    The second value is the "far side" threshold of the lap detectors: on a
    closed loop of length L the farthest point from any place is L/2 away,
    and every full circuit passes it.
    """
    from neurospatial.ops.distance import distance_field

    distances = distance_field(env.connectivity, list(sources), weight="distance")
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

    The circuit starts at the recording's first sample and ends at the first
    return to that sample's bin (a pass through its start zone) after a trip
    to the far side of the track. None when no recording completes one.
    """
    for run in runs:
        bins = position_bins[run]
        anchor = int(bins[0])
        from_anchor, far_distance = _distances_from(env, np.array([anchor]))
        zone = _start_zone(env, anchor)
        reached_far = np.maximum.accumulate(from_anchor[bins] >= far_distance)
        for stop in _pass_samples(bins, env, anchor, zone)[1:]:
            if reached_far[stop - 1]:
                found: NDArray[np.int64] = bins[:stop]
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
    """Split one recording into laps at passes through the template's start."""
    laps: list[Lap] = []
    boundaries = _pass_samples(position_bins, env, anchor, zone)
    if not boundaries:
        return laps
    lap_start = boundaries[0]
    for candidate in boundaries[1:]:
        lap_bins = position_bins[lap_start:candidate]
        overlap = _widened_overlap(template, lap_bins, adjacency)
        if overlap < min_overlap:
            continue  # a turn back or a partial circuit: keep extending
        lap_direction = _detect_lap_direction(env.bin_centers, lap_bins)
        if direction == "both" or lap_direction == direction:
            laps.append(
                Lap(
                    start_time=float(times[lap_start]),
                    end_time=float(times[candidate]),
                    direction=lap_direction,
                    overlap_score=float(overlap),
                )
            )
        lap_start = candidate
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

    First-class inbound/outbound labeler for linear / W-track tasks, so users no
    longer have to invent end-region names and stitch
    :func:`detect_runs_between_regions` together by hand. Outbound runs go from
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
        If ``start_region`` or any end region is not in ``env.regions``
        (raised by :func:`detect_runs_between_regions`), or if
        ``position_bins`` and ``times`` are not 1-D with the same length,
        times are not finite and strictly increasing, or a bin index is
        neither -1 nor a bin of ``env``.
    TypeError
        If position_bins does not hold integer bin indices.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_runs_between_regions : Underlying source-to-target run detector.
    runs_to_direction_labels : Generic run-to-label bridge.
    goal_pair_direction_labels : Arrow-notation labels for trialized tasks.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

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

    if isinstance(end_regions, str):
        end_region_names: list[str] = [end_regions]
    else:
        end_region_names = list(end_regions)
    _require_region_mask(
        env,
        start_region,
        operation="running_direction_labels()",
        argument="start_region",
    )
    for end_region in end_region_names:
        _require_region_mask(
            env,
            end_region,
            operation="running_direction_labels()",
            argument="end_regions",
            in_list=not isinstance(end_regions, str),
        )
    breaks = validate_break_before(
        break_before, len(times), call="running_direction_labels"
    )

    runs: list[tuple[Run, str]] = []
    for end_region in end_region_names:
        outbound_runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source=start_region,
            target=end_region,
            min_duration=min_duration,
            max_duration=max_duration,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
        )
        inbound_runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source=end_region,
            target=start_region,
            min_duration=min_duration,
            max_duration=max_duration,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
        )
        runs.extend((run, "outbound") for run in outbound_runs)
        runs.extend((run, "inbound") for run in inbound_runs)

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
# Functions from trials.py
# =============================================================================


def segment_trials(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    start_region: str,
    end_regions: list[str],
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
    end_regions : list[str]
        List of end region names (e.g., ['left', 'right'] for T-maze).
        Must all exist in env.regions.
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
        If min_duration is not positive.
    ValueError
        If max_duration < min_duration.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    See Also
    --------
    detect_region_crossings : Detect individual region crossings
    detect_runs_between_regions : Detect runs between two specific regions

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause.

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
    # Validate inputs
    _require_region_mask(
        env, start_region, operation="segment_trials()", argument="start_region"
    )

    if len(end_regions) == 0:
        raise ValueError(
            _format_error(
                "end_regions cannot be empty",
                fix="pass end_regions=['goal']",
                why="Why: each trial must finish at a named end region.",
            )
        )

    for region in end_regions:
        _require_region_mask(
            env,
            region,
            operation="segment_trials()",
            argument="end_regions",
            in_list=True,
        )

    # Prevent start_region from being in end_regions (typical neuroscience practice)
    if start_region in end_regions:
        raise ValueError(
            f"start_region '{start_region}' cannot be in end_regions. "
            f"Start and end regions must be spatially distinct for trial segmentation."
        )

    position_bins, times = _validated_samples(
        position_bins, times, env, call="segment_trials"
    )

    if min_duration <= 0:
        raise ValueError(f"min_duration must be positive, got {min_duration}")

    if max_duration <= min_duration:
        raise ValueError(
            f"max_duration must be greater than min_duration. "
            f"Got max_duration={max_duration}, min_duration={min_duration}"
        )

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
                env,
                start_region=start_region,
                end_regions=end_regions,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        )
    return results


def _segment_trials_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    start_region: str,
    end_regions: list[str],
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[Trial]:
    """Analyze one recording after the public input validation."""
    # Get region masks using existing functionality
    from neurospatial.ops.mapping import regions_to_mask

    start_mask = regions_to_mask(env, [start_region])
    end_masks = {region: regions_to_mask(env, [region]) for region in end_regions}

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
# Functions from similarity.py
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
    stereotyped behaviors, replay analysis, and comparing navigation strategies
    across trials or sessions.

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
          track. Needs a 1-D environment or a linearized track. Range [0, 1].
        - 'hausdorff': Hausdorff distance between paths, normalized. Range [0, 1].
        - 'dtw': Dynamic time warping distance, normalized. Range [0, 1].
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
    coordinate in a 1-D environment. Sensitive to order and timing. A 2-D
    or 3-D path has no single coordinate to correlate; use ``'dtw'`` or
    ``'hausdorff'`` there.

    **Hausdorff distance** measures maximum deviation between trajectories:

    .. math::

        d_H(A, B) = \\max\\{\\sup_{a \\in A} \\inf_{b \\in B} d(a, b),
                            \\sup_{b \\in B} \\inf_{a \\in A} d(a, b)\\}

    Converted to similarity as :math:`1 - d_H / d_{max}` where :math:`d_{max}`
    is the environment extent.

    **Dynamic time warping (DTW)** finds optimal alignment between trajectories
    allowing temporal shifts. Useful for comparing paths with different speeds.

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

    References
    ----------
    .. [1] Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell
           sequences depict future paths to remembered goals. *Nature*,
           497(7447), 74-79.
    .. [2] Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009).
           Hippocampal replay of extended experience. *Neuron*, 63(4), 497-507.
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

        # Normalize by path lengths
        # Theoretical max DTW = max(len1, len2) * max_bin_distance
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

    Parameters
    ----------
    seq1 : NDArray[np.float64], shape (n1, n_dims)
        First sequence of positions.
    seq2 : NDArray[np.float64], shape (n2, n_dims)
        Second sequence of positions.

    Returns
    -------
    float
        DTW distance.
    """
    n1, n2 = len(seq1), len(seq2)

    # Initialize cost matrix
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = euclidean(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n1, n2])


def detect_goal_directed_runs(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    goal_region: str,
    directedness_threshold: float = 0.7,
    min_progress: float = 20.0,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Detect goal-directed runs in a trajectory.

    Identifies segments where the animal moves efficiently toward a goal region.
    Useful for analyzing goal-directed replay, learning dynamics, and navigation
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
        Minimum directedness score to classify as goal-directed. Range [0, 1].
        Higher values require more efficient paths. Default is 0.7.
    min_progress : float, optional
        Minimum distance progress toward goal (physical units). Filters out
        short runs with minimal displacement. Default is 20.0.
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
        Goal-directed stretches as ``[start, stop)`` rows in seconds, in time
        order. A row means the directedness and net-progress criteria were
        met; it does not require reaching the goal region. Pass the rows as
        ``epochs=`` to restrict another analysis to them.

    Raises
    ------
    ValueError
        If goal_region not in env.regions, or if directedness_threshold
        not in [0, 1].
    ValueError
        If min_progress is negative.
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
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no segment spans a pause. A sample
    outside the environment (bin ``-1``) is missing and skipped as if its row
    were deleted; the gap around it counts toward ``max_gap``.

    Each recording supplies one candidate spanning its complete observed
    sample run; this function does not split repeated approaches into separate
    bouts. A returned row means the directedness and net-progress thresholds
    were met, even when the last sample is still outside the goal. To segment
    completed source-to-goal traversals, use ``detect_runs_between_regions``.

    **Directedness score** measures path efficiency:

    .. math::

        D = \\frac{d_{start} - d_{end}}{L_{path}}

    where :math:`d_{start}` and :math:`d_{end}` are distances to goal at
    trajectory start and end, and :math:`L_{path}` is total path length.

    A straight path toward goal has :math:`D \\approx 1.0`, while a random
    walk has :math:`D \\approx 0.0`. Negative values indicate movement away
    from goal.

    This metric is used to identify replay events where hippocampal place cells
    reactivate in sequences representing paths to remembered goals [1]_.

    Examples
    --------
    Detect goal-directed runs in a linear track:

    >>> import numpy as np  # doctest: +SKIP
    >>> from shapely.geometry import Point  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.behavior.segmentation import (
    ...     detect_goal_directed_runs,
    ... )  # doctest: +SKIP
    >>> positions = np.linspace(0, 100, 100)[:, None]  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>> goal_center = env.bin_centers[-1]  # doctest: +SKIP
    >>> goal_polygon = Point(float(goal_center[0]), 0.0).buffer(5.0)  # doctest: +SKIP
    >>> from neurospatial.regions import Regions  # doctest: +SKIP
    >>> regions = Regions(env.regions.values())  # doctest: +SKIP
    >>> regions.add("goal", polygon=goal_polygon)  # doctest: +SKIP
    >>> env = env.with_regions(regions)  # doctest: +SKIP
    >>> position_bins = np.arange(0, 40, dtype=np.int64)  # doctest: +SKIP
    >>> times = np.linspace(0, 10, len(position_bins))  # doctest: +SKIP
    >>> runs = detect_goal_directed_runs(  # doctest: +SKIP
    ...     position_bins,  # doctest: +SKIP
    ...     times,  # doctest: +SKIP
    ...     env,  # doctest: +SKIP
    ...     goal_region="goal",  # doctest: +SKIP
    ...     directedness_threshold=0.7,  # doctest: +SKIP
    ...     min_progress=10.0,  # doctest: +SKIP
    ... )  # doctest: +SKIP
    >>> print(f"Detected {len(runs)} goal-directed run(s)")  # doctest: +SKIP
    Detected 1 goal-directed run(s)

    References
    ----------
    .. [1] Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell
           sequences depict future paths to remembered goals. *Nature*,
           497(7447), 74-79.
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

    if min_progress < 0:
        raise ValueError(f"min_progress must be non-negative, got {min_progress}")

    position_bins, times = _validated_samples(
        position_bins, times, env, call="detect_goal_directed_runs"
    )
    breaks = validate_break_before(
        break_before, len(times), call="detect_goal_directed_runs"
    )

    from neurospatial.ops.distance import distance_field
    from neurospatial.ops.mapping import regions_to_mask

    goal_bin_indices = np.flatnonzero(regions_to_mask(env, [goal_region]))
    # Compute distance from each bin to nearest goal bin using graph distance,
    # once for every run. A single multi-source Dijkstra (via distance_field)
    # computes, for every bin, the shortest-path distance to the nearest goal
    # bin. This is equivalent to (but far cheaper than) looping over every
    # (bin, goal_bin) pair with nx.shortest_path_length; unreachable bins
    # remain np.inf.
    distances_to_goal = distance_field(
        env.connectivity, list(goal_bin_indices), weight="distance"
    )
    # A recording continues across bridged gaps between samples inside the
    # environment.
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
                env,
                distances_to_goal=distances_to_goal,
                directedness_threshold=directedness_threshold,
                min_progress=min_progress,
            )
        )
    return np.array(results, dtype=np.float64).reshape(-1, 2)


def _detect_goal_directed_runs_contiguous(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    distances_to_goal: NDArray[np.float64],
    directedness_threshold: float = 0.7,
    min_progress: float = 20.0,
) -> list[tuple[float, float]]:
    """Analyze one recording after the public input validation."""
    # Get start and end positions
    start_bin = position_bins[0]
    end_bin = position_bins[-1]

    d_start = distances_to_goal[start_bin]
    d_end = distances_to_goal[end_bin]

    # Check if start or end can reach goal
    # If either is infinite, cannot compute meaningful directedness
    if np.isinf(d_start) or np.isinf(d_end):
        # Cannot reach goal from start or end position
        return []

    # Compute path length (sum of graph distances between consecutive bins).
    # Adjacent bins (the common case) are resolved with a direct edge-weight
    # lookup; non-adjacent consecutive samples fall back to the geodesic
    # shortest-path length, preserving the original semantics. Disconnected
    # consecutive pairs contribute nothing (skipped).
    path_length = 0.0
    for src, dst in itertools.pairwise(position_bins):
        src_i, dst_i = int(src), int(dst)
        if src_i == dst_i:
            continue
        edge_data = env.connectivity.get_edge_data(src_i, dst_i)
        if edge_data is not None:
            path_length += edge_data["distance"]
            continue
        try:
            path_length += nx.shortest_path_length(
                env.connectivity, src_i, dst_i, weight="distance"
            )
        except nx.NetworkXNoPath:
            # Disconnected bins - skip this segment
            continue

    # Compute directedness
    directedness = 0.0 if path_length == 0 else (d_start - d_end) / path_length

    # Compute progress toward goal
    progress = d_start - d_end

    # Check if run meets criteria
    if directedness >= directedness_threshold and progress >= min_progress:
        return [(float(times[0]), float(times[-1]))]
    return []


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
