"""Decision analysis at choice points.

This module provides tools for analyzing animal behavior at choice points,
including pre-decision kinematics, decision boundary analysis, and Voronoi
labeling of goal regions.

Imports
-------
All functions are importable from `behavior.decisions`:

    from neurospatial.behavior.decisions import (
        # Decision analysis
        PreDecisionMetrics,
        DecisionBoundaryMetrics,
        DecisionAnalysisResult,
        compute_decision_analysis,
        compute_pre_decision_metrics,
        decision_region_entry_time,
        detect_boundary_crossings,
        distance_to_decision_boundary,
        extract_pre_decision_window,
        geodesic_voronoi_labels,
        pre_decision_heading_stats,
        pre_decision_speed_stats,
    )

Key Concepts
------------
Decision Analysis
~~~~~~~~~~~~~~~~~
- **Decision region**: Spatial zone where animal must choose (e.g., junction)
- **Pre-decision window**: Time window before entering decision region
- **Decision boundary**: Geodesic Voronoi boundary between potential goals
- **Boundary crossing**: When animal's nearest goal changes

When to Use
-----------
Use this module when analyzing:
- T-maze or Y-maze choice behavior
- Multi-goal navigation (spatial bandit tasks)
- Decision dynamics and commitment points

For VTE (Vicarious Trial and Error) analysis, see :mod:`neurospatial.behavior.vte`.

Example
-------
Decision analysis:

>>> from neurospatial.behavior import compute_decision_analysis
>>> result = compute_decision_analysis(
...     env,
...     times,
...     positions,
...     decision_region="center",
...     goal_regions=["left", "right"],
...     pre_window=1.0,
... )  # doctest: +SKIP
>>> if result.pre_decision.heading_circular_variance > 0.5:
...     print("High heading variability - possible deliberation")  # doctest: +SKIP

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import run_sample_bounds
from neurospatial._observation import (
    ObservationSupport,
    validated_observations,
    validated_signals,
)
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING, smoothed_speed
from neurospatial.behavior._region_bins import region_representative_bin
from neurospatial.behavior.segmentation import _safe_gather
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    # Decision analysis dataclasses
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
    "PreDecisionMetrics",
    # Decision analysis functions
    "compute_decision_analysis",
    "compute_pre_decision_metrics",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "geodesic_voronoi_labels",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
]


# =============================================================================
# Decision Analysis Data Structures
# =============================================================================


@dataclass(frozen=True)
class PreDecisionMetrics:
    """Metrics from the pre-decision window.

    The pre-decision window is the time period immediately before the animal
    enters a decision region (e.g., 1 second before entering a T-junction).

    Attributes
    ----------
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
        Low values may indicate pausing/deliberation.
    min_speed : float
        Minimum speed in window (units/s).
        Near-zero indicates a pause, common during VTE.
    heading_mean_direction : float
        Circular mean of heading direction (radians, -pi to pi).
    heading_circular_variance : float
        Circular variance of heading, range [0, 1].
        High values (> 0.5) indicate variable heading - possible scanning.
        Low values indicate stable, consistent heading.
    heading_mean_resultant_length : float
        Concentration of heading distribution, range [0, 1].
        Inverse of variance: high = concentrated, low = dispersed.
    window_duration : float
        Time from the first to the last observed sample in the window
        (seconds); 0.0 with fewer than two. Shorter than requested when the
        trajectory, or its time segment, starts late.
    n_samples : int
        Number of observed (finite) samples in the window; missing rows are
        not counted.
    is_valid : bool
        Whether observed movement and heading support interpretation of these
        metrics. Missing or insufficient observations have NaN metrics.

    Examples
    --------
    A 10 cm/s eastward approach with two lost frames in the 1 s window. Only
    observed samples count, and the duration runs from the first to the last
    of them:

    >>> import numpy as np
    >>> from neurospatial.behavior import compute_pre_decision_metrics
    >>> times = np.arange(40) / 10  # s
    >>> positions = np.column_stack([10 * times, np.zeros(40)])  # cm
    >>> positions[[22, 25]] = np.nan
    >>> metrics = compute_pre_decision_metrics(
    ...     times, positions, entry_time=3.0, window_duration=1.0, min_speed=5.0
    ... )
    >>> metrics.n_samples
    8
    >>> round(metrics.window_duration, 6)
    0.9
    >>> metrics.is_valid
    True
    """

    mean_speed: float
    min_speed: float
    heading_mean_direction: float
    heading_circular_variance: float
    heading_mean_resultant_length: float
    window_duration: float
    n_samples: int

    @property
    def is_valid(self) -> bool:
        """Whether enough observations support speed and heading metrics."""
        return bool(
            self.n_samples >= 2
            and self.window_duration > 0
            and np.isfinite(self.mean_speed)
            and np.isfinite(self.heading_circular_variance)
        )

    def suggests_deliberation(
        self,
        variance_threshold: float = 0.5,
        min_speed: float = 10.0,
    ) -> bool:
        """Check if metrics suggest deliberative behavior.

        Parameters
        ----------
        variance_threshold : float, default=0.5
            Circular variance above this suggests head scanning.
        min_speed : float, default=10.0
            Mean speed below this (units/s) suggests slowing.

        Returns
        -------
        bool
            True if valid observations show high heading variance AND low
            speed. False for missing evidence; inspect ``is_valid`` to
            distinguish missing evidence from an observed negative result.
        """
        return (
            self.is_valid
            and self.heading_circular_variance > variance_threshold
            and self.mean_speed < min_speed
        )


@dataclass(frozen=True, repr=False)
class DecisionBoundaryMetrics(ReadOnlyArraysMixin, ResultMixin):
    """Metrics related to decision boundaries between goals.

    Decision boundaries are the geodesic Voronoi edges between goal regions.
    An animal at the boundary is equidistant from multiple goals.

    Visualization (2-goal example)::

             Goal A          Goal B
               *               *
               |               |
               |    Boundary   |
               |       |       |
               |       |       |
               +-------+-------+
                       ^
                Decision point

    Attributes
    ----------
    goal_labels : NDArray[np.int_]
        Per-timepoint label of nearest goal (Voronoi region), shape (n_samples,).
        Values are indices into the goal_bins list.
    distance_to_boundary : NDArray[np.float64]
        Distance to nearest decision boundary at each timepoint, shape (n_samples,).
        Units match environment. Small values = near boundary = uncommitted.
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary.
    crossing_directions : list[tuple[int, int]]
        (from_goal_idx, to_goal_idx) for each crossing.
        Indicates which goal regions the animal moved between.
    """

    goal_labels: NDArray[np.int_]
    distance_to_boundary: NDArray[np.float64]
    crossing_times: list[float]
    crossing_directions: list[tuple[int, int]]

    def __post_init__(self) -> None:
        """Own read-only copies of goal labels and boundary distances."""
        self._own_arrays()

    @property
    def n_crossings(self) -> int:
        """Number of decision boundary crossings."""
        return len(self.crossing_times)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar decision-boundary metrics.

        For a human-readable string use ``str(result)``.
        """
        return {
            "n_crossings": self.n_crossings,
            "mean_distance_to_boundary": float(np.nanmean(self.distance_to_boundary)),
        }

    def __str__(self) -> str:
        return (
            f"Decision boundary: {self.n_crossings} crossing"
            f"{'s' if self.n_crossings != 1 else ''}, "
            f"mean distance to boundary: {np.nanmean(self.distance_to_boundary):.1f}"
        )


@dataclass(frozen=True, repr=False)
class DecisionAnalysisResult(ResultMixin):
    """Complete decision analysis for a trial.

    Attributes
    ----------
    entry_time : float
        Time of decision region entry (seconds).
    pre_decision : PreDecisionMetrics
        Metrics from pre-decision window.
    boundary : DecisionBoundaryMetrics or None
        Boundary metrics. None if only one goal (no boundary defined).
    chosen_goal : int or None
        Index of goal reached. None if trial timed out or goal not reached.
    """

    entry_time: float
    pre_decision: PreDecisionMetrics
    boundary: DecisionBoundaryMetrics | None
    chosen_goal: int | None

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar decision-analysis metrics.

        For a human-readable multi-line string use ``str(result)``.
        """
        return {
            "entry_time": self.entry_time,
            "pre_decision_mean_speed": self.pre_decision.mean_speed,
            "pre_decision_heading_variance": (
                self.pre_decision.heading_circular_variance
            ),
            "n_boundary_crossings": (
                self.boundary.n_crossings if self.boundary is not None else None
            ),
            "chosen_goal": self.chosen_goal,
        }

    def __str__(self) -> str:
        lines = [
            "Decision analysis:",
            f"  Entry time: {self.entry_time:.2f} s",
            f"  Pre-decision: speed={self.pre_decision.mean_speed:.1f}, "
            f"heading_var={self.pre_decision.heading_circular_variance:.2f}",
        ]
        if self.boundary is not None:
            lines.append(f"  {self.boundary}")
        if self.chosen_goal is not None:
            lines.append(f"  Chosen goal: {self.chosen_goal}")
        else:
            lines.append("  Chosen goal: none (timeout)")
        return "\n".join(lines)


# =============================================================================
# Pre-Decision Window Functions
# =============================================================================


def decision_region_entry_time(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    region: str,
) -> float:
    """Find time of first entry to a decision region.

    Parameters
    ----------
    position_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to position bins (seconds).
    env : Environment
        Environment with region definitions.
    region : str
        Name of decision region in env.regions.

    Returns
    -------
    float
        Time of first entry to the decision region (seconds).

    Raises
    ------
    ValueError
        If region not found in env.regions.
    ValueError
        If trajectory never enters the region.

    Examples
    --------
    >>> entry_time = decision_region_entry_time(
    ...     position_bins, times, env, "center"
    ... )  # doctest: +SKIP
    >>> print(f"Entered decision region at t={entry_time:.2f}s")  # doctest: +SKIP
    """
    _require_region_mask(
        env, region, operation="decision_region_entry_time()", argument="region"
    )

    # Get bins in the region
    region_bins = env.bins_in_region(region)
    region_bin_set = set(region_bins)

    # Find first entry
    for i, bin_idx in enumerate(position_bins):
        if bin_idx in region_bin_set:
            return float(times[i])

    raise ValueError(
        f"Trajectory never enters region '{region}'. "
        f"Check that the trajectory passes through the decision region, "
        f"or verify region bounds with env.regions['{region}']."
    )


def extract_pre_decision_window(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    entry_time: float,
    *,
    window_duration: float,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract trajectory segment before decision region entry.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps in seconds; finite and strictly increasing
        (a repeated timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions. A row with a non-finite coordinate (NaN, for
        a lost frame) is a missing sample.
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to extract (seconds).
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
        time. Entry 0 is ignored. The window never reaches back across a
        declared break.

    Returns
    -------
    window_times : NDArray[np.float64], shape (n_window,)
        Times within the pre-decision window.
    window_positions : NDArray[np.float64], shape (n_window, n_dims)
        Positions within the pre-decision window, NaN rows included. 1-D
        positions are returned with shape ``(n_window, 1)``.

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, or if ``break_before`` is not a Boolean array with one
        entry per sample.

    Notes
    -----
    The recording is split into time segments at gaps longer than
    ``max_gap``, at ``epochs`` edges and at declared breaks. The window holds
    the samples with ``max(entry_time - window_duration, segment_start) <= t
    < entry_time`` in the segment containing ``entry_time``, so it never
    extends across a pause and is shorter than requested when the segment
    starts late. If ``entry_time`` lies in no segment, the window is empty.

    A missing row inside the window stays in it as a missing row instead of
    cutting the window short: a lost frame shortly before entry does not
    shrink the window. Statistics computed from the window use only the
    observed samples.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import extract_pre_decision_window
    >>> times = np.arange(10) / 2  # s
    >>> positions = np.column_stack([10 * times, np.zeros(10)])  # cm
    >>> positions[6] = np.nan  # a lost frame
    >>> window_times, window_positions = extract_pre_decision_window(
    ...     times, positions, entry_time=4.0, window_duration=2.0
    ... )
    >>> window_times.tolist()
    [2.0, 2.5, 3.0, 3.5]
    >>> window_positions[:, 0].tolist()
    [20.0, 25.0, nan, 35.0]

    A new recording starting at 3.0 s ends the window there:

    >>> break_before = times == 3.0
    >>> extract_pre_decision_window(
    ...     times, positions, 4.0, window_duration=2.0, break_before=break_before
    ... )[0].tolist()
    [3.0, 3.5]
    """
    times, positions, support = validated_observations(
        times,
        positions,
        call="extract_pre_decision_window",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no pre-decision window was observed",
    )
    window = _pre_decision_window(
        times, support.interval_gates, entry_time, window_duration
    )
    return times[window], positions[window]


def _pre_decision_window(
    times: NDArray[np.float64],
    interval_gates: NDArray[np.bool_],
    entry_time: float,
    window_duration: float,
) -> slice:
    """Sample slice of the pre-decision window; empty if entry is in no segment.

    Segments are runs of ``interval_gates`` (``max_gap``, ``epochs`` and
    declared breaks), whatever the validity of their samples.
    """
    for first, last in run_sample_bounds(interval_gates):
        if times[first] <= entry_time <= times[last]:
            window_start = max(entry_time - window_duration, times[first])
            segment = times[first : last + 1]
            start = int(first + np.searchsorted(segment, window_start, side="left"))
            stop = int(first + np.searchsorted(segment, entry_time, side="left"))
            return slice(start, stop)
    return slice(0, 0)


def pre_decision_heading_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> tuple[float, float, float]:
    """Compute circular statistics on heading in a trajectory window.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates, ``n_dims >= 2``; heading uses x and y. A row
        with a non-finite coordinate (NaN, for a lost frame) is a missing
        sample: it has no heading, and no velocity is computed across it.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit.
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded from statistics.
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
        time. Entry 0 is ignored. No velocity spans a declared break.

    Returns
    -------
    mean_direction : float
        Circular mean heading in radians (-pi to pi).
    circular_variance : float
        Circular variance, range [0, 1].
        0 = all headings identical, 1 = uniform distribution.
    mean_resultant_length : float
        Mean resultant length, range [0, 1].
        1 = all headings identical, 0 = uniform distribution.

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, if ``break_before`` is not a Boolean array with one entry
        per sample, or if positions has fewer than two coordinates (x and y).

    Notes
    -----
    Each run of observed samples with gaps no longer than ``max_gap`` (inside
    ``epochs``, not crossing a missing row or a ``break_before``) is analyzed
    as a separate recording; no velocity or heading spans a pause.

    Circular statistics are computed directly:

    - mean_resultant_length = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2)
    - circular_variance = 1 - mean_resultant_length
    - mean_direction = atan2(mean(sin(theta)), mean(cos(theta)))

    If no movement provides a valid heading (including stationary samples
    below min_speed), all three statistics are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import pre_decision_heading_stats

    A 10 cm/s eastward run with one lost frame: the missing row has no
    heading, and every observed heading points East.

    >>> times = np.arange(10) / 10  # s
    >>> positions = np.column_stack([10 * times, np.zeros(10)])  # cm
    >>> positions[4] = np.nan
    >>> mean_dir, circ_var, mrl = pre_decision_heading_stats(
    ...     times, positions, min_speed=1.0
    ... )
    >>> float(mean_dir), float(circ_var), float(mrl)
    (0.0, 0.0, 1.0)

    A still animal provides no heading:

    >>> pre_decision_heading_stats(times, np.ones((10, 2)), min_speed=1.0)
    (nan, nan, nan)
    """
    from neurospatial.behavior._kinematics import _validate_velocity_positions

    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="pre_decision_heading_stats",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the heading statistics are NaN",
    )
    _validate_velocity_positions(raw_positions, call="pre_decision_heading_stats")
    return _heading_stats(times, positions, support.interval_mask, min_speed=min_speed)


def _heading_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    min_speed: float,
) -> tuple[float, float, float]:
    """Circular heading statistics over moving samples in usable intervals.

    ``positions`` are already validated by the public caller.
    """
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    if len(positions) < 2:
        return np.nan, np.nan, np.nan
    headings, speed = _movement_heading_and_speed(
        times, positions, interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
    )
    valid_headings = headings[
        (speed >= min_speed) & (speed > 0) & np.isfinite(headings)
    ]

    if len(valid_headings) == 0:
        return np.nan, np.nan, np.nan

    # Compute circular statistics directly
    cos_headings = np.cos(valid_headings)
    sin_headings = np.sin(valid_headings)
    mean_cos = float(np.mean(cos_headings))
    mean_sin = float(np.mean(sin_headings))

    mean_resultant_length = float(np.sqrt(mean_cos**2 + mean_sin**2))
    circular_variance = 1.0 - mean_resultant_length
    mean_direction = float(np.arctan2(mean_sin, mean_cos))

    return mean_direction, circular_variance, mean_resultant_length


def pre_decision_speed_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> tuple[float, float]:
    """Compute speed statistics for a trajectory window.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims) or (n_samples,)
        Position coordinates. A row with a non-finite coordinate (NaN, for a
        lost frame) is a missing sample: it has no speed, and no velocity is
        computed across it.
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
        time. Entry 0 is ignored. No velocity spans a declared break.

    Returns
    -------
    mean_speed : float
        Mean instantaneous speed (units/s).
    min_speed : float
        Minimum instantaneous speed (units/s).

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, or if ``break_before`` is not a Boolean array with one
        entry per sample.

    Notes
    -----
    Each run of observed samples with gaps no longer than ``max_gap`` (inside
    ``epochs``, not crossing a missing row or a ``break_before``) is analyzed
    as a separate recording; no velocity spans a pause.

    Only observed speeds enter the reductions. Both results are NaN when no
    interval is usable, including a single sample.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import pre_decision_speed_stats

    Two 10 cm/s runs recorded 50 cm apart and joined on one clock. Undeclared,
    the jump between them reads as a burst of speed; declared with
    ``break_before``, it does not.

    >>> times = np.arange(20) / 10  # s
    >>> x = np.r_[10 * times[:10], 10 * times[10:] + 50.0]  # cm
    >>> positions = np.column_stack([x, np.zeros(20)])
    >>> new_recording = times == 1.0
    >>> mean_speed, min_speed = pre_decision_speed_stats(
    ...     times, positions, break_before=new_recording
    ... )
    >>> round(mean_speed, 6), round(min_speed, 6)
    (10.0, 10.0)
    >>> round(pre_decision_speed_stats(times, positions)[0], 1)
    35.0
    """
    times, positions, support = validated_observations(
        times,
        positions,
        call="pre_decision_speed_stats",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the speed statistics are NaN",
    )
    return _speed_stats(times, positions, support.interval_mask)


def _speed_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Mean and minimum smoothed speed over usable intervals; NaN without any."""
    if not interval_mask.any():
        return np.nan, np.nan
    speeds = smoothed_speed(
        times, positions, interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
    )
    return float(np.nanmean(speeds)), float(np.nanmin(speeds))


def compute_pre_decision_metrics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    entry_time: float,
    *,
    window_duration: float,
    min_speed: float,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> PreDecisionMetrics:
    """Compute all pre-decision window metrics.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps in seconds; finite and strictly increasing
        (a repeated timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions, ``n_dims >= 2``. A row with a non-finite
        coordinate (NaN, for a lost frame) is a missing sample.
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to analyze (seconds).
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit.
        Minimum speed for valid heading (units/s).
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
        time. Entry 0 is ignored. The window never reaches back across a
        declared break.

    Returns
    -------
    PreDecisionMetrics
        Dataclass containing all pre-decision metrics.

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, or if ``break_before`` is not a Boolean array with one
        entry per sample.

    Notes
    -----
    The window is the one :func:`extract_pre_decision_window` returns: it lies
    in the time segment containing ``entry_time`` and never spans a pause, an
    ``epochs`` edge or a declared break. Missing rows inside it stay missing:
    speed and heading use only the observed samples, ``n_samples`` counts
    them, and ``window_duration`` runs from the first to the last of them.
    With fewer than two observed samples every metric is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import compute_pre_decision_metrics

    A 10 cm/s eastward approach with two lost frames in the 1 s window:

    >>> times = np.arange(40) / 10  # s
    >>> positions = np.column_stack([10 * times, np.zeros(40)])  # cm
    >>> positions[[22, 25]] = np.nan
    >>> metrics = compute_pre_decision_metrics(
    ...     times, positions, entry_time=3.0, window_duration=1.0, min_speed=5.0
    ... )
    >>> metrics.n_samples
    8
    >>> round(metrics.window_duration, 6), round(metrics.mean_speed, 6)
    (0.9, 10.0)
    >>> round(metrics.heading_circular_variance, 6)
    0.0
    """
    from neurospatial.behavior._kinematics import _validate_velocity_positions

    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_pre_decision_metrics",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the pre-decision metrics are NaN",
    )
    _validate_velocity_positions(raw_positions, call="compute_pre_decision_metrics")
    return _pre_decision_metrics(
        times,
        positions,
        support,
        entry_time,
        window_duration,
        min_speed=min_speed,
    )


def _pre_decision_metrics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float,
) -> PreDecisionMetrics:
    """Speed and heading metrics of the pre-decision window under ``support``."""
    window = _pre_decision_window(
        times, support.interval_gates, entry_time, window_duration
    )
    window_times = times[window]
    observed_times = window_times[support.sample_valid[window]]
    if observed_times.size < 2:
        return PreDecisionMetrics(
            mean_speed=np.nan,
            min_speed=np.nan,
            heading_mean_direction=np.nan,
            heading_circular_variance=np.nan,
            heading_mean_resultant_length=np.nan,
            window_duration=0.0,
            n_samples=int(observed_times.size),
        )
    # The window lies inside one gate segment, so its usable intervals are
    # exactly the run's usable intervals between its first and last sample.
    window_mask = support.interval_mask[window.start : window.stop - 1]
    window_positions = positions[window]
    mean_dir, circ_var, mrl = _heading_stats(
        window_times, window_positions, window_mask, min_speed=min_speed
    )
    mean_speed, min_speed_val = _speed_stats(
        window_times, window_positions, window_mask
    )
    return PreDecisionMetrics(
        mean_speed=mean_speed,
        min_speed=min_speed_val,
        heading_mean_direction=mean_dir,
        heading_circular_variance=circ_var,
        heading_mean_resultant_length=mrl,
        window_duration=float(observed_times[-1] - observed_times[0]),
        n_samples=int(observed_times.size),
    )


# =============================================================================
# Decision Boundary Functions
# =============================================================================


def geodesic_voronoi_labels(
    env: Environment,
    goal_bins: list[int] | NDArray[np.int_],
) -> NDArray[np.int_]:
    """Label each bin by its nearest goal using geodesic distance.

    Creates a Voronoi-like partition of the environment where each bin
    is assigned to the goal with the shortest geodesic path.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goal_bins : list[int] or NDArray[np.int_]
        Bin indices of goal locations.

    Returns
    -------
    NDArray[np.int_], shape (n_bins,)
        Index of nearest goal for each bin (indices into goal_bins).
        Bins unreachable from all goals have label -1.

    Notes
    -----
    Performance: O(n_goals * n_bins * log(n_bins)) using Dijkstra's algorithm.
    For large environments (n_bins > 5000) with many goals (> 10), this may
    take several seconds. Consider caching results if calling repeatedly.

    Examples
    --------
    >>> left_bin = int(env.bin_at([[10, 55]])[0])  # doctest: +SKIP
    >>> right_bin = int(env.bin_at([[90, 55]])[0])  # doctest: +SKIP
    >>> labels = geodesic_voronoi_labels(env, [left_bin, right_bin])  # doctest: +SKIP
    >>> # labels[i] == 0 means bin i is closer to left goal
    >>> # labels[i] == 1 means bin i is closer to right goal
    """
    from neurospatial.environment import _validate_target_bins
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = _validate_target_bins(
        goal_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="geodesic_voronoi_labels()",
        argument="goal_bins",
    )
    n_goals = len(goal_bins_arr)
    n_bins = env.n_bins

    # Compute distance field from each goal
    distances = np.full((n_goals, n_bins), np.inf)
    for i, goal_bin in enumerate(goal_bins_arr):
        distances[i] = distance_field(env.connectivity, [goal_bin], metric="geodesic")

    # Label by nearest goal
    labels = np.argmin(distances, axis=0)

    # Mark unreachable bins
    min_distances = np.min(distances, axis=0)
    labels_int: NDArray[np.int_] = labels.astype(np.int_)
    labels_int[np.isinf(min_distances)] = -1

    return labels_int


def distance_to_decision_boundary(
    env: Environment,
    position_bins: NDArray[np.int_],
    goal_bins: list[int] | NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute distance to nearest decision boundary for each trajectory point.

    The decision boundary is the Voronoi edge between goal regions - the set
    of points equidistant (geodesically) from multiple goals.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along the trajectory.
    goal_bins : list[int] or NDArray[np.int_]
        Bin indices of goal locations.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance to nearest decision boundary at each trajectory point.
        Units match environment. Small values = near boundary = uncommitted.

    Notes
    -----
    Distance to boundary is computed as the absolute difference between
    distances to the two nearest goals. At the boundary, this difference is 0.

    Examples
    --------
    >>> distances = distance_to_decision_boundary(
    ...     env, position_bins, goal_bins
    ... )  # doctest: +SKIP
    >>> commitment_mask = distances > 20.0  # Committed to a goal  # doctest: +SKIP
    """
    from neurospatial.environment import _validate_target_bins
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = _validate_target_bins(
        goal_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="distance_to_decision_boundary()",
        argument="goal_bins",
    )
    n_goals = len(goal_bins_arr)

    if n_goals < 2:
        # With single goal, there's no boundary - return infinity
        return np.full(len(position_bins), np.inf)

    # Compute distance from each goal to all bins
    all_distances = np.zeros((n_goals, env.n_bins))
    for i, goal_bin in enumerate(goal_bins_arr):
        all_distances[i] = distance_field(
            env.connectivity, [goal_bin], metric="geodesic"
        )

    # For each trajectory bin, the boundary distance is the gap between the two
    # nearest goals: d2 - d1 after sorting the per-bin goal distances. This is
    # computed in one vectorized pass over the trajectory.
    position_bins = np.asarray(position_bins).ravel()
    boundary_distances = np.full(len(position_bins), np.nan)

    # Out-of-bounds (including negative) trajectory bins map to NaN; only valid
    # bins are indexed into the goal-distance table to avoid fancy-index wrap.
    valid = (position_bins >= 0) & (position_bins < env.n_bins)
    if not np.any(valid):
        return boundary_distances

    valid_bins = position_bins[valid]

    # all_distances[:, valid_bins] -> (n_goals, n_valid); sort goals per sample.
    sorted_dists = np.sort(all_distances[:, valid_bins], axis=0)
    nearest = sorted_dists[0]
    second = sorted_dists[1]

    # When the nearest goal is unreachable (inf), the boundary distance is inf;
    # otherwise it is the gap to the second-nearest goal. Compute the gap only
    # where the nearest goal is finite to avoid inf - inf = nan.
    finite_nearest = np.isfinite(nearest)
    valid_boundary = np.full(valid_bins.shape, np.inf)
    valid_boundary[finite_nearest] = second[finite_nearest] - nearest[finite_nearest]
    boundary_distances[valid] = valid_boundary

    return boundary_distances


def detect_boundary_crossings(
    position_bins: NDArray[np.int_],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> tuple[list[float], list[tuple[int, int]]]:
    """Detect when trajectory crosses decision boundaries.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along the trajectory; -1 marks a missing or out-of-bounds
        position.
    voronoi_labels : NDArray[np.int_], shape (n_bins,)
        Voronoi label for each bin (from geodesic_voronoi_labels).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
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
        time. Entry 0 is ignored. No crossing spans a declared break.

    Returns
    -------
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary: the midpoint of
        the two samples on either side.
    crossing_directions : list[tuple[int, int]]
        (from_goal_idx, to_goal_idx) for each crossing.

    Raises
    ------
    ValueError
        If ``position_bins`` and ``times`` differ in length, if there are no
        samples, if ``times`` is not finite and strictly increasing, or if
        ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    Each time segment with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording; no
    crossing spans a pause. A crossing needs two consecutive samples with
    reachable labels, so a sample at bin -1 or with an unreachable label never
    forms one.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import detect_boundary_crossings
    >>> voronoi_labels = np.array([0, 1])  # bin 0 is nearer goal 0, bin 1 goal 1
    >>> position_bins = np.array([0, 0, 1, 1])
    >>> times = np.arange(4) / 8  # s
    >>> detect_boundary_crossings(position_bins, voronoi_labels, times)
    ([0.1875], [(0, 1)])

    The same samples from two recordings joined on one clock, where the
    second recording starts in bin 1, have no crossing:

    >>> new_recording = np.array([False, False, True, False])
    >>> detect_boundary_crossings(
    ...     position_bins, voronoi_labels, times, break_before=new_recording
    ... )
    ([], [])
    """
    position_bins = np.asarray(position_bins, dtype=np.int64)
    n_times = len(np.asarray(times))
    if len(position_bins) != n_times:
        raise ValueError(
            f"position_bins and times must have same length; got "
            f"{len(position_bins)} and {n_times}.\n"
            "Why: each trajectory sample needs its corresponding timestamp.\n"
            "Fix: pass one timestamp per position bin on the same clock."
        )
    times, _, supports = validated_signals(
        times,
        position_bins,
        valid=None,
        call="detect_boundary_crossings",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no boundary crossing was observed",
    )
    return _boundary_crossings(
        position_bins, voronoi_labels, times, supports[0].interval_gates
    )


def _boundary_crossings(
    position_bins: NDArray[np.int64],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
    interval_gates: NDArray[np.bool_],
) -> tuple[list[float], list[tuple[int, int]]]:
    """Boundary crossings within each run of ``interval_gates``."""
    crossing_times: list[float] = []
    crossing_directions: list[tuple[int, int]] = []
    for first, last in run_sample_bounds(interval_gates):
        run = slice(int(first), int(last) + 1)
        run_times, run_directions = _detect_boundary_crossings_contiguous(
            position_bins[run], voronoi_labels, times[run]
        )
        crossing_times.extend(run_times)
        crossing_directions.extend(run_directions)
    return crossing_times, crossing_directions


def _detect_boundary_crossings_contiguous(
    position_bins: NDArray[np.int_],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
) -> tuple[list[float], list[tuple[int, int]]]:
    """Detect label changes within one recording with labels indexed by bin."""
    # Get label for each trajectory point
    trajectory_labels = _safe_gather(voronoi_labels, position_bins, fill=-1)

    # Vectorized crossing detection
    # Find where labels change between consecutive frames
    prev_labels = trajectory_labels[:-1]
    curr_labels = trajectory_labels[1:]

    # Valid crossings: both labels are reachable (!= -1) and different
    valid_mask = (
        (prev_labels != -1) & (curr_labels != -1) & (prev_labels != curr_labels)
    )
    crossing_indices = np.where(valid_mask)[0]

    # Compute crossing times (midpoint between frames)
    crossing_times_arr = (times[crossing_indices] + times[crossing_indices + 1]) / 2
    crossing_times = crossing_times_arr.tolist()

    # Get crossing directions
    crossing_directions = [
        (int(prev_labels[idx]), int(curr_labels[idx])) for idx in crossing_indices
    ]

    return crossing_times, crossing_directions


# =============================================================================
# Composite Decision Analysis Function
# =============================================================================


def compute_decision_analysis(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    decision_region: str,
    goal_regions: list[str],
    pre_window: float = 1.0,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> DecisionAnalysisResult:
    """Compute complete decision analysis for a trajectory.

    Parameters
    ----------
    env : Environment
        Spatial environment with region definitions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates. A row with a non-finite coordinate (NaN, for a
        lost frame) is a missing sample.
    decision_region : str
        Name of decision region in env.regions (e.g., "center" for T-maze).
    goal_regions : list[str]
        Names of goal regions (e.g., ["left", "right"]).
    pre_window : float, default=1.0
        Duration of pre-decision window to analyze (seconds).
    min_speed : float or None, default=None
        In position units per second. None is 5 cm/s expressed in
        ``env.units`` (``0.05`` for meters, ``50.0`` for millimeters); with
        units that are not a metric length, pass it explicitly.
        Minimum speed for valid heading (units/s).
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
        time. Entry 0 is ignored. Neither the pre-decision window nor a
        boundary crossing spans a declared break.

    Returns
    -------
    DecisionAnalysisResult
        Complete decision analysis including pre-decision metrics,
        boundary metrics, and chosen goal.

    Raises
    ------
    ValueError
        If decision_region or any goal_region not found in env.regions.
    ValueError
        If positions and times have different lengths, if there are no
        samples, if ``times`` is not finite and strictly increasing, or if
        ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    The recording is split into time segments at gaps longer than
    ``max_gap``, at ``epochs`` edges and at declared breaks; no segment spans
    a pause. Pre-decision windows and boundary crossings respect these
    segments (see :func:`compute_pre_decision_metrics` and
    :func:`detect_boundary_crossings`). Missing positions fall in no bin, so
    their goal label is -1 and their boundary distance NaN. Per-sample goal
    labels and boundary distances keep the input length.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, Regions
    >>> from neurospatial.behavior import compute_decision_analysis

    A T-maze: a stem rising to a crossbar with a goal at each end. The animal
    walks up the stem and turns right.

    >>> stem = np.column_stack([np.full(30, 50.0), np.linspace(0, 50, 30)])
    >>> bar = np.column_stack([np.linspace(0, 100, 60), np.full(60, 55.0)])
    >>> env = Environment.from_samples(np.vstack([stem, bar]), bin_size=5.0)
    >>> regions = Regions()
    >>> _ = regions.add("center", point=(50.0, 55.0))
    >>> _ = regions.add("left", point=(5.0, 55.0))
    >>> _ = regions.add("right", point=(95.0, 55.0))
    >>> env = env.with_regions(regions)
    >>> times = np.arange(100) / 10  # s
    >>> positions = np.column_stack(
    ...     [
    ...         np.r_[np.full(30, 50.0), np.linspace(50, 95, 70)],
    ...         np.r_[np.linspace(5, 55, 30), np.full(70, 55.0)],
    ...     ]
    ... )  # cm
    >>> result = compute_decision_analysis(
    ...     env,
    ...     times,
    ...     positions,
    ...     decision_region="center",
    ...     goal_regions=["left", "right"],
    ...     min_speed=5.0,
    ... )
    >>> result.chosen_goal, result.pre_decision.n_samples
    (1, 10)
    """
    require_cartesian(
        env.coordinate_system,
        "compute_decision_analysis()",
        why="It reads positions as the animal's allocentric positions, with "
        "speeds and regions in the same Cartesian coordinates.",
        fix="use the allocentric environment the positions were recorded in",
    )
    from neurospatial.behavior._kinematics import _validate_velocity_positions

    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_decision_analysis",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the pre-decision metrics are NaN and no boundary crossing was observed",
    )
    _validate_velocity_positions(raw_positions, call="compute_decision_analysis")

    # Validate regions exist
    _require_region_mask(
        env,
        decision_region,
        operation="compute_decision_analysis()",
        argument="decision_region",
    )
    for goal_region in goal_regions:
        _require_region_mask(
            env,
            goal_region,
            operation="compute_decision_analysis()",
            argument="goal_regions",
            in_list=True,
        )

    # Get position bins
    position_bins = env.bin_at(positions)

    # Find entry time to decision region
    entry_time = decision_region_entry_time(
        position_bins, times, env, region=decision_region
    )

    from neurospatial.behavior._kinematics import resolve_min_speed

    min_speed = resolve_min_speed(env, min_speed, call="compute_decision_analysis()")
    # Compute pre-decision metrics
    pre_decision = _pre_decision_metrics(
        times, positions, support, entry_time, pre_window, min_speed=min_speed
    )

    # Compute boundary metrics. Each goal region is represented by its
    # centroid bin, as in trials_to_region_arrays.
    goal_bins = []
    for r in goal_regions:
        goal_bin = region_representative_bin(env, r)
        if goal_bin is not None:
            goal_bins.append(goal_bin)
        else:
            raise ValueError(
                f"Goal region '{r}' contains no bins. "
                f"Check that the region is within the environment bounds."
            )
    voronoi_labels = geodesic_voronoi_labels(env, goal_bins)

    position_bins = np.asarray(position_bins, dtype=np.int64)
    trajectory_labels = _safe_gather(voronoi_labels, position_bins, fill=-1)
    boundary_distances = distance_to_decision_boundary(env, position_bins, goal_bins)
    crossing_times, crossing_directions = _boundary_crossings(
        position_bins, voronoi_labels, times, support.interval_gates
    )

    boundary = DecisionBoundaryMetrics(
        goal_labels=trajectory_labels,
        distance_to_boundary=boundary_distances,
        crossing_times=crossing_times,
        crossing_directions=crossing_directions,
    )

    # Determine chosen goal (which goal region was reached)
    chosen_goal: int | None = None
    for i, goal_region in enumerate(goal_regions):
        goal_region_bins = set(env.bins_in_region(goal_region))
        # Check if trajectory ends in this goal region
        for bin_idx in reversed(position_bins):
            if bin_idx in goal_region_bins:
                chosen_goal = i
                break
        if chosen_goal is not None:
            break

    return DecisionAnalysisResult(
        entry_time=entry_time,
        pre_decision=pre_decision,
        boundary=boundary,
        chosen_goal=chosen_goal,
    )
