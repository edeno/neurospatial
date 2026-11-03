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
...     window_duration=1.0,
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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import DEFAULT_MAX_GAP, as_intervals
from neurospatial._observation import (
    ObservationSupport,
    validated_observations,
    validated_signals,
)
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.behavior._decision_windows import (
    first_entry,
    pre_decision_window,
    speed_stats,
    validated_entry_time,
    validated_window_duration,
)
from neurospatial.behavior._kinematics import (
    _movement_heading_and_speed,
    run_speed,
    validated_min_speed,
    validated_motion,
)
from neurospatial.behavior._position_bins import sample_bins
from neurospatial.behavior.segmentation import _region_names
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neurospatial.behavior._goals import Goal
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem

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


@dataclass(frozen=True, repr=False)
class PreDecisionMetrics(ResultMixin):
    """Metrics from the pre-decision window.

    The pre-decision window is the time period immediately before the animal
    enters a decision region (e.g., 1 second before entering a T-junction).

    Attributes
    ----------
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
        Low values may indicate pausing/deliberation.
    min_observed_speed : float
        Minimum observed speed in the window (units/s).
        Near-zero indicates a pause, common during VTE. Not to be confused
        with the ``min_speed`` heading gate passed to the analysis.
    heading_mean_direction : float
        Circular mean of heading direction (radians, -pi to pi).
    heading_circular_variance : float
        Circular variance of heading, range [0, 1].
        High values (> 0.5) indicate variable heading - possible scanning.
        Low values indicate stable, consistent heading.
    heading_mean_resultant_length : float
        Concentration of heading distribution, range [0, 1]:
        ``1 - heading_circular_variance``. High = concentrated, low =
        dispersed.
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

    _repr_units = MappingProxyType(
        {
            "heading_mean_direction": "rad",
            "window_duration": "s",
        }
    )

    mean_speed: float
    min_observed_speed: float
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
        max_mean_speed: float = 10.0,
    ) -> bool:
        """Check if metrics suggest deliberative behavior.

        A threshold screen, not a statistical test.

        Parameters
        ----------
        variance_threshold : float, default=0.5
            Circular variance at or above this suggests head scanning.
        max_mean_speed : float, default=10.0
            Mean speed at or below this (units/s) suggests slowing. A ceiling
            on ``mean_speed``, unrelated to the ``min_speed`` heading gate.

        Returns
        -------
        bool
            True if valid observations show ``heading_circular_variance >=
            variance_threshold`` AND ``mean_speed <= max_mean_speed``. False
            for missing evidence; inspect ``is_valid`` to distinguish missing
            evidence from an observed negative result.
        """
        return (
            self.is_valid
            and self.heading_circular_variance >= variance_threshold
            and self.mean_speed <= max_mean_speed
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar pre-decision metrics.

        For a human-readable string use ``str(result)``.
        """
        return {
            "mean_speed": self.mean_speed,
            "min_observed_speed": self.min_observed_speed,
            "heading_mean_direction": self.heading_mean_direction,
            "heading_circular_variance": self.heading_circular_variance,
            "heading_mean_resultant_length": self.heading_mean_resultant_length,
            "window_duration": self.window_duration,
            "n_samples": self.n_samples,
            "is_valid": self.is_valid,
        }

    def __str__(self) -> str:
        return (
            f"Pre-decision window: {self.n_samples} samples over "
            f"{self.window_duration:.2f} s, mean speed={self.mean_speed:.1f}, "
            f"heading variance={self.heading_circular_variance:.2f}"
        )


@dataclass(frozen=True, repr=False, eq=False)
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
    goal_labels : NDArray[np.int_], shape (n_samples,)
        Index of each sample's nearest goal (its geodesic Voronoi region),
        into the goal list. -1 where no goal is nearest: the sample is
        outside every bin, its bin is equidistant from the two nearest
        goals (on the boundary, undecided), or no goal is reachable.
    distance_to_decision_boundary : NDArray[np.float64], shape (n_samples,)
        Geodesic distance from each sample's bin to the decision boundary, in
        environment units (see :func:`distance_to_decision_boundary`). Small
        values = near the boundary = uncommitted; NaN for a sample outside
        every bin.
    crossing_times : NDArray[np.float64], shape (n_crossings,)
        Times when the nearest goal changed, in seconds (see
        :func:`detect_boundary_crossings`).
    crossing_directions : NDArray[np.intp], shape (n_crossings, 2)
        ``(from_goal_idx, to_goal_idx)`` for each crossing: the goals whose
        regions the animal moved between.

    All arrays are read-only copies owned by the result.
    """

    goal_labels: NDArray[np.int_]
    distance_to_decision_boundary: NDArray[np.float64]
    crossing_times: NDArray[np.float64]
    crossing_directions: NDArray[np.intp]

    def __post_init__(self) -> None:
        """Check shapes, then own read-only copies of every array."""
        goal_labels = np.asarray(self.goal_labels)
        distances = np.asarray(self.distance_to_decision_boundary, dtype=np.float64)
        crossing_times = np.asarray(self.crossing_times, dtype=np.float64)
        directions = np.asarray(self.crossing_directions, dtype=np.intp).reshape(-1, 2)
        if goal_labels.ndim != 1 or distances.shape != goal_labels.shape:
            msg = (
                "DecisionBoundaryMetrics needs goal_labels and "
                "distance_to_decision_boundary of the same shape (n_samples,); "
                f"got {goal_labels.shape} and {distances.shape}."
            )
            raise ValueError(msg)
        if crossing_times.ndim != 1 or len(directions) != len(crossing_times):
            msg = (
                "DecisionBoundaryMetrics needs one (from, to) row of "
                "crossing_directions per crossing time; got crossing_times of "
                f"shape {crossing_times.shape} and crossing_directions of shape "
                f"{directions.shape}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "goal_labels", goal_labels)
        object.__setattr__(self, "distance_to_decision_boundary", distances)
        object.__setattr__(self, "crossing_times", crossing_times)
        object.__setattr__(self, "crossing_directions", directions)
        self._own_arrays()

    @property
    def n_crossings(self) -> int:
        """Number of decision boundary crossings."""
        return len(self.crossing_times)

    @property
    def mean_distance_to_decision_boundary(self) -> float:
        """Mean ``distance_to_decision_boundary`` over the samples inside a bin.

        In environment units; NaN when no sample is inside a bin.
        """
        distances = self.distance_to_decision_boundary
        inside = distances[~np.isnan(distances)]
        return float(inside.mean()) if inside.size else np.nan

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar decision-boundary metrics.

        For a human-readable string use ``str(result)``.
        """
        return {
            "n_crossings": self.n_crossings,
            "mean_distance_to_decision_boundary": (
                self.mean_distance_to_decision_boundary
            ),
        }

    def __str__(self) -> str:
        return (
            f"Decision boundary: {self.n_crossings} crossing"
            f"{'s' if self.n_crossings != 1 else ''}, "
            "mean distance to boundary: "
            f"{self.mean_distance_to_decision_boundary:.1f}"
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
        Boundary metrics. None with a single goal region: no decision
        boundary is defined.
    chosen_goal : int or None
        Index into ``goal_regions`` of the first goal region the animal
        entered after ``entry_time``. Goals visited before the decision-region
        entry (such as the goal the trial started from) do not count. None if
        no goal region was entered after ``entry_time``.
    goal_regions : tuple of str
        Names of the goal regions, in the order that ``chosen_goal``,
        ``boundary.goal_labels`` and ``boundary.crossing_directions`` index.
    """

    _repr_units = MappingProxyType(
        {
            "entry_time": "s",
        }
    )

    entry_time: float
    pre_decision: PreDecisionMetrics
    boundary: DecisionBoundaryMetrics | None
    chosen_goal: int | None
    goal_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Store goal names as a tuple and check ``chosen_goal`` indexes them."""
        object.__setattr__(self, "goal_regions", tuple(self.goal_regions))
        if self.chosen_goal is not None and not (
            0 <= self.chosen_goal < len(self.goal_regions)
        ):
            msg = (
                f"chosen_goal={self.chosen_goal} does not index the "
                f"{len(self.goal_regions)} goal_regions {self.goal_regions}."
            )
            raise ValueError(msg)

    @property
    def chosen_goal_region(self) -> str | None:
        """Name of the chosen goal region; None if no goal was entered."""
        if self.chosen_goal is None:
            return None
        return self.goal_regions[self.chosen_goal]

    @property
    def pre_decision_mean_speed(self) -> float:
        """Mean speed in the pre-decision window."""
        return self.pre_decision.mean_speed

    @property
    def pre_decision_min_observed_speed(self) -> float:
        """Smallest observed speed in the pre-decision window."""
        return self.pre_decision.min_observed_speed

    @property
    def pre_decision_heading_variance(self) -> float:
        """Circular variance of heading in the pre-decision window."""
        return self.pre_decision.heading_circular_variance

    @property
    def n_boundary_crossings(self) -> int | None:
        """Decision-boundary crossings; None without a boundary."""
        return self.boundary.n_crossings if self.boundary is not None else None

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar decision-analysis metrics.

        For a human-readable multi-line string use ``str(result)``.
        """
        return {
            "entry_time": self.entry_time,
            "pre_decision_mean_speed": self.pre_decision_mean_speed,
            "pre_decision_min_observed_speed": self.pre_decision_min_observed_speed,
            "pre_decision_heading_variance": self.pre_decision_heading_variance,
            "n_boundary_crossings": self.n_boundary_crossings,
            "chosen_goal": self.chosen_goal,
            "chosen_goal_region": self.chosen_goal_region,
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
            lines.append(
                f"  Chosen goal: {self.chosen_goal} ({self.chosen_goal_region!r})"
            )
        else:
            lines.append("  Chosen goal: none entered")
        return "\n".join(lines)


# =============================================================================
# Pre-Decision Window Functions
# =============================================================================


@reads_pynapple_samples
def decision_region_entry_time(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    region: str,
) -> float:
    """Find time of first entry to a decision region.

    Parameters
    ----------
    env : Environment
        Environment with region definitions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where tracking was
        lost (such a sample is in no region).
    region : str
        Name of decision region in env.regions.

    Returns
    -------
    float
        Time of the first sample inside the decision region (seconds).

    Raises
    ------
    ValueError
        If region not found in env.regions, if ``positions`` do not have one
        row per time with the environment's number of coordinates, if
        ``times`` is not finite or not strictly increasing, or if the
        trajectory never enters the region.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, Regions
    >>> from neurospatial.behavior import decision_region_entry_time
    >>> env = Environment.from_samples(np.arange(101.0)[:, None], bin_size=10.0)
    >>> regions = Regions()
    >>> _ = regions.add("center", point=(50.0,))
    >>> env = env.with_regions(regions)
    >>> times = np.arange(11) / 10  # s
    >>> positions = 10.0 * np.arange(11)[:, None]  # 100 cm/s
    >>> decision_region_entry_time(env, times, positions, region="center")
    0.5
    """
    _require_region_mask(
        env, region, operation="decision_region_entry_time()", argument="region"
    )
    times, position_bins, _ = sample_bins(
        env, times, positions, call="decision_region_entry_time"
    )
    entry = first_entry(
        position_bins, env.bins_in_region(region), np.ones(times.shape, dtype=bool)
    )
    if entry is None:
        raise ValueError(_never_enters_message(region, epochs=None))
    return float(times[entry])


def _never_enters_message(region: str, *, epochs: Any) -> str:
    where = " inside epochs" if epochs is not None else ""
    return (
        f"Trajectory never enters region '{region}'{where}.\n"
        "Why: the decision-region entry time is the first sample inside the "
        "region.\n"
        "Fix: check that the trajectory passes through the decision region "
        f"(verify its bounds with env.regions['{region}'])"
        + (" during the epochs" if epochs is not None else "")
        + "."
    )


@reads_pynapple_samples
def extract_pre_decision_window(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    entry_time: float,
    *,
    window_duration: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
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
        Time of decision region entry, in seconds on the clock of ``times``;
        within ``[times[0], times[-1]]``.
    window_duration : float
        Duration of pre-decision window to extract (seconds); finite and
        positive.
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
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
        increasing, if ``break_before`` is not a Boolean array with one
        entry per sample, if ``window_duration`` is not finite and positive,
        or if ``entry_time`` is outside the recording.

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
    >>> times = np.arange(10) / 4  # s
    >>> positions = np.column_stack([10 * times, np.zeros(10)])  # cm
    >>> positions[6] = np.nan  # a lost frame
    >>> window_times, window_positions = extract_pre_decision_window(
    ...     times, positions, entry_time=2.0, window_duration=1.0
    ... )
    >>> window_times.tolist()
    [1.0, 1.25, 1.5, 1.75]
    >>> window_positions[:, 0].tolist()
    [10.0, 12.5, nan, 17.5]

    A new recording starting at 1.5 s ends the window there; its first
    frame was lost, so the window starts at its first observed frame:

    >>> break_before = times == 1.5
    >>> extract_pre_decision_window(
    ...     times, positions, 2.0, window_duration=1.0, break_before=break_before
    ... )[0].tolist()
    [1.75]
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
    window_duration = validated_window_duration(
        window_duration, name="window_duration", call="extract_pre_decision_window"
    )
    entry_time = validated_entry_time(
        times, entry_time, call="extract_pre_decision_window"
    )
    window = pre_decision_window(
        times, support.interval_gates, entry_time, window_duration
    )
    return times[window], positions[window]


@reads_pynapple_samples
def pre_decision_heading_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
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
        sample: it has no heading of its own and is bridged like a deleted
        row when the observed samples on either side are within ``max_gap``.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s with positions
        in cm, the speed below which Davidson, Kloosterman & Wilson (2009, Neuron
        63:497) and Pfeiffer & Foster (2013, Nature 497:74) counted a rat as stopped):
        this function sees no environment, so it cannot know the positions' unit.
        Minimum speed for valid heading (units/s); finite and non-negative. Stationary
        periods are excluded from statistics.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
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
    A missing row is skipped as if it were deleted: when the observed
    samples on either side are no more than ``max_gap`` apart, velocity and
    heading are computed across it. Only a longer gap, an ``epochs`` edge or
    a ``break_before`` splits the recording into separately analyzed runs,
    and no velocity or heading spans such a split.

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
    times, positions, support = validated_observations(
        times,
        positions,
        call="pre_decision_heading_stats",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the heading statistics are NaN",
        columns="xy",
    )
    call = "pre_decision_heading_stats"
    min_speed = validated_min_speed(min_speed, call=call)
    heading, _ = _movement_heading_and_speed(
        times,
        positions,
        support.interval_mask,
        smooth_window=DEFAULT_SPEED_SMOOTHING,
        min_speed=min_speed,
        speed=validated_motion(speed, name="speed", n_samples=len(times), call=call),
        headings=validated_motion(
            headings, name="headings", n_samples=len(times), call=call
        ),
    )
    return _heading_stats(heading)


def _heading_stats(headings: NDArray[np.float64]) -> tuple[float, float, float]:
    """Circular statistics of the headings of moving samples (NaN elsewhere)."""
    valid_headings = headings[np.isfinite(headings)]

    if len(valid_headings) == 0:
        return np.nan, np.nan, np.nan

    from neurospatial.stats.circular import circular_mean, mean_resultant_length

    # Circular variance is 1 - R, as stats.circular.circular_variance.
    mrl = mean_resultant_length(valid_headings)
    return circular_mean(valid_headings), 1.0 - mrl, mrl


@reads_pynapple_samples
def pre_decision_speed_stats(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
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
        lost frame) is a missing sample: it has no speed of its own and is
        bridged like a deleted row when the observed samples on either side
        are within ``max_gap``.
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
        time. Entry 0 is ignored. No velocity spans a declared break.

    Returns
    -------
    mean_speed : float
        Mean instantaneous speed (units/s).
    min_observed_speed : float
        Minimum instantaneous speed (units/s).

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, or if ``break_before`` is not a Boolean array with one
        entry per sample.

    Notes
    -----
    A missing row is skipped as if it were deleted: when the observed
    samples on either side are no more than ``max_gap`` apart, velocity is
    computed across it. Only a longer gap, an ``epochs`` edge or a
    ``break_before`` splits the recording into separately analyzed runs, and
    no velocity spans such a split.

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
    >>> mean_speed, min_observed_speed = pre_decision_speed_stats(
    ...     times, positions, break_before=new_recording
    ... )
    >>> round(mean_speed, 6), round(min_observed_speed, 6)
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
    speed = validated_motion(
        speed, name="speed", n_samples=len(times), call="pre_decision_speed_stats"
    )
    return speed_stats(run_speed(times, positions, support.interval_mask, speed=speed))


@reads_pynapple_samples
def compute_pre_decision_metrics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    entry_time: float,
    *,
    window_duration: float,
    min_speed: float,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
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
        Time of decision region entry, in seconds on the clock of ``times``;
        within ``[times[0], times[-1]]``.
    window_duration : float
        Duration of pre-decision window to analyze (seconds); finite and
        positive.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s with positions
        in cm, the speed below which Davidson, Kloosterman & Wilson (2009, Neuron
        63:497) and Pfeiffer & Foster (2013, Nature 497:74) counted a rat as stopped):
        this function sees no environment, so it cannot know the positions' unit.
        Minimum speed for valid heading (units/s); finite and non-negative.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
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
        increasing, if ``break_before`` is not a Boolean array with one
        entry per sample, if ``window_duration`` is not finite and positive,
        or if ``entry_time`` is outside the recording.

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
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_pre_decision_metrics",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the pre-decision metrics are NaN",
        columns="xy",
    )
    call = "compute_pre_decision_metrics"
    min_speed = validated_min_speed(min_speed, call=call)
    speed = validated_motion(speed, name="speed", n_samples=len(times), call=call)
    headings = validated_motion(
        headings, name="headings", n_samples=len(times), call=call
    )
    window_duration = validated_window_duration(
        window_duration, name="window_duration", call="compute_pre_decision_metrics"
    )
    entry_time = validated_entry_time(
        times, entry_time, call="compute_pre_decision_metrics"
    )
    return _pre_decision_metrics(
        times,
        positions,
        support,
        entry_time,
        window_duration,
        min_speed=min_speed,
        speed=speed,
        headings=headings,
    )


def _pre_decision_metrics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float,
    speed: NDArray[np.float64] | None,
    headings: NDArray[np.float64] | None,
) -> PreDecisionMetrics:
    """Speed and heading metrics of the pre-decision window under ``support``.

    Speed and heading are computed from the window's samples alone, as
    :func:`pre_decision_speed_stats` and :func:`pre_decision_heading_stats`
    compute them on :func:`extract_pre_decision_window`'s output; a caller's
    ``speed`` and ``headings`` are taken at the window's samples.
    """
    window = pre_decision_window(
        times, support.interval_gates, entry_time, window_duration
    )
    window_times = times[window]
    observed_times = window_times[support.sample_valid[window]]
    if observed_times.size < 2:
        return PreDecisionMetrics(
            mean_speed=np.nan,
            min_observed_speed=np.nan,
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
    window_speed = None if speed is None else speed[window]
    heading, _ = _movement_heading_and_speed(
        window_times,
        window_positions,
        window_mask,
        smooth_window=DEFAULT_SPEED_SMOOTHING,
        min_speed=min_speed,
        speed=window_speed,
        headings=None if headings is None else headings[window],
    )
    mean_dir, circ_var, mrl = _heading_stats(heading)
    mean_speed, min_observed_speed = speed_stats(
        run_speed(window_times, window_positions, window_mask, speed=window_speed)
    )
    return PreDecisionMetrics(
        mean_speed=mean_speed,
        min_observed_speed=min_observed_speed,
        heading_mean_direction=mean_dir,
        heading_circular_variance=circ_var,
        heading_mean_resultant_length=mrl,
        window_duration=float(observed_times[-1] - observed_times[0]),
        n_samples=int(observed_times.size),
    )


# =============================================================================
# Decision Boundary Functions
# =============================================================================

# Relative tolerance below which two geodesic distances to goals are a tie.
_TIE_RTOL = 1e-9


def geodesic_voronoi_labels(
    env: Environment[CartesianSystem],
    *,
    goals: Sequence[Any],
) -> NDArray[np.int_]:
    """Label each bin by its nearest goal using geodesic distance.

    Creates a Voronoi-like partition of the environment where each bin is
    assigned to the goal with the shortest geodesic path. The geodesic distance
    is the length of the shortest path between two positions that stays inside
    the environment: it goes around walls and never leaves the environment, and
    on a track it runs along the track's centerline. Bins are measured from
    their centers; a region goal is measured to its edge (0 inside it) and a
    goal point from the center of its bin.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goals : sequence of str or array-like of shape (n_dims,)
        The goals, each a region name or a point in environment coordinates,
        as in :func:`~neurospatial.behavior.distance_to_goal`, for example
        ``goals=["left_well", "right_well"]``. A polygon region is measured
        to its edge (0 inside it); a point region or a point to the point itself
        (to the bin containing it for the per-bin labels).

    Returns
    -------
    NDArray[np.int_], shape (n_bins,)
        Index of the nearest goal for each bin (indices into ``goals``),
        or -1 where no goal is nearest: the bin is equidistant from its two
        nearest goals (it lies on the decision boundary, so the animal there
        is undecided), or no goal is reachable from it.

    Raises
    ------
    ValueError
        If ``goals`` is empty, a region covers no bin, or a goal point is
        malformed or outside the environment.
    TypeError
        If ``goals`` is a single string instead of a sequence of goals.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment, whose bins are
        (distance, angle), not positions.

    Notes
    -----
    Two geodesic distances count as equal when they agree to a relative
    tolerance of 1e-9, far below a bin, so floating-point rounding along
    different paths cannot break a geometric tie. Labeling a tie -1 instead
    of the first goal keeps the labels independent of goal order and
    mirror-symmetric on a symmetric maze.

    A list that mixes region goals and point goals measures them differently:
    a region from its nearest edge, a point from the center of its bin. The
    boundary between a region and a point therefore lies halfway between the
    region's edge and the point, not between their centers, so a large region
    claims more bins than a point at its center would. When the goals should
    compete from their centers, give them all as points (or point regions).

    The shortest-path field of each goal is grown once and kept on the
    environment, so later calls with the same goals are lookups.

    Examples
    --------
    A 100 cm linear track with a goal at each end. The bin at 50 cm is
    equidistant from both goals, so it is undecided:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import geodesic_voronoi_labels
    >>> env = Environment.from_samples(np.arange(101.0)[:, None], bin_size=10.0)
    >>> geodesic_voronoi_labels(env, goals=[[0.0], [100.0]]).tolist()
    [0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1]
    """
    targets = _resolved_goals(env, goals, call="geodesic_voronoi_labels")
    n_goals = len(targets)
    distances = _goal_fields(env, targets, call="geodesic_voronoi_labels")

    labels_int: NDArray[np.int_] = np.argmin(distances, axis=0).astype(np.int_)
    sorted_distances = np.sort(distances, axis=0)
    nearest = sorted_distances[0]
    undecided = np.isinf(nearest)
    if n_goals > 1:
        second = sorted_distances[1]
        both_reachable = np.isfinite(second)
        undecided[both_reachable] |= (
            second[both_reachable] - nearest[both_reachable]
            <= _TIE_RTOL * second[both_reachable]
        )
    labels_int[undecided] = -1
    return labels_int


def _resolved_goals(env: Environment, goals: Any, *, call: str) -> list[Goal]:
    """Each goal of ``goals`` resolved on the geodesic engine."""
    from neurospatial._exceptions import _format_error
    from neurospatial.behavior._environment_paths import environment_paths
    from neurospatial.behavior._goals import resolve_goal

    if isinstance(goals, str):
        msg = (
            f"{call}(): goals must be a sequence of goals, for example "
            f"goals=[{goals!r}, 'other_well']; got the single name {goals!r}."
        )
        raise TypeError(msg)
    entries = list(goals)
    if not entries:
        raise ValueError(
            _format_error(
                f"{call}(): goals is empty.",
                why="the decision boundary lies between goals.",
                fix="pass the goals' region names, e.g. goals=['left', 'right'].",
            )
        )
    paths = environment_paths(env, call=call)
    return [resolve_goal(env, goal, paths=paths, call=call) for goal in entries]


def _goal_fields(
    env: Environment, goals: list[Goal], *, call: str
) -> NDArray[np.float64]:
    """Geodesic distance from every bin center to each goal.

    A region goal is measured to its edge, as :meth:`Goal.distances` measures
    it (0 for a bin center inside it); a goal point at the center of the bin
    containing it.

    Returns
    -------
    ndarray, shape (n_goals, n_bins)
        ``inf`` where a goal cannot be reached.
    """
    from neurospatial.behavior._environment_paths import environment_paths

    paths = environment_paths(env, call=call)
    centers = np.asarray(env.bin_centers, dtype=np.float64)
    return np.vstack(
        [
            goal.distances(env, paths, centers)
            if goal.area is not None
            else paths.from_bins(
                env.bin_at(np.asarray(goal.point)[None, :]).astype(np.intp)
            )
            for goal in goals
        ]
    )


@reads_pynapple_samples
def distance_to_decision_boundary(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    *,
    goals: Sequence[Any],
) -> NDArray[np.float64]:
    """Geodesic distance from each position to the decision boundary.

    The decision boundary is the set of points equidistant (geodesically)
    from the two nearest goals: the edge between their geodesic Voronoi
    regions.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row or a position
        outside the environment gives NaN.
    goals : sequence of str or array-like of shape (n_dims,)
        The goals, each a region name or a point in environment coordinates,
        as in :func:`~neurospatial.behavior.distance_to_goal`, for example
        ``goals=["left_well", "right_well"]``. A polygon region is measured
        to its edge (0 inside it); a point region or a point to the point itself
        (to the bin containing it for the per-bin labels).

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance to the decision boundary in environment units:
        ``(d2 - d1) / 2``, where ``d1`` and ``d2`` are the geodesic distances
        from the position to its nearest and second-nearest goal. 0 on
        the boundary; small values = uncommitted. NaN for a missing position
        or one outside the environment; inf where the second-nearest goal is
        unreachable or with fewer than two goals (there is no boundary).

    Raises
    ------
    ValueError
        If ``times`` or ``positions`` are invalid, ``goals`` is empty, a
        region covers no bin, or a goal point is malformed or outside the
        environment.
    TypeError
        If ``goals`` is a single string instead of a sequence of goals.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Notes
    -----
    Walking from the position toward the second-nearest goal along the path
    between the two goals shortens ``d2`` and lengthens ``d1`` by the
    distance walked, so the two meet after ``(d2 - d1) / 2``. This is the
    exact geodesic distance to the boundary for a position on the path
    between the goals (a linear track, the arms of a T-maze). Off that path,
    for example in a dead-end side arm, it is a lower bound: each geodesic
    distance changes by at most the distance walked.

    The geodesic distance is the length of the shortest path between two
    positions that stays inside the environment: it goes around walls and never
    leaves the environment, and on a track it runs along the track's
    centerline. Distances are measured from the position itself, as in
    :func:`~neurospatial.behavior.distance_to_goal`.

    Examples
    --------
    A 100 cm linear track with a goal at each end. The position at 30 cm is
    30 cm from one goal and 70 cm from the other, so the boundary at 50 cm
    is 20 cm away:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import distance_to_decision_boundary
    >>> env = Environment.from_samples(np.arange(101.0)[:, None], bin_size=10.0)
    >>> positions = np.array([[30.0], [50.0], [90.0]])
    >>> distance_to_decision_boundary(
    ...     env, np.arange(3.0), positions, goals=[[0.0], [100.0]]
    ... ).tolist()
    [20.0, 0.0, 40.0]
    """
    from neurospatial.behavior._environment_paths import environment_paths
    from neurospatial.behavior.navigation import _validated_route_samples

    call = "distance_to_decision_boundary"
    targets = _resolved_goals(env, goals, call=call)
    _, positions_arr, _ = _validated_route_samples(
        env, times, positions, None, call=call
    )
    located = env.bin_at(positions_arr) >= 0
    boundary_distances = np.full(len(positions_arr), np.nan)
    if len(targets) < 2:
        # With a single goal there is no boundary.
        boundary_distances[located] = np.inf
        return boundary_distances
    paths = environment_paths(env, call=call)
    distances = np.vstack(
        [goal.distances(env, paths, positions_arr[located]) for goal in targets]
    )
    # The boundary lies halfway between the two nearest goals: (d2 - d1) / 2
    # after sorting each position's goal distances.
    sorted_dists = np.sort(distances, axis=0)
    nearest, second = sorted_dists[0], sorted_dists[1]
    # When the nearest goal is unreachable (inf), the boundary distance is
    # inf; compute the gap only where it is finite to avoid inf - inf = nan.
    finite_nearest = np.isfinite(nearest)
    located_boundary = np.full(nearest.shape, np.inf)
    located_boundary[finite_nearest] = (
        second[finite_nearest] - nearest[finite_nearest]
    ) / 2
    boundary_distances[located] = located_boundary
    return boundary_distances


@reads_pynapple_samples
def detect_boundary_crossings(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    *,
    goals: Sequence[Any],
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Detect when trajectory crosses decision boundaries.

    A crossing is a change of nearest goal, by the bins of
    :func:`geodesic_voronoi_labels`.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : ArrayLike, shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row, or a position
        outside the environment, is missing.
    goals : sequence of str or array-like of shape (n_dims,)
        The goals, each a region name or a point in environment coordinates,
        as in :func:`geodesic_voronoi_labels`.
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No crossing spans a declared break.

    Returns
    -------
    crossing_times : NDArray[np.float64], shape (n_crossings,)
        Times when trajectory crossed a decision boundary: the midpoint of
        the last sample with one nearest goal and the first with the other.
    crossing_directions : NDArray[np.intp], shape (n_crossings, 2)
        ``(from_goal_idx, to_goal_idx)`` for each crossing, indices into
        ``goals``.

    Raises
    ------
    ValueError
        If ``times`` and ``positions`` differ in length, if there are no
        samples, if ``times`` is not finite and strictly increasing, if
        ``break_before`` is not a Boolean array with one entry per sample,
        or ``goals`` is empty or holds a goal point outside the environment.
    TypeError
        If ``goals`` is a single string instead of a sequence of goals.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Notes
    -----
    Each time segment with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording; no
    crossing spans a pause. A missing sample (a NaN row, or a position outside
    the environment) is skipped as if its row were deleted, so a crossing
    across a bridged gap is timed at the midpoint of the located samples on
    either side.

    A crossing is a change of nearest goal. A sample in a bin labeled -1
    has no nearest goal and is skipped too: walking through the bins on the
    boundary from goal 0's side to goal 1's side is one crossing, and
    leaving the boundary toward a goal (up the stem of a symmetric T-maze
    and into either arm) is none.

    Examples
    --------
    A 100 cm linear track with a goal at each end, crossed once at 8 Hz:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import detect_boundary_crossings
    >>> env = Environment.from_samples(np.arange(101.0)[:, None], bin_size=10.0)
    >>> positions = np.array([[30.0], [40.0], [60.0], [70.0]])
    >>> times = np.arange(4) / 8  # s
    >>> crossing_times, directions = detect_boundary_crossings(
    ...     env, times, positions, goals=[[0.0], [100.0]]
    ... )
    >>> crossing_times.tolist(), directions.tolist()
    ([0.1875], [[0, 1]])

    The same samples from two recordings joined on one clock, where the
    second recording starts at 60 cm, have no crossing:

    >>> new_recording = np.array([False, False, True, False])
    >>> crossing_times, directions = detect_boundary_crossings(
    ...     env,
    ...     times,
    ...     positions,
    ...     goals=[[0.0], [100.0]],
    ...     break_before=new_recording,
    ... )
    >>> crossing_times.size, directions.shape
    (0, (0, 2))
    """
    from neurospatial.behavior.navigation import _validated_route_samples

    call = "detect_boundary_crossings"
    labels = geodesic_voronoi_labels(env, goals=goals)
    times_arr, positions_arr, _ = _validated_route_samples(
        env, times, positions, None, call=call
    )
    position_bins = np.asarray(env.bin_at(positions_arr), dtype=np.int64)
    # A NaN row or a position outside the environment (bin -1) is missing.
    times_arr, _, supports = validated_signals(
        times_arr,
        position_bins,
        valid=position_bins >= 0,
        call=call,
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
        break_before=break_before,
        result="no boundary crossing was observed",
    )
    return _boundary_crossings(position_bins, labels, times_arr, supports[0])


def _boundary_crossings(
    position_bins: NDArray[np.int64],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
    support: ObservationSupport,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Boundary crossings within each run of ``support``.

    ``support`` treats every sample at bin -1 as missing, so it is skipped
    as if its row were deleted, and a crossing across it is timed between
    the located samples on either side when the gap between them is
    bridged.
    """
    crossing_times = [np.empty(0)]
    crossing_directions = [np.empty((0, 2), dtype=np.intp)]
    for run in support.run_samples():
        run_times, run_directions = _detect_boundary_crossings_contiguous(
            position_bins[run], voronoi_labels, times[run]
        )
        crossing_times.append(run_times)
        crossing_directions.append(run_directions)
    return np.concatenate(crossing_times), np.concatenate(crossing_directions)


def _detect_boundary_crossings_contiguous(
    position_bins: NDArray[np.int_],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Nearest-goal changes within one recording with labels indexed by bin.

    Every sample is in a bin: the caller removed the missing ones. Samples
    labeled -1 (undecided or unreachable) are skipped, so passing
    through the equidistant bins from one goal's side to the other's is one
    crossing, timed between the last sample on one side and the first on
    the other.
    """
    trajectory_labels = voronoi_labels[position_bins]
    decided = np.flatnonzero(trajectory_labels >= 0)
    labels = trajectory_labels[decided].astype(np.intp)
    changes = np.flatnonzero(labels[:-1] != labels[1:])
    before, after = decided[changes], decided[changes + 1]
    crossing_times = (times[before] + times[after]) / 2
    crossing_directions = np.column_stack([labels[changes], labels[changes + 1]])
    return crossing_times, crossing_directions.reshape(-1, 2)


# =============================================================================
# Composite Decision Analysis Function
# =============================================================================


@reads_pynapple_samples
def compute_decision_analysis(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    decision_region: str,
    goal_regions: str | list[str],
    window_duration: float = 1.0,
    min_speed: float | None = None,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
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
        Names of goal regions (e.g., ["left", "right"]); a single name is a
        one-goal list. With one goal there is no decision boundary and
        ``result.boundary`` is None.
    window_duration : float, default=1.0
        Duration of pre-decision window to analyze (seconds); finite and
        positive.
    min_speed : float or None, default=None
        Minimum speed for valid heading, in position units per second (finite and
        non-negative); None is 5 cm/s expressed in ``env.units`` (``0.05`` for meters,
        ``50.0`` for millimeters), the speed below which Davidson, Kloosterman & Wilson
        (2009, Neuron 63:497) and Pfeiffer & Foster (2013, Nature 497:74) counted a rat
        as stopped. With units that are not a metric length, pass it explicitly.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
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
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
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
        If decision_region or any goal_region not found in env.regions, or
        if goal_regions is empty.
    ValueError
        If positions and times have different lengths, if there are no
        samples, if ``times`` is not finite and strictly increasing, if
        ``break_before`` is not a Boolean array with one entry per sample, or
        if ``window_duration`` is not finite and positive.

    Notes
    -----
    The recording is split into time segments at gaps longer than
    ``max_gap``, at ``epochs`` edges and at declared breaks; no segment spans
    a pause. Pre-decision windows and boundary crossings respect these
    segments (see :func:`compute_pre_decision_metrics` and
    :func:`detect_boundary_crossings`). Missing positions fall in no bin, so
    their goal label is -1 and their boundary distance NaN. Per-sample goal
    labels and boundary distances keep the input length.

    The decision-region entry is the first sample inside both the region
    and ``epochs``, and the chosen goal is the first goal region occupied by
    a sample inside ``epochs`` from then on; samples outside ``epochs`` are
    never used.

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
    >>> result.chosen_goal_region, result.pre_decision.n_samples
    ('right', 10)

    The stem is equidistant from both goals (label -1), so walking up it and
    turning right never crosses from the left goal's side:

    >>> result.boundary.n_crossings
    0
    """
    require_cartesian(
        env.coordinate_system,
        "compute_decision_analysis()",
        why="It reads positions as the animal's allocentric positions, with "
        "speeds and regions in the same Cartesian coordinates.",
        fix="use the allocentric environment the positions were recorded in",
    )
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_decision_analysis",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the pre-decision metrics are NaN and no boundary crossing was observed",
        columns="xy",
    )
    window_duration = validated_window_duration(
        window_duration, name="window_duration", call="compute_decision_analysis"
    )

    # Validate regions exist
    _require_region_mask(
        env,
        decision_region,
        operation="compute_decision_analysis()",
        argument="decision_region",
    )
    goal_regions = _region_names(
        goal_regions,
        argument="goal_regions",
        why="the chosen goal and the decision boundary are defined by the goals "
        "the animal can choose between.",
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

    # The entry is the first decision-region sample inside the epochs.
    in_epochs = support.sample_in_epochs
    entry = first_entry(position_bins, env.bins_in_region(decision_region), in_epochs)
    if entry is None:
        raise ValueError(_never_enters_message(decision_region, epochs=epochs))
    entry_time = float(times[entry])

    from neurospatial.behavior._kinematics import resolve_min_speed

    call = "compute_decision_analysis"
    min_speed = resolve_min_speed(env, min_speed, call=call)
    # Compute pre-decision metrics
    pre_decision = _pre_decision_metrics(
        times,
        positions,
        support,
        entry_time,
        window_duration,
        min_speed=min_speed,
        speed=validated_motion(speed, name="speed", n_samples=len(times), call=call),
        headings=validated_motion(
            headings, name="headings", n_samples=len(times), call=call
        ),
    )

    # Compute boundary metrics. Each goal region is measured to its nearest
    # bin, as everywhere a region is a goal.
    position_bins = np.asarray(position_bins, dtype=np.int64)
    boundary = None
    if len(goal_regions) >= 2:
        voronoi_labels = geodesic_voronoi_labels(env, goals=goal_regions)
        crossing_times, crossing_directions = _boundary_crossings(
            position_bins,
            voronoi_labels,
            times,
            # A position outside every bin is missing too.
            support.restricted(times, position_bins >= 0),
        )
        boundary = DecisionBoundaryMetrics(
            goal_labels=np.where(position_bins >= 0, voronoi_labels[position_bins], -1),
            distance_to_decision_boundary=distance_to_decision_boundary(
                env, times, positions, goals=goal_regions
            ),
            crossing_times=crossing_times,
            crossing_directions=crossing_directions,
        )

    return DecisionAnalysisResult(
        entry_time=entry_time,
        pre_decision=pre_decision,
        boundary=boundary,
        chosen_goal=_first_goal_entered(
            env, position_bins[entry:], goal_regions, in_epochs[entry:]
        ),
        goal_regions=tuple(goal_regions),
    )


def _first_goal_entered(
    env: Environment,
    position_bins: NDArray[np.int64],
    goal_regions: list[str],
    observed: NDArray[np.bool_],
) -> int | None:
    """Index of the first goal region occupied by an observed sample; None if none.

    The caller passes the samples from the decision-region entry on, so goal
    occupancy before it (for example the goal the trial started from) does
    not count. A sample inside several goal regions picks the first of them
    in ``goal_regions`` order.
    """
    in_goal = np.stack(
        [np.isin(position_bins, env.bins_in_region(name)) for name in goal_regions]
    )
    reached = np.flatnonzero(observed & in_goal.any(axis=0))
    if reached.size == 0:
        return None
    return int(np.argmax(in_goal[:, reached[0]]))
