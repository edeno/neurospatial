"""Decision analysis and VTE (Vicarious Trial and Error) detection.

This module provides tools for analyzing animal behavior at choice points,
including pre-decision kinematics, decision boundary analysis, and VTE detection.

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
        # VTE
        VTETrialResult,
        VTESessionResult,
        compute_vte_index,
        compute_vte_trial,
        compute_vte_session,
        classify_vte,
        head_sweep_from_positions,
        head_sweep_magnitude,
        integrated_absolute_rotation,
        normalize_vte_scores,
        wrap_angle,
    )

Key Concepts
------------
Decision Analysis
~~~~~~~~~~~~~~~~~
- **Decision region**: Spatial zone where animal must choose (e.g., junction)
- **Pre-decision window**: Time window before entering decision region
- **Decision boundary**: Geodesic Voronoi boundary between potential goals
- **Boundary crossing**: When animal's nearest goal changes

VTE (Vicarious Trial and Error)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Head sweeping**: Looking back and forth between options (high IdPhi)
- **Pausing**: Slowing down or stopping at the choice point (low speed)
- **IdPhi**: Integrated absolute head rotation - sum of absolute heading changes
- **zIdPhi**: Z-scored IdPhi relative to session baseline
- **VTE index**: Combined measure of head sweeping and slowing

When to Use
-----------
Use this module when analyzing:
- T-maze or Y-maze choice behavior
- Multi-goal navigation (spatial bandit tasks)
- Decision dynamics and commitment points
- VTE detection (hesitation at choice points)

Example
-------
Decision analysis:

>>> from neurospatial.behavior import compute_decision_analysis
>>> result = compute_decision_analysis(
...     env,
...     positions,
...     times,
...     decision_region="center",
...     goal_regions=["left", "right"],
...     pre_window=1.0,
... )  # doctest: +SKIP
>>> if result.pre_decision.heading_circular_variance > 0.5:
...     print("High heading variability - possible deliberation")  # doctest: +SKIP

VTE analysis:

>>> from neurospatial.behavior import compute_vte_session
>>> result = compute_vte_session(
...     positions,
...     times,
...     trials,
...     decision_region="center",
...     window_duration=1.0,
... )  # doctest: +SKIP
>>> print(
...     f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}"
... )  # doctest: +SKIP

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
.. [2] Papale, A. E., et al. (2012). Interplay between hippocampal sharp-wave
       ripple events and vicarious trial and error behaviors. Neuron.
       DOI: 10.1016/j.neuron.2012.10.018
.. [3] Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
       DOI: 10.1038/nrn.2015.30
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.environment import Environment

from neurospatial.stats.circular import wrap_angle

__all__ = [  # noqa: RUF022
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
    # VTE dataclasses
    "VTESessionResult",
    "VTETrialResult",
    # VTE functions
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "integrated_absolute_rotation",
    "normalize_vte_scores",
    # Re-exported from stats.circular
    "wrap_angle",
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
        Actual duration of pre-decision window (seconds).
        May be shorter than requested if trajectory starts late.
    n_samples : int
        Number of samples in window.
    """

    mean_speed: float
    min_speed: float
    heading_mean_direction: float
    heading_circular_variance: float
    heading_mean_resultant_length: float
    window_duration: float
    n_samples: int

    def suggests_deliberation(
        self,
        variance_threshold: float = 0.5,
        speed_threshold: float = 10.0,
    ) -> bool:
        """Check if metrics suggest deliberative behavior.

        Parameters
        ----------
        variance_threshold : float, default=0.5
            Circular variance above this suggests head scanning.
        speed_threshold : float, default=10.0
            Mean speed below this (units/s) suggests slowing.

        Returns
        -------
        bool
            True if high heading variance AND low speed.
        """
        return (
            self.heading_circular_variance > variance_threshold
            and self.mean_speed < speed_threshold
        )


@dataclass(frozen=True)
class DecisionBoundaryMetrics:
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

    @property
    def n_crossings(self) -> int:
        """Number of decision boundary crossings."""
        return len(self.crossing_times)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Decision boundary: {self.n_crossings} crossing"
            f"{'s' if self.n_crossings != 1 else ''}, "
            f"mean distance to boundary: {np.nanmean(self.distance_to_boundary):.1f}"
        )


@dataclass(frozen=True)
class DecisionAnalysisResult:
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

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Decision analysis:",
            f"  Entry time: {self.entry_time:.2f} s",
            f"  Pre-decision: speed={self.pre_decision.mean_speed:.1f}, "
            f"heading_var={self.pre_decision.heading_circular_variance:.2f}",
        ]
        if self.boundary is not None:
            lines.append(f"  {self.boundary.summary()}")
        if self.chosen_goal is not None:
            lines.append(f"  Chosen goal: {self.chosen_goal}")
        else:
            lines.append("  Chosen goal: none (timeout)")
        return "\n".join(lines)


# =============================================================================
# VTE Data Structures
# =============================================================================


@dataclass(frozen=True)
class VTETrialResult:
    """VTE metrics for a single trial.

    Attributes
    ----------
    head_sweep_magnitude : float
        Sum of |delta_theta| in pre-decision window (radians).
        Also known as IdPhi (Integrated absolute Phi).
        High values indicate looking back and forth between options.
    z_head_sweep : float or None
        Z-scored head sweep magnitude relative to session baseline.
        None if session statistics not computed (single trial analysis).
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
    min_speed : float
        Minimum speed in pre-decision window (units/s).
        Near-zero indicates a pause.
    z_speed_inverse : float or None
        Z-scored inverse speed (higher = slower, more VTE-like).
        None if session statistics not computed.
    vte_index : float or None
        Combined VTE index: alpha * z_head_sweep + (1-alpha) * z_speed_inverse.
        None if session statistics not computed.
    is_vte : bool or None
        True if vte_index > threshold. None if not classified.
    window_start : float
        Start time of analysis window (seconds).
    window_end : float
        End time of window (decision region entry time, seconds).
    """

    head_sweep_magnitude: float
    z_head_sweep: float | None
    mean_speed: float
    min_speed: float
    z_speed_inverse: float | None
    vte_index: float | None
    is_vte: bool | None
    window_start: float
    window_end: float

    # Aliases for common terminology
    @property
    def idphi(self) -> float:
        """Alias for head_sweep_magnitude (IdPhi terminology)."""
        return self.head_sweep_magnitude

    @property
    def z_idphi(self) -> float | None:
        """Alias for z_head_sweep (zIdPhi terminology)."""
        return self.z_head_sweep

    def summary(self) -> str:
        """Human-readable summary.

        Returns
        -------
        str
            Formatted string with trial VTE metrics.
        """
        vte_str = (
            "VTE"
            if self.is_vte
            else "non-VTE"
            if self.is_vte is not None
            else "unclassified"
        )
        return (
            f"Trial [{self.window_start:.1f}-{self.window_end:.1f}s]: "
            f"IdPhi={self.head_sweep_magnitude:.2f} rad, "
            f"speed={self.mean_speed:.1f}, {vte_str}"
        )


@dataclass(frozen=True)
class VTESessionResult:
    """VTE analysis for an entire session.

    Attributes
    ----------
    trial_results : list[VTETrialResult]
        Per-trial VTE metrics with z-scores computed.
    mean_head_sweep : float
        Session mean of head sweep magnitude (for z-scoring).
    std_head_sweep : float
        Session std of head sweep magnitude.
    mean_speed : float
        Session mean of pre-decision mean speed.
    std_speed : float
        Session std of pre-decision mean speed.
    n_vte_trials : int
        Number of trials classified as VTE.
    vte_fraction : float
        Fraction of trials classified as VTE.
    """

    trial_results: list[VTETrialResult]
    mean_head_sweep: float
    std_head_sweep: float
    mean_speed: float
    std_speed: float
    n_vte_trials: int
    vte_fraction: float

    # Aliases for common terminology
    @property
    def mean_idphi(self) -> float:
        """Alias for mean_head_sweep (IdPhi terminology)."""
        return self.mean_head_sweep

    @property
    def std_idphi(self) -> float:
        """Alias for std_head_sweep (IdPhi terminology)."""
        return self.std_head_sweep

    def summary(self) -> str:
        """Human-readable summary.

        Returns
        -------
        str
            Formatted string with session VTE summary.
        """
        return (
            f"VTE session: {self.n_vte_trials}/{len(self.trial_results)} "
            f"VTE trials ({self.vte_fraction:.1%})\n"
            f"  Head sweep: mean={self.mean_head_sweep:.2f}, std={self.std_head_sweep:.2f}\n"
            f"  Speed: mean={self.mean_speed:.1f}, std={self.std_speed:.1f}"
        )

    def get_vte_trials(self) -> list[VTETrialResult]:
        """Return only trials classified as VTE.

        Returns
        -------
        list[VTETrialResult]
            List of trials where is_vte is True.
        """
        return [t for t in self.trial_results if t.is_vte]


# =============================================================================
# Pre-Decision Window Functions
# =============================================================================


def decision_region_entry_time(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    region: str,
) -> float:
    """Find time of first entry to a decision region.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to trajectory bins (seconds).
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
    ...     trajectory_bins, times, env, "center"
    ... )  # doctest: +SKIP
    >>> print(f"Entered decision region at t={entry_time:.2f}s")  # doctest: +SKIP
    """
    if region not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Region '{region}' not found in environment. "
            f"Available regions: {available}. "
            f"Add the region using env.regions.add_region()."
        )

    # Get bins in the region
    region_bins = env.bins_in_region(region)
    region_bin_set = set(region_bins)

    # Find first entry
    for i, bin_idx in enumerate(trajectory_bins):
        if bin_idx in region_bin_set:
            return float(times[i])

    raise ValueError(
        f"Trajectory never enters region '{region}'. "
        f"Check that the trajectory passes through the decision region, "
        f"or verify region bounds with env.regions['{region}']."
    )


def extract_pre_decision_window(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract trajectory segment before decision region entry.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps (seconds).
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to extract (seconds).

    Returns
    -------
    window_positions : NDArray[np.float64]
        Positions within the pre-decision window.
    window_times : NDArray[np.float64]
        Times within the pre-decision window.

    Notes
    -----
    If the requested window extends before the trajectory start,
    the returned window will be shorter than requested.

    Examples
    --------
    >>> window_pos, window_times = extract_pre_decision_window(
    ...     positions, times, entry_time=5.0, window_duration=2.0
    ... )  # doctest: +SKIP
    >>> # Returns data from t=3.0 to t<5.0
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    window_start = entry_time - window_duration
    window_start = max(window_start, times[0])  # Clamp to trajectory start

    # Select samples in window (before entry)
    mask = (times >= window_start) & (times < entry_time)

    return positions[mask], times[mask]


def pre_decision_heading_stats(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> tuple[float, float, float]:
    """Compute circular statistics on heading in a trajectory window.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded from statistics.

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

    Notes
    -----
    Circular statistics are computed directly:

    - mean_resultant_length = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2)
    - circular_variance = 1 - mean_resultant_length
    - mean_direction = atan2(mean(sin(theta)), mean(cos(theta)))

    If all samples are stationary (below min_speed), returns:
    mean_direction=0, circular_variance=1, mean_resultant_length=0.

    Examples
    --------
    >>> mean_dir, circ_var, mrl = pre_decision_heading_stats(
    ...     positions, times, min_speed=5.0
    ... )  # doctest: +SKIP
    >>> if circ_var > 0.5:
    ...     print("High heading variability")  # doctest: +SKIP
    """
    from neurospatial.ops.egocentric import heading_from_velocity

    positions = np.asarray(positions)
    times = np.asarray(times)

    if len(positions) < 2:
        return 0.0, 1.0, 0.0

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    # Filter out NaN (stationary periods)
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) == 0:
        # No valid headings: undefined direction, max variance
        return 0.0, 1.0, 0.0

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
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute speed statistics for a trajectory window.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).

    Returns
    -------
    mean_speed : float
        Mean instantaneous speed (units/s).
    min_speed : float
        Minimum instantaneous speed (units/s).

    Examples
    --------
    >>> mean_speed, min_speed = pre_decision_speed_stats(
    ...     positions, times
    ... )  # doctest: +SKIP
    >>> if min_speed < 1.0:
    ...     print("Animal paused during pre-decision window")  # doctest: +SKIP
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    if len(positions) < 2:
        return 0.0, 0.0

    # Compute velocities
    dt = np.diff(times)
    displacement = np.diff(positions, axis=0)
    velocity = displacement / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocity, axis=1)

    return float(np.mean(speeds)), float(np.min(speeds))


def compute_pre_decision_metrics(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float = 5.0,
) -> PreDecisionMetrics:
    """Compute all pre-decision window metrics.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps (seconds).
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to analyze (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

    Returns
    -------
    PreDecisionMetrics
        Dataclass containing all pre-decision metrics.

    Examples
    --------
    >>> metrics = compute_pre_decision_metrics(
    ...     positions, times, entry_time=5.0, window_duration=2.0
    ... )  # doctest: +SKIP
    >>> if metrics.suggests_deliberation():
    ...     print("Possible VTE behavior")  # doctest: +SKIP
    """
    # Extract window
    window_pos, window_times = extract_pre_decision_window(
        positions, times, entry_time, window_duration
    )

    # Handle edge case of empty or too-short window
    if len(window_pos) < 2:
        return PreDecisionMetrics(
            mean_speed=0.0,
            min_speed=0.0,
            heading_mean_direction=0.0,
            heading_circular_variance=1.0,
            heading_mean_resultant_length=0.0,
            window_duration=0.0,
            n_samples=len(window_pos),
        )

    # Compute heading stats
    mean_dir, circ_var, mrl = pre_decision_heading_stats(
        window_pos, window_times, min_speed=min_speed
    )

    # Compute speed stats
    mean_speed, min_speed_val = pre_decision_speed_stats(window_pos, window_times)

    # Actual window duration
    actual_duration = (
        float(window_times[-1] - window_times[0]) if len(window_times) > 1 else 0.0
    )

    return PreDecisionMetrics(
        mean_speed=mean_speed,
        min_speed=min_speed_val,
        heading_mean_direction=mean_dir,
        heading_circular_variance=circ_var,
        heading_mean_resultant_length=mrl,
        window_duration=actual_duration,
        n_samples=len(window_pos),
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
    >>> left_bin = env.bin_at([10, 55])  # doctest: +SKIP
    >>> right_bin = env.bin_at([90, 55])  # doctest: +SKIP
    >>> labels = geodesic_voronoi_labels(env, [left_bin, right_bin])  # doctest: +SKIP
    >>> # labels[i] == 0 means bin i is closer to left goal
    >>> # labels[i] == 1 means bin i is closer to right goal
    """
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = np.asarray(goal_bins)
    n_goals = len(goal_bins_arr)
    n_bins = env.n_bins

    # Compute distance field from each goal
    distances = np.full((n_goals, n_bins), np.inf)
    for i, goal_bin in enumerate(goal_bins_arr):
        # Ensure goal_bin is a Python int, not numpy scalar
        goal_bin_int = (
            int(goal_bin.item()) if hasattr(goal_bin, "item") else int(goal_bin)
        )
        distances[i] = distance_field(
            env.connectivity, [goal_bin_int], metric="geodesic"
        )

    # Label by nearest goal
    labels = np.argmin(distances, axis=0)

    # Mark unreachable bins
    min_distances = np.min(distances, axis=0)
    labels_int: NDArray[np.int_] = labels.astype(np.int_)
    labels_int[np.isinf(min_distances)] = -1

    return labels_int


def distance_to_decision_boundary(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    goal_bins: list[int] | NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute distance to nearest decision boundary for each trajectory point.

    The decision boundary is the Voronoi edge between goal regions - the set
    of points equidistant (geodesically) from multiple goals.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
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
    ...     env, trajectory_bins, goal_bins
    ... )  # doctest: +SKIP
    >>> commitment_mask = distances > 20.0  # Committed to a goal
    """
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = np.asarray(goal_bins)
    n_goals = len(goal_bins_arr)

    if n_goals < 2:
        # With single goal, there's no boundary - return infinity
        return np.full(len(trajectory_bins), np.inf)

    # Compute distance from each goal to all bins
    all_distances = np.zeros((n_goals, env.n_bins))
    for i, goal_bin in enumerate(goal_bins_arr):
        # Ensure goal_bin is a Python int, not numpy scalar
        goal_bin_int = (
            int(goal_bin.item()) if hasattr(goal_bin, "item") else int(goal_bin)
        )
        all_distances[i] = distance_field(
            env.connectivity, [goal_bin_int], metric="geodesic"
        )

    # For each trajectory bin, compute distance to boundary
    # Boundary distance = |d1 - d2| where d1, d2 are distances to two nearest goals
    boundary_distances = np.zeros(len(trajectory_bins))

    for i, bin_idx in enumerate(trajectory_bins):
        if bin_idx < 0 or bin_idx >= env.n_bins:
            boundary_distances[i] = np.nan
            continue

        # Get distances to all goals from this bin
        dists = all_distances[:, bin_idx]

        # Sort to find two nearest
        sorted_dists = np.sort(dists)

        # Boundary distance is difference between two nearest goals
        if np.isinf(sorted_dists[0]):
            boundary_distances[i] = np.inf
        else:
            # Extract scalars properly to avoid numpy deprecation warning
            d1 = float(
                sorted_dists[0].item()
                if hasattr(sorted_dists[0], "item")
                else sorted_dists[0]
            )
            d2 = float(
                sorted_dists[1].item()
                if hasattr(sorted_dists[1], "item")
                else sorted_dists[1]
            )
            boundary_distances[i] = d2 - d1

    return boundary_distances


def detect_boundary_crossings(
    trajectory_bins: NDArray[np.int_],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
) -> tuple[list[float], list[tuple[int, int]]]:
    """Detect when trajectory crosses decision boundaries.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along the trajectory.
    voronoi_labels : NDArray[np.int_], shape (n_bins,)
        Voronoi label for each bin (from geodesic_voronoi_labels).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).

    Returns
    -------
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary.
    crossing_directions : list[tuple[int, int]]
        (from_goal_idx, to_goal_idx) for each crossing.

    Examples
    --------
    >>> crossing_times, directions = detect_boundary_crossings(
    ...     trajectory_bins, voronoi_labels, times
    ... )  # doctest: +SKIP
    >>> print(f"Animal crossed boundary {len(crossing_times)} times")  # doctest: +SKIP
    """
    trajectory_bins = np.asarray(trajectory_bins)
    times = np.asarray(times)

    # Get label for each trajectory point
    trajectory_labels = voronoi_labels[trajectory_bins]

    crossing_times: list[float] = []
    crossing_directions: list[tuple[int, int]] = []

    for i in range(1, len(trajectory_labels)):
        prev_label = trajectory_labels[i - 1]
        curr_label = trajectory_labels[i]

        # Skip if either is unreachable (-1)
        if prev_label == -1 or curr_label == -1:
            continue

        # Check for crossing
        if prev_label != curr_label:
            # Interpolate crossing time (assume it happened at midpoint)
            crossing_time = (times[i - 1] + times[i]) / 2
            crossing_times.append(float(crossing_time))
            crossing_directions.append((int(prev_label), int(curr_label)))

    return crossing_times, crossing_directions


# =============================================================================
# Composite Decision Analysis Function
# =============================================================================


def compute_decision_analysis(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    decision_region: str,
    goal_regions: list[str],
    *,
    pre_window: float = 1.0,
    min_speed: float = 5.0,
) -> DecisionAnalysisResult:
    """Compute complete decision analysis for a trajectory.

    Parameters
    ----------
    env : Environment
        Spatial environment with region definitions.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    decision_region : str
        Name of decision region in env.regions (e.g., "center" for T-maze).
    goal_regions : list[str]
        Names of goal regions (e.g., ["left", "right"]).
    pre_window : float, default=1.0
        Duration of pre-decision window to analyze (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

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
        If positions and times have different lengths.

    Examples
    --------
    >>> result = compute_decision_analysis(
    ...     env,
    ...     positions,
    ...     times,
    ...     decision_region="center",
    ...     goal_regions=["left", "right"],
    ... )  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    # Validate inputs
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    # Validate regions exist
    if decision_region not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Decision region '{decision_region}' not found in environment. "
            f"Available regions: {available}."
        )

    for goal_region in goal_regions:
        if goal_region not in env.regions:
            available = list(env.regions.keys())
            raise ValueError(
                f"Goal region '{goal_region}' not found in environment. "
                f"Available regions: {available}."
            )

    # Get trajectory bins
    trajectory_bins = env.bin_at(positions)

    # Find entry time to decision region
    entry_time = decision_region_entry_time(
        trajectory_bins, times, env, decision_region
    )

    # Compute pre-decision metrics
    pre_decision = compute_pre_decision_metrics(
        positions, times, entry_time, pre_window, min_speed=min_speed
    )

    # Compute boundary metrics
    # Get the representative bin for each goal region (first bin in region)
    goal_bins = []
    for r in goal_regions:
        bins = env.bins_in_region(r)
        if len(bins) > 0:
            goal_bins.append(int(bins[0]))
        else:
            raise ValueError(
                f"Goal region '{r}' contains no bins. "
                f"Check that the region is within the environment bounds."
            )
    voronoi_labels = geodesic_voronoi_labels(env, goal_bins)

    trajectory_labels = voronoi_labels[trajectory_bins]
    boundary_distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)
    crossing_times, crossing_directions = detect_boundary_crossings(
        trajectory_bins, voronoi_labels, times
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
        for bin_idx in reversed(trajectory_bins):
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


# =============================================================================
# VTE Core Functions
# =============================================================================


def head_sweep_magnitude(headings: NDArray[np.float64]) -> float:
    """Compute integrated absolute head rotation (IdPhi).

    Sums the absolute value of heading changes across a trajectory window.
    High values indicate back-and-forth head movements ("scanning").

    Parameters
    ----------
    headings : NDArray[np.float64], shape (n_samples,)
        Heading angles in radians.

    Returns
    -------
    float
        Sum of absolute heading changes in radians.
        Returns 0.0 if fewer than 2 valid samples.

    Notes
    -----
    This is the core VTE metric from Papale et al. (2012) and Redish (2016).
    Also known as "IdPhi" (integrated absolute dphi/dt * dt).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.decisions import head_sweep_magnitude
    >>> # Constant heading: no head sweep
    >>> head_sweep_magnitude(np.zeros(10))
    0.0
    >>> # Alternating headings: high head sweep
    >>> headings = np.array([0, np.pi / 4, 0, np.pi / 4, 0])
    >>> head_sweep_magnitude(headings)  # 4 * pi/4 = pi
    3.141592653589793
    """
    # Filter NaN values
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) < 2:
        return 0.0

    delta = wrap_angle(np.diff(valid_headings))
    return float(np.sum(np.abs(delta)))


# Alias for backward compatibility and paper terminology
integrated_absolute_rotation = head_sweep_magnitude


def head_sweep_from_positions(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> float:
    """Compute head sweep magnitude (IdPhi) from position trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded.

    Returns
    -------
    float
        Sum of absolute heading changes (radians).
        Returns 0.0 if fewer than 2 valid heading samples.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.decisions import head_sweep_from_positions
    >>> # Straight line trajectory (low head sweep)
    >>> times = np.linspace(0, 2, 20)
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.ones(20) * 50])
    >>> head_sweep_from_positions(positions, times, min_speed=1.0)  # doctest: +SKIP
    0.0
    """
    from neurospatial.ops.egocentric import heading_from_velocity

    if len(positions) < 2:
        return 0.0

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    return head_sweep_magnitude(headings)


# =============================================================================
# VTE Z-Scoring Functions
# =============================================================================


def normalize_vte_scores(
    head_sweeps: NDArray[np.float64],
    speeds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score VTE metrics across trials.

    Parameters
    ----------
    head_sweeps : NDArray[np.float64], shape (n_trials,)
        Head sweep magnitude for each trial.
    speeds : NDArray[np.float64], shape (n_trials,)
        Mean speed for each trial.

    Returns
    -------
    z_head_sweeps : NDArray[np.float64], shape (n_trials,)
        Z-scored head sweeps.
    z_speed_inverse : NDArray[np.float64], shape (n_trials,)
        Z-scored inverse speed (higher = slower = more VTE-like).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Warns
    -----
    UserWarning
        If std is zero (no variation across trials), z-scores will be 0.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.decisions import normalize_vte_scores
    >>> head_sweeps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> speeds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> z_hs, z_sp = normalize_vte_scores(head_sweeps, speeds)
    >>> np.abs(np.mean(z_hs)) < 1e-10  # Mean is ~0
    True
    """
    if len(head_sweeps) != len(speeds):
        raise ValueError(
            f"head_sweeps and speeds must have same length. "
            f"Got {len(head_sweeps)} and {len(speeds)}."
        )

    # Z-score head sweeps
    mean_hs = np.mean(head_sweeps)
    std_hs = np.std(head_sweeps)
    if std_hs < 1e-10:
        warnings.warn(
            "No variation in head sweep magnitude across trials (std=0). "
            "All trials have identical head sweep behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_head_sweeps = np.zeros_like(head_sweeps)
    else:
        z_head_sweeps = (head_sweeps - mean_hs) / std_hs

    # Z-score inverse speed (invert so slower = higher score)
    mean_spd = np.mean(speeds)
    std_spd = np.std(speeds)
    if std_spd < 1e-10:
        warnings.warn(
            "No variation in speed across trials (std=0). "
            "All trials have identical speed behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_speed_inverse = np.zeros_like(speeds)
    else:
        # Invert: higher speed -> lower z, lower speed -> higher z
        z_speed_inverse = -(speeds - mean_spd) / std_spd

    return z_head_sweeps, z_speed_inverse


# =============================================================================
# VTE Classification Functions
# =============================================================================


def compute_vte_index(
    z_head_sweep: float,
    z_speed_inv: float,
    *,
    alpha: float = 0.5,
) -> float:
    """Compute combined VTE index.

    Parameters
    ----------
    z_head_sweep : float
        Z-scored head sweep magnitude.
    z_speed_inv : float
        Z-scored inverse speed.
    alpha : float, default=0.5
        Weight for head sweep (1-alpha for speed).
        Default 0.5 weights both equally.

    Returns
    -------
    float
        Combined VTE index.

    Examples
    --------
    >>> from neurospatial.behavior.decisions import compute_vte_index
    >>> compute_vte_index(1.0, 1.0, alpha=0.5)
    1.0
    >>> compute_vte_index(2.0, 0.0, alpha=1.0)  # Head sweep only
    2.0
    """
    return alpha * z_head_sweep + (1 - alpha) * z_speed_inv


def classify_vte(
    vte_index: float,
    *,
    threshold: float = 0.5,
) -> bool:
    """Classify trial as VTE based on VTE index.

    Parameters
    ----------
    vte_index : float
        Combined VTE index.
    threshold : float, default=0.5
        Classification threshold.
        Trial is VTE if vte_index > threshold.

    Returns
    -------
    bool
        True if trial is classified as VTE.

    Examples
    --------
    >>> from neurospatial.behavior.decisions import classify_vte
    >>> classify_vte(1.0, threshold=0.5)
    True
    >>> classify_vte(0.3, threshold=0.5)
    False
    """
    return vte_index > threshold


# =============================================================================
# VTE Composite Functions
# =============================================================================


def compute_vte_trial(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float = 5.0,
) -> VTETrialResult:
    """Compute VTE metrics for a single trial.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire trajectory.
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

    Returns
    -------
    VTETrialResult
        VTE metrics for the trial.
        Note: z-scores and classification are None for single trial analysis.
        Use compute_vte_session() for session-level analysis with z-scoring.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.decisions import compute_vte_trial
    >>> times = np.linspace(0, 3, 90)
    >>> positions = np.column_stack([np.linspace(0, 50, 90), np.ones(90) * 50])
    >>> result = compute_vte_trial(
    ...     positions, times, entry_time=2.0, window_duration=1.0
    ... )
    >>> result.window_start
    1.0
    >>> result.window_end
    2.0
    """
    # Extract pre-decision window
    window_positions, window_times = extract_pre_decision_window(
        positions, times, entry_time, window_duration
    )

    window_start = entry_time - window_duration
    window_end = entry_time

    # Compute head sweep magnitude
    head_sweep = head_sweep_from_positions(
        window_positions, window_times, min_speed=min_speed
    )

    # Compute speed statistics
    if len(window_positions) < 2:
        mean_spd = 0.0
        min_spd = 0.0
    else:
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocity, axis=1)
        mean_spd = float(np.mean(speeds))
        min_spd = float(np.min(speeds))

    return VTETrialResult(
        head_sweep_magnitude=head_sweep,
        z_head_sweep=None,  # Not computed for single trial
        mean_speed=mean_spd,
        min_speed=min_spd,
        z_speed_inverse=None,
        vte_index=None,
        is_vte=None,
        window_start=window_start,
        window_end=window_end,
    )


def compute_vte_session(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    trials: list[Trial],
    decision_region: str,
    env: Environment,
    *,
    window_duration: float = 1.0,
    min_speed: float = 5.0,
    alpha: float = 0.5,
    vte_threshold: float = 0.5,
) -> VTESessionResult:
    """Compute VTE metrics for all trials in a session.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire session.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    trials : list[Trial]
        Trial segmentation from segment_trials().
    decision_region : str
        Name of decision region in env.regions.
    env : Environment
        Environment with region definitions.
    window_duration : float, default=1.0
        Duration of pre-decision window in seconds.
        Typical values: 0.5-2.0s depending on maze size and task.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
    alpha : float, default=0.5
        Weight for head sweep in VTE index (1-alpha for speed).
        Default 0.5 weights both equally.
    vte_threshold : float, default=0.5
        Threshold for VTE classification.
        Trial is VTE if vte_index > threshold.

    Returns
    -------
    VTESessionResult
        Session-level VTE analysis with per-trial metrics.

    Examples
    --------
    >>> from neurospatial.behavior import compute_vte_session
    >>> result = compute_vte_session(
    ...     positions,
    ...     times,
    ...     trials,
    ...     decision_region="center",
    ...     env=env,
    ...     window_duration=1.0,
    ... )  # doctest: +SKIP
    >>> print(
    ...     f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}"
    ... )  # doctest: +SKIP
    """
    # First pass: compute raw metrics for all trials
    raw_head_sweeps: list[float] = []
    raw_speeds: list[float] = []
    trial_windows: list[tuple[float, float]] = []  # (window_start, window_end)
    raw_min_speeds: list[float] = []

    for trial in trials:
        # Get entry time to decision region
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_positions = positions[mask]
        trial_times = times[mask]

        if len(trial_positions) == 0:
            continue

        trial_bins = env.bin_at(trial_positions)

        try:
            entry_time = decision_region_entry_time(
                trial_bins, trial_times, env, decision_region
            )
        except ValueError:
            # Trial never enters decision region - skip
            continue

        # Extract pre-decision window
        window_positions, window_times = extract_pre_decision_window(
            positions, times, entry_time, window_duration
        )

        if len(window_positions) < 3:
            # Not enough samples for heading analysis - skip
            continue

        # Compute head sweep magnitude
        head_sweep = head_sweep_from_positions(
            window_positions, window_times, min_speed=min_speed
        )

        # Compute speed statistics
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocity, axis=1)
        mean_spd = float(np.mean(speeds))
        min_spd = float(np.min(speeds))

        raw_head_sweeps.append(head_sweep)
        raw_speeds.append(mean_spd)
        raw_min_speeds.append(min_spd)
        trial_windows.append((entry_time - window_duration, entry_time))

    # Convert to arrays
    head_sweeps_arr = np.array(raw_head_sweeps)
    speeds_arr = np.array(raw_speeds)

    if len(head_sweeps_arr) == 0:
        # No valid trials
        return VTESessionResult(
            trial_results=[],
            mean_head_sweep=0.0,
            std_head_sweep=0.0,
            mean_speed=0.0,
            std_speed=0.0,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

    # Compute session statistics
    mean_hs = float(np.mean(head_sweeps_arr))
    std_hs = float(np.std(head_sweeps_arr))
    mean_spd = float(np.mean(speeds_arr))
    std_spd = float(np.std(speeds_arr))

    # Second pass: compute z-scores and classify
    z_head_sweeps, z_speed_inv = normalize_vte_scores(head_sweeps_arr, speeds_arr)

    trial_results: list[VTETrialResult] = []
    n_vte = 0

    for i in range(len(head_sweeps_arr)):
        # Compute VTE index
        vte_idx = compute_vte_index(
            float(z_head_sweeps[i]), float(z_speed_inv[i]), alpha=alpha
        )

        # Classify
        is_vte = classify_vte(vte_idx, threshold=vte_threshold)
        if is_vte:
            n_vte += 1

        window_start, window_end = trial_windows[i]

        trial_results.append(
            VTETrialResult(
                head_sweep_magnitude=head_sweeps_arr[i],
                z_head_sweep=float(z_head_sweeps[i]),
                mean_speed=speeds_arr[i],
                min_speed=raw_min_speeds[i],
                z_speed_inverse=float(z_speed_inv[i]),
                vte_index=float(vte_idx),
                is_vte=is_vte,
                window_start=window_start,
                window_end=window_end,
            )
        )

    return VTESessionResult(
        trial_results=trial_results,
        mean_head_sweep=mean_hs,
        std_head_sweep=std_hs,
        mean_speed=mean_spd,
        std_speed=std_spd,
        n_vte_trials=n_vte,
        vte_fraction=n_vte / len(trial_results) if trial_results else 0.0,
    )
