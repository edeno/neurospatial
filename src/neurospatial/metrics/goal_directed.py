"""Goal-directed navigation metrics.

This module measures how directly an animal navigates toward a goal,
including instantaneous alignment, overall bias, and approach dynamics.

Key Concepts
------------
- **Goal alignment**: Cosine similarity between movement and goal direction.
  +1 = moving toward goal, -1 = moving away, 0 = orthogonal.
- **Goal bias**: Average alignment over trajectory. Positive = net approach.
- **Approach rate**: Rate of distance decrease toward goal (units/s).

Example
-------
>>> from neurospatial.metrics import compute_goal_directed_metrics
>>> result = compute_goal_directed_metrics(env, positions, times, goal)
>>> print(f"Goal bias: {result.goal_bias:.2f}")  # Range [-1, 1]
>>> if result.goal_bias > 0.5:
...     print("Strong goal-directed navigation")

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
       DOI: 10.1523/JNEUROSCI.3761-07.2007
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "GoalDirectedMetrics",
    "approach_rate",
    "compute_goal_directed_metrics",
    "goal_bias",
    "goal_direction",
    "goal_vector",
    "instantaneous_goal_alignment",
]


@dataclass(frozen=True)
class GoalDirectedMetrics:
    """Goal-directed navigation metrics for a trajectory.

    Attributes
    ----------
    goal_bias : float
        Mean instantaneous goal alignment, range [-1, 1].
        Interpretation:
        - > 0.5: Strong goal-directed navigation
        - 0 to 0.5: Weak goal-directed with some wandering
        - < 0: Net movement away from goal
    mean_approach_rate : float
        Mean rate of distance change in environment units per second.
        Negative values indicate approaching the goal.
        Interpretation: -10 cm/s means closing 10 cm per second on average.
    time_to_goal : float or None
        Time until goal region entered (seconds). None if goal not reached.
    min_distance_to_goal : float
        Closest approach to goal during trajectory, in environment units.
    goal_distance_at_start : float
        Distance to goal at trajectory start, in environment units.
    goal_distance_at_end : float
        Distance to goal at trajectory end, in environment units.
    goal_position : NDArray[np.float64]
        Goal position used for computation, shape (n_dims,).
    metric : str
        Distance metric used ("geodesic" or "euclidean").
    """

    goal_bias: float
    mean_approach_rate: float
    time_to_goal: float | None
    min_distance_to_goal: float
    goal_distance_at_start: float
    goal_distance_at_end: float
    goal_position: NDArray[np.float64]
    metric: str

    def is_goal_directed(self, threshold: float = 0.3) -> bool:
        """Return True if goal bias exceeds threshold.

        Parameters
        ----------
        threshold : float, default=0.3
            Goal bias threshold. Default 0.3 is a moderate threshold.

        Returns
        -------
        bool
            True if goal_bias > threshold.
        """
        return self.goal_bias > threshold

    def summary(self) -> str:
        """Human-readable summary for printing.

        Returns
        -------
        str
            Formatted string with goal-directed metrics.
        """
        lines = [
            "Goal-directed metrics:",
            f"  Goal bias: {self.goal_bias:.2f} (range [-1, 1])",
            f"  Approach rate: {self.mean_approach_rate:.1f} units/s",
            f"  Distance: {self.goal_distance_at_start:.1f} -> "
            f"{self.goal_distance_at_end:.1f} units",
            f"  Closest approach: {self.min_distance_to_goal:.1f} units",
        ]
        if self.time_to_goal is not None:
            lines.append(f"  Time to goal: {self.time_to_goal:.2f} s")
        else:
            lines.append("  Time to goal: not reached")
        return "\n".join(lines)


def goal_vector(
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute vector from each position to goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame (same coordinate system).
        Must have same number of dimensions as positions.

    Returns
    -------
    NDArray[np.float64], shape (n_samples, n_dims)
        Vector from each position to goal.

    Raises
    ------
    ValueError
        If goal dimensions don't match positions dimensions.

    Examples
    --------
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    >>> goal = np.array([50.0, 0.0])
    >>> goal_vector(positions, goal)
    array([[50., 0.], [40., 0.]])
    """
    goal = np.asarray(goal)
    positions = np.asarray(positions)

    if goal.shape[0] != positions.shape[1]:
        raise ValueError(
            f"Goal has {goal.shape[0]} dimensions but positions have "
            f"{positions.shape[1]} dimensions. Both must match."
        )

    return goal[np.newaxis, :] - positions


def goal_direction(
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute direction (angle) from each position to goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Angle in radians from each position to goal.
        Uses allocentric convention: 0=East, pi/2=North.

    Examples
    --------
    >>> positions = np.array([[0.0, 0.0]])
    >>> goal = np.array([1.0, 0.0])  # East
    >>> goal_direction(positions, goal)
    array([0.])
    """
    vec = goal_vector(positions, goal)
    return np.arctan2(vec[:, 1], vec[:, 0])


def instantaneous_goal_alignment(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> NDArray[np.float64]:
    """Compute instantaneous alignment between movement and goal direction.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float, default=5.0
        Minimum speed threshold in environment units per second.
        Samples below this speed are masked as NaN (stationary periods
        have undefined movement direction).

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of angle between velocity and goal direction.
        Range [-1, 1]. NaN for stationary periods (speed < min_speed).

    Examples
    --------
    >>> # Moving East toward goal at East
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> times = np.linspace(0, 10, 21)
    >>> goal = np.array([100.0, 0.0])
    >>> alignment = instantaneous_goal_alignment(positions, times, goal, min_speed=0.0)
    >>> np.nanmean(alignment) > 0.9  # High alignment
    True
    """
    from neurospatial.reference_frames import heading_from_velocity

    positions = np.asarray(positions)
    times = np.asarray(times)
    goal = np.asarray(goal)

    if len(positions) < 2:
        return np.full(len(positions), np.nan)

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Compute velocity heading
    velocity_heading = heading_from_velocity(positions, dt, min_speed=min_speed)

    # Compute goal direction at each position
    goal_heading = goal_direction(positions, goal)

    # Compute alignment as cos(velocity_heading - goal_heading)
    angle_diff = velocity_heading - goal_heading
    alignment = np.cos(angle_diff)

    return alignment


def goal_bias(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> float:
    """Compute mean alignment toward goal over trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float, default=5.0
        Minimum speed threshold. Stationary periods excluded.

    Returns
    -------
    float
        Mean goal alignment, range [-1, 1].
        Returns NaN if all samples are stationary or trajectory has < 2 points.

    Notes
    -----
    Interpretation:
    - > 0.5: Strong goal-directed navigation
    - 0 to 0.5: Weak goal-directed with some wandering
    - < 0: Net movement away from goal

    Examples
    --------
    >>> # Direct approach to goal
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> times = np.linspace(0, 10, 21)
    >>> goal = np.array([100.0, 0.0])
    >>> goal_bias(positions, times, goal, min_speed=0.0) > 0.8
    True
    """
    alignment = instantaneous_goal_alignment(
        positions, times, goal, min_speed=min_speed
    )

    # Compute mean ignoring NaN
    valid_alignment = alignment[~np.isnan(alignment)]

    if len(valid_alignment) == 0:
        return np.nan

    return float(np.mean(valid_alignment))


def approach_rate(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute rate of distance change toward goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric. Geodesic requires env parameter.
    env : Environment, optional
        Required when metric="geodesic".

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Rate of distance change (d(distance)/dt) in units per second.
        Negative values indicate approaching the goal.
        First value is NaN (no prior sample for derivative).

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.

    Examples
    --------
    >>> # Moving toward goal
    >>> positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
    >>> times = np.linspace(0, 5, 11)
    >>> goal = np.array([100.0, 0.0])
    >>> rates = approach_rate(positions, times, goal)
    >>> np.nanmean(rates) < 0  # Negative = approaching
    True
    """
    positions = np.asarray(positions)
    times = np.asarray(times)
    goal = np.asarray(goal)

    if metric == "geodesic" and env is None:
        raise ValueError(
            "env parameter is required when metric='geodesic'. "
            "Provide the Environment instance, or use metric='euclidean' "
            "for straight-line distances."
        )

    # Compute distance to goal at each timepoint
    if metric == "euclidean":
        goal_vec = goal_vector(positions, goal)
        distances = np.linalg.norm(goal_vec, axis=1)
    else:
        # Geodesic distance - env is guaranteed non-None by validation above
        from neurospatial.behavioral import distance_to_region

        assert env is not None  # Type guard for mypy
        trajectory_bins = env.bin_at(positions)
        goal_bin = env.bin_at(goal)
        distances = distance_to_region(
            env, trajectory_bins, goal_bin, metric="geodesic"
        )

    # Compute derivative (rate of change)
    dt = np.diff(times)
    d_distance = np.diff(distances)

    rates = np.full(len(positions), np.nan)
    rates[1:] = d_distance / dt

    return rates


def compute_goal_directed_metrics(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    min_speed: float = 5.0,
    goal_radius: float | None = None,
) -> GoalDirectedMetrics:
    """Compute comprehensive goal-directed navigation metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="euclidean"
        Distance metric for approach rate and distance computations.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
    goal_radius : float, optional
        Radius for goal arrival detection. If provided, time_to_goal
        is computed when animal comes within this distance of goal.
        If None, time_to_goal is None.

    Returns
    -------
    GoalDirectedMetrics
        Dataclass containing all goal-directed metrics.

    Raises
    ------
    ValueError
        If positions and times have different lengths.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import compute_goal_directed_metrics
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> result = compute_goal_directed_metrics(env, positions, times, goal)
    >>> print(result.summary())
    """
    positions = np.asarray(positions)
    times = np.asarray(times)
    goal = np.asarray(goal)

    # Validate inputs
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    # Compute goal bias
    bias = goal_bias(positions, times, goal, min_speed=min_speed)

    # Compute approach rate
    rates = approach_rate(positions, times, goal, metric=metric, env=env)
    mean_rate = float(np.nanmean(rates))

    # Compute distance to goal
    goal_vec = goal_vector(positions, goal)
    distances = np.linalg.norm(goal_vec, axis=1)

    min_dist = float(np.min(distances))
    start_dist = float(distances[0])
    end_dist = float(distances[-1])

    # Compute time to goal if radius specified
    time_to_goal: float | None = None
    if goal_radius is not None:
        arrival_mask = distances <= goal_radius
        if np.any(arrival_mask):
            arrival_idx = np.argmax(arrival_mask)
            time_to_goal = float(times[arrival_idx] - times[0])

    return GoalDirectedMetrics(
        goal_bias=bias,
        mean_approach_rate=mean_rate,
        time_to_goal=time_to_goal,
        min_distance_to_goal=min_dist,
        goal_distance_at_start=start_dist,
        goal_distance_at_end=end_dist,
        goal_position=goal.copy(),
        metric=metric,
    )
