"""Behavioral analysis and goal-directed navigation metrics.

This module provides functions for analyzing behavioral trajectories in spatial
environments, with a focus on goal-directed navigation and reinforcement learning
metrics. Functions include:

- Path progress and distance-to-goal computations
- Cost-aware navigation with terrain difficulty
- Trajectory curvature and turn sequence analysis
- Time-to-goal and temporal dynamics

These metrics are designed for neuroscience applications such as:
- Place cell analysis with behavioral covariates
- GLM regressors for neural decoding
- Navigation strategy classification
- Spatial value function estimation

All functions are fully vectorized for performance on large datasets (100k+ timepoints).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.segmentation import Trial


def trials_to_region_arrays(
    trials: list[Trial],
    times: NDArray[np.float64],
    env: Environment,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Convert trial list to per-timepoint start/goal bin arrays.

    Helper function to construct arrays for path_progress() and similar
    functions. Encapsulates the trial-to-timepoint mapping logic.

    Parameters
    ----------
    trials : list[Trial]
        Trial segmentation from segment_trials().
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    env : Environment
        Spatial environment (for region lookups).

    Returns
    -------
    start_bins : NDArray[np.int_], shape (n_samples,)
        Start bin index at each timepoint (-1 outside trials).
    goal_bins : NDArray[np.int_], shape (n_samples,)
        Goal bin index at each timepoint (-1 outside trials).

    Notes
    -----
    This function has a small loop over trials (typically 10-100), but
    avoids looping over timepoints (typically 100k+). The returned arrays
    can be passed to vectorized functions like path_progress().

    Examples
    --------
    >>> trials = segment_trials(trajectory_bins, times, env, ...)
    >>> start_bins, goal_bins = trials_to_region_arrays(trials, times, env)
    >>> progress = path_progress(env, trajectory_bins, start_bins, goal_bins)
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins)

    See Also
    --------
    path_progress : Compute normalized path progress
    distance_to_region : Distance to target region
    """
    # Initialize arrays with -1 (invalid bin)
    start_bins = np.full(len(times), -1, dtype=np.int_)
    goal_bins = np.full(len(times), -1, dtype=np.int_)

    # Loop over trials (small - typically 10-100)
    for trial in trials:
        # Create mask for timepoints within this trial
        mask = (times >= trial.start_time) & (times <= trial.end_time)

        # Get bins for start region
        start_region_bins = env.bins_in_region(trial.start_region)
        if len(start_region_bins) > 0:
            # Use first bin if multiple bins in region
            start_bins[mask] = start_region_bins[0]

        # Get bins for end region (handle None for failed trials)
        if trial.end_region is not None:
            end_region_bins = env.bins_in_region(trial.end_region)
            if len(end_region_bins) > 0:
                # Use first bin if multiple bins in region
                goal_bins[mask] = end_region_bins[0]
        # For failed trials (end_region=None), goal_bins remains -1

    return start_bins, goal_bins


def path_progress(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    start_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute normalized path progress from start to goal (0 → 1).

    Fully vectorized computation over entire session. For each timepoint t:
        progress[t] = distance(start_bins[t], trajectory_bins[t]) /
                      distance(start_bins[t], goal_bins[t])

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Current bin index at each timepoint.
    start_bins : NDArray[np.int_], shape (n_samples,)
        Start bin for each timepoint. Can be constant (single trial) or
        vary per-timepoint (multiple trials).
    goal_bins : NDArray[np.int_], shape (n_samples,)
        Goal bin for each timepoint. Can be constant or vary per-timepoint.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Path progress from 0 to 1 at each timepoint. NaN for invalid bins
        or disconnected paths.

    Notes
    -----
    This function is fully vectorized - no loops over timepoints. The only
    loop required is constructing start_bins/goal_bins arrays from trial
    information (see examples).

    For small environments (n_bins < 5000), precomputes full distance matrix:
        dist_matrix = geodesic_distance_matrix(env.connectivity, env.n_bins)
        progress = dist_matrix[start_bins, trajectory_bins] /
                   dist_matrix[start_bins, goal_bins]

    For large environments, computes distance fields per unique (start, goal) pair.

    **Edge cases**:
    - start_bin == goal_bin: Returns 1.0 (already at goal)
    - Disconnected paths: Returns NaN
    - Invalid bins (-1): Returns NaN
    - Detours (progress > 1): Clipped to 1.0

    Examples
    --------
    # Single trial - constant start/goal
    progress = path_progress(
        env,
        trajectory_bins,
        start_bins=np.full(len(trajectory_bins), 10),
        goal_bins=np.full(len(trajectory_bins), 50)
    )

    # Multiple trials - construct arrays once, compute once
    trials = segment_trials(trajectory_bins, times, env, ...)
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)

    # Vectorized computation - no loop over timepoints!
    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    See Also
    --------
    trials_to_region_arrays : Helper to construct start/goal arrays from trials
    distance_to_region : Distance to target region over time
    """
    # Check fitted state
    if not env._is_fitted:
        from neurospatial import EnvironmentNotFittedError

        raise EnvironmentNotFittedError("Environment", "path_progress")

    # Import distance functions
    from neurospatial.distance import (
        euclidean_distance_matrix,
        geodesic_distance_matrix,
    )

    n_samples = len(trajectory_bins)

    # Choose strategy based on environment size
    if env.n_bins < 5000:
        # Small environment - precompute full distance matrix
        if metric == "geodesic":
            dist_matrix = geodesic_distance_matrix(
                env.connectivity, env.n_bins, weight="distance"
            )
        else:  # euclidean
            dist_matrix = euclidean_distance_matrix(env.bin_centers)

        # Vectorized lookup: distance from start to current position
        distances_from_start = dist_matrix[start_bins, trajectory_bins]

        # Vectorized lookup: total distance from start to goal
        total_distances = dist_matrix[start_bins, goal_bins]

    else:
        # Large environment - compute per-unique-pair distance fields
        # This is more memory-efficient for large grids
        from neurospatial import distance_field

        # Initialize arrays
        distances_from_start = np.full(n_samples, np.nan)
        total_distances = np.full(n_samples, np.nan)

        # Find unique (start, goal) pairs
        # Filter out invalid bins (-1) before creating pairs
        valid_mask = (start_bins >= 0) & (goal_bins >= 0)
        valid_pairs = np.column_stack([start_bins[valid_mask], goal_bins[valid_mask]])
        unique_pairs = np.unique(valid_pairs, axis=0)

        # Compute distance field for each unique pair
        for start_bin, goal_bin in unique_pairs:
            # Find all timepoints with this (start, goal) pair
            pair_mask = (start_bins == start_bin) & (goal_bins == goal_bin)

            # Compute distance field from start
            start_dist_field = distance_field(
                env.connectivity,
                [int(start_bin)],
                metric=metric,
                bin_centers=env.bin_centers if metric == "euclidean" else None,
            )

            # Get distances for this pair
            distances_from_start[pair_mask] = start_dist_field[
                trajectory_bins[pair_mask]
            ]
            total_distances[pair_mask] = start_dist_field[goal_bin]

    # Compute progress
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = distances_from_start / total_distances

    # Handle edge cases

    # 1. start_bin == goal_bin: Return 1.0 (already at goal)
    same_start_goal = start_bins == goal_bins
    progress[same_start_goal] = 1.0

    # 2. Disconnected paths (total_distance = inf): Return NaN
    disconnected = np.isinf(total_distances)
    progress[disconnected] = np.nan

    # 3. Invalid bins (-1): Return NaN
    invalid_bins = (start_bins == -1) | (goal_bins == -1) | (trajectory_bins == -1)
    progress[invalid_bins] = np.nan

    # 4. Clip detours (progress > 1.0) to 1.0
    progress = np.clip(progress, 0.0, 1.0)

    return progress


def distance_to_region(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    target_bins: NDArray[np.int_] | int,
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute distance from each trajectory point to target region.

    Measures instantaneous distance to a goal region over time. Useful
    for analyzing approach behavior, goal-directed navigation, and
    spatial value functions.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time.
    target_bins : NDArray[np.int_] or int
        Target bin specification:
        - int: Single target bin (constant over time)
        - NDArray[np.int_], shape (n_samples,): Target bin at each timepoint
          (allows dynamic goals)
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to target at each timepoint.
        Units match environment coordinates (e.g., cm).

    Notes
    -----
    For scalar targets, delegates to env.distance_to() (already exists).
    For dynamic targets, precomputes distance matrix for vectorized lookup.

    Examples
    --------
    >>> # Distance to single region
    >>> goal_bins = env.bins_in_region("reward_zone")
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins[0])
    >>>
    >>> # Distance to dynamic goal (varies per trial)
    >>> trials = segment_trials(trajectory_bins, times, env, ...)
    >>> _, goal_bins = trials_to_region_arrays(trials, times, env)
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins)

    See Also
    --------
    path_progress : Normalized progress along path
    cost_to_goal : Distance with terrain/learned cost
    """
    # Check fitted state
    if not env._is_fitted:
        from neurospatial import EnvironmentNotFittedError

        raise EnvironmentNotFittedError("Environment", "distance_to_region")

    # Check if target_bins is scalar (int) vs array
    is_scalar_target = np.isscalar(target_bins)

    if is_scalar_target:
        # Scalar target - use existing env.distance_to()
        n_samples = len(trajectory_bins)
        target_int = int(target_bins)

        # Check if target is invalid
        if target_int == -1:
            # Invalid scalar target → all distances are NaN
            return np.full(n_samples, np.nan, dtype=np.float64)

        dist_field = env.distance_to([target_int], metric=metric)  # type: ignore[misc]
        distances = dist_field[trajectory_bins].astype(np.float64)

        # Handle invalid trajectory bins
        invalid_mask = trajectory_bins == -1
        distances[invalid_mask] = np.nan
        return distances

    else:
        # Array of targets - need distance matrix
        # Choose strategy based on environment size (same as path_progress)
        n_samples = len(trajectory_bins)

        if env.n_bins < 5000:
            # Small environment - precompute full matrix
            from neurospatial.distance import (
                euclidean_distance_matrix,
                geodesic_distance_matrix,
            )

            if metric == "geodesic":
                dist_matrix = geodesic_distance_matrix(
                    env.connectivity, env.n_bins, weight="distance"
                )
            else:
                dist_matrix = euclidean_distance_matrix(env.bin_centers)

            # Advanced indexing for vectorized lookup
            distances = dist_matrix[trajectory_bins, target_bins]
        else:
            # Large environment - compute per unique target
            unique_targets = np.unique(target_bins)
            unique_targets = unique_targets[unique_targets != -1]  # Filter invalid

            # Build distance array
            distances = np.full(n_samples, np.nan, dtype=np.float64)

            for target in unique_targets:
                mask = target_bins == target
                if np.any(mask):
                    dist_field = env.distance_to([int(target)], metric=metric)  # type: ignore[misc]
                    distances[mask] = dist_field[trajectory_bins[mask]]

        # Handle invalid bins
        invalid_mask = (trajectory_bins == -1) | (target_bins == -1)
        distances[invalid_mask] = np.nan

        return distances


def cost_to_goal(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_] | int,
    *,
    cost_map: NDArray[np.float64] | None = None,
    terrain_difficulty: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute cost-to-goal incorporating terrain difficulty and learned costs.

    Unlike simple geometric distance, cost-to-goal accounts for:
    - Movement difficulty (e.g., barriers, narrow passages)
    - Learned avoidance (e.g., punishment zones, risky areas)

    This is the spatial equivalent of a value function in RL.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time.
    goal_bins : NDArray[np.int_] or int
        Target bin(s). Can be scalar (constant goal) or array (dynamic goal).
    cost_map : NDArray[np.float64], shape (n_bins,), optional
        Per-bin traversal cost. Default: uniform cost (geodesic distance).
        Higher values = more costly to traverse.
    terrain_difficulty : NDArray[np.float64], shape (n_bins,), optional
        Movement difficulty multiplier per bin. Default: 1.0 (uniform).
        Example: narrow passages = 2.0, open space = 1.0.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cost-to-goal from current position. Lower cost = preferred path.

    Notes
    -----
    Cost is computed using Dijkstra's algorithm with modified edge weights:
        edge_cost = base_distance * terrain_difficulty[node] + cost_map[node]

    This aligns with TODO.md Section 5.1 (Cost-distance maps).

    **Static Cost Maps Only**: Current implementation supports static cost maps.
    Dynamic (time-varying) costs are deferred to future versions.

    Examples
    --------
    >>> # Simple: uniform cost (equivalent to geodesic distance)
    >>> cost = cost_to_goal(env, trajectory_bins, goal_bin)
    >>>
    >>> # Learned avoidance: avoid punishment zone
    >>> cost_map = np.ones(env.n_bins)
    >>> cost_map[punishment_bins] = 10.0  # High cost in punishment zone
    >>> cost = cost_to_goal(env, trajectory_bins, goal_bin, cost_map=cost_map)
    >>>
    >>> # Terrain difficulty: model narrow passages
    >>> difficulty = np.ones(env.n_bins)
    >>> difficulty[narrow_passage_bins] = 3.0  # 3x harder to traverse
    >>> cost = cost_to_goal(
    ...     env, trajectory_bins, goal_bin, terrain_difficulty=difficulty
    ... )

    See Also
    --------
    distance_to_region : Simple geometric distance
    """
    # Import required modules
    from neurospatial.distance import distance_field

    # Case 1: No cost modifications - use standard geodesic distance
    if cost_map is None and terrain_difficulty is None:
        return distance_to_region(env, trajectory_bins, goal_bins, metric="geodesic")

    # Case 2: Cost modifications - build weighted graph
    g_weighted = env.connectivity.copy()

    # Modify edge weights
    for u, v, data in g_weighted.edges(data=True):
        base_dist = data["distance"]

        # Apply terrain difficulty (multiplicative)
        if terrain_difficulty is not None:
            # Average difficulty between connected nodes
            difficulty = (terrain_difficulty[u] + terrain_difficulty[v]) / 2.0
            base_dist *= difficulty

        # Add cost (additive)
        if cost_map is not None:
            # Average cost between connected nodes
            cost = (cost_map[u] + cost_map[v]) / 2.0
            base_dist += cost

        # Update edge weight
        g_weighted[u][v]["weight"] = base_dist

    # Compute distance field(s) with modified weights
    if isinstance(goal_bins, (int, np.integer)):
        # Scalar goal - single distance field
        dist_field = distance_field(g_weighted, [int(goal_bins)], weight="weight")
        costs = dist_field[trajectory_bins]

        # Handle invalid bins
        invalid_mask = (trajectory_bins == -1) | (goal_bins == -1)
        costs[invalid_mask] = np.nan

        return costs
    else:
        # Dynamic goals - compute per unique goal
        unique_goals = np.unique(goal_bins)
        unique_goals = unique_goals[unique_goals != -1]  # Filter invalid

        # Build cost array
        costs = np.full(len(trajectory_bins), np.nan, dtype=np.float64)

        for goal in unique_goals:
            mask = goal_bins == goal
            if np.any(mask):
                dist_field = distance_field(g_weighted, [int(goal)], weight="weight")
                costs[mask] = dist_field[trajectory_bins[mask]]

        # Handle invalid bins
        invalid_mask = (trajectory_bins == -1) | (goal_bins == -1)
        costs[invalid_mask] = np.nan

        return costs


def time_to_goal(
    times: NDArray[np.float64],
    trials: list[Trial],
) -> NDArray[np.float64]:
    """Compute time remaining until goal arrival for each trial.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    trials : list[Trial]
        Trial segmentation from segment_trials(). Only successful trials
        (trial.success=True) are included; failed trials are excluded.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time in seconds until goal arrival. NaN outside successful trials
        and during failed trials.

    Notes
    -----
    For each successful trial:
    - time_to_goal[t] = trial.end_time - t (countdown to goal)
    - Clamped to 0.0 to handle any numerical issues

    For failed trials and outside trials:
    - time_to_goal = NaN

    Goal arrival time is defined as trial.end_time (when trial ends successfully).

    Examples
    --------
    >>> trials = segment_trials(
    ...     trajectory_bins, times, env, start_region="home", end_regions=["goal"]
    ... )
    >>> ttg = time_to_goal(times, trials)
    >>>
    >>> # Filter for approach phase (last 2 seconds before goal)
    >>> approach_mask = (ttg > 0) & (ttg <= 2.0)
    >>>
    >>> # Use in GLM
    >>> covariates = pd.DataFrame(
    ...     {
    ...         "time_to_goal": ttg,
    ...         "approaching_goal": approach_mask,
    ...     }
    ... )

    See Also
    --------
    segment_trials : Segment trajectory into trials
    path_progress : Normalized progress along path
    """
    # Initialize with NaN (default for outside trials and failed trials)
    ttg = np.full(len(times), np.nan, dtype=np.float64)

    # Process each trial
    for trial in trials:
        if not trial.success:
            continue  # Skip failed trials (leave as NaN)

        # Find timepoints within this trial
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_times = times[mask]

        # Compute countdown: time remaining until trial.end_time
        ttg[mask] = trial.end_time - trial_times

        # Clamp to 0.0 (handle any numerical issues)
        ttg[mask] = np.maximum(ttg[mask], 0.0)

    return ttg


def compute_trajectory_curvature(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64] | None = None,
    *,
    smooth_window: float | None = 0.2,
) -> NDArray[np.float64]:
    """Compute trajectory curvature from position data.

    Works for any dimensionality (1D, 2D, 3D, N-D). Computes signed
    angle between consecutive movement direction vectors.

    Parameters
    ----------
    trajectory_positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates over time in any dimensional space.
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps for temporal smoothing. If None, assumes uniform sampling.
    smooth_window : float, optional
        Temporal smoothing window in seconds. Default: 0.2s.
        Set to None for no smoothing.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Heading change at each timepoint in radians:
        - Positive values: counterclockwise turn (left in 2D top-down view)
        - Negative values: clockwise turn (right in 2D top-down view)
        - Zero: straight movement

    Notes
    -----
    This function wraps `compute_turn_angles()` from `metrics/trajectory.py`
    and adds:
    - Padding to match input length (n_samples)
    - Optional temporal smoothing

    `compute_turn_angles()` uses atan2(cross, dot) for proper signed angles in [-π, π],
    filters stationary periods automatically, and returns length (n_samples - 2).

    For N-D trajectories where N > 2, only the first 2 dimensions are used
    (consistent with `compute_turn_angles()` implementation).

    Examples
    --------
    >>> # 2D trajectory on any environment (grid, graph, continuous)
    >>> curvature = compute_trajectory_curvature(trajectory_positions)
    >>>
    >>> # Detect sharp turns (> 45 degrees)
    >>> sharp_left = np.where(curvature > np.pi / 4)[0]
    >>> sharp_right = np.where(curvature < -np.pi / 4)[0]
    >>>
    >>> # 3D trajectory (e.g., climbing, flying)
    >>> curvature_3d = compute_trajectory_curvature(trajectory_3d)
    >>>
    >>> # Smooth for noisy tracking data
    >>> curvature_smooth = compute_trajectory_curvature(
    ...     trajectory_positions, times, smooth_window=0.5
    ... )
    >>>
    >>> # Use as GLM regressor
    >>> covariates = pd.DataFrame(
    ...     {
    ...         "curvature": curvature,
    ...         "abs_curvature": np.abs(curvature),
    ...         "is_turning": np.abs(curvature) > np.pi / 6,
    ...     }
    ... )

    See Also
    --------
    graph_turn_sequence : Discrete turn labels for graph-based tracks
    """
    from neurospatial.metrics import compute_turn_angles

    # 1. Compute turn angles using existing function
    # Returns length (n_angles,) where n_angles <= n_samples - 2
    # Filters stationary periods automatically
    angles = compute_turn_angles(trajectory_positions)

    # 2. Pad to match input length (n_samples)
    # compute_turn_angles returns variable length due to duplicate filtering
    # Pad with 0 at start and end to reach n_samples
    n_samples = len(trajectory_positions)
    n_angles = len(angles)

    # Calculate padding needed
    if n_angles == 0:
        # Edge case: < 3 unique positions
        curvature = np.zeros(n_samples, dtype=np.float64)
    else:
        # Pad symmetrically: add (n_samples - n_angles) / 2 to each side
        # If uneven, add extra to the end
        pad_total = n_samples - n_angles
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        curvature = np.pad(
            angles, (pad_left, pad_right), mode="constant", constant_values=0.0
        )

    # 3. Optional temporal smoothing
    if smooth_window is not None and times is not None:
        from scipy.ndimage import gaussian_filter1d

        # Compute sigma from time resolution
        dt_median = np.median(np.diff(times))
        sigma = smooth_window / dt_median

        # Apply Gaussian smoothing
        curvature = gaussian_filter1d(curvature, sigma=sigma)

    return curvature


def graph_turn_sequence(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    start_bin: int,
    end_bin: int,
    *,
    min_samples_per_edge: int = 50,
) -> str:
    """Extract discrete turn sequence from trajectory.

    Analyzes which bins were traversed and classifies turns based on
    changes in movement direction. Works for any environment type.

    For continuous curvature analysis, use compute_trajectory_curvature().

    Parameters
    ----------
    env : Environment
        Spatial environment (any layout type).
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along trajectory (single trial or segment).
    start_bin, end_bin : int
        Start and end bin indices to orient path direction.
    min_samples_per_edge : int, default=50
        Minimum trajectory samples required on a transition to count.
        Filters out noise and brief crossings.

    Returns
    -------
    str
        Turn sequence like "left-right-left", or empty string "" if no
        turns detected. Each turn is separated by "-".

    Notes
    -----
    Algorithm:
    1. Infer which transitions were traversed from consecutive bin pairs
    2. Filter transitions with < min_samples_per_edge
    3. Orient path from start_bin to end_bin
    4. For each consecutive transition:
       - Extract direction vectors from bin positions
       - Compute cross product (2D) or rotation (N-D)
       - Classify: positive → "left", negative → "right"
    5. Join turns into sequence string

    **Dimensionality:**
    - 2D: "left"/"right" determined by cross product sign
    - 3D+: "left"/"right" determined by projection onto primary movement plane

    Examples
    --------
    >>> # Y-maze trajectory
    >>> env = Environment.from_graph(ymaze_graph, ...)
    >>> trials = segment_trials(trajectory_bins, times, env, ...)
    >>>
    >>> for trial in trials:
    ...     mask = (times >= trial.start_time) & (times <= trial.end_time)
    ...     start_bin = env.bins_in_region(trial.start_region)[0]
    ...     end_bin = env.bins_in_region(trial.end_region)[0]
    ...     turn_seq = graph_turn_sequence(
    ...         env, trajectory_bins[mask], start_bin, end_bin
    ...     )
    ...     print(f"Trial: {turn_seq}")  # "left" or "right"
    >>>
    >>> # Open field - also works
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> turn_seq = graph_turn_sequence(env, trajectory_bins, start_bin, end_bin)

    See Also
    --------
    compute_trajectory_curvature : Continuous curvature for any trajectory
    """
    raise NotImplementedError("graph_turn_sequence not yet implemented")
