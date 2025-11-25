# Implementation Plan: Goal-Directed Metrics and Behavioral Analysis

**Version**: v0.8.0
**Date**: 2025-11-24
**Status**: Planning

---

## Overview

Add core goal-directed navigation metrics and behavioral analysis functions to neurospatial, focusing on:

- Path progress along trajectories
- Distance/cost to goal regions
- Trajectory curvature analysis
- Time-to-goal computation

These features align with TODO.md Section 2.4 (Distance-to-goal), Section 3.3 (GLM regressors), and Section 5.1 (Cost-distance maps).

**Note**: This plan has been updated to fix API mismatches with existing code (correct parameter names, use existing functions like `env.distance_to()` and `compute_turn_angles()`, proper region-to-bin mapping).

---

## Scope

### ✅ In Scope

1. **Path Progress** - Normalized completion (0→1) along paths
2. **Distance to Region** - Dynamic distance to target regions over time
3. **Cost to Goal** - RL-aware cost incorporating terrain/learned avoidance (static cost maps)
4. **Time to Goal** - Temporal countdown to goal arrival
5. **Trajectory Curvature** - General continuous curvature for any environment
6. **Graph Turn Sequence** - Discrete turn labels for trajectory analysis
7. **Public API Fixes** - Export hidden segmentation functions

### ❌ Out of Scope

- Speed/velocity computation (user handles externally)
- Dynamic cost maps (time-varying)
- Successor representations
- Value function estimation
- Reward prediction error fields
- Phase 4 advanced RL features (deferred)

---

## Architecture

### Module Structure

```
src/neurospatial/
├── behavioral.py          # NEW: High-level behavioral metrics
│   ├── path_progress()
│   ├── distance_to_region()
│   ├── cost_to_goal()
│   ├── time_to_goal()
│   ├── compute_trajectory_curvature()
│   ├── graph_turn_sequence()
│   └── trials_to_region_arrays()  # Helper
│
├── environment/
│   └── trajectory.py      # EXISTING: Low-level trajectory primitives
│       ├── occupancy()
│       ├── bin_sequence()
│       └── transitions()
│
├── segmentation/          # EXISTING: Event detection
│   ├── detect_region_crossings()
│   ├── segment_by_velocity()     # Make public
│   ├── detect_goal_directed_runs()  # Make public
│   └── detect_runs_between_regions()  # Make public
│
└── distance.py            # EXISTING: Distance primitives
    ├── distance_field()
    ├── geodesic_distance_matrix()
    └── pairwise_distances()
```

### Design Rationale

**Why a new `behavioral.py` module?**

- Separates RL/navigation-specific metrics from general spatial primitives
- Aligns with TODO.md Section 3.3 (GLM regressors)
- Natural home for metrics requiring trajectory + time + regions
- Keeps Environment class focused on spatial structure

**Why NOT methods on Environment?**

- These metrics require multiple inputs beyond environment structure (trajectory + time + regions)
- They're analysis outputs, not environmental properties
- Follows existing pattern: `compute_place_field()` is a free function, not `env.place_field()`

**Why NOT in segmentation/?**

- Segmentation focuses on event detection (when did X happen?)
- Behavioral metrics compute continuous variables over time (how much progress? what speed?)
- Different conceptual purposes

---

## Technical Corrections from Code Review

This section documents API corrections made after reviewing existing neurospatial code:

### 1. **Correct Parameter Names**

- `distance_field()` uses `metric=` (not `method=`)
- `geodesic_distance_matrix()` uses `weight=` for edge attribute
- Euclidean distance requires `bin_centers=env.bin_centers` parameter

### 2. **Region → Bin Mapping**

- Regions don't have `.bin_indices` attribute
- Use `env.bins_in_region(region_name)` which returns `NDArray[np.int_]`
- For point regions: returns at most 1 bin
- For polygon regions: returns all bins whose centers fall within polygon

### 3. **Reuse Existing Functions**

- Use `env.distance_to(targets, metric=...)` for scalar targets (already exists in `environment/queries.py`)
- Use `compute_turn_angles()` from `metrics/trajectory.py` for curvature computation
- Don't reimplement low-level distance calculations

### 4. **Distance Matrix Memory Management**

- Full `n_bins × n_bins` float64 matrix is ~800 MB at 10k bins
- **Fallback strategy**:
  - For `n_bins < 5000`: precompute full matrix
  - For `n_bins >= 5000` and few unique (start, goal) pairs: compute per-pair distance fields
  - For large n_bins with many unique pairs: warn user and compute on-demand (slower but memory-safe)
- Consider adding `@lru_cache` for repeated calls with same environment

### 5. **Path Progress Edge Cases**

Must explicitly define behavior for:

- `start_bin == goal_bin`: Return `1.0` (already at goal)
- Disconnected paths (`distance(start, goal) = inf`): Return `NaN`
- Detours (progress > 1): Clip to `1.0` and document as "geodesic progress"
- Invalid bins (-1 from outside trials): Return `NaN`

### 6. **Curvature: Reuse Existing Code**

- `metrics.compute_turn_angles()` already exists (lines 31-165 in `metrics/trajectory.py`)
- Uses `atan2(cross, dot)` for proper signed angles [-π, π]
- Filters stationary periods automatically
- Returns length `(n_samples - 2)` not `n_samples`
- **New function should**:
  - Call `compute_turn_angles()` internally
  - Add temporal smoothing (optional)
  - Pad/align result to match `n_samples` length

### 7. **Dependencies**

- Import `PCA` from sklearn and `gaussian_filter1d` from scipy **inside functions** (not module-level)
- Be explicit about 2D vs N-D behavior
- For N-D curvature, document that it uses first 2 dimensions or PCA projection

### 8. **`time_to_goal()` Edge Cases**

- If `goal_region is not None` and trial ends in different region: all NaN for that trial
- Failed trials (`trial.success == False`): all NaN
- Outside trials: NaN
- After goal reached: 0.0

### 9. **`@check_fitted` Usage**

- Decorator designed for Environment methods, not free functions
- Will label errors as `Environment.function_name()` (slightly misleading)
- **Decision**: Use explicit checks in free functions:

```python
if not env._is_fitted:
    from neurospatial import EnvironmentNotFittedError
    raise EnvironmentNotFittedError("Environment", "path_progress")
```

---

## Function Specifications

### 1. Path Progress (Vectorized)

**Location**: `src/neurospatial/behavioral.py`

```python
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
```

**Implementation Strategy**:

1. **Check fitted state**:

   ```python
   if not env._is_fitted:
       raise EnvironmentNotFittedError("Environment", "path_progress")
   ```

2. **Choose strategy based on environment size**:
   - If `env.n_bins < 5000`: Precompute full distance matrix
   - Else: Compute per-unique-pair distance fields

3. **For small environments** (precompute matrix):

   ```python
   if metric == "geodesic":
       dist_matrix = geodesic_distance_matrix(env.connectivity, env.n_bins, weight="distance")
   else:  # euclidean
       dist_matrix = euclidean_distance_matrix(env.bin_centers)

   distances_from_start = dist_matrix[start_bins, trajectory_bins]
   total_distances = dist_matrix[start_bins, goal_bins]
   ```

4. **For large environments** (per-pair fields):

   ```python
   # Find unique (start, goal) pairs
   unique_pairs = np.unique(np.column_stack([start_bins, goal_bins]), axis=0)

   # Compute distance field for each unique pair
   # Index results by timepoint
   ```

5. **Compute progress**:

   ```python
   progress = distances_from_start / total_distances

   # Handle edge cases
   progress[start_bins == goal_bins] = 1.0  # Already at goal
   progress[np.isinf(total_distances)] = np.nan  # Disconnected
   progress[(start_bins == -1) | (goal_bins == -1)] = np.nan  # Invalid bins

   # Clip detours
   progress = np.clip(progress, 0.0, 1.0)
   ```

**Dependencies**:

- `geodesic_distance_matrix()` from `distance.py` (uses `weight="distance"`)
- `euclidean_distance_matrix()` from `distance.py`
- `EnvironmentNotFittedError` from `neurospatial`

---

### 2. Helper: Trials to Region Arrays

**Location**: `src/neurospatial/behavioral.py`

```python
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
```

**Implementation**:

```python
start_bins = np.full(len(times), -1, dtype=np.int_)
goal_bins = np.full(len(times), -1, dtype=np.int_)

for trial in trials:  # Loop over trials (small)
    mask = (times >= trial.start_time) & (times <= trial.end_time)

    # Get bins for start region
    start_region_bins = env.bins_in_region(trial.start_region)
    if len(start_region_bins) > 0:
        start_bins[mask] = start_region_bins[0]

    # Get bins for end region (handle None for failed trials)
    if trial.end_region is not None:
        end_region_bins = env.bins_in_region(trial.end_region)
        if len(end_region_bins) > 0:
            goal_bins[mask] = end_region_bins[0]
    # For failed trials (end_region=None), goal_bins remains -1

return start_bins, goal_bins
```

**Note**: For polygon regions with multiple bins, this takes the first bin. Future versions could support region centroids or allow user-specified representative bin. Failed trials (where `trial.end_region is None`) leave `goal_bins` as `-1`.

---

### 3. Distance to Region

**Location**: `src/neurospatial/behavioral.py`

```python
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
    >>> goal_bins = env.bins_in_region('reward_zone')
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins[0])
    >>>
    >>> # Distance to dynamic goal (varies per trial)
    >>> trials = segment_trials(trajectory_bins, times, env, ...)
    >>> _, goal_bins = trials_to_region_arrays(trials, times, env)
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins)

    See Also
    --------
    path_progress : Normalized progress along path
    env.distance_to : Underlying method for scalar targets
    cost_to_goal : Distance with terrain/learned cost
    """
```

**Implementation**:

```python
if isinstance(target_bins, int):
    # Scalar target - use existing env.distance_to()
    dist_field = env.distance_to([target_bins], metric=metric)
    distances = dist_field[trajectory_bins]

    # Handle invalid bins
    distances[(trajectory_bins == -1) | (target_bins == -1)] = np.nan
    return distances

else:
    # Array of targets - need distance matrix
    # Choose strategy based on environment size (same as path_progress)
    if env.n_bins < 5000:
        # Small environment - precompute full matrix
        if metric == "geodesic":
            dist_matrix = geodesic_distance_matrix(
                env.connectivity, env.n_bins, weight="distance"
            )
        else:
            dist_matrix = euclidean_distance_matrix(env.bin_centers)

        distances = dist_matrix[trajectory_bins, target_bins]
    else:
        # Large environment - compute per unique target
        unique_targets = np.unique(target_bins)
        unique_targets = unique_targets[unique_targets != -1]  # Filter invalid

        # Build distance array
        distances = np.full(len(trajectory_bins), np.nan)

        for target in unique_targets:
            mask = target_bins == target
            if np.any(mask):
                dist_field = env.distance_to([target], metric=metric)
                distances[mask] = dist_field[trajectory_bins[mask]]

    # Handle invalid bins
    distances[(trajectory_bins == -1) | (target_bins == -1)] = np.nan

    return distances
```

---

### 4. Cost to Goal

**Location**: `src/neurospatial/behavioral.py`

```python
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
    >>> cost = cost_to_goal(env, trajectory_bins, goal_bin,
    ...                     terrain_difficulty=difficulty)

    See Also
    --------
    distance_to_region : Simple geometric distance
    goal_reward_field : Compute reward gradient (inverse cost)
    """
```

**Implementation**:

```python
# Build modified graph with cost weights
if cost_map is not None or terrain_difficulty is not None:
    # Create weighted graph
    G_weighted = env.connectivity.copy()

    for u, v, data in G_weighted.edges(data=True):
        base_dist = data['distance']

        # Apply terrain difficulty
        if terrain_difficulty is not None:
            difficulty = (terrain_difficulty[u] + terrain_difficulty[v]) / 2
            base_dist *= difficulty

        # Add cost
        if cost_map is not None:
            base_dist += (cost_map[u] + cost_map[v]) / 2

        G_weighted[u][v]['weight'] = base_dist

    # Compute distance field with modified weights
    if isinstance(goal_bins, int):
        dist_field = distance_field(G_weighted, [goal_bins], weight='weight')
        return dist_field[trajectory_bins]
    else:
        # Multiple goals - compute per-goal and take minimum
        costs = []
        for goal_bin in np.unique(goal_bins):
            dist_field = distance_field(G_weighted, [goal_bin], weight='weight')
            costs.append(dist_field)

        cost_matrix = np.stack(costs)
        goal_indices = np.searchsorted(np.unique(goal_bins), goal_bins)
        return cost_matrix[goal_indices, trajectory_bins]
else:
    # No cost modifications - use standard distance
    return distance_to_region(env, trajectory_bins, goal_bins, metric="geodesic")
```

---

### 5. Time to Goal

**Location**: `src/neurospatial/behavioral.py`

```python
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
    >>> trials = segment_trials(trajectory_bins, times, env,
    ...                          start_region="home", end_regions=["goal"])
    >>> ttg = time_to_goal(times, trials)
    >>>
    >>> # Filter for approach phase (last 2 seconds before goal)
    >>> approach_mask = (ttg > 0) & (ttg <= 2.0)
    >>>
    >>> # Use in GLM
    >>> covariates = pd.DataFrame({
    ...     'time_to_goal': ttg,
    ...     'approaching_goal': approach_mask,
    ... })

    See Also
    --------
    segment_trials : Segment trajectory into trials
    path_progress : Normalized progress along path
    """
```

**Implementation**:

```python
ttg = np.full(len(times), np.nan)

for trial in trials:
    if not trial.success:
        continue  # Skip failed trials

    mask = (times >= trial.start_time) & (times <= trial.end_time)
    trial_times = times[mask]

    # Time remaining until trial.end_time
    ttg[mask] = trial.end_time - trial_times

    # Clamp to 0 (after goal reached)
    ttg[mask] = np.maximum(ttg[mask], 0.0)

return ttg
```

---

### 6. Trajectory Curvature (General)

**Location**: `src/neurospatial/behavioral.py`

```python
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
    >>> covariates = pd.DataFrame({
    ...     'curvature': curvature,
    ...     'abs_curvature': np.abs(curvature),
    ...     'is_turning': np.abs(curvature) > np.pi / 6,
    ... })

    See Also
    --------
    compute_turn_angles : Underlying function from metrics.trajectory
    graph_turn_sequence : Discrete turn labels for graph-based tracks
    """
```

**Implementation**:

```python
from neurospatial.metrics import compute_turn_angles

# 1. Compute turn angles using existing function
# Returns length (n_samples - 2), filters stationary periods, uses atan2
angles = compute_turn_angles(trajectory_positions)

# 2. Pad to match input length (n_samples)
# Pad with 0 at start and end
curvature = np.pad(angles, (1, 1), mode='constant', constant_values=0.0)

# 3. Optional temporal smoothing
if smooth_window is not None and times is not None:
    from scipy.ndimage import gaussian_filter1d

    # Compute sigma from time resolution
    dt_median = np.median(np.diff(times))
    sigma = smooth_window / dt_median

    # Apply Gaussian smoothing
    curvature = gaussian_filter1d(curvature, sigma=sigma)

return curvature
```

**Note**: Reuses `compute_turn_angles()` to avoid code duplication. The existing function already handles:

- Movement vector computation and normalization
- Stationary period filtering
- Signed angle calculation with `atan2(cross, dot)`
- 2D vs N-D trajectories

---

### 7. Graph Turn Sequence (Specialized)

**Location**: `src/neurospatial/behavioral.py`

```python
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
```

**Implementation** (see detailed algorithm in conversation above).

---

## Public API Updates

### Add to `src/neurospatial/__init__.py`

```python
# Behavioral analysis (NEW)
from neurospatial.behavioral import (
    compute_trajectory_curvature,
    cost_to_goal,
    distance_to_region,
    graph_turn_sequence,
    path_progress,
    time_to_goal,
    trials_to_region_arrays,
)

# Segmentation - add missing exports
from neurospatial.segmentation import (
    detect_laps,
    detect_region_crossings,
    segment_trials,
    # ADD THESE:
    detect_goal_directed_runs,
    detect_runs_between_regions,
    segment_by_velocity,
)
```

### Update `__all__`

```python
__all__ = [
    # ... existing exports ...

    # Behavioral analysis
    "compute_trajectory_curvature",
    "cost_to_goal",
    "distance_to_region",
    "graph_turn_sequence",
    "path_progress",
    "time_to_goal",
    "trials_to_region_arrays",

    # Behavioral segmentation (add missing)
    "detect_goal_directed_runs",
    "detect_runs_between_regions",
    "segment_by_velocity",
]
```

---

## Testing Strategy

### Test Structure

```
tests/
├── test_behavioral.py          # NEW: Test behavioral metrics
│   ├── test_path_progress_single_trial()
│   ├── test_path_progress_multiple_trials()
│   ├── test_path_progress_euclidean()
│   ├── test_distance_to_region_scalar_target()
│   ├── test_distance_to_region_dynamic_target()
│   ├── test_cost_to_goal_uniform()
│   ├── test_cost_to_goal_with_cost_map()
│   ├── test_cost_to_goal_terrain_difficulty()
│   ├── test_time_to_goal()
│   ├── test_compute_trajectory_curvature_2d()
│   ├── test_compute_trajectory_curvature_3d()
│   ├── test_graph_turn_sequence_ymaze()
│   ├── test_graph_turn_sequence_grid()
│   └── test_trials_to_region_arrays()
│
└── test_segmentation.py        # EXISTING: Verify public API
    ├── test_detect_goal_directed_runs_exported()
    ├── test_detect_runs_between_regions_exported()
    └── test_segment_by_velocity_exported()
```

### Test Fixtures (add to `conftest.py`)

```python
@pytest.fixture
def ymaze_environment():
    """Y-maze graph environment with 3 arms."""
    # Create simple Y-maze graph
    # Nodes: 0 (center), 1, 2, 3 (arms)
    # Return env with regions: "start", "left", "center", "right"

@pytest.fixture
def spatial_bandit_environment():
    """3-arm bandit environment (like user's example)."""
    # Create track graph with home + 3 goal wells
    # Return env with regions: "home", "goal_1", "goal_2", "goal_3"

@pytest.fixture
def tmaze_trajectory():
    """Simulated T-maze trajectory with multiple trials."""
    # Return: trajectory_bins, times, positions, ground_truth_progress

@pytest.fixture
def grid_trajectory():
    """Open field trajectory on grid environment."""
    # Return: env, trajectory_bins, times, positions
```

### Key Test Cases

1. **Path Progress**:
   - Single trial: constant start/goal
   - Multiple trials: varying start/goal per trial
   - Edge cases: same start/goal, disconnected paths
   - Euclidean vs geodesic metrics

2. **Distance to Region**:
   - Scalar target (constant goal)
   - Dynamic target (goal varies per trial)
   - Multiple goal bins (distance to nearest)

3. **Cost to Goal**:
   - Uniform cost (equivalent to geodesic distance)
   - Cost map with punishment zones
   - Terrain difficulty (narrow passages)
   - Combined cost + terrain

4. **Time to Goal**:
   - Successful trials
   - Failed trials (should be NaN)
   - Outside trials (should be NaN)

5. **Trajectory Curvature**:
   - 2D left/right turns
   - 3D trajectories
   - Straight paths (zero curvature)
   - Noisy trajectories with smoothing

6. **Graph Turn Sequence**:
   - Y-maze left/right choices
   - Grid environment directional changes
   - Multiple consecutive turns

---

## Documentation Updates

### Update `CLAUDE.md` Quick Reference

Add new section after "Trial segmentation":

```markdown
# Behavioral & Goal-Directed Metrics (v0.8.0+)

from neurospatial import (
    path_progress,           # Normalized progress (0→1) along path
    distance_to_region,      # Distance to goal region over time
    cost_to_goal,            # RL cost with terrain/avoidance
    time_to_goal,            # Time until goal arrival
    compute_trajectory_curvature,  # Continuous curvature analysis
    graph_turn_sequence,     # Discrete turn labels
    trials_to_region_arrays, # Helper for trial arrays
)

# Path progress for multiple trials (vectorized)
trials = segment_trials(trajectory_bins, times, env,
                        start_region="home", end_regions=["goal"])
start_bins, goal_bins = trials_to_region_arrays(trials, times, env)
progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

# Distance to goal over time
goal_bin = env.bins_in_region('reward_zone')[0]
dist = distance_to_region(env, trajectory_bins, goal_bin)

# Cost-to-goal with learned avoidance
cost_map = np.ones(env.n_bins)
cost_map[punishment_bins] = 10.0  # Avoid punishment zone
cost = cost_to_goal(env, trajectory_bins, goal_bins, cost_map=cost_map)

# Trajectory curvature (for GLM regressors)
curvature = compute_trajectory_curvature(trajectory_positions, times)
is_turning = np.abs(curvature) > np.pi / 4

# Turn sequence classification
for trial in trials:
    mask = (times >= trial.start_time) & (times <= trial.end_time)
    turn_seq = graph_turn_sequence(
        env, trajectory_bins[mask],
        start_bin=env.bins_in_region(trial.start_region)[0],
        end_bin=env.bins_in_region(trial.end_region)[0]
    )
    print(f"Trial: {turn_seq}")  # e.g., "left-right"
```

---

## Implementation Phases

### Phase 1: Public API Fixes (30 min)

1. Add missing segmentation functions to `__init__.py`
2. Update `__all__` list
3. Write basic import tests

### Phase 2: Foundation Functions (3-4 hours)

1. Implement `trials_to_region_arrays()` helper
2. Implement `path_progress()` (geodesic + euclidean)
3. Implement `distance_to_region()` (scalar + dynamic targets)
4. Write comprehensive tests
5. Update docstrings

### Phase 3: Time and Curvature (2-3 hours)

1. Implement `time_to_goal()`
2. Implement `compute_trajectory_curvature()` (2D + N-D)
3. Write tests for edge cases (3D, noisy data)
4. Validate smoothing works correctly

### Phase 4: Cost and Turn Analysis (3-4 hours)

1. Implement `cost_to_goal()` with cost maps
2. Add terrain difficulty support
3. Implement `graph_turn_sequence()`
4. Write integration tests with spatial bandit fixture
5. Validate on Y-maze and grid environments

### Phase 5: Documentation (2 hours)

1. Update CLAUDE.md Quick Reference
2. Add examples to function docstrings
3. Create tutorial notebook (optional)
4. Update TODO.md to mark items complete

---

## Design Decisions Record

### Q1: Multi-Trial Support

**Decision**: Vectorized API with `start_bins`/`goal_bins` arrays.

**Rationale**:

- User has small loop over trials (10-100), not timepoints (100k+)
- Heavy computation (distance lookups) is fully vectorized
- Optional helper `trials_to_region_arrays()` encapsulates trial loop
- Keeps functions simple and composable

### Q2: Cost Map Types

**Decision**: Static cost maps only (no time-varying).

**Rationale**:

- Simplifies implementation and API
- Covers 90% of use cases (learned avoidance, terrain difficulty)
- Dynamic cost maps deferred to future versions

### Q3: Turn Direction Generalization

**Decision**: Two-function design:

1. `compute_trajectory_curvature()` - general continuous angles
2. `graph_turn_sequence()` - discrete turn labels

**Rationale**:

- Curvature works for any environment/dimensionality
- Turn sequence useful for trial classification
- Different outputs for different use cases (GLM vs. choice analysis)

---

## Success Criteria

- [ ] All 7 new functions implemented with full docstrings
- [ ] All functions fully vectorized (no loops over timepoints)
- [ ] Works on 2D and N-D environments
- [ ] 100% test coverage for new code
- [ ] Public API updated with missing segmentation functions
- [ ] CLAUDE.md updated with examples
- [ ] All tests pass: `uv run pytest tests/test_behavioral.py -v`

---

## Future Enhancements (Post-v0.8.0)

These align with TODO.md but are deferred:

1. **Successor Representation** (TODO 5.3)
   - Expected future occupancy
   - Planning and replay analysis

2. **Value Function Estimation** (TODO 3.3)
   - TD learning from trajectories
   - Monte Carlo estimation

3. **Reward Prediction Error** (TODO 3.3)
   - RPE fields for neural comparison
   - TD error computation

4. **Dynamic Cost Maps**
   - Time-varying costs (attention, learning)
   - Cost map interpolation

5. **Advanced Turn Metrics**
   - Turn timing (reaction time at junctions)
   - Turn stereotypy (consistency across trials)
   - Turn anticipation (pre-turn heading changes)

---

## Notes

- All functions follow NumPy docstring format
- All functions are fully type-annotated for mypy
- Use explicit fitted checks (not `@check_fitted` decorator) for free functions
- All distance functions use `metric=` parameter (not `method=`)
- Region-to-bin mapping uses `env.bins_in_region()` (not `.bin_indices`)
- Reuses existing functions: `env.distance_to()`, `compute_turn_angles()`
- Performance: Functions precompute distance matrices for vectorized lookup (with memory fallback)

---

## Revision History

**2025-11-24 - Code Review Corrections**:

- Fixed parameter names: `method` → `metric` throughout
- Updated region mapping: `.bin_indices` → `env.bins_in_region()`
- Delegated to existing functions: `env.distance_to()`, `compute_turn_angles()`
- Added distance matrix memory management strategy (5k bin threshold)
- Documented all edge cases explicitly (disconnected paths, invalid bins, etc.)
- Added explicit fitted state checks (not using `@check_fitted` decorator)
- Specified dependencies import locations (inside functions, not module-level)
