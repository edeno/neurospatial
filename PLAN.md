# Spatial Operators Enhancement Plan

## Overview

This plan extends neurospatial's spatial analysis capabilities with geometric and topological operators focused on neuroscience use cases: trajectory analysis, environment validation, and field operations on spatial bins.

**Scope**: Pure spatial/geometric operations. No RL-specific abstractions.

**Principles**:

- Maintain existing architecture (Layout → Environment → Regions)
- Follow NumPy docstring format
- No duplicate functionality
- Test with existing test environments (plus maze, square environments)
- Use consistent naming conventions
- Provide helpful error messages with diagnostics

---

## Naming Conventions

**Decision**: Drop `compute_` prefix for consistency with existing neurospatial APIs and brevity. Use descriptive verb-noun patterns.

**Pattern examples**:

- Trajectory analysis: `heading()`, `curvature()`, `path_length()`
- Spatial queries: `bins_crossed_by_*()`, `boundary_distance_field()`
- Region operations: `region_bins()`, `region_mask()`, `region_union()`

**Rationale**: Existing APIs use `map_points_to_bins()`, `distance_field()`, `normalize_field()` without `compute_` prefix. New APIs follow this established pattern.

---

## 1. Trajectory Primitives

**Motivation**: Analyzing animal movement requires geometric analysis of trajectories. These primitives support studies of path integration, navigation strategies, and movement kinematics.

### 1.1 Heading & Angular Velocity

**API** (`spatial.py`):

```python
def heading(
    positions: NDArray[np.float64],
    *,
    differencing: Literal["forward", "central", "backward"] = "central"
) -> NDArray[np.float64]:
    """
    Compute heading angles from position trajectory.

    The heading angle represents the direction of movement at each point,
    measured counterclockwise from the positive x-axis.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        2D position coordinates over time (e.g., animal tracking data).
        Units typically cm or meters, matching your recording setup.
    differencing : {"forward", "central", "backward"}, default="central"
        Method for computing direction vectors:

        - "forward": heading[i] from positions[i] to positions[i+1]
        - "central": heading[i] from positions[i-1] to positions[i+1]
        - "backward": heading[i] from positions[i-1] to positions[i]

        Central differencing provides better noise robustness.

    Returns
    -------
    heading : NDArray[np.float64], shape (n_samples,)
        Heading angle in radians, range [-π, π). Counterclockwise from
        positive x-axis (standard mathematical convention).
        NaN at endpoints where differencing is undefined.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import heading
    >>> # Animal moving from origin to (10, 10) in straight line
    >>> trajectory = np.array([[0, 0], [5, 5], [10, 10]])
    >>> angles = heading(trajectory)
    >>> np.rad2deg(angles[1])  # Center point: 45 degrees
    45.0

    >>> # For analyzing head direction relative to movement
    >>> head_directions = load_head_direction_data()  # In degrees
    >>> movement_heading = heading(trajectory)
    >>> alignment = np.abs(np.deg2rad(head_directions) - movement_heading)

    Notes
    -----
    Heading is computed as arctan2(Δy, Δx) where Δ represents
    position differences. Central differencing uses a 2-point stencil
    providing better numerical stability for noisy trajectories.

    For 3D trajectories, use only x,y columns: `heading(positions[:, :2])`.

    See Also
    --------
    angular_velocity : Rate of heading change
    curvature : Path curvature from heading
    """
```

```python
def angular_velocity(
    positions: NDArray[np.float64],
    dt: float,
    *,
    window: int = 3
) -> NDArray[np.float64]:
    """
    Compute angular velocity (rate of heading change).

    Angular velocity quantifies how quickly the movement direction changes,
    useful for detecting turns, rotations, and navigation strategies.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        2D position coordinates over time.
    dt : float
        Time step between samples in seconds.

        **Examples**:
        - Video tracking at 30 Hz: dt = 1/30 ≈ 0.033 seconds
        - Position samples at 50 Hz: dt = 0.02 seconds
        - GPS data at 1 Hz: dt = 1.0 seconds

    window : int, default=3
        Window size for smoothing angular differences. Must be odd and >= 3.

        **Guidance**:
        - Larger values (5-7): Reduce noise, smooth out rapid turns
        - Smaller values (3): Preserve sharp turns, noise-sensitive
        - Use window ≈ sampling_rate × 0.1 seconds as starting point

    Returns
    -------
    angular_velocity : NDArray[np.float64], shape (n_samples,)
        Angular velocity in radians/second. NaN at boundaries.

        **Typical ranges**:
        - Slow turns: < π/6 rad/s (30 deg/s)
        - Moderate turns: π/6 to π/2 rad/s (30-90 deg/s)
        - Sharp turns: > π/2 rad/s (90 deg/s)

    Raises
    ------
    ValueError
        If dt <= 0 (time step must be positive).
        If window < 3 or window is even.

    Examples
    --------
    >>> from neurospatial import angular_velocity
    >>> trajectory = load_tracking_data()  # shape (1000, 2), 30 Hz video
    >>> omega = angular_velocity(trajectory, dt=1/30)
    >>> # Find frames with sharp turns (> 60 degrees/second)
    >>> sharp_turns = np.abs(omega) > np.deg2rad(60)
    >>> print(f"Sharp turns in {sharp_turns.sum()} frames")

    >>> # Analyze turning behavior near goal
    >>> goal_bins = env.region_bins("goal")
    >>> traj_bins = map_points_to_bins(trajectory, env)
    >>> near_goal = np.isin(traj_bins, goal_bins)
    >>> avg_turn_rate = np.nanmean(np.abs(omega[near_goal]))

    Notes
    -----
    Computed as the rate of change of heading angle, with unwrapping
    to handle 2π discontinuities. The window parameter applies smoothing
    to angular differences before computing the rate.

    See Also
    --------
    heading : Compute heading angles
    curvature : Path curvature (geometry-based alternative)
    """
```

### 1.2 Path Curvature

**API** (`spatial.py`):

```python
def curvature(
    positions: NDArray[np.float64],
    *,
    window: int | Literal["auto"] = "auto"
) -> NDArray[np.float64]:
    """
    Compute path curvature (inverse radius of osculating circle).

    Curvature quantifies how sharply a path bends, with applications
    to understanding navigation strategies and trajectory smoothness.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates over time.
    window : int or "auto", default="auto"
        Window size for local fitting. Must be >= 3 if specified as integer.

        If "auto", adapts to trajectory length: min(7, n_samples // 20).

        **Guidance**:
        - Noisy data (low-res tracking): Use 5-9 for smoothing
        - Clean data (high-res tracking): Use 3 for sharp feature detection
        - Short trajectories (< 100 points): Use 3 to avoid over-smoothing

    Returns
    -------
    curvature : NDArray[np.float64], shape (n_samples,)
        Signed curvature at each point. Units: 1/length (e.g., 1/cm if
        positions are in cm).

        **Interpretation**:
        - Positive: Left turn (counterclockwise)
        - Negative: Right turn (clockwise)
        - Near zero: Straight path
        - NaN: At boundaries where curvature cannot be computed

        **Magnitude**: |κ| = 1/r where r is turn radius.
        Sharp turns: |κ| > 0.1 cm⁻¹ (radius < 10 cm)

    Raises
    ------
    ValueError
        If window < 3 or window >= n_samples (trajectory too short).

    Examples
    --------
    >>> from neurospatial import curvature
    >>> # Circular path with radius 20 cm
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> circle = 20 * np.column_stack([np.cos(theta), np.sin(theta)])
    >>> k = curvature(circle)
    >>> np.nanmean(np.abs(k))  # Should be ≈ 1/20 = 0.05 cm⁻¹
    0.05

    >>> # Detect sharp turns in exploration trajectory
    >>> trajectory = load_tracking_data()
    >>> k = curvature(trajectory, window=5)
    >>> sharp_turns = np.abs(k) > 0.1  # Radius < 10 cm
    >>> print(f"Sharp turns: {np.nansum(sharp_turns)} frames")

    Notes
    -----
    Uses discrete curvature formula based on angle change per arc length:
    κ ≈ Δθ / Δs where Δθ is heading change and Δs is arc length.

    The osculating circle is the best-fit circle to the path at each point.
    Its radius is 1/|κ|.

    See Also
    --------
    heading : Compute heading angles
    angular_velocity : Rate of heading change
    """
```

### 1.3 Path Metrics

**API** (`spatial.py`):

```python
def path_length(
    positions: NDArray[np.float64],
    *,
    cumulative: bool = False
) -> float | NDArray[np.float64]:
    """
    Compute total or cumulative path length.

    Path length measures the actual distance traveled along a trajectory,
    as opposed to displacement (straight-line start-to-end distance).

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates over time.
    cumulative : bool, default=False
        If True, return cumulative distance at each point.
        If False, return total path length.

    Returns
    -------
    length : float or NDArray[np.float64]
        If cumulative=False: total path length (scalar).
        If cumulative=True: cumulative distance at each point,
        shape (n_samples,), with length[0] = 0.

        **Units match input positions** (e.g., if positions are in cm,
        length is in cm).

    Examples
    --------
    >>> from neurospatial import path_length, straightness
    >>> trajectory = load_tracking_data()  # shape (1000, 2), units: cm
    >>> total_dist = path_length(trajectory)
    >>> print(f"Animal traveled {total_dist:.1f} cm")
    Animal traveled 1234.5 cm

    >>> # Compute instantaneous speed
    >>> cum_dist = path_length(trajectory, cumulative=True)
    >>> dt = 1/30  # 30 Hz video
    >>> speed = np.diff(cum_dist) / dt  # cm/s
    >>> print(f"Average speed: {np.mean(speed):.1f} cm/s")

    >>> # Compare path length to displacement
    >>> displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    >>> efficiency = straightness(trajectory)
    >>> print(f"Path: {total_dist:.0f} cm, Displacement: {displacement:.0f} cm")
    >>> print(f"Efficiency: {efficiency:.2%}")

    Notes
    -----
    Computes Euclidean distance sum between consecutive points:

        total_length = Σ ||p[i+1] - p[i]||

    For geodesic path length (along environment graph), use the trajectory
    bin sequence and sum edge distances instead.

    See Also
    --------
    straightness : Path efficiency metric
    curvature : Path curvature analysis
    """
```

```python
def straightness(
    positions: NDArray[np.float64],
) -> float:
    """
    Compute path straightness (efficiency) metric.

    Straightness quantifies how direct a path is by comparing the
    straight-line distance to the actual path length. Used to assess
    navigation efficiency and goal-directed behavior.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates over time.

    Returns
    -------
    straightness : float
        Ratio of straight-line distance to path length.
        Range [0, 1] where 1 = perfectly straight, efficient path.

        **Interpretation**:
        - > 0.8: Highly direct, goal-directed path
        - 0.5-0.8: Moderately efficient
        - < 0.5: Tortuous, exploratory path

        Returns 1.0 if path has only 1 sample (undefined case).

    Examples
    --------
    >>> from neurospatial import straightness, path_length
    >>> # Direct path to goal
    >>> direct_path = np.array([[0, 0], [10, 0], [20, 0]])
    >>> straightness(direct_path)
    1.0

    >>> # Detour path
    >>> detour_path = np.array([[0, 0], [10, 10], [20, 0]])
    >>> straightness(detour_path)
    0.707...  # sqrt(2) ≈ 0.707

    >>> # Analyze trial-by-trial learning
    >>> trials = load_trial_trajectories()
    >>> trial_efficiency = [straightness(trial) for trial in trials]
    >>> plt.plot(trial_efficiency)
    >>> plt.xlabel("Trial number")
    >>> plt.ylabel("Path straightness")

    Notes
    -----
    Defined as:

        straightness = ||p_end - p_start|| / path_length

    where path_length is the sum of Euclidean distances between
    consecutive points.

    Values near 1 indicate efficient, direct paths (goal-directed behavior).
    Values near 0 indicate tortuous, inefficient paths (exploration, random search).

    See Also
    --------
    path_length : Compute path length
    """
```

**Tests**:

- Straight line: curvature ≈ 0, straightness = 1.0
- Circle with radius r: curvature ≈ 1/r
- Horizontal path (y constant): heading ≈ 0° or ±180°
- Vertical path (x constant): heading ≈ ±90°
- Cumulative length is monotonically increasing
- Angular velocity units check (radians/second)

---

## 2. Segment-Environment Intersection

**Motivation**: Identifying which bins a trajectory segment crosses is essential for analyzing movement patterns, boundary crossings, and computing transition statistics.

### 2.1 Segment Crossing

**API** (`spatial.py`):

```python
def bins_crossed_by_segment(
    segment: NDArray[np.float64],
    env: Environment,
    *,
    include_endpoints: bool = True
) -> NDArray[np.int64]:
    """
    Find bins crossed by a line segment.

    Useful for analyzing straight-line paths, checking line-of-sight,
    and understanding barrier crossings in spatial environments.

    Parameters
    ----------
    segment : NDArray[np.float64], shape (2, n_dims)
        Start and end points of segment: [[x_start, y_start], [x_end, y_end]].
    env : Environment
        Environment defining the spatial discretization.
    include_endpoints : bool, default=True
        If True, include bins containing start/end points even if
        segment doesn't cross their interior.

    Returns
    -------
    bin_indices : NDArray[np.int64], shape (n_crossed,)
        Indices of bins crossed by segment, in order from start to end.
        Returns empty array if segment doesn't cross any bins.

    Raises
    ------
    ValueError
        If segment shape is not (2, n_dims).
        If segment dimensionality doesn't match environment.

    Warnings
    --------
    UserWarning
        If segment extends outside environment bounds (only bins within
        environment are returned).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, bins_crossed_by_segment
    >>> data = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> # Check which bins a direct path crosses
    >>> segment = np.array([[0, 0], [10, 10]])
    >>> crossed = bins_crossed_by_segment(segment, env)
    >>> print(f"Crossed {len(crossed)} bins")

    >>> # Check if line-of-sight is clear (no barrier bins)
    >>> barrier_bins = env.region_bins("barrier")
    >>> has_clear_path = not np.any(np.isin(crossed, barrier_bins))

    Notes
    -----
    **Implementation strategy**:

    For regular grids (layouts with `grid_shape` attribute):
        Uses DDA (Digital Differential Analyzer) voxel traversal for
        exact, efficient bin sequence computation.

    For irregular layouts (graph, hexagonal, etc.):
        Samples segment at fine resolution (spacing = min_bin_size / 10)
        and maps to bins, removing consecutive duplicates.

    **Edge cases**:

    - Segment entirely outside environment: Returns empty array
    - Zero-length segment: Returns bin containing that point (if any)
    - Points exactly on bin boundaries: Uses lowest-index tie-breaking
    - Segment tangent to boundaries: Includes bins whose centers are
      nearest to the segment

    See Also
    --------
    bins_crossed_by_path : Find bins crossed by polyline
    map_points_to_bins : Map points to bins
    """
```

```python
def bins_crossed_by_path(
    path: NDArray[np.float64],
    env: Environment,
    *,
    include_endpoints: bool = True
) -> NDArray[np.int64]:
    """
    Find bins crossed by a polyline path.

    Analyzes multi-segment trajectories to determine the sequence of
    bins visited, useful for computing transition matrices and path
    statistics.

    Parameters
    ----------
    path : NDArray[np.float64], shape (n_points, n_dims)
        Sequence of waypoints defining the path. Consecutive points
        define line segments.
    env : Environment
        Environment defining the spatial discretization.
    include_endpoints : bool, default=True
        If True, include bins containing waypoints even if segments
        don't cross their interior.

    Returns
    -------
    bin_indices : NDArray[np.int64], shape (n_crossed,)
        Unique bin indices crossed by path, in order of traversal.
        Consecutive duplicate bins are removed to give visited sequence.

    Raises
    ------
    ValueError
        If path has fewer than 2 points.
        If path dimensionality doesn't match environment.

    Examples
    --------
    >>> from neurospatial import bins_crossed_by_path
    >>> # Analyze trial trajectory
    >>> trial_path = load_trial_trajectory()  # shape (500, 2)
    >>> crossed_bins = bins_crossed_by_path(trial_path, env)
    >>> print(f"Visited {len(crossed_bins)} unique bins in order")

    >>> # Compute transition counts between regions
    >>> start_bins = env.region_bins("start")
    >>> goal_bins = env.region_bins("goal")
    >>> entered_goal = np.isin(crossed_bins, goal_bins)
    >>> if entered_goal.any():
    ...     first_goal_bin = crossed_bins[entered_goal][0]
    ...     print(f"Entered goal at bin {first_goal_bin}")

    Notes
    -----
    Processes each segment (path[i] → path[i+1]) separately and merges
    results, removing consecutive duplicates to get the ordered sequence
    of visited bins.

    Equivalent to:
        all_bins = []
        for i in range(len(path) - 1):
            segment = path[i:i+2]
            all_bins.extend(bins_crossed_by_segment(segment, env))
        return remove_consecutive_duplicates(all_bins)

    See Also
    --------
    bins_crossed_by_segment : Single segment version
    map_points_to_bins : Map trajectory to bin sequence
    """
```

**Tests**:

- Horizontal segment on 5×5 grid crosses expected row
- Diagonal segment \[0,0\] → \[4,4\] crosses main diagonal bins
- Zero-length segment returns start bin only
- Multi-segment path preserves traversal order
- Out-of-bounds segment returns empty array with warning

---

## 3. Topology Validation

**Motivation**: Ensuring environment connectivity is critical for spatial analysis. Disconnected components cause failures in path finding, distance computation, and field operations.

### 3.1 Connectivity Assertions

**API** (`environment.py` public methods, import `GraphValidationError` from `neurospatial.layout.validation`):

```python
def assert_connected(self) -> None:
    """
    Assert that the environment connectivity graph is fully connected.

    Useful for validating that morphological operations (dilation,
    hole filling) haven't created unexpected disconnected regions.

    Raises
    ------
    GraphValidationError
        If graph has multiple connected components. Error message
        includes number of components, size of each, and example nodes
        for debugging.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.assert_connected()  # Passes if fully connected

    >>> # Environment with island raises informative error
    >>> env_with_gap = Environment.from_samples(
    ...     sparse_data, bin_size=2.0, dilate=False
    ... )
    >>> try:
    ...     env_with_gap.assert_connected()
    ... except GraphValidationError as e:
    ...     print(e)
    GraphValidationError: Graph has 2 connected components.
    Component sizes: [100, 5]
    Example nodes: Component 0: [0, 1, 2], Component 1: [105, 106, 107]
    Consider using dilate=True or increasing bin_size.

    Notes
    -----
    Uses NetworkX connected components analysis. For very large
    environments (>100K bins), this check may take a few seconds.

    Common causes of disconnected environments:
    - Sparse data with gaps in coverage
    - Aggressive filtering (high bin_count_threshold)
    - Multiple rooms without connecting passages
    - Barriers or walls in polygon-bounded environments

    See Also
    --------
    assert_no_small_components : Check for isolated bins
    """
```

```python
def assert_no_small_components(self, min_size: int = 2) -> None:
    """
    Assert that all connected components meet minimum size.

    Detects isolated bins or small disconnected regions that may
    result from noise, morphological operations, or data sparsity.

    Parameters
    ----------
    min_size : int, default=2
        Minimum number of bins required in each component.

        **Guidance**:
        - Use min_size=2 to detect isolated single bins
        - Use min_size=10 to detect small noise regions
        - Use min_size=20 for more aggressive filtering

    Raises
    ------
    GraphValidationError
        If any component has fewer than min_size bins. Error message
        includes component sizes and example nodes from small components.
    ValueError
        If min_size < 1.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=2.0, dilate=False)
    >>> # Check for isolated single bins
    >>> env.assert_no_small_components(min_size=2)

    >>> # Allow islands but ensure they're meaningful (>= 10 bins)
    >>> env.assert_no_small_components(min_size=10)

    Notes
    -----
    Useful for detecting problems caused by:
    - Isolated tracking artifacts (single-frame outliers)
    - Morphological operations creating small disconnected regions
    - Border effects in masked environments

    To fix small components:
    - Increase bin_size to smooth over gaps
    - Use dilate=True to connect nearby regions
    - Use close_gaps morphological operation
    - Manually filter input data to remove outliers

    See Also
    --------
    assert_connected : Check if fully connected
    """
```

### 3.2 Planarity Checking (2D only)

**API** (`environment.py`):

```python
def is_planar(self) -> bool:
    """
    Check if the environment connectivity graph is planar.

    Planarity checking validates that 2D spatial relationships are
    preserved correctly in the connectivity graph. Most 2D environments
    are planar; non-planar graphs may indicate bridges, multi-level
    structures, or layout bugs.

    Returns
    -------
    is_planar : bool
        True if graph is planar (can be embedded in 2D plane without
        edge crossings), False otherwise.

    Raises
    ------
    ValueError
        If environment is not 2D (n_dims != 2).

    Examples
    --------
    >>> env = Environment.from_samples(data_2d, bin_size=2.0)
    >>> assert env.is_planar()  # Verify 2D structure is preserved

    >>> # Composite with bridge may not be planar
    >>> from neurospatial import CompositeEnvironment
    >>> comp = CompositeEnvironment([env1, env2], bridges=[(10, 20)])
    >>> print(comp.is_planar())
    False  # Bridge creates non-planar crossing

    Notes
    -----
    Most 2D spatial environments are naturally planar:
    - Regular grids
    - Hexagonal tessellations
    - Simple polygon-bounded regions
    - Single-room mazes

    Non-planar graphs in 2D indicate:
    - CompositeEnvironment with bridges connecting remote locations
    - Multi-floor environments (should use 3D representation instead)
    - Possible layout engine bugs (edge connections not respecting geometry)

    Uses Kuratowski's theorem via NetworkX for planarity testing.

    See Also
    --------
    assert_connected : Validate graph connectivity
    """
```

**Tests**:

- 5×5 regular grid is connected and planar
- Grid with removed 3×3 center has 4 components (corners disconnected)
- Synthetic disconnected graph raises with correct diagnostics
- T-maze graph is planar
- Single-bin environment is trivially connected
- Composite with bridge is non-planar

---

## 4. Boundary Analysis

**Motivation**: Distance to boundaries is important for analyzing wall-following behavior, boundary avoidance, and edge effects in spatial fields.

### 4.1 Boundary Distance Field

**API** (`distance.py`):

```python
def boundary_distance_field(
    env: Environment,
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean"
) -> NDArray[np.float64]:
    """
    Compute distance from each bin to environment boundary.

    Useful for analyzing wall-following behavior, thigmotaxis
    (preference for edges), and detecting boundary cells in
    neural recordings.

    Parameters
    ----------
    env : Environment
        Environment to compute boundary distances for.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric:

        - "euclidean": Straight-line distance in coordinate space
          (as-the-crow-flies distance, faster computation)
        - "geodesic": Shortest path distance along connectivity graph
          (distance following legal paths, respects barriers)

    Returns
    -------
    distances : NDArray[np.float64], shape (n_bins,)
        Distance from each bin center to nearest boundary bin.
        Units match environment coordinates (e.g., cm).

    Examples
    --------
    >>> from neurospatial import Environment, boundary_distance_field
    >>> data = np.random.randn(1000, 2) * 50
    >>> env = Environment.from_samples(data, bin_size=5.0)
    >>> dists = boundary_distance_field(env, metric="euclidean")
    >>> # Find bins near walls (< 10 cm from boundary)
    >>> near_walls = dists < 10.0
    >>> print(f"{near_walls.sum()} bins within 10 cm of walls")

    >>> # Analyze occupancy vs wall distance
    >>> occupancy = compute_occupancy(trajectory, env)
    >>> plt.scatter(dists, occupancy, alpha=0.5)
    >>> plt.xlabel("Distance to boundary (cm)")
    >>> plt.ylabel("Occupancy (seconds)")

    Notes
    -----
    **Boundary detection**: Boundary bins are identified using graph
    topology. Nodes with fewer neighbors than the maximum degree are
    considered boundary nodes. This works for all environment types.

    - Euclidean: Straight-line distance to nearest boundary bin center.
      Fast computation: O(n_bins × n_boundary_bins) or O(n_bins log n_boundary_bins)
      using KDTree.

    - Geodesic: Shortest path distance along connectivity graph to
      nearest boundary bin. Respects barriers and walls.
      Slower computation: O((n_bins + n_edges) log n_bins) using Dijkstra.

    For large environments (>10,000 bins), prefer Euclidean for better
    performance unless geodesic distance is scientifically required.

    **Computational complexity**:
    - Euclidean: O(n_bins log n_boundary) using KDTree
    - Geodesic: O((V + E) log V) using multi-source Dijkstra

    See Also
    --------
    distance_field : Generic multi-source distance computation
    get_boundary_bins : Get boundary bin indices
    """
```

### 4.2 Boundary Identification Helper

**API** (`distance.py`):

```python
def get_boundary_bins(env: Environment) -> NDArray[np.int64]:
    """
    Get indices of bins on environment boundary.

    Boundary bins are those with fewer neighbors than the maximum degree,
    indicating they are on the edge of the environment.

    Parameters
    ----------
    env : Environment
        Environment to identify boundary bins for.

    Returns
    -------
    boundary_indices : NDArray[np.int64], shape (n_boundary_bins,)
        Sorted array of bin indices on boundary.

    Examples
    --------
    >>> from neurospatial import get_boundary_bins
    >>> boundary = get_boundary_bins(env)
    >>> print(f"{len(boundary)} bins on boundary")
    >>> # Compute average neural activity on boundary
    >>> boundary_activity = firing_rates[boundary].mean()

    Notes
    -----
    Uses find_boundary_nodes() from layout helpers, which identifies
    nodes with degree less than the maximum degree in the graph.

    For regular grids, boundary bins are those on the edges and corners.
    For irregular layouts, boundary is determined by connectivity topology.

    See Also
    --------
    boundary_distance_field : Compute distances to boundary
    """
```

**Tests**:

- Square environment: center bin has max distance
- Symmetric environments produce symmetric distance fields
- Boundary bins have distance ≈ 0 (within numerical tolerance)
- Geodesic ≥ Euclidean (triangle inequality)
- Circular arena: center distance ≈ radius

---

## 5. Region-Bin Queries (Performance Enhancement)

**Motivation**: Efficiently mapping regions to bins is essential for ROI analysis, masking operations, and spatial statistics. Current regions don't cache bin mappings, causing repeated expensive computations.

### 5.1 Cached Region Mapping

**Cache implementation**:

- Store cache as `Environment._region_bins_cache: dict` (private attribute)
- Cache key: `(region_name, method, self.regions._version)`
- Invalidation: Regions class gets `_version` counter, incremented on modification
- Manual clearing: `Environment.clear_region_cache()` method

**API** (`environment.py`):

```python
def region_bins(
    self,
    region_name: str,
    *,
    criterion: Literal["contains", "intersects"] = "contains",
    return_type: Literal["indices", "mask"] = "indices"
) -> NDArray[np.int64] | NDArray[np.bool_]:
    """
    Get bins contained in or intersecting a named region.

    Results are cached automatically; repeated calls with the same
    parameters are fast (O(1) lookup vs O(n_bins) computation).

    Parameters
    ----------
    region_name : str
        Name of region (must exist in self.regions).
    criterion : {"contains", "intersects"}, default="contains"
        Inclusion criterion:

        - "contains": Bin center must be inside region geometry
          (stricter, excludes edge bins)
        - "intersects": Bin must overlap region geometry
          (more inclusive, includes partial overlaps)

        For point regions, both behave identically (distance threshold).

    return_type : {"indices", "mask"}, default="indices"
        Output format:

        - "indices": Return sorted integer array of bin indices
          (good for fancy indexing: `field[indices]`)
        - "mask": Return boolean array shape (n_bins,)
          (good for masking: `field[mask] = 0`)

    Returns
    -------
    bins : NDArray[np.int64] or NDArray[np.bool_]
        If return_type="indices": shape (n_bins_in_region,), sorted bin indices
        If return_type="mask": shape (n_bins,), boolean mask

    Raises
    ------
    KeyError
        If region_name not in self.regions.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.regions.add("goal", point=[100, 100])
    >>> # Get bins in goal region
    >>> goal_bins = env.region_bins("goal")
    >>> print(f"Goal region contains {len(goal_bins)} bins")

    >>> # Mask a rate map to only show goal region
    >>> rate_map = compute_rate_map(spikes, env)
    >>> goal_mask = env.region_bins("goal", return_type="mask")
    >>> goal_rate_map = rate_map.copy()
    >>> goal_rate_map[~goal_mask] = 0  # Zero out non-goal bins

    >>> # Compare occupancy in different regions
    >>> start_bins = env.region_bins("start")
    >>> goal_bins = env.region_bins("goal")
    >>> occupancy = compute_occupancy(trajectory, env)
    >>> print(f"Start: {occupancy[start_bins].sum():.1f} s")
    >>> print(f"Goal: {occupancy[goal_bins].sum():.1f} s")

    Notes
    -----
    **Caching**: Results are cached per region. Second call with same
    parameters is ~100× faster (O(1) lookup vs O(n_bins) computation).
    Cache is invalidated automatically when regions are modified.

    **For point regions**: Uses distance threshold of 1.5 × typical bin size.
    Bins whose centers are within this distance are included.

    **For polygon regions**: Uses Shapely geometric predicates
    (contains_properly for "contains", intersects for "intersects").

    **Migrating from manual KDTree queries**:

    Old way (pre-v0.3.0):
        from scipy.spatial import KDTree
        tree = KDTree(env.bin_centers)
        region_center = env.regions['goal'].point
        nearby = tree.query_ball_point(region_center, r=10.0)

    New way (v0.3.0+):
        nearby = env.region_bins('goal')  # Cached and geometry-aware

    See Also
    --------
    region_union : Combine multiple regions
    region_intersection : Find overlapping bins
    region_difference : Set difference of regions
    clear_region_cache : Clear cache manually
    """
```

```python
def clear_region_cache(self) -> None:
    """
    Clear cached region-bin mappings.

    Useful for freeing memory or forcing recomputation after
    manual region modifications (not recommended).

    Examples
    --------
    >>> env.clear_region_cache()
    >>> # Next region_bins() call will recompute

    Notes
    -----
    Cache is automatically invalidated when regions are modified
    through standard methods (add, remove, update_region). Manual
    clearing is rarely needed.
    """
```

### 5.2 Region Set Operations

**API** (`environment.py`):

```python
def region_union(
    self,
    *region_names: str,
    criterion: Literal["contains", "intersects"] = "contains",
    return_type: Literal["indices", "mask"] = "mask"
) -> NDArray[np.int64] | NDArray[np.bool_]:
    """
    Get bins in the union of multiple regions.

    Union returns bins that are in ANY of the specified regions.

    Parameters
    ----------
    *region_names : str
        Names of regions to union (must exist in self.regions).
    criterion : {"contains", "intersects"}, default="contains"
        Inclusion criterion applied to each region independently.
    return_type : {"indices", "mask"}, default="mask"
        Output format (see region_bins documentation).

    Returns
    -------
    bins : NDArray[np.int64] or NDArray[np.bool_]
        Bins in any of the regions (set union).

    Examples
    --------
    >>> # Analyze occupancy in start or goal regions
    >>> start_or_goal = env.region_union("start", "goal")
    >>> occupancy = compute_occupancy(trajectory, env)
    >>> roi_occupancy = occupancy[start_or_goal].sum()

    Notes
    -----
    For combining regions with different criteria, call region_bins()
    separately:

        bins_a = env.region_bins("A", criterion="contains")
        bins_b = env.region_bins("B", criterion="intersects")
        union = np.union1d(bins_a, bins_b)

    See Also
    --------
    region_intersection : Bins in all regions
    region_difference : Bins in one region but not another
    """
```

```python
def region_intersection(
    self,
    *region_names: str,
    criterion: Literal["contains", "intersects"] = "contains",
    return_type: Literal["indices", "mask"] = "mask"
) -> NDArray[np.int64] | NDArray[np.bool_]:
    """
    Get bins in the intersection of multiple regions.

    Intersection returns bins that are in ALL of the specified regions
    (overlapping area).

    Parameters
    ----------
    *region_names : str
        Names of regions to intersect (must exist in self.regions).
    criterion : {"contains", "intersects"}, default="contains"
        Inclusion criterion applied to each region independently.
    return_type : {"indices", "mask"}, default="mask"
        Output format (see region_bins documentation).

    Returns
    -------
    bins : NDArray[np.int64] or NDArray[np.bool_]
        Bins in all of the regions (set intersection).

    Examples
    --------
    >>> # Find bins that are both near goal and high-occupancy
    >>> env.regions.add("high_occ", polygon=high_occ_polygon)
    >>> overlap = env.region_intersection("goal", "high_occ")
    >>> print(f"{overlap.sum() if isinstance(overlap, np.ndarray) else len(overlap)} bins in both regions")

    See Also
    --------
    region_union : Bins in any region
    region_difference : Bins in one region but not another
    """
```

```python
def region_difference(
    self,
    region_a: str,
    region_b: str,
    *,
    criterion: Literal["contains", "intersects"] = "contains",
    return_type: Literal["indices", "mask"] = "mask"
) -> NDArray[np.int64] | NDArray[np.bool_]:
    """
    Get bins in region_a but not in region_b.

    Set difference is useful for excluding subregions or comparing
    partially overlapping ROIs.

    Parameters
    ----------
    region_a : str
        Base region name (must exist in self.regions).
    region_b : str
        Region to subtract (must exist in self.regions).
    criterion : {"contains", "intersects"}, default="contains"
        Inclusion criterion applied to both regions.
    return_type : {"indices", "mask"}, default="mask"
        Output format (see region_bins documentation).

    Returns
    -------
    bins : NDArray[np.int64] or NDArray[np.bool_]
        Bins in region_a but not in region_b (set difference).

    Examples
    --------
    >>> # Analyze goal region excluding center
    >>> env.regions.add("goal", point=[100, 100])
    >>> env.regions.add("goal_center", point=[100, 100])  # Smaller radius
    >>> goal_periphery = env.region_difference("goal", "goal_center")
    >>> periphery_rate = firing_rates[goal_periphery].mean()

    See Also
    --------
    region_union : Bins in any region
    region_intersection : Bins in all regions
    """
```

**Tests**:

- Caching: second call with same region name is >10× faster
- Union/intersection follow set theory (verified with small examples)
- De Morgan's laws: NOT (A OR B) = (NOT A) AND (NOT B)
- Point regions include bins within threshold distance
- Polygon regions respect geometric boundaries
- Empty intersection returns empty array/all-False mask
- Single-region union equals original region

---

## 6. Field Operators (Mask-Aware)

**Motivation**: Current field operations don't explicitly handle inactive/masked bins. Neuroscience data often has regions excluded from analysis (walls, unexplored areas).

### 6.1 Masked Field Operations

**API** (`field_ops.py`):

```python
def masked_normalize(
    field: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
    *,
    eps: float = 1e-12
) -> NDArray[np.float64]:
    """
    Normalize field to sum to 1 over active (masked) bins only.

    Unlike normalize_field(), this handles spatial masks for fields
    defined over regions with inactive bins (walls, barriers, unexplored).

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values to normalize. Must be non-negative.
    mask : NDArray[np.bool_], shape (n_bins,), optional
        Boolean mask where True indicates active bins to include in
        normalization. If None, all bins are active (equivalent to
        normalize_field).
    eps : float, default=1e-12
        Small constant for numerical stability.

    Returns
    -------
    normalized : NDArray[np.float64], shape (n_bins,)
        Normalized field. Active bins sum to 1. Inactive bins are
        set to 0.

    Raises
    ------
    ValueError
        If all active bins have zero or negative values.
        If mask is all False (no active bins to normalize over).
        If field contains NaN or Inf values.
        If eps <= 0.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import masked_normalize
    >>> field = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mask = np.array([True, True, False, False])
    >>> normalized = masked_normalize(field, mask)
    >>> normalized
    array([0.333..., 0.666..., 0., 0.])
    >>> normalized[mask].sum()  # Active bins sum to 1
    1.0

    >>> # Normalize place field over explored region only
    >>> explored = env.region_bins("explored", return_type="mask")
    >>> place_field = spike_counts / occupancy
    >>> normalized_field = masked_normalize(place_field, explored)

    Notes
    -----
    For empty masks (all False), raises ValueError. Check explicitly
    if you want to handle this case:

        if mask.any():
            result = masked_normalize(field, mask)
        else:
            result = np.zeros_like(field)

    See Also
    --------
    normalize_field : Normalize without masking (simpler, faster)
    apply_field_mask : Apply mask with custom fill value
    """
```

```python
def apply_field_mask(
    field: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    fill_value: float = 0.0
) -> NDArray[np.float64]:
    """
    Apply mask to field, setting inactive bins to fill value.

    Simple utility for masking operations on spatial fields.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values.
    mask : NDArray[np.bool_], shape (n_bins,)
        Boolean mask where True indicates bins to keep.
    fill_value : float, default=0.0
        Value to use for masked-out (False) bins.

        **Common choices**:
        - 0.0: Zero out masked bins (default, good for plotting)
        - np.nan: Mark as missing data (preserves masked regions visually)
        - -np.inf: Flag as invalid for certain analyses

    Returns
    -------
    masked_field : NDArray[np.float64], shape (n_bins,)
        Field with inactive bins set to fill_value.

    Examples
    --------
    >>> field = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mask = np.array([True, True, False, False])
    >>> masked = apply_field_mask(field, mask, fill_value=np.nan)
    >>> masked
    array([1., 2., nan, nan])

    >>> # Mask rate map to show only explored region
    >>> rate_map = compute_rate_map(spikes, env)
    >>> explored = env.region_bins("explored", return_type="mask")
    >>> visible_map = apply_field_mask(rate_map, explored)

    Notes
    -----
    Equivalent to:
        result = field.copy()
        result[~mask] = fill_value

    Provided as convenience function for clarity in field operations.

    See Also
    --------
    masked_normalize : Normalize over active bins only
    """
```

### 6.2 Piecewise Field Construction

**API** (`field_ops.py`):

```python
def piecewise_field(
    regions: Sequence[NDArray[np.bool_]],
    values: Sequence[float],
    n_bins: int | None = None,
    *,
    overlap_policy: Literal["priority", "average", "raise"] = "priority"
) -> NDArray[np.float64]:
    """
    Construct field from piecewise constant values over regions.

    Useful for creating synthetic fields, defining reward landscapes,
    or building template spatial patterns.

    Parameters
    ----------
    regions : Sequence[NDArray[np.bool_]]
        List of boolean masks, each shape (n_bins,), defining regions.
        Regions may overlap depending on overlap_policy.
    values : Sequence[float]
        Constant value for each region. Must have same length as regions.
    n_bins : int, optional
        Total number of bins. If None, inferred from first region mask length.
        All region masks must have this length.
    overlap_policy : {"priority", "average", "raise"}, default="priority"
        How to handle bins in multiple regions:

        - "priority": Use value from first region in list (priority order)
        - "average": Average values across all containing regions
        - "raise": Raise ValueError if any overlaps exist

    Returns
    -------
    field : NDArray[np.float64], shape (n_bins,)
        Piecewise field. Bins not in any region have value 0.

    Raises
    ------
    ValueError
        If regions and values have different lengths.
        If n_bins is specified but regions have different lengths.
        If regions is empty.
        If overlap_policy="raise" and overlaps exist.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import piecewise_field
    >>> n_bins = 10
    >>> region_a = np.array([True]*3 + [False]*7)
    >>> region_b = np.array([False]*3 + [True]*4 + [False]*3)
    >>> field = piecewise_field([region_a, region_b], [1.0, 0.5])
    >>> field
    array([1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. ])

    >>> # Create reward landscape with goal and penalty regions
    >>> goal_region = env.region_bins("goal", return_type="mask")
    >>> penalty_region = env.region_bins("barrier", return_type="mask")
    >>> reward_field = piecewise_field(
    ...     [goal_region, penalty_region],
    ...     [10.0, -5.0],
    ...     overlap_policy="average"
    ... )

    Notes
    -----
    For overlapping regions with overlap_policy="priority", the order
    of regions matters: earlier regions take precedence.

    See Also
    --------
    apply_field_mask : Apply mask with custom fill value
    masked_normalize : Normalize over active bins
    """
```

**Tests**:

- Masked normalize preserves sum=1 over active bins
- Overlapping regions handled correctly per policy
- Empty mask raises with clear error message
- All-False mask raises ValueError
- Piecewise field with non-overlapping regions has correct values
- Priority policy respects region order

---

## 7. Graph Topology Operators

**Motivation**: Analyzing spatial neighborhoods and graph structure is useful for understanding connectivity, implementing local field operations, and defining spatial contexts.

### 7.1 Spatial Neighborhoods

**API** (`distance.py`):

```python
def spatial_neighbors(
    env: Environment,
    source: int,
    radius: int,
    *,
    exact: bool = False
) -> NDArray[np.int64]:
    """
    Find bins within graph-distance radius of a source bin.

    Spatial neighborhoods are useful for defining local contexts,
    analyzing spatial relationships, and implementing local field
    smoothing operations.

    Parameters
    ----------
    env : Environment
        Environment defining the spatial discretization.
    source : int
        Source bin index (must be valid bin in environment).
    radius : int
        Graph distance radius in number of connectivity hops.

        **Guidance**:
        - radius=1: Immediate neighbors (adjacent bins)
        - radius=2-3: Local neighborhood (typical for smoothing)
        - radius=5+: Regional context (may include many bins)

    exact : bool, default=False
        If True, return only bins at exactly radius distance (ring).
        If False, return all bins within radius (ball, inclusive).

    Returns
    -------
    neighbors : NDArray[np.int64], shape (n_neighbors,)
        Sorted array of bin indices within radius of source.

        - If exact=False: All bins with distance <= radius (including source)
        - If exact=True: Only bins with distance == radius (excluding source if radius > 0)

    Raises
    ------
    ValueError
        If source is not a valid bin index.
        If radius < 0.

    Examples
    --------
    >>> from neurospatial import spatial_neighbors
    >>> env = Environment.from_samples(data, bin_size=5.0)
    >>> # Get immediate neighbors of bin 42
    >>> adjacent = spatial_neighbors(env, source=42, radius=1, exact=True)
    >>> print(f"Bin 42 has {len(adjacent)} neighbors")

    >>> # Compute local average activity in 2-hop neighborhood
    >>> neighborhood = spatial_neighbors(env, source=42, radius=2)
    >>> local_activity = firing_rates[neighborhood].mean()

    >>> # Analyze how activity spreads from center bin
    >>> for r in range(1, 6):
    ...     ring = spatial_neighbors(env, source=center_bin, radius=r, exact=True)
    ...     avg_activity = firing_rates[ring].mean()
    ...     print(f"Radius {r}: {avg_activity:.2f} Hz")

    Notes
    -----
    Uses BFS (breadth-first search) to find bins at specified graph distance.
    Does not use edge weights; counts topological hops along connectivity.

    For distance measured in physical units (cm), use distance_field()
    and threshold instead:

        dists = distance_field(env.connectivity, sources=[source])
        nearby = np.where(dists <= radius_cm)[0]

    Computational complexity: O(V + E) where V = n_bins, E = n_edges.

    See Also
    --------
    distance_field : Physical distance from source(s)
    boundary_distance_field : Distance to environment boundary
    """
```

**Tests**:

- Linear graph (5 bins in row): radius=2 ball from center has 5 bins
- Grid graph: 1-hop neighbors count matches expected (4 for interior, less for edges)
- Ring (exact=True) doesn't include source when radius > 0
- Ball (exact=False) includes source
- 0-radius ball returns only source bin

---

## Implementation Strategy

### Phase 1: Trajectory & Geometry (Week 1)

1. Implement trajectory primitives in `spatial.py`
   - `heading()`, `angular_velocity()`, `curvature()`
   - `path_length()`, `straightness()`
2. Add comprehensive input validation and error messages
3. Add doctests and unit tests for straight/circular paths
4. Test with noisy synthetic trajectories

### Phase 2: Segment Crossing (Week 1-2)

1. Implement DDA algorithm for regular grids
2. Implement sampling-based approach for irregular layouts
3. Add layout detection logic (check for `grid_shape` attribute)
4. Test on 5×5 grid, hexagonal layout, graph layout
5. Verify traversal order correctness

### Phase 3: Topology & Validation (Week 2)

1. Import `GraphValidationError` from `layout.validation`
2. Implement `assert_connected()`, `assert_no_small_components()`
3. Add `is_planar()` with dimension checking
4. Test with disconnected synthetic graphs
5. Verify error messages include helpful diagnostics

### Phase 4: Boundary Analysis (Week 2-3)

1. Implement `boundary_distance_field()` using existing `distance_field()`
2. Add `get_boundary_bins()` helper
3. Integrate with `find_boundary_nodes()` from layout helpers
4. Test on square, circular, irregular environments
5. Benchmark performance (Euclidean vs geodesic)

### Phase 5: Regions & Caching (Week 3-4)

1. Add `_version` counter to `Regions` class
2. Implement cache in `Environment._region_bins_cache`
3. Add `region_bins()` with caching logic and `return_type` parameter
4. Implement set operations: union, intersection, difference
5. Add `clear_region_cache()` method
6. Test cache invalidation on region modifications
7. Benchmark caching performance (should be >10× faster)

### Phase 6: Field Operations (Week 4)

1. Implement `masked_normalize()` with comprehensive validation
2. Add `apply_field_mask()` utility
3. Implement `piecewise_field()` with overlap policies
4. Test with synthetic fields and real occupancy data
5. Verify sum-to-one property for normalized masked fields

### Phase 7: Graph Operators (Week 4-5)

1. Implement `spatial_neighbors()` using BFS
2. Test on various graph topologies
3. Add performance tests for large grids (100×100)

### Phase 8: Documentation & Polish (Week 5)

1. Review all docstrings for completeness
2. Add cross-references between related functions
3. Write integration examples showing workflows
4. Update `__all__` in relevant modules
5. Update CHANGELOG with all additions

---

## Testing Plan

### Test Environments

**Existing fixtures** (reuse from conftest.py):

- 5×5 regular grid
- Plus maze graph
- Square polygon environment

**New fixtures** (add to conftest.py):

- Circular arena (test boundary distance symmetry)
- T-maze with gap (test connectivity assertions)
- Hexagonal grid (test segment crossing on non-square layout)
- Line world (1D, test neighborhood operations)

### Test Coverage Requirements

Each new function requires:

1. **Unit tests**:
   - Happy path with typical inputs
   - Edge cases (empty, single-bin, zero-length)
   - Error cases with validation
   - Boundary conditions (NaN, Inf handling)

2. **Doctests**:
   - At least one working example in docstring
   - Realistic parameter values
   - Output shown or tested

3. **Integration tests**:
   - Combining multiple new functions
   - Using with existing neurospatial APIs
   - Performance benchmarks for expensive operations

4. **Property tests** (where applicable):
   - Monotonicity (cumulative path_length)
   - Bounds (straightness ∈ \[0, 1\])
   - Symmetry (distance fields on symmetric environments)
   - Conservation laws (masked normalization sums to 1)

### Performance Benchmarks

- Region caching: Should be >10× faster on repeated calls
- Segment crossing: Should handle 1000-point paths in <100ms on 100×100 grid
- Boundary distance field (Euclidean): <1 second for 10,000 bins
- Boundary distance field (geodesic): <5 seconds for 10,000 bins
- spatial_neighbors: <10ms for radius=5 on 100×100 grid

---

## Documentation Requirements

### Docstring Components (NumPy Format)

Each public function must include:

1. **Short description** (one line)
2. **Long description** (motivation, use cases, behavior)
3. **Parameters** section (types, defaults, guidance on choices)
4. **Returns** section (types, shapes, units, interpretation)
5. **Raises** section (all exceptions with conditions)
6. **Examples** section (at least one realistic workflow)
7. **Notes** section (algorithms, complexity, edge cases)
8. **See Also** section (cross-references to related functions)

### User Guidance in Docstrings

For parameters requiring user choices:

- Provide typical value ranges
- Give examples based on common recording setups (30 Hz video, etc.)
- Explain trade-offs (speed vs accuracy, smoothing vs detail)
- Link to neuroscience terminology when relevant

For return values:

- Specify units explicitly
- Give typical value ranges for neuroscience data
- Explain interpretation in scientific context

### Cross-References

Aggressively cross-reference related functions:

- Trajectory functions reference each other
- Region operations link to field masking operations
- Distance functions reference each other
- Each function links to integration examples

### Integration Examples

Create example workflows showing:

- Analyzing trajectory from recording to metrics
- Computing place field with region masking
- Validating environment and computing boundary distances
- Using region set operations for ROI analysis

---

## Public API Updates

### Module: `spatial.py`

Add to `__all__`:

- `heading`
- `angular_velocity`
- `curvature`
- `path_length`
- `straightness`
- `bins_crossed_by_segment`
- `bins_crossed_by_path`

### Module: `distance.py`

Add to `__all__`:

- `boundary_distance_field`
- `get_boundary_bins`
- `spatial_neighbors`

### Module: `field_ops.py`

Add to `__all__`:

- `masked_normalize`
- `apply_field_mask`
- `piecewise_field`

### Module: `environment.py`

Add public methods (no `__all__` changes needed):

- `assert_connected()`
- `assert_no_small_components()`
- `is_planar()`
- `region_bins()`
- `region_union()`
- `region_intersection()`
- `region_difference()`
- `clear_region_cache()`

### Module: `neurospatial/__init__.py`

Add to top-level imports:

```python
from neurospatial.spatial import (
    heading,
    angular_velocity,
    curvature,
    path_length,
    straightness,
    bins_crossed_by_segment,
    bins_crossed_by_path,
)
from neurospatial.distance import (
    boundary_distance_field,
    get_boundary_bins,
    spatial_neighbors,
)
from neurospatial.field_ops import (
    masked_normalize,
    apply_field_mask,
    piecewise_field,
)
```

---

## Non-Goals

**Explicitly out of scope for this enhancement**:

- RL state/action abstractions (state IDs, legal actions, step functions)
- Reward function modeling or value functions
- Policy representations or transition matrices for decision-making
- Gym/gymnasium integration or MDP export
- Any learning or optimization algorithms
- Visualization functions (defer to future plotting module)

**Why excluded**: These belong in a separate RL adapter package built on neurospatial (e.g., `neurospatial-rl`), not in the core spatial analysis library focused on neuroscience use cases.

---

## Migration Notes

### For Existing Users

No breaking changes to existing APIs. All additions are new functions or methods.

**Recommended updates**:

1. **Replace manual boundary detection**:

   ```python
   # Old way
   from neurospatial.layout.helpers.utils import find_boundary_nodes
   boundary = find_boundary_nodes(env.connectivity)

   # New way (v0.3.0+)
   from neurospatial import get_boundary_bins
   boundary = get_boundary_bins(env)
   ```

2. **Replace manual region-bin mapping**:

   ```python
   # Old way
   from scipy.spatial import KDTree
   tree = KDTree(env.bin_centers)
   nearby = tree.query_ball_point(region_center, r=10.0)

   # New way (v0.3.0+)
   nearby = env.region_bins('region_name')  # Cached automatically
   ```

3. **Use trajectory primitives instead of manual computation**:

   ```python
   # Old way
   dxy = np.diff(trajectory, axis=0)
   headings = np.arctan2(dxy[:, 1], dxy[:, 0])

   # New way (v0.3.0+)
   from neurospatial import heading
   headings = heading(trajectory, differencing="forward")
   ```

---

## Summary

This plan adds 23 new functions organized into 7 categories:

1. **Trajectory primitives** (5 functions): heading, angular_velocity, curvature, path_length, straightness
2. **Segment intersection** (2 functions): bins_crossed_by_segment, bins_crossed_by_path
3. **Topology validation** (3 functions): assert_connected, assert_no_small_components, is_planar
4. **Boundary analysis** (2 functions): boundary_distance_field, get_boundary_bins
5. **Region queries** (5 functions): region_bins, region_union, region_intersection, region_difference, clear_region_cache
6. **Field operators** (3 functions): masked_normalize, apply_field_mask, piecewise_field
7. **Graph topology** (1 function): spatial_neighbors

All additions:

- Support neuroscience spatial analysis use cases
- Maintain architectural consistency with existing codebase
- Follow NumPy docstring format with comprehensive documentation
- Include realistic examples and parameter guidance
- Have explicit error handling with helpful diagnostics
- Are tested thoroughly with multiple environments
- Use consistent naming conventions (no `compute_` prefix)
- Avoid RL-specific abstractions (stay focused on spatial geometry)
