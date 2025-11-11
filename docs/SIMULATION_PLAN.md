# Simulation Subpackage Plan (Practical Neuroscience Edition)

**Design Philosophy**: Simple, practical, coverage-focused simulation for testing place field algorithms.

**Key Insight**: Use neurospatial's graph structure to generate realistic, coverage-ensuring trajectories.

---

## Core Problems to Solve

1. **Coverage**: Place field estimation requires sampling most/all of the environment
2. **Goal-directed behavior**: Animals run from reward well to reward well, not random exploration
3. **Simplicity**: Ship minimal viable code, iterate based on usage

**Not solving**: Reinforcement learning, complex movement dynamics, RatInABox integration

---

## Raymond-Approved Design: One File, Three Functions

```python
# neurospatial/simulation.py (~200 lines total)

"""Generate synthetic neural data for testing and validation.

Practical simulation focused on what neuroscientists actually need:
- Trajectories that cover the environment
- Goal-directed movement between reward locations
- Place cells with known ground truth

Examples
--------
>>> env = Environment.from_samples(arena_data, bin_size=2.0)
>>>
>>> # Generate coverage trajectory
>>> positions, times = simulate_trajectory(env, duration=120.0)
>>>
>>> # Create place cell population
>>> place_cells = [PlaceCell(env, center=c) for c in env.bin_centers[::5]]
>>>
>>> # Generate spikes
>>> spike_trains = [generate_spikes(pc.firing_rate(positions), times)
...                 for pc in place_cells]
>>>
>>> # Validate detection
>>> for pc, spikes in zip(place_cells, spike_trains):
...     rate_map = compute_place_field(env, spikes, times, positions)
...     detected = field_centroid(rate_map, env)
...     error = np.linalg.norm(detected - pc.center)
...     print(f"Error: {error:.2f} cm")
"""

__all__ = ['simulate_trajectory', 'simulate_goal_directed', 'PlaceCell',
           'BoundaryCell', 'generate_spikes']
```

---

## 1. Trajectory Simulation - Use the Graph!

### Coverage Trajectory (Default)

**Problem**: OU process might miss parts of environment. We need good coverage.

**Solution**: Use graph structure to systematically visit bins.

```python
def simulate_trajectory(
    env: Environment,
    duration: float,
    *,
    speed: float = 0.08,  # m/s (8 cm/s)
    pause_at_boundaries: float = 0.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate trajectory with good spatial coverage.

    Creates realistic exploration by walking through the environment's
    graph structure, ensuring most/all bins are visited.

    Strategy:
    - Start at random bin
    - Perform biased random walk on connectivity graph
    - Bias toward unvisited or less-visited bins
    - Results in good coverage without looking artificial

    Parameters
    ----------
    env : Environment
        Spatial environment.
    duration : float
        Total simulation time in seconds.
    speed : float, optional
        Movement speed in environment units/second (default: 0.08 m/s).
    pause_at_boundaries : float, optional
        Pause duration at boundary bins in seconds (default: 0.0).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates sampled at 100 Hz.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.

    Notes
    -----
    The trajectory ensures good coverage by:
    1. Tracking occupancy of each bin
    2. Preferentially moving to less-visited neighbors
    3. Adding small position jitter within bins for realism

    This produces realistic-looking trajectories while guaranteeing
    sufficient sampling for place field estimation.

    Examples
    --------
    >>> # Open field - get good coverage
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> positions, times = simulate_trajectory(env, duration=180.0)
    >>>
    >>> # Check coverage
    >>> bin_counts = env.occupancy_counts(times, positions)
    >>> coverage = (bin_counts > 0).sum() / env.n_bins
    >>> print(f"Visited {coverage*100:.1f}% of bins")
    Visited 95.3% of bins

    >>> # Linear track with pauses
    >>> env_track = Environment.from_graph(track_graph, edge_order, bin_size=1.0)
    >>> positions, times = simulate_trajectory(
    ...     env_track, duration=120.0, pause_at_boundaries=0.5
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 0.01  # 100 Hz sampling
    n_samples = int(duration / dt)

    # Initialize trajectory
    current_bin = np.random.randint(env.n_bins)
    bin_sequence = [current_bin]
    occupancy_counts = np.zeros(env.n_bins)
    occupancy_counts[current_bin] = 1

    # Generate bin sequence by biased random walk
    boundary_bins = set(env.boundary_bins())

    while len(bin_sequence) < n_samples:
        # Get neighbors
        neighbors = list(env.connectivity.neighbors(current_bin))

        if not neighbors:
            # Isolated bin - stay here
            bin_sequence.append(current_bin)
            continue

        # Bias toward less-visited bins
        neighbor_weights = 1.0 / (1.0 + occupancy_counts[neighbors])
        neighbor_weights /= neighbor_weights.sum()

        # Choose next bin
        next_bin = np.random.choice(neighbors, p=neighbor_weights)

        # Move to next bin (interpolate for smooth motion)
        distance = env.distance_between(current_bin, next_bin)
        n_steps = max(1, int(distance / (speed * dt)))

        bin_sequence.extend([next_bin] * n_steps)
        occupancy_counts[next_bin] += n_steps

        # Pause at boundaries if requested
        if pause_at_boundaries > 0 and next_bin in boundary_bins:
            pause_steps = int(pause_at_boundaries / dt)
            bin_sequence.extend([next_bin] * pause_steps)
            occupancy_counts[next_bin] += pause_steps

        current_bin = next_bin

    # Trim to exact duration
    bin_sequence = np.array(bin_sequence[:n_samples])

    # Convert bin indices to continuous positions (with jitter)
    positions = env.bin_centers[bin_sequence].copy()

    # Add small random jitter within bins for realism
    jitter_std = env.bin_size * 0.2  # 20% of bin size
    positions += np.random.randn(*positions.shape) * jitter_std

    times = np.arange(n_samples) * dt

    return positions, times
```

### Goal-Directed Trajectory

**Problem**: Animals run from reward well to reward well in experiments.

**Solution**: Use graph shortest paths!

```python
def simulate_goal_directed(
    env: Environment,
    goals: list[NDArray[np.float64]],
    n_trials: int,
    *,
    speed: float = 0.08,
    pause_at_goal: float = 1.0,
    trial_order: Literal["sequential", "random", "alternating"] = "sequential",
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Generate goal-directed trajectory between reward locations.

    Simulates realistic experimental paradigms where animals run between
    reward wells (linear track, T-maze, plus maze, etc.).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goals : list of array-like, each shape (n_dims,)
        Reward locations in environment coordinates.
        Example: [[0, 0], [100, 100]] for two corners
    n_trials : int
        Number of goal-to-goal runs to simulate.
    speed : float, optional
        Movement speed in units/second (default: 0.08 m/s).
    pause_at_goal : float, optional
        Pause duration at each goal in seconds (default: 1.0).
    trial_order : {'sequential', 'random', 'alternating'}, optional
        How to sequence goals (default: 'sequential').
        - 'sequential': goal[0]→goal[1]→goal[0]→goal[1]...
        - 'random': random goal each trial
        - 'alternating': goal[0]→goal[1]→goal[2]→goal[0]→goal[1]...
    seed : int | None, optional
        Random seed.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates at 100 Hz.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.
    trial_ids : NDArray[np.int_], shape (n_times,)
        Trial number for each time point.

    Notes
    -----
    Uses Environment.shortest_path() to find routes between goals.
    Movement along path is smooth with constant speed.

    Examples
    --------
    >>> # Linear track - run back and forth
    >>> env = Environment.from_graph(track_graph, edge_order, bin_size=1.0)
    >>> goals = [
    ...     env.bin_centers[0],      # Start of track
    ...     env.bin_centers[-1]      # End of track
    ... ]
    >>> positions, times, trials = simulate_goal_directed(
    ...     env, goals, n_trials=20, trial_order='sequential'
    ... )

    >>> # T-maze alternation
    >>> env = Environment.from_polygon(tmaze_polygon, bin_size=2.0)
    >>> goals = [
    ...     [10, 80],   # Left arm
    ...     [90, 80],   # Right arm
    ... ]
    >>> positions, times, trials = simulate_goal_directed(
    ...     env, goals, n_trials=40, trial_order='alternating'
    ... )

    >>> # Plus maze - visit all arms
    >>> env = Environment.from_polygon(plus_maze_polygon, bin_size=2.0)
    >>> goals = [
    ...     [50, 10],   # South arm
    ...     [90, 50],   # East arm
    ...     [50, 90],   # North arm
    ...     [10, 50],   # West arm
    ... ]
    >>> positions, times, trials = simulate_goal_directed(
    ...     env, goals, n_trials=100, trial_order='random'
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 0.01  # 100 Hz

    # Convert goals to bin indices
    goal_bins = [env.bin_at(np.array(g)[np.newaxis, :])[0] for g in goals]

    # Generate trial sequence
    if trial_order == "sequential":
        trial_sequence = np.tile(np.arange(len(goals)), n_trials // len(goals) + 1)[:n_trials]
    elif trial_order == "alternating":
        trial_sequence = []
        for _ in range(n_trials):
            trial_sequence.append(np.arange(len(goals)))
        trial_sequence = np.concatenate(trial_sequence)[:n_trials]
    elif trial_order == "random":
        trial_sequence = np.random.choice(len(goals), n_trials)
    else:
        raise ValueError(f"Unknown trial_order: {trial_order}")

    # Build trajectory
    all_positions = []
    all_times = []
    all_trial_ids = []
    current_time = 0.0
    current_bin = goal_bins[trial_sequence[0]]

    for trial_idx, goal_idx in enumerate(trial_sequence):
        goal_bin = goal_bins[goal_idx]

        # Find shortest path
        path = env.shortest_path(current_bin, goal_bin)

        if path is None:
            # No path - skip this trial
            continue

        # Walk along path
        for i in range(len(path) - 1):
            bin_a, bin_b = path[i], path[i + 1]
            distance = env.distance_between(bin_a, bin_b)
            duration = distance / speed
            n_steps = max(1, int(duration / dt))

            # Interpolate positions
            pos_a = env.bin_centers[bin_a]
            pos_b = env.bin_centers[bin_b]
            alphas = np.linspace(0, 1, n_steps, endpoint=False)
            segment_positions = pos_a + alphas[:, np.newaxis] * (pos_b - pos_a)

            segment_times = current_time + np.arange(n_steps) * dt
            segment_trial_ids = np.full(n_steps, trial_idx)

            all_positions.append(segment_positions)
            all_times.append(segment_times)
            all_trial_ids.append(segment_trial_ids)

            current_time = segment_times[-1] + dt

        # Pause at goal
        if pause_at_goal > 0:
            pause_steps = int(pause_at_goal / dt)
            pause_positions = np.tile(env.bin_centers[goal_bin], (pause_steps, 1))
            pause_times = current_time + np.arange(pause_steps) * dt
            pause_trial_ids = np.full(pause_steps, trial_idx)

            all_positions.append(pause_positions)
            all_times.append(pause_times)
            all_trial_ids.append(pause_trial_ids)

            current_time = pause_times[-1] + dt

        current_bin = goal_bin

    positions = np.vstack(all_positions)
    times = np.concatenate(all_times)
    trial_ids = np.concatenate(all_trial_ids)

    return positions, times, trial_ids
```

---

## 2. Neural Models - Just the Essentials

### Place Cell

```python
class PlaceCell:
    """Gaussian place field model.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    center : array-like, shape (n_dims,), optional
        Field center. If None, random bin center chosen.
    width : float, optional
        Field width (Gaussian std dev). Default: 3 * bin_size.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate in Hz (default: 0.1).

    Attributes
    ----------
    center : ndarray, shape (n_dims,)
        True field center (ground truth for validation).

    Examples
    --------
    >>> # Single place cell
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> pc = PlaceCell(env, center=[50, 75], width=10.0)
    >>>
    >>> # Population covering environment
    >>> pcs = [PlaceCell(env, center=c, width=8.0)
    ...        for c in env.bin_centers[::5]]
    >>>
    >>> # Direction-selective place cell (using lambda)
    >>> def outbound_rate(pc, positions):
    ...     base_rate = pc._gaussian_rate(positions)
    ...     velocity = np.gradient(positions[:, 0])
    ...     return base_rate * (velocity > 0)
    >>>
    >>> # Just modify firing_rate afterward - no need for "condition" parameter
    """

    def __init__(
        self,
        env: Environment,
        center: NDArray[np.float64] | None = None,
        width: float | None = None,
        max_rate: float = 20.0,
        baseline_rate: float = 0.1,
    ):
        self.env = env
        self.center = (
            np.array(center) if center is not None
            else env.bin_centers[np.random.randint(env.n_bins)]
        )
        self.width = width if width is not None else 3 * env.bin_size
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate

    def firing_rate(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute firing rate at positions.

        Parameters
        ----------
        positions : ndarray, shape (n_times, n_dims)
            Position coordinates.

        Returns
        -------
        rates : ndarray, shape (n_times,)
            Firing rate in Hz at each position.
        """
        # Euclidean distance from center
        distances = np.linalg.norm(positions - self.center, axis=1)

        # Gaussian tuning
        rates = np.exp(-0.5 * (distances / self.width) ** 2)
        rates = rates * (self.max_rate - self.baseline_rate) + self.baseline_rate

        return rates
```

### Boundary Cell

```python
class BoundaryCell:
    """Distance-to-boundary tuned firing.

    Models border cells and boundary vector cells (BVCs).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    preferred_distance : float, optional
        Distance from boundary where firing peaks (default: 5.0).
    tolerance : float, optional
        Tuning width (default: 3.0).
    max_rate : float, optional
        Peak firing rate (default: 15.0).
    baseline_rate : float, optional
        Baseline rate (default: 0.1).
    distance_metric : {'geodesic', 'euclidean'}, optional
        Distance calculation method (default: 'geodesic').

    Attributes
    ----------
    preferred_distance : float
        Ground truth preferred distance.

    Examples
    --------
    >>> # Classic border cell
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> bc = BoundaryCell(env, preferred_distance=5.0)
    >>>
    >>> # Validate border_score metric
    >>> positions, times = simulate_trajectory(env, duration=180.0)
    >>> rates = bc.firing_rate(positions)
    >>> spikes = generate_spikes(rates, times)
    >>> rate_map = compute_place_field(env, spikes, times, positions)
    >>> score = border_score(rate_map, env)
    >>> print(f"Border score: {score:.2f}")  # Should be > 0.5
    """

    def __init__(
        self,
        env: Environment,
        preferred_distance: float = 5.0,
        tolerance: float = 3.0,
        max_rate: float = 15.0,
        baseline_rate: float = 0.1,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ):
        self.env = env
        self.preferred_distance = preferred_distance
        self.tolerance = tolerance
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate
        self.distance_metric = distance_metric

        # Pre-compute boundary bins
        self._boundary_bins = set(env.boundary_bins())

    def firing_rate(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute boundary-tuned firing rate."""
        # Map positions to bins
        bin_indices = self.env.bin_at(positions)

        # Compute distance to nearest boundary for each bin
        distances = np.zeros(len(positions))

        for i, bin_idx in enumerate(bin_indices):
            if bin_idx < 0:
                distances[i] = np.inf
                continue

            if bin_idx in self._boundary_bins:
                distances[i] = 0.0
            else:
                # Distance to nearest boundary bin
                if self.distance_metric == "geodesic":
                    # Use graph distances
                    dists = nx.single_source_dijkstra_path_length(
                        self.env.connectivity, bin_idx
                    )
                    boundary_dists = [
                        dists[b] for b in self._boundary_bins if b in dists
                    ]
                    distances[i] = min(boundary_dists) if boundary_dists else np.inf
                else:
                    # Euclidean distance
                    pos = self.env.bin_centers[bin_idx]
                    boundary_pos = self.env.bin_centers[list(self._boundary_bins)]
                    distances[i] = np.min(np.linalg.norm(boundary_pos - pos, axis=1))

        # Gaussian tuning around preferred distance
        rates = np.exp(-0.5 * ((distances - self.preferred_distance) / self.tolerance) ** 2)
        rates = rates * (self.max_rate - self.baseline_rate) + self.baseline_rate

        return rates
```

---

## 3. Spike Generation - Keep It Simple

```python
def generate_spikes(
    rates: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    refractory_period: float = 0.002,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Poisson spike train with refractory period.

    Parameters
    ----------
    rates : ndarray, shape (n_times,)
        Instantaneous firing rate in Hz.
    times : ndarray, shape (n_times,)
        Time points in seconds.
    refractory_period : float, optional
        Absolute refractory period in seconds (default: 2 ms).
        No spikes can occur within this window.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    spike_times : ndarray
        Times when spikes occurred.

    Examples
    --------
    >>> pc = PlaceCell(env, center=[50, 75])
    >>> positions, times = simulate_trajectory(env, duration=60.0)
    >>> rates = pc.firing_rate(positions)
    >>> spike_times = generate_spikes(rates, times)
    >>> print(f"Generated {len(spike_times)} spikes")
    """
    if seed is not None:
        np.random.seed(seed)

    dt = np.median(np.diff(times))

    # Poisson process
    spike_prob = rates * dt
    is_spike = np.random.rand(len(times)) < spike_prob
    spike_times = times[is_spike]

    # Enforce refractory period
    if len(spike_times) > 0 and refractory_period > 0:
        filtered_spikes = [spike_times[0]]
        for spike_time in spike_times[1:]:
            if spike_time - filtered_spikes[-1] >= refractory_period:
                filtered_spikes.append(spike_time)
        spike_times = np.array(filtered_spikes)

    return spike_times
```

---

## Complete Usage Examples

### Example 1: Validate Place Field Detection

```python
from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory, PlaceCell, generate_spikes
from neurospatial.fields import compute_place_field, detect_place_fields, field_centroid

# Create environment
env = Environment.from_samples(arena_data, bin_size=2.0)

# Create place cell population (every 5th bin)
place_cells = [
    PlaceCell(env, center=c, width=10.0, max_rate=25.0)
    for c in env.bin_centers[::5]
]

# Generate coverage trajectory
positions, times = simulate_trajectory(env, duration=180.0, speed=0.08)

# Generate spikes for each cell
spike_trains = [
    generate_spikes(pc.firing_rate(positions), times)
    for pc in place_cells
]

# Validate detection
errors = []
for pc, spike_times in zip(place_cells, spike_trains):
    # Detect field
    rate_map = compute_place_field(env, spike_times, times, positions)
    fields = detect_place_fields(rate_map, env, threshold=0.2)

    if len(fields) > 0:
        detected_center = field_centroid(rate_map, fields[0], env)
        error = np.linalg.norm(detected_center - pc.center)
        errors.append(error)
        print(f"Detection error: {error:.2f} cm")

print(f"\nMean error: {np.mean(errors):.2f} cm")
print(f"Expected: < {2 * env.bin_size:.2f} cm")
```

### Example 2: Linear Track with Direction Selectivity

```python
from neurospatial.simulation import simulate_goal_directed

# 1D track
env = Environment.from_graph(track_graph, edge_order, bin_size=1.0)

# Goals at track ends
goals = [env.bin_centers[0], env.bin_centers[-1]]

# Generate back-and-forth runs
positions, times, trials = simulate_goal_directed(
    env, goals, n_trials=30, trial_order='sequential', pause_at_goal=0.5
)

# Direction-selective place cells
def get_direction(positions):
    velocity = np.gradient(positions[:, 0])
    return velocity > 0

outbound = get_direction(positions)

# Cell fires only outbound
pc_out = PlaceCell(env, center=[85.0], width=5.0, max_rate=30.0)
rates_out = pc_out.firing_rate(positions) * outbound

# Cell fires only inbound
pc_in = PlaceCell(env, center=[85.0], width=5.0, max_rate=25.0)
rates_in = pc_in.firing_rate(positions) * ~outbound

# Generate spikes
spikes_out = generate_spikes(rates_out, times)
spikes_in = generate_spikes(rates_in, times)

# Analyze by direction
```

### Example 3: T-Maze Alternation

```python
# T-maze environment
env = Environment.from_polygon(tmaze_polygon, bin_size=2.0)

# Reward locations (left and right arms)
goals = [
    [20, 80],   # Left reward
    [80, 80],   # Right reward
]

# Alternating trials
positions, times, trials = simulate_goal_directed(
    env, goals, n_trials=40, trial_order='alternating', pause_at_goal=1.0
)

# Place cells along different parts of maze
stem_cell = PlaceCell(env, center=[50, 40], width=8.0)  # Stem
left_cell = PlaceCell(env, center=[20, 70], width=8.0)  # Left arm
right_cell = PlaceCell(env, center=[80, 70], width=8.0) # Right arm

# Generate and analyze
for cell, name in [(stem_cell, 'stem'), (left_cell, 'left'), (right_cell, 'right')]:
    rates = cell.firing_rate(positions)
    spikes = generate_spikes(rates, times)

    # Split by trial type
    left_trials = trials % 2 == 0
    right_trials = trials % 2 == 1

    # Compute separate rate maps
    # ...
```

### Example 4: Validate Border Score

```python
from neurospatial.simulation import BoundaryCell
from neurospatial.metrics import border_score

# Open field arena
env = Environment.from_samples(arena_data, bin_size=2.0)

# Create boundary cell
bc = BoundaryCell(env, preferred_distance=5.0, tolerance=3.0, max_rate=20.0)

# Generate trajectory with good coverage
positions, times = simulate_trajectory(env, duration=240.0)

# Generate spikes
rates = bc.firing_rate(positions)
spike_times = generate_spikes(rates, times)

# Compute rate map
rate_map = compute_place_field(env, spike_times, times, positions)

# Validate border score
score = border_score(rate_map, env, distance_metric='geodesic')
print(f"Border score: {score:.3f}")
assert score > 0.5, "Boundary cell should have high border score"

# Compare to place cell (should have low border score)
pc = PlaceCell(env, center=env.bin_centers[env.n_bins // 2])  # Center
pc_rates = pc.firing_rate(positions)
pc_spikes = generate_spikes(pc_rates, times)
pc_rate_map = compute_place_field(env, pc_spikes, times, positions)
pc_score = border_score(pc_rate_map, env)
print(f"Place cell border score: {pc_score:.3f}")
assert pc_score < 0.3, "Central place cell should have low border score"
```

---

## Implementation Plan

### Phase 1: Ship Minimum Viable (3 days)

**File**: `src/neurospatial/simulation.py` (~300 lines)

**Contents**:
1. `simulate_trajectory()` - Coverage-based graph walk
2. `PlaceCell` class - Gaussian fields
3. `generate_spikes()` - Poisson with refractory period

**Tests**: `tests/test_simulation.py`
- Coverage percentage > 90%
- Place field detection error < 2 * bin_size
- Refractory period enforcement

**Ship it. Get feedback.**

---

### Phase 2: Goal-Directed + Boundary Cells (2 days)

**Add**:
1. `simulate_goal_directed()` - Shortest paths between goals
2. `BoundaryCell` class - Distance-tuned

**Tests**:
- Trial segmentation works
- Paths use shortest_path() correctly
- Border score > 0.5 for boundary cells

**Ship it. Iterate.**

---

### Phase 3: Based on User Requests (TBD)

**Possible additions** (only if users ask):
- Grid cells (if gridness_score is priority)
- Replay sequences (if testing decoders)
- Head direction cells (if HD support added)
- Speed cells
- ...

**Don't build these until needed.**

---

## Key Design Decisions

### 1. Use the Graph, Luke

neurospatial already has:
- `connectivity` - neighbor relationships
- `shortest_path()` - optimal routes
- `distance_between()` - edge weights
- `bin_centers` - spatial coordinates

**Leverage these!** Don't reimplement movement dynamics.

### 2. Coverage > Realism

For place field estimation, **sampling the environment** is more important than perfect movement dynamics.

Simple biased random walk ensures good coverage without complex OU process math.

### 3. Goal-Directed Is the Real Use Case

Most experiments have reward locations. Animals don't randomly explore - they run to goals.

Use graph shortest paths = simple + realistic.

### 4. Direction Selectivity = User's Problem

Don't add "condition" parameters. Users can write:
```python
rates = pc.firing_rate(positions) * (velocity > 0)
```

This is clear, flexible, and doesn't complicate the API.

### 5. One File Until It Hurts

Start with `simulation.py` as a single module. Split into submodules only when it exceeds ~500 lines.

---

## What We're NOT Building

1. ~~RatInABox integration~~ - Users can use RatInABox directly if needed
2. ~~Ornstein-Uhlenbeck process~~ - Biased graph walk is simpler and ensures coverage
3. ~~Complex movement dynamics~~ - Not needed for testing place fields
4. ~~Spiking neuron models~~ - Poisson is sufficient
5. ~~RL value functions~~ - Out of scope
6. ~~Protocols and factories~~ - YAGNI

---

## Summary

**Simple. Practical. Ships in one week.**

Three functions:
1. `simulate_trajectory()` - Graph-based coverage
2. `simulate_goal_directed()` - Shortest paths between goals
3. `generate_spikes()` - Poisson with refractory period

Two classes:
1. `PlaceCell` - Gaussian fields
2. `BoundaryCell` - Distance-tuned (optional, Phase 2)

**Total**: ~300 lines, one file, solves real problems.

**Raymond would approve.**
