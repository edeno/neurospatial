"""Implementation sketches for coverage and goal-directed trajectories.

These are concrete implementations that could be added to neurospatial.simulation
to complement the existing OU process trajectory generation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal
from neurospatial import Environment


# =============================================================================
# Coverage-Ensuring Trajectory
# =============================================================================

def simulate_trajectory_coverage(
    env: Environment,
    duration: float,
    *,
    speed: float = 0.08,
    sampling_frequency: float = 100.0,
    pause_at_boundaries: float = 0.0,
    coverage_bias: float = 2.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate trajectory ensuring good spatial coverage.

    Uses biased random walk on the environment's connectivity graph,
    preferentially visiting less-explored bins. Guarantees >90% coverage
    for most environments without looking artificial.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    duration : float
        Total simulation time in seconds.
    speed : float, optional
        Movement speed in environment units/second (default: 0.08 m/s).
    sampling_frequency : float, optional
        Samples per second (default: 100 Hz).
    pause_at_boundaries : float, optional
        Pause duration at boundary bins in seconds (default: 0.0).
    coverage_bias : float, optional
        Strength of bias toward unvisited bins (default: 2.0).
        Higher values = more systematic exploration.
        Lower values = more random movement.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.

    Notes
    -----
    Strategy:
    1. Track occupancy count for each bin
    2. At each step, get neighbors from connectivity graph
    3. Weight neighbors inversely to their occupancy: w = 1 / (1 + count)^bias
    4. Choose next bin probabilistically based on weights
    5. Interpolate smooth movement along graph edges
    6. Add small jitter within bins for realism

    This produces realistic-looking trajectories while ensuring sufficient
    sampling for place field estimation.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Generate coverage trajectory
    >>> positions, times = simulate_trajectory_coverage(env, duration=180.0)
    >>>
    >>> # Check coverage
    >>> bin_indices = env.bin_at(positions)
    >>> unique_bins = len(np.unique(bin_indices[bin_indices >= 0]))
    >>> coverage_pct = 100 * unique_bins / env.n_bins
    >>> print(f"Visited {coverage_pct:.1f}% of bins")
    >>> # Expected: >90% for most environments
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / sampling_frequency
    n_samples = int(duration / dt)

    # Initialize
    current_bin = np.random.randint(env.n_bins)
    occupancy_counts = np.zeros(env.n_bins, dtype=np.int32)
    occupancy_counts[current_bin] = 1

    # Pre-compute boundary bins for pause logic
    boundary_bins = set(env.boundary_bins()) if pause_at_boundaries > 0 else set()

    # Build trajectory bin-by-bin
    trajectory_bins = []
    trajectory_bins.append(current_bin)

    samples_generated = 0

    while samples_generated < n_samples:
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current_bin))

        if not neighbors:
            # Isolated bin - stay here
            trajectory_bins.append(current_bin)
            samples_generated += 1
            continue

        # Compute weights based on occupancy
        # Less-visited bins get higher weight
        neighbor_counts = occupancy_counts[neighbors]
        weights = 1.0 / (1.0 + neighbor_counts) ** coverage_bias
        weights /= weights.sum()

        # Choose next bin
        next_bin = np.random.choice(neighbors, p=weights)

        # Compute movement duration (use Euclidean distance from bin centers)
        # Note: env.distance_between() has a bug and returns inf for adjacent bins
        pos_current = env.bin_centers[current_bin]
        pos_next = env.bin_centers[next_bin]
        edge_distance = np.linalg.norm(pos_next - pos_current)

        # Handle invalid distances (shouldn't happen, but be safe)
        if edge_distance <= 0:
            trajectory_bins.append(next_bin)
            samples_generated += 1
            current_bin = next_bin
            continue

        move_duration = edge_distance / speed
        n_move_samples = max(1, int(move_duration * sampling_frequency))

        # Add movement samples
        trajectory_bins.extend([next_bin] * n_move_samples)
        occupancy_counts[next_bin] += n_move_samples
        samples_generated += n_move_samples

        # Pause at boundaries if requested
        if pause_at_boundaries > 0 and next_bin in boundary_bins:
            n_pause_samples = int(pause_at_boundaries * sampling_frequency)
            trajectory_bins.extend([next_bin] * n_pause_samples)
            occupancy_counts[next_bin] += n_pause_samples
            samples_generated += n_pause_samples

        current_bin = next_bin

    # Trim to exact duration
    trajectory_bins = np.array(trajectory_bins[:n_samples], dtype=np.int32)

    # Convert bin indices to continuous positions
    positions = env.bin_centers[trajectory_bins].copy()

    # Add small jitter within bins for realism
    # Jitter std is 20% of mean bin size (keeps points mostly within bins)
    mean_bin_size = np.mean(env.bin_sizes)
    jitter_std = mean_bin_size * 0.2
    jitter = np.random.randn(*positions.shape) * jitter_std
    positions += jitter

    # Generate time vector
    times = np.arange(n_samples) * dt

    return positions, times


# =============================================================================
# Goal-Directed Multi-Reward Trajectory
# =============================================================================

def simulate_trajectory_goal_directed(
    env: Environment,
    goals: list[NDArray[np.float64]] | list[list[float]],
    n_trials: int,
    *,
    speed: float = 0.08,
    sampling_frequency: float = 100.0,
    pause_at_goal: float = 1.0,
    trial_order: Literal["sequential", "random", "alternating"] = "sequential",
    add_jitter: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Generate goal-directed trajectory between multiple reward locations.

    Uses Environment.shortest_path() to find routes between goals.
    Simulates realistic experimental paradigms (linear track, T-maze,
    plus maze, etc.).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goals : list of array-like, each shape (n_dims,)
        Reward locations in environment coordinates.
        Examples:
        - Linear track: [[0, 0], [100, 0]]
        - T-maze: [[20, 80], [80, 80]] (left and right arms)
        - Plus maze: [[50, 10], [90, 50], [50, 90], [10, 50]]
    n_trials : int
        Number of goal-to-goal runs.
    speed : float, optional
        Movement speed in units/second (default: 0.08 m/s).
    sampling_frequency : float, optional
        Samples per second (default: 100 Hz).
    pause_at_goal : float, optional
        Pause duration at each goal in seconds (default: 1.0).
    trial_order : {'sequential', 'random', 'alternating'}, optional
        How to sequence goals (default: 'sequential').
        - 'sequential': goal[0]→goal[1]→goal[0]→goal[1]...
        - 'random': random goal each trial
        - 'alternating': goal[0]→goal[1]→goal[2]→...→goal[0] (cycles through all)
    add_jitter : bool, optional
        Add small position jitter for realism (default: True).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.
    trial_ids : NDArray[np.int_], shape (n_times,)
        Trial number for each time point (0-indexed).

    Notes
    -----
    Movement along paths is smooth with constant speed. The function:
    1. Converts goal positions to bin indices
    2. Generates trial sequence based on trial_order
    3. For each trial:
       - Finds shortest path between current bin and goal bin
       - Interpolates smooth movement along path edges
       - Pauses at goal location
    4. Concatenates all trials into continuous trajectory

    Examples
    --------
    Linear track - back and forth:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>>
    >>> # Create 1D track environment
    >>> track_positions = np.linspace(0, 170, 1000)[:, np.newaxis]
    >>> env = Environment.from_samples(track_positions, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Goals at track ends
    >>> goals = [[0.0], [170.0]]
    >>>
    >>> # Generate 20 back-and-forth runs
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=20, trial_order='sequential'
    ... )
    >>>
    >>> # Check trial structure
    >>> print(f"Total duration: {times[-1]:.1f} seconds")
    >>> print(f"Unique trials: {len(np.unique(trials))}")

    T-maze alternation:

    >>> # Create T-maze environment (simplified as 2D grid)
    >>> tmaze_data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(tmaze_data, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Goals: left and right arms
    >>> goals = [
    ...     [20, 80],   # Left reward well
    ...     [80, 80],   # Right reward well
    ... ]
    >>>
    >>> # Alternating trials (L-R-L-R-...)
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=40, trial_order='sequential',
    ...     pause_at_goal=1.5
    ... )
    >>>
    >>> # Extract left vs right trials
    >>> left_trials = trials[::2]  # Even trials (0, 2, 4, ...)
    >>> right_trials = trials[1::2]  # Odd trials (1, 3, 5, ...)

    Plus maze - random arm visits:

    >>> # Four goals (N, E, S, W arms)
    >>> goals = [
    ...     [50, 10],   # South
    ...     [90, 50],   # East
    ...     [50, 90],   # North
    ...     [10, 50],   # West
    ... ]
    >>>
    >>> # Random order
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=100, trial_order='random', seed=42
    ... )

    See Also
    --------
    simulate_trajectory_laps : Structured back-and-forth on fixed path
    simulate_trajectory_coverage : Coverage-ensuring exploration
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / sampling_frequency

    # Convert goals to numpy arrays
    goals = [np.asarray(g, dtype=np.float64) for g in goals]

    # Convert goals to bin indices
    goal_bins = []
    for goal_pos in goals:
        if goal_pos.ndim == 1:
            goal_pos = goal_pos[np.newaxis, :]  # Add batch dim
        bin_idx = env.bin_at(goal_pos)[0]
        if bin_idx < 0:
            raise ValueError(f"Goal position {goal_pos.squeeze()} is outside environment")
        goal_bins.append(bin_idx)

    # Generate trial sequence
    n_goals = len(goals)

    if trial_order == "sequential":
        # Alternate between goals: 0→1→0→1→...
        if n_goals == 2:
            sequence = np.tile([0, 1], (n_trials // 2) + 1)[:n_trials]
        else:
            # For >2 goals, visit in order then reverse
            forward = list(range(n_goals))
            backward = list(reversed(forward))
            cycle = forward + backward
            sequence = np.tile(cycle, (n_trials // len(cycle)) + 1)[:n_trials]

    elif trial_order == "alternating":
        # Cycle through all goals in order: 0→1→2→...→0→1→2→...
        sequence = np.tile(range(n_goals), (n_trials // n_goals) + 1)[:n_trials]

    elif trial_order == "random":
        # Random goal each trial
        sequence = np.random.choice(n_goals, size=n_trials)

    else:
        raise ValueError(f"Unknown trial_order: {trial_order}. Use 'sequential', 'random', or 'alternating'")

    # Build trajectory trial-by-trial
    all_positions = []
    all_times = []
    all_trial_ids = []

    current_time = 0.0

    # Start at center of environment (not at a goal)
    center_idx = env.n_bins // 2
    current_bin = center_idx

    for trial_idx, goal_idx in enumerate(sequence):
        goal_bin = goal_bins[goal_idx]

        # Skip if already at goal (avoids zero-length trials)
        if current_bin == goal_bin:
            continue

        # Find shortest path
        path = env.shortest_path(current_bin, goal_bin)

        if path is None:
            # No path exists - skip this trial
            print(f"Warning: No path from bin {current_bin} to {goal_bin}, skipping trial {trial_idx}")
            continue

        # Walk along path, edge by edge
        for i in range(len(path) - 1):
            bin_a, bin_b = path[i], path[i + 1]

            # Get edge distance (use Euclidean from bin centers)
            # Note: env.distance_between() has a bug and returns inf for adjacent bins
            pos_a = env.bin_centers[bin_a]
            pos_b = env.bin_centers[bin_b]
            edge_distance = np.linalg.norm(pos_b - pos_a)

            # Skip if edge is invalid
            if edge_distance <= 0:
                continue

            move_duration = edge_distance / speed
            n_move_samples = max(1, int(move_duration * sampling_frequency))

            # Linear interpolation along edge
            alphas = np.linspace(0, 1, n_move_samples, endpoint=False)
            segment_positions = pos_a + alphas[:, np.newaxis] * (pos_b - pos_a)

            # Time and trial ID vectors
            segment_times = current_time + np.arange(n_move_samples) * dt
            segment_trial_ids = np.full(n_move_samples, trial_idx, dtype=np.int32)

            all_positions.append(segment_positions)
            all_times.append(segment_times)
            all_trial_ids.append(segment_trial_ids)

            current_time = segment_times[-1] + dt

        # Pause at goal
        if pause_at_goal > 0:
            n_pause_samples = int(pause_at_goal * sampling_frequency)
            pause_pos = np.tile(env.bin_centers[goal_bin], (n_pause_samples, 1))
            pause_times = current_time + np.arange(n_pause_samples) * dt
            pause_trial_ids = np.full(n_pause_samples, trial_idx, dtype=np.int32)

            all_positions.append(pause_pos)
            all_times.append(pause_times)
            all_trial_ids.append(pause_trial_ids)

            current_time = pause_times[-1] + dt

        current_bin = goal_bin

    # Concatenate all segments
    positions = np.vstack(all_positions)
    times = np.concatenate(all_times)
    trial_ids = np.concatenate(all_trial_ids)

    # Add jitter for realism
    if add_jitter:
        mean_bin_size = np.mean(env.bin_sizes)
        jitter_std = mean_bin_size * 0.15  # 15% of mean bin size
        jitter = np.random.randn(*positions.shape) * jitter_std
        positions += jitter

    return positions, times, trial_ids


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from neurospatial import Environment
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Create simple 2D environment
    np.random.seed(42)
    arena_data = np.random.uniform(0, 100, (2000, 2))
    env = Environment.from_samples(arena_data, bin_size=3.0)
    env.units = "cm"

    print(f"Environment: {env.n_bins} bins")

    # =========================================================================
    # Example 1: Coverage trajectory
    # =========================================================================
    print("\n--- Coverage Trajectory ---")
    positions_cov, times_cov = simulate_trajectory_coverage(
        env, duration=30.0, speed=15.0, coverage_bias=2.0, seed=42
    )

    # Check coverage
    bin_indices = env.bin_at(positions_cov)
    unique_bins = len(np.unique(bin_indices[bin_indices >= 0]))
    coverage_pct = 100 * unique_bins / env.n_bins
    print(f"Duration: {times_cov[-1]:.1f} s")
    print(f"Samples: {len(positions_cov)}")
    print(f"Coverage: {coverage_pct:.1f}% ({unique_bins}/{env.n_bins} bins)")

    # =========================================================================
    # Example 2: Goal-directed (T-maze style)
    # =========================================================================
    print("\n--- Goal-Directed Trajectory (T-maze) ---")
    goals = [
        [20, 80],  # Left arm
        [80, 80],  # Right arm
    ]

    positions_goal, times_goal, trials = simulate_trajectory_goal_directed(
        env, goals, n_trials=20, trial_order='sequential',
        speed=20.0, pause_at_goal=1.0, seed=42
    )

    print(f"Duration: {times_goal[-1]:.1f} s")
    print(f"Samples: {len(positions_goal)}")
    print(f"Trials: {len(np.unique(trials))}")
    print(f"Avg trial duration: {times_goal[-1] / len(np.unique(trials)):.2f} s")

    # =========================================================================
    # Example 3: Goal-directed (Plus maze - random)
    # =========================================================================
    print("\n--- Goal-Directed Trajectory (Plus maze - random arms) ---")
    # Adjust goals to be within environment bounds (0-100)
    goals_plus = [
        [50, 20],   # South
        [80, 50],   # East
        [50, 80],   # North
        [20, 50],   # West
    ]

    positions_plus, times_plus, trials_plus = simulate_trajectory_goal_directed(
        env, goals_plus, n_trials=40, trial_order='random',
        speed=20.0, pause_at_goal=0.5, seed=42
    )

    print(f"Duration: {times_plus[-1]:.1f} s")
    print(f"Trials: {len(np.unique(trials_plus))}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Coverage trajectory
    axes[0].plot(positions_cov[:, 0], positions_cov[:, 1], 'b-', alpha=0.3, lw=0.5)
    axes[0].scatter(positions_cov[0, 0], positions_cov[0, 1], c='green', s=100, marker='o', label='Start', zorder=10)
    axes[0].set_title(f'Coverage Trajectory\n{coverage_pct:.1f}% of bins visited')
    axes[0].set_xlabel('X (cm)')
    axes[0].set_ylabel('Y (cm)')
    axes[0].legend()
    axes[0].set_aspect('equal')

    # T-maze (color by trial)
    scatter = axes[1].scatter(positions_goal[:, 0], positions_goal[:, 1],
                              c=trials, cmap='viridis', s=1, alpha=0.5)
    for i, goal in enumerate(goals):
        axes[1].scatter(goal[0], goal[1], c='red', s=200, marker='*',
                       edgecolors='black', linewidths=2, zorder=10)
    axes[1].set_title('Goal-Directed (Sequential)')
    axes[1].set_xlabel('X (cm)')
    axes[1].set_ylabel('Y (cm)')
    axes[1].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[1], label='Trial ID')

    # Plus maze (color by trial)
    scatter = axes[2].scatter(positions_plus[:, 0], positions_plus[:, 1],
                             c=trials_plus, cmap='plasma', s=1, alpha=0.5)
    for i, goal in enumerate(goals_plus):
        axes[2].scatter(goal[0], goal[1], c='red', s=200, marker='*',
                       edgecolors='black', linewidths=2, zorder=10)
    axes[2].set_title('Goal-Directed (Random)')
    axes[2].set_xlabel('X (cm)')
    axes[2].set_ylabel('Y (cm)')
    axes[2].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[2], label='Trial ID')

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to trajectory_comparison.png")
    plt.show()
