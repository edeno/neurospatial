"""
Proof-of-concept implementation of core primitives.

This demonstrates that the proposed primitives work and compose correctly.
"""

import numpy as np
from numpy.typing import NDArray
import networkx as nx
from typing import Literal, Callable

# Mock Environment class for POC
class MockEnv:
    def __init__(self, connectivity: nx.Graph, n_bins: int, bin_centers: NDArray):
        self.connectivity = connectivity
        self.n_bins = n_bins
        self.bin_centers = bin_centers


def neighbor_reduce(
    field: NDArray[np.float64],
    env: MockEnv,
    *,
    op: Literal["sum", "mean", "max", "min"] | Callable = "mean",
    weights: NDArray[np.float64] | None = None,
    include_self: bool = False,
) -> NDArray[np.float64]:
    """Core primitive: aggregate over graph neighborhoods."""
    if field.shape[0] != env.n_bins:
        raise ValueError(f"field shape mismatch")

    G = env.connectivity
    result = np.zeros_like(field)

    # Define operation
    if isinstance(op, str):
        op_map = {
            'sum': np.sum,
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
        }
        op_func = op_map[op]
    else:
        op_func = op

    # Handle weighted case
    if weights is not None:
        # Simple dense implementation for POC
        for i in range(env.n_bins):
            result[i] = np.sum(weights[i, :] * field)
        if include_self:
            result += field
        return result

    # Unweighted: iterate over neighbors
    for bin_idx in range(env.n_bins):
        neighbor_indices = list(G.neighbors(bin_idx))

        if len(neighbor_indices) == 0:
            result[bin_idx] = field[bin_idx] if include_self else np.nan
            continue

        if include_self:
            neighbor_indices.append(bin_idx)

        neighbor_values = field[neighbor_indices]
        result[bin_idx] = op_func(neighbor_values)

    return result


def accumulate_along_path(
    field: NDArray[np.float64],
    path: NDArray[np.int_],
    *,
    op: Literal["sum", "prod"] = "sum",
    discount: float | None = None,
    reverse: bool = False,
) -> NDArray[np.float64]:
    """Core primitive: accumulate values along path."""
    n_steps = len(path)
    path_values = field[path]

    if reverse:
        path_values = path_values[::-1]

    accumulated = np.zeros(n_steps, dtype=np.float64)

    if op == 'sum':
        accumulated[0] = path_values[0]
        for i in range(1, n_steps):
            if discount is not None:
                accumulated[i] = path_values[i] + discount * accumulated[i-1]
            else:
                accumulated[i] = path_values[i] + accumulated[i-1]

    elif op == 'prod':
        accumulated[0] = path_values[0]
        for i in range(1, n_steps):
            if discount is not None:
                accumulated[i] = path_values[i] * (discount * accumulated[i-1])
            else:
                accumulated[i] = path_values[i] * accumulated[i-1]

    if reverse:
        accumulated = accumulated[::-1]

    return accumulated


def propagate(
    sources: int | list[int],
    initial_values: float | NDArray[np.float64],
    env: MockEnv,
    *,
    decay: float = 1.0,
    max_steps: int | None = None,
) -> NDArray[np.float64]:
    """Core primitive: propagate values through graph with decay."""
    G = env.connectivity
    n_bins = env.n_bins

    # Parse sources
    if isinstance(sources, int):
        source_bins = [sources]
    else:
        source_bins = list(sources)

    # Parse initial values
    if isinstance(initial_values, (int, float)):
        init_vals = [float(initial_values)] * len(source_bins)
    else:
        init_vals = list(initial_values)

    # Initialize
    field = np.zeros(n_bins, dtype=np.float64)
    for src, val in zip(source_bins, init_vals):
        field[src] = val

    # BFS propagation
    active = set(source_bins)
    step = 0
    threshold = 1e-10

    while active and (max_steps is None or step < max_steps):
        next_active = set()

        for bin_idx in active:
            current_value = field[bin_idx] * decay

            if current_value < threshold:
                continue

            for neighbor in G.neighbors(bin_idx):
                if current_value > field[neighbor]:
                    field[neighbor] = current_value
                    next_active.add(neighbor)

        active = next_active
        step += 1

    return field


# ============================================================================
# HIGHER-LEVEL OPERATIONS BUILT FROM PRIMITIVES
# ============================================================================

def bellman_backup(
    values: NDArray[np.float64],
    rewards: NDArray[np.float64],
    env: MockEnv,
    gamma: float = 0.95,
) -> NDArray[np.float64]:
    """RL Bellman backup: r + γ * E[V(s')]."""
    expected_values = neighbor_reduce(values, env, op='mean', include_self=False)
    return rewards + gamma * expected_values


def laplacian(
    field: NDArray[np.float64],
    env: MockEnv,
) -> NDArray[np.float64]:
    """Graph Laplacian: field - mean(neighbors)."""
    return field - neighbor_reduce(field, env, op='mean', include_self=False)


def smooth(
    field: NDArray[np.float64],
    env: MockEnv,
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """Smooth field by averaging with neighbors."""
    neighbor_mean = neighbor_reduce(field, env, op='mean', include_self=False)
    return (1 - alpha) * field + alpha * neighbor_mean


def discounted_return(
    rewards: NDArray[np.float64],
    trajectory: NDArray[np.int_],
    gamma: float = 0.95,
) -> NDArray[np.float64]:
    """Compute discounted return along trajectory."""
    return accumulate_along_path(
        rewards, trajectory, op='sum', discount=gamma, reverse=True
    )


def successor_representation_column(
    env: MockEnv,
    source_bin: int,
    gamma: float = 0.95,
) -> NDArray[np.float64]:
    """Compute one column of successor representation matrix."""
    return propagate(
        sources=source_bin,
        initial_values=1.0,
        env=env,
        decay=gamma,
        max_steps=None
    )


# ============================================================================
# DEMONSTRATION
# ============================================================================

def create_test_grid(rows: int, cols: int) -> MockEnv:
    """Create a simple grid environment for testing."""
    n_bins = rows * cols
    G = nx.grid_2d_graph(rows, cols)

    # Relabel nodes to integers
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    G = nx.relabel_nodes(G, mapping)

    # Add distance and vector attributes (simplified)
    for u, v in G.edges():
        G.edges[u, v]['distance'] = 1.0
        G.edges[u, v]['vector'] = (1.0, 0.0)

    # Create bin centers
    bin_centers = np.array([
        [i % cols, i // cols] for i in range(n_bins)
    ], dtype=np.float64)

    return MockEnv(G, n_bins, bin_centers)


def demo_primitives():
    """Demonstrate that primitives work and compose correctly."""
    print("=" * 70)
    print("PRIMITIVES PROOF-OF-CONCEPT DEMONSTRATION")
    print("=" * 70)

    # Create 5x5 grid
    env = create_test_grid(5, 5)
    print(f"\nCreated {env.n_bins}-bin grid environment")
    print(f"Graph has {env.connectivity.number_of_edges()} edges")

    # ========================================================================
    # 1. neighbor_reduce - THE FUNDAMENTAL PRIMITIVE
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. NEIGHBOR_REDUCE - Aggregate over graph neighborhoods")
    print("=" * 70)

    field = np.arange(25, dtype=float)
    print(f"\nOriginal field:\n{field.reshape(5, 5)}")

    # Mean smoothing
    smoothed = neighbor_reduce(field, env, op='mean')
    print(f"\nNeighbor mean:\n{smoothed.reshape(5, 5)}")

    # Laplacian (difference from neighbors)
    lap = field - smoothed
    print(f"\nLaplacian (field - neighbor_mean):\n{lap.reshape(5, 5)}")

    # Local max (peak detection helper)
    local_max = neighbor_reduce(field, env, op='max')
    is_peak = field > local_max
    print(f"\nIs local maximum? {is_peak.sum()} peaks")
    print(f"Peak bins: {np.where(is_peak)[0].tolist()}")

    # ========================================================================
    # 2. accumulate_along_path - TRAJECTORY OPERATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. ACCUMULATE_ALONG_PATH - Process sequences")
    print("=" * 70)

    # Create reward field (goal at end)
    rewards = np.zeros(25)
    rewards[24] = 10.0  # Goal at bottom-right corner

    # Trajectory: diagonal path from (0,0) to (4,4)
    trajectory = np.array([0, 6, 12, 18, 24])  # Diagonal
    print(f"\nTrajectory: {trajectory}")
    print(f"Reward at each step: {rewards[trajectory]}")

    # Discounted returns (reverse accumulation)
    gamma = 0.9
    returns = accumulate_along_path(
        rewards, trajectory, op='sum', discount=gamma, reverse=True
    )
    print(f"\nDiscounted returns (γ={gamma}):")
    for i, (bin_idx, ret) in enumerate(zip(trajectory, returns)):
        print(f"  Step {i}, bin {bin_idx}: return = {ret:.4f}")

    # Path integral (forward accumulation)
    cumulative = accumulate_along_path(
        rewards, trajectory, op='sum', reverse=False
    )
    print(f"\nCumulative reward: {cumulative}")

    # ========================================================================
    # 3. propagate - VALUE PROPAGATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. PROPAGATE - Spread values through graph")
    print("=" * 70)

    # Propagate from goal with decay
    goal_bin = 24
    value_field = propagate(
        sources=goal_bin,
        initial_values=1.0,
        env=env,
        decay=0.8,
        max_steps=10
    )
    print(f"\nValue field (propagated from bin {goal_bin}):")
    print(f"{value_field.reshape(5, 5)}")

    # Multi-source propagation
    sources = [0, 4, 20, 24]  # Four corners
    multi_field = propagate(
        sources=sources,
        initial_values=[1.0, 0.8, 0.6, 0.4],
        env=env,
        decay=0.9,
        max_steps=5
    )
    print(f"\nMulti-source propagation (from corners):")
    print(f"{multi_field.reshape(5, 5)}")

    # ========================================================================
    # 4. COMPOSING PRIMITIVES - Bellman Backup
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. COMPOSITION - Bellman backup from primitives")
    print("=" * 70)

    # Initialize random values
    values = np.random.rand(25)

    # Compute Bellman backup
    gamma = 0.95
    backed_up = bellman_backup(values, rewards, env, gamma=gamma)

    print(f"\nOriginal values (random):")
    print(f"{values.reshape(5, 5)}")
    print(f"\nRewards (goal at {goal_bin}):")
    print(f"{rewards.reshape(5, 5)}")
    print(f"\nBellman backup (r + γ*E[V(s')]):")
    print(f"{backed_up.reshape(5, 5)}")

    # Verify composition
    manual_backup = rewards + gamma * neighbor_reduce(values, env, op='mean')
    print(f"\nVerification: Manual == Bellman? {np.allclose(backed_up, manual_backup)}")

    # ========================================================================
    # 5. VALUE ITERATION using primitives
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. VALUE ITERATION - Iterative Bellman backup")
    print("=" * 70)

    # Run value iteration
    V = np.zeros(25)
    gamma = 0.95

    print(f"\nRunning value iteration (γ={gamma})...")
    for iteration in range(20):
        V_new = bellman_backup(V, rewards, env, gamma=gamma)
        delta = np.max(np.abs(V_new - V))
        V = V_new

        if iteration % 5 == 0:
            print(f"  Iteration {iteration:2d}: max change = {delta:.6f}")

        if delta < 1e-6:
            print(f"  Converged at iteration {iteration}")
            break

    print(f"\nFinal value function:")
    print(f"{V.reshape(5, 5)}")

    # ========================================================================
    # 6. SUCCESSOR REPRESENTATION using propagate
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. SUCCESSOR REPRESENTATION - Using propagate")
    print("=" * 70)

    # Compute SR for center bin
    center_bin = 12  # Center of 5x5 grid
    sr_column = successor_representation_column(env, center_bin, gamma=0.9)

    print(f"\nSuccessor representation (from bin {center_bin}):")
    print(f"{sr_column.reshape(5, 5)}")
    print(f"\nInterpretation: Expected discounted occupancy starting from center")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Primitives compose into complex operations")
    print("=" * 70)
    print("""
    ✓ neighbor_reduce → smoothing, Laplacian, peak detection, Bellman backup
    ✓ accumulate_along_path → returns, path integrals, likelihood
    ✓ propagate → value fields, diffusion, successor representation

    These 3 primitives enable:
    - RL algorithms (value iteration, policy evaluation)
    - Trajectory analysis (returns, smoothness, compression)
    - Spatial analysis (diffusion, influence maps)
    - Neural analysis (replay detection, field operations)

    All operations are graph-aware and work on arbitrary connectivity.
    """)


if __name__ == '__main__':
    demo_primitives()
