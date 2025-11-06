# Primitive Operations Implementation Proposal

This document proposes implementation of fundamental graph-based spatial primitives for neurospatial.

## Design Principles

1. **Composability**: Primitives should compose into higher-level operations
2. **Type Safety**: Full type hints, mypy-compliant
3. **Performance**: Vectorized where possible, sparse-aware
4. **Graph-Aware**: All operations respect connectivity structure
5. **Error Handling**: Clear, diagnostic error messages
6. **Consistency**: Follow neurospatial naming conventions

## Module Structure

```
src/neurospatial/
├── primitives.py          # Core primitives (NEW)
├── sequence_ops.py        # Sequence operations (NEW)
├── graph_ops.py           # Graph matrix operations (NEW)
└── distance.py            # Existing (extend)
```

---

## 1. Core Spatial Primitives

### `neighbor_reduce` - THE FUNDAMENTAL OPERATION

**Location**: `src/neurospatial/primitives.py`

```python
from typing import Literal, Callable
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from neurospatial.environment.core import Environment

# Type alias for reduction operations
ReduceOp = Literal["sum", "mean", "max", "min", "prod", "std", "var"]

def neighbor_reduce(
    field: NDArray[np.float64],
    env: Environment,
    *,
    op: ReduceOp | Callable[[NDArray[np.float64]], float] = "mean",
    weights: NDArray[np.float64] | None = None,
    include_self: bool = False,
    normalize_weights: bool = True,
) -> NDArray[np.float64]:
    """
    Aggregate field values over graph neighborhoods.

    This is the fundamental primitive for all local spatial operations.
    Applies reduction operation over each bin's neighbors in the
    connectivity graph.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values at each bin.
    env : Environment
        Environment defining spatial connectivity graph.
    op : {"sum", "mean", "max", "min", "prod", "std", "var"} or callable
        Reduction operation to apply over neighbors:

        - "sum": Sum neighbor values
        - "mean": Average neighbor values (default)
        - "max": Maximum neighbor value
        - "min": Minimum neighbor value
        - "prod": Product of neighbor values
        - "std": Standard deviation of neighbor values
        - "var": Variance of neighbor values
        - callable: Custom function taking array → scalar

    weights : NDArray[np.float64], shape (n_bins, n_bins), optional
        Weight matrix for neighbors. If provided, computes weighted
        aggregation. Typically edge weights from transition matrix.
        Must be sparse-compatible. If None, uniform weights.
    include_self : bool, default=False
        If True, include bin's own value in aggregation.
        If False, aggregate over neighbors only.
    normalize_weights : bool, default=True
        If True and weights provided, normalize weights per bin
        to sum to 1 (probability distribution). If False, use raw weights.

    Returns
    -------
    reduced : NDArray[np.float64], shape (n_bins,)
        Field with reduction applied over each bin's neighborhood.

    Raises
    ------
    ValueError
        If field shape doesn't match env.n_bins.
        If weights shape is not (n_bins, n_bins).
        If op is unknown string.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, neighbor_reduce
    >>> data = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> field = np.random.rand(env.n_bins)

    >>> # Local averaging (smoothing)
    >>> smoothed = neighbor_reduce(field, env, op='mean')

    >>> # Laplacian (difference from neighbors)
    >>> laplacian = field - neighbor_reduce(field, env, op='mean')

    >>> # Bellman backup (weighted sum)
    >>> T = env.transitions(method='random_walk')  # Transition matrix
    >>> rewards = np.zeros(env.n_bins)
    >>> rewards[goal_bin] = 1.0
    >>> values = neighbor_reduce(
    ...     field=rewards,
    ...     env=env,
    ...     op='sum',
    ...     weights=T.toarray(),  # Use transition probabilities
    ...     normalize_weights=False
    ... )

    >>> # Local maximum (peak detection helper)
    >>> local_max = neighbor_reduce(field, env, op='max')
    >>> is_peak = field > local_max  # Bins greater than all neighbors

    Notes
    -----
    **Implementation**:

    For unweighted operations, iterates over graph neighbors:

        result[i] = op([field[j] for j in neighbors(i)])

    For weighted operations with sparse matrix W:

        result[i] = Σⱼ W[i,j] * field[j]

    **Performance**:

    - Unweighted: O(n_bins * avg_degree)
    - Weighted sparse: O(nnz) where nnz is number of edges
    - Uses scipy.sparse for weighted operations when available

    **Common patterns**:

    - Smoothing: `neighbor_reduce(field, env, op='mean')`
    - Laplacian: `field - neighbor_reduce(field, env, op='mean')`
    - Local statistics: `neighbor_reduce(field, env, op='std')`
    - RL value backup: `neighbor_reduce(values, env, op='sum', weights=T)`

    See Also
    --------
    local_statistic : Multi-hop neighborhood statistics
    propagate : Value propagation with decay
    """
    # Validate inputs
    if field.shape[0] != env.n_bins:
        raise ValueError(
            f"field shape {field.shape} doesn't match env.n_bins={env.n_bins}"
        )

    G = env.connectivity
    result = np.zeros_like(field)

    # Define operation function
    if isinstance(op, str):
        op_map = {
            'sum': np.sum,
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'prod': np.prod,
            'std': np.std,
            'var': np.var,
        }
        if op not in op_map:
            raise ValueError(
                f"Unknown op '{op}'. Valid: {list(op_map.keys())}"
            )
        op_func = op_map[op]
    else:
        op_func = op

    # Handle weighted case
    if weights is not None:
        if weights.shape != (env.n_bins, env.n_bins):
            raise ValueError(
                f"weights shape {weights.shape} must be ({env.n_bins}, {env.n_bins})"
            )

        # Use sparse matrix multiply if possible
        import scipy.sparse as sp
        if sp.issparse(weights):
            W = weights
        else:
            W = sp.csr_matrix(weights)

        if normalize_weights:
            # Normalize rows to sum to 1
            row_sums = np.array(W.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            W = sp.diags(1.0 / row_sums) @ W

        # Weighted aggregation (typically only makes sense for sum/mean)
        result = W @ field

        if include_self:
            result += field

        return result

    # Unweighted case - iterate over neighbors
    for bin_idx in range(env.n_bins):
        neighbor_indices = list(G.neighbors(bin_idx))

        if len(neighbor_indices) == 0:
            # Isolated node
            result[bin_idx] = field[bin_idx] if include_self else np.nan
            continue

        if include_self:
            neighbor_indices.append(bin_idx)

        neighbor_values = field[neighbor_indices]
        result[bin_idx] = op_func(neighbor_values)

    return result
```

---

### `accumulate_along_path` - TRAJECTORY ACCUMULATION

```python
def accumulate_along_path(
    field: NDArray[np.float64],
    path: NDArray[np.int_],
    *,
    op: Literal["sum", "prod", "max", "min"] = "sum",
    discount: float | None = None,
    reverse: bool = False,
) -> NDArray[np.float64]:
    """
    Accumulate field values along a path (bin sequence).

    Fundamental primitive for computing returns, path integrals,
    trajectory likelihood, and cumulative statistics along sequences.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values at each bin (e.g., rewards, log-probabilities).
    path : NDArray[np.int_], shape (n_steps,)
        Sequence of bin indices defining the path.
        Must be valid bin indices in range [0, n_bins).
    op : {"sum", "prod", "max", "min"}, default="sum"
        Accumulation operation:

        - "sum": Cumulative sum (e.g., total reward)
        - "prod": Cumulative product (e.g., likelihood)
        - "max": Running maximum
        - "min": Running minimum

    discount : float, optional
        Discount factor applied at each step (0 < discount ≤ 1).

        - For sum: accumulated[t] = field[t] + discount * accumulated[t-1]
        - For prod: accumulated[t] = field[t] * (discount * accumulated[t-1])

        If None, no discounting (equivalent to discount=1.0).
    reverse : bool, default=False
        If True, accumulate from end to start (backward).
        If False, accumulate from start to end (forward).

    Returns
    -------
    accumulated : NDArray[np.float64], shape (n_steps,)
        Accumulated values at each step along the path.

        **Interpretation**:

        - forward sum: accumulated[i] = Σₖ₌₀ⁱ discount^k * field[path[k]]
        - backward sum: accumulated[i] = Σₖ₌ᵢⁿ discount^(k-i) * field[path[k]]

    Raises
    ------
    ValueError
        If path contains invalid bin indices (< 0 or >= n_bins).
        If discount is not None and not in (0, 1].
        If op is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, accumulate_along_path
    >>> env = Environment.from_samples(data, bin_size=2.0)

    >>> # Discounted return (RL)
    >>> rewards = np.zeros(env.n_bins)
    >>> rewards[goal_bin] = 10.0
    >>> trajectory = env.bin_sequence(times, positions)
    >>> returns = accumulate_along_path(
    ...     rewards,
    ...     trajectory,
    ...     op='sum',
    ...     discount=0.95,
    ...     reverse=True  # Backward for returns-to-go
    ... )

    >>> # Path integral (total reward collected)
    >>> total_reward = accumulate_along_path(
    ...     rewards,
    ...     trajectory,
    ...     op='sum'
    ... )[-1]  # Final accumulated value

    >>> # Trajectory likelihood
    >>> log_probs = np.log(posterior + 1e-10)  # Log probabilities
    >>> log_likelihood = accumulate_along_path(
    ...     log_probs,
    ...     trajectory,
    ...     op='sum'
    ... )
    >>> likelihood = np.exp(log_likelihood)

    >>> # Running maximum value seen
    >>> values = compute_state_values(env)
    >>> max_value_seen = accumulate_along_path(
    ...     values,
    ...     trajectory,
    ...     op='max'
    ... )

    Notes
    -----
    **Implementation**:

    For sum with discount γ:

        acc[0] = field[path[0]]
        acc[i] = field[path[i]] + γ * acc[i-1]

    For product:

        acc[0] = field[path[0]]
        acc[i] = field[path[i]] * acc[i-1]

    **Common use cases**:

    - RL returns: `accumulate_along_path(rewards, traj, 'sum', γ, reverse=True)`
    - Path cost: `accumulate_along_path(costs, traj, 'sum')`
    - Likelihood: `accumulate_along_path(log_probs, traj, 'sum')`
    - TD targets: Combine with bootstrapping for TD(λ)

    See Also
    --------
    propagate : Spread values through graph
    neighbor_reduce : Aggregate over neighborhoods
    """
    n_bins = field.shape[0]
    n_steps = len(path)

    # Validate path
    if np.any(path < 0) or np.any(path >= n_bins):
        invalid = path[(path < 0) | (path >= n_bins)]
        raise ValueError(
            f"path contains invalid bin indices: {invalid}. "
            f"Valid range: [0, {n_bins})"
        )

    # Validate discount
    if discount is not None and (discount <= 0 or discount > 1):
        raise ValueError(
            f"discount must be in (0, 1], got {discount}"
        )

    # Get field values along path
    path_values = field[path]

    # Reverse if needed
    if reverse:
        path_values = path_values[::-1]

    # Accumulate
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

    elif op == 'max':
        accumulated[0] = path_values[0]
        for i in range(1, n_steps):
            accumulated[i] = max(path_values[i], accumulated[i-1])

    elif op == 'min':
        accumulated[0] = path_values[0]
        for i in range(1, n_steps):
            accumulated[i] = min(path_values[i], accumulated[i-1])

    else:
        raise ValueError(
            f"Unknown op '{op}'. Valid: 'sum', 'prod', 'max', 'min'"
        )

    # Reverse back if needed
    if reverse:
        accumulated = accumulated[::-1]

    return accumulated
```

---

### `propagate` - VALUE PROPAGATION

```python
def propagate(
    sources: int | NDArray[np.int_] | NDArray[np.bool_],
    initial_values: float | NDArray[np.float64],
    env: Environment,
    *,
    decay: float = 1.0,
    max_steps: int | None = None,
    method: Literal["bfs", "dijkstra"] = "bfs",
) -> NDArray[np.float64]:
    """
    Propagate values from sources through graph with exponential decay.

    Fundamental primitive for value iteration, diffusion, successor
    representation, and distance fields. Spreads values from source
    bins to neighbors, with multiplicative decay at each step.

    Parameters
    ----------
    sources : int, array of int, or bool array
        Source bins where propagation starts:

        - int: Single source bin index
        - array of int: Multiple source bin indices
        - bool array, shape (n_bins,): Mask of source bins

    initial_values : float or NDArray[np.float64]
        Initial values at source bins:

        - float: Same value for all sources
        - array: Per-source values (must match number of sources)

    env : Environment
        Environment defining connectivity graph.
    decay : float, default=1.0
        Multiplicative decay factor per step (0 < decay ≤ 1).

        **Common values**:

        - 1.0: No decay (uniform spread)
        - 0.95-0.99: RL discount factors
        - 0.5-0.8: Strong localization

    max_steps : int, optional
        Maximum propagation steps. If None, propagates until
        convergence (values < 1e-10) or all bins reached.

        **Guidance**: Set to horizon length for RL, or None for
        steady-state distributions.

    method : {"bfs", "dijkstra"}, default="bfs"
        Propagation algorithm:

        - "bfs": Breadth-first search (hop-based, uniform edge costs)
        - "dijkstra": Dijkstra's algorithm (weighted by edge 'distance')

    Returns
    -------
    propagated : NDArray[np.float64], shape (n_bins,)
        Field with propagated values. Bins reached from sources have
        decayed values, unreached bins have 0.

    Raises
    ------
    ValueError
        If sources contains invalid bin indices.
        If initial_values is array with wrong shape.
        If decay not in (0, 1].

    Examples
    --------
    >>> from neurospatial import Environment, propagate
    >>> env = Environment.from_samples(data, bin_size=2.0)

    >>> # Discounted distance field (RL value initialization)
    >>> goal_bin = 42
    >>> value_field = propagate(
    ...     sources=goal_bin,
    ...     initial_values=1.0,
    ...     env=env,
    ...     decay=0.95,
    ...     max_steps=100
    ... )

    >>> # Successor representation (single column)
    >>> sr_column = propagate(
    ...     sources=bin_i,
    ...     initial_values=1.0,
    ...     env=env,
    ...     decay=0.95,
    ...     max_steps=None  # Until convergence
    ... )

    >>> # Multi-source diffusion (multiple goals)
    >>> goal_bins = [10, 50, 90]
    >>> goal_values = [1.0, 0.5, 0.8]  # Different reward values
    >>> diffused = propagate(
    ...     sources=goal_bins,
    ...     initial_values=goal_values,
    ...     env=env,
    ...     decay=0.9
    ... )

    Notes
    -----
    **Algorithm** (BFS variant):

    1. Initialize: field[sources] = initial_values, others = 0
    2. For each step k = 0 to max_steps:
       - For each bin with value > threshold:
         - Propagate: neighbor_value = max(neighbor_value, value * decay)
       - Stop if no updates or max_steps reached

    **Convergence**:

    For decay < 1, converges geometrically. Max iterations ≈ -log(ε)/log(decay)
    where ε is tolerance (1e-10).

    **Use cases**:

    - Value iteration: Iteratively call with updated values
    - Successor representation: Propagate from each bin
    - Influence maps: Decay from important locations
    - Diffusion: Model value/information spread

    See Also
    --------
    distance_field : Multi-source geodesic distances
    neighbor_reduce : Local aggregation
    accumulate_along_path : Path-based accumulation
    """
    G = env.connectivity
    n_bins = env.n_bins

    # Validate decay
    if decay <= 0 or decay > 1:
        raise ValueError(f"decay must be in (0, 1], got {decay}")

    # Parse sources
    if isinstance(sources, (int, np.integer)):
        source_bins = [int(sources)]
    elif isinstance(sources, np.ndarray) and sources.dtype == bool:
        if sources.shape[0] != n_bins:
            raise ValueError(
                f"Boolean sources array must have shape ({n_bins},), "
                f"got {sources.shape}"
            )
        source_bins = np.where(sources)[0].tolist()
    else:
        source_bins = np.asarray(sources, dtype=int).tolist()

    # Validate source bins
    for src in source_bins:
        if src < 0 or src >= n_bins:
            raise ValueError(
                f"Invalid source bin {src}. Valid range: [0, {n_bins})"
            )

    # Parse initial values
    if isinstance(initial_values, (int, float, np.number)):
        init_vals = np.full(len(source_bins), float(initial_values))
    else:
        init_vals = np.asarray(initial_values, dtype=np.float64)
        if init_vals.shape[0] != len(source_bins):
            raise ValueError(
                f"initial_values has {init_vals.shape[0]} elements but "
                f"{len(source_bins)} sources provided"
            )

    # Initialize field
    field = np.zeros(n_bins, dtype=np.float64)
    for src, val in zip(source_bins, init_vals):
        field[src] = val

    # Propagation
    if method == 'bfs':
        # BFS-based propagation
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
                    # Update if this path gives higher value
                    if current_value > field[neighbor]:
                        field[neighbor] = current_value
                        next_active.add(neighbor)

            active = next_active
            step += 1

    elif method == 'dijkstra':
        # Dijkstra-like propagation using edge distances
        import heapq

        # Priority queue: (-value, bin_idx)
        # Use negative value so higher values have priority
        pq = [(-init_vals[i], src) for i, src in enumerate(source_bins)]
        heapq.heapify(pq)

        visited = set()
        step = 0

        while pq and (max_steps is None or step < max_steps):
            neg_value, bin_idx = heapq.heappop(pq)
            value = -neg_value

            if bin_idx in visited:
                continue

            visited.add(bin_idx)

            # Propagate to neighbors
            for neighbor in G.neighbors(bin_idx):
                if neighbor in visited:
                    continue

                # Get edge distance
                edge_dist = G.edges[bin_idx, neighbor].get('distance', 1.0)

                # Decay based on distance
                decayed_value = value * (decay ** edge_dist)

                if decayed_value > field[neighbor]:
                    field[neighbor] = decayed_value
                    heapq.heappush(pq, (-decayed_value, neighbor))

            step += 1

    else:
        raise ValueError(f"Unknown method '{method}'. Valid: 'bfs', 'dijkstra'")

    return field
```

---

## 2. Sequence Operations

### `pairwise_sequence_distances`

**Location**: `src/neurospatial/sequence_ops.py`

```python
def pairwise_sequence_distances(
    sequence: NDArray[np.int_],
    env: Environment,
    *,
    lag: int = 1,
    metric: Literal["geodesic", "euclidean", "hops"] = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distances between bins at specified lag in a sequence.

    Fundamental primitive for trajectory smoothness, jump detection,
    temporal compression, and path length computation.

    Parameters
    ----------
    sequence : NDArray[np.int_], shape (n_steps,)
        Sequence of bin indices (trajectory).
    env : Environment
        Environment for computing distances.
    lag : int, default=1
        Lag between bins to compare:

        - 1: Consecutive bins (default)
        - k: Distance between sequence[i] and sequence[i+k]

    metric : {"geodesic", "euclidean", "hops"}, default="geodesic"
        Distance metric:

        - "geodesic": Shortest path through connectivity graph
        - "euclidean": Straight-line distance in space
        - "hops": Graph hop count (unweighted shortest path)

    Returns
    -------
    distances : NDArray[np.float64], shape (n_steps - lag,)
        Distance from sequence[i] to sequence[i+lag] for each i.
        Length is n_steps - lag (can't compute distance for last lag steps).

    Examples
    --------
    >>> from neurospatial import Environment, pairwise_sequence_distances
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> trajectory = env.bin_sequence(times, positions)

    >>> # Detect jumps (large distances between consecutive bins)
    >>> distances = pairwise_sequence_distances(trajectory, env, lag=1)
    >>> jumps = distances > 20.0  # cm threshold
    >>> print(f"Found {jumps.sum()} jumps")

    >>> # Path length (total distance traveled)
    >>> distances = pairwise_sequence_distances(trajectory, env, metric='euclidean')
    >>> path_length = distances.sum()

    >>> # Temporal compression (replay analysis)
    >>> decoded_traj = decode_position(spikes, time_windows)
    >>> spatial_dist = pairwise_sequence_distances(decoded_traj, env).sum()
    >>> temporal_dist = time_windows[-1] - time_windows[0]
    >>> compression_ratio = spatial_dist / temporal_dist

    Notes
    -----
    For consecutive bins (lag=1), this is equivalent to:

        [distance(sequence[i], sequence[i+1]) for i in range(n-1)]

    **Performance**: O(n_steps * distance_computation_cost)

    For geodesic with large graphs, consider caching distance matrix.

    See Also
    --------
    accumulate_along_path : Cumulative path integration
    """
    n_steps = len(sequence)
    if lag >= n_steps:
        raise ValueError(f"lag {lag} >= sequence length {n_steps}")

    distances = np.zeros(n_steps - lag, dtype=np.float64)

    if metric == "geodesic":
        # Use NetworkX shortest path
        G = env.connectivity
        for i in range(n_steps - lag):
            bin_from = sequence[i]
            bin_to = sequence[i + lag]

            try:
                distances[i] = nx.shortest_path_length(
                    G, bin_from, bin_to, weight='distance'
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distances[i] = np.inf

    elif metric == "euclidean":
        # Direct Euclidean distance
        for i in range(n_steps - lag):
            bin_from = sequence[i]
            bin_to = sequence[i + lag]
            distances[i] = np.linalg.norm(
                env.bin_centers[bin_to] - env.bin_centers[bin_from]
            )

    elif metric == "hops":
        # Unweighted shortest path
        G = env.connectivity
        for i in range(n_steps - lag):
            bin_from = sequence[i]
            bin_to = sequence[i + lag]

            try:
                distances[i] = nx.shortest_path_length(
                    G, bin_from, bin_to, weight=None
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distances[i] = np.inf

    else:
        raise ValueError(f"Unknown metric '{metric}'")

    return distances
```

---

## 3. Higher-Level Primitives Built from Core

### Example: Bellman Backup

```python
def bellman_backup(
    values: NDArray[np.float64],
    rewards: NDArray[np.float64],
    env: Environment,
    *,
    transitions: NDArray[np.float64] | None = None,
    gamma: float = 0.95,
) -> NDArray[np.float64]:
    """
    Compute Bellman backup: r + γ * E[V(s')].

    Built from neighbor_reduce primitive.
    """
    if transitions is None:
        # Uniform transition to neighbors
        expected_values = neighbor_reduce(
            values, env, op='mean', include_self=False
        )
    else:
        # Weighted transition
        expected_values = neighbor_reduce(
            values, env, op='sum', weights=transitions, normalize_weights=False
        )

    return rewards + gamma * expected_values


def successor_representation(
    env: Environment,
    gamma: float = 0.95,
    method: Literal["propagate", "analytic"] = "propagate",
) -> NDArray[np.float64]:
    """
    Compute successor representation: M = (I - γT)^(-1).

    Can use propagate primitive or matrix inversion.
    """
    if method == "propagate":
        # Build SR column by column using propagate
        SR = np.zeros((env.n_bins, env.n_bins), dtype=np.float64)
        for i in range(env.n_bins):
            SR[:, i] = propagate(
                sources=i,
                initial_values=1.0,
                env=env,
                decay=gamma,
                max_steps=None  # Until convergence
            )
        return SR

    else:  # analytic
        # Get transition matrix
        T = env.transitions(method='random_walk')

        # Solve: SR = (I - γT)^(-1)
        import scipy.sparse as sp
        I = sp.eye(env.n_bins)
        SR = sp.linalg.inv(I - gamma * T)
        return SR.toarray()
```

---

## Implementation Plan

### Phase 1: Core Primitives (Week 1)
- [ ] Implement `neighbor_reduce` in `primitives.py`
- [ ] Implement `accumulate_along_path` in `primitives.py`
- [ ] Implement `propagate` in `primitives.py`
- [ ] Write comprehensive tests for each
- [ ] Add to `__init__.py` public API

### Phase 2: Sequence Operations (Week 2)
- [ ] Implement `pairwise_sequence_distances` in `sequence_ops.py`
- [ ] Implement `sequence_correlation` in `sequence_ops.py`
- [ ] Implement `sequence_windowing` in `sequence_ops.py`
- [ ] Tests and documentation

### Phase 3: Higher-Level Functions (Week 3)
- [ ] Implement `bellman_backup` using primitives
- [ ] Implement `successor_representation` using primitives
- [ ] Implement `local_statistic` (multi-hop neighbor_reduce)
- [ ] Examples and tutorials

### Phase 4: Integration (Week 4)
- [ ] Add examples to documentation
- [ ] Performance benchmarks
- [ ] Integration tests with existing features
- [ ] User guide updates

---

## Testing Strategy

Each primitive needs:
1. **Unit tests**: Correctness on simple cases
2. **Edge cases**: Empty inputs, single bin, disconnected graphs
3. **Integration tests**: Compose primitives into known operations
4. **Performance tests**: Scaling with graph size
5. **Doctest examples**: All examples in docstrings must pass

Example test structure:
```python
# tests/test_primitives.py

def test_neighbor_reduce_mean():
    """Test neighbor_reduce with mean operation."""
    # Create simple 3x3 grid
    env = create_test_grid(3, 3)
    field = np.arange(9, dtype=float)

    result = neighbor_reduce(field, env, op='mean')

    # Center bin (4) should average its 4 neighbors: (1,3,5,7)
    expected_center = (1 + 3 + 5 + 7) / 4
    assert np.isclose(result[4], expected_center)


def test_accumulate_along_path_discounted():
    """Test discounted return calculation."""
    env = create_linear_track(10)
    rewards = np.zeros(10)
    rewards[-1] = 1.0  # Goal at end

    path = np.arange(10)  # Straight path to goal
    returns = accumulate_along_path(
        rewards, path, op='sum', discount=0.9, reverse=True
    )

    # Should be geometric series: [0.9^9, 0.9^8, ..., 0.9^0]
    expected = 0.9 ** np.arange(9, -1, -1)
    np.testing.assert_allclose(returns, expected)


def test_bellman_backup_composition():
    """Test that Bellman backup composes from primitives correctly."""
    env = create_test_grid(5, 5)
    values = np.random.rand(25)
    rewards = np.random.rand(25)
    gamma = 0.95

    # Method 1: Using bellman_backup
    result1 = bellman_backup(values, rewards, env, gamma=gamma)

    # Method 2: Manual composition
    expected_values = neighbor_reduce(values, env, op='mean')
    result2 = rewards + gamma * expected_values

    np.testing.assert_allclose(result1, result2)
```

---

## Performance Considerations

1. **Vectorization**: Use NumPy operations where possible
2. **Sparse matrices**: Use scipy.sparse for transition matrices
3. **Caching**: Cache distance matrices for repeated queries
4. **Early stopping**: Propagate stops when values < threshold
5. **JIT compilation**: Consider numba for hot paths

Example optimization:
```python
# Before: Python loop (slow)
for i in range(n_bins):
    neighbors = list(G.neighbors(i))
    result[i] = np.mean(field[neighbors])

# After: Vectorized with sparse matrix (fast)
result = adjacency_matrix @ field / degree_vector
```
