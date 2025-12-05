# Transition Matrix Construction for Replay Analysis

**Version**: 0.3.0
**Status**: Planning
**Target**: Hippocampal replay and theta sequence analysis

---

## Overview

This document specifies a new module `neurospatial.transitions` providing utilities for constructing transition matrices from behavioral trajectories and synthetic models. These are essential primitives for hippocampal replay analysis, theta sequence decoding, and spatial navigation models.

**Scope**: Transition matrix **construction** only. HMM inference algorithms (Forward-Backward, Viterbi) are explicitly out of scope.

**Target use cases**:

- Building empirical transition models from behavioral data
- Generating synthetic null models for statistical testing
- Analyzing forward vs reverse replay directionality
- Modeling momentum/inertia in movement sequences
- Regularizing sparse transition estimates
- Quantifying transition uncertainty

---

## Design Principles

1. **Sparse matrices**: All transition matrices use `scipy.sparse.csr_matrix` for memory efficiency
2. **Column-stochastic**: Each column sums to 1 (P(next_state | current_state))
3. **Graph-aware**: All functions respect environment connectivity (no transitions between non-adjacent bins)
4. **Smoothing**: Laplace smoothing prevents zero probabilities
5. **Neuroscience focus**: Examples and parameters tuned for hippocampal replay analysis
6. **NumPy docstrings**: Comprehensive documentation with realistic examples

---

## Module Structure

**New file**: `src/neurospatial/transitions.py`

**Dependencies**:

- `numpy` - Array operations
- `scipy.sparse` - Sparse matrix construction
- `networkx` - Graph connectivity
- `sklearn.decomposition.PCA` - For directional analysis

**Exports** (9 functions):

1. `empirical_transition_matrix` - Build from behavioral trajectory
2. `uniform_transition_matrix` - Uniform over graph neighbors (null model)
3. `distance_biased_transition_matrix` - Synthetic with spatial bias
4. `diffusion_transition_matrix` - Heat kernel / diffusion prior
5. `smooth_transition_matrix` - Regularize sparse estimates
6. `mix_transition_matrices` - Weighted combination
7. `transition_entropy` - Quantify uncertainty
8. `directional_transition_matrices` - Separate forward/reverse models
9. `momentum_transition_matrix` - Second-order with inertia

---

## Performance and Computational Limits

**Matrix sizes and memory usage**:

| n_bins | Dense size | Sparse (10% full) | Typical environment |
|--------|-----------|-------------------|---------------------|
| 50     | 20 KB     | 2 KB              | Small linear track  |
| 100    | 80 KB     | 8 KB              | Medium arena        |
| 500    | 2 MB      | 200 KB            | Large 2D arena      |
| 1000   | 8 MB      | 800 KB            | Very large / 3D     |
| 5000   | 200 MB    | 20 MB             | High-res / volumetric |

**Computational complexity**:

| Function | Time complexity | Memory | Large environment limit |
|----------|----------------|--------|-------------------------|
| `empirical_transition_matrix` | O(T) | O(nnz) | No limit (sparse) |
| `uniform_transition_matrix` | O(E) | O(nnz) | No limit (sparse) |
| `distance_biased_transition_matrix` | O(n²) or O(nE) | O(n²) or O(nnz) | ~1000 bins (geodesic) |
| `diffusion_transition_matrix` | O(n³) | O(n²) | **500 bins max** |
| `smooth_transition_matrix` | O(nnz) | O(nnz) | No limit (sparse) |
| `mix_transition_matrices` | O(k·nnz) | O(nnz) | No limit (sparse) |
| `transition_entropy` | O(nnz) | O(n) | No limit (sparse) |
| `directional_transition_matrices` | O(T) | O(nnz) | No limit (sparse) |
| `momentum_transition_matrix` | O(T) | O(T) | No limit (callable) |

**Notes**:

- T = trajectory length, n = n_bins, E = number of edges, nnz = non-zero entries, k = number of matrices
- Geodesic distances require all-pairs shortest paths: O(n²log(n) + nE)
- Diffusion matrix uses dense matrix exponential - **not recommended for n_bins > 500**
- Euclidean distances can be computed efficiently with only neighbor pairs: O(E)

---

## API Specification

### 1. Empirical Transition Matrix

```python
def empirical_transition_matrix(
    trajectory_bins: NDArray[np.int64],
    n_bins: int,
    *,
    smoothing: float = 0.01,
    normalize: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Build transition matrix from observed behavioral trajectory.

    Estimates P(next_bin | current_bin) by counting transitions in the
    observed sequence and normalizing.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_timepoints,)
        Sequence of bin indices from behavioral trajectory.
    n_bins : int
        Total number of bins in environment.
    smoothing : float, default=0.01
        Laplace smoothing constant added to all transition counts.
        Prevents zero probabilities for unobserved transitions.

        **Guidance based on data quantity**:
        - Abundant data (>1000 transitions): 0.001-0.01
        - Moderate data (100-1000): 0.01-0.05 (default)
        - Sparse data (<100): 0.05-0.2
    normalize : bool, default=True
        If True, normalize each column to sum to 1 (stochastic matrix).
        If False, return raw counts + smoothing.

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix where T[i, j] = P(next_bin=i | current_bin=j).
        Each column sums to 1 if normalize=True.

    Raises
    ------
    ValueError
        If trajectory_bins contains invalid indices (< 0 or >= n_bins).
        If smoothing is negative.
        If trajectory is too short (< 2 timepoints).

    Notes
    -----
    **Algorithm**:

    1. Count transitions: C[i, j] = count of (j → i) in trajectory
    2. Add smoothing: C[i, j] += smoothing
    3. Normalize columns: T[:, j] = C[:, j] / sum(C[:, j])

    **Smoothing rationale**: Laplace smoothing prevents zero probabilities
    which cause numerical issues in log-likelihood computations. Typical
    values: 0.01 to 0.1.

    **Complexity**: O(T) where T = len(trajectory_bins)

    **Use case**: Primary method for building transition models from real
    behavioral data for replay analysis.

    Examples
    --------
    >>> # Build from linear track trajectory
    >>> trajectory = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
    >>> T = empirical_transition_matrix(trajectory, n_bins=5)
    >>> T.shape
    (5, 5)
    >>> # Check transition from bin 2 to neighbors
    >>> T[:, 2].toarray().ravel()
    array([0.0, 0.33, 0.0, 0.67, 0.0])  # Mostly to bins 1 and 3

    >>> # Build from 2D arena trajectory
    >>> trajectory_2d = env.bin_at(positions)  # Map positions to bins
    >>> T = empirical_transition_matrix(trajectory_2d, env.n_bins)
    >>> # Use for replay decoding
    >>> replay_likelihood = T[decoded_bins[1:], decoded_bins[:-1]].diagonal()

    See Also
    --------
    directional_transition_matrices : Separate forward/reverse models
    smooth_transition_matrix : Regularize sparse estimates
    uniform_transition_matrix : Null model for comparison
    """
```

### 2. Uniform Transition Matrix

```python
def uniform_transition_matrix(
    G: nx.Graph,
    n_bins: int,
) -> scipy.sparse.csr_matrix:
    """
    Build transition matrix with uniform probability over graph neighbors.

    For each bin, assigns equal probability to all connected neighbors in the
    graph. This provides a structure-aware null model that respects environment
    connectivity.

    Parameters
    ----------
    G : nx.Graph
        Environment connectivity graph (typically env.connectivity).
    n_bins : int
        Total number of bins in environment.

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix where T[i, j] = 1/deg(j) if edge (j, i) exists,
        else 0. Each column sums to 1.

    Raises
    ------
    ValueError
        If graph contains isolated nodes (degree 0).
        If n_bins doesn't match graph node count.

    Notes
    -----
    **Algorithm**:

    For each node j:
    - Find neighbors N(j)
    - Set T[i, j] = 1 / |N(j)| for all i in N(j)
    - Set T[i, j] = 0 otherwise

    **Use case**: Null model for statistical testing. If real replay is more
    structured than uniform random walk, this provides baseline for comparison.

    **Relation to graph Laplacian**: This is the transition matrix for a
    simple random walk on the graph (related to normalized Laplacian).

    Examples
    --------
    >>> # Null model for linear track
    >>> T_null = uniform_transition_matrix(env.connectivity, env.n_bins)
    >>> # Interior bins: P=0.5 to each neighbor (left/right)
    >>> # Boundary bins: P=1.0 to single neighbor

    >>> # Compare empirical vs null
    >>> T_empirical = empirical_transition_matrix(trajectory, env.n_bins)
    >>> kl_div = divergence(T_empirical[:, i], T_null[:, i], kind="kl")
    >>> # High KL divergence → structured beyond random walk

    See Also
    --------
    empirical_transition_matrix : Data-driven transitions
    distance_biased_transition_matrix : Null model with spatial bias
    """
```

### 3. Distance-Biased Transition Matrix

```python
def distance_biased_transition_matrix(
    env: Environment,
    *,
    distance_scale: float = 10.0,
    distance_type: Literal["euclidean", "geodesic"] = "euclidean",
) -> scipy.sparse.csr_matrix:
    """
    Build synthetic transition matrix biased toward nearby bins.

    Transition probability decays exponentially with distance between bins,
    providing a spatially-structured null model. Transitions are restricted
    to graph neighbors only - this function applies distance weighting to
    existing edges, not all bin pairs.

    Parameters
    ----------
    env : Environment
        Spatial environment defining bin locations and connectivity.
    distance_scale : float, default=10.0
        Characteristic distance scale (σ) for exponential decay.
        Transitions decay as exp(-distance / σ).
        Units match environment units (e.g., cm).
    distance_type : {"euclidean", "geodesic"}, default="euclidean"
        Type of distance to use:

        - "euclidean": Straight-line distance between bin centers (fast).
          Ignores barriers/walls. Use for open fields.
        - "geodesic": Shortest path distance along graph edges (slow).
          Respects obstacles. Use for mazes.

        **Start with euclidean (default)**.

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix where T[i, j] ∝ exp(-d(i,j) / distance_scale)
        for neighbors i of j. Non-neighbors have T[i, j] = 0.
        Each column normalized to sum to 1.

    Raises
    ------
    ValueError
        If distance_scale <= 0.
        If distance_type not in {"euclidean", "geodesic"}.

    Notes
    -----
    **Algorithm**:

    1. For each bin j and its neighbors N(j):
       - Compute distance d(i, j) for i in N(j)
       - Compute weights: W[i, j] = exp(-d(i, j) / distance_scale)
    2. Normalize: T[i, j] = W[i, j] / sum_{k in N(j)} W[k, j]
    3. Non-neighbors have T[i, j] = 0

    **Important**: This only weights existing edges by distance. It does not
    create long-range transitions between non-neighbors.

    **Parameter guidance**:

    - Small distance_scale (< bin_size): Favors nearest neighbors
    - Medium distance_scale (≈ 5-10 × bin_size): Realistic local bias
    - Large distance_scale (> environment size): Approaches uniform

    **Use case**: More realistic null model than uniform_transition_matrix,
    accounts for spatial locality bias in hippocampal replay.

    Examples
    --------
    >>> # Linear track with local bias
    >>> T = distance_biased_transition_matrix(
    ...     env, distance_scale=20.0, distance_type="euclidean"
    ... )
    >>> # Transitions favor nearby bins exponentially

    >>> # 2D arena with geodesic distances
    >>> T = distance_biased_transition_matrix(
    ...     env, distance_scale=15.0, distance_type="geodesic"
    ... )
    >>> # Respects environment boundaries (geodesic wraps around obstacles)

    See Also
    --------
    uniform_transition_matrix : Unbiased null model
    diffusion_transition_matrix : Alternative distance-based model
    """
```

### 4. Diffusion Transition Matrix

```python
def diffusion_transition_matrix(
    env: Environment,
    *,
    diffusion_time: float = 1.0,
    mode: Literal["transition", "density"] = "transition",
) -> scipy.sparse.csr_matrix:
    """
    Build transition matrix from heat diffusion on environment graph.

    Uses matrix exponential of graph Laplacian to model diffusion process.
    This provides a principled distance-based prior derived from the heat
    equation.

    Parameters
    ----------
    env : Environment
        Spatial environment defining connectivity.
    diffusion_time : float, default=1.0
        Diffusion time parameter (t). Larger values = more spread.
        Approximately: variance ∝ t.

        **Typical ranges**:
        - 0.1-0.5: Highly local (1-2 bins)
        - 0.5-2.0: Moderate spread (good starting range)
        - 2.0-5.0: Broad transitions
        - >5.0: Nearly uniform

        **Rule of thumb**: Start with t = (bin_size)² / 10
    mode : {"transition", "density"}, default="transition"
        Normalization mode (passed to compute_diffusion_kernels).

        - "transition": Column-stochastic (sum_i T[i,j] = 1)
        - "density": Continuous density normalization

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Diffusion-based transition matrix. T[i, j] represents probability
        of being at bin i after diffusion time t, starting from bin j.

    Raises
    ------
    ValueError
        If diffusion_time <= 0.
        If mode not in {"transition", "density"}.
        If n_bins > 500 (matrix exponential too expensive).

    Notes
    -----
    **Algorithm**:

    Uses existing `compute_diffusion_kernels()` function internally:
    1. Compute graph Laplacian L
    2. Exponentiate: K = exp(-t * L)
    3. Normalize according to mode

    **Relation to Gaussian kernel**: For regular grids, this approximates
    a Gaussian kernel with σ² = 2 * diffusion_time.

    **Use case**: Regularization prior for Bayesian decoding, synthetic
    replay generation with smooth spatial structure.

    **Performance warning**: O(n³) for dense matrix exponential. The function
    will raise ValueError for n_bins > 500. For large environments, use
    distance_biased_transition_matrix instead.

    Examples
    --------
    >>> # Short diffusion time: local transitions
    >>> T = diffusion_transition_matrix(env, diffusion_time=0.5)
    >>> # Transitions concentrated near diagonal

    >>> # Long diffusion time: global transitions
    >>> T = diffusion_transition_matrix(env, diffusion_time=5.0)
    >>> # Transitions spread throughout environment

    >>> # Use as prior in Bayesian decoding
    >>> posterior = likelihood * T[:, prev_bin]
    >>> posterior /= posterior.sum()

    See Also
    --------
    neurospatial.kernels.compute_diffusion_kernels : Underlying implementation
    distance_biased_transition_matrix : Faster distance-based alternative
    """
```

### 5. Smooth Transition Matrix

```python
def smooth_transition_matrix(
    T_empirical: scipy.sparse.csr_matrix,
    T_prior: scipy.sparse.csr_matrix,
    *,
    prior_weight: float = 0.1,
) -> scipy.sparse.csr_matrix:
    """
    Regularize empirical transition matrix with smooth prior.

    Combines empirical estimates with a smooth prior (e.g., diffusion or
    distance-biased) to handle sparse data. This is a Bayesian MAP estimate
    with Dirichlet prior.

    Parameters
    ----------
    T_empirical : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Empirical transition matrix from data (e.g., from empirical_transition_matrix).
    T_prior : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Prior transition matrix (e.g., from diffusion_transition_matrix).
    prior_weight : float, default=0.1
        Weight for prior vs empirical. Range [0, 1].

        - 0: Pure empirical (no smoothing)
        - 1: Pure prior (ignore data)
        - 0.1-0.3: Typical range (light to moderate smoothing)

    Returns
    -------
    T_smooth : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Smoothed transition matrix. Each column normalized to sum to 1.

    Raises
    ------
    ValueError
        If T_empirical and T_prior have different shapes.
        If prior_weight not in [0, 1].

    Notes
    -----
    **Algorithm**:

    1. Compute weighted sum: T = (1 - α) * T_empirical + α * T_prior
    2. Normalize columns: T[:, j] = T[:, j] / sum(T[:, j])

    where α = prior_weight.

    **Parameter guidance**:

    - Limited data (< 100 trajectory samples): prior_weight = 0.2-0.5
    - Moderate data (100-1000 samples): prior_weight = 0.05-0.2
    - Large data (> 1000 samples): prior_weight = 0.01-0.05

    **Use case**: Regularize transition estimates when behavioral trajectory
    is short or when some bins are rarely visited.

    Examples
    --------
    >>> # Build empirical matrix from limited data
    >>> T_emp = empirical_transition_matrix(short_trajectory, n_bins=100)
    >>> # Too sparse, many zeros

    >>> # Build smooth prior
    >>> T_prior = diffusion_transition_matrix(env, diffusion_time=1.0)
    >>>
    >>> # Regularize
    >>> T_smooth = smooth_transition_matrix(T_emp, T_prior, prior_weight=0.2)
    >>> # Now has non-zero probabilities everywhere, but respects data

    >>> # Adaptive smoothing: stronger for rarely-visited bins
    >>> visit_counts = np.bincount(trajectory, minlength=n_bins)
    >>> for j in range(n_bins):
    ...     if visit_counts[j] < 10:
    ...         # Increase prior weight for sparse columns
    ...         T_smooth[:, j] = 0.5 * T_emp[:, j] + 0.5 * T_prior[:, j]

    See Also
    --------
    empirical_transition_matrix : Build from data
    diffusion_transition_matrix : Smooth prior from diffusion
    distance_biased_transition_matrix : Alternative prior
    """
```

### 6. Mix Transition Matrices

```python
def mix_transition_matrices(
    matrices: list[scipy.sparse.csr_matrix],
    weights: NDArray[np.float64] | None = None,
) -> scipy.sparse.csr_matrix:
    """
    Combine multiple transition matrices with weighted averaging.

    Useful for creating composite models (e.g., mixing forward/reverse,
    combining multiple behavioral sessions, or ensemble models).

    Parameters
    ----------
    matrices : list[scipy.sparse.csr_matrix]
        List of transition matrices to combine. All must have same shape.
    weights : NDArray[np.float64], shape (n_matrices,), optional
        Weights for each matrix. Must be non-negative and sum to 1.
        If None, uses uniform weights (1/n_matrices for each).

    Returns
    -------
    T_mixed : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Weighted mixture of input matrices. Each column normalized to sum to 1.

    Raises
    ------
    ValueError
        If matrices is empty.
        If matrices have inconsistent shapes.
        If weights has wrong length.
        If weights don't sum to 1 (tolerance 1e-6).
        If weights contain negative values.

    Notes
    -----
    **Algorithm**:

    1. Compute weighted sum: T = Σ_k w_k * T_k
    2. Normalize columns: T[:, j] = T[:, j] / sum(T[:, j])

    **Use case**: Combine models from different conditions (e.g., mix
    forward and reverse models for bidirectional replay analysis, or
    combine multiple recording sessions).

    Examples
    --------
    >>> # Mix forward and reverse models equally
    >>> T_fwd, T_rev = directional_transition_matrices(trajectory, env)
    >>> T_mixed = mix_transition_matrices([T_fwd, T_rev], weights=[0.5, 0.5])
    >>> # Balanced model for bidirectional replay

    >>> # Weighted ensemble from multiple sessions
    >>> T_list = [empirical_transition_matrix(traj, n_bins) for traj in sessions]
    >>> session_lengths = [len(traj) for traj in sessions]
    >>> weights = np.array(session_lengths) / sum(session_lengths)
    >>> T_ensemble = mix_transition_matrices(T_list, weights=weights)
    >>> # Weight by data quantity

    >>> # Mix empirical with null model
    >>> T_emp = empirical_transition_matrix(trajectory, n_bins)
    >>> T_null = uniform_transition_matrix(env.connectivity, n_bins)
    >>> T_mixed = mix_transition_matrices([T_emp, T_null], weights=[0.9, 0.1])

    See Also
    --------
    smooth_transition_matrix : Special case of mixing (empirical + prior)
    directional_transition_matrices : Build forward/reverse for mixing
    """
```

### 7. Transition Entropy

```python
def transition_entropy(
    T: scipy.sparse.csr_matrix,
    *,
    base: float = 2.0,
) -> NDArray[np.float64]:
    """
    Compute Shannon entropy of transition distribution for each bin.

    Quantifies uncertainty/randomness in transitions. High entropy = many
    possible next bins with similar probability. Low entropy = deterministic
    transitions.

    Parameters
    ----------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix where T[:, j] is distribution over next states
        given current state j.
    base : float, default=2.0
        Logarithm base for entropy computation.

        - base=2: Entropy in bits
        - base=e: Entropy in nats
        - base=10: Entropy in dits

    Returns
    -------
    entropy : NDArray[np.float64], shape (n_bins,)
        Shannon entropy for each bin's transition distribution.
        entropy[j] = -Σ_i T[i,j] * log_base(T[i,j])

    Raises
    ------
    ValueError
        If base <= 0.
        If T contains negative values (invalid probability).

    Notes
    -----
    **Algorithm**:

    For each column j of T:
    H[j] = -Σ_i P(i|j) * log_base(P(i|j))

    where P(i|j) = T[i, j], using the convention 0 * log(0) = 0.

    **Sparse matrix handling**: Efficiently computes entropy by only
    iterating over non-zero entries. Zero entries contribute 0 to sum.

    **Interpretation**:

    - entropy ≈ 0: Deterministic (one dominant next bin)
    - entropy ≈ log_base(k): Uniform over k neighbors
    - entropy ≈ log_base(n_bins): Maximally uncertain (uniform over all bins)

    **Use case**: Identify bins with deterministic vs exploratory transitions,
    quantify model complexity, compare empirical vs null model structure.

    Examples
    --------
    >>> # Uniform transitions: high entropy
    >>> T_uniform = uniform_transition_matrix(env.connectivity, env.n_bins)
    >>> H = transition_entropy(T_uniform)
    >>> # For linear track interior bins with 2 neighbors: H ≈ 1.0 bit

    >>> # Empirical transitions: variable entropy
    >>> T_emp = empirical_transition_matrix(trajectory, env.n_bins)
    >>> H = transition_entropy(T_emp)
    >>> # Low entropy → stereotyped behavior
    >>> # High entropy → exploratory behavior

    >>> # Identify deterministic vs random bins
    >>> H = transition_entropy(T_emp)
    >>> deterministic_bins = np.where(H < 0.5)[0]  # Low entropy
    >>> random_bins = np.where(H > 1.5)[0]  # High entropy

    >>> # Compare empirical vs null model
    >>> H_emp = transition_entropy(T_empirical)
    >>> H_null = transition_entropy(T_null)
    >>> structure_score = H_null.mean() - H_emp.mean()
    >>> # Positive score → empirical is more structured (lower entropy)

    See Also
    --------
    neurospatial.field_ops.divergence : Compare two distributions
    """
```

### 8. Directional Transition Matrices

```python
def directional_transition_matrices(
    trajectory_bins: NDArray[np.int64],
    env: Environment,
    *,
    smoothing: float = 0.01,
    reference_trajectory: NDArray[np.int64] | None = None,
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Build forward and reverse transition matrices from behavioral trajectory.

    Splits empirical transitions into two classes based on movement direction
    relative to a reference trajectory or the trajectory's own principal direction.
    Generalizes linear track forward/reverse replay to arbitrary 2D/3D environments.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_timepoints,)
        Sequence of bin indices from behavioral trajectory.
    env : Environment
        Spatial environment defining bin centers and connectivity.
    smoothing : float, default=0.01
        Laplace smoothing constant added to all transition counts.

        **Guidance based on data quantity**:
        - Abundant data (>1000 transitions): 0.001-0.01
        - Moderate data (100-1000): 0.01-0.05 (default)
        - Sparse data (<100): 0.05-0.2
    reference_trajectory : NDArray[np.int64], optional
        Reference trajectory defining "forward" direction. If None, uses
        principal direction of `trajectory_bins` itself (first PCA component
        of bin centers).

    Returns
    -------
    T_forward : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix for forward movements. Each column sums to 1.
    T_reverse : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix for reverse movements. Each column sums to 1.

    Raises
    ------
    ValueError
        If trajectory_bins contains invalid indices.
        If smoothing is negative.
        If trajectory is too short (< 2 timepoints).
        If reference_trajectory provided but has invalid indices.

    Warnings
    --------
    UserWarning
        If reference_trajectory is None and PCA first component explains
        <80% of variance (ambiguous global direction).

    Notes
    -----
    **Algorithm**:

    1. **Define reference direction**:
       - If reference_trajectory provided:
         * Compute local tangent at each bin using central differencing:
           tangent[i] = bin_centers[traj[i+1]] - bin_centers[traj[i-1]]
         * Handle endpoints with forward/backward differencing
         * Normalize to unit vectors
       - Else:
         * Run PCA on trajectory bin centers
         * Use PC1 as global direction (same for all bins)
         * Warn if PC1 explains <80% variance (ambiguous direction)

    2. **Classify each transition** (deterministic):
       - For transition (bin_i → bin_j):
         v = bin_centers[j] - bin_centers[i]
         d = v · ref_direction[i]
         If d > 1e-6: "forward"
         If d < -1e-6: "reverse"
         If |d| ≤ 1e-6: split 50/50 deterministically (alternate or hash-based)

    3. **Build separate matrices**:
       - Count forward and reverse transitions separately
       - Add smoothing and normalize independently

    **Linear track behavior**: Forward corresponds to increasing position,
    reverse to decreasing position.

    **2D arena behavior**: Forward/reverse defined relative to principal
    movement axis (e.g., outbound vs inbound on Y-maze, or radial in/out).

    **Use case**: Analyze forward vs reverse replay separately, test if
    replay direction predicts upcoming behavior, quantify directional biases.

    Examples
    --------
    >>> # Linear track: forward = right, reverse = left
    >>> trajectory = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])  # out and back
    >>> T_fwd, T_rev = directional_transition_matrices(trajectory, env)
    >>> # T_fwd captures 0→1→2→3→4 transitions
    >>> # T_rev captures 4→3→2→1→0 transitions

    >>> # 2D arena with reference trajectory
    >>> reference = outbound_trajectory  # bins from nest to goal
    >>> T_fwd, T_rev = directional_transition_matrices(
    ...     trajectory_bins, env, reference_trajectory=reference
    ... )
    >>> # T_fwd captures movements toward goal
    >>> # T_rev captures movements toward nest

    >>> # Analyze replay directionality
    >>> for replay_event in replay_events:
    ...     LL_fwd = compute_sequence_likelihood(replay_event, T_fwd)
    ...     LL_rev = compute_sequence_likelihood(replay_event, T_rev)
    ...     if LL_fwd > LL_rev:
    ...         print("Forward replay")
    ...     else:
    ...         print("Reverse replay")

    See Also
    --------
    empirical_transition_matrix : Builds single matrix without directional split
    momentum_transition_matrix : Second-order transitions with inertia
    """
```

### 9. Momentum Transition Matrix

```python
def momentum_transition_matrix(
    trajectory_bins: NDArray[np.int64],
    env: Environment,
    *,
    momentum_weight: float = 0.7,
    smoothing: float = 0.01,
) -> Callable[[int, int], NDArray[np.float64]]:
    """
    Build second-order transition function with directional momentum.

    Transition probability from current bin depends on previous bin, encoding
    the tendency to continue moving in the same direction (momentum/inertia).

    Returns a callable function rather than a matrix to avoid awkward indexing
    of the (n_bins * n_bins, n_bins) shape.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_timepoints,)
        Sequence of bin indices from behavioral trajectory.
    env : Environment
        Spatial environment defining bin centers and connectivity.
    momentum_weight : float, default=0.7
        Weight for continuing in same direction vs uniform over neighbors.
        Range [0, 1]:

        - 0: No momentum (reduces to first-order model)
        - 1: Pure momentum (only forward direction)
        - 0.5-0.9: Typical range (moderate to strong inertia)
    smoothing : float, default=0.01
        Laplace smoothing constant.

        **Guidance based on data quantity**:
        - Abundant data (>1000 transitions): 0.001-0.01
        - Moderate data (100-1000): 0.01-0.05 (default)
        - Sparse data (<100): 0.05-0.2

    Returns
    -------
    T_momentum : Callable[[int, int], NDArray[np.float64]]
        Function that takes (prev_bin, curr_bin) and returns probability
        distribution over next_bin as array of shape (n_bins,).

        Usage: probs = T_momentum(prev_bin, curr_bin)

    Raises
    ------
    ValueError
        If trajectory_bins contains invalid indices.
        If momentum_weight not in [0, 1].
        If smoothing is negative.
        If trajectory is too short (< 3 timepoints for second-order).

    Notes
    -----
    **Algorithm**:

    1. Extract all (prev, curr, next) triplets from trajectory
    2. For each (prev, curr) state:
       - Compute "forward" direction: vector from prev to curr
       - Find neighbors of curr from graph
       - Score each neighbor by alignment with forward direction:
         score[neighbor] = cos(angle(forward, curr→neighbor))
       - Combine momentum and uniform:
         P(next | prev, curr) ∝ momentum_weight * score + (1 - momentum_weight) * uniform
    3. Add smoothing and normalize

    **Design rationale**: Returns a callable instead of a matrix to avoid the
    awkward (n_bins², n_bins) indexing. The function signature makes the
    dependency on both prev_bin and curr_bin explicit and natural.

    **Memory**: Internal storage is typically O(T) where T = trajectory length,
    much more compact than a full (n_bins², n_bins) matrix.

    **Use case**: Model realistic trajectory dynamics for synthetic replay
    generation, test if neural sequences respect momentum constraints,
    improve Bayesian decoding with directional priors.

    Examples
    --------
    >>> # Build momentum model
    >>> T_momentum = momentum_transition_matrix(trajectory, env, momentum_weight=0.8)
    >>>
    >>> # Query: what's P(next | prev=1, curr=2)?
    >>> probs = T_momentum(prev_bin=1, curr_bin=2)
    >>> next_bin = np.random.choice(env.n_bins, p=probs)
    >>>
    >>> # Generate synthetic trajectory
    >>> prev, curr = 0, 1
    >>> trajectory = [prev, curr]
    >>> for _ in range(100):
    ...     probs = T_momentum(prev, curr)
    ...     next_bin = np.random.choice(env.n_bins, p=probs)
    ...     trajectory.append(next_bin)
    ...     prev, curr = curr, next_bin

    >>> # Linear track with strong momentum
    >>> trajectory = np.array([0, 1, 2, 3, 2, 1, 0])
    >>> T = momentum_transition_matrix(trajectory, env, momentum_weight=0.8)
    >>>
    >>> # Query P(next | prev=1, curr=2)
    >>> probs = T(prev_bin=1, curr_bin=2)
    >>> # Higher probability for next=3 (forward) than next=1 (reverse)
    >>> print(f"P(3|1,2) = {probs[3]:.2f}, P(1|1,2) = {probs[1]:.2f}")
    P(3|1,2) = 0.82, P(1|1,2) = 0.18

    >>> # 2D arena: momentum toward goal
    >>> trajectory_2d = env.bin_at(positions)
    >>> T = momentum_transition_matrix(trajectory_2d, env, momentum_weight=0.6)
    >>> # Transitions favor continuing in same direction

    See Also
    --------
    directional_transition_matrices : Separate forward/reverse matrices
    empirical_transition_matrix : First-order transitions (no momentum)
    """
```

---

## Implementation Notes

### Sparse Matrix Construction

All functions return `scipy.sparse.csr_matrix` (Compressed Sparse Row format):

**Why CSR?**

- Efficient row slicing for querying P(· | current_state)
- Memory efficient for typical environments (adjacency is sparse)
- Fast matrix-vector products for likelihood computations

**Typical construction pattern**:

```python
from scipy.sparse import csr_matrix, lil_matrix

# Build using LIL (efficient for incremental construction)
T_lil = lil_matrix((n_bins, n_bins), dtype=np.float64)

# Populate (example: uniform over neighbors)
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    if len(neighbors) > 0:
        prob = 1.0 / len(neighbors)
        for neighbor in neighbors:
            T_lil[neighbor, node] = prob

# Convert to CSR for efficient access
T = csr_matrix(T_lil)
```

### Normalization Strategy

All transition matrices are **column-stochastic**: each column sums to 1.

**Convention**: `T[i, j]` = P(next_state=i | current_state=j)

**Normalization code**:

```python
# Normalize each column to sum to 1
for j in range(n_bins):
    col_sum = T[:, j].sum()
    if col_sum > 0:
        T[:, j] /= col_sum
    else:
        # Handle zero columns (isolated nodes)
        # Option 1: Self-loop
        T[j, j] = 1.0
        # Option 2: Raise error
        raise ValueError(f"Bin {j} has no outgoing transitions")
```

### Graph Connectivity Enforcement

All transition matrices respect environment connectivity:

**Constraint**: `T[i, j]` can be non-zero only if:

1. `i == j` (self-loop), OR
2. Edge `(j, i)` exists in `env.connectivity`

**Enforcement (optimized with sparse adjacency masking)**:

```python
# Create adjacency mask (sparse)
adj_mask = nx.adjacency_matrix(G, nodelist=range(n_bins))
adj_mask.setdiag(1)  # Allow self-loops

# Enforce connectivity by element-wise multiplication (O(nnz))
T = T.multiply(adj_mask)  # Sparse element-wise product
```

**Old inefficient approach (O(n²))**:

```python
# DON'T DO THIS - too slow for large graphs
for j in range(n_bins):
    neighbors = set(env.connectivity.neighbors(j))
    neighbors.add(j)  # Allow self-loops
    for i in range(n_bins):
        if i not in neighbors:
            T[i, j] = 0.0
```

### Smoothing Implementation Details

Laplace smoothing prevents zero probabilities:

**Efficient algorithm (add smoothing ONLY to allowed edges)**:

```python
from scipy.sparse import lil_matrix

# Build sparse transition matrix
T_lil = lil_matrix((n_bins, n_bins), dtype=np.float64)

# Add smoothing ONLY to allowed edges (neighbors + self-loops)
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    neighbors.append(node)  # self-loop
    for neighbor in neighbors:
        T_lil[neighbor, node] += smoothing

# Then add empirical counts
for i in range(len(trajectory) - 1):
    curr, next = trajectory[i], trajectory[i+1]
    T_lil[next, curr] += 1

# Normalize columns
T = csr_matrix(T_lil)
for j in range(n_bins):
    col_sum = T[:, j].sum()
    if col_sum > 0:
        T[:, j] /= col_sum
```

**Wrong approach (wasteful)**:

```python
# DON'T DO THIS - adds smoothing to all n² entries, then zeros most of them
T += smoothing  # Bad: adds to ALL entries
# Then enforce connectivity (zeros non-neighbors) - wastes computation
```

**Effect**: Even unobserved transitions get probability `smoothing / Z` where `Z` is normalization constant.

---

## Testing Plan

### Test Fixtures

**Reuse existing** (from `conftest.py`):

- 5×5 regular grid
- Plus maze graph
- Linear track (1D)

**Add new fixtures**:

```python
@pytest.fixture
def linear_track_trajectory():
    """Linear track trajectory: out (0→9) and back (9→0)."""
    outbound = np.arange(10)
    inbound = np.arange(9, -1, -1)
    return np.concatenate([outbound, inbound])

@pytest.fixture
def circular_arena_env():
    """Circular 2D arena for directional analysis."""
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta)]) * 50
    return Environment.from_samples(positions, bin_size=5.0)
```

### Test Coverage Requirements

Each function requires tests for:

1. **Basic functionality**:
   - Correct shape and type (csr_matrix)
   - Column normalization (sum to 1)
   - Connectivity enforcement (zeros for non-neighbors)

2. **Edge cases**:
   - Empty trajectory
   - Single-bin trajectory
   - Isolated nodes
   - All transitions to single bin (deterministic)

3. **Parameter validation**:
   - Invalid smoothing (negative)
   - Invalid weights (don't sum to 1, negative)
   - Mismatched shapes
   - Invalid bin indices

4. **Numerical correctness**:
   - Known trajectories with expected probabilities
   - Comparison with hand-computed values
   - Entropy bounds (0 ≤ H ≤ log(n_neighbors))

5. **Integration**:
   - Chain multiple functions (e.g., build → smooth → mix)
   - Use with actual replay decoding workflow

### Example Test

```python
def test_empirical_transition_matrix_linear_track():
    """Test empirical matrix on simple linear track."""
    # Trajectory: 0 → 1 → 2 → 1 → 0
    trajectory = np.array([0, 1, 2, 1, 0])
    n_bins = 3

    T = empirical_transition_matrix(trajectory, n_bins, smoothing=0.0)

    # Shape and type
    assert T.shape == (n_bins, n_bins)
    assert isinstance(T, scipy.sparse.csr_matrix)

    # Column normalization
    for j in range(n_bins):
        assert np.isclose(T[:, j].sum(), 1.0)

    # Expected probabilities (with zero smoothing)
    # From bin 1: went to 2 once, to 0 once → P(2|1) = P(0|1) = 0.5
    assert np.isclose(T[2, 1], 0.5)
    assert np.isclose(T[0, 1], 0.5)

    # From bin 0: went to 1 once (ignore last state) → P(1|0) = 1.0
    assert np.isclose(T[1, 0], 1.0)
```

---

## Public API Updates

### New Module: `neurospatial/transitions.py`

Exports all 9 functions to `__all__`:

```python
__all__ = [
    "empirical_transition_matrix",
    "uniform_transition_matrix",
    "distance_biased_transition_matrix",
    "diffusion_transition_matrix",
    "smooth_transition_matrix",
    "mix_transition_matrices",
    "transition_entropy",
    "directional_transition_matrices",
    "momentum_transition_matrix",
]
```

### Top-Level Imports: `neurospatial/__init__.py`

Add to main package imports:

```python
# Transition matrix construction
from neurospatial.transitions import (
    empirical_transition_matrix,
    uniform_transition_matrix,
    distance_biased_transition_matrix,
    diffusion_transition_matrix,
    smooth_transition_matrix,
    mix_transition_matrices,
    transition_entropy,
    directional_transition_matrices,
    momentum_transition_matrix,
)
```

---

## Migration Guide

### For New Users

**Basic workflow**:

```python
from neurospatial import Environment, empirical_transition_matrix
import numpy as np

# Create environment
env = Environment.from_samples(position_data, bin_size=5.0)

# Map trajectory to bins
trajectory_bins = env.bin_at(trajectory_positions)

# Build transition matrix
T = empirical_transition_matrix(trajectory_bins, env.n_bins)

# Use for replay analysis
replay_bins = decode_replay_event(spike_data)  # Your decoder
likelihood = T[replay_bins[1:], replay_bins[:-1]].diagonal()
```

**Directional analysis**:

```python
from neurospatial import directional_transition_matrices

# Build forward/reverse models
T_fwd, T_rev = directional_transition_matrices(trajectory_bins, env)

# Analyze replay directionality using likelihood ratio
for replay in replay_events:
    # Compute log-likelihood under each model
    LL_fwd = sum(np.log(T_fwd[replay[i+1], replay[i]]) for i in range(len(replay)-1))
    LL_rev = sum(np.log(T_rev[replay[i+1], replay[i]]) for i in range(len(replay)-1))
    score = LL_fwd - LL_rev
    direction = "forward" if score > 0 else "reverse"
    print(f"{direction} replay (score: {score:.2f})")
```

**Regularization**:

```python
from neurospatial import (
    empirical_transition_matrix,
    diffusion_transition_matrix,
    smooth_transition_matrix,
)

# Limited data: regularize with smooth prior
T_emp = empirical_transition_matrix(short_trajectory, env.n_bins)
T_prior = diffusion_transition_matrix(env, diffusion_time=1.0)
T_smooth = smooth_transition_matrix(T_emp, T_prior, prior_weight=0.2)
```

### For Existing Users

**No breaking changes**. All additions are new functions.

**Recommended workflow enhancements**:

1. **Replace manual transition counting**:

   ```python
   # Old way
   T = np.zeros((n_bins, n_bins))
   for i in range(len(trajectory) - 1):
       T[trajectory[i+1], trajectory[i]] += 1
   T /= T.sum(axis=0, keepdims=True)

   # New way
   from neurospatial import empirical_transition_matrix
   T = empirical_transition_matrix(trajectory, n_bins)
   ```

2. **Add directional analysis**:

   ```python
   # Separate forward/reverse replay
   T_fwd, T_rev = directional_transition_matrices(trajectory, env)
   # Analyze replay events by direction
   ```

3. **Use structure-aware null models**:

   ```python
   # Old: uniform over all bins (ignores connectivity)
   T_null = np.ones((n_bins, n_bins)) / n_bins

   # New: uniform over graph neighbors (respects walls/boundaries)
   T_null = uniform_transition_matrix(env.connectivity, n_bins)
   ```

---

## Implementation Strategy

### Phase 1: Core Functions (Priority 1)

**Functions**: 1-3, 5, 7

1. `empirical_transition_matrix`
2. `uniform_transition_matrix`
3. `distance_biased_transition_matrix`
4. `smooth_transition_matrix`
5. `transition_entropy`

**Rationale**: Most commonly used, minimal dependencies, foundation for others.

**Estimated effort**: 2-3 days

- Implementation: 1 day
- Tests: 1 day
- Documentation: 0.5 day

### Phase 2: Diffusion and Mixing (Priority 2)

**Functions**: 4, 6

1. `diffusion_transition_matrix` (uses existing `compute_diffusion_kernels`)
2. `mix_transition_matrices`

**Rationale**: Depends on Phase 1, but independent of directional analysis.

**Estimated effort**: 1 day

- Implementation: 0.5 day (reuses existing code)
- Tests: 0.5 day

### Phase 3: Directional Analysis (Priority 3)

**Functions**: 8, 9

1. `directional_transition_matrices`
2. `momentum_transition_matrix`

**Rationale**: Most complex, requires PCA and vector geometry, builds on Phase 1.

**Estimated effort**: 3-4 days

- Implementation: 2 days (directional classification logic, callable design for momentum)
- Tests: 1-1.5 days (more edge cases)
- Documentation: 0.5 day

### Total Timeline

**Sequential**: ~6-8 days
**With code review**: +2 days
**Total**: ~8-10 days (2 work weeks)

---

## Summary

**9 transition matrix construction functions** organized into three categories:

1. **Basic construction** (3 functions): empirical, uniform, distance-biased
2. **Regularization and mixing** (3 functions): diffusion, smooth, mix
3. **Analysis** (1 function): entropy
4. **Directional** (2 functions): directional split, momentum

**Key features**:

- All sparse matrices (memory efficient) or callables (momentum)
- Graph-aware (respects connectivity)
- Smoothing support (handles sparse data)
- NumPy docstrings (comprehensive documentation)
- Neuroscience-focused (hippocampal replay use cases)
- No HMM inference (construction only, explicit scope boundary)
- Optimized implementations (sparse adjacency masking, efficient smoothing)
- Performance limits documented (diffusion: n_bins ≤ 500)

**Enables new workflows**:

- Empirical transition models from behavior
- Synthetic null models for statistical testing
- Forward vs reverse replay analysis
- Momentum-aware sequence generation with natural callable API
- Regularized decoding with smooth priors
- Directional bias quantification

**Zero breaking changes**: All additions to new module `transitions.py`, optional imports.
