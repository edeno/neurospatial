# Implementation Plan: Advanced Behavioral Analysis Metrics

**Created**: 2025-12-05
**Revised**: 2025-12-05 (post code-review and UX-review)
**Status**: Ready for implementation
**Scope**: Reachability/information theory, manifold embeddings, diffusion baselines, geodesic VTE
**Dependencies**: BEHAV_PLAN.md (path efficiency, goal-directed, decision analysis, VTE)

---

## Overview

This plan implements **four advanced behavioral analysis directions** that leverage neurospatial's unique capabilities: environment geometry, connectivity graphs, Laplacian operators, diffusion kernels, egocentric transforms, geodesic distances, and visibility/viewshed modeling.

**Key Principle**: These metrics go beyond traditional Euclidean behavioral analysis by operating in the **space the animal actually experiences**—respecting walls, barriers, visibility, and graph topology.

### Relationship to BEHAV_PLAN.md

This plan **extends** the behavioral trajectory metrics from BEHAV_PLAN.md:

| BEHAV_PLAN.md | BEHAV_PLAN2.md (this plan) |
|---------------|----------------------------|
| Path efficiency (traveled vs shortest) | Deviation-from-diffusion (planned vs random) |
| Goal-directed metrics (goal bias) | Information-theoretic reachability |
| Decision analysis (pre-decision window) | Manifold-space trajectory analysis |
| VTE (IdPhi head sweep) | Geodesic VTE (topology-aware deliberation) |

**Implementation order**: Complete BEHAV_PLAN.md modules first, then implement this plan.

### Critical Prerequisites from BEHAV_PLAN.md

**Module 4 (Geodesic VTE Extensions) requires these functions from BEHAV_PLAN.md vte.py:**

```python
# These MUST exist before implementing geodesic VTE:
from neurospatial.metrics.vte import wrap_angle, head_sweep_magnitude
from neurospatial.metrics.vte import VTETrialResult, VTESessionResult
```

Modules 1-3 can be implemented independently of BEHAV_PLAN.md.

---

## Existing Code Inventory

### Already Implemented (DO NOT Reimplement)

| Capability | Function/Module | Location |
|------------|-----------------|----------|
| **Graph Laplacian** | `compute_diffusion_kernels()` | `kernels.py:34-154` |
| **Differential operator** | `compute_differential_operator()` | `differential.py:31-144` |
| **Gradient/Divergence** | `gradient()`, `divergence()` | `differential.py:147-375` |
| **Kernel application** | `apply_kernel()` | `kernels.py:157-249` |
| **Geodesic distance matrix** | `geodesic_distance_matrix()` | `distance.py:32-80` |
| **Distance field** | `distance_field()` | `distance.py:180-365` |
| **Viewshed computation** | `compute_viewshed()` | `visibility.py:682-838` |
| **Field of view** | `FieldOfView` dataclass | `visibility.py:86-450` |
| **Viewed location** | `compute_viewed_location()` | `visibility.py:468-565` |
| **Egocentric transforms** | `allocentric_to_egocentric()` | `reference_frames.py:164-267` |
| **Egocentric bearing** | `compute_egocentric_bearing()` | `reference_frames.py:337-392` |
| **Heading from velocity** | `heading_from_velocity()` | `reference_frames.py:524-621` |
| **Path progress** | `path_progress()` | `behavioral.py:182-356` |
| **Distance to region** | `distance_to_region()` | `behavioral.py:359-493` |

**Note**: `neighbors_within()` was originally listed but does not exist with the expected signature.
Use `distance_field()` with thresholding instead:

```python
# CORRECT approach for reachable set:
from neurospatial.distance import distance_field
dist_field = distance_field(G, [position_bin], metric="geodesic")
reachable_bins = np.where(dist_field <= radius)[0]
```

### NOT Implemented (New in This Plan)

| Capability | New Module |
|------------|------------|
| Reachability & future entropy | `reachability.py` |
| Laplacian eigenfunctions & manifold embedding | `manifold.py` |
| Diffusion deviation metrics | `diffusion_baseline.py` |
| Geodesic VTE measures | Extends `vte.py` from BEHAV_PLAN.md |

---

## Design Decisions

### Parameter Naming Convention

**Consistent with BEHAV_PLAN.md**: Use `metric` parameter (not `distance_type`).

```python
# Correct:
metric: Literal["geodesic", "euclidean"] = "geodesic"
```

### Diffusion Time Parameter

**Decision**: Use `tau` for diffusion time scale, consistent with diffusion kernel literature.

```python
# Diffusion kernel: P(τ) = exp(-τL)
tau: float = 1.0  # Diffusion time scale
```

**Relationship to `bandwidth_sigma` in `compute_diffusion_kernels()`:**

The existing `kernels.py` uses `bandwidth_sigma` where `t = sigma² / 2`.
To convert from user-facing `tau` to internal `bandwidth_sigma`:

```python
# kernel = exp(-t * L) where t = sigma² / 2
# If user specifies tau, then t = tau, so sigma = sqrt(2 * tau)
bandwidth_sigma = np.sqrt(2.0 * tau)
```

### Parameter Calibration Guide

**Choosing `tau` (diffusion time scale):**

| Environment | Bin Count | Suggested `tau` Range |
|-------------|-----------|----------------------|
| Small maze | ~100 bins | 0.5 - 2.0 |
| Medium maze | ~500 bins | 1.0 - 5.0 |
| Large maze | ~1000+ bins | 2.0 - 10.0 |

**Interpretation**: `tau` represents how many effective graph-hops the random walker takes.
Larger `tau` = more spread in future distribution = lower entropy contrast between locations.

**Calibration**: Plot `future_entropy` vs `tau` for sample positions. Decision points show
entropy peaks at characteristic `tau` values.

**Choosing `radius` (geodesic radius for reachability):**

| Goal | Suggested `radius` |
|------|-------------------|
| Local neighborhood | 3-5 × bin_size |
| Regional scale | 0.25 × environment_diameter |
| Captures next 1-2 decision points | Validate empirically |

**Too small**: Misses upcoming choice points, underestimates entropy.
**Too large**: Everything reachable, metrics become uniform.
**Sweet spot**: Captures next 1-2 decision points ahead.

### Manifold Embedding Dimensions

**Decision**: Default to 3 dimensions for manifold embedding (sufficient for most 2D environments).

```python
n_components: int = 3  # Number of Laplacian eigenfunctions
```

### VTE Extensions

**Decision**: Geodesic VTE functions extend (not replace) the basic VTE module from BEHAV_PLAN.md.

---

## New Modules

### Module 1: Reachability & Information-Theoretic Behavior

**File**: `src/neurospatial/metrics/reachability.py`

#### Module Docstring

```python
"""Reachability and information-theoretic metrics for spatial behavior.

This module quantifies the space of future actions available to an animal
based on environment geometry, using graph-theoretic and information-theoretic
measures.

Key Concepts
------------
- **Reachable set**: Bins within geodesic distance r from current position
- **Future path entropy**: Uncertainty about future positions given diffusion dynamics
- **Geodesic options**: Number of distinct first-step directions on shortest paths
  (Note: This differs from graph degree - see `n_geodesic_options` docs)

When to Use
-----------
Use these metrics when analyzing:
- Exploration behavior (high entropy = many options)
- Constrained navigation (low entropy = corridor)
- Decision point identification (high branching)
- Cognitive load estimation (entropy as proxy for planning complexity)

Interpreting Future Entropy
---------------------------
Future entropy quantifies navigational uncertainty:

- **0 bits**: Only one possible future position (dead end, forced path)
- **1 bit**: Two equally-likely options (T-junction)
- **2 bits**: Four equally-likely options (open field center)
- **log2(N) bits**: Maximum for N reachable bins (uniform distribution)

Behavioral meaning:

- High entropy = Exploratory context, many choices available
- Low entropy = Constrained navigation, corridor-like
- Entropy drop = Approaching decision requires commitment

Cognitive interpretation (speculative):

- Higher entropy may correlate with increased deliberation
- Entropy profile along trajectory reveals planning demands

Quick Start (Typical Neuroscience Workflow)
-------------------------------------------
1. Calibrate parameters on sample data:

   - Use tau=1.0, radius=0.25*environment_diameter as starting point
   - Validate: Plot entropy profile, verify peaks at known decision points

2. Compute metrics for full session::

       results = [compute_reachability_metrics(env, pos, tau=1.0, radius=50.0)
                  for pos in trial_positions]

3. Extract behavioral correlates::

       decision_trials = [r for r in results if r.is_decision_point()]
       mean_entropy = np.mean([r.future_entropy for r in results])

4. Statistical test: Compare mean_entropy between task conditions using t-test

Example
-------
>>> from neurospatial.metrics import compute_reachability_metrics
>>> result = compute_reachability_metrics(env, position, tau=1.0, radius=50.0)
>>> print(f"Future entropy: {result.future_entropy:.2f} bits")
>>> if result.future_entropy > 2.0:
...     print("High uncertainty - many future paths available")

References
----------
.. [1] Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
       Computational Harmonic Analysis, 21(1), 5-30.
.. [2] Nadler, B., et al. (2006). Diffusion maps, spectral clustering and
       reaction coordinates of dynamical systems.
"""
```

#### Data Structures

```python
@dataclass(frozen=True)
class ReachabilityMetrics:
    """Reachability metrics at a single position.

    Attributes
    ----------
    position_bin : int
        Current position bin index.
    radius : float
        Geodesic radius used for reachable set.
    tau : float
        Diffusion time scale used for entropy computation.
    reachable_bins : NDArray[np.int_]
        Indices of bins within geodesic radius, shape (n_reachable,).
    n_reachable : int
        Number of reachable bins.
    reachable_fraction : float
        Fraction of total environment reachable (0 to 1).
    future_distribution : NDArray[np.float64]
        Probability of being at each bin after diffusion time tau, shape (n_bins,).
    future_entropy : float
        Shannon entropy of diffusion distribution (bits).
        High values indicate many equally-likely future positions.
        See module docstring for interpretation guide.
    n_geodesic_options : int
        Number of distinct first-step directions on shortest paths within radius.

        **Important**: This differs from graph degree (immediate neighbors):

        - Graph degree: How many physical exits exist from this bin
        - Geodesic options: How many distinct path choices matter for reaching
          destinations within radius

        Example: 4-way junction with 2 goals → degree=4, geodesic_options=2

        High values indicate a decision point where the animal must choose
        between meaningfully different paths.
    """

    # Input parameters (first for logical ordering)
    position_bin: int
    radius: float
    tau: float

    # Primary outputs
    reachable_bins: NDArray[np.int_]
    n_reachable: int
    reachable_fraction: float

    # Derived outputs
    future_distribution: NDArray[np.float64]
    future_entropy: float
    n_geodesic_options: int

    def is_decision_point(self, min_options: int = 3) -> bool:
        """Check if position qualifies as a decision point.

        Parameters
        ----------
        min_options : int, default=3
            Minimum geodesic options to qualify as decision point.

            **Choosing threshold**:

            - 2: Any bifurcation (e.g., T-junction in linear track)
            - 3: True multi-choice (e.g., center of plus maze) [default]
            - 4+: Complex decision points (radial arm maze)

            **Recommendation**: Start with 3 for typical maze tasks.
            Validate by comparing to hand-labeled decision regions.

        Returns
        -------
        bool
            True if n_geodesic_options >= min_options.

        See Also
        --------
        future_entropy : Alternative decision point detection via uncertainty
        """
        return self.n_geodesic_options >= min_options

    def is_constrained(self, max_entropy: float = 1.0) -> bool:
        """Check if position is in a constrained region (corridor).

        Parameters
        ----------
        max_entropy : float, default=1.0
            Maximum entropy (bits) to qualify as constrained.

            **Interpretation**:

            - < 1.0 bit: Highly constrained (corridor, dead end)
            - 1.0-2.0 bits: Moderately constrained
            - > 2.0 bits: Open area with many options

        Returns
        -------
        bool
            True if future_entropy <= max_entropy.
        """
        return self.future_entropy <= max_entropy

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Reachability: {self.n_reachable} bins ({self.reachable_fraction:.1%}), "
            f"entropy={self.future_entropy:.2f} bits, options={self.n_geodesic_options}"
        )


@dataclass(frozen=True)
class ReachabilityTrajectory:
    """Reachability metrics along a trajectory.

    Attributes
    ----------
    future_entropy : NDArray[np.float64]
        Entropy at each timepoint, shape (n_samples,).
    n_reachable : NDArray[np.int_]
        Number of reachable bins at each timepoint, shape (n_samples,).
    n_geodesic_options : NDArray[np.int_]
        Number of geodesic options at each timepoint, shape (n_samples,).
    mean_entropy : float
        Mean entropy over trajectory.
    decision_point_indices : NDArray[np.int_]
        Indices where n_geodesic_options >= threshold.
    """

    future_entropy: NDArray[np.float64]
    n_reachable: NDArray[np.int_]
    n_geodesic_options: NDArray[np.int_]
    mean_entropy: float
    decision_point_indices: NDArray[np.int_]

    @property
    def n_decision_points(self) -> int:
        """Number of decision points encountered."""
        return len(self.decision_point_indices)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Trajectory: mean_entropy={self.mean_entropy:.2f} bits, "
            f"{self.n_decision_points} decision points"
        )
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `reachable_set` | `(env, position_bin, radius) -> NDArray[np.int_]` | Bins within geodesic radius | `distance_field()` + threshold |
| `future_distribution` | `(env, position_bin, tau) -> NDArray[np.float64]` | Diffusion kernel row | `compute_diffusion_kernels()` |
| `future_entropy` | `(distribution) -> float` | Shannon entropy of distribution | `scipy.stats.entropy()` |
| `count_geodesic_options` | `(env, position_bin, radius) -> int` | Distinct first-step directions | `distance_field()` + neighbor analysis |
| `compute_reachability_metrics` | `(env, position, tau, radius) -> ReachabilityMetrics` | All metrics at position | All above |
| `compute_reachability_trajectory` | `(env, positions, tau, radius, ...) -> ReachabilityTrajectory` | Metrics along trajectory | `compute_reachability_metrics()` |

#### Implementation Notes

```python
def reachable_set(
    env: Environment,
    position_bin: int,
    radius: float,
) -> NDArray[np.int_]:
    """Find bins within geodesic radius of position.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position_bin : int
        Current position bin index.
    radius : float
        Geodesic radius to consider.

    Returns
    -------
    NDArray[np.int_], shape (n_reachable,)
        Indices of reachable bins.
    """
    from neurospatial.distance import distance_field

    # Compute distance field from position
    dist_field = distance_field(
        env.connectivity, [position_bin], metric="geodesic"
    )

    # Threshold to get reachable bins
    reachable_bins = np.where(dist_field <= radius)[0]
    return reachable_bins


def future_distribution(
    env: Environment,
    position_bin: int,
    tau: float,
) -> NDArray[np.float64]:
    """Compute probability distribution over future positions.

    Uses the diffusion kernel P(τ) = exp(-τL) where L is the graph Laplacian.
    The i-th row gives the probability of transitioning from bin i to all other bins.

    Parameters
    ----------
    env : Environment
        Spatial environment with connectivity graph.
    position_bin : int
        Current position bin index.
    tau : float
        Diffusion time scale. Larger values = more spread.
        Typical range: 0.1 to 10.0 depending on environment size.
        See module docstring for calibration guide.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Probability of being at each bin after time tau.
        Sums to 1.0.

    Notes
    -----
    The bandwidth parameter for compute_diffusion_kernels is related to tau via:
    bandwidth_sigma = sqrt(2 * tau)

    Performance: O(n_bins^3) for kernel computation. Consider caching
    for repeated calls with same tau.
    """
    from neurospatial.kernels import compute_diffusion_kernels

    # Convert tau to bandwidth_sigma for compute_diffusion_kernels
    # The relationship is: kernel = exp(-t * L) where t = sigma^2/2
    # Therefore: sigma = sqrt(2 * tau)
    bandwidth_sigma = np.sqrt(2.0 * tau)

    # Compute kernel (NOTE: caller should cache for efficiency)
    kernel = compute_diffusion_kernels(
        env.connectivity,
        bandwidth_sigma=bandwidth_sigma,
        mode="transition",
    )

    # Return the row for current position
    return kernel[position_bin, :]


def future_entropy(distribution: NDArray[np.float64]) -> float:
    """Compute Shannon entropy of future position distribution.

    Parameters
    ----------
    distribution : NDArray[np.float64], shape (n_bins,)
        Probability distribution over bins. Should sum to 1.

    Returns
    -------
    float
        Entropy in bits. Higher values indicate more uncertainty.

    Notes
    -----
    Entropy is computed as: H = -sum(p * log2(p)) for p > 0.
    Maximum entropy for n bins is log2(n) (uniform distribution).

    For delta functions (single bin with probability ~1), returns 0.0.
    """
    from scipy.stats import entropy as scipy_entropy

    # Handle delta function (zero entropy)
    if np.max(distribution) > 0.999:
        return 0.0

    # Filter zero probabilities to avoid log(0)
    p = distribution[distribution > 0]
    return float(scipy_entropy(p, base=2))


def count_geodesic_options(
    env: Environment,
    position_bin: int,
    radius: float,
) -> int:
    """Count distinct first-step directions on shortest paths within radius.

    A geodesic option is a neighbor of the current position that lies on
    the shortest path to at least one bin within the radius. This counts
    how many meaningfully different path choices exist.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position_bin : int
        Current position bin index.
    radius : float
        Geodesic radius to consider.

    Returns
    -------
    int
        Number of distinct geodesic options.

    Notes
    -----
    **Important distinction from graph degree**:

    - Graph degree: Number of immediate neighbors (physical exits)
    - Geodesic options: Number of distinct paths that matter for reaching
      destinations within radius

    Example: At a 4-way junction but with only 2 goals within radius on
    opposite sides → graph degree = 4, geodesic options = 2

    **Algorithm**:

    1. Compute distance field from position
    2. Find all bins within radius
    3. For each neighbor, check if it's on a shortest path to any reachable bin
    4. Count neighbors that are on at least one shortest path

    Performance: O(V * degree) where V = n_bins, degree = mean graph degree
    """
    from neurospatial.distance import distance_field

    G = env.connectivity

    # Get immediate neighbors
    neighbors = list(G.neighbors(position_bin))

    if len(neighbors) == 0:
        return 0

    # Compute distance field from position
    dist_from_pos = distance_field(
        G, [position_bin], metric="geodesic"
    )

    # Find reachable bins
    reachable_bins = np.where(dist_from_pos <= radius)[0]

    if len(reachable_bins) <= 1:
        # Only the position itself is reachable
        return 0

    # For each neighbor, check if it's on a shortest path to any reachable bin
    # A neighbor n is on a shortest path to bin b if:
    #   dist(pos, b) = edge_weight(pos, n) + dist(n, b)
    options_used = set()

    for neighbor in neighbors:
        # Get edge weight
        edge_weight = G[position_bin][neighbor].get('distance', 1.0)

        # Compute distance from this neighbor to all bins
        dist_from_neighbor = distance_field(
            G, [neighbor], metric="geodesic"
        )

        # Check if this neighbor is on shortest path to any reachable bin
        for bin_idx in reachable_bins:
            if bin_idx == position_bin:
                continue

            total_dist = dist_from_pos[bin_idx]
            via_neighbor_dist = edge_weight + dist_from_neighbor[bin_idx]

            # Allow small tolerance for floating point
            if np.isclose(total_dist, via_neighbor_dist, atol=1e-6):
                options_used.add(neighbor)
                break  # Found one path through this neighbor, move to next

    return len(options_used)
```

#### Error Messages

```python
# Environment not fitted
if not env._is_fitted:
    raise RuntimeError(
        "Environment must be fitted before computing reachability metrics. "
        "Use Environment.from_samples() or another factory method."
    )

# Position outside environment
if not env.contains(position):
    raise ValueError(
        f"Position {position} is outside environment bounds. "
        f"Use env.contains() to check validity before computing metrics."
    )

# Invalid tau
if tau <= 0:
    raise ValueError(
        f"tau must be positive, got {tau}. "
        f"Typical values are 0.1 to 10.0 depending on environment size."
    )
```

---

### Module 2: Manifold Embedding & Trajectory Analysis

**File**: `src/neurospatial/metrics/manifold.py`

#### Module Docstring

```python
"""Manifold embedding for trajectory analysis in cognitive-map coordinates.

This module embeds the environment into low-dimensional coordinates using
Laplacian eigenfunctions (spectral embedding / diffusion maps). Trajectories
analyzed in this space reflect the *intrinsic maze geometry* rather than
Euclidean layout.

Key Concepts
------------
- **Laplacian eigenfunctions**: Natural coordinates respecting connectivity
- **Manifold embedding**: φ(i) = (λ₁ᵗφ₁(i), λ₂ᵗφ₂(i), ..., λₘᵗφₘ(i))
- **Manifold velocity**: Rate of change in embedded coordinates
- **Manifold path efficiency**: Straightness in cognitive-map space

When to Use
-----------
Use manifold analysis when:
- Testing if behavior is "straight" in cognitive space (even if curved in Euclidean)
- Comparing exploration vs goal-directed navigation
- Detecting learned shortcuts (straight manifold paths through complex mazes)
- Analyzing behavior relative to hippocampal map-like representations

Example
-------
>>> from neurospatial.metrics import compute_manifold_embedding, manifold_trajectory
>>> embedding = compute_manifold_embedding(env, n_components=3, tau=1.0)
>>> result = manifold_trajectory(env, positions, embedding)
>>> print(f"Manifold path efficiency: {result.path_efficiency:.2f}")
>>> if result.path_efficiency > 0.8:
...     print("Straight path in cognitive space - well-learned navigation")

References
----------
.. [1] Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
       Computational Harmonic Analysis, 21(1), 5-30.
.. [2] Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017).
       The hippocampus as a predictive map. Nature Neuroscience.
"""
```

#### Data Structures

```python
@dataclass(frozen=True)
class ManifoldEmbedding:
    """Laplacian eigenfunction embedding of environment.

    Attributes
    ----------
    coordinates : NDArray[np.float64]
        Embedded coordinates for each bin, shape (n_bins, n_components).
    eigenvalues : NDArray[np.float64]
        Laplacian eigenvalues, shape (n_components,).
        Sorted ascending (eigenvalue[0] ≈ 0 for connected graph).
    eigenvectors : NDArray[np.float64]
        Laplacian eigenvectors, shape (n_bins, n_components).
        Column i is eigenvector for eigenvalue[i].
    n_components : int
        Number of embedding dimensions.
    tau : float
        Diffusion time scale used for embedding.
    """

    coordinates: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    n_components: int
    tau: float

    def embed_positions(
        self, positions: NDArray[np.float64], env: Environment
    ) -> NDArray[np.float64]:
        """Embed positions into manifold coordinates.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Allocentric positions.
        env : Environment
            Environment for bin lookup.

        Returns
        -------
        NDArray[np.float64], shape (n_samples, n_components)
            Manifold coordinates for each position.
        """
        bins = env.bin_at(positions)
        return self.coordinates[bins]

    def manifold_distance(self, bin_i: int, bin_j: int) -> float:
        """Compute distance between bins in manifold space.

        Parameters
        ----------
        bin_i, bin_j : int
            Bin indices.

        Returns
        -------
        float
            Euclidean distance in manifold coordinates.
        """
        return float(np.linalg.norm(
            self.coordinates[bin_i] - self.coordinates[bin_j]
        ))


@dataclass(frozen=True)
class ManifoldTrajectoryResult:
    """Trajectory analysis in manifold space.

    Attributes
    ----------
    manifold_positions : NDArray[np.float64]
        Trajectory in manifold coordinates, shape (n_samples, n_components).
    manifold_velocity : NDArray[np.float64]
        Velocity in manifold space, shape (n_samples-1, n_components).
    manifold_speed : NDArray[np.float64]
        Speed in manifold space, shape (n_samples-1,).
    manifold_curvature : NDArray[np.float64]
        Curvature in manifold space, shape (n_samples-2,).
    path_efficiency : float
        Ratio of manifold displacement to manifold path length.
        1.0 = straight line in manifold space.
    total_manifold_distance : float
        Total path length in manifold space.
    net_manifold_displacement : float
        Straight-line distance from start to end in manifold space.
    """

    manifold_positions: NDArray[np.float64]
    manifold_velocity: NDArray[np.float64]
    manifold_speed: NDArray[np.float64]
    manifold_curvature: NDArray[np.float64]
    path_efficiency: float
    total_manifold_distance: float
    net_manifold_displacement: float

    def is_straight(self, threshold: float = 0.8) -> bool:
        """Check if trajectory is straight in manifold space.

        Parameters
        ----------
        threshold : float, default=0.8
            Efficiency threshold for "straight" classification.

        Returns
        -------
        bool
            True if path_efficiency > threshold.
        """
        return self.path_efficiency > threshold

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Manifold trajectory: efficiency={self.path_efficiency:.2f}, "
            f"distance={self.total_manifold_distance:.2f}, "
            f"displacement={self.net_manifold_displacement:.2f}"
        )
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `compute_laplacian_eigenfunctions` | `(env, n_components) -> tuple[eigenvalues, eigenvectors]` | Compute Laplacian spectrum | `scipy.sparse.linalg.eigsh()` |
| `compute_manifold_embedding` | `(env, n_components, tau) -> ManifoldEmbedding` | Create diffusion map embedding | `compute_laplacian_eigenfunctions()` |
| `manifold_velocity` | `(manifold_positions, dt) -> NDArray` | Velocity in embedded space | `np.diff()` |
| `manifold_curvature` | `(manifold_positions) -> NDArray` | Curvature in embedded space | Frenet-Serret |
| `manifold_path_efficiency` | `(manifold_positions) -> float` | Displacement / path length | Vectorized numpy |
| `manifold_trajectory` | `(env, positions, embedding) -> ManifoldTrajectoryResult` | Complete analysis | All above |

#### Implementation Notes

```python
def compute_laplacian_eigenfunctions(
    env: Environment,
    n_components: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Laplacian eigenfunctions for environment graph.

    Parameters
    ----------
    env : Environment
        Spatial environment with connectivity graph.
    n_components : int, default=6
        Number of eigenfunctions to compute.
        The first eigenfunction (constant) is always included.

    Returns
    -------
    eigenvalues : NDArray[np.float64], shape (n_components,)
        Eigenvalues sorted ascending. eigenvalues[0] ≈ 0 for connected graph.
    eigenvectors : NDArray[np.float64], shape (n_bins, n_components)
        Corresponding eigenvectors. Column i is eigenvector for eigenvalue[i].

    Raises
    ------
    RuntimeError
        If environment graph is disconnected. This would result in
        multiple zero eigenvalues and degenerate eigenvectors.

    Warnings
    --------
    ResourceWarning
        For environments with >1000 bins, computation may be slow.
        Consider downsampling via larger bin_size.

    Notes
    -----
    Uses the unnormalized graph Laplacian L = D - W where D is degree matrix
    and W is adjacency matrix. For the normalized Laplacian, eigenvectors
    would need rescaling.

    The first eigenvector is constant (all 1/sqrt(n)) with eigenvalue 0.
    Subsequent eigenvectors capture increasingly fine-grained structure.

    **Sign convention**: Eigenvectors are made deterministic by ensuring
    the first non-zero element is positive. This matters for reproducibility
    since eigsh can return eigenvectors with arbitrary sign.

    Performance: O(n_bins * n_components) using iterative eigensolver.
    For >1000 bins, consider using fewer components or larger bin_size.
    """
    import networkx as nx
    from scipy.sparse.linalg import eigsh
    import warnings

    n_bins = env.n_bins

    # Performance warning for large environments
    if n_bins > 1000:
        warnings.warn(
            f"Computing Laplacian eigenfunctions for {n_bins} bins. "
            f"This may be slow. Consider using larger bin_size to reduce bins.",
            ResourceWarning,
            stacklevel=2,
        )

    # Check for disconnected graph
    if not nx.is_connected(env.connectivity):
        n_components_graph = nx.number_connected_components(env.connectivity)
        raise RuntimeError(
            f"Environment graph is disconnected ({n_components_graph} components). "
            f"Laplacian eigenfunctions require a connected graph. "
            f"\nTo fix:\n"
            f"- Use Environment.from_samples() with dilate=True to connect nearby bins\n"
            f"- Use fill_holes=True to close gaps\n"
            f"- Manually check env.connectivity for isolated regions"
        )

    # Get Laplacian matrix
    L = nx.laplacian_matrix(env.connectivity).astype(np.float64)

    # Compute smallest eigenvalues/vectors (Laplacian is positive semi-definite)
    # We want the smallest eigenvalues, so use which='SM' (smallest magnitude)
    eigenvalues, eigenvectors = eigsh(
        L, k=n_components, which='SM', return_eigenvectors=True
    )

    # Sort by eigenvalue (eigsh doesn't guarantee order)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fix sign convention: ensure first non-zero element is positive
    # This makes results deterministic (eigsh can return ±v)
    for i in range(eigenvectors.shape[1]):
        first_nonzero_idx = np.argmax(np.abs(eigenvectors[:, i]) > 1e-10)
        if eigenvectors[first_nonzero_idx, i] < 0:
            eigenvectors[:, i] *= -1

    return eigenvalues, eigenvectors


def compute_manifold_embedding(
    env: Environment,
    n_components: int = 3,
    tau: float = 1.0,
    skip_first: bool = True,
) -> ManifoldEmbedding:
    """Create diffusion map embedding of environment.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    n_components : int, default=3
        Number of embedding dimensions.
    tau : float, default=1.0
        Diffusion time scale. Controls how much eigenvalues are dampened.
        Larger tau emphasizes global structure (low-frequency eigenfunctions).
    skip_first : bool, default=True
        Skip the first (constant) eigenvector. Usually True since it
        contains no structural information.

    Returns
    -------
    ManifoldEmbedding
        Embedding with coordinates for each bin.

    Notes
    -----
    The diffusion map embedding is:
        Ψ(i) = (λ₁ᵗ φ₁(i), λ₂ᵗ φ₂(i), ..., λₘᵗ φₘ(i))

    where λₖ are eigenvalues and φₖ are eigenvectors of the Laplacian.
    The τ parameter controls the time scale of the diffusion process.
    """
    # Request one extra component if skipping first
    n_compute = n_components + 1 if skip_first else n_components

    eigenvalues, eigenvectors = compute_laplacian_eigenfunctions(env, n_compute)

    if skip_first:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]

    # Apply diffusion time scaling: λᵗ
    # For Laplacian eigenvalues, we use exp(-τλ) as the scaling
    # (consistent with diffusion kernel exp(-τL))
    scaling = np.exp(-tau * eigenvalues)

    # Compute embedding coordinates
    coordinates = eigenvectors * scaling[np.newaxis, :]

    return ManifoldEmbedding(
        coordinates=coordinates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_components=n_components,
        tau=tau,
    )


def manifold_trajectory(
    env: Environment,
    positions: NDArray[np.float64],
    embedding: ManifoldEmbedding,
    times: NDArray[np.float64] | None = None,
) -> ManifoldTrajectoryResult:
    """Analyze trajectory in manifold space.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Allocentric positions.
    embedding : ManifoldEmbedding
        Pre-computed manifold embedding.
    times : NDArray[np.float64], optional
        Timestamps for velocity computation. If None, assumes uniform dt=1.

    Returns
    -------
    ManifoldTrajectoryResult
        Complete manifold trajectory analysis.
    """
    # Embed trajectory
    manifold_pos = embedding.embed_positions(positions, env)

    # Compute dt
    if times is not None:
        dt = np.diff(times)
    else:
        dt = np.ones(len(positions) - 1)

    # Velocity in manifold space
    velocity = np.diff(manifold_pos, axis=0) / dt[:, np.newaxis]

    # Speed (magnitude of velocity)
    speed = np.linalg.norm(velocity, axis=1)

    # Curvature (requires at least 3 points)
    if len(manifold_pos) >= 3:
        # Use discrete curvature: |v(t+1) - v(t)| / |v(t)|
        dv = np.diff(velocity, axis=0)
        v_mag = speed[:-1]

        # Numerical stability: use relative threshold based on typical speed
        # This avoids spurious high curvature values when animal is stationary
        median_speed = np.median(speed[speed > 0]) if np.any(speed > 0) else 1.0
        min_speed_threshold = 0.01 * median_speed  # 1% of median speed

        # Mask out low-speed regions where curvature is ill-defined
        valid_mask = v_mag > min_speed_threshold
        curvature = np.full(len(dv), np.nan)
        curvature[valid_mask] = np.linalg.norm(dv[valid_mask], axis=1) / v_mag[valid_mask]
    else:
        curvature = np.array([])

    # Path efficiency
    total_distance = np.sum(speed * dt)
    net_displacement = np.linalg.norm(manifold_pos[-1] - manifold_pos[0])

    if total_distance > 1e-10:
        efficiency = net_displacement / total_distance
    else:
        efficiency = np.nan

    return ManifoldTrajectoryResult(
        manifold_positions=manifold_pos,
        manifold_velocity=velocity,
        manifold_speed=speed,
        manifold_curvature=curvature,
        path_efficiency=float(efficiency),
        total_manifold_distance=float(total_distance),
        net_manifold_displacement=float(net_displacement),
    )
```

---

### Module 3: Diffusion Baseline Comparison

**File**: `src/neurospatial/metrics/diffusion_baseline.py`

#### Module Docstring

```python
"""Compare trajectories against diffusion-based random walk baselines.

This module provides a principled null model for navigation behavior based
on random walks shaped by environment geometry. It quantifies how "planned"
or "intentional" behavior is by comparing to what an unbiased explorer would do.

Key Concepts
------------
- **Diffusion null model**: Random walk on environment graph via exp(-τL)
- **Expected next position**: Weighted average of neighbors by transition probability
- **Diffusion deviation**: Distance between actual and expected positions
- **Navigational intentionality**: Systematic deviation from random exploration

When to Use
-----------
Use diffusion baselines when:
- Quantifying goal-directed vs exploratory behavior
- Detecting shortcuts (deviation toward goal)
- Identifying avoidance (deviation away from region)
- Measuring planning/intentionality in navigation

Example
-------
>>> from neurospatial.metrics import compute_diffusion_deviation
>>> result = compute_diffusion_deviation(env, positions, times, tau=1.0)
>>> print(f"Mean deviation: {result.mean_deviation:.2f} cm")
>>> if result.deviation_significance > 2.0:
...     print("Significant deviation from random walk - intentional navigation")

References
----------
.. [1] Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
.. [2] Nadler, B., et al. (2006). Diffusion maps and spectral clustering.
"""
```

#### Data Structures

```python
@dataclass(frozen=True)
class DiffusionDeviationResult:
    """Comparison of trajectory against diffusion baseline.

    Attributes
    ----------
    deviation : NDArray[np.float64]
        Distance between actual and expected positions at each step,
        shape (n_samples-1,). Units match environment.
    deviation_geodesic : NDArray[np.float64]
        Geodesic distance version of deviation, shape (n_samples-1,).
    expected_positions : NDArray[np.float64]
        Diffusion-expected positions, shape (n_samples-1, n_dims).
    actual_positions : NDArray[np.float64]
        Actual next positions, shape (n_samples-1, n_dims).
    mean_deviation : float
        Mean Euclidean deviation across trajectory.
    mean_deviation_geodesic : float
        Mean geodesic deviation across trajectory.
    deviation_significance : float
        Z-score of mean deviation relative to null distribution.
        Values > 2 indicate significant deviation from random walk.
    tau : float
        Diffusion time scale used.
    """

    deviation: NDArray[np.float64]
    deviation_geodesic: NDArray[np.float64]
    expected_positions: NDArray[np.float64]
    actual_positions: NDArray[np.float64]
    mean_deviation: float
    mean_deviation_geodesic: float
    deviation_significance: float
    tau: float

    def is_intentional(self, threshold: float = 2.0) -> bool:
        """Check if behavior shows significant intentionality.

        Parameters
        ----------
        threshold : float, default=2.0
            Z-score threshold for significance.

        Returns
        -------
        bool
            True if deviation_significance > threshold.
        """
        return self.deviation_significance > threshold

    def summary(self) -> str:
        """Human-readable summary."""
        intent = "intentional" if self.is_intentional() else "random-like"
        return (
            f"Diffusion deviation: mean={self.mean_deviation:.2f}, "
            f"z={self.deviation_significance:.2f} ({intent})"
        )


@dataclass(frozen=True)
class DirectedDeviationResult:
    """Deviation decomposed by direction relative to goal.

    Attributes
    ----------
    deviation_toward_goal : NDArray[np.float64]
        Component of deviation in direction of goal, shape (n_samples-1,).
        Positive = deviating toward goal (shortcutting).
    deviation_orthogonal : NDArray[np.float64]
        Component orthogonal to goal direction, shape (n_samples-1,).
    mean_toward_goal : float
        Mean deviation toward goal. Positive = systematic approach.
    mean_orthogonal : float
        Mean orthogonal deviation. Should be ~0 for unbiased.
    goal_seeking_index : float
        Ratio of toward-goal to total deviation magnitude.
        Range [-1, 1]. Positive = goal-seeking, negative = avoiding.
    """

    deviation_toward_goal: NDArray[np.float64]
    deviation_orthogonal: NDArray[np.float64]
    mean_toward_goal: float
    mean_orthogonal: float
    goal_seeking_index: float

    def is_goal_seeking(self, threshold: float = 0.3) -> bool:
        """Check if deviation is systematically toward goal."""
        return self.goal_seeking_index > threshold

    def is_avoiding(self, threshold: float = -0.3) -> bool:
        """Check if deviation is systematically away from goal."""
        return self.goal_seeking_index < threshold
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `expected_next_position` | `(env, position_bin, tau) -> NDArray[np.float64]` | Diffusion-weighted mean of neighbors | `compute_diffusion_kernels()` |
| `compute_diffusion_deviation` | `(env, positions, times, tau) -> DiffusionDeviationResult` | Full deviation analysis | `expected_next_position()` |
| `decompose_deviation_by_goal` | `(deviation, positions, goal) -> DirectedDeviationResult` | Project deviation onto goal direction | Vector projection |
| `null_deviation_distribution` | `(env, tau, n_samples) -> NDArray[np.float64]` | Bootstrap null distribution | Monte Carlo |

#### Implementation Notes

```python
def expected_next_position(
    env: Environment,
    position_bin: int,
    tau: float,
) -> NDArray[np.float64]:
    """Compute expected next position under diffusion dynamics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position_bin : int
        Current position bin index.
    tau : float
        Diffusion time scale.

    Returns
    -------
    NDArray[np.float64], shape (n_dims,)
        Expected position (weighted average of bin centers by transition prob).
    """
    # Get transition probabilities from current bin
    p = future_distribution(env, position_bin, tau)

    # Weighted average of bin centers
    expected = np.sum(p[:, np.newaxis] * env.bin_centers, axis=0)

    return expected


def compute_diffusion_deviation(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    tau: float = 1.0,
    *,
    compute_significance: bool = True,
    n_bootstrap: int = 1000,
) -> DiffusionDeviationResult:
    """Compare trajectory to diffusion baseline.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps.
    tau : float, default=1.0
        Diffusion time scale. Should match typical step duration.
    compute_significance : bool, default=True
        Whether to compute bootstrap significance test.
    n_bootstrap : int, default=1000
        Number of bootstrap samples for significance.

    Returns
    -------
    DiffusionDeviationResult
        Deviation analysis results.

    Notes
    -----
    Performance: O(n_steps * n_bins^2) due to kernel computation.
    For large environments (>1000 bins), consider caching the kernel.
    """
    from neurospatial.distance import geodesic_distance_matrix

    n_steps = len(positions) - 1
    expected = np.zeros((n_steps, positions.shape[1]))
    actual = positions[1:]

    # Compute expected positions
    bins = env.bin_at(positions[:-1])
    for i, pos_bin in enumerate(bins):
        expected[i] = expected_next_position(env, pos_bin, tau)

    # Euclidean deviation
    deviation = np.linalg.norm(actual - expected, axis=1)

    # Geodesic deviation - use precomputed distance matrix for efficiency
    actual_bins = env.bin_at(actual)
    expected_bins = env.bin_at(expected)

    # Compute full geodesic distance matrix once (O(V^2 log V) via Dijkstra)
    # This is more efficient than V separate single-source queries
    dist_matrix = geodesic_distance_matrix(env.connectivity)

    # Extract geodesic distances between actual and expected bins
    deviation_geo = np.array([
        dist_matrix[int(a), int(e)]
        for a, e in zip(actual_bins, expected_bins)
    ])

    mean_dev = float(np.mean(deviation))
    mean_dev_geo = float(np.nanmean(deviation_geo))

    # Significance test via bootstrap
    if compute_significance:
        null_dist = null_deviation_distribution(env, tau, n_bootstrap)
        z_score = (mean_dev - np.mean(null_dist)) / np.std(null_dist)
    else:
        z_score = np.nan

    return DiffusionDeviationResult(
        deviation=deviation,
        deviation_geodesic=deviation_geo,
        expected_positions=expected,
        actual_positions=actual,
        mean_deviation=mean_dev,
        mean_deviation_geodesic=mean_dev_geo,
        deviation_significance=float(z_score),
        tau=tau,
    )
```

---

### Module 4: Geodesic VTE Extensions

**File**: `src/neurospatial/metrics/vte.py` (extends BEHAV_PLAN.md module)

#### Additional Docstring Section

```python
"""
... (existing VTE docstring from BEHAV_PLAN.md)

Geodesic VTE Extensions
-----------------------
This module also provides topology-aware VTE metrics that go beyond
simple head sweep angles to measure deliberation over geodesically
plausible future paths.

- **Geodesic goal directions**: Optimal path direction to each goal
- **Geodesic VTE score**: Head sweeps aligned to geodesic path options
- **Visibility-driven VTE**: Head sweeps toward newly visible passages
- **Branch sampling**: Fraction of time oriented toward each path branch

Example (Geodesic VTE)
----------------------
>>> from neurospatial.metrics import compute_geodesic_vte
>>> result = compute_geodesic_vte(
...     env, positions, headings, times,
...     goal_bins=[goal_left_bin, goal_right_bin],
...     pre_window=1.0,
... )
>>> print(f"Geodesic VTE: {result.geodesic_vte_score:.2f}")
>>> for goal_idx, sampling in enumerate(result.branch_sampling):
...     print(f"  Goal {goal_idx}: {sampling:.1%} sampling")

References
----------
.. [4] (New) Topology-aware VTE is a novel extension not yet in literature.
"""
```

#### Additional Data Structures

```python
@dataclass(frozen=True)
class GeodesicGoalDirections:
    """Geodesic optimal directions to each goal.

    Attributes
    ----------
    goal_directions : NDArray[np.float64]
        Allocentric direction to each goal via geodesic path,
        shape (n_goals,). In radians, [-π, π].
    goal_distances : NDArray[np.float64]
        Geodesic distance to each goal, shape (n_goals,).
    goal_bins : NDArray[np.int_]
        Bin indices of goals, shape (n_goals,).
    position_bin : int
        Current position bin.
    """

    goal_directions: NDArray[np.float64]
    goal_distances: NDArray[np.float64]
    goal_bins: NDArray[np.int_]
    position_bin: int

    def nearest_goal(self) -> int:
        """Index of nearest goal by geodesic distance."""
        return int(np.argmin(self.goal_distances))


@dataclass(frozen=True)
class GeodesicVTEResult:
    """Geodesic (topology-aware) VTE analysis.

    Attributes
    ----------
    geodesic_vte_score : float
        Sum of |heading change| weighted by geodesic alignment.
        High values indicate deliberation over geodesic paths.
    standard_vte_score : float
        Standard IdPhi for comparison.
    branch_sampling : NDArray[np.float64]
        Fraction of time oriented toward each goal branch, shape (n_goals,).
    goal_directions_over_time : NDArray[np.float64]
        Geodesic direction to each goal at each timepoint,
        shape (n_samples, n_goals).
    heading_goal_alignment : NDArray[np.float64]
        Cosine similarity between heading and each goal direction,
        shape (n_samples, n_goals).
    mean_alignment_variance : float
        Variance of alignment across goals. High = considering multiple options.
    """

    geodesic_vte_score: float
    standard_vte_score: float
    branch_sampling: NDArray[np.float64]
    goal_directions_over_time: NDArray[np.float64]
    heading_goal_alignment: NDArray[np.float64]
    mean_alignment_variance: float

    def is_deliberating(
        self,
        vte_threshold: float = 1.0,
        variance_threshold: float = 0.3,
    ) -> bool:
        """Check if metrics indicate deliberation over multiple paths.

        Parameters
        ----------
        vte_threshold : float, default=1.0
            Minimum geodesic VTE score.
        variance_threshold : float, default=0.3
            Minimum alignment variance (considering multiple options).

        Returns
        -------
        bool
            True if both thresholds exceeded.
        """
        return (
            self.geodesic_vte_score > vte_threshold
            and self.mean_alignment_variance > variance_threshold
        )

    def dominant_branch(self) -> int:
        """Index of most-sampled branch."""
        return int(np.argmax(self.branch_sampling))

    def summary(self) -> str:
        """Human-readable summary."""
        deliberating = "deliberating" if self.is_deliberating() else "committed"
        return (
            f"Geodesic VTE: score={self.geodesic_vte_score:.2f}, "
            f"variance={self.mean_alignment_variance:.2f} ({deliberating}), "
            f"dominant_branch={self.dominant_branch()}"
        )


@dataclass(frozen=True)
class VisibilityVTEResult:
    """Visibility-driven VTE analysis.

    Attributes
    ----------
    visibility_vte_score : float
        Head sweeps weighted by newly-visible passage detection.
    newly_visible_passages : list[tuple[float, int]]
        (time, passage_bin) for each newly visible passage.
    head_sweep_to_passage : NDArray[np.float64]
        For each newly visible passage, magnitude of head sweep toward it.
    passage_sampling_fraction : float
        Fraction of newly visible passages that were looked at.
    """

    visibility_vte_score: float
    newly_visible_passages: list[tuple[float, int]]
    head_sweep_to_passage: NDArray[np.float64]
    passage_sampling_fraction: float

    @property
    def n_passages_detected(self) -> int:
        """Number of newly visible passages."""
        return len(self.newly_visible_passages)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Visibility VTE: score={self.visibility_vte_score:.2f}, "
            f"{self.n_passages_detected} passages, "
            f"{self.passage_sampling_fraction:.1%} sampled"
        )
```

#### Additional Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `compute_geodesic_goal_directions` | `(env, position_bin, goal_bins) -> GeodesicGoalDirections` | Geodesic-optimal direction to each goal | `distance_field()`, gradient |
| `compute_geodesic_vte` | `(env, positions, headings, times, goal_bins, ...) -> GeodesicVTEResult` | Topology-aware VTE | `compute_geodesic_goal_directions()` |
| `compute_visibility_vte` | `(env, positions, headings, times, ...) -> VisibilityVTEResult` | Visibility-driven VTE | `compute_viewshed()` |
| `branch_sampling_fraction` | `(headings, goal_directions, threshold) -> NDArray[np.float64]` | Time fraction oriented toward each branch | Cosine similarity |

#### Implementation Notes

```python
def compute_geodesic_goal_directions(
    env: Environment,
    position_bin: int,
    goal_bins: NDArray[np.int_] | list[int],
) -> GeodesicGoalDirections:
    """Compute geodesic-optimal direction to each goal from position.

    The geodesic direction is the direction of the first step on the
    shortest path to each goal.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    position_bin : int
        Current position bin.
    goal_bins : array-like of int
        Bin indices of goals.

    Returns
    -------
    GeodesicGoalDirections
        Directions and distances to each goal.
    """
    from neurospatial.distance import distance_field
    import networkx as nx

    goal_bins = np.asarray(goal_bins)
    n_goals = len(goal_bins)

    directions = np.zeros(n_goals)
    distances = np.zeros(n_goals)

    G = env.connectivity
    pos = env.bin_centers[position_bin]

    for i, goal_bin in enumerate(goal_bins):
        # Compute distance field from goal
        dist_field = distance_field(G, [int(goal_bin)], metric="geodesic")
        distances[i] = dist_field[position_bin]

        if np.isinf(distances[i]):
            # Unreachable goal
            directions[i] = np.nan
            continue

        # Find neighbor with minimum distance to goal (first step of shortest path)
        neighbors = list(G.neighbors(position_bin))
        if not neighbors:
            directions[i] = np.nan
            continue

        neighbor_dists = [dist_field[n] for n in neighbors]
        best_neighbor = neighbors[np.argmin(neighbor_dists)]

        # Direction to best neighbor
        neighbor_pos = env.bin_centers[best_neighbor]
        delta = neighbor_pos - pos
        directions[i] = np.arctan2(delta[1], delta[0])

    return GeodesicGoalDirections(
        goal_directions=directions,
        goal_distances=distances,
        goal_bins=goal_bins,
        position_bin=position_bin,
    )


def compute_geodesic_vte(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    times: NDArray[np.float64],
    goal_bins: NDArray[np.int_] | list[int],
    *,
    alignment_threshold: float = 0.5,
) -> GeodesicVTEResult:
    """Compute topology-aware VTE metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    headings : NDArray[np.float64], shape (n_samples,)
        Head direction at each timepoint (radians).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps.
    goal_bins : array-like of int
        Bin indices of potential goals.
    alignment_threshold : float, default=0.5
        Cosine threshold for counting as "oriented toward" a branch.

    Returns
    -------
    GeodesicVTEResult
        Geodesic VTE analysis.
    """
    goal_bins = np.asarray(goal_bins)
    n_samples = len(positions)
    n_goals = len(goal_bins)

    # Compute geodesic directions at each timepoint
    trajectory_bins = env.bin_at(positions)
    goal_dirs = np.zeros((n_samples, n_goals))

    for t in range(n_samples):
        gd = compute_geodesic_goal_directions(env, trajectory_bins[t], goal_bins)
        goal_dirs[t] = gd.goal_directions

    # Compute heading-goal alignment (cosine similarity)
    alignment = np.cos(headings[:, np.newaxis] - goal_dirs)

    # Branch sampling: fraction of time oriented toward each goal
    oriented = alignment > alignment_threshold
    branch_sampling = np.mean(oriented, axis=0)

    # Geodesic VTE score: head sweep magnitude weighted by alignment variance
    from neurospatial.metrics.vte import head_sweep_magnitude, wrap_angle

    head_changes = np.abs(wrap_angle(np.diff(headings)))

    # Weight by alignment variance at each step (considering multiple options)
    alignment_variance = np.var(alignment[:-1], axis=1)
    geodesic_vte = float(np.sum(head_changes * alignment_variance))

    # Standard VTE for comparison
    standard_vte = head_sweep_magnitude(headings)

    return GeodesicVTEResult(
        geodesic_vte_score=geodesic_vte,
        standard_vte_score=standard_vte,
        branch_sampling=branch_sampling,
        goal_directions_over_time=goal_dirs,
        heading_goal_alignment=alignment,
        mean_alignment_variance=float(np.mean(alignment_variance)),
    )
```

---

## File Structure

```
src/neurospatial/metrics/
├── __init__.py                    # Update exports
├── trajectory.py                  # EXISTING (no changes)
├── circular.py                    # EXISTING (no changes)
├── path_efficiency.py             # FROM BEHAV_PLAN.md
├── goal_directed.py               # FROM BEHAV_PLAN.md
├── decision_analysis.py           # FROM BEHAV_PLAN.md
├── vte.py                         # FROM BEHAV_PLAN.md + EXTENSIONS (this plan)
├── reachability.py                # NEW (this plan)
├── manifold.py                    # NEW (this plan)
└── diffusion_baseline.py          # NEW (this plan)

tests/metrics/
├── test_path_efficiency.py        # FROM BEHAV_PLAN.md
├── test_goal_directed.py          # FROM BEHAV_PLAN.md
├── test_decision_analysis.py      # FROM BEHAV_PLAN.md
├── test_vte.py                    # FROM BEHAV_PLAN.md + EXTENSIONS
├── test_reachability.py           # NEW (this plan)
├── test_manifold.py               # NEW (this plan)
└── test_diffusion_baseline.py     # NEW (this plan)
```

---

## Dependencies Between Modules

### External Dependencies (already exist)

```
kernels.py
    └── compute_diffusion_kernels()  # Graph Laplacian + matrix exp

differential.py
    └── compute_differential_operator()  # For gradient computation

distance.py
    └── distance_field(), geodesic_distance_matrix()
    NOTE: neighbors_within() does NOT exist - use distance_field() + threshold

visibility.py
    └── compute_viewshed(), FieldOfView

reference_frames.py
    └── heading_from_velocity(), compute_egocentric_bearing()
```

### Internal Dependencies (this plan)

```
reachability.py
    └── uses: compute_diffusion_kernels() from kernels
    └── uses: distance_field() from distance (NOT neighbors_within - see note above)
    └── uses: scipy.stats.entropy for information metrics

manifold.py
    └── uses: scipy.sparse.linalg.eigsh() for Laplacian eigenfunctions
    └── uses: networkx.laplacian_matrix(), networkx.is_connected()
    └── NOTE: Could potentially reuse differential_operator for Laplacian

diffusion_baseline.py
    └── uses: reachability.future_distribution()
    └── uses: geodesic_distance_matrix() from distance (precomputed for efficiency)

vte.py (extensions)
    └── uses: distance_field() from distance
    └── uses: compute_viewshed() from visibility
    └── uses: wrap_angle(), head_sweep_magnitude() from vte.py (BEHAV_PLAN.md)
```

### Dependency on BEHAV_PLAN.md Modules

```
This plan DEPENDS ON:
├── vte.py (BEHAV_PLAN.md)
│   └── wrap_angle()
│   └── head_sweep_magnitude()
│   └── VTETrialResult, VTESessionResult
│
├── decision_analysis.py (BEHAV_PLAN.md)
│   └── geodesic_voronoi_labels()  # For branch detection
│   └── extract_pre_decision_window()
│
└── path_efficiency.py (BEHAV_PLAN.md)
    └── traveled_path_length()  # For comparison metrics
```

---

## Implementation Order

### Phase 1: Reachability Metrics (Foundation)

**Prerequisite**: BEHAV_PLAN.md modules NOT required

- [ ] Create `metrics/reachability.py` with module docstring
- [ ] Implement `ReachabilityMetrics` dataclass with helper methods
- [ ] Implement `reachable_set()` using `distance_field()` + threshold
- [ ] Implement `future_distribution()` using `compute_diffusion_kernels()`
- [ ] Implement `future_entropy()` using `scipy.stats.entropy()`
- [ ] Implement `count_geodesic_options()` using shortest path analysis
- [ ] Implement `compute_reachability_metrics()` combining all
- [ ] Implement `ReachabilityTrajectory` and `compute_reachability_trajectory()`
- [ ] Add comprehensive error messages
- [ ] Write tests in `tests/metrics/test_reachability.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 2: Manifold Embedding

**Prerequisite**: Phase 1 (uses similar patterns)

- [ ] Create `metrics/manifold.py` with module docstring
- [ ] Implement `compute_laplacian_eigenfunctions()` using `scipy.sparse.linalg.eigsh()`
- [ ] Implement `ManifoldEmbedding` dataclass with helper methods
- [ ] Implement `compute_manifold_embedding()` with diffusion time scaling
- [ ] Implement `manifold_velocity()` and `manifold_curvature()`
- [ ] Implement `manifold_path_efficiency()`
- [ ] Implement `ManifoldTrajectoryResult` and `manifold_trajectory()`
- [ ] Write tests in `tests/metrics/test_manifold.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 3: Diffusion Baseline

**Prerequisite**: Phase 1 (`future_distribution()`)

- [ ] Create `metrics/diffusion_baseline.py` with module docstring
- [ ] Implement `expected_next_position()` using `future_distribution()`
- [ ] Implement `DiffusionDeviationResult` dataclass
- [ ] Implement `compute_diffusion_deviation()` with significance testing
- [ ] Implement `null_deviation_distribution()` for bootstrap
- [ ] Implement `DirectedDeviationResult` and `decompose_deviation_by_goal()`
- [ ] Write tests in `tests/metrics/test_diffusion_baseline.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 4: Geodesic VTE Extensions

**Prerequisite**: BEHAV_PLAN.md vte.py module complete

- [ ] Add geodesic VTE docstring section to `vte.py`
- [ ] Implement `GeodesicGoalDirections` dataclass
- [ ] Implement `compute_geodesic_goal_directions()` using `distance_field()`
- [ ] Implement `GeodesicVTEResult` dataclass with helper methods
- [ ] Implement `compute_geodesic_vte()` with branch sampling
- [ ] Implement `VisibilityVTEResult` dataclass
- [ ] Implement `compute_visibility_vte()` using `compute_viewshed()`
- [ ] Implement `branch_sampling_fraction()`
- [ ] Write tests for geodesic VTE in `tests/metrics/test_vte.py`
- [ ] Update exports in `metrics/__init__.py`

### Phase 5: Integration

- [ ] Update `.claude/QUICKSTART.md` with advanced metrics examples
- [ ] Update `.claude/API_REFERENCE.md` with new imports
- [ ] Run full test suite: `uv run pytest tests/metrics/`
- [ ] Run type checker: `uv run mypy src/neurospatial/metrics/`
- [ ] Run linter: `uv run ruff check src/neurospatial/metrics/`

---

## Testing Strategy

### Unit Tests

Each function tested independently with:

- Basic functionality with known inputs/outputs
- Edge cases (disconnected graphs, single bin, uniform distribution)
- Parameter validation (error messages are helpful)

### Integration Tests

- Cross-module consistency: reachability uses kernels correctly
- Manifold embedding preserves graph structure
- Diffusion baseline matches kernel computation
- Geodesic VTE consistent with standard VTE

### Regression Tests

**Reachability**:

- Corridor: low entropy, few geodesic options (≤2)
- Junction: high entropy, many geodesic options (≥3)
- Uniform grid: entropy ≈ log2(reachable bins)

**Manifold**:

- Straight Euclidean path in open field: straight manifold path
- Path around obstacle: shorter in manifold than Euclidean
- Disconnected graph: eigenvector structure reflects components

**Diffusion Baseline**:

- Random walk: deviation ≈ 0 (matches null)
- Direct path to goal: positive deviation toward goal
- Avoidance: negative deviation away from region

**Geodesic VTE**:

- High standard VTE + aligned with geodesic paths → high geodesic VTE
- Random head sweeps not aligned with paths → low geodesic VTE
- Equal sampling of branches → high alignment variance

---

## API Design Principles

1. **Consistent Signatures**: All main functions take `(env, positions, times, ...)` as first args
2. **Keyword-Only Options**: All optional parameters are keyword-only after positional
3. **Metric Parameter**: Functions supporting geodesic/Euclidean use `metric=`
4. **Return Dataclasses**: Complex results return frozen dataclasses with helper methods
5. **NumPy Docstrings**: All functions have complete NumPy-format docstrings
6. **Vectorized**: No loops over timepoints where possible; all operations vectorized
7. **Helpful Errors**: Error messages explain what's wrong AND how to fix it
8. **Builds on Existing**: Reuse existing kernels, distance, visibility modules

---

## Estimated Effort

| Module | New LOC | Reused Functions | Test LOC |
|--------|---------|------------------|----------|
| `reachability.py` | ~350 | 4 | ~300 |
| `manifold.py` | ~400 | 3 | ~350 |
| `diffusion_baseline.py` | ~350 | 4 | ~300 |
| `vte.py` extensions | ~400 | 5 | ~350 |
| **Total** | **~1,500** | **16** | **~1,300** |

---

## References

### Mathematical Framework

- **Diffusion maps**: Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
  Applied and Computational Harmonic Analysis, 21(1), 5-30.
  - Used in: `compute_manifold_embedding()`, `future_distribution()`

- **Spectral clustering**: Nadler, B., et al. (2006). Diffusion maps, spectral
  clustering and reaction coordinates of dynamical systems.
  - Used in: `compute_laplacian_eigenfunctions()`

- **Hippocampal maps**: Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J.
  (2017). The hippocampus as a predictive map. Nature Neuroscience.
  - Motivation for manifold analysis of behavior

- **VTE**: Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
  - Extended with geodesic-aware metrics

### Existing Code

- `neurospatial.kernels`: Diffusion kernel computation
- `neurospatial.differential`: Laplacian via differential operator
- `neurospatial.distance`: Geodesic distance functions
- `neurospatial.visibility`: Viewshed computation
- `neurospatial.reference_frames`: Heading and egocentric transforms

---

## Summary: Why These Metrics Are Novel

Traditional behavioral analysis uses:

- Euclidean distances (ignores walls)
- 2D trajectories (ignores topology)
- Heuristic VTE (left/right head sweeps)
- Hand-defined decision regions

This plan enables:

- **Graph-aware distances** respecting environment structure
- **Manifold embeddings** reflecting cognitive-map geometry
- **Diffusion baselines** providing principled null models
- **Geodesic VTE** measuring deliberation over actual path options
- **Visibility-driven VTE** connecting head sweeps to information gathering

These metrics operate in the **space the animal actually experiences**, enabling quantitative tests of navigation theories grounded in optimal control, random walks, information theory, and manifold geometry.
