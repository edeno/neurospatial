# Advanced Behavioral Metrics: Implementation Tasks

**Source**: [BEHAV_PLAN2.md](BEHAV_PLAN2.md)
**Created**: 2025-12-05
**Revised**: 2025-12-05 (post code-review and UX-review)
**Status**: Ready for implementation (after BEHAV_PLAN.md)

---

## Overview

Implementation of advanced behavioral analysis metrics leveraging neurospatial's unique capabilities: environment geometry, connectivity graphs, Laplacian operators, diffusion kernels, and visibility modeling.

**Modules to implement**:

1. Reachability & Information Theory (`reachability.py`)
2. Manifold Embedding (`manifold.py`)
3. Diffusion Baseline (`diffusion_baseline.py`)
4. Geodesic VTE Extensions (`vte.py` additions)

**Estimated LOC**: ~1,500 new + ~1,300 tests

---

## Prerequisites

**MUST complete BEHAV_PLAN.md first** (at minimum: `vte.py` module)

```
BEHAV_PLAN.md modules required:
├── vte.py (required for Phase 4)
│   └── wrap_angle()
│   └── head_sweep_magnitude()
│
├── decision_analysis.py (optional, helpful for context)
│
└── path_efficiency.py (optional, for comparison)
```

---

## Dependencies

```
reachability.py (no internal dependencies)
    │
    └──► manifold.py (can be parallel with reachability)
    │
    └──► diffusion_baseline.py (depends on reachability.future_distribution)
              │
              └──► vte.py extensions (depends on BEHAV_PLAN.md vte.py)
```

**External dependencies** (already exist):

- `neurospatial.kernels`: `compute_diffusion_kernels()`
- `neurospatial.differential`: `compute_differential_operator()`
- `neurospatial.distance`: `distance_field()`, `geodesic_distance_matrix()`
  - **NOTE**: `neighbors_within()` does NOT exist - use `distance_field()` + threshold instead
- `neurospatial.visibility`: `compute_viewshed()`, `FieldOfView`
- `neurospatial.reference_frames`: `heading_from_velocity()`
- `scipy.sparse.linalg`: `eigsh()` for Laplacian eigenfunctions
- `scipy.stats`: `entropy()` for information metrics
- `networkx`: `laplacian_matrix()`, `is_connected()` for graph operations

---

## Milestone 1: Reachability & Information Theory

**Goal**: Quantify future action space and navigational uncertainty.

**File**: `src/neurospatial/metrics/reachability.py`

**Priority**: HIGH (foundation for diffusion baseline)

### M1.1: Module Setup

- [ ] Create `reachability.py` with module docstring
- [ ] Add imports: numpy, typing, dataclasses, scipy.stats
- [ ] Add internal imports from `neurospatial.kernels`, `neurospatial.distance`

### M1.2: Data Structures

- [ ] Implement `ReachabilityMetrics` frozen dataclass
  - [ ] Fields (input parameters first):
    - [ ] `position_bin`, `radius`, `tau` (inputs)
    - [ ] `reachable_bins`, `n_reachable`, `reachable_fraction` (primary outputs)
    - [ ] `future_distribution`, `future_entropy`, `n_geodesic_options` (derived outputs)
  - [ ] Method: `is_decision_point(min_options=3) -> bool` (renamed from min_branches)
  - [ ] Method: `is_constrained(max_entropy=1.0) -> bool`
  - [ ] Method: `summary() -> str`
- [ ] Implement `ReachabilityTrajectory` frozen dataclass
  - [ ] Fields: `future_entropy`, `n_reachable`, `n_geodesic_options`, `mean_entropy`, `decision_point_indices`
  - [ ] Property: `n_decision_points`
  - [ ] Method: `summary() -> str`

### M1.3: Core Functions

- [ ] Implement `reachable_set(env, position_bin, radius) -> NDArray[np.int_]`
  - [ ] Use `distance_field()` from distance module + threshold
  - [ ] **NOTE**: `neighbors_within()` does NOT exist - see BEHAV_PLAN2.md for correct implementation
  - [ ] Return bin indices within geodesic radius
- [ ] Implement `future_distribution(env, position_bin, tau) -> NDArray[np.float64]`
  - [ ] Convert tau to bandwidth_sigma: `sqrt(2 * tau)`
  - [ ] Use `compute_diffusion_kernels()` directly from kernels module
  - [ ] **NOTE**: `env.compute_kernel()` does NOT exist - call kernels module directly
  - [ ] Return row of transition matrix for position_bin
- [ ] Implement `future_entropy(distribution) -> float`
  - [ ] Use `scipy.stats.entropy(p, base=2)` for bits
  - [ ] Filter zero probabilities to avoid log(0)
  - [ ] Handle delta functions: return 0.0 if max(p) > 0.999
- [ ] Implement `count_geodesic_options(env, position_bin, radius) -> int` (renamed from count_branches)
  - [ ] Find distinct first-step neighbors on shortest paths to reachable bins
  - [ ] Use `distance_field()` from position and from each neighbor
  - [ ] Count neighbors that lie on at least one shortest path

### M1.4: Composite Functions

- [ ] Implement `compute_reachability_metrics(env, position, tau, radius) -> ReachabilityMetrics`
  - [ ] Combine all core functions
  - [ ] Map position to bin via `env.bin_at()`
- [ ] Implement `compute_reachability_trajectory(env, positions, tau, radius, ...) -> ReachabilityTrajectory`
  - [ ] Loop over trajectory (or vectorize where possible)
  - [ ] Compute decision point indices based on branch threshold

### M1.5: Error Handling

- [ ] Add helpful error messages:
  - [ ] Environment not fitted
  - [ ] Position outside environment
  - [ ] Invalid tau (must be positive)
  - [ ] Invalid radius (must be positive)

### M1.6: Tests

**File**: `tests/metrics/test_reachability.py`

- [ ] Test `reachable_set()` returns correct bins
- [ ] Test `future_distribution()` sums to 1.0
- [ ] Test `future_entropy()` with uniform distribution (should be log2(n))
- [ ] Test `future_entropy()` with delta distribution (should be 0)
- [ ] Test `count_geodesic_options()` at corridor (should be 1-2)
- [ ] Test `count_geodesic_options()` at T-junction (should be 3)
- [ ] Test `count_geodesic_options()` differs from graph degree (see BEHAV_PLAN2.md)
- [ ] Test `is_decision_point()` threshold logic
- [ ] Test `is_constrained()` threshold logic
- [ ] Test error messages are helpful

### M1.7: Exports

- [ ] Add to `src/neurospatial/metrics/__init__.py`:
  - [ ] `ReachabilityMetrics`
  - [ ] `ReachabilityTrajectory`
  - [ ] `reachable_set`
  - [ ] `future_distribution`
  - [ ] `future_entropy`
  - [ ] `count_geodesic_options` (renamed from count_branches)
  - [ ] `compute_reachability_metrics`
  - [ ] `compute_reachability_trajectory`

**Success criteria**:

- [ ] Corridor position: entropy < 1.5 bits, geodesic_options ≤ 2
- [ ] Junction position: entropy > 2.0 bits, geodesic_options ≥ 3
- [ ] All tests pass: `uv run pytest tests/metrics/test_reachability.py -v`

---

## Milestone 2: Manifold Embedding

**Goal**: Embed environment into cognitive-map coordinates for trajectory analysis.

**File**: `src/neurospatial/metrics/manifold.py`

**Priority**: HIGH

### M2.1: Module Setup

- [ ] Create `manifold.py` with module docstring
- [ ] Add imports: scipy.sparse.linalg, networkx, numpy
- [ ] Reference Coifman & Lafon (2006) in docstring

### M2.2: Data Structures

- [ ] Implement `ManifoldEmbedding` frozen dataclass
  - [ ] Fields: `coordinates`, `eigenvalues`, `eigenvectors`, `n_components`, `tau`
  - [ ] Method: `embed_positions(positions, env) -> NDArray`
  - [ ] Method: `manifold_distance(bin_i, bin_j) -> float`
- [ ] Implement `ManifoldTrajectoryResult` frozen dataclass
  - [ ] Fields: `manifold_positions`, `manifold_velocity`, `manifold_speed`, `manifold_curvature`, `path_efficiency`, `total_manifold_distance`, `net_manifold_displacement`
  - [ ] Method: `is_straight(threshold=0.8) -> bool`
  - [ ] Method: `summary() -> str`

### M2.3: Core Functions

- [ ] Implement `compute_laplacian_eigenfunctions(env, n_components=6) -> tuple[eigenvalues, eigenvectors]`
  - [ ] Use `nx.laplacian_matrix()` to get sparse Laplacian
  - [ ] Use `scipy.sparse.linalg.eigsh(..., which='SM')` for smallest eigenvalues
  - [ ] Sort by eigenvalue (eigsh doesn't guarantee order)
  - [ ] **Fix sign convention**: ensure first non-zero element is positive for reproducibility
  - [ ] **Check connectivity**: raise RuntimeError if graph is disconnected
  - [ ] **Add performance warning**: ResourceWarning for >1000 bins
- [ ] Implement `compute_manifold_embedding(env, n_components=3, tau=1.0, skip_first=True) -> ManifoldEmbedding`
  - [ ] Apply diffusion time scaling: `exp(-tau * eigenvalues)`
  - [ ] Skip first (constant) eigenvector if `skip_first=True`
- [ ] Implement `manifold_velocity(manifold_positions, dt) -> NDArray`
  - [ ] `np.diff(positions, axis=0) / dt`
- [ ] Implement `manifold_curvature(manifold_positions) -> NDArray`
  - [ ] Discrete curvature: `|dv| / |v|`
  - [ ] **Numerical stability**: use relative threshold (1% of median speed)
  - [ ] Return NaN for low-speed regions where curvature is ill-defined

### M2.4: Composite Functions

- [ ] Implement `manifold_path_efficiency(manifold_positions) -> float`
  - [ ] `net_displacement / total_path_length`
  - [ ] Return NaN if total_path_length ≈ 0
- [ ] Implement `manifold_trajectory(env, positions, embedding, times=None) -> ManifoldTrajectoryResult`
  - [ ] Embed trajectory via `embedding.embed_positions()`
  - [ ] Compute velocity, curvature, efficiency

### M2.5: Tests

**File**: `tests/metrics/test_manifold.py`

- [ ] Test `compute_laplacian_eigenfunctions()` returns sorted eigenvalues
- [ ] Test first eigenvalue ≈ 0 for connected graph
- [ ] Test `compute_laplacian_eigenfunctions()` raises RuntimeError for disconnected graph
- [ ] Test eigenvector sign convention is deterministic (run twice, compare)
- [ ] Test `compute_manifold_embedding()` has correct shape
- [ ] Test `embed_positions()` maps positions to manifold
- [ ] Test `manifold_distance()` is non-negative
- [ ] Test `manifold_curvature()` returns NaN for stationary positions
- [ ] Test `manifold_path_efficiency()` is 1.0 for straight path
- [ ] Test `is_straight()` threshold logic

### M2.6: Exports

- [ ] Add to `src/neurospatial/metrics/__init__.py`:
  - [ ] `ManifoldEmbedding`
  - [ ] `ManifoldTrajectoryResult`
  - [ ] `compute_laplacian_eigenfunctions`
  - [ ] `compute_manifold_embedding`
  - [ ] `manifold_velocity`
  - [ ] `manifold_curvature`
  - [ ] `manifold_path_efficiency`
  - [ ] `manifold_trajectory`

**Success criteria**:

- [ ] Eigenvalues sorted ascending, first ≈ 0
- [ ] Straight Euclidean path: efficiency > 0.9
- [ ] All tests pass: `uv run pytest tests/metrics/test_manifold.py -v`

---

## Milestone 3: Diffusion Baseline

**Goal**: Compare trajectories against random walk null model.

**File**: `src/neurospatial/metrics/diffusion_baseline.py`

**Priority**: MEDIUM

**Dependencies**: M1 (`future_distribution()`)

### M3.1: Module Setup

- [ ] Create `diffusion_baseline.py` with module docstring
- [ ] Add imports from `neurospatial.metrics.reachability`, `neurospatial.distance`

### M3.2: Data Structures

- [ ] Implement `DiffusionDeviationResult` frozen dataclass
  - [ ] Fields: `deviation`, `deviation_geodesic`, `expected_positions`, `actual_positions`, `mean_deviation`, `mean_deviation_geodesic`, `deviation_significance`, `tau`
  - [ ] Method: `is_intentional(threshold=2.0) -> bool`
  - [ ] Method: `summary() -> str`
- [ ] Implement `DirectedDeviationResult` frozen dataclass
  - [ ] Fields: `deviation_toward_goal`, `deviation_orthogonal`, `mean_toward_goal`, `mean_orthogonal`, `goal_seeking_index`
  - [ ] Method: `is_goal_seeking(threshold=0.3) -> bool`
  - [ ] Method: `is_avoiding(threshold=-0.3) -> bool`

### M3.3: Core Functions

- [ ] Implement `expected_next_position(env, position_bin, tau) -> NDArray[np.float64]`
  - [ ] Use `future_distribution()` from reachability
  - [ ] Weighted average of bin centers by transition probability
- [ ] Implement `null_deviation_distribution(env, tau, n_samples=1000) -> NDArray[np.float64]`
  - [ ] Monte Carlo bootstrap of random walk deviations
  - [ ] Return distribution of mean deviations
- [ ] **Performance optimization**: Use `geodesic_distance_matrix()` (precomputed once)
  - [ ] More efficient than repeated `geodesic_distance_between_points()` calls

### M3.4: Composite Functions

- [ ] Implement `compute_diffusion_deviation(env, positions, times, tau, ...) -> DiffusionDeviationResult`
  - [ ] Compute expected positions at each step
  - [ ] Compute Euclidean and geodesic deviations
  - [ ] Bootstrap significance test (optional)
- [ ] Implement `decompose_deviation_by_goal(deviation, positions, goal) -> DirectedDeviationResult`
  - [ ] Project deviation onto goal direction
  - [ ] Compute toward-goal and orthogonal components
  - [ ] Compute goal-seeking index

### M3.5: Tests

**File**: `tests/metrics/test_diffusion_baseline.py`

- [ ] Test `expected_next_position()` is weighted average
- [ ] Test `compute_diffusion_deviation()` with random walk (deviation ≈ 0)
- [ ] Test `compute_diffusion_deviation()` with direct path (deviation > 0)
- [ ] Test `decompose_deviation_by_goal()` sign conventions
- [ ] Test `is_intentional()` threshold logic
- [ ] Test `is_goal_seeking()` and `is_avoiding()` logic

### M3.6: Exports

- [ ] Add to `src/neurospatial/metrics/__init__.py`:
  - [ ] `DiffusionDeviationResult`
  - [ ] `DirectedDeviationResult`
  - [ ] `expected_next_position`
  - [ ] `null_deviation_distribution`
  - [ ] `compute_diffusion_deviation`
  - [ ] `decompose_deviation_by_goal`

**Success criteria**:

- [ ] Random walk: `deviation_significance < 1.0`
- [ ] Goal-directed: `deviation_significance > 2.0`
- [ ] All tests pass: `uv run pytest tests/metrics/test_diffusion_baseline.py -v`

---

## Milestone 4: Geodesic VTE Extensions

**Goal**: Extend VTE metrics with topology-aware deliberation measures.

**File**: `src/neurospatial/metrics/vte.py` (extends BEHAV_PLAN.md module)

**Priority**: MEDIUM

**Dependencies**: BEHAV_PLAN.md `vte.py` module complete

### M4.1: Module Updates

- [ ] Add geodesic VTE section to `vte.py` docstring
- [ ] Add imports from `neurospatial.distance`, `neurospatial.visibility`
- [ ] Document new concepts: geodesic goal directions, branch sampling

### M4.2: Data Structures

- [ ] Implement `GeodesicGoalDirections` frozen dataclass
  - [ ] Fields: `goal_directions`, `goal_distances`, `goal_bins`, `position_bin`
  - [ ] Method: `nearest_goal() -> int`
- [ ] Implement `GeodesicVTEResult` frozen dataclass
  - [ ] Fields: `geodesic_vte_score`, `standard_vte_score`, `branch_sampling`, `goal_directions_over_time`, `heading_goal_alignment`, `mean_alignment_variance`
  - [ ] Method: `is_deliberating(vte_threshold=1.0, variance_threshold=0.3) -> bool`
  - [ ] Method: `dominant_branch() -> int`
  - [ ] Method: `summary() -> str`
- [ ] Implement `VisibilityVTEResult` frozen dataclass
  - [ ] Fields: `visibility_vte_score`, `newly_visible_passages`, `head_sweep_to_passage`, `passage_sampling_fraction`
  - [ ] Property: `n_passages_detected`
  - [ ] Method: `summary() -> str`

### M4.3: Core Functions

- [ ] Implement `compute_geodesic_goal_directions(env, position_bin, goal_bins) -> GeodesicGoalDirections`
  - [ ] Use `distance_field()` from each goal
  - [ ] Find first-step direction on shortest path
  - [ ] Handle unreachable goals (return NaN direction)
- [ ] Implement `branch_sampling_fraction(headings, goal_directions, threshold=0.5) -> NDArray[np.float64]`
  - [ ] Compute cosine similarity between heading and each goal direction
  - [ ] Return fraction of time oriented toward each branch

### M4.4: Composite Functions

- [ ] Implement `compute_geodesic_vte(env, positions, headings, times, goal_bins, ...) -> GeodesicVTEResult`
  - [ ] Compute geodesic directions at each timepoint
  - [ ] Compute heading-goal alignment
  - [ ] Weight head sweeps by alignment variance
  - [ ] Compare to standard VTE
- [ ] Implement `compute_visibility_vte(env, positions, headings, times, ...) -> VisibilityVTEResult`
  - [ ] Use `compute_viewshed()` to detect newly visible passages
  - [ ] Check if head sweeps align with newly visible passages
  - [ ] Compute passage sampling fraction

### M4.5: Tests

**File**: `tests/metrics/test_vte.py` (extend existing)

- [ ] Test `compute_geodesic_goal_directions()` at T-junction
- [ ] Test geodesic direction differs from Euclidean when obstacle present
- [ ] Test `branch_sampling_fraction()` with known orientations
- [ ] Test `compute_geodesic_vte()` weighting by alignment variance
- [ ] Test `is_deliberating()` threshold logic
- [ ] Test `compute_visibility_vte()` detects passages

### M4.6: Exports

- [ ] Add to `src/neurospatial/metrics/__init__.py`:
  - [ ] `GeodesicGoalDirections`
  - [ ] `GeodesicVTEResult`
  - [ ] `VisibilityVTEResult`
  - [ ] `compute_geodesic_goal_directions`
  - [ ] `branch_sampling_fraction`
  - [ ] `compute_geodesic_vte`
  - [ ] `compute_visibility_vte`

**Success criteria**:

- [ ] High standard VTE + aligned sweeps → high geodesic VTE
- [ ] Random sweeps not aligned → low geodesic VTE
- [ ] All tests pass: `uv run pytest tests/metrics/test_vte.py -v`

---

## Milestone 5: Integration and Documentation

**Goal**: Integrate all modules and update documentation.

**Priority**: LOW (after M1-M4 complete)

### M5.1: Full Test Suite

- [ ] Run all tests: `uv run pytest tests/metrics/ -v`
- [ ] Run type checker: `uv run mypy src/neurospatial/metrics/`
- [ ] Run linter: `uv run ruff check src/neurospatial/metrics/`
- [ ] Fix any issues

### M5.2: Integration Tests

- [ ] Test reachability uses kernels correctly
- [ ] Test manifold embedding preserves graph structure
- [ ] Test diffusion baseline matches kernel computation
- [ ] Test geodesic VTE consistent with standard VTE
- [ ] Test round-trip: simulated trajectory → metrics → verify expected values

### M5.3: Documentation Updates

- [ ] Update `.claude/QUICKSTART.md` with advanced metrics examples
  - [ ] Reachability/entropy example
  - [ ] Manifold embedding example
  - [ ] Diffusion deviation example
  - [ ] Geodesic VTE example
- [ ] Update `.claude/API_REFERENCE.md` with new imports
  - [ ] List all new functions and dataclasses
  - [ ] Group by module
- [ ] Update `.claude/ADVANCED.md` with theoretical background
  - [ ] Diffusion maps section
  - [ ] Information-theoretic behavior section

### M5.4: Final Validation

- [ ] All tests pass
- [ ] No type errors
- [ ] No linting errors
- [ ] Examples in docstrings work

**Success criteria**:

- [ ] `uv run pytest tests/metrics/ -v` all pass
- [ ] `uv run mypy src/neurospatial/metrics/` no errors
- [ ] `uv run ruff check src/neurospatial/metrics/` no errors

---

## Implementation Notes

### Critical Design Decisions

1. **Parameter naming**: Use `tau` for diffusion time scale (not `t` or `time`)

2. **Kernel computation**: Call `compute_diffusion_kernels()` directly from kernels module

   ```python
   from neurospatial.kernels import compute_diffusion_kernels
   # Convert tau to bandwidth_sigma
   bandwidth_sigma = np.sqrt(2.0 * tau)
   # NOTE: env.compute_kernel() does NOT exist - call kernels module directly
   kernel = compute_diffusion_kernels(
       env.connectivity, bandwidth_sigma=bandwidth_sigma, mode="transition"
   )
   ```

3. **Laplacian computation**: Use scipy's sparse eigensolver

   ```python
   from scipy.sparse.linalg import eigsh
   import networkx as nx
   L = nx.laplacian_matrix(env.connectivity).astype(np.float64)
   eigenvalues, eigenvectors = eigsh(L, k=n_components, which='SM')
   # Sort and fix sign convention - see BEHAV_PLAN2.md
   ```

4. **Reachable set computation**: Use `distance_field()` + threshold

   ```python
   from neurospatial.distance import distance_field
   # NOTE: neighbors_within() does NOT exist
   dist_field = distance_field(env.connectivity, [position_bin], metric="geodesic")
   reachable_bins = np.where(dist_field <= radius)[0]
   ```

5. **Geodesic distance matrix**: Precompute once for efficiency

   ```python
   from neurospatial.distance import geodesic_distance_matrix
   # More efficient than repeated single-source queries
   dist_matrix = geodesic_distance_matrix(env.connectivity)
   ```

6. **Edge case handling**:
   - Disconnected graph: raise RuntimeError (not just NaN) for Laplacian
   - Zero velocity: return NaN for curvature (relative threshold)
   - Zero standard deviation: warn and return NaN significance
   - Delta distribution: return 0.0 entropy

### Import Patterns

```python
# Existing neurospatial modules
from neurospatial.kernels import compute_diffusion_kernels
from neurospatial.distance import distance_field, geodesic_distance_matrix
# NOTE: neighbors_within() does NOT exist - use distance_field() + threshold
from neurospatial.visibility import compute_viewshed, FieldOfView

# From BEHAV_PLAN.md modules
from neurospatial.metrics.vte import wrap_angle, head_sweep_magnitude

# External
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy as scipy_entropy
import networkx as nx
```

### Error Message Template

```python
raise ValueError(
    f"<What's wrong>. "
    f"Got <actual values>. "
    f"\nTo fix:\n"
    f"- <Suggestion 1>\n"
    f"- <Suggestion 2>"
)
```

---

## Execution Order

| Order | Milestone | Dependencies | Notes |
|-------|-----------|--------------|-------|
| 1 | M1: Reachability | None | Foundation |
| 2 | M2: Manifold | None (parallel with M1) | Can start early |
| 3 | M3: Diffusion Baseline | M1.3 (`future_distribution`) | After M1 core |
| 4 | M4: Geodesic VTE | BEHAV_PLAN.md vte.py | After BEHAV_PLAN.md |
| 5 | M5: Integration | M1, M2, M3, M4 | Last |

**Recommended approach**:

1. Complete BEHAV_PLAN.md first (path_efficiency, goal_directed, decision_analysis, vte)
2. Implement M1 and M2 in parallel
3. Implement M3 after M1 core functions complete
4. Implement M4 after BEHAV_PLAN.md vte.py complete
5. Final integration in M5

---

## References

- **BEHAV_PLAN2.md**: Full implementation details, dataclass definitions, function signatures
- **BEHAV_PLAN.md**: Basic behavioral metrics (prerequisite for M4)
- **Mathematical framework**: Coifman & Lafon (2006), Stachenfeld et al. (2017), Redish (2016)
- **Existing code**: `neurospatial.kernels`, `neurospatial.distance`, `neurospatial.visibility`
