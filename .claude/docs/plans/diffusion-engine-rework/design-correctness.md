# Design — Diffusion kernel correctness fix (grid-independent bandwidth)

**Status:** revised after design review; all layouts + hard operators fully specified;
mesh sub-fork decided (E2, §3.E). Ready for implementation planning.
**Date:** 2026-07-08
**Scope:** PR1 of the diffusion-engine rework. Correctness only.
**Sibling docs (future):** `design-performance.md` (PR2: cached eigenbasis + spectral
truncation), `design-mrf-gam.md` (penalized-Poisson encoding estimator).
**Target release:** 0.7.0 (minor bump; behavior-changing correctness fix, pre-1.0).

---

## 1. Problem

`neurospatial`'s `diffusion_kde` smoothing bandwidth is **not** the physical σ its
docstring claims ([`fields.py::compute_kernel`](../../../src/neurospatial/environment/fields.py)
: "Smoothing bandwidth in physical units (σ)"). The *effective* smoothing width scales
with bin size, so the same `bandwidth` produces different physical smoothing at different
grid resolutions — a reproducibility bug: place-field widths / spatial information are not
comparable across bin sizes or with other tools.

### Root cause (`ops/smoothing.py::compute_diffusion_kernels`)

1. Gaussian edge weight `w = exp(-d²/2σ²)` — σ leaks into the graph operator.
2. `mode="density"` folds the mass matrix into the Laplacian (`L_vol = M⁻¹ L`) *before*
   `expm`, which on a finite-difference Laplacian double-counts the spacing (`≈ −∇²/h`).
3. Cartesian grids default to 8-connectivity, oversmoothing under a proper weighting.

### Evidence

Point-source std, requested σ = 5, measured in physical units (script:
[`scripts/diffusion_grid_bandwidth_check.py`](../../../scripts/diffusion_grid_bandwidth_check.py),
committed with this work; becomes the regression test):

| bin_size | ns `density` (encoding path) | ns `transition` | NLD `1/d²` (≡ FV on uniform grids) |
|---:|---:|---:|---:|
| 0.5 | 3.53 | 2.49 | **5.000** |
| 1.0 | 4.95 | 4.95 | **5.000** |
| 2.0 | 6.79 | 9.61 | **5.000** |
| 4.0 | 8.52 | 16.95 | **5.000** |

1D theory `σ·√h·exp(-h²/4σ²)` matches the measured `density` column to 3 decimals. In 2D
(8-connected) the `density` kernel overshoots to 8.5 at `bin_size=1` for σ=5 (~70% too
much) and drifts with resolution. The finite-volume operator (below) reads exactly 5.000
at every bin size, 1D and 2D.

---

## 2. Goal / non-goals

**Goal:** make `bandwidth` the true physical σ for the diffusion kernel. Fix in place (one
operator path; no parallel v1/v2).

**Verification status (the contract boundary).** The finite-volume operator is the *intended*
physical-σ construction for all layouts. Its epistemic status differs by geometry, and the
tests — not the prose — are the boundary; a failure is a geometry/spec correction *before*
production callers route through the new operator, not a silent approximation:

- **Cartesian 1D/2D (uniform spacing) — empirically confirmed** by
  [`scripts/diffusion_grid_bandwidth_check.py`](../../../scripts/diffusion_grid_bandwidth_check.py)
  (measured σ == `bandwidth` to 3 decimals). **Nonuniform / custom `grid_edges`** inherit
  `_GridMixin.bin_sizes`' uniform approximation and are **excluded from PR1's physical-σ
  guarantee** (§3.B, tracked follow-up).
- **Hex, polar, graph — analytic expectation / implementation-gating smoke tests.** σ
  recovery is an acceptance criterion checked on first implementation, verified by the first
  smoke tests — not something this doc asserts as already proven.
- **Mesh (E2 centroid TPFA) — expected on well-shaped refined meshes only.** σ recovery is
  *required* on well-shaped flat triangulations; skewed meshes are allowed approximation
  error (skew-guarded, §3.E) and may require a future circumcentric/cotangent operator.

**Non-goals (deferred):** performance (dense `expm` → cached eigenbasis + truncation) →
PR2; MRF-GAM estimator → separate design; changing `env.connectivity` (stays 8-connected
for pathfinding). **`min_occupancy` is explicitly out of PR1 and out of the grid-invariance
guarantee**: it is documented as seconds ([_smoothing.py:259](../../../src/neurospatial/encoding/_smoothing.py))
but the KDE paths compare it to `occupancy_density = K @ occupancy`, which under the new
density kernel `H·M⁻¹` is **seconds per cell volume** on non-uniform grids — a pre-existing
unit mismatch that PR1 neither fixes nor worsens for uniform grids. Flagged in the CHANGELOG
and a tracked follow-up.

---

## 3. Design

### 3.A Unified operator — finite-volume (TPFA) mass-weighted heat kernel

All geometries use **one** operator. Discretize the heat equation `∂u/∂t = ∇²u` by the
two-point flux finite-volume method:

```
H(σ) = exp(−t · L),   t = σ²/2,   L = M⁻¹ (D − W)
W_ij = A_ij / d_ij      (A_ij = measure of the face shared by bins i,j;  d_ij = center distance)
D    = diag(W · 1)
M    = diag(cell volume per bin)
```

`L`'s continuum limit is `−∇²` on any **K-orthogonal** discretization (center-to-center
line ⟂ shared face), which every regular lattice here satisfies. Consequences:

- **`L·1 = 0`** (constant preserved) ⇒ **`H·1 = 1`**: `H` is natively **row-stochastic**.
- **M-self-adjoint**: `M_i H_ij = M_j H_ji` ⇒ `Σ_i M_i H_ij = M_j` (**M-weighted mass
  conserved**).
- **Real spectrum** via the symmetric conjugate `S = M^{-1/2}(D−W)M^{-1/2}`, with
  `H = M^{-1/2} exp(−tS) M^{1/2}` (the clean seam PR2 eigendecomposes).

This single operator subsumes the old "FD vs FEM vs special-polar" split. Two properties
fall out for free that the previous design hand-coded:

- **Diagonal edges vanish automatically**: Moore/diagonal neighbors share only a corner
  ⇒ `A_ij = 0` ⇒ `W_ij = 0`. No face-adjacency filter needed.
- **No hex lattice constant**: the face/volume geometry is the physical-σ construction with
  no hand-tuned constant (analytic; acceptance-tested per §2).

### 3.B Per-geometry finite-volume data

Each layout supplies `(A_ij, d_ij, M_i)`; the central module builds `L`. `d_ij` is already
the edge `"distance"`. What's new is `A_ij` (face measure) and `M_i` (cell volume).

| geometry | `A_ij` (shared face) | `M_i` (cell volume) | notes |
|---|---|---|---|
| Cartesian (regular/masked/image/polygon) | ∏ of the *other* dims' bin widths (corner-only ⇒ 0) | ∏ bin widths | diagonals auto-excluded |
| Hexagonal | shared hex edge length | hex cell area | constant automatic |
| Egocentric-polar | radial nbr: arc `r_face·Δθ`; angular nbr: `Δr` | sector area `½(r_out²−r_in²)Δθ` | K-orthogonal ⇒ analytically exact; seam edges handled |
| Graph / linear-track | 1 (unit cross-section) | bin length | `d_ij` via junction contraction (§3.F) |
| Triangular mesh | shared triangle edge length | triangle area | well-shaped only; skew-guarded (§3.E) |

**Canonical-`M` invariant.** `M` is a single per-bin volume array used identically by the
operator, the density normalization, and the `apply_kernel` adjoint. It is `env.bin_sizes`,
which **already** returns true volumes for polar (annular sector,
[polar.py:437](../../../src/neurospatial/environment/polar.py)), hex, mesh, and graph;
only the generic `_GridMixin` ([mixins.py:387](../../../src/neurospatial/layout/mixins.py))
assumes uniform Cartesian cells. `A_ij` is computed under the same geometry convention as
`M`. Nonuniform-Cartesian masked `grid_edges` therefore inherit `bin_sizes`' existing
uniform approximation (operator + contract stay consistent, so it is not made worse);
making `_GridMixin.bin_sizes` nonuniform-aware is a tracked follow-up, out of PR1 scope.

### 3.C Modes & orientation contract (resolves the transition-orientation finding)

`H` (row-stochastic) **averages an intensive field** (rate/temperature: `H @ rate`);
`Hᵀ` (column-stochastic, since `H·1=1`) **propagates an extensive quantity** (counts/mass:
`Hᵀ @ counts` conserves `Σ counts`). These are the backward/forward duality of the same
operator; they coincide **only** when `M` is uniform (`H` symmetric). The library consumes
both orientations, so each mode picks the right one:

- **`compute_kernel(mode="transition")` → `Hᵀ`** (column-stochastic, `Σ_i K_ij = 1`).
  Consumed as `kernel @ field` for mass-conserving forward smoothing — `env.smooth`,
  `occupancy(bandwidth=…)` ([trajectory.py:495](../../../src/neurospatial/environment/trajectory.py)),
  `apply_kernel(mode="forward")`. Preserves the documented `smoothed.sum() == field.sum()`
  contract for **extensive** inputs (`resample_field(method="diffuse")` is *intensive* — see
  the `H`-average bullet).
- **`compute_kernel(mode="density")` → `H·M⁻¹`** (`= M⁻¹Hᵀ`; `ρ = K @ counts` diffuses the
  count-density `counts/M`; `Σ_i M_i K_ij = 1`, today's density contract).
- **`transitions(method="diffusion", normalize=True)` → `H`** (row-stochastic,
  `P(next=j | current=i)`, [trajectory.py:838](../../../src/neurospatial/environment/trajectory.py)).
  Obtained as `compute_kernel(mode="transition").T` (or a dedicated row-stochastic path);
  `transitions()` is updated to transpose rather than assume symmetry.
- **`compute_kernel(mode="average")` / `env.smooth(mode="average")` → `H`** (row-stochastic;
  `kernel @ rate` **averages an intensive field**). **New public mode in PR1** — the correct
  smoother for rate maps / probability **densities**, especially on non-uniform `M` where
  `density` (`H·M⁻¹`) would volume-bias them. Same `H` as `transitions()`, exposed for
  smoothing. **Discrete probability *mass*** over bins (a posterior summing to 1) stays on
  `transition` (`Hᵀ`), which preserves `Σ == 1` — `average` is for intensive quantities only.

For **uniform `M`** (Cartesian, hex) `H` is symmetric, so `Hᵀ = H` and the transition kernel
equals the `transitions()` matrix — the **orientation** ambiguity vanishes (transpose is a
no-op). The density kernel is still `H·M⁻¹ = H/m` (a per-cell rescaling absorbed by the
integrate-to-1 normalization), **not** identical to the transition kernel — only orientation
coincides, not the three kernels. And the smoothed **values still differ from the pre-fix
operator even on a uniform grid**: that difference *is* the correction (effective σ becomes
the requested σ), so "reduces to today's values" would be wrong. For **non-uniform `M`**
(polar, mesh) the orientations genuinely differ too; returning the wrong one is exactly the
failure the review flagged.

- **Intensive-field averaging (`H`) is in PR1 — public `mode="average"`.** A field that is already *intensive*
  (a rate map) must be **averaged** by the row-stochastic `H`, not run through `density`
  (`H·M⁻¹`), which reweights by inverse source-cell volume on non-uniform `M`.
  `smooth_rate_map(method="binned")` does exactly this — it smooths an intensive raw rate via
  `env.smooth()` (default `density`) ([_smoothing.py:711](../../../src/neurospatial/encoding/_smoothing.py)),
  so on polar/mesh it would be volume-biased. `resample_field(method="diffuse")` has the same
  issue — it smooths place-fields (intensive) via `mode="transition"` (`Hᵀ`) on `dst_env`
  ([binning.py:799](../../../src/neurospatial/ops/binning.py)). PR1 therefore adds a **private
  `H`-averaging path** and routes **both** `binned` (single + batch + JAX) **and** diffuse
  `resample_field` through it, **and** exposes it as the public `mode="average"` (above) so
  external callers get a correct intensive-field smoother — nothing about it is deferred. For
  `resample_field` the `H`-average must use **masked normalization** — smooth
  `value·valid_mask`, smooth `valid_mask`, divide, then re-impose outside-source `NaN`
  (mirroring `binned`) — because the current zero-fill-then-single-smooth
  ([binning.py:810](../../../src/neurospatial/ops/binning.py)) lets an uncovered bin
  contribute a real 0 and biases covered values **down** near the uncovered region.

### 3.D Dispatch & API (resolves the graph-only-dispatch + polar-classification findings)

- **Entry point is env-level**, since geometry requires the environment, not a bare graph.
  New `ops/diffusion.py`:
  - `diffusion_kernel(env, sigma, *, mode) -> ndarray` — resolves geometry, builds `L`, and
    returns, for the public modes `{"transition","density","average"}`:
    - `mode="transition"` → **`Hᵀ`** — column-stochastic *smoothing* kernel; `kernel @ field`
      conserves `Σ field`.
    - `mode="density"` → **`H·M⁻¹`** — count→density kernel; `Σ_i M_i K_ij = 1`.
    - `mode="average"` → **`H`** — row-stochastic intensive-field smoother (`H @ rate`); new
      public mode in PR1, same `H` as `transitions()`.
    `transitions(method="diffusion")` reuses that **row-stochastic `H`**, obtained as the
    transpose of the transition kernel (`H = (Hᵀ)ᵀ`) — the one non-uniform-`M` path that must
    NOT reuse the smoothing kernel verbatim.
  - `heat_kernel(L, M, sigma) -> H` builds the base operator; `Hᵀ`, `H·M⁻¹`, and `H` are
    views of it (PR2 swaps only `heat_kernel`'s internals).
  - **`average` threads every mode surface** — the plan enumerates each:
    `Literal["transition","density","average"]` types, `valid_modes` sets, the
    `(bandwidth, mode)` kernel cache keys, `EnvironmentProtocol` / `compute_kernel` /
    `smooth` signatures, and the low-level `compute_diffusion_kernels(…, mode=…)`.
- **Geometry resolution**: each layout engine gains a small
  `finite_volume_geometry(env) -> (face_measures, distances, volumes)` (or a
  `_diffusion_geometry` tag + a per-geometry builder in `ops/diffusion.py`). **Polar
  dispatches on the `EgocentricPolarEnvironment` type / `_POLAR`, not the layout engine** —
  it is built on a `MaskedGrid` layout ([polar.py:191](../../../src/neurospatial/environment/polar.py))
  and would otherwise be misclassified Cartesian.
- **`compute_diffusion_kernels(graph, …)` — graph-only dispatch**: the current public, graph-only
  signature cannot carry geometry. Resolution: `env.compute_kernel` routes to
  `ops/diffusion.py`; `compute_diffusion_kernels` is **reframed** as a lower-level
  primitive `compute_diffusion_kernels(graph, *, volumes, sigma, mode)` that reads the face
  measure from a **graph edge attribute** `graph.edges[u, v]["A"]` — single source of truth,
  exactly like `"distance"`, with no redundant `face_measures` arg to drift out of alignment.
  An **explicit `A=0`** means no diffusion across that edge (e.g. corner-touching Cartesian
  bins); a **missing** `A` on an existing edge **raises** (a broken geometry builder must not
  silently degrade to an identity-ish kernel). `volumes` is a **node-ordered array** aligned
  to nodes `0..n-1`. The old
  `(graph, bandwidth_sigma, bin_sizes, mode)` Gaussian form is **removed** (no v1/v2). Its
  known in-repo callers (a **public export** + tests) — [ops/__init__.py](../../../src/neurospatial/ops/__init__.py)
  export, [tests/ops/test_ops_smoothing.py](../../../tests/ops/test_ops_smoothing.py),
  [tests/benchmarks/test_performance.py:260](../../../tests/benchmarks/test_performance.py),
  and a mock in [test_encoding_spatial.py:5052](../../../tests/encoding/test_encoding_spatial.py) —
  all updated in this PR. `apply_kernel`'s **forward** (`K @ field`) is unchanged; its
  **adjoint** is the intended operator only w.r.t. the M-weighted inner product, which no
  longer equals the plain transpose when `M` is non-uniform. PR1 adds an **adjoint regression
  test on a non-uniform-`M` geometry** and states that inner-product contract (no production
  caller uses density adjoints today).

### 3.E Mesh operator — decided: E2 (resolves finding 3)

Bins are triangle **centroids** with triangle-adjacency
([triangular_mesh.py:163](../../../src/neurospatial/layout/engines/triangular_mesh.py)),
so a *cell-centered* finite-volume operator is the right family — but TPFA is only exact on
a K-orthogonal dual, and centroids are not circumcenters. Options:

- **(E1) Circumcentric-Voronoi finite volume** — exact orthogonal operator (physical σ),
  but obtuse triangles put the circumcenter outside the cell (negative/again-special
  handling), and the "centers" are then circumcenters, not the centroid `bin_centers`.
- **(E2) TPFA on centroids** — reuse §3.A directly; consistent and physical-σ for
  well-shaped (near-equilateral) meshes, with an O(1) error that grows with triangle
  skew. σ-recovery tested on a refined well-shaped mesh.
- **(E3) Vertex cotangent → project to triangles** — exact FEM on vertices, then average
  the 3 vertex values per triangle; adds a projection step and lives on a different node
  set than the bins.

**Decided: (E2)** — keeps the single unified operator (no separate mesh code path), is
exact in the well-shaped limit the tests exercise, and documents the skew-dependent error.
Escalate to (E1) only if a real skewed-mesh use case needs it.

**Skew guard (resolves finding 3).** TPFA is exact only on a K-orthogonal dual, which
centroid meshes approximate. So the builder computes a per-mesh non-orthogonality measure
(e.g. max angle between the centroid-connection line and the shared-edge normal, or the
fraction of obtuse triangles) and **warns loudly** when it exceeds a threshold — turning
"exact physical σ" into an *honest, self-reporting* claim rather than a silent
approximation. The σ-recovery guarantee is asserted (and tested) only for meshes within the
well-shaped regime. **The implementation plan pins the exact metric and threshold** —
candidate: fraction of non-acute triangles above X%, or a max centroid-line/face-normal
angle above θ — before any test is written, so "well-shaped" is a checkable predicate, not a
subjective label.

### 3.F Graph / linear-track junctions (resolves finding 5)

Current junction (inter-segment) edge `"distance"` is the Euclidean **chord** between
embedded bin centers ([graph.py:296](../../../src/neurospatial/layout/helpers/graph.py)),
not the path length through the junction, so `A/d` oversmooths across bends. Adopt the
NLD-style **junction contraction**: Dijkstra from each bin-center over the substrate
(bin-center, bin-edge, junction nodes), treating other bin-centers as sinks, to get true
along-track `d_ij` (ref: `non_local_detector .../diffusion.py::_neighbor_centers`). A
straight track (no junctions) is unaffected.

### 3.G Normalization / mass (resolves the component-graph finding)

Clip tiny negative round-off, renormalize. On **disconnected** graphs renormalize **per
connected component** using M-weighted sums, so a point source beside a wall cannot leak
mass across it. **Component structure is derived from the diffusion weight matrix `W`
(nonzero `A_ij`), NOT `env.connectivity`.** With default 8-connectivity, corner-touching
bins are linked in `env.connectivity` but have `A_ij = 0`, so they are *separate* diffusion
components — using `env.connectivity`'s components would shift mass between truly-decoupled
regions after clipping. (Consequence, documented: a region joined only through corner
contact does not diffuse across that corner — physically correct, zero-flux interface.)

### 3.H Routing / in-place rewrite

`env.compute_kernel` → `ops/diffusion.py`. `ops/smoothing.py::compute_diffusion_kernels`
reframed (§3.D). Routed through the new path: `env.smooth`; `encoding/_smoothing.py`
(`_diffusion_kde` + `_binned`-via-`H` + batch + JAX); `transitions(method="diffusion")` (→
`H`); and `ops/binning.py::resample_field(method="diffuse")` (→ `H`-average, §3.C).

---

## 4. Testing

σ-recovery **tolerances are fixed a priori** from the expected discretization error (a few
%), **not** tuned to pass: a geometry that misses its principled tolerance is an
operator/spec correction, not a tolerance bump. The plan pins the specific per-geometry
numbers (Cartesian starts at `rtol=0.02`) before code — that pinning is a plan deliverable,
not a design-doc guess.

- **Grid-independence regression**: measured σ ≈ `bandwidth` across bin sizes, 1D + 2D,
  Cartesian/masked/polygon. The CI test is a **hardened** form of
  `scripts/diffusion_grid_bandwidth_check.py` — **deterministic explicit grid/mask
  construction** (not random point sampling), prints replaced by **assertions with named
  tolerances** (e.g. `rtol=0.02`). The script stays as runnable exploratory evidence.
- **Resolution-independence**: hex, polar, graph/linear-track each recover σ across
  resolutions.
- **Mesh**: σ recovered on a well-shaped flat triangulation at two resolutions (E2); the
  skew guard warns on a deliberately skewed mesh.
- **σ-measurement protocol** (shared by every resolution-independence test): measure the
  smoothed point-source std in **physical Cartesian (or local-tangent) coordinates**, never
  bin-index space; place the source in the **interior**, away from domain boundaries, the
  polar seam, and `r=0` (unless a test targets the seam specifically); for polar, convert bin
  centers to Cartesian before the second moment. Keeps a smoke-test failure attributable to
  the operator, not a coordinate/boundary artifact.
- **Orientation contract**: on a **non-uniform-`M`** geometry (polar or mesh),
  assert `compute_kernel(mode="transition")` is **column**-stochastic (`Σ_i K_ij = 1`) and
  `kernel @ counts` conserves `Σ counts`; assert `transitions(method="diffusion")` is
  **row**-stochastic (`Σ_j T_ij = 1`); assert `mode="density"` integrates to 1 under
  M-weighting. On a uniform grid the transition kernel and `transitions()` coincide
  (symmetric `H`); the density kernel differs by the normalized-away mass factor; and **all**
  values differ from the *pre-fix* operator (the intended σ correction) — so the test pins the
  new operator, not the old numbers.
- **Binned averaging (`H`)**: on a non-uniform-`M` geometry (polar/mesh),
  `smooth_rate_map(method="binned")` uses the private row-stochastic `H` path, not `density`;
  assert the smoothed intensive rate is the volume-*unbiased* **valid-bin-normalized** `H`
  average (current `binned` smooths the rate + a `{0,1}` validity mask, **not** occupancy
  weights; matches `H`, not `H·M⁻¹`) — this preserves current `binned` semantics, only
  removing the volume bias.
- **Diffuse resample (`H`, masked)**: on a non-uniform-`M` destination env,
  `resample_field(method="diffuse")` uses the **masked** `H`-average (smooth `v·valid` /
  smooth `valid`); assert covered values are **not biased down** by adjacent uncovered bins
  (vs. the zero-fill single-smooth), and outside-source bins stay `NaN`.
- **Adjoint (non-uniform `M`)**: `apply_kernel(mode="adjoint")` on the density kernel obeys
  the stated M-weighted inner-product contract (regression on polar/mesh).
- **Component structure**: a corner-only-connected pair (8-connected
  `env.connectivity`, `A_ij=0`) forms two diffusion components — no mass crosses the corner;
  per-component renormalization uses the `W`-graph components.
- **No-leakage**: point source beside a masked wall puts ~0 mass across it.
- **Golden-value updates**: enumerate + recompute tests pinning specific smoothed numbers
  (expected fallout) — done during the implementation plan.

## 5. Migration & docs

- 0.6.0 → 0.7.0. CHANGELOG: `bandwidth` is now the true physical σ; 2D place fields smooth
  ~1.5–1.7× less for the same value; "to approximate old behavior scale bandwidth by
  ~`√(bin_size)`" (rough, mode-dependent). Fix docstrings in `fields.py`,
  `ops/smoothing.py`, `encoding/_smoothing.py`, and `environment/_protocols.py` — the last's
  `bin_sizes` docstring wrongly says shape `(n_dims,)` / bin widths, but it is a **per-bin
  volume `(n_bins,)`**, the very quantity PR1 uses as `M`.
- **Mode input-type contract**: document that `mode="density"` (`H·M⁻¹`) takes an
  **extensive** input (counts / mass / occupancy) and returns a **density** (integrates to
  1) — it must **not** be applied to an already-**intensive** field (rate map, probability
  density), which on non-uniform `M` divides by cell volume twice. Intensive fields are
  *averaged* by the row-stochastic `H` via the **new public `mode="average"`** (§3.C). Today `fields.py::smooth`/`compute_kernel` describe density mode as smoothing
  "continuous density fields / rate maps" ([fields.py:177](../../../src/neurospatial/environment/fields.py));
  that docstring is corrected to the input-type contract. On uniform `M` the *orientation*
  distinction vanishes, but density is still `H/m`: for a direct `env.smooth(mode="density")`
  the constant volume factor changes units/values when `m ≠ 1` — it only cancels in ratio
  estimators (the KDE `spike_density/occupancy_density`).
- **`env.smooth` default stays `mode="density"` (deliberate)** — flipping the default to
  `average` would silently change every mode-less `env.smooth()` call, an orthogonal breaking
  change; PR1 keeps `density` for compatibility, documents `average` as the correct choice for
  intensive rate maps, and leaves any default-flip as a tracked follow-up. Called out in the
  migration notes.
- **Public API break (finding 5)**: `neurospatial.ops.compute_diffusion_kernels` is
  exported ([ops/__init__.py:188](../../../src/neurospatial/ops/__init__.py)); reframing its
  `(graph, bandwidth_sigma, bin_sizes, mode)` signature to finite-volume inputs is a
  **breaking change to a public low-level function** — called out explicitly in the
  CHANGELOG (not buried as internal cleanup).

## 6. Risks / open items

- **Mesh exactness (§3.E)** — decided E2 (TPFA-centroid): accurate for well-shaped
  meshes, skew-dependent error documented; E1 (circumcentric, exact) held in reserve.
- **Polar seam** under M-weighting: ensure seam faces use the realized angular step
  (matches `_add_circular_connectivity` / `_fix_polar_edge_geometry`).
- **Face-measure extraction** currently isn't stored on edges — each geometry must compute
  `A_ij` (cheap for lattices; triangle-edge length for mesh; arc/radial for polar).
- **Golden-value churn** — intended consequence of the fix.
- **`mode="average"` (public intensive-`H` smoother) is in PR1 scope** (§3.C) — resolves the
  intensive-field gap on non-uniform `M`; nothing deferred.

## 7. References

- NLD engine: `~/Documents/GitHub/non_local_detector/.../likelihoods/diffusion.py`
  (`build_laplacian`, `diffuse`, `to_density`, `_neighbor_centers` junction contraction).
- Finite-volume TPFA / K-orthogonality: standard cell-centered FV heat discretization.
  Circumcentric dual: Voronoi finite volume.
- Experiment: `scripts/diffusion_grid_bandwidth_check.py`.
- Memory: `diffusion-kde-grid-dependent-bandwidth`, `nld-diffusion-mrf-learnings`.
