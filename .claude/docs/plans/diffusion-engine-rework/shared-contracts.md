# Shared contracts — diffusion engine rework

[← back to PLAN.md](PLAN.md) · [design spec](design-correctness.md)

Cross-phase contracts. Both phases depend on these; **do not weaken**. The authoritative
math rationale is [design-correctness.md](design-correctness.md) §3; this file is the
executable contract.

## Index

- [C1. Finite-volume operator](#c1-finite-volume-operator)
- [C2. Kernel modes & orientation](#c2-kernel-modes--orientation)
- [C3. Per-geometry finite-volume data](#c3-per-geometry-finite-volume-data)
- [C4. Canonical `M`](#c4-canonical-m)
- [C5. Component structure from `W`](#c5-component-structure-from-w)
- [C6. Low-level `compute_diffusion_kernels` API](#c6-low-level-compute_diffusion_kernels-api)

---

## C1. Finite-volume operator

Single operator for all geometries:

```
H(σ) = exp(−t · L),   t = σ² / 2,   L = M⁻¹ (D − W)
W[i,j] = A[i,j] / d[i,j]     # A = shared-face measure, d = center distance; symmetric
D      = diag(W · 1)
M      = diag(volumes)       # per-bin cell volume, == env.bin_sizes (see C4)
```

Properties of the **raw** heat kernel (pre-clip / pre-normalization) — the reason the three
mode kernels stay mutually consistent. Verified on the raw operator to numerical tolerance at
full rank; they are **not** re-asserted on the per-mode-normalized returned kernels (those are
checked against their own C2 contracts; per-mode normalization in
[D4](designs.md#d4-heat_kernel--per-component-renormalization) is what makes each returned
kernel meet its contract):

- `L · 1 = 0` ⇒ `H · 1 = 1` (row-stochastic).
- M-self-adjoint: `M[i]·H[i,j] = M[j]·H[j,i]` ⇒ `Σ_i M[i]·H[i,j] = M[j]` (⇒ `density`
  integrates to 1 at full rank).
- Symmetric conjugate `S = M^(−1/2)(D−W)M^(−1/2)` has a real spectrum;
  `H = M^(−1/2) exp(−tS) M^(1/2)`. **PR2 (out of scope here) eigendecomposes `S`; keep this
  factorization reachable** — Phase 1's `heat_kernel` must take `(L or its S-conjugate, M)`
  so PR2 swaps only its internals.

Phase 1 implements `H` via dense `scipy.sparse.linalg.expm(−t·L)` (same cost class as today;
PR2 replaces it). Clip round-off negatives, then **normalize each mode to its own contract**
(average → row-stochastic; transition → column-stochastic; density → M-weighted columns
integrate to 1). Row-normalizing once and reusing would break density's integrate-to-1 after
clipping — normalize per mode ([D4](designs.md#d4-heat_kernel--per-component-renormalization)).

## C2. Kernel modes & orientation

`H` averages **intensive** fields (rates); `Hᵀ` propagates **extensive** quantities (mass).
They coincide **only** when `M` is uniform. One operator, three public kernels:

| `mode` | kernel returned | stochasticity | consumer / use | phase |
| --- | --- | --- | --- | --- |
| `"transition"` | `Hᵀ` | column (`Σ_i K[i,j]=1`) | `kernel @ field` mass-conserving smoothing of **extensive** data (occupancy, counts); `env.smooth`, `apply_kernel(forward)` | 1 |
| `"density"` | `H·M⁻¹` | `Σ_i M[i]·K[i,j]=1` | count→**density** (`ρ = K @ counts`); KDE `spike_density/occupancy_density` | 1 |
| `"average"` | `H` | row (`Σ_j K[i,j]=1`) | `kernel @ rate` average of an **intensive** field (rate maps, probability *densities*); NOT discrete probability *mass* (use `transition`) | 2† |

† *phase* = when the mode is **publicly** exposed on `env.smooth`/`compute_kernel`; the
low-level `H` view exists from Phase 1 (C6, D4).

- `transitions(method="diffusion")` returns the **row-stochastic `H`** = `Hᵀ.T` (Phase 1
  updates it to transpose, not assume symmetry).
- **Do not** return `H` from `mode="transition"` (that reintroduces the orientation bug on
  non-uniform `M`). Uniform `M`: `Hᵀ = H`, so all orientations coincide — but `density` is
  still `H/m ≠ H`, and every mode's *values* differ from the pre-fix operator (that is the
  σ correction), so tests must pin the new operator, not today's numbers.

## C3. Per-geometry finite-volume data

Each layout supplies `(A per edge, volumes per node)`; formulas + code in
[designs.md#per-geometry-finite-volume-builders](designs.md#d1-per-geometry-finite-volume-builders).
Dispatch key:

| geometry | layout engine class (verified) | dispatch |
| --- | --- | --- |
| cartesian | `RegularGridLayout`, `MaskedGridLayout`, `ImageMaskLayout`, `ShapelyPolygonLayout` (all subclass `_GridMixin`) | engine type / `isinstance(_GridMixin)` |
| hex | `HexagonalLayout` | engine type |
| graph | `GraphLayout` (linear track) | engine type |
| mesh | `TriangularMeshLayout` | engine type |
| polar | `EgocentricPolarEnvironment` (built on `MaskedGridLayout`) | **env type / `_POLAR`**, NOT engine |

Polar MUST dispatch on the environment (`_POLAR` flag / `EgocentricPolarEnvironment`), or it
is misclassified cartesian ([polar.py:191](../../../src/neurospatial/environment/polar.py)).

## C4. Canonical `M`

`M` (per-bin volume) is **one array** used identically by the operator, the density
normalization, and the `apply_kernel` adjoint. It **is `env.bin_sizes`**, which already
returns true volumes for polar (annular sector,
[polar.py:437](../../../src/neurospatial/environment/polar.py)), hex
([hexagonal.py:347](../../../src/neurospatial/layout/engines/hexagonal.py)), mesh
([triangular_mesh.py:406](../../../src/neurospatial/layout/engines/triangular_mesh.py)), and
graph ([graph.py:407](../../../src/neurospatial/layout/engines/graph.py)). Only the generic
`_GridMixin.bin_sizes` ([mixins.py:387](../../../src/neurospatial/layout/mixins.py)) assumes
uniform cells — so **nonuniform-Cartesian `grid_edges` are out of scope** and excluded from
the physical-σ guarantee (operator + contract stay mutually consistent because both read the
same `bin_sizes`).

## C5. Component structure from `W`

Connected components for any component-aware step use the **`W` matrix** (nonzero `A`),
NOT `env.connectivity`. Corner-touching 8-connected bins have `A=0` ⇒ separate diffusion
components; using `env.connectivity` would shift mass between decoupled regions. A region
joined only by corner contact does not diffuse across the corner (zero-flux, physically
correct). **At full rank (Phase 1) the heat kernel is block-diagonal across `W`-components,
so the per-mode normalization (C1) is within-component automatically — no explicit component
loop is needed.** The `W`-component structure is asserted by the corner-split test and
becomes load-bearing under **PR2** truncation (where clipping removes real cross-block
lobes). Components: `scipy.sparse.csgraph.connected_components(W, directed=False)`.

## C6. Low-level `compute_diffusion_kernels` API

`ops/smoothing.py::compute_diffusion_kernels(graph, *, volumes, sigma, mode)`:

- Reads the face measure from **edge attribute** `graph.edges[u, v]["A"]` — single source of
  truth, exactly like `"distance"`. No `face_measures` argument.
- **Explicit `A == 0`** ⇒ no diffusion across that edge. **Missing `A`** on an existing edge
  ⇒ **raise** (a broken geometry builder must not silently degrade to an identity-ish
  kernel). A **negative, NaN, or infinite `A`** ⇒ **raise** (it would corrupt `W = A/d` and
  the C1 invariants); only `A >= 0` finite is accepted.
- `volumes`: node-ordered `(n_bins,)` array aligned to nodes `0..n−1`.
- **Input validation (preserve + extend today's `bandwidth <= 0` check at
  [smoothing.py:149](../../../src/neurospatial/ops/smoothing.py)):** raise on non-finite
  `sigma` or `sigma <= 0` (it feeds `sigma**2` into `expm`);
  on `volumes.shape != (n_bins,)`; on any non-finite or non-positive `volumes` entry (D4
  divides by `volumes`, so a zero/negative/NaN volume must fail loudly, not produce
  `inf`/`nan`); on **node IDs not contiguous `0..n−1`** (they are used directly as sparse-
  matrix indices — a non-integer or gappy label set must fail clearly, not cryptically); and
  on any edge **missing `"distance"`** or with a non-finite / non-positive `"distance"`
  (mirrors the `"A"` handling — `NaN` **and** `inf` rejected).
- `mode ∈ {"transition","density","average"}` (C2). Returns the dense `(n_bins, n_bins)`
  kernel. **All three are implemented at the low level in Phase 1** (D4 returns the `H`
  view for `"average"`); the **public** `mode="average"` on `env.smooth`/`compute_kernel`
  and the `binned`/`resample` rerouting are **Phase 2**. So C6 is fully satisfied by Phase 1
  even though Phase 1 exposes only `transition`/`density` publicly.
- **Public API break**: replaces the old `(graph, bandwidth_sigma, bin_sizes, mode)`
  Gaussian-weight signature. Exported from `neurospatial.ops`
  ([ops/__init__.py:188](../../../src/neurospatial/ops/__init__.py)) — call out in CHANGELOG.
