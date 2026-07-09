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

Invariants (tested):

- `L · 1 = 0` ⇒ `H · 1 = 1` (`H` row-stochastic).
- M-self-adjoint: `M[i]·H[i,j] = M[j]·H[j,i]` ⇒ `Σ_i M[i]·H[i,j] = M[j]`.
- Symmetric conjugate `S = M^(−1/2)(D−W)M^(−1/2)` has a real spectrum;
  `H = M^(−1/2) exp(−tS) M^(1/2)`. **PR2 (out of scope here) eigendecomposes `S`; keep this
  factorization reachable** — Phase 1's `heat_kernel` must take `(L or its S-conjugate, M)`
  so PR2 swaps only its internals.

Phase 1 implements `H` via dense `scipy.sparse.linalg.expm(−t·L)` (same cost class as today;
PR2 replaces it). Clip round-off negatives to 0, then renormalize per component (C5).

## C2. Kernel modes & orientation

`H` averages **intensive** fields (rates); `Hᵀ` propagates **extensive** quantities (mass).
They coincide **only** when `M` is uniform. One operator, three public kernels:

| `mode` | kernel returned | stochasticity | consumer / use | phase |
| --- | --- | --- | --- | --- |
| `"transition"` | `Hᵀ` | column (`Σ_i K[i,j]=1`) | `kernel @ field` mass-conserving smoothing of **extensive** data (occupancy, counts); `env.smooth`, `apply_kernel(forward)` | 1 |
| `"density"` | `H·M⁻¹` | `Σ_i M[i]·K[i,j]=1` | count→**density** (`ρ = K @ counts`); KDE `spike_density/occupancy_density` | 1 |
| `"average"` | `H` | row (`Σ_j K[i,j]=1`) | `kernel @ rate` average of an **intensive** field (rate maps, probability *densities*); NOT discrete probability *mass* (use `transition`) | 2 |

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

| geometry | layouts (engine class) | dispatch |
| --- | --- | --- |
| cartesian | `RegularGrid`, `MaskedGrid`, `ImageMask`, `ShapelyPolygon` | engine type |
| hex | `Hexagonal` | engine type |
| graph | `Graph` (linear track) | engine type |
| mesh | `TriangularMesh` | engine type |
| polar | `EgocentricPolarEnvironment` (built on `MaskedGrid`) | **env type / `_POLAR`**, NOT engine |

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

Per-component mass renormalization uses connected components of the **`W` matrix** (nonzero
`A`), NOT `env.connectivity`. Corner-touching 8-connected bins have `A=0` ⇒ separate
diffusion components; using `env.connectivity` would shift mass between decoupled regions
after clipping. A region joined only by corner contact does not diffuse across the corner
(zero-flux, physically correct).

## C6. Low-level `compute_diffusion_kernels` API

`ops/smoothing.py::compute_diffusion_kernels(graph, *, volumes, sigma, mode)`:

- Reads the face measure from **edge attribute** `graph.edges[u, v]["A"]` — single source of
  truth, exactly like `"distance"`. No `face_measures` argument.
- **Explicit `A == 0`** ⇒ no diffusion across that edge. **Missing `A`** on an existing edge
  ⇒ **raise** (a broken geometry builder must not silently degrade to an identity-ish
  kernel).
- `volumes`: node-ordered `(n_bins,)` array aligned to nodes `0..n−1`.
- `mode ∈ {"transition","density","average"}` (C2). Returns the dense `(n_bins, n_bins)`
  kernel.
- **Public API break**: replaces the old `(graph, bandwidth_sigma, bin_sizes, mode)`
  Gaussian-weight signature. Exported from `neurospatial.ops`
  ([ops/__init__.py:188](../../../src/neurospatial/ops/__init__.py)) — call out in CHANGELOG.
