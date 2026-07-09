# Design — Diffusion engine PR2: cached eigenbasis + matrix-free apply-path

**Status:** approved design, pre-implementation.
**Scope:** PR2 of the diffusion-engine rework. **Performance only, behavior-preserving.**
**Depends on:** the shipped correctness operator (v0.7.0, [design-correctness.md](design-correctness.md);
`src/neurospatial/ops/diffusion.py`).
**Sibling (future):** `design-mrf-gam.md` (the (c) estimator).
**Target release:** 0.8.0 (minor — adds one public method `env.diffuse`; otherwise behavior-preserving).

---

## 1. Problem

The shipped operator builds a **dense** `exp(-tL)` per `(bandwidth, mode)` call via
`scipy.sparse.linalg.expm` ([ops/diffusion.py:65-101](../../../src/neurospatial/ops/diffusion.py),
`_raw_heat_operator`): `O(n_bins³)` time, `O(n_bins²)` memory, materialized every call and
cached only as the finished `(n,n)` matrix. It warns above ~3000 bins and is impractical on
large/fine grids. This retires the deferred `expm_multiply`/spectral-engine goal.

Key fact: the eigenbasis of the operator depends only on **geometry** `(W, volumes)`, not on
`sigma` or `mode`. Caching it amortizes **one** eigensolve across every neuron, bandwidth,
and mode; bandwidth-aware truncation then makes both time and memory `O(n·rank)` with
`rank ~ area/sigma²` (independent of `n_bins`).

## 2. Goal / non-goals

**Goal:** replace the dense `expm` with a cached, per-component, bandwidth-aware **truncated
eigenbasis** and a **matrix-free apply-path** that never materializes `(n,n)`. Output must be
**equivalent to the shipped dense operator within tolerance** (§7).

**Non-goals:** any change to the *operator* or its results (correctness is shipped); the MRF-GAM
estimator; nonuniform-Cartesian `bin_sizes`. No change to mode semantics or the three-mode
orientation contract.

## 3. Architecture

Two pieces replace `_raw_heat_operator`'s dense `expm`:

- **Cached eigenbasis (geometry-only).** The operator `L = M⁻¹(D−W)` is non-symmetric, so
  eigendecompose the **symmetric conjugate** `S = M^{-1/2}(D−W)M^{-1/2}` (PSD, real spectrum);
  `H = M^{-1/2} exp(-tS) M^{1/2}` (the seam Phase 1 kept, [ops/diffusion.py:63-64](../../../src/neurospatial/ops/diffusion.py)).
  Cache the per-component truncated `(Q, Λ, component_labels)` on the Environment via
  `versioned_cached_property` ([decorators.py:180](../../../src/neurospatial/environment/decorators.py)),
  so it builds once and auto-invalidates on the `_state_version` bump geometry changes trigger
  ([core.py:909](../../../src/neurospatial/environment/core.py)).
- **Matrix-free apply-path.** `H @ F = M^{-1/2} · Q · (e^{-tΛ} ⊙ (Qᵀ · (M^{1/2} · F)))`,
  `t = σ²/2`, for a batch `F` of shape `(n_bins, n_fields)`. `O(n·rank·n_fields)` time,
  `O(n·rank)` memory. `Hᵀ @ F` is the same with the `M^{±1/2}` powers swapped.

## 4. Eigensolver + truncation

Mirror NLD's `heat_kernel_rank`/`cached_eigenbasis`
(`~/Documents/GitHub/non_local_detector/.../likelihoods/diffusion.py`), adapted for the
M-conjugation (NLD's Laplacian is mass-free; ours needs `S`):

- **Per-component** decomposition (resolves the truncation-leak fork): decompose each
  `W`-connected-component's block of `S` separately (`_components_from_W` already exists,
  [ops/diffusion.py:176](../../../src/neurospatial/ops/diffusion.py)), so every eigenvector is
  component-local and **truncation cannot leak mass across a wall/hole by construction**.
- **Bandwidth-aware rank.** Keep modes with `e^{-tλ} ≥ tol` (default `tol=1e-6`); rank tracks
  `~area/σ²`, not `n_bins`. Resolve adaptively (Weyl's law probe) via truncated
  `scipy.sparse.linalg.eigsh`; the constant (λ=0) null mode per component is **always kept**
  (mass conservation).
- **Dense fallback.** When the resolved rank ≥ `dense_fraction·n` (default `0.5`), use dense
  `scipy.linalg.eigh` on the block (truncation stops paying off). Small grids therefore just
  do one cached dense `eigh` — already a large win over per-call `expm`.
- **Caching detail.** The eigenbasis is built at the geometry's max useful rank and **sliced
  per σ** at apply time (rank resolved from `sigma, tol`), so one cached basis serves all
  bandwidths. Cache key: geometry (via `_state_version`) + `(tol, dense_fraction)`.

## 5. Per-mode normalization under truncation

Truncation makes clipping **load-bearing** (real negative lobes, not round-off) and we do not
materialize the matrix, so — exactly like NLD's `diffuse` + `to_density` — we clip and
renormalize the **output field** (`n`-vector per column), never the kernel:

1. apply `H` → **clip result `≥ 0`** → **renormalize mass per `W`-component** (component-local
   modes make this leak-free; the constant mode conserves each component's mass).
2. per-mode, via normalization vectors computed **once per σ** (`r = H@1` row sums,
   `m = Hᵀ@M` M-weighted column sums):
   - `average(F)   = (H @ F) ⊘ r`            (row-stochastic; intensive)
   - `transition(F) = Hᵀ @ (F ⊘ r)`          (column-stochastic; extensive, conserves Σ)
   - `density(c)   = H @ (c ⊘ m)`            (M-weighted columns integrate to 1)

All three reduce to `H`/`Hᵀ` applications to `F` and to the vectors `{1, M}`. This reproduces
`heat_kernel_from_W`'s three-mode contract ([ops/diffusion.py:104-173](../../../src/neurospatial/ops/diffusion.py))
without `(n,n)`. The masked `H`-average for `binned`/`resample` (Phase 2, `smooth(v·valid)/
smooth(valid)`) composes on top by calling `average` twice.

## 6. Consumer routing + `compute_kernel` fate

- **Keep `env.compute_kernel`** (dense-matrix return) for power users / backward-compat — it
  **materializes** from the cached eigenbasis on demand (`Q (e^{-tΛ} …) Qᵀ` with the M-powers,
  then the existing `heat_kernel_from_W` clip+normalize). Same result; a huge-grid caller who
  explicitly asks for the matrix still pays `O(n²)` (their choice). **No public API break.**
- **New matrix-free method `env.diffuse(fields, bandwidth, *, mode)`** — the apply-path (§3, §5),
  batched over `fields`. Route the hot paths through it: `env.smooth`
  ([fields.py:150-299](../../../src/neurospatial/environment/fields.py)), `_diffusion_kde`
  (+ batch + JAX, [encoding/_smoothing.py:569,729,847,934](../../../src/neurospatial/encoding/_smoothing.py)),
  and `resample_field(method="diffuse")` ([ops/binning.py:790-815](../../../src/neurospatial/ops/binning.py))
  call `env.diffuse` instead of `compute_kernel(...) @ field`. Same numbers, no `(n,n)`.
- `transitions(method="diffusion")` still needs the row-stochastic matrix — it materializes via
  `compute_kernel` (it is inherently a matrix, not an apply); unchanged.

## 7. Behavior-preservation gate + testing

PR2 is an optimization, so the gate is **output equivalence to the shipped dense operator**:

| Test | Asserts |
| --- | --- |
| `test_apply_matches_dense_full_rank` | `env.diffuse` == `_raw_heat_operator` dense path within `rtol=1e-8` at full rank (all modes) |
| `test_apply_matches_dense_truncated` | truncated apply == full-rank apply within the truncation tol (~`1e-6`) |
| `test_compute_kernel_unchanged` | `compute_kernel` (now eigenbasis-materialized) matches the pre-PR2 dense kernel within `rtol=1e-8`; **all existing Phase 1/2 diffusion tests pass unchanged** |
| `test_grid_independence_preserved` | measured σ == `bandwidth` still holds (regression from Phase 1) |
| `test_no_leakage_truncated` | point source beside a wall: 0 mass across it **under truncation** (component-local modes) |
| `test_eigenbasis_cached_and_invalidated` | basis built once, reused across σ/mode; rebuilt after a geometry change (`_state_version` bump) |
| `test_perf_large_grid` (slow) | baseline-capture the old dense `expm` time/mem on a ~10k-bin grid, then assert the apply-path's reduction (time and peak memory) |

## 8. Risks / open items

- **`eigsh` fragility** — shift-invert on the near-singular `S` can fail; adopt NLD's
  deterministic-`v0` + no-shift-invert fallback and a warning.
- **M-conjugation precision** — `M^{±1/2}` scaling on ill-conditioned `volumes` (tiny sectors
  near polar `r=0`); verify the full-rank equivalence test on polar/mesh specifically.
- **float32/JAX paths** — the encoding JAX consumers pass through `env.diffuse`; ensure the
  apply-path is dtype-stable (the eigenbasis is float64; cast at the boundary).
- **Perf claim is measured, not assumed** — `test_perf_large_grid` captures a real baseline.

## 9. References

- NLD spectral engine: `~/Documents/GitHub/non_local_detector/.../likelihoods/diffusion.py`
  (`diffusion_eigenbasis`, `heat_kernel_rank`, `_adaptive_heat_kernel_basis`,
  `cached_eigenbasis`, `cached_heat_kernel_eigenbasis`, `diffuse`, `to_density`).
- The seam: `ops/diffusion.py::_raw_heat_operator` + `heat_kernel_from_W`.
- Memory: [[nld-diffusion-mrf-learnings]], [[diffusion-kde-grid-dependent-bandwidth]].
