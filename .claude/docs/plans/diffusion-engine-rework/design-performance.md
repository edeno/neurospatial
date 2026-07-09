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
`rank ~ measure(domain)/σ^d` (`d` = domain dimension: `length/σ` in 1D, `area/σ²` in 2D),
independent of `n_bins`.

## 2. Goal / non-goals

**Goal:** replace the dense `expm` with a cached, per-component, bandwidth-aware **truncated
eigenbasis** and a **matrix-free apply-path** that never materializes `(n,n)`. The apply-path
is a **pure linear operator** equivalent to the shipped dense operator within tolerance (§7);
`env.smooth` on signed fields is preserved, and positivity is a per-consumer concern (§5), not
baked into the smoother.

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
  `O(n·rank)` memory. `Hᵀ @ F` is the same with the `M^{±1/2}` powers swapped. It is a **pure
  linear operator** — no positivity projection (§5) — so `env.smooth` on signed fields is
  preserved exactly (within truncation tolerance).

## 4. Eigensolver + truncation

Mirror NLD's `heat_kernel_rank`/`cached_eigenbasis`
(`~/Documents/GitHub/non_local_detector/.../likelihoods/diffusion.py`), adapted for the
M-conjugation (NLD's Laplacian is mass-free; ours needs `S`):

- **Per-component** decomposition (resolves the truncation-leak fork): decompose each
  `W`-connected-component's block of `S` separately (`_components_from_W` already exists,
  [ops/diffusion.py:176](../../../src/neurospatial/ops/diffusion.py)), so every eigenvector is
  component-local and **truncation cannot leak mass across a wall/hole by construction**.
- **Bandwidth-aware rank.** Keep modes with `e^{-tλ} ≥ tol` (default `tol=1e-6`); rank tracks
  `~measure(domain)/σ^d` (`length/σ` in 1D, `area/σ²` in 2D), not `n_bins`. Resolve adaptively
  (Weyl's law probe) via truncated
  `scipy.sparse.linalg.eigsh`; the constant (λ=0) null mode per component is **always kept**
  (mass conservation).
- **Dense fallback + memory guard.** When the resolved rank ≥ `dense_fraction·n` (default
  `0.5`), use dense `scipy.linalg.eigh` on the block — which **does** materialize an `(n,n)`
  eigenvector matrix (the inherent cost of near-full-rank smoothing: a light bandwidth on a
  large grid genuinely needs most modes). This is guarded by the **same large-matrix
  `UserWarning`** as `compute_kernel` (GB estimate) and then proceeds — never a *silent*
  `(n,n)` allocation. Small components hit this cheaply; a huge near-full-rank request is the
  caller's explicit choice, not a hidden regression of the "no `(n,n)`" goal.
- **Growable single-basis cache (resolves under-ranking without unbounded memory).** A first
  large-σ call needs FEW modes; a later small-σ call needs MORE — so a fixed basis would
  under-rank the later call. The cache holds **one basis per geometry: the max rank requested
  so far.** A call resolves `rank_σ` from `(σ, tol)`; if the cached basis has `≥ rank_σ` modes
  it **slices** (modes ascending), else it recomputes at `rank_σ` and **replaces** the cached
  basis. Keeping every rank (a `{rank → basis}` dict) would make memory `O(n·Σ ranks)`, not the
  promised `O(n·rank)`, since a smaller basis is fully dominated (sliceable) by a larger one —
  so we evict on replace. One entry, grown by replacement, bounds cache memory at
  `O(n·max_rank_seen)`. `versioned_cached_property` keys only on `_state_version`
  ([decorators.py:282](../../../src/neurospatial/environment/decorators.py)), so the property
  **owns this single-entry cache** and drops it wholesale on any geometry change. This is the
  **truncated apply-path cache only**, capped below the dense-fallback threshold
  (`dense_fraction·n`); the **full-rank basis `compute_kernel` needs (§6) lives in a separate
  slot (or is uncached)** so one `compute_kernel` call can **never** grow the `env.diffuse`
  cache to a full `(n,n)` eigenvector matrix and defeat the memory goal.

## 5. Per-mode application — LINEAR, no positivity projection

`env.diffuse` is a **pure linear operator** — required, because `env.smooth` accepts **signed**
fields via a linear `kernel @ field` ([fields.py:347](../../../src/neurospatial/environment/fields.py)),
so a positivity projection would change its result.

**Mass is conserved without clipping or renormalization.** Because the null (λ=0) mode is
**always kept per component** (§4), the truncated operator satisfies `H_trunc @ 1 = 1` and
`Σ_i M_i H_trunc[i,j] = M_j` **exactly** (dropping high-frequency modes never touches the null
mode; `H_trunc` stays M-self-adjoint via the symmetric `exp(-tS_trunc)`). So there is **no
output clip and no mass renormalization** in the apply-path — the only deviation from the dense
operator is the near-lossless truncation (dropped modes contribute ≤ `tol·‖F‖_M` in the
**M-weighted norm** — clean in the eigenbasis; **raw per-bin error carries a volume-conditioning
factor** `κ(M) = sqrt(max vol / min vol)`, worst on polar `r→0` / skewed mesh). This is where
we **diverge from NLD's `diffuse`**, which clips because it diffuses nonnegative densities; ours
must stay linear.

Per-mode, via normalization vectors computed once per σ (`r = H@1` row sums, `m = Hᵀ@M`
M-weighted column sums — each a fixed vector, so every mode is **linear in F**):

- `average(F)   = (H @ F) ⊘ r`      (row-stochastic; intensive)
- `transition(F) = Hᵀ @ (F ⊘ r)`    (column-stochastic; extensive, conserves Σ)
- `density(c)   = H @ (c ⊘ m)`      (M-weighted columns integrate to 1)

All reduce to `H`/`Hᵀ` applied to `F` and to `{1, M}` — no `(n,n)`. This reproduces
`heat_kernel_from_W`'s three-mode contract ([ops/diffusion.py:104-173](../../../src/neurospatial/ops/diffusion.py)).

**Positivity is the consumer's job — the denominator policy.** The shipped dense kernel is
entrywise-clipped ([ops/diffusion.py:161](../../../src/neurospatial/ops/diffusion.py)), so
`kernel @ nonneg ≥ 0`; the un-clipped linear apply-path can leave ≤`tol` negative lobes under
truncation. **A `max(den, 0)` floor does NOT rescue a strict `den > 0` guard** — a tiny-positive
dense denominator flipped negative by a lobe still floors to `0` and fails `> 0`, spuriously
emitting a `NaN`. Two gate types:

- **Strict support gates** (`binned`'s `weights_smoothed > 0`, resample's `den > 0`): derive
  support from the **`W`-component structure**, NOT the smoothed denominator's sign. Within a
  connected `W`-component the heat kernel is entrywise positive, so the dense `den[i] > 0`
  **iff** `i`'s component contains any valid input mass — a boolean that is **exact and
  truncation-proof**. Gate on that (`_components_from_W` labels + the input valid mask), and
  where support holds divide by `max(den, ε)` (`ε` tiny) only to avoid dividing by truncation
  noise. Reproduces the dense support exactly ([_smoothing.py:716](../../../src/neurospatial/encoding/_smoothing.py),
  [binning.py:823](../../../src/neurospatial/ops/binning.py)).
- **Magnitude gates** (`_diffusion_kde`'s `occupancy_density > occupancy_threshold`, threshold
  possibly `> 0`; NumPy + batch + JAX): floor `max(occupancy_density, 0)` and compare to the
  threshold. Robust **away from** the threshold; bins **within ~`tol` of the threshold may
  flip** — part of the stated approximation contract, so the test asserts agreement only for
  comfortably-above/below-threshold bins ([_smoothing.py:611](../../../src/neurospatial/encoding/_smoothing.py)).

The **numerator is NOT floored** — diffuse resampling can carry a **signed** intensive field,
so clipping the numerator would be a behavior change; only the inherently-nonnegative
denominator gets a floor. Separately, where the *result itself* is semantically nonnegative,
the consumer clips it: `_diffusion_kde` clips its smoothed density/rate `≥ 0` (the shipped
clipped-kernel path did, so decode still sees nonnegative rates). `env.smooth` (general linear
smoother) clips nothing.

**Approximation contract (`env.smooth`).** `env.smooth` stays a pure linear operator, so on
nonnegative inputs (counts, occupancy, probability mass —
[fields.py:207](../../../src/neurospatial/environment/fields.py)) it may return
tolerance-level negatives — bounded by `tol·‖field‖_M` in the **M-weighted norm** (raw per-bin
floor can be `~ -tol·κ(M)·max(|field|)` under volume conditioning) — instead of the dense
kernel's exact 0-floor. This is the **stated PR2 approximation**: documented in the `env.smooth`
docstring + CHANGELOG and tested **relative to the dense output** (which absorbs `κ(M)`),
verified on polar/mesh where `κ(M)` is worst — not against a raw `-tol·max` bound. Callers
needing a strict 0-floor clip the result themselves (as the density consumers do).

## 6. Consumer routing + `compute_kernel` fate

- **Keep `env.compute_kernel`** (dense-matrix return) for power users / backward-compat — it
  materializes from the **full-rank** eigenbasis (all modes for the geometry: `Q (e^{-tΛ}) Qᵀ`
  with the M-powers), then normalizes via a **`_normalize_modes(H, volumes, mode)` helper
  extracted from `heat_kernel_from_W`** — splitting its clip + per-mode normalization from its
  internal `_raw_heat_operator` `expm` build ([ops/diffusion.py:159](../../../src/neurospatial/ops/diffusion.py)),
  so the materialized `H` is normalized **without recomputing `expm`**. **Full rank, not
  the truncated basis** — so it equals the shipped dense operator within eigensolve precision
  (`rtol≈1e-8`), *not* merely within the truncation `tol=1e-6`. A huge-grid caller who
  explicitly asks for the matrix pays `O(n²)`/`O(n³)` (same as today; their choice, same
  large-matrix warning). **No public API break, no accuracy change.** Truncation is used
  **only** by the fast `env.diffuse` apply-path. This full-rank basis is held **separately from
  (never replaces)** the truncated apply cache (§4), so calling `compute_kernel` does not
  poison later `env.diffuse` calls with an `(n,n)` basis.
- **New matrix-free method `env.diffuse(fields, bandwidth, *, mode)`** — the apply-path (§3, §5),
  batched over `fields`. Route the hot paths through it: `env.smooth`
  ([fields.py:150-299](../../../src/neurospatial/environment/fields.py)), `_diffusion_kde`
  (+ batch + JAX, [encoding/_smoothing.py:569,729,847,934](../../../src/neurospatial/encoding/_smoothing.py)),
  and `resample_field(method="diffuse")` ([ops/binning.py:790-815](../../../src/neurospatial/ops/binning.py))
  call `env.diffuse` instead of `compute_kernel(...) @ field`. **Same within truncation
  tolerance** (§5 approximation contract; ≤`tol` deviation, ≤`tol` negatives before the
  consumers' floors), no `(n,n)`.
- `transitions(method="diffusion")` still needs the row-stochastic matrix — it materializes via
  `compute_kernel` (it is inherently a matrix, not an apply); unchanged.

## 7. Behavior-preservation gate + testing

PR2 is an optimization, so the gate is **output equivalence to the shipped dense operator**:

| Test | Asserts |
| --- | --- |
| `test_apply_matches_dense_full_rank` | `env.diffuse(F)` == `compute_kernel-materialized @ F` within `rtol=1e-8` at full rank, all modes, on **signed** `F` |
| `test_env_diffuse_is_linear` | `diffuse(a·F1 + b·F2) == a·diffuse(F1) + b·diffuse(F2)` on signed fields — **no positivity projection** (guards the linearity contract) |
| `test_apply_matches_dense_truncated` | truncated apply == full-rank apply within the truncation tol (~`1e-6`); mass conserved **exactly** per component (null mode kept) |
| `test_compute_kernel_full_rank_exact` | `compute_kernel` (now **full-rank** eigenbasis-materialized) matches the pre-PR2 dense kernel within `rtol=1e-8`; **all existing Phase 1/2 diffusion tests pass unchanged** |
| `test_diffusion_kde_nonnegative` | the density consumer clips its own output ≥ 0 after `env.diffuse`; smoothed rate matches the shipped KDE within tol and is nonnegative |
| `test_denominator_support_no_spurious_nan` | under truncation: **strict** support gates (`binned`, diffuse-`resample`) use **`W`-component support** — exact vs dense, **no spurious `NaN`** even where the dense `den` was tiny-positive (a `max(den,0)` floor would still fail `>0`); the **magnitude** gate (`_diffusion_kde` NumPy+batch+JAX) agrees with dense for bins comfortably above/below `occupancy_threshold`; **numerator not floored** (signed-safe for resample) |
| `test_env_smooth_nonneg_within_tol` | `env.smooth` on a nonnegative field: negatives bounded **relative to the dense output** (M-weighted `tol`), verified on **polar/mesh** where `κ(M)` is worst — not a raw `-tol·max` bound; stays linear on signed fields |
| `test_grid_independence_preserved` | measured σ == `bandwidth` still holds (regression from Phase 1) |
| `test_no_leakage_truncated` | point source beside a wall: 0 mass across it **under truncation** (component-local modes) |
| `test_cache_grows_with_smaller_sigma` | a large-σ call then a small-σ call: the small-σ call is **not under-ranked** — the single cached basis is recomputed+**replaced** at the larger rank (the smaller one evicted); result within tol of dense |
| `test_eigenbasis_single_basis_and_invalidated` | one max-rank basis reused/sliced across σ/mode (not an accumulating dict); replaced on growth; dropped wholesale after a geometry change (`_state_version` bump) |
| `test_compute_kernel_does_not_poison_apply_cache` | a `compute_kernel` (full-rank) call does **not** grow the truncated `env.diffuse` cache to `(n,n)`; a subsequent `env.diffuse` still uses the bounded truncated basis |
| `test_perf_large_grid` (slow) | baseline-capture the old dense `expm` time/peak-mem on a ~10k-bin grid, then assert the apply-path's reduction |

## 8. Risks / open items

- **Null-mode retention is load-bearing** — exact mass conservation *and* linearity without a
  renorm (§5) both depend on the λ=0 mode being kept in **every** component under truncation.
  The rank resolver must assert `rank ≥ n_components` (as NLD does) and never drop a null mode.
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
