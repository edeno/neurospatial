# Diffusion PR2 — Cached eigenbasis + matrix-free apply-path (Performance) Implementation Plan

**Status:** Not started.

**Goal:** Replace the per-call dense `scipy.expm` in the diffusion operator with a cached,
per-component, bandwidth-aware truncated symmetric eigenbasis plus a new matrix-free **linear**
apply-path `env.diffuse`, so smoothing scales to large/fine grids without materializing an
`(n,n)` kernel — behavior-preserving to within a stated truncation tolerance.

**Architecture:** Eigendecompose the symmetric conjugate `S = M^{-1/2}(D−W)M^{-1/2}`
per `W`-component, cache it on the Environment (geometry-only, `_state_version`-invalidated),
and apply `H = M^{-1/2} Q e^{-tΛ} Qᵀ M^{1/2}` to field batches directly. `env.diffuse` is a
**pure linear operator** (no clip/renorm; the always-kept null mode conserves mass exactly
under truncation), so `env.smooth` on signed fields is preserved; positivity is enforced by
the consumers that need it. `compute_kernel` is **left unchanged** (its existing dense-`expm`
path), so it stays byte-identical, is memory-safe, and never touches the truncated eigenbasis
cache. Full rationale + the M-norm/denominator/cache contracts: **[design-performance.md](design-performance.md)**
(source of truth, §3–§8).

**Tech stack:** `numpy`, `scipy.sparse` / `scipy.sparse.linalg.eigsh` / `scipy.linalg.eigh`,
`networkx`; the existing `ops/diffusion.py` finite-volume engine and the Environment
`versioned_cached_property` invalidation.

**Out of scope:** the MRF-GAM estimator (its own cycle); any change to the operator or its
*results* (correctness shipped in v0.7.0); nonuniform-Cartesian `bin_sizes`.

## Inputs to read first

- [design-performance.md](design-performance.md) — the design (§3 architecture, §4 eigensolver/truncation, §5 linear apply + denominator policy, §6 routing, §7 tests, §8 risks).
- [ops/diffusion.py:65-101](../../../src/neurospatial/ops/diffusion.py) — `_raw_heat_operator` (the dense `expm` seam being replaced); [104-173](../../../src/neurospatial/ops/diffusion.py) `heat_kernel_from_W` (clip+per-mode normalize; extract `_normalize_modes` from here); [176-197](../../../src/neurospatial/ops/diffusion.py) `_components_from_W`; [203-244](../../../src/neurospatial/ops/diffusion.py) `diffusion_kernel` (geometry dispatch to reuse).
- [fields.py:46-178](../../../src/neurospatial/environment/fields.py) — `compute_kernel` (+ `_kernel_cache`, `valid_modes` at :148); [181-352](../../../src/neurospatial/environment/fields.py) — `smooth` (`kernel @ field` at :351, `valid_modes` at :338) — reroute to `env.diffuse`.
- [encoding/_smoothing.py:568-630](../../../src/neurospatial/encoding/_smoothing.py) `_diffusion_kde` (`kernel @ spike_counts`/`kernel @ occupancy` at :604/:607, gate `occupancy_density > occupancy_threshold` at :614), [731-786](../../../src/neurospatial/encoding/_smoothing.py) `_diffusion_kde_batch`, JAX at [:494](../../../src/neurospatial/encoding/_smoothing.py)/[:918](../../../src/neurospatial/encoding/_smoothing.py); [664-729](../../../src/neurospatial/encoding/_smoothing.py) `_binned` (`weights_smoothed > 0` at :723), `_binned_batch` [791-](../../../src/neurospatial/encoding/_smoothing.py).
- [ops/binning.py:600-830](../../../src/neurospatial/ops/binning.py) — `resample_field`; diffuse branch (`valid` at :820, `den > 0` at :825).
- [environment/decorators.py:179-290](../../../src/neurospatial/environment/decorators.py) — `versioned_cached_property` (keys only on `_state_version` at :282); [core.py:290,296,909,1246](../../../src/neurospatial/environment/core.py) — `_state_version`, `_kernel_cache`, the bump, `clear_cache`.
- NLD reference: `~/Documents/GitHub/non_local_detector/src/non_local_detector/likelihoods/diffusion.py` — `diffusion_eigenbasis` (per-component + eigsh shift-invert + deterministic `v0` + no-shift-invert fallback), `heat_kernel_rank` (Weyl adaptive probe), `_require_rank_covers_components`.

## Tasks

**T1 — Baseline capture (run BEFORE the refactor).** A script/test that pickles the shipped
dense operator's outputs on a fixture set — `compute_kernel` matrices and `kernel @ field`
results for {1D track, 2D open field, 2D masked-with-wall, hex, polar, small mesh} × σ ∈
{small, large} × mode ∈ {transition, density, average}, plus signed-field smooths — and records
dense-`expm` wall-time + peak memory on a ~10k-bin grid. Saved to `tests/.../conftest` fixtures
/ a pickle. The equivalence and perf gates compare against this. (Optimization idiom: capture
first.)

**T2 — Symmetric eigenbasis builder (`ops/diffusion.py`).** Add:

```python
def _symmetric_conjugate(W, volumes):
    """S = M^{-1/2}(D-W)M^{-1/2} (symmetric PSD), sparse."""
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt_m = scipy.sparse.diags(1.0 / np.sqrt(volumes))
    return (inv_sqrt_m @ (scipy.sparse.diags(d) - W) @ inv_sqrt_m).tocsr()

def heat_kernel_rank(eigvals_ascending, sigma, tol):
    """Smallest rank keeping every mode with e^{-tλ} >= tol; >= n_components (null modes)."""
    t = sigma**2 / 2.0
    keep = int(np.searchsorted(-np.exp(-t * eigvals_ascending), -tol, side="right"))
    return max(keep, 1)  # per-component call; null mode (λ≈0) always survives e^{-tλ}=1>=tol

def _component_eigenbasis(W, volumes, *, max_rank, tol, dense_fraction):
    """Per-W-component eigenbasis of S, component-local. Returns (Q, Λ, labels).
    Each block: dense eigh if block small / rank>=dense_fraction*n_block, else truncated eigsh
    (deterministic v0; on shift-invert failure fall back to which='SM' with a UserWarning,
    mirroring NLD). Null mode kept per block (assert rank >= 1 per component)."""
    ...  # see design-performance.md §4; mirror NLD diffusion_eigenbasis, adapted to S
```

Adaptive rank uses the Weyl probe (NLD `_adaptive_heat_kernel_basis`): compute a small `k`,
grow toward the `heat_kernel_rank` estimate, stop at `dense_fraction·n`. **Assert `rank ≥
n_components`** (null-mode retention is load-bearing for linearity+mass, §8). Eigenvectors are
component-local (zero outside their block) so truncation cannot leak across walls.

**T3 — Truncated eigenbasis cache on the Environment.** A `_state_version`-invalidated cache
owned by a `versioned_cached_property` (the decorator keys only on `_state_version`,
[decorators.py:282](../../../src/neurospatial/environment/decorators.py), so the property returns
a small mutable holder, not a single array). It holds **one truncated basis** — the max rank
requested so far, **strictly below `dense_fraction·n`** — grown by **replace** (a call needing
more modes recomputes+replaces, evicting the smaller). This is the **only** persistent eigenbasis
cache. A request resolving rank `≥ dense_fraction·n` (a near-full-rank `env.diffuse`) builds a
**transient** dense `eigh` — applied then dropped, never cached (the pathological
light-bandwidth-on-huge-grid case; caching it would hold O(n²)). `compute_kernel` does **not
touch this cache** (T5), so it cannot poison it. Invalidated wholesale on `_state_version` bump,
like `_kernel_cache`.

**T4 — Matrix-free linear apply-path (`ops/diffusion.py` + `env.diffuse`).** Add:

```python
def apply_heat_operator(Q, Lam, volumes, sigma, F, *, mode, transpose=False):
    """H @ F (or Hᵀ @ F) via the eigenbasis; PURE LINEAR, no clip/renorm.
    F is ALWAYS 2-D (n_bins, n_fields); env.diffuse coerces a 1-D field first —
    a 1-D F would make `sqrt_m * F` (with sqrt_m of shape (n,1)) broadcast to (n,n)."""
    t = sigma**2 / 2.0
    coeff = np.exp(-t * Lam)                      # (rank,)
    sqrt_m = np.sqrt(volumes)[:, None]
    if not transpose:                             # H @ F = M^{-1/2} Q diag(coeff) Qᵀ M^{1/2} F
        return (Q @ (coeff[:, None] * (Q.T @ (sqrt_m * F)))) / sqrt_m
    return sqrt_m * (Q @ (coeff[:, None] * (Q.T @ (F / sqrt_m))))  # Hᵀ @ F

# per-mode (r = H@1, m = Hᵀ@M computed once per σ, each a COLUMN vector (n,1) that
# broadcasts over the n_fields axis; each mode LINEAR in F):
#   average(F)    = apply(H, F) / r
#   transition(F) = apply(Hᵀ, F / r)
#   density(c)    = apply(H, c / m)
```

`env.diffuse(fields, bandwidth, *, mode="density")` (new method on the Environment fields
mixin): validate `bandwidth`/`mode`; **coerce a 1-D `fields` to `(n, 1)`** and squeeze the
result back on return (so callers like `env.smooth` can pass a 1-D field); resolve geometry via
the existing `_finite_volume_geometry` ([ops/diffusion.py:247](../../../src/neurospatial/ops/diffusion.py));
get/grow the cached `(Q, Λ)` at `rank_σ`; apply, batched over columns. **No output clip.**
Docstring states the linearity + approximation contract. **Also declare `diffuse` on
`EnvironmentProtocol`** ([_protocols.py:309](../../../src/neurospatial/environment/_protocols.py),
next to `compute_kernel`/`smooth`) so the protocol-cast consumers in T6 type-check.

**T5 — Leave `compute_kernel` UNCHANGED (deviation from spec §6, reconcile the spec).** Keep
`compute_kernel` on its existing dense-`expm` path — `_raw_heat_operator`, `heat_kernel_from_W`,
`_kernel_cache` ([ops/diffusion.py:65-173](../../../src/neurospatial/ops/diffusion.py)). Do
**not** route it through the eigenbasis. Rationale (surfaced in review): materializing
`compute_kernel` from a *cached* full-rank basis would hold both the `(n,n)` eigenvectors **and**
the cached `(n,n)` kernel — **doubling** persistent O(n²) memory vs today — and materializing
from a *transient* full basis has no benefit over the existing `expm` (both O(n³) per new σ,
both cache only the kernel) while introducing ~`1e-10` numerical drift. So `compute_kernel` stays
**byte-identical**, is memory-safe, and — because it never touches the truncated eigenbasis cache
(T3) — cannot poison it. No `_normalize_modes` extraction is needed. **Update spec §6** to state
`compute_kernel` is unchanged (the `env.diffuse` apply-path is the sole new/eigenbasis surface).
`_raw_heat_operator` is retained (compute_kernel uses it, and it is the equivalence oracle for
`env.diffuse` in tests) — no dead parallel path.

**T6 — Route the smoothing consumers to `env.diffuse` + apply the denominator policy (§5).**

- `env.smooth` ([fields.py:351](../../../src/neurospatial/environment/fields.py)): replace
  `compute_kernel(...) @ field` with `env.diffuse(field, bandwidth, mode=mode)`. Linear, **no
  clip**. Update the docstring with the approximation contract (`≥ -tol` relative to dense).
- `_diffusion_kde` (+ `_diffusion_kde_batch` + JAX) **and their outer dispatch**: today
  `smooth_rate_map` / `smooth_rate_maps_batch` ([_smoothing.py:217,399](../../../src/neurospatial/encoding/_smoothing.py))
  precompute `kernel = env.compute_kernel(...)` ([:494](../../../src/neurospatial/encoding/_smoothing.py))
  and thread it via a `kernel=` parameter ([:225,407](../../../src/neurospatial/encoding/_smoothing.py))
  into `_diffusion_kde*` ([:574](../../../src/neurospatial/encoding/_smoothing.py)). **Remove that
  precomputed-`(n,n)` path for the `diffusion_kde` method** — drop the `kernel` parameter from
  `smooth_rate_map`/`smooth_rate_maps_batch`/`_diffusion_kde*` (the eigenbasis cache now provides
  the cross-neuron reuse the arg gave), and replace `kernel @ spike_counts` / `kernel @ occupancy`
  ([:604,607](../../../src/neurospatial/encoding/_smoothing.py)) with
  `env.diffuse(..., mode="density")` — otherwise batch/JAX silently keep materializing `(n,n)`.
  **Mind the axis:** `env.diffuse` takes `(n_bins, n_fields)`, but the batch `spike_counts` is
  `(n_neurons, n_bins)` ([:401](../../../src/neurospatial/encoding/_smoothing.py); today it does
  `(kernel @ spike_counts.T).T`), so pass `spike_counts.T` and **transpose the result back** to
  `(n_neurons, n_bins)`; `occupancy` is `(n_bins,)` → `(n_bins, 1)`. (`gaussian_kde` keeps its
  own dense weight matrix — unchanged; `binned` is the next bullet.) Then the **magnitude gate** — floor
  `max(occupancy_density, 0)` before `> occupancy_threshold`
  ([:614](../../../src/neurospatial/encoding/_smoothing.py)); clip the output density/rate `≥ 0`
  (decode nonnegativity).
- `_binned` (+ batch + JAX): replace the two `env.smooth` calls with `env.diffuse(mode="average")`;
  **strict support gate** — derive `weights_smoothed > 0` support from `_components_from_W` +
  the input valid mask (not the smoothed sign, [:723](../../../src/neurospatial/encoding/_smoothing.py)).
- `resample_field(method="diffuse")` ([binning.py:820-825](../../../src/neurospatial/ops/binning.py)):
  replace the `apply_kernel(compute_kernel(mode="transition"))` with `env.diffuse(mode="average")`;
  W-component support gate for `den > 0`; **numerator not floored** (signed-safe).

**T7 — Migration + docs (ship with the PR).** CHANGELOG: `env.diffuse` added; the cached
eigenbasis + bandwidth-aware truncation (large-grid speed/memory); the `env.smooth`
approximation contract (linear, `≥ -tol` relative to dense); **the removed `kernel=` parameter
on `smooth_rate_map` / `smooth_rate_maps_batch`** — an intentional break (no backward-compat
shim per the project default; the eigenbasis cache now handles cross-neuron reuse), called out
explicitly so callers passing `kernel=` update. Docstrings: `env.smooth`, `env.diffuse`.
Version 0.7.0 → **0.8.0** (`pyproject.toml`). Note the `compute_kernel` result is unchanged.

## Deliberately not in this plan

- **MRF-GAM estimator** — separate design cycle.
- **Any operator/result change** — PR2 is behavior-preserving; the operator (finite-volume `S`)
  and every mode's *contract* are fixed by v0.7.0.
- **Nonuniform-Cartesian `bin_sizes`** — excluded from the physical-σ guarantee (tracked follow-up).
- **`transitions(method="diffusion")` rewrite** — it needs the row-stochastic *matrix*, so it
  keeps materializing via `compute_kernel`; unchanged.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_apply_matches_dense_full_rank` | `env.diffuse(F)` == `compute_kernel-materialized @ F` within `rtol=1e-8` at full rank, all modes, **signed** `F` |
| `test_env_diffuse_is_linear` | `diffuse(a·F1 + b·F2) == a·diffuse(F1) + b·diffuse(F2)` on signed fields (no positivity projection) |
| `test_apply_matches_dense_truncated` | truncated apply == full-rank within tol in the **M-weighted / dense-relative** norm (polar/mesh); mass conserved **exactly** per component |
| `test_compute_kernel_unchanged` | `compute_kernel` is **byte-identical** to pre-PR2 (unchanged dense-`expm` path); **all Phase 1/2 diffusion tests pass** |
| `test_denominator_support_no_spurious_nan` | strict gates (`binned`, resample) use `W`-component support — no spurious `NaN` even where dense `den` tiny; value asserted only where `den ≫ tol`; magnitude gate (KDE ×3 paths) agrees comfortably off-threshold; numerator not floored |
| `test_diffusion_kde_nonnegative` | KDE clips its output ≥ 0; rate matches shipped within tol, nonnegative |
| `test_env_smooth_nonneg_within_tol` | `env.smooth(nonneg)` negatives bounded **relative to dense** (M-weighted), verified polar/mesh; linear on signed |
| `test_grid_independence_preserved` | measured σ == `bandwidth` still holds (Phase 1 regression) |
| `test_no_leakage_truncated` | point source beside a wall → 0 mass across it **under truncation** (component-local modes) |
| `test_cache_grows_with_smaller_sigma` | large-σ then small-σ: small-σ recomputes+**replaces** the single truncated basis (not under-ranked); within tol |
| `test_eigenbasis_single_basis_and_invalidated` | one max-rank truncated basis reused/sliced; replaced on growth; dropped on `_state_version` bump |
| `test_compute_kernel_does_not_poison_apply_cache` | a `compute_kernel` (full-rank) call does **not** grow the truncated `env.diffuse` cache to `(n,n)`; later `env.diffuse` still bounded |
| `test_null_mode_retained` | resolved rank `≥ n_components`; `H_trunc @ 1 == 1` and `Σ_i M_i H_trunc[i,j] == M_j` exactly under truncation |
| `test_perf_large_grid` (slow) | baseline-capture (T1) dense-`expm` time/peak-mem on ~10k bins vs the apply-path; assert the reduction |

## Fixtures

`conftest.py`: reuse the Phase 1/2 geometry fixtures (1D track, 2D open + masked-wall, hex,
polar, well-shaped mesh); add a **signed** test field (for linearity), a **partially-masked**
field (for support gates), and the T1 baseline pickle. The ~10k-bin grid for `test_perf_large_grid`
is synthesized, marked `slow`.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented; "Deliberately not in this plan" honored (no operator/result change, no MRF).
- `env.diffuse` is **linear** (no clip/renorm); positivity only in the named consumers; the null-mode `rank ≥ n_components` assertion is present.
- The truncated apply cache is never grown to `(n,n)`: `compute_kernel` doesn't touch it (unchanged dense-`expm`), and a near-full-rank `env.diffuse` uses a **transient** dense `eigh` (applied then dropped, not cached).
- Validation slice passes; `test_perf_large_grid` marked slow; equivalence tests compare against the T1 baseline, not tautologies.
- Docstrings / test names / module names don't reference this plan or "PR2/phase".
- CHANGELOG + version bump + `env.smooth`/`env.diffuse` docstrings shipped in this PR.
- The dense-`expm` `_raw_heat_operator` path's fate (kept as test oracle vs removed) is explicit — no dead parallel path left ambiguous.
