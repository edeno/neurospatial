# Design — MRF-GAM place-field estimator (penalized-Poisson GLM)

**Status:** approved design, revised after a repository-contract review (population API,
disconnected components, param defaults, active-bin storage, rank cache, degenerate data,
rename migration, diagnostic shapes). Pre-implementation.
**Scope:** item (c) of the diffusion-engine roadmap — a new **estimator** on the spatial-rate
functions, ported from `non_local_detector`'s `sorted_spikes_mrf`.
**Depends on:** the shipped PR2 eigenbasis (`ops/diffusion.py`, v0.8.0). Verified against
main (v0.8.0) 2026-07-10.
**Reference:** `~/Documents/GitHub/non_local_detector/.../likelihoods/sorted_spikes_mrf.py`;
memory [[nld-diffusion-mrf-learnings]].
**Target release:** 0.9.0 (adds `method="glm"` + `pooled` on `compute_spatial_rate(s)`;
**hard-renames** `smoothing_method → method` across **all** smoothing encoders).

---

## 1. Problem

The shipped estimator (`compute_spatial_rate(s)`, all methods) is a **ratio**:
`rate = smoothed_spikes / smoothed_occupancy` — NaNs at low/zero occupancy, hand-set
smoothness, not a likelihood. A penalized-Poisson GAM fixes all three, and its smoothness
penalty **is** the diffusion energy, so it reuses PR2's cached eigenbasis.

## 2. Goal / non-goals

**Goal:** a `method="glm"` estimator on **both** `compute_spatial_rate` (singular) and
`compute_spatial_rates` (plural — the batched population entry the decoder uses), fitting a
penalized-Poisson GAM place field: occupancy is a **log-offset (exposure), never a
denominator**; `λ` chosen by **REML**; proper likelihood (deviance). NumPy/SciPy core (base
install) + optional JAX accel. **`λ` is shared across the population by default (`pooled=True`);
a per-neuron `λ_k` option (`pooled=False`) is the final in-scope capability** (§5 — implemented
last so the shared-λ core lands and is reviewed first).

**Scope note (supersedes earlier deferrals).** Two items this design earlier deferred are now
**in scope**, per project direction: (a) the `smoothing_method → method` rename is applied to
**ALL** smoothing encoders (spatial, view, egocentric + their result classes), so `method` is
uniform across the API with no cross-result-class inconsistency; (b) **per-neuron λ
(`pooled=False`)** ships as the final phase, not a post-release follow-up.

**Non-goals (this PR):**
- Replacing the ratio estimators (default stays `method="diffusion_kde"`).
- **Adding `glm` to the *other* encoders** (`compute_view_rate`, `compute_egocentric_rate`, …) —
  only `compute_spatial_rate(s)` gain the estimator; the others are renamed for uniformity but
  keep their ratio-only `Literal`.
- Clusterless/mark-space; changing the operator/eigenbasis.

## 3. Model

Active interior bins `i` (`0..n_bins-1`, **active bins only** — `env.n_bins`,
[core.py:1030](../../../src/neurospatial/environment/core.py)), neurons `k`. Spike counts
`n_ik` (pixellated to active bins), occupancy `o_i` (seconds; the exposure). Reduced-rank
basis `B` (`n_bins × rank`), penalty weights `d` (`rank`):

```
n_ik ~ Poisson( μ_ik ),  μ_ik = o_i · exp(η_ik),  η_k = B γ_k
penalty_k = (λ/2) · γ_kᵀ diag(d) γ_k
```

- `B = M^{-1/2}Q`, `d = Λ` — the generalized finite-volume Laplacian eigenpair
  `(D−W)v = λMv`, from PR2's `S = M^{-1/2}(D−W)M^{-1/2}` eigensolver (§6.1). Penalizing
  `γᵀdiag(d)γ` **is** the **M-weighted** diffusion energy `ηᵀ(D−W)η = ηᵀMLη` — note the mass
  matrix (`L = M⁻¹(D−W)`, so `ηᵀLη` alone is wrong); `ηᵀ(D−W)η = γᵀBᵀ(D−W)Bγ = γᵀΛγ`. The `M`
  matters on nonuniform-volume polar/mesh layouts.
- **One unpenalized null mode PER connected component** (not one globally): a fragmented mask
  has `n_components` zero-eigenvalue modes, each an intercept for its component, all with
  `d=0` (unpenalized). Consequences propagate to the rank (§6.1) and the REML rank term (§5).
- **Place field = `max(exp(η_k), _RATE_FLOOR)`** (Hz; the floor and dead-component fallback are
  defined in §7), stored in **active-bin order, shape `(n_bins,)`** — NOT a full regular grid
  (neurospatial rate arrays are active-bin; full-grid would break result methods/decoding and is
  undefined for mesh/graph). Plotting scatters into a grid. `firing_rate` composes with
  `decode_position` exactly like the ratio estimator's.

## 4. Fit (NumPy/SciPy core; optional JAX)

Penalized-Poisson **IRLS/Newton batched over the neuron axis** (shared `B`, `o`; per-neuron
`n_k`, `γ_k`):

```
η = B γ ; μ = o[:,None]·exp(clip(η))
grad_k    = Bᵀ(n_k − μ_k) − λ d ⊙ γ_k
hessian_k = Bᵀ diag(μ_k) B + diag(λ d) + jitter·I
step_k    = solve(hessian_k, grad_k)                 # np.linalg.solve batches over k
```

- **Step-halving** for monotone descent of the penalized objective.
- **Converge on the penalized deviance, NOT the coefficient step** — the unpenalized
  per-component null modes drift forever in undersampled arenas, so `max|Δγ|` never settles
  though `exp(η)` has (NLD's documented trap).
- Warm-start from a constant log-rate; the linear predictor is **clipped to `±_ETA_CLIP` before
  `exp`** to avoid IRLS overflow; float64 core. Optional JAX path (`_core_jax` pattern, `jit`,
  batched) runs float32 (rate error `~1e-7` vs the float64 reference) and is verified to match
  NumPy within tolerance.
- **Solver / REML controls are FIXED internal constants** (not public params — keep the `glm`
  API surface to `penalty`/`rank`), named module constants matching the reference so
  implementations and tests can't diverge:
  - `max_iter=100`, `_LOG_PENALTY_BOUNDS=(-8.0, 20.0)`, `reml_xatol=1e-3`.
  - `_ETA_CLIP=30.0`, `_MAX_STEP_HALVINGS=30`, `_HESSIAN_JITTER=1e-10` (ridge on the Hessian
    diagonal).
  - **Convergence tolerance is dtype-dependent.** The fit converges on the **relative
    penalized-objective decrease** with tol `1e-10` on the float64 core, but the float32 JAX path
    **floors it to `_FIT_TOL_FLOOR=1e-6`** (`tol = max(tol, 1e-6)`) and uses a step-halving
    descent slack **`_DESCENT_TOL=1e-5`** — `1e-10` sits below float32 objective noise (`~1e-7`),
    so without the floor convergence never triggers and halving fires on rounding noise
    ([sorted_spikes_mrf.py:85-89](../../../../non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py),
    [sorted_spikes_mrf.py:257](../../../../non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py)).
- **Final-fit nonconvergence WARNS (does not raise).** `max_iter` is internal, so a user cannot
  raise it; silently returning `converged=False` in a diagnostic field would let unreliable place
  fields pass unnoticed. So warn on the **returned convergence flag — `if not converged: warn(...)`
  — not on `n_iter == max_iter`** (a fit can satisfy the tolerance on its final permitted
  iteration; the reference triggers from the flag). `n_iter` is diagnostic context in the message.
  Emit a `UserWarning` (matching the reference, [sorted_spikes_mrf.py:865-875](../../../../non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py))
  that reports `n_iter` + the max coefficient step and recommends the **available** remedies —
  **reduce `rank`** (fewer, better-conditioned modes) or **supply / increase `penalty`** (a fixed
  positive λ stabilizes the Hessian). Fields are still returned (`converged=False` stays on the
  result for programmatic checks); it is a warning, not a hard failure. (A REML *total* failure —
  no finite objective in the λ interval, §5 — still **raises**; that is distinct from Newton
  nonconvergence at a chosen λ.)
- **Degenerate data** — see §7.

## 5. λ selection — REML (single shared λ default; per-neuron `pooled=False` option)

Default: one shared `λ` by **REML** (negative Laplace-approximate restricted marginal
likelihood, Wood 2011). The df term is **per-neuron**, so the pooled objective is

```
Σ_k [ −loglik_k + penalty_k − ½·r·log λ + ½·log|H_k| ]
  =  Σ_k[−loglik_k + penalty_k]  −  ½·n_units·r·log λ  +  ½·Σ_k log|H_k|
```

(`r = penalty_rank`). **The `n_units` factor is required** — placing `−½ r log λ` once (outside
the sum) selects a population-size-dependent, mis-scaled λ. Minimized over `log λ` by
`scipy.optimize.minimize_scalar(method="bounded")`; `log|H_k|` via `scipy.linalg.cholesky`
(**`+inf` for a non-PD `H`** so the search rejects it, and raise a clear error if no finite
objective exists in the interval).

- **`penalty_rank r = r_eff − n_live_components` (structural), with the per-component null
  weights zeroed exactly.** The finite-volume operator has **exactly one null (constant) mode per
  `W`-component**, and the resolver keeps only live-component modes (§6.1), so the retained live
  basis contains **exactly `n_live_components` structural nulls**. **Set those designated
  per-component null `d` entries to exactly `0`** (they are the component constant eigenvectors —
  intercepts, which must be unpenalized), then `r = r_eff − n_live_components`.
  - **Why not the reference's relative rule.** NLD's `_penalty_rank = count(d > 1e-12·max(d))`
    ([sorted_spikes_mrf.py:436](../../../../non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py))
    needs at least one genuinely positive eigenvalue to set the scale. In an **all-null truncated
    basis** `max(d)` is itself noise (`eigh` on a 3-node path gives a null of `~4e-17`), so
    two such components at `rank=2` count as `r=2` — bypassing the `r==0` REML-skip and returning
    an arbitrary λ. The structural identity gives `r = 2 − 2 = 0` correctly.
  - **Structural guarantee (by construction, not a recount).** The basis is built as
    `[n_live_components intercepts (d=0) | n_fill smoothness modes]`, where the fills are the
    smallest-`Λ` live modes **selected by strict positivity** `Λ > _NULL_TOL` (the eigensolver
    clips negatives to 0, so nulls sit at `0..~1e-15 < _NULL_TOL`). Hence `d` has **exactly
    `n_live_components` zeros and all remaining entries strictly `> 0`** — never an extra zero — so
    `r = r_eff − n_live_components` equals the true penalty rank and the REML df term is exact. A
    relative `count(d > 1e-12·max(d)) == r` check is **deliberately avoided** (it false-fails a
    genuine low-frequency mode on a bottlenecked graph); positivity-selection is authoritative.
- **No penalized modes (`r == 0`)**: when `r_eff == n_live_components` (only null modes, e.g. a
  tiny/fragmented live basis), every `d` is a null weight, λ has **no effect**, and the REML
  objective is flat in λ — bounded minimization would return an arbitrary penalty. **Skip REML,
  set `penalty = None`, `reml_objective = None`, and warn** (regression-tested). REML runs only
  when `r > 0`.
- **`penalty=<float>` escape hatch** skips REML (incl. `0` = unpenalized — with an
  identifiability warning, §7). The result **records the supplied `penalty`** (the model actually
  fitted) and sets only `reml_objective=None`. `penalty=None` in the result is reserved for
  automatic-REML-skipped (`r==0`) and no-data cases (§7).
- **Per-neuron λ (`pooled=False`) — the final in-scope capability**, added last (§2). Default
  stays `pooled=True` (one shared λ, everything above). `pooled=False` selects an independent
  `λ_k` per unit by minimizing **each unit's** REML score (drop the `Σ_k` — the pooled objective
  is already a per-unit sum), then a per-unit final fit at each `λ_k`. **The basis `B`, weights
  `d`, and penalty rank `r` are shared geometry — identical for every unit**; only the counts
  differ. Consequences:
  - **`r == 0` is a population-level property** (shared basis), so REML is skipped for the whole
    population regardless of `pooled`; `penalty` stays a scalar `None` (not a per-unit vector).
  - **A fixed `penalty=<float>` takes precedence over `pooled`**: REML is skipped entirely and the
    result records that **scalar** λ (the model fitted), exactly as `pooled=True`. `pooled` only
    affects the automatic-REML path.
  - When `r > 0` and REML runs, `penalty`/`reml_objective` become `(n_units,)` vectors, but
    per-unit λ is meaningful **only for informative units** (`Σ n_k > 0`). **A zero-spike unit is
    statistically unidentified for per-unit λ** — its intercept → −∞, non-null coeffs → 0, and the
    `r·log λ` terms cancel, so the bounded optimizer's point is driven by clipping/jitter/bounds,
    **not data**. So those units get a **documented fallback: the pooled λ over the informative
    units** (not the optimizer's point), `reml_objective[k] = nan` (sentinel), the field floors,
    and a per-unit boolean **`penalty_selected_by_reml`** marks them (`False`) so a persisted λ is
    never mistaken for a unit estimate (§6.3, §8).
  - **If NO unit is informative** (all-zero-spike population), the pooled objective is *equally*
    unidentified — so `pooled=False` **does not run pooled REML**; it takes the shared
    **all-zero-spike degenerate path** (§7): scalar `penalty=None`, `reml_objective=None`, floor
    fields. (`r==0` likewise → scalar `None`.)
  - Shapes: shared-λ (`pooled=True`) keeps scalar `penalty`/`reml_objective` exactly as today;
    only `pooled=False` with automatic REML **and ≥1 informative unit** widens them to vectors
    (§6.3, §8).

## 6. Eigenbasis access + API

### 6.1 Rank-based, live-component eigenbasis resolver (new)

PR2's `_symmetric_eigenbasis(S, rank)` builds the eigenpair; the env cache `_diffusion_eigenbasis`
([fields.py:536](../../../src/neurospatial/environment/fields.py)) is keyed by **`(sigma, tol)`**
(not rank), and its selection is **global** across components
([diffusion.py:445](../../../src/neurospatial/ops/diffusion.py)). Two things the resolver must
handle:

1. **Rank-based access.** Add a rank-based entry (`env`-level, `versioned_cached_property`-
   invalidated) that builds `S` from the geometry (`_symmetric_conjugate`) and requests the
   geometry eigenbasis at a rank, reusing/growing the truncated cache (it holds a max-rank basis
   grown by replace); a request with `rank ≥ dense_fraction·n` follows the **transient dense-`eigh`**
   policy. The eigensolver **raises** on `rank < n_components`
   ([diffusion.py:419](../../../src/neurospatial/ops/diffusion.py)), so any request floors at
   `n_components`.
2. **Live-component selection BEFORE the rank budget is spent.** Modes are **component-local**;
   global smallest-`rank` selection would spend budget on **dead** (unvisited, `Σo=0`) components
   — each contributes a null + low modes — leaving `< R` *useful* modes and making fit quality
   depend on an unused region. So identify **live** components (`Σo>0` per `W`-component) from
   occupancy, **keep only live-component modes**, and **over-request** from the (still-cached,
   PR2-reused) geometry basis until `r_eff = max(n_live_components, min(n_live_bins, R))` live
   modes are retained (`R` the cap, default 250). The **effective live basis** is those modes
   restricted to live bins: `B = (M^{-1/2}Q)[live_bins, live_modes]`, `d = Λ[live_modes]`. **`rank`,
   `coefficients`, and `penalty_weights` are all defined on this live basis**, and the fit runs on
   `live_bins` only. **Zero the designated per-component null weights exactly** — one `λ≈0`
   constant mode per live component — so `d` has exactly `n_live_components` structural zeros and
   `penalty_rank = r_eff − n_live_components` is exact (§5, Finding 1); the intercept directions
   carry zero penalty. The geometry eigenbasis stays cached (reuses PR2); the MRF selects a subset.
   Returns `(B, d, live_bins, n_live_components)`. **Perf caveat** (§11): over-requesting against
   the *global* basis can approach a near-full eigendecomposition when a large dead component
   carries many low modes; the component-local-eigenpair mitigation is the fallback if the `R` cap
   stops holding.

### 6.2 `compute_spatial_rate` / `compute_spatial_rates`

**Hard-rename `smoothing_method → method`** on both, values `{"diffusion_kde", "gaussian_kde",
"binned", "glm"}` (one estimator axis). No keyword/property alias (project default). **Method
params default `None` (sentinel)** so validation can distinguish omitted from explicit:

- Ratio methods resolve `bandwidth` (default `5.0` when `None`), `min_occupancy` (default
  `0.0`), `fill_value`. `glm` resolves `penalty` (`None`→REML) and `rank` — the **requested cap
  `R` on the live basis** (`None`→`250`). The **effective retained rank** is
  `r_eff = max(n_live_components, min(n_live_bins, R))` (§6.1, computed on **live** components,
  not the global grid), and `result.rank` reports `r_eff`.
- **Mutually-exclusive validation**: an *explicitly set* ratio param with `method="glm"` (or
  `penalty`/`rank` with a ratio method) raises `ValueError` naming which param applies to which
  method. (Sentinel `None` = "not set", so a bare `method="glm"` doesn't false-trip on the old
  `bandwidth=5.0` default.)
- **Value-domain validation** (mirrors the reference's dedicated validators,
  [sorted_spikes_mrf.py:99,128](../../../../non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py)):
  - `penalty` (when set): a **finite, non-negative scalar**; **reject `bool`** (`isinstance(x,
    bool)` before the numeric coercion — `True`/`False` are ints in Python), reject `NaN`/`inf`
    and negatives. `ValueError`.
  - `rank` (when set): a **positive integer**; **reject `bool`**, reject non-integral floats
    (`int(x) != x`) and `< 1`. `ValueError`.
  - **`rank` is clamped both ways, never rejected for magnitude** (`r_eff = max(n_live_components,
    min(n_live_bins, R))`, §6.1). **Above** available live bins → capped at `min(n_live_bins, R)`.
    **Below `n_live_components`** → raised to `n_live_components`, so **every live component keeps
    its intercept** (an unpenalized null mode per component; a smaller basis would drop components
    entirely). Either clamp is **reported** via `result.rank == r_eff` (the field the user reads
    back). Both directions documented in the `rank` docstring.

### 6.3 Result classes + diagnostic shapes

`SpatialRateResult` / `SpatialRatesResult`: rename field `smoothing_method → method`; make
`bandwidth: float | None` (**`None` for `glm`**). Add optional GAM fields (`None` for ratio):

| field | plural (`SpatialRatesResult`) | singular (index `i`) |
| --- | --- | --- |
| `coefficients` γ | `(rank, n_units)` | `[:, i]` → `(rank,)` |
| `penalty` λ | scalar (shared) — supplied or REML-selected; `None` only when no penalty applied (§7) | same scalar |
| `penalty_weights` d | `(rank,)` | same |
| `rank` | int (effective live rank `r_eff`, §6.1) | int |
| `deviance` | `(n_units,)` | `[i]` |
| `converged` | **scalar** bool | same scalar |
| `n_iter` | **scalar** int | same scalar |
| `reml_objective` | scalar, or `None` whenever REML did not run (fixed λ, `r==0`, no-data) | same |
| `pooled` | `bool` — the input `pooled` flag; **persisted** as the only reliable NWB source (§8, cannot be inferred from scalar outputs) | same |

`converged` and `n_iter` are **batch-level scalars**, not per-unit: the batched Newton uses one
shared stopping criterion (the `max` relative-objective decrease across neurons, matching the
reference), so there is a single convergence flag and iteration count. (Per-unit convergence
would require a redesign that freezes converged units while others continue — out of scope.)

**`pooled=False` (§5) widens two fields only:** `penalty` and `reml_objective` become `(n_units,)`
vectors (per-unit λ_k / score) — but still scalar `None` when `r == 0` (population-level skip).
`converged`/`n_iter` stay batch scalars; all other fields keep their shared-λ shapes. Under the
default `pooled=True` nothing changes.

**`deviance` is the unpenalized Poisson deviance per unit** — the *model-fit* quantity, distinct
from the penalized-deviance *convergence* objective: `D_k = 2·Σ_i [n_ik·log(n_ik/μ_ik) − (n_ik −
μ_ik)]`, where **`μ_ik = o_i · firing_rate_ik`** (the **stored** rate map, so deviance describes
what is reported — not the unfloored `o·exp(Bγ)`), with `0·log(0/μ)=0`, summed over **exposed**
bins (`o_i > 0`; zero-exposure bins have `μ=0` and are excluded). Tests verify this formula
independently, not just "a value exists".

Indexing/iteration stamps `unit_id` (existing v0.6 behavior). Terminal verbs (`to_dataframe`,
`summary_table`, `peak_locations`, `summary`) unchanged; `summary_table` gains the GAM scalar
columns when present. Composes with `decode_position` unchanged.

## 7. Degenerate data (defined, model-consistent outputs)

**Rate floor.** Define a single module constant **`_RATE_FLOOR = 1e-10`** (matches decoding's
`min_rate`, [likelihood.py:47](../../../src/neurospatial/decoding/likelihood.py), so a floored
place field is consistent with what the decoder expects — not NLD's `1e-15`). `firing_rate` is
`max(exp(η), _RATE_FLOOR)`; note this floor is the **one** place `firing_rate` may differ from
`exp(Bγ)`, and it is documented on the field.

**Dead components are excluded by the live-basis selection (§6.1)** — their bins are not in the
design (`B`, `o`, `n`), so the fit never sees an unidentified intercept and **`coefficients`
cover only live modes**. Post-fit, dead bins are set to `_RATE_FLOOR`; this is the documented
region where `firing_rate` has **no** underlying coefficient (the only break from
`firing_rate == max(exp(Bγ), floor)`). `deviance` is over exposed (live) bins.

The whole-map degenerate cases must keep `firing_rate` **consistent with `coefficients`** — zero
coefficients would give `exp(Bγ)=1`, not the floor, so we do **not** report zero coefficients
there. Per case (all warn, none raise):

| case | `coefficients` | `firing_rate` | `deviance` | `penalty` / `reml_objective` | `converged` |
| --- | --- | --- | --- | --- | --- |
| **no neurons** (`n_units==0`) | `(r_eff, 0)` | `(0, n_bins)` | `(0,)` | `None` / `None` | `True` |
| **zero total occupancy** (no live comp) | **empty** `(0, n_units)` — no live basis, so **no coefficients**; all bins are the floor fallback | `_RATE_FLOOR` everywhere | `(n_units,)` zeros | `None` / `None` | `True` |
| **all-zero-spike population** (`Σn==0` over all units) | fitted (`penalty_diag=(fixed λ or 0)·d`) → low intercept | `_RATE_FLOOR` everywhere | `≈0` | **fixed float if supplied, else `None`** / `None` (REML *selection* skipped; fixed-penalty contract respected — §5) | fitted |
| **dead component** (`Σo_comp==0`) | live modes only (§6.1) | `_RATE_FLOOR` on that comp; `exp(Bγ)`-fitted elsewhere | fitted, exposed bins | fitted / fitted | fitted |
| **zero-spike neuron** (`Σn_k==0`, population has ≥1 informative unit) | fitted (low intercept) | `exp(η)` floored → near-floor | finite | shared (or pooled-λ fallback if `pooled=False`, §5) | shared |

- **`penalty` / `reml_objective` when REML did not run** — `reml_objective` is always `None`
  here. `penalty` records the value **actually applied**: for a **fixed `penalty=<float>`** the
  result reports **that float** (Finding 1 — the model fitted is recorded); `penalty=None` only
  for **automatic-REML `r==0`** (§5) and the **no-data cases** (no neurons / zero total
  occupancy) where no penalty is meaningful.
- **`penalty=0`** (unpenalized) identifiability: warn **iff the exposed live design is
  rank-deficient** — `rank(B[exposed_live_bins, :]) < n_coefficients` (all live modes,
  **including** the component intercepts, not the penalized rank `r`). This is a property of the
  design/exposure, shared across neurons. **Per-neuron finite-optimum failures are NOT part of
  this warning**: a zero-spike unit has no finite intercept optimum even with full-column-rank
  `B`, but that is its own degenerate case (table above) already handled by the `_RATE_FLOOR`
  fallback — folding it in would make a design-level warning fire per-neuron. (Matches the §10
  test: warning ⇔ rank deficiency.)

## 8. Persistence + downstream migration (hard rename)

The `smoothing_method → method` rename touches persistence and the decoder:

- **NWB** ([io/nwb/_fields.py:616,792](../../../src/neurospatial/io/nwb/_fields.py)): write the
  metadata key `"method"` (was `"smoothing_method"`); **bump the encoding-model schema version**.
  Reader reads `"method"` **only** — **no** fallback to the old `"smoothing_method"` key (clean
  break, no back-compat shim; pre-0.9 tables are rejected with a clear error). Write `bandwidth` as
  nullable (`None` for glm). **GAM diagnostics round-trip**, with concrete placement:
  - **Metadata scalars** (batch-level): `rank`, `n_iter`, `converged`, and the `(rank,)`
    `penalty_weights` vector. `penalty` λ and `reml_objective` are metadata scalars **under
    `pooled=True`**; **under `pooled=False` they move to per-unit table columns** (each is
    `(n_units,)`). A reader keys on the stored `pooled` flag (or on whether the value is scalar vs
    vector) to place them.
  - **Per-unit table columns**: `deviance` (`(n_units,)`), `coefficients` as a **fixed-length
    `(rank,)` per-unit vector** column, and (when `pooled=False`) per-unit `penalty` /
    `reml_objective` / **`penalty_selected_by_reml`** (the fallback mask — so a persisted λ is never
    read back as a unit estimate when it was the pooled-λ fallback).
  - Round-trip preserves shapes; a ratio-method result persists these as absent/`None`.
- **Decoder — functional + class paths.** `decode_session` / `decode_session_summary`
  ([decoding/session.py:100,180](../../../src/neurospatial/decoding/session.py)): rename the
  forwarded `smoothing_method → method`; accept `method="glm"` (+ `penalty`/`rank`) through the
  encoding step. **`BayesianDecoder`** ([decoding/estimator.py:151-153](../../../src/neurospatial/decoding/estimator.py))
  stores `bandwidth`/`smoothing_method`/`min_occupancy` as config fields: rename its field
  `smoothing_method → method`, add `penalty`/`rank` fields (nullable), make `bandwidth`/
  `min_occupancy` nullable, add the same **method-specific validation** as §6.2, and forward the
  glm params through both `decode_session` and `decode_session_summary`. So `"glm"` works via the
  functional *and* the class decoder.
- CHANGELOG: the hard rename (breaking, **all smoothing encoders** — spatial/view/egocentric),
  the nullable `bandwidth`, and that `glm` is spatial-only.

## 9. Layouts & composition

All finite-volume layouts (eigenbasis exists per layout; §6.1 handles `n_components`).
Occupancy is the existing `compute_occupancy` exposure. `firing_rate` composes with
`decode_position` / population paths unchanged.

## 10. Testing

- **Zero-occupancy finiteness** — glm gives finite rates (floored at `_RATE_FLOOR`) where the
  ratio estimator NaNs (headline).
- **Disconnected mask** — a fully-visited multi-component mask does not raise; `result.rank ==
  r_eff = max(n_live_components, min(n_live_bins, R))` (live-basis, §6.1); per-component
  intercepts; a **dead (unvisited) component** is dropped pre-fit, its bins → `_RATE_FLOOR` + warn.
- **Live-component rank budget** (§6.1) — with a dead component present, the resolver still returns
  `r_eff` *live-component* modes (a naïve global smallest-rank selection would spend budget on the
  dead component's null mode); assert live-mode count and that every returned column indexes a live
  bin (regression guard).
- **`penalty` records the fitted model** (§5/§7) — a **fixed `penalty=<float>`** fit yields
  `result.penalty == <float>` and `reml_objective is None` (not discarded to `None`); a REML fit
  yields `penalty ==` the selected λ with `reml_objective` finite; no-data / auto-`r==0` yield
  `penalty is None` (regression guard).
- **`r==0` structural rank + REML skip** (§5) — the decisive case: **two 3-node-path components
  at `rank=2`** (an all-null retained basis, per-component null `~4e-17`). The structural identity
  gives `r = r_eff − n_live_components = 2 − 2 = 0` → REML skipped, `penalty=None`, rates finite;
  the **relative rule `count(d > 1e-12·max(d))` would wrongly give `r=2`** and return an arbitrary
  λ (the P1 regression guard). Also assert the `n_live_components` null `d` entries are **exactly
  `0.0`**.
- **`penalty=0` identifiability** (§7) — the warning fires **iff**
  `rank(B[exposed_live_bins,:]) < n_coefficients` (full column rank ⇒ no warn), not on a bin-count
  heuristic.
- **REML pooled scaling** — the selected λ is invariant to duplicating the population (the
  `n_units` df factor is present); a version omitting it shifts λ with `n_units` (regression guard).
- **`deviance` formula** — recompute `2·Σ[n·log(n/μ) − (n−μ)]` (with `0·log0=0`, exposed bins)
  independently and match `result.deviance`; not just "a value exists".
- **Recovers distinct place fields per neuron** (population fit); peaks near simulated centers.
- **REML** recovers a sensible λ; fixed extreme λ → monotone smoothing.
- **Newton convergence** on penalized deviance under step-halving; null-mode drift doesn't stall.
- **Degenerate cases** (§7): zero spikes, zero exposure, no neurons, `penalty=0` warning.
- **Agreement with the ratio estimator** on a well-sampled arena.
- **NumPy vs JAX parity** (within float tol) — and the float32 path actually **converges**
  (`converged is True`, `n_iter < max_iter`), confirming the `_FIT_TOL_FLOOR=1e-6` floor is
  applied; a `tol=1e-10` float32 fit *without* the floor would run to `max_iter` (regression guard).
- **Nonconvergence warns** (§4) — a non-converged fit (**warn keyed on `not converged`, not
  `n_iter == max_iter`**) emits a `UserWarning` naming `n_iter` and recommending reduce-`rank` /
  increase-`penalty`; `converged=False` on the result; fields still returned (no raise). Distinct
  from a REML total-failure **raise**.
- **Param value-domain validation** (§6.2) — `penalty` rejects `True`/`False`, negative, `NaN`,
  `inf`; `rank` rejects `True`/`False`, `2.5`, `0`, `-1`; each a `ValueError`. **`rank` clamps
  both ways without raising**: `rank > n_live_bins` → `result.rank == min(n_live_bins, R)`;
  `rank < n_live_components` → `result.rank == n_live_components` (every component keeps an
  intercept).
- **API**: mutually-exclusive param errors; `method="glm"` doesn't false-trip on default
  `bandwidth`; default `method="diffusion_kde"` unchanged; ratio results have `None` GAM fields.
- **Decoding**: `method="glm"` works end-to-end through **both** the functional path
  (`decode_session` / `decode_session_summary`) **and** the `BayesianDecoder` class (renamed
  `method` field, `penalty`/`rank` forwarded, nullable `bandwidth`/`min_occupancy`).
- **NWB round-trip**: glm result (with GAM diagnostics + `bandwidth=None`) writes and reads back;
  pre-0.9 `"smoothing_method"`-key tables remain rejected (clean break).
- **All layouts** smoke-tested (1D track, 2D open+masked, hex, polar, mesh).
- **`pooled=False`** recovers **distinct** per-neuron λ_k on a population with unit-varying
  smoothness (variance across λ_k > 0); `pooled=True` (default) is scalar-λ and byte-identical to
  the shared-λ result; `r==0` under `pooled=False` still skips REML population-wide (scalar
  `penalty=None`).

## 11. Risks / open items

- **REML robustness** — degenerate regions (over-large rank vs data, many zero-occupancy bins):
  `+inf` on non-PD `H`, error if no finite objective in the interval.
- **Rank cap default (250)** drops the highest-frequency modes on large grids (NLD's perf
  choice); `rank=` overrides; the `max(n_live_components, …)` floor keeps it valid on fragmented
  masks (one intercept per live component).
- **Over-request can defeat the rank cap's perf goal** (Finding 5). The §6.1 baseline grows the
  **global** (still PR2-cached) geometry basis until `r_eff` *live* modes are found. A large
  **dead** component full of low eigenvalues forces the global eigendecomposition toward near-full
  rank just to discard that component's modes — so the effective work can exceed the `R=250` bound
  even though only `r_eff` modes are kept. Mitigation (preferred if it bites): compute
  **component-local eigenpairs for live components only** (per-live-`W`-component `eigsh` at
  `min(R, n_comp_bins)`), which holds the `R` bound regardless of dead mass but needs a
  per-component eigenbasis cache. Baseline is acceptable when the dead fraction is small; revisit
  under the perf-benchmark task.
- **Hard rename** breaks callers passing `smoothing_method=` and reading `.smoothing_method`
  across **all** smoothing encoders (spatial/view/egocentric) and their result classes;
  documented in the CHANGELOG. Pre-0.9 NWB tables do **not** read (clean break, no defensive fallback).
- **float32 JAX vs float64 NumPy** — parity within `~1e-6`; core stays float64.
- **In-flight `min-occupancy-seconds` work** (not on this repo's main; see git note) — if it
  lands first, re-verify the `method` validation's ratio-param semantics and the result-class
  field additions; glm is largely insulated (no `min_occupancy`; only adds fields).

## 12. References

- NLD `sorted_spikes_mrf.py` (`mrf_penalized_poisson_fit`, `mrf_reml_objective`,
  `select_penalty_by_reml`, `_penalty_rank`, the zero-occupancy/empty-population special cases).
- PR2 eigenbasis: `ops/diffusion.py::_symmetric_eigenbasis` / `_symmetric_conjugate`.
- Wood, S. (2011), REML for GAM smoothing-parameter estimation.
