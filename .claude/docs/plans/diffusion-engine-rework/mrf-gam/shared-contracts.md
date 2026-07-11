# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts referenced by ≥2 phases. Each lives here once; phases link in by anchor.
Rationale for each is in the spec ([../design-mrf-gam.md](../design-mrf-gam.md)); this
file is the *implementation* contract (names, shapes, signatures).

- [The `method` parameter + validation](#method-param) — phase-0, phase-3, phase-4.
- [Module constants](#constants) — phase-2, phase-3, phase-5.
- [Resolver return: `MRFBasis`](#mrfbasis) — phase-1, phase-2, phase-3, phase-6.
- [Fit return: `MRFFit`](#mrffit) — phase-2, phase-3, phase-5, phase-6.
- [Result-class GAM fields](#result-fields) — phase-3, phase-4, phase-6.

---

## <a id="method-param"></a>The `method` parameter + validation

**One estimator axis.** `smoothing_method → method` is a hard rename (no alias) on every
smoothing encoder and result class. Values:

- `compute_spatial_rate` / `compute_spatial_rates`: `Literal["diffusion_kde", "gaussian_kde", "binned", "glm"]`, default `"diffusion_kde"`.
- All other renamed encoders (`compute_view_rate(s)`, `compute_egocentric_rate(s)`, `detect_place_fields`, `is_place_cell`, `compute_directional_place_fields`): `Literal["diffusion_kde", "gaussian_kde", "binned"]` — **no `"glm"`**. Egocentric default stays `"binned"`.

**Method-param sentinel.** On `compute_spatial_rate(s)`, the method-specific params default to
`None` (sentinel) so validation distinguishes "omitted" from "explicit":

```python
method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
*,
bandwidth: float | None = None,        # ratio methods; resolves to 5.0
min_occupancy: float | None = None,    # ratio methods; resolves to 0.0
fill_value: float | None = None,       # ratio methods
penalty: float | None = None,          # glm; None → REML
rank: int | None = None,               # glm; None → 250 (requested cap R)
```

**Validation (spec §6.2), in this order:**

1. **Mutual exclusivity** — an *explicitly set* (`not None`) ratio param with `method="glm"`, or `penalty`/`rank` with a ratio method, raises `ValueError` naming which param belongs to which method. (Sentinel `None` = "not set", so bare `method="glm"` does not false-trip on a resolved `bandwidth`.)
2. **Value domains** (only when set) — mirror the reference validators (appendix.md):
   - `penalty`: reject `bool` (`isinstance(x, bool)` **before** numeric coercion), then require finite and `≥ 0`. Else `ValueError`.
   - `rank`: reject `bool`, then require integral (`int(x) == x`) and `≥ 1`. Else `ValueError`.
   - `rank` is **clamped, never rejected, for magnitude**: effective `r_eff = max(n_live_components, min(n_live_bins, R))`. Both clamps are reported via `result.rank`.

**Do not weaken:** the `bool`-before-numeric check is load-bearing (`True`/`False` are `int` in Python). The mutual-exclusivity check keys on `is None`, not on value.

---

## <a id="constants"></a>Module constants

Named module-level constants (spec §4, §7). Home: `encoding/_glm.py` for the **fit** constants
below. **Fixed, not public params** — the glm API surface is only `penalty`/`rank`.

```python
_RATE_FLOOR = 1e-10          # matches decoding min_rate (likelihood.py:47)
_MAX_ITER = 100              # Newton iterations
_FIT_TOL = 1e-10             # float64 relative penalized-objective decrease
_FIT_TOL_FLOOR = 1e-6        # float32 floor: max(tol, this) on the JAX path
_DESCENT_TOL = 1e-5          # float32 step-halving "objective increased" slack
_ETA_CLIP = 30.0             # clip linear predictor before exp
_MAX_STEP_HALVINGS = 30      # per Newton iteration
_HESSIAN_JITTER = 1e-10      # ridge on the Hessian diagonal
_LOG_PENALTY_BOUNDS = (-8.0, 20.0)   # REML search bounds in log λ
_REML_XATOL = 1e-3           # scipy.optimize.minimize_scalar xatol
```

**Resolver constant — canonical home is `ops/diffusion.py`, not `_glm.py`.** `_DEFAULT_MAX_RANK
= 250` (the requested-rank cap `R` when `rank=None`) lives in `ops/diffusion.py` because the
pure `select_live_basis` there returns `MRFBasis` and must not import from the higher encoding
tier. Phase 2/3 **import** it from `ops.diffusion` rather than redefining it, so there is one
source of truth. There is **no `_NULL_TOL`**: the null-vs-fill split is done **structurally**
(each live component's constant null is excluded by overlap with its mass-weighted constant, see
`_null_mode_mask`), never by an absolute eigenvalue cutoff — a cutoff would be coordinate-scale-
dependent (Laplacian eigenvalues carry units of `1/length²`, so a genuine smoothness mode can be
positive yet below any fixed tolerance).

The float64 NumPy core uses `_FIT_TOL`; the float32 JAX path uses `max(_FIT_TOL, _FIT_TOL_FLOOR)`
and `_DESCENT_TOL` (spec §4 — `1e-10` is below float32 objective noise `~1e-7`).

---

## <a id="mrfbasis"></a>Resolver return: `MRFBasis`

Produced by the phase-1 resolver; consumed by the phase-2 fit. `NamedTuple` (mirrors the
project's `DiffusionGeometry`), all arrays float64, **live-bin order**:

```python
class MRFBasis(NamedTuple):
    B: NDArray[np.float64]          # (n_live_bins, r_eff) — [ intercepts | M^{-1/2}Q fill ][live_bins]
    d: NDArray[np.float64]          # (r_eff,) — [ n_live_components zeros | fill-mode eigenvalues ]
    live_bins: NDArray[np.intp]     # (n_live_bins,) — indices into the active-bin array (env.n_bins order)
    n_live_components: int          # count of W-components with Σ occupancy > 0
```

`B` and `d` are float64 in live-bin order; `live_bins` (np.intp) maps that order back to
`env.n_bins` order.

**Invariants (do not weaken):**

- **Column layout is fixed:** `B[:, :n_live_components]` are the **intercepts**, `B[:, n_live_components:]` the **fill** (smoothness) modes; `d[:n_live_components] == 0.0` **exactly** (bit-exact, not thresholded), and `d[n_live_components:]` are the fill modes' **strictly-positive** eigenvalues (the eigensolver's values, clipped to ≥ 0; a genuine nonconstant mode's eigenvalue is accurately positive, well above the ~ε·‖S‖ error floor). Exactly one constant null is excluded **per live component**, identified **structurally** — by overlap with that component's mass-weighted constant, `_null_mode_mask` — **not** by an absolute eigenvalue cutoff (which would be coordinate-scale-dependent). `select_live_basis` **raises** if the supplied eigenbasis lacks `n_fill` non-null live modes, if a live component's null is absent, or if a selected fill weight is ≤ 0 (its eigenvalue clipped to 0 — a numerically (near-)disconnected component) — never silently returning a narrow or under-penalized `B`. `_mrf_basis`'s growth loop **verifies null coverage** (the numerical null can sort after a positive mode) and keeps growing until every live null is present.
- **Each intercept column is the exact MASS-NORMALIZED constant** on its live component: `B[bins_of_c, j] = 1/sqrt(Σ_c volumes)`, 0 elsewhere (= `M^{-1/2}` of the S-null `q0_c ∝ sqrt(vol)·1_c`). Constructed structurally, **not** read off the spectrum.
- `r_eff == B.shape[1] == d.shape[0] == max(n_live_components, min(n_live_bins, R))`.
- Structural penalty rank `r = r_eff − n_live_components` (spec §5) — **exact by construction** (exactly `n_live_components` structural-null columns excluded, `n_fill` fill columns kept), **not** a relative-threshold recount.
- **Occupancy is validated** at the resolver boundary: non-finite or negative occupancy **raises** `ValueError` (a NaN would otherwise silently drop a whole component; a negative could cancel one live). Occupancy is time/samples per bin.
- Dead (unvisited, `Σo == 0`) components contribute **no** rows to `B` and **no** modes — their bins are absent from `live_bins`.
- Degenerate: zero total occupancy → `B` is `(0, 0)`, `live_bins` empty, `n_live_components == 0` (spec §7 "zero total occupancy" row).

Signature (spec §6.1; see [designs.md](designs.md#resolver)):

```python
# environment/fields.py (Environment method; reuses PR2 geometry + eigensolver)
def _mrf_basis(self, occupancy: NDArray[np.float64], *, rank: int) -> MRFBasis: ...
```

---

## <a id="mrffit"></a>Fit return: `MRFFit`

Produced by `fit_mrf_gam(basis, counts, occupancy, *, penalty, pooled=True, backend="numpy")` —
**no `rank` arg** (the basis is the single source of truth for rank); `pooled` is wired in phase-6,
`backend` in phase-5 (both in the signature from phase-2 so phase-3 forwards them unchanged);
`counts`/`occupancy` **arrive already restricted to `basis.live_bins`** and are validated, never
re-sliced. `MRFFit` arrays are **always NumPy** (public return-type conversion is phase-3's job).
`NamedTuple`:

```python
class MRFFit(NamedTuple):
    coefficients: NDArray[np.float64]   # (r_eff, n_units) — γ on the live basis
    log_rate: NDArray[np.float64]       # (n_live_bins, n_units) — η = B γ (unclipped store is exp-of-clip)
    penalty: float | NDArray | None     # λ applied; scalar shared/fixed, (n_units,) per-unit (pooled=False), or None
    penalty_weights: NDArray[np.float64] # (r_eff,) — d (echoed for the result)
    rank: int                           # r_eff, DERIVED from basis.B.shape[1] (Finding 5)
    penalty_rank: int                   # r = r_eff − n_live_components
    deviance: NDArray[np.float64]        # (n_units,) — unpenalized Poisson deviance (spec §6.3)
    converged: bool                     # batch-level scalar (all(per-unit) when looped)
    n_iter: int                         # batch-level scalar (max(per-unit) when looped)
    reml_objective: float | NDArray | None  # None if REML didn't run; (n_units,) per-unit (nan for fallback units)
    penalty_selected_by_reml: NDArray | None # (n_units,) bool — pooled=False only; False = pooled-λ fallback unit
    pooled: bool                        # the input pooled flag — the ONLY reliable source for NWB (Finding 3)
```

**`pooled` is a stored field, not inferable (Finding 3).** For fixed-penalty / `r==0` / all-zero
cases the scalar outputs are identical under `pooled=True` and `pooled=False`, so NWB cannot
reconstruct the flag from the values — it must be carried on `MRFFit` and the result, and persisted
explicitly.

**Invariants:** `converged`/`n_iter` are **batch scalars** (one shared stopping criterion, spec
§6.3), never per-unit. `penalty` records the value **actually applied** (a supplied fixed
`penalty` is echoed, not discarded); `reml_objective is None` ⟺ REML did not run.

**`penalty is None` vs. the fit's `penalty_diag`.** `MRFFit.penalty` is `None` for REML-skip
(`r==0`) and no-data cases, but the final Newton fit **never** receives `None` — it uses
`penalty_diag = np.zeros_like(d)` (unpenalized; correct because `r==0` means every weight is a
structural null). The `None` lives only on the returned `MRFFit`, not inside the solver.

**Boundary orientation** (the fit is **bin-major**, the encoding API **unit-major**): the
phase-3 orchestrator transposes counts in (`(n_units, n_bins) → (n_live_bins, n_units)` restricted
to `live_bins`) and rates out (`log_rate (n_live_bins, n_units) →` result `firing_rates
(n_units, n_bins)`). See [designs.md → Boundary orientation](designs.md#module-layout).

**`pooled=False` (phase-6) widens `penalty`/`reml_objective`** to `float | NDArray[(n_units,)] |
None` — a `(n_units,)` vector **only** on the automatic-REML path (`penalty is None`, `r > 0`;
informative units get their λ_k, zero-spike units the pooled-λ fallback). It stays a **scalar**
when `pooled=True`, when a **fixed `penalty=<float>` is supplied** (fixed penalty beats `pooled`),
or `None` at `r==0` (population-level). Every other field keeps its shared-λ shape.

---

## <a id="result-fields"></a>Result-class GAM fields

Added to `SpatialRateResult` (singular) and `SpatialRatesResult` (plural). All `None` for the
ratio methods; populated for `method="glm"`. `method` replaces `smoothing_method`; `bandwidth`
widens to `float | None` (`None` for glm).

| field | plural `SpatialRatesResult` | singular (index `i` stamps `unit_id`) |
| --- | --- | --- |
| `method` | `str` | `str` |
| `bandwidth` | `float \| None` | `float \| None` |
| `coefficients` | `(rank, n_units)` | `[:, i]` → `(rank,)` |
| `penalty` | scalar, `(n_units,)` (pooled=False), or `None` | scalar or `[i]` |
| `penalty_weights` | `(rank,)` | same |
| `rank` | `int` (effective `r_eff`) | same |
| `deviance` | `(n_units,)` | `[i]` |
| `converged` | scalar `bool` | same |
| `n_iter` | scalar `int` | same |
| `reml_objective` | scalar, `(n_units,)` (pooled=False; `nan` for fallback units), or `None` | scalar or `[i]` |
| `penalty_selected_by_reml` | `(n_units,)` bool (pooled=False only) or `None` | `[i]` or `None` |
| `pooled` | `bool` (glm) or `None` (ratio) — persisted; the only reliable NWB source | same |

**Phasing of the fields.** Most GAM fields are added in **phase-3**. **`pooled` and
`penalty_selected_by_reml` are added in phase-6** (with the `pooled` param): phase-3/phase-4 glm
results are implicitly shared-λ and carry neither; phase-6 adds them and their NWB persistence. The
reader defaults a **missing `pooled` key method-conditionally** — `True` for `method=="glm"`
(matches pre-phase-6 glm behavior), **`None` for ratio methods** (where `pooled` is meaningless) —
so both phase-4-era glm and legacy ratio files read correctly. The vector shapes of
`penalty`/`reml_objective` likewise appear only in phase-6.

`firing_rate` (existing field) for glm is `max(exp(η), _RATE_FLOOR)` in **active-bin order,
shape `(n_bins,)`** — dead/non-live bins are `_RATE_FLOOR`. Terminal verbs (`to_dataframe`,
`summary_table`, `peak_locations`, `summary`, `to_xarray`) are unchanged; `summary_table` gains
the GAM scalar columns when present (spec §6.3).
