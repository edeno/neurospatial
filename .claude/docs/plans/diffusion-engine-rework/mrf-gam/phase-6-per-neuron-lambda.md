# Phase 6 — Per-neuron λ (`pooled=False`)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#reml)

Add per-neuron REML smoothing: `pooled=False` selects an independent `λ_k` per unit instead of
one shared λ. Purely additive — the default stays `pooled=True` (shared λ), so nothing existing
changes.

**Inputs to read first:**

- [designs.md#reml](designs.md#reml) — the shared-λ REML; per-neuron runs the same objective per unit.
- `src/neurospatial/encoding/_glm.py` (phase-2) — `fit_mrf_gam`, `select_penalty_by_reml`.
- [shared-contracts.md#result-fields](shared-contracts.md#result-fields) — `penalty` widens to a per-unit vector when `pooled=False`.

**Contracts referenced:**

- [`MRFFit`](shared-contracts.md#mrffit) / [Result GAM fields](shared-contracts.md#result-fields) — `penalty` becomes `float | NDArray | None` and `reml_objective` `float | NDArray | None` when `pooled=False`; add `reml_at_boundary: bool | NDArray | None` for weakly identified/beyond-interval REML optima. **Existing shared-λ field shapes are unchanged** (backward-compatible with phases 2–5).

## Tasks

**Key fact: the basis `B`, weights `d`, and penalty rank `r` are shared geometry — identical for
every unit; only the counts differ.** So `r == 0` is a **population-level** property (not per-unit),
and per-unit λ is meaningful only for **informative** units (`Σ n_k > 0`).

- Add `pooled: bool = True` to `compute_spatial_rate` / `compute_spatial_rates` with **strict validation** (Finding 3): reject non-`bool` (`isinstance(pooled, bool)` — rejects strings, ints incl. `0`/`1`, arrays, `None`) with a clear `ValueError`; and reject `pooled=False` with a **ratio method** (`pooled` is glm-only), same validation family as [method-param](shared-contracts.md#method-param). `pooled=True` with a ratio method is a harmless no-op default (don't error on the default).
- **Fixed penalty takes precedence over `pooled` (Finding 6).** If `penalty=<float>` is supplied, REML is skipped entirely and the result records that **scalar** λ — identical to `pooled=True`. `pooled` only affects the automatic-REML (`penalty=None`) path. Document + test.
- In `_glm.py`, per-neuron REML branch to `fit_mrf_gam` for `pooled=False, penalty is None, penalty_rank > 0`: **partition units into informative (`Σ n_k > 0`) and zero-spike**.
  - **No informative unit at all** → this is the **all-zero-spike population** degenerate case ([designs.md#degenerate](designs.md#degenerate), Finding 3): **do not run pooled REML** (it is equally unidentified); take the shared degenerate path — scalar `penalty=None`, `reml_objective=None`, floor fields. `pooled=False` reuses that behavior, does not special-case it.
  - **Informative units:** `select_penalty_by_reml` **per unit** (the pooled objective is already a per-unit sum — minimize each unit's score over `log λ`), then a per-unit final fit at each `λ_k`.
  - **Zero-spike units (with ≥1 informative unit present) are statistically unidentified for per-unit λ (Finding 5)** — intercept → −∞, non-null coeffs → 0, the `r·log λ` terms cancel. Assign the **fallback λ = the pooled λ over the informative units** (shared-λ REML on the informative subset, computed once), **not** the optimizer's arbitrary point; their field still floors to `_RATE_FLOOR`.
- **Boundary/weak-identification diagnostic (mgcv comparison follow-up).** Change the internal
  NumPy and JAX selectors to return `(penalty, objective, at_boundary)`, where
  `at_boundary = min(result.x - lower, upper - result.x) <= _REML_BOUNDARY_TOL` on the **log-λ**
  scale and `_REML_BOUNDARY_TOL = 5 * _REML_XATOL = 5e-3`. REML-skip returns
  `(None, None, None)`. Carry **`reml_at_boundary: bool | NDArray[bool] | None`** on `MRFFit` and
  public results: scalar for pooled REML, `(n_units,)` for per-unit REML, `None` when REML did not
  run. A zero-spike fallback unit inherits the pooled-informative search's boundary flag while
  `penalty_selected_by_reml=False` continues to identify its provenance. Emit one warning naming
  the boundary side and affected unit ids (or the pooled fit). A boundary λ remains the finite
  applied penalty; this phase does **not** silently expand the interval or substitute pooled λ for
  an informative low-count unit.
- **Fallback diagnostics + provenance (Finding 4).** Add a per-unit boolean field **`penalty_selected_by_reml: NDArray[bool] | None`** to `MRFFit` / the result (`None` unless `pooled=False` produced a per-unit vector; `True` for informative units, `False` for fallback units). For fallback units, store **`reml_objective[k] = np.nan`** (documented sentinel — their λ is not a per-unit REML minimum). This mask is carried through **singular indexing** (`rates[i].penalty_selected_by_reml`), **`summary_table`** (a boolean column), and **NWB** (a per-unit column), so a round-tripped λ never looks unit-estimated when it was a fallback.
- **`penalty_rank == 0` behaves exactly as `pooled=True`** — skip REML population-wide, scalar `penalty=None`, `penalty_diag=zeros`. Reuse the shared-λ Newton fit per distinct λ_k; do not fork it.
- **Aggregate batch diagnostics when per-unit fits are looped (Finding 6):** `converged = all(per_unit_converged)`, `n_iter = max(per_unit_n_iter)`, and **one** `UserWarning` naming the non-converged unit ids (not one per unit). These stay batch scalars ([MRFFit](shared-contracts.md#mrffit)).
- **Forward `pooled` + store it (Findings 2/3).** The public functions pass `pooled=pooled` into `fit_mrf_gam(..., pooled=pooled)`; `MRFFit.pooled` and the **result's new `pooled` field** record it (`None` for ratio results). This is the **only reliable NWB source** — for fixed-penalty / `r==0` / all-zero cases the scalar outputs are identical under both settings, so `pooled` cannot be inferred (Finding 3).
- **Result plumbing** ([shared-contracts.md#mrffit](shared-contracts.md#mrffit)): `penalty` / `reml_objective` become `(n_units,)` float arrays and `reml_at_boundary` a matching bool array when `pooled=False, penalty is None, r > 0` **and ≥1 informative unit** (informative units their λ_k, zero-spike units the fallback λ with `reml_objective=nan`); **scalar** otherwise for pooled automatic REML, and `None` for fixed penalty / `r==0` / no-REML cases. `summary_table` reports per-unit λ, boundary status, and `penalty_selected_by_reml` when vector; indexing slices all three per-unit fields.
- **Singular `compute_spatial_rate(..., pooled=False)` plumbing (Finding 6).** The singular function is a **separate path** from the plural — it must **unwrap the one-element vectors to scalars** on `SpatialRateResult`: `penalty` / `reml_objective` become the single float (`reml_objective` may be `nan` for a fallback unit), `reml_at_boundary` and `penalty_selected_by_reml` the single bools, `pooled` carried. Specify + test that `compute_spatial_rate(env, spikes_k, ..., pooled=False)` equals `compute_spatial_rates(env, [spikes_k], ..., pooled=False)[0]` field-for-field.
- **NWB — schema migration (Finding 4).** This adds new persisted fields, so **bump the encoding-model schema version again** (beyond phase-4's bump). Writer: add **`pooled` and scalar `reml_at_boundary`** to metadata; when `penalty`/`reml_objective` are vectors, persist them, **`reml_at_boundary`**, and `penalty_selected_by_reml` as **per-unit table columns** ([spec §8](../design-mrf-gam.md)). Reader **backward-compat (method-conditional — Finding 4):** a file with no `pooled` key reads as **`pooled=True` only when `method=="glm"`** (the only glm behavior that existed pre-phase-6); a **ratio-method** file reads as **`pooled=None`**. Missing phase-6 boundary metadata/columns read as `reml_at_boundary=None`; missing vector columns/mask retain scalar `penalty`/`reml_objective` and `penalty_selected_by_reml=None`. Round-trip preserves scalar-vs-vector and the mask/boundary diagnostic.
- Docs: `compute_spatial_rate` docstring + CHANGELOG — `pooled=False` for per-neuron smoothing; note the cost (REML per unit), fixed-penalty precedence, zero-spike fallback, shared-basis `r==0` behavior, and that a boundary warning means λ itself is weakly identified — the fitted field remains finite but its λ-sensitivity should be checked (not asserted stable).

## Deliberately not in this phase

- **Changing the shared-λ default** — `pooled=True` stays default and shape-/value-identical to phases 2–5.
- **A per-neuron JAX fast path** beyond looping/batching the existing fit — only if the perf test shows the loop is unacceptable; otherwise reuse phase-5's fit per distinct λ.
- **Adaptive REML bounds, curvature-based shrinkage, or an arbitrary low-spike cutoff.** This
  phase reports boundary optima but does not silently change the search interval or replace an
  informative unit's λ. Use the measured REML surface, not a hard spike-count threshold, in a
  later empirical-Bayes/partial-pooling design.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pooled_false_per_unit_lambda` | population with deliberately different per-unit smoothness → `pooled=False` recovers **distinct** finite `λ_k` (variance across units > 0), each near its unit-specific optimum. |
| `test_pooled_true_unchanged` | `pooled=True` (default) → existing scalar `penalty`/`reml_objective` values and types remain identical to phase 5; the only additive output is scalar `reml_at_boundary`. |
| `test_pooled_false_shapes` | `r>0`, all units informative: `penalty`/`reml_objective` are `(n_units,)` **finite**, `reml_at_boundary` is `(n_units,)` bool, and `penalty_selected_by_reml` all `True`; `summary_table` and indexing preserve them. |
| `test_pooled_false_r0_scalar_none` | `r==0` population (two-3-node-paths) with `pooled=False` → `penalty is None` **scalar** (not a vector); rates finite (shared-basis semantics). |
| `test_pooled_false_zero_spike_fallback` | a zero-spike unit inside a `pooled=False`, `r>0` population gets the **fallback λ = pooled λ over informative units** (assert `penalty[zs] == pooled_lambda_of_informative`, **not** the per-unit optimizer output), `penalty_selected_by_reml[zs] == False`, `reml_objective[zs]` is `nan` (sentinel), near-floor field; informative units keep their own `λ_k` and `penalty_selected_by_reml == True` (Findings 4/5). |
| `test_reml_boundary_diagnostic` | weak-signal data with a high-bound optimum returns the finite applied λ, `reml_at_boundary is True`, and one warning naming the upper boundary; an interior fixture returns `False`. Fixed penalty / `r==0` / no-data return `None`. Pooled and per-unit shapes are both asserted. |
| `test_pooled_false_all_zero_spike` | a `pooled=False` population with **no** informative unit reuses the all-zero-spike degenerate path — scalar `penalty is None`, `reml_objective is None`, floor fields (**no pooled REML run** — Finding 3). |
| `test_fixed_penalty_precedence` | `pooled=False, penalty=2.5` → REML skipped, `result.penalty == 2.5` **scalar** (not a vector), `reml_objective is None` — fixed penalty beats `pooled` (Finding 6). |
| `test_aggregate_diagnostics` | with a looped per-unit fit where one unit fails to converge: `converged is False`, `n_iter == max` per-unit, and **one** `UserWarning` names the failed unit id(s) (not one per unit) (Finding 6). |
| `test_pooled_validation` | `pooled=False` with `method="binned"` raises; with `method="glm"` accepted; `pooled=True` with a ratio method is a no-op (no raise). |
| `test_pooled_type_strict` | `pooled` rejects `"true"`, `1`, `0`, `np.array([True])`, `None` with `ValueError` (only real `bool` accepted; Finding 3). |
| `test_singular_pooled_false_equals_plural` | `compute_spatial_rate(env, spikes_k, ..., pooled=False)` equals `compute_spatial_rates(env, [spikes_k], ..., pooled=False)[0]` field-for-field; numeric REML fields and `reml_at_boundary`/`penalty_selected_by_reml` are scalars (unwrapped), `pooled` carried. |
| `test_result_carries_pooled` | glm results expose `.pooled` (`True`/`False`); ratio results have `.pooled is None`; it survives NWB round-trip and indexing (Finding 3). |
| `test_pooled_false_nwb_roundtrip` | a `pooled=False` result round-trips with `penalty`/`reml_objective`/`reml_at_boundary`/`penalty_selected_by_reml` as per-unit columns; pooled automatic REML stores scalar boundary metadata, while fixed-penalty results keep it absent/`None`. |
| `test_nwb_phase4_era_glm_reads` | a **phase-4-era glm file** with no phase-6 keys reads back as **`pooled=True`**, scalar `penalty`/`reml_objective`, `reml_at_boundary is None`, and `penalty_selected_by_reml is None`. Back it with a checked-in tiny phase-4-schema file. |
| `test_nwb_legacy_ratio_pooled_none` | a **phase-4-era ratio-method file** with no `pooled` key reads back as **`pooled=None`** (not `True`) — `pooled` is meaningless for ratio; method-conditional default (Finding 4). |

## Fixtures

- Reuse `simulate_place_fields`; add a variant with per-unit smoothness (some units sharp, some broad, fixed seed) so distinct `λ_k` is recoverable, a weak-signal fixture whose REML minimum is reproducibly at the upper bound, and the two-3-node-paths `r==0` env for scalar-`None` tests.
- A **checked-in tiny phase-4-era NWB file** (written at the phase-4 schema version, no `pooled` / `reml_at_boundary` key, vector columns, or mask) to back `test_nwb_phase4_era_file_reads` — the same way phase-0's legacy-key file backs its read test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Existing `pooled=True` outputs are shape- and value-identical to before; its new
  `reml_at_boundary` diagnostic is additive, and `pooled=False` remains additive.
- `r==0` is population-level (scalar `None`), **not** per-unit; a **fixed penalty stays a scalar**
  regardless of `pooled`; per-unit penalties are finite, and only fallback units use the documented
  `nan` sentinel in `reml_objective` (never `None` inside a vector).
- **Zero-spike units use the pooled-λ fallback, not the optimizer's arbitrary point** (Finding 5); aggregate `converged`/`n_iter` follow the all/max rule with a single failed-unit warning (Finding 6).
- Boundary optima are warned and persisted through `reml_at_boundary`; they are not silently
  expanded, replaced, or treated as well-identified interior λ values.
- The Newton fit is reused per λ, not forked; NWB **bumps the schema version** and reads phase-4-era files (missing `pooled` → `True`) — a backward-read test exists (Finding 4).
- `pooled` is **strictly boolean-validated** (rejects `"true"`/`1`/`0`/arrays/`None`) and glm-only (Finding 3).
- "Deliberately not in this phase" honored — default unchanged, no speculative JAX fork.
- Tests recover distinct per-unit λ and assert the scalar-vs-vector shape rules (not tautologies); fixtures shared, seed fixed.
- No plan references in code/tests; docstring + CHANGELOG updated.
