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

- [`MRFFit`](shared-contracts.md#mrffit) / [Result GAM fields](shared-contracts.md#result-fields) — `penalty` becomes `float | NDArray | None` and `reml_objective` `float | NDArray | None` when `pooled=False`; **shared-λ shapes are unchanged** (backward-compatible with phases 2–5).

## Tasks

**Key fact: the basis `B`, weights `d`, and penalty rank `r` are shared geometry — identical for
every unit; only the counts differ.** So `r == 0` is a **population-level** property (not per-unit),
and per-unit λ is meaningful only for **informative** units (`Σ n_k > 0`).

- Add `pooled: bool = True` to `compute_spatial_rate` / `compute_spatial_rates` (glm-only; a `ValueError` if set `False` with a ratio method, same validation family as [method-param](shared-contracts.md#method-param)).
- **Fixed penalty takes precedence over `pooled` (Finding 6).** If `penalty=<float>` is supplied, REML is skipped entirely and the result records that **scalar** λ — identical to `pooled=True`. `pooled` only affects the automatic-REML (`penalty=None`) path. Document + test.
- In `_glm.py`, per-neuron REML branch to `fit_mrf_gam` for `pooled=False, penalty is None, penalty_rank > 0`: **partition units into informative (`Σ n_k > 0`) and zero-spike**.
  - **Informative units:** `select_penalty_by_reml` **per unit** (the pooled objective is already a per-unit sum — minimize each unit's score over `log λ`), then a per-unit final fit at each `λ_k`.
  - **Zero-spike units are statistically unidentified for per-unit λ (Finding 5)** — intercept → −∞, non-null coeffs → 0, the `r·log λ` terms cancel, so the bounded optimizer's point is noise. Assign the **fallback λ = the pooled λ over the informative units** (compute the shared-λ REML on the informative subset once), **not** the optimizer's arbitrary point. Their field still floors to `_RATE_FLOOR`. If **no** unit is informative, use the whole-population pooled λ (or `None` at `r==0`). Record which units used the fallback (e.g. a boolean note / in the warning), and document that their λ is not data-driven.
- **`penalty_rank == 0` behaves exactly as `pooled=True`** — skip REML population-wide, scalar `penalty=None`, `penalty_diag=zeros`. Reuse the shared-λ Newton fit per distinct λ_k; do not fork it.
- **Aggregate batch diagnostics when per-unit fits are looped (Finding 6):** `converged = all(per_unit_converged)`, `n_iter = max(per_unit_n_iter)`, and **one** `UserWarning` naming the non-converged unit ids (not one per unit). These stay batch scalars ([MRFFit](shared-contracts.md#mrffit)).
- **Result plumbing** ([shared-contracts.md#mrffit](shared-contracts.md#mrffit)): `penalty` / `reml_objective` become `(n_units,)` float arrays when `pooled=False, penalty is None, r > 0` (informative units their λ_k, zero-spike units the fallback λ); **scalar** otherwise (`pooled=True`, or fixed penalty, or `r==0`→`None`). `summary_table` reports per-unit λ when vector; `rates[i].penalty` slices `penalty[i]` when vector, else the shared scalar. Extend `SpatialRatesResult.__getitem__` accordingly.
- **NWB** (extends phase-4): when `penalty`/`reml_objective` are vectors, persist them as **per-unit table columns** (not metadata scalars); the reader keys on the stored `pooled` flag / scalar-vs-vector to place them ([spec §8](../design-mrf-gam.md)). Add a `pooled` scalar to the metadata. Round-trip preserves scalar-vs-vector.
- Docs: `compute_spatial_rate` docstring + CHANGELOG — `pooled=False` for per-neuron smoothing; note the cost (REML per unit), the fixed-penalty precedence, the zero-spike fallback (λ not data-driven), and the shared-basis `r==0` behavior.

## Deliberately not in this phase

- **Changing the shared-λ default** — `pooled=True` stays default and shape-/value-identical to phases 2–5.
- **A per-neuron JAX fast path** beyond looping/batching the existing fit — only if the perf test shows the loop is unacceptable; otherwise reuse phase-5's fit per distinct λ.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pooled_false_per_unit_lambda` | population with deliberately different per-unit smoothness → `pooled=False` recovers **distinct** finite `λ_k` (variance across units > 0), each near its unit-specific optimum. |
| `test_pooled_true_unchanged` | `pooled=True` (default) → scalar `penalty`/`reml_objective` identical (value + type) to the phase-2/3 result (no regression). |
| `test_pooled_false_shapes` | `r>0`: `penalty`/`reml_objective` are `(n_units,)` **finite** (no `nan`/`None`); `summary_table` has per-unit λ; `rates[i].penalty == penalty[i]`. |
| `test_pooled_false_r0_scalar_none` | `r==0` population (two-3-node-paths) with `pooled=False` → `penalty is None` **scalar** (not a vector of `None`/`nan`); rates finite (guards the Finding 7 shared-basis semantics). |
| `test_pooled_false_zero_spike_fallback` | a zero-spike unit inside a `pooled=False`, `r>0` population gets the **fallback λ = the pooled λ over informative units** (assert `penalty[zero_spike] == pooled_lambda_of_informative`, **not** the per-unit optimizer output), near-floor field; informative units keep their own `λ_k` (Finding 5). |
| `test_fixed_penalty_precedence` | `pooled=False, penalty=2.5` → REML skipped, `result.penalty == 2.5` **scalar** (not a vector), `reml_objective is None` — fixed penalty beats `pooled` (Finding 6). |
| `test_aggregate_diagnostics` | with a looped per-unit fit where one unit fails to converge: `converged is False`, `n_iter == max` per-unit, and **one** `UserWarning` names the failed unit id(s) (not one per unit) (Finding 6). |
| `test_pooled_validation` | `pooled=False` with `method="binned"` raises; with `method="glm"` accepted. |
| `test_pooled_false_nwb_roundtrip` | a `pooled=False` (vector-penalty) result round-trips through NWB with `penalty`/`reml_objective` as per-unit columns; a `pooled=True` / fixed-penalty result keeps them as metadata scalars. |

## Fixtures

- Reuse `simulate_place_fields`; add a variant with per-unit smoothness (some units sharp, some broad, fixed seed) so distinct `λ_k` is recoverable, and the two-3-node-paths `r==0` env for the scalar-`None` test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `pooled=True` default is shape- and value-identical to before (no regression); `pooled=False` is additive.
- `r==0` is population-level (scalar `None`), **not** per-unit; a **fixed penalty stays a scalar** regardless of `pooled`; no `None`/`nan` inside the per-unit vector.
- **Zero-spike units use the pooled-λ fallback, not the optimizer's arbitrary point** (Finding 5); aggregate `converged`/`n_iter` follow the all/max rule with a single failed-unit warning (Finding 6).
- The Newton fit is reused per λ, not forked; NWB places scalar-vs-vector correctly.
- "Deliberately not in this phase" honored — default unchanged, no speculative JAX fork.
- Tests recover distinct per-unit λ and assert the scalar-vs-vector shape rules (not tautologies); fixtures shared, seed fixed.
- No plan references in code/tests; docstring + CHANGELOG updated.
