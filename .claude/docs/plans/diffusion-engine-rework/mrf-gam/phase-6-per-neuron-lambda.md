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

- Add `pooled: bool = True` to `compute_spatial_rate` / `compute_spatial_rates` (glm-only; a `ValueError` if set with a ratio method, same validation family as [method-param](shared-contracts.md#method-param)).
- In `_glm.py`, add a per-neuron REML branch: when `pooled=False`, run `select_penalty_by_reml` **per unit** (the objective is already per-unit summable — drop the `np.sum` over units and minimize each unit's score), then a final per-unit fit at each `λ_k`. Reuse the shared-λ machinery; do not fork the Newton fit.
- Result plumbing: `penalty` / `reml_objective` become `(n_units,)` arrays when `pooled=False`; `summary_table` reports per-unit λ; indexing a plural result slices `penalty[i]`. Shared-λ path keeps scalars.
- Skip/`r==0` semantics per unit: a unit whose basis has `penalty_rank==0` gets `penalty=None` (or `nan` in the vector — pick one and document); zero-spike units behave as in phase-2.
- Docs: `compute_spatial_rate` docstring + CHANGELOG — `pooled=False` for per-neuron smoothing; note the cost (REML per unit).

## Deliberately not in this phase

- **Changing the shared-λ default** — `pooled=True` stays default and shape-identical to phases 2–5.
- **A per-neuron JAX fast path** beyond looping the existing JAX fit — only if the parity/perf test shows the loop is unacceptable; otherwise reuse phase-5's batched fit per λ.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pooled_false_per_unit_lambda` | on a population with deliberately different smoothness per unit, `pooled=False` recovers **distinct** `λ_k` (variance across units > 0); each near its unit-specific optimum. |
| `test_pooled_true_unchanged` | `pooled=True` (default) produces scalar `penalty`/`reml_objective` identical to the phase-2/3 result (no regression). |
| `test_pooled_false_shapes` | `penalty`/`reml_objective` are `(n_units,)`; `summary_table` has per-unit λ; `rates[i].penalty` is the scalar `λ_i`. |
| `test_pooled_validation` | `pooled=False` with `method="binned"` raises; with `method="glm"` is accepted. |
| `test_pooled_false_degenerate` | a zero-spike / `r==0` unit inside a `pooled=False` population gets the documented per-unit degenerate value; the rest fit normally. |

## Fixtures

- Reuse `simulate_place_fields`; add a variant with per-unit smoothness (some units sharp, some broad) so distinct `λ_k` is recoverable (fixed seed).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `pooled=True` default is shape- and value-identical to before (no regression); `pooled=False` is additive.
- The Newton fit is reused per λ, not forked; per-unit degenerate semantics are documented and tested.
- "Deliberately not in this phase" honored — default unchanged, no speculative JAX fork.
- Tests recover distinct per-unit λ (not a tautology) and assert shape changes; fixtures shared, seed fixed.
- No plan references in code/tests; docstring + CHANGELOG updated.
