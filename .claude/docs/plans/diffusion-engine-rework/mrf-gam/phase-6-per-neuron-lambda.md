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

**Key fact (Finding 7): the basis `B`, weights `d`, and penalty rank `r` are shared geometry —
identical for every unit; only the counts differ.** So `r == 0` is a **population-level** property
(not per-unit), and there is **no per-unit `None`/`nan`**: when `r > 0` every unit gets a finite
`λ_k`; when `r == 0` REML is skipped population-wide and `penalty` stays a **scalar `None`**.

- Add `pooled: bool = True` to `compute_spatial_rate` / `compute_spatial_rates` (glm-only; a `ValueError` if set `False` with a ratio method, same validation family as [method-param](shared-contracts.md#method-param)).
- In `_glm.py`, add a per-neuron REML branch to `fit_mrf_gam`: when `pooled=False` **and `penalty_rank > 0`**, run `select_penalty_by_reml` **per unit** (the pooled objective is already a per-unit sum — minimize each unit's score over `log λ` independently), then a per-unit final fit at each `λ_k`. **When `penalty_rank == 0`, behave exactly as `pooled=True`** — skip REML population-wide, scalar `penalty=None`, `penalty_diag=zeros`. Reuse the shared-λ Newton fit (loop or batch per distinct λ_k); do not fork it.
- **Zero-spike unit** (`r > 0`): its per-unit REML still returns a finite in-bounds `λ_k` (its likelihood is flat toward the rate floor — no special-casing); the field floors as in phase-2. Documented, not a `None`.
- **Result plumbing** ([shared-contracts.md#mrffit](shared-contracts.md#mrffit)): `penalty` / `reml_objective` become `(n_units,)` **finite** float arrays when `pooled=False` and `r > 0`; **scalar `None`** when `r == 0`; **scalar** (unchanged) when `pooled=True`. `summary_table` reports per-unit λ when vector; `rates[i].penalty` slices `penalty[i]` when vector, else the shared scalar. Extend `SpatialRatesResult.__getitem__` accordingly.
- **NWB** (extends phase-4): when `penalty`/`reml_objective` are vectors, persist them as **per-unit table columns** (not metadata scalars); the reader keys on the stored `pooled` flag / scalar-vs-vector to place them ([spec §8](../design-mrf-gam.md)). Add a `pooled` scalar to the metadata. Round-trip preserves scalar-vs-vector.
- Docs: `compute_spatial_rate` docstring + CHANGELOG — `pooled=False` for per-neuron smoothing; note the cost (REML per unit) and the shared-basis `r==0` behavior.

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
| `test_pooled_false_zero_spike` | a zero-spike unit inside a `pooled=False`, `r>0` population gets a **finite** `λ_k` and a near-floor field; the rest fit normally. |
| `test_pooled_validation` | `pooled=False` with `method="binned"` raises; with `method="glm"` accepted. |
| `test_pooled_false_nwb_roundtrip` | a `pooled=False` result round-trips through NWB with `penalty`/`reml_objective` as per-unit columns; a `pooled=True` result keeps them as metadata scalars. |

## Fixtures

- Reuse `simulate_place_fields`; add a variant with per-unit smoothness (some units sharp, some broad, fixed seed) so distinct `λ_k` is recoverable, and the two-3-node-paths `r==0` env for the scalar-`None` test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `pooled=True` default is shape- and value-identical to before (no regression); `pooled=False` is additive.
- `r==0` is treated as population-level (scalar `None`), **not** per-unit; no `None`/`nan` inside the per-unit vector; zero-spike units get finite `λ_k`.
- The Newton fit is reused per λ, not forked; NWB places scalar-vs-vector correctly.
- "Deliberately not in this phase" honored — default unchanged, no speculative JAX fork.
- Tests recover distinct per-unit λ and assert the scalar-vs-vector shape rules (not tautologies); fixtures shared, seed fixed.
- No plan references in code/tests; docstring + CHANGELOG updated.
