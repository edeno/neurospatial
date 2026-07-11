# Phase 2 — Penalized-Poisson fit + REML (NumPy/SciPy core)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#fit)

Implement the statistical core: batched penalized-Poisson Newton/IRLS, REML λ selection, and
deviance — the float64 NumPy/SciPy reference path. Consumes an [`MRFBasis`](shared-contracts.md#mrfbasis)
and spike counts + occupancy, produces an [`MRFFit`](shared-contracts.md#mrffit). Internal only;
phase-3 wires it to the public API.

**Inputs to read first:**

- [designs.md#fit](designs.md#fit), [#reml](designs.md#reml), [#deviance](designs.md#deviance) — the NumPy skeletons.
- [appendix.md](appendix.md) — reference `_newton_fit_jax` (`:242`), `_reml_score_jax` (`:445`), `select_penalty_by_reml` (`:550`), `_penalized_hessian` (`:229`), the `converged` warning (`:865-875`).
- `src/neurospatial/encoding/_smoothing.py` — the module's NumPy idiom / imports to match.
- `src/neurospatial/decoding/likelihood.py:47` — `min_rate = 1e-10`, the source of `_RATE_FLOOR`.

**Contracts referenced:**

- [Module constants](shared-contracts.md#constants) — define them here in `encoding/_glm.py`; float64 core uses `_FIT_TOL` (not the float32 floor).
- [`MRFBasis`](shared-contracts.md#mrfbasis) (input) / [`MRFFit`](shared-contracts.md#mrffit) (output) — honor the structural `penalty_rank = r_eff − n_live_components`; **do not** recompute rank from a relative threshold.

**Designs referenced:** [designs.md#fit](designs.md#fit), [#reml](designs.md#reml), [#deviance](designs.md#deviance), [#module-layout](designs.md#module-layout).

## Tasks

- Create `encoding/_glm.py`: the [constants](shared-contracts.md#constants), `MRFFit`, and the orchestrator `fit_mrf_gam(basis: MRFBasis, counts, occupancy, *, penalty, rank) -> MRFFit`. `counts`/`occupancy` are restricted to `basis.live_bins` before fitting. Compute `penalty_rank = basis.d.size − basis.n_live_components` structurally.
- Create `encoding/_glm_numpy.py`:
  - `_newton_fit_numpy(counts, occupancy, B, penalty_diag, *, max_iter, tol) -> (coeffs, eta, mu, n_iter, max_step, converged)` per [designs.md#fit](designs.md#fit). Warm-start from a constant log-rate projected onto `B`; clip η to `±_ETA_CLIP`; step-halve (≤ `_MAX_STEP_HALVINGS`) for monotone descent; converge on the **relative penalized-objective decrease** (batch-scalar `converged`/`n_iter` = one shared max-reduce stop), never the coefficient step.
  - `_reml_objective_numpy(...)` + `select_penalty_by_reml(...)` per [designs.md#reml](designs.md#reml): `scipy.optimize.minimize_scalar(method="bounded", bounds=_LOG_PENALTY_BOUNDS, options={"xatol": _REML_XATOL})`; `log|H_k|` via `scipy.linalg.cholesky` with a non-PD `H_k` → `+inf`; **skip REML and return `(None, None)` when `penalty_rank == 0`**; raise a clear `ValueError` when no finite objective exists in the interval.
- Wire `fit_mrf_gam`: if `penalty is None` → `select_penalty_by_reml` (returns `(None, None)` at `r==0`); else fixed λ, `reml_objective=None`, **record the supplied `penalty`**. Then the final `_newton_fit_numpy` at the chosen λ — **when the effective penalty is `None`, pass `penalty_diag = np.zeros_like(d)` to the fit** (never `None`), while `MRFFit.penalty` stays `None` (Finding 4). Compute the floored `firing_rate = max(exp(η), _RATE_FLOOR)` on live bins from the fit's **final** `η` (recomputed post-loop, not the pre-update array — Finding 3), and the deviance ([designs.md#deviance](designs.md#deviance)) from that stored rate.
- `_newton_fit_numpy` takes `max_iter`/`tol` **positionally** (or bind via `functools.partial`) so REML's `minimize_scalar(args=…)` can supply them (Finding 4); it returns the **accepted (step-halved)** max step as its diagnostic, and recomputes `(η, μ)` from the updated coefficients after the loop (Finding 3) — see [designs.md#fit](designs.md#fit).
- **Nonconvergence warning:** after the final fit, `if not converged: warnings.warn(...)` (keyed on the flag, **not** `n_iter == max_iter`) naming `n_iter` + max coefficient step and recommending reduce-`rank` / increase-`penalty` (spec §4). Fields still returned.
- Docstrings (NumPy-style) on every public-within-module function documenting array shapes.

## Deliberately not in this phase

- **Public wiring / result classes / param validation / degenerate whole-map dispatch** — phase-3. This phase's functions take a `MRFBasis` + arrays and return `MRFFit`; they don't touch `compute_spatial_rate`.
- **JAX** — phase-5. NumPy float64 is the correctness reference; the float32 tol-floor logic (`_FIT_TOL_FLOOR`, `_DESCENT_TOL`) is defined as constants here but only *used* by the JAX path.
- **Per-neuron λ** — phase-6 (`pooled=False`); this phase is single-shared-λ only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recovers_place_fields` | population fit on simulated Poisson spikes with distinct centers → per-neuron peak bins near the simulated centers. |
| `test_reml_selects_sensible_lambda` | REML λ is finite, within `exp(_LOG_PENALTY_BOUNDS)`; a fixed extreme λ → monotonically smoother field (compare field variance). |
| `test_r0_skips_reml` | two-3-node-paths `MRFBasis` (all-null `d`, `penalty_rank==0`) with `penalty=None` → `penalty is None`, `reml_objective is None`, rates finite; **not** an arbitrary λ. |
| `test_fixed_penalty_recorded` | `penalty=2.5` → `MRFFit.penalty == 2.5`, `reml_objective is None` (not discarded). |
| `test_penalty_rank_structural` | `MRFFit.penalty_rank == r_eff − n_live_components`; independent of any relative-threshold count. |
| `test_reml_pooled_scaling` | REML λ invariant to duplicating the population (the `n_units` df factor present); a variant omitting it shifts λ with `n_units`. |
| `test_deviance_formula` | independently recompute `2·Σ[n·log(n/μ) − (n−μ)]` (0·log0=0, exposed bins, `μ = o·firing_rate`) and match `MRFFit.deviance`. |
| `test_convergence_on_deviance` | undersampled arena (null-mode drift) still converges (`converged True`); a coeff-step criterion would not. |
| `test_fit_returns_consistent_triple` | the returned `(coeffs, eta, mu)` satisfy `eta == clip(B @ coeffs)` and `mu == occupancy[:,None]·exp(eta)` exactly (guards Finding 3 — no stale pre-update arrays); the reported step equals the **accepted** halved step, not the raw Newton step. |
| `test_r0_final_fit_unpenalized` | at `r==0`, the final fit runs with `penalty_diag == 0` (never `None`) and returns finite rates; `MRFFit.penalty is None` (guards Finding 4). |
| `test_reml_objective_callable` | `select_penalty_by_reml` runs end-to-end (no `TypeError` from `minimize_scalar` arg passing — guards Finding 4). |
| `test_nonconvergence_warns` | a fit forced non-converged (tiny `max_iter` via the internal constant patched, or a pathological input) emits `UserWarning` keyed on `not converged`; fields still returned; no raise. |
| `test_zero_spike_neuron` | a unit with zero total spikes → finite near-floor rate, finite deviance, shared λ. |

## Fixtures

- **Simulated-data fixture** in `conftest.py`: `simulate_place_fields` producing Poisson counts for N units with known centers over a 2D open field (fixed seed). Edge fixtures: zero-spike unit, undersampled arena (sparse occupancy), the two-3-node-paths `MRFBasis`.
- No real data needed at this layer (pure statistics on synthetic counts); real-data agreement is covered in phase-3's ratio-estimator-agreement test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `penalty_rank` is structural everywhere; no relative-threshold rank survives.
- `converged`/`n_iter` are batch scalars; the warning keys on the flag.
- REML skip at `r==0` returns `(None, None)`; fixed penalty is recorded.
- "Deliberately not in this phase" honored — no public wiring, no result-class edits, no JAX.
- Tests recompute quantities independently (deviance, rank, λ scaling) — not "a value exists"; fixtures shared, seed fixed.
- No plan/phase references in code or test names.
