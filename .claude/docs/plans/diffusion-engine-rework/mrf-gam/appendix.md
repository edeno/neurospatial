# Appendix — reference source map

[← back to PLAN.md](PLAN.md)

The estimator is ported from `non_local_detector`. **Reference file** (commit `89c943c`):
`~/Documents/GitHub/non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_mrf.py`.

Port the *algorithm*, not the storage convention: NLD stores **full-grid** rate maps (off-track
bins = 0); neurospatial stores **active-bin** arrays (`env.n_bins`). The resolver's `live_bins`
indexing replaces NLD's interior/pixellation bookkeeping.

## Function / constant map

| ref line | symbol | maps to (this plan) |
| --- | --- | --- |
| `:78` `_FIT_DTYPE` | float32 fit dtype | JAX path only (phase-5); NumPy core is float64. |
| `:80,82,84` `_ETA_CLIP`/`_MAX_STEP_HALVINGS`/`_HESSIAN_JITTER` | solver guards | [constants](shared-contracts.md#constants). |
| `:88,89` `_FIT_TOL_FLOOR`/`_DESCENT_TOL` | float32 tol floor + descent slack | [constants](shared-contracts.md#constants); phase-5. |
| `:91,96` `_LOG_PENALTY_BOUNDS`/`_DEFAULT_MAX_RANK` | REML bounds, rank cap | [constants](shared-contracts.md#constants). |
| `:99` `_as_positive_int` | positive-int validator (rejects `bool`) | `rank` validation (phase-3), [method-param](shared-contracts.md#method-param). |
| `:115,128` `_as_positive_float`/`_as_nonnegative_float` | float validators | `penalty` validation (phase-3). |
| `:160` `_validate_mrf_problem` | shape/degenerate guards | degenerate dispatch ([designs](designs.md#degenerate)), phase-3. |
| `:229` `_penalized_hessian` | `Bᵀ diag(μ) B + diag(penalty·d)` | fit Hessian ([designs](designs.md#fit)), phase-2. |
| `:242` `_newton_fit_jax` | batched Newton/IRLS + step-halving | `_newton_fit_numpy` (phase-2) / `_glm_jax` (phase-5). Converge on penalized objective; batch-scalar `converged`. |
| `:354` `mrf_penalized_poisson_fit` | fit entry (+ diagnostics) | `fit_mrf_gam` orchestrator ([designs](designs.md#module-layout)), phase-2. |
| `:436` `_penalty_rank` | **relative** `count(d > 1e-12·max)` — **NOT ported as-is** | replaced by the structural `r = r_eff − n_live_components` (spec §5, [MRFBasis](shared-contracts.md#mrfbasis)); the reference's relative rule is the P1 bug we avoid. |
| `:445` `_reml_score_jax` | REML score at one λ (loglik + penalty − ½·r·logλ + ½·logdet) | `_reml_objective_numpy` (phase-2). |
| `:477` `mrf_reml_objective` | REML objective wrapper | folded into `_reml_objective_numpy`. |
| `:550` `select_penalty_by_reml` | bounded `minimize_scalar` over log λ | `select_penalty_by_reml` (phase-2, [designs](designs.md#reml)). |
| `:665` `fit_sorted_spikes_mrf_encoding_model` | top-level orchestration + degenerate special-cases | split across `fit_mrf_gam` (phase-2) + degenerate dispatch (phase-3). |
| `:840-899` | REML/fit call + `converged` **warning** + diagnostics dict | nonconvergence warning (phase-2/3, spec §4); GAM diagnostics → result fields ([result-fields](shared-contracts.md#result-fields)). |

## Key divergences from the reference

1. **Storage:** active-bin arrays, not full-grid. `firing_rate` is `(n_bins,)`; dead bins `_RATE_FLOOR`.
2. **Penalty rank:** structural (spec §5), not the relative-threshold `_penalty_rank`. This is a deliberate correctness fix, not a port.
3. **Rate floor:** `_RATE_FLOOR = 1e-10` (matches neurospatial decoding `min_rate`, `likelihood.py:47`), not NLD's `1e-15`.
4. **NumPy-core-first:** the float64 NumPy/SciPy path is the correctness reference and base-install path; JAX (NLD's only path) is an optional accel mirror here.
5. **Rank clamp:** `r_eff = max(n_live_components, min(n_live_bins, R))` — both-ways clamp with `result.rank` reporting; the reference caps only at the top.
