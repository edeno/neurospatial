# Phase 5 — JAX acceleration (float32 mirror + parity)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#jax)

Add the optional float32 JAX path mirroring the phase-2 NumPy fit/REML, dispatched by the
existing encoding backend-awareness. Correctness is already established in NumPy; this phase is a
speed path that must **match the float64 core within tolerance** and **actually converge** in
float32.

**Inputs to read first:**

- [designs.md#jax](designs.md#jax) — the mirror's deltas from the NumPy core.
- `src/neurospatial/encoding/_core_jax.py` + `src/neurospatial/encoding/_backend.py` — the established `_core_numpy`/`_core_jax` dispatch pattern to follow.
- [appendix.md](appendix.md) — reference `_newton_fit_jax` (`:242`, tol floor `:257`), `_reml_score_jax` (`:445`).

**Contracts referenced:**

- [Module constants](shared-contracts.md#constants) — the float32 path uses `max(_FIT_TOL, _FIT_TOL_FLOOR)` and `_DESCENT_TOL`; float64 constants unchanged.
- [`MRFFit`](shared-contracts.md#mrffit) — the JAX path returns the identical NamedTuple (arrays cast back to float64 at the boundary).

**Designs referenced:** [designs.md#jax](designs.md#jax).

## Tasks

- **Baseline capture (before the JAX path):** run the phase-2 NumPy fit on the representative simulated population fixture and pickle `coefficients`, `firing_rate`, `penalty`, `deviance`, plus wall-clock/peak-memory (to the scratchpad or a test-local artifact). This is the parity + speedup reference.
- Create `encoding/_glm_jax.py`: mirror `_newton_fit_numpy` / `_reml_objective_numpy` with `jnp`, `jax.jit` (static `max_iter`), batched over units, `_FIT_DTYPE = jnp.float32`. Apply `tol = max(tol, _FIT_TOL_FLOOR)` and the `_DESCENT_TOL` step-halving slack. Cast inputs to float32 at entry, results back to float64 at exit.
- Dispatch: `fit_mrf_gam` **already takes `backend=` from phase-2** (phase-3 forwards the `get_backend_name(backend)`-resolved value). Phase-5 makes `backend="jax"` route the **fit compute** to `_glm_jax` (was a NumPy-compute fallback in phases 2–4) — JAX only when the extra is installed; else the NumPy core. **No signature or phase-3 change** — only the `backend="jax"` branch's body.
- **The public return contract is already fixed by phase-3** (NumPy core + convert output arrays to JAX when the resolved backend is `"jax"`). Phase-5 **only accelerates the fit compute** — it must **not** change what `compute_spatial_rates(method="glm", backend=…)` returns. The phase-3 `test_glm_backend_jax_return` contract test must stay green.
- **Comparison (after):** re-run on the identical fixture; assert `coefficients`/`firing_rate`/`deviance` match the pickled NumPy baseline within `~1e-6` (rate) and report the wall-clock/memory delta.
- Docs: note in the `compute_spatial_rate` docstring / CHANGELOG that glm accelerates with the optional JAX extra (float32; parity `~1e-6`).

## Deliberately not in this phase

- **Any change to the NumPy core's numerics** — the JAX path mirrors it; if they disagree, the NumPy core is truth and the JAX path is fixed, not the reverse.
- **Per-neuron λ** — phase-6 (its JAX mirror, if any, rides along there).
- **New public params** — dispatch is automatic (backend-driven), not a user flag.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_jax_numpy_parity` | on the simulated population: JAX `firing_rate`/`coefficients`/`deviance` match the NumPy core within `~1e-6`; `penalty`/`rank`/`penalty_rank` identical. Mark `slow` if JAX warm-up dominates. |
| `test_jax_converges_float32` | the float32 JAX fit reports `converged is True` and `n_iter < _MAX_ITER` — proving `_FIT_TOL_FLOOR` is applied; a variant passing raw `tol=1e-10` (no floor) runs to `_MAX_ITER` (guards the floor). |
| `test_jax_absent_uses_numpy` | with the JAX extra unavailable/backend off, `fit_mrf_gam` uses the NumPy core and still produces a valid `MRFFit` (no import error). |
| `test_backend_return_matches_ratio` | `compute_spatial_rates(method="glm", backend="jax")` returns the **same array-type convention** as `method="diffusion_kde", backend="jax"` (both JAX arrays, or both NumPy — whichever the ratio path does); `firing_rates.dtype` honors `dtype` (Finding 5). |
| `test_reml_parity` | REML-selected λ agrees between paths within `_REML_XATOL`-consistent tolerance. |

## Measured reality (post-implementation correction)

The `~1e-6` parity and the "no-floor runs to `_MAX_ITER`" convergence claims above
were pre-implementation estimates; measurement corrected them (metrics: relative
L2 + relative error above a rate threshold — raw relative error at near-zero
floored rates is meaningless). The shipped code, docstring, CHANGELOG, and tests
reflect the corrected reality:

- **Parity is `~1e-6` at a *fixed* λ** (isolating the Newton fit): rate relative
  L2 `~7e-7`, JAX-float64 vs NumPy-float64 `~1e-14` (confirming a faithful
  mirror). **Automatic REML is `~1e-3`**, dominated by float32 selecting a
  slightly different λ near the flat REML minimum (scientifically identical). So
  `test_jax_numpy_parity` runs at a fixed λ, and the phase-3 exact-match value
  tolerance in `test_glm_backend_jax_return` (written when `backend="jax"` fell
  back to the NumPy core) was relaxed to a float32-appropriate `~2e-3`.
- **`_FIT_TOL_FLOOR` is a *performance* floor, not a converge/diverge switch.**
  With monotone step-halving the float32 objective plateaus to bit-identical, so
  `rel_decrease` hits ~0 and even `tol=1e-10` converges (just in more Newton
  iterations, e.g. 22 vs 8); only `tol=0` runs to `_MAX_ITER`. So
  `test_jax_converges_float32` asserts the robust claims (floored converges, and
  floored `n_iter` < un-floored `n_iter`) rather than "no-floor runs to
  `_MAX_ITER`".
- **Numerical-stability fallback (added):** a design rank-deficient on the
  exposed bins has a Hessian a float32 solve cannot handle at a small/zero
  penalty (it can saturate to ~1e12 Hz while reporting convergence), so
  `fit_mrf_gam` runs any rank-deficient fit on the float64 core regardless of
  backend. Full-rank designs (the norm) keep the fast JAX path; REML never
  selects a blow-up penalty.

## Fixtures

- Reuse phase-2/3 `simulate_place_fields` (fixed seed) for parity. The pickled NumPy baseline is produced by the baseline-capture task (regenerated deterministically, not checked in unless large).
- Guard JAX tests with the project's optional-extra skip (`pytest.importorskip("jax")` or the repo's existing marker) so the base-install suite stays green.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The NumPy core is unchanged; the JAX path is a faithful mirror; parity + convergence tests pass with real numbers (the baseline/comparison pair exists, not a smoke test).
- Dispatch degrades gracefully without the extra; JAX tests are skip-guarded.
- "Deliberately not in this phase" honored — no core numeric change, no `pooled`, no user flag.
- Tests assert measured parity + convergence, not "it ran"; the speedup delta is reported.
- No plan references in code/tests; CHANGELOG updated.
