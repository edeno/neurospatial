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
- Dispatch: `fit_mrf_gam` selects the NumPy vs JAX **fit** via `get_backend_name(backend)` ([_backend.py:165](../../../src/neurospatial/encoding/_backend.py), same helper phase-3 uses) — JAX only when the extra is installed and the resolved backend is `"jax"`; otherwise the NumPy core.
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
