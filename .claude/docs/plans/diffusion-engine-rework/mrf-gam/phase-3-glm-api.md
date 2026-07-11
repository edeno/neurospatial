# Phase 3 — Wire `method="glm"` into the public API + result classes

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#degenerate)

Land the user-facing feature: `method="glm"` on `compute_spatial_rate` and
`compute_spatial_rates`, GAM fields on the two spatial result classes, method-specific
validation, and the degenerate-data handling. After this phase `glm` works end-to-end through the
encoding functions (decoder/persistence follow in phase-4).

**Inputs to read first:**

- `src/neurospatial/encoding/spatial.py:1894` (`compute_spatial_rate`), `:2237` (`compute_spatial_rates`), `:2135,:2542` (`_validate_smoothing_parameters` call sites), `:211,:843` (result classes), `:1724` (`summary_table`), `:168` (`to_dataframe`).
- [designs.md#degenerate](designs.md#degenerate) — the degenerate-case dispatch table.
- [shared-contracts.md#method-param](shared-contracts.md#method-param) — validation order; [#result-fields](shared-contracts.md#result-fields) — the fields to add.

**Contracts referenced:**

- [The `method` parameter + validation](shared-contracts.md#method-param) — add `"glm"` to the two spatial encoders' `Literal`; add the sentinel-`None` method params + mutual-exclusivity + value-domain validation. **Do not weaken** the `bool`-before-numeric and `is None` checks.
- [Result GAM fields](shared-contracts.md#result-fields) — add to `SpatialRateResult` / `SpatialRatesResult`; widen `bandwidth` to `float | None`.
- [`MRFBasis`](shared-contracts.md#mrfbasis) / [`MRFFit`](shared-contracts.md#mrffit) — consumed via `env._mrf_basis` (phase-1) + `fit_mrf_gam` (phase-2).

**Designs referenced:** [designs.md#degenerate](designs.md#degenerate), [#module-layout](designs.md#module-layout).

## Tasks

- **Params:** on `compute_spatial_rate` and `compute_spatial_rates`, add `"glm"` to the `method` `Literal`; convert `bandwidth`/`min_occupancy`/`fill_value` to sentinel `None` defaults and add `penalty: float | None = None`, `rank: int | None = None` ([contract](shared-contracts.md#method-param)). Resolve ratio defaults (`bandwidth→5.0`, `min_occupancy→0.0`) after validation so existing ratio behavior is unchanged.
- **Validation:** implement the mutual-exclusivity + value-domain checks ([contract](shared-contracts.md#method-param)); house the glm value-domain validators alongside `_validate_smoothing_parameters` in `encoding/_smoothing.py` (or `_glm.py`). Order: mutual-exclusivity → value domains → resolve defaults.
- **Dispatch (with explicit orientation + dtype):** when `method == "glm"`, compute occupancy (`compute_occupancy`, `(n_bins,)`) + the per-unit binned counts. The encoding side is **unit-major** `(n_units, n_bins)`; the fit is **bin-major**. So, **in this order** (Finding 1 — `basis` before any `basis.live_bins` use):
  - **build the basis first:** `basis = env._mrf_basis(occupancy, rank=R)` → `basis.live_bins` (global active-bin indices).
  - **counts in (restrict here, once):** `counts_fit = counts.T[basis.live_bins, :]` → `(n_live_bins, n_units)`; `occ_fit = occupancy[basis.live_bins]`. **This is the only live-bin restriction** — `fit_mrf_gam` does not re-slice (Finding 1).
  - **fit:** `resolved = get_backend_name(backend)`; `fit = fit_mrf_gam(basis, counts_fit, occ_fit, penalty=penalty, backend=resolved)` (**no `rank=`**; the basis fixes the rank. `backend` is forwarded so phase-5 accelerates the compute with no phase-3 change — Finding 2; `pooled` stays default `True` until phase-6 adds the public param). `MRFFit` comes back NumPy.
  - **rates out:** allocate `firing_rates = np.full((n_units, n_bins), _RATE_FLOOR)`; scatter `max(exp(fit.log_rate), _RATE_FLOOR).T` (`(n_units, n_live_bins)`) into columns `basis.live_bins`. Singular path builds `(n_bins,)`.
  - **dtype:** cast the assembled `firing_rates` to the requested `dtype` (`{np.float32, np.float64}`) at the result boundary, like the ratio path; the glm core stays float64.
  - **backend (define the contract NOW — Finding 4):** resolve with `get_backend_name(backend)` ([_backend.py:165](../../../src/neurospatial/encoding/_backend.py)) — **not** a raw `backend != "numpy"` check. In phase-3 the glm fit always runs the **NumPy core**; if the resolved backend is `"jax"`, **convert the assembled `firing_rates` (and occupancy) to JAX arrays at the return boundary**, exactly matching what the ratio path returns for `backend="jax"`. This fixes the public return contract before phase-5 exists, so phase-5 can accelerate the fit internally without changing it.
  - Ratio methods keep their current path untouched. See [designs.md → Boundary orientation](designs.md#module-layout).
- **Degenerate dispatch** ([designs.md#degenerate](designs.md#degenerate)): no-neurons, zero-total-occupancy, **all-zero-spike population** (`counts.sum()==0` → skip REML *selection*, floor fields, `reml_objective=None`, **`penalty` = supplied fixed float else `None`** — the fixed-penalty contract holds even with no data, Finding 4), dead-component (warn), zero-spike neuron, and the `penalty=0` rank-deficiency warning (`matrix_rank(B[exposed_live_bins]) < r_eff`). Each warns, none raise.
- **Result classes:** add the GAM fields ([contract](shared-contracts.md#result-fields)) to `SpatialRateResult` / `SpatialRatesResult` (all `None`/defaults for ratio results); widen `bandwidth` to `float | None`; set `bandwidth=None` for glm. Indexing a plural result stamps `unit_id` and slices `coefficients[:, i]`/`deviance[i]` (extend `__getitem__` at `spatial.py:982`). `summary_table` (`:1724`) gains the GAM scalar columns when present; `to_dataframe`/`summary`/`to_xarray` unchanged in shape.
- **Docs:** QUICKSTART + `compute_spatial_rate` docstring — a `method="glm"` example (occupancy-as-offset, `penalty=None` REML default, finite where ratio NaNs); CHANGELOG — the new estimator (spatial-only).

## Deliberately not in this phase

- **Decoder + NWB glm support** — phase-4. This phase makes `compute_spatial_rate(s)` produce glm results; persisting them / decoding with them is phase-4. (glm results simply aren't NWB-saved in this phase's tests.)
- **JAX-accelerated *fit*** — phase-5; the NumPy `fit_mrf_gam` from phase-2 is used here even when `backend="jax"` (this phase only defines the JAX **return contract** by converting the NumPy output, per the backend task above; phase-5 swaps the compute without changing that contract).
- **Per-neuron λ** — phase-6.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_glm_finite_where_ratio_nans` (headline) | low-occupancy arena: `method="glm"` all-finite; `method="diffusion_kde"` has NaN in the same bins. |
| `test_glm_result_fields` | glm `SpatialRatesResult` exposes `coefficients (rank,n_units)`, scalar `penalty`, `penalty_weights (rank,)`, `rank`, `deviance (n_units,)`, scalar `converged`/`n_iter`, `reml_objective`; `bandwidth is None`; `method == "glm"`. |
| `test_ratio_result_gam_fields_none` | a `diffusion_kde` result has all GAM fields `None` and `bandwidth` a float. |
| `test_indexing_stamps_and_slices` | `rates[i]` → singular with `unit_id == unit_ids[i]`, `coefficients (rank,)`, `deviance` scalar. |
| `test_mutual_exclusivity` | `bandwidth=5.0` + `method="glm"` raises; `penalty=1.0` + `method="binned"` raises; bare `method="glm"` (no ratio kwargs) does **not** raise. |
| `test_value_domain` | `penalty` rejects `True`/`-1`/`nan`/`inf`; `rank` rejects `True`/`2.5`/`0`; `rank` clamps both ways (`> n_live_bins` and `< n_live_components`) with `result.rank == r_eff`, no raise. |
| `test_degenerate_cases` | no-neurons / zero-occupancy / **all-zero-spike population** / dead-component / zero-spike neuron each produce the [designs.md#degenerate](designs.md#degenerate) row and warn (none raise); all-zero-spike + `penalty=None` → `penalty is None`, `reml_objective is None`, floor fields. |
| `test_all_zero_spike_fixed_penalty_public` | `compute_spatial_rate(env, no_spikes, ..., method="glm", penalty=3.0)` → `result.penalty == 3.0` (fixed-penalty contract holds through the public API with no data), `reml_objective is None`, floor fields (guards Finding 4). |
| `test_penalty0_identifiability` | `penalty=0` warns **iff** `matrix_rank(B[exposed_live_bins]) < r_eff`. |
| `test_agreement_with_ratio` | on a well-sampled arena glm and the ratio estimator agree qualitatively (peak co-location, correlation above a threshold). |
| `test_glm_orientation` | glm `firing_rates` is `(n_units, n_bins)` (unit-major, matching the ratio result); per-unit peak bins land where each simulated unit's field is — i.e. the transpose is correct, not swapped. Singular `firing_rate` is `(n_bins,)`. |
| `test_glm_dtype` | `dtype=np.float32` → `firing_rates.dtype == float32`; `np.float64` → float64; values agree within float32 tol. |
| `test_glm_backend_jax_return` | `method="glm", backend="jax"` returns the **same array-type** as `method="diffusion_kde", backend="jax"` (both JAX arrays, or both NumPy — whatever the ratio path does), resolved via `get_backend_name`; values match the `backend="numpy"` glm result. Defines the contract before phase-5. Skip-guard on the JAX extra. |
| `test_default_method_unchanged` | omitting `method` → `"diffusion_kde"`, byte-identical to the pre-phase result. |
| `test_all_layouts_smoke` | glm runs on 1D track, 2D open+masked, hex, polar, mesh (finite output). Mark `slow` if the mesh/polar builds are heavy. |

## Fixtures

- **Simulated:** reuse phase-2's `simulate_place_fields`; add a low-occupancy arena (occupancy zero in a band) for the headline test, and per-layout tiny envs for the smoke test (`conftest.py`).
- **Real-data smoke:** at least one small real-position slice (reuse whatever the existing encoding tests use for `compute_spatial_rate`) run through `method="glm"` to confirm no real-world NaN/shape surprises.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Ratio-method paths are untouched (default behavior byte-identical); glm is additive.
- Validation order + `bool`/`is None` checks match the [contract](shared-contracts.md#method-param); degenerate rows match [designs.md#degenerate](designs.md#degenerate).
- "Deliberately not in this phase" honored — no decoder/NWB glm, no JAX, no `pooled`.
- Tests assert the headline finiteness contrast, field shapes/None-ness, validation branches, and degenerate rows — not tautologies; fixtures shared; the real-data smoke runs.
- Docstrings/tests don't reference the plan; QUICKSTART + CHANGELOG updated.
