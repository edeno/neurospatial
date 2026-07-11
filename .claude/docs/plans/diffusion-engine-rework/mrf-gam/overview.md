# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

The design rationale lives in the committed spec:
[../design-mrf-gam.md](../design-mrf-gam.md). This overview covers what the *plan*
adds: integration points, scope split across PRs, metrics, risks. Section refs
below (`spec §N`) point into that spec.

## Current codebase integration points

Anchored to `feat/mrf-gam-estimator` off `main` (v0.8.0), verified 2026-07-10.

**Encoding — the rename surface (phase-0) and the glm host (phase-3):**

- `src/neurospatial/encoding/spatial.py:1894` — `compute_spatial_rate`: gains `method="glm"`; param `smoothing_method → method`.
- `src/neurospatial/encoding/spatial.py:2237` — `compute_spatial_rates`: same; this is the batched entry the decoder uses.
- `src/neurospatial/encoding/spatial.py:211,300` — `SpatialRateResult` + its `smoothing_method` field → `method`; gains GAM fields (phase-3).
- `src/neurospatial/encoding/spatial.py:843,955` — `SpatialRatesResult` + field → `method`; gains GAM fields.
- `src/neurospatial/encoding/spatial.py:3320` (`detect_place_fields`), `:3507` (`is_place_cell`) — param `smoothing_method → method` (public), internal forward becomes `compute_spatial_rate(method=…)`. **No glm** (their `Literal` stays the three ratio methods).
- `src/neurospatial/encoding/spatial.py:3166` (`compute_directional_place_fields`) — **directional**: keep the deferred bucket, but its param is renamed too for uniformity (see Non-Goals note). Internal forward updated.
- `src/neurospatial/encoding/view.py:1095,1372` (`compute_view_rate`/`compute_view_rates`), `:216,:544` (result fields) — param + field `smoothing_method → method`. No glm.
- `src/neurospatial/encoding/egocentric.py:1311,1599` (`compute_egocentric_rate`/`compute_egocentric_rates`), classes `:108,:508` — param + field rename. **Default is `"binned"` here, not `"diffusion_kde"`** — preserve it. No glm.
- `src/neurospatial/encoding/_smoothing.py:544` — `_validate_smoothing_parameters(method, bandwidth)` already uses `method` internally; only the ratio-method validation moves here / grows a glm sibling.

**Eigenbasis reuse (phase-1) — reuses PR2, adds nothing to the operator:**

- `src/neurospatial/ops/diffusion.py:290` — `_symmetric_conjugate`: builds `S = M^{-1/2}(D−W)M^{-1/2}`.
- `src/neurospatial/ops/diffusion.py:382` — `_symmetric_eigenbasis(S, rank)`: per-component eigenpairs, **global** smallest-`rank` selection at `:444-448`; **raises** on `rank < n_components` at `:419`.
- `src/neurospatial/ops/diffusion.py:201` — `_components_from_W`.
- `src/neurospatial/environment/fields.py:512` — `_diffusion_geometry` → `DiffusionGeometry(W, volumes, n_components, labels)` (the resolver reuses `W`/`volumes`/`labels`).
- `src/neurospatial/environment/fields.py:535` — `_diffusion_eigenbasis` cache (keyed by `(sigma, tol)`); phase-1 adds a rank-keyed sibling.
- `src/neurospatial/environment/core.py:1030` — `n_bins` (active bins; the rate-array length).

**Persistence + decoder (phase-0 rename, phase-4 glm):**

- `src/neurospatial/io/nwb/_fields.py:434` (`write_spatial_rates`), `:616` (writes key `"smoothing_method"`), `:697` (`read_place_field`), `:782-792` (reader requires the key) — SpatialRatesResult-specific.
- `src/neurospatial/decoding/session.py:92,100` (`decode_session` + param), `:348-356` (`_build_encoding_model`), `:566-574` (`_encode_and_bin`) — forward into `compute_spatial_rates`.
- `src/neurospatial/decoding/estimator.py:41` (`BayesianDecoder`), fields `:151-153` (`bandwidth`/`smoothing_method`/`min_occupancy`), forward `:351-353`.
- `src/neurospatial/decoding/likelihood.py:47` — `min_rate = 1e-10`: the source of `_RATE_FLOOR`.

**Incidental rename touches:** `src/neurospatial/simulation/validation.py:27,248,674`; `src/neurospatial/__init__.py:155` (doctest).

## Scope and dependency policy

### Goals

- A `method="glm"` penalized-Poisson GAM estimator on `compute_spatial_rate` **and** `compute_spatial_rates` (spec §2): occupancy as a log-offset (never a denominator), `λ` by REML, proper deviance — finite rates where the ratio estimator NaNs.
- Reuse PR2's cached finite-volume eigenbasis as the smoothness penalty basis (spec §3, §6.1); **do not** change the operator or eigensolver.
- Uniform `method` param across **all** smoothing encoders (spatial/view/egocentric) + result classes + NWB + decoder — one name, one meaning (v0.6 API contract).
- NumPy/SciPy core on the base install; optional JAX accel with verified parity.

### Non-Goals

- **glm on the other encoders.** Only `compute_spatial_rate(s)` gain the estimator. `compute_view_rate`, `compute_egocentric_rate`, `compute_directional_place_fields` keep their ratio-only `Literal` (no `"glm"`). They are renamed for API uniformity but gain no new estimator.
- Clusterless / mark-space; changing the diffusion operator or eigenbasis (spec §2).
- Per-unit convergence flags (batch-level scalar `converged`/`n_iter` only — spec §6.3).

### Dependency policy

No new required dependencies. JAX stays an **optional extra** (`_core_jax` pattern); the NumPy/SciPy core is the correctness reference and the base-install path. SciPy (`scipy.optimize.minimize_scalar`, `scipy.linalg.cholesky`) is already a core dep.

## Metrics

- **Correctness (headline):** on a synthetic arena with low-occupancy bins, `method="glm"` returns all-finite rates where `method="diffusion_kde"` yields NaN (spec §10).
- **Statistical recovery:** population fit recovers distinct per-neuron place-field peaks near simulated centers; REML selects a sensible `λ`; fixed extreme `λ` → monotone smoothing.
- **Reference fidelity:** deviance, penalty-rank, REML objective match an independent recomputation (not "a value exists").
- **JAX parity:** float32 JAX vs float64 NumPy agree within `~1e-6`; the float32 path actually converges (proves the `_FIT_TOL_FLOOR` floor is applied).
- **No regression:** default `method="diffusion_kde"` behavior byte-identical after phase-0 rename; full suite green at every phase boundary.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Rename cascade misses a `smoothing_method` reader → broken tests | Phase-0 is a single atomic behavior-preserving PR across all sites (grep-complete list in [phase-0](phase-0-rename.md)); suite must be green before merge. |
| `r==0` all-null basis returns arbitrary λ | Structural rank `r = r_eff − n_live_components` with per-component nulls zeroed exactly (spec §5); the decisive two-3-node-paths test guards it ([phase-2](phase-2-fit-reml.md)). |
| Dead components consume the rank budget | Live-component selection *before* the rank budget is spent (spec §6.1); resolver returns only live-component modes ([phase-1](phase-1-eigenbasis-resolver.md)). |
| Over-request approaches a full eigendecomposition on large dead components | Documented perf caveat (spec §11); component-local-eigenpair mitigation is the named fallback, revisited only if the `R` cap stops holding. Not blocking for correctness. |
| float32 objective noise stalls convergence | dtype-dependent tol: `_FIT_TOL_FLOOR=1e-6` + `_DESCENT_TOL=1e-5` on the JAX path (spec §4); parity test asserts convergence ([phase-5](phase-5-jax-accel.md)). |
| Degenerate data (no neurons / zero occupancy / dead comp / zero-spike) produces inconsistent fields | Per-case table with model-consistent outputs (spec §7); each row is a test ([phase-3](phase-3-glm-api.md)). |

## Rollout Strategy

Ships all-at-once as **0.9.0** — no feature flag, no deprecation window (per project policy: hard rename, no back-compat shim). The `method` param default stays `"diffusion_kde"`, so callers who never touched `smoothing_method` and never pass `method="glm"` see identical behavior; only callers *naming* `smoothing_method=` or reading `.smoothing_method` break, and the CHANGELOG documents the rename. Old NWB files still **read** (defensive fallback on the old key). The release is cut after the feature phases land, not per-phase.

## Open Questions

1. **Component-local eigenpairs vs global over-request** (spec §11) — deferred: the global-basis baseline ships; the per-component `eigsh` mitigation is implemented only if the perf-benchmark task shows the `R` cap breaking. Not a correctness question.
2. **JAX `_core_jax` module boundary** — resolved in [designs.md](designs.md#module-layout): NumPy fit in `_glm_numpy.py`, JAX mirror in `_glm_jax.py`, dispatched by the existing backend-awareness machinery (`encoding/_backend.py`).

## Estimated Effort

Rough diff sizing (implementation + tests), for expectation-setting only:

- phase-0 rename: wide but mechanical — ~9 src files + ~20 test files, mostly find-replace + forward-call fixes. Low logic, high surface.
- phase-1 resolver: ~150 LOC src + tests.
- phase-2 fit+REML: ~300 LOC src + tests (the statistical core).
- phase-3 glm API + results: ~250 LOC src + tests (dispatch, result fields, validation, degenerate cases).
- phase-4 persistence+decoder: ~150 LOC src + tests.
- phase-5 JAX: ~200 LOC src + parity tests.
- phase-6 per-neuron λ: ~120 LOC src + tests.
