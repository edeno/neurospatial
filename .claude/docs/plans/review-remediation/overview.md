# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

This plan operationalizes [.claude/reviews/ROADMAP.md](../../../reviews/ROADMAP.md). The roadmap's nine themes map onto the 25 phase PRs as follows; exact `file:line` refs live in each phase file (verified there against live source), not here.

| Roadmap phase | Phase PRs |
| --- | --- |
| 0 — silent-correctness criticals | 1–13 (by subsystem) |
| 1 — numerical-robustness guards | folded into 2,3,6,8,9 (dt/finite guards ship with the module they protect) |
| 2 — API crashes + spike seam | 14,15,16; the Graph/Polygon `to_file` crash and `interpolate` reshape ship in 9 and 2 |
| 3 — statistical validity | folded into 1,4,5 (phase-precession p-value in 5; shuffle p-value in 4) |
| 4 — ergonomics & composability | 17,18,19,20,21 |
| 5 — consistency / naming | 22 |
| 6 — docs runnable + CI | 23 |
| 7 — domain capability gaps | 24 |
| 8 — test backfill | 25 (plus each correctness phase ships its own regression tests) |

## Current codebase integration points

Module-level touch map. Each correctness phase is scoped to a single subsystem so its PR reviews in isolation.

- `src/neurospatial/decoding/` — `assemblies.py` (reactivation stats), `posterior.py`/`likelihood.py` (dimension checks), `estimates.py`/`metrics.py` (NaN handling). Phase 1.
- `src/neurospatial/environment/` — `trajectory.py` (occupancy, timestamps), `fields.py` (interpolate), `transforms.py` (rebin), `factories.py`/`core.py` (polar; phase 19). Phases 2, 19.
- `src/neurospatial/simulation/` — `models/place_cells.py`, `spikes.py`. Phase 3.
- `src/neurospatial/stats/` — `circular.py`, `shuffle.py`, `surrogates.py`. Phase 4.
- `src/neurospatial/encoding/` — `directional.py`, `_directional_binning.py`, `phase_precession.py`, `egocentric.py`, `view.py`. Phases 5, 22.
- `src/neurospatial/behavior/` — `segmentation.py`, `decisions.py`, `navigation.py`, `vte.py`. Phase 6.
- `src/neurospatial/events/` — `intervals.py`, `regressors.py`, `detection.py`. Phases 7, 22.
- `src/neurospatial/ops/` — `binning.py`, `egocentric.py`, `alignment.py`, `basis.py`, `transforms.py`. Phase 8.
- `src/neurospatial/io/` — `files.py`, `nwb/_environment.py`, `nwb/_adapters.py`, `nwb/_behavior.py`, `nwb/_pose.py`; new `nwb/_units.py`. Phases 9, 15.
- `src/neurospatial/layout/engines/` — `image_mask.py`, `masked_grid.py`, `regular_grid.py`, `graph.py`; `helpers/utils.py`. Phase 10.
- `src/neurospatial/regions/` — `io.py`, `ops.py`, `core.py`. Phase 11.
- `src/neurospatial/annotation/` — `_boundary_inference.py`, `_track_state.py`, `validation.py`, `converters.py`. Phase 12.
- `src/neurospatial/animation/` — `transforms.py`, `overlays.py`, `_parallel.py`. Phase 13.
- `src/neurospatial/__init__.py` and submodule `__init__.py` — exports for the spike binner, NWB unit readers, lazy `__getattr__`, re-exports. Phases 14, 15, 21.
- `.claude/QUICKSTART.md`, `.claude/API_REFERENCE.md`, `docs/` — runnable-example fixes. Phase 23 (and each phase updates docstrings it touches).

## Scope and dependency policy

### Goals

- Eliminate every silent-wrong-result path the review found: invalid input (NaN/Inf, wrong dtype, mismatched lengths, out-of-range indices) must fail loudly or be handled correctly, never flow through to plausible-but-wrong numbers.
- Make the canonical spike workflows (NWB → bin spikes → decode/GLM/reactivation) achievable through the public API without bare-pynwb / hand-rolled `np.histogram` glue.
- Bring cross-module API conventions (naming, units, result objects, argument order) into consistency, and make every first-contact documentation example runnable and CI-gated.
- Backfill the validation/failure-branch tests whose absence let these bugs survive.

### Non-Goals

- No new analysis methods beyond closing identified domain gaps (ripple/theta) — and those are scope-gated in phase 24.
- No performance optimization except where a fix incidentally requires it (Theme "performance-memory" suggestions are out of scope for this plan).
- No migration to JAX/xarray internals; `to_xarray()` (phase 20) is an output adapter only.
- No backwards-compatibility shims: per the project's pre-1.0 policy this is replace-in-place (see Rollout).

### Dependency policy

No new required runtime dependencies. `to_xarray()` (phase 20) imports `xarray` lazily inside the method and is optional. Ripple detection (phase 24) must not add a hard dependency — if it needs filtering beyond `scipy`, scope it out and point to `ripple_detection`.

## Dependency ordering

- **Phases 1–13 are mutually independent** (disjoint subsystems) and may ship in any order or in parallel. Recommended order = scientific blast radius: 1, 2, 3, 5, 4, then 6–13.
- **Phase 14 (spike binner)** has no deps; **phase 15 (read_units)** depends on nothing but pairs naturally with 14; **phase 16 (occupancy fill)** depends on phase 2 (occupancy fix) landing first.
- **Phase 17 (result mixin)** should land before or with phase 20 (`to_xarray`) since `to_xarray` attaches to the mixin; phase 18 (direction labels) is independent.
- **Phase 19 (polar env)** subsumes the polar Theme-4 correctness items; do not also patch them piecemeal in a correctness phase — they are deliberately deferred here.
- **Phase 22 (naming)** touches files across many subsystems; land it after the correctness phases for those files to avoid churn/conflicts.
- **Phase 23 (docs CI)** should land after the API changes in 14–22 so examples reflect the final API.
- **Phase 25 (test backfill)** lands last for anything not already covered by per-phase regression tests.

## Metrics

- `uv run pytest` green at every phase boundary; each correctness phase adds a regression test that **fails before the fix and passes after** (named in its Validation slice).
- Zero remaining `except Exception:`/bare-`except` that downgrades a scientific result silently in the touched files (grep audit per phase).
- After phase 23: `uv run pytest --doctest-modules src/neurospatial/` and the docs-example CI job pass; no QUICKSTART/README snippet raises on paste.
- After phases 14–15: the "NWB → place fields → decode → animate" journey is writable with zero bare-pynwb calls and zero hand-rolled spike-binning loops.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A "fix" silently changes a scientific number users already depend on | Every correctness phase ships a regression test pinning the corrected value; PR review (`scientific-code-change-audit` lens) compares before/after on a fixture. |
| New validation raises on inputs that previously "worked" (silently wrong) | Intended per pre-1.0 policy; phase docstrings + CHANGELOG note the new error and the correct usage. |
| Polar-env redesign (19) is a wide blast radius (factories, core, io, encoding) | Gated on a design decision (Open Q3); ships as its own PR; correctness phases deliberately do **not** also touch polar paths. |
| Result-object contract (17) churns many classes | Gated on Open Q2; introduce the mixin additively (new methods, no removals) so existing accessors keep working. |
| Naming changes (22) break user code | Pre-1.0 replace-in-place; CHANGELOG lists every rename; done in one PR so the break is atomic and documented. |

## Rollout Strategy

Replace-in-place, no deprecation cycle (project is v0.3.x; CLAUDE.md default is "just change the code"). Each phase is one PR, merged when green and independently reviewed. Behavior changes that turn previously-silent-wrong paths into raised errors are documented in that phase's docstring updates and a CHANGELOG entry. No feature flags.

## Open Questions

1. **(Phase 16) `min_occupancy` NaN reconciliation — where does the fix live?** RESOLVED (2026-06-02): add `fill_value: float | None = None` (opt-in, NOT `0.0`) to `compute_spatial_rate(s)` — the default `None` preserves current NaN behavior, and users opt into zero-fill with `fill_value=0.0`; keep the decoder-side defensive zero-rate handling of residual NaN bins (one-time warning), compatible with phase 1's `validate=True`. The documented golden path passes `fill_value=0.0` explicitly.
2. **(Phase 17) Result-object contract shape.** RESOLVED (2026-06-02): a `ResultMixin` guaranteeing `.to_dataframe()`, `.summary()`, and `.plot()`, with `.plot()` **optional per result type** (base may provide a `NotImplementedError`/`raise` default; not every result must implement it). `SpatialResultMixin` re-parents under `ResultMixin` (additive — existing accessors preserved). Backfill `DirectionalPlaceFields`/`PlaceFieldsResult`.
3. **(Phase 19) Polar egocentric env: new type vs. marked wrapper.** RESOLVED (2026-06-02): a **separate** `EgocentricPolarEnvironment` class implementing the shared environment protocol — NOT a subclass of `Environment`. `from_polar_egocentric` stays the public entry-point name and returns the new type (kept working as a module-level factory or moved onto the new class). Geodesic: fix the polar edge lengths.
4. **(Phase 24) Ripple/theta scope.** RESOLVED (2026-06-02, split): **implement `encoding.theta_phase` in-library** (scipy Hilbert on theta-band-filtered LFP); **scope OUT ripple detection** — direct users to the `ripple_detection` package and feed its returned intervals into `peri_event_histogram` (no hard dependency added).

## Estimated Effort

- Correctness phases (1–13): mostly small, surgical diffs — typically 1 fix + 1 regression test per finding, ~50–200 LOC each including tests. Phase 1 (reactivation) and phase 2 (occupancy index) are the most algorithmically involved.
- API phases (14, 15, 20, 21): new public functions, ~100–250 LOC each with tests and docstrings.
- Gated phases (16, 17, 19): scope depends on the decision; 17 and 19 are the largest (multi-file).
- Phase 22 (naming): wide but mechanical. Phase 23 (docs): high file count, low per-file complexity. Phase 25 (tests): bounded by the Theme-6 list.
