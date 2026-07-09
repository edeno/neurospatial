# Overview — Scope, integration, risks

[← back to PLAN.md](PLAN.md)

Full rationale and epistemic status: [design-correctness.md](design-correctness.md) (committed,
`0de7d6a`). This file is the cross-phase execution context; contracts are in
[shared-contracts.md](shared-contracts.md), algorithms in [designs.md](designs.md).

**Terminology (spec ↔ plan).** The design spec's **"PR1"** = this plan's entire correctness
effort = **Phase 1 + Phase 2** (two PRs). Spec **"PR2"** = performance (eigenbasis +
truncation), a separate plan; spec **"(c)"** = MRF-GAM, a separate plan. So where the spec
says `mode="average"` "ships in PR1, nothing deferred," that means **Phase 2 here** — within
the correctness effort, not pushed out to the performance/MRF work. The spec and this plan
are **both committed** on branch `fix/diffusion-grid-bandwidth` (spec `0de7d6a`, stable; the
plan's own commit hash advances with each fix so it is not pinned here); treat the spec as
the frozen source of truth and the plan as its executable
decomposition.

## Current codebase integration points

- `src/neurospatial/ops/diffusion.py` — **new** module: the finite-volume operator + dispatch (Phase 1).
- [ops/smoothing.py:52-234](../../../src/neurospatial/ops/smoothing.py) — `compute_diffusion_kernels` **rewritten** (new signature, edge-`"A"`); `_assign_gaussian_weights_from_distance` **deleted**. `apply_kernel` (:237+) **code**
  untouched, but its **adjoint** under the new (non-symmetric on non-uniform `M`) density
  kernel is **regression-tested** in Phase 1 (spec §4) — "untouched code" ≠ "unverified behavior".
- [ops/__init__.py:99,188](../../../src/neurospatial/ops/__init__.py) — public export of `compute_diffusion_kernels` (signature break; CHANGELOG).
- [fields.py:124-142](../../../src/neurospatial/environment/fields.py) — `compute_kernel` re-routed to `ops/diffusion.py`; [fields.py:48,155,285](../../../src/neurospatial/environment/fields.py) — mode Literals/`valid_modes` gain `"average"` (Phase 2). `smooth`/`compute_kernel` docstrings updated.
- [trajectory.py:495](../../../src/neurospatial/environment/trajectory.py) — `occupancy(bandwidth=)`: unchanged (`mode="transition"` = `Hᵀ`, still mass-conserving). [trajectory.py:820-870](../../../src/neurospatial/environment/trajectory.py) — `transitions(method="diffusion")`: returns row-stochastic `H` via transpose (Phase 1).
- [encoding/_smoothing.py:665,789,847,934](../../../src/neurospatial/encoding/_smoothing.py) — `binned` paths routed to `mode="average"` (Phase 2). The KDE paths (`_diffusion_kde` :569 etc.) keep `density` — correct (extensive).
- [ops/binning.py:790-815](../../../src/neurospatial/ops/binning.py) — `resample_field(method="diffuse")` → masked `H`-average (Phase 2).
- [_protocols.py:50,94-101,307](../../../src/neurospatial/environment/_protocols.py) — cache-key/Literal gain `"average"`; `bin_sizes` docstring fixed (Phase 1).
- Layout `bin_sizes` (the canonical `M`, C4): [mixins.py:387](../../../src/neurospatial/layout/mixins.py) (generic, uniform — untouched, nonuniform-Cartesian out of scope), [polar.py:413](../../../src/neurospatial/environment/polar.py), [hexagonal.py:347](../../../src/neurospatial/layout/engines/hexagonal.py), [graph.py:407](../../../src/neurospatial/layout/engines/graph.py), [triangular_mesh.py:406](../../../src/neurospatial/layout/engines/triangular_mesh.py) — all read as-is, not modified.

## Scope and dependency policy

### Goals

- `bandwidth` is the true physical σ for the diffusion kernel on every supported layout,
  independent of bin size/resolution.
- One operator path (finite-volume `H`), no parallel v1/v2.
- Correct orientation per consumer: extensive smoothing (`Hᵀ`), count→density (`H·M⁻¹`),
  intensive averaging (`H`), row-stochastic transitions (`H`).

### Non-Goals

- **Performance** (dense `expm` → cached eigenbasis + spectral truncation) — PR2, separate plan.
- **MRF-GAM** penalized-Poisson encoding estimator — separate design.
- **Nonuniform-Cartesian `grid_edges`** physical-σ guarantee — excluded (C4); `_GridMixin.bin_sizes` stays uniform.
- **`min_occupancy` unit fix** (seconds vs seconds/volume) — out of the grid-invariance guarantee; documented, tracked follow-up.
- **Flipping `env.smooth`'s default mode** — stays `density`.

### Dependency policy

No new runtime dependencies (`scipy.sparse`, `scipy.sparse.linalg`, `networkx`, `numpy`
already present). JAX paths in `encoding/_smoothing.py` stay optional-extra as today.

## Metrics

- Grid-independence: measured point-source σ within **rtol 0.02** of `bandwidth` across bin
  sizes (Cartesian 1D/2D — empirically established by
  [scripts/diffusion_grid_bandwidth_check.py](../../../scripts/diffusion_grid_bandwidth_check.py)).
- Per-geometry σ-recovery within a-priori tolerances pinned in each test (hex/polar/graph/mesh).
- Contract tests green: `transition` column-stochastic, `transitions()` row-stochastic,
  `density` integrates to 1, `average` row-stochastic, `W`-component split, no-leakage.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A non-Cartesian geometry's σ claim fails at implementation | It's *analytic + smoke-test-gated* (spec §2); a miss is an operator/spec correction, caught by the σ-recovery test **before** routing production callers — not a tolerance bump |
| Mesh TPFA inaccurate on skewed triangulations | E2 accepted; skew guard (D3) warns loudly; σ asserted only well-shaped |
| Operator rewrite changes existing users' smoothed values (even uniform grids) | Intended (the σ correction); loud CHANGELOG + migration note + 0.7.0 bump |
| Golden-value tests pinning old smoothed numbers break | Expected fallout; enumerate + recompute in Phase 1 (they encode the old, wrong operator) |
| Public `compute_diffusion_kernels` signature break hits a downstream caller | Documented as a public break in CHANGELOG; in-repo callers updated in Phase 1 |

## Rollout Strategy

Replace-in-place, no deprecation window (pre-1.0; correctness fix). Ships across two PRs
(Phase 1 then Phase 2), each green and independently reviewable. Version 0.6.0 → 0.7.0 with a
migration note; users approximating the old amount of smoothing scale `bandwidth` by
~`√(bin_size)` (rough, mode-dependent).

## Open Questions

1. Exact hex side-length / center-distance source — derive from `bin_centers`/`bin_sizes` or
   a layout-stored value? Current best: derive geometrically in `_hex_fv` (D1); the
   σ-recovery test validates. Deferred to implementation, not blocking.
2. Per-geometry a-priori tolerances (hex/polar/graph/mesh) — set from each discretization's
   expected error when the test is written; **not** tuned to pass (spec §4).

## Estimated Effort

Phase 1: ~500–800 LOC (new `ops/diffusion.py` ~300–400, `compute_diffusion_kernels` rewrite,
5 geometry builders, junction contraction, migration, ~14 tests + fixtures). Phase 2:
~150–250 LOC (mode threading, 2 caller reroutes, ~5 tests). Golden-value recompute count TBD
at Phase 1 (grep tests pinning smoothed numbers).
