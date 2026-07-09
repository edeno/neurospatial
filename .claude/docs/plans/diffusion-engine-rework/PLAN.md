# Diffusion Engine Rework — Correctness (PR1) Implementation Plan

**Status:** Not started.

Makes the diffusion-smoothing `bandwidth` the true physical σ on every environment layout,
fixing a grid-size-dependence bug where the same `bandwidth` produced different amounts of
smoothing at different bin sizes (2D overshoots ~70% at bin_size=1). Replaces the kernel with
a single finite-volume operator, gives each smoothing consumer the correct orientation
(extensive vs intensive vs transition), and ships a public `mode="average"` intensive-field
smoother. Design: [design-correctness.md](design-correctness.md) (committed `0de7d6a`).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — self-contained (inputs,
   contracts/designs, tasks, validation, fixtures, review).
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design / code?** [designs.md](designs.md).
4. **Need broader scope / integration / risks?** [overview.md](overview.md).
5. **Need upstream (NLD) refs / FV background?** [appendix.md](appendix.md).
6. **Need the why?** [design-correctness.md](design-correctness.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, rollout.
- [shared-contracts.md](shared-contracts.md) — C1–C6: operator, modes, per-geometry data, canonical `M`, `W`-components, low-level API.
- [designs.md](designs.md) — D0–D5: module assembly, per-geometry builders, junction contraction, mesh skew guard, `heat_kernel`, masked `H`-average.
- Phases (each ships as a separable PR):
  - [phase-1-operator-and-geometries.md](phase-1-operator-and-geometries.md) — finite-volume operator, all geometries, `transition`/`density` modes, reframed low-level API, migration + full σ-recovery test suite.
  - [phase-2-intensive-fields-and-average-mode.md](phase-2-intensive-fields-and-average-mode.md) — public `mode="average"` (`H`), route `binned` + diffuse-`resample_field` through the masked `H`-average.
- [appendix.md](appendix.md) — NLD `_neighbor_centers` reference; TPFA/finite-volume background.
- [design-correctness.md](design-correctness.md) — approved design spec (source of truth).
