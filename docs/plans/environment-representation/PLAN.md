# Environment Representation Implementation Plan

**Committed**: 2026-10-13

**Status:** In progress on `feature/environment-representation`.

| Phase | Status |
| --- | --- |
| 1 — Binning snapshot | Complete |
| 2a — Data locators | Complete |
| 2b — Shapes and plots | Pending |
| 3 — Save binnings | Pending |
| 4 — Derived environments | Pending |

Validation evidence: [validation.md](validation.md).

Environments will store their bins, point lookup and optional structure as data,
built once by a layout and then kept. An environment made by subsetting,
transforming, saving or loading then keeps every feature its data supports:
smoothing, gradients, plotting, linearization and true bin sizes. Today such an
environment gets a stand-in layout that loses those features. These five phases
prove the design on every built-in layout, with the grid, hexagonal and track
types as the focus. The continuous arena, an immutable environment value,
`combine`, wall-aware analyses and removal of the layout engine follow as later
stages, listed in the roadmap in [overview.md](overview.md#roadmap-after-phase-4).

## Reading order

When working with an agent, **load only the part you need**:

1. **Working on a specific phase?** Open the matching phase file. Each one is
   self-contained: it lists the files to read first, the contracts and designs
   it depends on, its tasks, its validation slice and its fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need an algorithm or a format?** [designs.md](designs.md).
4. **Need the broader scope, risks or roadmap?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — the problem, the target design, integration points, goals, risks, the roadmap for stages A–F, and open questions.
- [shared-contracts.md](shared-contracts.md) — `SpatialBins`, `FacesSource`, `Locator`, the components, `Binning`, publication, the `BinningLayout` facade, codecs, and the derivation rules.
- [designs.md](designs.md) — building a binning from a layout, the locator moves, plotting from data, the file format, and the subset and transform algorithms.
- Phases (each ships as a separate PR):
  - [phase-1-binning-snapshot.md](phase-1-binning-snapshot.md) — publish one `Binning` per environment; environment code reads capabilities from it. No output changes.
  - [phase-2a-data-locators.md](phase-2a-data-locators.md) — data-backed locators and track linearization, each with one implementation. No output changes.
  - [phase-2b-shapes-and-plots.md](phase-2b-shapes-and-plots.md) — hexagon and triangle shapes, outlines, and plotting from data. No output changes.
  - [phase-3-save-binnings.md](phase-3-save-binnings.md) — files and NWB store the binning; loading never rebuilds; fixes NWB read-back.
  - [phase-4-derived-environments.md](phase-4-derived-environments.md) — subset and transforms derive a binning under explicit rules; fixes translated and subset environments.
