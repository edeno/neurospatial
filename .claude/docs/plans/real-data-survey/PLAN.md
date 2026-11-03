# Real-data survey Implementation Plan

**Status:** In progress — phases 1–3 merged (#260, #261, #263); next: phase 4 (gap synthesis).

Build a tracked `datasets/` folder that loads public spatial-neuroscience
recordings into plain arrays, runs a fixed set of neurospatial operations on
each, and records the outcome in a dataset × operation table. Every place a
loader or operation needed a workaround, where neurospatial disagrees with an
independent tool or with what we expected beforehand, or where its warnings fire
on good data or stay silent on known defects, goes into a gap log. The
log, not the scripts, is the product: it is the evidence for what neurospatial
is missing. The first batches cover seven datasets chosen for variety (2-D grid
cells, head direction, 1-D figure-8 with replay, 3-D bat flight, imaging, a
behavior-only graph maze, and NWB files with misplaced or mislabelled data).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
5. **Need dataset access details (IDs, URLs, known quirks)?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, dependency policy, risks
- [shared-contracts.md](shared-contracts.md) — the `Recording` record, loader module contract, operation/outcome contract, gap-log entry format
- [designs.md](designs.md) — fetch/cache helper, record validation, operation registry, driver, warning calibration, reference checks
- Phases (each ships as a separable PR):
  - [phase-1-harness-reference-sets.md](phase-1-harness-reference-sets.md) — harness + the two datasets with independent reference numbers (000582 grid cells, Mouse32 head direction) + reference cross-checks
  - [phase-2-tracks-graphs-3d.md](phase-2-tracks-graphs-3d.md) — 000552 figure-8 (1-D, replay), Rosenberg labyrinth (graph maze, behavior only), Yartsev bat (3-D)
  - [phase-3-imaging-quirky-nwb.md](phase-3-imaging-quirky-nwb.md) — 002030 miniscope (imaging input), 000405 + 001701 (misplaced and mislabelled NWB data)
  - [phase-4-gap-synthesis.md](phase-4-gap-synthesis.md) — rank the gap log into proposed neurospatial changes and pick the next batch
- [appendix.md](appendix.md) — per-dataset access details and known quirks, from the survey report
