# Researcher-First Rebuild Implementation Plan

**Status:** Phase 1 done (2d7017c2, PR #36); next: Phase 2a.

Rebuild the valuable outcomes of the archived `feat/public-api-curation` branch on a fresh branch from `main`, putting researcher effort first. The result: the four-array call `compute_spatial_rate(env, spike_times, times, positions)` and its siblings work with no setup and handle recording gaps correctly. Every verified scientific bug is fixed. Errors say how to fix the problem. Every advertised example runs in CI. The public surface is coherent and discoverable, and it's protected by one snapshot test rather than a governance apparatus.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Executing any phase?** Read [executing.md](executing.md) first: branch and PR workflow, definition of done, and what to do when reality differs from the plan. Then open the matching phase file. Each phase file is self-contained for its tasks.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope, decisions, ordering or risks?** [overview.md](overview.md).

## Files

- [executing.md](executing.md) — how every phase is executed and when it is done.
- [overview.md](overview.md) — goals, non-goals, settled design decisions, ordering, risks.
- [shared-contracts.md](shared-contracts.md) — the time-window semantics, error-message contract, input conventions and API snapshot rule shared by several phases.
- **Phases.** Each ships as one PR, in this order. The full dependency list is in [overview → Rollout Strategy](overview.md#rollout-strategy).
  - **Phase 1 and Phase 2: fixes.**
    - [phase-1-port-fixes.md](phase-1-port-fixes.md) — enable CI on this branch; port the verified scientific fixes from the archive branch.
    - [phase-2a-main-bugs-geometry-io.md](phase-2a-main-bugs-geometry-io.md) — W-maze linearization, mirrored polar plots, NWB unit conversion and units, simulation field width, PETH and VTE bugs.
    - [phase-2b-main-bugs-operators-behavior.md](phase-2b-main-bugs-operators-behavior.md) — finite-volume calculus and basis operators, heading interpolation, the behavior stationary filters.
  - **Phase 3: time windows.**
    - [phase-3a-time-windows-core.md](phase-3a-time-windows-core.md) — interval helpers, the extended `interval_valid_mask`, the silence warning and spatial rates.
    - [phase-3b-time-windows-frame-families.md](phase-3b-time-windows-frame-families.md) — the shared frame kernel for the directional, view and egocentric rate families.
    - [phase-3c-time-windows-decoding-events.md](phase-3c-time-windows-decoding-events.md) — per-run decode time bins; event filtering by `epochs` and `spike_window`.
    - [phase-3d-time-windows-segmentation.md](phase-3d-time-windows-segmentation.md) — laps, trials, crossings and environment sequence methods never span an invalid interval.
    - [phase-3e-time-windows-kinematics.md](phase-3e-time-windows-kinematics.md) — kinematics, `heading_from_velocity(times, …)`, `add_positions`, with docs and notebooks.
  - **Phase 4: errors and docs.**
    - [phase-4a-errors.md](phase-4a-errors.md) — `NeurospatialError`, first-run error messages, detection of silent mistakes.
    - [phase-4b-docs-that-run.md](phase-4b-docs-that-run.md) — docstring completeness; README, quickstart, CLAUDE.md and docstring examples executed in CI.
  - **Checkpoint:** a researcher-workflow review must pass before Phase 5a ([overview → Rollout Strategy](overview.md#rollout-strategy)).
  - **Phase 5: classification.**
    - [phase-5a-object-vector-frames.md](phase-5a-object-vector-frames.md) — the `label_cell_types` fix, the allocentric object-vector map and results, frame-specific functions, the simulator.
    - [phase-5b-significance-predicates.md](phase-5b-significance-predicates.md) — opt-in shuffle significance, one predicate contract, `has_place_field` and `is_place_cell(criterion=…)`, bias documentation.
  - **Phase 6: API surface.**
    - [phase-6a-namespaces-snapshot.md](phase-6a-namespaces-snapshot.md) — curated namespaces, governance remnants removed, the API snapshot test.
    - [phase-6b-argument-conventions.md](phase-6b-argument-conventions.md) — the `(times, positions)` order, the swap-detecting validator, `PositionLike` removal, decoder `unit_ids`.
    - [phase-6c-data-holders.md](phase-6c-data-holders.md) — simulator and NWB data holders; `Session` and `load_session` deleted.
  - **Phase 7: output.**
    - [phase-7-output-polish.md](phase-7-output-polish.md) — population summaries, single/batch `summary_table` parity, visible thresholds, `overwrite=False` for animation export.
