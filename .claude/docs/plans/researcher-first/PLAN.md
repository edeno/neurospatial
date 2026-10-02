# Researcher-First Rebuild Implementation Plan

**Status:** Not started.

Rebuild the valuable outcomes of the archived `feat/public-api-curation` branch on a fresh branch from `main`, putting researcher effort first. The result: the four-array call `compute_spatial_rate(env, spike_times, times, positions)` and its siblings work with no setup and handle recording gaps correctly. Every verified scientific bug is fixed. Errors say how to fix the problem. Every advertised example runs in CI. The public surface is small enough to learn, and it's protected by one snapshot test rather than a governance apparatus.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists the upstream files to read, the contracts it depends on, the tasks, the validation slice and the fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope, risks, decisions or the archive branch?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, settled design decisions, archive-branch provenance, risks.
- [shared-contracts.md](shared-contracts.md) — the time-window semantics, error-message contract, input conventions and API snapshot rule shared by several phases.
- Phases (each ships as a separable PR, in order):
  - [phase-1-port-fixes.md](phase-1-port-fixes.md) — port the verified scientific fixes from the archive branch.
  - [phase-2-main-bugs.md](phase-2-main-bugs.md) — fix the correctness bugs verified on `main`: W-maze linearization, mirrored polar plots, NWB unit conversion, simulation field width and calculus gradient scaling.
  - [phase-3a-time-windows-rates.md](phase-3a-time-windows-rates.md) — the interval helpers, the extended `interval_valid_mask`, and gap-aware, stream-overlap handling for every rate family.
  - [phase-3b-time-windows-decoding-events.md](phase-3b-time-windows-decoding-events.md) — per-run decode time bins and event filtering by `epochs` and `spike_window` (requires 3a).
  - [phase-3c-time-windows-behavior.md](phase-3c-time-windows-behavior.md) — behavior and segmentation never span an invalid interval (requires 3a).
  - [phase-4-errors-docs.md](phase-4-errors-docs.md) — errors that teach, complete docstrings for the flagship functions, and examples executed in CI.
  - [phase-5-classification-ovc.md](phase-5-classification-ovc.md) — document the threshold classifiers' bias, add opt-in shuffle significance, and add an allocentric object-vector mode.
  - [phase-6-api-surface.md](phase-6-api-surface.md) — curated namespaces, one API snapshot test, consistent argument conventions, and a simulation-to-analysis path.
  - [phase-7-output-polish.md](phase-7-output-polish.md) — population summaries, `summary_table` parity between single and batch results, and `overwrite=False` for animation export.
