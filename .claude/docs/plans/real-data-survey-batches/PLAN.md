# Real-data survey batches Implementation Plan

**Status:** Phases 1–5 merged (#268, #269, #271, #273, #281) and the environment-fidelity study (#274); phase 6 not started (its first task is replaced by the dataset-loaders plan).

Extend the `datasets/` survey harness from eight recordings to most of the
open-dataset catalog, in four batches. The first batch is the one the
maintainer's pending decisions wait on: grid-cell populations for the grid-score
choice, linear and circular tracks for the origin-share and heading-coverage
warnings, and a shuffle-null study of the spatial-information bias. The other
three batches add breadth: more DANDI NWB files, non-NWB archives, and CRCNS
sets. Each batch adds loaders, runs the fixed operations and hazard probes, logs
every gap, and updates `datasets/FINDINGS.md`. No batch changes
`src/neurospatial/`; the agreed library changes are planned separately.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, designs it depends on, tasks, validation slice, and fixtures.
2. **Need a per-component design?** [designs.md](designs.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
4. **Need dataset access details (asset ids, URLs, sizes, known layout)?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals, non-goals, harness integration points, dependency policy, risks, the decisions this plan feeds
- [designs.md](designs.md) — streamed NWB opener, safe pickle inspection, MATLAB v7 reading, the information-bias null study, the gated-decision evidence
- Phases (each ships as a separable PR):
  - [phase-1-decision-datasets.md](phase-1-decision-datasets.md) — Gardner grid modules, Vollan linear track and open field, 000044 circular and linear tracks (streamed), 000447 W-track (streamed), the information-bias null study, and the evidence for the three gated decisions
  - [phase-2-dandi-nwb-breadth.md](phase-2-dandi-nwb-breadth.md) — nine small DANDI / EBRAINS NWB recordings (head direction in arenas, pixels, `conversion`, pose, objects, linear tracks)
  - [phase-3-archive-formats.md](phase-3-archive-formats.md) — non-NWB archives: egocentric boundary cells, 3-D lattice, water maze, Axona, MATLAB, NPY, track-graph `.npz`
  - [phase-4-crcns.md](phase-4-crcns.md) — CRCNS hippocampal sets behind an account, including a same-session cross-check against DANDI 000044
  - [phase-5-faithful-environments.md](phase-5-faithful-environments.md) — every loader declares its apparatus (outline, track graph, bars, room) so environments are built from stated geometry, compared with the samples-built ones
  - [phase-6-correctness.md](phase-6-correctness.md) — cut to one simulation check per domain (maintainer, 2026-11-03): planted cells on the real path, simulated behavior in each declared environment, planted environment misfits, and the visualization checks; a full rerun first
- [appendix.md](appendix.md) — per-dataset access details verified at planning time, and what is still unverified
