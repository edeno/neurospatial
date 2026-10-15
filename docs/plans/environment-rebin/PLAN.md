# Stage D — Re-binning and physical-wall consumers

**Status:** D0 baseline captured on 2026-10-14; consumer design in progress.
**Base:** `feature/environment-combine` at `1699ff1e` ([PR #71](https://github.com/edeno/neurospatial/pull/71)).
**First branch:** `feature/environment-rebin`.

The representation roadmap calls for separate changes per consumer: arena
re-binning, border scores, and visibility/ray casting. Implement and validate
these in order, with separate logical commits and stacked branches. Creating
Stage D PRs requires a later publication instruction.

## Scientific contract

- An analysis crop, absent position sample, or inactive bin does not create a
  physical wall. Use the retained arena rather than the active-bin outline.
- Explicit physical footprints establish their exterior/hole boundaries;
  explicit internal walls are additional barriers. Position-inferred domains
  and analysis masks do not establish measured walls.
- Re-binning returns new geometry and bin IDs. It does not automatically copy
  fields, counts, rates or fitted encoders. Center-based whole-bin selection
  remains distinct from exact clipping.
- Construction specifications are owned typed data, independent of descriptive
  provenance. Loading must not replay historical build parameters or infer
  missing measured geometry.
- Preserve existing factor-based coarsening, including truncation warnings and
  refusal when cells cannot represent an existing physical wall.
- Native behavior outside the intentional consumer changes must match the
  recorded Stage C baseline.

## Decisions pending

1. Require physical geometry for physical-wall consumers, or additionally expose
   an explicit analysis-boundary approximation mode? No implicit fallback is
   proposed.
2. Should new-resolution re-binning preserve the supported binning family and
   frame, or always build a world-aligned rectangular grid?

## Implementation sequence

### D0 — Baseline and design

Captured 4,072 environment probes, including 584 expected refusals, in
`/private/tmp/environment-stage-d-base-1699ff1e.npz`. Relevant existing tests:
**130 passed**, with 14 existing warnings, in
`/private/tmp/stage-d-base-tests.log`.

The deterministic [probe](probe_stage_d.py), run on `1699ff1e`, constructs a
10-by-10 physical arena with an internal wall at x=5 and a crop ending at x=4.
A ray starting at (2.5, 4.5) should hit the physical wall at (5, 4.5). Existing
code instead returns approximately x=10 on the full environment and x=4 on the
crop. A field at x=3.5 changes border score from -0.4359 to 1.0 solely because of
the crop. `rebin(bin_size=...)` is currently rejected as an unknown keyword.
Full probe output: `/private/tmp/stage-d-observations-1699ff1e.json`.

### D1 — Stored construction specification and arena re-binning

- Specify the supported initial families and new bin-size units/alignment.
- Own and serialize the construction specification independently of provenance.
- Preserve it through supported subset, clipping and transforms; a combine does
  not invent one shared specification.
- Construct candidate bins over the retained arena, select centers, and apply
  existing complete-face wall constraints before publication.
- Verify repeated re-binning, copies, dict/file/NWB round trips, transformed
  frames, holes, disconnected domains, partial cells, invalid sizes, and missing
  specifications. Never infer a specification from a descriptive layout name.

### D2 — Physical-wall border scores

- Derive wall evidence from asserted arena geometry, not `boundary_bins`.
- Define contact, distance, normalization and missing support explicitly while
  keeping the estimator's scientific interpretation documented.
- Test internal walls, holes, analysis crops, resolution changes, unavailable
  footprints, disconnected graphs and population/result wrappers.
- Keep the general `boundary_bins` query as an analysis-geometry query.

### D3 — Physical-wall rays and visibility

- Use geometric intersections with retained walls, rather than stepping until
  `bin_at` first returns -1. Preserve first-hit occlusion by physical barriers.
- Share wall semantics across viewed location, viewshed and cue line of sight.
- Specify rays starting outside/on walls, no hit before maximum distance,
  tangent/corner contacts and nonfinite observations.
- Preserve analysis-bin support when exporting visible-bin IDs; absence of an
  analysis bin must not occlude a ray to a known wall or cue.

## Validation

Use the existing isolated environment, without installing into the base:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache \
UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache \
uv run --no-sync pytest -p no:napari -n 0 -q ...
```

Add independent geometry oracles on deterministic simulated inputs. Recheck
confirmed defects before fixes and preserve permanent regressions. Run relevant
source doctests, typing, API/convention checks, Ruff and the behavior comparison
at consumer boundaries. Real recordings are downloaded separately; no recording
is currently cached under `data/`. GUI/GPU coverage must not be inferred from CPU
checks.
