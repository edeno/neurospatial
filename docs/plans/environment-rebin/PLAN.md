# Stage D — Re-binning and physical-wall consumers

**Status:** D0/D1 complete and validated on 2026-10-15.
D2 is complete and validated on `feature/environment-border-walls`,
stacked on rebin commit `bf5121e5`. D3 is implemented and under final validation on
`feature/environment-wall-visibility`, stacked on border commit `8a8693d5`.
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

## Approved decisions

1. Physical-wall operations also offer an explicitly requested analysis-boundary
   approximation. There is no silent fallback to analysis boundaries.
2. New-size re-binning preserves supported binning families and grid frames;
   unsupported families or missing specifications refuse. The first supported
   family is rectangular grids from samples, masks, images and polygon grids.
   The grid origin, affine frame and diagonal-navigation choice are typed owned
   data; the known arena supplies the domain. New sizes are physical lengths
   along the retained axes, including after unequal scaling.

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

## D1 validation

New-size regressions failed before implementation (`stage-d1-before.log`). The
accepted gate passed **779 tests**, including the 22 new rebin/specification
cases, existing factor/arena tests, polar classification, binning/codecs,
JSON and NWB round trips, API and argument conventions. Runtime warnings were
errors; the 70 warnings were existing user/dependency warnings. Source/contract
mypy passed (242 files). The Cartesian doctests and runnable environment guide
passed **10 cases**, with 4 explicit skips and 20 unrelated cases deselected.

The Stage C baseline comparison reports only the intended `Environment.rebin`
signature/docstring change; all other probes match. `GridBinningSpec` is the
intentional public export snapshot addition. No performance improvement is
claimed. No real recording was cached, and GUI/GPU checks were not run.

Local evidence: `/private/tmp/stage-d1-accepted.log`,
`/private/tmp/stage-d1-docs.log`, `/private/tmp/stage-d1-types.log`,
`/private/tmp/stage-d1-compare.log`. The new-size family scope is rectangular;
other families refuse until a reviewed specification is implemented. Native
integer coarsening remains unchanged.

## D2 estimator contract

Physical mode uses exact planar declared footprint rings and explicit internal
walls. Position/mask inference never asserts a wall. It measures the length of
wall covered by the union of thresholded field cells, divided by total wall
length. Mean field-center distance is normalized by the diameter of the declared walls,
so an analysis crop cannot become a new wall or change the distance scale.
Geodesic mode includes physical center-to-wall seed lengths, then paths on the
retained graph. Missing represented routes return NaN; they are not discarded.
This is the documented generalized screen, not the exact original Solstad
estimator or a calibrated null test. Analysis mode preserves the existing
boundary-bin formula and normalization exactly.

Direct metric and label calls require their chosen geometry. General summary
tables retain other metrics and record unavailable physical-border geometry in
attributes alongside NaN scores; they do not fall back to analysis mode. The
initial physical-wall implementation requires polygonal planar walls. Curved
measured tracks refuse with an actionable explanation.

Rotation regression exposed loss of contact from separately rounded transformed
polygons, plus frame-dependent normalization. Contact now uses the retained
source frame and normalization uses the wall diameter, invariant to rigid frame
changes. Independently transformed combined children lack one certified contact
frame and explicitly refuse physical scores; the uniform-field witness lost
wall coverage (0.9654349 instead of 1) in a world-frame intersection. Analysis
mode remains an explicit available approximation, not a hidden repair.

## D2 validation

The initial physical-wall tests failed before implementation. The accepted
border/result/API gate passed **347 tests** (4 unrelated slow cases deselected),
with runtime warnings treated as errors. The source/contract mypy gate passed
(244 files). Border, shared-metric and spatial-result doctests passed **48 cases**
with 2 explicit skips. The runnable environment guide passed. Legacy scientific
controls now request analysis mode explicitly, including population and result
wrappers, rather than asserting physical walls from samples.

Permanent regressions include hand-calculated coverage/distance, crop invariance
of Euclidean scores, internal walls, mask holes, unavailable paths, summaries,
copy/storage, frame changes and refusal of uncertified independent frames.
The API snapshot changes only the intended `boundary_source` argument.
Logs: `/private/tmp/stage-d2-release.log`,
`/private/tmp/stage-d2-release-types.log`, `/private/tmp/stage-d2-doctests.log`,
`/private/tmp/stage-d2-guide.log`.

## D3 delivered behavior and limits

Wall hits and cue occlusion use exact planar geometric intersections. Physical
mode reads asserted footprint rings and internal walls; explicit analysis mode
uses the effective analysis-cell domain and known internal walls. A maximum
search endpoint is not a wall. No hit or invalid/outside observer returns NaN;
starting on a wall returns the origin. A target at first contact is visible,
while an earlier contact blocks it. Stored source frames and locally capped
inverse-rounding bounds preserve boundary contact under affine frames.

Viewshed coverage still samples rays and bins, strictly before the first contact;
it does not claim exact visible-cell coverage. Known-wall intersections prevent
skipping a thin barrier. Missing analysis bins never terminate physical rays.

The view encoders and model forward the boundary choice; encoder provenance
records it. Endpoint binning uses a one-sided limit in the retained cell-owner
frame. Counts and occupancy share the resulting frame bins. Independently
transformed combined children without one owner frame refuse physical wall-bin
assignment; analysis mode remains explicit. Fixed-distance gaze is unchanged.
Encoder/model searches cover the supplied geometry instead of a fixed range.

Regression review caught two transformed-query defects after the first passing
gate: a cue on a rotated wall became occluded, and an angular sweep encoded two
bins beyond a closed wall. Source-frame contacts and owner lookup now resolve
both; the suite covers offsets 0, 100 and 1e9, a clearly behind-wall cue,
transformed subsets/clips and dict restoration.
