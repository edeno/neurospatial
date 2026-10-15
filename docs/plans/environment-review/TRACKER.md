# Environment stack review tracker

Review supplied on 2026-10-15. Starting Stage F commit: `1c4cd754`, PR #77.
This file tracks the supplied review, our reproductions, repairs and verification.
The review's severity and introduction claims are hypotheses until checked here.

## Working rules

- Reproduce a defect before repairing it, with a regression test where useful.
- Fix an existing stack regression on its owning PR and merge forward, preserving
  published branch history. Keep unrelated historical bugs in separate PRs.
- Keep numerical corrections separate from mechanical architecture changes.
- A geometry or analysis limitation must give an actionable refusal; it must not
  silently invent unobserved activity, physical walls or path distances.
- `Reported` means not yet independently checked; `Confirmed` means reproduced;
  `Fixed` requires a regression check and a recorded commit. `Design` identifies
  a recommendation rather than a demonstrated defect.

## Correctness and documentation

| ID | Review priority | Finding / acceptance criterion | Owning PR or proposed home | Status / evidence |
| --- | --- | --- | --- | --- |
| R01 | High | Place-field components preserve actual diagonal connections after subset, arena filtering, new-size rebin and storage. | #68 graph-based field labeling; #77 cleanup | Confirmed: parent field extraction reconstructs lattice connectivity from build parameters. Repair in progress. |
| R02 | High | Place fields cannot merge across graph connections removed by physical walls. | #68, carried through #73/#77 | Confirmed: same lattice shortcut ignores removed graph edges. Repair with R01. |
| R03 | High | Polygon clipping behaves consistently in meters/centimeters and after rotation; overlay artifacts do not create invalid cells. | #68 | Reproduction and repair in progress in isolated worktree. |
| R04 | High | Physical border scores and wall-hit maps distinguish unobserved wall coverage from measured zero activity, with stable geometry under rotation and useful setup guidance. | #73 border scoring / #74 wall ownership | Investigation in progress; scientific policy must be explicit. |
| R05 | High | Polar factory creates exactly the requested angular partition when the step divides the span, including 61 bins. | Separate fix against main | Reported. |
| R06 | Medium | Clipped and combined environments expose correct analysis boundary bins. | #68 / #71 | Reported: empty boundary arrays. |
| R07 | Medium | Every successfully saved supported track loads with equivalent lookup; coincident embedded centers have consistent construction/storage validation. | #67 or earliest owning change | Reported. |
| R08 | Medium | Clipping never silently replaces declared navigation path lengths with Euclidean lengths. | #68 / #70 explicit bins | Reported: 3.0 becomes 1.0. |
| R09 | Medium | Automatic bridges cannot traverse an intervening third component. | #71 | Reported. |
| R10 | Medium | Copies and pickle omit disposable analysis caches and start fresh caches on restore. | #70 / #77 | Reported: cached kernel inflates saved object from 2.3 MB to 56.8 MB. |
| R11 | Medium | Track/maze containment rejects points arbitrarily far from the track. | Separate fix against main | Reported. |
| R12 | Medium | Partial polar angle ranges do not wrap merely because circularity defaults to true. | Separate fix against main | Reported; review default/explicit circularity contract. |
| R13 | Medium | Saving an extensionless session name containing dots preserves the full name and cannot overwrite another session. | Separate fix against main | Reported: `rat1.day1` writes `rat1.json`. |
| R14 | Low | Long supported transform chains either save/load equivalently or refuse before writing an unreadable record. | #67 | Reported: 33 transforms fail on load. |
| R15 | Low | Mesh logical grid shape remains stable across supported storage. | #67 / #77 | Reported. |
| R16 | Low | User graph attributes named `id`/`source` survive storage or receive an explicit supported restriction. | #67 | Reported. |
| R17 | Low | Clip mappings have one storage authority; legacy duplicate records remain readable if required. | #68 | Reported. |
| R18 | Low | Malformed nested or archive input produces actionable public errors rather than raw `RecursionError`/`BadZipFile`. | #67 | Reported. |
| R19 | Low | Explicit grid faces are validated against authoritative geometry. | #70 | Reported. |
| R20 | Low | Explicit pairs crossing physical walls cannot disappear without an explicit construction decision. | #70 | Reported. |
| R21 | Low | A drawing Outline cannot silently assert a physical arena footprint. | #70 | Reported. |
| R22 | Low | A trajectory point exactly on a wall cannot bypass wall-crossing rules. | #68 | Reported. |
| R23 | Low | Visibility of a cue on a slanted wall is stable under bounded floating-point rounding. | #74 | Reported. |
| R24 | Low | Loaded Cartesian/polar environments have truthful static coordinate types. | #77 | Reported. |
| R25 | Low | Shared operations accept polar values statically wherever they accept them at runtime. | #77 | Reported: unparameterized annotations default to Cartesian. |
| R26 | Low | Read-only connectivity remains exportable through standard NetworkX node-link JSON. | #70 | Reported. |
| R27 | Docs/CI | Replace removed serialization/smoothing API anchors; strict docs build passes. | #77 | Fixed: strict build passes; local check of 8,033 built fragments goes from two invalid links to zero. CI failure was the anchor-check step. |
| R28 | Docs | Initialization examples match E1006 raised by direct `Environment()` construction. | #77 | Fixed: factory-only construction documented; new normal/raises examples execute. |
| R29 | Docs | Contributor instructions describe the capability checks actually performed. | #77 | Fixed: guide now describes owned face inputs and actual grid/track extraction. |
| R30 | Docs | Test descriptions explain current behavior rather than obsolete Phase 2/3/3.3 milestones. | #77 | Fixed: batch encoding, spike inputs and NWB test descriptions name their behavior; experimental trajectory phases are retained. |

## Architecture recommendations

These require behavior baselines and dependency checks, and belong in a separate
stacked PR after scientific fixes. Optional API redesigns are not prerequisites
for repairing numerical defects.

| ID | Recommendation / acceptance criterion | Status |
| --- | --- | --- |
| A01 | Separate owned geometry, validation and codecs from native builders; preserve public import paths and enforce package-level dependency rules. | User approved implementation after correctness repairs, in a new stacked PR. |
| A02 | Expose read-only `env.binning`; numerical functions consume data while Environment owns analysis caches. Remove external private-state reads/writes and casts where feasible. | User approved implementation after correctness repairs, in a new stacked PR. |
| A03 | Behavior follows owned geometry/connectivity rather than descriptive type names or construction parameters. Includes segmentation, boundaries, circularity and rebin capability. | R01/R02/R06/R12 cover defects; audit remaining cases. |
| A04 | Remove parallel/dead paths: duplicate rebin implementations, unused provider protocols/decorators/state versions/validators, inconsistent Cartesian guards and duplicate publication validation. | Reported; verify each path is actually dead before removal. |
| A05 | Consolidate wrapper traversal and numerical tolerances where their contracts match; include clipped children in combined diffusion traversal. | Reported drift; avoid imposing one tolerance on different physical quantities. |
| A06a | Put subset/rebin in appropriately named derivation modules. | Design. |
| A06b | Move shared time-support helpers from environment trajectory into `_intervals.py`. | Design. |
| A06c | Remove the `ops.transforms` / `environment._affine` dependency cycle and import-linter exception. | Design. |
| A06d | Consolidate wall evidence and place wall-hit ownership with geometry consumers. | Design; coordinate with R04/R23. |
| A06e | Extract publication and summary formatting from public core without changing ownership or public signatures. | Design. |
| A07 | Reconsider read-only graph implementation based on measured costs and JSON compatibility. | Optional design; R26 is the concrete compatibility defect. |
| A08 | Consider an `env.plot.*` accessor. | Optional API redesign; tracked, not selected. |
| A09 | Audit function-local imports; change only those needed for dependency clarity or measured import cost. | Optional cleanup; not a correctness defect. |

## Findings the supplied review says are already fixed

Hex/mesh subset measures; capabilities retained by derivations; NWB boundary and
track linearization; loading without builder replay; graph attribute ownership;
metadata setters; density-mode smoothing documentation. No new repair is implied.
Their passing regressions remain part of the broader gate.

## Verification record

- Initial branch/stack check: PRs #67–#76 have no failed current checks. PR #77
  has a failed documentation build and no other reported failed check.
- Isolated scratch branches are based on the owning PRs for field connectivity
  (#68), polygon overlays (#68), and physical-wall coverage (#73). Tests import
  each worktree explicitly through `PYTHONPATH` using the existing isolated `.venv`.
- Commit hashes, before/after test results, propagated PR heads and CI outcomes
  will be recorded here as repairs land. No overall completion is claimed yet.

## Approved scientific decisions

- Physical border scores require represented, observed wall coverage; unavailable
  coverage produces an actionable refusal rather than invented zero firing.
  A view-map ray with a physical hit outside retained bins also refuses, while
  raw wall intersections and unobserved gaze directions retain their own meaning.
- Polar circularity will default to automatic: full 2π wraps, partial ranges
  remain open, and explicitly wrapping a partial range is rejected.
