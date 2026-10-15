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
| R01 | High | Place-field components preserve actual diagonal connections after subset, arena filtering, new-size rebin and storage. | #68 graph-based field labeling; #77 cleanup | Fixed: graph-based labeling `0df50a01`; 18 regressions, 389 focused tests. Parent integration passes. |
| R02 | High | Place fields cannot merge across graph connections removed by physical walls. | #68, carried through #73/#77 | Fixed with R01: components follow retained graph edges rather than reconstructed lattice adjacency. |
| R03 | High | Polygon clipping behaves consistently in meters/centimeters and after rotation; overlay artifacts do not create invalid cells. | #68 | Fixed: `27241165` keeps per-parent overlay ownership and local precision; 883 tests and 4,051 unchanged probes. |
| R04 | High | Physical border scores and wall-hit maps distinguish unobserved wall coverage from measured zero activity, with stable geometry under rotation and useful setup guidance. | #73 border scoring / #74 wall ownership | Fixed in source commits `ed3517fd`, `cf5b3092`, `a6531c59`: border coverage and selected positive-duration view exposure checked separately; independent review approved. |
| R05 | High | Polar factory creates exactly the requested angular partition when the step divides the span, including 61 bins. | Separate fix against main | Fixed in PR #78: `f45e9619`; deterministic bin-count regressions pass. |
| R06 | Medium | Clipped and combined environments expose correct analysis boundary bins. | #68 / #71 | Fixed: polygon boundary extraction `14f6460b`; combined regressions `44574f63` verify the inherited fix. |
| R07 | Medium | Every successfully saved supported track loads with equivalent lookup; coincident embedded centers have consistent construction/storage validation. | #67 or earliest owning change | Fixed on #67: `fbde0b88`; coincident supported track centers round-trip; independent review approved. |
| R08 | Medium | Clipping never silently replaces declared navigation path lengths with Euclidean lengths. | #68 / #70 explicit bins | Fixed: `6ba4ed4c` preserves unchanged declared path endpoints and refuses unsupported endpoint changes; 245 tests. |
| R09 | Medium | Automatic bridges cannot traverse an intervening third component. | #71 | Fixed in `160dea5d` plus independently reviewed `dedc3efc`; tiny intervening balls and exact tangencies covered; 768 broader tests. |
| R10 | Medium | Copies and pickle omit disposable analysis caches and start fresh caches on restore. | #70 / #77 | Fixed: `bbb086d5`; caches excluded before copy/pickle rather than discarded afterward; four failures reproduced, 154 tests pass. |
| R11 | Medium | Track/maze containment rejects points arbitrarily far from the track. | #68 | Fixed on #68: `1219b6ff`; declared width required for containment, nearest projection retained; 35-case persistence/derivation matrix. |
| R12 | Medium | Partial polar angle ranges do not wrap merely because circularity defaults to true. | Separate fix against main | Fixed in PR #78: `12b3d737`; approved automatic full-circle/partial-range policy. |
| R13 | Medium | Saving an extensionless session name containing dots preserves the full name and cannot overwrite another session. | Separate fix against main | Fixed in PR #79: `4fcf4726`; independent dotted session prefixes retained. |
| R14 | Low | Long supported transform chains either save/load equivalently or refuse before writing an unreadable record. | #71 | Fixed: `9a70a12b`; affine wrappers do not consume composite branching depth; 33 transforms round-trip; 79 tests. |
| R15 | Low | Mesh logical grid shape remains stable across supported storage. | #67 / #77 | Fixed on #67: `e2b5ee73`; native simplex grid shape and mask retained, with inconsistent duplicate context rejected. |
| R16 | Low | User graph attributes named `id`/`source` survive storage or receive an explicit supported restriction. | #67 | Fixed on #67: `3d8b642c`; optional reserved-attribute sidecars preserve user data and native node-link topology; independently reviewed. |
| R17 | Low | Clip mappings have one storage authority; legacy duplicate records remain readable if required. | #70 | Fixed: `6e3c8da2`; equal immutable parent mappings share one record; legacy duplicate validation retained; 161 tests. |
| R18 | Low | Malformed nested or archive input produces actionable public errors rather than raw `RecursionError`/`BadZipFile`. | #67 | Fixed on #67: `b892ddf2` and `45f7c403`; contextual malformed-input errors include UTF-8 and deflate failures; 442 IO tests, independent review approved. |
| R19 | Low | Explicit grid faces are validated against authoritative geometry. | #70 | Fixed: `2cb663de` and `9c769517`; authoritative grid geometry checked with stable norms, including 1e±200 scales; 475 relevant tests. |
| R20 | Low | Explicit pairs crossing physical walls cannot disappear without an explicit construction decision. | #70 | Fixed: `22e88747`; explicit neighbor pairs crossing walls raise with guidance instead of silently disappearing. |
| R21 | Low | A drawing Outline cannot silently assert a physical arena footprint. | #70 | Fixed with R20: an Outline supplies drawing/domain geometry without asserting a physical footprint. |
| R22 | Low | A trajectory point exactly on a wall cannot bypass wall-crossing rules. | #68 | Fixed on #68: `ea438c34` plus `c0522e98`; possible approach sides carried through sampled and unsampled wall contacts; independent review approved. |
| R23 | Low | Visibility of a cue on a slanted wall is stable under bounded floating-point rounding. | #74 | Fixed on #74 through `4735deec`: local cue rounding and cancellation handled without accepting earlier crossings or wall overlap; 105 strict tests and 3,000 deterministic cue probes pass; independent review approved. |
| R24 | Low | Loaded Cartesian/polar environments have truthful static coordinate types. | #77 | Fixed on #77: `d689394e`; loaded coordinate kind typed as unknown (`Any`), Cartesian factories remain specific. |
| R25 | Low | Shared operations accept polar values statically wherever they accept them at runtime. | #77 | Fixed with R24: shared annotations accept both coordinate systems; five original static errors reproduced, 257-file typing gate passes; independent review approved. |
| R26 | Low | Read-only connectivity remains exportable through standard NetworkX node-link JSON. | #70 | Fixed: `a33da946` and `f6fe3254`; detached read-only JSON metadata remains copyable/pickleable and cannot mutate owned graph state; independent review approved. |
| R27 | Docs/CI | Replace removed serialization/smoothing API anchors; strict docs build passes. | #77 | Fixed: strict build passes; local check of 8,033 built fragments goes from two invalid links to zero. CI failure was the anchor-check step. |
| R28 | Docs | Initialization examples match E1006 raised by direct `Environment()` construction. | #77 | Fixed: factory-only construction documented; new normal/raises examples execute. |
| R29 | Docs | Contributor instructions describe the capability checks actually performed. | #77 | Fixed: guide now describes owned face inputs and actual grid/track extraction. |
| R30 | Docs | Test descriptions explain current behavior rather than obsolete Phase 2/3/3.3 milestones. | #77 | Fixed: batch encoding, spike inputs and NWB test descriptions name their behavior; experimental trajectory phases are retained. |
| R31 | Additional | Native grid diffusion faces remain valid in small coordinate units. | PR #80 against main | Fixed: `cacb3a73`; remove the absolute axis-displacement cutoff; physical published-face tests pass. |
| R32 | Additional | Factor rebinning preserves independent named regions. | #67 | Fixed: `74a79a7b`; 18 failing ownership regressions reproduced, 166 tests pass. |
| R33 | CI baseline | Generated UTF-8 animation HTML is read correctly on Windows. | PR #81 against main; merged into #78–#80 | Fixed: `b147b034`; two cp1252 failures reproduced; all 16 existing module tests pass normally and with cp1252 simulated. |

## Additional verification findings

- R34 (#76): identical arena vertices could round differently in matrix batches,
  breaking polygon ring closure after chained affine maps. Fixed in `47f0b077`
  by mapping each identical source vertex once, without changing tolerances;
  three reproduced failures and two new regressions pass, with 235 owner-module
  checks and independent tiny-geometry probes.
- R35 (#67): Windows/Python 3.10 can exhaust the C stack before cyclic metadata
  raises `RecursionError`. An iterative preflight now rejects container cycles
  before decoding while allowing shared acyclic records and long valid affine
  histories. Verification: 478 IO tests and 15 Python 3.10 ingress tests pass;
  independent review approved; published as `9436ae14` and merged forward without rewriting history.

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
| A05 | Consolidate traversal/tolerances only where contracts match. | Clipped traversal is intentionally a certificate boundary, not a defect: `45a19995` documents and tests 66 certified/uncertified cases. Unwrapping to parent diffusion geometry would be incorrect. Other consolidation remains scoped to matching contracts. |
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
- Track containment requires a declared physical width. Unknown width refuses
  `contains()` with guidance; `bin_at()` continues nearest-track projection.

## Integration progress

- Published owner heads: #67 `9436ae14`, #68 `6b8e530e`, #70 `31bc3ff2`,
  #71 `7dffa014`, #72 `43d671ab`, #73 `58a88588`, #74 `ec1aa183`,
  #76 `a2450fde`. Repairs are merged forward without rewriting published history.
- Baseline fixes have separate PRs: polar count/circularity #78; dotted file
  prefixes #79; small-unit native grid faces #80; Windows HTML decoding #81.
- Stage F CPU verification at `8c35617e`: 13,846 passed, 335 skipped using
  the locked dev/NWB/xarray environment plus CI's widget dependencies; slow,
  napari and wall-clock checks were excluded. GUI/GPU execution remains unrun.
  The subsequent cycle-preflight change passed the malformed-ingress gate and
  passed a fresh complete IO gate: 590 passed, seven skipped. Source and static contracts pass for 257 files; Ruff, formatting and whitespace checks pass.
- Parent CI is rerunning on the new heads. Earlier green runs and cancelled
  superseded jobs are not treated as proof for these updated commits.
- Structural work is isolated in `refactor/environment-responsibilities`,
  stacked above #77. It retains separate before/after behavior captures and
  independent reviews; optional plot namespacing and graph redesign remain
  tracked rather than silently added.

## Pre-merge review (2026-10-15)

An independent per-PR review of #67–#83 was run before landing. Findings
reproduced on the stack top and repaired on their owning PRs, then merged
forward:

| ID | Owning PR | Finding | Status / evidence |
| --- | --- | --- | --- |
| R37 | #74 | An observer tracked exactly on a wall viewed its own position (regression vs main): gaze hit, viewshed, cues and view-rate bins. | Fixed `92f01345`: a zero-distance start contact is ignored when the ray departs into the domain; corner/internal-wall/rotated cases tested. |
| R38 | #74 | Ray-cast view rates refused exactly clipped cells on curved/slanted walls (floating-point overlay fragments). | Fixed `49ec39d0`: ownership by per-cell coordinate budget at the hit; centre-inclusion cells still refuse (approved policy); refusal names `grid.clip(polygon).binning`. |
| R39 | #73 | `summary_table()` labeled a cell `"place"` when the physical border score was unavailable. | Fixed `322e9ca8`: `"undetermined"` in single-unit and population tables. |
| R40 | #73 | A combined component without declared walls scored its wall fields against other components' walls (−1.0). | Fixed `9373e7ae`: physical scoring refuses and names the component. |
| R41 | #70 | `from_bins` accepted hexagon/triangle `KnownFaces` inconsistent with cell geometry. | Fixed `1c0e37e3`: shared-edge length and centre distance validated. |
| R42 | #77 | Cartesian-only domain functions accepted polar environments (mypy and runtime), e.g. `mean_position` at the ±π seam. | Fixed `af14a148`, `586cee4a`: `Environment[CartesianSystem]` typing plus runtime `CartesianOnlyError`; guard kept below `ops` in the layer contracts. |
| R43 | #67 | Saved environments from 0.6.0 cannot be loaded; migration guide was silent. | Intended (maintainer); documented `337016e6`. |

Open follow-ups (not blocking): mesh environments saved on one OS refuse to
load on another (Delaunay rebuild at load, #67); factor rebin of masked/polygon
grids drops valid coarse cells and `CHANGELOG` claims unchanged behavior (#83);
circular `rebin(factor=2)` drops perimeter bins (#68); combine overlap
certification is O(n²) (#71); no warning when `rebin(bin_size=)` leaves arena
area unbinned and coarse rebin can connect across thin outline barriers (#72);
a single arena with internal walls but unknown footprint can still mis-score
outer-edge border fields (#73); polar `DecodingResult.to_dataframe()` and
`decode_position_summary` now refuse because they include `mean_position`.
