# Stage E — Preserve geometry through affine transforms

**Status:** E0–E2 complete on 2026-10-15. E3–E5 in progress.
**Branch:** `feature/environment-affine`, stacked on the repaired Stage D
`feature/environment-wall-visibility` at `98bf9790` ([PR #74](https://github.com/edeno/neurospatial/pull/74)).
This is a local implementation branch; opening another PR is a separate action.

## Entry status

Stage D's independent review is complete: shared component walls and slanted
wall-hit ownership were repaired, and the reviewer rechecked both fixes and
missing-support safeguards. The repaired stack passed **13,556 CPU tests**
(31 skipped), a **543-test** focused gate including slow plot snapshots, and
source/contract typing (**245 files**). The final CPU log is
`/private/tmp/stage-d-review-full-final.log`.

The base PR's CI failures were reproduced from the Linux/Windows/slow-job logs.
Portable geometry/plot assertions and explicit UTF-8 test reads are published;
all stacked PRs now receive checks, with obsolete runs cancelled per PR.
**Remote matrix verification is running/queued, not established as passing.**
Stage E has started with that matrix pending. Local phase gates proceed; the
remote matrix remains a publication gate. Before the documentation-example
correction, the base slow-test job and multiple Ubuntu/Windows matrix jobs
passed; the current heads are rerunning. A further arena test-oracle
repair evaluates the same filtered lookup batch; its final 150 publication/
validation checks pass without changing library behavior. Full source doctests
passed 596 cases (126 skipped), and all 4,072 environment probes match the prior
Stage D snapshot. Planning does not claim that a queued CI job has passed.

## Problem and user workflow

A scientist may rotate a recording into room coordinates, reflect an image's
vertical axis, or convert pixel dimensions into physical units. These operations
should move existing cells, preserve their IDs, and retain their exact shapes.
A rectangular grid also needs its original axes: rotating it should not remove
linear interpolation or turn a cell plot into a scatter plot.

Changing geometry and retaining numerical operations are separate questions.
A shear turns rectangles into parallelograms. Those cells can still be located,
drawn and measured exactly, but the current diffusion method assumes that the
line between neighboring cell centers is perpendicular to their shared face.
Stage E must check that assumption rather than silently apply the old formula.

The intended workflow keeps `env.apply_transform(Affine2D(...))` and `AffineND`.
The returned environment has the same bin and edge ordering, transformed cells,
arena, walls and regions. Queries accept world coordinates. Grid operations
internally use the saved grid coordinate system. Unsupported numerical
operations raise an error explaining which geometric requirement failed.

This stage changes geometry capabilities, not firing rates or fitted models.
Users still decide whether existing bin-aligned values retain their scientific
meaning after changing units or deforming an arena.

## Evidence from the current implementation

The deterministic planning probe uses a 3-by-4 grid, query `(1.25, 1.75)` and
field `2*x + 3*y`; native linear interpolation returns `7.75`.

| Transform | Bin IDs at transformed centers | Linear interpolation | Field renderer | Faces available |
| --- | --- | --- | --- | --- |
| 90-degree rotation | Preserved | Refused | `PathCollection` | Yes |
| x reflection | Preserved | Refused | `PathCollection` | Yes |
| x shear by 0.3*y | Preserved | Refused | `PathCollection` | No |

Observed at `21b2c5f4`; local output is
`/private/tmp/stage-e-plan-probe.log`. These are confirmed capability gaps,
not evidence that current rotation/reflection lookup or diffusion is broken.

Relevant implementation:

- [`_derive.py`](../../../src/neurospatial/layout/_derive.py):
  `_transform_binning` drops the rectilinear component except for positive
  diagonal transforms; face preservation uses a transform-class decision.
- [`_capabilities.py`](../../../src/neurospatial/layout/_capabilities.py):
  `RectilinearGrid` contains edges and a mask but no world-coordinate frame.
- [`_diffusion.py`](../../../src/neurospatial/layout/_diffusion.py):
  `FaceGeometry` contains areas, flux distances and volumes, but no normals.
- [`visualization.py`](../../../src/neurospatial/environment/visualization.py)
  and [`cartesian.py`](../../../src/neurospatial/environment/cartesian.py):
  cell rendering and linear interpolation require the corresponding components.
- [Current derivation contract](../environment-representation/shared-contracts.md#derivation-rules):
  the existing preservation rules remain the compatibility floor.

## Scope and invariants

1. Retain every bin ID, authoritative `edge_pairs` order and orientation, active
   mask, immediate-parent mapping and component identity. Do not reverse edges
   or flip an array to make reflected coordinates increase.
2. Own all new arrays; preserve read-only behavior through caller mutation,
   deepcopy, pickle, dict/file and NWB restoration. Loading never runs builders
   or replays descriptive provenance.
3. Preserve current translations, similarities and supported axis-aligned
   scaling, including their numerical outputs within recorded tolerances.
4. Keep exact supported cell geometry available independently of face validity.
   A missing diffusion capability must not remove lookup, plotting or occupancy.
5. Preserve the distinction between navigation distances and flux distances.
   Scalar path lengths and bridge conductance do not specify paths or faces.
6. Keep physical walls independent of selected bins. Carry the Stage D shared
   room-wall and slanted wall-hit regressions through all supported transforms.
7. Preserve explicit physical units and coordinate-frame metadata. An affine
   matrix alone does not supply a new unit label; keep the current metadata API.

Initial implementation covers full-dimensional Cartesian rectangular grids
in 2-D/3-D, plus existing exact planar polygons/triangles/hexagonal cells.
Existing higher-dimensional and embedded-track guarantees remain available;
new general affine track-length rules, curved walls, 3-D clipping, polar affine
coordinates and a general non-orthogonal diffusion solver are outside scope.
Stage F's removal of the layout facade remains separate.

## Data contracts

E0 settled the public names `FaceNormals` / `Binning.face_normals` and
`FramedGrid` / `Binning.grid_frame`. `FaceGeometry` keeps its four-field tuple
contract. E1 binning records use codec version 2; readers accept legacy binning
version 1 with absent optional components. E2 must increment the binning version
again when adding framed grids, retaining support for earlier records. Existing
other codecs stay at version 1, as do the new component codecs. Environment-v3 and NWB 3.0 envelopes remain
unchanged: the nested version makes older readers reject the new binning record.
No promise is made that an older reader can consume version 2.

Independent E0 review clarified that normal alignment means positive edge
orientation, not perpendicularity. Existing approximate mesh faces remain
available under their current rules. Full-cell subsets retain their framed grid;
partial or split polygon clipping drops it rather than inventing tensor cells.
Inverse composition restores geometry and lookup within storage precision; it
does not restore numerical faces already discarded as `MissingFaces`. Retaining
uncertified source faces for future recertification would need a separate design.
Prefer two small optional components over changes to existing tuple iteration
or a new parallel environment hierarchy.

**A framed grid component** stores a source `RectilinearGrid` and an owned
homogeneous forward matrix from that grid's coordinates into world coordinates.
Its mask and row-major order remain unchanged after reflection. Flatten nested
frames by composition, subject to the existing conditioning/precision checks.
The source grid, locator and matrix must agree at publication and loading.

Keep `binning.rectilinear` truthful for consumers that still require a grid
aligned with world axes. Provide a shared internal grid-context accessor for
migrated consumers, returning native axes plus forward/inverse coordinate maps.
Do not make old consumers mistake source edges for world edges. Preserve the
current fast path for positive axis-aligned transforms where it is equivalent.

`GridBinningSpec` from Stage D describes *future construction*. It is not the
geometry of current cells after clipping/subsetting, so it cannot substitute
for the framed current-grid component.

**A face-normal component** stores one owned unit normal per positive-area
physical face, in authoritative edge order and directed from its first bin
toward its second. Zero-area navigation edges have no physical normal and must
not gain conductance. Validate dimensions, finiteness, normalization, alignment
and agreement with certified face geometry. Initially keep `FaceGeometry`'s
existing four fields and tuple behavior unchanged.

Capture normals from actual native shared-face geometry. Do not infer missing
normals from center links for arbitrary providers. Explicit-bin construction can
accept validated normal data; old records without it retain today's guaranteed
transform classes but gain no unsupported numerical capability.

Subset normal rows with the same edge map used for faces. Exact clipping must
recompute normals only when its current face certificate is valid. Combined
components retain their own normals; navigation bridges do not acquire invented
normals. Both components require versioned codecs and consistency checks. E1
verifies legacy version-1 records without normals and refuses a newer component hidden
in a downgraded record. Never promise that an older reader understands new
geometry records.

## Geometric and numerical rules

For an invertible Cartesian linear map `L`, translation `t`, cell volume `V`,
face area `a` and unit normal `n`:

- Transform points by `x_new = L*x + t`.
- Transform full-dimensional measures by `V_new = abs(det(L))*V`.
- Transform the physical area vector by
  `s_new = abs(det(L)) * inverse(L).T @ (a*n)`.
  Then `a_new = norm(s_new)` and `n_new = s_new/a_new`.
  The absolute determinant handles reflections while normals keep the edge's
  physical first-bin-to-second-bin orientation.

This area-vector rule is often called Nanson's formula. A deterministic planning
check (seed 417, 100 maps each in 2-D/3-D) matched transformed edge lengths and
cross-product areas within rtol=atol=1e-12, including reflections. Repeat those
independent oracles on the published implementation and its stored normals. Use scaled or
logarithmic arithmetic where needed; refuse unrepresentable values rather than
publishing zero/NaN measures.

For each newly enabled positive face, compare its transformed normal with the
transformed center-separation direction. Require their normalized tangential
residual to meet a documented numerical tolerance (initial target `1e-12`,
consistent with the current similarity classification). Derive the new flux
distance from certified physical geometry, not from the navigation weight.
Where scalar flux distance cannot be related to the stored geometry, refuse
new non-similarity numerical support instead of guessing.

A failed check produces `MissingFaces` with a reason naming the first affected
face and a useful remedy. Smoothing, diffusion, gradient and divergence must all
honor it. Keep exact cells, sizes, lookup, plotting and unsmoothed occupancy.
Current native approximate-mesh warnings and guaranteed similarity behavior
remain unchanged; this stage does not use that approximation as permission to
enable additional deformations.

For planar hexagons under transforms that no longer make regular hexagons,
retain the exact transformed vertices as polygon cells. Triangle vertices
already survive general affine transforms; preserve that behavior. All exact
shape conversions must retain bin identities and clipping/component mappings.

## Grid consumers

- **Lookup:** map world positions into retained source coordinates and use the
  original ownership rules. Preserve half-open boundaries, holes, nonfinite
  rows and coordinate-width errors. Never renumber reflected bins.
- **Linear interpolation:** map query positions into source axes, then evaluate
  the same tensor-product interpolation. Do not change Euclidean nearest-center
  behavior under a shear. Preserve missing/inactive-cell handling and NaNs.
- **Linear time allocation:** map movement-segment endpoints into source axes;
  affine maps preserve their time fractions. Preserve arena entry/exit gates,
  recording gaps, and matching spike/count and dwell-time allocation.
- **Plotting:** map grid corner coordinates into world coordinates and draw a
  `QuadMesh`. Draw polygon-converted cells exactly. Avoid an axis-aligned image
  extent for a rotated or sheared grid.
- **Other consumers:** audit every `rectilinear`/grid-context read in encoding,
  metrics, animation and storage. Each must map coordinates explicitly, retain
  a truthful existing restriction, or use the generic bin representation.
  Storage/export must never label source axes as world coordinates silently.
- **Rebinning:** retain Stage D's independent construction specification and
  physical axis-length convention; a current framed-grid component must not
  overwrite the retained arena or construction anchor.

## Implementation sequence and acceptance gates

Each phase captures its baseline first, ships as a logical commit and receives
an independent review before proceeding. Do not open implementation PRs until
publication is requested.

| Phase | Deliverable | Acceptance gate |
| --- | --- | --- |
| E0 | Freeze scope, component names, codec/version policy and baseline | Record current environment probes, numerical outputs and renderer types; add deterministic rotation/reflection/shear reproducers and agree on expected changes. |
| E1 | Owned face normals and serialization | Native 2-D/3-D geometry oracles; exact edge order, subset/clip mapping, explicit-bin validation, legacy loading and ownership/round-trip tests pass. No existing numerical behavior changes. |
| E2 | Framed current grids and exact transformed shapes | Compose/invert transforms without renumbering; preserve source/world consistency through loading, subsets and clipping. Lookup parity holds; hexagon polygons and triangle vertices match transformed originals. |
| E3 | Transform faces using geometry and certify numerical support | Hand-calculated areas/normals/flux distances; existing guaranteed cases unchanged; valid new cases enabled, skew cases refused consistently by every numerical consumer. |
| E4 | Migrate grid interpolation, time allocation and rendering | Rotated/reflected queries recover the same interpolated values, time fractions and `QuadMesh` cells. Missing support is not filled and physical walls are not lost. |
| E5 | Workflow documentation and final compatibility gate | Full relevant CPU/NWB tests, strict numerical cases, source doctests, typing, API snapshot, formatting and baseline comparison pass; validation limits and intentional changes are recorded. |

E0 must record runtime and memory before any performance-related refactoring.
Repeat the same measurements afterward; make no improvement claim without them.
If a component or numerical contract requires a major change beyond this plan,
bring the concrete tradeoff back to the user before implementing it.

## Test matrix and completion criteria

Use deterministic simulated geometry first. Include native, subsetted, clipped,
combined, transformed-then-subsetted, nested-transform and restored environments.
Test translations, rotations, reflections, uniform/unequal scaling, shear,
rotation followed by unequal scaling, inverse composition, near-singular maps,
large offsets and small well-resolved cells. Preserve distinct navigation and
flux weights and reject unsupported bridge/track transformations.

Scientific checks include exact affine-linear interpolation, correct cell
measures and oriented edge alignment, shared wall retention, observer-side gaze
bin ownership, count/exposure consistency, disconnected components, constant
fields, volume-weighted mass conservation and the appropriate bandwidth/time
scaling under similarities. Include zero-area edges and isolated cells.

Run grid/mesh geometry and wall-contact tests with `RuntimeWarning` as an error.
Use plotting tolerances and geometry membership checks that are independent of
platform-specific Qhull numbering; retain exact identity/ordering assertions
within each environment. Validate dict/file/NWB, pickle and deepcopy ownership.
Corrupted normal/frame records must be rejected before publication.

Then use the smallest available real tracking/image-mask slice, expanding only
when it adds coverage. If no recording is cached, report that limit explicitly.
Include a workflow that starts from position data or an image mask without
asserting physical walls. GUI/GPU checks remain separate from CPU validation.

Completion requires runnable user examples, a migration/capability table and a
validation record distinguishing unchanged behavior, intentional capability
additions, numerical refusals and checks that were not run. Until those gates
pass, Stage E remains planned or in progress.
