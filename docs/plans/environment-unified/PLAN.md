# Stage F — One public Environment and direct geometry consumers

**Status:** Planning complete; implementation has not started.
**Branch:** `feature/environment-unified`, stacked on `feature/environment-affine`
at `936b29ae` ([Stage E PR #76](https://github.com/edeno/neurospatial/pull/76)).
**User decision, 2026-10-15:** Merge the public Cartesian and polar classes during
Stage F. Place that merge where the dependency audit supports it; retaining two
public classes is not the final design. No Stage F PR is requested by this plan.

## Problem and intended workflow

Scientists currently use `Environment` for room/track coordinates and obtain a
separate `EgocentricPolarEnvironment` for distance and angle relative to an
animal. Shared code has to accommodate both public types. At the same time,
many operations still reach through `env.layout` to access information already
stored in the environment's geometry. Native face sources can retain a builder,
and loading constructs a layout-shaped wrapper around stored data.

Stage F will expose one public `Environment`. The construction methods still
say what the coordinates mean: `Environment.from_samples(...)` for Cartesian
positions and `Environment.from_polar_egocentric(...)` for radius/angle bins.
Both return the same runtime class. Metadata, lookup data, cell measures,
numerical capabilities and coordinate semantics belong to the owned value.
Numerical operations and storage read those data directly.

Merging classes must not reinterpret an angle as a length. Polar bins retain
radius in declared length units and angle in radians, with zero ahead and
positive angles to the animal's left. Unsupported Cartesian operations still
refuse before interpreting polar coordinates. This stage unifies the public
object; it does not add polar affine transforms, polar clipping or new polar
trajectory estimators.

## Entry evidence and CI status

Stage E is complete and independently reviewed. Its final local gates passed
13,727 CPU tests (31 skipped), 596 source doctests (126 skipped), 26 executable
Markdown/API checks, 900 strict geometry/storage tests and typing for 248 files.
The [Stage E record](../environment-affine/validation.md) gives commands,
exclusions, baselines and limits. Those results precede Stage F implementation.

Read-only GitHub inspection on 2026-10-15 found that the parent PR stack is not
fully green:

| PRs | Observed check state / confirmed sampled failure |
| --- | --- |
| #67 | 24 successful checks, one skipped |
| #68 | No failed checks in the inspected rollup; cancelled duplicate runs are present |
| #70–#72 and #74 | Failed Python 3.10–3.12 matrix jobs; sampled #74 Ubuntu/Python 3.12 fails `tests/regions/test_core.py::test_regions_add_point_and_remove` with ambiguous NumPy array truth during equality |
| #73 | Matrix failures; sampled Ubuntu/Python 3.13 fails the spatial threshold-summary test because it requests physical-wall metrics on an inferred footprint |
| #76 | Latest inspected rollup: 12 successful checks, 12 running, one skipped; no completed matrix pass is claimed |

Failure logs: `/private/tmp/neurospatial-parent-ci-312.log` and
`/private/tmp/neurospatial-border-ci-313.log`. The CI run/job URLs are recorded
in [inventory.md](inventory.md). They are observations, not a diagnosis of every
failed job. Fix and verify failures on their owning branches in separate
changes before F1 starts. F0 inventory and planning can proceed meanwhile.
Do not hide them by
weakening geometry checks, silently selecting analysis boundaries, or disabling
older supported Python versions. This planning change does not repair them.

No real tracking/image-mask recording is cached under `data/`. Carry that
validation limit forward; use the smallest available real slice when provided.

## Contracts to preserve

1. Factories run builders once. Environment state, face sources, locators,
   caches and saved data retain no `LayoutEngine` or opaque provider callback.
   Custom-provider adapters keep Stage B's owned built-in snapshot contract.
2. Bin IDs, authoritative `edge_pairs` order/orientation, masks, immediate clip
   mappings and component sources do not change. Navigation distances remain
   distinct from flux distances. Bridges do not acquire physical normals.
3. Polar annular-sector measures, physical graph distances, open/circular
   smoothing, seam handling and boundary-bin semantics remain unchanged.
   One/two angular bins must retain their stored circularity; graph edges alone
   cannot distinguish the open and circular cases.
4. Coordinate semantics become per-instance data, not `_POLAR` on a class,
   `isinstance(env, ...)`, a layout name or mutable provenance. Store one owned
   Cartesian/polar descriptor in the published value; validate it against the
   arena and binning at publication/loading. Reuse the current coordinate
   policies rather than adding a second geometry hierarchy.
5. Preserve Cartesian-only refusals and their exception types. Preserve polar
   instance `hasattr` behavior for currently unavailable Cartesian operations;
   the class may define those methods, but polar attribute access and explicit
   unbound calls must remain guarded. Keep shared graph-based operations usable.
6. Native and framed tensor grids, hexagonal logical indexing, and track linear
   indexing are distinct contexts. A helper for one must not silently claim
   another. In particular, retain existing native hex FFT context while keeping
   Stage E's framed-grid raster restrictions.
7. Loading never runs builders or replays construction parameters. Existing
   Environment-v3, nested binning versions 1–3, and supported NWB records keep
   loading, with owned arrays and unchanged scientific results. Descriptive
   legacy layout labels may remain in records without controlling behavior.
8. Native face construction stays lazy where it is lazy today. Capture minimum
   owned geometry needed to compute faces without a builder; do not replace
   `LayoutFaces` with hidden delegation or eagerly assemble every operator.
9. Optional libraries remain optional and lazy at their integration boundaries.
   No new runtime dependency is planned. `import-linter` is a dev dependency.
10. Existing scientific APIs and data arguments retain their names, order,
    defaults, result meaning and warning locations except the listed class/
    layout-surface migration. Do not use this stage to redesign estimators.

## Target architecture

Keep a single owned environment value rather than introducing a parallel
model. Its essential authority is binning, arena, coordinate descriptor,
metadata/regions and descriptive provenance. Connectivity and other public
views derive from that authority and retain environment-owned caches.

| Layer | Responsibility | Dependency boundary |
| --- | --- | --- |
| Geometry data and validation | Bins, locators, faces, coordinate semantics, arenas, frames and mappings | No environment API, native builders, encoders or optional storage/UI libraries |
| Numerical/domain functions | Operators, lookup/interpolation, allocation, metrics and analysis using data or narrow consumer protocols | No layout facade or concrete environment-class discrimination |
| Construction and integration adapters | Build native geometry once, normalize plugin snapshots, decode/encode storage, render supported geometry | Operate on data; no stored callbacks or builder replay |
| Public API | One `Environment`, factory/method entry points, stable public import paths | Delegates to the functions/adapters; does not duplicate their algorithms |

These are dependency roles, not a requirement to rename every package. Keep
`neurospatial.binning` exports stable. Split builder extraction out of the data
module and validation/plot functions out of `BinningLayout` before deleting that
class. Retain those algorithms in their appropriate data/adapter layer.

Audit public re-exports explicitly: for example, `ops/transforms.py` currently
contains affine primitives and an environment-construction convenience function.
Separate the dependency roles while preserving documented public import paths;
do not exempt the whole numerical package from architecture checks.

### Native face ownership

`Binning.faces` still uses the existing faces-source contract. Replace
builder-backed `LayoutFaces` with closed built-in data-backed sources. Native
builders and the new sources share extracted pure geometry functions, so there
is one formula per geometry family. Grid axes/masks, polar geometry/circularity,
planar cells and track-junction data must supply the same measures as now.

F0 inventories the exact minimal inputs per family before choosing the record
set; do not introduce a separate record for every engine by default. Sources
own their arrays and retain no callable or layout object. Environment caches own
materialization. Storage can continue materializing a source into `KnownFaces`,
as it currently does for native layouts, without a new wire codec merely to
store an internal lazy recipe. Missing geometry retains its existing reason.

### The public-class merge

Place the public merge in **F3**, after owned coordinate state and migrated
consumers exist, and before removing the remaining host/facade machinery.
Internal preparation may temporarily keep the old classes; F3 ends with one
public runtime class.

| Existing use | Stage F target |
| --- | --- |
| `Environment.from_samples(...)` and the other Cartesian factories | Same call; returns Cartesian `Environment` |
| `Environment.from_polar_egocentric(...)` | Same arguments; returns polar `Environment` |
| `EgocentricPolarEnvironment.create(...)` | Migrate to `Environment.from_polar_egocentric(...)` |
| Import or annotate `EgocentricPolarEnvironment` | Remove the old public class/export; use `Environment` with factory-inferred coordinate typing |
| `isinstance(polar, Environment)` | Becomes true; internal coordinate dispatch must not use this test |
| `env.layout` / direct `Environment(layout=...)` | Read supported environment/binning properties; construct through `from_layout` or `from_bins` |
| `env.layout_type` / `env.layout_parameters` | May remain as owned descriptive metadata; never supply capabilities |

Do not add a forwarding polar constructor or alias that leaves two public class
identities. The documented public factory stays discoverable. Preserve the
unfitted/default-construction checks unless an explicit F0 API audit identifies
a necessary change; removing the layout argument need not remove that state.

Preserve static coordinate safety with one generic runtime class: factory
results carry an internal Cartesian/polar type parameter, shared methods accept
both, and Cartesian-only methods restrict their `self` type. A strict standalone
mypy prototype passed for this pattern on 2026-10-15. That is feasibility
evidence only: actual factories, decorators, protocols and supported Python
versions require F3 contract tests. No new public wrapper classes are needed,
and ordinary users should not have to write generic parameters.

Retain runtime guards as well: annotations do not protect untyped callers,
unbound method calls, structural inputs, or restored records. If complete static
coverage requires a different implementation, review it at F0 rather than
silently weakening either check.

### Coordinate restoration and schemas

Normalize coordinate kind and circularity once at construction/loading. Freeze
current legacy circularity precedence in F0, including known arenas, saved
`circular_angle`, one/two-bin ambiguity and graph-derived legacy behavior.
Subsequent calculations read the normalized descriptor rather than parameters.
Copies and metadata changes retain it unchanged.

Prefer writing the existing authoritative coordinate-kind/circularity metadata
from that descriptor. If representing it requires a new stored field, define
and test a versioned schema change before coding it; never insert new authority
into an old version that an older reader would silently ignore. Round-trip
compatibility is part of F1/F3, not deferred to final cleanup. Class removal does
not justify removing supported scientific file readers. Historical pickle class
paths need an explicit F0 audit; promise only the migration support that tests
establish, rather than recreating the old public class accidentally.

## Implementation sequence

Each implementation phase captures the affected behavior first, commits a
logical change and receives independent review. Fix confirmed bugs separately
from behavior-preserving moves. No Stage F runtime code is changed by this plan.

| Phase | Work | Acceptance gate |
| --- | --- | --- |
| F0 — Freeze inventory and migration | Classify the access inventory, record scientific/API/typing/performance baselines, settle the exact constructor/type/codec diff and native face-source inputs | Complete member/consumer capability table and reviewed intended differences; resolve prerequisite CI failures before F1; no speculative provider API |
| F1 — Publish owned state without builders | Extract publication/validation; normalize per-instance coordinates; introduce minimum data-backed native face sources and direct construction/loading | Construction/first-use laziness, native face-array parity, circularity and ownership/copy/dict/file/NWB gates; native builders no longer reachable through published state |
| F2 — Migrate consumers | Move animation, encoding/grid metrics, simulation, shuffling, numerical adapters and writers to owned components/context helpers | Every inventory entry resolved or justified as construction-only/non-environment use; numerical/rendering/refusal baselines unchanged, including native hex indexing and framed raster refusals |
| F3 — Merge public classes | Move polar factories/semantics onto the common class; migrate exports, annotations, results and restoration atomically; remove the old public class | Same runtime class for both factories; typed and untyped Cartesian refusal gates; no mixed length/radian calculations; migration/API snapshot lists only reviewed changes; supported storage still loads |
| F4 — Delete the bridge and mixins | Flatten thin public forwarding methods, replace the broad host protocol with narrow consumers, remove stored layouts, `_OwnedLayout`, `BinningLayout` and builder-bearing adapters | No environment-reachable builders/facades; construction-only layout protocol; public wrapper signatures/decorators stable; all validation and plot implementations still covered |
| F5 — Enforce and document | Add dev-only import-linter contracts and CI gate; execute migrated workflows, audit storage, compare final baselines and run full checks | Architecture, typing, API, strict geometry, CPU/NWB, doctests, runnable documentation and ownership gates pass; intentional differences and unrun checks recorded |

If the F0 audit shows that moving one consumer earlier removes a dependency
cycle, move that preparatory work within F1/F2. Keep the public merge after its
coordinate/ownership prerequisites; it must happen within F rather than being
postponed to another stage.

## Validation and completion

- Record a fresh Stage F baseline on its exact base. The Stage E capture is
  available at `/private/tmp/environment-stage-e-final-rechecked.npz`.
  Update the probe harness before deleting its polar import: retain numerical
  case labels and refusal probes, and compare the public-class migration
  separately. Do not make differences disappear by dropping polar probes.
- Test every built-in family plus explicit/plugin snapshots, subsets, clipped
  and split cells, combined components/bridges, affine frames, native/restored
  hex grids, embedded tracks and missing faces. Preserve graph/edge IDs.
- For polar data, cover open/circular axes, one/two/many angular bins, seam
  points, zero radius, nonfinite rows, metadata units, regional operations,
  plotting, Gaussian/diffusion kernels and egocentric rate results. Freeze
  supported/refused behavior rather than enabling a Cartesian calculation.
- Repeat Stage D wall-contact/observer-side allocation and Stage E interpolation,
  source-plane, count/exposure, inverse-chain, polygon-hole and numerical-face
  regressions with RuntimeWarnings as errors. Keep missing observations distinct
  from zero and retain physical-wall versus selected-bin semantics.
- Verify caller mutation, array/header protection, deepcopy, pickle and dict/
  file/NWB ownership without builders. Corrupted coordinate/frame/face metadata
  must fail before publication. Test scalar path distances independently of flux
  geometry and every existing MissingFaces refusal.
- Run the public API snapshot and argument conventions; source plus all static
  consumer contracts; source doctests and executable Markdown; CPU and NWB
  suites; Ruff/format; import-linter; and import-light optional-dependency checks.
  GUI/GPU tests remain a distinct reported gate, not implied by CPU success.
- Before changing lazy publication/dispatch, measure construction, first face/
  smoothing materialization, batch lookup (including one million positions),
  runtime and peak memory on deterministic identical inputs. Repeat afterward;
  investigate regressions and report deltas. No speedup claim without evidence.
- Run a small real tracking/image-mask slice if available; report its absence
  explicitly. Keep the public workflow's actual coordinate/unit meanings clear.

Completion requires one public class, no stored layout bridge or environment
mixins, passing dependency contracts and scientific comparisons, executable
migration examples and a validation record. Planning, a prototype, or a queued
CI run does not satisfy an implementation gate. Publishing or merging Stage F
remains a separate action.
