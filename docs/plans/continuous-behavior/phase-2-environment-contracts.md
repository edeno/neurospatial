# Phase 2 — Environment contracts for continuous behavior

**Created:** 2026-10-14

**Committed:** 2026-10-14

**Status:** Partially implemented. The representation phases are complete
(PR #67). Slice 2D's physical walls came through arena stages A and D
(#68, #73, #74). The others are partial: 2A has `Environment.bin_sources` (#71),
the validated caller `bin_mapping` in `compare_spatial_rates` (#119) and exact
derivation correspondences from `Environment.bin_mapping_from` (#121); 2B has
measure dimensions (#67, #101) and affine measures (#76); 2C has
zero-conductance path-only bridges (#71); 2E has containment-gated feature
sampling (#118). 2F interval states, optional external `bin_ids`, directed
movement operators and the feature persistence payload have not started.
Evidence is in [TASKS.md](TASKS.md).

**Status reviewed:** 2026-10-20

[← PLAN.md](PLAN.md) · [shared contracts](shared-contracts.md) ·
[environment representation plan](../environment-representation/PLAN.md)

## Objective and ownership

Make spatial identity, measures, movement, physical boundaries and changing
features explicit enough for behavior and neural analyses to share results.
Examples include comparing routes across sessions, sampling terrain or stimuli,
opening a door, and relating fields computed on different spatial supports.

The [environment representation phases 1–4](../environment-representation/PLAN.md)
own stored binnings, locators, capability preservation and persistence. Complete
that refactor under its existing contracts; this phase does not enlarge those
phases or change their behavior-preserving requirements. The work below consists
of follow-on extensions and integrations with roadmap stages A–D. Proposed
amendments are recorded here; the existing representation contracts are not
changed by this planning document.

These slices may introduce public APIs after the representation refactor. Review
names, inputs and exports against [public API rules](../../../.claude/rules/public-api.md)
and [contributing principles](../../contributing.md#design-principles) before
implementation. Use module functions, narrow consumer protocols and optional
delegating environment methods. Do not dispatch by layout type strings.

## Reuse inventory and present limitations

| Existing component | Reuse and limitation |
| --- | --- |
| [Published geometry](../../../src/neurospatial/environment/core.py#L71) and [publication](../../../src/neurospatial/environment/core.py#L921) | Geometry and cache lifetime belong to the environment. Frozen graph structure does not freeze edge attributes, units, or regions; Environment-valued snapshots require roadmap stage B. |
| [Subset implementation](../../../src/neurospatial/environment/transforms.py#L588) | Preserves node attributes but renumbers computational indices. Builder-specific source indices are not cross-session identities. Capability preservation is owned by representation phase 4. |
| [Affine transforms](../../../src/neurospatial/ops/transforms.py#L1345) | Reuse coordinate transforms. Existing measure scaling special-cases tracks and otherwise assumes ambient volume. |
| [FaceGeometry and DiffusionGeometry](../../../src/neurospatial/layout/_diffusion.py#L28) | Keep finite-volume validation and symmetric conductances. Zero-area pairs already carry no diffusion. |
| [Empirical transitions](../../../src/neurospatial/environment/trajectory.py#L649) | Reuse directed count matrices. Current adjacency filtering adds both directions and excludes nonlocal moves by default. |
| [Boundary bins](../../../src/neurospatial/environment/metrics.py#L87), [border analysis](../../../src/neurospatial/encoding/border.py#L108) and [ray casting](../../../src/neurospatial/ops/visibility.py#L585) | Current consumers derive boundaries from discretization; physical-wall integration belongs to arena stages A/D. |
| [Rebin](../../../src/neurospatial/environment/transforms.py#L28) and [field resampling](../../../src/neurospatial/ops/binning.py#L623) | Reuse where contracts apply. Rebin currently supports regular grids only; resampling is not a mass-conserving correspondence. |
| [Intervals](../../../src/neurospatial/_intervals.py) and [trajectory epochs](../../../src/neurospatial/environment/cartesian.py#L429) | Reuse time validation, gap semantics and interval selection; a new timeline framework is unnecessary. |
| [Point lookup](../../../src/neurospatial/environment/cartesian.py#L131) and [nearest-bin lookup](../../../src/neurospatial/layout/mixins.py#L134) | `bin_at` and `contains` do not prove physical containment on every layout. Track lookup remains unbounded nearest-bin assignment through representation phases 1–4. |

## Delivery slices and dependencies

| Slice | Can start after | Additional dependency |
| --- | --- | --- |
| 2A. IDs and correspondences | Representation phases 1–4 | General rebin correspondences wait for exact support from stage D; subset/transform mappings do not. |
| 2B. Coordinates and measures | Representation phases 1–4 | Embedded surfaces need supplied measures; no terrain builder is required. |
| 2C. Movement and faces | Representation phases 1–4 | Compound-arena integration follows stage C; existing bins and explicit movement operators suffice initially. |
| 2D. Physical boundaries | Representation phases 1–4 | Physical-wall consumers require stage A geometry and stage D integration. |
| 2E. Aligned features | 2A | Containment queries require a verified footprint/locator capability or a validated caller footprint; verified grid/polygon containment and explicit nearest queries need not wait for all stage A work. Persistence uses representation phase 3 codecs; side values do not require physical walls. |
| 2F. Interval states | 2A, 2E for fixed-geometry owned feature arrays | Environment-valued snapshots and their identity-keyed caches require existing roadmap stage B. Changed movement uses 2C; changed walls use 2D only for consumers that require walls. |

Behavior analyses that only consume positions, times and a static environment
remain independently deliverable. Queries still require the selected lookup
capability. [Stage B](../environment-representation/overview.md#roadmap-after-phase-4)
is complete: an `Environment` is an immutable value, so state tables can
reference environments directly and cache by their identity.

## 2A — Stable IDs and explicit correspondences

Decisions and tasks:

- Keep array indices contiguous for numerical work. Add optional unique external
  `bin_ids`, scoped to a declared binning identity; generated labels do not assert
  anatomical or physical correspondence across independent recordings.
- Preserve IDs through copying, persistence, coordinate transforms and whole-bin
  selection. A subset returns or exposes exact parent/child index mappings,
  including dropped-bin sentinels. Choose one retrieval convention before adding
  public methods; do not change existing return types implicitly.
- Record source and destination identities, mapping direction and support on
  correspondences. Equal centers or equal array lengths are insufficient evidence
  of alignment; independently built environments require supplied registration.
- A rebin creates new bins and new IDs. Report an exact many-to-one mapping only
  when every retained source cell belongs wholly to one destination cell. Report
  unsupported or not-defined correspondence when this condition is unproven.
- Split source cells require validated overlap measures and a declared allocation
  assumption, or reallocation from original observations. Counts cannot be split
  exactly from a spatial histogram without within-bin information. Never substitute
  nearest-center assignments for overlap geometry or claim exact count transfer.
- Add ID and mapping persistence as a follow-on schema extension, preserving
  files produced by the representation refactor. Missing optional IDs remain valid.

Validation: reject duplicates, missing labels and reversed/mismatched identities;
subset IDs and lookups agree exactly; transform/inverse preserves identity;
nested coarsening preserves additive totals on retained support. Cropping reports
discarded support, and uncovered destinations remain missing rather than zero.

## 2B — Coordinate dimension and measure dimension

The existing [dimension contract](../environment-representation/shared-contracts.md#spatialbins)
is the starting point. The follow-on amendment generalizes it beyond tracks.

- Preserve `n_dims` as coordinate width and `measure_dim` as the dimension of
  bin measure: a track in XY has `(2, 1)`, a surface in XYZ has `(3, 2)`.
- Accept supplied positive measures for supported custom bins. Coordinate
  dimension alone cannot establish cell area, volume or finite-volume faces.
- Scale measures under similarities by `scale ** measure_dim`. Preserve existing
  guaranteed transforms; a non-similarity transform of an embedded object needs
  sufficient geometry or an explicit refusal. Do not infer surface Jacobians from
  the ambient determinant.
- Carry coordinate frame, physical units, measure dimension and quantity meaning
  with alignment-sensitive data. Document mass/count, density and intensive-field
  transfer separately; this adds no statistical estimators or tuning metrics.

Validation: uniform scaling doubles embedded-track lengths and quadruples
embedded-surface areas at scale two; rotations/translations preserve measures;
density integrals respect bin measure; unsupported anisotropic surface scaling
refuses instead of treating area as volume.

## 2C — Movement connections and finite-volume faces

- Retain undirected spatial neighbor pairs and finite-volume conductances as
  distinct from empirical transitions and action-conditioned movement operators.
  The [two-distance contract](../environment-representation/shared-contracts.md#spatialbins)
  remains authoritative during the representation refactor.
- Require explicit conductance or face geometry for movement-only connections.
  A jump/bridge can belong to navigation while its face measure is zero. Native
  hex geometry must not infer a shared face from a nonlocal edge's length.
- Initially permit a validated pair superset with zero conductance for path-only
  pairs. Keep the existing face order/orientation contract; independently indexed
  faces require a separate numerical design and are not required here.
- Accept directed movement/count operators separately, with explicit bin
  alignment and semantics. State whether weights mean counts, probabilities or
  costs. Validate shape, finite values and the applicable normalization; do not
  silently symmetrize an operator used for directional path analysis.
- Keep known movement links separate from observed jumps and tracking artifacts.
  Removing a sensory fact or changing route preference does not delete geometry.
  Existing default behavior changes need explicit release notes and API review.

Validation: a path-only bridge shortens navigation distance without transferring
diffusive mass; a one-way operator forbids its reverse; empirical reverse counts
remain independent; positive-conductance components agree with diffusion support;
copy/save/load preserves pair ordering and declared connection semantics.

## 2D — Physical boundaries and analysis support

Integrate with [arena stages A/D](../environment-representation/overview.md#roadmap-after-phase-4),
without implementing a second arena model in continuous behavior.

- Distinguish physical walls, missing/unknown footprint and analysis selections.
  Selecting whole bins changes analysis support; it does not add a physical wall.
  Cell-face no-flux treatment at a selection boundary remains a numerical boundary
  condition and must not be exported as an observed wall.
- Let consumers request the boundary capability they need. Unknown walls must
  produce a clear refusal or an explicitly selected analysis-boundary mode.
- Preserve original physical geometry through analysis subsets. Implement clipped
  cells, altered measures and faces only under stage A's clipping contract.
- Migrate wall-distance, border and visibility consumers individually under stage
  D, documenting each intended behavior change. Basic point/route analyses can
  continue without physical-wall information.

Validation: cropping an interior region creates no new wall; a genuine barrier
changes wall distance/visibility; unknown footprint is distinguishable from an
empty wall set; physical geometry survives persistence and coordinate alignment.

## 2E — External aligned features

- Use validated arrays or tables outside the geometry, keyed by binning identity
  and, when supplied, `bin_ids`. Without external IDs, exact array alignment
  requires the same declared binning and ordering. Label feature columns with
  unique `feature_ids`. Support per-bin scalar/categorical values and directional
  side slots; a wall-facing slot need not correspond to a navigation edge.
- Declare field units, missing values, and side direction frame/order. Shape
  equality does not authorize alignment. Subsets follow exact correspondence;
  transforms preserve scalar values and explicitly update geometric directions.
- Gate `method="containment"` on a verified footprint/locator capability or a
  validated caller-provided footprint in the query's coordinate frame and units.
  Reject unknown containment rather than treating a successful nearest-bin
  lookup as proof that the query is inside. Validate any supplied footprint's
  relationship to the bins and use it before assigning values.
- Verified grid/polygon containment can ship without the full stage A roadmap.
  A track or abstract graph with unknown footprint cannot claim containment.
  Permit an explicit `method="nearest"` earlier, documenting its distance policy
  and that assigned values do not establish physical support.
- Apply the same containment gate to direct and egocentric queries; finite
  transformed coordinates are not sufficient support. Mask invalid indices
  before field lookup. Footprint membership does not imply known physical walls;
  wall-aware consumers retain the separate capability requirements of slice 2D.
- Keep measured features, stimulus schedules and incomplete knowledge separate.
  A knowledge mask selects available observations without changing occupancy,
  cell measures, locator support or movement feasibility.
- Define a small persistence payload and validation protocol before choosing any
  public container. Do not add terrain-specific environment subclasses or require
  a universal feature registry. Unsupported feature remapping stays explicit.

Validation: reordered IDs realign correctly; wrong identities fail; a side-slot
rotation follows the declared frame; missing data remain missing; changing a
height/texture/knowledge array leaves environment geometry and caches unchanged.
Add a track regression: a track from `(0, 0)` to `(10, 0)` currently assigns
`(1000, 1000)` to a bin and reports `contains=True`. Containment sampling must
reject an unknown footprint or mark that query unsupported under a validated
footprint; explicit nearest sampling remains separately labeled. Verify both
direct and egocentric cases, plus supported grid/polygon queries that need no
blanket stage A dependency.

## 2F — Feature and topology states by interval

- Default to an external table of nonoverlapping half-open intervals
  `[start, stop)`, state IDs and references to immutable feature/topology snapshots.
  Use `times` in seconds, existing interval validation and a declared time origin.
- Deliver fixed-geometry feature-only tables first if useful: own read-only
  feature arrays and immutable units/frame/state metadata, aligned to one fixed
  binning.
- Environment-valued snapshots reference immutable `Environment` values (stage B,
  complete) and may reuse caches by snapshot identity. Implement no second
  environment representation.
- Feature-only changes reuse one fixed binning. Topology changes select immutable
  movement/environment snapshots by read-only reference; do not copy geometry
  per lookup. Shared IDs require a common declared binning;
  changed spatial support requires an explicit correspondence or no mapping.
- Reject overlap, missing declared IDs and incompatible coordinate units/frames.
  Uncovered intervals are unknown and produce invalid output rather than selecting
  the nearest state. Partition intervals at state changes;
  a transition straddling a change needs a declared exclusion or assignment rule.
- Cache by immutable snapshot identity and all relevant operator inputs. Do not
  mutate an environment to replay a session or clear a shared cache by hand.
- Keep state lookup as a small adapter consumed only where needed. A dedicated
  timeline object, continuously moving meshes, and cognitive inference frameworks
  are deferred until multiple scientific workflows justify them.

Validation: samples at a change select the new state; gaps stay uncovered;
feature changes preserve bin IDs; opening a door changes only the declared
operator/walls; snapshot reuse cannot leak cached geometry between states.
Before Environment registration ships, test attempted mutation of edge attributes,
units, and regions after registration. Mutations must be refused or require a new
immutable Environment under stage B; no accepted snapshot or its cached operators
may change. Separately mutate caller-owned feature inputs after registering an
early fixed-geometry table and verify its owned values and metadata are unchanged.

## Completion and review

Each slice supplies contract tests for the invariants above, runnable public-API
examples, relevant file/NWB round trips, and documented refusals. Run focused
tests plus required API, typing and documentation checks for its implementation;
record intentional changes to existing behavior. Review the diff independently.
Completion requires useful consumer integration, not just unused data containers.
