# Stage A — Continuous arena representation

**Status:** Draft. The boundary-selection policy, wall-connectivity policy and
initial clipping scope await the user's decisions. No implementation has begun.

**Branch:** `feature/environment-arena`, stacked on
`feature/environment-representation` at `944c818d`.

This is the next stage in the [representation roadmap](../environment-representation/overview.md#roadmap-after-phase-4).
Stages B–F remain separate work. The completed representation plan and its
validation record remain unchanged.

## Scientific outcome

An environment distinguishes the continuous arena, the bins chosen to represent
it, and the support selected for an analysis. A physical wall and the edge of an
analysis selection have different meanings. A track centerline alone does not
specify the physical footprint of the track.

For known arena geometry, point assignment first checks the arena, then its
binning. Whole-bin selection keeps complete cells. Clipping creates new cell
geometry and updates every quantity that depends on it.

## Decisions required before the implementation plan is finalized

| Decision | Recommendation | Alternative and consequence |
| --- | --- | --- |
| Whole-bin selection at a boundary | Select bins whose centers are strictly inside. This preserves the current polygon factory's selection rule. | Require a specified minimum overlap fraction. This needs exact cell footprints and an explicit fraction, and can select different bins. |
| Physical internal walls | Remove blocked navigation pairs and their matching finite-volume face records. | Store wall geometry without changing connectivity; paths and diffusion could still cross the wall. |
| Initial clipping scope | Implement exact axis-aligned rectangular clipping of Cartesian rectilinear bins. Design general polygon clipping separately. | Include general polygon clipping now, with an additional design pass for disconnected cell pieces, centers, lookup and diffusion validity. |

These recommendations are not yet accepted decisions. Any restrictions below
are proposals to review with the finalized plan, not implemented guarantees.

## Current evidence

At the branch base, deterministic probes reproduce these mismatches:

- A circle of radius 5, built with `from_polygon(..., bin_size=2)`, assigns
  `(4.9, 1.8)` and `(4.5, 2.5)` to bin 20 even though both are outside the
  polygon. `contains` also returns true. The polygon builder selects cell
  centers; point lookup subsequently reads only the rectilinear locator.
- A straight track from `(0, 0)` to `(100, 0)`, with 2-unit bins, assigns
  `(1000, 1000)` to bin 49 and reports containment. Its nearest-bin locator
  has no distance limit, and the factory has no track-width argument.

Relevant implementation points:

- [Polygon construction](../../../src/neurospatial/layout/engines/shapely_polygon.py):
  `ShapelyPolygonLayout.build` selects centers with `shapely.contains`.
- [Cartesian queries](../../../src/neurospatial/environment/cartesian.py):
  `bin_at` and `contains` call the locator independently.
- [Time allocation](../../../src/neurospatial/environment/trajectory.py): linear
  occupancy has its own grid-segment allocation path. Gating sampled points
  alone cannot correct a segment passing through a hole or excluded cell part.
- [Subsetting](../../../src/neurospatial/layout/_derive.py): `subset_binning`
  restricts bin data and removes the drawing outline. Physical walls cannot
  be recovered from the child's outline or selected bins.
- [Environment metadata](../../../src/neurospatial/environment/core.py): units,
  frame and regions are currently environment-owned mutable metadata.
- [Diffusion geometry](../../../src/neurospatial/layout/_diffusion.py): ordered
  pair arrays and face arrays must agree; zero conductance alone does not
  remove a navigation edge.

The existing baseline script recorded **4,046 probes** at `944c818d`, including
592 exception outcomes, before any implementation. The local artifact is
`/private/tmp/environment-arena-base-944c818d.npz`. It is not a portable fixture;
recreate it from that commit when repeating the comparison.

## Proposed contracts

### Arena and coordinate metadata

- Add a small arena interface and the roadmapped `OpenArena`, `TrackArena`,
  `MaskArena` and `EgocentricSurround` representations. Keep geometry as owned,
  immutable data, with no reference back to an environment or layout builder.
- Reuse the existing Cartesian and polar coordinate-system implementations.
  Units, frame and regions keep one authority on the environment in Stage A.
  Moving them into an immutable environment value belongs to Stage B.
- Distinguish a known physical footprint from unknown geometry. Never infer
  physical walls from sampled occupancy, a bounding box, provenance, a nearest
  center, or a subset mask. Factory-by-factory inference rules must be listed
  and tested before publication changes.
- A graph without an explicit positive track width has an unknown physical
  footprint. Preserve its current abstract graph operations; a query requiring
  physical containment must report that the footprint is unknown. Known-width
  tracks use distance to the actual centerline segments, not bin centers.
  Proposed initial footprint: constant full width, with round end caps and the
  union of incident segment footprints at junctions. Use exact point-to-segment
  distance for containment rather than an approximate polygon buffer.
- An egocentric distance cutoff or angular window is an analysis boundary,
  not a physical wall. Preserve circular seam semantics.
  Adding its domain representation does not make the polar sibling inherit
  Cartesian-only queries, subsetting or affine transforms.
- Domain geometry and asserted physical walls are separate knowledge. A mask
  can define a continuous analysis domain without proving it is the arena's
  measured footprint. The finalized factory matrix must state both explicitly.

### Lookup, support and coverage

- `env.contains` continues to mean assignment to an active bin. It must agree
  exactly with `env.bin_at(positions) >= 0`. Physical arena containment is a
  separate query and may be true where an analysis has no active bin.
- For known geometry, the effective lookup is arena containment followed by
  the stored locator. Locate only valid finite rows; preserve `-1`, dimensions,
  bin identity and existing ownership of boundaries between bins.
- Specify physical-boundary inclusion separately from whole-bin selection.
  Proposed physical containment includes the boundary; the locator still
  determines whether it belongs to an active bin. This does not change
  half-open grid-cell ownership.
- State the coverage guarantee precisely: whole-bin approximation can leave
  parts of an arena without an active bin. Never promise every arena point
  maps to a bin. Clipping and overlap selection need their own guarantees.
- Cover occupancy and encoding paths that bypass `bin_at`, including linear
  time allocation. Segments through holes or outside a known arena must not
  silently accrue occupancy there. Nearest-center mapping remains an explicit
  approximation, with its existing separate API.
  Audit remaining `env.layout` exact-lookup readers as well; a gate only on the
  convenience method must not leave another exact API assigning outside points.

### Walls and derivation

- Whole-bin subsets retain their parent's physical arena. Dropping a bin never
  creates a physical wall. A crop boundary must not replace the arena boundary.
- Transforms map arena geometry together with binning. Preserve the existing
  proven transform restrictions, including refusal of track non-similarities.
  Copy, deepcopy and pickle preserve ownership and geometry.
- If wall blocking is chosen, preserve ordered pair/face consistency and
  rebuild connectivity before publishing the geometry once.
- Proposed initial wall restriction: accept walls supported by cell boundaries;
  refuse a wall cutting through a cell interior until splitting or rebuilding
  can represent both sides. Pruning neighboring edges cannot repair one bin
  spanning an impenetrable wall. Define corner and endpoint contact explicitly.
  A wall must block complete shared faces, with doorway endpoints aligned to
  cell vertices. A wall covering only part of one face needs explicit opening
  geometry and navigation rules; refuse it rather than blocking an entire
  doorway or retaining conductance through a closed portion.
- Border-score and ray-casting migrations remain Stage D. Their behavior can
  change indirectly through corrected containment, but Stage A must not claim
  that these consumers already use physical walls.

### Clipping

- Keep existing whole-bin `subset` semantics distinct from a clipping API.
  Clipping an analysis region changes support without inventing physical walls.
- Under the recommended initial scope, intersect axis edges with rectangular
  limits, retain positive-size cell intersections, and recompute centers,
  sizes, lookup, navigation distances, face areas and flux distances.
- Exact physical clipping initially requires the existing arena domain to be
  representable by the rectilinear cells as well. A circle discretized into a
  square grid does not meet that precondition: clipping its grid edges would
  still leave square-cell area outside the circle. Refuse known curved or
  oblique arena boundaries until general clipping can intersect those too.
  Cell-aligned masks, including holes, remain eligible. For an unknown physical
  footprint, a clip can establish exact discrete cell geometry only; it must
  retain the unknown-footprint status and make no physical-area claim.
- Retain the parent's allowed neighbor policy, including zero-area diagonal
  links where applicable. All graph and face arrays use the same order.
  Never turn a path-only link into a conducting face. Arbitrary partial face
  conductances do not locate their openings; refuse those clips unless stored
  physical face geometry determines the clipped opening.
- Do not route partial edge cells through a builder that assumes uniform
  widths. Publish the resulting data directly. Audit rectilinear consumers
  before retaining interpolation or linear-time-allocation capabilities.
- Explicitly refuse unsupported dimensionality, bin geometry or clipping
  shape with a supported alternative. Do not approximate a polygon clip by
  selecting centers and call it clipping.
- General polygon clipping needs a separate design for disconnected pieces
  and representative centers. Exact shapes or areas alone do not prove that
  the current two-point finite-volume solver is valid.

### Persistence

- Proposed format change: write `Environment-v3` and NWB `3.0`, with one shared
  arena codec representation used by dictionaries, files and NWB. Old readers
  must not silently discard arena-gated lookup or walls.
- Proposed compatibility: read existing `Environment-v2` / NWB `2.0` binnings
  without rebuilding, with an explicitly unknown physical footprint. Do not
  guess walls from the stored bins or descriptive build parameters.
- Validate geometry, coordinate dimension, effective center lookup and
  arena/binning consistency before publishing. Reject unknown required codecs
  with an actionable error. Preserve existing corruption checks.
  New-format records must explicitly encode known or unknown geometry; a
  missing arena record is corruption, not permission to discard containment.
- Persist derived arenas as data; no layout factory or provenance replay on
  loading. Missing physical geometry remains unknown after every round trip.

The public signatures and persistence-version proposal require review with the
final plan before implementation. They are not approved by this draft.

## Proposed implementation sequence

Each phase gets a logical commit, focused validation, and independent review.
Fix confirmed review findings before advancing. Every phase remains **not
started** until the decisions above are resolved and its contract is concrete.

1. **Arena data, codecs and contract fixtures.** Finalize the public surface,
   factory matrix, unknown-footprint behavior and boundary rules. Add the shared
   arena codecs and reviewed transport/version handling before enabling new
   lookup semantics. Add deterministic simulated arenas with holes, masks,
   known-width tracks and polar windows.
2. **Publication and assignment.** Attach the arena through the one geometry
   publication path; reconcile all exact lookup and occupancy entry points.
   Include subset, copy, supported transforms and their arena-aware persistence
   in this phase. A save must never silently drop newly enabled semantics.
3. **Walls and connectivity.** Implement the chosen wall rule with the stated
   representability restrictions. Validate both paths and diffusion across
   barriers and through openings.
4. **Exact clipping.** Implement the approved supported geometry, maintaining
   real sizes, graph and valid numerical capabilities. Verify direct-build
   or analytical oracles rather than testing only the new implementation.
5. **User workflows and broad validation.** Finish documentation and exercised
   examples, verify the full transport/compatibility matrix, and review the
   accumulated API snapshot changes. Run broad validation and reconcile every
   baseline delta. Each earlier phase already tests its persistence behavior
   and updates its API snapshot when introducing a public API.

## Validation required before claiming Stage A complete

| Area | Independent oracle or invariant |
| --- | --- |
| Continuous containment | Analytic rectangles, circle/holed polygon predicates, exact centerline distances and masked cell unions; inside, outside, boundary, NaN, empty and wrong-width inputs. |
| Bin support | `contains == (bin_at >= 0)`; dropped-bin points stay `-1`; known-arena exterior is never assigned; unknown footprints do not become invented walls. |
| Coverage | Explicit counterexamples for center selection near boundaries; positive-measure coverage within the supported clipped cell domain; refusal of curved-domain cases that a grid-edge clip cannot represent exactly. |
| Occupancy | Hand-computed dwell times for segments crossing arena edges and holes, including a segment with both sampled endpoints inside. Unsupported intervals stay missing rather than becoming zero activity. |
| Walls | No navigation path or positive diffusion conductance crosses a closed wall; doorways stay passable; interior-cut walls follow the reviewed split/refusal policy. |
| Clipping | Analytic partial-cell lengths/areas, center-to-bin identity, ordered pairs/faces, flux-distance and conservation checks. Include one-cell and repeated clips. |
| Derivation | Whole-bin subsets keep physical walls; transformations preserve lookup under mapped probes; original environments remain unchanged. |
| Persistence | Dict, file and actual NWB file round trips preserve arena, lookup, walls, measures and allowed analyses; v2 fixtures retain unknown footprints; corrupted arena/binning combinations fail before publication. |
| Ownership | Caller mutation, deepcopy and pickle cannot mutate published arena geometry. |
| Workflow | Seeded occupancy → rates → decoding workflow; smallest available recorded-data slice if a suitable fixture exists. Report real-data validation as unavailable if it does not. |

Use the existing isolated environment with `uv run --no-sync`. Before the final
phase is accepted, run relevant CPU tests, doctests, executable documentation,
API and argument-convention checks, Ruff, and source/typing-contract mypy.
Compare against the base behavior capture and classify every change. Measure
construction and batched lookup before and after on identical inputs, since
containment adds work to the lookup path. Report timings and any unrun checks;
do not claim an unmeasured speed improvement.
