# General polygon clipping — Stage A design

[Stage A plan](PLAN.md)

**Status:** Implemented and independently reviewed. The user approved
general polygon clipping, splitting disconnected pieces with a parent mapping,
and numerical operations only on certified geometry. Center-based whole-bin
selection and connectivity-blocking walls are accepted too. The design and
implementation passed independent geometry, numerical and persistence reviews;
see the [validation record](validation.md).

## What clipping means

The scientific operation is to retain a continuous analysis region while
accounting for the true amount of space in every retained bin. Selection by
center is a different operation: it keeps entire bins and their old measures.

Let `C_i` be a stored parent cell, `P` the requested clip, and `D` the known
polygonal arena domain. The retained geometry is `C_i ∩ P ∩ D`, including a
known computational domain without claiming a measured footprint. When no
polygonal domain is available and the arena domain is unknown, use `C_i ∩ P`
and preserve that unknown status. A known domain lacking the materialization
needed for exact clipping is explicitly refused. A request
cannot reconstruct inactive bins that are absent from the parent.

The output covers exactly the represented parent support intersected with this
effective clip, up to floating-point geometry arithmetic. It need not cover
every part of the physical arena: center-based parent construction can already
have omitted boundary cells. Do not silently rebuild those missing cells.

Here "exact" excludes rasterization, center-selection approximations and
silently dropped slivers. It does not promise exact real arithmetic or
bit-identical overlays across different GEOS versions.

## Recommended scientific choices

| Choice | Proposal | Why it matters |
| --- | --- | --- |
| One cell becomes disconnected | Create one child bin per interior-connected positive-area polygon, with a child-to-parent map. | Lobes touching only at a point remain separate states. Bin count can increase. |
| Bin representative | Use the area centroid if strictly interior; otherwise a deterministic interior representative point. | A concave polygon or a polygon with a hole can have its centroid outside its own support. |
| Numerical capabilities | Certify the existing two-point geometry or publish `MissingFaces`. | Exact polygon areas alone do not justify physical diffusion, gradients or divergence. |
| New solver | Do not add a general-mesh diffusion solver in Stage A without separate authorization. | A larger stencil changes the numerical contract and needs its own validation. |

Representatives are chosen from the resulting geometry, not preferentially
inherited from the parent. Keeping an old center whenever it remains inside
can give different centers for direct and repeated rectangular clips. Use
stored representatives after loading; transform them with the geometry rather
than recomputing a new representative during a transform.

## Supported inputs and explicit limits

The general operation accepts valid, nonempty, finite Cartesian 2-D `Polygon`
or `MultiPolygon` inputs, including holes and disconnected regions. Invalid
input is rejected with the geometry defect and a fix. Do not silently call
`buffer(0)`, snap coordinates, repair self-intersections or choose a precision
grid. An empty retained area is an actionable error, matching the existing
nonempty-environment contract.

Source cells must have exact polygonal geometry available from stored data:

| Source | Geometry authority |
| --- | --- |
| Cartesian rectilinear grid or mask | Shared axis edges and active-cell indices. |
| Native regular hexagons | A canonical shared lattice, verified against stored centers, locator and shapes. |
| Triangular mesh | Stored vertex IDs and active triangle indices. |
| Previously clipped cells | Stored connected polygon cells, including interior rings. |
| Supported derived cells | The same retained components and exact mapped shared vertices. |

A drawing outline is not cell geometry. Refuse missing exact source geometry
rather than invent Voronoi cells or rerun a factory. A transformed grid whose
exact cells are no longer published is unsupported until a proven stored-data
materializer is provided. This does not bring Stage E's general grid transform
work into Stage A.

Tracks have length measures, polar bins have radial/angular semantics, and
3-D cells need volumetric clipping. These are not planar polygon clips and
remain explicit refusals for this API. Track arenas and egocentric domains
still receive their other Stage A containment contracts.

## Public surface and mapping

Proposed entry point: `env.clip(polygon, *, name=None) -> Environment`. It is a
Cartesian operation, alongside `subset`, delegating to a module function.
Use an explicit Shapely box for a rectangle rather than adding a second bounds
signature or a selection/clipping mode flag.

The returned environment exposes an optional immutable `clip_mapping` record
with `parent_bin_indices`, shape `(n_child_bins,)`, and `n_parent_bins`.
The record is present after a clip and identifies the input bin for each
output bin. It owns its integer array and contains no reference to the parent
environment. It is stored with the environment payload, not in descriptive
build parameters.

- Order children by parent index, then a deterministic geometric key for each
  connected piece. Canonicalize ring ordering for this key. Do not promise the
  same numbering after different operation histories or geometry-library
  versions. Saving and loading preserve the stored order exactly.
- A new clip replaces the mapping with one to its immediate input; no ancestry
  chain is exposed by this public record. Copy and transform preserve it. A
  whole-bin subset restricts its rows and still refers to the same most-recent
  clip source. Ownership locators separately retain the prior snapshots needed
  to preserve boundary assignment; that is not a public ancestry/mapping API.
- Each map index is in `[0, n_parent_bins)`. Repeated values are expected when
  a parent splits. The array length must equal the published child bin count.
- Do not transfer a field, trained decoder, occupancy or counts automatically.
  Mapping a parent's rate to two children assumes spatial uniformity; copying
  counts duplicates events. Recompute from the recording for the new support
  where possible. A generic field-transfer API is outside Stage A.

The reviewed public name and mapping placement preserve the usual
environment-returning operation and give
the mapping one persistent authority, without a second result-loading API.

## Geometry and publication algorithm

1. Validate source components, coordinate kind, input polygon and any known
   planar arena domain. Reject overlapping source-cell interiors or ambiguous
   ownership. Materialize only required active cells and retain parent IDs.
2. Build shared source vertices and edges once. Grids use their stored edge
   coordinates; triangles use stored vertex IDs. Native hexagons need integer
   lattice vertex keys and one coordinate evaluation per shared vertex.
   Independently translating trigonometric drawing polygons can produce
   microscopic overlaps and zero shared-edge length for intended neighbors.
   Verify any lattice reconstruction against published geometry using a
   documented, locally bounded round-off allowance; otherwise refuse it.
3. Node the union of source boundaries and the effective clip/domain
   boundaries into one planar line arrangement. Polygonize that linework;
   classify each positive-area atomic face by a strict interior probe. Assign
   it only if one parent owns its interior and the effective clip retains it.
   Ambiguous ownership is an error, not a tie resolved by bin order.
4. Union retained atoms within each parent, then extract interior-connected Polygon
   components. Preserve holes. Discard only empty or zero-area line/point
   intersections. Retain every positive-area piece; do not apply an arbitrary
   area threshold or merge a sliver into a neighbor.
5. Compute component ordering, representatives, areas and mapping. Verify
   interior membership, disjoint interiors and area/union conservation against
   an independent intersection oracle. Refuse unusable floating-point
   geometry if no valid finite interior representative can be certified.
6. Derive open interfaces from the shared arrangement, preserving source
   neighbor permissions and asserted walls. Build ordered child pairs, their
   navigation distances, and certified faces or a `MissingFaces` explanation.
   Insert nodes in child order and edges deterministically, then take the
   resulting graph's edge iteration as authoritative for pair and face arrays.
   Splitting nodes can change that iteration relative to source-pair order.
7. Publish one new binning and arena reference through the existing publication
   path. The analysis crop does not create physical walls or replace the
   parent's arena. Copy units/frame; follow the existing whole-bin subset
   policy of dropping analysis regions, without mutating the parent's regions.

Global noding is a topology strategy, not a claim that floating-point errors
disappear. Validate the result. If an overlay is ambiguous or invalid, report
the offending parent/geometry; do not silently change its precision or mask it.
Measure the cost and peak memory before deciding on batching or spatial-index
optimizations. Do not claim that independent tile overlays are equivalent
without checking their shared boundaries.

## Stored cell geometry, lookup and rendering

Add one immutable `PolygonCells` component containing an ordered tuple of
connected Shapely Polygons. Each can contain holes. These polygons are the
single authority for shape, cell area and piece membership. Add
`ClippedLocator(base, cells, mapping)`: at the instant of a clip it references
the new environment's same immutable `PolygonCells` and `ClipMapping` records,
with no duplicate polygon buffers. First ask the immediate input's stored
locator for an owner, then check only child polygons mapped to that owner.
It may memoize an STRtree; the memo is rebuildable and never serialized. This
is a data component, not another layout builder or public plugin mechanism.

Each locator's references are immutable snapshots in its own coordinate frame
and bin numbering. A later transform publishes mapped current polygons but
retains the inner locator's original polygons under an inverse-transform
wrapper. A subset publishes restricted current polygons/map rows while its
inner locator retains the original row count under a subset wrapper. These
are deliberately different snapshots, not inconsistent duplicates to overwrite.

- `SpatialBins.sizes` equals polygon areas, with `measure_dim == 2`.
- Each representative lies strictly inside exactly its own polygon.
- Non-finite rows return `-1`; empty/wrong-width behavior follows the locator
  contract. Lookup must check geometry, including holes, rather than nearest
  representatives.
- Interior membership is unambiguous. Shared-boundary ownership must be
  deterministic: preserve an available parent owner first, then use the
  stored child order for remaining ties. New clip-created outer boundaries
  use closed-domain membership. Do not add an epsilon-wide buffer.
- For points on inherited source boundaries, preserve the parent's ownership
  and outside sentinel, including grid upper-edge exclusions. If the parent
  owner was dropped, do not assign the point to a different parent. The
  saved `base` locator and map carry this rule without retaining an entire
  parent environment. Nested clipping/subset/transform locators are permitted,
  as in the existing derivation design. Do not flatten them until equivalence
  is proved: a subset may deliberately discard a shared-boundary owner.
- Subset restricts polygon cells and mapping. Transform maps stored polygons
  and representatives without renumbering. Every derived codec preserves the
  same geometry/ownership authority.
- Use compound plotting paths with correctly oriented hole rings in both
  `plot` and `plot_field`. Hole interiors must remain unfilled. Consumer paths
  that cannot draw polygon cells must refuse explicitly rather than pretend
  their bounding boxes or centroids are exact cell shapes.

Retain a `RectilinearGrid` only after proving the result is that full Cartesian
product with its active mask and midpoint representatives. General polygon
cells do not inherit grid interpolation or time-allocation shortcuts. A
rectangular optimization may publish that component only if its bin numbering,
lookup boundaries, areas and mapping match the general operation's contract.

## Adjacency and physical walls

Create a child pair only where its two source parents already had an allowed
pair and the corresponding contact survives clipping. Positive-length open
shared interfaces support geometric adjacency. Preserve parent path-only
contacts as path-only; do not turn them into conducting faces. Retain an
explicit parent diagonal/corner link only if its contact point survives and no
asserted wall blocks it. Do not connect disconnected pieces of the same parent
merely because they share a parent ID.

All closed physical interfaces remove navigation pairs and matching face
records. Missing source neighbor permissions remain missing. No geometric
overlay may restore a blocked parent edge.

Stage A's wall-insertion restriction remains full supported
cell faces, with aligned doorway endpoints. General polygon clipping does not
implicitly authorize a new algorithm for walls cutting through a cell or
ending halfway along a face. Refuse those unresolved wall geometries until
their representation and opening rules are reviewed. Clipping an already
supported wall preserves its surviving geometry.

Navigation distances remain graph-discretization distances. Adjacency through
an open interface establishes a connection; it does not prove a straight
segment between representatives lies inside a concave pair of cells. The
public graph path is a list of bins, not a continuous collision-free route.
Document this existing distinction; do not claim exact physical geodesics
from representative-to-representative chord weights.

## Finite-volume capabilities

The existing numerical contract stores one face measure `A` and flux distance
`d` per pair and uses the symmetric conductance `A / d`. Gradients/divergence
share that geometry. This representation is not a general polygon-mesh
diffusion solver. TPFA consistency requires appropriate orthogonality;
[DuMux's primary discretization documentation](https://dumux.org/docs/doxygen/releases/3.9/group___c_c_tpfa_discretization.html)
describes the consistency lost when the omitted non-orthogonal term is
nonzero. Stage A applies a conservative eligibility test rather than treating
positive weights or mass conservation as a proof of correctness.

Proposed certification for the current isotropic two-point scheme:

1. Every active cell is a valid convex polygon with a strictly interior
   representative. Cells with holes or concavity remain geometrically
   supported, but conservatively fail this numerical certificate.
2. Each conducting pair has a positive-length planar shared opening with
   known physical geometry. Require one connected straight aperture;
   conservatively refuse multiple separated openings, even if collinear.
   Asserted walls have already been removed. Unlocated partial conductances or
   arbitrary inherited flux weights are not geometric evidence.
3. Representatives lie on opposite sides of that interface, their connecting
   vector is perpendicular to it within a stated scale-relative round-off
   allowance, and the connecting segment crosses an open part of the shared
   interface. Distances are finite and strictly positive.
4. After orthogonality is certified, use actual open interface length for `A`
   and Euclidean representative distance `norm(x_j - x_i)` for `d` (equal to
   normal separation in exact admissible geometry). Do not project a skew
   vector to make an ineligible pair pass. Cell measures are actual polygon
   areas. Keep ordered pair/face arrays aligned; explicitly permitted
   path-only pairs have `A == 0`.
5. Check affine-field fluxes, constant preservation and volume-weighted mass
   conservation on independent analytic fixtures. These tests supplement the
   geometry checks; conservation alone does not certify the discretization.

If any cell or conducting pair fails, publish exact cells/lookup/areas with a
`MissingFaces` reason naming the first failure. Do not silently delete a real
opening, replace `d` with a chord or projected number, truncate a negative
weight, or claim a graph surrogate is physical diffusion. All consumers using
faces must give the same actionable refusal. Ordinary non-diffusion queries
and exact plotting remain available.

The current scheme closes exposed boundaries with homogeneous no-flux
conditions because only interior faces enter the operator. A crop therefore
introduces a numerical analysis-boundary condition without asserting a physical
wall. Dirichlet or absorbing boundaries are outside this API. Affine-field
validation checks conducting-face fluxes; it must not demand a zero global
Laplacian for an affine field incompatible with no-flux exterior boundaries.

This intentionally conservative certificate can reject some valid specialized
meshes. Broader numerical support needs a separate reviewed certificate or a
general-mesh solver. Centroid recentering can lose parent orthogonality, even
when the parent was a valid grid or hexagonal mesh. That is an explicit
capability change, not an implementation accident to hide.

Numerical tolerances must bound arithmetic error using local feature lengths
and coordinate representability. They are not a permitted physical skew
angle, and must be capped below the nearest meaningful local feature gap.
If coordinate round-off prevents distinguishing those scales, certification
fails. Existing layout warning tolerances do not constitute this certificate.
The implementation must state its error bound and verify it with shifted,
rescaled, near-orthogonal and deliberately skewed fixtures before publication.

## Occupancy, interpolation and scientific results

Separate observation validity from geometric support before linear allocation.
An observation-valid interval has finite endpoints, valid timing and the
recording-window/gap support required by the existing estimator. Its start
point need not already map to a retained bin. Do not reuse the current
start-bin-based interval mask to drop an outside-to-inside segment wholesale.

Intersect each observation-valid trajectory segment with the clipped polygon
cells, convert the retained parameter intervals to durations, and resolve
boundary ownership once. Holes and outside portions receive no allocated
occupancy. A segment can enter and leave a cell multiple times; endpoint-only
gating and assigning the entire segment to one parent are wrong. Preserve the
existing missing-time and observation-gap semantics independently of this
geometric subdivision.

Use the same retained time intervals for spike/event counting and occupancy,
including entry, exit, re-entry and the existing final-endpoint convention.
Spatial rate estimators expose this through `time_allocation="linear"`;
their default remains `"start"`. Retained time windows are half-open: entry
is included subject to spatial bin ownership, exit is excluded, and adjacent
retained windows merge. Events
at an internal shared cell boundary use the locator's bin owner. Preserve
these choices in analysis settings and observed time intervals. Explicit
linear spatial encoding initially requires exact planar cell geometry.
If an owned boundary spike falls in a bin with zero retained dwell time,
refuse the undefined rate explicitly; do not reassign its owner or silently
discard the spike from the estimated rates.
Spikes outside retained intervals do not contribute to a numerator whose
occupancy denominator excludes them. A known physical wall crossing cannot be
made valid by its zero area; linear allocation must refuse an unresolved
barrier-crossing interval rather than silently interpolate through the wall.
Analysis crop crossings, which are not physical barriers, are subdivided.

Nearest interpolation is still a named approximation based on centers. It
must not be described as polygon membership or silently used as an exact
occupancy allocator. Audit encoding, decoding, resampling and exact lookup
entry points so they cannot bypass the effective arena/cell gate. Retain grid
linear interpolation only when a valid rectilinear component survives;
general polygon interpolation is not introduced by this plan.

Decoder environment-compatibility checks must compare ordered polygon cells
and their effective domain/wall/ownership semantics as well as centers,
measures, units/frame and weighted connectivity. Different clipped cells can
share those existing summaries. Do not reuse a fitted model merely because
its array length still matches. The current equal-per-supported-bin default
prior remains explicit: splitting a bin changes that prior. Workflows seeking
uniform physical density should supply an area-weighted prior deliberately.

Raw events and dwell time can be reassigned to the new support. Existing
per-bin arrays and fitted decoder models cannot infer how activity was
distributed inside a split parent. Keep unsupported intervals/bins distinct
from zero activity, and expose the mapping for deliberate downstream work.

## Persistence and validation

Use the Stage A versioned environment payload for all transports.
Encode per-cell WKB as concatenated `uint8` bytes plus integer offsets, keeping
holes and stored cell order. Keep small typed record tables for polygon-cell
and clip-map snapshots only; top-level geometry and each `ClippedLocator`
reference their appropriate records. Assign IDs deterministically during
traversal, decode each record once, and validate reference types and ranges.
This is not a general object-graph serialization framework.

A fresh clip shares its top-level cells/map with its new locator. Subsequent
subset/transform wrappers can refer to different current and historical
snapshots; validate each locator against its own frame and row count, not
against the final environment's arrays. Nested bases retain exactly the data
required for inherited ownership. Spatial-index memos are absent. Readback
must not run a layout builder or redo clipping.

Validate finite coordinates, valid positive-area connected cells, no interior
overlap, representatives, stored area agreement, mapping ranges, locator
ownership and interface/face consistency before publication. Bounds on
floating-point error must use local feature scales; a large coordinate origin
does not permit moving a boundary across a nearby small cell. Existing v2
records retain unknown footprints through the Stage A compatibility path.

The simulated design probes are in [probe_polygon_clipping.py](probe_polygon_clipping.py).
Required regression families include disconnected intersections, concave and
holed cells, thin positive slivers, corner-only contacts, native hexagon shared
topology, repeated clips, whole-bin subsets, all supported transforms, exact
hole rendering, boundary ties, dict/file/NWB readback, corrupt rings/areas/maps,
trajectory re-entry intervals and analytic orthogonal/non-orthogonal fluxes.
Compare direct and repeated clipping by geometry, measure and membership at
interior probes; do not demand cross-history identity or bitwise overlay parity.

Reproduce the design evidence in the existing isolated environment:

```bash
UV_CACHE_DIR=/private/tmp/neurospatial-uv-cache MPLCONFIGDIR=/private/tmp/neurospatial-mpl-cache MPLBACKEND=Agg uv run --no-sync python -W error::RuntimeWarning docs/plans/environment-arena/probe_polygon_clipping.py
```

| Simulated case | Measured result |
| --- | --- |
| One source cell, disconnected retained lobes | Two children, immediate parent mapping `[0, 0]`; component order survives reversed input geometry order. |
| Concave cell and cell with a hole | Centroids lie outside support; deterministic fallback representatives are strictly interior. |
| Boundary contacts and thin cell | Line/point contacts contribute no cell; a positive-area `3.63797880709e-12` sliver survives. |
| Four cells, oblique polygon and hole | Total area `8.32`; independent union/intersection symmetric-difference area `7.65e-16`. |
| Repeated versus direct clip | Maximum per-original-parent area discrepancy `3.77e-16` on the fixed fixture. |
| Two openings around a hole | Orthogonal representative link misses both openings and crosses excluded geometry; alignment alone is insufficient. |
| Native-style hexagonal cells | Independent trigonometric vertices miss 10 of 16 intended interfaces and create ten microscopic overlaps; shared lattice vertices remove both in this fixture. |
| Two clipped grid trapezoids | Scalar operator gives normal flux `0.0620173672946` for `u=y` through a vertical face; analytic normal flux is zero. All three smoothing-kernel positivity/normalization checks still pass. |
| Canonical clipped hexagons | Recentered scalar operator gives false normal flux `0.106240556803`; original anchors give zero in this fixture. |
| Rectangular clip | Areas `(0.75, 1)`, face length `1`, distance `0.875`; normal flux for `u=y` is correctly zero. |
| Prefer-parent-center policy | Sequential clipping yields `(0.2, 0.5)` while direct clipping to the same rectangle yields `(0.2, 0.4)`. Geometry-based centroid selection avoids that history dependence. |

This capture used Python 3.13.5, NumPy 2.3.4, Shapely 2.1.2 and GEOS 3.13.1.
The script prints its versions for subsequent comparisons.

These are simulated design probes, including deliberate invalid numerical
constructions passed to the existing operator. They are not production clipping
tests, a general overlay proof, or evidence that a clipping implementation is
complete. The prototype's fixed-fixture area allowance is not the production
local-feature tolerance contract.

## Phase-4 expansion and completion gate

General clipping was implemented in reviewable parts within Stage A phase 4:

1. Shared cell geometry and persistent boundary ownership, with analytic source
   topology tests before any clipping behavior changes.
2. Global overlay, connected-piece splitting, ordering and parent mapping.
3. Exact polygon lookup, rendering, derivation and all transport round trips.
4. Open-interface adjacency, wall preservation and numerical certification.
5. Linear occupancy and a seeded rates/decoding workflow on the new support;
   record before/after construction and lookup cost and reconcile baseline
   changes.

Each part received independent review. General polygon clipping, including
disconnected pieces and holes, now satisfies the reviewed API, boundary
ownership, transport and numerical contracts. The
[completed validation record](validation.md) includes analytic regression
suites, simulated and recorded-tracking workflows, classified baseline changes
and measured construction/lookup costs. A general-mesh numerical solver remains
outside Stage A.
