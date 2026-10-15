# Broad geometry and coupling design

Implemented Stage C contract, reviewed on 2026-10-14. It expands combining
beyond planar cells while refusing geometry that cannot be certified. The user
chose automatic nearest-neighbor bridges and removal of `CompositeEnvironment`
after migration. Implementation defaults and limits are recorded below.

## Separate three meanings of overlap

1. **Cell overlap:** the same spatial region is counted by two bins. Reject
   positive common measure between components. Comparing centers is insufficient.
2. **Lookup ambiguity:** two child locators, after their arena restrictions,
   claim the same query point. This can occur without overlapping cell interiors:
   shared closed boundaries, snapping around tracks, or unlimited nearest lookup.
3. **Arena overlap:** continuous arenas can overlap even when retained analysis
   bins do not, for example two disjoint subsets of one arena. Arena overlap alone
   is not a reason to reject cell-based components.

The combined lookup must query child locator snapshots with their original arena
restrictions. It must not pick a globally nearest center and invent a new domain.
The baseline probe shows that two unlimited nearest locators both claim a point
between their centers. Two finite-radius locators can do so as well. Unknown
geometry cannot be certified merely because a sample of query points was unique.

## Geometry support matrix

| Geometry | Implemented overlap evidence | Lookup and support limits |
| --- | --- | --- |
| Planar grids, masks, hexagons, meshes, clipped polygons | Exact retained cell polygons and positive-area intersection, including holes and disconnected pieces. | Keep original boundary ownership and arena containment. A polygonal drawing alone does not replace the child locator. |
| Axis-aligned 1D/3D rectilinear cells | Interval/box intersections in all coordinate dimensions. Contact on a boundary has zero length/volume. | Use active cell IDs, not the bounding box of the entire masked grid. |
| Rotated/sheared rectilinear cells in 3D | Retain the inverse affine map and use separating axes for convex parallelotopes in a common frame. | Scale-aware contact checks refuse unresolved positive overlap. AABB intersection alone is not proof of true overlap. Transformed cells above 3D require separated bounds or an axis-aligned representation. Stage E numerical support remains separate. |
| Embedded tracks with stored segments | Use measured-width capsule supports, with segment distance and a conservative numerical-clearance bound. Refuse nearly parallel or unresolved contacts. | Existing lookup snaps to centers. Disjoint centerlines do not establish disjoint lookup support. A measured width supplies a capsule domain; transformed capsules use conservative enclosures. Overlapping support bounds are refused even if centerlines only cross at a point. Active subsets may therefore be conservatively refused when the retained arena is broader than their lookup support. |
| Nearest-only bins with a finite radius | Accepted support is the union of closed balls around centers, intersected with the arena; use analytic ball separation where possible. | A nearest rule establishes query support, not physical cell shapes or face measures. Overlapping component supports require restriction or refusal. |
| Unbounded nearest-only bins and abstract tracks | A supplied exact domain restriction can bound accepted support; otherwise each component may claim all finite space. | Multiple unrestricted nearest leaves cannot satisfy unambiguous combined lookup. Wrapped subsets may partition an unbounded parent domain and must be certified using their actual masks/maps; otherwise refuse with guidance to supply a domain or finite support. |
| Mixed component families | Use a certified common representation for the pair, or prove disjoint support with conservative outer bounds. | An overlapping bound means inconclusive, not proof of overlap. Report that certification is unavailable and how to restrict/transform the inputs. |

All inputs must have matching coordinate dimension, coordinate kind, units/frame
labels and measure dimension. Do not sum lengths and areas in one scalar volume
array. Matching unknown labels is not proof of calibration; the API retains the
caller's declaration. Cartesian and egocentric-polar coordinates cannot be mixed.
Combining polar charts is a separate contract, not an accidental consequence of
both having two columns.

Certification must follow retained subset/transform wrappers. Two complementary
subsets of one unlimited nearest locator can have disjoint unbounded Voronoi
support; a blanket refusal based on their underlying nearest leaves is wrong.
Track lookup follows active linear bins and masks. Support certification may use
the broader retained arena as a conservative enclosure; a subset does not remove
that arena or its coordinate-conversion graph. Such enclosing-support checks can
refuse a disjoint active subset, but cannot silently accept ambiguous support.

Certification follows planar/box checks, then finite-ball or measured-track
supports. Mixed pairs use a supported common representation or separated outer
bounds. Conservative overlap is a refusal to certify, not proof that the actual
restricted supports overlap. The tests include positive, negative, transformed
and boundary cases; they do not establish a general-purpose geometry solver.

## Boundary and uncertainty decisions

Implemented rule: reject positive-measure overlap; on a certified shared boundary,
retain input component order as the final tie-break, after applying each child's
own boundary rules. That preserves the existing first-match convention only on
zero-measure boundaries. Do not extend it to overlapping interiors or overlapping
snap regions.

Use local cell/segment scales and coordinate precision for tolerances. Reject
uncertain cases rather than hide positive overlap in a user-selected large
epsilon. The implementation must distinguish an overlap witness from a failure
to certify disjointness. A bounding-box broad phase is safe for rejection of
candidate pairs from further checking, not for declaring actual intersection.

## Automatic bridges and physical meaning

Retain `auto_bridge=True` and the optional `max_mnn_distance` cutoff. For each
pair of components, find mutually nearest centers using their declared Cartesian
coordinates. Break exact ties by local bin ID. Store endpoint component/local IDs,
combined IDs and straight-line inferred distance. The initial API does not
accept replacement bridge paths: stored lengths are validated against inferred
endpoint separation, and loading never reruns inference.

Auto bridges express navigational adjacency. They do not fill gaps for lookup,
create occupancy exposure or physical cells, remove internal walls, or assert a
measured passage. Automatic bridges crossing known 2D physical walls are blocked.
Unsupported wall-intersection geometry outside 2D refuses with guidance to turn
off automatic bridging.

Regions do not collide silently. The implemented default qualifies child region
names with stable component keys and retain an immutable source mapping. Combined
bin identity likewise uses `(component_key, local_bin_id)`; input display names
are not unique identifiers.

## Numerical edges are not navigation edges

Current code has one ordered bin-pair array. `FaceGeometry` has arrays of that
same edge count, and calculus computes an inverse-distance gradient column for
every entry (`ops/calculus.py:compute_differential_operator`). Filling bridge
face areas with zero stops diffusion but still produces a purported gradient
across a gap. Filling areas with `conductance * distance` invents a physical
interface. Neither is a sufficient representation of the roadmap contract.

Implemented representation: retain owned child binnings with derived bin offsets,
plus separate bridge coupling records for declared conductance. Assemble scalar
diffusion from the block diagonal child conductance matrices, then add explicit
symmetric bridge terms. Existing child face pairs and bin offsets already supply
the necessary identities. No new public or stored face-index map is needed for
the first contract; any internal lookup map is derived. A future face-indexed
calculus API would require separate review.

- Path-only bridges contribute to shortest paths; they contribute nothing to
  diffusion. Child numerical blocks remain independent when valid.
- Explicit conductance adds a symmetric exchange term: flux from i to j is
  `c * (f_i - f_j)`, with the two opposite contributions divided by their cell
  measures. `bridge_conductances` uses the existing diffusion geometric
  convention, with units `length ** (measure_dim - 2)`. It is not a physical
  measure-per-time exchange coefficient. Existing diffusion parameters set the
  numerical time scale.
- A conductance alone does not define a face normal, area, path derivative or
  flux density. Spatial gradient/divergence must either expose only physical
  faces through a reviewed edge-index contract, or refuse combined environments
  with nonphysical bridges. They must not claim bridge flux density exists.
- Missing child face geometry remains missing. If any retained child lacks valid
  numerical geometry, refuse whole-environment smoothing/diffusion; do not
  silently make that child an isolated numerical component. Users can analyze a
  certified subset separately. A bridge conductance cannot certify a child.

The implemented first contract keeps existing `edge_pairs` semantics for
navigation, enables scalar diffusion on certified child blocks and explicit
geometric conductances, and refuses gradient/divergence where the numerical
edge set cannot satisfy their current public contract. A later, separately
reviewed face-index API can expose physical-face calculations without ambiguity.
The existing calculus edge contract is unchanged; nonphysical bridges cause an
explicit refusal.

## Storage, derivation and capability retention

Store canonical child data and composition identity, not nested live environments.
Add versioned built-in composition locator/arena/data codecs, validate recursion,
child offsets, counts, dimensions and bridge endpoints, and bound malformed
recursive input. Serialized bridge results must survive loading without inference.
Avoid storing the same authoritative geometry twice without consistency checks.

A combined arena retains child domains and physical-wall evidence. It must not
turn every component outline into a physical wall or claim a complete footprint
when a component's footprint is unknown. Lookup uses each child's actual domain.
Exact planar plotting can concatenate retained cells; 3D drawing follows existing
3D capabilities. A combined object does not pretend to be one regular grid or
one global track coordinate. Keep per-component linearization accessible through
the source mapping, with any global conversion API separately designed.

Subset drops incident bridges and composes the source mapping. A supported affine
transform applies consistently to all components, regions and stored domains;
path weights obey Stage B's similarity restrictions. Explicit geometric
conductances scale by
`scale ** (measure_dim - 2)` under similarities. Non-similarity transforms of
positive couplings, or scaling that underflows/overflows, refuse. Clipping
is supported only where every retained component supplies the exact required cell
geometry; split-child mapping must compose with the component source mapping.
Rebinning remains Stage D unless an existing operation can prove preservation.
The first implementation refuses clipping whenever a retained bridge remains;
clip components first or combine with `auto_bridge=False`. It does not invent
connections to split child bins.

## Recorded decisions

- Shared-boundary ties use input component order, after child lookup. Unknown or
  numerically uncertain support refuses rather than using sampled evidence.
- Known physical walls block inferred bridges. No bridge creates a corridor,
  exposure in a gap, or a physical face.
- The result is an ordinary owned `Environment`, with stable keyed sources and
  qualified region names. These are implementation defaults from the reviewed
  design; they do not require unique display names.
- Geometric conductance enables symmetric scalar coupling, with physical
  gradient/divergence refusal where bridge faces are absent. Missing child
  geometry still blocks numerical operations for retained bins.
- The user explicitly chose immediate removal of `CompositeEnvironment` after
  migration, rather than a forwarding constructor.
