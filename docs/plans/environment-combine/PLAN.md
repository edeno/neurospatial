# Stage C — Combine owned environments

**Status:** C0 started: baseline captured and broader geometry design drafted.
No production implementation has changed. The user requested the broader design
before implementation on 2026-10-14 and approved retaining automatic
nearest-neighbor bridges. Sections marked proposed need review before coding.

**Branch:** `feature/environment-combine`, stacked on
`feature/environment-immutable` at `bab36f5f`.

**PR stack:** [representation #67](https://github.com/edeno/neurospatial/pull/67)
→ [arena #68](https://github.com/edeno/neurospatial/pull/68)
→ [immutable values #70](https://github.com/edeno/neurospatial/pull/70).
There is no Stage C PR; this turn authorizes opening the completed Stage B PR.

## Outcome and scope

`Environment.combine` returns an ordinary owned `Environment`. Scientists can
query, plot, save and analyze it using the same API as other environments.
Component bins retain their geometry and acquire deterministic combined IDs.
Combining does not resample rates, concatenate time series, transfer fitted
models, or imply that two recordings share a coordinate system.

The representation roadmap already requires rejection of overlapping bins and
path-only bridges unless conductance is supplied. The user additionally chose:

- Keep automatic mutual-nearest-neighbor bridges, rather than require explicit
  endpoints for every connection.
- Design support for 3D cells, embedded tracks and nearest-only bins before
  implementing a planar-only subset.

Read [geometry-and-coupling.md](geometry-and-coupling.md) for that expanded
contract and the decisions still needing agreement. A geometry refusal is an
explicit part of the contract, not silent loss of a component's capabilities.

## Proposed user workflow

1. Construct components from positions, calibrated masks, grids, tracks or
   explicit bins, as today. Supply consistent units/frame declarations. Align
   coordinates with an actual transform before combining if necessary.
2. Give each component a stable key in an ordered mapping. This avoids using
   display names as identity; duplicate or empty environment names remain legal.
3. Call the proposed `Environment.combine(components, *, auto_bridge=True,
   max_mnn_distance=None, name="", ...)`. Additional bridge-conductance and
   overlap-certificate arguments are not fixed until the design is agreed.
4. Inspect the returned component-to-bin mapping and inferred bridge table.
   `auto_bridge=False` leaves components disconnected; a maximum distance limits
   the automatically proposed connections. Automatic bridges describe graph
   navigation, not a measured corridor or permission to cross a physical wall.
5. Use the normal environment APIs. Supported numerical methods operate only on
   certified geometry/couplings. Save with `to_file` or NWB; readers never rerun
   bridge inference or rebuild a component layout.

The proposed result owns immutable component keys, bin offsets and bridge
records. Input order fixes IDs. Child internal edge order is concatenated first;
bridges follow in a deterministic order. `edge_pairs` remains authoritative.
No rates or models are copied automatically.

## Implementation phases

### C0 — Behavior capture and design (current)

- Record the Stage B environment baseline and current composite tests.
- Capture deterministic legacy overlap, units, lookup, bridges and API behavior.
- Specify broad geometry support, ambiguity handling, physical walls, path
  lengths versus conductance, persistence and derivation rules.
- Review the design independently; resolve major decisions before code changes.

### C1 — Owned component identity and lookup

- Add the smallest immutable composition data needed to retain child bin IDs,
  child locators, arena restrictions and their offsets. Never retain live
  `Environment` callbacks or child analysis caches.
- Preserve child lookup exactly, including nonfinite sentinels, holes, masks,
  thresholds and outside values. Reject overlap/ambiguity according to the
  reviewed certification policy; do not substitute global nearest-center lookup.
- Add dedicated built-in codec records and validation for composition. Custom
  adapters still normalize to supported built-ins; this is not a second plugin
  registry. Validate all child counts, offsets and shared geometry on load.
- Prove copies, metadata/region replacements, pickle and dict/file/NWB reads
  retain ownership. Exercise the existing public contract kit.

### C2 — Geometry certification and automatic bridges

- Implement reviewed overlap checks for planar cells, 3D boxes, tracks and
  bounded nearest supports, with explicit refusals for uncertified cases.
- Preserve automatic MNN inference and its distance cutoff. Specify ties by
  component order and local bin ID, with a regression for symmetric inputs.
- Infer only after validating component units, frames, dimensions, measures and
  overlap. Publish immutable bridge records so inference runs only once.
- Bridge inference does not create cell volume, a physical corridor or a
  finite-volume face. Internal walls remain part of each child's arena.

### C3 — Numerical coupling and ordinary environment integration

- Keep navigation edges separate from certified numerical couplings. Implement
  the reviewed bridge conductance contract; test mass conservation, constants,
  symmetry/detailed balance and absence of diffusion across path-only bridges.
- Preserve child faces and missing-face reasons. Never manufacture face geometry
  for a component with only centers and scalar sizes.
- Define exact plotting and linear time allocation from retained component
  geometry. Unsupported interpolation/rebinning/track-linearization operations
  must refuse with an actionable explanation.
- Verify subset, transform, clip and storage keep component identity and bridge
  semantics, including removal of bridge endpoints.

### C4 — Migrate and retire CompositeEnvironment

- Replace existing callers and examples with the agreed combine workflow.
- Migrate exports, typing, API snapshots and persistence guidance together.
  Decide immediate removal versus a temporary forwarding factory before coding;
  do not carry a second independent environment implementation forward.
- Run independent review, scientific regressions, the environment baseline,
  composite/consumer suites, doctests, executable guides, Ruff and mypy.
- Commit logical phases. Opening or publishing a Stage C PR requires a later
  instruction; the current request only opens the completed Stage B PR.

## Evidence and acceptance

See [validation.md](validation.md) for the initial evidence. Completion requires:

- No unapproved changes to existing native environments.
- Independently specified boundary oracles, component offsets and bridge IDs.
- Overlap witnesses with distinct centers, 3D face contact versus volume overlap,
  crossing/collinear tracks, finite and unbounded nearest lookup, and missing
  geometry. Samples alone cannot prove disjointness.
- MNN tie/cutoff cases, disconnected graphs, units/frame mismatch and conflicting
  region names. Inferred bridges never silently enable numerical transport.
- Geometry/ownership/storage checks after subset, transform and clipping.
- Preserved wall meaning: joining analysis components must not erase a physical
  barrier, while an analysis cut must not invent one.
