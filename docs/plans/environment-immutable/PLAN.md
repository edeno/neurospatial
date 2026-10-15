# Stage B — Immutable environment values

**Status:** B0–B4 complete and independently reviewed. Validation results and
remaining execution limits are recorded in [validation.md](validation.md).
The user approved immediate read-only metadata, mutable standalone region
builders with immutable environment snapshots, and migration of custom
providers to immutable built-in snapshots. No Stage B PR publication is
authorized.

**Branch:** `feature/environment-immutable`, stacked on
`feature/environment-arena` at `a88360e5`.

**PR stack:** [representation #67](https://github.com/edeno/neurospatial/pull/67)
→ [arena #68](https://github.com/edeno/neurospatial/pull/68) → Stage B.

This implements Stage B of the [representation roadmap](../environment-representation/overview.md#roadmap-after-phase-4).
Stage A is [complete and validated](../environment-arena/validation.md).
Combining environments, physical-wall consumer migrations, general affine
numerics and removal of the layout bridge remain stages C–F.

## Scientific outcome

An environment shared by an encoder, decoder or analysis cannot change because
another caller edits its units, coordinate frame, named regions or input data.
Changes produce a new environment. Existing immutable binning and arena data
can be shared; mutable caches remain private to each environment.

Immutability does not imply numerical equality or a new cache-key scheme.
Preserve environment identity behavior and existing name/string comparison;
do not introduce structural equality, structural hashing or global caches.

## Captured behavior before Stage B

Small deterministic probes on the Stage A tree confirmed:

- `Region(name, "point", array)` shares the supplied NumPy buffer. Changing
  the caller's array changes the region coordinates.
- The environment constructor retains its supplied `Regions` container by
  reference. The collection is mutable.
- `env.units` and `env.frame` can be assigned after construction.
- Assigning `env.units = "m"` changes the label while leaving bin coordinates
  unchanged. It is not coordinate conversion.

Relevant code is [Region construction](../../../src/neurospatial/regions/core.py),
[environment publication and metadata](../../../src/neurospatial/environment/core.py)
and the [factories](../../../src/neurospatial/environment/factories.py).
The original `layout_parameters` property returned its live dictionary;
provenance and the retained layout bridge are now owned. Factories consistently
accept units/frame at construction.

## Proposed contracts and decisions

The mutation policy, region collection model and custom-provider snapshot
requirement below are approved. The replacement methods, B3 public binning
records and B4 installed adapter contract have received independent review.

| Decision | Proposed default | Alternative or constraint |
| --- | --- | --- |
| Mutation compatibility | **Approved:** reject assignment to published metadata immediately; migrate internal callers, examples and tests together. | No deprecation period. |
| Metadata replacement | Set units/frame/name at construction and add `with_metadata(*, name=UNSET, units=UNSET, frame=UNSET)`. Omitted arguments preserve values; explicit `None` clears units/frame; name stays a string. | Independently reviewed contract. Replacements preserve concrete type and geometry without republication, with fresh environment-owned analysis caches. |
| Unit meaning | Units and frame describe the supplied coordinates. Metadata replacement corrects declarations only and never scales coordinates. | Automatic coordinate conversion is outside Stage B. A future conversion API must transform all geometry, measures, walls and regions consistently; pixels require calibration. |
| Region collection | **Approved:** keep standalone `Regions` usable as a mutable builder; an environment owns an immutable snapshot. | Standalone editing remains available; environment snapshots reject mutation and own their data. |
| Region replacement | `with_regions(regions)` replaces the whole collection and returns a new environment. | Merging or updating names must be explicit; do not silently combine old and new regions. |
| Name and provenance | Include the public name in immutable metadata. Own build provenance and expose immutable data or detached copies. | Keep provenance descriptive; it cannot become a recipe for rebuilding geometry. |
| Custom scientific providers | **Approved:** require owned immutable snapshots for custom layouts, track converters and arenas, even when older providers need migration. | Live delegation and generic deepcopy are not proofs of immutable scientific state. Define supported data/codec contracts before admitting new provider roles. |

The region choice affects the existing `.add`, `.set`, `.update_region`, mapping
assignment and deletion APIs. Specify the returned collection's public type and
read helpers before implementation. An immutable snapshot must preserve ordered
iteration and cannot contain caller-owned writable coordinate buffers or mutable
nested metadata. Enforce or refine the documented JSON metadata contract with
explicit regression cases; do not quietly accept mutable arbitrary objects.

The snapshot remains a `Regions` instance, preserving existing read helpers and
ordered iteration. `Regions(env.regions.values())` creates a mutable editing
builder. Sharing immutable geometry also shares geometry-owned private lookup
memos (KD-trees/Delaunay); replacement methods refresh environment-owned
analysis caches, not the geometry or those memos.

## Implementation phases

Every phase captures relevant behavior before changing it, receives independent
review, repairs confirmed findings and ends in a logical commit. The full stage
is not complete when only the metadata slice is complete.

### B0 — Capture and API agreement

- Record the existing environment baseline on this stack. The original Stage A
  capture remains historical evidence; do not overwrite it or use an older
  checkout's metadata behavior as the new baseline.
- Add focused ownership probes for point regions, nested metadata, input region
  collections, published provenance, graph attributes and retained layouts.
- Capture shallow/deep copying, pickle and dict/file/NWB round trips, including
  Cartesian and polar environments, derived environments and legacy records.
- Resolve the decisions above and review example workflows before code changes.

### B1 — Owned values and construction-time metadata

- Fix confirmed region buffer ownership and restoration defects with failing
  regression tests first. Preserve read-only state through deepcopy and pickle.
- Define one immutable authority for public metadata. Audit name, units, frame,
  provenance and public geometry views; private caches can still populate lazily.
- Protect public cached results as well as input buffers. Differential and
  divergence operators, smoothing kernels, boundary-bin masks and bin/edge
  attribute results must be immutable values or detached exports. Caller edits
  and rebinding cached-property names must not poison later calculations. Review
  sparse-matrix ownership explicitly; freezing one `.data` array is insufficient
  if callers can replace buffers on the shared cached matrix.
- Audit the retained `env.layout` bridge and delegated custom providers. Later
  mutation of a builder or provider must not change an existing environment's
  scientific behavior. Review snapshot/provider contracts and compatibility
  before claiming this guarantee; generic deepcopy of opaque providers is not
  proof of immutability. Keep the bridge in Stage B; its removal remains Stage F.
- Accept keyword-only units/frame consistently across factories and construct
  metadata before publication. Preserve the existing advisory units registry.
- Migrate factory, copy, subset, clip, rebin, transform and loading paths away
  from post-construction assignments. Metadata-only replacement shares supported
  immutable geometry and creates fresh private caches.
- Preserve existing numerical geometry, lookup, boundary ownership and
  derivation behavior. A declaration correction is not a geometric transform.

Primary touchpoints: `environment/core.py`, `factories.py`, `_protocols.py`,
`_clipping.py`, `transforms.py`, `ops/transforms.py`, `regions/core.py`,
`io/files.py`, `io/nwb/_environment.py`, constructors and executable examples.

### B2 — Immutable environment regions

- Implement the approved collection model and `with_regions` replacement.
  Later edits to a builder must not affect an existing environment.
- Preserve region query helpers and validate coordinate dimension at
  publication. Regions may be outside or partly outside the arena.
- Retain the current derivation policy unless separately approved: copies
  retain regions, transforms map them, and subset/clip/rebin drop them.
- Update docs and error messages so users can replace a region collection
  without modifying an environment held by another analysis.
- Verify that saved/loaded environments and decoder-held references preserve
  the same immutable public state. Existing caches remain environment-owned.

### B3 — Explicit bin construction and editable graph export

Implemented after the B1/B2 contract review:

- `Environment.from_bins(...)` uses one authoritative data representation and
  the existing publication/validation path. Review the public binning types and
  exact signature before exposing private implementation types.
- Lookup must be explicit. Diffusion needs valid faces; clipping needs actual
  cells. Centers, scalar sizes or nearest-center lookup cannot establish those
  capabilities or infer physical walls.
- `to_networkx()` returns a detached editable graph, including independently
  owned nested attributes, with preserved bin identity and weighted edges.
- Resolve the current facade's NetworkX edge-iteration-order restriction.
  Arbitrary pair-array order is not always representable as graph iteration
  order. Keep authoritative pair/face alignment; decide how graph equivalence
  is validated without silently changing stored identities or numerical arrays.
- Start with supported data-backed components. General independent polygon
  cells need a real locator/validation contract, not fabricated clip ancestry.

### B4 — Third-party codecs and public contract tests

- Define supported extension roles and their validation obligations before
  adding a versioned entry-point registry. Registration alone is insufficient:
  the facade and codec reader currently hard-code supported locator types.
- Keep built-in codec identifiers and payloads compatible. Reject duplicate,
  unknown or unsupported versions deterministically with actionable errors.
- State the installed-package requirement: a custom record can load only when
  the required codec and dependencies are available. Loading never runs a layout
  builder or restores a runtime cache as scientific geometry.
- Publish lightweight contract tests for ownership, lookup dimensions,
  nonfinite inputs, sentinels, caller-supplied boundary oracles, capability
  refusals, copy/pickle behavior and codec round trips/missing codecs.
- Exercise the kit on built-ins and a minimal external-style component. A codec
  cannot make invalid numerical geometry eligible for smoothing or diffusion.

## Compatibility and validation

Use the existing isolated environment through `uv run --no-sync`. No additional
runtime dependency is proposed. Review public signatures and update the API
snapshot only for approved changes.

Required checks include:

- Seeded samples, calibrated masks, tracks and polar environments through
  construction, derivation and persistence. Positions/masks remain direct inputs.
- Caller mutation, nested metadata mutation, constructor aliasing, shallow/deep
  copies and pickle. Graph exports and exported metadata must be independent.
- Cache-poisoning probes through every public operator/kernel/attribute export,
  cached-property rebinding, and retained layout/custom-provider references.
- Identity/boundary-preserving bin lookup, occupancy and rate/decoding workflows;
  new metadata error behavior is classified separately from numerical parity.
- Actual dict/file/NWB round trips and supported legacy reads. Do not change the
  schema merely to enforce in-memory ownership; revise it only if the approved
  stored representation requires it.
- Consumer suites, executable docs, doctests, public API and argument contracts,
  Ruff, source mypy and typing contracts. Record unrun GUI/GPU checks honestly.
- Benchmark metadata replacement on large binnings to verify that it does not
  rebuild or duplicate all geometry. Report measurements without claiming an
  unmeasured improvement.

## Relationship to other work

The [continuous-behavior environment plan](../continuous-behavior/phase-2-environment-contracts.md)
depends on Stage B for environment-valued immutable snapshots and their
identity-based caches. Stable bin IDs, correspondences, feature alignment and
time-indexed environmental states remain that plan's separate slices. This
stage does not add another environment wrapper or timeline system.

Stage A's measured lookup overhead is a separate performance follow-up. Capture
and preserve its current geometry behavior; do not mix a lookup redesign into
metadata immutability.

## Approved B3/B4 details

- `Environment.from_bins(binning, *, name="", units=None, frame=None,
  regions=None, arena=None)` accepts a `Binning` or an installed provider adapter.
  Explicit construction is Cartesian; existing polar factories remain supported.
- Supplied neighbor order and orientation are preserved. `env.edge_pairs` is
  authoritative for edge arrays; NetworkX graph iteration order may differ.
- Hex locators restore conceptual grid shape and the exact active mask, including
  saved environments. This intentionally fixes the previous loss of hexagonal
  spatial-autocorrelation support after loading.
- Transforms preserve declared non-Euclidean path weights for rigid maps and
  multiply them for uniform scaling. Shear and unequal scaling refuse when only
  scalar path lengths are known. Native Euclidean computations retain their
  existing arithmetic and rounding.
- Installed entry-point objects in `neurospatial.snapshot_adapters.v1` provide
  versioned adapters for locator, track, arena and whole-binning data. The first
  contract normalizes to existing built-in snapshots; it does not add arbitrary
  runtime lookup algorithms. Original custom payloads require the adapter;
  canonical resaves do not.
- `assert_environment_contract` checks independent caller-supplied lookup
  answers, ownership, copy/pickle/storage and missing-face refusals without a
  pytest dependency. Dedicated plugin tests cover missing/conflicting codecs.
