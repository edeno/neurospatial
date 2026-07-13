# Public API Curation Plan

**Status:** Ready for staged execution

**Created:** 2026-07-12

**Release stance:** Version-agnostic. This plan defines readiness gates, not a
release number or date.

## Execution Entry Point

This file defines intent and policy. [TASKS.md](TASKS.md) is the authoritative
agent runbook: it defines exact artifacts, ordering, safe defaults, validation,
progress recording, and the few changes that require a maintainer decision.

A coding agent starts at task A.0, then executes the first unchecked, unblocked
task in document order. It works in independently reviewable increments and
records commands and results in
`artifacts/PROGRESS.md`, and does not infer that completing an inventory
authorizes changing the runtime API. Additive inventory, documentation, tests,
and tooling can proceed autonomously. Removing or hard-deprecating a public
path, changing scientific semantics/defaults, adding a required dependency, or
introducing a material new facade requires an accepted entry in
`artifacts/DECISIONS.md`.

No user study is required by this plan. Namespace comparison uses documented
researcher journeys, maintainer walkthroughs, and objective runtime, static-tool,
documentation-search, and import-cost checks.

## Objective

Make neurospatial's public surface small enough to learn, stable enough to cite,
and broad enough to support real spatial-neuroscience workflows without hiding
advanced capabilities.

The curated API must serve both single-neuron analyses and modern systems
neuroscience. Hundreds of simultaneously recorded neurons are routine now, and
low-thousands populations are a realistic operating target. Population-scale
entry points, unit identity, memory behavior, and summary/chunked alternatives
are therefore API concerns, not later performance details.

## Current Baseline

The initial audit found:

- 405 exports across the top level and twelve major subpackage roots.
- 70 `__all__` declarations across the source tree.
- 28 top-level exports, including lazy namespaces and convenience re-exports.
- Seven duplicate names across the audited major roots. Duplicate re-exports
  are a smaller problem than overall breadth and task discoverability.
- API pages are generated for every source module, including private modules.
- The `detect_region_crossings` compatibility shim still advertises removal in
  0.7 in both its docstring and warning even though the package has moved beyond
  that version. The false promise should be corrected immediately while the
  compatibility path remains in place; the wider deprecation audit follows.
- Cross-domain conventions are not yet uniform: public stochastic APIs use both
  `rng` and `seed`, result tables use identifiers such as `unit_id`, `neuron`,
  and domain-specific coordinates, and singular/plural families are not governed
  by one documented design grammar.
- Five prominent top-level exports—`SpikeTrains`, `Session`, `load_session`,
  `BayesianDecoder`, and `restrict`—are runtime-discoverable through PEP 562 but
  currently resolve to `Any` under mypy. Runtime laziness and static typing must
  work together.
- The current Pynapple adapter is an array-extraction surface: `TsGroup` becomes
  spike arrays plus unit IDs, `Tsd` / `TsdFrame` becomes times plus values, and
  `IntervalSet` becomes start/end arrays. It does not currently claim to preserve
  Pynapple metadata or time support, so its loss behavior must be explicit.
- Population xarray conversion already demonstrates a strong target pattern:
  real `unit_id` labels, named dimensions, bin-center coordinates, and occupancy
  as a data variable rather than unstructured metadata. Other result families
  need to be checked against the same vocabulary and preservation rules.

These counts are a reproducible starting point, not reduction quotas. Curation
is judged by task clarity and contract quality, not by minimizing a number.

## Success Criteria

The work is complete when:

1. Conventional Python exports expose the recommended runtime surface, while one
   curated repository metadata file classifies every supported symbol and path,
   including compatibility paths that intentionally remain importable without
   appearing in `__all__`, as **stable**, **advanced**, or **experimental**. It
   also records internal modules that must be excluded from public documentation.
2. One approved API design grammar defines the shared vocabulary, function-family
   patterns, data-semantics/lazy-source profiles, dimension names, temporal and
   segment conventions, copy/mutation rules, result profiles, and randomness
   conventions used across public domains.
3. Every stable symbol has one canonical import path, one anchored reference
   entry, an executable example or guide reference, an owning domain, and
   contract coverage. The project maintainer is accountable by default; an
   owner override is recorded only when responsibility differs.
4. Stable contracts cover more than signatures: shapes, axis roles/order,
   coordinate representation/reference frame, units, ID/index and segment
   semantics, result fields, missing-data behavior, warnings/errors, identity
   propagation, serialization schemas, determinism, copy/view behavior,
   partial-read/cache/materialization behavior, dtype/range/device coercion, and
   optional-dependency behavior are recorded where relevant.
5. Pynapple, NWB, pandas, and xarray boundaries declare what identity/index and
   segment structure, metadata, time support, coordinate representation/frame,
   units, dtype/scaling, partial-read/cache behavior, and resource ownership they
   preserve; any lossy conversion is explicit and tested.
6. Population workflows expose batch-first entry points, preserve `unit_ids`
   separately from positional indices and retain segment/order semantics, and
   document computational and memory scaling. When full outputs can become
   impractical, an existing bounded-memory or summary alternative is documented
   and tested; if none exists, the practical limit is explicit and the missing
   implementation has a dispositioned gap-report entry.
7. The main API navigation contains stable and advanced public APIs only.
   Experimental APIs are visibly labeled; internal modules are absent.
8. Duplicate re-exports are either intentional compatibility aliases or have a
   documented migration path.
9. CI fails when the curated metadata, `__all__`, lazy export mappings, static
   declarations, stable contract snapshot, generated docs, runtime imports,
   behavioral conformance tests, or supported interoperability matrix disagree.
10. Representative neuroscience tasks each map to one canonical entry point on
    a task-oriented API page, and namespace choices are based on documented
    discovery checks rather than a predetermined flat or nested style.
11. Cleanup occurs only when the compatibility policy says its declared
    conditions have been satisfied. Soft deprecation does not imply removal,
    and this plan does not tie cleanup to a named release.

## Scope

### In scope

- Top-level exports and every subpackage `__all__`.
- Canonical import paths, names, calling conventions, and result types.
- Stable, advanced, experimental, and internal support semantics.
- Scientific compatibility dimensions visible to callers.
- Cross-domain naming, signature, result, dimension, randomness, mutation, and
  ownership conventions.
- API-reference generation, navigation, and task discovery.
- Compatibility policy, existing deprecation debt, and migration records.
- Import-time behavior caused by public re-exports.
- Eager/lazy materialization and resource-lifetime behavior at I/O and
  interoperability boundaries.
- Supported-version matrices and loss guarantees for Pynapple, NWB, pandas, and
  xarray integrations.
- Population-scale API behavior, including batch, summary, streaming/chunking,
  identity, and serialization contracts.
- A gap report identifying workflows that may deserve separate design plans.

### Out of scope

- Rewriting algorithms or changing scientific defaults without a separate
  scientific review.
- Adding new workflow facades, session QC systems, or configuration frameworks
  as part of curation. This plan may identify those needs but does not design or
  implement them.
- Adopting Pynapple as a required internal data model or making it a core
  dependency.
- Adopting Astropy quantities/coordinates, MNE containers, SpikeInterface
  extractors, NiBabel proxies, Dask collections, or any new universal lazy-array
  abstraction solely because they are useful design precedents. A material new
  data model or dependency requires a separate design decision.
- Promising universal Array API, GPU, Dask, or distributed support where the
  numerical kernels and scientific semantics have not been reviewed and tested.
- Turning `Session` into a stateful analysis god-object.
- Removing useful advanced capabilities solely to reduce the symbol count.
- Moving implementation files merely to make the source tree look smaller.
- Scheduling or conditioning any named release.
- Removing an existing public path before the agreed compatibility conditions
  are satisfied.

## Design Principles

1. **Task first.** Stable entry points correspond to researcher goals such as
   computing population spatial rates, decoding a session, detecting trials, or
   loading NWB data.
2. **Consistency compounds.** A researcher who learns one domain's vocabulary,
   array orientation, option placement, and result behavior should be able to
   predict the next domain's API. Deliberate scientific differences are named,
   not hidden behind superficial uniformity.
3. **Batch first, singular convenient.** Population functions are first-class
   APIs rather than loops left to users. Singular wrappers remain useful for
   exploration but must agree with the batch path.
4. **One canonical home.** Compatibility aliases may remain, but documentation
   teaches exactly one import path.
5. **Array-first core with explicit coercion.** Core numerical functions continue
   accepting arrays and specifically documented array-like protocols.
   Conversion, copying, dtype changes, and device transfers are observable
   contract behavior; “array-like” is not a blanket backend promise.
6. **Identity is never incidental.** Population inputs and outputs distinguish
   stable identifiers from positional indices and preserve `unit_ids`, segment
   identity, ordering, and aligned unit metadata through filtering, indexing,
   serialization, xarray conversion, and NWB round trips.
7. **Interoperability is loss-aware.** Adapters preserve scientific meaning or
   state exactly what they extract or discard. A convenience conversion is not
   described as a round trip unless identity, metadata, time support, units, and
   schema survive it.
8. **Scale is explicit.** Public docs state relevant complexity and memory
   shape. Intrinsically expensive methods identify practical limits and point to
   existing blocked, approximate, or summary alternatives; if none exists, they
   link the dispositioned gap rather than implying that curation implements one.
9. **Result behavior follows profiles.** Primary computations return stable,
   inspectable results. Tensor, table, fitted-model, scalar/report, and streaming
   results implement only the conversions and terminal operations meaningful for
   their profile; no universal mixin is imposed for symmetry alone.
10. **Lazy behavior is observable.** Lazy imports, deferred data reads, chunked
    results, and file-backed objects document available structure before loading,
    partial-read behavior, compute triggers, caching, dtype/scaling changes,
    resource ownership, and whether materialization copies data.
11. **Advanced is still public.** Graph operators, basis methods, visibility,
   simulation, animation, and other building blocks remain documented and
   searchable without competing with common tasks in beginner navigation.
12. **Evidence before removal.** Repository usage, documentation, tests, release
   history, downstream compatibility evidence, and scientific maintainer review
   are considered before deprecation.
13. **Conventional mechanisms first.** Recommended exports are expressed with
   standard Python mechanisms (`__all__`, normal imports, and PEP 562 lazy
   exports), not a package-private runtime policy framework. Supported documented
   and compatibility paths are also part of the compatibility baseline even when
   they are intentionally omitted from wildcard exports. Support policy lives in
   repository metadata and is checked against all of those mechanisms.
14. **Separate boundaries from scientific logic.** I/O and interoperability
   adapters normalize external objects into documented scientific types before
   computation. Friendly entry points may validate and normalize inputs, while
   numerical kernels remain strict enough that coercion and scientific choices
   are not silently guessed.
15. **Prefer composable, familiar pieces.** Functions and standard scientific
   types are the default. A public custom class, workflow object, or shared
   abstraction must earn its surface by preserving invariants, identity,
   reusable state, or resource ownership that simpler pieces cannot express.
   Task-oriented entry points remain thin orchestration over reusable lower-level
   APIs rather than duplicating algorithms.

## Sources of Truth

The curation work will use these artifacts with non-overlapping authority:

1. Package and subpackage `__all__` declarations, together with explicitly
   declared PEP 562 lazy-export mappings, are authoritative for the recommended
   exported surface. Documented public imports and supported compatibility paths
   remain part of the supported surface until reviewed, even when they are
   intentionally absent from `__all__`. Static declarations or stubs must expose
   the recommended surface and any compatibility paths they claim to support.
2. `docs/api/public_api_metadata.json` is the manually reviewed policy layer for
   facts that cannot be safely derived: canonical paths, tiers, owning domains,
   optional owner overrides, intentional aliases, lifecycle state, applicable
   grammar/result/contract profiles, optional extras, interoperability claims,
   and deprecation conditions. It is validated by
   `docs/api/public_api_metadata.schema.json` and does not duplicate generated
   signatures or object kinds.
3. `docs/api/conventions.md` is authoritative for the cross-symbol API design
   grammar: vocabulary, function families, dimensions, temporal semantics,
   mutability/ownership, result profiles, randomness, and interoperability
   conventions. The curated metadata links each symbol to a grammar profile or
   records a reviewed exception.
4. The generated API inventory records discoverable structure and evidence from
   the source tree, runtime exports, documentation, and tests. It is an audit
   product, not a second policy database.
5. A generated contract snapshot records inspectable stable details such as
   signatures plus explicitly declared result/schema contracts. It is reviewed
   like an API change, not treated as an opaque golden file.
6. API documentation and navigation are generated from the curated metadata,
   discovered exports, design grammar, and source docstrings.

These sources are distinct on purpose: Python exports define the recommended
runtime surface, existing documented and compatibility imports establish the
conservative support baseline, the curated metadata records support policy, the
design grammar records library-wide consistency, the inventory reports discovered
facts, and the snapshot records selected observable behavior.

## Support Tiers

| Tier | Meaning | Compatibility rule |
| --- | --- | --- |
| Stable | Recommended researcher-facing entry point with demonstrated value, complete documentation, contract coverage, and characterized applicable scale behavior | Breaking changes require a declared migration path and completion of the project's compatibility process. |
| Advanced | Supported building block for custom analyses, with documented intended users and practical limits | Changes are announced and documented; any narrower guarantee is explicit per symbol rather than inferred from the tier name. |
| Experimental | Interface whose value, shape, performance, or coverage is still being evaluated | May change with prominent documentation and release notes; every symbol has an owning domain, evaluation plan, and promotion/continuation/removal criterion. The project maintainer is accountable unless an override is recorded. |
| Internal | Implementation detail, backend kernel, or private helper | Omitted from public navigation; no compatibility guarantee. |

An API is not made experimental merely because it is specialized. Mature
specialized analyses belong in the advanced tier. Experimental labeling reflects
interface uncertainty.

Experimental status is expressed through curated metadata and a documentation
badge. Existing APIs are not moved into a new namespace solely to display the
status, because that would itself create migration work.

Stable promotion is evidence-based rather than release-based. It requires a
canonical task, supported contract profile, complete reference and executable
example, owning domain, relevant interoperability coverage, and scale
characterization where data size or population size materially affects the
contract. A specialized API can satisfy these criteria without being common
enough for beginner navigation.

## Compatibility Dimensions

The inventory records applicable observable behavior for each stable API:

| Dimension | Examples |
| --- | --- |
| Import and naming | Canonical path, intentional aliases, exception classes |
| Calling convention | Parameter names, positional/keyword status, defaults, accepted protocols, return-kind stability |
| Array contract | Shapes, axis roles/order, dtype and value-range policy, unit-major/time-major/bin-major conventions |
| Scientific semantics | Coordinate representation and reference frame, units, identifier/index and segment semantics, angular conventions, interval closure, normalization |
| Missing/invalid data | NaN/Inf handling, dropped samples/spikes, empty inputs, masking behavior |
| Result schema | Result class, fields, dimensions, identity labels, summary keys |
| Mutability and ownership | Copy/view rules, in-place effects, read-only claims, parent/child mutation, file-handle ownership |
| Execution and coercion | Eager/lazy evaluation, partial reads, caching, materialization, dtype/scaling changes, memory order, host/device transfer, accepted array protocols |
| Diagnostics | Warning/error categories, convergence flags, exclusions, fallback behavior |
| Reproducibility | RNG inputs, determinism, resolved method parameters, provenance fields |
| Interoperability | DataFrame/xarray schemas, NWB names, serialization compatibility |
| Optional dependencies | Minimal-install import behavior and actionable missing-extra errors |
| Scale | Time/memory complexity, full-output size, chunking/summary behavior |

Not every dimension applies to every symbol. The curated metadata identifies the
contract profile, and the human-readable inventory explains omissions.

## API Design Grammar

Runtime exports and curated policy metadata answer “what is public?” The design
grammar answers “how should a public API feel?” It is reviewed before
symbol-by-symbol cleanup so that curation does not replace local inconsistencies
with a different set of local decisions.

The grammar defines:

- A canonical vocabulary for recurring concepts such as environments,
  timestamps, positions, spikes, units, epochs, bins, events, trials, and random
  number generators. Existing alternatives are inventoried before one term is
  selected.
- Function-family templates for compute, fit, predict/decode, detect, transform,
  convert, load/read, save/write, summarize, and plot operations. Scientific
  operations are generalized across dimensionality or singular/batch inputs only
  when their semantics and validation genuinely agree.
- Signature rules: primary data may remain positional where readable; optional
  controls are keyword-only; the same concept uses the same parameter name and
  interpretation across domains. Boolean flags do not change the fundamental
  return type when an explicit conversion method or separate function is
  clearer. Scientifically consequential choices are required or have a visible,
  justified default; mode switches and interacting parameters are minimized.
- Forwarding rules: public scientific entry points do not hide controls behind
  undocumented `**kwargs`. Deliberate forwarding at plotting or adapter
  boundaries names the target API, documents collision/precedence behavior, and
  remains separately testable.
- Boundary rules: load/read and adapter functions handle external formats and
  return documented standard or neurospatial types; scientific kernels do not
  open files or depend on optional container implementations. Friendly
  normalization is separated from strict computation so accepted coercions are
  explicit.
- Structural-interface rules: use duck typing or `typing.Protocol` where callers
  need behavior rather than a concrete implementation. Each protocol claim names
  and tests the required attributes; it does not imply universal array/backend
  support.
- Object-design rules: public custom classes record the invariant, identity,
  reusable fitted state, or resource lifetime that justifies them over a
  function, array, DataFrame, xarray object, or dataclass. Methods that are
  invalid in hidden workflow states are avoided. Model configuration, learned
  state, fitting behavior, predictions, and result/provenance objects are
  distinguished; fitted-state requirements and mutation are explicit and tested.
- Canonical raw-array orientations and labeled dimension/coordinate names. The
  inventory explicitly resolves current alternatives such as `neuron`, `unit`,
  and `unit_id`, plus time-major versus unit-major population arrays.
- Coordinate rules distinguish array indexing, coordinate representation
  (Cartesian, polar, graph/node, or another named representation), reference
  frame (environment, world, allocentric, egocentric, or body-centered), and
  physical units. Conversions name their source/target semantics and preserve or
  explicitly transform each component.
- Identity and segmentation rules distinguish `unit_id` from `unit_index`, node
  or bin identity from array position, and `segment_id` from `segment_index`.
  Selection and storage ordering are observable; multi-segment inputs preserve
  segment identity or require an explicit concatenation policy rather than
  silently joining time series.
- Temporal rules including base units, timestamp precision, sample alignment,
  time support, epoch representation, and interval closure.
- Mutation and ownership rules for containers, arrays returned by properties,
  selections, metadata tables, result objects, and file-backed data.
- Coercion rules for accepted input dtype and value range, output dtype, scaling,
  precision, contiguity, sparse/ragged inputs, accepted array protocols, and
  CPU/device movement. Copying, rescaling, and lossy conversions are documented.
- Lazy-source rules declare which structure is available before loading; whether
  basic slicing performs a partial read; compute/materialization triggers; cache
  policy; scaling and dtype effects; resource ownership/closure; observable
  materialization state; and operations unavailable before materialization.
- Reproducibility rules centered on an explicit `rng` convention and avoidance
  of hidden global random state. Existing public `seed` surfaces are
  compatibility inputs to inventory, not silently reinterpreted. Private
  backend-adapter parameters such as sklearn-facing `random_state` are recorded
  separately and do not count as public vocabulary inconsistencies.
- Error and warning patterns, including stable exception categories and messages
  that identify the invalid input or value, violated condition or scientific
  consequence, and corrective action. Library code raises exceptions rather than
  catching an error only to print it and continue ambiguously.

A deliberate exception is allowed when uniformity would obscure a real
scientific distinction. Every exception records the affected family, rationale,
and conformance tests.

## Data Semantics and Lazy-Source Profiles

Stable APIs use applicable contract profiles so the same scientific concept is
not re-specified inconsistently in every docstring:

| Profile | Required declarations |
| --- | --- |
| Spatial array | Coordinate representation, reference frame, physical units, axis roles/order, dtype/value range, and conversion behavior |
| Population | Stable IDs versus positional indices, ordering, metadata alignment, filtering/missing-unit behavior, and serialization labels |
| Segmented time | Time unit/support, segment IDs versus indices, boundary semantics, ordering, and explicit preservation or concatenation behavior |
| Lazy source | Pre-load shape/dtype/identity, supported partial slicing, materialization trigger, caching, scaling/copy behavior, resource ownership, and state-dependent operations |

These are contract profiles, not proposed base classes. The audit first maps
existing arrays, containers, NWB objects, and adapters onto them. Introducing a
new proxy, quantity, coordinate, segmented-data, or extractor hierarchy remains
a separately reviewed design change.

## Result Profiles and Labeled Schemas

Stable results use one or more approved profiles rather than inheriting methods
that do not terminate a meaningful workflow:

| Profile | Required behavior |
| --- | --- |
| Labeled tensor/map | Stable array fields and dimensions; `to_xarray()` when labeled conversion is supported |
| Tabular/event | Stable row meaning, identifier columns, units, and `to_dataframe()` |
| Fitted model | Explicit model specification, fit mutation/return behavior, learned-state inspection, pre-fit errors, prediction/decode/transform behavior, and resolved configuration/provenance |
| Scalar/report | Stable named metrics through `summary()` or an equivalent report object |
| Streaming/chunked | Documented iterator/chunk semantics, ordering, resource ownership, and equivalence to full evaluation |

Conversion methods preserve the result's identity, identifier/index distinction,
segment structure, units, coordinate representation/reference frame, ordering,
and missing-data meaning. Essential scientific arrays such as occupancy, bin
edges, or validity masks are coordinates or data variables in labeled outputs,
not only unstructured attributes that may be dropped during common xarray
operations. Attributes remain appropriate for provenance and descriptive
metadata.

The grammar chooses one identifier vocabulary for cross-domain tables and
labeled arrays. Domain-specific coordinates remain distinct when they represent
different scientific quantities; consistency does not collapse direction,
position, egocentric bearing, or view into an ambiguous generic coordinate.

## Interoperability and Loss Contracts

Pandas schema behavior is a core interoperability contract because pandas is a
required dependency. Pynapple, NWB, and xarray are optional-integration contracts.
Accepted array protocols and their coercion behavior are execution contracts, not
dependencies. All of these boundaries must state what they support without
conflating core, optional, and protocol guarantees.

- Each adapter is classified as **lossless conversion**, **documented
  extraction**, or **lossy conversion**. A lossy path identifies discarded
  fields and provides a strict or richer alternative when one is approved.
- The Pynapple audit covers `TsGroup` unit identity and metadata; `Tsd` /
  `TsdFrame` timestamps, columns, time support, and metadata; and `IntervalSet`
  boundaries, interval metadata, units, and closure semantics.
- Direct Pynapple inputs and their documented array equivalents produce
  numerically equivalent results and identical surviving identity labels.
- NWB contracts cover unit-table identity, time-series timestamps, spatial units,
  environment schemas, lazy materialization, and ownership/closure of file
  handles.
- DataFrame and xarray conversions use stable schemas with round-trip tests for
  fields declared reversible. Non-reversible presentation formats are labeled as
  such.
- A checked-in compatibility matrix distinguishes the declared support range from
  the versions actually tested. Unless the project declares a different policy,
  CI covers the declared minimum and the locked/current version for each optional
  integration; an integration with no declared lower bound records only the
  locked/current version as tested until a bound is chosen. Pandas schema tests
  use the corresponding core-dependency rows. “Import succeeds” is not sufficient
  interoperability coverage.

If preserving information requires a materially new container or adapter API,
the inventory records the gap and routes it to a separate design review rather
than expanding curation silently.

## API Evolution and Deprecation States

Support tier and lifecycle state are separate axes. A stable API can become
soft-deprecated without losing compatibility, and an experimental API can be
removed without first being promoted.

- **Recommended:** taught in current documentation and open to compatible
  enhancement.
- **Soft-deprecated:** still documented and tested, safe for existing code, and
  not scheduled for removal; new code is directed to a preferred alternative.
  It emits no runtime warning solely because it is soft-deprecated.
- **Hard-deprecated:** intended to change or be removed after declared
  compatibility conditions are met; runtime/static diagnostics, migration
  guidance, rationale, and a feedback route are provided where feasible.
- **Permanent alias:** an intentional supported alternate path whose maintenance
  and discoverability cost has been accepted.

Hard deprecation is used only when removal or behavioral change is actually
intended. Compatibility conditions may include supported-version reach,
downstream compatibility evidence, migration tooling, or scientific maintainer
review; they do not name or schedule a release in this plan. Behavior-changing
scientific bug fixes still require explicit impact review even when maintainers
consider the prior behavior incorrect.

## Top-Level and Canonical Import Policy

- The top level is reserved for core data containers, stable public exception
  classes, and lazy domain namespaces.
- Domain computations are canonical at their subpackage root, for example
  `neurospatial.encoding.compute_spatial_rates`.
- Specialized building blocks are canonical in the narrowest owning public
  module when a subpackage-root export would overwhelm discovery.
- Existing top-level workflow conveniences remain compatibility candidates until
  the inventory evaluates their usage.
- The current sparse policy is the provisional runtime default. A small
  task-facade documentation/autocomplete prototype may be compared without first
  adding runtime aliases, but that comparison is non-blocking. Adopting a
  materially new facade requires evidence strong enough to dislodge the default,
  an explicit canonical-path decision, and a separate design review.
- New aliases are not added merely for symmetry or to make both import styles
  work. Autocomplete, `dir()`, static typing, documentation search, import time,
  and journey-to-symbol coverage are considered together.
- Public exceptions are explicitly tiered and included in the contract; error
  types are part of user-visible control flow.
- A symbol can have multiple reachable paths during migration, but only one
  canonical path.

## Population-Scale Requirements

The curation target covers present-day populations of hundreds of neurons and
should not structurally preclude low-thousands populations.

### API requirements

- Population entry points accept ragged spike trains and array-like population
  inputs without requiring user-written per-unit loops.
- Unit identity and metadata remain aligned across selection, batching,
  iteration, summaries, persistence, and interoperation. Public APIs do not
  conflate `unit_id` with positional `unit_index`, and they declare output order
  after selection or filtering.
- Multi-segment/session inputs preserve segment identity and time support or
  require the caller to choose an explicit concatenation/alignment policy.
- Return shapes use explicit, consistent dimension order and expose labeled
  conversion where available.
- Public functions state whether inputs are viewed, copied, coerced to a new
  dtype, materialized from a lazy source, or transferred between devices. A
  generic “array-like” annotation is narrowed to the protocols actually tested.
- APIs that materialize `n_units x n_bins`, `n_time x n_bins`, or larger arrays
  state that cost plainly.
- Where the scientific result is naturally sparse or unit-specific, APIs and
  conversions state whether they preserve sparsity or densify it, including the
  resulting allocation. This does not require sparse output for intrinsically
  dense spatial maps.
- When an existing summary, blocked, iterator, or streaming path is available, it
  is documented and agrees numerically with the full path within the documented
  tolerance. When no such path exists and full output can become impractical, the
  practical limit is explicit and the missing implementation is routed to the
  gap report rather than silently expanding this plan.
- No minimal import eagerly loads optional GUI, NWB, JAX, or notebook stacks.
- Specialized acceleration remains optional; correct NumPy behavior is the
  baseline contract unless an API explicitly requires another backend.
- File-backed or deferred population inputs document chunk boundaries, compute
  triggers, ordering, cache behavior, file-handle lifetime, and the memory
  consequences of conversion to NumPy, pandas, or xarray. If partial slicing is
  promised, a representative slice must not read or allocate the complete
  population.

### Scale audit and benchmarks

The inventory records known asymptotic shapes and allocation hotspots without
blocking documentation on new measurements. Phase 4 benchmark reporting covers
applicable primary population workflows at least at:

- 1 unit, to guard singular/batch equivalence.
- 100 units, representing common current recordings.
- 1,000 units, representing the planned population-scale envelope.

Benchmark scenarios also fix representative time samples, spatial bins, spike
density, dtype, segment structure, storage/chunk shape, and backend so results
are comparable. The benchmark report includes peak memory, wall time, output
size, whether partial access materialized a full source, whether a conversion
densified sparse/unit-specific data, and whether a bounded-memory path exists.

The initial measurements establish baselines rather than universal performance
promises. Regression gates are added only after workload-specific variance and
hardware sensitivity are understood. Intrinsically superlinear methods are not
required to meet an arbitrary latency target, but their scaling and practical
alternatives must be honest.

## Researcher Journeys Used for Curation

The stable-tier proposal is checked against:

1. Arrays to single-unit and population spatial rates.
2. NWB or pynapple session to population spatial rates.
3. Population encoding-model fit and Bayesian decoding, including summary mode.
4. Place, grid, border, directional, view, and egocentric coding analyses.
5. Trial, lap, region-crossing, navigation, and decision analyses.
6. Peri-event and population event-alignment analyses.
7. Cross-session alignment and result comparison.
8. Simulation and ground-truth validation.
9. Static, interactive, and exported visualization.

Each journey records its canonical starting object, primary function, result
type/profile, coordinate representation/frame/units, identity/index, segment and
time-support behavior, scale/materialization behavior, and next-step methods.

## Contract-Test Strategy

New curation tests follow an outside-in order while preserving the repository's
existing domain-oriented test layout:

1. **Structural API guards** check exports, metadata, signatures, documentation,
   minimal imports, and static discovery. They run first and fail quickly.
2. **Public-interface workflow tests** import only canonical public paths and
   exercise the supported researcher journeys with small representative data.
   They are concise enough to serve as executable usage documentation, avoid
   private attributes, and minimize mocks.
3. **Project integration tests** exercise real interactions among neurospatial
   components and with declared optional dependencies, including preservation
   and resource-lifetime contracts. They use the existing integration/extra
   markers and run after the fast public-interface suite.
4. **Domain, property, fuzz, and benchmark tests** continue to cover internal
   logic, extensive input spaces, numerical invariants, and scale. Slow or
   environment-specific suites remain independently selectable.

This plan does not require a wholesale rearrangement of existing tests. New
contract tests are placed by purpose, and each test focuses on one observable
behavior. Structural validators receive checked-in negative fixtures, while
semantic tests include applicable failure-path assertions. When a guard cannot be
exercised durably with a negative fixture, one representative deliberate mutation
per guard family is recorded to show that the diagnostic detects the intended
break.

## Phased Work

Phases are ordered by dependency and acceptance criteria, not by release number
or calendar date.

### Immediate execution tranche: visible corrections

These tasks land independently before the governance phase:

- Correct the false “removed in 0.7” promise for
  `detect_region_crossings` while retaining and testing the compatibility path.
- Restore concrete static types for the five lazy top-level exports without
  sacrificing PEP 562 runtime laziness or minimal-import behavior.
- Draft the Researcher API page from the nine existing journeys, leading with
  population workflows where appropriate.
- Remove private modules from primary navigation while preserving generated
  pages that remain valid link targets.
- Explicitly classify the complete current top-level surface, including
  `Session`, `load_session`, and every other lazy export.
- Audit public `seed` versus `rng` usage and migration options without renaming a
  public keyword.

Acceptance criteria:

- Current warnings/docs are truthful, headline lazy imports are statically
  typed, and the task-oriented page and primary navigation build strictly.
- No public path, compatibility shim, scientific behavior, or RNG keyword is
  removed or renamed by this tranche.
- Completing the tranche does not wait for curated metadata, a scale benchmark, or
  namespace comparison.

### Phase 0: Governance and existing deprecation debt

Deliverables:

- Add the curated policy metadata schema and a minimal metadata file containing
  the current top-level contract, seeded by the tranche's explicit
  classification of `Session` and the other lazy exports. Keep this repository
  policy out of the installed package and avoid duplicating derivable signatures
  or object kinds.
- Draft the API design grammar, data-semantics/lazy-source profiles, result
  profiles, tier promotion semantics, top-level policy and namespace comparison,
  compatibility dimensions, interoperability/loss categories, lifecycle states,
  and process for intentional contract changes. Record gated choices for
  maintainer decision.
- Inventory every remaining active deprecation. Recommend case by case whether
  to retain, revise, or remove it under the compatibility policy; runtime
  behavior changes remain gated.
  Classify safe indefinite compatibility paths as permanent aliases or soft
  deprecations rather than inventing removal promises.
- Define the schema for core-dependency and optional-integration support/tested
  versions, array-protocol execution claims, and adapter loss guarantees without
  importing optional packages while reading the metadata.
- Define how the project maintainer records rationale and evidence for metadata,
  snapshot, scientific, and breaking changes; do not require a multi-reviewer
  process that the current team does not have.

Acceptance criteria:

- No new deprecation uses an undefined removal process.
- Existing overdue notices no longer make promises the project is not following.
- The curated metadata can represent aliases, optional dependencies, contract
  profiles, result profiles, lifecycle state, and interoperability guarantees;
  its schema and entries validate with development tooling and without adding an
  installed-package dependency or importing neurospatial domain modules.
- The design grammar can express reviewed scientific exceptions rather than
  forcing superficial uniformity.
- No public symbol is removed merely to finish this phase.

### Phase 1: Inventory and contract proposal

Deliverables:

- Generate an inventory of top-level and subpackage exports with canonical
  module, kind, signature, aliases, tier proposal, owning domain, optional owner
  override, docs/examples/tests, optional dependencies, grammar/result profiles,
  contract dimensions, and known scaling notes.
- Bootstrap that inventory in an explicit incomplete-policy mode, use its missing
  policy queue to populate the full curated metadata catalog, and then switch the
  normal inventory check to strict policy completeness. Discovery must not require
  the complete catalog that it is intended to seed.
- Identify duplicate names, singular/plural pairs, deprecated aliases, public
  symbols sourced from private modules, and inconsistent population axes or
  identity behavior.
- Audit cross-domain vocabulary, function-family signatures, positional versus
  keyword-only controls, required scientific choices, interacting parameters,
  hidden `**kwargs`, return-kind changes, result methods, table columns, labeled
  dimensions, coordinate representation/reference frame, physical units,
  identifier/index and segment semantics, temporal conventions, mutation/copy
  behavior, RNG names, dtype/value-range coercion, accepted array protocols, and
  eager/lazy materialization.
- Audit architectural fit for stable candidates: I/O versus computation,
  friendly normalization versus strict kernels, functions versus stateful
  workflow objects, standard scientific types versus custom classes, and
  concrete-type checks versus tested structural protocols. Preserve fitted
  objects where learned state is explicit; flag methods that are valid only in
  undocumented hidden states.
- For every lazy/file-backed stable candidate, record pre-load structure,
  supported partial reads, materialization and cache behavior, dtype/scaling,
  resource ownership, observable state, and unavailable operations. This is an
  audit of existing behavior, not authorization for a new proxy hierarchy.
- Classify every Pynapple, NWB, pandas, and xarray boundary as lossless,
  extraction, or lossy. Keep pandas in the core-dependency rows; keep Pynapple,
  NWB, and xarray in optional-integration rows; and record array protocols
  separately as execution/coercion contracts. Record declared support separately
  from the minimum and locked/current versions actually tested, and map dropped
  identity/index distinctions, segment structure, metadata, time support,
  coordinate representation/frame, units, or schema fields.
- Record known population allocation/scale evidence and mark unknown behavior
  for later measurement; do not block documentation on benchmark machinery.
- Keep the sparse runtime namespace as the provisional default. A
  namespace-first versus small task-facade documentation/autocomplete comparison
  is optional and non-blocking.
- Review tier proposals against the researcher journeys, repository usage, and
  available downstream compatibility evidence.

Acceptance criteria:

- A checked-in script reproduces the inventory and baseline counts.
- Each proposed stable symbol has a contract profile and owning domain; the
  project maintainer is accountable unless an override is recorded.
- Each stable symbol either conforms to its grammar/result profile or records a
  reviewed scientific exception.
- Every applicable stable array/container declares representation, frame, units,
  axes, dtype/range, identity/index, segment, and lazy-source semantics through a
  profile or a reviewed inapplicable rationale.
- Every core or optional integration has separate declared-support and
  tested-version rows, every adapter has a loss classification, and accepted array
  protocols are recorded as execution/coercion contracts rather than dependencies.
- Known scale risks are recorded and unknowns are queued for Phase 4 rather than
  guessed.
- No symbol is removed in this phase.

### Phase 2: Documentation and discoverability without breaking changes

Deliverables:

- Refine the tranche's task-oriented Researcher API page with metadata-backed
  tiers and canonical paths.
- Generate the primary reference from curated metadata plus discovered public
  exports rather than every source file.
- Separate stable and advanced navigation, visibly label experimental APIs, and
  omit internals.
- Mark compatibility aliases as aliases and stop teaching them in examples.
- Present singular and population paths together, with population APIs first for
  systems-neuroscience workflows.
- Document scale, full-output size, coordinate/frame/unit and ID/index semantics,
  segment behavior, materialization triggers, practical limits, and existing
  bounded-memory alternatives near each primary population API. Where none exists,
  link the dispositioned gap-report entry.
- Publish the API design grammar, data-semantics/lazy-source profiles, result
  profiles, lifecycle labels, supported interoperability matrix, adapter loss
  behavior, and lazy/resource-ownership semantics.
- Preserve the tranche's concrete types for lazy headline exports and extend
  discoverability checks across lazy namespaces through `dir()`, IDE completion,
  static typing, and documentation search.
- If the optional namespace comparison is performed, use maintainer walkthroughs
  plus objective autocomplete, `dir()`, static-typing, documentation-search, and
  import-cost checks. Otherwise retain the sparse namespace; deferral cannot
  block documentation acceptance.

Acceptance criteria:

- Every stable symbol has an anchored reference entry and is reachable from the
  task map.
- No internal module appears in the main API navigation.
- Link and executable-snippet checks cover all journey-to-symbol mappings.
- Each documented round trip has an executable preservation test or is labeled
  as an extraction/non-reversible presentation.
- Every selected journey has one unambiguous canonical entry point reachable
  from the task map and the applicable runtime/static discovery surfaces. If a
  namespace comparison was performed, its findings identify the tested variant
  and unresolved ambiguity; otherwise the current sparse namespace remains the
  documented default.

### Phase 3: Canonicalize names and import paths

Deliverables:

- Select one canonical path for every duplicated public export.
- Record intentional permanent aliases separately from migration aliases.
- Add deprecations only where the compatibility policy requires them.
- Normalize high-value inconsistencies that obstruct task discovery; avoid a
  repository-wide cosmetic rename.
- Apply approved grammar changes to high-value families, including parameter
  vocabulary, keyword-only controls, RNG naming, dimensions/identifiers, result
  profiles, and explicit conversion methods. Preserve scientific distinctions
  and use compatibility shims where required.
- Resolve approved interoperability losses or relabel the affected functions as
  documented extraction. Materially new lossless containers or adapters remain
  separate design projects.
- Maintain a migration table with old path, new path, deprecation condition,
  lifecycle state, compatibility status, and eventual removal eligibility.

Acceptance criteria:

- Documentation and examples use canonical paths only.
- Migration aliases have tests for warning category, message, stack level, and
  replacement behavior where warnings are part of the policy.
- Singular and batch routes remain numerically equivalent where documented.
- Grammar migrations preserve or deliberately translate identity/index and
  segment semantics, coordinate representation/frame, units, time support,
  missing-data meaning, and result schemas.
- No unannounced stable contract break occurs.

### Phase 4: Enforce the contract in CI

Stage 4A lands the low-cost, high-value guards first:

- Generate the reviewed stable-contract snapshot from runtime inspection and
  curated metadata.
- Compare curated metadata, runtime exports, lazy/static declarations, snapshot,
  and generated docs in CI.
- Detect accidental changes to stable signatures and declared result schemas.
- Test the minimal installation and each optional-extra import boundary.
- Preserve runtime and static discoverability for lazy exports.

Stage 4B begins with readable outside-in tests of the supported researcher
journeys, then adds contract-specific guards only where curated metadata declares
that a dimension applies:

- Add public-interface workflow tests that use canonical imports, representative
  data, and no private implementation details; keep them small and minimally
  mocked so they double as living usage documentation.
- Add behavioral conformance tests for applicable grammar dimensions, including
  coordinate representation/frame, identifier/index and segment semantics,
  copy/view behavior, mutation, exception classes, fitted-state errors, RNG
  determinism, dtype/value-range/device coercion, partial reads, caching,
  eager/lazy materialization, and file-handle ownership.
- Add metamorphic tests for coordinate round trips, singular/batch,
  full/chunked/partial-read, arrays/Pynapple, and result/conversion equivalence
  where the contract declares equivalence.
- Exercise the defined interoperability test cells—normally the declared minimum
  and locked/current versions—and preservation/loss assertions, not only
  successful imports. Do not interpret an open-ended support declaration as a CI
  job for every intermediate release.
- Add population-scale benchmark reporting for fixed 1-, 100-, and 1,000-unit
  scenarios, including accidental full materialization and densification checks,
  and selected regression guards.
- Track import time and memory so convenience exports do not pull in optional
  stacks.

Stage 4B does not create every compatibility dimension × every stable symbol as
a test matrix. Inapplicable or unverified dimensions are recorded with a
rationale. Timing gates are added only after workload variance is characterized;
benchmark reporting does not require a raw-time CI threshold.

Acceptance criteria:

- Structural validators have durable negative fixtures and semantic guards cover
  applicable failure paths. Guard families that cannot use a persistent negative
  fixture have one representative deliberate failing run with an actionable
  diagnostic recorded in the progress log.
- Stage 4A protects metadata/export, docs, minimal imports, and lazy typing even
  while Stage 4B remains incomplete.
- Intentional contract changes have an explicit review/update workflow.
- Stable behavior cannot change merely by regenerating a structural snapshot;
  affected semantic tests and rationale must change explicitly.
- Scale checks report peak memory and output allocation, not wall time alone.
- Stable lazy-source checks detect a promised partial read that instead loads the
  complete source; sparse/unit-specific conversions report densification.
- CI distinguishes noisy benchmark variance from structural regressions such as
  unintended dense allocation.

### Phase 5: Eligible cleanup and policy maintenance

Deliverables:

- Remove only aliases and paths whose declared compatibility conditions have
  been satisfied.
- Publish migration documentation alongside every cleanup.
- Keep soft-deprecated paths documented and tested unless they enter a separately
  approved hard-deprecation process; soft deprecation alone never makes a path
  eligible for removal.
- Review experimental symbols for promotion, continued experimentation, or
  removal against their documented evidence criteria.
- Refresh the task map, inventory, scale baselines, and contract snapshot as the
  library evolves.

Acceptance criteria:

- Every removal is traceable to curated metadata and completed compatibility
  conditions.
- The supported stable surface, migration state, and known scale limits are
  published and current.
- The published design grammar, result profiles, integration matrix, and loss
  declarations match runtime behavior.
- Cleanup timing is chosen independently of this plan.

## Follow-On Gap Report

Phase 1 may identify missing facades, session QC, configuration/provenance
objects, richer lossless interoperability containers, backend/Array API support,
or new bounded-memory algorithms. Those findings are recorded with repository
or downstream evidence, affected journeys, preservation requirements, and scale
impact.
Material additions receive their own design and scientific review rather than
expanding this curation plan.

## Required Artifacts

Exact paths, producing task IDs, schemas, and validation commands are defined in
[TASKS.md](TASKS.md). The required artifact set is:

- `docs/api/public_api_metadata.json` and its JSON schema with support,
  owning-domain, optional owner-override, lifecycle, and contract-profile policy.
- `docs/api/conventions.md` with the approved API design grammar, result
  profiles, data-semantics/lazy-source profiles, and reviewed exceptions.
- Reproducible inventory script and human-readable inventory report.
- Reviewed stable-contract snapshot.
- Researcher API journey/task map.
- Deprecation debt ledger and migration table.
- Namespace-default record plus any non-blocking comparison of tested
  import/navigation variants.
- Core-dependency and optional-integration support/import matrix, with accepted
  array protocols represented separately as execution contracts.
- Interoperability preservation/loss ledger covering Pynapple, NWB, pandas, and
  xarray, including representation/frame/units, ID/index/segments, partial
  reads/caching, dtype/scaling, and resource ownership.
- `tests/test_api_semantics.py` and the public-workflow/interoperability tests as
  the behavioral and metamorphic conformance suite.
- Population-scale benchmark matrix and report.
- Follow-on workflow gap report.

## Risks and Mitigations

- **Risk: symbol-count targets cause useful APIs to disappear.** Curate by task
  and support tier, not by quota.
- **Risk: population APIs look concise but hide prohibitive allocations.** Make
  dimension sizes, complexity, peak memory, existing bounded-memory alternatives,
  and explicit limitations or gap-report dispositions part of the contract.
- **Risk: singular examples dominate despite population use.** Teach singular
  and batch paths together and lead systems-neuroscience journeys with batch
  APIs.
- **Risk: aliases keep the conceptual surface large forever.** Distinguish
  permanent convenience aliases from migration aliases and attach explicit
  eligibility conditions to the latter.
- **Risk: snapshots become opaque golden files.** Generate them from inspected
  runtime exports plus curated metadata, keep them human-reviewable, and require
  rationale for updates.
- **Risk: a design grammar erases real scientific differences.** Standardize
  recurring software concepts while recording reviewed exceptions for distinct
  coordinate frames, statistical meanings, or analysis families.
- **Risk: “coordinates” conflates representation, frame, units, and array
  indexing.** Record these independently and make conversions state which
  components they preserve or transform.
- **Risk: “conversion” implies preservation that an adapter does not provide.**
  Classify boundaries as lossless, extraction, or lossy and test declared fields.
- **Risk: lazy or file-backed APIs hide computation and resource lifetime.** Make
  pre-load structure, partial reads, compute triggers, caching, dtype/scaling,
  copying, chunking, observable state, and ownership part of the contract.
- **Risk: external precedents encourage a new abstraction hierarchy.** Adopt
  their testable semantics first; a new quantity, coordinate, extractor, proxy,
  or distributed collection requires separate design evidence.
- **Risk: a flat task facade becomes a second permanent API.** Prototype in docs
  first, select one canonical path from documented discovery results, and avoid
  symmetry aliases.
- **Risk: broad array-like claims imply unsupported GPU or distributed behavior.**
  Name and test accepted protocols, dtype coercion, and device transfers per
  workflow; keep NumPy as the baseline.
- **Risk: new facades duplicate algorithms.** Keep new workflow design in
  follow-on plans and require numerical equivalence if approved.
- **Risk: advanced users perceive reduced prominence as removal.** Keep advanced
  APIs documented, searchable, and supported.
- **Risk: import curation changes optional-dependency behavior.** Test minimal
  installation and each optional-extra boundary explicitly.
- **Risk: scale benchmarks create brittle CI.** Establish variance first and
  gate structural allocations or robust regressions rather than noisy raw time.
- **Risk: the aggregate contract matrix costs more to maintain than the API.**
  Land structural guards first, add semantic/interop/scale checks only for
  declared contract dimensions, and record inapplicable dimensions instead of
  generating a universal cross-product.

## Governance Questions Resolved During Phase 0

1. The project's compatibility process and removal eligibility conditions.
2. The canonical vocabulary, signature families, coordinate
   representation/reference-frame rules, physical units, temporal and segment
   rules, ID/index distinctions, dimensions, dtype/range behavior, RNG
   convention, lazy-source profile, and mutability/ownership rules in the design
   grammar.
3. Initial tier, lifecycle state, grammar/result profile, and contract profile
   for each current top-level export.
4. Stable result fields, fitted-state behavior, result profiles, table
   identifiers, segment labels, and labeled schema dimensions that callers may
   rely on.
5. The loss classification plus declared-support and tested-version rows for each
   current core or optional integration, and the separately tested execution
   contract for each accepted array protocol.
6. Which advanced APIs receive narrower compatibility guarantees and what those
   guarantees are.
7. Whether repository evidence is strong enough to dislodge the current sparse
   namespace as the provisional default for common researcher tasks.
8. Which population workflows require new bounded-memory implementations; those
   become separate follow-on projects.

## External Design Precedents

These projects inform the plan without becoming requirements by imitation:

- [Pynapple API](https://pynapple.org/api.html),
  [metadata](https://pynapple.org/user_guide/03_metadata.html), and
  [releases](https://pynapple.org/releases.html): domain-native time-series,
  interval, population, metadata, generalized-analysis, and lazy-I/O patterns.
- [Astropy coordinates](https://docs.astropy.org/en/stable/coordinates/representations.html),
  [units](https://docs.astropy.org/en/stable/units/index.html), and
  [unified I/O](https://docs.astropy.org/en/stable/io/overview.html): separate
  coordinate representation, reference frame, physical units, and high-level
  versus format-specific I/O.
- [SpikeInterface core](https://spikeinterface.readthedocs.io/en/latest/modules/core.html):
  stable unit/channel IDs versus indices, multiple segments, metadata
  propagation, on-demand access, and unit-specific sparsity at population scale.
- [Scikit-image array conventions](https://scikit-image.org/docs/stable/user_guide/numpy_images.html)
  and [dtype contracts](https://scikit-image.org/docs/stable/user_guide/data_types.html):
  explicit axis/coordinate vocabulary, accepted value ranges, conversions, and
  output dtype behavior for array-first algorithms.
- [NiBabel ArrayProxy](https://nipy.org/nibabel/reference/nibabel.arrayproxy.html)
  and [memory behavior](https://nipy.org/nibabel/images_and_memory.html): a small
  testable lazy-array protocol, partial slicing, caching, dtype/scaling, and
  file-resource semantics.
- [MNE-Python Epochs](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)
  and [memory-efficient I/O](https://mne.tools/stable/documentation/implementation.html#memory-efficient-i-o):
  domain containers with explicit shapes, identity/metadata, conversion methods,
  and visible preload-dependent behavior—including the cost of methods that are
  unavailable in some states.
- [NetworkX functional API and graph views](https://networkx.org/documentation/stable/reference/functions.html):
  algorithms over shared graph interfaces plus explicit view/copy behavior.
- [Statsmodels model-fit-results workflow](https://www.statsmodels.org/stable/gettingstarted.html):
  separation of model specification, fitting, fitted results, prediction, and
  summaries.
- [Dask Array best practices](https://docs.dask.org/en/latest/array-best-practices.html):
  explicit chunks and materialization, storage-aligned access, and a preference
  for NumPy when parallel/lazy complexity is unnecessary.
- [NumPy NEP 52](https://numpy.org/neps/nep-0052-python-api-cleanup.html) and the
  [SciPy API policy](https://docs.scipy.org/doc/scipy/reference/): explicit
  public/private boundaries, one canonical location, coherent namespaces, and
  lazy submodule discovery.
- [Scikit-learn estimator conventions](https://scikit-learn.org/stable/developers/develop.html):
  predictable vocabulary, signatures, object profiles, and common conformance
  checks.
- [Xarray data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html)
  and [alignment semantics](https://docs.xarray.dev/en/stable/user-guide/computation.html#automatic-alignment):
  named dimensions, coordinates, identity-aware alignment, and labeled
  interoperability.
- [pandas copy-on-write design](https://pandas.pydata.org/pdeps/0007-copy-on-write.html):
  consistent user-visible copy/view and mutation behavior independent of internal
  optimization.
- [Python PEP 387](https://peps.python.org/pep-0387/): behavioral compatibility,
  soft deprecation, and evidence-aware incompatible changes.
- [Scientific Python SPEC 1](https://scientific-python.org/specs/spec-0001/) and
  [SPEC 7](https://scientific-python.org/specs/spec-0007/): discoverable lazy
  loading and explicit RNG APIs without global random state.
- [Scientific Python design](https://learn.scientific-python.org/development/principles/design/),
  [process](https://learn.scientific-python.org/development/principles/process/),
  and [testing](https://learn.scientific-python.org/development/principles/testing/)
  recommendations: conventional and composable interfaces, separation of I/O
  from scientific logic, explicit errors and return types, outside-in public
  behavior tests, and independently runnable test suites.
- [Python Array API purpose and scope](https://data-apis.org/array-api/latest/purpose_and_scope.html):
  explicit shape, dtype, NaN/Inf, empty-input, device, and interchange semantics
  without assuming all array libraries behave identically.
- [PyTorch feature classifications](https://pytorch.org/blog/pytorch-feature-classification-changes/):
  readiness tiers that include demonstrated value, documentation, performance,
  feedback, and compatibility expectations.
