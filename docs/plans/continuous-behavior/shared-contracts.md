# Shared contracts for continuous behavior and neural analysis

[Back to PLAN.md](PLAN.md)

**Created**: 2026-10-14
**Committed**: 2026-10-14
**Status**: Proposed implementation contracts

These contracts extend the existing array-first interfaces across the six
workstreams. The [environment representation contracts](../environment-representation/shared-contracts.md)
continue to govern that refactor. Implementation must resolve explicitly marked
API decisions before exporting new names.

## Arrays, clocks, and observation boundaries

New temporal operations take data positionally and settings after `*`. Use
`times` in seconds, `positions` shaped `(n_samples, n_dims)`, `position_bins`
for sample-aligned bin indices, and `headings` in radians. Proposed new joint
trajectory calls start with `(times, positions, *, ...)`.

Each call operates on one finite, strictly increasing clock. Restarting
recording-local clocks remain separate calls/tables until explicit clock
alignment is supplied. Row and distance-sample numbers never become invented
timestamps. Existing APIs accepting duplicate times need a documented migration
decision; new interpolation/derivative APIs reject ambiguous duplicate times.

Optional `break_before`, a Boolean array shaped `(n_samples,)`, declares a
recording or observation break before a sample regardless of time difference.
The first sample starts a run implicitly. Confirm this proposed keyword before
export; avoid multiple equivalent recording-boundary inputs.

Reuse `_intervals` and `environment.trajectory` helpers to combine explicit
breaks, `epochs`, timestamp gaps, and required observation validity. Epochs are
half-open `[start, stop)` intervals on the same clock as `times`.

- A position sample requires finite values in all coordinate columns used by
  the operation. Signal-feature validity can differ by feature.
- Motion/interpolation requires valid endpoints within one run. Removing a
  missing row does not make its neighboring finite samples adjacent.
- Occupancy retains its declared allocation rule, including start-sample
  allocation. Acquisition support and motion validity are distinct definitions;
  changes to existing occupancy gates require scientific validation.
- A rate estimator's spike numerator and occupancy denominator use matching
  resolved observation intervals and analysis gates.
- Mask invalid bins before indexing: `-1` must not select the final value.
- Support cutoffs censor events; they do not invent arrivals, departures,
  failures, or zero signals.

Derived sample-aligned `run_ids` are integers local to a call, with `-1` where
there is no continuous motion/interpolation support. A separate sample-validity
mask preserves isolated finite positions, which can support point queries but
provide no motion interval. External recording/subject identity remains caller
metadata. Local run IDs do not identify recordings across independent calls.

## Distance, sampling, and movement quantities

Cumulative observed distance sums Euclidean displacements between supported
continuous samples. Run-local and aggregate observed progression are separate;
neither adds displacement across a break. Graph geodesic distance retains its
distinct environment/binning assumptions. State whether distance uses XY, XYZ,
or other supplied coordinates.

Physical-distance resampling proceeds independently within each valid run:

1. Anchor at its first position and sample multiples of positive `spacing`
   expressed in declared spatial units.
2. Interpolate positions and times with identical source brackets and weights.
   Skip zero-length intervals for moving interpolation while preserving original
   adjacent brackets and distinct plateau arrival/departure times. An exact
   interior plateau anchor uses the last stationary time before movement.
3. Retain bracketing source indices, interpolation weights, times, run identity,
   and actual run-local distance.
4. Omit an off-grid final endpoint by default. Optional endpoint retention
   flags that sample and records actual spacing; never duplicate an endpoint
   already on the grid.
5. Never extrapolate or interpolate across unsupported intervals. Tracking
   repair, if later added, is explicit and preserves imputed/observed identity.

Define stationary/singleton output without fabricated motion or division by
zero. A stationary run retains its first position/time and separate observed
run-time bounds. Direction and curvature are undefined without displacement.
Distinguish turning angle (radians), angular velocity (radians/second), and
geometric curvature (inverse spatial units). Signed 2D turning and unsigned 3D
curvature have different dimensional contracts.
The legacy N-D turning kernel combines an XY cross product with a dot product
over all supplied coordinates. Preserving that output is a compatibility
decision; it must not be described as a pure XY projection. New planar quantities
declare their coordinate dimensions and reject unsupported input explicitly.

## Occurrences, eligibility, and history

Crossings/encounters preserve occurrence identity, original sample indices,
times, source/destination, run, and observed/censored bounds. Reuse the scalar
immutable `RegionVisit`/`RegionTransition` patterns. Sample-observed crossings
and optional interpolated crossings have distinct stated time semantics.

Seconds of occupancy, sample transitions including self-transitions, and counts
of location changes are separate measurements. Crossing aggregates count
occurrences independently of sampling-rate-dependent stationary repetitions.
Retain caller-provided jump/artifact/condition annotations without assigning
universal biological labels.

Coverage requires declared eligible bins/physical support. Measure-weighted
coverage divides visited eligible measure by total eligible measure. A
visited-only environment cannot silently define available space. State what
observation declares a visit; sample visits do not assert traversal of every
intermediate bin.
Physical weights require verified cell measures with declared measure dimension
and units, or explicitly supplied validated measures. Positive `env.bin_sizes`
alone do not establish area: current hex/mesh subsets can report lengths and an
isolated-bin fallback. Reject unsupported physical weighting until the relevant
representation fix, while allowing count-based coverage independently.

Complete history uses observations strictly before each reference occurrence
within one supported run by default. An explicit "since last observed" summary
may reach across a break, but it reports incomplete/censored history: unseen
visits and travel during the break remain unknown. First visits, incomplete
lookback, and undefined revisits remain identifiable.

Labels obtained by forward path matching or a whole-recording repertoire are
post-hoc descriptions. Extracting preceding samples does not make match
membership a causal predictor. Predictive workflows require prefix-only label
assignment and references fitted within training data. Labels/fitting must not
use future or held-out information implicitly.

## Ordered path matching and direction

Temporal orientation and physical travel direction are separate controls.
Advancing both paths forward does not by itself establish compatible movement.
Same-direction matching also requires supported, nonzero local tangents in the
declared coordinate space. By default their normalized dot product is positive;
an optional stricter angular bound is in radians and recorded in settings.
Tangent estimation and its boundary handling must be explicit and contained
within each run. Undefined tangents cannot pass this direction gate.

Reverse temporal comparison records its index orientation and applies the
declared tangent convention. Spatial tolerance, angular compatibility, and
minimum overlap length jointly define an accepted segment. This is local
compatibility, not a global clockwise/counterclockwise or route-identity verdict.
Tests include permissive spatial tolerances, opposite directions at the same
location, and small reversed loops; positional proximity alone must not pass.

## Spatial identity and correspondence

Array position remains a computational index. Optional persistent scientific
IDs are unique within a declared identity namespace; independently generated
environments have no implied correspondence.

Subsets/transforms preserve exact parent-child mappings. Rebinning uses explicit
many-to-one maps or justified overlap weights where available. Unsupported
correspondence is reported, not inferred by row order or nearest center.
Count/mass transfer, density transfer, and scalar averaging have separate
conservation contracts.

Field comparisons verify geometry or explicit correspondence. Descriptive
`env_fingerprint` is not a spatial equality check. Unit identity also requires
explicit correspondence across independently acquired recordings.
Initial rate comparisons accept verified one-to-one correspondence, including
reordering or matched subsets. Reject many-to-one, split, and overlap mappings
at this boundary. Coarsening requires separate re-estimation on a common
binning or a separately specified count/exposure transfer; averaging rate values
is not equivalent to combining their observations.

Coordinate and measure dimensions remain explicit. Cell adjacency, movement
links, diffusion faces, and empirical transitions have distinct roles.
Path-only links invent no shared face or conductance. Physical walls remain
distinct from analysis selections.

## Features and environmental conditions

Use `feature_ids` for labeled scalar columns. Features can have bin or ordered
side axes, with side facts at walls independent of navigation edges. Declare
feature units and required coordinate frames separately from geometric arrays.

Sampling preserves input sample identity and returns values plus validity.
Unknown space/nonfinite features stay missing. Initially represent categories
as numeric codes with an explicit code-to-label mapping, preserving zero as a
valid category; linear/circular interpolation of categories is rejected.
Interpolation policies distinguish continuous measurements and sharp physical
steps. Transform body-relative offsets explicitly before spatial lookup.
Containment sampling requires a verified locator/footprint capability or a
validated caller-supplied physical footprint. A nearest-center track locator is
not containment, and positive `contains()` output is not proof when the locator
has no footprint contract. Refuse containment for unsupported layouts; offer
explicit nearest-bin queries with their distance/support policy stated. Verified
grids and polygons can support containment before the broader arena roadmap.

Changing features/topology initially use caller-provided half-open state
intervals and a mapping from `snapshot_id` to immutable geometry/feature states.
Reject overlap, unknown IDs, and incompatible units/frames. Uncovered time
produces unknown samples by default. Never interpolate between snapshots or
implicitly carry the preceding state through a gap. Reuse immutable environments
by reference; do not copy full geometry per lookup or mutate it in place.
Environment-valued state snapshots and identity-keyed caches depend on the
existing roadmap's immutable-value stage B. Read-only geometry publication in
representation phases 1–4 does not freeze units, regions, or graph attributes.
A frozen wrapper is insufficient. Earlier fixed-geometry feature schedules may
use owned immutable feature arrays and captured source context, without storing
or caching mutable environments as snapshots.

## Continuous event alignment

Retain `event_index`, the original input event row, separately from optional
external event labels. Events at identical times remain distinct occurrences.
Signal columns use `feature_ids`. Aligned axes are
`(event_index, relative_time, feature_id)`.

Interpolation respects signal support, per-feature validity, and explicit
boundaries. Per-cell masks and contributing counts distinguish partial windows,
missing measurements, and zeros. Event filtering reports retained/excluded
identity. Interpolation does not cross a state boundary that changes a signal's
definition.
Feature metadata declares value kind independently of units: linear, circular
(radians), or categorical with code/label mapping. Interpolation and reduction
policies are feature-specific. Linear summaries use supported sums/counts;
circular summaries use sine/cosine sums and counts, with NaN plus an explicit
reason when normalized resultant length is at or below the recorded numerical
threshold (proposed default `1e-12`) or count is zero.
Categorical summaries return label counts/proportions and never an arithmetic
mean of codes. Summary-only and retained outputs agree on these reductions.

## Conditions and neural comparison

Label-to-epoch conversion declares interval allocation, splits at support
boundaries, and leaves missing labels outside conditions. No final-sample
duration is inferred without explicit recording support. Caller-declared
conditions may overlap; comparisons state when disjoint conditions are required.

Reuse epoch-conditioned encoding and fitted decoder interfaces. Align unit IDs,
spatial correspondence, and common observed support; retain occupancy and
excluded-bin diagnostics. Insufficient support has an explicit status and
undefined metric. Conditional fields alone do not identify independent history
coding: examples inspect speed, heading, occupancy, and supplied trial/recording
holdouts.
Before rate comparison ships, encoding results capture immutable estimation-time
spatial context: coordinate system, units/frame, bin coordinates/order/measures,
identity/correspondence where present, and relevant topology/operator provenance.
Capture owned minimal arrays/scalars once, or reuse a genuinely immutable stored
binning when available. Preserve context through unit indexing and exports.
The live `result.env` reference cannot reconstruct an earlier context. Legacy or
manually constructed maps with missing context require explicit verified source
context or are rejected; do not infer provenance from current environment values.

## Results, exports, and bounded computation

Results own read-only copies of derived arrays and preserve settings, support,
units/frame, labels, and source mappings. Retain no raw-recording buffers,
recording closures, or mutable aliases.

Existing neural rate/decoder export rules remain in force. New non-neural
results need a narrow documented domain extension: sample/crossing/match tables
use occurrence identity, and feature/event arrays use their own axes. Do not
manufacture `unit_id` or reuse neural `(unit_id, bin)` axes. Prefer domain
methods such as `samples_table()`, `crossings_table()`, and `matches_table()`
where existing terminal verbs would misrepresent rows. New continuous-event
exports do not change spike-PSTH exports.

Use existing dependencies; xarray remains optional and lazy. Verify names and
axis definitions in each phase's API review. Sparse path storage scales with
retained matches. Chunks carry required boundary context and reset at support
or state breaks; chunked and unchunked outputs agree. Add summary-only modes
for demonstrated large outputs without a distributed-execution framework.
