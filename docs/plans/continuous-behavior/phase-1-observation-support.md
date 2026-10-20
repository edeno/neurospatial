# Phase 1 — Observation support and distance-based preprocessing

**Created:** 2026-10-14
**Committed:** 2026-10-14
**Status:** Implemented — all three milestones delivered in PRs #112, #114
and #120. `break_before` reaches every function that takes `max_gap`
(#121).
**Status reviewed:** 2026-10-20

[Back to PLAN.md](PLAN.md) · [Shared contracts](shared-contracts.md) ·
[Environment representation plan](../environment-representation/PLAN.md)

Scientists must be able to compare movement recorded at different sampling
rates, with pauses, missing tracking, and different running speeds. This phase
establishes which motion was observed and derives distance-based samples without
inventing movement between recordings. It also separates geometric curvature
from the turning angles currently returned under that name.

This is a foundation and reliability phase. Functions continue to accept arrays;
an optional small result carries derived support and provenance. No mandatory
trajectory or session class is introduced.

## Read and reuse first

- `src/neurospatial/_intervals.py`: `as_intervals`, `intervals_contain`,
  `run_sample_bounds`, and `run_time_bounds`; preserve half-open epoch semantics.
- `src/neurospatial/environment/trajectory.py`: `interval_valid_mask` at line 97,
  `observed_interval_mask` at line 249, and `observed_runs` at line 298.
- `src/neurospatial/_validation.py`: `validate_times_positions` at line 154;
  existing callers accept non-decreasing clocks, unlike some behavior APIs.
- `src/neurospatial/behavior/_kinematics.py`: `interval_velocity`.
- `src/neurospatial/behavior/trajectory.py`: `compute_step_lengths`,
  `compute_turn_angles`, and `compute_trajectory_curvature`.
- `src/neurospatial/behavior/navigation.py`: `traveled_path_length` at line 1269.
- `src/neurospatial/ops/egocentric.py`: per-run velocity heading and the
  timestamps-free `heading_from_body_orientation` interpolation behavior.
- `src/neurospatial/_results.py`: `ReadOnlyArraysMixin` and `read_only_copy`.
- `tests/behavior/test_kinematics_gaps.py`, `test_segmentation_gaps.py`,
  `tests/environment/test_trajectory_gaps.py`, and `tests/events/test_add_positions_gaps.py`.
- `.claude/rules/public-api.md`, `.claude/rules/tests.md`, and the shared contracts.

The environment representation work supplies geometry and physical bin sizes.
This phase needs no new layout engine and must not depend on that refactor being
complete. Use current environment capabilities where needed.

## Scientist workflows

1. Compute travel distance across several observation epochs without counting
   the displacement between the last point of one epoch and the first of another.
2. Resample a path every chosen physical distance while retaining the original
   times, so route analysis can ignore speed and later recover speed or spikes.
3. Compare a curved path recorded at two frame rates without confusing smaller
   per-sample turn angles with a straighter path.

## Contracts and initial decisions

### Observations and clocks

- New time-dependent calls use `(times, positions, *, ...)`, with `times` in
  seconds and `positions` with shape
  `(n_samples, n_dims)`. Clocks are finite and strictly increasing within a call.
  Reject duplicate or restarted clocks with an actionable message; separately
  recorded clocks can be processed separately or mapped to a common clock.
- Non-finite coordinates and an optional validity mask mark missing observations;
  they are not replaced with zeros, interpolated automatically, or silently dropped.
- An interval is usable only when both endpoint observations are valid, elapsed
  time is positive and within `max_gap`, it is fully inside an epoch, and it
  crosses no explicit recording boundary. Optional `break_before` has shape
  `(n_samples,)`; its first sample is an implicit break, and later true values
  declare breaks regardless of elapsed time. Derived `run_ids` use `-1` where
  there is no continuous observation support.
- Keep observation support separate from speed, region, and task-cohort filters.
  A stationary observed interval has zero displacement and positive exposure.
- Derived run bounds refer to the original sample array. An isolated observed
  point can have a valid position but provides no velocity or travel interval.
  Make this distinction explicit in outputs rather than inventing duration.
- Keep `interval_valid_mask` as the rate-map numerator/denominator authority.
  Refactoring helpers must preserve spike/occupancy agreement; new endpoint
  validity rules are not silently imposed on every existing rate-map caller.

### Distance and resampling

- Initial resampling uses Euclidean arclength along the observed polyline, in
  the coordinate units. Do not call this geodesic distance or infer traversed
  graph edges between observed points. Existing geodesic step lengths remain
  a separate operation.
- Accumulate distance only over usable intervals. Report per-run distance and
  observed cumulative distance across runs; restarting a run adds no jump.
  Missing samples have an explicit invalid output marker.
- Place samples at `0, spacing, 2 * spacing, ...` independently in each run. Omit an
  unmatched terminal remainder by default; an explicit endpoint option can
  include it and must mark the final spacing as shorter than `spacing`.
- Skip zero-length intervals when locating moving interpolation brackets, but
  keep the original adjacent source indices and times. A plateau's arrival and
  departure times belong to different neighboring moving segments; do not replace
  both with one collapsed timestamp knot. At an exact interior distance anchor,
  use the last stationary time before movement; moving-run endpoints retain
  their observed times. A wholly stationary run yields one position at its first
  time, no velocity, and separate observed run-time bounds preserving its duration.
- Return interpolated original-clock `times`, positions, distance coordinate,
  run identity, source left/right indices, and interpolation weights. Exact
  samples use the same left/right index. The source array is never retained.
- Time is interpolated with the same weight as position along a valid segment.
  No global synthetic constant-speed timestamps are substituted for measured time.
- Numerical thresholds use the same units as positions; optional unit/frame
  metadata records the convention without adding an automatic conversion system.

### Honest turning quantities

- Preserve `compute_trajectory_curvature` and `compute_turn_angles` numerical
  outputs during this phase. In 2D, the legacy kernel returns signed turning
  angles in radians. For N-D with N > 2 it computes
  `atan2(v1_x * v2_y - v1_y * v2_x, sum(v1 * v2))`: the cross term uses XY while
  the dot product uses all dimensions. This is neither an XY projection nor a
  geometric N-D turning angle or curvature; document the actual legacy formula.
- For example, `[[0, 0, 0], [1, 0, 1], [1, 1, 2]]` gives a legacy middle value
  of `pi / 4`, while projecting onto XY gives `pi / 2`. Correct the description
  without silently changing this output. A numerical correction/migration of
  legacy N-D behavior requires a separate compatibility decision.
- Introduce a clearly named sample-aligned planar turning-angle operation and a
  separate planar geometric-curvature operation. Both require exactly two
  coordinate columns and reject N-D rather than projecting. Public names are
  provisional pending API review.
- Add angular velocity in radians per second as a distinct supported-time
  derivative. Define its heading, stationary, and irregular-time conventions;
  do not divide per-sample turns by a guessed global sampling rate.
- Initial geometric curvature is signed planar `d(theta) / ds`, with inverse
  coordinate units, computed on observed movement support. Reject dimensions
  other than 2 rather than silently projecting or claiming N-D curvature.
- Define endpoint, stationary, and insufficient-neighbor values explicitly as
  unsupported/NaN. Smoothing is opt-in and must state whether its window is in
  time or distance; irregular time must not silently become sample-index time.
- Defer legacy deprecation/removal to a separately documented compatibility
  decision. Defer general 3-D curvature/torsion until an explicit workflow needs it.

## Implementation tasks

- [x] Record existing outputs for the gap-free kinematics fixtures and existing
  spike/occupancy support cases before moving shared logic.
  `4c3eaf1a` (#112) and `2501a4d9` (#114).
- [x] Add a small internal observation-support resolver, using `_intervals.py`
  primitives. Resolve finite-coordinate masks, explicit boundaries, gap/epoch
  gates, original-index run bounds, and isolated sample status in one place.
  `_observation.resolve_observation_support` (`276dfe93`, #112), tested in
  `tests/test_observation_support.py`. Breaks also route through the shared
  `_intervals` gate (#121, `tests/test_interval_helpers.py`).
- [x] Keep the resolver independent of `Environment`; optional bin validity is
  a caller gate. Avoid introducing another public interval-set abstraction.
  `_observation.py` imports only `_intervals`, `_results` and `_validation`.
- [x] Fix `Environment.transitions(times=..., positions=...)` to exclude
  unsupported/outside-bin pairs without rejecting the entire recording or
  bridging removed samples. Preserve sample-pair/self-transition semantics and
  validate truly malformed indices. Define the separate explicit `bins=`
  sentinel/boundary policy in API review; do not recover lost timestamps by
  guessing from a compressed sequence.
  Inside-to-inside pairs only (`c77e3833`, #85), tested in
  `tests/environment/test_missing_data.py`. `bins=` with `-1` is still refused.
  `Environment` trajectory methods accept `break_before` (#121,
  `tests/environment/test_trajectory_breaks.py`).
- [x] Add optional support/result holders only where several aligned arrays are
  returned. Own derived arrays as read-only copies; provide `summary()` and a
  domain-specific `samples_table()` or `support_table()` where useful. Follow
  shared contracts for exports; never add artificial `unit_id` columns.
  `CumulativeDistanceResult` and `ResampledPathResult` (#112).
- [x] Implement observed step lengths and cumulative distance in
  `behavior/trajectory.py`, reusing the Euclidean step kernel and support mask.
  Keep interval-aligned and sample-aligned outputs distinguishable.
  `compute_cumulative_distance` (`dad1919e`, #112), tested in
  `tests/behavior/test_cumulative_distance.py`.
- [x] Extend `traveled_path_length` with a backward-compatible time/support path,
  or delegate to a new observed-distance function. Existing positions-only calls
  retain their documented meaning for a single continuous path.
  This shipped as a documented breaking change instead: it now takes
  `(times, positions)` (`437b153e`, #112), and positions-only calls are gone.
- [x] Add a distance-resampling function in `behavior/trajectory.py` or a small
  sibling module if that prevents the existing module from becoming unwieldy.
  Data arguments are positional; spacing, endpoint policy, and support controls
  are keyword-only. No multiprocessing or DataFrame requirement.
  `behavior/resampling.py::resample_by_distance` (`f5dc1087`, #112).
- [x] Use monotone cumulative distance and `searchsorted`/interpolation rather
  than heuristic index searches. Handle empty, singleton, duplicate-position,
  and stationary paths without division by zero.
  `f5dc1087` and `7df37b3a`, tested in `tests/behavior/test_distance_resampling.py`.
- [x] Add explicit resampling provenance and a helper/example for recovering
  original aligned signals; do not automatically interpolate discrete labels.
  `ResampledPathResult.interpolate_signal` (`bea3eafb`, #112).
- [x] Add correctly named planar turning outputs, angular velocity, and planar
  curvature. State derivative and smoothing conventions in docstrings/result
  metadata. Correct legacy N-D descriptions to the mixed-coordinate formula,
  with no numerical change; keep its migration separate.
  `fb109fe3`, `bf097d3f` and `f562ed9c` (#112), tested in
  `tests/behavior/test_planar_turning.py` and `test_legacy_turning_outputs.py`.
- [x] Audit heading/body-orientation consumers and document per-run invocation
  or reuse the support resolver where a compatible extension is appropriate.
  `f562ed9c` (#112); `heading_from_velocity(break_before=)` in `4ec2afc6`
  (#118); missing and slow samples handled in `14e8a5c3` (#120). Both heading
  functions now live in `behavior` (#121).
- [x] Export accepted names from `behavior/__init__.py`; update API snapshots,
  argument-convention tests, and the trajectory analysis guide.
  `tests/data/public_api.txt`; `aaddfead` (#112).

## Scientific invariant validation

- **Break invariance:** concatenating two recordings with an explicit boundary
  yields the same observed distance and per-run samples as separate calls.
  A short elapsed gap still breaks when `break_before` declares a boundary.
- **Missing-observation invariance:** inserting NaN coordinates or excluded
  epochs never contributes displacement, heading, or curvature across the gap.
- **Transition support:** one missing/outside position does not reject an entire
  trajectory or create a transition across it; valid sample-pair counts remain
  unchanged and malformed explicit bin input is still diagnosed.
- **Stationarity:** positive dwell with repeated positions yields zero distance;
  it does not create infinite speed or lose the stationary interval's duration.
- **Plateau provenance:** `times=[0, 1, 10, 11]`, positions along X at
  `[0, 1, 1, 2]`, and spacing `0.5` yield times `0.5, 10, 10.5` at distances
  `0.5, 1, 1.5`. The moving brackets remain original intervals `(0, 1)` and
  `(2, 3)`; the exact plateau anchor owns index 2. A stationary run's separate
  time bounds retain both observation endpoints.
- **Sampling invariance:** dense and sparse samples of a straight polyline produce
  the same distance-resampled positions and correctly interpolated source times.
- **Provenance reconstruction:** each output position/time reconstructs from its
  source bracket and weight; every bracket belongs to one usable original interval.
- **Physical scaling:** scaling coordinates by `a > 0` scales length/step by `a`
  and geometric curvature by `1 / a`; turning angles remain unchanged.
- **Curvature fixtures:** a straight path is zero on supported interior samples;
  a sufficiently sampled circle approaches signed `1 / radius`; reversing travel
  reverses signed planar curvature. Irregular timestamps do not alter geometry.
- **Angular-rate units:** a supported constant angular rotation has the same
  radians-per-second rate under irregular sampling; time scaling changes the
  rate inversely without changing geometric curvature.
- **Compatibility:** old gap-free outputs and existing spike/occupancy masks
  remain unchanged unless a separately documented bug fix intentionally changes them.
- **Legacy N-D compatibility:** the three-point XYZ example retains `pi / 4` in
  both legacy APIs, its explicitly projected XY input gives `pi / 2`, and new
  planar turning/curvature methods reject the XYZ input.

Use analytic line, circle, paused-path, and epoch-boundary fixtures. These tests
check scientific properties rather than repeat implementation expressions.

## Milestones and completion

1. Support resolver and observed distance ship with compatibility checks.
2. Distance resampling ships with reconstructable time/index provenance.
3. Honest turning quantities and the scientist workflow example ship.

The phase is complete when later phases can consume support and distance outputs
without guessing boundaries, each output has explicit units, and an example
compares intermittent recordings at different speeds. Run the focused behavior,
interval, and event-gap tests, plus required public API checks; add encoding
support tests only when shared rate-map helpers change.

## Deferred scope

No tracking-error imputation policy, filtering framework, mandatory experiment
container, automatic unit conversion, graph-constrained interpolation, or
behavioral inference model belongs in this phase. A caller may smooth a path
before resampling, but the package must not hide that scientific choice.
