# Continuous behavior and neural analysis implementation tasks

**Created**: 2026-10-14
**Committed**: 2026-10-14
**Status**: Partially implemented. Area 1 observation support is delivered
(PRs #112, #114, #120), and `break_before` reaches every function that takes
`max_gap` (#121). Area 5 continuous event alignment and static/egocentric feature
sampling are delivered (#118); interval states remain. Area 6 same-environment
conditioned comparisons are delivered (#119). Area 2 is partly covered by
environment stages A–E (#67–#76). Areas 3 and 4 have not started, and M0 and M5
are partial.
**Status reviewed**: 2026-10-20
**Source**: [PLAN.md](PLAN.md)

This tracker covers six accepted workstreams. Detailed tasks, inventories, and
validation live in phase files. Checkboxes record implementation completion,
not creation of a plan. A box is checked only when both the implementing code
and its tests exist; evidence follows each item. An unchecked item with a note
is partial.

## M0 Contracts and baseline

- [ ] Review [shared contracts](shared-contracts.md); finalize API keywords,
  result/export names, and domains against existing conventions.
  *Partial:* finalized for the delivered slices (`break_before`, `value_kinds`,
  `window_coverage`, `samples_table`/`aligned_table`, recorded in
  `.claude/rules/public-api.md`). Names for areas 3–4 and states remain.
- [ ] Capture current behavior for invalid coordinates, recording boundaries,
  duplicate times, outside bins, stationary movement, and incomplete windows.
  *Partial:* outputs were pinned per slice before the migration (`4c3eaf1a` turning
  and step lengths, #112; `2501a4d9` speed and traveled length, #114). There is
  no consolidated baseline record.
- [ ] Include review counterexamples: mutable snapshots/result context, unknown
  track containment, incorrect subset measures, permissive reversed-path
  matches, angular summaries, aggregating rate mappings, and mixed N-D turns.
  *Partial:* five are tested. Result context is in
  `tests/encoding/test_rate_estimation_context.py` (#119). Track containment is
  in `tests/ops/test_feature_sampling.py` (#118). Angular wrap is `3344bfc9`
  (#118). Aggregating mappings are in `test_compare_spatial_rates.py` (#119).
  Mixed N-D turns are in `test_legacy_turning_outputs.py` (#112). Subset measures
  and reversed-path matches remain (areas 3–4).
- [ ] Identify intentional scientific changes and migration/changelog needs.
  *Partial:* each breaking change in the delivered slices has a CHANGELOG
  "Unreleased → Breaking" entry. Changes for areas 3–4 and states are pending.
- [ ] Define deterministic fixtures for maze, free-foraging, and changing-state
  workflows, with independent behavioral and neural random seeds.
  *Partial:* the linear-track outbound/inbound fixture with separately seeded
  spikes is in `docs/user-guide/workflows.md` (#119). The free-foraging box is in
  `docs/user-guide/behavior-around-events.md` (#118). A changing-state fixture
  is absent.
- [x] Keep representation phases 1–4 in their own baseline-governed stream.
  Evidence: [environment-representation](../environment-representation/PLAN.md),
  merged in #67, with `scripts/environment_baseline.py`.

## Area 1 Observation support and movement primitives

Source: [phase-1-observation-support.md](phase-1-observation-support.md).

- [ ] Reuse validity, epochs, gaps, and explicit boundaries across operations.
  *Partial:* `_observation.resolve_observation_support` (`276dfe93`, #112;
  `tests/test_observation_support.py`) is used by behavior kinematics and
  navigation (#112, #114, #120), continuous events (#118) and `labels_to_epochs`
  (#119). Encoders, significance tests, decoders, segmentation, visits, goal
  metrics, `Environment` trajectory methods, `events.add_positions` and
  `generate_population_spikes` accept `break_before` through the shared
  `_intervals` gate (#121; `tests/encoding/test_declared_breaks.py`,
  `tests/decoding/test_declared_breaks.py`, `tests/behavior/test_declared_breaks.py`).
  Remaining: `behavior.distance_to_reward` (position interpolation), `behavior.time_efficiency` and `ops.visibility_occupancy` take neither `max_gap` nor `break_before`.
- [x] Fix missing-position transitions and movement helpers without bridging gaps.
  Evidence: `Environment.transitions(times=, positions=)` counts only
  inside-to-inside pairs (`c77e3833`, #85; `tests/environment/test_missing_data.py`).
  `compute_speed` is `3968625c` (#114). `traveled_path_length` is `437b153e`
  (#112). Curvature, MSD and `approach_rate` are `d7baec8f`, `058b83d4` and
  `001bb0a5` (#120).
- [x] Add observed cumulative distance and distance resampling with exact
  source/time provenance and deterministic stationary handling.
  Evidence: `compute_cumulative_distance` (`dad1919e`) and `resample_by_distance`
  / `ResampledPathResult.interpolate_signal` (`f5dc1087`, `bea3eafb`), in #112.
  Tests: `tests/behavior/test_cumulative_distance.py` and
  `test_distance_resampling.py`.
- [x] Clarify turning angle, angular velocity, curvature units, and dimensions;
  document the mixed-coordinate legacy N-D formula and preserve its output
  until an explicit numerical migration is approved.
  Evidence: `compute_planar_turn_angles`, `compute_planar_curvature` and
  `compute_movement_angular_velocity` (`fb109fe3`, `bf097d3f`), with the legacy
  formula documented in `f562ed9c` (#112). Tests: `tests/behavior/test_planar_turning.py`
  and `test_legacy_turning_outputs.py`. `compute_trajectory_curvature` now takes
  `(times, positions)` with unchanged gap-free output (`d7baec8f`, #120).
- [ ] Verify adapter support and a runnable preprocessing example.
  *Partial:* an executed example of resampling across recordings and frame rates
  is in `docs/user-guide/trajectory-and-behavioral-analysis.md` (`aaddfead`, #112).
  The plan does not say which adapters it means, and no adapter changes were
  found.

## Area 2 Environment correctness and follow-on contracts

Source: [phase-2-environment-contracts.md](phase-2-environment-contracts.md).

- [x] Complete applicable representation phases 1–4 and their baseline checks.
  Evidence: [environment-representation](../environment-representation/PLAN.md)
  is complete and merged (#67).
- [ ] Add optional scientific bin IDs and validated derivation correspondences.
  *Partial:* combined environments expose `Environment.bin_sources` (#71), and
  `compare_spatial_rates(bin_mapping=...)` validates caller-supplied one-to-one
  correspondences by geometry (#119). Optional external `bin_ids` and
  subset/rebin mapping outputs remain.
- [ ] Enforce measure dimensions and supported transfer/conservation rules.
  *Partial:* measure dimensions (#67, per-bin `measure_dims` in #101), exact
  affine measures (#76) and
  similarity scaling of bridge conductance (#71) exist. Documented
  mass/density/intensive transfer rules remain.
- [ ] Define path-only connections, movement operators, and diffusion faces.
  *Partial:* zero-conductance path-only bridges exist in `Environment.combine`
  (#71; `tests/environment/test_combine.py`). No directed movement/count
  operator input was found.
- [x] Implement physical-boundary needs through the existing arena roadmap.
  Evidence: arena stage A (#68), physical border scores (#73) and wall-aware
  visibility (#74). See [environment-rebin](../environment-rebin/PLAN.md).
- [ ] Validate feature alignment and immutable state intervals with area 5.
  *Partial:* static feature arrays align by bin under verified containment
  (#118). Interval states have not started.
- [ ] Gate containment on a verified footprint and environment snapshot/cache
  reuse on immutable-value stage B; allow independent fixed-geometry features.
  *Partial:* feature sampling refuses layouts without a known domain
  (`ops/feature_sampling.py`, #118; track tests in
  `tests/ops/test_feature_sampling.py`). Stage B immutable environments are
  merged (#70). Snapshot selection and cache reuse by state are not implemented.

## Area 3 Exploration and occurrence records

Source: [phase-3-exploration-and-crossings.md](phase-3-exploration-and-crossings.md).
Not started: no crossing-record, coverage, first-arrival or revisit code exists.

- [ ] Add bin crossings/encounters, reusing region visits and censoring.
  Visits and segmentation accept `break_before` (#121).
- [ ] Derive crossing counts and directionality independently of self-samples.
- [ ] Add eligible-space coverage, first arrivals, discovery, and revisits.
- [ ] Require verified physical measures or validated supplied weights for
  weighted coverage; reject unsupported hex/mesh subsets without blocking counts.
- [ ] Add causal previous-location and time/distance-since-visit features.
- [ ] Document occupancy/transitions/crossings/coverage and an intermittent
  exploration workflow.

## Area 4 Ordered paths

Source: [phase-4-path-analysis.md](phase-4-path-analysis.md).
Not started: no `behavior/paths.py` or ordered path-matching code exists.

- [ ] Add sparse ordered matches with explicit thresholds, direction,
  self-exclusion, and support policies.
- [ ] Separate index orientation from supported tangent compatibility and test
  reversals under permissive spatial tolerances and small loops.
- [ ] Add named repertoire-comparison variants and reference/candidate rules;
  include path-bundle distance as one method.
- [ ] Extract preceding/matched/following segments with support and causality.
- [ ] Integrate or clarify existing similarity coordinate/label semantics.
- [ ] Verify invariants and storage scaling in two navigation examples.

## Area 5 Continuous events and spatial features

Source: [phase-5-event-alignment-and-features.md](phase-5-event-alignment-and-features.md).

- [x] Add continuous event alignment with identity, feature validity, partial
  windows, and explicit interpolation.
  Evidence: `events.align_continuous_to_events` (`957c4b34`, `377839ec`,
  `3ac44e16`, #118). Tests: `tests/events/test_continuous_sampling.py`,
  `test_continuous_coverage.py` and `test_continuous_alignment.py`.
- [x] Add per-feature value kinds and reductions: linear sums, circular
  sine/cosine sums with undefined-result handling, and categorical counts.
  Evidence: `68fc049e`, `377839ec` and `3344bfc9` (#118). Tests:
  `tests/events/test_continuous_features.py` and `test_continuous_alignment.py`.
- [x] Add static/egocentric feature sampling with units, frames, and masks.
  Evidence: `ops.sample_spatial_features` and `ops.sample_egocentric_features`
  (`d217cb34`, `34c2f9f6`, #118). Tests: `tests/ops/test_feature_sampling.py`.
  Only containment is available. The plan's optional nearest mode, per-feature
  kinds and time-support gating are not part of the sampling API.
- [ ] Select immutable states by intervals, preserving uncovered time as unknown.
  Not started.
- [x] Add domain-correct exports without changing PSTH/rate-map axes.
  Evidence: `ContinuousEventResult.events_table`, `aligned_table`,
  `time_course_table` and `category_counts_table` (`bac67ea3`, #118). Tests:
  `tests/events/test_continuous_exports.py`.
- [ ] Verify chunks/event/state cutoffs and encounter/change examples.
  *Partial:* chunked/streamed parity (`fe4a668d`) and event-window cutoffs are
  tested, and the reward example is in
  `docs/user-guide/behavior-around-events.md` (#118). State cutoffs and the
  encounter/change examples remain.

## Area 6 Conditioned neural workflows

Source: [phase-6-neural-comparisons.md](phase-6-neural-comparisons.md).

- [x] Convert labels to contiguous valid epochs with explicit allocation.
  Evidence: `behavior.labels_to_epochs` (`5f9f3777`, `5355c731`, #119). Tests:
  `tests/behavior/test_labels_to_epochs.py`.
- [x] Reuse existing condition-specific encoding and decoder fit/predict.
  Evidence: `tests/encoding/test_conditioned_rates.py` (`57efe6e5`, #119),
  including decoder reuse of condition epochs. The executed workflow is in
  `docs/user-guide/workflows.md` (`ad10874f`).
- [x] Capture immutable estimation-time spatial context in rate results and
  preserve it through indexing/exports; reject unknown legacy provenance unless
  verified source context is supplied.
  Evidence: `3f5f599d` (#119). Rate results and NWB `env=` overrides refuse
  environments that do not match their bins. Tests:
  `tests/encoding/test_rate_estimation_context.py` and
  `tests/nwb/test_fields_roundtrip.py`.
- [x] Compare aligned units/bins/common support with occupancy diagnostics.
  Evidence: `encoding.compare_spatial_rates` and `RateComparisonResult`
  (`8a3e7980`, `0b0ed9c9`, #119). Tests: `tests/encoding/test_compare_spatial_rates.py`
  and `test_rate_comparison_result.py`. Population follow-on: the per-location
  `population_vector_correlation` is `95973398` (#120). It pairs
  `SpatialRatesResult` inputs by `unit_ids`, and assembly activation matches
  columns by label (#121; `tests/encoding/test_population_vector_identity.py`,
  `tests/decoding/test_assembly_unit_identity.py`).
- [x] Accept only one-to-one comparison mappings initially and reject
  coarsening/overlap; document separate re-estimation on common bins.
  Evidence: `test_merging_bins_is_refused_because_rates_cannot_be_averaged` and
  `test_splitting_a_bin_is_refused`. Re-estimation is documented under "A
  comparison neurospatial refuses" in `docs/user-guide/workflows.md` (#119).
- [ ] Expose insufficient coverage and supplied speed/heading differences.
  *Partial:* insufficient common support returns NaN with a reason
  (`test_too_few_common_bins_is_nan_with_a_reason`). Duration, occupied bins
  and speed are compared in `docs/user-guide/workflows.md`. A heading
  diagnostic remains.
- [ ] Verify holdouts, causal labels, and matching spike/dwell support without
  adding a general covariate model.
  *Partial:* held-out and transfer decoding are executed in `workflows.md`, and
  spike/dwell gating is in `test_conditioned_rates.py` (#119). Causal (prefix-only)
  label checks remain and depend on areas 3–4.

## M5 Integrated validation and release

- [ ] Run T/W maze crossings/history, conditioned fields, and held-out decoding.
  *Partial:* linear-track conditioned fields and held-out decoding are in
  `workflows.md` (#119). Crossings/history (area 3) remain.
- [ ] Run intermittent foraging coverage/revisits/paths and event profiles.
  *Partial:* event profiles with a tracking dropout are in
  `behavior-around-events.md` (#118). Coverage, revisits and paths remain.
- [ ] Run changing-state correspondence/features and neural comparisons.
  Not started. Only `compare_spatial_rates(topology="may_differ")` exists for a
  door change on the same bins.
- [ ] Verify scientific invariants, chunk agreement, units, ownership, and exports.
  *Partial:* covered by the per-slice tests above for areas 1, 5 and 6.
- [ ] Update examples, user guides, API exports/snapshots, and changelog.
  *Partial:* done for the delivered slices (`tests/data/public_api.txt`,
  `CHANGELOG.md`, the user guides above).
- [ ] Run required targeted and repository checks, including geometry baselines.
- [ ] Confirm six-area scope without deferred statistical/modeling frameworks.
