# Continuous behavior and neural analysis implementation tasks

**Created**: 2026-10-14
**Committed**: 2026-10-14
**Status**: Not started
**Source**: [PLAN.md](PLAN.md)

This tracker covers six accepted workstreams. Detailed tasks, inventories, and
validation live in phase files. Checkboxes record implementation completion,
not creation of a plan.

## M0 Contracts and baseline

- [ ] Review [shared contracts](shared-contracts.md); finalize API keywords,
  result/export names, and domains against existing conventions.
- [ ] Capture current behavior for invalid coordinates, recording boundaries,
  duplicate times, outside bins, stationary movement, and incomplete windows.
- [ ] Include review counterexamples: mutable snapshots/result context, unknown
  track containment, incorrect subset measures, permissive reversed-path
  matches, angular summaries, aggregating rate mappings, and mixed N-D turns.
- [ ] Identify intentional scientific changes and migration/changelog needs.
- [ ] Define deterministic fixtures for maze, free-foraging, and changing-state
  workflows, with independent behavioral and neural random seeds.
- [ ] Keep representation phases 1–4 in their own baseline-governed stream.

## Area 1 Observation support and movement primitives

Source: [phase-1-observation-support.md](phase-1-observation-support.md).

- [ ] Reuse validity, epochs, gaps, and explicit boundaries across operations.
- [ ] Fix missing-position transitions and movement helpers without bridging gaps.
- [ ] Add observed cumulative distance and distance resampling with exact
  source/time provenance and deterministic stationary handling.
- [ ] Clarify turning angle, angular velocity, curvature units, and dimensions;
  document the mixed-coordinate legacy N-D formula and preserve its output
  until an explicit numerical migration is approved.
- [ ] Verify adapter support and a runnable preprocessing example.

## Area 2 Environment correctness and follow-on contracts

Source: [phase-2-environment-contracts.md](phase-2-environment-contracts.md).

- [ ] Complete applicable representation phases 1–4 and their baseline checks.
- [ ] Add optional scientific bin IDs and validated derivation correspondences.
- [ ] Enforce measure dimensions and supported transfer/conservation rules.
- [ ] Define path-only connections, movement operators, and diffusion faces.
- [ ] Implement physical-boundary needs through the existing arena roadmap.
- [ ] Validate feature alignment and immutable state intervals with area 5.
- [ ] Gate containment on a verified footprint and environment snapshot/cache
  reuse on immutable-value stage B; allow independent fixed-geometry features.

## Area 3 Exploration and occurrence records

Source: [phase-3-exploration-and-crossings.md](phase-3-exploration-and-crossings.md).

- [ ] Add bin crossings/encounters, reusing region visits and censoring.
- [ ] Derive crossing counts and directionality independently of self-samples.
- [ ] Add eligible-space coverage, first arrivals, discovery, and revisits.
- [ ] Require verified physical measures or validated supplied weights for
  weighted coverage; reject unsupported hex/mesh subsets without blocking counts.
- [ ] Add causal previous-location and time/distance-since-visit features.
- [ ] Document occupancy/transitions/crossings/coverage and an intermittent
  exploration workflow.

## Area 4 Ordered paths

Source: [phase-4-path-analysis.md](phase-4-path-analysis.md).

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

- [ ] Add continuous event alignment with identity, feature validity, partial
  windows, and explicit interpolation.
- [ ] Add per-feature value kinds and reductions: linear sums, circular
  sine/cosine sums with undefined-result handling, and categorical counts.
- [ ] Add static/egocentric feature sampling with units, frames, and masks.
- [ ] Select immutable states by intervals, preserving uncovered time as unknown.
- [ ] Add domain-correct exports without changing PSTH/rate-map axes.
- [ ] Verify chunks/event/state cutoffs and encounter/change examples.

## Area 6 Conditioned neural workflows

Source: [phase-6-neural-comparisons.md](phase-6-neural-comparisons.md).

- [ ] Convert labels to contiguous valid epochs with explicit allocation.
- [ ] Reuse existing condition-specific encoding and decoder fit/predict.
- [ ] Capture immutable estimation-time spatial context in rate results and
  preserve it through indexing/exports; reject unknown legacy provenance unless
  verified source context is supplied.
- [ ] Compare aligned units/bins/common support with occupancy diagnostics.
- [ ] Accept only one-to-one comparison mappings initially and reject
  coarsening/overlap; document separate re-estimation on common bins.
- [ ] Expose insufficient coverage and supplied speed/heading differences.
- [ ] Verify holdouts, causal labels, and matching spike/dwell support without
  adding a general covariate model.

## M5 Integrated validation and release

- [ ] Run T/W maze crossings/history, conditioned fields, and held-out decoding.
- [ ] Run intermittent foraging coverage/revisits/paths and event profiles.
- [ ] Run changing-state correspondence/features and neural comparisons.
- [ ] Verify scientific invariants, chunk agreement, units, ownership, and exports.
- [ ] Update examples, user guides, API exports/snapshots, and changelog.
- [ ] Run required targeted and repository checks, including geometry baselines.
- [ ] Confirm six-area scope without deferred statistical/modeling frameworks.
