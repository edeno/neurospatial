# Phase 3 — Crossings, exploration, and revisit history

**Created:** 2026-10-14
**Committed:** 2026-10-14
**Status:** Not started

[Back to PLAN.md](PLAN.md) · [Shared contracts](shared-contracts.md) ·
[Environment representation plan](../environment-representation/PLAN.md)

Scientists need to distinguish time spent somewhere from the number of times an
animal arrived there, quantify how exploration unfolds, and recover the recent
movement history preceding a choice. This phase adds occurrence-preserving
crossing outputs and exploration summaries over observed time or travel distance.
It extends existing visit records instead of redefining trials or experiments.

## Read and reuse first

- `src/neurospatial/behavior/visits.py`: `RegionVisit`, `RegionTransition`,
  `detect_region_visits`, and `region_transitions`.
- `src/neurospatial/behavior/segmentation.py`: `detect_region_crossings`,
  `_detect_region_crossings_contiguous`, and `detect_runs_between_regions`.
- `src/neurospatial/environment/cartesian.py`: `bin_sequence`,
  `bin_sequence_with_runs`, and `occupancy`.
- `src/neurospatial/environment/trajectory.py`: empirical transitions at line 646,
  self/adjacency filtering at line 815, and shared support helpers.
- `src/neurospatial/behavior/trajectory.py`: `compute_home_range`; this measures
  occupancy concentration, not fraction of the arena discovered.
- `src/neurospatial/ops/binning.py`: `regions_to_mask` for bin-center membership.
- `src/neurospatial/environment/queries.py`: `bin_sizes` for spatial weights.
- [Current subset size fallback](../../../src/neurospatial/environment/transforms.py#L527):
  verify physical measures rather than accepting approximate edge lengths as area.
- `tests/behavior/test_region_visits.py`, `tests/behavior/test_segmentation_gaps.py`,
  `tests/environment/test_transitions.py`, and public API convention tests.
- [Phase 1](phase-1-observation-support.md) for observation support, distance,
  original-index provenance, and units; `.claude/rules/public-api.md`.

Use current point lookup for count-based coverage and crossings. Physical
coverage additionally requires verified cell measures; positive `env.bin_sizes`
alone are insufficient. Current hexagonal/mesh subsets return length estimates,
so their physical-weighted slice waits for representation phase 4 or validated
caller-supplied physical weights. Other slices proceed independently; this phase
introduces no additional geometry snapshot.

## Scientist workflows

1. Compare directed crossing counts after excluding slow approaches or particular
   events, without rebuilding a path that bridges the removed observations.
2. Measure when an animal first discovers a region and how much observed travel
   it takes to visit a chosen fraction of an independently defined arena.
3. Compare revisits and choices conditioned on prior region, prior visit count,
   or time/distance since a previous encounter.
4. Use those behavioral rows as explicit covariates or event cohorts for neural
   analyses, retaining the original event and recording identities.

## Contracts and initial decisions

### Crossings and visits

- Keep `RegionVisit` and entry-to-entry `RegionTransition` semantics. Source dwell
  is part of entry-to-entry duration; departure-to-arrival runs remain a distinct
  `detect_runs_between_regions` operation.
- Add a sampled-bin crossing operation whose data arguments use
  `(times, position_bins, env, *, ...)`, with sample-aligned integer bins.
  Count only actual changes inside
  one observed run; consecutive repeats do not become crossings.
- Each crossing retains a local occurrence identifier, run identifier, original
  source/target sample indices, source/target bins, and observed crossing time.
  External subject/session labels remain caller metadata, not fabricated units.
- Invalid bins break continuity. A selected cohort or filtered crossing table
  cannot imply a new transition across omitted occurrences. Detect first, derive
  history on the full observed sequence, and filter event rows afterward.
- Crossing time is the first sample in the destination, consistent with existing
  sampled detections. Flag this resolution; do not claim a precise physical
  boundary intersection or fabricate intermediate bins skipped by sampling.
- Graph adjacency is an explicit filter, not the definition of valid movement.
  Preserve non-adjacent observed crossings and let callers label jumps/artifacts.
- Continuous polygon/proximity encounters are a separate mode operating on raw
  positions and their support. They allow independently overlapping targets;
  retain `detect_region_visits`' disjoint-bin membership contract unchanged.
- A polygon boundary policy and proximity threshold convention must be explicit.
  Start-inside observations and restarts are censored visits, not invented entries.
  Sub-frame geometric crossing interpolation is deferred from the first release.

### Counts, exposure, and directionality

- Derive raw directed crossing counts from occurrence records. Keep self-dwell,
  raw counts, row-conditioned transition probabilities, and count/exposure rates
  as separately named quantities; one must not silently replace another.
- Preserve sparse matrices and bin identities. Excluded bins still have known
  coordinates; an empty count row is not evidence of zero occupancy.
- Add top-k crossing concentration and paired-edge directionality summaries with
  explicit eligible crossings and minimum count thresholds. Return their counts
  and denominators, not only a dimensionless statistic.
- Directionality uses paired forward/reverse counts; reversing bin labels or
  coordinate axes cannot change its unsigned magnitude. Statistical null tests
  are outside this implementation plan.

### Exploration and revisits

- Coverage requires an explicit eligible bin mask or independently supplied arena
  support. Do not infer the denominator from observed occupied bins, particularly
  when `Environment.from_samples` inferred its active mask from those observations.
- Initial coverage is sample-observed bin discovery. Do not count unobserved bins
  along straight chords between samples. Describe sampling-resolution limits.
- Support both count-weighted bin coverage and physical measure-weighted coverage.
  Physical weighting requires verified geometry/measure capability or supplied
  weights with declared measure dimension, units and bin alignment. Validate
  positive finite eligible measures, but do not mistake positivity or ambient
  coordinate dimension for proof of physical area/volume. Graph lengths need an
  explicitly stated denominator interpretation.
- Refuse physical weighting when measures are approximations or their meaning
  is unknown. In particular, current hexagonal/mesh `_SubsetLayout.bin_sizes`
  supplies edge lengths and an isolated-bin fallback of `1.0`; it cannot provide
  area weights. Gate only this unsupported weighted slice on representation
  phase 4, or accept independently validated supplied weights. Count coverage
  does not inherit that dependency. Report weight source, units and dimension.
- Curves include first observed discovery, cumulative eligible fraction, and
  observed distance/time axes. Observed time sums valid intervals; wall-clock
  event times are retained separately and include pauses by definition.
- Unknown/never observed first-arrival values are censored or missing, not zero.
  A requested coverage fraction not reached returns an explicit unreached status.
- Revisit history is derived from distinct visits/crossings, never frame counts.
  Complete within-run history is the default. An explicit "since last observed"
  mode may include a recording pause but reports incomplete/censored history:
  intervening visits during that pause are unknown.
- Distance since last observed encounter means observed distance. Report missing
  movement support and do not claim total intervening travel is known.
- Prior visit counts and prior-source frequencies use events strictly preceding
  the current entry. The current event and future events cannot enter predictors.

## Implementation tasks

- [ ] Add sampled-bin crossing records or an array-backed result in
  `behavior/visits.py`; reuse phase 1 support rather than compressing arrays.
- [ ] Provide a `crossings_table()` export and immutable owned derived arrays
  where a holder is useful. Identifiers describe crossings, bins, targets, and
  runs; follow shared contracts and do not add a fake `unit_id`.
- [ ] Add continuous target encounters in a small behavior module or a clear
  separate operation in `visits.py`. Use existing region geometries where
  appropriate and accept direct proximity targets without creating a layout.
- [ ] Preserve current disjoint-region visit detection and censoring behavior;
  share membership/run logic only when existing outputs remain equivalent.
- [ ] Add aggregation from crossing records to sparse counts. Provide explicit
  diagonal, adjacency, cohort, and normalization choices without rereading paths.
- [ ] Document existing `env.transitions` as sample-pair transitions. Fix or
  explicitly isolate outside-bin rejection and gap-provenance limitations where
  touched; do not silently change its counts to the new crossing interpretation.
- [ ] Add concentration and paired-edge directionality functions returning
  denominators and selected-edge counts; use algorithm parameters keyword-only.
- [ ] Add exploration operations in `behavior/exploration.py`: coverage curve,
  first arrival, and distance/time to requested coverage. Names are provisional.
- [ ] Require the eligible mask at the main coverage boundary; validate its
  shape, positive total weight, empty support, and coordinate/bin compatibility.
- [ ] Resolve physical weights through verified cell geometry/capability or
  explicitly supplied aligned measures. Refuse unknown or approximate measures
  with a fix naming supplied weights or the required representation capability;
  do not infer proof from a layout tag or use nearest-center remapping.
- [ ] Add revisit intervals and causal history features from full event records.
  Return validity/censoring and original event identifiers with the arrays.
- [ ] Keep history features composable with generic covariates; do not add a
  classifier, route-memory verdict, or domain-specific novelty model.
- [ ] Export accepted APIs; update argument/public API snapshots and the
  trajectory guide with occupancy-versus-crossings examples.
- [ ] Add a runnable example comparing exploration and revisit history for two
  paths having equal occupancy but different arrival order.

## Scientific invariant validation

- **Occupancy versus crossings:** long dwell increases exposure while leaving
  entry count unchanged; doubling tracking frame rate does not double crossings.
- **Continuity:** an outside/invalid sample, epoch gap, or explicit recording
  boundary cannot create a crossing or connect unrelated source/target entries.
- **Filtering order:** excluding event rows after history construction does not
  manufacture consecutive transitions or change the retained rows' prior counts.
- **Membership:** independent overlapping encounter zones work without weakening
  the disjoint-region visit API. Start-inside and gap-restarted visits are censored.
- **Coverage denominator:** changing the animal's observed path cannot change
  the eligible arena or its total weight. Unequal-area bins give correct weighted
  fractions; ineligible bins never advance coverage.
- **Physical-measure guard:** three equal-area 17 cm hexagons, with two adjacent
  and one isolated after selection, currently return subset weights `[17, 17, 1]`.
  Reject these as physical areas. Visiting the isolated cell gives `1 / 3` with
  supplied true equal areas or verified post-refactor geometry, never `1 / 35`.
  Count-weighted coverage still succeeds before the representation fix.
- **First arrival:** known path order gives exact sampled discovery times and
  distances; unreachable requested fractions are reported as unreached.
- **History causality:** extending a trajectory after a selected event cannot
  change that event's preceding counts, source frequency, or revisit predictors.
- **Direction:** swapping source and destination leaves unsigned directionality
  unchanged and reverses any signed preference; pure dwell adds no directed edge.
- **Relabeling:** a consistent permutation of bin identifiers permutes count
  outputs but leaves scalar coverage/concentration measures unchanged.
- **Break equivalence:** separately detected recordings with external labels
  agree with one common-clock call containing explicit boundaries.

Use tiny paths with known dwell and crossing order, unequal-size bins, censored
visits, overlapping polygons, and short-gap recording boundaries. Reuse current
visit fixtures and add only tests for new scientific behavior.

## Milestones and completion

1. Crossing/encounter outputs ship with censoring and immutable occurrence identity.
2. Sparse crossing summaries and independently defined coverage curves ship.
3. Causal revisit/history outputs and the scientist example ship.

The phase is complete when filtering events does not reconstruct movement,
coverage states its denominator, and history features remain linked to the full
observed sequence. Run focused visits, segmentation, transitions, exploration,
and public API checks; broader checks are required only for shared helper changes.

## Deferred scope

No paper-specific jump dictionary, automatic approach categorization, universal
trial ontology, latent memory inference, statistics/surrogate framework, or
coverage derived from a fitted utilization distribution belongs in this phase.
