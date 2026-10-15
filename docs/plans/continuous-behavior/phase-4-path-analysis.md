# Phase 4 — Ordered path matching and repeated-path analysis

**Created:** 2026-10-14
**Committed:** 2026-10-14
**Status:** Not started

[Back to PLAN.md](PLAN.md) · [Shared contracts](shared-contracts.md) ·
[Environment representation plan](../environment-representation/PLAN.md)

Scientists need to compare spatial paths independently of running speed, locate
repeated segments in free behavior, and recover the history surrounding shared
routes. This phase adds ordered continuous-coordinate matching and repertoire
comparisons. Path-bundle distance is one explicit method within that family,
not a new ontology for all behavior or an automatic route-clustering system.

## Read and reuse first

- `src/neurospatial/behavior/segmentation.py`: `trajectory_similarity` at line
  2199 and `_dtw_distance` at line 2392; understand their distinct semantics.
- `src/neurospatial/behavior/navigation.py`: `graph_turn_sequence`; it reduces
  repeated bin occurrences and is not an ordered continuous-path matcher.
- `src/neurospatial/behavior/trajectory.py`: turn/step primitives.
- `src/neurospatial/ops/alignment.py` and `ops/transforms.py`: coordinate-frame
  alignment belongs upstream; matching must not silently rotate one route.
- `src/neurospatial/_results.py`: owned read-only arrays and scalar summaries.
- [Phase 1](phase-1-observation-support.md): distance resampling, original-clock
  provenance, observation support, coordinate units, and recording boundaries.
- [Phase 3](phase-3-exploration-and-crossings.md): occurrence identity, cohort
  filters, and causal movement history.
- `.claude/rules/public-api.md`, public API tests, and the shared contracts.

Algorithm reference: `HermundstadLab/pathbundle`, commit
`eae70655747a620b7d7dc6d6e71c35b28ab46562`, especially `src/bundle.py` and
`src/equal.py`. Study the candidate-search and counting ideas; do not vendor or
copy code. That snapshot has no declared license and assumes study-specific
DataFrame indices, pixel units, and unbroken concatenated recordings.

## Scientist workflows

1. Find repeated route segments across sessions while controlling speed through
   physical-distance resampling and preserving recording boundaries.
2. Distinguish paths that share a location from paths that share an extended,
   ordered direction of travel; compare repertoire changes across epochs.
3. Extract preceding/shared/following segments for matched routes, producing
   inputs for neural covariates or external held-out prediction models.

## Contracts and initial decisions

### Inputs and comparison geometry

- Time-dependent functions use `(times, positions, *, ...)` and support/provenance
  from phase 1; shape-only functions accept positional continuous `positions`.
  Cross-recording comparisons use
  explicit separate inputs or run/recording labels; no session class is required.
- Shape comparisons first use common Euclidean arclength spacing. Require equal
  step sizes or resample explicitly; do not silently warp distance coordinates.
- Positions must share a declared coordinate frame and compatible spatial units.
  Coordinate registration is an explicit upstream operation. Translation,
  rotation, and bin relabeling cannot be hidden in a matching score.
- No `Environment` is required for geometric matching. Environments may supply
  masks, regions, and frame metadata without quantizing positions to bin centers.
- Keep physical tolerance, minimum matched length, maximum extension length,
  temporal orientation, moving-direction tolerance, and self-exclusion distinct.
  Names are provisional and algorithm parameters are keyword-only.

### Ordered segment matching

- Build spatial-neighbor candidates with SciPy `cKDTree`/`KDTree`, then compare
  ordered samples. Forward index progression is the default temporal orientation;
  it does not establish compatible physical movement by itself.
- Gate every compared moving step by local tangent compatibility as well as
  positional tolerance. Tangents come from nonzero Euclidean displacements of
  supported neighboring samples within each run. Compute the unsigned angle
  from normalized N-D vector dot products, clamped to `[-1, 1]`; do not apply
  the legacy XY-cross/all-dimension-dot turning kernel or invent a signed N-D angle.
- The default gate requires positive normalized dot product, equivalently angle
  strictly below `pi / 2`. Optional provisional `max_direction_difference` is in
  radians with `0 < value <= pi / 2` and permits a stricter bound. Acceptance is
  strict; perpendicular/opposing tangents and undefined stationary directions
  cannot pass. Do not derive a tangent through a run, gap or epoch boundary.
- An explicitly selected reversed temporal comparison advances the candidate
  backward, labels that orientation, and maps original indices correctly. Compare
  tangents along the chosen index progression: at candidate index `j`, the
  reversed step is `positions[j - 1] - positions[j]`, not the outgoing forward
  tangent at `j`. Reversed comparison and bidirectional extension are separate.
- This is a local moving-direction contract. Self-retracing or complex globally
  reversed routes can contain compatible local segments; do not promise a verdict
  on whole-route winding. A single colocated sample provides no moving match.
- Seed radius and continuation tolerance may differ. Document inclusive/exclusive
  comparisons and apply the same choice to candidates and extension.
- Extend only while both observations remain supported, positional tolerance
  holds and the compared step passes the moving-direction gate.
  Cap both paths at run, requested epoch, and input boundaries. A match may compare
  two sessions but may never travel through the join between sessions.
- Offer explicit forward-only versus bidirectional extension. Forward-only can
  ship first; if bidirectional is absent, reject that request rather than emulate
  it with undocumented offsets.
- Same-recording self-exclusion uses explicit overlapping source intervals or a
  documented time/distance separation. It must not be implicitly set to maximum
  match length. Equal sample indices in different recordings are unrelated.
- Return sparse segment records: reference/candidate identities, run identities,
  start/stop indices, temporal orientation, matched physical length, and distance
  summary. Record the angular threshold/gate in metadata. Map back to original
  indices/times through phase 1 provenance.
- Canonicalize duplicates from adjacent seeds using a documented segment rule,
  or expose raw seed matches distinctly. Do not count each seed as a new visit.

### Path-repertoire comparison

- Separate matching from aggregation so researchers can inspect which segments
  contribute to a score and reuse the matches with another estimator.
- Implement an explicitly named revised count-based path-bundle method first.
  Record the estimator variant, reference selection, overlap suppression,
  self-count rule, threshold lengths, and cohort in the result metadata.
- Select nonoverlapping reference starts with an explicit deterministic rule;
  suppress overlapping candidate occurrences separately. The authors' released
  code suppresses candidates while using all eligible reference points, which
  is a distinct variant and must not be silently treated as identical.
- For count-based comparison, define `f_ij` as matched occurrence counts from
  eligible references in interval `i` to eligible candidates in interval `j`;
  use `1 - (f_ij + f_ji) / (2 * sqrt(f_ii * f_jj))` only with its documented rules.
- Return the four counts, eligible reference counts, and unsupported/empty
  statuses alongside distance. Do not assume every parameterization is a metric
  or clip out-of-range values to hide violated normalization assumptions.
- A historical length-weighted measure and compatibility with the authors' exact
  all-reference variant are deferred unless needed for reproducibility. Reserve
  separate variant names; never change one method's formula silently.
- Existing Jaccard, Hausdorff, and DTW functions remain available for their
  distinct questions. Document that Pearson correlation of arbitrary bin numbers
  is not a geometric invariant and cannot substitute for ordered path matching.

### Context extraction

- Extract pre/matched/post windows from sparse match records, with explicit time
  or distance coordinates and original occurrence identities.
- Each window is contained in one observed run; partial/unsupported windows carry
  a mask or are rejected by an explicit complete-window policy.
- Return aligned arrays and group labels for external modeling. Do not infer a
  memory state, fit nearest-neighbor predictors, cluster latent routes, or use
  future movement when constructing a preceding-history feature.
- Forward-match membership and whole-recording repertoire labels are post-hoc.
  Selecting a match by its future segment then extracting a pre-window does not
  produce a causal route predictor. Predictive examples use prefix-only label
  assignment and fit their reference repertoire within training folds.

## Implementation tasks

- [ ] Create `behavior/paths.py` with array-first matching kernels and a small
  owned result. Keep the module separate from trial/lap segmentation.
- [ ] Design `PathMatches` or another provisional result with sparse records and
  `matches_table()`, `summary()`, and a focused plot if useful. Do not retain raw
  input arrays or invent unit identities for paths.
- [ ] Implement candidate search and bounded forward extension. Use phase 1 run
  bounds to cap both sides; validate coordinates before building the neighbor tree.
- [ ] Implement supported nonzero tangents and the normalized N-D dot-product
  gate at every accepted step. Handle roundoff, undefined directions and strict
  perpendicular rejection; reversed progression selects the correct source step.
- [ ] Add distinct self-overlap, temporal/distance exclusion, temporal orientation,
  angular tolerance and minimum-length controls. Expose the reference/candidate
  selections as arrays so results can be audited and reproduced.
- [ ] Implement deterministic duplicate handling and original-provenance mapping.
  Store lengths as safe integer step counts or physical floats, not narrow
  `int16` counters whose validity depends on an unstated maximum length.
- [ ] Bound candidate batches to control temporary memory; avoid a full time-by-
  time distance matrix. Allow explicit reference subsampling while recording it.
- [ ] Add count-based path-bundle aggregation as a separately named method;
  implement reference and candidate overlap suppression independently.
- [ ] Compute symmetric epoch comparisons once where the selected estimator
  permits it. Return counts and status diagnostics before adding plotting helpers.
- [ ] Add context-window extraction using the generic alignment/support work
  where applicable; keep time-based and distance-based windows distinguishable.
- [ ] Export accepted names and update public API snapshots and conventions.
- [ ] Document the differences among occupancy overlap, pairwise DTW, repeated
  segment matches, and repertoire distance in the trajectory analysis guide.
- [ ] Add a runnable example with repeated directional routes, reversals,
  crossings, pauses, and changed speed; use it as a scientific demonstration,
  not a benchmark against one paper's particular arena.

## Scientific invariant validation

- **Ordered geometry:** identical moving paths pass both positional and tangent
  gates. Known ordered crossings and diverging routes have auditable endpoints;
  no single-position seed satisfies a positive matched-length requirement.
- **Physical direction:** coincident reverse straight traversals and oppositely
  traversed, well-sampled circles seeded at the same position fail the forward
  tangent gate even when positional radius exceeds their diameter. Correctly
  oriented backward comparisons can pass and retain reversed source indices.
  Complex self-retracing paths are checked locally, without a global winding claim.
- **Angular domain:** perpendicular/opposing tangents fail the strict default;
  a stricter threshold rejects its boundary. Test normalized vectors in 2D/3D,
  roundoff clamping, stationary/zero-length steps and run-end undefined tangents.
- **Support:** matching across recordings is allowed, extension across a recording
  boundary is forbidden. The same applies to NaNs and excluded epochs.
- **Speed:** differing speeds produce the same shape matches after distance
  resampling while their original times remain different and reconstructable.
- **Rigid transforms:** applying the same translation/rotation to both paths
  leaves Euclidean scores unchanged. Scaling both positions and physical
  thresholds scales matched length without changing occurrence counts.
- **Index identity:** equal numeric indices in different recordings do not cause
  self-exclusion. Permuting bin labels does not affect continuous path matches.
- **Overlap accounting:** nearby seeds do not multiply occurrence counts under
  the selected suppression rule; reference and candidate choices are testable
  separately. Diagonal/empty comparisons have explicit estimator behavior.
- **Formula diagnostics:** analytic repeated-route repertoires produce the
  documented counts and symmetry; normalization failures are visible rather than
  silently clipped. Do not test only the final scalar.
- **History:** extending later behavior cannot alter extracted preceding windows;
  unsupported windows are not filled through gaps or borrowed from another run.
- **Predictive labels:** changing held-out or future segments cannot alter a
  prefix-only label or the references fitted on training data; post-hoc match
  cohorts are explicitly identified as descriptive.
- **Scale:** measure bounded memory on long synthetic repeated/irregular paths;
  compare to a simple exhaustive reference only on tiny fixtures.

## Milestones and completion

1. Sparse ordered segment matching ships with explicit direction and boundaries.
2. Auditable revised count-based repertoire distance ships with reference policy.
3. Context extraction and the cross-speed scientist example ship.

The phase is complete when users can inspect contributing segment occurrences,
compare recordings without losing support, and choose a measure whose assumptions
are stated. Run focused path, resampling, and public API checks; benchmark candidate
batching only when new changes justify repeating the memory/performance check.

## Deferred scope

No automatic latent route clustering, paper-specific terrain ontology, pretrained
behavior model, mandatory session object, full pairwise dense matrix, vendor-code
copying, or statistical surrogate framework belongs in this phase.
