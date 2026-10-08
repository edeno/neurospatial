# neurospatial — API & Design Review (for neuroscience users)

## Executive summary

The current array-based design supports all five documented journeys without
reading implementation code. Population rate maps, shared occupancy,
gap-aware decoding and ordinary count matrices now compose with relatively
little scientific glue. Public tutorials still require joining recipes, and
the primary linearization tutorial teaches direction separation that the
runtime does not provide. A simulation-duration/metadata contradiction and
uncalibrated statistical interpretation wording also need correction. The
checkpoint is on hold for Phase 4c; it does not reopen the settled array,
frame or classifier-default decisions. See
[measurements and exact evidence](RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07.md).

## User journeys

### Place fields from NWB → decode → posterior animation

**workable** (baseline: painful). Component readers, selected epoch arrays,
population fields, reusable model arrays and posterior frame times compose.
Six unit IDs and the excluded 20–35 second window survive the path. An
independent first attempt mistakenly passed the population result directly
to the session decoder; public docs enabled the `.firing_rates` correction.
Choosing units/epochs and bypassing the guide's Session recommendation remain
navigation work, addressed by Phase 6c's joined holder recipe.

### Linear track → linearization → trials → directional fields

**painful** (baseline: painful), despite successful execution of 12 trials and
two planted direction maps. Notebook 05 falsely claims a return traversal
receives different coordinates/bins and a monotonically increasing linear
position. Correct fields require explicit regions, trials and direction labels
from another example. Phase 4c must correct the mental model before this
journey qualifies as easier for a first-timer.

### Egocentric frame → object-vector map and candidate classification

**workable** (baseline: painful). Velocity headings, documented angular
conventions, bearing/distance arrays, polar maps and a yes/no screen all work.
The place-cell control also passes the information-only criterion, already
owned by Phase 5b. Score evaluation requires reshaping a flat polar map;
guidance must distinguish frame, estimator and candidate screen without
changing the settled threshold criterion.

### Simulation → encode/decode → assembly/reactivation statistics

**workable via open-field simulation; painful via the linear convenience
helper** (baseline: painful). The corrected choice of documented helper yields
60 seconds of data; manual and golden decoding match exactly, and counts feed
PCA/EV directly. The requested duration is ignored by the linear helper, and
assembly/reactivation capabilities are discoverable only through public
runtime docs. Decoding and count-based statistics are sibling branches;
zero-variance preprocessing, controls and statistical interpretation need an
explicit population example.

### Events → PSTH/raster → GLM regressors → positions

**smooth for continuous recordings; workable across gaps** (baseline: painful).
Example 26 and public helpers support the full task with ordinary arrays and
tables. The gap probe reveals different retained-event cohorts and closure
rules among PSTH, raster and count/indicator regressors. Share an explicit
full-window-selected event cohort and document boundaries; do not silently
change the deferred helpers' numerical contracts.

## Design axes

### Mental model & core abstractions

Environment plus raw arrays and domain results is a workable core. Linear
coordinates encode graph geometry, while direction is a separate temporal
label; the tutorial currently blurs that distinction.

**Strengths**

- Shared discretization and occupancy make maps comparable.
- Results expose metrics and ordinary array/table outputs.

**Weaknesses**

- [high] Notebook 05 teaches direction/history semantics contradicted by its
  own current runtime.
- [medium] Population statistics are not explained as a sibling of decoding.

### Onboarding & the golden path

First fields and one-call decoding are approachable. The shortcut should not
misrepresent its duration, and joined recipes should show the complete handoff.

**Strengths**

- README/quickstart provide a complete field and useful next steps.
- E1006 explains the factory-only construction pattern.

**Weaknesses**

- [high] `linear_track_session(duration=...)` surprises both scale and metadata.
- [medium] NWB, track directions and assembly statistics require joining docs.

### Cross-module API consistency & predictability

Encoding order and rate/count matrix orientation compose. Named boundaries,
model types and result parity are the remaining inconsistencies.

**Strengths**

- Seconds, Hz and SEM count units are documented.
- The manual and golden decoder outputs match exactly on the same data.

**Weaknesses**

- [medium] PSTH/raster/regressors use different closure/cohort rules.
- [medium] Session decoder model wording and single/batch accessors differ;
  Phases 6b and 7 already own most of that work.

### Discoverability & namespacing

Public exports and runtime help suffice without private implementation reads.
Navigation should expose existing capabilities and follow complete tasks.

**Strengths**

- Domain subpackages have usable public entry points.
- Public notebook companions provide runnable scientific examples.

**Weaknesses**

- [medium] The API index/guide omits the assembly/reactivation branch.
- [medium] Interoperability still favors Session before its planned deletion.

### Domain fit & vocabulary

Most vocabulary matches the requested analyses. Scientific effects,
candidate screens and statistical significance must remain distinguishable.

**Strengths**

- Occupancy, egocentric frames, EV/REV and assembly patterns are exposed.
- The OVC tutorial already acknowledges place-cell false positives.

**Weaknesses**

- [high] Activation/EV result docs give uncalibrated significance statements.
- [medium] A significant PCA dimension may have no core assembly members;
  the population example should explain the observed distinction.

### Ecosystem fit & interoperability

Ordinary NumPy arrays and pandas tables make adoption workable. Eager file
reading and identity preservation are useful; loaders must carry context
without requiring a Session facade.

**Strengths**

- Real NWB input becomes usable arrays after the file closes.
- Population rows retain input unit IDs.

**Weaknesses**

- [medium] Users must choose NWB epochs and establish position units themselves.
- [medium] Common unit/event cohorts across branches require explicit handling.

### Composability across modules

Rate maps feed decoding and count matrices feed assembly statistics cleanly.
The remaining seams concern interpretation, cohorts and output dimensions.

**Strengths**

- `fill_value=0.0` makes the manual model handoff explicit and finite.
- Gap/epoch restrictions do not require stitching disjoint samples together.

**Weaknesses**

- [medium] PSTH retention does not provide an event mask for related branches.
- [medium] Native 1D rate plotting fails after successful computation.

## Prioritized recommendations

**High**

1. **Repair duration and truthful metadata.** Phase 4c.1 must reconcile the
   documented duration with laps and preserve stream alignment; otherwise
   even a minimal simulation silently changes recording scale.
2. **Correct scientific interpretations.** Phase 4c.2–4c.3 must distinguish
   geometry from direction and effect/activity cutoffs from significance.
3. **Repeat the checkpoint before Phase 5a.** Keep the gate held until those
   paths can be recommended from public docs without the misleading claims.

**Medium**

1. **Publish complete branch-specific recipes.** Add a population-statistics
   branch in Phase 6a and preserve the joined NWB recipe through Phase 6c.
2. **Show one retained-event cohort.** Phase 6b should document a reusable
   full-window mask for PSTH, rasters and positioned events, with explicit
   endpoint handling rather than blanket API unification.
3. **Finish output parity and native 1D plotting.** Phase 7 owns the rate-result
   summaries, visible units/thresholds and dimensionality support.
4. **Clarify estimator-dependent candidate criteria.** Phase 5b should retain
   its planned bias examples and avoid claiming raw-binned estimates are
   universally lower or more conservative than smoothed estimates.

**Low**

1. **Make animation limitations visible.** Explain the unsupported HTML
   colorbar keywords and show timestamp labels for nonuniform decoder frames.

## What's working (keep)

- Four-array computation and ordinary NumPy/pandas interchange.
- One shared population occupancy map and preserved unit identities.
- Automatic pauses and explicit half-open analysis/recording windows.
- One-call decoding plus transparent manual count/model paths.
- Helpful first-run errors and viridis static plots.
- Explicit candidate-screen limitations and opt-in significance work in Phase 5b.
