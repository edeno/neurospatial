# neurospatial — API and design review after Phase 4c

## Executive summary

All five repeated journeys complete from public documentation and help with
zero implementation/test reads. The track route is smooth, and corrected lap
duration and statistical interpretation remove the earlier scientific
contradictions. A new tutorial display error mixes centimeters with posterior
bin indices; **the checkpoint remains HOLD for Phase 4d before Phase 5a**.
The numerical decoder and public plot contract already agree and need no
redesign. See [exact evidence](RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07_POST_4C.md).

## User journeys

- **NWB → epochs → population maps → decode → table/plot/animation: workable.**
  Eager component readers, detected cm units, explicit ephys/analysis windows,
  six preserved IDs and a normalized posterior work across a real tracking gap.
  Joining several pages remains Phase 6c navigation work.
- **Graph track → linearization → trials → directional fields: smooth.**
  The current joined recipe separates geometry and labels, completes twelve
  trials, reuses repeated bins and recovers both planted centers within one bin.
- **Egocentric/object-vector map and candidate screen: workable.**
  Heading/angular conventions and polar arrays compose. Short-fixture candidate
  acceptance includes the place control, consistent with Phase 5's existing
  criterion/frame limitations rather than biological ground-truth accuracy.
- **Linear simulation → encode/decode → assembly/EV: workable after recovery.**
  The requested 45 s duration is real; manual/session posterior arrays agree.
  Counts feed statistics as a sibling branch. Current docs correctly distinguish
  selected dimensions, empty core members and descriptive EV/activity. The
  tutorial's spatial-overlay error requires one documented public-call recovery
  and blocks a recommendation of the advertised figure until Phase 4d.
- **Events → PSTH/raster → regressors → positions: workable.**
  An explicit retained cohort aligns outputs; complete-window PSTH eligibility
  differs from endpoint-inclusive positioned events as help explains. Phase 6b
  should publish the joined cohort/boundary recipe.

## Design axes

**Mental model/core abstractions.** Environment plus arrays and domain results
is coherent. Geometry, temporal direction, count matrices and posterior bins
have separate roles. The remaining high-priority contradiction is a tutorial
that labels posterior-bin coordinates as physical position.

**Onboarding/golden path.** Factory recovery and four-array encoding are
workable. Track direction now has a complete recipe. Lap simulation no longer
changes recording scale behind truthful-looking metadata. Correct the decoder
figure next; retain the later joined NWB/statistics recipe work.

**Cross-module consistency.** Rate/count orientations and manual/session
posteriors agree. Spatial bins and centimeters must stay explicit at plot
handoffs. Single/batch accessor/summary parity and event closure/cohorts remain
assigned to Phases 6b/7.

**Discoverability/namespacing.** Public exports/runtime help suffice without
private code. Assembly/reactivation navigation remains thin; the guide still
foregrounds Session/load_session. Phases 6a/6c own those gaps.

**Domain fit/vocabulary.** The corrected activity/EV language no longer implies
calibrated probability from a heuristic cutoff. Selected dimensions need not
contain thresholded core neurons. Synthetic candidate screens and in-sample
decoding remain descriptive. Wrong spatial plot labels still contradict the
domain meaning, even when numeric decoding is perfect.

**Ecosystem/interoperability.** Real NWB input, ordinary arrays/pandas and stable
unit IDs compose after file close. Acquisition windows are explicit rather
than guessed from spike extrema; missing tracking remains a gap. The future
holder recipe should preserve these successful semantics.

**Composability.** Encoding feeds decoding; count matrices separately feed
assembly/EV statistics. Full-window event selection can align raster/regressor/
position branches. No new mandatory bundle, private workflow helper, time type
or API shim is needed for the observed repairs.

## Prioritized recommendations

**High:** implement the bounded Phase 4d tutorial/guard correction, then repeat
the checkpoint. Do not start Phase 5a while a perfect decode is falsely
visualized by an advertised example.

**Medium:** retain existing Phase 5 criterion/frame work; Phase 6a statistics
navigation/common-unit recipe; Phase 6b cohort/model handoffs; Phase 6c eager
reader/holder recipe; Phase 7 native 1D plotting and result/table parity.

**Low:** remove residual ambiguity from notebook 05's introductory automatic-
handling sentence; continue explicit units/time labels in rendered artifacts.

## What to preserve

Four-array input, public-help recovery, automatic gap handling, explicit
analysis/recording windows, unit identities, shared population occupancy,
ordinary count matrices, truthful simulation clocks, manual/golden numerical
agreement, correct geometry/direction explanation and descriptive statistics.
