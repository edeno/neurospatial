# neurospatial — API & Design Review after Phase 4d

## Executive summary

**The researcher-workflow checkpoint passes.** Reviewed `d10607a5` on
2026-10-07, all five independent public-doc journeys execute with zero
implementation/test reads: graph tracks are smooth, the other four workable.
The original all-painful qualitative baseline is clearly improved, without
a numerical call-reduction claim. Phase 4d closes the last held scientific
display contradiction. Navigation, frame/criterion work, event/model/holder
handoffs and result parity remain assigned to Phases 5–7; a bounded continuous
heatmap image-centering issue is added to Phase 7 presentation scope.

The lead has prior implementation context and supplies controlled reruns;
the two reviewers supply fresh discovery. See the
[checkpoint](RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07_POST_4D.md)
for exact counts, scripts, scratch recovery and inference limits.

## User journeys

### NWB → epochs → encoding/decoding → table and plots: workable

Eager disk-NWB components survive file close; IDs persist in the table.
Selected epochs cross a genuine tracking gap, intersect explicit acquisition
coverage and produce a normalized posterior with no excluded rows. Real static
and 12-frame HTML outputs contain actual-position markers and session labels.
Composition spans several pages/docstrings; the explicit occupancy prior and
minimum-fps warning are disclosed user choices/limits.

### Graph geometry → trials → directional fields: smooth

The joined recipe repeats geometric bins across directions, then supplies
explicit outbound/inbound trial labels. Both planted peaks recover within
one 5 cm bin. The example clearly covers a single edge rather than certifying
branch assignment or navigation metrics.

### Heading/frame → object-vector maps/candidate screens: workable

Public frame checks establish units and signs, and maps/scores/screens execute.
Both synthetic object-vector and place models pass the current candidate
criteria; limited occupancy and differing map/screen criteria are retained
honestly. Phases 5a/5b carry frame-explicit functions and criterion/bias work.

### Linear simulation → population decoding → assembly/EV: workable

The requested 48-second clock is truthful. Batch encoding feeds decoding;
manual and session posteriors agree exactly. Public counts feed assembly/EV
statistics, while docs distinguish MP dimensions, core members, projection
scale and descriptive EV/REV. The current posterior cells need no coordinate
recovery. Statistics navigation still depends on exports/help.

### Events → PSTH/raster → design columns/positioned events: workable

Explicit full-window selection gives one eleven-event cohort for the main
PSTH/raster/position route. Six design columns are built without claiming a
GLM fit. A small gap probe separately confirms point-position versus full-window
PSTH semantics. Public array glue remains for the shared cohort and regressors.

## Design axes

### Onboarding and golden paths

README and the four-array path lead to a first field; current guides/examples
then support all five complete journeys without implementation access.
**Strength:** the joined graph and actual-bin posterior recipes make their
next steps explicit. **Weakness [medium]:** NWB and population/statistics
onboarding still spans examples and public result help. Phases 6a/6c should
join those entry paths while preserving the working array-first calls.

### Mental model

Geometry, arrays, recording windows and numerical results compose coherently.
The joined direction and decoder-overlay recipes now express the operations
they perform. **Strength:** time/space and units have verifiable contracts.
**Weakness [medium]:** current object-vector frame/criterion names still need
the planned Phase 5 clarification.

### Consistency and predictability

Manual/session decoding agrees, explicit coverage metadata survives and gaps
do not create decoder rows. **Strength:** array/result numerical handoffs work.
**Weakness [medium]:** single/batch summaries and xarray are asymmetric;
precomputed session models use the explicit `.firing_rates` handoff. Phases
6b/7 own those seams.

### Discoverability and navigation

Current examples/help make every route executable without source reading.
**Strength:** the graph workflow is complete in one guide.
**Weakness [medium]:** assembly/EV exports and result help remain necessary
because API/narrative navigation omits that branch. Phase 6a should publish
the count-statistics route and consistent unit selection across periods.

### Domain fit and vocabulary

Occupancy, Hz, cm, radians, candidate thresholds, EV effects and reference
dimensions are interpreted accurately in the executed routes.
**Strength:** corrected statistical language avoids unsupported significance.
**Weakness [medium]:** candidate criteria/coverage cannot establish biological
identity; Phases 5a/5b should preserve that distinction in frame and bias docs.

### Ecosystem fit and interoperability

Real HDF5 input, eager NumPy components, pandas tables and standalone HTML/PNG
outputs work. **Strength:** nondefault NWB IDs and acquisition clocks survive.
**Weakness [medium]:** component/holder, unit and clock selection need a joined
Phase 6c guide. External animal NWB files and an optional-install matrix were
not exercised here.

### Composability across modules

Segmentation labels feed encoding; population results/counts feed decoding
and statistics; retained events feed rasters and positioned tables.
**Strength:** the main array/result pipeline is usable.
**Weakness [medium]:** event cohort glue and explicit model handoffs still
need Phase 6b. Native 1D rate-result plotting and table metadata need Phase 7.

## Prioritized recommendations

**High:** no unresolved checkpoint blocker; start Phase 5a after the
checkpoint report/planning PR merges.

**Medium:** complete the existing Phase 5 frame/criterion/bias work, Phase 6
navigation/cohort/model/holder handoffs and Phase 7 native 1D/result parity.
Keep existing scientific defaults unless a phase explicitly changes them.

**Low:** center continuous posterior image columns on the public timestamps.
The known four-row fixture has a 37.5 ms maximum pixel offset; actual/MAP
lines, spatial coordinates and physical errors are already correct. This
bounded presentation task stays in Phase 7 and does not reopen the held
cm-versus-bin error. Retain optional HTML colorbar/overlay accessibility polish.

## What's working (keep)

- Array-first analysis, explicit recording windows and per-run gap behavior.
- Eager NWB reads and stable unit identity in exported results.
- Joined graph geometry/trial-label workflow and planted-peak verification.
- Shared numerical population decoding and descriptive statistical interpretation.
- Truthful posterior bin labels, actual-bin overlays and MAP-derived plot times.
