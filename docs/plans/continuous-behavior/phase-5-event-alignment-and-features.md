# Phase 5 — Continuous event alignment and spatial features

**Created:** 2026-10-14

**Committed:** 2026-10-14

**Status:** Not started

[← PLAN.md](PLAN.md) · [shared contracts](shared-contracts.md)

## Outcome

Align continuous signals to events and sample spatial features on valid paths:
speed at rewards, heading near turns, egocentric wall cues during free foraging,
and feature/accessibility changes during navigation.

This is an implementation plan. All new names and signatures below are
**provisional proposals, not implemented APIs**. Final names require public API
review before code or public export changes.

## Dependencies and starting points

- [Phase 1](phase-1-observation-support.md) supplies validated time support,
  contiguous valid runs, distance preprocessing, and owned outputs.
- [Phase 2 slices 2E/2F](phase-2-environment-contracts.md) supply aligned features
  and states. Owned field arrays on fixed geometry can land earlier; registering
  Environment-valued snapshots require existing representation roadmap stage B.
- This phase can proceed independently of route matching once those contracts
  are available. Phase 6 consumes its features and support metadata.
- Reuse [event alignment](../../../src/neurospatial/events/alignment.py),
  [event result ownership](../../../src/neurospatial/events/_core.py), and
  [time support](../../../src/neurospatial/_intervals.py).
- Reuse [point containment](../../../src/neurospatial/environment/cartesian.py),
  [field interpolation](../../../src/neurospatial/environment/fields.py), and
  [egocentric transforms](../../../src/neurospatial/ops/egocentric.py).
- Read [public API rules](../../../.claude/rules/public-api.md) before defining
  results; existing neural rate and PSTH contracts must remain intact.

## Proposed continuous event interface

<!-- docs-test: skip Proposed API; functions are not implemented. -->
```python
align_continuous_to_events(
    times, signals, event_times, *, window, dt,
    feature_ids=None, feature_units=None, event_ids=None,
    value_kinds=None, category_labels=None, interpolation=None,
    resultant_threshold=1e-12, max_gap=0.5, epochs=None,
    valid=None, break_before=None,
    coverage="complete", retain_aligned=True,
)
```

`times` is seconds; `signals` is `(n_time, n_features)`, accepting a scalar
signal as `(n_time,)`. Unique `feature_ids` identify measurements, not neurons.
`event_times` remains in input order; duplicate times identify separate events.
Optional `event_ids` preserve caller labels separately from original event indices.
`window` is `[start, stop)` relative to each event. `dt` defines explicit query
times starting at `start` and strictly less than `stop`; this samples a signal
rather than calculating spike counts or silently averaging temporal bins.

Use the phase 1 support resolver to permit interpolation only within one valid
run. Never extrapolate at recording edges or interpolate across excluded signal
observations, declared state boundaries, or a gap exceeding `max_gap`.
Trajectory-derived features include required coordinate failures in their
validity; independently measured signals retain their own acquisition support.
Default `coverage="complete"` keeps an event only when its whole window is
observed for all requested features. Proposed explicit `coverage="partial"`
retains supported cells and reports coverage per feature and relative time.
Both policies must report original event indices and reasons for exclusion.
Test full-window interval containment independently of the finite query grid;
unsampled gaps or an unobserved tail after the last query still reject completeness.

Declare per-feature `value_kinds` as linear, circular, or categorical; default
bare numerical columns to linear. Units do not infer value kinds. Optional
per-feature interpolation overrides default to linear for linear values,
shorter-arc interpolation in radians for circular values, and previous-sample
holding for categories. Reject linear/circular interpolation of category codes;
retain validated code labels including zero; reject conflicting circular units.
Use one finite increasing clock per call. `break_before` is a boolean sample-axis
array declaring a new run even when adjacent timestamps are close; validity
masks may be per feature. Invalid values are excluded by feature; a finite zero
is a valid observation.
Overlapping event windows may reuse source samples and are reported as such.

## Proposed result contract

Use a separate `ContinuousEventResult` with derived aligned values shaped
`(event_index, relative_time, feature_id)` when requested, a support array of the
same shape, feature identifiers and units, relative times, original event
indices, counts of excluded events, and serializable analysis settings.
For linear values, return means and counts. Circular summaries accumulate sine,
cosine, and count at each relative time; mean direction is `atan2(sum_sin, sum_cos)`.
Return NaN when normalized resultant length is at/below recorded finite
`resultant_threshold` in `[0, 1]`, or count is zero. Categories return code counts and
labels, never a numerical mean. Summary-only mode retains these sufficient
statistics without the event tensor; retained and streamed reductions must agree.
Results retain no full input signal or recompute closure.

Follow `ReadOnlyArraysMixin`: own arrays and call `_own_arrays()` last in
`__post_init__`; separately own/freeze nested metadata and category count buffers.
Exports are writable copies; `_own_arrays()` alone does not freeze nested mappings.
Use `summary()` for a flat scalar dict and `plot(ax=None, ...)` returning `Axes`.
Keep valid coverage counts beside the mean; no confidence intervals,
significance tests, or group uncertainty are added in this phase.

Review a non-neural `to_dataframe()` exception: retained values use
`(event_index, relative_time, feature_id)` plus support; scalar summaries use
`(relative_time, feature_id)`, and category counts add `category_code`.
Never invent `unit_id` or a unit-oriented `summary_table()`. Optional xarray
exports use these domain axes; existing PSTH classes remain unchanged.
NumPy/pandas suffice for computation and required exports.

## Proposed feature sampling interfaces

<!-- docs-test: skip Proposed API; functions are not implemented. -->
```python
sample_spatial_features(
    env, fields, times, positions, *, feature_ids=None,
    feature_units=None, value_kinds=None, category_labels=None,
    method="containment", footprint=None, max_gap=0.5,
    epochs=None, valid=None, break_before=None,
)
sample_egocentric_features(
    env, fields, times, positions, headings, targets, *,
    feature_ids=None, feature_units=None, value_kinds=None,
    category_labels=None, method="containment", footprint=None,
    max_gap=0.5, epochs=None, valid=None, break_before=None,
)
```

`fields` is `(n_bins, n_features)` or a single `(n_bins,)` array. It can hold
height, texture, illumination, cue identity, or a measured scalar map.
Reuse per-feature kinds and category mappings above; unknown values stay missing.
`targets` contains egocentric offsets, with the documented coordinate convention
and the same distance units as positions. Start with the existing 2D heading
transform; reject unsupported orientation dimensionality explicitly.

Containment requires a verified footprint/locator capability or a validated caller
`footprint` in the same frame and units, with a validated relationship to the
bins. Current track/abstract graph lookup can
be unbounded nearest assignment: a successful `bin_at` is not containment proof.
Refuse unknown containment; verified grid/polygon queries and explicit nearest
sampling can land earlier. Nearest declares its distance policy and result kind.
Mask `-1` before indexing; outside/missing queries stay NaN with false support.
Grid-linear sampling is explicit; sharp steps/categories are never smoothed by default.

Outputs keep the original sample axis: `(time, feature)` for direct queries and
`(time, target, feature)` for egocentric queries, plus validity, feature units,
query/source indices, coordinate frame identity, and analysis settings.
Support combines position/heading, recording, field, and spatial-query validity.
Field NaNs remain unobserved. Use a small owned result or named array return;
no new `Field`, experiment session, or timeline object is required.

## Time-dependent features and environment snapshots

Extend the static array-first call with updates and `(start, stop, snapshot_id)`
records; snapshot/field mappings follow the shared contract separately.

Before roadmap stage B, interval states can select owned read-only feature arrays
and captured immutable binning/units/frame metadata on fixed geometry. Current
Environment graph attributes, units, and frame remain mutable despite frozen
graph structure. Registering
Environment-valued snapshots and caching by their identity wait for stage B.
Then reuse immutable environments by reference; do not copy full geometry per
lookup or substitute a frozen wrapper/parallel snapshot representation.
Own feature/result arrays; advanced correspondence uses the phase 2 slices.

Intervals are half-open, ordered, and non-overlapping. Shared boundaries belong
to the interval starting there. Uncovered times produce invalid samples; missing
snapshot IDs, overlapping records, or incompatible units/frames raise errors.
Never infer physical connectivity from a cue-field mask or remove bins for unknown
cues. Record any mapping needed when snapshots have different bin identities.

For updates on unchanged geometry, default to previous observed field values
inside explicitly supplied validity intervals. Linear interpolation is opt-in
for continuous fields on compatible geometry, with two valid bracketing updates.
Do not interpolate across snapshot transitions, carry an update into a new state,
or assume an update remains valid through unspecified recording gaps.
Expose these choices as parameters and support metadata, not hidden assumptions.

## Implementation tasks

- [ ] Audit reusable interpolation and cohort-selection helpers; extract only
  the shared support logic needed by continuous and spike event alignment.
- [ ] Define the continuous result/table exception and add focused API contract
  checks without changing neural `unit_id`, rate, or PSTH conventions.
- [ ] Implement validated relative query grids and run-bounded linear,
  previous-sample, and circular sampling with original event indices.
- [ ] Implement complete and partial coverage explicitly; add summary-only
  accumulation with bounded temporary memory and numerical parity checks.
- [ ] Validate per-feature kinds, interpolation, category labels, and circular
  resultant thresholds; stream linear sums, circular sums, and category counts.
- [ ] Implement static containment sampling and safe invalid-bin indexing;
  compose egocentric queries through existing transforms.
- [ ] Add snapshot interval validation, immutable snapshot reuse, and feature
  updates on compatible geometry; record state/query provenance.
- [ ] Route common feature validity through phase 1; avoid a second interpretation
  of epochs, timestamp gaps, or missing coordinates.
- [ ] Add plots/tables and feature-kind/unit documentation; optional imports
  stay at exports, and category counts retain explicit code labels.
- [ ] Add examples described below and update API snapshots only after review.

## Validation invariants

| Case | Required evidence |
| --- | --- |
| Irregular sampling | Linear signals reproduce analytic values at supported queries. |
| Wrapped angle | +179°/−179° summaries point near ±180°; retained/streamed outputs agree. |
| Undefined/category mean | Near-zero circular resultant is NaN; categories use code counts. |
| Invalid run | No event query or field interpolation crosses an excluded interval. |
| Complete window | An unsupported interval/tail between or after queries rejects completeness. |
| Event cohort | Original indices preserve unsorted and duplicate events. |
| Partial coverage | Means use only supported values; counts match support exactly. |
| Outside query | `-1` never indexes a field and valid zeros stay supported. |
| Unknown footprint | A far-off-track point cannot pass containment via nearest lookup. |
| State boundary | An exact boundary selects the new snapshot; no mixed geometry. |
| Ownership | Caller mutation cannot change results or stored snapshot metadata. |
| Chunk reduction | Summary-only and retained-event means agree on identical support. |
| Dependencies | Base computation and required exports work without xarray installed. |

## Workflow and review slice

Use deterministic fixtures: maze reward/turn events, free-foraging egocentric
wall/texture samples, and a door/height change with two states. Show supported
and excluded/partial events so users can inspect each coverage decision.

Review shared support/public API contracts; run focused event/feature tests,
existing events/encoding contracts, affected examples, ruff, and mypy.

## Completion boundary

Exclude tracking/camera pipelines, null models, group bootstraps, general
statistics, and forecasting models. Keep physical geometry, sampled cues, and
observation support distinct throughout sampling and event alignment.
