# Phase 6 — Behavioral conditions and neural comparisons

**Created:** 2026-10-14

**Committed:** 2026-10-14

**Status:** Partially implemented (PR #119). Label-to-epoch conversion,
conditioned estimation, captured estimation context, one-to-one
`compare_spatial_rates` and held-out decoding are delivered. A heading
diagnostic, external feature/basis handoff documentation, and the
free-foraging and door/cue-change workflow examples remain. Only the
linear-track example exists. The shipped signatures match
the proposal for `labels_to_epochs`; the comparison is
`encoding.compare_spatial_rates`.

**Status reviewed:** 2026-10-20

[← PLAN.md](PLAN.md) · [shared contracts](shared-contracts.md)

## Outcome

Turn behavioral labels into valid epochs, estimate condition fields, compare
matched units/supported bins, and evaluate fitted decoders on held-out data.
Examples cover incoming-arm history, free-foraging movement states, and
navigation before/after accessibility or cue changes.

This document proposes implementation work. New names and signatures are
**provisional and not implemented**; public exports require API review.
The outcome is numerical results and inspectable support through existing models.

## Dependencies and reuse

- [Phase 1](phase-1-observation-support.md) establishes valid runs, interval
  boundaries, distance preprocessing, and provenance.
- [Phases 3](phase-3-exploration-and-crossings.md) and
  [4](phase-4-path-analysis.md) provide optional direction/route/history labels.
  Labels may also be supplied by users; route matching is not mandatory.
- Cross-binning correspondence uses [phase 2A](phase-2-environment-contracts.md).
  Same-environment comparisons can land after the estimation-context fix below,
  without waiting for immutable Environment values under roadmap stage B.
- Phase 5 provides optional spatial/egocentric covariates and change snapshots.
  Basic conditioned estimation needs only phase 1 and existing estimators;
  result comparisons additionally require captured estimation-time context.
- Reuse [spatial estimators](../../../src/neurospatial/encoding/spatial.py),
  [directional estimators](../../../src/neurospatial/encoding/directional.py),
  and [egocentric estimators](../../../src/neurospatial/encoding/egocentric.py).
- Reuse [unit selection and ownership](../../../src/neurospatial/_results.py),
  [support/provenance](../../../src/neurospatial/encoding/_provenance.py),
  [field metrics](../../../src/neurospatial/encoding/_field_metrics.py), and
  [fitted decoding](../../../src/neurospatial/decoding/estimator.py).
- Apply [public API rules](../../../.claude/rules/public-api.md), including
  positional data, keyword-only settings, and unit-oriented terminal verbs.

## Labels become contiguous valid epochs

<!-- docs-test: skip Proposed API; functions are not implemented. -->
```python
labels_to_epochs(
    times, labels, *, valid=None, break_before=None,
    max_gap=0.5, epochs=None, missing_label=None,
)
```

`labels` is one categorical value per original sample; its meaning may be route,
prior arm, movement direction, task phase, or door state. Use one finite increasing
clock per call. `break_before` declares a new run independently of timestamp gaps.
Missing labels are explicitly excluded. Equality does not join occurrences
separated by invalid coordinates, excluded windows, recording gaps, or a
different intervening label.

Assign each observed interval `[times[i], times[i + 1])` the label at its start.
Retain it only when the shared support resolver accepts the entire interval.
Merge adjacent retained intervals with the same label within one valid run.
The final sample has no interval and does not create inferred exposure; a
singleton contributes no epoch. Keep this distinct from sample-based plotting.

Return a label-keyed mapping of owned `(n_intervals, 2)` half-open arrays, support
settings, and original interval indices; preserve first-seen label order.
No session/conditional-field container is required. Named results use phase 5's
non-neural table exception instead of fabricated unit IDs.

Pass original `times`, `positions`, and spike trains with `epochs=` to existing
estimators. Do not filter samples and accidentally introduce adjacency across
excluded periods. Apply the identical epoch support to spike counts and dwell.
Keep spike recording coverage separate from behavioral observation coverage.

## Conditioned estimation stays a composition

Start with iteration over epochs and existing batch encoders. Add a delegating
wrapper only for repeated validation/support mistakes; return existing results
keyed by condition without another rate-fitting implementation.

Use one declared spatial bin correspondence across conditions. Distinguish
world position, movement heading, measured head orientation, and route progress.
Movement heading must not be reported as measured head direction.
Route/history labels are conditioning variables, not extra spatial dimensions
appended to coordinates or artificial branches added to an environment graph.

Retain occupancy, spatial support, unit identity, estimator choices, selected
epochs, and the label's definition/source. A finite smoothed prediction outside
observed coverage is not a supported observation.
Changing accessibility can alter the smoothing graph even when physical bin
locations match; record snapshot/topology identity and make that choice explicit.

## Proposed result-level rate comparison

<!-- docs-test: skip Proposed API; functions are not implemented. -->
```python
compare_spatial_rates(
    rates_a, rates_b, *, unit_selection="intersection",
    bin_mapping=None, support="intersection", min_common_bins=2,
    metric="pearson", source_context=None,
)
```

Start with spatial results; directional/polar results need coordinate review.
Match units by `unit_ids`, never silently by row position. Report units excluded
from either condition; an explicit exact-match mode raises on unequal sets.
Unit IDs must already identify the same recorded cells. The helper does not
perform cell registration across sessions or infer identities from rates.

First extend existing rate results with owned immutable estimation-time context:
coordinates, bin order/sizes, declared binning identity, units/frame, and relevant
topology/operator metadata. Preserve it through indexing and exports. Reuse a
read-only representation `Binning` if ready; otherwise copy only the necessary
arrays/scalars once at estimation, without another equality/hash system.
The current live `rates.env` reference cannot prove past units/frame or geometry:
changing its metadata after estimating A must not relabel A when estimating B.
For legacy/manually constructed results, require caller-verified `source_context`
for both maps or refuse missing provenance; never manufacture it from current env.

Check this captured context, not shape equality or `env_fingerprint`, before
comparing bins. Initially accept only validated one-to-one bin mappings for
reordering or subset correspondence, recording direction and omitted bins.
Reject many-to-one/coarsening/overlap mappings; never average rates to merge bins.
Fine rates `[0, 100]` with exposures `[100, 1]` would merge to about 0.99 Hz from
counts/exposure, not their 50 Hz scalar average. Re-estimate on a common environment.
Separately justified count/exposure transfer is a future proposal, especially
for smoothed/GAM maps whose raw count information cannot be recovered from rates.

Default comparison uses both conditions' positive sufficient occupancy and
finite per-unit rates. Intersect observed support, including any supplied
analysis mask; never convert unsupported predictions into observed bins.
Return retained bins/counts and coverage diagnostics alongside each score.
Compute support per matched unit when finite rates differ across units rather
than allowing one invalid cell to remove every population comparison bin.

Return an owned `RateComparisonResult` with aligned `unit_ids`, per-unit scores,
common-bin counts, derived contrast maps, per-unit bin support, excluded unit/bin
information, and serializable settings. Keep signed contrasts distinct from
firing rates so they cannot be mistaken for a decoder's encoding models.
Insufficient common bins and zero-variance maps return NaN with a reason;
they do not look like zero similarity or demonstrated remapping.
Keep the existing `field_stability` API stable unless separately reviewed;
the new helper can apply explicit support before reusing its metric kernels.

`summary_table()` is one row per unit indexed by `unit_id`, with scalar scores
and coverage; `summary()` is a flat scalar dict. `to_dataframe()` is dense
`(unit_id, bin)` contrast/support data; optional xarray uses those labeled axes.
Own stored arrays/context; exports are writable copies and xarray remains optional.
Plots return `Axes`. `UnitSelectionMixin` supplies positional `[]`, label-based
`sel(unit_id=...)`, and matched unit identity on indexed children.

## Population and behavioral diagnostics

Report condition occupancy/speed/heading on analyzed epochs and common bins,
with duration, retained samples/intervals, bin coverage, and comparison masks.
Show confounding differences without a universal matching method or claims
of independent route/history coding.

Existing `population_vector_correlation` correlates cell rows within one matrix;
it is not cross-condition population-vector comparison. Defer a new helper until
examples establish bin-wise axes/support distinct from per-unit field scores.
Descriptive correlation thresholds do not classify cell types.

## Held-out prediction and external covariates

Use existing immutable `BayesianDecoder.fit(..., epochs=training_epochs)` and
`predict(..., epochs=test_epochs)` behavior. Keep train/test original arrays and
unit labels. Pass support epochs derived from explicit run breaks until those
breaks are accepted directly by the existing estimator/decoder interfaces.
Show trial- or session-blocked partitions; history features and smoothing must
not borrow samples across fold boundaries. Make any history buffer explicit.
Route labels selected by future matched segments are post-hoc and are unsuitable
as causal predictors. Predictive examples assign labels from observed prefixes
and fit reference repertoires within training folds; descriptive analyses retain
post-hoc labels with that qualification.

Report supported per-unit field scores and numerical decoder errors/coverage;
omit p-values, significance verdicts, confidence intervals, and group inference.
Distinguish performance within a condition from transfer to a new condition.
Show train and test spatial coverage so error changes are assessable.

Export time-aligned route/direction/history/terrain arrays and existing spatial
bases for external models. Mask invalid bins before basis indexing and retain
time support. Current `method="glm"` is a spatial MRF/GAM estimator on binned
counts and exposure; it does not become an arbitrary time-covariate GLM here.

## Implementation tasks

- [x] Implement label-to-epoch conversion through phase 1 support, with original
  interval provenance and no cross-gap merging.
  `behavior.labels_to_epochs` (`5f9f3777`, #119), tested in
  `tests/behavior/test_labels_to_epochs.py`.
- [x] Add examples estimating condition maps with existing estimator semantics;
  preserve unit IDs and epoch/spike-window settings.
  `docs/user-guide/workflows.md` (`ad10874f`, #119) and
  `tests/encoding/test_conditioned_rates.py` (`57efe6e5`).
- [x] Capture owned estimation-time context in existing rate results and carry
  it through indexing/exports; validate explicit legacy context or refuse it.
  `3f5f599d` (#119), tested in `tests/encoding/test_rate_estimation_context.py`
  and `tests/nwb/test_fields_roundtrip.py`.
- [x] Validate unit/bin context and one-to-one mappings; reject aggregation.
  `compare_spatial_rates` (`8a3e7980`, `718878c7`, `ec1592a9`, #119), tested in
  `tests/encoding/test_compare_spatial_rates.py`.
- [x] Implement common observed support and per-unit numerical field scores;
  retain omitted-bin reasons and distinguish insufficient data from zero scores.
  `8a3e7980` (#119); see
  `test_too_few_common_bins_is_nan_with_a_reason`.
- [x] Add rate comparison tables/plots following terminal verb contracts.
  `RateComparisonResult` (`0b0ed9c9`, #119), tested in
  `tests/encoding/test_rate_comparison_result.py`.
- [ ] Add condition duration, occupancy, speed, and heading diagnostics using
  existing behavioral operations and numerical summaries.
  *Partial:* duration, occupied bins and mean speed are compared in
  `workflows.md` (#119). A heading diagnostic remains.
- [x] Demonstrate held-out fit/predict and state transfer without training leakage.
  The "Decode held-out laps" section of `workflows.md` is executed (#119), and
  `test_decoder_reuses_condition_epochs` covers it.
- [ ] Document external feature/basis handoffs; introduce no general GLM solver.
  Not started: no handoff documentation was found.
- [x] Review public names, update snapshots/contracts, and add workflow docs.
  `tests/data/public_api.txt` and `workflows.md` (#119). Identity-aware
  `unit_ids` for `population_vector_correlation` (added in `95973398`, #120)
  and for assembly activation are in progress (refactor/api-consistency).

## Validation and workflow slice

| Invariant | Test or executable example |
| --- | --- |
| Same label after gap | Produces separate epochs even when timestamps are close. |
| Label boundary | Uses half-open exposure; final/singleton samples add no dwell. |
| Unit permutation | Reordered inputs produce identical scores under label matching. |
| Incompatible bins | Shape equality cannot bypass geometry/frame/unit validation. |
| Estimation context | Later env units/frame/graph mutation cannot relabel maps or exports. |
| Legacy context | Missing provenance is refused unless verified source context is supplied. |
| Mapping aggregation | Rates `[0, 100]`, exposure `[100, 1]` cannot merge via `bin_mapping`. |
| Unobserved bin | Finite extrapolated rates are excluded from observed comparisons. |
| Per-unit failure | Invalid data in one unit does not corrupt other units' support. |
| Insufficient data | Returns NaN plus reason, not a remapping-like numeric zero. |
| Spike/dwell support | Condition selection gates numerator and denominator identically. |
| Held-out history | Feature construction cannot use another trial/fold's samples. |
| Ownership/export | Inputs cannot mutate outputs; xarray is optional. |

Use a T/W maze with prior-arm labels, a free-foraging arena with movement/head
orientation differences, and a door/cue-change example sharing physical bin
correspondence. Each reports numerical scores and coverage, including one
deliberately unsupported comparison. Review focused support/unit/encoding and
decoder tests, public API checks, executable examples, ruff, and mypy.

Complete reusable estimator compositions and inspectable numerical comparisons;
exclude null models, group bootstraps, general statistics, cell registration,
and a new general GLM framework.
