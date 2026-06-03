# Review Remediation Implementation Plan

**Status:** Not started.

This plan executes the findings of the neurospatial review (see [.claude/reviews/ROADMAP.md](../../../reviews/ROADMAP.md)): the correctness fixes that stop silently-wrong scientific output, the numerical-robustness and statistical-validity guards, the API/crash repairs around the spike→time-bin→decode seam, and the ergonomics, naming, documentation, and test-coverage work that follows. Each phase ships as one independently-reviewable PR. Correctness phases (1–13) are design-settled and fully specified; the four design decisions (phases 16, 17, 19, 24) are **resolved** and baked into those phase files (recorded in [overview.md](overview.md#open-questions)).

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained: inputs to read, contracts/designs it depends on, tasks, validation slice, fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md) — validation helpers and the result-object contract reused across phases.
3. **Need broader scope / dependency order / risks / open design decisions?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points, dependency ordering, risks, open design decisions, effort.
- [shared-contracts.md](shared-contracts.md) — `_validate_finite`/`_validate_lengths` helpers and the `ResultMixin` contract reused by ≥2 phases.

### Phases (each ships as a separable PR)

**Correctness — silent-wrong-result fixes (design-settled, full detail):**
- [phase-1-decoding-correctness.md](phase-1-decoding-correctness.md) — reactivation/assembly stats, posterior n_bins check, NaN in HPD/confusion.
- [phase-2-environment-occupancy.md](phase-2-environment-occupancy.md) — occupancy active-bin index, `interpolate` reshape, timestamp validation.
- [phase-3-simulation-groundtruth.md](phase-3-simulation-groundtruth.md) — PlaceCellModel width normalization & `width=0`, Poisson timestamp validation.
- [phase-4-stats-circular.md](phase-4-stats-circular.md) — weighted-circular validation/NaN-cofilter, shuffle p-value NaN.
- [phase-5-encoding-directional.md](phase-5-encoding-directional.md) — Rayleigh sample-size, NaN heading binning, classifier error-swallowing, phase-precession p-value.
- [phase-6-behavior-binning.md](phase-6-behavior-binning.md) — `-1` bin wraparound, dt guards, geodesic-vs-Euclidean efficiency.
- [phase-7-events-validation.md](phase-7-events-validation.md) — `match_by` Cartesian product, `distance_to_reward` NaN.
- [phase-8-ops-robustness.md](phase-8-ops-robustness.md) — `resample_field` NaN, `heading_from_velocity` dt, `map_probabilities` broad-except, Kabsch reflection.
- [phase-9-io-roundtrip.md](phase-9-io-roundtrip.md) — Graph/Polygon `to_file` crash, NWB `coordinate_kind`, rate≤0 timestamps, length/unit checks.
- [phase-10-layout-indexing.md](phase-10-layout-indexing.md) — ImageMask axis order, MaskedGrid dtype, index-helper raising, int32 overflow.
- [phase-11-regions-validation.md](phase-11-regions-validation.md) — RLE bounds, `point_tolerance`, `region_center`, CVAT broad-except.
- [phase-12-annotation-geometry.md](phase-12-annotation-geometry.md) — alpha-shape non-polygon, edge-order staleness, validate-don't-crash.
- [phase-13-animation-overlays.md](phase-13-animation-overlays.md) — napari int truncation, circular heading interpolation, NaN bounds.

**API seam, ergonomics, consistency, docs, tests:**
- [phase-14-spike-binner.md](phase-14-spike-binner.md) — public `bin_spikes_in_time(..., *, orient=)`.
- [phase-15-nwb-units.md](phase-15-nwb-units.md) — `read_units`/`read_spikes` NWB readers.
- [phase-16-occupancy-fill.md](phase-16-occupancy-fill.md) — **[gate resolved]** reconcile `min_occupancy` NaN with `decode_position`.
- [phase-17-result-mixin.md](phase-17-result-mixin.md) — **[gate resolved]** unified result-object contract; backfill bare dataclasses.
- [phase-18-direction-labels.md](phase-18-direction-labels.md) — lap/run → direction-labels bridge; `running_direction_labels`.
- [phase-19-polar-env.md](phase-19-polar-env.md) — **[gate resolved]** promote polar egocentric env to a distinct type.
- [phase-20-xarray-interop.md](phase-20-xarray-interop.md) — `to_xarray()` results; `DecodingResult.error_against`.
- [phase-21-lazy-imports.md](phase-21-lazy-imports.md) — top-level `__getattr__` lazy submodule access.
- [phase-22-naming-consistency.md](phase-22-naming-consistency.md) — `random_state`→`rng`, unified `window`, keyword-only/env-first, OVC delegate.
- [phase-23-docs-runnable.md](phase-23-docs-runnable.md) — fix crashing examples; execute docs/QUICKSTART in CI.
- [phase-24-domain-capabilities.md](phase-24-domain-capabilities.md) — **[gate resolved]** ripple detection, theta-phase helper.
- [phase-25-test-backfill.md](phase-25-test-backfill.md) — untested validation/failure branches that let bugs survive.
