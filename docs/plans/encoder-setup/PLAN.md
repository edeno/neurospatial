# Consolidate encoder setup

**Committed**: 2026-10-14
**Status**: Implemented — Phases 0–3 delivered in
[PR #111](https://github.com/edeno/neurospatial/pull/111) (merged 2026-10-19).
Phase 0 parity evidence is in the PR description, not in the repository, and no
independent review is recorded (the PR has no GitHub reviews).
**Status reviewed**: 2026-10-20

## Problem and scope

The single-unit and population entry points for spatial, directional, view and
egocentric encoding repeat backend checks, input conversion, trajectory/spike
validation, time-window resolution and result provenance. A change to those
rules can drift between eight paths. Object-vector entry points already delegate
to the egocentric family's private implementation and belong in the same checks.

Current setup locations:

| Family | Entry points | Existing reusable pieces |
| --- | --- | --- |
| Spatial | `compute_spatial_rate`, `compute_spatial_rates` in `encoding/spatial.py` | `_binning`, `_smoothing`, `_validation`, `_provenance` |
| Directional | `compute_directional_rate`, `compute_directional_rates` in `encoding/directional.py` | `_directional_binning`, `_validation`, `_provenance` |
| View | `compute_view_rate`, `compute_view_rates` in `encoding/view.py` | `_view_binning`, `_smoothing`, `_validation`, `_provenance` |
| Egocentric | `compute_egocentric_rate`, `compute_egocentric_rates` in `encoding/egocentric.py` | Shared object-vector implementations, `_egocentric_binning`, `_validation`, `_provenance` |

Retain public names, signatures, defaults, result types, optional dependencies,
unit identity, scientific behavior and error/warning context. This is a private
refactor, with no new public encoder class and no simultaneous change to the
environment representation.

## Proposed boundaries

1. Reuse existing helpers first. Adopt `_backend.validate_and_resolve_backend`,
   which already checks the supported set and resolves the backend. Keep call
   order where it affects error precedence or optional-dependency loading.
2. Share spike-input normalization and validation between single and population
   paths, retaining population labels and the distinction between generated and
   supplied labels. Single-unit entry points must still reject population input.
3. Share time-window resolution and normalized trajectory validation only where
   inputs and semantics agree. Keep environment checks and family parameters in
   their family modules. Spatial setup currently resolves support inside
   `_resolve_spatial_time_support`; avoid resolving its windows a second time.
4. Have each family's single and population functions consume the same prepared
   trajectory/binning data. Prefer the existing prepared objects; introduce a
   small typed private record only when existing structures cannot carry the
   needed values. Do not retain raw recording arrays in published results.
5. Keep family-specific scientific operations explicit: spatial support and speed
   gates, circular directional bins, view visibility/ray casting, polar object
   geometry, and GAM validation/fitting. Existing count, occupancy, smoothing,
   NumPy/JAX and provenance helpers remain the computation boundaries.

The tradeoff is deliberate: some family orchestration stays duplicated to keep
scientific differences readable. A universal encoder pipeline with callbacks or
dynamic keyword dictionaries would obscure validation and add a larger type/API
surface than this problem needs. Do not route single-unit calls through full
population result construction if that adds fitting work or changes diagnostics.

## Ordered implementation and acceptance checks

### Phase 0: Capture behavior

*Status: delivered, evidence outside the repository.* The PR #111 description
reports an equivalence harness of 296 cases (dev) and 316 (all extras, with
JAX). It covers every encoder entry point, errors and their order, warnings and
the file they point at, and a 120 s, 10-unit slice of the J16 recording. It
found 0 differences against `main` after every commit. The harness and the
baseline artifacts were not committed.

Run the current encoder tests and save deterministic numeric/provenance fixtures
for all eight entry points plus allocentric object-vector wrappers. Cover binned,
diffusion and Gaussian methods where supported, spatial GAM with fixed and
automatic penalties (pooled and per-unit), and installed NumPy/JAX backends.
Use separately seeded stimulus and spike generators. Include labeled inputs,
unit reordering, silent units, zero units, epochs, empty/disjoint spike windows,
trajectory gaps, speed gates, boundary samples, unsupported bins and NaN versus
zero-filled maps. Include epoch and recording-window endpoints. Capture
warnings/errors where setup order is observable. Use the project `uv`
environment and record the commit, working-tree diff, installed optional extras
and explicit parameters alongside the baseline artifacts.

### Phase 1: Consolidate independent setup helpers

*Status: done (#111).* `abd31ce0` adopts `validate_and_resolve_backend` and
adds the shared `as_labelled_spike_trains` and `validate_spike_trains`. `4a88e8f3` moves the
spike and time-bin helpers to `_spikes.py`/`_time_bins.py`, guarded by
`tests/test_spike_imports.py` and `tests/test_spikes.py`. `819933bc` points the
bandwidth warning at the caller.

Adopt the existing backend resolver and consolidate spike normalization/validation
first, one helper at a time. Reuse `_spikes` and `_validation` rather than building
parallel versions.
Run the matching family tests and compare Phase 0 outputs after each migration.
Keep caller names and useful warning stack locations. Review this boundary
independently before moving binning or support logic.

### Phase 2: Unify setup within each family

*Status: done (#111).* Shared preparation landed per family: directional
`523a579a`, spatial `906bb8fb`, view `e676eb29`, object-vector/egocentric
`7f7aab7e`, with named count records in `2629a091`.

Proceed directional, spatial, view, then egocentric/object-vector. Each family
gets one preparation path reused by its single and population functions; keep
distinct fit kernels and result construction where their semantics differ.
Compare values, occupancy/support, labels, windows, settings and GAM diagnostics
after each family. Commit each family as a logical chunk and review it
independently before the next.

### Phase 3: Validate the complete workflows

*Status: done per the PR #111 description.* The full `--all-extras` suite gave
16116 passed. mypy was clean on dev, `--platform win32` and all extras, ruff
was clean, and the public API snapshot was unchanged. The J16 slice was the
real-data check. No runtime or memory was measured, and none was needed because
no performance gain is claimed. The independent review is not recorded.

Run encoder contracts, xarray/NetCDF and available NWB round trips, decoder
fit/session parity, scientist workflows and executable examples. Run Ruff,
mypy, public API snapshot and argument-convention checks. Require exact parity
on the same backend when arithmetic is unchanged; any intentional numerical
tolerance or behavior change needs an explicit explanation and separate scope.
Check one smallest existing real-data slice if available, record its source,
and report when no suitable fixture is available. Measure runtime and peak
memory on the same representative inputs before claiming any performance gain.

Completion means all setup migrations have parity evidence, duplication removed
without a new public abstraction, and independent reviews resolved. Phase 0
must precede implementation.
