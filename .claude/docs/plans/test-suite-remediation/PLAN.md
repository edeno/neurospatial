# Test-Suite Remediation Plan

**Status:** Not started.

A full-suite audit of `tests/` (301 files, ~160k LOC, 6 parallel reviewer agents) surfaced five real bugs / silent-contract violations in shipping code, recurring "plumbing-tested, behavior-untested" patterns in encoding/decoding/simulation, a load-bearing animation-backend parity claim that does not actually compare backends, severely under-tested core primitives (`map_points_to_bins`, 5 layout engines), missing NWB disk round-trips for everything except `Environment`, and assorted hygiene drift. This plan decomposes the remediation into 8 independently-shippable PRs ordered by user-visible-risk first, then coverage, then quality, then hygiene/mocks.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, metrics, open questions, estimated effort.
- Phases (each ships as a separable PR):
  - [phase-1-red-table-bugs.md](phase-1-red-table-bugs.md) — fix the five real bugs surfaced by the audit, each with a regression test.
  - [phase-2-encoding-recovery.md](phase-2-encoding-recovery.md) — end-to-end recovery tests through `compute_*_rate` functions; NumPy↔JAX directional parity.
  - [phase-3-decoding-stats-correctness.md](phase-3-decoding-stats-correctness.md) — closed-form posterior, Radon angle recovery, shuffle pairing destruction, Rayleigh pinning, long-trajectory underflow.
  - [phase-4-animation-parity.md](phase-4-animation-parity.md) — rewrite `test_backend_consistency.py` to actually compare pixels across napari/video/html/widget; dedup `test_video_overlay.py`.
  - [phase-5-core-primitives.md](phase-5-core-primitives.md) — `map_points_to_bins` behavioral tests, dedicated layout-engine test files, `ops/transforms.py` helpers, polar-egocentric angular seam, heading edge cases.
  - [phase-6-nwb-roundtrip.md](phase-6-nwb-roundtrip.md) — `NWBHDF5IO` write→close→reopen tests for laps, trials, region_crossings, events, fields.
  - [phase-7-thresholds-hygiene.md](phase-7-thresholds-hygiene.md) — tighten loose thresholds, strengthen simulation→detection assertions, env_2d connectivity checks, delete empty stubs, rename `test_m10_*`, inline `test_validation_new.py`, add `sungod` maze tests.
  - [phase-8-mock-removal.md](phase-8-mock-removal.md) — add real-path tests for ipywidgets / napari / pynwb-adapter scenarios; drop `monkeypatch(builtins.__import__)` tests.
