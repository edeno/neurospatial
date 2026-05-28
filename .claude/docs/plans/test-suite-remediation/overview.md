# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

The audit identified specific lines in shipping code and tests that this plan touches. Phases reference these by file:line; the list below is the canonical anchor set.

**Source — behavior changes (Phase 1 only):**
- [src/neurospatial/ops/egocentric.py:569](../../../src/neurospatial/ops/egocentric.py) — `compute_egocentric_distance(metric="geodesic")` loop iterates `targets_3d[0]`, ignoring time-varying targets. Behavior fix.
- [src/neurospatial/events/regressors.py:10](../../../src/neurospatial/events/regressors.py) — module docstring promises `exponential_kernel`. Function is not defined anywhere. Either implement or remove docstring entry.
- [src/neurospatial/encoding/egocentric.py:1610](../../../src/neurospatial/encoding/egocentric.py) — `object_vector_score` is exported (`__all__` at line 100) and has zero tests. Add tests; no source change.
- [CLAUDE.md](../../../CLAUDE.md) Tier 4 section — documents `encoding/place.py`, `head_direction.py`, `object_vector.py`, `spatial_view.py` which do not exist (actual: `spatial.py`, `directional.py`, `egocentric.py`, `view.py`). Doc fix.

**Source — read-only (everything else is test-only):**
- [src/neurospatial/encoding/spatial.py](../../../src/neurospatial/encoding/spatial.py) — `compute_spatial_rate`, Phase 2 wraps it in recovery tests.
- [src/neurospatial/encoding/directional.py](../../../src/neurospatial/encoding/directional.py) — `compute_directional_rate{,s}`, Phase 2 adds NumPy↔JAX parity + recovery.
- [src/neurospatial/encoding/egocentric.py](../../../src/neurospatial/encoding/egocentric.py) — `compute_egocentric_rate`, Phase 2 adds convention-pinning test.
- [src/neurospatial/encoding/view.py](../../../src/neurospatial/encoding/view.py) — `compute_view_rate` with `gaze_model={"fixed_distance","ray_cast","boundary"}`, Phase 2 adds geometric-correctness test.
- [src/neurospatial/encoding/phase_precession.py](../../../src/neurospatial/encoding/phase_precession.py) — Phase 2 pins slope magnitude.
- [src/neurospatial/decoding/posterior.py](../../../src/neurospatial/decoding/posterior.py) — `normalize_to_posterior`, Phase 3 closed-form test.
- [src/neurospatial/decoding/likelihood.py](../../../src/neurospatial/decoding/likelihood.py) — `log_poisson_likelihood`, Phase 3 long-trajectory underflow test.
- [src/neurospatial/decoding/trajectory.py](../../../src/neurospatial/decoding/trajectory.py) — `detect_trajectory_radon`, Phase 3 angle recovery.
- [src/neurospatial/stats/shuffle.py](../../../src/neurospatial/stats/shuffle.py) — Phase 3 pairing-destruction tests.
- [src/neurospatial/stats/circular.py](../../../src/neurospatial/stats/circular.py) — Phase 3 Rayleigh pinning.
- [src/neurospatial/ops/binning.py](../../../src/neurospatial/ops/binning.py) — `map_points_to_bins`, Phase 5 thorough behavioral tests.
- [src/neurospatial/layout/engines/](../../../src/neurospatial/layout/engines/) — `masked_grid.py`, `image_mask.py`, `shapely_polygon.py`, `hexagonal.py`, `graph.py`. Phase 5 adds dedicated test files.
- [src/neurospatial/ops/transforms.py](../../../src/neurospatial/ops/transforms.py) — `flip_y_data`, `convert_to_cm`, `convert_to_pixels`. Phase 5 adds tests (currently zero references in test tree).
- [src/neurospatial/io/nwb/](../../../src/neurospatial/io/nwb/) — writers for events/laps/trials/fields. Phase 6 adds disk round-trip tests.

**Tests — modified or added (per-phase index):**
- Phase 1: `tests/encoding/test_object_vector_score.py` (new), `tests/ops/test_reference_frames.py`, `tests/events/test_regressors.py`, `tests/simulation/models/test_object_vector_cells.py:498-516`.
- Phase 2: `tests/encoding/test_encoding_spatial.py`, `test_encoding_directional.py`, `test_compute_egocentric_rate.py`, `test_encoding_view_binning.py`, `test_phase_precession_metrics.py`, `test_jax_compute_dispatch.py`.
- Phase 3: `tests/decoding/test_posterior.py`, `test_likelihood.py`, `test_trajectory.py`, `tests/stats/test_stats_shuffle.py`, `test_circular_metrics.py`.
- Phase 4: `tests/animation/test_backend_consistency.py` (rewrite), `test_overlay_visual_regression.py`, delete `test_video_overlay.py` (merge into `test_video_overlays.py`).
- Phase 5: `tests/ops/test_binning.py`, new `tests/layout/test_masked_grid_layout.py`, `test_image_mask_layout.py`, `test_shapely_polygon_layout.py`, `test_hexagonal_layout.py`, `test_graph_layout.py`, new `tests/ops/test_transforms_helpers.py`, `tests/environment/test_polar_egocentric.py`.
- Phase 6: `tests/nwb/test_events.py`, `test_fields.py`, `test_trials.py`, `test_environment.py` (polygon round-trip).
- Phase 7: `tests/segmentation/test_laps.py`, `tests/simulation/test_validation_sim.py`, `test_integration.py`, `tests/simulation/mazes/test_barnes.py`, `test_cheeseboard.py`, `test_watermaze.py`, `test_honeycomb.py`, `tests/encoding/test_grid_cell_metrics.py`, `tests/behavior/test_path_efficiency.py`, `tests/animation/test_video_performance.py`, `tests/animation/test_benchmarks.py`, `tests/animation/test_frame_naming.py`, `tests/animation/test_rendering_validation.py`, rename `tests/test_m10_old_files_deleted.py`, inline `tests/test_validation_new.py`, new `tests/simulation/mazes/test_sungod.py`.
- Phase 8: `tests/animation/test_backend_consistency.py` (real-widget parity), `tests/nwb/test_adapters.py`, `tests/nwb/test_events.py:216-247`, `tests/nwb/test_pose.py:689`.

## Scope and dependency policy

### Goals

- Eliminate the five real bugs / silent-contract violations identified by the audit, each with a regression test.
- Make recovery and parity tests run **through the public compute functions on simulated ground truth**, not against hand-built result objects.
- Replace loose / wrong-direction thresholds (`>= 0`, `>= 2 of 5`, `0.8 < speedup < 1.5`) with bracketed assertions tied to ground truth or analytic reference values.
- Replace "claim parity, test one backend at a time" with actual cross-backend output comparison in animation.
- Add dedicated behavioral tests for under-tested core primitives (`map_points_to_bins`, 5 layout engines, `ops/transforms.py` helpers).
- Add real-disk NWB round-trip coverage matching the `Environment` baseline.
- Reduce mock surface so that CI without optional dependencies (Qt, napari, pynwb, ipywidgets) still exercises real code paths somewhere.
- Pay down hygiene: rename scaffolding-named test files, delete empty stubs, dedup near-duplicate test files, inline orphan `_new`-suffixed files.

### Non-Goals

- **No source refactors beyond the five red-table bug fixes.** Phases 2–8 are test-only. If an executor identifies a source bug while writing tests, surface in PR description; do not bundle.
- **No removal of existing weak tests in Phases 2–6.** Add strong tests alongside; let the weak ones become redundant. Phase 7 removes empty stubs and dedup files; Phase 8 leaves existing mock tests in place where they still document API surface.
- **No new dependencies.** Existing test-stack (pytest, hypothesis, pytest-mpl, importorskip for optional) is sufficient.
- **No coverage-percentage chasing.** Each phase ships specific named tests with named asserted behaviors. We do not track or assert against a `coverage --fail-under` number.
- **No backwards-compatibility shims for the geodesic-distance fix.** The current behavior is silently wrong; no caller can be intentionally depending on it. Per CLAUDE.md default: "just change the code."

### Dependency policy

No new runtime or dev dependencies. `pytest-mpl` (already present) gains additional baseline images in Phase 4. `pytest.mark.slow` is used to mark integration tests that take > 5s — see Phase 2/3 fixtures.

## Metrics

- **Bug count (red table):** 5 → 0 after Phase 1.
- **Backend parity:** Phase 4 adds ≥4 pixel-level cross-backend asserts in `test_backend_consistency.py` (currently 0).
- **NumPy↔JAX parity surface:** Phase 2 adds parity for `compute_directional_rate`, `compute_directional_rates`, and `gaussian_kde` smoothing path. After Phase 2, every public `compute_*_rate{,s}` has parity tests for all supported smoothing methods.
- **NWB writer disk-roundtrip coverage:** 1/5 (Environment only) → 5/5 after Phase 6.
- **Layout-engine dedicated test files:** 2/7 (RegularGrid, TriangularMesh) → 7/7 after Phase 5.
- **Recovery-test pattern:** Phase 2 adds end-to-end recovery for ≥4 cell types (place, HD, object-vector, phase-precession) where currently 0 exist that drive the public compute function.
- **Threshold quality:** Phase 7 replaces ≥10 specifically-identified loose-threshold asserts (see phase file).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Recovery tests added in Phase 2 are noisy under Poisson sampling and flake on CI. | Each recovery test fixes seed, simulates a large-N session (≥ 5000 spikes), and asserts `||detected - true|| < one bin_size` rather than a tight tolerance. Mark `@pytest.mark.slow` for runs ≥ 5s. |
| Phase 4 pixel comparison across napari/video/html/widget fails on different CI runners (font rendering, antialiasing). | Compare at low resolution (32×32 fields), use `np.allclose(atol=2)` on uint8 pixel arrays, and gate napari + ipywidgets paths behind `importorskip`. |
| Phase 5 layout-engine tests introduce new dependencies (Shapely) into the test path. | Shapely is already a runtime dependency for `ShapelyPolygonLayout`. The new test files just add `importorskip("shapely")` where needed — no install changes. |
| Phase 6 NWB disk round-trips slow down CI. | Mark `@pytest.mark.slow`; existing `Environment` round-trip (which already does this) sets the precedent and is not currently flaky. |
| Phase 8 real-widget tests are platform-dependent (Qt on macOS/Linux/Windows CI differs). | Use existing `pytest.importorskip("napari")` and `pytest.importorskip("ipywidgets")` patterns. Failing gracefully on missing Qt is already handled by commit c7c2808 (PlaybackController). |
| `exponential_kernel` decision (implement vs. remove docstring entry) made by executor without prior context. | Open Question 1 below — executor greps for callers; if none, remove docstring entry. Default decision documented in Phase 1. |

## Rollout Strategy

Sequential PRs in the order Phase 1 → Phase 8. Each phase ships independently on its own branch and merges to `main` (or the active dev branch) before the next phase begins. There is no feature-flag or parallel-running requirement — each phase's changes are additive to the test suite except where Phase 7 deletes empty stubs and Phase 4 deletes the duplicate `test_video_overlay.py`. Phase 1's source changes (geodesic-distance fix, optional `exponential_kernel`) are bug fixes; downstream callers can only benefit.

The three "source-touching" red-table items in Phase 1 ship in a single PR alongside their regression tests. The two pure-doc/test items in Phase 1 (CLAUDE.md Tier 4, OVC convention test) ship in the same PR.

### Cross-phase dependencies

While each phase is independently reviewable, some phases consume artifacts produced by earlier phases. **The sequential rollout above honors all of these by construction.** A reviewer who wants to merge phases out-of-order must address these first:

- **Phase 4 → Phase 8.** Phase 8's real-widget/napari parity tests strengthen the parity tests that Phase 4 introduces. Out-of-order: Phase 8 has nothing to extend.
- **Phase 6 → Phase 2.** Phase 6 Task 5 (`test_fields_survive_disk_roundtrip`) builds a `SpatialRateResult` for the round-trip test. Phase 2 introduces the `place_cell_session` conftest fixture that makes this construction one line. Out-of-order: Phase 6 must synthesize the fixture inline, doubling the test setup.
- **Phase 7 → Phase 1.** Phase 7 Task 6 parametrizes the left/right-symmetry assertion that Phase 1 introduces in `tests/simulation/models/test_object_vector_cells.py`. Out-of-order: Phase 7 must add the base test it intends to extend.

## Open Questions

1. **`exponential_kernel` — implement or remove docstring entry?** The module docstring at [src/neurospatial/events/regressors.py:10](../../../src/neurospatial/events/regressors.py) advertises the function but no implementation exists.
   - **Default for Phase 1:** Executor greps the repo (and `examples/`) for `exponential_kernel` callers. If zero callers, remove the docstring entry. If any caller, implement (signature: `exponential_kernel(times, event_times, tau)` returning `(n_times,)` array, computing `sum_e exp(-(t - t_e)/tau) for t_e <= t`).
   - The repo has no prior usage outside docstrings (verified during planning). Executor confirms with `grep -rn exponential_kernel src/ tests/ examples/`.

2. **Phase 4 pixel tolerance across backends.** Different renderers (matplotlib for video, OpenGL for napari, embedded PNGs for html, ipywidgets PNG bytes for widget) will not be bit-identical. The plan fixes `atol=2` on uint8 RGB at 32×32 — if this turns out to be too tight for napari (which uses OpenGL), executor relaxes to `atol=5` and documents in the PR.

3. **Layout-engine ImageMaskLayout fixture data.** Phase 5 tests for `ImageMaskLayout` need a small mask image. Plan suggests synthesizing a 16×16 boolean mask in `conftest.py`; if real-data parity is wanted, a tiny PNG committed to `tests/layout/fixtures/` is also acceptable.

## Estimated Effort

LOC sanity check by phase (test code only except Phase 1):

| Phase | Test LOC (added) | Source LOC (changed) | Files touched |
|---|---|---|---|
| 1 | ~250 | ~30 | 5 src, 4 test, 1 doc |
| 2 | ~500 | 0 | 6 test |
| 3 | ~400 | 0 | 5 test |
| 4 | ~350 (incl. dedup −200) | 0 | 3 test (1 deleted) |
| 5 | ~800 | 0 | 8 test (5 new) |
| 6 | ~300 | 0 | 4 test |
| 7 | ~200 | 0 | ~12 test (1 renamed, 1 inlined, 1 new) |
| 8 | ~250 | 0 | 4 test |

Total: ~3050 test LOC added, ~30 source LOC changed across ~50 files.
