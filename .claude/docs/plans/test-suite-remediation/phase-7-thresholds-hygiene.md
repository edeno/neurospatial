# Phase 7 — Threshold tightening + simulation strengthening + hygiene

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Sweep across the suite to (a) replace loose thresholds with bracketed assertions or known-answer values, (b) add ground-truth-recovery thresholds where the audit found "tests of simulator → detector that pass on random data," (c) add `env_2d.connectivity` is_connected checks to open-field mazes, (d) delete or fix empty test stubs, and (e) handle naming hygiene (rename `test_m10_*`, inline `test_validation_new.py`, add `sungod` maze tests).

This is a sweep of small, mostly-independent edits. The PR is large in file count but small in net LOC.

**Inputs to read first:**

- [tests/segmentation/test_laps.py:16-44, 43-44, 79, 146](../../../tests/segmentation/test_laps.py) — current `len(laps) >= 2` on a 4-lap simulation.
- [tests/simulation/test_validation_sim.py:66-87](../../../tests/simulation/test_validation_sim.py) — current `center_errors >= 0` no-op assertion.
- [tests/simulation/test_integration.py:238-281, 283-348](../../../tests/simulation/test_integration.py) — `>= 2/5` detection threshold; `error_long <= error_short + bin_size` degraded assertion.
- [tests/simulation/mazes/test_barnes.py](../../../tests/simulation/mazes/test_barnes.py), `test_cheeseboard.py`, `test_watermaze.py` — no `env_2d.connectivity` is_connected check.
- [tests/simulation/mazes/test_honeycomb.py:210-219](../../../tests/simulation/mazes/test_honeycomb.py) — `test_hexagonal_connectivity` lies about what it tests.
- [tests/encoding/test_grid_cell_metrics.py:262, 292](../../../tests/encoding/test_grid_cell_metrics.py) — `score > 0.0` for hex pattern, `abs(score) < 0.3` for noise.
- [tests/encoding/test_encoding_phase_precession.py:180](../../../tests/encoding/test_encoding_phase_precession.py) — `slope < 0` only.
- [tests/behavior/test_path_efficiency.py:316-322](../../../tests/behavior/test_path_efficiency.py) — empty `TestPathEfficiencyResult` and `TestComputePathEfficiency` stubs.
- [tests/animation/test_video_performance.py:365, 442](../../../tests/animation/test_video_performance.py) — `speedup > 1.5` and `0.8 < speedup < 1.5` flaky perf assertions.
- [tests/animation/test_benchmarks.py:188, 195](../../../tests/animation/test_benchmarks.py) — `speedups[2] >= 1.2`, `speedups[4] >= 1.4` flaky.
- [tests/animation/test_frame_naming.py:155-187](../../../tests/animation/test_frame_naming.py) — `contextlib.suppress(Exception)` + `if mock_run.called` conditional assertions.
- [tests/animation/test_rendering_validation.py](../../../tests/animation/test_rendering_validation.py) — 36 lines, one test (shape error). Audit recommended expanding or deleting.
- [tests/test_m10_old_files_deleted.py](../../../tests/test_m10_old_files_deleted.py) — scaffolding-named.
- [tests/test_validation_new.py](../../../tests/test_validation_new.py) — orphan `_new` suffix; 105 lines, 7 tests.
- [src/neurospatial/simulation/mazes/sungod.py](../../../src/neurospatial/simulation/mazes/sungod.py) — 369 LOC with zero test file.

## Tasks

### 1. Tighten lap-detection thresholds

In [tests/segmentation/test_laps.py](../../../tests/segmentation/test_laps.py):
- Line 16-44, `test_detect_laps_circular_track_auto`: simulation produces exactly 4 laps. Change `assert len(laps) >= 2` → `assert len(laps) in {3, 4}` (allow one boundary lap to be dropped, but no more).
- Line 79, `test_detect_laps_direction_clockwise`: similar — pin to the known lap count from the simulation.
- Line 146, region-method test: pin similarly.

### 2. Strengthen simulation→detection thresholds

In [tests/simulation/test_validation_sim.py:66-87](../../../tests/simulation/test_validation_sim.py): replace `assert (center_errors >= 0).all()` with `assert center_errors.mean() < 3 * bin_size` and `assert correlations.mean() > 0.5`. Choose thresholds based on the known signal-to-noise of the validation simulator (run once locally and pick a threshold below the observed mean by ~20% to leave a noise budget).

In [tests/simulation/test_integration.py:238-281](../../../tests/simulation/test_integration.py): `test_place_field_detection_accuracy`. Replace `len(detected) >= 2 and all in [0, 110]` with: for each true center, find the closest detected field; assert at least 4 of 5 are matched within `2 * bin_size`. (If five-fields is too tight, three-of-five is the minimum acceptable threshold.)

### 3. Lap simulation/detection round-trip

In [tests/simulation/test_trajectory_sim.py](../../../tests/simulation/test_trajectory_sim.py) or `tests/segmentation/test_laps.py`, add `TestLapRoundTrip::test_simulate_then_detect_recovers_lap_count`:
- `simulate_trajectory_laps(n_laps=5, ...)` → positions+times → `detect_laps(positions, times, ...)` → `assert detected_count == 5` (or `in {4, 5}` if one boundary lap can be dropped).
- Pins the two halves of the lap stack to each other.

### 4. Open-field maze connectivity checks

In each of `tests/simulation/mazes/test_barnes.py`, `test_cheeseboard.py`, `test_watermaze.py`: add `test_env_2d_connectivity_is_connected`:

```python
def test_env_2d_connectivity_is_connected(self, maze):
    import networkx as nx
    assert nx.is_connected(maze.env_2d.connectivity)
```

(Adjust to whatever the env-2d attribute is called — read each maze class first.)

### 5. Fix the lying `test_hexagonal_connectivity` test

In [tests/simulation/mazes/test_honeycomb.py:210-219](../../../tests/simulation/mazes/test_honeycomb.py), `test_hexagonal_connectivity`:
- **Option A**: implement a real hex-degree check on the pre-discretized graph if it's accessible. Hex interior cells have degree 6.
- **Option B**: rename the test to `test_honeycomb_graph_is_connected` to match what it actually checks. Drop the misleading comment.

Default: Option B if Option A requires significant new fixture work.

### 6. Strengthen object-vector test left/right symmetry

Phase 1 added a left/right symmetry test for object-ahead. Extend in [tests/simulation/models/test_object_vector_cells.py](../../../tests/simulation/models/test_object_vector_cells.py):
- Parametrize the symmetry assertion over `preferred_direction ∈ {0, π/2, π, -π/2}`. At each preferred direction, the rate at `heading=preferred+δ` should equal the rate at `heading=preferred-δ` for any δ. 4 parametrizations.

### 7. Tighten grid-score thresholds

In [tests/encoding/test_grid_cell_metrics.py:262, 292](../../../tests/encoding/test_grid_cell_metrics.py):
- Line 292, perfect hex pattern: `score > 0.0` → `0.4 < score < 1.5`.
- Line 262, noise: `abs(score) < 0.3` → keep, but add a positive lower bound: `-0.3 < score < 0.3` (already implied by `abs`, but make explicit so a regression returning `+ε` still fails).

### 8. Tighten phase-precession sign assertion

In [tests/encoding/test_encoding_phase_precession.py:180](../../../tests/encoding/test_encoding_phase_precession.py): the existing `slope < 0` assertion stays, but Phase 2's slope-magnitude test (`TestPhasePrecessionSlopeMagnitude::test_slope_magnitude_recovered`) is now the primary correctness check. Add a comment in this test pointing readers to the Phase 2 magnitude tests:

```python
# Sign-only check; for magnitude recovery, see TestPhasePrecessionSlopeMagnitude.
assert result.slope < 0
```

### 9. Delete empty path-efficiency stubs

In [tests/behavior/test_path_efficiency.py:316-322](../../../tests/behavior/test_path_efficiency.py): `TestPathEfficiencyResult` and `TestComputePathEfficiency` are empty classes. Delete both. The other tests in this file already exercise `compute_path_efficiency` adequately.

If executor finds that the public `compute_path_efficiency` function lacks behavioral coverage in the rest of the file, add minimal coverage (straight path → efficiency=1; U-turn → efficiency=0.5) before deleting the stubs.

### 10. Fix or guard flaky perf thresholds

In [tests/animation/test_video_performance.py:365](../../../tests/animation/test_video_performance.py): `speedup > 1.5`. Replace with `print(f"speedup={speedup:.2f}")` and remove the assertion. Move the test to a `@pytest.mark.benchmark` class that's excluded from default CI.

At line 442: `0.8 < speedup < 1.5`. The upper bound is an anti-improvement guard — **delete** the assertion entirely. Keep as a `print`.

In [tests/animation/test_benchmarks.py:188, 195](../../../tests/animation/test_benchmarks.py): same pattern — convert to `print`-only under `@pytest.mark.benchmark`.

### 11. Fix `test_frame_naming.py` conditional assertions

In [tests/animation/test_frame_naming.py:155-187](../../../tests/animation/test_frame_naming.py): remove the `contextlib.suppress(Exception)` wrapper and the `if mock_run.called:` guard. The test should fail loudly if the subprocess wasn't called, not silently pass.

```python
# Replace:
#   with contextlib.suppress(Exception):
#       render_video(...)
#   if mock_run.called:
#       cmd = mock_run.call_args[0][0]
#       assert "frame_" in pattern
# With:
render_video(...)  # must succeed
assert mock_run.called, "render_video did not invoke ffmpeg subprocess"
cmd = mock_run.call_args[0][0]
pattern = ...  # extract pattern arg from cmd
assert "frame_" in pattern
assert "%0" in pattern
```

### 12. Expand or delete `test_rendering_validation.py`

Phase 4 already cross-checked this file is not duplicating new parity work. Decide here:
- If `field_to_rgb_for_napari` (or equivalent) is the only function in `src/neurospatial/animation/_rendering.py` that's not covered by Phase 4's parity tests: expand `test_rendering_validation.py` with one test asserting RGB values match a hand-computed expectation for a known field.
- Otherwise: delete the file and let Phase 4 own rendering validation.

Read both files to make this judgment.

### 13. Rename `test_m10_old_files_deleted.py`

Rename [tests/test_m10_old_files_deleted.py](../../../tests/test_m10_old_files_deleted.py) → `tests/test_deprecated_modules_removed.py`.

Update the module docstring at the top (lines 1-9) to drop the "Milestone 10" reference; replace with a one-liner: `"""Guard tests that assert deprecated module paths no longer resolve."""`.

The CLAUDE.md Tier 4 fix (Phase 1) already addresses the documented-but-deleted modules list this file enforces.

### 14. Inline `test_validation_new.py` or rename

Read [tests/test_validation_new.py](../../../tests/test_validation_new.py) — 105 lines, 7 tests. Determine:
- If it's testing the same surface as `tests/test_type_validation.py`: move the 7 tests into `test_type_validation.py` and delete `test_validation_new.py`.
- If it's testing a distinct domain: rename to a domain-meaningful name (read the tests to determine — likely `test_input_validation.py` or `test_argument_validation.py`).

Either way, the `_new` suffix is removed.

### 15. Add tests for `sungod` maze

Create [tests/simulation/mazes/test_sungod.py](../../../tests/simulation/mazes/test_sungod.py) (new file, ~80 LOC) mirroring the pattern of `tests/simulation/mazes/test_barnes.py`:
- `test_construct_default`: maze instantiates without error.
- `test_env_2d_is_connected`: `nx.is_connected(maze.env_2d.connectivity)`.
- `test_n_bins_reasonable`: bin count is within expected range for the maze geometry.
- `test_region_positions`: spot-check that named regions (if any) are at sensible coordinates.

Read [src/neurospatial/simulation/mazes/sungod.py](../../../src/neurospatial/simulation/mazes/sungod.py) first to learn the maze geometry and what assertions are scientifically meaningful.

## Deliberately not in this phase

- **No removal of audit-cited weak tests beyond the empty stubs in Task 9.** The "tests that construct results from synthetic data and read accessors" pattern is left in place; Phases 2 and 3 added strong tests alongside.
- **No widget / napari / NWB-adapter mock removal.** Phase 8.
- **No code consolidation of `tests/animation/test_overlays.py` (125 tests) or the per-backend overlay files.** Audit flagged duplication; deferred. Phase 4 only handled `test_video_overlay.py` (6 tests) → `test_video_overlays.py` dedup.
- **No moving `tests/test_properties.py` (2167 lines).** Audit confirmed it's well-organized as-is.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/segmentation/test_laps.py::test_detect_laps_*` (3 tests, tightened) | Detected lap count matches simulated count within ±1. |
| `tests/simulation/test_validation_sim.py::test_validate_simulation` (tightened) | `center_errors.mean() < 3 * bin_size`, `correlations.mean() > 0.5`. |
| `tests/simulation/test_integration.py::test_place_field_detection_accuracy` (tightened) | ≥ 4 of 5 true centers matched within `2 * bin_size`. |
| `tests/segmentation/test_laps.py::TestLapRoundTrip::test_simulate_then_detect_recovers_lap_count` | `simulate_trajectory_laps → detect_laps` recovers the simulated count. |
| `tests/simulation/mazes/test_{barnes,cheeseboard,watermaze}.py::test_env_2d_connectivity_is_connected` (3 tests) | Open-field mazes have connected 2D environments. |
| `tests/simulation/mazes/test_honeycomb.py::test_honeycomb_graph_is_connected` (renamed) | Honeycomb graph is connected (test name now matches assertion). |
| `tests/simulation/models/test_object_vector_cells.py::test_left_right_symmetry[*]` (4 parametrizations) | OVC rate symmetric under heading reflection across `preferred_direction`. |
| `tests/encoding/test_grid_cell_metrics.py` (tightened thresholds at lines 262, 292) | Grid score in `[0.4, 1.5]` for perfect hex; in `[-0.3, 0.3]` for noise. |
| `tests/behavior/test_path_efficiency.py` (stubs deleted) | Empty `TestPathEfficiencyResult` and `TestComputePathEfficiency` are gone. |
| `tests/animation/test_video_performance.py`, `test_benchmarks.py` (perf asserts removed) | Benchmarks print but don't assert flaky speedups. |
| `tests/animation/test_frame_naming.py` (conditional removed) | ffmpeg subprocess invocation is required, not conditional. |
| `tests/test_deprecated_modules_removed.py` (renamed) | Old `test_m10_*` file is gone; new name is in place. |
| `tests/test_validation_new.py` (inlined or renamed) | Orphan `_new` suffix is gone. |
| `tests/simulation/mazes/test_sungod.py` (4 tests) | `sungod` maze has dedicated test file. |

## Fixtures

No new shared fixtures. Existing per-maze fixtures in `tests/simulation/mazes/conftest.py` are extended only if `sungod` needs setup that isn't inline-constructible.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 8 (mock removal).
- Validation slice tests pass; no new slow tests.
- Tests aren't trivial — threshold tightening uses bracketed assertions tied to ground truth, not just changes of inequality direction. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones (in particular, the renamed `test_deprecated_modules_removed.py` docstring drops the "Milestone 10" reference).
- Old code paths flagged for removal in this phase are actually removed: empty `TestPathEfficiencyResult`/`TestComputePathEfficiency` stubs, `test_m10_*` file (renamed), `test_validation_new.py` (renamed/inlined), `0.8 < speedup < 1.5` upper bound, `contextlib.suppress` and conditional `if mock_run.called` blocks.
- User-facing documentation listed as tasks is updated, not deferred (none in this phase).
