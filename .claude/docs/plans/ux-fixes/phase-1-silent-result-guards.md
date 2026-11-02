# Phase 1 — Warn or raise on wrong-but-plausible results

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

**Inputs to read first:**

- [src/neurospatial/_intervals.py:1158-1211](../../../../src/neurospatial/_intervals.py) — `IntervalGates` and `narrowed`.
- [src/neurospatial/_intervals.py:1699-1759](../../../../src/neurospatial/_intervals.py) — `warn_excluded_intervals`, the single exit every analysis passes through.
- [src/neurospatial/encoding/_spatial_counts.py:60-130](../../../../src/neurospatial/encoding/_spatial_counts.py) — the existing spike-drop warnings and the 50% threshold.
- [src/neurospatial/decoding/posterior.py:553-712](../../../../src/neurospatial/decoding/posterior.py) — `_prepare_decode_inputs`, including the NaN-rate warning to mirror.
- [src/neurospatial/encoding/directional.py:2230-2330](../../../../src/neurospatial/encoding/directional.py) — angular width validation.
- [src/neurospatial/_exceptions.py:68](../../../../src/neurospatial/_exceptions.py) — `_format_error`.
- `docs/reviews/UX_REVIEW_2026-11-02.md` §1 — the reproductions (S1–S9).

**Contracts referenced:**

- [Discarded-data warning](shared-contracts.md#discarded-data-warning) — the threshold, message and mechanics. Do not weaken.
- [Zero-warning golden path](shared-contracts.md#zero-warning-golden-path) — this phase creates the test.

**Designs referenced:** [designs.md](designs.md), sections S1–S9.

## Tasks

- **Golden-path test first.** Add `tests/test_golden_path_warnings.py` as
  specified in the contract. It must pass on `main` before any guard lands.
  It proves the guards add no noise.
- **Shared constant.**
  - Add `DISCARD_WARN_FRACTION = 0.05` to `src/neurospatial/_constants.py`,
    with a one-line comment giving the reason (designs S1).
  - Replace `_DROP_WARN_THRESHOLD` (`encoding/_spatial_counts.py:66-68`)
    with it, and delete the old constant and its comment (S6).
- **S1 and S2 in `_intervals.py`.**
  - Add `IntervalGates.bin_dropped`, set in `narrowed`.
  - Add the outside-environment branch and `_masked_sample_fraction` to
    `warn_excluded_intervals`, with the `extent=` keyword.
  - Update the 11 call sites (`grep -rn "warn_excluded_intervals(" src/`)
    that apply a bin gate to pass `extent`. Code is in designs S1 and S2.
- **S3.** Replace `_check_smoothing_bandwidth` with `_check_angular_widths`
  (designs S3), keeping its one call site. Add an "Angular widths are in
  `angle_unit`" sentence to the `bin_size`/`bandwidth` parameter docs of
  `compute_directional_rate(s)` and `head_direction_cell_significance`.
- **S4 and S5 in decoding.**
  - Zero-rate warning in `_prepare_decode_inputs` (designs S4).
  - `dt` vs `times` refusal in `decode_position` and `decode_position_summary`.
  - The milliseconds heuristic in `bin_spikes_in_time` (designs S5).
- **S7.** Make `occupancy` keyword-only in the four metric functions (designs
  S7). Update every in-repo caller, docstring example and notebook:
  `grep -rn "spatial_information(\|sparsity(\|selectivity(\|information_per_second(" src docs examples`.
- **S8.** Loop check in `detect_laps` (designs S8).
- **S9.** Warn when a spatial bandwidth is below the bin length (designs S9).
- **Docs.**
  - `docs/errors.md` gets a "Data discarded at the boundary" section listing
    the new warnings and how to silence each (`warn_on_drop=False` where it
    exists).
  - CHANGELOG "Unreleased": S3, S5, S7 and S8 are **Breaking** (they raise,
    or change the signature). The rest go under "Changed".

## Deliberately not in this phase

- Raising when *every* interval is excluded (ms times, DatetimeIndex) belongs
  to phase 6. This phase only adds partial-exclusion warnings.
- Wrong-kind inputs (list of trains to a single-unit function, result object
  as an array) belong to phase 6.
- Renaming `bin_size` → `dt` on PSTH functions belongs to phase 3a. Do not
  touch PSTH signatures here.
- No changes to estimator numerics.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_golden_path_warnings.py::test_documented_workflow_emits_no_warnings` | The full golden path runs under `simplefilter("error")`. |
| `test_outside_environment_positions_warn` | `open_field_session(rng=0)` positions ×8 on its cm env → one `UserWarning` whose text contains the kept seconds, the session seconds and both x ranges. `n_spikes_outside_intervals` is unchanged. |
| `test_brief_tracking_dropout_is_silent` | 1% of samples set to NaN in short runs → no warning. |
| `test_boolean_masked_samples_warn` | `times[speed > 5]` → warning naming `min_speed=`. The same call with `min_speed=5.0` on full arrays → no warning. |
| `test_dropped_frames_are_silent` | 30 Hz times with 0.5% of frames removed at random → no warning. |
| `test_angular_bin_size_above_pi_raises` | `compute_directional_rate(..., bin_size=6)` → `ValueError` mentioning `angle_unit='deg'`. `bin_size=6, angle_unit="deg"` → 60 bins. |
| `test_wide_angular_bandwidth_warns` | `bandwidth=10` (rad) warns. `bandwidth=np.pi/6` is silent. |
| `test_zero_filled_rates_warn` | `decode_position(env, counts, np.nan_to_num(rates.firing_rates), dt=)` warns. Passing `rates` itself is silent. |
| `test_decode_dt_mismatch_raises` | `times` at 25 ms with `dt=25` → `ValueError` quoting both values. |
| `test_bin_spikes_dt_in_ms_warns` | `bin_spikes_in_time(spikes_in_s, dt=25)` warns "milliseconds". |
| `test_partial_spike_drop_warns_above_5pct` | Spikes shifted so 28% fall outside → warning. 2% outside → silent. |
| `test_metric_occupancy_is_keyword_only` | `spatial_information(rate, occ)` → `TypeError`. `spatial_information(rate, occupancy=occ)` gives the same value as before the change. |
| `test_detect_laps_refuses_linear_track` | `linear_track_session` env → `ValueError` naming `detect_trials`. A loop env still detects laps. |
| `test_tiny_spatial_bandwidth_warns` | `bandwidth=0.05` on a 2 cm env warns. `method="binned", bandwidth=0` is silent. |

## Fixtures

Simulated only:

- `open_field_session(rng=0, duration=120.0)`, or `seed=0` until phase 3a
  lands.
- `linear_track_session(seed=0)`.
- A small synthetic HD cell, built as in
  `docs/reviews/UX_REVIEW_2026-11-02.md` or `tests/encoding` fixtures.
- A loop environment from an existing behavior test fixture
  (`grep -rn "circular\|loop" tests/behavior/conftest.py`).

Put shared fixtures in the nearest `conftest.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
