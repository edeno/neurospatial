# Phase 3e — Time windows: kinematics, `heading_from_velocity(positions, times)` and `add_positions`

**Requires:** Phases 3d and 2b (3d owns the new keywords on `compute_pre_decision_metrics` and the VTE functions; this phase only forwards them).

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics) · [3a](phase-3a-time-windows-core.md) · [3d](phase-3d-time-windows-segmentation.md)

This is the fifth Phase 3 PR (the split is tabled in [3a](phase-3a-time-windows-core.md)). It makes every position-only **kinematic quantity** (velocity, speed, heading, turn angle, dwell) and `events.add_positions` respect recording gaps. It is independent of 3b, 3c and 3d. It requires Phase 2b because it changes the signature of `heading_from_velocity` and of Phase 2b's `_velocity_heading_and_speed`, in code Phase 2b rewrote.

Follow [executing.md](executing.md) for branching, commits, CHANGELOG bullets, the definition of done and the PR.

**One rule for this PR (shared with 3d):** each maximal run of valid intervals is analyzed **as if it were a separate recording**.

- **Kinematic quantities** are never computed across an invalid interval. At run boundaries, samples get the treatment the function already gives the first or last sample of a recording. A sample that belongs to no run (both neighbouring intervals invalid) gets NaN.
- For a gap-free input with `epochs=None`, there is one run covering every sample, so outputs are identical to the base commit. The gap gate is the existing `max_gap=0.5` (added where absent) and `epochs=None`, which together form the contract's position-only row.

**Inputs to read first:**

- **3a's merged code is the source of truth**, not the 3a plan text:
  - [src/neurospatial/_intervals.py](../../../../src/neurospatial/_intervals.py): `run_sample_bounds`, `run_time_bounds`.
  - [src/neurospatial/environment/trajectory.py](../../../../src/neurospatial/environment/trajectory.py): `observed_interval_mask(times, *, max_gap, epochs)` and `observed_runs(times, *, max_gap, epochs) -> list[slice]`, which normalize a raw `epochs` argument themselves.
- [src/neurospatial/ops/egocentric.py:655](../../../../src/neurospatial/ops/egocentric.py#L655). This is `heading_from_velocity(positions, dt, *, min_speed, bandwidth, allow_all_nan)`. It takes a **scalar** `dt` (velocity at :768), so the step across a pause reads as a very fast movement and produces a fake heading. Its `src/` callers are `decisions.py:464`, `vte.py:365`, `navigation.py:1780` and `simulation/spikes.py:429`; `simulation/models/head_direction_cells.py:107` uses it in a docstring example.
- [src/neurospatial/behavior/decisions.py](../../../../src/neurospatial/behavior/decisions.py): `pre_decision_heading_stats` (:402, median dt at :460), `pre_decision_speed_stats` (:488, inline speed at 523–526), and `compute_pre_decision_metrics` (:531), which calls both on the extracted window (:585, :590).
- [src/neurospatial/behavior/vte.py](../../../../src/neurospatial/behavior/vte.py):
  - `head_sweep_magnitude` (:271), which drops NaN headings and then diffs across the hole at :311;
  - `head_sweep_from_positions` (:319, median dt at :360);
  - `compute_vte_trial` (:534, inline speed at 598–602);
  - `compute_vte_session` (:617, inline speed at 728–732).
- [src/neurospatial/behavior/navigation.py](../../../../src/neurospatial/behavior/navigation.py):
  - `heading_direction_labels` (:1107, inline speed at 1192–1196);
  - `compute_path_efficiency` (:1570, path length at :1619);
  - `instantaneous_goal_alignment` (:1732, median dt at :1776);
  - `goal_bias` (:1791);
  - `approach_rate` (:1844, `_positive_dt` at :1924);
  - `compute_goal_directed_metrics` (:1933).
- [src/neurospatial/behavior/trajectory.py](../../../../src/neurospatial/behavior/trajectory.py): `compute_home_range` (:360, dwell from `np.diff(times)` at :454 and a last-sample median at :457) and `compute_trajectory_curvature` (:754, Gaussian sigma from the median dt at :865).
- [src/neurospatial/events/detection.py:20](../../../../src/neurospatial/events/detection.py#L20). This is `add_positions`. It sorts the trajectory by time, then its `interp1d(..., fill_value="extrapolate")` (205–211) **invents** positions for events inside a pause or outside the tracked span.
- Evidence (session scratchpad): `branch-triage.md` §A2 and `catA/gap4.py`. On 3a's two-epoch geometry (50 Hz), `heading_from_velocity` with a scalar `dt = 0.02` s reads the cross-gap step as about 1,911 cm/s.
- Archive reference only: `e9eee2c5` (trajectory and heading).
- **Files Phase 2b already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `ops/egocentric.py`: Phase 2b Task 3 rewrote `_interpolate_heading_circular` (shorter arc, uniform in angle). Phase 2b Task 4 added `_velocity_heading_and_speed(positions, dt, *, bandwidth)`, which `heading_from_velocity` and `instantaneous_goal_alignment` share, and made `instantaneous_goal_alignment` NaN below `min_speed`.
  - `behavior/decisions.py`: Phase 2b Task 5 made `pre_decision_heading_stats` use `_velocity_heading_and_speed` and exclude samples below `min_speed`.
  - `behavior/vte.py`: Phase 2a Task 7 made `compute_vte_session` cut each window from the trial's own samples and clamp `window_start` to `trial.start_time`. Keep that clamp.
  - `behavior/decisions.py`, `behavior/vte.py`: if 3d has merged, it added `max_gap`/`epochs` to `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session`, and made `extract_pre_decision_window` return a single-run window (Task 3 says how to share the keywords).

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics): the position-only row. No velocity, speed, heading or dwell spans an invalid interval.
- [Error-message contract](shared-contracts.md#error-message-contract). `epochs` errors come from `as_intervals`. The new `heading_from_velocity` `times` validation (1-D, same length as `positions`, strictly increasing) follows the contract.
- [Input conventions](shared-contracts.md#input-conventions). New arguments are keyword-only. `heading_from_velocity` swaps its scalar `dt` for `times` **in the same position**; the `(times, positions)` reordering of behavior functions belongs to Phase 6b.

**Designs referenced:** none.

## Inventory (this PR's slice)

None of these has `max_gap` or `epochs` on `main`. Every row adds keyword-only `max_gap: float | None = 0.5, epochs=None` after the existing keywords and applies the rule above.

| Function (file:line) | Current gap behavior on `main` | Change in 3e |
| --- | --- | --- |
| `ops.egocentric.heading_from_velocity` ops/egocentric.py:655 | scalar `dt`; teleport heading; low-speed interpolation bridges the pause | **`dt` replaced by `times`**; per-run velocity, smoothing and interpolation |
| `pre_decision_heading_stats` dec:402 | median-dt heading; the teleport heading counts | `_velocity_heading_and_speed` per run (Task 2); invalid intervals give NaN, so the teleport heading is excluded like a stationary sample |
| `pre_decision_speed_stats` dec:488 | speed across the pause ≈ 0 drags the minimum and mean | per-interval speed with NaN on invalid intervals; nan-aware reductions |
| `head_sweep_from_positions` vte:319 | teleport heading plus the hole-bridging diff inflate IdPhi | heading per run; IdPhi = sum over runs of `head_sweep_magnitude` |
| `heading_direction_labels` nav:1107 | a post-pause sample gets speed ≈ 0 by accident, or the teleport direction when `min_speed=0` | a sample whose backward interval is invalid is labeled like the first sample (`"stationary"`) |
| `compute_path_efficiency` nav:1570 | the jump adds to path length; two fake turn angles | `traveled_length`, `efficiency` and `angular_efficiency` are NaN when the samples include an invalid interval (the path is unknowable); `time_efficiency` stays wall-clock |
| `instantaneous_goal_alignment` nav:1732 | teleport heading | gap-aware heading via `_velocity_heading_and_speed` per run; samples in no run are NaN, as are samples below `min_speed` (Phase 2b) |
| `goal_bias` nav:1791 | inherits | forwards; nan-aware mean already |
| `approach_rate` nav:1844 | a made-up rate at the first post-pause sample | NaN on invalid intervals |
| `compute_goal_directed_metrics` nav:1933 | inherits; `time_to_goal` is wall-clock | forwards; `time_to_goal` unchanged (wall-clock time is known across a gap) |
| `compute_trajectory_curvature` traj:754 | fake turn angles at the jump, then smoothed across it | when `times` is given: per run; samples in no run are NaN |
| `compute_home_range` traj:360 | the last pre-pause sample gets ≈1000 s of dwell | dwell from valid intervals only; the last sample of each run gets that run's median `dt` (today's last-sample rule) |
| `events.add_positions` detection.py:20 | linear interpolation **across the pause**, and extrapolation outside the span | NaN position for an event outside every observed run (Task 4) |

That is 13 of the 29 position-only "Change" functions; 3d owns the other 16. Three of 3d's rows also change here, because they compute kinematics on the extracted window: `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session` (Task 3).

The "no change" list (for example `mean_square_displacement` and `time_efficiency`) and its reasons are in [3d](phase-3d-time-windows-segmentation.md#inventory-this-prs-slice).

## Tasks

### 1. Per-interval velocity helper: `src/neurospatial/behavior/_kinematics.py` (new)

Only this PR creates this module. 3d does not need it.

```python
"""Per-interval kinematics that never span an invalid interval."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interval_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Velocity of each interval, NaN where the interval is invalid.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``observed_interval_mask`` output.

    Returns
    -------
    ndarray, shape (n_samples - 1, n_dims)
    """
    dt = np.diff(times)
    velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocity[~interval_mask] = np.nan
    return velocity
```

### 2. `heading_from_velocity(positions, times, ...)` and its shared helper

New signature: `heading_from_velocity(positions, times, *, max_gap=0.5, epochs=None, min_speed=0.0, bandwidth=0.0, allow_all_nan=False)`.

- **Signature.** `dt` (scalar) is replaced by `times`, shape `(n_samples,)`, in the same position. This is not backward compatible (overview decision 1). Validate `times`: 1-D, the same length as `positions`, finite and strictly increasing. This replaces today's `dt <= 0` check.
- **Shared helper.** Phase 2b's `_velocity_heading_and_speed(positions, dt, *, bandwidth)` becomes `_velocity_heading_and_speed(positions, times, *, interval_mask, bandwidth)`:
  - for each run (`run_sample_bounds(interval_mask)`), it computes the velocity with that run's own `np.diff(times[run])`; the last sample of a run repeats the previous interval's velocity, as today (:771);
  - Gaussian smoothing (`bandwidth`, in seconds) is applied per run; the sigma in samples uses that run's median `dt`, which is today's conversion applied per run;
  - samples in no run get NaN heading and NaN speed.

  `heading_from_velocity`, `instantaneous_goal_alignment` and `pre_decision_heading_stats` keep sharing it.
- **Interpolation.** `heading_from_velocity` runs `_interpolate_heading_circular` (Phase 2b Task 3) **per run**, so a low-speed stretch is never filled from the other side of a pause.
- **All-NaN check.** The all-below-`min_speed` check (lines 780–810) runs over the samples that belong to runs.

Update the callers:

- `vte.py:360–365` (`head_sweep_from_positions`) drops its `np.median(np.diff(times))` and passes `times`, `max_gap` and `epochs`.
- `pre_decision_heading_stats` and `instantaneous_goal_alignment` drop their median `dt` and pass `times` and `interval_mask=observed_interval_mask(times, max_gap=max_gap, epochs=epochs)` to `_velocity_heading_and_speed`. A NaN speed (no run) fails the `speed >= min_speed` test, so those samples are excluded exactly like stationary ones.
- `simulation/spikes.py:428–429` passes `times` instead of its median `dt`.
- `simulation/models/head_direction_cells.py:107` (docstring example) passes `times`.

### 3. Inline speed sites and the other kinematic functions

Replace the inline speed computations with `interval_velocity(times, positions, observed_interval_mask(times, max_gap=max_gap, epochs=epochs))` plus nan-aware reductions:

- `pre_decision_speed_stats` (dec:523–526): `np.nanmean`/`np.nanmin` of `‖v‖`, returning NaN when no interval is valid;
- `compute_vte_trial` (vte:598–602);
- `compute_vte_session` (vte:728–732);
- `heading_direction_labels` (nav:1192–1196), where the backward-interval speed and heading are NaN for invalid intervals and so get the `"stationary"` label, the same rule as the first sample. When the caller passes precomputed `speed`/`heading` instead of `positions`/`times`, nothing changes;
- `approach_rate` (nav:1924–1928).

**Composite functions.** `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session` forward `max_gap` and `epochs` to the heading and speed helpers they call. 3d independently forwards the same keywords to `extract_pre_decision_window`. **Whichever of 3d and 3e merges first adds the keyword-only `max_gap: float | None = 0.5, epochs=None`** (after the existing keywords) with its own forwarding. The second rebases onto it, keeps the existing parameters and docstring entries, and adds only its own forwarding. Neither declares them twice. `goal_bias` forwards to `instantaneous_goal_alignment`; `compute_goal_directed_metrics` forwards to the functions it calls.

`head_sweep_from_positions` sums `head_sweep_magnitude(headings[run])` over `observed_runs(times, ...)`. This stops the NaN-dropping diff at vte:311 from bridging a pause. `head_sweep_magnitude` itself is unchanged; it takes no `times`.

`compute_path_efficiency`:

- Compute `mask = observed_interval_mask(times, max_gap=max_gap, epochs=epochs)`.
- If `not mask.all()`, set `traveled_length`, `efficiency` and `angular_efficiency` to NaN. The path through a pause is unknowable, and summing only the observed steps would silently underestimate it.
- `shortest_length` and `time_efficiency` are unchanged.
- Document this in Returns and in `PathEfficiencyResult`'s Attributes.

`compute_trajectory_curvature` (when `times` is not None) computes turn angles and the Gaussian smoothing per run; samples in no run are NaN. With `times=None` there is no gap information, so nothing changes.

`compute_home_range` (when `times` is not None):

- The dwell is `dt` on valid intervals and 0 on invalid ones.
- The last sample of each run gets that run's median `dt`, which is today's single-recording rule applied per run.
- Samples in no run get 0.

### 4. `events.add_positions`

- It gains keyword-only `max_gap=0.5, epochs=None`, after `timestamp_column`.
- Keep `interp1d` for the in-span values. Compute the observed runs on the **sorted** trajectory (`sorted_times`, :196) and set an event's position to NaN unless it lies in a run's closed span `[times[first], times[last]]`:

  ```python
  from neurospatial._intervals import run_time_bounds
  from neurospatial.environment.trajectory import observed_interval_mask

  runs = run_time_bounds(
      sorted_times, observed_interval_mask(sorted_times, max_gap=max_gap, epochs=epochs)
  )
  r = np.searchsorted(runs[:, 0], event_times, side="right") - 1
  observed = (r >= 0) & (event_times <= runs[np.maximum(r, 0), 1])
  interpolated[~observed, :] = np.nan
  ```

- **Boundary behavior, stated in the docstring.** The span is closed at both ends because both end samples were observed: an event exactly on the last sample before a pause (or on `times[-1]`) gets that sample's position, while an event strictly inside the pause interval gets NaN. A NaN event time gives NaN (as today). A sample in no run (both neighbouring intervals invalid) is not interpolated around, so an event exactly on it gets NaN.
- This also replaces extrapolation outside the tracked span. Extrapolated positions are fabricated, so this is a deliberate, documented change; it gets a CHANGELOG bullet.

### 5. Public docstrings for this PR's functions (own task)

Docstrings ship with the change that alters behavior, so each of 3d and 3e documents its own functions; this task covers the 13 rows above and the kinematic forwarding in the three composite functions.

- **Parameters:** `max_gap`/`epochs` entries using 3a's text without `spike_window` (see 3a Task 7, or `Environment.occupancy`'s merged docstring). For `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session`, add them only if 3d has not already (Task 3).
- **Notes:** one sentence: "Each run of samples with gaps no longer than `max_gap` (inside `epochs`) is analyzed as a separate recording; no velocity or heading spans a pause."
- **`heading_from_velocity`.** `Parameters` replaces `dt` with `times`; its Examples use `times`.
- **NaN cases.** `PathEfficiencyResult`, `add_positions`, `approach_rate`, `instantaneous_goal_alignment` and `compute_trajectory_curvature` document when they return NaN.

### 6. Docs that call `heading_from_velocity(positions, dt, ...)`, and the CHANGELOG check (own task)

Update every call to pass `times`. From `git grep -l "heading_from_velocity("` on `da631a47`:

- `CLAUDE.md` (pattern 7);
- `.claude/QUICKSTART.md`;
- `docs/api/index.md`;
- `docs/snippets.yml`;
- `examples/22_spatial_view_cells.{py,ipynb}`, `examples/24_object_vector_cells.{py,ipynb}` and `examples/25_head_direction_tuning.{py,ipynb}`, re-synced as jupytext pairs, together with their `docs/examples/` mirrors.

Leave the historical `CHANGELOG.md` entries and the `docs/plans`/`docs/reviews` files unchanged. Phase 4b executes `CLAUDE.md` and the snippets; check them now with `uv run python scripts/test_doc_snippets.py`.

Each earlier commit added its own CHANGELOG bullet ([executing.md](executing.md)). Check that `[Unreleased]` has a section "Changed — behavior analyses respect recording gaps" (3d adds to the same section; whichever merges first creates it) listing:

- the teleport-heading bug fixed (on the two-epoch geometry, the cross-gap step read as about 1,911 cm/s);
- the new keywords on the 13 functions;
- `heading_from_velocity(positions, times)`;
- the `add_positions` NaN rule, including the end of extrapolation;
- the `compute_path_efficiency` NaN rule.

## Deliberately not in this phase

- **Segment detectors, `extract_pre_decision_window`, `env.bin_sequence`/`transitions`.** These belong to 3d.
- **Argument order.** `(positions, times)` becomes `(times, positions)`, including `heading_from_velocity`'s, in Phase 6b. This PR only adds keywords and swaps `dt` for `times` in place.
- **Unrelated fixes the audit found while reading these files.** Phase 2a Task 7 fixed the `compute_vte_session` window clamp; Phase 2b Tasks 4–5 fixed stationary samples in `instantaneous_goal_alignment`/`goal_bias` and `pre_decision_heading_stats`. Do not re-fix them here.
- **Encoding, decoding, PETH, `_intervals.py`, `interval_valid_mask`, `observed_runs`.** These belong to 3a–3c.
- **Position-only functions without `times`** (`traveled_path_length`, `compute_step_lengths`, `compute_turn_angles`, `trajectory_similarity`, `cost_to_goal`, `head_sweep_magnitude`, …). With no timestamps there is no gap to detect. Callers pass one run at a time.
- **`heading_from_body_orientation`.** It takes no `times` (pose-based heading per sample). Its interpolation of NaN headings can still bridge a pause; callers pass one run at a time.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/ops/test_reference_frames.py::test_heading_from_velocity_ignores_pause` | On `two_epoch_recording`, `heading_from_velocity(positions, times)` at the last pre-pause sample equals the heading of the previous interval. No heading equals the cross-gap direction `atan2(Δy, Δx)` of the jump. **On `main`** (scalar `dt = 0.02` s), the cross-gap step reads as ≈ 1,911 cm/s and gives the teleport heading. |
| `tests/ops/test_reference_frames.py::test_heading_from_velocity_isolated_sample_nan` | A sample with both neighbouring intervals longer than `max_gap` gets a NaN heading. |
| `tests/ops/test_reference_frames.py::test_heading_interpolation_stays_in_run` | Run A ends in a low-speed stretch and run B starts moving north. With `min_speed` above the stretch's speed, the filled headings in run A equal run A's last moving heading; none is pulled toward run B's heading. |
| `tests/ops/test_reference_frames.py::test_heading_from_velocity_rejects_bad_times` | `times` of the wrong length, non-increasing, or containing NaN raises `ValueError` with a `Fix:` line. Replaces the old `dt=0.0`/`dt=-0.1` cases. |
| `tests/behavior/test_kinematics_gaps.py::test_speed_stats_exclude_pause` | `pre_decision_speed_stats` over samples that straddle the pause gives `min_speed` equal to the minimum within-run speed, not ≈ 0. |
| `tests/behavior/test_kinematics_gaps.py::test_head_sweep_sums_runs` | `head_sweep_from_positions` over two runs equals the sum of the two single-run values (`rtol=1e-12`). |
| `tests/behavior/test_kinematics_gaps.py::test_path_efficiency_nan_across_pause` | `compute_path_efficiency` over samples that include the pause gives NaN `traveled_length`, `efficiency` and `angular_efficiency`, and a finite `shortest_length`. A single-run input is unchanged. |
| `tests/behavior/test_kinematics_gaps.py::test_home_range_dwell_excludes_pause` | `compute_home_range(pb, times=t)` on `two_epoch_recording` (with `pb = env.bin_at(positions)`): total dwell = 99.98 + 99.98 + 2 × 0.02 = 200.0 s (± 1e-9). The bin of the last pre-pause sample gets 0.02 s, not about 1000 s. |
| `tests/behavior/test_kinematics_gaps.py::test_heading_direction_labels_post_pause_sample` | With `min_speed=0`, the first post-pause sample is labeled `"stationary"`, not the teleport direction. |
| `tests/behavior/test_kinematics_gaps.py::test_goal_alignment_nan_outside_runs` | `instantaneous_goal_alignment` on `two_epoch_recording` is NaN at an isolated sample and never equals the cosine of the teleport heading at the last pre-pause sample. |
| `tests/behavior/test_kinematics_gaps.py::test_gap_free_outputs_unchanged` | Parametrized over every function in the inventory (and `compute_pre_decision_metrics`, `compute_vte_trial`, `compute_vte_session`), on gap-free input sampled at 50 Hz. The output equals a golden captured on this PR's base commit (see Review; the base calls `heading_from_velocity` with `dt`), with `rtol=1e-12`. |
| `tests/events/test_add_positions_gaps.py::test_event_in_pause_gets_nan` | Fixture `pause_events` (below). Events at `[50., 600., 1150., 5000.]` → positions finite, NaN, finite, NaN. **On `main`**, 600 is interpolated across the pause and 5000 is extrapolated. |
| `tests/events/test_add_positions_gaps.py::test_event_on_run_edges` | Events exactly at the last pre-pause sample (99.98), at `times[0]` and at `times[-1]` get that sample's position exactly. An event at 99.99 (inside the gap interval) gets NaN. |

**Existing tests and examples this phase changes.** Fix each and list it in the PR description:

- **`heading_from_velocity(positions, dt)` calls:** `tests/ops/test_reference_frames.py` (16 calls, including the `dt=0.0`/`dt=-0.1` validation tests at :835–:838 and :867), `tests/encoding/test_compute_egocentric_rate.py:1091` and `:1180`, `tests/encoding/test_compute_view_rate.py:1744`, plus Phase 2b's tests of `_velocity_heading_and_speed`, `instantaneous_goal_alignment` and `pre_decision_heading_stats`, which pass `dt`.
- **Coarsely sampled inputs.** Under the new default `max_gap=0.5`, any existing test that samples at 1 Hz or slower now has a gap at every interval. A probe on `da631a47` (a pytest plugin wrapping each inventory function and recording calls whose `times` has a step longer than 0.5 s; session scratchpad `remed3/gapspy2.py`) found **30 tests** that pass such `times` to this PR's functions: `tests/events/test_detection.py` 20 (almost every `add_positions` test uses `times = np.array([0.0, 1.0, 2.0, ...])`, so every event would get NaN), `tests/behavior/test_goal_directed.py` 4, `tests/behavior/test_binning_dt_correctness.py` 2, and one each in `tests/behavior/test_behavioral.py`, `test_path_efficiency.py`, `test_trajectory_metrics.py` and `tests/events/test_nwb.py` (`test_write_events_from_add_positions`). By function: `add_positions` 21, `approach_rate` 3, `instantaneous_goal_alignment` 2, and one each for `goal_bias`, `compute_goal_directed_metrics`, `compute_path_efficiency`, `compute_home_range` and `compute_trajectory_curvature`. Decide per test whether the coarse sampling is incidental (pass `max_gap=None`, the documented way to treat long intervals as continuous, or densify `times`) or the point of the test (update the expected value and cite the CHANGELOG bullet). Do not change the default to make tests pass. List every changed test and the choice in the PR description. The `add_positions` docstring example must also use sampling at or faster than 2 Hz, or pass `max_gap=None`.

None of these tests exceeds 1 s. No `slow` marks are needed.

## Fixtures

- `two_epoch_recording` from `tests/conftest.py` (3a). This PR uses it for all kinematic tests, so it adds nothing to `tests/behavior/conftest.py` (3d owns `pause_track` and `lap_track` there).
- **`pause_events`**, module-scoped in `tests/events/test_add_positions_gaps.py`: the `times` and `positions` of `two_epoch_recording`, plus `events = pd.DataFrame({"timestamp": [50., 600., 1150., 5000.]})`.
- **Gap-free variants** for the unchanged-output test: `continuous_recording` (3a).

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff; this is the review step of [executing.md → Definition of done](executing.md#definition-of-done). Confirm:

- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also confirm:

- **Goldens come from the base commit, not from `main`.** Before changing code, run `git worktree add ../ns-base $(git merge-base HEAD feat/researcher-first)`. In that worktree, run a capture script that records every inventory function's output on the gap-free input (calling `heading_from_velocity(positions, dt)` there). Commit the goldens as a small `.npz` under `tests/behavior/data/`, then `git worktree remove ../ns-base`. Note the base SHA in the test module docstring.
- No `np.median(np.diff(times))` remains as a velocity `dt` in `src/neurospatial/behavior/`, `src/neurospatial/ops/` or `src/neurospatial/simulation/` (`git grep -n "median(np.diff(times))" src/neurospatial`). Medians used only for smoothing-window widths are allowed.
- `git grep -n "heading_from_velocity(" -- src docs examples CLAUDE.md .claude/QUICKSTART.md` shows no call that passes a scalar `dt`.
- `uv run pytest --doctest-modules src/neurospatial/behavior src/neurospatial/ops src/neurospatial/simulation src/neurospatial/events -n 0` passes with the new `heading_from_velocity` signature.
