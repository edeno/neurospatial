# Phase 3c — Time windows: position-only behavior analyses

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics) · [3a](phase-3a-time-windows-rates.md) · [3b](phase-3b-time-windows-decoding-events.md)

This is the third of three Phase 3 PRs (see the split table in [3a](phase-3a-time-windows-rates.md)). It needs 3a's `neurospatial/_intervals.py` and the extended `interval_valid_mask`. It does not depend on 3b.

**One rule for this whole PR:** each maximal run of valid intervals is analyzed **as if it were a separate recording**.

- **Segment and event detectors** (laps, trials, runs, crossings, velocity epochs, boundary crossings) run on each run separately, and their results are concatenated in time order. A segment in progress when a run ends gets exactly the treatment the detector already gives a segment in progress at the end of a recording. It is never extended across the gap, so it is not counted as complete.
- **Kinematic quantities** (velocity, speed, heading, turn angle, dwell) are never computed across an invalid interval. At run boundaries, samples get the treatment the function already gives the first or last sample of a recording.

For a gap-free input with `epochs=None`, there is one run covering every sample, so outputs are identical to `main`. The gap gate is the existing `max_gap=0.5` (added where absent) and `epochs=None`, which together form the contract's position-only row. Speed and bounds keep each function's existing `min_speed` and `-1`-bin handling.

**Inputs to read first:**

- [phase-3a-time-windows-rates.md](phase-3a-time-windows-rates.md), Tasks 1–2. These give `as_intervals`, `run_sample_bounds`, and `interval_valid_mask(times, *, max_gap, epochs)` with no env.
- [src/neurospatial/behavior/segmentation.py](../../../../src/neurospatial/behavior/segmentation.py). It defines:
  - `_positive_dt` (:133);
  - `detect_region_crossings` (:322), which takes `np.diff(in_region)` across all samples at :488;
  - `detect_runs_between_regions` (:518), with an exit loop at 680–703;
  - `segment_by_velocity` (:772), which uses a median-dt smoothing window at 904–905, a moving average at 914–919, and `_emit` at 945–966;
  - `detect_laps` (:1049), whose region method pairs consecutive entries at 1262–1290 and whose auto method takes its template from the first 10% of samples at 1293–1296;
  - `running_direction_labels` (:1523);
  - `segment_trials` (:1680), whose state machine runs at 1909–1976 and whose end-of-recording rule emits an in-progress trial as `success=False`;
  - `detect_goal_directed_runs` (:2230), which treats the whole input as one candidate at 2421–2424.
- [src/neurospatial/behavior/decisions.py](../../../../src/neurospatial/behavior/decisions.py). It defines `extract_pre_decision_window` (:352), `pre_decision_heading_stats` (:402, which takes a median dt at :460), `pre_decision_speed_stats` (:488, inline speed at 523–526), `compute_pre_decision_metrics` (:531), `detect_boundary_crossings` (:764, which uses the midpoint time at :812) and `compute_decision_analysis` (:828).
- [src/neurospatial/behavior/vte.py](../../../../src/neurospatial/behavior/vte.py). It defines:
  - `head_sweep_magnitude` (:271), which drops NaN headings and then diffs across the hole at :311;
  - `head_sweep_from_positions` (:319, median dt at :360);
  - `compute_vte_trial` (:534, inline speed at 598–602);
  - `compute_vte_session` (:617, inline speed at 728–732).
- [src/neurospatial/behavior/navigation.py](../../../../src/neurospatial/behavior/navigation.py). It defines:
  - `heading_direction_labels` (:1107, inline speed at 1192–1196);
  - `time_efficiency` (:1358);
  - `compute_path_efficiency` (:1570, path length at :1619);
  - `instantaneous_goal_alignment` (:1732, median dt at :1776);
  - `goal_bias` (:1791);
  - `approach_rate` (:1844, `_positive_dt` at :1924);
  - `compute_goal_directed_metrics` (:1933).
- [src/neurospatial/behavior/trajectory.py](../../../../src/neurospatial/behavior/trajectory.py). It defines `compute_home_range` (:360, dwell from `np.diff(times)` at :454 and a last-sample median at :457), `mean_square_displacement` (:485) and `compute_trajectory_curvature` (:754, Gaussian sigma from the median dt at :865).
- [src/neurospatial/ops/egocentric.py:655](../../../../src/neurospatial/ops/egocentric.py#L655). This is `heading_from_velocity(positions, dt, ...)`. It takes a **scalar** `dt` (velocity at :768), so the step across a pause reads as a very fast movement and produces a fake heading. Its callers are `decisions.py:464`, `vte.py:365`, `navigation.py:1780` and `simulation/spikes.py:429`.
- [src/neurospatial/environment/trajectory.py:494](../../../../src/neurospatial/environment/trajectory.py#L494). This holds `bin_sequence` (:494), `bin_sequence_with_runs` (:567), `_bin_sequence` (:622), `transitions` (:787) and `_empirical_transitions` (:965). In `_empirical_transitions`, the pairs `bins[:-lag] → bins[lag:]` are taken around :1115 and cross pauses.
- [src/neurospatial/events/detection.py:20](../../../../src/neurospatial/events/detection.py#L20). This is `add_positions`. Its `interp1d(..., fill_value="extrapolate")` (205–211) **invents** positions for events inside a pause or outside the tracked span.
- Evidence (session scratchpad): `branch-triage.md` §A2 and `catA/gap3.py`. The setup is 10 Hz tracking. Epoch A covers `[0, 100)` s: the animal sits in `source` (x < 10) and leaves at t = 98. Epoch B covers `[1100, 1110)` s with the animal in `target` (x > 90). On `main`:
  - a target entry is reported at t = 1100;
  - a run spanning 98 → 1100 s is marked a success;
  - with `max_duration=2000`, a trial spanning 0 → 1100 s is marked a success;
  - a movement epoch spans 98.1 → 1100 s.

  Separately, `catA/gap4.py` (the 50 Hz two-epoch geometry of 3a's fixture) shows that `heading_from_velocity` with a scalar `dt = 0.02` s reads the cross-gap step as about 1,911 cm/s.
- Archive reference only: `7b862673` (crossings and laps), `f038a558` (trials), `19eaa2b3` (runs and velocity) and `e9eee2c5` (trajectory and heading).
- **Files earlier phases already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `ops/egocentric.py`: Phase 2 Task 6 rewrote `_interpolate_heading_circular` (shorter-arc, uniform in angle). Phase 2 Task 10 added `_velocity_heading_and_speed(positions, dt, *, bandwidth)`, which `heading_from_velocity`, `instantaneous_goal_alignment` and `pre_decision_heading_stats` share.
  - `behavior/vte.py`: Phase 2 Task 9 made `compute_vte_session` cut each window from the trial's own samples and clamp `window_start` to `trial.start_time`. Keep that clamp.
  - `behavior/navigation.py` and `behavior/decisions.py`: Phase 2 Tasks 10–11 made `instantaneous_goal_alignment` NaN, and `pre_decision_heading_stats` exclude, samples below `min_speed`.
  - `events/alignment.py` is not touched here. `CHANGELOG.md`: append after the Phase 1, 2 and 3a sections.

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics). This PR implements the position-only row and the Segmentation paragraph. "Not counted as complete" is realized as follows:
  - the detector's own end-of-recording treatment applies at the gap: trials and runs get `success=False`, and velocity epochs are truncated;
  - a segment is never extended across the gap.
- [Error-message contract](shared-contracts.md#error-message-contract). `epochs` errors come from `as_intervals`.
- [Input conventions](shared-contracts.md#input-conventions). New arguments are keyword-only. **Argument order is not changed here**: the behavior `(times, positions)` reordering belongs to Phase 6.

**Designs referenced:** none.

## Inventory (this PR's slice)

All functions below consume `times` with `positions` or `position_bins`. None has `max_gap` or `epochs` on `main`. Every "Change" row adds keyword-only `max_gap: float | None = 0.5, epochs=None` (after the existing keywords) and applies the rule above.

| Function (file:line) | Current gap behavior on `main` | Change in 3c |
| --- | --- | --- |
| `Environment.bin_sequence` trajectory.py:494 | dedup merges a same-bin pair across a pause; a different-bin pair reads as a step | samples outside runs dropped; no dedup or step across an invalid interval |
| `Environment.bin_sequence_with_runs` :567 | a same-bin run can span the pause (its duration then includes it) | runs also split at invalid intervals |
| `Environment.transitions` (empirical) :787 | counts the last-pre-pause → first-post-pause pair | a pair `(k, k+lag)` counts only if intervals `k … k+lag-1` are all valid |
| `detect_region_crossings` seg:322 | stamps an unobserved entry/exit at the first post-pause sample | per run |
| `detect_runs_between_regions` seg:518 | a run spans the pause, or "times out" at the first post-pause sample | per run; a run in progress at the run end ends there with `success=False` |
| `segment_by_velocity` seg:772 | the moving average bridges the pause; epochs end at `times[i+1]` after the pause | per run (smoothing, median dt and hysteresis are each per run) |
| `detect_laps` seg:1049 | a region lap pairs entries across the pause; auto windows straddle it | per run; the auto template is still taken from the first 10% of the whole input |
| `segment_trials` seg:1680 | a trial spans 0 → 1100 s (success when `max_duration` is large) | per run; a trial in progress at the run end is emitted as `success=False`, as at the end of a recording |
| `detect_goal_directed_runs` seg:2230 | the whole input is one run, and the jump adds a geodesic shortcut | one candidate per run |
| `running_direction_labels` seg:1523 | inherits from runs | forwards `max_gap`, `epochs` |
| `detect_boundary_crossings` dec:764 | a label change across the pause is stamped at its midpoint (about 600 s) | per run |
| `extract_pre_decision_window` dec:352 | the window can straddle the pause | restricted to the run that contains `entry_time` |
| `pre_decision_heading_stats` dec:402 | median-dt heading; the teleport heading counts | `_velocity_heading_and_speed` per run (see Task 3); invalid intervals give NaN, so the teleport heading is excluded like a stationary sample |
| `pre_decision_speed_stats` dec:488 | speed across the pause ≈ 0 drags the minimum and mean | per-interval speed with NaN on invalid intervals; nan-aware reductions |
| `compute_pre_decision_metrics` dec:531 | inherits; `actual_duration` includes the pause | forwards; duration measured within the window's run |
| `compute_decision_analysis` dec:828 | inherits | forwards |
| `head_sweep_from_positions` vte:319 | teleport heading plus the hole-bridging diff inflate IdPhi | heading per run; IdPhi = sum over runs of `head_sweep_magnitude` |
| `compute_vte_trial` vte:534 | inline speed across the pause | shared per-interval speed helper |
| `compute_vte_session` vte:617 | same | same; windows via the gap-aware `extract_pre_decision_window`, still clamped to the trial start (Phase 2) |
| `heading_direction_labels` nav:1107 | a post-pause sample gets speed ≈ 0 by accident, or the teleport direction when `min_speed=0` | a sample whose backward interval is invalid is labeled like the first sample (`"stationary"`) |
| `compute_path_efficiency` nav:1570 | the jump adds to path length; two fake turn angles | `traveled_length`, `efficiency` and `angular_efficiency` are NaN when the samples include an invalid interval (the path is unknowable); `time_efficiency` stays wall-clock |
| `instantaneous_goal_alignment` nav:1732 | teleport heading | gap-aware heading via `_velocity_heading_and_speed` per run; samples in no run are NaN, as are samples below `min_speed` (Phase 2) |
| `goal_bias` nav:1791 | inherits | inherits (nan-aware mean already) |
| `approach_rate` nav:1844 | a made-up rate at the first post-pause sample | NaN on invalid intervals |
| `compute_goal_directed_metrics` nav:1933 | inherits; `time_to_goal` is wall-clock | forwards; `time_to_goal` unchanged (wall-clock time is known across a gap) |
| `compute_trajectory_curvature` traj:754 | fake turn angles at the jump, then smoothed across it | when `times` is given: per run; samples in no run are NaN |
| `compute_home_range` traj:360 | the last pre-pause sample gets ≈1000 s of dwell | dwell from valid intervals only; the last sample of each run gets that run's median `dt` (today's last-sample rule) |
| `ops.egocentric.heading_from_velocity` ops/egocentric.py:655 | scalar `dt`; teleport heading; low-speed interpolation bridges the pause | **`dt` replaced by `times`**; per-run velocity, smoothing and interpolation |
| `events.add_positions` detection.py:20 | linear interpolation **across the pause**, and extrapolation outside the span | NaN position for an event outside every valid interval |

These consume `times` but need **no change**:

| Function | Why |
| --- | --- |
| `mean_square_displacement` traj:485 | Pairs form only at real time lags between two observed samples, so a pair that straddles a pause is a genuine displacement. No quantity is interpolated across the pause. |
| `time_efficiency` nav:1358 | It is a wall-clock duration over the input, which is known across a gap. |
| `decision_region_entry_time` dec:289 | It returns the first in-region sample. Under the per-run rule, that is the same sample as on `main`. |
| `distance_to_reward` events/regressors.py:508 | "Last/next reward" is well defined across a gap, and the distance uses the current observed position. |
| `laps_to_direction_labels`, `runs_to_direction_labels`, `goal_pair_direction_labels`, `trials_to_region_arrays`, `time_to_goal` | They only map existing segments onto samples. Once the detectors stop producing spanning segments, they are correct. |

## Tasks

### 1. Shared private helpers: `src/neurospatial/behavior/_runs.py` (new)

```python
"""Split position-only data into observed runs and compute per-interval kinematics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import as_intervals, run_sample_bounds


def observed_interval_mask(
    times: NDArray[np.float64], *, max_gap: float | None, epochs: Any
) -> NDArray[np.bool_]:
    """Per-interval validity for position-only analyses (gap and epochs gates)."""
    from neurospatial.environment.trajectory import interval_valid_mask

    return interval_valid_mask(
        np.asarray(times, dtype=np.float64),
        max_gap=max_gap,
        epochs=as_intervals(epochs, name="epochs"),
    )


def observed_runs(
    times: NDArray[np.float64], *, max_gap: float | None, epochs: Any
) -> Iterator[slice]:
    """Yield one sample slice per maximal run of valid intervals, in time order."""
    mask = observed_interval_mask(times, max_gap=max_gap, epochs=epochs)
    for first, last in run_sample_bounds(mask):
        yield slice(int(first), int(last) + 1)


def interval_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Velocity of each interval, NaN where the interval is invalid.

    Returns
    -------
    ndarray, shape (n_samples - 1, n_dims)
    """
    dt = np.diff(times)
    velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocity[~interval_mask] = np.nan
    return velocity
```

Every segmentation detector is refactored the same way:

1. Rename the current body to a private `_<name>_contiguous(...)`, keeping its arguments and logic as they are.
2. The public function validates once, then concatenates the per-run results:

   ```python
   results = []
   for run in observed_runs(times, max_gap=max_gap, epochs=epochs):
       results.extend(_<name>_contiguous(position_bins[run], times[run], env, ...))
   return results
   ```

**No detector re-implements gap logic.**

### 2. Segmentation detectors (segmentation.py, decisions.py)

Apply the Task 1 refactor to:

- `detect_region_crossings`;
- `detect_runs_between_regions`;
- `segment_by_velocity` (its `positions`, `times` and smoothing all per run);
- `segment_trials`;
- `detect_goal_directed_runs`;
- `detect_boundary_crossings`, where the loop runs over `position_bins[run]`, `voronoi_labels[run]` and `times[run]`.

`running_direction_labels` forwards `max_gap` and `epochs` to both of its `detect_runs_between_regions` calls (1643–1662).

`detect_laps` needs two adjustments:

- **`method="region"`.** Call `_detect_region_crossings_contiguous` per run and pair only the entries *within* that run.
- **`method="auto"`.** Compute the template from the first 10% of the **whole** input, as today (1293–1296), so that template selection is unchanged. Then search each run. In the run that contains sample 0, the search starts at `template_size - run.start` (as today); every other run starts at 0.

`_positive_dt` (seg:133) stays as it is: it validates monotonicity and does no gap logic.

`extract_pre_decision_window`:

- Find the run `r` with `times[first_r] <= entry_time <= times[last_r]`.
- Return the samples in `[max(entry_time - window_duration, times[first_r]), entry_time)` from that run only.
- Return empty arrays when `entry_time` is in no run. That is today's empty-window behavior.

`compute_pre_decision_metrics`, `compute_decision_analysis`, `compute_vte_trial` and `compute_vte_session` forward `max_gap` and `epochs` to it.

### 3. Kinematics: `heading_from_velocity` and the inline speed sites

`heading_from_velocity(positions, times, *, max_gap=0.5, epochs=None, min_speed=0.0, bandwidth=0.0, allow_all_nan=False)`:

- **Signature.** `dt` (scalar) is replaced by `times` of shape `(n_samples,)`. This is not backward compatible (overview decision 1).
- **Per-run computation.** Phase 2's shared `_velocity_heading_and_speed` changes from `(positions, dt, *, bandwidth)` to `(positions, times, *, interval_mask, bandwidth)`: for each run it computes the velocity with the per-run `np.diff(times[run])`, and the last sample of a run repeats the previous interval's velocity, as today (:771). Samples in no run get NaN heading and NaN speed. `heading_from_velocity`, `instantaneous_goal_alignment` and `pre_decision_heading_stats` keep sharing it.
- **Smoothing.** Apply the optional Gaussian smoothing per run.
- **Interpolation.** Run `_interpolate_heading_circular` per run.
- **Samples in no run** stay NaN.
- **All-NaN check.** The all-below-`min_speed` check (lines 780–810) runs over the samples that belong to runs.

Update the four callers:

- `vte.py:360–365` drops its `np.median(np.diff(times))` and passes `times`, `max_gap` and `epochs`. `decisions.py` (`pre_decision_heading_stats`) and `navigation.py` (`instantaneous_goal_alignment`) call `_velocity_heading_and_speed` since Phase 2; they drop their median `dt` and pass `times` and the observed-interval mask instead. A NaN speed (no run) fails the `speed >= min_speed` test, so those samples are excluded exactly like stationary ones.
- `simulation/spikes.py:428–429` passes `times`.

Also update `simulation/models/head_direction_cells.py:107` (docstring example).

Replace the inline speed computations with `interval_velocity` plus nan-aware reductions:

- `pre_decision_speed_stats` (dec:523–526): `np.nanmean`/`np.nanmin` of `‖v‖`, returning NaN when no interval is valid;
- `compute_vte_trial` (vte:598–602);
- `compute_vte_session` (vte:728–732);
- `heading_direction_labels` (nav:1192–1196), where the backward-interval speed and heading are NaN for invalid intervals and so get the `"stationary"` label, the same rule as the first sample;
- `approach_rate` (nav:1924–1928).

`head_sweep_from_positions` sums `head_sweep_magnitude(headings[run])` over the runs. This stops the NaN-dropping diff at vte:311 from bridging a pause. `head_sweep_magnitude` itself is unchanged; it takes no `times`.

`goal_bias` inherits through `instantaneous_goal_alignment`.

`compute_path_efficiency`:

- Compute `mask = observed_interval_mask(times, ...)`.
- If `not mask.all()`, set `traveled_length`, `efficiency` and `angular_efficiency` to NaN. The path through a pause is unknowable, and summing only the observed steps would silently underestimate it.
- `shortest_length` and `time_efficiency` are unchanged.
- Document this in Returns.

`compute_trajectory_curvature` (when `times` is not None) computes per run, as described above. `compute_home_range` (when `times` is not None) changes as follows:

- The dwell is `dt` on valid intervals and 0 on invalid ones.
- The last sample of each run gets that run's median `dt`, which is today's single-recording rule applied per run.
- Samples in no run get 0.

### 4. `Environment` sequence methods (environment/trajectory.py)

- **`bin_sequence` and `bin_sequence_with_runs`.** Both gain `max_gap=0.5, epochs=None`.
  - In `_bin_sequence` (:622), compute `mask = interval_valid_mask(times, max_gap=..., epochs=...)`.
  - Drop samples that belong to no run.
  - Treat an invalid interval exactly like the existing outside-sample split: it starts a new run, and dedup never merges across it. That is the existing `gap_splits_runs` index-gap logic (733–736), extended to time gaps.
  - The `BinSequenceWithRuns` duration formula then never includes a pause.
- **`transitions`.** It gains `max_gap=0.5, epochs=None`, which are used only when `times` is given. In `_empirical_transitions`, after building `bins` (dedup off, aligned to samples), keep only pairs whose intervals are all valid:

  ```python
  invalid_before = np.concatenate([[0], np.cumsum(~mask)])
  pair_ok = invalid_before[lag:] == invalid_before[:-lag]  # no invalid interval in k .. k+lag-1
  source_bins, target_bins = bins[:-lag][pair_ok], bins[lag:][pair_ok]
  ```

  When `bins` is given directly with no `times`, nothing changes: no time information means no gap gate.

### 5. `events.add_positions`

- It gains `max_gap=0.5, epochs=None`.
- Keep `interp1d` for the in-span values, but set an event's position to NaN unless it lies inside a valid interval:

  ```python
  k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
  ok = (t >= times[0]) & (t <= times[-1]) & mask[k]
  ```

- This also replaces extrapolation outside the tracked span. Extrapolated positions are fabricated, so this is a deliberate, documented change; put it in the CHANGELOG.

### 6. Public docstrings for every touched public function (own task)

- **Coverage.** All 29 "Change" rows get `max_gap`/`epochs` Parameters entries, using 3a's text without `spike_window`, plus one Notes sentence: "Each run of samples with gaps no longer than `max_gap` (inside `epochs`) is analyzed as a separate recording; no segment, velocity or heading spans a pause."
- **`heading_from_velocity`.** Its `Parameters` replaces `dt` with `times`. Its Examples use `times`.
- **Result Notes.** `PathEfficiencyResult` and `add_positions` document their NaN cases.

### 7. Docs that call `heading_from_velocity(positions, dt, ...)` and the CHANGELOG (own task)

Update every call to pass `times`. From `git grep -l "heading_from_velocity("` on this branch:

- `CLAUDE.md` (pattern 7);
- `.claude/QUICKSTART.md`;
- `docs/api/index.md`;
- `docs/snippets.yml`;
- `examples/22_spatial_view_cells.{py,ipynb}`, `examples/24_object_vector_cells.{py,ipynb}` and `examples/25_head_direction_tuning.{py,ipynb}`, re-synced as jupytext pairs, together with their `docs/examples/` mirrors.

Leave the historical `CHANGELOG.md` entries and the `docs/plans`/`docs/reviews` files unchanged.

Add a `CHANGELOG.md` `[Unreleased]` section, "Changed — behavior analyses respect recording gaps", that lists:

- the spanning-segment and teleport-heading bugs fixed (with the audit numbers above);
- the new keywords;
- `heading_from_velocity(positions, times)`;
- the `add_positions` NaN rule;
- the `compute_path_efficiency` NaN rule.

## Deliberately not in this phase

- **Argument order.** `(positions, times)` becomes `(times, positions)`, and `heading_from_velocity`'s position, in Phase 6. This PR only adds keywords and swaps `dt` for `times` in place.
- **Unrelated fixes the audit found while reading these files.** Phase 2 fixed them (Tasks 9–11): the `compute_vte_session` window clamp, stationary samples in `instantaneous_goal_alignment`/`goal_bias`, and the `pre_decision_heading_stats` stationary filter. Do not re-fix them here.
- **Encoding, decoding, PETH, `_intervals.py`.** These belong to 3a and 3b.
- **Position-only functions without `times`** (`traveled_path_length`, `compute_step_lengths`, `compute_turn_angles`, `trajectory_similarity`, `cost_to_goal`, …). With no timestamps there is no gap to detect. Callers pass one run at a time.
- **The per-sample GLM regressors and `align_spikes_to_events`.** Deferred; see 3b and the overview's Open Questions.

## Validation slice

The fixtures are described under Fixtures. Each "fails on `main`" row is a regression test for the A2 repro.

| Test | Asserts |
| --- | --- |
| `tests/behavior/test_segmentation_gaps.py::test_no_crossing_reported_across_pause` | On `pause_track`, `detect_region_crossings(pb, t, env, region_name="target")` returns `[]`. **Fails on `main`**, which reports an entry at t = 1100. |
| `tests/behavior/test_segmentation_gaps.py::test_runs_do_not_span_pause` | `detect_runs_between_regions(..., source="source", target="target", max_duration=2000)`: no run has `start_time < 100 <= 1100 <= end_time`. The run that starts near 98 s ends at or before 99.9 with `success=False`. **Fails on `main`**, where 98 → 1100 is a success. |
| `tests/behavior/test_segmentation_gaps.py::test_trials_do_not_span_pause` | `segment_trials(..., start_region="source", end_regions=["target"], max_duration=2000)`: every trial lies within one run, and the epoch-A trial has `end_time <= 99.9` and `success=False`. **Fails on `main`**, where 0 → 1100 is a success. |
| `tests/behavior/test_segmentation_gaps.py::test_velocity_epochs_do_not_span_pause` | `segment_by_velocity(pos, t, min_speed=5.0, min_duration=0.1)`: every epoch has `end_time <= 99.9` or `start_time >= 1100`. **Fails on `main`** (98.1 → 1100). |
| `tests/behavior/test_segmentation_gaps.py::test_region_laps_do_not_pair_entries_across_pause` | `lap_track`, which has start-region entries at t ≈ 50 (run A) and t ≈ 1150 (run B), gives `detect_laps(method="region")` → no lap spanning `[100, 1100]`. **Fails on `main`**, which reports one lap from 50 to 1150. |
| `tests/behavior/test_segmentation_gaps.py::test_auto_laps_template_unchanged` | On a gap-free circular track, `detect_laps(method="auto")` is identical to `main` (same lap list), so the per-run search does not change template selection. |
| `tests/behavior/test_segmentation_gaps.py::test_boundary_crossing_not_in_pause` | `detect_boundary_crossings` on a label change across the pause → no crossing time in `(100, 1100)`. **Fails on `main`** (≈ 600 s). |
| `tests/behavior/test_segmentation_gaps.py::test_epochs_equal_slicing` | For `segment_trials`, `detect_region_crossings` and `segment_by_velocity` on a gap-free 200 s track, `epochs=[(0., 100.)]` gives exactly the same result as slicing the arrays to `times <= 100`. |
| `tests/behavior/test_segmentation_gaps.py::test_gap_free_outputs_unchanged` | Parametrized over every function in the "Change" table that has an existing test fixture: on gap-free input the output equals the pre-change output (recorded as a golden value in the test from `main`, `rtol=1e-12`). This catches accidental behavior change in the `_contiguous` refactor. |
| `tests/ops/test_reference_frames.py::test_heading_from_velocity_ignores_pause` | On `two_epoch_recording`, `heading_from_velocity(positions, times)` at the last pre-pause sample equals the heading of the previous interval. No heading equals the cross-gap direction `atan2(Δy, Δx)` of the jump. **On `main`** (scalar `dt = 0.02` s), the cross-gap step reads as ≈ 1,911 cm/s and gives the teleport heading. |
| `tests/ops/test_reference_frames.py::test_heading_from_velocity_isolated_sample_nan` | A sample with both neighbouring intervals longer than `max_gap` gets a NaN heading. |
| `tests/behavior/test_kinematics_gaps.py::test_speed_stats_exclude_pause` | `pre_decision_speed_stats` over samples that straddle the pause gives `min_speed` equal to the minimum within-run speed, not ≈ 0. |
| `tests/behavior/test_kinematics_gaps.py::test_head_sweep_sums_runs` | `head_sweep_from_positions` over two runs equals the sum of the two single-run values (`rtol=1e-12`). |
| `tests/behavior/test_kinematics_gaps.py::test_path_efficiency_nan_across_pause` | `compute_path_efficiency` over samples that include the pause gives NaN `traveled_length`, `efficiency` and `angular_efficiency`, and a finite `shortest_length`. A single-run input is unchanged. |
| `tests/behavior/test_kinematics_gaps.py::test_home_range_dwell_excludes_pause` | `compute_home_range(pb, times=t)` total dwell = 99.98 + 99.98 + 2 × 0.02 = 200.0 s (± 1e-9). The bin of the last pre-pause sample gets 0.02 s, not about 1000 s. |
| `tests/behavior/test_kinematics_gaps.py::test_heading_direction_labels_post_pause_sample` | With `min_speed=0`, the first post-pause sample is labeled `"stationary"`, not the teleport direction. |
| `tests/environment/test_bin_sequence.py::test_transitions_skip_pause_pair` | `env.transitions(times=t, positions=p, allow_teleports=True, normalize=False)` on `two_epoch_recording` does not include the last-pre → first-post pair; its total count `.sum()` is `n_samples - 2 = 9998` (one interval dropped). **Fails on `main`** (count 9999). |
| `tests/environment/test_bin_sequence.py::test_runs_split_at_pause` | `bin_sequence_with_runs`: when the last pre-pause and first post-pause samples share a bin, they end up in two runs, and no run duration exceeds 100 s. |
| `tests/events/test_add_positions_gaps.py::test_event_in_pause_gets_nan` | Events at `[50., 600., 1150., 5000.]` → positions are finite, NaN, finite, NaN. **On `main`**, 600 is interpolated across the pause and 5000 is extrapolated. |

None of these tests exceeds 1 s. No `slow` marks are needed.

## Fixtures

Session-scoped fixtures go in `tests/behavior/conftest.py`. `two_epoch_recording` comes from 3a.

- **`pause_track`.** This is `catA/gap3.py` rebuilt:
  - Sampled at 10 Hz from `np.arange(n) / 10` (epoch A, `[0, 100)`) and `1100 + np.arange(100) / 10` (epoch B, `[1100, 1110)`).
  - The animal sits at x = 5 until t = 98, then moves right at 12.5 cm/s. In epoch B it sits at x = 95; y = 5 throughout.
  - `env` is `Environment.from_samples` on a 100 × 10 cm grid of points with `bin_size=2.0`. Regions are `source = box(0, 0, 10, 10)` and `target = box(90, 0, 100, 10)`. The fixture also provides `pb = env.bin_at(pos)`.
- **`lap_track`.** A 1-D circular sequence of bins (a `from_graph` ring or a small grid ring) that passes through a `start` region at t ≈ 10 and 50 in run A, and at t ≈ 1150 and 1190 in run B. Each run holds two complete laps.
- **Gap-free variants.** Each fixture is regenerated over a single continuous span, for the equivalence and unchanged-output tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

Also confirm:

- No `np.median(np.diff(times))` remains as a velocity `dt` in `src/neurospatial/behavior/` (`git grep -n "median(np.diff(times))" src/neurospatial/behavior`). Medians used only for smoothing window widths are allowed.
- Every detector's public function contains no state machine; that logic lives only in `_<name>_contiguous`.
- `uv run pytest --doctest-modules src/neurospatial/behavior src/neurospatial/ops` passes with the new `heading_from_velocity` signature.
