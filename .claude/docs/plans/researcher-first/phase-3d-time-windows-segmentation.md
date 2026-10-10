# Phase 3d — Time windows: segmentation and environment sequence methods

**Requires:** Phase 3a.

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics) · [3a](phase-3a-time-windows-core.md)

This is the fourth Phase 3 PR (the split is tabled in [3a](phase-3a-time-windows-core.md)). It makes every position-only **segment detector** and the `Environment` sequence methods respect recording gaps. It is independent of 3b, 3c and 3e. Its sibling [3e](phase-3e-time-windows-kinematics.md) covers the kinematic quantities (velocity, speed, heading) under the same rule.

Follow [executing.md](executing.md) for branching, commits, CHANGELOG bullets, the definition of done and the PR.

**One rule for this PR (shared with 3e):** each maximal run of valid intervals is analyzed **as if it were a separate recording**.

- **Segment and event detectors** (laps, trials, runs, crossings, velocity epochs, boundary crossings) run on each run separately, and their results are concatenated in time order. A segment in progress when a run ends gets exactly the treatment the detector already gives a segment in progress at the end of a recording. It is never extended across the gap, so it is not counted as complete.
- For a gap-free input with `epochs=None`, there is one run covering every sample, so outputs are identical to the base commit. The gap gate is the existing `max_gap=0.5` (added where absent) and `epochs=None`, which together form the contract's position-only row. Speed and bounds keep each function's existing `min_speed` and `-1`-bin handling.

**Inputs to read first:**

- **3a's merged code is the source of truth**, not the 3a plan text:
  - [src/neurospatial/_intervals.py](../../../../src/neurospatial/_intervals.py): `as_intervals`, `run_sample_bounds`.
  - [src/neurospatial/environment/trajectory.py](../../../../src/neurospatial/environment/trajectory.py): `interval_valid_mask(times, *, max_gap, epochs)` (no env needed), and the position-only helpers `observed_interval_mask(times, *, max_gap, epochs)` and `observed_runs(times, *, max_gap, epochs) -> list[slice]`, which normalize a raw `epochs` argument themselves.
- [src/neurospatial/behavior/segmentation.py](../../../../src/neurospatial/behavior/segmentation.py). It defines:
  - `_positive_dt` (:133);
  - `detect_region_crossings` (:322), which takes `np.diff(in_region)` across all samples at :488. It still accepts the compatibility argument order `(..., region_name, env)` through `arg3`/`arg4`;
  - `detect_runs_between_regions` (:518), with an exit loop at 680–703;
  - `segment_by_velocity` (:772), which uses a median-dt smoothing window at 904–905, a moving average at 914–919, and `_emit` at 945–966;
  - `detect_laps` (:1049). Its region method pairs consecutive entries at 1262–1290; the auto method takes its template from the first 10% of samples and starts searching after it (1293–1296); the reference method searches from sample 0 (:1299–1301);
  - `running_direction_labels` (:1523), which calls `detect_runs_between_regions` twice per end region (1643–1662);
  - `segment_trials` (:1680), whose state machine runs at 1909–1976 and whose end-of-recording rule emits an in-progress trial as `success=False`;
  - `detect_goal_directed_runs` (:2230), which treats the whole input as one candidate at 2421–2424.
- [src/neurospatial/behavior/decisions.py](../../../../src/neurospatial/behavior/decisions.py):
  - `extract_pre_decision_window` (:352);
  - `compute_pre_decision_metrics` (:531), which calls it at :568;
  - `detect_boundary_crossings` (:764). Its `voronoi_labels` has shape `(n_bins,)` (one label per **bin**, from `geodesic_voronoi_labels`), not one per sample, and it returns a tuple of two lists `(crossing_times, crossing_directions)`. It stamps a crossing at the midpoint time (:812);
  - `compute_decision_analysis` (:828), which calls `compute_pre_decision_metrics` (:917) and `detect_boundary_crossings` (:938).
- [src/neurospatial/behavior/vte.py](../../../../src/neurospatial/behavior/vte.py): `compute_vte_trial` (:534) and `compute_vte_session` (:617) call `extract_pre_decision_window` (:581 and :714).
- [src/neurospatial/environment/trajectory.py:494](../../../../src/neurospatial/environment/trajectory.py#L494). This holds `bin_sequence` (:494), `bin_sequence_with_runs` (:567), `_bin_sequence` (:622, with the `gap_splits_runs` index-gap logic at 733–736), `transitions` (:787) and `_empirical_transitions` (:965). `_empirical_transitions` builds per-sample bins with `self.bin_sequence(times, positions, dedup=False, outside_value=-1)` at :1080 and pairs `bins[:-lag] → bins[lag:]` at :1115, which crosses pauses.
- Evidence (session scratchpad): `branch-triage.md` §A2 and `catA/gap3.py`. The setup is 10 Hz tracking. Epoch A covers `[0, 100)` s: the animal sits in `source` (x < 10) and leaves at t = 98. Epoch B covers `[1100, 1110)` s with the animal in `target` (x > 90). On `main`:
  - a target entry is reported at t = 1100;
  - a run spanning 98 → 1100 s is marked a success;
  - with `max_duration=2000`, a trial spanning 0 → 1100 s is marked a success;
  - a movement epoch spans 98.1 → 1100 s.
- Archive reference only: `7b862673` (crossings and laps), `f038a558` (trials) and `19eaa2b3` (runs and velocity).
- **Files earlier phases already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `behavior/vte.py`: Phase 2a Task 7 made `compute_vte_session` cut each window from the trial's own samples and clamp `window_start` to `trial.start_time`. Keep that clamp. Phase 2a precedes this PR in the PLAN order.
  - `behavior/decisions.py`, `behavior/vte.py`: if 3e has merged, it added `max_gap`/`epochs` to `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session` (Task 3 says how to share them).

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics). This PR implements the position-only row and the Segmentation paragraph. "Not counted as complete" is realized as follows:
  - the detector's own end-of-recording treatment applies at the gap: trials and runs get `success=False`, and velocity epochs are truncated;
  - a segment is never extended across the gap.
- [Error-message contract](shared-contracts.md#error-message-contract). `epochs` errors come from `as_intervals`.
- [Input conventions](shared-contracts.md#input-conventions). New arguments are keyword-only. **Argument order is not changed here**: the behavior `(times, positions)` reordering belongs to Phase 6b.

**Designs referenced:** none.

## Inventory (this PR's slice)

All functions below consume `times` with `positions` or `position_bins`. None has `max_gap` or `epochs` on `main`. Every row adds keyword-only `max_gap: float | None = 0.5, epochs=None` after the existing keywords and applies the rule above.

| Function (file:line) | Current gap behavior on `main` | Change in 3d |
| --- | --- | --- |
| `Environment.bin_sequence` trajectory.py:494 | dedup merges a same-bin pair across a pause; a different-bin pair reads as a step | samples outside runs dropped; no dedup or step across an invalid interval |
| `Environment.bin_sequence_with_runs` :567 | a same-bin run can span the pause (its duration then includes it) | runs also split at invalid intervals |
| `Environment.transitions` (empirical) :787 | counts the last-pre-pause → first-post-pause pair | a pair `(k, k+lag)` counts only if intervals `k … k+lag-1` are all valid |
| `detect_region_crossings` seg:322 | stamps an unobserved entry/exit at the first post-pause sample | per run |
| `detect_runs_between_regions` seg:518 | a run spans the pause, or "times out" at the first post-pause sample | per run; a run in progress at the run end ends there with `success=False` |
| `segment_by_velocity` seg:772 | the moving average bridges the pause; epochs end at `times[i+1]` after the pause | per run (smoothing, median dt and hysteresis are each per run) |
| `detect_laps` seg:1049 | a region lap pairs entries across the pause; auto and reference windows straddle it | per run; the auto template is still taken from the first 10% of the whole input |
| `segment_trials` seg:1680 | a trial spans 0 → 1100 s (success when `max_duration` is large) | per run; a trial in progress at the run end is emitted as `success=False`, as at the end of a recording |
| `detect_goal_directed_runs` seg:2230 | the whole input is one run, and the jump adds a geodesic shortcut | one candidate per run |
| `running_direction_labels` seg:1523 | inherits from runs | forwards `max_gap`, `epochs` |
| `detect_boundary_crossings` dec:764 | a label change across the pause is stamped at its midpoint (about 600 s) | per run |
| `extract_pre_decision_window` dec:352 | the window can straddle the pause | restricted to the run that contains `entry_time` |
| `compute_pre_decision_metrics` dec:531 | inherits; `actual_duration` includes the pause | forwards to `extract_pre_decision_window`; duration measured within the window's run |
| `compute_decision_analysis` dec:828 | inherits | forwards to `compute_pre_decision_metrics` and `detect_boundary_crossings` |
| `compute_vte_trial` vte:534 | window can straddle the pause | forwards to `extract_pre_decision_window` |
| `compute_vte_session` vte:617 | same | same; still clamped to the trial start (Phase 2a) |

That is 16 of the 29 position-only "Change" functions. 3e owns the other 13, and also adds per-run kinematics inside `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session`.

These consume `times` but need **no change**:

| Function | Why |
| --- | --- |
| `mean_square_displacement` traj:485 | Pairs form only at real time lags between two observed samples, so a pair that straddles a pause is a genuine displacement. No quantity is interpolated across the pause. |
| `time_efficiency` nav:1358 | It is a wall-clock duration over the input, which is known across a gap. |
| `decision_region_entry_time` dec:289 | It returns the first in-region sample. Under the per-run rule, that is the same sample as on `main`. |
| `distance_to_reward` events/regressors.py:508 | "Last/next reward" is well defined across a gap, and the distance uses the current observed position. |
| `laps_to_direction_labels`, `runs_to_direction_labels`, `goal_pair_direction_labels`, `trials_to_region_arrays`, `time_to_goal` | They only map existing segments onto samples. Once the detectors stop producing spanning segments, they are correct. |

## Tasks

### 1. The per-run refactor pattern

3a already provides `observed_runs`, so this PR adds no shared module. Every segment detector is refactored the same way:

1. Rename the current body to a private `_<name>_contiguous(...)`, keeping its arguments and logic as they are.
2. The public function validates once (including resolving `detect_region_crossings`'s compatibility argument order), then concatenates the per-run results:

   ```python
   from neurospatial.environment.trajectory import observed_runs

   results = []
   for run in observed_runs(times, max_gap=max_gap, epochs=epochs):
       results.extend(_<name>_contiguous(position_bins[run], times[run], env, ...))
   return results
   ```

**No detector re-implements gap logic**: the runs come only from `observed_runs`.

### 2. Segmentation detectors (segmentation.py, decisions.py)

Apply the Task 1 refactor to:

- `detect_region_crossings`;
- `detect_runs_between_regions`;
- `segment_by_velocity` (its `positions`, `times` and smoothing all per run);
- `segment_trials`;
- `detect_goal_directed_runs`;
- `detect_boundary_crossings`. Slice only the per-sample arrays, `position_bins[run]` and `times[run]`. Pass the whole `voronoi_labels` (shape `(n_bins,)`, indexed by bin) to every call. The contiguous function returns `(crossing_times, crossing_directions)`, so extend **both** lists per run and return the pair:

  ```python
  crossing_times: list[float] = []
  crossing_directions: list[tuple[int, int]] = []
  for run in observed_runs(times, max_gap=max_gap, epochs=epochs):
      t, d = _detect_boundary_crossings_contiguous(position_bins[run], voronoi_labels, times[run])
      crossing_times.extend(t)
      crossing_directions.extend(d)
  return crossing_times, crossing_directions
  ```

`running_direction_labels` forwards `max_gap` and `epochs` to both of its `detect_runs_between_regions` calls (1643–1662).

`detect_laps` needs three adjustments, one per method:

- **`method="region"`.** Call `_detect_region_crossings_contiguous` per run and pair only the entries *within* that run. The `np.searchsorted(times, crossing.time)` lookups (1270–1271) use that run's `times[run]`, and the lap bins come from `position_bins[run]`.
- **`method="auto"`.** Compute the template from the first 10% of the **whole** input, as today (1293–1296), so template selection is unchanged. Then search each run, starting at run-local index `max(0, template_size - run.start)`. For the run containing sample 0 that is `template_size`, as today; a run that lies entirely inside the template region is not searched; every later run starts at 0.
- **`method="reference"`.** The template is the caller's `reference_lap`, as today. Search each run from run-local index 0, which is today's `search_start = 0` (:1301) applied per run.

The sliding-window loop (from :1308) moves unchanged into `_detect_laps_search_contiguous(position_bins_run, times_run, env, template, search_start, ...)`.

`_positive_dt` (seg:133) stays as it is: it validates monotonicity and does no gap logic.

### 3. `extract_pre_decision_window` and its callers

`extract_pre_decision_window`:

- Find the run `r` with `times[first_r] <= entry_time <= times[last_r]`, using `observed_runs`.
- Return the samples in `[max(entry_time - window_duration, times[first_r]), entry_time)` from that run only.
- Return empty arrays when `entry_time` is in no run. That is today's empty-window behavior.

`compute_pre_decision_metrics`, `compute_decision_analysis`, `compute_vte_trial` and `compute_vte_session` forward `max_gap` and `epochs` to it. `compute_decision_analysis` also forwards them to `detect_boundary_crossings`.

**Sharing keywords with 3e.** 3e independently adds per-run kinematics inside `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session`, so both PRs need `max_gap`/`epochs` on those three functions. Whichever of 3d and 3e merges first adds the keyword-only `max_gap: float | None = 0.5, epochs=None` (after the existing keywords) with its own forwarding. The second rebases onto it, keeps the existing parameters and docstring entries, and adds only its own forwarding. Neither declares them twice.

### 4. `Environment` sequence methods (environment/trajectory.py)

- **`bin_sequence` and `bin_sequence_with_runs`.** Both gain `max_gap=0.5, epochs=None`.
  - In `_bin_sequence` (:622), compute `mask = interval_valid_mask(times, max_gap=max_gap, epochs=as_intervals(epochs, name="epochs"))`.
  - Drop samples that belong to no run.
  - Treat an invalid interval exactly like the existing outside-sample split: it starts a new run, and dedup never merges across it. That is the existing `gap_splits_runs` index-gap logic (733–736), extended to time gaps.
  - The `BinSequenceWithRuns` duration formula then never includes a pause.
- **`transitions`.** It gains `max_gap=0.5, epochs=None`, which are used only when `times` is given.
  - `_empirical_transitions` (:1080) builds its per-sample bins with `self.bin_sequence(times, positions, dedup=False, outside_value=-1)`. Under the new default that call would drop samples in no run and break the one-to-one alignment between `bins` and `times` that the pair filter below needs. That internal call must pass **`max_gap=None, epochs=None`**, so it keeps returning one bin per sample, and the gap gate is applied to the pairs instead.
  - After building `bins`, compute `mask = interval_valid_mask(times, max_gap=max_gap, epochs=as_intervals(epochs, name="epochs"))` on the same `times` and keep only pairs whose intervals are all valid:

    ```python
    invalid_before = np.concatenate([[0], np.cumsum(~mask)])
    pair_ok = invalid_before[lag:] == invalid_before[:-lag]  # no invalid interval in k .. k+lag-1
    source_bins, target_bins = bins[:-lag][pair_ok], bins[lag:][pair_ok]
    ```

  - When `bins` is given directly with no `times`, nothing changes: no time information means no gap gate.

### 5. Public docstrings for this PR's functions (own task)

Old single-PR plans grouped every behavior docstring into one task. Docstrings ship with the change that alters behavior, so each of 3d and 3e documents its own functions; this task covers the 16 rows above.

- **Parameters:** `max_gap`/`epochs` entries using 3a's text without `spike_window` (see 3a Task 7, or `Environment.occupancy`'s merged docstring). For `compute_pre_decision_metrics`, `compute_vte_trial` and `compute_vte_session`, add them only if 3e has not already (Task 3).
- **Notes:** one sentence: "Each run of samples with gaps no longer than `max_gap` (inside `epochs`) is analyzed as a separate recording; no segment spans a pause." Detectors add: "A segment in progress when a run ends is treated as at the end of a recording (for trials and runs, `success=False`)."
- `extract_pre_decision_window` documents that the window never extends before the start of the run containing `entry_time`.
- `bin_sequence` documents that samples in no run are dropped; `transitions` documents the pair rule.

### 6. CHANGELOG check (own task)

Each earlier commit added its own bullet ([executing.md](executing.md)). Check that `[Unreleased]` has a section "Changed — behavior analyses respect recording gaps" (3e adds to the same section; whichever merges first creates it) listing:

- the spanning-segment bugs fixed, with the audit numbers above (a target entry at 1100 s, a 98 → 1100 s run and a 0 → 1100 s trial marked successful, a 98.1 → 1100 s movement epoch);
- the new keywords on the 16 functions;
- `bin_sequence` dropping samples in no run, and `transitions` dropping pairs across invalid intervals.

## Deliberately not in this phase

- **Kinematics** (velocity, speed, heading, turn angle, dwell), `heading_from_velocity`, `add_positions`, and their docs and notebooks. These belong to 3e, even inside `compute_pre_decision_metrics` and the VTE functions.
- **Argument order.** `(positions, times)` becomes `(times, positions)` in Phase 6b. This PR only adds keywords.
- **Unrelated fixes the audit found while reading these files.** Phase 2a Task 7 fixed the `compute_vte_session` window clamp. Do not re-fix it.
- **Encoding, decoding, PETH, `_intervals.py`, `interval_valid_mask`, `observed_runs`.** These belong to 3a–3c.
- **Position-only functions without `times`** (`traveled_path_length`, `compute_step_lengths`, `compute_turn_angles`, `trajectory_similarity`, `cost_to_goal`, …). With no timestamps there is no gap to detect. Callers pass one run at a time.
- **The per-sample GLM regressors and `align_spikes_to_events`.** Deferred; see 3c and the overview's Open Questions.

## Validation slice

The fixtures are described under Fixtures. Each "fails on `main`" row is a regression test for the A2 repro.

| Test | Asserts |
| --- | --- |
| `tests/behavior/test_segmentation_gaps.py::test_no_crossing_reported_across_pause` | On `pause_track`, `detect_region_crossings(pb, t, env, region_name="target")` returns `[]`. **Fails on `main`**, which reports an entry at t = 1100. |
| `tests/behavior/test_segmentation_gaps.py::test_runs_do_not_span_pause` | `detect_runs_between_regions(..., source="source", target="target", max_duration=2000)`: no run has `start_time < 100 <= 1100 <= end_time`. The run that starts near 98 s ends at or before 99.9 with `success=False`. **Fails on `main`**, where 98 → 1100 is a success. |
| `tests/behavior/test_segmentation_gaps.py::test_trials_do_not_span_pause` | `segment_trials(..., start_region="source", end_regions=["target"], max_duration=2000)`: every trial lies within one run, and the epoch-A trial has `end_time <= 99.9` and `success=False`. **Fails on `main`**, where 0 → 1100 is a success. |
| `tests/behavior/test_segmentation_gaps.py::test_velocity_epochs_do_not_span_pause` | `segment_by_velocity(pos, t, min_speed=5.0, min_duration=0.1)`: every epoch has `end_time <= 99.9` or `start_time >= 1100`. **Fails on `main`** (98.1 → 1100). |
| `tests/behavior/test_segmentation_gaps.py::test_region_laps_do_not_pair_entries_across_pause` | `lap_track`, which has start-region entries at t ≈ 50 (run A) and t ≈ 1150 (run B), gives `detect_laps(method="region")` → no lap spanning `[100, 1100]`. **Fails on `main`**, which reports one lap from 50 to 1150. |
| `tests/behavior/test_segmentation_gaps.py::test_reference_laps_do_not_span_pause` | On `lap_track` with `reference_lap` = the bins of run A's first lap, `detect_laps(method="reference")` returns no lap with `start_time < 100` and `end_time > 1100`, and finds laps in both runs. |
| `tests/behavior/test_segmentation_gaps.py::test_auto_laps_template_unchanged` | On the gap-free `lap_track`, `detect_laps(method="auto")` equals the golden lap list (see the last row), so the per-run search does not change template selection. |
| `tests/behavior/test_segmentation_gaps.py::test_boundary_crossing_not_in_pause` | `pause_track` with `voronoi_labels = np.where(env.bin_centers[:, 0] < 50, 0, 1)` (shape `(n_bins,)`): `detect_boundary_crossings(pb, voronoi_labels, t)` returns two lists of equal length, and no crossing time lies in `(100, 1100)`. **Fails on `main`** (one crossing at ≈ 600 s). |
| `tests/behavior/test_segmentation_gaps.py::test_pre_decision_window_stays_in_run` | On `pause_track`, `extract_pre_decision_window(pos, t, entry_time=1100.5, window_duration=1001.0)` returns exactly the 5 samples `1100.0 … 1100.4`. `compute_pre_decision_metrics` with the same arguments reports `n_samples == 5` and `window_duration <= 0.5`. **Fails on `main`** (probe, scratchpad `remed3/prewin.py`): the window starts at 99.5 s and holds 10 samples, 5 of them before the pause. |
| `tests/behavior/test_segmentation_gaps.py::test_epochs_equal_slicing` | For `segment_trials`, `detect_region_crossings` and `segment_by_velocity` on the gap-free `pause_track`, `epochs=[(0., 100.)]` gives exactly the same result as slicing the arrays to `times <= 100`. |
| `tests/environment/test_bin_sequence.py::test_transitions_skip_pause_pair` | `env.transitions(times=t, positions=p, allow_teleports=True, normalize=False)` on `two_epoch_recording` does not include the last-pre → first-post pair; its total count `.sum()` is `n_samples - 2 = 9998` (one interval dropped). **Fails on `main`** (count 9999). |
| `tests/environment/test_bin_sequence.py::test_transitions_lag_pairs_skip_pause` | With `lag=3` on the same input, the total count is `9997 - 3 = 9994`: the three pairs whose intervals include the gap interval are dropped. |
| `tests/environment/test_bin_sequence.py::test_runs_split_at_pause` | `bin_sequence_with_runs`: when the last pre-pause and first post-pause samples share a bin, they end up in two runs, and no run duration exceeds 100 s. `bin_sequence(..., dedup=False)` on input with an isolated sample (both neighbouring intervals longer than `max_gap`) omits that sample. |
| `tests/behavior/test_segmentation_gaps.py::test_gap_free_outputs_unchanged` | Parametrized over every function in the inventory, on the gap-free fixture variants. The output equals a golden captured on this PR's base commit (see Review), compared exactly for segment lists and with `rtol=1e-12` for floats. This catches accidental behavior change in the `_contiguous` refactor. |

**Existing tests this phase changes.** The new default `max_gap=0.5` turns every interval longer than 0.5 s into a gap, and many existing tests sample at 1 Hz (`times = np.arange(n)` or `[0.0, 1.0, 2.0, …]`). A probe ran `tests/behavior`, `tests/events`, `tests/environment`, `tests/segmentation`, `tests/ops` and `tests/simulation` on `da631a47` (3347 passed) with a pytest plugin that wraps each inventory function and records every call whose `times` has a step longer than 0.5 s (session scratchpad `remed3/gapspy2.py`). **80 tests** pass such `times` to this PR's functions (`bin_sequence` 30, `segment_trials` 13, `detect_region_crossings` 10, `bin_sequence_with_runs` 10, `transitions` 6, `detect_goal_directed_runs` 6, `detect_boundary_crossings` 4, and 1–2 each for the rest):

| Test file | Tests |
| --- | --- |
| `tests/environment/test_bin_sequence.py` | 21 |
| `tests/environment/test_trajectory_extended.py` | 12 |
| `tests/segmentation/test_trials.py` | 11 |
| `tests/environment/test_trajectory_gaps.py` | 7 |
| `tests/behavior/test_detect_region_crossings_argorder.py` | 5 |
| `tests/behavior/test_binning_dt_correctness.py`, `tests/segmentation/test_regions.py`, `tests/segmentation/test_similarity.py` | 4 each |
| `tests/behavior/test_decision_analysis.py`, `tests/environment/test_transitions.py`, `tests/segmentation/test_integration.py` | 3 each |
| `tests/behavior/test_behavioral_integration.py`, `tests/behavior/test_vte.py`, `tests/environment/test_occupancy.py` | 1 each |

Some of them only check input validation and raise before any gate; the rest change. Before changing code, run the full suite once with the new default stubbed in, and list every failure. For each, decide whether the coarse sampling is incidental (pass `max_gap=None`, the documented way to treat long intervals as continuous, or densify `times`) or the point of the test (update the expected value and cite the CHANGELOG bullet). Do not change the default to make tests pass. List every changed test and the choice made in the PR description. `tests/environment/test_trajectory_gaps.py` already tests gap handling; read it first, because some of its cases may now be covered by this PR's rule.

None of these tests exceeds 1 s. No `slow` marks are needed.

## Fixtures

Session-scoped fixtures go in `tests/behavior/conftest.py`. `two_epoch_recording` comes from 3a (`tests/conftest.py`). 3e does not use the two fixtures below, so no other Phase 3 PR adds to this conftest.

- **`pause_track`.** This is `catA/gap3.py` rebuilt:
  - Sampled at 10 Hz: `t = np.r_[np.arange(1000) / 10, 1100 + np.arange(100) / 10]` (epoch A `[0, 100)` with 1000 samples, epoch B `[1100, 1110)` with 100).
  - The animal sits at x = 5 until t = 98, then moves right at 12.5 cm/s. In epoch B it sits at x = 95; y = 5 throughout.
  - `env` is `Environment.from_samples` on a 100 × 10 cm grid of points with `bin_size=2.0`. Regions are `source = box(0, 0, 10, 10)` and `target = box(90, 0, 100, 10)`. The fixture also provides `pb = env.bin_at(pos)`.
- **`lap_track`.** A 1-D circular sequence of bins (a `from_graph` ring or a small grid ring) that passes through a `start` region at t ≈ 10 and 50 in run A, and at t ≈ 1150 and 1190 in run B. Each run holds two complete laps.
- **Gap-free variants.** Each fixture is regenerated over a single continuous span, for the equivalence and unchanged-output tests.

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

- **Goldens come from the base commit, not from `main`.** Before changing code, run `git worktree add ../ns-base $(git merge-base HEAD feat/researcher-first)`. In that worktree, run a capture script that builds the gap-free fixture inputs with the same code as `tests/behavior/conftest.py` and records every inventory function's output. Commit the goldens as literals or a small `.npz` under `tests/behavior/data/`, then `git worktree remove ../ns-base`. Note the base SHA in the test module docstring.
- Every detector's public function contains no state machine; that logic lives only in `_<name>_contiguous`.
- `_empirical_transitions` calls `bin_sequence(..., max_gap=None, epochs=None)`; `git grep -n "bin_sequence(" src/neurospatial` shows no other internal caller relying on the per-sample alignment.
- `uv run pytest --doctest-modules src/neurospatial/behavior src/neurospatial/environment -n 0` passes.
