# Phase 3b — Time windows: one frame kernel for the directional, view and egocentric rates

**Requires:** Phase 3a.

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#time-window-semantics) · [3a](phase-3a-time-windows-core.md)

This is the second Phase 3 PR (the split is tabled in [3a](phase-3a-time-windows-core.md)). It brings the three per-sample ("frame-binned") rate families to the spatial family's standard: one shared validity mask, applied identically to spike counts and occupancy. It is independent of 3c, 3d and 3e.

Follow [executing.md](executing.md) for branching, commits, CHANGELOG bullets, the definition of done and the PR.

**Inputs to read first:**

- **3a's merged code is the source of truth**, not the 3a plan text:
  - [src/neurospatial/_intervals.py](../../../../src/neurospatial/_intervals.py): `resolve_time_windows`, `run_time_bounds`.
  - [src/neurospatial/environment/trajectory.py](../../../../src/neurospatial/environment/trajectory.py): `interval_valid_mask(times, *, start_bin, max_gap, epochs, spike_window)` (no env needed) and `start_allocated_occupancy`.
  - [src/neurospatial/encoding/_binning.py](../../../../src/neurospatial/encoding/_binning.py): `_warn_if_population_silent`, and the spatial kernel this PR's kernel mirrors.
  - The `spike_window` field 3a added to all eight rate result classes, and the `spike_window_assumed` property and `summary()` keys on `SpatialResultMixin`.
- [src/neurospatial/encoding/_directional_binning.py:44](../../../../src/neurospatial/encoding/_directional_binning.py#L44). This holds `compute_directional_occupancy` (occupancy at 177–197, with no gap gate), `_precompute_directional_bins` at :200, `_bin_spikes_with_precomputed_directional_bins` at :246, `bin_directional_spike_train` at :289, and `bin_directional_spike_trains` at :367.
- [src/neurospatial/encoding/_view_binning.py:48](../../../../src/neurospatial/encoding/_view_binning.py#L48). This holds:
  - `_precompute_view_bins`;
  - `_bin_spikes_with_precomputed_view_bins` at :108;
  - `compute_occupancy` at :169 (occupancy block at 295–323);
  - `bin_view_spike_train` at :326;
  - `bin_view_spike_trains` at :475 (occupancy at 596–603).

  None of these has a gap gate.
- [src/neurospatial/encoding/_egocentric_binning.py:375](../../../../src/neurospatial/encoding/_egocentric_binning.py#L375). This holds `compute_egocentric_occupancy` (occupancy at 518–531), `bin_egocentric_spike_train` at :534, and `bin_egocentric_spike_trains` at :700 (occupancy at 851–857; the per-neuron `_bin_single_neuron` closure follows it). None of these has a gap gate.
- Public functions:
  - [directional.py:1629](../../../../src/neurospatial/encoding/directional.py#L1629), [:1866](../../../../src/neurospatial/encoding/directional.py#L1866) and [:2184](../../../../src/neurospatial/encoding/directional.py#L2184): `compute_directional_rate`, `compute_directional_rates` and `is_head_direction_cell` (forwards at :2281). Result constructors at :1856, :2102 (no neurons) and :2168.
  - [view.py:1097](../../../../src/neurospatial/encoding/view.py#L1097), [:1373](../../../../src/neurospatial/encoding/view.py#L1373) and [:1735](../../../../src/neurospatial/encoding/view.py#L1735): `compute_view_rate`, `compute_view_rates` and `is_spatial_view_cell` (forwards at :1807). Result constructors at :1362, :1672 (no neurons; its occupancy call is at :1654) and :1718.
  - [egocentric.py:1311](../../../../src/neurospatial/encoding/egocentric.py#L1311), [:1600](../../../../src/neurospatial/encoding/egocentric.py#L1600) and [:2141](../../../../src/neurospatial/encoding/egocentric.py#L2141): `compute_egocentric_rate`, `compute_egocentric_rates` and `is_object_vector_cell` (forwards at :2230). Result constructors at :1590, :1931 (no neurons; its occupancy call is at :1904) and :1994.
  - Plural `__getitem__` (child constructors) at directional.py:1155, view.py:637 and egocentric.py:713; plural `to_xarray` attrs at directional.py:1131, view.py:595 and egocentric.py:670.
- Evidence (session scratchpad, not in the repo): `branch-triage.md` §A2 and `catA/gap.py`. Two epochs `[0, 100)` and `[1100, 1200)` s at 50 Hz with a true rate of 5 Hz:

  | Family on `main` | Result |
  | --- | --- |
  | Directional | Occupancy totals 1200 s, the worst bin holds 1003 s, and the minimum rate is 0.019 Hz |
  | View | The worst bin holds 1000 s |
  | Egocentric | 0.024 Hz |

- Archive reference only, never cherry-picked: `aad59e74` (directional), `a0ff995a` (view), `d373cc18` (egocentric). These commits are entangled with `TemporalSupport` and do not apply to `main`.
- **Files earlier phases already changed** (line numbers above are from `da631a47`; re-locate by symbol):
  - `encoding/directional.py`, `view.py` and `egocentric.py`: Phase 1 Task 7 routes spike-group input through `as_spike_trains_with_ids` in the three plural functions, and Phase 1 Task 9 changed the directional `to_xarray` attrs. Phase 2a changed the polar plots in `directional.py` and `egocentric.py` (it precedes this PR in the PLAN order). Keep all of these when adding the new keywords.
  - All eight result classes: 3a added the `spike_window` field.

**Contracts referenced:**

- [Time-window semantics](shared-contracts.md#time-window-semantics): the spike+position row; the single implementation point (no family implements its own gap logic); the boundary rule for per-sample (frame) counters (a spike exactly at `times[-1]` is not counted); the visible spike-window assumption; the population-silence warning.
- [Error-message contract](shared-contracts.md#error-message-contract): `epochs`/`spike_window` errors come from `resolve_time_windows`.
- [Input conventions](shared-contracts.md#input-conventions). The new keywords are keyword-only, in the order `max_gap, epochs, spike_window`. The directional raw form stays `(spike_times, times, headings, *, ...)`.

**Designs referenced:** none.

## Inventory (this PR's slice)

"Bridges gaps" means an interval with `dt > 0.5 s` is charged to one bin as occupancy and its spikes are counted.

| Function (file:line) | Current gap behavior on `main` | Change in 3b |
| --- | --- | --- |
| `compute_directional_rate` directional.py:1629 | **bridges gaps** (no gate; NaN headings only) | add `max_gap=0.5, epochs=None, spike_window=None`; shared mask |
| `compute_directional_rates` directional.py:1866 | **bridges gaps** | same, plus silence warning |
| `is_head_direction_cell` directional.py:2184 | **bridges gaps** | add and forward the three keywords |
| `compute_view_rate` view.py:1097 | **bridges gaps** (gates only on view-bin validity) | add the three keywords; shared mask with `start_bin=view_bins` |
| `compute_view_rates` view.py:1373 | **bridges gaps** | same, plus silence warning |
| `is_spatial_view_cell` view.py:1735 | **bridges gaps** | add and forward |
| `compute_egocentric_rate` egocentric.py:1311 | **bridges gaps** (gates only on polar-bin validity) | add the three keywords; shared mask with `start_bin=polar_bins` |
| `compute_egocentric_rates` egocentric.py:1600 | **bridges gaps** | same, plus silence warning |
| `is_object_vector_cell` egocentric.py:2141 | **bridges gaps** | add and forward |

## Tasks

### 1. One frame-binning kernel

All three families assign every sample a bin (`frame_bins`, with `-1` for invalid). Each charges interval `k` to `frame_bins[k]` and looks up spikes at the most recent frame. Today each family carries its own copy of this logic: 6 occupancy blocks and 4 spike-lookup helpers. Replace them all with:

- **Occupancy:** 3a's `start_allocated_occupancy(frame_bins, np.diff(times), mask, n_bins)`, imported from `neurospatial.environment.trajectory`.
- **Spike counts:** one new kernel in `encoding/_binning.py`, next to the spatial kernel `_bin_spike_train_with_stats`:

```python
def count_spikes_by_frame(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    frame_bins: NDArray[np.intp],
    interval_mask: NDArray[np.bool_],
    n_bins: int,
) -> NDArray[np.float64]:
    """Count spikes per bin from the most recent frame, gated by interval validity.

    Interval ``k`` is the half-open ``[times[k], times[k+1])``. A spike at ``t``
    lies in interval ``frame = searchsorted(times, t, "right") - 1``, takes that
    frame's bin, and is kept iff the interval is valid and
    ``frame_bins[frame] >= 0``. A spike before ``times[0]`` or at or after
    ``times[-1]`` lies in no interval and is not counted. Interval validity is
    the SAME mask used for the occupancy denominator
    (``start_allocated_occupancy``), so numerator and denominator drop identical
    intervals.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    spikes = spike_times[(spike_times >= times[0]) & (spike_times < times[-1])]
    frame = np.searchsorted(times, spikes, side="right") - 1  # <= n_samples - 2
    bins = frame_bins[frame]
    keep = interval_mask[frame] & (bins >= 0)
    return np.bincount(bins[keep], minlength=n_bins).astype(np.float64)
```

This snippet was probed: on 3000 frames with 5000 random spikes (none exactly at `times[-1]`) it matches the old `<=`/clip version exactly, and it differs only for a spike at `times[-1]`.

Every kernel-level function below builds its mask with one call:

```python
mask = interval_valid_mask(times, start_bin=frame_bins, max_gap=max_gap,
                           epochs=epochs, spike_window=spike_window)
```

`epochs` and `spike_window` arrive already normalized (the public function calls `resolve_time_windows` once and threads the arrays down).

### 2. Per family

- **Directional (`_directional_binning.py`).**
  - Add `directional_frame_bins(headings, bin_size, *, angle_unit) -> (frame_bins, bin_centers)`. It is built from `_precompute_directional_bins` (:200) as follows:
    - `np.digitize(wrapped, bin_edges) - 1` on finite headings;
    - `>= n_bins → 0`;
    - non-finite headings → `-1`.
  - `compute_directional_occupancy` (:44) keeps its validation (lines 115–175) and gains keyword-only `max_gap=0.5, epochs=None, spike_window=None`. It then builds the mask and calls `start_allocated_occupancy`.
  - `bin_directional_spike_train(s)` (:289, :367) gain the same keywords and call `count_spikes_by_frame`.
  - **Remove `_bin_spikes_with_precomputed_directional_bins` (:246).** It is replaced by `count_spikes_by_frame`. Remove `_precompute_directional_bins` too if `directional_frame_bins` absorbs all its callers.
  - The NaN-heading exclusion is preserved, because `-1` frames fail the bounds gate.
- **View (`_view_binning.py`).**
  - `compute_occupancy` (:169), `bin_view_spike_train` (:326) and `bin_view_spike_trains` (:475) gain `max_gap=0.5, epochs=None, spike_window=None`.
  - After `_precompute_view_bins`, each builds the mask with `start_bin=view_bins` and uses `start_allocated_occupancy` together with `count_spikes_by_frame`.
  - **Remove `_bin_spikes_with_precomputed_view_bins` (:108)** and the two hand-written occupancy blocks (295–323 and 596–603).
- **Egocentric (`_egocentric_binning.py`).**
  - `compute_egocentric_occupancy` (:375), `bin_egocentric_spike_train` (:534) and `bin_egocentric_spike_trains` (:700) change the same way, with `start_bin=bin_indices` (the polar bins). `env` may be `None` (Euclidean metric); the mask needs no env.
  - **Remove the inline occupancy blocks (518–531 and 851–857) and the `_bin_single_neuron` closure in `bin_egocentric_spike_trains`.**
  - `n_jobs` keeps its meaning: the joblib map calls `count_spikes_by_frame` per neuron.

### 3. Public functions and predicates

Each public function gains `max_gap: float | None = 0.5, epochs=None, spike_window=None` (keyword-only), calls `resolve_time_windows` once, and threads the normalized arrays down. Placement, from the actual signatures:

| Function | Insert after | Insert before |
| --- | --- | --- |
| `compute_directional_rate` | `angle_unit` | `backend` |
| `compute_directional_rates` | `angle_unit` | `n_jobs` |
| `is_head_direction_cell` | `angle_unit` | `min_mvl` |
| `compute_view_rate`, `compute_view_rates` | `gaze_offsets` | `method` |
| `is_spatial_view_cell` (has no `gaze_offsets`) | `view_distance` | `method` |
| `compute_egocentric_rate`, `compute_egocentric_rates`, `is_object_vector_cell` | `metric` | `method` (`min_info` for the predicate) |

- The directional functions have no `method`; the egocentric ones have `metric` between `n_direction_bins` and `method`. The table follows the code, not a pattern.
- The predicates forward the three keywords at :2281, :1807 and :2230.
- **Zero-neuron paths.** `compute_view_rates` (its occupancy call at view.py:1654) and `compute_egocentric_rates` (egocentric.py:1904) compute occupancy separately when there are no neurons. Pass the keywords there too, so an empty population reports the same occupancy as a non-empty one. `compute_directional_rates` computes occupancy before branching, so it needs no extra change.
- **Boundary rule.** In every family, a spike exactly at `times[-1]` is no longer counted. On `main` it is counted by the directional kernel (probe: `times = [0, 0.5, 1]`, spike at `1.0` → 1 count from `bin_directional_spike_train`) and by the egocentric kernel without any gate.

### 4. Population-silence warning in the three plural functions

Call 3a's `_warn_if_population_silent` in `compute_directional_rates`, `compute_view_rates` and `compute_egocentric_rates`, exactly as 3a calls it in `compute_spatial_rates`:

- after the spike trains and times are validated;
- only when `spike_window is None`;
- with observed runs `run_time_bounds(times, interval_valid_mask(times, max_gap=max_gap, epochs=E))` (no speed or bounds gate).

Not in the singular functions or the predicates.

### 5. Results record the spike window

3a added the field, the property and the `summary()` keys. This PR sets and propagates the value:

- **Construction sites.** Every result built by a public function passes `spike_window=S`: directional.py:1856, :2102 and :2168; view.py:1362, :1672 and :1718; egocentric.py:1590, :1931 and :1994. The predicates forward `spike_window`, so their internal results record it.
- **Indexing.** `__getitem__`/`__iter__` on `DirectionalRatesResult` (:1155), `ViewRatesResult` (:637) and `EgocentricRatesResult` (:713) pass `spike_window` to the child.
- **`to_xarray()` attrs.** The three plural attrs dicts (directional.py:1131, view.py:595, egocentric.py:670) add the same two entries as 3a's spatial dict: `attrs["spike_window_assumed"] = int(self.spike_window_assumed)` and, only when it is not None, `attrs["spike_window"] = self.spike_window.ravel()`. Keep Phase 1 Task 9's directional attrs.

### 6. Public docstrings (own task)

- **Functions:** the nine in the inventory, plus the `Parameters` of the kernel-level public helpers whose signatures changed (`compute_directional_occupancy`, `bin_directional_spike_train(s)`, view `compute_occupancy`, `bin_view_spike_train(s)`, `compute_egocentric_occupancy`, `bin_egocentric_spike_train(s)`).
- **Parameter text:** 3a's `max_gap`/`epochs`/`spike_window` text, verbatim (see 3a Task 7 or `compute_spatial_rate`'s merged docstring).
- **Plural functions:** the same `Warns` entry as `compute_spatial_rates` (a heuristic: at least 5 units silent for at least 60 s; it cannot detect an outage for one unit or a shorter one, and its silence is not proof of coverage).
- **Notes** on all nine: "An interval is analyzed only if it passes the gap, speed and bounds checks and lies inside `epochs ∩ spike_window`. The same intervals are removed from the spike counts and the occupancy."

### 7. CHANGELOG check (own task)

Check that the "Changed — recording gaps and time windows in rate maps" section 3a created now also lists:

- the bug fixed: directional, view and egocentric rates charged a recording pause to one bin (for example 0.02 Hz instead of 5 Hz);
- the new keywords on these nine functions;
- the population-silence warning on their plural functions;
- **behavior change:** a spike exactly at `times[-1]` is no longer counted by these families. On `main` the directional kernel counted it, gated by the last interval, and the egocentric kernel counted it ungated.

No README or quickstart change: 3a's subsection already describes the keywords for every rate family.

## Deliberately not in this phase

- **`_intervals.py`, `interval_valid_mask`, the spatial family, the result fields and the silence-warning helper.** 3a owns them; call them, do not re-edit them.
- **Decoding, PETH** (3c); **segmentation and env sequence methods** (3d); **kinematics, `heading_from_velocity`, `add_positions`** (3e).
- **New gating knobs** (for example `min_speed` for these families). The contract adds only `epochs`/`spike_window` and reuses `max_gap`.
- **Allocentric object-vector maps.** Phase 5a adds them; this PR changes the existing egocentric functions only.
- **Performance work.** The kernel replacement must not regress runtime (benchmark below), but this PR does not optimize.

## Validation slice

Fixtures come from 3a (`two_epoch_recording`, `continuous_recording`). "Pooled rate" is defined in 3a's validation slice.

| Test | Asserts |
| --- | --- |
| `tests/encoding/test_recording_gaps.py::test_rate_family_recovers_true_rate_across_pause` | Extend 3a's spatial test: parametrized over directional (`bandwidth=None`), view (`fixed_distance`, `view_distance=5`) and egocentric (object `(50, 50)`, `distance_range=(0, 100)`), each singular and plural. Pooled rate = `5.0 ± 5%`; total occupancy ≤ 199.96 s, and for directional/egocentric `== 199.96 ± 1e-6`; max per-bin occupancy < 10 s. **Fails on `main`** for directional (total 1200 s, worst bin ≈1003 s), view (worst bin ≈1000 s) and egocentric (pooled ≈0.8 Hz). |
| `tests/encoding/test_recording_gaps.py::test_predicates_forward_time_windows` | For `is_head_direction_cell`, `is_spatial_view_cell` and `is_object_vector_cell`, a call with `epochs=[(0, 100)]`, `spike_window=(0, 1200)` and `max_gap=1.0`: spy on the underlying `compute_*` (via `monkeypatch` wrapping, which records kwargs) and assert the three keywords arrive unchanged. |
| `tests/encoding/test_time_windows.py::test_epochs_equal_slicing` | Parametrized over the three families. On `continuous_recording`, `epochs=[(0., 100.)]` gives `firing_rate` and `occupancy` equal (`rtol=1e-12`, `equal_nan=True`) to slicing samples to `times <= 100` and spikes to `< 100`. |
| `tests/encoding/test_time_windows.py::test_spike_window_restores_true_rate` | 3a's spatial test, parametrized over the three plural functions: pooled rate `2.5 ± 5%` plus the silence warning without `spike_window`; `5.0 ± 5%` and no warning with `spike_window=(100., 200.)`. |
| `tests/encoding/test_time_windows.py::test_silence_warning_not_in_singular_or_predicates` | The 3a silence scenario passed to each singular function and predicate raises no warning (`warnings.simplefilter("error")`). |
| `tests/encoding/test_time_windows.py::test_zero_unit_occupancy_respects_windows` | `compute_view_rates` and `compute_egocentric_rates` with an empty spike list on `two_epoch_recording` give the same occupancy as with one unit (`rtol=1e-12`), and total occupancy ≤ 199.96 s. **Fails on `main`** for view (the zero-neuron path bridges the pause). |
| `tests/encoding/test_interval_mask_alignment.py::test_spike_at_last_sample_not_counted` | Extend 3a's spatial case: `bin_directional_spike_train`, `bin_view_spike_train`, `bin_egocentric_spike_train`, and `count_spikes_by_frame` directly. With `times = [0, 0.5, 1.0]` and spikes `[0.5, 1.0]`, the total count is 1. `main` counts 2 for directional (probe: old frame kernel `[0, 1, 1]`, new `[0, 1, 0]`). |
| `tests/encoding/test_time_windows.py::test_results_record_spike_window` | Parametrized over the three families, singular and plural (including the zero-neuron plural call). By default `spike_window is None` and `spike_window_assumed is True`, also in `summary()`. With `spike_window=(100., 200.)`: `result.spike_window` array-equals `[[100., 200.]]`, `spike_window_assumed is False`, `summary()["spike_window"] == [[100.0, 200.0]]`, and `rates[0].spike_window` equals the parent's. |
| `tests/encoding/test_spatial_xarray_interop.py::test_frame_family_spike_window_attrs_roundtrip` | In the `test_xarray.yml` job (the workflow runs only this file, `tests/decoding/test_xarray_interop.py` and `tests/decoding/test_result.py`, so the test must live here). Plural directional, view and egocentric: the same assertions as 3a's spatial round-trip test. |
| `tests/benchmarks/test_performance.py::TestFrameFamilyRates` (`@pytest.mark.slow`, uses the `benchmark` fixture) | `compute_directional_rates`, `compute_view_rates` and `compute_egocentric_rates` on 50 units × 60 s at 50 Hz. Run `uv run pytest tests/benchmarks -m "slow and not napari" -n 0` (pytest-benchmark disables itself under xdist) on the base commit and on the branch; the branch must be no slower. Record both numbers in the PR description. |

**Existing tests this phase changes.** A probe ran `tests/encoding`, `tests/simulation` and `tests/public_api` on `da631a47` (2921 passed) with a pytest plugin that records every call passing `times` with a step longer than 0.5 s into a frame-family kernel (scratchpad `remed3/gapspy.py`). Twelve tests do. Those that do more than raise on invalid input will change under the new default `max_gap=0.5`:

- `tests/encoding/test_encoding_directional_binning.py`: `TestNonFiniteHeadingMasking::test_nan_heading_not_folded_into_bin0[nan|inf]` and `test_inf_heading_rejected_or_dropped[nan|inf]`;
- `tests/encoding/test_encoding_view_binning.py`: `TestViewOccupancyNonUniformSampling::test_last_frame_not_counted` and `TestTimesValidation::test_duplicate_times_allowed`;
- `tests/encoding/test_encoding_base.py::TestSummaryEmptyResult::test_empty_batch_summary_does_not_crash`.

The validation-only cases (`test_insufficient_samples`, `test_non_monotonic_times`, the `test_unsorted_times_raises_error` variants) raise before any gate and should be unaffected. For each changed test, decide whether the coarse sampling is incidental (pass `max_gap=None`, or densify `times`) or the point of the test (update the expected value and cite the CHANGELOG bullet). The plugin only sees calls that reach the kernel modules, so also re-run the full suite and list every other failure. Tests calling the removed `_bin_spikes_with_precomputed_*` helpers are deleted with them; none exist on `da631a47` (`git grep` finds no test reference).

## Fixtures

- `two_epoch_recording` and `continuous_recording` from `tests/conftest.py` (3a).
- **Population variants** are built in the tests by offsetting the spike train per unit (`+0.04·u` s), as in 3a.
- **Benchmark data:** 50 units of seeded Poisson spikes at 5 Hz over 60 s; `times = np.arange(3000) / 50`; positions and headings from the `continuous_recording` formulas. Build it in a module-scoped fixture in `tests/benchmarks/test_performance.py`.

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

- Every family obtains its mask from `interval_valid_mask` exactly once per call, and applies that one array to both the counts and the occupancy. `grep -n "np.diff(times)" src/neurospatial/encoding/_*binning.py` shows no hand-rolled occupancy accumulation; occupancy goes through `start_allocated_occupancy` and counts through `count_spikes_by_frame`.
- `_bin_spikes_with_precomputed_directional_bins`, `_bin_spikes_with_precomputed_view_bins` and the `_bin_single_neuron` closure are gone.
- **Gap-free outputs unchanged.** Run `scientific-code-change-audit` on the kernel replacement. Capture goldens on this PR's base commit (`git worktree add ../ns-base $(git merge-base HEAD feat/researcher-first)`; run a capture script there for `firing_rate`/`occupancy` of all six rate functions on `continuous_recording`, save to the scratchpad; `git worktree remove ../ns-base`). On the branch they match at `rtol=1e-12`. `continuous_recording` has no spike at `times[-1]` (its last spike is 199.9 s; `times[-1]` is 199.98 s).
- The xarray test runs without a skip under `uv sync --all-extras`: `uv run pytest tests/encoding/test_spatial_xarray_interop.py -n 0`.
- The benchmark numbers are in the PR description.
