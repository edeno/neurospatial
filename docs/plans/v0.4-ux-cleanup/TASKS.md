# v0.4 UX Cleanup — Task Breakdown

**Committed**: 2026-05-08

Reference: see [PLAN.md](PLAN.md) for design rationale, milestone ordering, and gating criteria. This document tracks per-task progress with file paths, concrete change descriptions, and acceptance criteria.

Source review: [`docs/reviews/UX_REVIEW_2026-05-08.md`](../../reviews/UX_REVIEW_2026-05-08.md). Each task cites the review section number(s) it closes.

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` complete

---

## M0 — Onboarding hotfixes

**Goal**: Every visible code block on the user's first-30-minutes path runs under the current API. CI prevents regression.

**Risk**: Low. Pure documentation and CI.

**Verification**: `uv run pytest tests/ -q` green; new doc-snippet smoke CI job green; manually copy-paste each fixed visible first-run snippet into a fresh Python session and confirm it runs.

- [x] **0.1** Fix `data_samples=` → `positions=` in [README.md:148, 225](../../../README.md). Closes review §1.1. Acceptance: copy-paste of each snippet into a fresh session runs without error. _Landed in `a30dc4e`._

- [x] **0.2** Add required `frame_times=` argument to all README animation snippets at [README.md:430-440](../../../README.md). Use `frame_times = np.arange(len(fields)) / 30.0` as the canonical example. Closes review §1.2. _Landed in `a30dc4e`; covered all four `animate_fields` calls plus the large-scale-dataset section._

- [x] **0.3** Fix four broken snippets in [.claude/QUICKSTART.md](../../../.claude/QUICKSTART.md). Closes review §1.3. _Landed in `a30dc4e`; an additional `bin_at` shape bug at the "Your First Environment" snippet was caught by the M0 0.10 smoke harness during authoring and fixed in `842e8c6`._
  - Line 66: change `from neurospatial import validate_environment` → `from neurospatial.layout.validation import validate_environment`.
  - Line 91: change `Environment.from_graph(G, bin_size=2.0)` → full call with `edge_order` and `edge_spacing` (use the README plus-maze example as the canonical form).
  - Line 191: define `position_bins = env.bin_sequence(times, positions)` before the `segment_trials(trajectory_bins, ...)` call; rename `trajectory_bins` → `position_bins`.
  - Line 739: swap `bin_indices = env.bin_sequence(trajectory, times)` → `bin_indices = env.bin_sequence(times, positions)`.

- [x] **0.4** Fix four broken snippets in [docs/getting-started/quickstart.md](../../../docs/getting-started/quickstart.md). Closes review §1.4. _Landed in `a30dc4e`._
  - Lines 194, 202: rename `layout_type=` → `layout=`. For line 202 (triangular), additionally either remove the example or rewrite to use a supported factory path; `from_samples()` does not currently expose triangular. Document the decision in the task PR. _Resolution: removed the triangular `from_samples` block (only `RegularGrid` and `Hexagonal` are supported there) and pointed readers at `from_layout` / `user-guide/layout-engines.md` for the rest._
  - Line 283: change `env.regions[name].polygon.contains(...)` → `env.regions[name].data.contains(...)` (since `Region.data` holds the polygon for `kind="polygon"`). Optionally use the higher-level `region_membership` API instead. _Resolution: replaced the manual loop with the high-level `env.bins_in_region(name)` API._
  - Line 287: define `position_bin_indices = env.bin_sequence(times, positions)` before reference. _Done; also fixed an additional broken `distance_between(bin_idx, target_bin)` snippet at line 258 (was passing bin indices to a coords-taking method) by switching to `env.distance_to([target_bin])`._

- [x] **0.5** Fix top-level package docstring examples in [`src/neurospatial/__init__.py:140, 190`](../../../src/neurospatial/__init__.py). Closes review §10.10. _Landed in `a30dc4e`._
  - Line 140: change `env.bin_sequence(trajectory)` → `env.bin_sequence(times, trajectory)` with `times` defined; mark as `# doctest: +SKIP` if needed.
  - Line 190: change `env.distance_between(0, 100)` (passes bin indices to a coords-taking method) → `env.path_between(start_bin=0, goal_bin=100)` for graph distance, or `env.distance_between(point_a, point_b)` with coordinates defined. _Resolution: used `float(env.distance_to([100])[0])`. Also removed a stray invalid `units='cm'` kwarg passed to `Environment.from_samples` / `Environment.from_polygon` and replaced it with `env.units = 'cm'` after construction._

- [x] **0.6** Update [examples/README.md](../../../examples/README.md) to cover all 22 notebooks. Closes review §1.7. _Landed in `76a236d`._
  - Replace 01-08 table with 01-22 table including title, prerequisites, estimated time.
  - Fix the misnamed reference to "08_complete_workflow.ipynb" (actual: `08_spike_field_basics.ipynb`).
  - Add a "Goal → Notebook" mapping for the most common neuroscience tasks.

- [x] **0.7** Move `docs/examples/21_directional_place_fields.{py,ipynb}` to canonical home and verify [docs/sync_notebooks.py](../../../docs/sync_notebooks.py) treats it consistently. Closes review §1.8 / §10.7. _Landed in `76a236d` via `git mv` to `examples/`; ran `sync_notebooks.py` to confirm the sync pairs the new canonical source with `docs/examples/21_*.ipynb` cleanly._

- [x] **0.8** Re-execute notebooks 15 and 20 with all outputs cached. Closes review §1.18. Acceptance: `nbformat` shows non-empty `outputs` for at least one cell per code cell. _Landed in `842e8c6`. Notebook 15: 20/21 code cells now have outputs (was 0/21). Notebook 20: 23/24 (was 0/24). The unexecuted cell in each is a no-op cell (e.g. pure import) with no output by design._

- [x] **0.9** Fix `examples/research/distance_to_goal_benchmark.py:11` removed top-level `map_points_to_bins` import. Change to `from neurospatial.ops.binning import map_points_to_bins`. Closes review §10.22. _Local fix applied; **no commit** — `examples/research/` is gitignored at `.gitignore:185`, so the file is not shipped to users. End-user impact of the original audit finding is therefore zero. Documented in `76a236d` commit message so the issue is not re-opened._

- [x] **0.10** **CI smoke test for curated doc snippets.** Add `scripts/test_doc_snippets.py`, an explicit snippet manifest (for example `docs/snippets.yml`), and `.github/workflows/test_docs.yml` that: _Landed in `b9a4df8`._
  1. Extracts only manifest-listed first-run snippets from `README.md`, `.claude/QUICKSTART.md`, `docs/getting-started/quickstart.md`, and the top-level package docstring.
  2. Uses a Markdown fenced-block parser for `.md` files and Python `doctest` / `ast.get_docstring` extraction for `src/neurospatial/__init__.py`; do not treat the docstring as fenced Markdown.
  3. Runs each snippet in an isolated subprocess with deterministic shared fixture setup declared in the manifest.
  4. Requires placeholder-only prose examples such as `load_timestamps()` to be rewritten into runnable examples or marked as `skip` in the manifest with a reason.
  Closes review §10.11. Acceptance: workflow is green on the fixed snippets from tasks 0.1-0.5, and the helper has a regression test showing that a known broken manifest-listed snippet fails the job.

---

## M1 — Stop silent wrong results

**Goal**: No public path silently produces an output that looks plausible but is wrong.

**Risk**: Medium. Behavior changes for users who relied on silent fallbacks.

**Verification**: Add at least one regression test per fix that asserts the new loud failure (warning / exception / metadata).

- [x] **1.1** **`subset()` round-trip.** Rewrite [Environment.subset()](../../../src/neurospatial/environment/transforms.py) to return a `MaskedGrid`, reusing the existing layout serializer instead of registering a one-off `subset` layout kind in [src/neurospatial/layout/factories.py](../../../src/neurospatial/layout/factories.py). Preserve the current boolean-mask `bins` input contract.
  Closes review §9.1. _Landed in `3f93811`. Grid-parent envs route through `Environment.from_mask` so the rebuilt subset has `_layout_type_used="MaskedGrid"` and round-trips through `to_file`/`from_file` cleanly. Graph (1-D) envs keep the inline `SubsetLayout` fallback for now (still not serializable, but graph-subset round-trip is out of M1 scope; existing call sites preserved). Behavior change: connectivity is now the natural grid-neighbor graph among selected cells, not the strict induced subgraph; identical for the typical polygon/region/contiguous selections, different only for pathological selections that broke parent edges in non-grid-derived ways. 4 regression tests cover bins-mask round-trip, polygon round-trip, units/frame preservation, and the canonical layout type._
  Acceptance: round-trip test in `tests/environment/test_serialization.py`:

  ```python
  mask = np.zeros(env.n_bins, dtype=bool)
  mask[[0, 5, 10]] = True
  env_sub = env.subset(bins=mask)
  env_sub.to_file(tmp / "sub")
  env_loaded = Environment.from_file(tmp / "sub")
  assert env_loaded.n_bins == env_sub.n_bins
  assert np.array_equal(env_loaded.bin_centers, env_sub.bin_centers)
  ```

- [x] **1.2** **`bin_at` vs `map_points_to_bins` divergence.** Standardize trajectory semantics on `-1` for points outside the environment. `Environment.bin_sequence` and `Environment.occupancy` must report the same `-1` assignment for out-of-env samples. Keep "nearest in-env bin" only for explicit interpolation queries, not trajectory occupancy.
  Files: [src/neurospatial/environment/queries.py:42-100](../../../src/neurospatial/environment/queries.py) (`bin_at`), [src/neurospatial/environment/trajectory.py](../../../src/neurospatial/environment/trajectory.py) (`bin_sequence`, `occupancy`), [src/neurospatial/layout/mixins.py](../../../src/neurospatial/layout/mixins.py) (`map_points_to_bins`).
  Closes review §6.10 / §9.14.
  Acceptance: regression test that constructs an env, drops a point at `(-5, -5)`, and asserts `bin_sequence` returns `-1` and `occupancy` excludes or reports that sample consistently with the documented `-1` trajectory convention. _Landed in `2e3dfed`. `Environment.occupancy` switched from `map_points_to_bins` (KDTree nearest-neighbor with a heuristic 10x-typical-spacing outside check) to `self.bin_at` (geometric containment, -1 for any point outside the active mask), matching `Environment.bin_sequence`. `map_points_to_bins` docstring updated to flag the silent-wrong path under its default heuristic and point trajectory callers at `bin_at`. `SubsetLayout.point_to_bin_index` upgraded to the standard layout contract (batch input, -1 for outside) as a stopgap (M1 1.1 replaces it). Several test fixtures (`minimal_20x20_grid_env` and three inline test envs) rebuilt as dense grids because they were silently relying on the KDTree smearing to "fill in" bins outside the original two-corner active mask. 2 new TestOccupancyAgreesWithBinSequenceOnOutsideSamples regressions._

- [x] **1.3** **Polar env safety.** Add `is_polar: bool` property and `coordinate_kind: Literal["cartesian", "polar"]` attribute to `Environment`. `from_polar_egocentric` sets `coordinate_kind="polar"`. `Environment.distance_to`, `Environment.distance_between`, `Environment.bin_at` (when called with (x, y) shape input on a polar env), `Environment.contains`, and `apply_transform` raise `ValueError("This environment is polar; method X assumes Cartesian coordinates")`. `plot_field` does NOT raise — it switches axis labels to `"Distance"` / `"Angle (rad)"` and skips `set_aspect("equal")` so `EgocentricRateResult.plot()` still renders polar firing fields correctly.
  Files: [src/neurospatial/environment/core.py](../../../src/neurospatial/environment/core.py), [src/neurospatial/environment/factories.py:773-774](../../../src/neurospatial/environment/factories.py), [src/neurospatial/environment/queries.py](../../../src/neurospatial/environment/queries.py), [src/neurospatial/environment/visualization.py](../../../src/neurospatial/environment/visualization.py).
  Closes review §8.4.
  Acceptance: tests that construct a polar env and assert clear `ValueError` from each Cartesian-assuming method. _Landed in `7377271` (initial wiring) and the M1 review-response commit (additional polar guards). Added `Environment.coordinate_kind` (default `"cartesian"`), `is_polar` property, and `_check_cartesian` helper. `from_polar_egocentric` flips the flag. Wired into `bin_at`, `distance_between`, `distance_to(metric="euclidean")`, `contains` (added in review-response), and `apply_transform` in `ops/transforms.py` (added in review-response — translate/rotate are Cartesian-only operations). `distance_to(metric="geodesic")`, `neighbors`, `path_between`, and `reachable_from` continue to work because they read only the connectivity graph. `plot_field` does NOT raise on polar — instead the axis labels switch to `"Distance"` / `"Angle (rad)"` and the `set_aspect("equal")` call is skipped, so `EgocentricRateResult.plot()` and direct `env.plot_field` on a polar env both render correctly. `EnvironmentProtocol` extended with `coordinate_kind`, `is_polar`, `_check_cartesian`. Persistence: `to_file`/`from_file` and `Environment.copy` carry `coordinate_kind` across; the default value is omitted from serialized JSON so older Cartesian envs round-trip unchanged. `from_dict` review-response fix: previously dropped `coordinate_kind` so polar envs round-tripped as Cartesian; now restored. 11 regression tests cover the flag, the raises (incl. `contains` and `apply_transform`), the polar plot labels, and the round-trip. Drive-by fix: `tests/environment/test_visualization.py:330` was passing unsorted spike_times to `compute_spatial_rate`; M1 1.7's sortedness check rejected it; sorted in place._

- [x] **1.4** **`detect_place_fields` silent drop.** Replace the silent `if mean_rate > max_mean_rate: return []` early-exit at [encoding/spatial.py:2046-2047](../../../src/neurospatial/encoding/spatial.py) with a `PlaceFieldsResult` dataclass returning either the fields or `excluded_reason="mean_rate_above_threshold"` plus a `warnings.warn(..., UserWarning)`. _Landed in `14291b3` (warning) and the M1 review-response commit (PlaceFieldsResult dataclass). The function now returns a `PlaceFieldsResult` frozen dataclass with `fields: list[NDArray[np.int64]]`, `excluded_reason: str | None`, `n_excluded: int`. The result is iterable / sized / indexable / truthy like the underlying list, so existing `for f in detect_place_fields(...)` / `len(...)` / `result[i]` / `if result:` patterns keep working without changes. `population_coverage`'s `place_fields` field type is updated to `list[PlaceFieldsResult]` to match. `field_density_map`'s signature widened to accept either form (legacy hand-built `list[list[NDArray]]` or canonical `list[PlaceFieldsResult]`). M2 task 2.10's "fold this into PlaceFieldsResult" is therefore complete._
  Closes review §6.4.
  Note: because v0.4 carries no backwards compatibility surface, M1 introduces the final structured return instead of a temporary list-returning compromise. M2 task 2.10 only harmonizes the result-class naming / helper methods if needed.
  Acceptance: regression test that passes a mean_rate=10 Hz cell and asserts both the warning and `result.excluded_reason == "mean_rate_above_threshold"`.

- [x] **1.5** **Batch metric `nan` substitution.** [encoding/_metrics.py:548-550, 672-674](../../../src/neurospatial/encoding/_metrics.py) (`batch_grid_scores`, `batch_border_scores`) and [encoding/border.py:243-244](../../../src/neurospatial/encoding/border.py) (Dijkstra failure):
  - Emit a `UserWarning` summarizing failure count when any neuron failed.
  - Add a `failures: NDArray[bool]` mask to the return so callers can identify which neurons failed.
  Closes review §6.5 / §6.6.
  Acceptance: feeding a deliberately-broken neuron (e.g., empty firing rate) produces a warning with count and the `failures` mask flags the right index. _Landed in `dea5404` (initial) and the M1 review-response commit (border_score raise-on-failure). New `BatchScoresResult` frozen dataclass carrying parallel `scores` and `failures` arrays; both batch functions now return it. `BatchScoresResult` implements `__len__`, `__getitem__`, `__array__`, `.shape`, `.dtype`, and `.n_failures` so callers using the result like an ndarray (`np.isnan(result)`, `result[mask]`, `result.shape`) keep working. `border.py:243` Dijkstra-failure path: review-response changed it from `warnings.warn(...) + return np.nan` to `raise RuntimeError(...)`. The single-neuron `border_score` now surfaces the failure loudly; `batch_border_scores` catches it, increments the failure counter, and continues — preserving the per-neuron summary semantics while ending the silent-NaN substitution that hid Dijkstra failures from single-neuron callers. Docstring updated; 2 new regression tests verify the raise and the batch-catches-it behaviors. `SpatialRatesResult.classify()` and `.to_dataframe()` updated to read `.scores` from the wrapper. 5 total tests cover the M1 1.5 surface (3 from initial + 2 review-response)._

- [x] **1.6** **`@check_fitted` / fitted-state validation at entry of public env-consuming functions.** Add fitted-state validation at the top of [compute_spatial_rate](../../../src/neurospatial/encoding/spatial.py), [compute_egocentric_rate](../../../src/neurospatial/encoding/egocentric.py), [compute_view_rate](../../../src/neurospatial/encoding/view.py), [decode_position](../../../src/neurospatial/decoding/posterior.py). _Landed in `83cc42b`. Also wired into the plural variants `compute_spatial_rates`, `compute_view_rates`, `compute_egocentric_rates`. New shared helper `validate_env_fitted` in `encoding/_validation.py`. Egocentric variants check is conditional on `env is not None` because the euclidean distance path explicitly accepts `env=None`. 7 regression tests added. Message text M3 follow-up tracked in M3 task 3.1._
  Use the new free-function `EnvironmentNotFittedError` variant from M3 task 3.2. Until that lands, raise `EnvironmentNotFittedError("Environment", "compute_spatial_rate")` (deferred message rewrite to M3).
  Closes review §6.3.

- [x] **1.7** **`spike_times` validation.** Add `_validate_spike_times` to [encoding/_validation.py](../../../src/neurospatial/encoding/_validation.py): asserts 1-D, finite, sorted, non-negative. Call from `compute_spatial_rate(s)`, `compute_directional_rate(s)`, `compute_egocentric_rate(s)`, `compute_view_rate(s)`. Closes review §6.14. _Landed in `ac30a17`. Empty arrays accepted by default (silent neuron is valid input). Decreasing-times error includes up to five offending indices and the suggestion to call `np.sort(spike_times)`. 10 regression tests added._
  Acceptance: tests that pass shuffled spike times and assert clear `ValueError("spike_times must be sorted in ascending order")`.

- [x] **1.8** **`decode_position` output validation.** Change [decoding/posterior.py:237](../../../src/neurospatial/decoding/posterior.py) default from `validate=False` to `validate=True`. Add validation for non-negative `spike_counts`. Keep posterior row-sum validation strict when `validate=True`: each posterior row must sum to 1 within tolerance or `_validate_output` raises `ValueError`. Closes review §6.14 / §5.5. _Landed in `8143baa`. `_validate_inputs` extended to also reject negative entries in both `spike_counts` (must be non-negative integers) and `encoding_models` (firing rates are non-negative by construction). Error messages quote the count of bad entries and the worst value so failures on large arrays are actionable. `validate=False` remains available as an explicit opt-out for hot loops; 4 regression tests pin the new behavior._

---

## M2 — API consolidation (hard breaks)

**Goal**: One spelling per concept. Canonical argument order from CLAUDE.md actually enforced. Result-class shapes uniform.

**Risk**: High. Land as one big PR (or two coordinated PRs split by surface area: encoding+ops, then behavior+events). The migration doc draft must be ready before merge.

**Verification**: All tests pass. Migration doc cross-check script (M7 task 7.1 deferred reference) confirms each rename. New parameter names appear consistently across all files (grep test).

### M2.A — Parameter-name unification

- [ ] **2.1** **Distance-metric kwarg rename.** Rename `distance_metric`, `distance_type`, `use_geodesic` → `metric` everywhere. Standardize physical-distance APIs to legal values `{"euclidean", "geodesic"}`.
  Files (rename `distance_metric` → `metric`): [encoding/egocentric.py:950, 1215](../../../src/neurospatial/encoding/egocentric.py), [encoding/border.py:31](../../../src/neurospatial/encoding/border.py), [encoding/spatial.py:458, 946](../../../src/neurospatial/encoding/spatial.py), [simulation/models/place_cells.py:166](../../../src/neurospatial/simulation/models/place_cells.py), [simulation/models/object_vector_cells.py:227](../../../src/neurospatial/simulation/models/object_vector_cells.py), [simulation/models/boundary_cells.py:149](../../../src/neurospatial/simulation/models/boundary_cells.py).
  Files (rename `distance_type` → `metric`): [behavior/trajectory.py:171, 408](../../../src/neurospatial/behavior/trajectory.py), [behavior/navigation.py:1216](../../../src/neurospatial/behavior/navigation.py).
  Files (replace `use_geodesic: bool` with `metric: Literal["euclidean", "geodesic"]`): [encoding/_field_metrics.py:744](../../../src/neurospatial/encoding/_field_metrics.py).
  Files (replace `{"euclidean", "graph"}` legal-value set with `{"euclidean", "geodesic"}` — `"graph"` is renamed to `"geodesic"`): [decoding/metrics.py:34](../../../src/neurospatial/decoding/metrics.py), [encoding/_field_metrics.py:99](../../../src/neurospatial/encoding/_field_metrics.py).
  Files (keep `reachable_from(metric=...)` legal values as `{"hops", "geodesic"}`): [environment/queries.py:581](../../../src/neurospatial/environment/queries.py). Rename nothing except the kwarg if needed; do not replace `"hops"` with `"euclidean"` or `"geodesic"`, because `"hops"` means graph-edge count while `"geodesic"` sums edge distances.
  Closes review §3.1, §3.2.
  Acceptance: `grep -rn "distance_metric\|distance_type\|use_geodesic\|\"graph\"" src/neurospatial/` returns no public-API hits; `"hops"` is allowed only in `reachable_from` and its tests/docs.

- [ ] **2.2** **Bandwidth kwarg rename.** Rename `smoothing_sigma`, `kernel_bandwidth` → `bandwidth` everywhere.
  Files: [encoding/directional.py:1321, 1525, 1785](../../../src/neurospatial/encoding/directional.py), [ops/egocentric.py:590](../../../src/neurospatial/ops/egocentric.py), [environment/trajectory.py:63](../../../src/neurospatial/environment/trajectory.py).
  Closes review §3.3.

- [ ] **2.3** **Velocity-threshold kwarg rename.** Rename `velocity_threshold`, `speed_threshold`, `threshold` (in `segment_by_velocity`) → `min_speed`.
  Files: [behavior/segmentation.py:338, 546](../../../src/neurospatial/behavior/segmentation.py), [behavior/decisions.py:143](../../../src/neurospatial/behavior/decisions.py).
  Closes review §3.4.

- [ ] **2.4** **Trajectory-data kwarg rename.** Rename `data` → `positions` in `PositionOverlay`, `data` → `headings` in `HeadDirectionOverlay`.
  Files: [animation/overlays.py:164, 512](../../../src/neurospatial/animation/overlays.py).
  Closes review §3.8.

### M2.B — Result-class consolidation

- [ ] **2.5** **Result-class field rename.** `EgocentricRateResult.ego_env` → `env`; `ViewRateResult.view_occupancy` → `occupancy`. Do not add an `env` field to `DirectionalRateResult`; it is an angular histogram result, not an `Environment`-backed spatial result, and keeps `bin_centers` / `bin_size` as its domain representation.
  Files: [encoding/egocentric.py:199](../../../src/neurospatial/encoding/egocentric.py), [encoding/view.py:212](../../../src/neurospatial/encoding/view.py), [encoding/directional.py:166-170](../../../src/neurospatial/encoding/directional.py).
  Closes review §3.10.
  Acceptance: write a polymorphic helper for environment-backed results and assert it works on `SpatialRateResult`, `EgocentricRateResult`, `ViewRateResult`, and `DecodingResult`. Assert separately that `DirectionalRateResult` exposes angular-domain fields and does not claim `result.env`.

- [ ] **2.6** **`PeriEventResult.firing_rate` attribute.** Convert from method to attribute (cached on construction). The current method (`firing_rate(self) -> NDArray`) computes from histogram + bin_size; move that computation into `__post_init__` or a `@cached_property`.
  Files: [events/_core.py:75](../../../src/neurospatial/events/_core.py).
  Closes review §3.11.
  Acceptance: `result.firing_rate.max()` works on `PeriEventResult` exactly as on `SpatialRateResult`.

- [ ] **2.7** **Result-class method singular/plural normalization.** Pick one convention per result-class category (single-neuron result → singular methods; batch result → plural methods). Rename:
  - `SpatialRateResult.peak_firing_rates()` → `peak_firing_rate()`.
  - `SpatialRateResult.peak_locations()` → `peak_location()`.
  - `ViewRateResult.peak_view_locations()` → `peak_view_location()`.
  Files: [encoding/spatial.py:208, 229](../../../src/neurospatial/encoding/spatial.py), [encoding/view.py:262, 638](../../../src/neurospatial/encoding/view.py), [encoding/_base.py](../../../src/neurospatial/encoding/_base.py) (mixin alias).
  Closes review §3.13.

- [ ] **2.8** **`DecodingResult.uncertainty` rename.** Rename to `posterior_entropy` (matching the free function in `decoding/estimates.py`). Update tests.
  Files: [decoding/_result.py:172](../../../src/neurospatial/decoding/_result.py).
  Closes review §3.12.

- [ ] **2.9** **`is_X_cell` consistency.** Rename result-method abbreviations to match free functions:
  - `DirectionalRateResult.is_hd_cell` → `is_head_direction_cell`.
  - `EgocentricRateResult.is_ovc` → `is_object_vector_cell`.
  - `ViewRateResult.is_view_cell` → `is_spatial_view_cell`.
  Closes review §4.3.

- [x] **2.10** **New `PlaceFieldsResult` dataclass.** [encoding/spatial.py:1944](../../../src/neurospatial/encoding/spatial.py) `detect_place_fields` returns a frozen dataclass with `fields: list[NDArray[np.int64]]`, `excluded_reason: str | None`, `n_excluded: int`. This task finalizes any naming / helper-method polish after M1 task 1.4 introduces the structured return.
  Closes review §7.7. _Folded into M1 1.4 in the M1 review-response commit `312b7a5`. The full dataclass landed there (with `__len__`/`__getitem__`/`__iter__`/`__bool__` so existing list-style callers keep working). No further naming or helper-method polish needed in M2._

- [ ] **2.11** **`bin_sequence` shape-shifting fix.** Split [environment/trajectory.py:331-342](../../../src/neurospatial/environment/trajectory.py) `bin_sequence(return_runs=True)` into two methods: `bin_sequence(times, positions) -> NDArray` (always returns indices) and `bin_sequence_with_runs(times, positions) -> BinSequenceWithRuns` (a dataclass with `bins`, `run_starts`, `run_lengths`).
  Closes review §7.2.

- [ ] **2.12** **`spatial_autocorrelation` shape-shifting fix.** Same treatment for [encoding/grid.py:150](../../../src/neurospatial/encoding/grid.py): split based on `method`. Either two functions or always-tuple return.
  Closes review §7.1.

- [ ] **2.13** **Misc result-type cleanup.**
  - [behavior/trajectory.py:404](../../../src/neurospatial/behavior/trajectory.py) `mean_square_displacement` → returns `MSDResult` with `lags: NDArray`, `msd: NDArray`.
  - [encoding/grid.py:1170](../../../src/neurospatial/encoding/grid.py) `grid_orientation` → absorb into `GridProperties`; remove standalone function or have it return `GridProperties` (with only orientation populated).
  - [behavior/segmentation.py:543](../../../src/neurospatial/behavior/segmentation.py) `segment_by_velocity` → returns `list[Run]` (matching siblings).
  - [encoding/spatial.py:1652-1691](../../../src/neurospatial/encoding/spatial.py) `compute_directional_place_fields`: rename `DirectionalPlaceFields.fields` → `firing_rates`; add `env` and `occupancy` fields to match other result classes.
  Closes review §7.3, §7.4, §7.8, §7.9.

### M2.C — Argument-order consolidation

- [ ] **2.14** **Encoding `env`-first canonicalization.** Reorder to `env` first in:
  - [encoding/population.py:137](../../../src/neurospatial/encoding/population.py) `population_coverage`.
  - [encoding/spatial.py:1944](../../../src/neurospatial/encoding/spatial.py) `detect_place_fields`.
  - [encoding/_field_metrics.py:49, 94, 736, 915](../../../src/neurospatial/encoding/_field_metrics.py) `field_size`, `rate_map_centroid`, `field_shift_distance`, `compute_field_emd`.
  Closes review §2.4-2.7, §2.17.

- [ ] **2.15** **`compute_directional_rate` / `is_head_direction_cell` documented exception.** Keep signatures heading-domain native: no `env` positional argument and no internal `Environment` shim. Document in CLAUDE.md and both function docstrings that directional encoding operates on circular heading bins rather than spatial environments.
  Files: [encoding/directional.py:1313, 1512, 1779](../../../src/neurospatial/encoding/directional.py).
  Closes review §2.1.
  Acceptance: signature tests assert `compute_spatial_rate`, `compute_egocentric_rate`, and `compute_view_rate` take `env` first, while `compute_directional_rate` and `is_head_direction_cell` do not; docs include the rationale for the exception.

- [ ] **2.16** **Egocentric ops reorder.** [ops/egocentric.py:187, 313](../../../src/neurospatial/ops/egocentric.py) `allocentric_to_egocentric`, `egocentric_to_allocentric` reorder to `(positions, headings, targets)`.
  Closes review §2.14.

- [ ] **2.17** **Behavioral functions reorder.**
  - [behavior/segmentation.py:329](../../../src/neurospatial/behavior/segmentation.py) `detect_runs_between_regions`: switch first arg from `positions` to `position_bins`; force `source` and `target` after `*`.
  - [behavior/navigation.py:443, 604, 712](../../../src/neurospatial/behavior/navigation.py) `path_progress`, `distance_to_region`, `cost_to_goal`: reorder to `(position_bins, times, env, *, ...)`.
  - [behavior/decisions.py:261, 801](../../../src/neurospatial/behavior/decisions.py) `decision_region_entry_time`, `compute_decision_analysis`: force region kwargs after `*`.
  - [behavior/vte.py:591](../../../src/neurospatial/behavior/vte.py) `compute_vte_session`: reorder to `(position_bins, times, env, *, decision_region, trials, ...)`.
  Closes review §2.8-2.12.

- [ ] **2.18** **Events reorder.** [events/regressors.py:487](../../../src/neurospatial/events/regressors.py) `distance_to_reward`: swap `times` and `positions` to canonical `(env, times, positions, reward_times, ...)`.
  Closes review §2.13.

- [ ] **2.19** **Decoding trajectory align.** [decoding/trajectory.py:195, 355](../../../src/neurospatial/decoding/trajectory.py) `fit_isotonic_trajectory` and `fit_linear_trajectory`: align signatures to `(env, posterior, times, *, ...)`. Make `env` keyword-only with `None` allowed iff Cartesian-only metric. Standardize on `method` kwarg name (`fit_isotonic_trajectory(estimate_method)` and `fit_linear_trajectory(fitting_method)` both become `method`).
  Closes review §2.16, §11.5.

- [ ] **2.20** **`is_object_vector_cell` raw-data signature.** Rewrite [encoding/egocentric.py:1671](../../../src/neurospatial/encoding/egocentric.py) `is_object_vector_cell` to take `(env, spike_times, times, positions, headings, object_positions, *, ...)` like sister classifiers. Keep a private `_is_object_vector_cell_from_tuning(tuning_curve, peak_rate, ...)` for internal use.
  Closes review §2.15.

- [ ] **2.21** **`*` separator and required-kwarg position cleanup.**
  - [environment/visualization.py:544-572](../../../src/neurospatial/environment/visualization.py) `Environment.animate_fields`: move `frame_times` to be the first positional after `*` (or accept it as the second positional after `fields`).
  - All `Environment.from_*` factories: make `name` keyword-only via `*,` separator. Files: [environment/factories.py](../../../src/neurospatial/environment/factories.py).
  - [simulation/trajectory.py:59](../../../src/neurospatial/simulation/trajectory.py) `simulate_trajectory_ou`: add `*` after `duration` (data positional, settings keyword).
  - [simulation/session.py:128](../../../src/neurospatial/simulation/session.py), [simulation/examples.py:13](../../../src/neurospatial/simulation/examples.py), [simulation/spikes.py:12, 160-169](../../../src/neurospatial/simulation/spikes.py): add `*` separator.
  - All `*CellModel.__init__` in [simulation/models/](../../../src/neurospatial/simulation/models/): add `*` after positional `env`.
  - [encoding/directional.py:1322](../../../src/neurospatial/encoding/directional.py) `compute_directional_rate(bin_size=...)`: move after `*`.
  Closes review §5.1-5.3, §2.21.

### M2.D — Function-name disambiguation

- [ ] **2.22** **Remove `path_efficiency` (float-returning) function.** Keep `compute_path_efficiency` (returning `PathEfficiencyResult`). Update callers.
  Files: [`behavior/__init__.py:38, 50`](../../../src/neurospatial/behavior/__init__.py), [behavior/navigation.py:1270, 1527](../../../src/neurospatial/behavior/navigation.py).
  Closes review §4.2.

- [ ] **2.23** **Re-export hygiene.** Audit `__all__` lists; remove cross-domain re-exports so each public symbol has exactly one canonical import path:
  - [`encoding/__init__.py:46-56, 113`](../../../src/neurospatial/encoding/__init__.py): remove re-exports of `circular_mean`, `rayleigh_test`, `mean_resultant_length`, `compute_egocentric_bearing`, `FieldOfView`, `compute_viewed_location`, `compute_viewshed`. Users import from `stats.circular`, `ops.egocentric`, `ops.visibility` only.
  - [`decoding/__init__.py:188-198`](../../../src/neurospatial/decoding/__init__.py): remove re-exports of `shuffle_*`, `compute_shuffle_pvalue`, `generate_poisson_surrogates`. Users import from `stats` only.
  Closes review §3.14-3.17, §11.

- [ ] **2.24** **`events.__init__` lazy → eager.** Convert [`events/__init__.py:107`](../../../src/neurospatial/events/__init__.py) lazy `__getattr__` to eager imports for consistency with sister modules. (Or, if there's a lazy-import reason, document it.)
  Closes review §3.18.

---

## M3 — Errors and validation

**Goal**: One canonical exception per failure class. Numeric error messages name units.

**Risk**: Medium. Behavior changes for users who catch specific exceptions.

**Verification**: Test suite includes one regression test per migrated check.

- [ ] **3.1** **Free-function `EnvironmentNotFittedError` variant.** Extend [environment/decorators.py:19-92](../../../src/neurospatial/environment/decorators.py) so `EnvironmentNotFittedError` accepts either `(class_name, method_name)` or `(function_name, *, is_function=True)` and formats accordingly. Or split into two classes. Closes review §6.2.

- [ ] **3.2** **Migrate manual "not fitted" checks to `EnvironmentNotFittedError`.** Migrate sites:
  - [composite.py:73](../../../src/neurospatial/composite.py) (currently `ValueError`).
  - [animation/core.py:323-327](../../../src/neurospatial/animation/core.py) (currently `RuntimeError`).
  - [encoding/population.py:236-240, 388-392](../../../src/neurospatial/encoding/population.py) (currently `RuntimeError`).
  - [ops/transforms.py:1439-1443](../../../src/neurospatial/ops/transforms.py) (currently `RuntimeError`).
  - [ops/alignment.py:113, 115](../../../src/neurospatial/ops/alignment.py) (currently `ValueError`).
  - [io/nwb/_environment.py:185-190](../../../src/neurospatial/io/nwb/_environment.py) (currently `ValueError`).
  - [behavior/navigation.py:525, 659, 899](../../../src/neurospatial/behavior/navigation.py): keep `EnvironmentNotFittedError` but switch to the free-function variant from task 3.1 so messages don't say "Environment.path_progress()".
  Closes review §6.1, §6.2.
  Acceptance: a unit test does `try: ...; except EnvironmentNotFittedError: ok` for each migrated site.

- [ ] **3.3** **`@check_fitted` coverage.** Apply the decorator (or equivalent runtime check) to:
  - Methods that succeed silently on unfitted: `Environment.clear_cache`, `Environment.components`.
  - Methods that raise `AttributeError` on unfitted: `Environment.copy`, `Environment.occupancy`, `Environment.rebin`, `Environment.plot_1d`.
  - Exclude `Environment.save` / `Environment.load` from this task because M5 task 5.9 removes both methods outright.
  Closes review §6.21.

- [ ] **3.4** **Custom exception classes.** Add to `src/neurospatial/_exceptions.py` (new module) or wherever `EnvironmentNotFittedError` lives:
  - `RegionNotFoundError(KeyError)`.
  - `BinIndexOutOfRangeError(ValueError)`.
  - `IncompatibleEnvironmentError(ValueError)`.
  - `LayoutNotBuiltError(RuntimeError)`.
  Export `GraphValidationError` from public namespace (currently in `layout/validation.py`, used 22 times, not exported).
  Closes review §6.13.

- [ ] **3.5** **Units in error messages.** Sweep ~30 raise sites. Representative:
  - [encoding/_smoothing.py:409, 412](../../../src/neurospatial/encoding/_smoothing.py): `bandwidth must be positive (in environment units, e.g., cm), got {bandwidth}`.
  - [simulation/spikes.py:514](../../../src/neurospatial/simulation/spikes.py): `modulation_freq must be positive (Hz), got {modulation_freq}`.
  - [simulation/session.py:317-320](../../../src/neurospatial/simulation/session.py): `duration must be positive (seconds), got {duration}`.
  - [simulation/examples.py:130-139](../../../src/neurospatial/simulation/examples.py): `duration` (s), `arena_size` (env units), `bin_size` (env units).
  - [simulation/trajectory.py:664-681](../../../src/neurospatial/simulation/trajectory.py): `speed_mean`, `speed_std`, `pause_duration`, `sampling_frequency` units.
  - [simulation/models/object_vector_cells.py:247, 252, 257, 262](../../../src/neurospatial/simulation/models/object_vector_cells.py): `preferred_distance` (cm), `distance_width` (cm), `max_rate` (Hz), `baseline_rate` (Hz).
  - [simulation/models/head_direction_cells.py:182-185](../../../src/neurospatial/simulation/models/head_direction_cells.py): `max_rate`, `baseline_rate` (Hz).
  - [encoding/grid.py:1141, 1438](../../../src/neurospatial/encoding/grid.py): `bin_size` units.
  Closes review §6.15.

- [ ] **3.6** **Stack-context errors.** Thread a `context: str` argument through [encoding/_binning.py:106-109](../../../src/neurospatial/encoding/_binning.py) so length-mismatch errors say "in compute_spatial_rate: ...". Closes review §6.16. Pattern already exists in [encoding/_validation.py:75-131](../../../src/neurospatial/encoding/_validation.py) `validate_trajectory`; replicate.

- [ ] **3.7** **Warning hygiene.** Sweep `warnings.warn` calls; standardize on `category=UserWarning` for data-quality, `category=RuntimeWarning` for numerical fallbacks, `stacklevel=2` everywhere.
  Files: [stats/circular.py:298, 323, 666, 684, 855, 873](../../../src/neurospatial/stats/circular.py), [regions/io.py](../../../src/neurospatial/regions/io.py), [regions/plot.py:90, 122, 130](../../../src/neurospatial/regions/plot.py), [encoding/_field_metrics.py:903](../../../src/neurospatial/encoding/_field_metrics.py).
  Closes review §6.18.

- [ ] **3.8** **`print()` → `logger.*`.** Replace production prints with module-level `logger.info` / `logger.debug`. Add module-level loggers where missing.
  Files: [animation/backends/video_backend.py:42, 44, 180, 281, 306-320, 345, 420](../../../src/neurospatial/animation/backends/video_backend.py), [animation/backends/html_backend.py:552, 749, 785, 790, 831](../../../src/neurospatial/animation/backends/html_backend.py), [animation/backends/widget_backend.py:925](../../../src/neurospatial/animation/backends/widget_backend.py), [animation/_timing.py:57](../../../src/neurospatial/animation/_timing.py), [simulation/spikes.py:328-331](../../../src/neurospatial/simulation/spikes.py).
  Closes review §6.19.

- [ ] **3.9** **Region overwriting.** Change [regions/core.py:230-235](../../../src/neurospatial/regions/core.py) `Regions.__setitem__` to raise `RegionNotFoundError` (or `RegionAlreadyExistsError`) on duplicate key instead of warning-and-overwriting. Add `Regions.set(name, ...)` as the idempotent replace path. Closes review §6.22.

---

## M4 — Coordinate convention safety

**Goal**: Heading conventions and (x,y) vs (row,col) orderings are unambiguous.

**Risk**: Medium. `is_1d` rename and `peak_coords` reorder are user-visible.

- [ ] **4.1** **`is_1d` rename.** Rename `Environment.is_1d` to `Environment.is_linearized_track` (current semantics: linearized 1D track in a 2D world). Add `Environment.n_dims_intrinsic` if useful for users wanting "n_dims == 1" semantics. Update all callers in `src/`, `tests/`, `examples/`, `docs/`, `CLAUDE.md`.
  Closes review §8.3.

- [ ] **4.2** **`GridProperties.peak_coords` reorder.** Change [encoding/grid.py:114-117](../../../src/neurospatial/encoding/grid.py) `peak_coords` from `(row_offset, col_offset)` to `(x_offset, y_offset)`. Document the breaking change loudly in CHANGELOG (likely the most invisible-but-important breakage).
  Closes review §8.1.

- [ ] **4.3** **Heading-convention docstrings.** Each function whose `headings` argument has a sign convention must state it explicitly:
  - [encoding/directional.py:1338](../../../src/neurospatial/encoding/directional.py) `compute_directional_rate`: "headings: angles in radians, allocentric (0=East, π/2=North)".
  - [encoding/directional.py:1804](../../../src/neurospatial/encoding/directional.py) `is_head_direction_cell`: same.
  - [encoding/view.py:909](../../../src/neurospatial/encoding/view.py) `compute_view_rate`: confirm "0=East".
  - [encoding/egocentric.py:971](../../../src/neurospatial/encoding/egocentric.py) `compute_egocentric_rate`: clarify that input headings are allocentric and the egocentric transform happens internally.
  - [encoding/view.py](../../../src/neurospatial/encoding/view.py) `is_spatial_view_cell`, [encoding/egocentric.py](../../../src/neurospatial/encoding/egocentric.py) `is_object_vector_cell`: same.
  - [ops/egocentric.py:600-602](../../../src/neurospatial/ops/egocentric.py) `heading_from_velocity`: state output convention and `min_speed` units.
  Closes review §8.5, §8.6.

- [ ] **4.4** **`Environment.units` validation (lightweight).** Keep free-form string for v0.4.0 but add a `Environment._validate_units()` method that accepts a small registry (`{"cm", "m", "mm", "px", None}`) and warns on unknown values. Document `units` as advisory.
  Closes review §8.8.

- [ ] **4.5** **`simulate_trajectory_ou` units.** Make [simulation/trajectory.py:70](../../../src/neurospatial/simulation/trajectory.py) `speed_units` required (no `None` default). Default speed values in cm/s rather than m/s. If `speed_units` doesn't match the constructed env's `units`, raise.
  Closes review §8.9.

---

## M5 — Environment class polish

**Goal**: `Environment` behaves predictably under mutation, factory selection, and serialization.

**Risk**: Medium.

- [ ] **5.1** **Mutability via version counter.** Add `_state_version: int = 0` to `Environment`. Increment on `_setup_from_layout` and on `subset()`/`apply_transform()`/`rebin()`. Cached properties verify the version on access; mismatch → recompute. Mutations of `bin_centers` or `connectivity` are detected via a hash check on first access after the documented mutation paths.
  Files: [environment/core.py](../../../src/neurospatial/environment/core.py), [environment/metrics.py](../../../src/neurospatial/environment/metrics.py), [environment/fields.py](../../../src/neurospatial/environment/fields.py).
  Closes review §9.2.
  Note: this is the lighter alternative to `@dataclass(frozen=True)`. If the maintainer chooses frozen instead (per PLAN open question #2), this task is replaced by a full migration to immutable Environment.

- [ ] **5.2** **`SubsetLayout` KDTree caching.** Cache the KDTree on the layout instance; invalidate on env state-version change.
  Files: [environment/transforms.py:613-617](../../../src/neurospatial/environment/transforms.py).
  Closes review §9.3.
  Acceptance: benchmark `bin_at` on a subset env over 10⁴ queries; assert at most one KDTree construction.

- [ ] **5.3** **`from_image` / `from_mask` rename.** Rename to `from_pixel_mask(image_mask, pixel_size, ...)` and `from_grid_mask(active_mask, grid_edges, ...)`. The old names `from_image` and `from_mask` are removed outright (no aliases). Update CLAUDE.md to remove the misleading "bin_size is REQUIRED" rule (it is required only for grid-inferring factories).
  Files: [environment/factories.py:482-487, 558](../../../src/neurospatial/environment/factories.py).
  Closes review §9.6, §9.4.

- [ ] **5.4** **`from_graph` examples.** Add 2-3 `Examples` blocks to [environment/factories.py:353-389](../../../src/neurospatial/environment/factories.py) `from_graph` docstring (T-maze, plus-maze, linear track). Closes review §9.7.

- [ ] **5.5** **Region API consolidation.**
  - `Regions.add(name, ...)` raises on duplicate.
  - `Regions.update_region(name, ...)` raises on missing.
  - `Regions.__setitem__(name, region)` raises on duplicate (M3 task 3.9).
  - `Regions.__delitem__(name)` and `Regions.remove(name)` both raise on missing.
  - Add `Regions.set(name, ...)` as the idempotent path.
  Files: [regions/core.py](../../../src/neurospatial/regions/core.py).
  Closes review §9.9, §9.10.

- [ ] **5.6** **`bin_attributes`, `edge_attributes`, `differential_operator` cost.** Convert from `@cached_property` to methods (`get_bin_attributes()`, `get_edge_attributes()`, `get_differential_operator()`) so the cost is visible in code and tab completion.
  Files: [environment/metrics.py:236-252, 299-316](../../../src/neurospatial/environment/metrics.py), [environment/fields.py:91-103](../../../src/neurospatial/environment/fields.py).
  Closes review §9.11, §9.12.

- [ ] **5.7** **Remove `mask_for_region`; standardize on `region_mask`.** Delete `Environment.mask_for_region` outright. `region_mask` is the canonical method (accepts name / list / `Region` / `Regions`). Use `covers` as the containment predicate (inclusive of polygon-boundary bins). Migrate all callers in `src/`, `tests/`, `examples/`, `docs/`. No alias.
  Files: [environment/regions.py](../../../src/neurospatial/environment/regions.py).
  Closes review §9.8.

- [ ] **5.8** **`__repr__` and `__str__`.** Fix the `name=None` repr bug for empty-string names (use `repr(self.name)` to make empty string visible). Add `__str__` returning `info()`.
  Files: [environment/core.py:387-390](../../../src/neurospatial/environment/core.py).
  Closes review §9.18, §9.19.

- [ ] **5.9** **Remove `Environment.save`/`load`.** Delete both methods outright. The pickle-based serialization path is replaced by `to_file`/`from_file` (JSON metadata + npz arrays). Audit the codebase and tests for any internal use of `save`/`load` and migrate them to `to_file`/`from_file` in the same task. No `DeprecationWarning` — the methods simply do not exist in v0.4.0.
  Files: [environment/serialization.py:239-241](../../../src/neurospatial/environment/serialization.py).
  Closes review §9.20.

---

## M6 — Examples and docs site

**Goal**: A neuroscientist with NWB data + spikes can find a runnable starting point in <30 seconds.

**Risk**: Low-medium. New notebooks add surface area.

- [ ] **6.1** **Notebook: object-vector cells.** Add `examples/23_object_vector_cells.ipynb` mirroring `22_spatial_view_cells.ipynb`. Use `compute_egocentric_rate`, `is_object_vector_cell`, `object_vector_score`, `plot_object_vector_tuning`. Closes review §1.12. Slot 21 was reserved by the original review for OVC, but `21_directional_place_fields.ipynb` is already published there (moved to `examples/` in M0 task 0.7); OVC therefore takes the next free slot 23.

- [ ] **6.2** **Notebook: head-direction tuning.** Add `examples/24_head_direction_tuning.ipynb`. Use `compute_directional_rate`, `is_head_direction_cell`, `rayleigh_test`, `plot_head_direction_tuning`. Closes review §1.11.

- [ ] **6.3** **Notebook: peri-event PSTH.** Add `examples/25_peri_event_psth.ipynb`. Use `peri_event_histogram`, `align_spikes_to_events`, GLM regressors. Closes review §1.10.

- [ ] **6.4** **Notebook: NWB loading.** Add `examples/26_loading_from_nwb.ipynb`. Use a small public NWB file from DANDI (e.g., a Buzsaki-lab open-field session). Closes review §1.9.

- [ ] **6.5** **Bandit dataset friendly skip.**
  - Add tracked `data/README.md` with download instructions (Zenodo DOI per PLAN open question #3).
  - Update `.gitignore` with `!data/README.md` so the README is tracked while downloaded data files remain ignored.
  - Add `examples/19_real_data_bandit_task.py` graceful skip when `data/` is empty: print URL, exit 0.
  Closes review §1.6, §10.25.

- [ ] **6.6** **README front-door place-field example.** Add a "First place field" section to [README.md](../../../README.md) immediately after the existing Quickstart. ~30 lines: simulate trajectory + spikes, build env, call `compute_spatial_rate`, plot. Use the canonical `simulate_trajectory_ou` + `PlaceCellModel` + `generate_population_spikes` pattern. Closes review §1.5.

- [ ] **6.7** **Glossary page.** Add `docs/glossary.md` defining: `bin`, `node`, `cell` (graph), `field`, `rate map`, `tuning curve`, `place cell`, `place field`, `egocentric`, `object-vector`, `allocentric`, `view field`, `linearization`, `occupancy`, `spike-triggered`. Cross-link from [docs/getting-started/core-concepts.md](../../../docs/getting-started/core-concepts.md) and from the README. Closes review §10.13.

- [ ] **6.8** **`mkdocs.yml` nav update.** Add the eight orphaned user-guide pages and four orphaned `docs/*.md` pages to nav. Closes review §10.4.

- [ ] **6.9** **`docs/api/index.md` expansion.** Add sections (with mkdocstrings auto-gen) for `encoding`, `decoding`, `behavior`, `events`, `ops.egocentric`, `ops.visibility`, `ops.basis`, `stats`, `animation`, `io.nwb`. Closes review §10.5.

- [ ] **6.10** **`docs/examples/index.md` rewrite.** Document all 22 notebooks (review §10.6). Synchronize with `examples/README.md` from M0 task 0.6.

- [ ] **6.11** **Notebook narrative consistency.** Convert "see **02_layout_engines.ipynb**" bold text to relative MD links. Add a "Prerequisites" line at the top of each notebook 09-22 pointing to upstream notebooks. Closes review §1.19, §1.20.

- [ ] **6.12** **Notebook regen CI.** Add a CI job that re-executes a representative notebook (e.g., 11_place_field_analysis) on PRs to catch silent regressions in the example surface. Use `papermill` or `jupyter nbconvert --execute`.

---

## M7 — v0.4.0 release prep

**Goal**: v0.4.0 is shippable. Migration is documented and verified.

**Risk**: Low.

- [ ] **7.1** **`docs/migrating-to-v0.4.md`.** Create a single migration page with one row per breaking change. Columns: `Category` (kwarg / function / signature / return-type / behavior), `From (v0.3)`, `To (v0.4)`, `Migration example`. Cross-check script `scripts/check_migration_doc.py` greps the page for `To` cells and confirms each one actually exists in the codebase. The script does **not** check `From` cells for absence — `From` names are gone from v0.4.0 by construction (no aliases, no deprecation). Run the script in CI.

- [ ] **7.2** **`CHANGELOG.md`** under `## [0.4.0]`:
  - **Breaking changes**: every renamed kwarg, every reordered signature, every removed function, every removed method, every removed alias / re-export, every result-class field rename. With a link to the migration doc.
  - **Added**: notebooks (M6 6.1-6.4), glossary (6.7), `docs/api/index.md` expansion (6.9), CI doc-snippet test (M0 0.10), CI notebook regen (M6 6.12), `is_polar` flag (M1 1.3), `PlaceFieldsResult` (M2 2.10), `BinSequenceWithRuns` (M2 2.11), `MSDResult` (M2 2.13), free-function `EnvironmentNotFittedError` variant (M3 3.1), four custom error classes (M3 3.4), `from_pixel_mask` and `from_grid_mask` factories (M5 5.3).
  - **Changed**: behavior changes from M1 (loud failures replacing silent ones) and M3 (canonical exception types).
  - **Fixed**: the silent-failure list from M1, the `repr(env)` `name=None` bug (M5 5.8).
  - **Removed**: `Environment.save`/`load` (M5 5.9), `path_efficiency` float-returning function (M2 2.22), all cross-domain re-exports (M2 2.23), `from_image` and `from_mask` factory aliases (M5 5.3, replaced by `from_pixel_mask` / `from_grid_mask`).
  - No "Deprecated" section. There are no deprecations in this release; everything is a clean delete-and-replace.

- [ ] **7.3** **Version bump.**
  - [pyproject.toml](../../../pyproject.toml) `version = "0.4.0"`.
  - Citation strings: [README.md](../../../README.md) line 567, [docs/index.md](../../../docs/index.md) line 109, [examples/README.md](../../../examples/README.md) line 240.
  - Header strings: [README.md](../../../README.md) line 85 ("Tested Dependency Versions"), line 591 ("Alpha" / "Beta" decision).
  - Module docstring: [src/neurospatial/ops/visibility.py:9](../../../src/neurospatial/ops/visibility.py) "(v0.4.0+)".
  Closes review §10.1.

- [ ] **7.4** **`examples/_style.py`.** Shared matplotlib styling: Wong/Okabe-Ito palette, fixed `figsize`, font sizes. Each example notebook imports it at top. Closes review §10.17.

- [ ] **7.5** **Final smoke.**
  - Full `uv run pytest` green.
  - Full `mkdocs build --strict` green.
  - Full notebook re-execution on a fresh `uv venv` environment.
  - `pip install -e .` from a clean checkout produces a working interactive session.
  - README quickstart end-to-end runs without modification.
  - Migration doc cross-check script (task 7.1) green.

- [ ] **7.6** **Tag and release.** `git tag v0.4.0`, push, build wheel/sdist, upload to PyPI (or whatever the project's release path is). Update GitHub release notes from `CHANGELOG.md`.

---

## Tracking links

- Source review: [`docs/reviews/UX_REVIEW_2026-05-08.md`](../../reviews/UX_REVIEW_2026-05-08.md).
- Plan: [PLAN.md](PLAN.md).
- Sibling plans for context:
  - [encoding-cleanup/PLAN.md](../encoding-cleanup/PLAN.md) — overlapping work on encoding entry-points.
  - [environment-refactor/PLAN.md](../environment-refactor/PLAN.md) — backlog item that this plan partially addresses (M5).

## Notes for implementers

- Land milestones in order; M2 must precede M3, M4, M5, M6.
- Each milestone PR's description should reference the closing TASKS task IDs and the review section numbers.
- Treat the migration doc (M7 task 7.1) as a living document edited within each PR that introduces a breaking change.
- Hard breaks should not surprise the maintainer mid-flight; if a task in M2 turns out to break more than expected, raise it in the open-questions section of PLAN.md before merging.
