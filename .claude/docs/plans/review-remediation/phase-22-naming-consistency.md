# Phase 22 — Cross-module API-convention consistency

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

One PR. Mechanical but wide: it renames public arguments, inserts keyword-only
`*` separators, reorders sibling parameters, unifies one event-window contract,
single-sources the object-vector-cell classification criterion, and refreshes two
stale `Protocol` stubs. **No behavior change to any happy-path call that already
used keyword arguments** — every change is signature, name, argument order, or
docstring. The one functional-looking change (OVC delegation) is a deliberate
*bug-removal-by-construction*: two classifiers that compute different things are
collapsed onto one implementation so they can no longer disagree.

**Ordering (per [overview](overview.md#dependency-ordering)):** land Phase 22
**after** the correctness phases that touch these same files — Phase 5 (`encoding/`),
Phase 7 (`events/`), Phase 8 (`ops/`), Phase 1 (`decoding/`), Phase 6 (`behavior/`),
Phase 10 (`layout/`), Phase 11 (`regions/`), Phase 12 (`annotation/`), Phase 3
(`simulation/`), Phase 2 (`environment/`). Those phases edit the *bodies* of several
functions whose *signatures* this phase rewrites; sequencing 22 last keeps the
rename diff conflict-free and atomic.

Pre-1.0 policy is replace-in-place — **no deprecation shims, no aliases** — but
**every renamed public name gets a CHANGELOG entry** under `## Unreleased`
(`CHANGELOG.md` already has that section). The rename tasks below each name their
CHANGELOG line.

**Inputs to read first:**

- [src/neurospatial/decoding/assemblies.py:368-375](../../../../src/neurospatial/decoding/assemblies.py) — `detect_assemblies(spike_counts, *, algorithm, n_components, z_threshold, random_state=None)`. Rename `random_state` → `rng`. The body at lines ~440-470 passes `random_state` into the sklearn estimator constructor; update the internal seed plumbing too.
- [src/neurospatial/ops/basis.py:117,347,575,760,1005](../../../../src/neurospatial/ops/basis.py) — five public functions take `random_state: int | np.random.Generator | None`, all **positionally**, with **no `*` separator**: `select_basis_centers` (117), `geodesic_rbf_basis` (347), `heat_kernel_wavelet_basis` (575), `chebyshev_filter_basis` (760), `spatial_basis` (1005). Lines 192-195 normalize `random_state` into a `Generator`; the inner `_select_centers_*` helpers already use a local name `rng`.
- [src/neurospatial/events/regressors.py:29-34,211-215,347-352](../../../../src/neurospatial/events/regressors.py) — the three regressor signatures to unify: `time_to_nearest_event(..., *, signed, max_time=None)` (29), `event_count_in_window(sample_times, event_times, window: tuple[float,float])` — **`window` positional, not keyword-only** (211), `event_indicator(..., *, window: float = 0.0)` — scalar half-width (347).
- [src/neurospatial/encoding/egocentric.py:388-461,1869-1979,1833-1866](../../../../src/neurospatial/encoding/egocentric.py) — `EgocentricRateResult.is_object_vector_cell(self, min_info=0.3)` returns `egocentric_spatial_information() > min_info` (388-461). The **free** `is_object_vector_cell(env, spike_times, times, positions, headings, object_positions, *, distance_range, n_distance_bins, n_direction_bins, metric, score_threshold=0.3, min_peak_rate=5.0)` (1869-1979) classifies via `_is_object_vector_cell_from_tuning` → `object_vector_score` + `min_peak_rate` (1833-1866). **These two criteria genuinely differ** (spatial-information threshold vs. score+peak-rate). The free function already builds the result via `compute_egocentric_rate`; it just classifies it with a *different* rule.
- [src/neurospatial/encoding/view.py:861-869,1468-1482](../../../../src/neurospatial/encoding/view.py) — `compute_view_rate(..., *, gaze_model=..., view_distance=10.0, ...)` declares `gaze_model` **before** `view_distance` (867-869); the sibling `is_spatial_view_cell(..., *, view_distance=10.0, gaze_model=..., ...)` declares them **reversed** (1474-1476).
- [src/neurospatial/encoding/spatial.py:1042](../../../../src/neurospatial/encoding/spatial.py) — `SpatialRatesResult.classify(...)`; sibling batch classifiers are `EgocentricRatesResult.detect_ovcs` ([egocentric.py:792](../../../../src/neurospatial/encoding/egocentric.py)) and `ViewRatesResult.detect_view_cells` ([view.py:714](../../../../src/neurospatial/encoding/view.py)). Naming `classify` vs `detect_*` is inconsistent.
- [src/neurospatial/decoding/trajectory.py:195-216,275](../../../../src/neurospatial/decoding/trajectory.py) — `fit_isotonic_trajectory(env, posterior, times, *, increasing, method)`; `env` is mandatory-positional but immediately `del env` at line 275 (kept "for signature parity"). **(This is the real unused-mandatory-`env` footgun; SUMMARY mis-attributed it to `spatial.py:1042`.)**
- [src/neurospatial/decoding/metrics.py:29-35,228-234](../../../../src/neurospatial/decoding/metrics.py) — `decoding_error(decoded_positions, actual_positions, env=None, *, metric)` places `env` **3rd positional** (32); `confusion_matrix(env, posterior, actual_bins, *, summary_method=...)` uses `summary_method` (233) where `fit_isotonic_trajectory`/`fit_linear_trajectory` use `method` (trajectory.py:201) for the same `{"map","expected"}` choice.
- [src/neurospatial/ops/normalize.py:161-165](../../../../src/neurospatial/ops/normalize.py) — `combine_fields(fields, weights=None, mode="mean")`; algorithm params positional, no `*`.
- [src/neurospatial/ops/smoothing.py:70-75](../../../../src/neurospatial/ops/smoothing.py) — `compute_diffusion_kernels(graph, bandwidth_sigma, bin_sizes=None, mode="transition")`; no `*`.
- [src/neurospatial/ops/distance.py:55-59,410-414](../../../../src/neurospatial/ops/distance.py) — `geodesic_distance_matrix(G, n_states, weight="distance")` (55), `pairwise_distances(G, nodes, weight="distance")` (410); `weight` positional, no `*`.
- [src/neurospatial/behavior/vte.py:593-604](../../../../src/neurospatial/behavior/vte.py) — `compute_vte_session(positions, times, env, *, ...)` places `env` **3rd** (596); sibling `compute_decision_analysis(env, positions, times, *, ...)` is env-first ([decisions.py:797-806](../../../../src/neurospatial/behavior/decisions.py)).
- [src/neurospatial/behavior/segmentation.py:1749-1755](../../../../src/neurospatial/behavior/segmentation.py) — `detect_goal_directed_runs` raises **`KeyError`** for a missing region (1752); sibling validators raise `ValueError`.
- [src/neurospatial/layout/engines/triangular_mesh.py:77-89](../../../../src/neurospatial/layout/engines/triangular_mesh.py) — `def build(self, boundary_polygon, point_spacing)`; **no `*`**, and **no `@capture_build_params`** decorator (siblings `masked_grid.py:60` and `image_mask.py:57` have it).
- [src/neurospatial/annotation/io.py:13-19,72-75](../../../../src/neurospatial/annotation/io.py) — `regions_from_labelme(json_path, calibration=None, *, label_key, points_key)` (13); `calibration` positional before `*`. `regions_from_cvat(xml_path, calibration=None)` (72) — no `*` at all.
- [src/neurospatial/annotation/_boundary_inference.py:74-82](../../../../src/neurospatial/annotation/_boundary_inference.py) — `boundary_from_positions(positions, method=None, *, config, ...)`; `method` (algorithm choice) positional before `*`.
- [src/neurospatial/annotation/converters.py:21-31](../../../../src/neurospatial/annotation/converters.py) — `shapes_to_regions(shapes_data, names, region_types, calibration=None, simplify_tolerance=None, *, ...)`; `calibration` and `simplify_tolerance` positional before `*`.
- [src/neurospatial/simulation/trajectory.py:62-76,460-469,598-608](../../../../src/neurospatial/simulation/trajectory.py) — `simulate_trajectory_ou(env, duration, *, ...)` **has** `*` (65); `simulate_trajectory_sinusoidal(env, duration, sampling_frequency=500.0, ...)` (460) and `simulate_trajectory_laps(env, n_laps, speed_mean=0.1, ...)` (598) have **no `*`**.
- [src/neurospatial/simulation/spikes.py:173-182](../../../../src/neurospatial/simulation/spikes.py) — `generate_population_spikes(models, positions, times, *, refractory_period, seed, show_progress, headings=None)`; `headings` (animal state) sits **after** the algorithm params.
- [src/neurospatial/regions/core.py:542-560,562-591](../../../../src/neurospatial/regions/core.py) — `region_center(self, region_name)` (562) uses `region_name`; sibling `area(self, name)` (542) uses `name`.
- [src/neurospatial/behavior/navigation.py:1083-1104](../../../../src/neurospatial/behavior/navigation.py) — `heading_direction_labels(..., min_speed=5.0)`; docstring says only "Minimum speed threshold" with **no unit**.
- [src/neurospatial/environment/_protocols.py:378-389,718-811](../../../../src/neurospatial/environment/_protocols.py) — stale stubs vs real impls: `occupancy` stub has `max_gap=None` (385) but the real method defaults `max_gap=0.5` ([trajectory.py:100](../../../../src/neurospatial/environment/trajectory.py)); `animate_fields` stub still has `fps: int = 30` (724), lacks `speed`, and makes `frame_times` optional (`= None`, 742). Real impl ([visualization.py:562-591](../../../../src/neurospatial/environment/visualization.py)) has `*, frame_times: NDArray (required), ..., speed: float = 1.0`, **no `fps`**.

**Contracts referenced:**

- [Input-validation helpers](shared-contracts.md#input-validation-helpers) — this phase does **not** add validation; it only unifies the *type* of the `events` window. The NaN/Inf guards in `events/regressors.py` land in Phase 7.

## Tasks

Grouped by change-type. Each task lists exact verified refs and the concrete new
signature. Public renames carry their CHANGELOG line inline.

### Task group A — RNG argument rename (`random_state` → `rng`)

The rest of the library (`stats/`, `simulation/`) already uses `rng`. Rename the
six public surfaces below; keep accepting `int | np.random.Generator | None`.

1. **`detect_assemblies`** — [decoding/assemblies.py:368-375](../../../../src/neurospatial/decoding/assemblies.py).
   New signature:
   ```python
   def detect_assemblies(
       spike_counts: NDArray[np.float64],
       *,
       algorithm: Literal["ica", "pca", "nmf"] = "ica",
       n_components: int | Literal["auto"] = "auto",
       z_threshold: float = 2.0,
       rng: int | np.random.Generator | None = None,
   ) -> AssemblyDetectionResult:
   ```
   Update the docstring `random_state` parameter block → `rng`, and the body where
   `random_state` is forwarded into the sklearn estimator (search the function body
   for `random_state=`). sklearn's own constructor kwarg stays `random_state=` —
   only the *public* neurospatial parameter is renamed; convert via
   `np.random.default_rng(rng)` / pass an int seed into sklearn as it does today.

   > Note: `detect_assemblies` is also edited by phases 1 and 25; land in order
   > 1 → 22 → 25; the edits are disjoint (REV math / signature rename / clamp guard).

2. **The five `ops/basis.py` functions** — [ops/basis.py:117,347,575,760,1005](../../../../src/neurospatial/ops/basis.py).
   Rename `random_state` → `rng` in each public signature **and** add the `*`
   separator (this overlaps Task group D — do both in one edit per function). Concrete
   new signatures:
   ```python
   def select_basis_centers(
       env: Environment,
       n_centers: int,
       *,
       method: Literal["kmeans", "farthest_point", "random", "grid"] = "kmeans",
       rng: int | np.random.Generator | None = None,
   ) -> NDArray[np.int_]: ...

   def geodesic_rbf_basis(
       env: Environment,
       centers: NDArray[np.int_] | None = None,
       *,
       sigma: float | Sequence[float] = 5.0,
       n_centers: int | None = None,
       center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
       normalize: Literal["unit", "max", "none"] = "unit",
       rng: int | np.random.Generator | None = None,
   ) -> NDArray[np.float64]: ...

   def heat_kernel_wavelet_basis(env, ..., *, ..., rng=None) -> ...: ...
   def chebyshev_filter_basis(env, ..., *, ..., rng=None) -> ...: ...
   def spatial_basis(env, *, ..., rng=None) -> ...: ...
   ```
   For `heat_kernel_wavelet_basis`/`chebyshev_filter_basis`/`spatial_basis`, read
   each current signature and move every defaulted algorithm parameter behind `*`
   while renaming `random_state` → `rng`. Internal calls already pass
   `random_state=random_state` into `select_basis_centers`; change those forwarded
   keywords to `rng=rng`. Update each `random_state : ...` docstring block → `rng : ...`.

3. **CHANGELOG** (one entry covering the group), under `## Unreleased` →
   add an `### API changes` subsection:
   ```
   - Renamed the `random_state` argument to `rng` in `detect_assemblies`
     (`neurospatial.decoding`) and in `select_basis_centers`,
     `geodesic_rbf_basis`, `heat_kernel_wavelet_basis`, `chebyshev_filter_basis`,
     and `spatial_basis` (`neurospatial.ops.basis`), for consistency with the
     rest of the library. Both `int` seeds and `np.random.Generator` instances
     are still accepted. Pre-1.0: no alias is kept — update call sites to `rng=`.
   ```

### Task group B — Unify the events `window` contract on `(start, end)`

Today three sibling regressors express "window" three incompatible ways:
`time_to_nearest_event` uses `max_time` (scalar, symmetric clip),
`event_count_in_window` uses `window: tuple[float, float]` (positional),
`event_indicator` uses `window: float` (scalar half-width). Standardize on the
explicit `(start, end)` tuple, keyword-only, named `window`, everywhere.

1. **`event_indicator`** — [events/regressors.py:347-484](../../../../src/neurospatial/events/regressors.py).
   New signature:
   ```python
   def event_indicator(
       sample_times: NDArray[np.float64],
       event_times: NDArray[np.float64],
       *,
       window: tuple[float, float] = (0.0, 0.0),
   ) -> NDArray[np.bool_]:
   ```
   Body: replace the symmetric `window_starts = sample_times - window` /
   `window_ends = sample_times + window` (lines 473-474) with
   `start, end = window; window_starts = sample_times + start; window_ends = sample_times + end`.
   Replace the `if window < 0:` half-width validation (453-458) with the
   `if start > end:` check used by `event_count_in_window` (regressors.py:312-317).
   Update the docstring (scalar half-width → `(start, end)`) and the inline examples
   (`window=0.5` → `window=(-0.5, 0.5)`; `window=0.0` impulse → `window=(0.0, 0.0)`).

2. **`event_count_in_window`** — [events/regressors.py:211-215](../../../../src/neurospatial/events/regressors.py).
   Make `window` keyword-only to match the others:
   ```python
   def event_count_in_window(
       sample_times: NDArray[np.float64],
       event_times: NDArray[np.float64],
       *,
       window: tuple[float, float],
   ) -> NDArray[np.int64]:
   ```
   No body change (it already uses `(start, end)`).

3. **`time_to_nearest_event`** — [events/regressors.py:29-34](../../../../src/neurospatial/events/regressors.py).
   Leave `max_time` as-is. **Rationale (record in the docstring `Notes`):**
   `time_to_nearest_event` returns a *continuous signed time-to-event regressor*,
   not a windowed count/indicator — its `max_time` is a symmetric clip, semantically
   distinct from a `(start, end)` masking window, so forcing a tuple here would be a
   false unification. The `(start, end)` `window` contract applies to the two
   *windowing* regressors (`event_count_in_window`, `event_indicator`); document that
   distinction so the asymmetry is intentional and discoverable.

4. **Docs** — fix the `event_indicator` example in
   [.claude/QUICKSTART.md:863-866](../../../../.claude/QUICKSTART.md) and
   [src/neurospatial/events/__init__.py:58-64](../../../../src/neurospatial/events/__init__.py)
   (module docstring), which pass a tuple to the old scalar param. After this task the
   tuple form is correct, so the examples become runnable — verify and adjust the
   surrounding prose. (The broader QUICKSTART example sweep is Phase 23; this task only
   fixes the snippet whose contract this phase changes.)

5. **CHANGELOG** (`### API changes`):
   ```
   - Unified the time-window argument across the events GLM regressors:
     `event_indicator` and `event_count_in_window` now both take a keyword-only
     `window=(start, end)` tuple (relative seconds). `event_indicator` previously
     took a scalar symmetric half-width; rewrite `window=w` as `window=(-w, w)`.
     `event_count_in_window`'s `window` is now keyword-only. `time_to_nearest_event`
     keeps its distinct scalar `max_time` (it is a continuous regressor, not a
     windowing function).
   ```

### Task group C — Single-source the object-vector-cell criterion + view-cell parity

1. **Make the free `is_object_vector_cell` delegate to the result method** —
   [encoding/egocentric.py:1869-1979](../../../../src/neurospatial/encoding/egocentric.py).
   The free function currently classifies with `object_vector_score` + `min_peak_rate`
   (via `_is_object_vector_cell_from_tuning`), which **cannot agree** with
   `EgocentricRateResult.is_object_vector_cell`'s spatial-information rule. Replace the
   body's classification step so the result method is the single source of truth.
   New signature (drops `score_threshold`/`min_peak_rate`, adds `min_info`):
   ```python
   def is_object_vector_cell(
       env: Environment,
       spike_times: NDArray[np.float64],
       times: NDArray[np.float64],
       positions: NDArray[np.float64],
       headings: NDArray[np.float64],
       object_positions: NDArray[np.float64],
       *,
       distance_range: tuple[float, float] = (0.0, 50.0),
       n_distance_bins: int = 10,
       n_direction_bins: int = 12,
       metric: Literal["euclidean", "geodesic"] = "euclidean",
       min_info: float = 0.3,
   ) -> bool:
   ```
   New body:
   ```python
   try:
       result = compute_egocentric_rate(
           env, spike_times, times, positions, headings, object_positions,
           distance_range=distance_range,
           n_distance_bins=n_distance_bins,
           n_direction_bins=n_direction_bins,
           metric=metric,
       )
   except (ValueError, RuntimeError):
       return False
   return result.is_object_vector_cell(min_info=min_info)
   ```
   Note: the `except (ValueError, RuntimeError): return False` swallow is a separate
   correctness concern owned by **Phase 5** (encoding-directional silent-False); do
   **not** alter that behavior here — keep the existing try/except exactly. Update the
   free function's docstring to describe the spatial-information criterion and
   cross-reference `EgocentricRateResult.is_object_vector_cell`. Leave
   `object_vector_score` and `_is_object_vector_cell_from_tuning` in place (they remain
   the public score primitive and an internal helper); just stop the free classifier
   from using a divergent rule. If `_is_object_vector_cell_from_tuning` ends up with no
   remaining callers after this edit, remove it (grep first).
   **CHANGELOG** (`### API changes`):
   ```
   - `is_object_vector_cell` (free function in `neurospatial.encoding.egocentric`)
     now delegates to `EgocentricRateResult.is_object_vector_cell`, using the
     egocentric-spatial-information criterion. It replaces the old
     `score_threshold`/`min_peak_rate` parameters with a single `min_info`
     (default 0.3, matching the result method), so the quick-check and the
     result-object classification can no longer disagree.
   ```

2. **Align `view_distance`/`gaze_model` order across siblings** —
   [encoding/view.py:1474-1476](../../../../src/neurospatial/encoding/view.py).
   `compute_view_rate` is the primary function and declares `gaze_model` first
   (867-869); reorder the convenience `is_spatial_view_cell` to match (**`gaze_model`
   before `view_distance`**):
   ```python
   def is_spatial_view_cell(
       env, spike_times, times, positions, headings,
       *,
       gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
       view_distance: float = 10.0,
       smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
       bandwidth: float = 5.0,
       min_info: float = 0.5,
   ) -> bool:
   ```
   Both are keyword-only, so this is source-compatible for keyword callers. Also fix
   the CLAUDE.md "Most Common Patterns" §9 snippet, which shows `view_distance=` before
   `gaze_model=` for `compute_view_rate` — reorder it to `gaze_model=` first to match
   the canonical order. **CHANGELOG** (`### API changes`):
   ```
   - Reordered the keyword-only `gaze_model`/`view_distance` parameters in
     `is_spatial_view_cell` to match `compute_view_rate` (`gaze_model` first).
     Keyword callers are unaffected.
   ```

3. **Unify batch-classifier method naming** — [encoding/spatial.py:1042](../../../../src/neurospatial/encoding/spatial.py).
   Rename `SpatialRatesResult.classify` → `SpatialRatesResult.detect_cell_types`
   to match the `detect_*` sibling convention (`detect_ovcs`, `detect_view_cells`).
   Update its docstring `See Also`, the `to_dataframe` reference to `classify()`
   ([spatial.py:~1152](../../../../src/neurospatial/encoding/spatial.py), "labels from
   `classify()`") and any `result.classify(` call in QUICKSTART/API_REFERENCE (grep).
   **CHANGELOG** (`### API changes`):
   ```
   - Renamed `SpatialRatesResult.classify` to `SpatialRatesResult.detect_cell_types`
     for naming parity with `EgocentricRatesResult.detect_ovcs` and
     `ViewRatesResult.detect_view_cells`.
   ```

### Task group D — Keyword-only `*` / env-first / positional-algorithm-params sweep

For each function below, insert a `*` so every defaulted *algorithm* parameter is
keyword-only, and (where noted) reorder to env-first / animal-state-before-params.
Keyword callers are unaffected; positional callers must update (CHANGELOG covers the
public ones).

1. **`ops/normalize.py:161`** — `combine_fields`:
   ```python
   def combine_fields(
       fields: Sequence[NDArray[np.float64]],
       *,
       weights: Sequence[float] | None = None,
       mode: Literal["mean", "max", "min"] = "mean",
   ) -> NDArray[np.float64]:
   ```

2. **`ops/smoothing.py:70`** — `compute_diffusion_kernels`:
   ```python
   def compute_diffusion_kernels(
       graph: nx.Graph,
       bandwidth_sigma: float,
       *,
       bin_sizes: NDArray | None = None,
       mode: Literal["transition", "density"] = "transition",
   ) -> NDArray[np.float64]:
   ```

3. **`ops/distance.py:55,410`** — make `weight` keyword-only:
   ```python
   def geodesic_distance_matrix(G: nx.Graph, n_states: int, *, weight: str = "distance") -> ...
   def pairwise_distances(G: nx.Graph, nodes: ..., *, weight: str = "distance") -> ...
   ```

4. **`decoding/trajectory.py:195`** — `fit_isotonic_trajectory`: make the unused `env`
   keyword-only with default `None` so callers no longer pass a placeholder positionally:
   ```python
   def fit_isotonic_trajectory(
       posterior: NDArray[np.float64],
       times: NDArray[np.float64],
       *,
       env: Environment | None = None,
       increasing: bool | None = None,
       method: Literal["map", "expected"] = "expected",
   ) -> IsotonicFitResult:
   ```
   Keep the `del env` at line 275 and the docstring note that it is consulted by no
   code path (parity placeholder). **CHANGELOG** (`### API changes`):
   ```
   - `fit_isotonic_trajectory` no longer takes a mandatory leading `env`; `env`
     is now an optional keyword-only argument (it was unused). Drop the `None`
     you previously passed as the first positional argument.
   ```

5. **`decoding/metrics.py:29,228`** —
   - `decoding_error`: keep `env` positional but document that env-first is the
     canonical order *for env-consuming functions*; here `env` is optional and the
     two position arrays are the primary inputs. **Do not move `env` to first** —
     `decoded_positions`/`actual_positions` are the data and `env` only gates the
     geodesic metric. Instead make `env` keyword-only to remove the env-3rd-positional
     ambiguity:
     ```python
     def decoding_error(
         decoded_positions: NDArray[np.float64],
         actual_positions: NDArray[np.float64],
         *,
         env: Environment | None = None,
         metric: Literal["euclidean", "geodesic"] = "euclidean",
     ) -> NDArray[np.float64]:
     ```
   - `confusion_matrix`: rename `summary_method` → `method` to match
     `fit_isotonic_trajectory`'s `method` for the identical `{"map","expected"}`
     choice. (Note `fit_linear_trajectory.method` is a *different* literal set,
     `{"map","sample"}` — the shared name is the convention, not the value set.)
     Update the body's
     validation/branch references (`summary_method` at lines 305-345) and docstring.
     ```python
     def confusion_matrix(
         env: Environment,
         posterior: NDArray[np.float64],
         actual_bins: NDArray[np.int64],
         *,
         method: Literal["map", "expected"] = "map",
     ) -> NDArray[np.float64]:
     ```
   **CHANGELOG** (`### API changes`):
   ```
   - `decoding_error`'s `env` is now keyword-only (was an optional 3rd positional).
   - `confusion_matrix`'s `summary_method` argument was renamed to `method`,
     matching `fit_isotonic_trajectory`.
   ```

6. **`behavior/vte.py:593`** — `compute_vte_session`: move `env` first to match
   `compute_decision_analysis(env, positions, times, ...)`:
   ```python
   def compute_vte_session(
       env: Environment,
       positions: NDArray[np.float64],
       times: NDArray[np.float64],
       *,
       decision_region: str,
       trials: list[Trial],
       window_duration: float = 1.0,
       min_speed: float = 5.0,
       alpha: float = 0.5,
       vte_threshold: float = 0.5,
   ) -> VTESessionResult:
   ```
   Update the internal callee order (the function forwards `positions, times` into
   `compute_vte_trial` — verify those calls still pass by keyword/position correctly).
   **CHANGELOG** (`### API changes`):
   ```
   - `compute_vte_session` is now env-first — `(env, positions, times, *, ...)` —
     matching `compute_decision_analysis`. Update positional callers.
   ```

7. **`behavior/segmentation.py:1749-1755`** — `detect_goal_directed_runs`: change the
   missing-region `raise KeyError(...)` (1752) to `raise ValueError(...)` so it matches
   sibling region-validation (e.g. `detect_goal_directed_runs`'s neighbors raise
   `ValueError`). Keep the message text. *(This is an exception-**type** consistency
   fix, not new validation — the branch already exists.)* **CHANGELOG** (`### API changes`):
   ```
   - `detect_goal_directed_runs` now raises `ValueError` (was `KeyError`) when the
     requested region is absent, matching sibling behavior validators.
   ```

8. **`layout/engines/triangular_mesh.py:77`** — `TriangularMeshLayout.build`: add the
   `@capture_build_params` decorator (import from `neurospatial.layout.base` as
   `masked_grid.py:8` / `image_mask.py:8` do) and make `point_spacing` keyword-only to
   match sibling `build` signatures:
   ```python
   @capture_build_params
   def build(self, boundary_polygon: Polygon, *, point_spacing: float) -> None:
   ```
   Verify the factory that calls this `build` (grep `TriangularMeshLayout(` / `.build(`)
   passes `point_spacing=` by keyword; update if positional. (The `image_mask`
   `bin_size`-vs-`pixel_size` naming item from the SUMMARY is **not** in this phase —
   see "Deliberately not in this phase".)

9. **`annotation/io.py:13,72`**, **`_boundary_inference.py:74`**, **`converters.py:21`** —
   move pre-`*` non-primary params behind the separator:
   - `regions_from_labelme(json_path, *, calibration=None, label_key="label", points_key="points")`
   - `regions_from_cvat(xml_path, *, calibration=None)` (add a `*`)
   - `boundary_from_positions(positions, *, method=None, config=None, buffer_fraction=None, simplify_fraction=None, **method_kwargs)` — move the `method` algorithm choice behind `*`.
   - `shapes_to_regions(shapes_data, names, region_types, *, calibration=None, simplify_tolerance=None, multiple_boundaries="last", validate=True, min_area=1e-6)`.
   **CHANGELOG** (`### API changes`):
   ```
   - `calibration` (and `method`/`simplify_tolerance` where present) are now
     keyword-only in `regions_from_labelme`, `regions_from_cvat`,
     `boundary_from_positions`, and `shapes_to_regions`.
   ```

10. **`simulation/trajectory.py:460,598`** — add `*` after the leading required args to
    match `simulate_trajectory_ou`:
    ```python
    def simulate_trajectory_sinusoidal(env, duration, *, sampling_frequency=500.0, speed=10.0, period=None, pause_duration=0.0, pause_at_peaks=True, seed=None) -> ...
    def simulate_trajectory_laps(env, n_laps, *, speed_mean=0.1, speed_std=0.02, outbound_path=None, inbound_path=None, pause_duration=0.5, sampling_frequency=500.0, seed=None, return_metadata=False) -> ...
    ```

11. **`simulation/spikes.py:173`** — `generate_population_spikes`: move `headings`
    (animal state) ahead of the algorithm params, keyword-only with default:
    ```python
    def generate_population_spikes(
        models: list[NeuralModel],
        positions: NDArray[np.float64],
        times: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
        refractory_period: float = 0.002,
        seed: int | None = None,
        show_progress: bool = True,
    ) -> list[NDArray[np.float64]]:
    ```
    All keyword-only, so keyword callers unaffected; reorder is cosmetic but aligns with
    the canonical "animal state before algorithm params" rule.

12. **`regions/core.py:562`** — rename `region_center`'s parameter `region_name` → `name`
    to match sibling `area(name)`:
    ```python
    def region_center(self, name: str) -> NDArray[np.float64] | None:
    ```
    Update the body (`region_name` → `name` at lines 582-585) and the docstring
    `Parameters`/`Raises`. **Do not** touch the `| None` return-type behavior claim —
    that (it never returns `None`) is a correctness item owned by Phase 11.
    **CHANGELOG** (`### API changes`):
    ```
    - `Regions.region_center`'s parameter was renamed from `region_name` to `name`,
      matching `Regions.area`. Update keyword callers.
    ```

### Task group E — Protocol-stub refresh

Bring the `Protocol` stubs into exact agreement with the real `Environment` methods
so type-checkers and IDEs report the true signatures. No runtime behavior.

1. **`occupancy`** — [_protocols.py:378-389](../../../../src/neurospatial/environment/_protocols.py):
   change `max_gap: float | None = None` (385) → `max_gap: float | None = 0.5` to match
   [trajectory.py:100](../../../../src/neurospatial/environment/trajectory.py). Update
   the docstring default mention if present.

2. **`animate_fields`** — [_protocols.py:718-811](../../../../src/neurospatial/environment/_protocols.py):
   rewrite the stub to match [visualization.py:562-591](../../../../src/neurospatial/environment/visualization.py):
   - Make `frame_times: NDArray[np.float64]` the **first keyword-only** parameter
     (required, no default).
   - Remove `fps: int = 30`.
   - Add `speed: float = 1.0`.
   - Drop `frame_times: ... = None` from its old position near the end.
   - Keep the remaining params in the real method's order. Update the docstring to
     drop the `fps` block and add `speed`/required-`frame_times`.

### Task group F — Unit docstrings

1. **`heading_direction_labels`** — [behavior/navigation.py:1102-1103](../../../../src/neurospatial/behavior/navigation.py):
   expand the `min_speed : float, default=5.0` docstring from "Minimum speed threshold."
   to state the unit and the assumption, matching sibling unit notes:
   ```
   min_speed : float, default=5.0
       Minimum speed threshold below which a sample is labeled "stationary".
       In the same spatial-units-per-second as `positions`/`times` (e.g. cm/s
       if positions are in cm). The 5.0 default assumes cm/s.
   ```
   Docstring-only; no signature or behavior change, no CHANGELOG entry.

### Task group G — CHANGELOG assembly

Add a single `### API changes` subsection under the existing `## Unreleased` heading
in [CHANGELOG.md](../../../../CHANGELOG.md) (currently has `### Bug fixes` and
`### Documentation`), collecting every entry quoted in tasks A.3, B.5, C.1, C.2, C.3,
D.4, D.5, D.6, D.7, D.9, D.12. Keep them as bullet points in task order.

## Deliberately not in this phase

- **The `except (ValueError, RuntimeError): return False` silent-False swallow** in the
  free `is_object_vector_cell` (and the sibling `is_head_direction_cell` /
  `has_phase_precession` / `is_spatial_view_cell` swallows) — that is a *correctness*
  bug (Theme 1) owned by **Phase 5**. Task C.1 preserves the existing try/except verbatim.
- **`region_center` returning `| None` it never produces** — correctness item, **Phase 11**.
  This phase only renames the parameter.
- **`detect_goal_directed_runs` / `-1` bin wraparound, occupancy index mismatch,
  NaN-heading binning** and every other *behavior* bug listed in SUMMARY for these files
  — owned by the per-subsystem correctness phases (2, 5, 6, 7, 8). This phase changes
  signatures/names/docstrings only; the exception-type swap in D.7 is the sole change to
  control flow and it changes only the *class* of an already-raised error.
- **`ImageMaskLayout.build` `bin_size` vs factory `pixel_size` naming** — that is a
  layout-engine naming item bundled with the `image_mask` axis-order correctness work in
  **Phase 10**; not touched here to avoid double-editing `image_mask.py`.
- **NWB writer `description` placement** (io Theme 7) — **Phase 9/15** territory; this
  phase does not touch `io/nwb/`.
- **Adding NaN/Inf validation to the events regressors** — **Phase 7**. Task group B only
  changes the *type* of the window argument.
- **Renaming `to_dataframe`/`plot`/`summary` or introducing a result mixin** — **Phase 17**.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_detect_assemblies_rng_keyword` | `detect_assemblies(counts, rng=0)` runs and two calls with `rng=0` give identical patterns; `rng=1` differs. Calling with the removed `random_state=` raises `TypeError`. |
| `test_basis_rng_keyword_deterministic` | `geodesic_rbf_basis(env, n_centers=8, rng=0)` and `select_basis_centers(env, 8, rng=0)` are reproducible across two calls; passing `method=`/`rng=` positionally now raises `TypeError` (keyword-only). |
| `test_event_indicator_window_tuple` | `event_indicator(samples, events, window=(-0.5, 0.5))` matches the old `window=0.5` result on a fixture; `window=(0.0, 0.0)` reproduces exact-match impulse; `start>end` raises `ValueError`. |
| `test_event_count_in_window_keyword_only` | `event_count_in_window(samples, events, window=(-5.0, 0.0))` still equals the documented `array([0,2,3,1])`; passing `window` positionally raises `TypeError`. |
| `test_ovc_free_function_agrees_with_result_method` | For a fixture trajectory+spikes, `is_object_vector_cell(env, ..., min_info=m)` equals `compute_egocentric_rate(env, ...).is_object_vector_cell(min_info=m)` for several `m` (0.1, 0.3, 0.6) — they can no longer disagree. Removed `score_threshold=`/`min_peak_rate=` kwargs raise `TypeError`. |
| `test_is_spatial_view_cell_param_order` | `is_spatial_view_cell(env, ..., gaze_model="ray_cast", view_distance=15.0)` runs; signature inspection shows `gaze_model` precedes `view_distance`, matching `compute_view_rate`. |
| `test_spatialrates_detect_cell_types_renamed` | `SpatialRatesResult.detect_cell_types()` returns the labels the old `.classify()` returned (pin on a fixture); `.classify` attribute no longer exists. |
| `test_fit_isotonic_trajectory_env_optional` | `fit_isotonic_trajectory(posterior, times)` runs with no `env`; result equals the old `fit_isotonic_trajectory(None, posterior, times)`. |
| `test_confusion_matrix_method_renamed` | `confusion_matrix(env, post, bins, method="map")` equals old `summary_method="map"` output; `summary_method=` raises `TypeError`. |
| `test_compute_vte_session_env_first` | `compute_vte_session(env, positions, times, decision_region=..., trials=...)` runs; first positional param is `env` (signature inspection). |
| `test_detect_goal_directed_runs_raises_valueerror` | Missing region raises `ValueError` (not `KeyError`). |
| `test_triangular_mesh_build_capture_params` | After `TriangularMeshLayout().build(poly, point_spacing=2.0)`, the captured build params include `point_spacing`; positional `point_spacing` raises `TypeError`. |
| `test_region_center_name_kwarg` | `regions.region_center(name="goal")` works; `region_name=` raises `TypeError`. |
| `test_protocol_stub_matches_impl` | `inspect.signature` of `EnvironmentProtocol.animate_fields` has required keyword-only `frame_times`, no `fps`, and a `speed` param; `EnvironmentProtocol.occupancy` has `max_gap` default `0.5` — each equal to the real `Environment` method's corresponding parameter. |

Mark none slow; all are fast unit/signature tests. Use `inspect.signature(...).parameters`
for keyword-only / ordering / default assertions rather than brittle string matching.

## Fixtures

- Reuse existing `conftest.py` factories: a small `Environment.from_samples` 2D env, a
  short synthetic trajectory (`times`, `positions`, `headings`), a Poisson spike train,
  and a population spike-count matrix. No new checked-in data.
- For the OVC-agreement test, synthesize a trajectory that orbits one object so the
  egocentric rate map is non-degenerate (so both classifiers return a meaningful, equal
  boolean across `min_info` values) — a `conftest` fixture `ovc_session` returning
  `(env, spike_times, times, positions, headings, object_positions)`.
- For `test_protocol_stub_matches_impl`, compare `inspect.signature` of the protocol
  method against the bound `Environment` method on the small env fixture — no rendering,
  no napari import (use `dry_run` semantics only if the method is actually invoked, which
  this test does not require).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:

- Every task A–G is implemented with the exact signatures quoted; no signature left
  half-renamed (e.g. body still references `random_state`/`summary_method`/`region_name`).
- **No behavior change** beyond the two intended ones: (a) the OVC free function now uses
  the spatial-information criterion (Task C.1) and (b) `detect_goal_directed_runs` raises
  `ValueError` instead of `KeyError` (Task D.7). A reviewer should diff each touched
  function body and confirm only signatures/forwarded-keywords/docstrings moved otherwise.
  Run `scientific-code-change-audit` lens on C.1 specifically (it is the one change that
  alters which neurons get classified).
- The "Deliberately not in this phase" list is honored — no validation added to events
  regressors, no `io/nwb/` edits, no `image_mask` naming change, no silent-False swallow
  removed.
- Every public rename in the CHANGELOG `### API changes` block corresponds to a real
  signature change, and vice-versa (no orphan CHANGELOG line, no undocumented rename).
- Validation-slice tests pass and use `inspect.signature` (not tautologies); `TypeError`
  assertions actually pin the *removed* keyword, not a typo.
- `uv run pytest` is green; `uv run ruff check . && uv run ruff format .` and
  `uv run mypy src/neurospatial/` pass (the protocol-stub refresh must keep mypy happy
  against the real `Environment`).
- Docstrings, test names, and module names do not reference this plan or "Phase 22".
- `_is_object_vector_cell_from_tuning` is removed if and only if it has no remaining
  callers after Task C.1 (grep-verified); no orphan left behind.
- The QUICKSTART/CLAUDE.md snippets touched by Tasks B.4 and C.2 are updated, not deferred.
