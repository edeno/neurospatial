# Phase 5a — `label_cell_types` fix and frame-explicit object-vector analysis

**Requires:** Phases 4b, 4c and 4d merged, and a passing repeated researcher-workflow checkpoint ([overview → Rollout Strategy](overview.md#rollout-strategy)). The post-4c checkpoint on 2026-10-07 is held for the decoder-tutorial coordinate correction.

[← back to PLAN.md](PLAN.md) · [executing](executing.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Read [executing.md](executing.md) first: branch and PR workflow, definition of done, CHANGELOG-per-commit, and what to do when the plan and reality disagree. This file holds only what is specific to Phase 5a.

This phase:

- fixes `label_cell_types`, which labels random spikes "border" against its own docstring;
- adds the allocentric object-vector map that Høydal et al. (2019) describe, as its own functions;
- gives each object-vector frame its own functions with truthful signatures: the allocentric functions take no `headings`, and the egocentric functions require them (decision 6; [error contract](shared-contracts.md#error-message-contract), rule 1);
- renames the object-vector result classes to frame-neutral names that record their frame;
- makes the simulator allocentric by default.

Phase 5b builds shuffle significance and the single predicate contract on top of these names. This phase leaves every threshold criterion, default value and comparison operator as it is on `main`.

**Inputs to read first** (line numbers verified on `main` at `da631a47`; earlier phases shift them, so search by name):

- [src/neurospatial/encoding/spatial.py](../../../../src/neurospatial/encoding/spatial.py) — `SpatialRatesResult.label_cell_types` :1959 (labeling at :2089-2100).
- [src/neurospatial/encoding/egocentric.py](../../../../src/neurospatial/encoding/egocentric.py) — Høydal citations :60, :436, :476, :1481, :1819; `EgocentricRateResult` and its method `is_object_vector_cell` :421; `egocentric_spatial_information`; `compute_egocentric_rate` :1311 / `_rates` :1600; free `is_object_vector_cell` :2141 (takes `headings` positionally on `main`); `plot_object_vector_tuning` :2248 (theta setup :2322-2323).
- [src/neurospatial/encoding/_egocentric_binning.py:163](../../../../src/neurospatial/encoding/_egocentric_binning.py) — `_compute_egocentric_coords`; the egocentric bearing at :244. Callers: `compute_egocentric_occupancy` :375, `bin_egocentric_spike_train` :534, `bin_egocentric_spike_trains` :700.
- [src/neurospatial/simulation/models/object_vector_cells.py](../../../../src/neurospatial/simulation/models/object_vector_cells.py) — egocentric-only direction tuning (:444), Høydal citation :61.
- [src/neurospatial/ops/egocentric.py:65](../../../../src/neurospatial/ops/egocentric.py) — Høydal cited for egocentric transforms.
- **Files earlier phases already changed** (search by name):
  - `encoding/egocentric.py`: Phase 1 Task 7 (spike-group input to `compute_egocentric_rates`, `resolve_unit_ids(..., input_ids=)`), Phase 2a Task 2 (egocentric polar plot orientation; keep it), Phase 3b (`max_gap`/`epochs`/`spike_window` on every function and predicate, population-silence warning).
  - `encoding/_egocentric_binning.py`: Phase 3b replaced the inline occupancy blocks and `_bin_single_neuron` with `start_allocated_occupancy` and `count_spikes_by_frame`. Thread `headings: NDArray | None` through those call sites.
  - `ops/egocentric.py`: Phase 2b Task 3 (heading interpolation) and Phase 3e changed `heading_from_velocity`; this phase edits only the module docstring citation.
  - Phase 4b's `tests/docs/test_flagship_docstrings.py` runs every `FLAGSHIP` docstring example and forbids `+SKIP`; `tests/docs/test_docstring_sections.py` requires Parameters, Returns and Examples sections and every signature parameter documented for root callables and `FLAGSHIP`. Use Phase 4a's `_format_error` for new messages.

**Contracts referenced:**

- [Overview decision 6](overview.md#settled-design-decisions) — object-vector cells are allocentric by default.
- [Error-message contract](shared-contracts.md#error-message-contract), rule 1 — an argument needed in one mode and forbidden in another means **separate functions**. So no object-vector function in this phase takes `direction_frame=` or `headings=None`; the allocentric and egocentric frames are different functions. Results may carry a `direction_frame` attribute, because a result reports what was computed rather than choosing a mode.
- [Input conventions](shared-contracts.md#input-conventions) — the allocentric raw form is `(env, spike_times, times, positions, object_positions, *, ...)`; the egocentric raw form keeps `(env, spike_times, times, positions, headings, object_positions, *, ...)`.

## Evidence (measured on `main`, `da631a47`)

**`label_cell_types`.** The `ou_trajectory` fixture (Fixtures, seed 0), 20 independent homogeneous 0.5 Hz Poisson units (spikes seed 1), `bin_size=5`:

| Duration | Method | Labels on `main` | "border" units: spatial info (bits/spike) | border score |
| --- | --- | --- | --- | --- |
| 10 min | `diffusion_kde` (bandwidth 5) | 19 border, 1 unclassified | 0.082–0.132 (all < 0.5) | 0.54–0.79 |
| 10 min | `binned` | 9 border, 11 unclassified | 0.099–0.143 (all < 0.5) | 0.55–0.88 |
| 2 min | `diffusion_kde` | 20 border | 0.42–0.69 (15 of 20 ≥ 0.5) | 0.72–1.00 |

**Frame-specific tuning.** A prototype allocentric map recovered frame-specific tuning. For a cell whose field sits 20 cm East of the object, the allocentric map's information was 2.86 bits/spike and the egocentric map's 1.63. For an egocentric cell (object ahead), the egocentric map gave 2.94 and the allocentric map 1.77. Each frame's peak direction was within one bin (0.26 rad) of the truth. These are prototype numbers, not `main` numbers: `main` has no allocentric map. Record the observed values in the commit body.

**Signatures on `main`** (`inspect.signature`):

```text
is_object_vector_cell(env, spike_times, times, positions, headings, object_positions, *, distance_range=(0.0, 50.0), n_distance_bins=10, n_direction_bins=12, metric='euclidean', min_info=0.3)
EgocentricRateResult.is_object_vector_cell(self, min_info=0.3)
EgocentricRatesResult.classify(self, *, min_info=0.3)
```

## Tasks

**5a.1 `label_cell_types` applies its documented spatial-information gate (bug fix).**

- **Bug (verified, Evidence above).** On `main`, `SpatialRatesResult.label_cell_types` (`spatial.py:2089-2100`) requires `spatial_info >= min_spatial_info` only for `"place"`. Its `min_spatial_info` parameter says "Neurons below this are labeled 'unclassified'", and its Returns say grid and border cells "pass spatial info threshold".
- **Regression test first** (`test_label_cell_types_noise_is_unclassified`), run against the unmodified code; record the 19 "border" labels in the commit body.
- **Fix.** Gate every label on spatial information, then apply the existing precedence:

  ```python
  tuned = spatial_info >= min_spatial_info  # NaN -> False
  labels = np.full(n_neurons, "unclassified", dtype="<U14")
  labels[tuned] = "place"
  labels[tuned & (border_scores_arr >= min_border_score)] = "border"
  labels[tuned & (grid_scores_arr >= min_grid_score)] = "grid"
  ```

  NaN scores compare `False`, so the explicit `~np.isnan` terms go. Rewrite the Notes "Classification priority" list so each of grid, border and place says "and spatial_info >= min_spatial_info".
- **Expected after the fix.** At 10 min the 20 noise units are all "unclassified". At 2 min the 15 units whose plug-in information is ≥ 0.5 stay "border". That is the threshold bias Phase 5b documents, not a remaining bug.
- `label_cell_types` stays threshold-only. Its keyword defaults and signature change in Phase 5b (keyword-only, resolved from `PLACE_GRID_BORDER_THRESHOLDS`), not here.
- CHANGELOG `### Fixed`: the gate, with the 19/20 → 0/20 "border" numbers.

**5a.2 Allocentric object-vector map and frame-neutral results.**

- **Choice:** separate functions, not a `direction_frame=` keyword. The allocentric map needs no headings, so `(env, spike_times, times, positions, object_positions)` follows the input conventions without a `None` slot.
- **The literature:** Høydal et al. 2019 (Nature 568:400, PMID 30944479) describe allocentric object vectors in MEC. Wang et al. 2018 (Science 362:945, PMID 30467169) describe egocentric bearing to items in LEC. Both maps are needed.
- **New public names in `encoding.__all__`:** `compute_object_vector_rate` and `compute_object_vector_rates`, with the same keywords as `compute_egocentric_rate(s)` minus `headings` (including Phase 3b's `max_gap`, `epochs`, `spike_window` and Phase 1's `unit_ids=` on the plural).
- **Rename (decision 1, no aliases):**
  - `EgocentricRateResult` → `ObjectVectorRateResult`;
  - `EgocentricRatesResult` → `ObjectVectorRatesResult`;
  - `egocentric_spatial_information()` → `spatial_information()`.

  Both `compute_egocentric_rate(s)` and `compute_object_vector_rate(s)` return these classes.
- **New required field** `direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)`. It has no default, so a result cannot be mislabelled. Indexing a plural result (`rates[i]`) passes it to the child. It appears in `summary()`.
- **Direction convention:** "direction from the animal to the object". For `allocentric`, 0 = East and +π/2 = North. For `egocentric`, 0 = ahead and +π/2 = left. So `egocentric = wrap(allocentric − heading)`. Document that the object-centred vector (object → animal) has direction `preferred_direction() + π`.
- **Binning core:**
  - rename `_compute_egocentric_coords` to `_compute_object_coords(positions, headings: NDArray | None, object_positions, *, metric, env)`. This private helper is shared by both public families; the `None` is internal plumbing, never a public signature;
  - `headings is None` means allocentric: `bearings_all = np.arctan2(dy, dx)` of `object_positions[None] - positions[:, None]`;
  - thread `headings: NDArray | None` through `compute_egocentric_occupancy`, `bin_egocentric_spike_train(s)` and a shared private `_object_vector_rate(...)` used by both public families. Check `encoding.__all__`: if any of those binning helpers is public, give the allocentric case a separate public wrapper rather than a public `headings=None`.
- **Interval mask.** Factor the family's Phase 3b mask construction into a private `_object_vector_interval_mask(...)` that both frames call. Phase 5b's significance functions reuse it, so the shuffle windows are exactly the intervals the observed map used.

**5a.3 Frame-specific free predicates.** Split `main`'s free `is_object_vector_cell` by frame. Each keeps `main`'s threshold behavior (`min_info=0.3`, the same comparison, and its `try/except → False`); Phase 5b changes those for every predicate at once.

```text
is_object_vector_cell(env, spike_times, times, positions, object_positions, *, <compute_object_vector_rate keywords>, min_info=0.3)
is_egocentric_object_vector_cell(env, spike_times, times, positions, headings, object_positions, *, <compute_egocentric_rate keywords>, min_info=0.3)
```

- `is_object_vector_cell` is allocentric, matching the cell type's defining paper. It has **no** `headings` and no `direction_frame` parameter.
- `is_egocentric_object_vector_cell` is new in `encoding.__all__`. Its `headings` is an ordinary required positional argument.
- Each docstring names the other function in See Also and in its summary paragraph ("for egocentric bearing to the object, use `is_egocentric_object_vector_cell`"), cites its frame's paper, and states that the threshold verdict also equals `compute_*_rate(...).is_object_vector_cell(...)`.
- The method `ObjectVectorRateResult.is_object_vector_cell` and `ObjectVectorRatesResult.classify` work on either frame; their docstrings say they test tuning in `result.direction_frame`.

**5a.4 Plot and citations.**

- **`plot_object_vector_tuning`** reads `result.direction_frame`. Allocentric uses `set_theta_zero_location("E")` and `set_theta_direction(1)`, with the axis label "direction to object (allocentric, 0 = East)". The egocentric branch is Phase 2a Task 2's fix; do not rewrite it.
- **Citations:**
  - module docstrings (`egocentric.py:1-64`, `ops/egocentric.py:65`, the simulator :61): cite Høydal 2019 for allocentric and Wang 2018 for egocentric. Keep Deshmukh & Knierim 2011 as LEC background. Cite Alexander et al. 2020 (Sci Adv 6:eaaz2322, PMID 32128423) only as related egocentric *boundary*-vector coding;
  - `ops/egocentric.py`: replace the Høydal citation.
  - The "How was 0.3 chosen? … (Hoydal et al., 2019)" justification at `egocentric.py:436` and its twin in the free function are Phase 5b's (bias documentation); leave them.

**5a.5 Simulation.** `ObjectVectorCellModel(..., direction_frame: Literal["allocentric", "egocentric"] = "allocentric")`. This is a model-configuration keyword on a constructor; it selects what the model *is*, not which arguments a call needs.

- **Allocentric** directional tuning uses `arctan2(object − position)` and needs no headings: `firing_rate(positions)` works.
- **Egocentric** keeps `compute_egocentric_bearing` and its `headings` requirement when `preferred_direction` is set (on `main`, `simulation/models/object_vector_cells.py:351` and :444). Raise a missing `headings` as a contract error naming `direction_frame="allocentric"` as the alternative.
- `firing_rate(positions, times=None, headings=None)` is the shared `NeuralModel` interface (`simulation/models/base.py:111`) that `generate_population_spikes` calls on every model. Changing that protocol is out of scope.
- **`ground_truth`** gains `"direction_frame"`.
- The default matches the cell type's definition and `is_object_vector_cell` (decision 6; no users). Update every internal construction that relied on egocentric tuning (`simulation/examples.py`, `examples/24_object_vector_cells.py`) to pass `direction_frame="egocentric"` explicitly or switch to the allocentric analysis.

**5a.6 Documentation** (each is part of this PR; each commit adds its own CHANGELOG bullet per [executing.md](executing.md), and this task checks they are all present):

- CLAUDE.md:
  - pattern 8 shows both frames: `compute_object_vector_rate(env, spike_times, times, positions, object_positions, ...)` and `compute_egocentric_rate(env, spike_times, times, positions, headings, object_positions, ...)`;
  - "Cell-type API": the shipped object-vector predicates are `is_object_vector_cell` (allocentric) and `is_egocentric_object_vector_cell`;
  - "Peak / preferred accessors": `EgocentricRateResult` / `EgocentricRatesResult` → `ObjectVectorRateResult` / `ObjectVectorRatesResult` (both frames);
  - "Terminal verbs" / `to_xarray` lists naming `EgocentricRatesResult`.
- `.claude/QUICKSTART.md` "Object-Vector Cells"; `.claude/API_REFERENCE.md` (renamed classes and new functions).
- `docs/migration/v0.6.md` (:33, :49, :77 name `EgocentricRatesResult`): update the class names.
- `docs/glossary.md`: object-vector cell (allocentric); egocentric object-vector (bearing) cell.
- `examples/24_object_vector_cells.py`: both frames. Run `uv run jupytext --sync` on it, then `uv run python docs/sync_notebooks.py`.
- Phase 4b's `FLAGSHIP` tuple: add `compute_object_vector_rate`, `compute_object_vector_rates` and `is_egocentric_object_vector_cell`. Their examples use ≤ 60 s of simulated data. Every edited example stays self-contained and executable (no `+SKIP`).
- `CHANGELOG.md` `[Unreleased]`:
  - `### Fixed`: `label_cell_types` gate (19/20 → 0/20 "border");
  - `### Added`: `compute_object_vector_rate(s)`, `is_egocentric_object_vector_cell`, the `direction_frame` result field;
  - `### Changed` (breaking): the class renames, `spatial_information()`, `is_object_vector_cell` is allocentric and takes no `headings`, the simulator's default `direction_frame="allocentric"`.

## Deliberately not in this phase

- **Shuffle significance, `criterion=`, threshold constants, `>=` comparisons, removing the `try/except → False` blocks, and the bias documentation.** Phase 5b.
- **`has_place_field` / `is_place_cell(criterion=…)`.** Phase 5b.
- **The mirrored egocentric polar plot.** Phase 2a owns it; this phase only adds the allocentric branch.
- **Renaming `EgocentricPolarEnvironment`, `view_spatial_information`, or the `encoding/egocentric.py` module.** Separate naming work.
- **A `direction_frame=` keyword on any analysis function.** Ruled out by the error contract; the frames are separate functions.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/encoding/test_label_cell_types.py::test_label_cell_types_noise_is_unclassified` | 10-min noise fixture, `diffusion_kde`, `bandwidth=5`: all 20 labels are `"unclassified"`. **Fails on `main`** (19 `"border"`, spatial info 0.082–0.132 bits/spike) |
| `…::test_label_cell_types_border_requires_information` | precomputed `border_scores=[0.9, 0.9]` for `[allocentric field cell, one noise unit]` (10 min): labels `["border", "unclassified"]`. Same with `grid_scores` for `"grid"` |
| `…::test_label_cell_types_short_recording_bias` | 2-min noise fixture: the units labelled `"border"` are exactly those with spatial information ≥ 0.5 (Evidence: 15 of 20; record the observed count) |
| `tests/encoding/test_object_vector_frames.py::test_allocentric_equals_zero_heading_egocentric` | `compute_object_vector_rate(...)` firing rate equals `compute_egocentric_rate(..., headings=zeros, ...)` exactly (`assert_array_equal`, NaN-aware) |
| `…::test_allocentric_recovers_east_field` | field 20 cm East of object: `wrap(preferred_direction() − π)` within π/6 (prototype 0.26); `spatial_information()` allocentric ≥ 1.3× egocentric (prototype 2.86 / 1.63) |
| `…::test_egocentric_recovers_ahead_ovc` | egocentric model, preferred direction 0: egocentric `preferred_direction()` within π/6 of 0 (prototype 0.26); egocentric information ≥ 1.3× allocentric (2.94 / 1.77) |
| `…::test_result_records_frame` | `.direction_frame` is `"allocentric"` / `"egocentric"` for the two families, singular, plural and `rates[0]`; constructing a result without it raises `TypeError` |
| `…::test_frame_signatures_are_truthful` | `inspect.signature`: `compute_object_vector_rate(s)`, `is_object_vector_cell` have no `headings` or `direction_frame` parameter; `compute_egocentric_rate(s)` and `is_egocentric_object_vector_cell` have a required positional `headings` (default `inspect.Parameter.empty`). `is_object_vector_cell(..., headings=h)` raises `TypeError` |
| `…::test_free_predicates_match_methods` | `is_object_vector_cell(...)` equals `compute_object_vector_rate(...).is_object_vector_cell()`; `is_egocentric_object_vector_cell(...)` equals `compute_egocentric_rate(...).is_object_vector_cell()`; each method equals `classify()[i]` on the plural |
| `…::test_plot_allocentric_north_is_up` | a field peaked at +π/2 in an allocentric result is drawn above the centre: display dy > 0, \|dx\| < 0.1·dy |
| `tests/simulation/models/test_object_vector_cells.py::test_allocentric_model_needs_no_headings` | `firing_rate(positions)` works with `preferred_direction` set; peak rate when the object is due West of the animal for `preferred_direction=π`; `ground_truth["direction_frame"] == "allocentric"` |
| `…::test_egocentric_model_requires_headings` | `direction_frame="egocentric"` without headings → `ValueError` naming `direction_frame="allocentric"` with a `Fix:` line |
| Phase 4b `tests/docs/test_flagship_docstrings.py` and `test_docstring_sections.py` | pass with the three new `FLAGSHIP` entries |

Mark every test over 5 s `@pytest.mark.slow`; run them with `uv run pytest -m "slow and not napari" -n 4`. Update existing tests that construct `EgocentricRate(s)Result`, call `egocentric_spatial_information()`, or call `is_object_vector_cell(..., headings, ...)` (`grep -rn "EgocentricRate\|egocentric_spatial_information\|is_object_vector_cell" tests`): egocentric calls move to `is_egocentric_object_vector_cell`.

## Fixtures

Add to `tests/encoding/conftest.py` (session-scoped). It reproduces the Evidence numbers exactly, and Phase 5b reuses it:

```python
def _ou_trajectory(duration_s: float, seed: int, fs: float = 30.0, arena: float = 100.0):
    rng = np.random.default_rng(seed)
    n, dt = int(duration_s * fs), 1.0 / fs
    pos = np.empty((n, 2)); pos[0] = arena / 2; v = np.zeros(2)
    for i in range(1, n):
        v += -v * dt + 15.0 * np.sqrt(2.0 * dt) * rng.standard_normal(2)
        p = pos[i - 1] + v * dt
        for d in range(2):
            if p[d] < 0 or p[d] > arena:
                v[d] = -v[d]; p[d] = np.clip(p[d], 0, arena)
        pos[i] = p
    vel = np.gradient(pos, dt, axis=0)
    return np.arange(n) * dt, pos, np.arctan2(vel[:, 1], vel[:, 0])
```

- `ou_10min = _ou_trajectory(600.0, seed=0)`, plus `ou_2min = _ou_trajectory(120.0, seed=0)` for the fast tests.
- `noise_trains(duration, seed=1)`: 20 trains, each `np.sort(rng.uniform(t0, t1, rng.poisson(0.5 * duration)))`.
- Allocentric field cell: `PlaceCellModel(env, center=obj + [20, 0], width=6, max_rate=10)`, spikes `generate_poisson_spikes(..., seed=3)`.
- Egocentric OVC: `ObjectVectorCellModel(env, object_positions=obj, preferred_distance=20, distance_width=5, preferred_direction=0.0, direction_frame="egocentric", max_rate=10)`, spikes `seed=5`.
- `obj = [[50, 50]]`; `env = Environment.from_samples(pos, bin_size=5.0)`.

## Review

Before opening the PR, dispatch `code-reviewer` (or an equivalent independent reviewer) against the diff. Confirm:

- Every task is implemented as specified, and no object-vector analysis function has a `direction_frame=` keyword or a `headings=None` slot.
- The "Deliberately not in this phase" list is honored; nothing from Phase 5b leaked in.
- Validation slice tests pass; slow tests are marked.
- Tests aren't trivial: they exercise the asserted behavior, not tautologies, and shared setup is in fixtures (`testing-anti-patterns`).
- Docstrings, test names and module names don't reference this plan or its milestones.
- No `EgocentricRate(s)Result` or `egocentric_spatial_information` reference remains (`grep -rn` over `src`, `tests`, `docs`, `examples`, `CLAUDE.md`, `.claude/`).
- User-facing documentation listed as tasks is updated, not deferred.
