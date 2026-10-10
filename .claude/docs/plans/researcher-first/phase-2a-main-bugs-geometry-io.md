# Phase 2a — Fix the geometry, I/O, simulation and event bugs verified on `main`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

**Requires:** [Phase 1](phase-1-port-fixes.md). Task 6 uses Phase 1 Task 7's `resolve_unit_ids(..., input_ids=)` and Phase 1's `make_spike_group` fixture. Independent of [Phase 2b](phase-2b-main-bugs-operators-behavior.md).

Branching, commits, the CHANGELOG rule, the definition of done and the PR workflow are in [executing.md](executing.md). This file adds only what is specific to Phase 2a.

**Inputs to read first:**

- [src/neurospatial/layout/engines/graph.py:70-153, 316-339](../../../../src/neurospatial/layout/engines/graph.py) — `GraphLayout.build` and `to_linear`. `to_linear` calls `track_linearization.get_linearized_position` on `_build_params_used["graph_definition"]`. Task 1.
- `track_linearization/core.py` (installed 2.4.0) at `:1001-1014` and `:1206`. The nearest segment is a *position* in `graph.edges()`, but it is then looked up as the `edge_id` attribute. `make_track_graph` (`utils.py:94-95`) assigns `edge_id` = enumeration index, which is the only numbering under which this lookup is correct. Task 1.
- [src/neurospatial/environment/factories.py:64-75, 127-186, 740, 915-932, 978-1012](../../../../src/neurospatial/environment/factories.py). These assign `edge_id` in `edge_order` order: the W maze gets `edge_id`s 0,2,1,3,4 in enumeration order. Two comments claim `edge_id` is "not consumed by `to_linear()`". Task 1.
- [src/neurospatial/encoding/egocentric.py:2248, 2320-2323](../../../../src/neurospatial/encoding/egocentric.py), [encoding/directional.py:2296-2347, 2385-2388](../../../../src/neurospatial/encoding/directional.py), [stats/circular.py:1594, 1686-1692, 1797-1800](../../../../src/neurospatial/stats/circular.py) — the three polar axis setups. Task 2.
- [src/neurospatial/io/nwb/_behavior.py:122-147, 360-383](../../../../src/neurospatial/io/nwb/_behavior.py), [_pose.py:116-125](../../../../src/neurospatial/io/nwb/_pose.py), [_environment.py:1040-1060, 1100-1123, 1126-1179](../../../../src/neurospatial/io/nwb/_environment.py), [_adapters.py:26](../../../../src/neurospatial/io/nwb/_adapters.py) — raw `.data[:]` reads, the unit-string copy, `environment_from_position`'s `units` parameter and `_get_position_units` (silent-fallback Notes `:1149-1158`, return `:1178-1179`). Tasks 3 and 4.
- [src/neurospatial/simulation/models/place_cells.py:30, 201-207](../../../../src/neurospatial/simulation/models/place_cells.py), [simulation/validation.py:59, 163, 217-220](../../../../src/neurospatial/simulation/validation.py), [simulation/session.py:192](../../../../src/neurospatial/simulation/session.py), [simulation/examples.py:43](../../../../src/neurospatial/simulation/examples.py). `env.bin_sizes` is a per-bin *volume* (an area in 2-D), but these lines use it as a length. Task 5.
- [src/neurospatial/events/alignment.py:343-500](../../../../src/neurospatial/events/alignment.py) — `population_peri_event_histogram` (`n_units` at `:455`, `resolve_unit_ids` at `:461`, `enumerate(spike_trains)` at `:481`). Task 6.
- [src/neurospatial/behavior/vte.py:617-760](../../../../src/neurospatial/behavior/vte.py) — `compute_vte_session` (trial mask at `:688-690`, window from the whole session at `:714-716`, window record at `:737`). Task 7.
- **Files Phase 1 already changed** (re-locate line numbers by symbol):
  - `io/nwb/_environment.py`: Phase 1 Task 10 rewrote the writer, reader and `_ReconstructedLayout`. Tasks 3 and 4 here edit only `_get_position_units` and `environment_from_position`.
  - `stats/circular.py`: Phase 1 Task 4 changed three p-values. Task 2 here edits only the plot setup.
  - `environment/factories.py`: Phase 1 Task 5 edited a docstring only.
  - `_results.py`: Phase 1 Task 7 added `input_ids=` and the duplicate-label check to `resolve_unit_ids`. Task 6 calls it.
  - `tests/simulation/test_integration.py`: Phase 1 Task 1 marked `test_place_field_detection_accuracy` `xfail(strict=True)`, naming this phase. Task 5 resolves it.
  - `CHANGELOG.md`: Phase 1 created `### Fixed` under `## [Unreleased]`. Append to it.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — every new raise or warning in Tasks 3, 4 and 5 states what was wrong, why it matters, and a `Fix:` line.
- [Input conventions → Population identity](shared-contracts.md#input-conventions) — Task 6: a labelled `TsGroup`'s index becomes `unit_ids`. Labels are never overridden: a `unit_ids=` passed with a labelled group must equal its index, or the call raises. Duplicate labels raise. Both rules live in Phase 1 Task 7's `resolve_unit_ids(..., input_ids=)`.

**Designs referenced:** none.

## Tasks

Commits, regression-test-first and the per-commit CHANGELOG bullet follow [executing.md → While you work](executing.md#while-you-work). Phase-specific notes:

- Run a new regression test against the unmodified code with `uv run pytest <nodeid> -n 0` and quote the failure in the commit body. Do not use `git stash` for this.
- Tasks 5–7 were each verified on this branch (`da631a47`) with a minimal probe before being added. The measured numbers are quoted in each task and reused as the regression-test expectations.

1. **Linearize with `edge_id` equal to the enumeration index (`layout/engines/graph.py`).** The root cause is the ordering of `edge_id`, not the type of the node labels: integer labels fail identically. Fix it at the one place every graph environment passes through (`from_graph`, `maze`, `linear_track`, and `from_file` rebuilds). At the top of `build`, after the argument checks (`:106-111`), add:

   ```python
   # track_linearization finds a point's nearest segment by its position in
   # ``graph.edges()`` but looks that segment up by its ``edge_id`` attribute,
   # so the two must coincide. Linearize a copy numbered by enumeration; the
   # caller's graph is never mutated.
   track_graph = graph_definition.copy()
   for enumeration_index, (u, v) in enumerate(track_graph.edges()):
       track_graph.edges[u, v]["edge_id"] = enumeration_index
   self._build_params_used["graph_definition"] = track_graph
   ```

   Then use `track_graph` in place of `graph_definition` at `:115`, `:125` and `:131`. That way `_get_graph_bins` (which numbers segments by enumeration, `helpers/graph.py:116-118`), `to_linear`, `plot` and `ops/diffusion.py:1403` (`_graph_fv`) all read the same graph.

   **Old path removed in the same commit:** factories no longer assign `edge_id`.
   - `_add_edge_with_distance` (`factories.py:64-75`) loses its `edge_id` parameter.
   - Drop the `edge_id` counters at `:127-186` and the `edge_id=i` at `:740`.
   - In `maze(track_graph=...)` (`:915-932`), keep only the `distance` fill.
   - Replace the two false comments (`:66-71` docstring and `:919-924`) with one sentence: "`GraphLayout.build` numbers `edge_id` by `graph.edges()` order; any `edge_id` on the input graph is ignored."
   - Add that sentence to the `graph` parameter of `from_graph` (`:986-988`).

2. **Polar plots follow the library's angle conventions.**
   - **Egocentric** (`egocentric.py:2322-2323`). The convention is 0 = ahead and +π/2 = left (CLAUDE.md; `environment/polar.py:152-153`). Keep `set_theta_zero_location("N")`, set `set_theta_direction(1)`, and fix the comment ("counter-clockwise: +π/2 = left of the animal is drawn on the left"). On `main` a field at +π/2 is drawn at display dx = +80.3 px, which is right of centre.
   - **Allocentric head direction** (`directional.py:2387-2388`). The data convention is 0 = East, π/2 = North (`directional.py:1667-1668`; `ops/egocentric.py` `heading_from_velocity`). The current North-up, clockwise axes therefore draw a North-preferring cell pointing East: π/2 lands at dx > 0, dy ≈ 0. Set `set_theta_zero_location("E")` and `set_theta_direction(1)` explicitly, so a caller-supplied axis is also normalized. Rewrite the docstring sentences at `:2309-2311` and the Notes at `:2340-2347`: "angles are drawn as in the arena, 0 = East (right), π/2 = North (up), counter-clockwise".
   - **`plot_circular_basis_tuning`** (`circular.py:1797-1800`). Same change and same evidence: its own example fits head-direction data (`:1688`), and `circular_basis_metrics` returns `arctan2(beta_sin, beta_cos)` in the same math convention. Add a Notes line stating the orientation.

3. **NWB position readers apply `conversion` and `offset` and normalize unit names.** NWB defines the value in `unit` as `data * conversion + offset`.
   - In `io/nwb/_adapters.py`, next to `timestamps_from_series`, add:

     ```python
     def scaling_from_series(series: Any) -> tuple[float, float]:
         """(conversion, offset) mapping stored values to ``series.unit``."""
         return float(getattr(series, "conversion", 1.0)), float(getattr(series, "offset", 0.0))


     def data_from_series(series: Any) -> NDArray[np.float64]:
         """Materialize ``series.data`` in its declared unit (``data * conversion + offset``)."""
         data = np.asarray(series.data[:], dtype=np.float64)
         conversion, offset = scaling_from_series(series)
         return data * conversion + offset if (conversion, offset) != (1.0, 0.0) else data


     def require_unscaled_for_lazy(series: Any, *, context: str) -> None:
         """Refuse a lazy handle whose stored values are not in ``series.unit``."""
         conversion, offset = scaling_from_series(series)
         if (conversion, offset) != (1.0, 0.0):
             raise ValueError(
                 f"{context}(lazy=True) would return the stored values of '{series.name}', "
                 f"but the series declares conversion={conversion} and offset={offset}, so "
                 f"stored values are not in '{series.unit}'.\n"
                 f"Fix: call {context}(..., lazy=False); values are converted on read."
             )
     ```

   - Use `data_from_series` at `_behavior.py:141` (`read_position`), `_behavior.py:364` (`read_head_direction`, before the degree→radian step) and `_pose.py:123` (`read_pose`, per body part). Call `require_unscaled_for_lazy` on the lazy branches (`_behavior.py:122-138`; `_pose.py:116-120`, per series). `Session.from_nwb` (`recording.py:456`) and the NWB overlays (`_overlays.py:73, 150`) read eagerly, so they inherit the fix.
   - **Unit names.** In `_environment.py`, add a module mapping `_NWB_UNIT_ALIASES = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "m": "m", "centimeter": "cm", "centimeters": "cm", "cm": "cm", "millimeter": "mm", "millimeters": "mm", "mm": "mm", "pixel": "px", "pixels": "px", "px": "px"}`. `_get_position_units` (`:1179`) returns `_NWB_UNIT_ALIASES.get(unit.strip().lower(), unit)`. NWB's default `"meters"` then becomes the registry value `"m"` (`environment/core.py:398-400`) instead of a free-form string that triggers a warning. Task 4 adds the empty-unit warning to the same function.
   - **Docstrings.** `read_position`, `read_head_direction` and `read_pose` Returns sections, plus the `lazy` parameter: "values are in the series' `unit`: stored × `conversion` + `offset`; `lazy=True` raises when that map is not the identity". `environment_from_position` `units`: "auto-detected from the series `unit`, with NWB long names mapped to `m` / `cm` / `mm` / `px`".

4. **`environment_from_position` warns when it assumes `"cm"` (`io/nwb/_environment.py:1126-1179`).** This edits the same `_get_position_units` as Task 3 and lands after it.
   - **Bug (verified).** Probe `p2extra/g_nwbunit.py`: a `SpatialSeries` with `unit=""` gives `env.units == "cm"` and **no warning**. The Notes call this "a silent fallback". (pynwb fills `"meters"` when `unit` is omitted, so this case is an explicitly empty unit.) Non-empty unrecognized units (`"a.u."`, `"inches"`) already warn through the `env.units` registry check (`environment/core.py:398-425`), which names the value, so they need no change.
   - **Fix.** When `spatial_series.unit` is empty or None, emit one `UserWarning` and still return `"cm"`:

     > Position series '{name}' declares no unit (unit={unit!r}); assuming 'cm'. A wrong unit mislabels every distance, speed and bin size derived from this environment.
     > Fix: pass units='cm' (or 'm', 'mm', 'px') to environment_from_position to state the real unit.

     `environment_from_position` calls `_get_position_units` only when `units is None` (`:1108`), so passing `units=` silences it. Replace the "silent fallback" Notes (`:1149-1158`) with "warns, then assumes `cm`", and add the same to the `units` parameter (`:1050-1052`).

5. **Simulated place-field width uses a linear bin spacing.**
   - At `place_cells.py:201-207`, replace `3.0 * np.mean(env.bin_sizes)` with three times the median nearest-neighbour distance between bin centres. That equals `bin_size` on a regular grid and the bin length on a linearized track. Reuse `neurospatial.ops.binning._estimate_typical_bin_spacing(cKDTree(env.bin_centers), env.bin_centers)`.
   - When the spacing is not finite (a single-bin env), raise `ValueError` with `Fix: pass width=<sigma in environment units>`.
   - Apply the same spacing at `validation.py:217-220`: `max_center_error = 2.0 * spacing`.
   - Correct the docstrings: `place_cells.py:30`, `validation.py:59` and `:163`, `simulation/session.py:192`, `simulation/examples.py:43`. Each becomes "3 × bin spacing (median distance between neighbouring bin centres; equals `bin_size` on a regular grid)".
   - **Old test removed:** `tests/simulation/test_models.py:68-74` (`test_default_width`) asserts the buggy `3 * mean(bin_sizes)`. Replace it.
   - **Blast radius.** On a 2-D grid with bin 2 cm the default width drops from 12 cm to 6 cm (from 75 to 15 cm at bin 5), so everything built on the default simulators changes numerically:
     - `PlaceCellModel` and the session simulators (`open_field_session`, `linear_track_session`, `tmaze_alternation_session`, `simulate_session`) are used by 9 test files: `tests/decoding/test_decode_session.py`, `tests/decoding/test_estimator.py`, `tests/encoding/test_encoding_spatial.py` and `tests/simulation/test_{examples,integration,models,session,spikes,validation_sim}.py`.
     - They are also used by 10 examples (`examples/08, 11, 12, 15, 20, 21, 22, 24, 25, 27_*.py`), by `docs/user-guide/workflows.md` (snippets executed by `scripts/test_doc_snippets.py`), and by the README "Simulation > Quick Example" block (`docs/snippets.yml` id `readme_simulation_quick_example`). Notebooks 11 and 20 are re-executed in CI (`test_notebooks.yml`).
     - **Run them:**
       - `uv run python scripts/test_doc_snippets.py`;
       - the two CI notebooks, as `test_notebooks.yml` runs them (from the repository root): `MPLBACKEND=Agg uv run --extra notebooks jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 examples/11_place_field_analysis.ipynb` and the same for `examples/20_bayesian_decoding.ipynb`;
       - the 9 test files above, with `-m "not napari"` so slow tests run too.
     - **Re-baseline only tests whose premise was the old width.** A threshold tuned to 3·area-wide fields may legitimately move. Name each re-baselined test, with its old and new value, in the commit body. A test that fails for any other reason is a regression: stop and investigate.
     - `tests/simulation/test_integration.py:252` (`test_place_field_detection_accuracy`) computes its match tolerance as `2 * mean(env.bin_sizes)`, the same area-as-length bug. Use the bin spacing there too, and remove the `xfail(strict=True)` that Phase 1 Task 1 added. If it still fails with the corrected width and tolerance, report the measured match count instead of loosening it.
     - **Stored notebook outputs.** `mkdocs.yml:93` has `execute: false`, so the docs show the outputs stored in each `.ipynb`. Every example that simulates with the default width (at least `11`, `12`, `15` and `20`, which call the session simulators) now has stale outputs. Re-execute those notebooks, commit them, and run `uv run python docs/sync_notebooks.py`; `docs.yml` checks `git diff --exit-code docs/examples`.

6. **`population_peri_event_histogram` accepts a pynapple `TsGroup` (`events/alignment.py:343`).**
   - **Bug (verified).** The loop `for unit_idx, spike_times in enumerate(spike_trains)` (`:481`) iterates a `TsGroup`'s *keys*. Probe `p2extra/c_peth.py`, three units with keys `3, 7, 9`: a real `nap.TsGroup` and the `UserDict` double both raise `AxisError: axis -1 is out of bounds for array of dimension 0`. The same trains as a list give mean rates `[4.06, 10.0, 2.08]` Hz.
   - **Fix.** Mirror `compute_spatial_rates` (`encoding/spatial.py:3408-3419`) as Phase 1 Task 7 left it: `trains, extracted_ids = as_spike_trains_with_ids(spike_trains)` before the empty check; use `trains` everywhere below; call `resolve_unit_ids(unit_ids, n_units, input_ids=extracted_ids, context="population_peri_event_histogram")` (`:461`). That resolver raises when both are given and differ, and on a repeated label. Document in the `spike_trains` parameter that a `TsGroup` is accepted, that its index becomes `unit_ids`, and that a `unit_ids=` passed with it must equal that index. ([Phase 6b](phase-6b-argument-conventions.md) renames this parameter to `spike_times`; [Phase 3c](phase-3c-time-windows-decoding-events.md) adds `epochs`/`spike_window` to the same function.)

7. **`compute_vte_session` clamps each pre-decision window to its trial (`behavior/vte.py:714-737`).**
   - **Bug (verified).** The entry time is found within the trial (`:688-690`), but the window is cut from the **whole session** (`extract_pre_decision_window(positions, times, …)` at `:714`), so it reaches into the inter-trial interval or the previous trial. Probe `p2extra/d_vte.py`: a trial starting at 5.0 s that enters the decision region at 5.267 s gets the window `[4.267, 5.267]`. Its head sweep is **7.689 rad**, all of it from pre-trial zig-zagging. The same window clamped to `[5.0, 5.267]` gives 0.000 rad.
   - **Fix.** Extract the window from the trial's samples: `extract_pre_decision_window(trial_positions, trial_times, entry_time, window_duration)`. Record `(max(entry_time - window_duration, trial.start_time), entry_time)` at `:737`. Windows that become shorter than 3 samples are skipped by the existing rule (`:717-719`). Docstring: `window_duration` "is clipped at the trial start; samples before `trial.start_time` are never used". `compute_vte_trial` has no trial bounds and is unchanged.

8. **User-facing documentation check (no separate commit unless something is missing).** Per [executing.md](executing.md#while-you-work), each of Tasks 1–7 already appended its own bullet under `## [Unreleased]` → `### Fixed` in `CHANGELOG.md`, with the symptom and the numbers from the Validation slice. Before opening the PR, check that all seven are present and that these are marked as **behavior changes**:
   - Task 1: `to_linear` on graphs whose `edge_order` differs from `graph.edges()` order (the W maze) now returns along-track distance; any `edge_id` on an input graph is ignored;
   - Task 2: head-direction and circular-basis polar plots now draw 0 = East, counter-clockwise; egocentric polar plots draw +π/2 (left) on the left;
   - Task 3: NWB position, head-direction and pose readers return values in the series' `unit` (stored × `conversion` + `offset`); `lazy=True` raises for a non-identity scaling; `"meters"` becomes `"m"`;
   - Task 4: the new empty-unit warning;
   - Task 5: the default simulated field width and the default `max_center_error` shrink on 2-D grids (3 × bin spacing, not 3 × bin area);
   - Task 6: `population_peri_event_histogram` accepts a `TsGroup` and takes its `unit_ids`; a conflicting `unit_ids=` raises;
   - Task 7: VTE windows stop at the trial start.

   A missing bullet is added in a `docs: complete phase 2a changelog` commit.

## Deliberately not in this phase

- **The operator and behavior bugs** (finite-volume calculus, graph bases, heading interpolation, goal alignment, pre-decision heading statistics) are [Phase 2b](phase-2b-main-bugs-operators-behavior.md).
- **Object-vector-cell classification bias and an allocentric object-vector mode** are [Phase 5a](phase-5a-object-vector-frames.md), and so are the Høydal 2019 citation fixes.
- **The ported archive fixes,** including NWB environment *geometry* persistence, are [Phase 1](phase-1-port-fixes.md). Tasks 3 and 4 here edit other functions of the same `_environment.py`.
- **Gap handling, errors and executable docs, API surface, output polish** are Phases 3 (from [3a](phase-3a-time-windows-core.md)), 4 ([4a](phase-4a-errors.md), [4b](phase-4b-docs-that-run.md)), 6 ([6a](phase-6a-namespaces-snapshot.md)–[6c](phase-6c-data-holders.md)) and [7](phase-7-output-polish.md).
- **A second warning for non-empty unrecognized NWB units.** The `env.units` registry check already warns and names the value (Task 4 probe).
- **The labelled-group handling in `peri_event_histogram` (single unit).** It takes one 1-D train, so there are no keys to mis-iterate.
- **Upstream `track_linearization`.** Its positional-index-as-`edge_id` lookup should be reported upstream; Task 1 does not depend on that.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/environment/test_factory_presets.py::test_w_maze_to_linear_matches_track_distance[str labels, int labels]` | W maze `bl(0,0) bm(50,0) br(100,0) al(0,50) am(50,50) ar(100,50)`, bin 5: `to_linear` of `(25,0),(75,0),(0,40),(50,25),(100,25),(100,45)` == `[25, 75, 140, 175, 225, 245]` (`main`: `[25, 175, 114.03, 175, 225, 245]`) |
| `…::test_w_maze_to_linear_agrees_with_bin_at` | 2000 on-track samples (seed 0, t∈[0.02, 0.98] per edge): fraction where `linear_point_to_bin_ind(to_linear(x)) != bin_at(x)` == 0.0 (`main`: 0.3775) |
| `…::test_from_graph_ignores_input_edge_ids` | the same graph via `from_graph` with `edge_id`s following `edge_order` gives the correct values above; the caller's `edge_id`s are unchanged after the call |
| `…::test_w_maze_to_linear_survives_file_roundtrip` | `to_file` then `from_file`, then the same six expected values (`main`: 175 and 114.03) |
| `…::test_plus_and_t_maze_to_linear_unchanged` (guard) | plus and T mazes give identical `to_linear` before and after |
| `tests/encoding/test_encoding_egocentric.py::test_object_vector_plot_draws_left_on_left` | `EgocentricRateResult` peaked at +π/2 (polar env 0–50, 10×12 bins): the peak marker's display dx < 0 (`main`: +80.3 px) |
| `tests/encoding/test_encoding_directional.py::test_head_direction_plot_north_is_up` | after `plot_head_direction_tuning`, `ax.transData` maps (π/2, r) to dx≈0, dy>0 and (0, r) to dx>0, dy≈0 (`main`: π/2 at dx>0, dy≈0) |
| `tests/stats/test_stats_circular.py::test_circular_basis_plot_north_is_up` | `plot_circular_basis_tuning(1.0, 0.0)`: (π/2, r) maps to dx≈0, dy>0 (`main`: dx = +248 px, dy = 0) |
| `tests/nwb/test_behavior.py::test_read_position_applies_conversion_and_offset` | pixel data 0–500, `unit="meters"`, `conversion=0.002`, `offset=0.1` → `positions.min(0) == [0.1, 0.1]`, `positions.max(0) == [1.1, 1.1]` (`main`: max `[500, 500]`) |
| `…::test_read_position_lazy_refuses_scaled_series` | the same series with `lazy=True` → `ValueError` containing `Fix:`; an identity-scaled series stays lazy (guard) |
| `…::test_read_head_direction_applies_conversion` | degrees stored ×0.5 with `conversion=2.0` → radians equal `deg2rad(2·stored)` |
| `tests/nwb/test_pose.py::test_read_pose_applies_conversion_and_offset` | body-part coordinates equal stored × conversion + offset |
| `tests/nwb/test_environment.py::test_environment_from_position_uses_converted_meters` | `environment_from_position(nwbfile, bin_size=0.05)` (the converted extent is 1 m, so a cm-scale bin would give a single bin): `env.units == "m"`, with no units-registry warning; `bin_centers` lie within [0.1 − 0.05, 1.1 + 0.05] (`main`: units `"meters"`, extent 500 × 500) |
| `tests/nwb/test_environment.py::test_empty_unit_warns_and_assumes_cm` | `unit=""` → one `UserWarning` containing `"declares no unit"` and `"Fix: pass units="`; `env.units == "cm"`. Passing `units="cm"` emits no warning (before the fix: silent) |
| `tests/simulation/test_models.py::test_default_width_is_three_bin_spacings[1,2,5]` | `from_samples` grid on 0–100 cm: `width == 3 * np.diff(env.layout.grid_edges[0])[0]` at rtol 1e-12, and ≈ 3·bin_size within 5% (`main`: 3, 12, 75 cm for bin 1, 2, 5) |
| `…::test_default_width_on_track_uses_bin_length` | W-maze env with bin 5 → `width` ≈ 15 |
| `…::test_default_width_single_bin_raises` | 1-bin env, `width=None` → `ValueError` with `Fix: pass width=` |
| `tests/simulation/test_validation_sim.py::test_default_center_error_threshold` | the default `max_center_error` equals 2 × spacing (4.0 cm at bin 2; `main`: 8.0) |
| `tests/simulation/test_integration.py::test_place_field_detection_accuracy` (slow; Phase 1's `xfail` removed) | passes with the spacing-based tolerance and the corrected default width, or its measured match count is reported in the PR (`main`: 0 of 5 within 8.0 cm) |
| `uv run python scripts/test_doc_snippets.py`; notebooks 11 and 20 re-executed | all pass, including `readme_simulation_quick_example` and the `workflows.md` snippets |
| `tests/events/test_alignment.py::test_population_psth_accepts_tsgroup` | spike-group double with keys `3, 7, 9` → `unit_ids == [3, 7, 9]` and `firing_rates` equal to the list input's (mean rates `[4.06, 10.0, 2.08]` Hz on the probe data); a real `nap.TsGroup` gives the same (`@pytest.mark.pynapple`). `unit_ids=[3, 7, 9]` is accepted; `unit_ids=[9, 7, 3]` raises `ValueError` listing both orders. Before the fix: `AxisError` |
| `tests/behavior/test_vte.py::test_session_window_clamped_to_trial_start` | the Task 7 probe path: `trial_results[0].window_start == 5.0` and `head_sweep_magnitude == 0.0` within 1e-12 (before the fix: window start 4.267, head sweep 7.689 rad) |

Run the slice with `-n 0`. Then satisfy [executing.md → Definition of done](executing.md#definition-of-done). Phase-specific extra checks:

- `uv run pytest tests/nwb -n 0` and `uv run pytest tests/events/test_alignment.py -m "pynapple" -n 0` under `uv sync --all-extras`. Check with `-rs` that nothing skips for a missing extra.
- The Task 5 blast-radius runs (doc snippets, notebooks 11 and 20, the 9 simulation-dependent test files including their slow tests).

## Fixtures

- **W maze, plus maze and T maze.** Built inline from the node coordinates in the table. Reuse `tests/conftest.py`'s session-scoped `tmaze_env` where it fits.
- **Polar plots.** Use the `Agg` backend. Get display offsets from `ax.transData.transform` relative to `(0, 0)`. Build the egocentric field with `Environment.from_polar_egocentric((0, 50), (−π, π), 5.0, 2π/12)` and a von-Mises-in-angle × Gaussian-in-distance rate peaked at (25, +π/2).
- **NWB.** In-memory `NWBFile` with a `Position` `SpatialSeries`, as in the triage repro `claim6_conversion_offset.py`: 100 rows cycling `(0,0),(500,0),(500,500),(0,500),(250,250)` at 30 Hz. Add a `CompassDirection` series and an ndx-pose series with non-identity `conversion`. These run in `test_nwb.yml`. Environments built from the converted (meter-scale) positions use `bin_size=0.05`.
- **NWB empty unit.** The in-memory `NWBFile` above with `SpatialSeries(unit="")`.
- **Simulation.** `from_samples` grids on 0–100 cm at bin 1, 2 and 5; the W maze above at bin 5; a one-bin env for the error case.
- **Population PETH.** Three seeded uniform trains (seed 0; 500, 1000 and 200 spikes over 100 s) with keys `3, 7, 9`, events every 5 s from 5 to 90 s, `window=(-1, 1)`, `bin_size=0.1`. Use Phase 1's `make_spike_group` fixture (`tests/conftest.py`).
- **VTE.** The probe path inline: `t = np.arange(0, 10, 1/30)`, zig-zag before 5 s, then east at 30 cm/s into `box(45, 40, 60, 60)`, one `Trial(5.0, 9.9, …)`.
- **Probe scripts** live in the session scratchpad (`p2extra/`); the tests reproduce them inline and do not depend on those files.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Every test re-baselined in Task 5 is named in its commit body, with a reason tied to the old width.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
