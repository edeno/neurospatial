# Phase 23 — Runnable documentation + CI example gate

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase closes **Theme 5 — Documentation / example drift** from
`.claude/reviews/SUMMARY.md`. It has two halves that ship in one PR:

1. **Fix every non-runnable example** so it executes against the *current*
   API (the API that exists *after* phases 14–22 land — see
   "Deliberately not in this phase").
2. **Add a CI gate** that actually executes the examples (doctests + the
   curated-snippet harness) so paste-and-crash documentation cannot recur.

> **Ordering dependency.** This phase lands **after phases 14–22**. Those
> phases change behavior and signatures; this phase only rewrites docs and
> examples to match whatever the final API is. Before editing, re-read each
> referenced signature in the source — if an earlier phase already touched it,
> the *current source* is authoritative, not the snippet below.

---

**Inputs to read first:**

- [scripts/test_doc_snippets.py](../../../../scripts/test_doc_snippets.py) — the existing curated snippet runner (manifest-driven, extracts fenced ```python``` blocks / package-docstring REPL groups and executes them). The CI gate extends this, it does not replace it.
- [docs/snippets.yml](../../../../docs/snippets.yml) — the snippet manifest (14 entries today). New corrected blocks get registered here.
- [.github/workflows/test_docs.yml](../../../../.github/workflows/test_docs.yml) — runs `scripts/test_doc_snippets.py` on py3.10 + py3.13. **No `--doctest-modules` job exists anywhere in CI** (verified: `grep -rn doctest .github/ pyproject.toml` is empty) — that is the gap for docstring examples.
- [.github/workflows/test.yml:76-81](../../../../.github/workflows/test.yml) — main pytest job; `addopts` in `pyproject.toml:139` excludes `slow`/`napari` and does **not** include doctests.
- [pyproject.toml:131-141](../../../../pyproject.toml) — `[tool.pytest.ini_options]`; `testpaths = ["tests"]`, custom markers registered with `--strict-markers`.

**Verified API facts (re-confirm against source before editing — phases 14–22 may have moved them):**

- `PositionOverlay` field is `positions` (not `data`); `times` is an optional field — `src/neurospatial/animation/overlays.py:263-264`.
- `BodypartOverlay` fields are `data` (dict), `times`, `skeleton: Skeleton | None` — there is **no** `skeleton_color`/`skeleton_width` field — `src/neurospatial/animation/overlays.py:438-440`.
- `Skeleton(name, nodes, edges, *, edge_color=..., edge_width=...)`; convenience `Skeleton.from_edge_list(edges, name="skeleton", *, node_order=None, **kwargs)` — `src/neurospatial/animation/skeleton.py:118-135,337-344`. Importable as `from neurospatial.animation import Skeleton`.
- `path_progress(position_bins, env, *, start_bins, goal_bins, metric=...)` — `position_bins` first, `env` second, the rest keyword-only — `src/neurospatial/behavior/navigation.py:483-490`.
- `cost_to_goal(position_bins, env, *, goal_bins, cost_map=None, terrain_difficulty=None)` — `src/neurospatial/behavior/navigation.py:754-760`.
- `compute_vte_session(positions, times, env, *, decision_region, trials, window_duration=1.0, min_speed=5.0, alpha=0.5, vte_threshold=0.5)` — `env` is the 3rd positional; `decision_region` and `trials` are keyword-only — `src/neurospatial/behavior/vte.py` (sig block).
- `event_indicator(sample_times, event_times, *, window: float = 0.0)` — `window` is a **scalar half-width**, not a tuple — `src/neurospatial/events/regressors.py` (sig). (By contrast `event_count_in_window(..., window: tuple[float, float])` genuinely takes a tuple, so QUICKSTART:870 is already correct — do not touch it.)
- `circular_basis_metrics(beta_sin, beta_cos, cov_matrix=None)` — single optional `cov_matrix`, **no** `var_sin`/`var_cos`/`cov_sin_cos` — `src/neurospatial/stats/circular.py` (sig).
- Phase-precession lives at `neurospatial.encoding.phase_precession`; `neurospatial.metrics` does not exist.
- `ViewRatesResult.peak_view_location()` is **singular** (batch returns `(n_neurons, n_dims)`) — `src/neurospatial/encoding/view.py:625`; there is no `peak_view_locations()`.
- `DirectionalRateResult.mean_vector_length()` (no `.mrl()`) — `src/neurospatial/encoding/directional.py:380`.
- `EgocentricRatesResult` (batch, from `compute_egocentric_rates`) has only `preferred_distances()`, `preferred_directions()`, `detect_ovcs()`, `plot(idx)` — **not** the singular `preferred_distance()`/`is_object_vector_cell()`/`plot()`, which live on the single-neuron `EgocentricRateResult` — `src/neurospatial/encoding/egocentric.py:250,388,677,792,206,626`.
- `bin_sequence(times, positions, *, dedup=True, outside_value=-1)` — `times` first — `src/neurospatial/environment/trajectory.py:373-376`.
- `Environment.plot(ax=None, ...)` plots the *layout*; field rendering is `Environment.plot_field(field, ...)` — `src/neurospatial/environment/visualization.py:134,298`.
- `graph_convolve(...)` is the public name; there is no `convolve` — `src/neurospatial/ops/graph.py:288`.
- `load_labelme_json(json_path, *, pixel_to_world=None, ...)` — kwarg is `pixel_to_world`, not `transform` — `src/neurospatial/regions/io.py:85-88`. (The sibling `points_in_any_region`/`regions_containing_points` *do* take `transform=` — `regions/ops.py:196,253` — so only the `load_labelme_json(...)` call in the `ops.py` module docstring is wrong.)
- `plot_regions(regions, *, ax=None, region_names=None, default_kwargs=None, world_to_pixel=None, add_legend=True, **per_region_kwargs)` — `regions` first, everything else keyword-only — `src/neurospatial/regions/plot.py:31-39`.
- Annotation public symbol is `RegionType`, not `Role` — `src/neurospatial/annotation/__init__.py:10,28`.
- `BoundaryConfig` defaults: `method="alpha_shape"`, `buffer_fraction=0.02`, `simplify_fraction=0.01` — `src/neurospatial/annotation/_boundary_inference.py:68-70`.
- `PlaceCellModel`/`BoundaryCellModel` default `baseline_rate=0.01` (not `0.001`) — `src/neurospatial/simulation/models/place_cells.py:168`, `boundary_cells.py:157`.
- `neurospatial.differential` and `neurospatial.reference_frames` do not exist; the differential operator lives in `neurospatial.ops.calculus`, egocentric transforms in `neurospatial.ops.egocentric`.
- `from_samples`/`from_polygon` morphological kwarg is `close_gaps` (not `close`) — `src/neurospatial/environment/factories.py:127`.
- Behavior segmentation lives in `neurospatial.behavior.segmentation` / is re-exported from `neurospatial.behavior`; `neurospatial.segmentation` does not exist.
- There is no `write_intervals()`; the NWB interval writers are `write_laps`, `write_region_crossings`, `write_trials`, `write_events` — `src/neurospatial/io/nwb/_events.py:207,414,561,1001`.

---

## Tasks

### A. Overlay docs (animation)

- **`.claude/QUICKSTART.md:674-693`** — `PositionOverlay(data=...)` → `positions=...`, and the `BodypartOverlay` raw-list `skeleton=`:
  - Lines 674-679 / 691-692: replace `data=trajectory` / `data=traj1` / `data=traj2` with `positions=...`. (Optionally add `times=...` where a timestamp array is in scope; `times` defaults to `None`, so it is not strictly required.)
  - Lines 683-688: build a `Skeleton` first, then pass it. Corrected block:
    ```python
    from neurospatial.animation import BodypartOverlay, Skeleton

    skeleton = Skeleton.from_edge_list(
        [("tail", "body"), ("body", "nose")],
        name="rodent",
        edge_color="white",
        edge_width=2.0,
    )
    bodypart_overlay = BodypartOverlay(
        data={"nose": nose_traj, "body": body_traj, "tail": tail_traj},
        skeleton=skeleton,
        colors={"nose": "yellow", "body": "red", "tail": "blue"},
    )
    ```
- **`docs/animation_overlays.md:105-126`** and **`:567-578`** — same fix: convert the inline `skeleton=[(...), ...]` list into a `Skeleton.from_edge_list([...], edge_color="white", edge_width=2.0)` (the `skeleton_color`/`skeleton_width` kwargs at 124-125 and 578 fold into the `Skeleton`, then are deleted from the `BodypartOverlay(...)` call).
- **`docs/animation_overlays.md:993`** — the `PositionOverlay` dataclass-reference block lists `data: NDArray...`; change the attribute name to `positions: NDArray...` to match the real field.
- **`docs/animation_overlays.md:1006-1011`** — the `BodypartOverlay` dataclass-reference block lists `skeleton: list[tuple[str, str]] | None`, `skeleton_color: str = "white"`, `skeleton_width: float = 2.0`. Replace with the real fields: `skeleton: Skeleton | None = None` and **remove** the two `skeleton_*` lines.

### B. Behavior / events non-runnable examples

- **`src/neurospatial/behavior/navigation.py:538-554`** (`path_progress` docstring) — the doctest at 540-545 already uses the correct `path_progress(env, position_bins, start_bins=..., goal_bins=...)` order? It does **not**: the real signature is `(position_bins, env, *, start_bins, goal_bins)`. Rewrite both doctest groups to `position_bins` first, `env` second:
  ```python
  >>> progress = path_progress(
  ...     position_bins,
  ...     env,
  ...     start_bins=np.full(len(position_bins), 10),
  ...     goal_bins=np.full(len(position_bins), 50),
  ... )  # doctest: +SKIP
  ```
  and the multi-trial example at 552-554 to `path_progress(position_bins, env, start_bins=start_bins, goal_bins=goal_bins)`.
- **`src/neurospatial/behavior/navigation.py:796-798`** (`cost_to_goal` docstring) — example passes `cost_to_goal(env, position_bins, goal_bin, cost_map=...)` (env first, positional `goal_bin`). Correct to the real order with keyword-only `goal_bins`:
  ```python
  >>> cost = cost_to_goal(
  ...     position_bins, env, goal_bins=goal_bins, cost_map=cost_map
  ... )  # doctest: +SKIP
  ```
  (The simpler example at 791 already uses `cost_to_goal(position_bins, env, goal_bins=goal_bin)` — verify it stays correct.)
- **`src/neurospatial/behavior/vte.py:638-649`** and **`.claude/QUICKSTART.md:267-273`** (`compute_vte_session` argument order) — **deferred to phase 22.** Phase 22 (Task D.6) makes `compute_vte_session` env-first (`(env, positions, times, *, decision_region, trials, ...)`) **and** corrects these same blocks as part of that rename. Phase 23 lands *after* phase 22, so it does **not** prescribe a concrete pre-22 edit here (in particular it must **not** re-introduce the pre-22 `positions, times, env` order). Phase 23's only job for these two blocks is to **verify they execute against the post-22 (env-first) API** and register them in the snippet/doctest gate; if they still fail, that is a phase-22 bug, not a doc edit.
- **`src/neurospatial/events/__init__.py:58-64`** (package docstring) and **`.claude/QUICKSTART.md:863-866`** (`event_indicator` `window`) — **deferred to phase 22.** Phase 22 (Task B.4) makes `event_indicator`'s `window` a keyword-only `(start, end)` tuple **and** itself fixes these two `event_indicator(..., window=(-0.5, 1.0))` snippets so the tuple form becomes correct. Phase 23 lands *after* phase 22, so it does **not** prescribe a concrete pre-22 edit here (in particular it must **not** re-introduce the pre-22 scalar `window=0.5` form). Phase 23's only job for these blocks is to **verify they execute against the post-22 `(start, end)`-tuple API** and register them in the gate; a remaining failure is a phase-22 bug, not a doc edit. **Leave QUICKSTART:870 (`event_count_in_window(..., window=(-2.0, 0.0))`) unchanged** — that function genuinely takes a tuple (and phase 22 only makes its `window` keyword-only, not its type).

### C. Stats

- **`.claude/QUICKSTART.md:797-802`** — `circular_basis_metrics(beta_sin, beta_cos, var_sin=..., var_cos=..., cov_sin_cos=...)` uses three nonexistent kwargs. Replace with the single `cov_matrix` argument (the 2×2 covariance submatrix for the `[sin, cos]` coefficients):
  ```python
  cov = fit.cov_params()
  amplitude, preferred_direction, p_value = circular_basis_metrics(
      beta_sin, beta_cos, cov_matrix=cov[1:3, 1:3]
  )
  ```
  (`beta_sin, beta_cos = fit.params[1], fit.params[2]`, so the matching covariance slice is rows/cols 1–2.)
- **`src/neurospatial/stats/circular.py:31-34`** (module docstring "Phase precession analysis?" pointer) — change `neurospatial.metrics.phase_precession` → `neurospatial.encoding.phase_precession`.

### D. Encoding

- **`.claude/QUICKSTART.md:564`** and **`:583`** — `result.peak_view_locations()` / `batch.peak_view_locations()` (plural) → `peak_view_location()` (singular; the batch method, returns `(n_neurons, n_dims)`). Leave the single-neuron call at 578 (`single.peak_view_location()`) unchanged — it is already correct.
- **`.claude/API_REFERENCE.md:346`** — `.mrl()` → `.mean_vector_length()` in the `DirectionalRateResult` method list (keep the "Mean resultant length (tuning strength)" gloss).
- **`.claude/QUICKSTART.md:459-468`** — the "Classify object-vector cells from result metrics" block calls `result.is_object_vector_cell(...)`, `result.preferred_distance()`, `result.preferred_direction()`, `result.plot()` — singular, single-neuron methods — but the `result` in scope (line 441) is the **batch** `EgocentricRatesResult`, which has only `preferred_distances()`/`preferred_directions()`/`detect_ovcs()`/`plot(idx)`. Fix by re-anchoring the block to a single-neuron result. Either:
  - introduce a `single = compute_egocentric_rate(env, spike_times, times, positions, headings, object_positions, ...)` line and call the singular methods on `single`; **or**
  - rewrite the block to use the batch API (`result.detect_ovcs(min_info=0.5)`, `result.preferred_distances()`, `result.preferred_directions()`, `result.plot(idx=0)`).
  Prefer the first (it matches the section's "canonical result object" intent). Make the chosen variable name unambiguous so it cannot collide with the batch `result`.

### E. Rename-corrupted prose (decoding)

A global find-replace of "uncertainty"/"entropy" → the identifier `posterior_entropy` corrupted English prose. Restore the words while leaving genuine identifiers intact:

- **`src/neurospatial/decoding/trajectory.py:12`** — "…with optional Monte Carlo posterior_entropy." → "…with optional Monte Carlo **uncertainty**."
- **`:102`** — "…with **uncertainty** quantification via Monte Carlo sampling." (currently `posterior_entropy quantification`).
- **`:373`** — "Optionally uses Monte Carlo sampling to quantify **uncertainty** in the slope estimate." (currently `posterior_entropy`).
- **`:386`** — "Number of Monte Carlo samples from the posterior for **uncertainty** estimation…".
- **`:390`** — `"map"` bullet: "Fast but ignores **uncertainty**."
- **`:392`** — `"sample"` bullet: "Provides **uncertainty** estimate via slope_std."
- **`src/neurospatial/decoding/estimates.py:203`** — the `posterior_entropy` *function name* on the `def` line is a real identifier — **do not rename it**. The corruption is in the docstring summary at **:203** ("Posterior entropy in bits (**entropy** measure)." — currently `(posterior_entropy measure)`) and **:205** ("Measures the **entropy** in the position estimate." — currently `the posterior_entropy in`). Fix only the prose words, keep the `def posterior_entropy(` signature.

> Verification note: grep each file for `posterior_entropy` after editing; every remaining occurrence must be a genuine symbol reference (function name, attribute, parameter), never a sentence word.

### F. Ops

- **`src/neurospatial/ops/graph.py:360-381`** (`graph_convolve` docstring examples) — `from neurospatial.ops.graph import convolve` and the two `convolve(env, field, ...)` calls reference a nonexistent function. Replace `convolve` with `graph_convolve` in the import (line 360) and both call sites (lines 370, 382).
- **`src/neurospatial/ops/basis.py:55,69,87,419,1053`** — every `env.bin_sequence(trajectory, times)` has the arguments reversed; correct to `env.bin_sequence(times, trajectory)` (signature is `bin_sequence(times, positions)`).
- **`src/neurospatial/ops/basis.py` (the `env.plot(place_field, ...)` line in the full-GLM example, ~line 87 region per the SUMMARY)** — `env.plot(...)` first positional is `ax`, not a field. Change to `env.plot_field(place_field, title="Fitted Place Field")`.

### G. Cross-module doc drift (environment / behavior / io / regions / annotation / simulation)

- **`src/neurospatial/environment/core.py:1099`** — See-Also `neurospatial.differential.compute_differential_operator` → `neurospatial.ops.calculus.compute_differential_operator`.
- **`src/neurospatial/environment/factories.py:944`** — See-Also `neurospatial.reference_frames : Functions for egocentric transforms.` → `neurospatial.ops.egocentric : Functions for egocentric transforms.`
- **`docs/user-guide/environments.md:43`** — `dilate`, `fill_holes`, `close` → `dilate`, `fill_holes`, `close_gaps` (the real factory kwarg).
- **`docs/user-guide/trajectory-and-behavioral-analysis.md`** — every `from neurospatial.segmentation import ...` (lines 3 prose ref, 236, 274, 315, 365, 486 — grep to enumerate) → `from neurospatial.behavior.segmentation import ...` (or the `neurospatial.behavior` re-export; verify which symbols are re-exported before choosing). Update the line-3 prose module name likewise.
- **`src/neurospatial/io/nwb/_events.py:608`** and **`:780`** — both reference `write_intervals()`, which does not exist. Re-point to an actual writer for "store these intervals under a different name", e.g. `write_region_crossings()` / `write_events()` (pick the one whose schema matches the surrounding context), or drop the dangling clause. Confirm the chosen function exists in `_events.py` and is exported from `neurospatial.io.nwb.__init__`.
- **`src/neurospatial/regions/plot.py:44-66`** — the `Parameters` block lists `ax` first then `regions`, contradicting the real signature `plot_regions(regions, *, ax=None, ...)`. Reorder the documented parameters to match (`regions` first; `ax`, `region_names`, `default_kwargs`, `world_to_pixel`, `add_legend` as keyword-only), and fix any example call that passes `ax` positionally to use `plot_regions(regions, ax=ax, ...)`.
- **`src/neurospatial/regions/ops.py:37`** (module docstring) — the `load_labelme_json(Path("annotations.json"), transform=transform)` call uses the wrong kwarg; change `transform=transform` → `pixel_to_world=transform` **only on the `load_labelme_json` call**. The `points_in_any_region(..., transform=transform)` / `regions_containing_points(..., transform=transform)` calls in the same example are correct (those functions do take `transform=`) — leave them.
- **`docs/user-guide/video-annotation.md:263-271`** — `from neurospatial.annotation import Role` → `from neurospatial.annotation import RegionType`, and update the type hint `role: Role` → `role: RegionType` (and any prose using the symbol name `Role`).
- **`src/neurospatial/annotation/core.py:100-102`** (`annotate_video` docstring, `boundary_config`) — "uses BoundaryConfig defaults (**convex_hull**, 2% buffer, 1% simplify)" → "(**alpha_shape**, 2% buffer, 1% simplify)" to match the real `BoundaryConfig` default `method="alpha_shape"`.
- **`src/neurospatial/simulation/session.py:165-169`** — the `baseline_rate` defaults documented for place and boundary cells are `0.001`; the real model defaults are `0.01`. Correct the place bullet (line ~165) and the boundary bullet (line ~169) to `default: 0.01`. Verify the grid bullet's documented default against `simulation/models/grid_cells.py` and correct if it disagrees.
- **`src/neurospatial/simulation/examples.py:44`** — `baseline_rate : float - Baseline firing rate in Hz (default: 0.001)` → `(default: 0.01)` (matches `PlaceCellModel`).

### H. CI example gate

- **Add a doctest job.** Doctests are not executed anywhere today. Add a step to `test_docs.yml` (or a small dedicated job) that runs:
  ```yaml
  - name: Run module doctests
    env:
      MPLBACKEND: Agg
    run: uv run pytest --doctest-modules src/neurospatial/
  ```
  This is what catches the corrected docstring examples (graph_convolve, basis.py, navigation, vte, circular, decoding). Doctests already in the tree use `# doctest: +SKIP` for examples that need external fixtures; the corrected examples above either keep `+SKIP` (when they reference undefined fixtures) **or** are made self-contained so they execute. Prefer self-contained where cheap; otherwise keep `+SKIP` but ensure the *syntax / kwarg names* are still valid (a `+SKIP` doctest is not parsed for kwargs, so for those, coverage comes from the snippet harness instead — see next bullet).
- **Register the corrected QUICKSTART/README blocks in the snippet harness.** Add `docs/snippets.yml` entries (with minimal `setup:` fixtures) for the blocks fixed in this phase that are *not* protected by `+SKIP` doctests: the overlay block (QUICKSTART:670-693), the OVC classify block (459-468), the view batch block (552-583), the VTE block (267-293), the events GLM block (855-870), and the circular-basis block (778-806). Each entry points at the markdown `source`, the fenced-block `index`, and a `setup:` prelude that defines the fixtures the snippet consumes (`positions`, `times`, `trajectory`, `spike_times`, `headings`, `object_positions`, `fields`, `frame_times`, etc.). Use the smallest synthetic arrays that let the block execute under `MPLBACKEND=Agg`.
- **Wire both into the existing `test_docs.yml`** so the gate runs on every PR (the snippet runner already runs there; the doctest step is the addition). Do not duplicate the job into `test.yml`.

### I. Worked end-to-end example (cross-reference only — do not author here)

- The single worked example `NWB → read_units → bin_spikes_in_time → fields → decode → animate` depends on APIs introduced/finalized in **phases 14 and 15** (`read_units`, `bin_spikes_in_time`). Do **not** write or execute that example in this phase. Add a one-line cross-reference in `overview.md` (or this file's "Deliberately not in this phase") noting it is authored and added to the snippet manifest in the phase that lands those functions, so the example gate covers it then.

---

## Deliberately not in this phase

- **Any API behavior or signature change.** All renames/reorders that the examples now reflect were made in **phases 14–22**. This phase only rewrites docs to match the *post-14–22* API. If a snippet still fails because the underlying function is wrong, that is a bug for the owning phase, not a doc edit — do not "fix" it by papering over the example.
- **Renaming `posterior_entropy` the function/attribute/parameter** (`decoding/estimates.py:203` `def`, and the `n_samples` doc that legitimately names the measure). Only prose words are restored; the public symbol stays.
- **Authoring the NWB→…→animate worked example** (Task I) — deferred to the phase that lands `read_units` / `bin_spikes_in_time` (phases 14/15). Cross-reference only.
- **Broadening doctest coverage to `tests/` or to non-`src` markdown** beyond the curated `snippets.yml` set. Keep the gate scoped to `src/neurospatial/` doctests + the explicitly-enumerated snippet manifest, consistent with the existing harness's opt-in philosophy.
- **Theme 7 API-convention drift** (RNG naming, `*` separators, env-first ordering) — separate phase; only the *documentation* consequences that overlap Theme 5 are in scope here.
- **Concrete pre-22 edits to the two snippet families phase 22 already rewrites** — the `event_indicator` `window` snippet (`events/__init__.py:58-64`, `QUICKSTART.md:863-866`) and the `compute_vte_session` argument-order snippet (`behavior/vte.py:638-649`, `QUICKSTART.md:267-273`). These blocks are corrected by **phase 22** (Tasks B.4 and D.6); phase 23 only **verifies they execute against the post-22 API** and registers them in the doctest/snippet gate, and must **not** re-introduce the pre-22 form (scalar `window=0.5`, or `positions, times, env` order).

---

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest --doctest-modules src/neurospatial/` (new CI step) | All non-`+SKIP` docstring examples execute; specifically the corrected `graph_convolve`, `ops/basis.py`, `navigation.path_progress`/`cost_to_goal`, `vte.compute_vte_session`, `stats/circular`, and `decoding/trajectory`+`estimates` docstrings raise no error / `+SKIP` ones still parse. |
| `uv run python scripts/test_doc_snippets.py` (existing runner, extended manifest) | Every registered snippet — including the newly-added overlay, OVC-classify, view-batch, VTE, events-GLM, and circular-basis blocks — runs to completion under `MPLBACKEND=Agg` and reports `pass` (not `skip`). |
| `tests/test_doc_snippets_helper.py` (existing self-test) | The snippet-extraction/manifest helpers still pass after new entries are added. |
| Representative snippet test: overlay block | `PositionOverlay(positions=...)` and `BodypartOverlay(data=..., skeleton=Skeleton.from_edge_list([...]))` construct without `TypeError`/`AttributeError`, and `env.animate_fields(..., overlays=[...], backend="video")` (or a headless render path) produces output. |
| Representative snippet test: OVC-classify block | The re-anchored single-neuron result exposes `is_object_vector_cell()`, `preferred_distance()`, `preferred_direction()`, `plot()` without `AttributeError`. |
| Grep gate (cheap regression) | `grep -rn "posterior_entropy" src/neurospatial/decoding/` returns **only** genuine symbol references, no sentence words; `grep -rn "PositionOverlay(data=" .claude docs` and `grep -rn "skeleton_color\|skeleton_width" docs src/neurospatial/animation` return nothing; `grep -rn "neurospatial.metrics.phase_precession\|neurospatial.differential\|neurospatial.reference_frames\|neurospatial.segmentation\|write_intervals\|\.mrl()\|peak_view_locations\|var_sin" src docs .claude` returns nothing. |

Mark no test `slow`; doctest + snippet runs are fast under `Agg`. Do **not** add the doctest run to `napari`-marked paths.

## Fixtures

- Snippet fixtures are synthesized **inline in `docs/snippets.yml` `setup:` blocks** (the harness's native mechanism), not in `conftest.py`. Use minimal synthetic data: a small `Environment.from_samples(positions, bin_size=2.0)`, a short `positions`/`times` trajectory, one or two `spike_times` arrays, `headings = heading_from_velocity(...)`, a couple of `object_positions`, and a tiny `fields`/`frame_times` stack. No external files, no NWB, no video — anything needing those keeps a `+SKIP`/`skip:` with a reason.
- Doctests reuse the existing `# doctest: +SKIP` convention for fixture-dependent examples; self-contained corrected examples build their own arrays inline (the `graph_convolve`/`basis.py` examples already construct a 3×3 grid).

## Review

Before opening the PR, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:

- **Examples actually execute** — the new `--doctest-modules` job and the extended `scripts/test_doc_snippets.py` run both pass in CI, and the reviewer has seen green output (evidence, not assertion).
- Every task A–H is applied; each edited snippet matches the *current* source signature (re-verified against `src/`, since phases 14–22 may have shifted line numbers).
- The grep gate finds no surviving drift markers (`data=` overlays, `skeleton_color/width`, `convolve`, `.mrl()`, `peak_view_locations`, `var_sin`, stale module paths, `write_intervals`).
- "Deliberately not in this phase" is honored: no public symbol renamed, no behavior changed, the NWB worked example is only cross-referenced.
- The `posterior_entropy` prose fix did **not** touch the function name/parameter; every remaining `posterior_entropy` is a real identifier.
- New `snippets.yml` entries report `pass` (not `skip`) and their `setup:` blocks are minimal, not copy-pasted bloat; no entry weakens the gate by skipping the very block it claims to cover.
- Tests aren't tautologies — the snippet/doctest assertions exercise real construction and execution paths, not mock echoes.
- Docstrings, manifest ids, and CI step names don't reference this plan or its phase numbers.
