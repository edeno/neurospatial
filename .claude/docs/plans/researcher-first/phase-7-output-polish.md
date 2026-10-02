# Phase 7 — Output polish: summaries, single/batch parity, and safe animation export

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md#error-message-contract)

**Line numbers** were read on `main` at `da631a47`. Phases 1–6 land first, so re-locate each site by its symbol name. Every item below was **re-verified on `main`** by probe (60 s, 30 Hz trajectory, 3 units, four rate families).

**Inputs to read first:**

- `src/neurospatial/encoding/_base.py:333-390` (`SpatialResultMixin.summary`) and `:465-509` (base `summary_table`). Every rate result inherits both. `__repr__` and `_repr_html_` (`_results.py:499-545`) are built from `summary()`, so `summary()` must stay cheap and must never raise.
- `src/neurospatial/encoding/spatial.py:2205-2362` (`SpatialRatesResult.summary_table`), `:1959-2100` (`label_cell_types`) and `:1627-1717` (`to_xarray`). Also the sibling batch classes:
  - `view.py:976`, `:602`;
  - `egocentric.py:1086`, `:675`;
  - `directional.py:1507`, `:1136`.
- `src/neurospatial/_results.py:323` — `build_population_dataset`, the shared xarray constructor that the four batch `to_xarray` methods already call.
- `src/neurospatial/animation/core.py:138-556` (dispatcher); `animation/backends/video_backend.py:78, :296, :387` (`render_video`, `dry_run`, `"-y"`); `animation/backends/html_backend.py:446, :737-751, :795, :840` (`render_html`, the `animation.html` default, `frames_dir`, `write_text`); `environment/visualization.py:562` (`Environment.animate_fields`).
- `src/neurospatial/io/files.py:297-303` — the `overwrite=False` guard and wording to reuse.
- **What earlier phases changed in these files** (search by symbol):
  - Phase 5 renamed `EgocentricRateResult`/`EgocentricRatesResult` to `ObjectVectorRateResult`/`ObjectVectorRatesResult` and added `compute_object_vector_rate(s)`. "Egocentric" below means the object-vector family, in both frames.
  - Phase 5 (5.5) fixed `label_cell_types` so every label requires `spatial_info >= min_spatial_info`. Phase 5 (5.4) made threshold keywords default to `None`, resolved from one private default constant per family. Reuse those constants here; do not define new ones.
  - Phase 5 keeps every rate result free of its inputs: shuffle significance is the separate `*_significance` free functions, so tables have no significance columns. Phase 5 also renamed field detection to `has_place_field`; `is_place_cell` now requires `criterion=`.
  - Phase 3a added `spike_window` (field) and `spike_window_assumed` (property) to every rate result, reported in `summary()` and in the batch `to_xarray()` attrs. 7.1 and 7.2 must keep both.
  - Phase 1 Task 9 changed the directional `to_xarray` attrs (`bandwidth` only when not None), and Phase 1 Task 11 rewrote `tests/animation/test_video_backend.py`'s parallel-render test.
  - Phase 4 added `_format_error` and rewrote the `io/files.py:299` message with a `Fix:` line. `check_writable` (7.4) replaces that site and keeps its wording.
  - `CHANGELOG.md`: append after the earlier phases' sections.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract): the new `FileExistsError` and `ValueError` messages have `Fix:` lines built with Phase 4's `_format_error`.
- [API snapshot](shared-contracts.md#api-snapshot): run `tests/test_public_api_snapshot.py`. `animate_fields` and result methods are not in any `__all__`, so no snapshot change is expected; if one appears, regenerate it and explain it in the PR.

**Designs referenced:** none.

## Verified on `main` (what this phase fixes, and what it drops)

| Archive-review claim | On `main` | Action |
| --- | --- | --- |
| Batch `total_occupancy=299.9` sums occupancy over units | **False.** All four batch classes store one shared `(n_bins,)` occupancy; `total_occupancy == 59.97` for a 60 s session. | Dropped. Add a guard test that pins the shared-occupancy semantics. |
| Batch `peak_firing_rate` is a max over units but shares its name with the singular per-unit value | True for all four families (`_base.py:376`) | 7.1 |
| Singular `summary_table()` lacks the batch metrics | True for all four: spatial has 4 columns vs 9, view 4 vs 6, egocentric 3 vs 5, directional 2 vs 7 | 7.2 |
| `print(R.summary_table())` hides `spatial_info` behind `...` | True (9 columns at 80-character width) | 7.3 |
| `cell_type` is shown without its thresholds | True; `df.attrs` is empty | 7.3 |
| Singular `to_xarray` raises a bare `AttributeError` | True for all four singular classes | 7.2 |
| `summary_table(unit_ids=...)` raises "unexpected keyword" | **False**: it works on all four batch classes on `main`. | Dropped. The bare length-mismatch message gets its `Fix:` line in Phase 4. |
| Video and HTML export overwrite existing files | True: ffmpeg `-y` (`video_backend.py:387`); unconditional `write_text` (`html_backend.py:795, :840`); and the `None` → `"animation.html"` default lands in the working directory | 7.4 |
| Empty ledger fields (`unavailable_unit_ids=()`) | Archive only; there are no such fields on `main` | Dropped |

## Tasks

### 7.1 Population summary keys say what was aggregated

In `SpatialResultMixin.summary` (`_base.py:333-390`), batch results report:

- `n_units` (renamed from `n_neurons`);
- `n_bins`;
- `max_peak_firing_rate` (renamed from `peak_firing_rate`; the max over units);
- `total_occupancy` (now documented as "seconds in the occupancy map shared by all units, *not* summed over units");
- `method` when present.

Singular results keep `peak_firing_rate` and add the cheap headline metrics through a hook:

```python
def summary(self) -> dict[str, Any]:
    rates = _to_numpy(self._get_rates())
    occupancy = _to_numpy(self.occupancy)
    peak = float("nan") if rates.size == 0 else float(np.nanmax(np.asarray(self.peak_firing_rate())))
    out: dict[str, Any] = {}
    if rates.ndim > 1:
        out["n_units"] = int(rates.shape[0])
    out["n_bins"] = int(rates.shape[-1])
    out["max_peak_firing_rate" if rates.ndim > 1 else "peak_firing_rate"] = peak
    if rates.ndim == 1:
        out.update(self._headline_metrics())
    out["total_occupancy"] = float(np.nansum(occupancy))
    if hasattr(self, "method"):
        out["method"] = self.method
    out["spike_window_assumed"] = self.spike_window_assumed
    out["spike_window"] = None if self.spike_window is None else self.spike_window.tolist()
    return out

def _headline_metrics(self) -> dict[str, float]:
    """O(n_bins) per-unit scalars shown in the repr; subclasses extend."""
    return {}
```

Overrides of `_headline_metrics`, each O(n_bins) and NaN-safe (an all-NaN map gives `nan`, never an exception). Grid and border scores are deliberately excluded because they are too slow for a repr.

- `SpatialRateResult`: `spatial_info`, `sparsity`.
- `ViewRateResult`: `view_spatial_info`.
- `ObjectVectorRateResult`: `preferred_distance`, `preferred_direction`.
- `DirectionalRateResult`: `preferred_direction`, `mean_vector_length`.

Update the `summary` docstring example (`_base.py:362-364`) and `tests/encoding/test_encoding_base.py:734-736` (`n_neurons` → `n_units`).

### 7.2 One column builder for single and batch results, plus a singular `to_xarray`

- **Spatial.** Move the body of `SpatialRatesResult.summary_table` into a module-level builder that works on `np.atleast_2d(rates)`, so one implementation serves both classes. Use the free `_metrics.batch_*` functions that the batch `spatial_information`/`sparsity`/`grid_scores`/`border_scores` methods already delegate to. Do not construct a one-unit `SpatialRatesResult`: a singular `method="glm"` slice of a lone fallback unit fails `_check_gam_result_invariant` (`spatial.py:406-411`).

  ```python
  def _spatial_summary_frame(
      result: SpatialRateResult | SpatialRatesResult,
      *,
      index: Sequence[Hashable],
      include_classification: bool,
  ) -> pd.DataFrame:
      """Per-unit summary columns shared by the single and batch spatial results."""
      import pandas as pd
      from neurospatial.encoding._metrics import (
          batch_border_scores, batch_grid_scores, batch_sparsity, batch_spatial_information,
      )

      rates = np.atleast_2d(_to_numpy(result._get_rates()))
      occupancy = _to_numpy(result.occupancy)
      peaks = np.atleast_2d(result.peak_location())
      spatial_info = np.asarray(batch_spatial_information(rates, occupancy))
      grid = batch_grid_scores(result.env, rates).scores
      border = batch_border_scores(result.env, rates).scores  # same defaults as SpatialRatesResult.border_scores
      columns: dict[str, Any] = {
          "peak_rate": np.atleast_1d(result.peak_firing_rate()),
          "spatial_info": spatial_info,
          "sparsity": np.asarray(batch_sparsity(rates, occupancy)),
          "grid_score": grid,
          "border_score": border,
          "peak_x": peaks[:, 0],
          "peak_y": peaks[:, 1] if peaks.shape[1] > 1 else np.full(len(rates), np.nan),
          **_glm_summary_columns(result),  # the existing spatial.py:2346-2360 block; scalars broadcast
      }
      if include_classification:
          columns["cell_type"] = _label_from_scores(spatial_info, grid, border, **PLACE_GRID_BORDER_THRESHOLDS)
      df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
      df.attrs["method"] = result.method
      df.attrs["units"] = {"peak_rate": "Hz", "spatial_info": "bits/spike",
                           "peak_x": result.env.units or "", "peak_y": result.env.units or ""}
      if include_classification:
          df.attrs["classification_thresholds"] = dict(PLACE_GRID_BORDER_THRESHOLDS)
      return df
  ```

- **Labeling rule and thresholds.** `_label_from_scores` is the labeling rule from `label_cell_types` (`spatial.py:2089-2100`), extracted **exactly as it stands after Phase 5**; `label_cell_types` calls it too. `PLACE_GRID_BORDER_THRESHOLDS = {"min_spatial_info": 0.5, "min_grid_score": 0.4, "min_border_score": 0.5}` is Phase 5's private default constant for `label_cell_types`. Its `None` threshold keywords resolve from it, so the attrs and the defaults cannot drift. Use Phase 5's constants for view (`min_info=0.5`), object-vector (`min_info=0.3`) and directional (`min_mvl=0.4, alpha=0.05`) the same way. Defaults are unchanged (decision 5).
- **Both `summary_table`s become thin wrappers.** `SpatialRatesResult.summary_table(unit_ids=None, include_classification=True)` validates the `unit_ids` length (keep that check), then calls the builder. `SpatialRateResult` gains `summary_table(include_classification=True)`, which calls the builder with `index=self._row_unit_ids()`.
- **The other three families.** Apply the same pattern: `_view_summary_frame`, `_egocentric_summary_frame` and `_directional_summary_frame`, each built from the current batch body (`view.py:976`, `egocentric.py:1086`, `directional.py:1507`) and computing from `np.atleast_2d` rates. The singular classes stop inheriting the 3-column base table. Then delete the base `summary_table` (`_base.py:465-509`), since nothing uses it anymore; check `grep -rn "summary_table" src/` first.
- **One `to_xarray` for all eight classes.** Move it into `SpatialResultMixin` and delete the four batch copies:

  ```python
  def to_xarray(self) -> Any:
      """Labeled ``xr.Dataset`` with dims ``("unit_id", "bin")``; one unit for a singular result."""
      from neurospatial._results import build_population_dataset

      rates = np.asarray(_to_numpy(self._get_rates()), dtype=np.float64)
      env = getattr(self, "env", None)
      return build_population_dataset(
          np.atleast_2d(rates),
          self._row_unit_ids(),
          env=env,
          bin_centers=None if env is not None else np.asarray(self.bin_centers, dtype=np.float64),
          occupancy=np.asarray(_to_numpy(self.occupancy), dtype=np.float64),
          attrs=self._xarray_attrs(),
      )
  ```

  Each family defines `_xarray_attrs()` once, as a module-level function used by both of its classes. Move each batch class's current `attrs` dict in unchanged, for example the spatial `units_attr`/`method`/`env`/`software_version`/`bandwidth` block at `spatial.py:1697-1710`. That includes Phase 3a's `spike_window_assumed` (an int) and `spike_window` (a flat array, present only when set), so a singular result now carries them too. A standalone singular result with `unit_id=None` gets a single `<NA>` unit coordinate. Document this in the docstring: pass `unit_id` or index a batch result if you need a NetCDF-safe label.

### 7.3 Readable tables: column order, units and visible thresholds

- **Column order.** Pandas truncates the *middle* columns, so every builder orders its columns: rate and information first, coordinates and secondary scores in the middle, the classification column **last**. `method` moves from a column to `df.attrs["method"]`, since it is constant per table.
  - Spatial: `peak_rate, spatial_info, sparsity, grid_score, border_score, peak_x, peak_y, [glm…], cell_type`.
  - View: `peak_rate, view_spatial_info, peak_x, peak_y, is_spatial_view_cell`.
  - Egocentric: `peak_rate, preferred_distance, preferred_direction_deg, preferred_direction, is_object_vector_cell`.
  - Directional: `peak_rate, mean_vector_length, preferred_direction_deg, tuning_width_deg, preferred_direction, tuning_width, is_head_direction_cell`.

- **Attrs.** `df.attrs["units"]` and `df.attrs["classification_thresholds"]` are set on every table (see 7.2).
- **Docstrings.** Each `summary_table` docstring gains a Notes paragraph: "Labels are fixed-threshold heuristics; the thresholds used are in `df.attrs['classification_thresholds']`. For shuffle significance, call the family's `*_significance` function with the raw arrays." Point to Phase 5's bias notes; don't duplicate them.

### 7.4 `overwrite=False` for animation export

- **Shared helper.** Add `check_writable(path, *, overwrite, what, argument)` to `src/neurospatial/_validation.py` and use it at `io/files.py:297-303`, so the environment writer and the animation backends share one implementation and one wording:

  ```python
  def check_writable(paths: Sequence[Path], *, overwrite: bool, what: str, argument: str) -> None:
      """Raise FileExistsError if any of ``paths`` exists and ``overwrite`` is False.

      A directory counts as existing only when it is non-empty.
      """
      if overwrite:
          return
      existing = [p for p in paths if (p.is_dir() and any(p.iterdir())) or p.is_file()]
      if existing:
          raise FileExistsError(_format_error(
              f"Refusing to overwrite existing {what}: {', '.join(map(str, existing))}.",
              fix=f"pass overwrite=True to replace it, or choose a different {argument}.",
          ))
  ```

- **Signatures.** Add `overwrite: bool = False` as an explicit keyword to `Environment.animate_fields` (`visualization.py:562`), the dispatcher `animate_fields` (`animation/core.py:138`), `render_video` (`video_backend.py:78`) and `render_html` (`html_backend.py:446`). The dispatcher passes it only to those two backends; napari and widget never receive it.
- **Where the check runs.** At the top of each backend, after `save_path` defaults are resolved and **before** the `dry_run` estimate (`video_backend.py:296`) or any frame rendering:
  - `render_video` checks `[Path(save_path)]`;
  - `render_html` checks `[output_path]`, plus `frames_dir` (default `output_path.with_suffix("")`, `html_backend.py:746-750`) when `embed=False`.

  A dry run on an existing target therefore raises instead of reporting an estimate for a render that would fail.
- **ffmpeg flag.** In the command at `video_backend.py:386-387`, replace `"-y"` with `"-y" if overwrite else "-n"`. `-n` makes ffmpeg refuse to overwrite even if the file appears between the check and the encode.

### 7.5 User-facing documentation

- `CHANGELOG.md` `[Unreleased]` (breaking):
  - the batch `summary()` keys `n_neurons`→`n_units` and `peak_firing_rate`→`max_peak_firing_rate`;
  - the singular `summary()` and `summary_table()` gaining metrics;
  - `method` moving to `df.attrs`;
  - the new column order;
  - `df.attrs["classification_thresholds"]`;
  - singular `to_xarray()`;
  - `animate_fields(overwrite=False)` (re-running a script with the same `save_path` now raises `FileExistsError`).
- `CLAUDE.md` "v0.6 API Naming Contract":
  - `to_xarray()` exists on every rate result, singular ones included, with a one-unit `unit_id` axis;
  - singular and batch `summary_table()` have identical columns;
  - `method`, units and thresholds live in `df.attrs`;
  - pattern 3 gains `overwrite=True` in the video call.
- `docs/getting-started/quickstart.md` "Inspect and plot the result": update the `result.summary()` comment to list `spatial_info`.
- `docs/user-guide/animation.md`: an "Overwriting existing files" note.
- The `animate_fields`, `render_video` and `render_html` docstrings: document `overwrite` and `Raises FileExistsError`.

## Deliberately not in this phase

- **The cell-type labeling rule.** On `main`, `label_cell_types` (`spatial.py:2089-2100`) applies the spatial-information gate only to `"place"`, so units with uniformly random spikes are labelled `"border"`, which contradicts its docstring. Phase 5 (5.5) fixes it. This phase extracts the fixed rule into `_label_from_scores` and only makes the thresholds visible.
- Shuffle significance columns. Significance is a separate free computation on raw arrays (Phase 5), not something a result can compute. Classification defaults are also out (decision 5).
- A multi-line `__str__`, NaN-bin counts in the repr, resolved-default display, and rounding of float noise such as `peak_x=37.999995`. These are possible follow-ups, not regressions.
- Colorblind overlay defaults, `tqdm.auto`, ffmpeg progress, and automatic `clear_cache()`. These are animation UX items with no correctness impact.
- `summary_table(unit_ids=)` and the bare length-mismatch message: the first works on `main`, and Phase 4 handles the second.
- Removing the deprecated `SpatialRatesResult.detect_cell_types` alias (Phase 5 removes it, decision 1).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/encoding/test_result_summaries.py::test_population_summary_names_aggregation` (parametrized over the 4 families) | `R.summary()` keys are exactly `{"n_units", "n_bins", "max_peak_firing_rate", "total_occupancy", "spike_window_assumed", "spike_window"}`, plus `"method"` for spatial and view. `n_units == 3`; `max_peak_firing_rate == np.nanmax(R.peak_firing_rate())`. `"n_neurons" not in repr(R)`. |
| `…::test_total_occupancy_is_shared_not_summed` | For spatial and directional on the 60 s at 30 Hz fixture: `R.occupancy.ndim == 1` and `R.summary()["total_occupancy"] == pytest.approx(59.967, abs=1e-3)`, which equals `R[0].summary()["total_occupancy"]`. |
| `…::test_singular_summary_headline_metrics` | `R[0].summary()["spatial_info"] == pytest.approx(R[0].spatial_information())`, and similarly for the view, egocentric and directional keys. `repr(R[0])` contains `spatial_info=`. On an all-NaN map, `summary()` returns NaN values without raising. |
| `…::test_summary_table_single_matches_batch` (4 families, plus spatial with `method="glm"`) | `pd.testing.assert_frame_equal(R[i].summary_table(), R.summary_table().iloc[[i]])` for i in 0..2, and `R[i].summary_table().attrs == R.summary_table().attrs`. |
| `…::test_summary_table_column_order_and_attrs` | Spatial columns equal the 7.3 list exactly; `"method" not in df.columns`. `df.attrs["classification_thresholds"] == {"min_spatial_info": 0.5, "min_grid_score": 0.4, "min_border_score": 0.5}`. `"spatial_info"` and `"cell_type"` are both in `df.to_string(max_cols=6)`. |
| `…::test_thresholds_attrs_match_resolved_defaults` | For each family, the attrs values equal Phase 5's default constant, and `classify()` / `label_cell_types()` with no threshold keywords equal the same calls with the constant's values passed explicitly. (The signature defaults are `None` since Phase 5.) |
| `tests/encoding/test_spatial_xarray_interop.py::test_singular_to_xarray_matches_batch_row` (4 families) | `xr.testing.assert_identical(R[i].to_xarray(), R.to_xarray().isel(unit_id=[i]))`. A standalone `compute_spatial_rate(...)` result gives `ds.sizes == {"unit_id": 1, "bin": env.n_bins}`. This file already runs in the `test_xarray.yml` job. |
| `tests/animation/test_overwrite.py::test_video_refuses_existing_file` | With a pre-existing `out.mp4` (content `b"keep"`), `env.animate_fields(fields, frame_times=ft, backend="video", save_path=p)` raises `FileExistsError` matching `"overwrite=True"` and leaves the bytes unchanged. `dry_run=True` raises too. With `overwrite=True` it succeeds and the file differs. Requires ffmpeg: skip if it is absent. |
| `…::test_html_refuses_existing_file_and_frames_dir` | The same checks for `.html`. With `embed=False`, a non-empty `frames_dir` raises even when the HTML file is absent; an empty one does not. |
| `…::test_check_happens_before_rendering` | Monkeypatch `neurospatial.animation._parallel.parallel_render_frames` (video) and `neurospatial.animation.rendering.render_field_to_image_bytes` (html) to raise `AssertionError`. The `FileExistsError` is raised first. |
| `…::test_env_to_file_wording_shared` | `env.to_file(p)` twice: the second raises `FileExistsError` containing `"Fix: pass overwrite=True"`. That is the same helper the animation tests hit. |
| `…::test_ffmpeg_never_overwrites_without_opt_in` | Spy on `subprocess.run`: the ffmpeg command contains `"-n"` and not `"-y"` when `overwrite=False`, and `"-y"` when `True`. |

Mark the `method="glm"` parity parametrization `slow` only if it takes > 5 s on the fixture; measure it first. The other tests use the 3-unit, 60 s fixture and run in about 1 s each.

## Fixtures

- In `tests/encoding/conftest.py` (or reuse an existing equivalent): a module-scoped `rate_family_results` fixture. It builds:
  - the 60 s at 30 Hz Lissajous trajectory (`50 + 40·[sin(2πt/20), cos(2πt/13)]` cm);
  - `env = from_samples(positions, bin_size=4.0, units="cm")`;
  - headings from the velocity angle;
  - one object at `(30, 30)`;
  - three seeded uniform spike trains of 200, 50 and 300 spikes;
  - the four batch results (`compute_spatial_rates`, `compute_view_rates`, `compute_egocentric_rates`, `compute_directional_rates`).

  The same arrays are the ones used in the verification probes above.
- Animation tests: a 5-frame `fields` array on that env with `frame_times = arange(5) / 30`, written under `tmp_path`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
