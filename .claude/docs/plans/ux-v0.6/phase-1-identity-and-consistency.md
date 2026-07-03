# Phase 1 — Unit Identity & API Consistency

**Goal:** Make unit identity durable end-to-end, make terminal verbs mean one thing everywhere, and enforce the naming contract. Mostly additive; two documented behavior changes + several deprecations. May split into **1a** (identity) and **1b** (naming/contract) PRs.

**Governing:** [`api-contract.md`](./api-contract.md). Every rename ships a deprecated alias (removed 0.7). Every deprecation gets a test for both old-warns and new-clean.

---

### 1.1 — Thread `unit_ids` through population compute + results
- **Files:** `encoding/_spikes.py` (carry ids), `encoding/spatial.py`, `directional.py`, `view.py`, `egocentric.py` (batch fns + result dataclasses), `events/alignment.py` (`population_peri_event_histogram`).
- **Change:** add optional `unit_ids=None` kwarg to every batch compute (default `np.arange(n)`); add `unit_ids: NDArray` field (+ optional `unit_table: pd.DataFrame | None`) to every population result dataclass, set in `__post_init__`.
- **Frozen-dataclass mechanics (gaps):** result classes are `@dataclass(frozen=True)` (`spatial.py:84,207,760`), so defaulting `unit_ids`/`unit_id` in `__post_init__` must use `object.__setattr__`. Adding `unit_ids: NDArray` also affects `__eq__`/`__hash__` (NumPy arrays are unhashable and compare elementwise) — declare the array fields `field(compare=False)` (and keep results unhashable as today, or define `__eq__` explicitly) so equality/round-trip tests don't break.
- **Identity through indexing (finding):** "survives `__getitem__`/iteration" is **not** automatic — `SpatialRatesResult.__getitem__`/iteration (`encoding/spatial.py:879,911`) returns a single-unit `SpatialRateResult` that currently has **no** id. So also add a singular **`unit_id: int | str | None = None`** field to every *single-unit* result class (`SpatialRateResult`, `DirectionalRateResult`, `ViewRateResult`, `EgocentricRateResult`), and make the batch `__getitem__`/iteration set `unit_id = self.unit_ids[i]` on the child. Codify this in [api-contract.md → Unit identity](./api-contract.md).
- **Tests:** indexing `rates[i].unit_id == rates.unit_ids[i]`; iteration preserves order/labels; `classify`/`subset` keep `unit_ids` aligned.
- **Back-compat:** optional param, default arange → no break.
- **Tests:** ids round-trip from `read_units` → `compute_spatial_rates(..., unit_ids=ids)` → `result.unit_ids`; positional `result[i]` documented as positional; label access via 1.4/`sel`.

### 1.2 — `to_xarray()` → labeled `xr.Dataset` on all batch results  ⚑ behavior change (D1)
- **Files:** `encoding/spatial.py:993` (+ implement on `DirectionalRatesResult`, `ViewRatesResult`, `EgocentricRatesResult`), `decoding/_result.py:462`.
- **Test migration (H3):** update `tests/encoding/test_spatial_xarray_interop.py` (locks integer neuron coords) **and** `tests/decoding/test_xarray_interop.py` + `tests/decoding/test_result.py` (lock decode xarray coords). Add an **xarray-present CI job** — xarray is an optional extra, so the `DataArray`→`Dataset` break is otherwise untested on default installs.
- **Change:** return `xr.Dataset` with **dims `("unit_id", "bin")`**, `unit_id` as the **index/dimension coordinate** (real ids), `bin_center_x`/`bin_center_y` as **non-index coords on `bin`**; decoding adds a `time` dim/index. **Validate `unit_ids` are unique** before indexing (raise a clear error on duplicates). `da.sel(unit_id=…)` then selects by label. See [api-contract.md → to_xarray](./api-contract.md) (finding #5).
- **Transition (D1 = clean break):** 0.6 returns `Dataset`; **no** `to_dataset()` alias, **no** DataArray shim. Loud CHANGELOG `### Breaking changes`.
- **Tests:** `.sel(unit_id=specific)` returns that unit; `bin_center_x` present; dataset attrs carry units/bandwidth/env hash.

### 1.3 — Split terminal verbs; PSTH results get `ResultMixin`  ⚑ behavior change
- **Files:** the canonical `ResultMixin` is **`src/neurospatial/_results.py:45`** (L2 — `encoding/_base.py:162` only defines the `SpatialResultMixin(ResultMixin)` subclass); all batch result classes; `events/_core.py:27,83` (inherit `ResultMixin` from `_results.py`).
- **Test/doctest migration (H3):** the dense-`to_dataframe` break touches **~11 `tests/**` files** referencing `to_dataframe`, plus inline doctests (e.g. `spatial.py:1488-1490` `>>> len(df) == 3`, `>>> "cell_type" in df.columns`). Grep `to_dataframe` across `tests/` and `>>>`-doctests in every result class and update them — the doctest CI job gates the green-CI DoD.
- **Change:** add `summary_table()` (one row/unit, `unit_id`-indexed) to every batch result; make `to_dataframe()` consistently **dense tidy** (one row/(unit,bin), carries `unit_id`). `PeriEventResult`/`PopulationPeriEventResult` inherit `ResultMixin` and implement `to_dataframe`/`summary`/`plot`.
- **Transition (clean break in 0.6 — D1 + DoD #4):** batch `to_dataframe()` **becomes dense tidy immediately** in 0.6 (one row/(unit,bin), carries `unit_id`); per-unit summaries are available **only** via the new `summary_table()`. **No shim** — announced loudly in CHANGELOG `### Breaking changes`. (Resolves the PLAN/DoD contradiction the review flagged: a shim that silently changes a return shape under the same call is worse than a clean, documented break. Renamed *callables* still ship aliases; return-shape changes do not.)
- **Tests:** `summary_table()` is one row/unit (`unit_id`-indexed); `to_dataframe()` is one row/(unit,bin) carrying `unit_id` on both single and batch results; PSTH `to_dataframe()`/`summary()` exist and round-trip.

### 1.4 — Enforce the naming contract
- **`detect_region_crossings` arg order** (`behavior/segmentation.py:321`, current `(position_bins, times, region_name: str, env, *, direction)`): target `(position_bins, times, env, *, region_name, direction)`. A keyword-only `region_name` would reject the old **4-positional** calls before any remap (finding #3), so use a **transitional signature for one release**: `def detect_region_crossings(position_bins, times, arg3, arg4=None, *, region_name=None, direction=...)`. If `arg4 is not None` (old `(…, region_name, env)`) **or** `isinstance(arg3, str)`, treat as old order (`region_name=arg3, env=arg4`), emit `DeprecationWarning`, remap; else `env=arg3` + keyword `region_name`. Collapse to the clean keyword-only signature in 0.7.
- **Batch classifiers (H2 — two verbs, not one):** add `result.classify(*, ...) -> NDArray[bool]` (single-type predicate) on `SpatialRatesResult`/`DirectionalRatesResult`/`ViewRatesResult`/`EgocentricRatesResult`; deprecate the **bool-returning** `detect_hd_cells`/`detect_view_cells`/`detect_ovcs` as aliases of `classify`. Separately, **rename the multi-class `detect_cell_types` (`spatial.py:1303`, returns `NDArray[str]` labels) → `label_cell_types`** with a deprecated alias — do **NOT** fold it into `classify` (different return type; would break `df[col=="place"]`). See [api-contract.md → Cell-type API](./api-contract.md).
- **Add `is_place_cell`** (free in `encoding/spatial.py` + `SpatialRateResult.is_place_cell()`), mirroring `is_spatial_view_cell`.
- **Collapse `peak_view_location` → `peak_location`** (`view.py:257`), alias old; ensure batch accessors are plural (`peak_locations`).
- **`decode_position` accepts a result object:** `encoding_models` may be a `SpatialRatesResult` (duck-typed `.firing_rates`) or an array (`decoding/posterior.py:269`) — removes the `np.stack([r.firing_rate …])` glue.
- **Tests:** old arg-order warns+works; `.classify()` parity with old per-domain methods; `is_place_cell` agrees with `detect_place_fields`; `decode_position(env, counts, rates_result, dt)` works.

### 1.5 — Experiment-shaped factory presets
- **Files:** `environment/factories.py` (+ export). Topology rule per [api-contract.md → Factory presets](./api-contract.md) (finding #6).
- **`Environment.open_field(positions, bin_size, *, …)`** → `from_samples` (flips `fill_holes=True`). The only positions-only preset.
- **`Environment.linear_track(*, endpoints | node_positions, bin_size, …)`** and **`Environment.maze(kind, *, track_graph | node_positions, bin_size, …)`** (`kind ∈ {"w","plus","t"}`) take an **explicit topology spec** (endpoints, node coords, a `networkx` track graph, or an annotation) and delegate to `from_graph`, or wrap the existing `simulation` maze constructors. Positions **cannot** infer track/maze topology (finding #6) — these are not positions-only.
- Demote morphology knobs in `from_samples` docstring into an "Advanced / cleanup" group.
- **Tests:** `open_field` ≡ the documented `from_samples` call; `linear_track`/`maze` build the expected graph (node/edge counts, `is_linearized_track=True`) from an explicit spec; passing positions-only to a track/maze preset raises a clear error.

### 1.6 — Write the naming contract into CLAUDE.md
- Copy [`api-contract.md`](./api-contract.md)'s naming section into `CLAUDE.md` so future domains can't regress it.

### 1.7 — Migration guide + CHANGELOG breaking-change copy
- Draft the `### Breaking changes` CHANGELOG block **and** a short copy-paste migration snippet for the two clean breaks: `to_xarray()` `DataArray`→`Dataset` (`da.sel(neuron=i)` → `da.sel(unit_id=…)`) and batch `to_dataframe()` summary → `summary_table()`. External users hit a *silent* return-shape change, so they need explicit remediation, not just a note.
- **Tests:** the migration snippet itself runs under the doc-snippet harness.

**PR deliverable(s):** 1a (1.1–1.3 identity/verbs) and 1b (1.4–1.7 naming/presets/migration); CHANGELOG `### Breaking changes` + `### Changed` with the deprecation table.
