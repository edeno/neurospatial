# v0.6 API & Design Contract (shared by all phases)

This is the rulebook every phase adheres to. After v0.6 it is copied into `CLAUDE.md` so new domains can't regress it.

## Design principles (Scientific-Python guide + maintainer refinements)

1. **Array-first, standard types.** Raw NumPy arrays are the universal input/output contract. Every public function works with plain arrays. Prefer `xarray.Dataset` and frozen dataclasses-of-arrays over new bespoke container classes. Metadata (unit ids, units, region labels) rides *alongside* arrays (as params, dataclass fields, or xarray coords), never locked inside a custom class.
2. **Return-type stability.** A function returns the same type regardless of optional args. No mode flags that change the return shape (no `return_posterior="full"|"map"`, no `to_dataframe(kind=...)`). Vary *computation/availability*, not *type*; or split into separately-named methods/functions.
3. **Duck typing / Protocols, not `isinstance`.** Accept any object exposing the needed attributes (define `SpikeTrainsLike`, `PositionLike`, etc.). Convert third-party objects (pynapple) to arrays at the boundary. Optional deps stay optional.
4. **Layered:** thin friendly layer (result objects, presets, decoder, bundle) over a strict array-core; every result-object computation is *also* reachable as a function on arrays.
5. **Immutable:** result objects and the `Session` bundle are `frozen`; "modify" returns a new object.
6. **Errors:** raise (don't print); `warnings.warn` for likely-mistakes (never `print`); messages name the parameter, the received vs expected value/shape/type, and the fix; no permissive defaults where stricter is clearer; no internal-doc (`CLAUDE.md`) references in user-facing errors.
7. **Keyword-only** for numerical params and flags (after `*`).
8. **I/O separation:** `io/` performs I/O and returns standard types; scientific modules never import `pynwb`/`pynapple`.

## Naming contract (enforced across all result classes & domains)

### Argument order
- **Spatial encoding** functions: `func(env, spike_times, times, positions, headings?, object_positions?, *, ...)` — env first.
- **Directional encoding** (documented exception): `func(spike_times, times, headings, *, ...)` — no `env` (heading is angular).
- **Behavioral segmentation:** `func(position_bins, times, env, *, region_params...)` — env in slot 3, region names keyword-only after `*`.
- **Egocentric ops:** `func(positions, headings, targets)`.

### Cell-type API (one learnable rule)
- **Single-neuron predicate:** `is_<celltype>_cell(...)` as a free function AND a result method. Cell types: `place`, `head_direction`, `object_vector`, `spatial_view`, `border`, `grid` (where applicable). **Add the missing `is_place_cell`.**
- **Batch classification:** `result.classify(*, ...) -> NDArray[bool]` on *every* plural result class. Old per-domain names (`detect_cell_types`, `detect_ovcs`, `detect_view_cells`, `detect_hd_cells`) become deprecated aliases. Never ship unspelled acronyms.

### Peak / preferred accessors
- Cartesian peak location: `peak_location()` (single) / `peak_locations()` (batch). **Collapse `peak_view_location` → `peak_location`.**
- Genuinely non-Cartesian peaks keep domain names: `preferred_direction()`, `preferred_distance()` (angle/radius), plural for batch.

### Terminal verbs (identical name + semantics on every result class)
- `to_dataframe()` → **dense tidy**, one row per `(unit, bin)` (or `(field, bin)`), always carrying a `unit_id` column. For plotting / detailed inspection.
- `summary_table()` → **one row per unit**, `unit_id`-indexed, scalar metric columns. The default a 1000-neuron user wants.
- `to_xarray()` → labeled `xr.Dataset`. Precise index design (so `.sel(unit_id=…)` works): population rate datasets use **dims `("unit_id", "bin")`** with `unit_id` as the **dimension/index coordinate** (the `unit_ids` array) and `bin` as the bin-index dimension; `bin_center_x`/`bin_center_y` are non-index coords *on* the `bin` dim; decoding adds a `time` dim/index coord. **Validate `unit_ids` are unique** before building the index (raise a clear error on duplicates, since label selection requires uniqueness). Attrs carry units/bandwidth/env-hash/software-version.
- `summary()` → flat dict of scalar headline metrics.
- `plot(ax=None, ...) -> Axes`.
- **PSTH results** (`PeriEventResult`, `PopulationPeriEventResult`) gain `ResultMixin` and these verbs.

### Unit identity
- `unit_ids: NDArray` is a field on every population result (default `np.arange(n)` via `__post_init__`). Threaded from `read_units`/`SpikeTrains` through `compute_*` into results and into xarray coords.
- Single-unit results carry a singular `unit_id: int | str | None = None`. Indexing/iterating a batch result (`rates[i]`, `for r in rates`) **sets the child's `unit_id = unit_ids[i]`**, so an indexed single-unit result keeps its label (it is not dropped at `__getitem__`).
- Optional `unit_table: pd.DataFrame | None` (region, quality, depth, inclusion) rides alongside where available.

### Factory presets (experiment vocabulary over `from_*`)
- **`Environment.open_field(positions, bin_size, *, ...)`** — positions-based, delegates to `from_samples` (flips `fill_holes=True`). This is the only positions-only preset.
- **Track/maze presets need an explicit topology** — positions alone cannot infer a linear/W/plus/T graph. `Environment.linear_track(*, endpoints | node_positions, bin_size, ...)` and `Environment.maze(kind, *, track_graph | node_positions, bin_size, ...)` take a topology spec (endpoints, node coords, a `networkx` track graph, or an annotation file) and delegate to `from_graph` — or wrap the existing `simulation` maze constructors. They do **not** accept positions as the sole argument. Each preset's docstring names the `from_*`/constructor it delegates to and the defaults it flips.

## Deprecation mechanics

- Renames/reorders: keep old callable/param as an alias that emits `DeprecationWarning("X is deprecated since 0.6, use Y; removed in 0.7")` and forwards. For `detect_region_crossings` arg-order, detect the old order (3rd positional is `str`) and warn+remap.
- Behavior changes (`to_xarray` → `Dataset`, batch `to_dataframe` → dense): **clean break in 0.6** (D1 + DoD #4), no transition shim — announced in CHANGELOG `### Breaking changes`. See Phase 1.
- Every deprecation: a test asserting (a) old form still works + warns, (b) new form is warning-free.
