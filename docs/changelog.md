# Changelog

All notable changes to neurospatial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Corrected

- **`smoothing_method="binned"` is not a dense-kernel memory mitigation.**
  Earlier notes (through 0.6.0) recommended `binned` to avoid the dense
  `(n_bins, n_bins)` kernel, but `binned` smooths its rate map through the same
  diffusion kernel (via `env.smooth`), so all smoothing methods build a dense
  kernel. The only memory mitigation is fewer bins (a larger `bin_size`).

### UX hardening — fail-loud errors, one count convention, docs & viewer polish (0.8.0)

A broad usability pass: silent failures now raise or warn with actionable
messages, a few APIs were made consistent, and the docs/viewers were polished.

#### Breaking Changes

- **`add_positions` is now `add_positions(events, *, times, positions)`** —
  reordered to the canonical `(data, times, positions)` order and made
  **keyword-only** (a bare 1-D trajectory made a positional swap
  shape-indistinguishable). *Migration:*
  `add_positions(events, times=times, positions=positions)`.
- **Assembly functions take `(n_time_bins, n_neurons)`** — `detect_assemblies`,
  `assembly_activation`, `pairwise_correlations`, `reactivation_strength` now
  match `decode_position` / `bin_spikes_in_time`. *Migration:* feed the default
  `bin_spikes_in_time(...)` output as-is, or transpose an old
  `(n_neurons, n_time_bins)` matrix.
- **Behavior result `summary()` returns a `dict`** (was a formatted string).
  *Migration:* use `str(result)` for the human-readable form.
- **Fail-loud instead of silent sentinels** — `bin_at` raises on wrong-dimension
  points on a graph/track env; `heading_from_velocity` raises when every sample
  is below `min_speed` (opt out with `allow_all_nan=True`); graph queries reject
  a multi-point coordinate batch.

#### Added

- **`units=` / `frame=` keyword args** on `from_samples`, `open_field`,
  `linear_track`, and `maze` set the metadata at construction.
- **Graph queries accept coordinates** (not just bin indices) — `neighbors`,
  `path_between`, `reachable_from` map a coordinate via `bin_at`.
- **`ResultMixin` on every result class** — concise repr/HTML and a scalar
  `summary()` dict.
- `dir(neurospatial.ops)` surfaces the lazily-exported ops (autocomplete).

#### Fixed

- **`min_occupancy` thresholds raw occupancy seconds, not smoothed density** —
  fixes silent all-zero place fields; consistent across all smoothing methods
  (single and batch); `min_occupancy=0.0` unchanged.
- **Grid allocation is preflighted** — a transposed `(2, N)` trajectory raises
  fast instead of OOMing; a likely-transposed array warns rather than
  false-rejecting a valid low-sample N-D environment.
- Non-finite query rows map to the `-1` sentinel per row (one NaN no longer
  empties the whole map).
- `to_file` no-overwrite default; `occupancy()` shape-before-monotonicity;
  `from_graph` edge-`distance` validation.
- `SpatialRateResult.plot()` colorbar docs corrected + "Firing Rate (Hz)" label;
  actionable batch `plot()` error.
- **Interactive-viewer accessibility** — HTML player keyboard-focus guard,
  `:focus-visible`, aria-live toggled off during autoplay; napari region dock and
  track-builder layout/labels.

#### Documentation

- Version bumped to 0.8.0; dead links and fork-clone fixed; value-first runnable
  quickstart; grouped/collapsible nav; new advanced architecture page; `viridis`
  defaults; example-notebook links normalized + internal link-check in docs CI.

## [0.6.0] - 2026-07-03

### Breaking Changes

- `to_xarray()` now returns a labeled `xarray.Dataset` instead of an
  `xarray.DataArray`.
  - Population rate results (`SpatialRatesResult`, `DirectionalRatesResult`,
    `ViewRatesResult`, `EgocentricRatesResult`) use dims `("unit_id", "bin")`.
    The rate matrix lives in the `firing_rate` data variable and `unit_id`
    stores real per-unit identity labels.
  - Decode results (`DecodingResult`) use dims `("time", "bin")`; they have no
    `unit_id` axis.
  - Duplicate `unit_ids` raise `ValueError` because label-based xarray
    selection requires unique labels.
- Batch encoding `to_dataframe()` is now dense tidy: one row per `(unit, bin)`,
  always carrying `unit_id`, bin-center coordinates, `firing_rate`, and
  `occupancy`.
- Per-unit metric tables moved to `summary_table()`, which is indexed by
  `unit_id`. The old `neuron_ids=` relabeling keyword is replaced by
  `unit_ids=` on `summary_table()`.

### Added

- Real unit identity on population results via `unit_ids`, plus singular
  `unit_id` on indexed/iterated single-unit results.
- `summary_table()` on batch encoding results and population PSTH results.
- Experiment-shaped environment presets:
  `Environment.open_field(...)`, `Environment.linear_track(...)`, and
  `Environment.maze(...)`.
- `SpatialRatesResult.label_cell_types()` for multi-class labels, with
  `SpatialRatesResult.classify()` reserved for the boolean place-cell predicate.
- `decode_position()` accepts population rate result objects directly, and
  preserves their dtype (a `float32` rate result decodes in `float32` rather
  than being promoted back to `float64`).
- Memory-safe summary decoding for long sessions. `decode_position_summary`
  returns a new `DecodingSummary` result that streams over time and reduces
  each block to per-time scalars/vectors (`map_position` / `map_bin`,
  `mean_position`, `posterior_entropy`, `peak_prob`) without ever
  materializing the full `(n_time, n_bins)` posterior. `decode_session_summary`
  is the matching one-call encode -> bin -> decode wrapper, and now also
  streams the time-binning so the full count matrix is never materialized
  either. `DecodingSummary` carries the standard terminal verbs
  (`to_dataframe()`, `summary()`, `plot()`, `to_xarray()`).
- `decode_session()` and `decode_session_summary()` gain a keyword-only `dtype`
  parameter (`np.float32` / `np.float64`, default `np.float64`) that is honored
  **end-to-end** — a single `dtype=np.float32` controls both the encoding-model
  working set and the posterior, halving the decode working set on the golden
  path (no silent promotion back to `float64`). Default `np.float64` leaves
  every existing caller byte-for-byte unchanged; any other dtype raises
  `ValueError`.
- `decode_position()` gains keyword-only `dtype` (`np.float32` / `np.float64`,
  default `np.float64`) and a **hybrid** `time_chunk`. `time_chunk=None` (the
  default) keeps the full-matmul path byte-for-byte unchanged;
  `decode_position(time_chunk=N)` computes the Poisson likelihood blockwise
  directly into the preallocated posterior, cutting the transient peak to ~1×
  over the returned posterior (tolerance-equal to the default, not byte-exact).
  `decode_session()` forwards both.
- The summary decoders (`decode_position_summary` / `decode_session_summary`)
  **reject `time_chunk=None`** (raising a clear `ValueError`): a `None` value
  would set the streaming block to the full session length and materialize the
  full `(n_time, n_bins)` posterior, defeating their never-materialize contract.
  `time_chunk` must be a positive integer (default `1024`); use
  `decode_position` / `decode_session` if you want the full posterior.
- `compute_spatial_rates()` gains a keyword-only `dtype` (`np.float32` /
  `np.float64`, default `np.float64`) to halve the memory of stored rate maps.
- Speed filtering on the encode path: `compute_spatial_rate` /
  `compute_spatial_rates` gain keyword-only `speed` / `min_speed` (forwarded by
  **both** `decode_session` and `decode_session_summary`). When `min_speed` is
  set, **one shared per-interval speed gate** filters **both** the spike
  numerator and the occupancy denominator, so a `min_speed` knob can never bias
  firing rates by filtering only one side.
- `population_coverage()` gains a keyword-only `n_jobs` parameter to
  parallelize per-neuron coverage; results are identical regardless of
  `n_jobs`.

### Fixed

- `decode_session` / `decode_session_summary` now **validate `dt`** (finite,
  `> 0`; non-numeric and `bool` rejected) up front with a clear `ValueError`,
  matching `bin_spikes_in_time`. Previously the shared decode-grid builder
  bypassed that guard, so an invalid `dt` leaked a cryptic downstream error
  (`dt=0` -> `ZeroDivisionError`, `dt=NaN` -> "cannot convert float NaN to
  integer", `dt<0` -> a misleading "span smaller than one bin" message).
- `bin_spikes_in_time` now **validates `dt` consistently** via the same shared
  helper: a non-numeric `dt` (including a numeric string like `"0.1"`) and a
  `bool` (`dt=True`) now raise a clear `ValueError` ("dt must be a finite
  number > 0, ..."). Previously a numeric string leaked a raw `TypeError` and
  `dt=True` was silently accepted as a chunk size of `1`.
- An **unparseable `dtype`** (e.g. `dtype="bogus"`) now raises a clear
  `ValueError` naming `dtype` across the decode/encode entry points
  (`decode_position`, `decode_position_summary`, `compute_spatial_rates`,
  `decode_session`, `decode_session_summary`) instead of a raw NumPy
  `TypeError: data type 'bogus' not understood`.
- `decode_position` now **preserves `float32`** when handed a rate-result
  object (anything exposing `.firing_rates`). Previously the friendly object
  path promoted a `float32` `.firing_rates` to `float64`, silently losing part
  of the `dtype=np.float32` memory win the raw-array path already delivered.
  The object path now matches the raw-array path byte-for-byte: `float32` stays
  `float32`, `float64` stays `float64`, an integer rate map is promoted to
  `float64`, and a `None` / dict / non-2-D `.firing_rates` still raises the
  same clear `ValueError`.
- `time_chunk` is now **validated as a positive integer (not `bool`)** across
  `normalize_to_posterior`, `decode_position`, `decode_position_summary`, and
  `decode_session_summary`, raising a clear `ValueError` naming the value and
  its type. Previously a float (`1.5`) or string (`"2"`) leaked a raw
  `TypeError`, and `True` was silently accepted as a chunk size of `1`.

### Changed

- **Firing-rate numerator/denominator alignment (behavior change for gappy /
  out-of-bounds data).** `compute_spatial_rate` / `compute_spatial_rates` gain
  a keyword-only `max_gap` (default `0.5 s`); spikes inside large tracking gaps
  and out-of-bounds excursions are now excluded from the numerator so it drops
  the **identical** set of intervals that `env.occupancy` drops from the
  denominator. Rate maps now differ (and are more correct) for sessions with
  large tracking gaps or out-of-bounds samples; pass `max_gap=None` to disable
  gap gating on both sides.
- `SpatialRatesResult.summary_table()` no longer double-computes grid and
  border scores (single pass, faster for large populations).
- The dense smoothing kernels (`diffusion_kde` and `gaussian_kde`) are
  `O(n_bins²)` memory by construction. For very large bin counts they now emit
  a loud `UserWarning` (with the estimated size) and **proceed** — there is no
  hard limit and no opt-out parameter. To reduce memory, use
  `smoothing_method="binned"` (or fewer bins / a larger `bin_size`).

### Deprecated

- `EgocentricRatesResult.detect_ovcs(...)` -> `classify(...)`
- `ViewRatesResult.detect_view_cells(...)` -> `classify(...)`
- `DirectionalRatesResult.detect_hd_cells(...)` -> `classify(...)`
- `SpatialRatesResult.detect_cell_types(...)` -> `label_cell_types(...)`
- `ViewRateResult.peak_view_location()` -> `peak_location()`
- `ViewRatesResult.peak_view_location()` -> `peak_locations()`
- `detect_region_crossings(position_bins, times, region_name, env, ...)` ->
  `detect_region_crossings(position_bins, times, env, *, region_name, ...)`

## [0.1.0] - 2024-11-03

### Added

- Initial release of neurospatial
- Core `Environment` class with factory methods
- Multiple layout engines (regular grid, hexagonal, triangular, graph-based)
- Region support for defining ROIs
- Composite environment functionality
- Alignment and transformation tools
- Comprehensive test suite
- NumPy-style docstrings throughout

### Features

- Automatic active bin detection from data samples
- NetworkX-based connectivity graphs
- 1D linearization for track-based experiments
- Spatial queries (bin_at, neighbors, shortest_path, distance_between)
- Visualization with matplotlib
- Morphological operations (dilation, closing, hole filling)

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/edeno/neurospatial/compare/v0.5.0...v0.6.0
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
