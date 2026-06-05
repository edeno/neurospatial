# Changelog

All notable changes to neurospatial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This section documents changes planned for v0.6.0. Until the release is cut,
the published package may still report v0.5.x.

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

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0
