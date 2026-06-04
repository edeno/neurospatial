# Changelog

All notable changes to neurospatial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `decode_position()` accepts population rate result objects directly.

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
