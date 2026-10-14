---
paths:
  - "src/neurospatial/**/*.py"
  - "tests/**/*.py"
---

# Public API contract

A public name has one meaning everywhere. New functions and result classes
follow these rules. `tests/test_public_api_snapshot.py`,
`tests/test_argument_conventions.py` and the contract tests in `tests/encoding/`
(`test_terminal_verbs_contract.py`, `test_naming_contract.py`,
`test_predicate_contract.py`, `test_unit_identity.py`) enforce parts of it.

## Names and arguments

- `positions` `(n_time, n_dims)`, not `trajectory` or `xy`; `position_bins`, not
  `trajectory_bins`; `times` in seconds, not `timestamps`; `spike_times`, not
  `spikes` or `spike_trains`; `headings` in radians, not `head_direction`;
  `targets` for egocentric operations and `object_positions` for encoding;
  `bandwidth` in physical units, not `sigma`; `method` for estimators;
  `metric="euclidean" | "geodesic"`, not `use_geodesic`.
- Spell out acronyms in public names (no `detect_ovcs`).
- Data arguments are positional. Algorithm parameters, thresholds and ranges are
  keyword-only after `*`.

## Result classes

Each terminal verb means the same thing on every result class:

- `to_dataframe()`: dense tidy frame, one row per `(unit, bin)`, always with a
  `unit_id` column.
- `summary_table()`: one row per unit, indexed by `unit_id`, scalar metrics
  only. Singular and population results have identical columns; primary metrics
  come first and classification last. Constants (`method`, `units`,
  `classification_thresholds`, `direction_frame`) go in `df.attrs`, not columns.
- `to_xarray()`: rate results use dims `("unit_id", "bin")`, with a one-unit
  axis for singular results and `unit_id` as the label index; duplicate
  `unit_ids` raise. Decode results use `("time", "bin")` with no unit axis.
  Never conflate the two. PSTH results have no `to_xarray()`.
- `summary()` returns a flat dict of scalars; `plot(ax=None, ...)` returns the
  `Axes`.

Results keep no raw input arrays or recompute closures. Rate and peri-event
results inherit `ReadOnlyArraysMixin` and call `self._own_arrays()` last in
`__post_init__`, so their arrays are read-only copies; exports
(`to_dataframe`, `to_xarray`) hand out writable copies.
Results own `analysis_settings` and return independent dictionaries, including
nested recording windows, so inspecting or editing exported settings cannot
change training support or recorded analysis choices.

## Cell types

- Free predicates exist only for place, head-direction, object-vector
  (allocentric and egocentric) and spatial-view cells. Do not add
  `is_border_cell` or `is_grid_cell`: border and grid are labels from
  `SpatialRatesResult.label_cell_types()`.
- `has_place_field` detects a field; it is not a cell-type verdict.
- Result methods and batch classifiers are threshold screens with inclusive
  `>=` thresholds (Rayleigh p-values stay `< alpha`). Shuffle verdicts come from
  the `*_significance` functions or `criterion="shuffle"` on raw arrays, and
  seeded results agree between single and population calls when `unit_id`
  matches.
- `classify()` returns a bool array for one cell type and `label_cell_types()`
  returns strings. A bool predicate and a string labeler never share a name.

## Peaks and unit identity

- `peak_location()` works on every result, in that result's own coordinates.
  Plural `peak_locations()` exists only for Cartesian results; directional and
  object-vector results use `preferred_direction(s)` and
  `preferred_distance(s)` instead.
- Population results carry `unit_ids` (default `np.arange(n)`), and batch
  `compute_*` functions accept keyword-only `unit_ids=`; a wrong length raises.
  Indexing or iterating a population result stamps each child's `unit_id`.
- `[]` selects by position on every unit container, `SpikeTrains` included, and
  `sel(unit_id=...)` selects by label. New population results get both from
  `UnitSelectionMixin` and call `self._unit_position(idx)` in `__getitem__`.
