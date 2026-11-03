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
- One name per concept. `tests/test_argument_conventions.py` enforces each row;
  the old name is never kept as an alias.

  | Concept | Name | Not |
  | --- | --- | --- |
  | Randomness | `rng: np.random.Generator \| int \| None`, keyword-only, read as `np.random.default_rng(rng)`, so `rng=0` and `rng=np.random.default_rng(0)` give the same result | `seed`, `line_seed` |
  | One region | `region=` (a name) | `region_name`, `goal_region` |
  | Several regions | `regions=` | |
  | Two region roles in one call | a role prefix on each: `decision_region=` and `goal_regions=` | |
  | Time-bin width | `dt=` (seconds); spatial, angular and distance widths stay `bin_size` | `bin_size` for seconds |
  | Pre-entry window length | `window_duration=` | `pre_window` |
  | Paired inputs | suffix `_a` / `_b` | `_1` / `_2`, `_i` / `_j`, `angles1` / `angles2` |
  | Unit selection | `unit_ids=` (unit labels; as a selector, an array of integer ids, never a boolean mask) | `cells` |
  | Temporal smoothing width | `smooth_window=` is seconds; a width counted in bins is `smooth_bins=` | `smoothing_window` |
  | Significance level | `alpha=` means only this; the VTE head-sweep weight is `head_sweep_weight=` | `alpha` as a mixing weight |
  | Matplotlib target | `ax=`, keyword-only, never positional; a pair of axes is `ax=(raster_ax, psth_ax)` | `axes=`, positional `ax` |
  | Field resampling | `f(source_env, target_env, values, *, ...)` | `resample_field(field, src_env, dst_env)` |
  | Graph input in `ops` | `graph=` | `G` |
  | Sample times | `times` | `t` |

- Spell out acronyms in public names (no `detect_ovcs`).
- Data arguments are positional. Algorithm parameters, thresholds and ranges are
  keyword-only after `*`. Metric functions whose two array arguments share a
  shape take the second one keyword-only (`spatial_information(rate,
  occupancy=occupancy)`), so a swap cannot run silently.

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
- `summary()` returns a flat dict of scalars; `plot(*, ax=None, ...)` (after
  `idx` on population results) returns the `Axes`. `Environment.plot` and `plot_field` also take
  `backend="fastplotlib"`, which returns the fastplotlib figure; typing
  overloads keep the `Axes` return for the default.

- Properties vs methods: a no-argument scalar that is cheap to compute is a
  property (`hd.mean_vector_length`, `hd.preferred_direction`); anything that
  takes options or does real work stays a method. A single-unit name that a
  population result shares with the same meaning (`peak_firing_rate()`,
  `spatial_information()`, `peak_location()`) stays a method on both, so the
  two levels read alike. Every `summary()` key reads back from the result as
  written, and the repr says how: `name=` is a non-callable attribute,
  `name()=` a method that needs no argument (`spatial_information()=2.807`),
  `name[key]=` an entry of a mapping attribute and `a.b=` an attribute of an
  attribute (`observed.score`). Give a summary value its own property (`n_units`,
  `n_bins`, `total_occupancy`) rather than a key nothing else exposes; counts
  that depend on a method's options (`n_place` at a default `alpha`) stay out
  of the repr. `test_repr_names_are_accessible` checks every public result.
- Population `plot(idx=None, *, ax=None)`: with no unit, results with one
  axis per unit (PSTH, tuning, directional) draw a units-by-bins heatmap
  sorted by peak with a colorbar in Hz; 2-D map results raise
  `unit_index_required` from `_results.py`.

Non-neural results such as `ResampledPathResult` and `ContinuousEventResult`
export domain tables named for their rows (`samples_table()`,
`aligned_table()`) and do not implement `to_dataframe()`.

Results keep no raw input arrays or recompute closures. Rate and peri-event
results inherit `ReadOnlyArraysMixin` and call `self._own_arrays()` last in
`__post_init__`, so their arrays are read-only copies; exports
(`to_dataframe`, `to_xarray`) hand out writable copies.
Decode summaries, batch scores, behavior arrays, assembly results and
trajectory-fit arrays use this ownership contract too. Directional place fields
own the arrays in their mappings and prevent replacing entries. Pickle and
deepcopy preserve read-only state.
Results own `analysis_settings` and return independent dictionaries, including
nested recording windows, so inspecting or editing exported settings cannot
change training support or recorded analysis choices.

## Cell types

- Free predicates exist only for place, head-direction, object-vector
  (allocentric and egocentric) and spatial-view cells. Do not add
  `is_border_cell` or `is_grid_cell`: border and grid are labels from
  `SpatialRatesResult.label_cell_types()`.
- `has_place_field` detects a field; it is not a cell-type verdict.
- The free predicates on raw arrays are verdicts: they default to
  `criterion="shuffle"` and keep `criterion="threshold"` as the fast screen.
  Their first docstring line starts "Test whether ...". Result methods,
  `classify()` and `label_cell_types()` are threshold screens with inclusive
  `>=` thresholds (p-values stay `< alpha`), take no `criterion=`, and start
  "Screen ...". Seeded shuffle verdicts agree between single and population
  calls when `unit_id` matches.
- Constant defaults are written in the signature (`min_info=0.5`,
  `alpha=0.05`, `n_shuffles=1000`), never as a `None` that resolves to a
  constant; `tests/test_argument_conventions.py` enforces it.
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
