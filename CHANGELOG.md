# Changelog

## [Unreleased]

### Breaking

These `Environment` changes come with before-and-after code on the environment
architecture [migration
page](https://edeno.github.io/neurospatial/migration/environment-architecture/).

- `Environment.occupancy` (with the default `time_allocation="start"`) now
  counts an interval only when both of its samples are inside the
  environment, the rule `compute_spatial_rate` already used for its spike
  counts and occupancy. Before, an interval that ended at a tracking dropout
  (NaN) or outside the environment kept its time in its start bin, so spike
  counts divided by `env.occupancy` used a different denominator than the
  rate maps. Fill short dropouts during tracking preprocessing to keep that
  time.

- `Environment.animate_fields` raises `TypeError` for an option that no
  backend accepts, naming the closest option, instead of ignoring it: a
  misspelled `frame_label=` used to change nothing. `fps=` raises too (use
  `speed=`). The unused `contrast_limits`, `show_colorbar` and
  `colorbar_label` options are removed.

- `Environment.from_polar_egocentric` refuses an `angle_range` wider than one
  turn (2π) with a `ValueError` that shows a one-turn range. Angles a whole
  turn apart name one direction, so after angle wrapping the bins past the
  first turn would never hold a sample; such a range used to be accepted.

- A track built without `track_width` (`Environment.linear_track`,
  `Environment.maze`, `Environment.from_graph`) no longer assigns 2-D positions
  to bins. `bin_at`, `contains`, `occupancy`, `bin_sequence`,
  `bin_sequence_with_runs`, `transitions(times=..., positions=...)`, `points=`
  queries, point regions, and every analysis that maps positions with them
  (rate maps, decoding, behavior metrics) raise `NotImplementedError`.
  Before, a position any distance from the track, such as a reflection 5 m
  away, was put in its nearest bin and counted as occupancy and spikes there;
  `distance_to(points=[[500, 500]])` measured from the nearest bin. Pass the
  measured full width when you build the track, for example
  `Environment.linear_track(endpoints=[(0, 0), (200, 0)], bin_size=2.0,
  track_width=10.0)`; positions farther than half the width from the
  centerline then get bin `-1` and add no occupancy. A track you already
  have, including one saved by an earlier version and read with
  `Environment.from_file`, `from_dict` or `from_nwb`, takes its width with the
  new `env.with_track_width(10.0)`, which keeps its bins; the error message
  shows that call. `linear_to_nd` and
  graph queries on bin indices do not need the width. `to_linear` does not
  either, because it projects every position onto the nearest point of the
  centerline, however far off the track: drop off-track samples (bin `-1`
  from `bin_at` on a track with a width) before using linear positions. The
  simulated mazes in `neurospatial.simulation.mazes` with a corridor width
  (linear track, T, Y, W, plus, radial arm, repeated T and Y, crossword,
  Hamlet, Hampton Court and rat hexmaze) now pass it to their track, and
  `tmaze_alternation_session` builds 10 cm corridors. The Sungod, honeycomb
  and small hex mazes have no single corridor width; their tracks take the
  widest one (the widest arm, one platform, the widest corridor between
  barriers), so a sample within half that width of the centerline counts as
  on the track and a narrower corridor can admit a few samples just outside
  its walls.

- Pickle files of an `Environment` written by an earlier version can no longer
  be loaded. Pickle still works for passing an environment between processes
  of the same version (for example with `multiprocessing`), but it is not a
  storage format. An environment state from another version raises a
  `TypeError` that says to use files instead.
  Save with `env.to_file(...)` or NWB export and reload with
  `Environment.from_file(...)`.

- Old import paths are removed. The compatibility modules
  `neurospatial.environment.transforms`, `._coordinates`, `._metadata`,
  `._ownership` and `._from_bins`, the former private geometry modules under
  `neurospatial.layout`, the private names re-exported by `neurospatial.arena`
  and `neurospatial.environment._HAS_SHAPELY` are gone (shapely is a required
  dependency). The modules that implement `Environment` methods are private:
  `neurospatial.environment.fields`, `.queries`, `.metrics`, `.trajectory`,
  `.regions`, `.visualization`, `.polar`, `.cartesian` and `.factories` are
  now `_fields`, `_queries` and so on, and
  `neurospatial.environment.serialization` is removed. Call the `Environment`
  methods (for example `env.smooth`, `env.occupancy`, `env.to_file`) instead,
  and import `BinSequenceWithRuns` from `neurospatial.environment`. The public
  arena classes in `neurospatial.arena` are unchanged.

- `neurospatial.ops.binning` is renamed `neurospatial.ops.mapping`, so it is
  no longer confused with `neurospatial.binning`, which holds the `Binning`
  records. It maps points to bins, regions to bin masks and fields between
  environments. Import `regions_to_mask`, `resample_field` and
  `TieBreakStrategy` from `neurospatial.ops` or `neurospatial.ops.mapping`.

- `Environment` is an immutable value. Two environments are equal only when
  they are the same object. `env == "name"` used to compare the environment's
  name; it now returns `False` without an error, so a check such as
  `if env == "session1":` silently takes the other branch. Write
  `env.name == "session1"` instead. Environments can be used in sets and as
  dictionary keys. `Environment.copy()` is removed because an environment
  cannot change after it is built; `copy.copy(env)` and `copy.deepcopy(env)`
  return the same object. Use `with_metadata`, `with_regions`, `subset` or
  `rebin` to get a changed environment.

- `EnvironmentNotFittedError` (error code E1004), the `check_fitted`
  decorator and the `neurospatial.environment.decorators` module are removed.
  An environment always has its bins and connectivity; calling
  `Environment()` directly still raises E1006. Passing something that is not
  an environment to an analysis function raises `TypeError` naming the
  corrected call.

- `Environment.clear_cache()` takes no arguments and releases every cached
  analysis (KD-tree, smoothing kernels, eigenbases, operators, boundary bins
  and attribute tables), including the search structures behind `bin_at` on
  tracks, triangular meshes and clipped environments. On a track, `bin_at`
  and `nearest_bin` share one KD-tree. The `kdtree=`, `kernels=` and
  `cached_properties=` options and
  `neurospatial.ops.binning.clear_kdtree_cache` are removed; call
  `env.clear_cache()` instead.

- `Environment.to_linear()` and `Environment.linear_to_nd()` raise `TypeError`
  instead of `AttributeError` when the environment is not a linearized track.
  The message is unchanged.

- Egocentric polar environments (`Environment.from_polar_egocentric`) report
  `env.layout_type == "Polar"` instead of `"MaskedGrid"`, and their printed
  summary says `Polar`. Their bins are distance-by-angle sectors, not
  Cartesian grid cells. Files saved before this change load as `"Polar"` too.
  To detect a polar environment, test
  `env.coordinate_kind == "polar"` instead of the layout name.

- A polar environment takes the choice of whether its angle wraps around only
  from its stored `EgocentricSurround` arena. Old files (`Environment-v2`
  files and NWB environment schema 2.0) stored no arena, so a polar
  environment in one of them is refused with a message to re-save it.
  Cartesian environments in those files still load.

- Custom layout engines must implement `binning()`, which returns the
  `Binning` the environment keeps: the bins, their neighbor pairs, point
  lookup, the faces between cells (or why they are missing), any grid, track,
  shape and outline records and, when the cells do not show it, the
  `dimension_ranges` the bins cover. An engine needs only `build(...)` and
  `binning()`: the environment derives its connectivity graph, `grid_edges`,
  `grid_shape`, `active_mask` and `dimension_ranges` from the `Binning` and
  no longer reads an engine's `connectivity`, `dimension_ranges` or grid
  attributes, so attributes an engine adds to its own graph do not reach
  `env.connectivity`. Lookup and faces come only from `binning()`, so to
  change them, change what `binning()` returns. Two labels
  are read when an engine has them: its `layout_type` label, and the
  build arguments recorded with `neurospatial.layout.base.capture_build_params`
  (`env.layout_parameters`). All built-in engines implement `binning()`, a
  subclass of a built-in engine inherits a working one, and `LayoutEngine`
  lists the method. Objects without `binning()` still need an installed
  snapshot adapter. The private `neurospatial.layout._capabilities` module
  (`RectilinearGridProvider`) is removed; read `layout.binning().rectilinear`.

- `env.connectivity` is rebuilt from the bins every time an environment is
  built, derived or read back, so it holds only the standard node and edge
  attributes, as numbers and tuples, and `env.connectivity.graph` is an empty
  read-only mapping. Reading an attribute returns the stored value without a
  copy; editing any of them is refused. `env.to_networkx()` and
  `env.connectivity.copy()` return independent, editable graphs. A file, dict
  or NWB record still stores the graph for other tools, but loading rebuilds
  it from the bins: attributes added to a saved graph are not read back.

- Derived environments keep the labels of the environment they came from.
  `subset()` and `clip()` keep the named regions (they used to drop them), and
  `subset()` keeps the name. `rebin()` keeps the name (no `_rebinned` suffix).
  `apply_transform()` keeps the name (no `_transformed` suffix) and no longer
  invents a frame: the new frame is the `frame=` you pass, or `None`.

- `env.layout_parameters` (and the `layout_parameters` record in files, dicts
  and NWB) holds only the scalar build choices, such as `bin_size`,
  `connect_diagonal_neighbors` or `circular_angle`. Input data is no longer
  kept: the positions given to `from_samples`, the masks of `from_grid_mask`,
  `from_pixel_mask` and polar environments, polygons, the track graph and
  edge order of `from_graph`, and nested values such as `dimension_ranges`
  (read `env.dimension_ranges`). A saved environment no longer grows with the
  length of the trajectory it was built from (a 100,000-sample `from_samples`
  file went from 7.5 MB to 0.46 MB, the same size as a 10-sample one). The
  track graph is still owned by the environment: read it from
  `env.binning.linearization.graph` (and `.edge_order`).

- Factories take their data positionally and every option by keyword:
  - `Environment.from_samples(positions, bin_size=2.0)` and
    `Environment.open_field(positions, bin_size=2.0)`, not
    `from_samples(positions, 2.0)`. Their signatures list every option, so a
    misspelled option raises `TypeError` instead of reaching the layout
    engine.
  - `Environment.from_graph(graph, edge_order, bin_size=2.0)`. `edge_spacing`
    was a required positional argument before `bin_size`; it is now a keyword
    that defaults to `0.0` (edges end to end), as in `linear_track` and
    `maze`.
  - `Environment.maze(kind="plus", node_positions=..., bin_size=...)`.
  - `Environment.from_polygon(polygon, bin_size=2.0)` and
    `Environment.from_pixel_mask(mask, pixel_size=0.5)`.
  - `Environment.from_polar_egocentric(distance_range=(0, 50),
    angle_range=(-np.pi, np.pi), distance_bin_size=5.0,
    angle_bin_size=np.pi / 8)`. Four positional numbers were easy to swap.

  An option passed by position raises a `TypeError` that names the keyword
  and shows the corrected call: `Environment.from_samples(positions, 4.0)`
  says to pass `bin_size=4.0` by keyword, and
  `Environment.from_graph(graph, edge_order, 0.0, 2.0)` says to pass
  `edge_spacing=0.0, bin_size=2.0`. Other argument errors, such as a
  misspelled option, keep Python's own message.

- `Environment.from_samples`, `Environment.open_field` and
  `neurospatial.io.nwb.environment_from_position` refuse option values that
  the call would ignore, instead of ignoring them without a word. A
  regular-grid-only option (`dilate`, `fill_holes`, `close_gaps`,
  `add_boundary_bins`, `connect_diagonal_neighbors`) set to anything but its
  default raises `ValueError` with `layout="Hexagonal"`. With
  `infer_active_bins=False` every bin in the extent is active, so changing
  `bin_count_threshold`, `dilate`, `fill_holes` or `close_gaps` from its
  default raises `ValueError`. The defaults are accepted with any layout, and
  the signatures show them (`dilate: bool = False`,
  `bin_count_threshold: int = 0`, `connect_diagonal_neighbors: bool = True`;
  `fill_holes` defaults to `True` in `open_field`). `open_field` also accepts
  `fill_holes=False` with `layout="Hexagonal"` or `infer_active_bins=False`,
  because neither fills holes. A `layout=` other than `"RegularGrid"` or
  `"Hexagonal"` raises `ValueError` listing the supported
  layouts and the factory to use instead; it used to raise
  `NotImplementedError`. Errors from `environment_from_position` name that
  function.

- `Environment.from_layout` takes a built layout engine instead of a layout
  name and a dictionary of build parameters:
  `Environment.from_layout(create_layout("TriangularMesh",
  boundary_polygon=arena, point_spacing=2.5))`. A layout name or an engine
  class raises `TypeError` with the corrected call. `layout_type` and
  `layout_parameters` come from the engine's own record, which includes the
  defaults it resolved but never its input data: a built-in engine's
  positions, masks, polygons, graphs and boundary coordinates are left out,
  whether or not they were given. Grids, polygons, masks and tracks keep
  their dedicated factories.

- `Environment.from_nwb` only loads an environment saved in the file:
  `Environment.from_nwb(nwbfile, name="linear_track")`. The record name is
  `name=` (it was `scratch_name=`), matching `env.to_nwb(nwbfile, name)` and
  `read_environment`, and defaults to `"spatial_environment"`. To build an
  environment from the file's Position data, call
  `neurospatial.io.nwb.environment_from_position(nwbfile, bin_size=...)`
  instead of `from_nwb(nwbfile, bin_size=...)`. It takes `bin_size` as a
  keyword and lists every `from_samples` option (`name=`, `regions=`,
  `arena=`, `layout=`, `fill_holes=` and so on) instead of forwarding
  `**kwargs`. `from_nwb(nwbfile, bin_size=...)` raises a `TypeError` that
  shows that call. When the file has no saved environment of that name,
  `from_nwb` and `read_environment` raise a `KeyError` that lists the saved
  environments and, if the file has Position data, names it and points to
  `environment_from_position`.

- Every region query now uses one rule for which bins a region covers. A
  point region covers the bin that contains the point. A polygon region covers
  the bins whose centers lie inside it or on its edge; pass
  `include_boundary=False` to count only centers strictly inside. The rule is
  shared by `bins_in_region`, `region_mask`, `region_membership`,
  `distance_to` with a region name, `subset(region_names=...)`,
  `subset(polygon=...)` and `neurospatial.ops.mapping.regions_to_mask`, so
  they agree on every layout. Before, they disagreed:
  - `region_membership()` and `regions_to_mask()` gave a point region no bins
    (`regions_to_mask` also warned). They now give the bin that contains the
    point. Time spent at a point region, and visits, crossings and laps found
    through `regions_to_mask`, now count that bin.
  - `bins_in_region()`, `subset(region_names=...)` and `subset(polygon=...)`
    left out bins whose centers lie exactly on a polygon's edge, while
    `region_mask` included them. All of them now include those bins.
    `bins_in_region` and `subset` take `include_boundary=False` for the old
    interior-only selection (`subset` refuses it together with `mask=`).
    Analyses built on `bins_in_region` can change when a polygon
    edge passes through bin centers: the decision-region entry time in
    `decision_region_entry_time`, `compute_decision_analysis` and
    `compute_vte_session`, the goal bins and decision boundary of
    `compute_decision_analysis`, the start and goal bins of
    `trials_to_region_arrays`, and the region boundary of
    `neurospatial.events.distance_to_boundary(region_name=...)`.
  - `subset(region_names=[...])` keeps the bin that contains a point region
    instead of raising `ValueError`.
  - `distance_to("reward")` with a point region measures the distance to the
    bin that contains the point. It used to warn that the region had no bins
    and then raise. A region that covers no bin, such as a point outside the
    environment, raises a `ValueError` that says so.
  - On egocentric polar environments a point region is read in the same
    (distance, angle) coordinates as `bin_centers`; `bins_in_region` and
    `region_mask` used to refuse point regions there.
  - A polygon region on an environment whose bins are not 2-D raises
    `ValueError` with a fix in `region_membership` and `regions_to_mask`; it
    used to raise `NotImplementedError`.

- Graph and distance queries take their target by keyword: `bins=` for bin
  indices, `points=` for coordinates and, for `distance_to`, `region=` for a
  region name. Exactly one is required. A bin index and a coordinate look the
  same as numbers, and the old positional argument guessed between them:
  `env.distance_to([50.7, 12.3])` measured the distance to bins 50 and 12,
  and `env.neighbors([0, 1])` treated bins 0 and 1 as the point (0, 1).
  - `env.distance_to(targets)` is `env.distance_to(bins=...)`,
    `env.distance_to(points=...)` or `env.distance_to(region="reward")`.
  - `env.neighbors(i)`, `env.reachable_from(i, ...)` and
    `env.rings(center_bin=i, ...)` take `bins=i` or one point,
    `points=[[x, y]]`.
  - `env.path_between(a, b)` is `env.path_between(bins=[a, b])` or
    `env.path_between(points=[[x0, y0], [x1, y1]])`.
  - `env.distance_between(p1, p2)` is
    `env.distance_between(points=[p1, p2])` or
    `env.distance_between(bins=[a, b])`; `edge_weight=` is keyword-only.

  `bins=` takes an int or an integer array. Floats and booleans raise
  `TypeError` and point to `points=`; `env.bin_center_of` refuses them too and
  points to `env.bin_at`, with `(distance, angle)` rows on polar environments.
  `BinIndexOutOfRangeError` names the same lookup. `bin_center_of` also refuses a 2-D
  array of indices, for which it used to return an array of shape
  `(a, b, n_dims)`; pass a 1-D array. `points=` has shape `(n_points, n_dims)`, one row per
  point, in the same coordinates as `bin_centers` (distance and angle on a
  polar environment). Points with NaN or inf coordinates, or outside the
  environment, raise `ValueError` that names the first such row.
  `distance_between` used to return `inf` for a point outside the
  environment; it now raises. It also works on egocentric polar environments,
  where it used to raise `CartesianOnlyError`: the path length follows the
  graph, and points are (distance, angle). A target passed by position or
  under its old name (`targets=`, `bin_index=`, `center_bin=`, ...) raises a
  `TypeError` that says which keyword applies and shows the corrected call:
  `env.neighbors(5)` suggests `env.neighbors(bins=5)`, and
  `env.distance_to("reward")` suggests `env.distance_to(region="reward")`.

- Behavior functions that take bin targets check them the way the queries
  check `bins=`: `distance_to_region(target_bins=)`, `cost_to_goal(goal_bins=)`,
  `path_progress(start_bins=, goal_bins=)`, `geodesic_voronoi_labels`,
  `distance_to_decision_boundary` and `goal_reward_field`. Floats and
  booleans raise `TypeError` instead of being
  truncated to bin indices, and out-of-range bins raise
  `BinIndexOutOfRangeError` (a `ValueError`). `-1`, the value `bin_at` gives
  a position outside the environment, is never accepted as a single target,
  whether it is an int or a one-element array such as
  `env.bin_at([[x, y]])`: `target_bins=-1` and `goal_bins=-1` used to return
  all-NaN distances and now raise. Only an array with one entry per sample
  (shape `(len(position_bins),)`, as built by `trials_to_region_arrays`) is
  per sample, and there `-1` still marks a sample with no target and gives
  NaN; an array of any other length raises `ValueError`. `path_progress`
  also accepts one start and one goal bin for every sample.
  `approach_rate(..., metric="geodesic")` raises for a goal outside the
  environment; it used to return all-NaN rates. `shortest_path_length(...,
  metric="geodesic")` measures with `env.distance_between`: a start outside
  the environment used to return the distance from the last bin without a
  warning, and a goal outside raised an unrelated graph error; both now raise
  `ValueError` naming the point. So `compute_path_efficiency` and
  `subgoal_efficiency` with `metric="geodesic"` raise that `ValueError` when
  the trajectory starts outside the environment. `geodesic_voronoi_labels`
  and `distance_to_decision_boundary` raise `ValueError` for an empty goal
  list; `distance_to_decision_boundary` used to return `inf` for every sample.

- `Environment.subset(bins=mask)` is now `env.subset(mask=mask)`, so `bins=`
  means integer bin indices everywhere. `mask=` is still a boolean array with
  one entry per bin. Integer values in `mask=` raise `ValueError` that shows how
  to build a mask from bin indices.

- Cartesian-only methods stay visible on egocentric polar environments and
  refuse when called. `hasattr(polar_env, "subset")` is now `True` and
  `dir(polar_env)` lists `subset`, `rebin`, `interpolate` and the other
  Cartesian-only methods; calling one raises `CartesianOnlyError`.
  `CartesianOnlyError` is now a `TypeError` (the operation does not apply to
  this kind of environment) and no longer an `AttributeError` or
  `NotImplementedError`; import it as `neurospatial.CartesianOnlyError`.
  Replace `hasattr` checks with `env.coordinate_kind == "polar"`, and
  `except (AttributeError, NotImplementedError)` with
  `except CartesianOnlyError`. Every Cartesian-only refusal now raises this
  error with one message layout: what was refused, why it needs Cartesian
  positions, and what works on a polar environment. That includes the
  Cartesian modes of shared methods (`distance_to(metric="euclidean")`,
  `occupancy(time_allocation="linear")`), analysis functions such as
  `spatial_autocorrelation` and `field_shape_metrics`,
  `compute_egocentric_distance(metric="geodesic")` (it reported
  `Environment.bin_at()`), and `Environment.combine` (it raised a plain
  `TypeError`). A function that takes two environments names the polar one,
  for example `resample_field(dst_env=...)`. Where a polar environment has an
  equivalent, the message ends with the code to run instead, also available as
  `error.example`. The suggested code keeps containment: rows outside the
  environment get bin `-1` (or NaN from `interpolate`), never the nearest
  edge bin.

- `bin_at`, `contains`, `occupancy`, `bin_sequence`, `bin_sequence_with_runs`
  and `transitions(times=..., positions=...)` work on egocentric polar
  environments, where they used to raise `CartesianOnlyError`. Each row is
  `(distance, angle)` and gets the bin whose cell holds it, or `-1` outside
  the distance and angle ranges, the same lookup the `points=` queries use.
  `occupancy` gives each interval to the bin of its first sample, as on any
  environment, so egocentric occupancy (seconds with the object at each
  distance and direction) comes straight from per-sample `(distance, angle)`
  rows; `time_allocation="linear"` still refuses polar environments.
  `CARTESIAN_OPERATIONS` no longer lists these methods. Analysis functions
  that read positions as the animal's allocentric positions and relied on
  `bin_at` to refuse polar environments now refuse them with their own name:
  `compute_spatial_rate`, `compute_spatial_rates`, `place_cell_significance`,
  `compute_directional_place_fields`, the view-rate functions,
  `compute_decision_analysis`, `compute_vte_session`, the geodesic modes of
  `compute_step_lengths`, `mean_square_displacement` and `decoding_error`,
  the `distance_to_reward` and `distance_to_boundary` event regressors,
  `simulate_trajectory_ou` (and so `simulate_session`), and the boundary,
  object-vector, spatial-view and geodesic place-cell models.

- Three `neurospatial.ops` functions that only renamed an `Environment`
  method are removed; call the method instead.
  `map_points_to_bins(env, points, ...)` is `env.nearest_bin(points, ...)`,
  `compute_differential_operator(env)` is `env.get_differential_operator()`,
  and `apply_transform_to_environment(env, transform, ...)` is
  `env.apply_transform(transform, ...)`. The arguments and results are the
  same, with one difference: `map_points_to_bins` refused polar environments,
  and `env.nearest_bin` accepts them.

- `neurospatial.layout.validation.validate_environment` is removed. Every
  environment is checked when it is built, so a built environment always
  passes it. Set units and frame when you build the environment, for example
  `Environment.from_samples(positions, bin_size=2.0, units="cm", frame="room")`,
  or later with `env.with_metadata(units=..., frame=...)`.

- The built-in layout engines no longer have `point_to_bin_index`,
  `finite_volume_geometry` or `is_grid_compatible`, and the `LayoutEngine`
  protocol no longer lists the first and last. An environment never used
  them: it locates points and builds cell faces from the engine's
  `binning()`. Use `env.bin_at(points)`, or
  `engine.binning().locator.locate(points)` on a built engine;
  `engine.binning().faces.build()` for the cell faces; and
  `binning.rectilinear is not None` to ask whether the bins form a
  rectangular grid.

- A custom layout engine that records its build arguments with
  `neurospatial.layout.base.capture_build_params` declares its input-data
  arguments in the decorator: `@capture_build_params(data=("positions",))`.
  They are left out of `env.layout_parameters`. Any other argument that is not
  a scalar or a flat sequence of scalars now raises a `TypeError` naming it
  when the engine is built; it used to be dropped without notice. The
  built-in engines declare their data the same way, replacing their
  `_data_parameters` attribute, and their recorded choices are unchanged.
  When a saved environment's `layout_parameters` hold input data, as files
  from earlier versions do, the data is dropped on reading and the scalar
  choices are kept; the bins are stored in the file.

- One layout label. `LayoutType` (now also with `POLAR` and `COMBINED`) is
  the only vocabulary: a built-in engine's `layout_type` is its `LayoutType`
  value (`RegularGridLayout().layout_type == "RegularGrid"`, it was the
  category `"grid"`), `env.binning.kind` holds the same label, and
  `env.layout_type` reports it. Engine categories such as `"grid"` or
  `"mask"`, the private `_layout_type_tag` and the `layout_category` field of
  saved files are gone. User spellings are reported canonically:
  `from_samples(..., layout="regular-grid")` gives
  `env.layout_type == "RegularGrid"` (it gave `"regular-grid"`), and
  `create_layout` also accepts `"grid"`, `"polygon"` and `"mesh"`. The new
  `neurospatial.layout.find_layout_type(name)` returns the `LayoutType` a
  spelling names, or `None`. `from_bins` spells a `Binning.kind` that names
  a layout canonically. It refuses `"Polar"`, which is reserved for
  `from_polar_egocentric`, and `"Combined"` on bins that are not joined
  components of `combine`. Derived
  environments keep their parent's label instead of `"MaskedGrid"` or
  `"subset"` (subset), `"PolygonCells"` (clip), `"RegularGrid"` (rebin of
  any grid) or `"<label>_transformed"` (apply_transform); `combine` reports
  `"Combined"` (it was `"combined"`). The new `env.derivations` lists the
  derivations applied, oldest first, for example `("subset", "rebin")`.
  `layout_parameters` keeps the parent's build choices through `subset`,
  `clip` and `apply_transform` (they used to be replaced by
  `{"source": ..., "original_n_bins": ...}` or gain `"transformed": True`),
  and `rebin(bin_size=...)` records `bin_size` and
  `connect_diagonal_neighbors` as `rebin(factor)` does. Files from earlier
  versions load with the canonical label and their derivation markers as
  `derivations`. A subclass of a built-in engine that sets its own
  `layout_type` reports it everywhere; it used to report the parent's
  label as `env.layout_type`.

- Assigning or deleting a public attribute of an `Environment` raises an
  `AttributeError` that says what to call instead. That includes new names:
  `env.session_id = "s1"` used to attach a value silently and now says to keep
  your own labels next to the environment, for example in a dict keyed by
  `env.name`.

- Building from any layout engine, including a custom one, and `rebin` now
  check that each bin center is located in its own bin, as `from_bins`,
  `subset` and the readers already did. The built-in engines pass; a custom
  engine whose binning fails the check is refused with a `ValueError`.

- `GridCellModel(grid_orientation=θ)` (and `simulate_session(...,
  orientation=θ)`) now means the angle of the field lattice: the direction
  from a firing field to its neighbors, in radians from +x counterclockwise,
  modulo π/3. This is the orientation `grid_properties` reports, so
  simulating at θ and analysing the rate map returns θ. It used to set the
  angle of the first of the three plane waves, which puts the lattice π/6
  (30°) further counterclockwise, so simulated rate maps and spikes for a
  given `grid_orientation` are rotated by π/6. To reproduce an earlier
  simulation, pass `grid_orientation=old + np.pi / 6`.

- Grid orientation values change units and frame.
  `grid_properties().orientation` (and `SpatialRateResult.grid_properties()`)
  read the autocorrelogram as an
  image, with its first axis as y, but `spatial_autocorrelation` puts the
  environment's x on the first axis, so the reported angle was the lattice
  angle reflected about 45°. It was also in degrees. Orientation is now the
  angle of the field lattice in the environment's frame, in radians from +x
  (East) counterclockwise toward +y (North), in [0, π/3): a hexagonal
  lattice repeats every π/3. `orientation_std` is in radians too.
  `peak_coords` columns are now offsets along x then y, as documented; they
  used to come out y first, so code that read `peak_coords[:, 0]` as the x
  offset was reading y. Convert old orientation values with
  `np.radians((90 - old) % 60)`; this is approximate, because old values
  also averaged in a spare peak that could lie on the second ring of fields
  (see Fixed), so they were off by up to about 5°. The migration page has
  before-and-after values. A grid cell simulated with
  `GridCellModel(grid_orientation=θ)` is reported with orientation θ.

- `Environment` method options are keyword-only, as factory options already
  were. `bandwidth` is keyword-only in `smooth`, `diffuse`, `compute_kernel`,
  `compute_gaussian_kernel` and `diffusion_accuracy`: write
  `env.smooth(rate, bandwidth=5.0)`, not `env.smooth(rate, 5.0)`. In
  `plot`, `plot_1d` and `plot_field`, everything after the field and `ax` is
  keyword-only (`env.plot_field(rate, ax, cmap="magma")`), and `to_nwb` takes
  `name=` by keyword, as `from_nwb` does. A positional call raises a
  `TypeError` that names the keyword and shows the corrected call, as the
  factories do: `env.smooth(rate, 5.0)` suggests
  `env.smooth(field, bandwidth=5.0)`. `env.subset(bins=mask)` says to pass
  `mask=`. The migration page has before-and-after calls.

- `env.animate_fields(overlay_trajectory=...)` is removed. No backend drew it
  (napari dropped it and the others ignored it), although the animation guide
  said napari showed it as a track. Pass
  `overlays=[PositionOverlay(positions=..., times=..., trail_length=...)]`
  instead.

- The coordinate argument of `bin_at`, `contains`, `nearest_bin`,
  `to_linear` and `interpolate` is named `positions`, the name analysis
  functions use (it was `points_nd`, `points` or `nd_position`). Only a call
  that passed it by keyword changes: `env.bin_at(points_nd=xy)` becomes
  `env.bin_at(xy)` or `env.bin_at(positions=xy)`. The `points=` query
  target of `distance_to`, `neighbors` and the other graph queries is
  unchanged.

- A nonzero `edge_spacing` on a track with one edge (for example
  `Environment.linear_track(endpoints=..., edge_spacing=5.0)`) raises
  `ValueError`. A gap goes between consecutive edges, so the value used to be
  recorded in `layout_parameters` and ignored. Remove it, or use
  `node_positions` with more than two points for a track with segments.

- A misspelled region name in `animate_fields(show_regions=[...])` (every
  backend), `regions_containing_points(region_names=...)` or
  `plot_regions(region_names=...)` raises `RegionNotFoundError` with the
  closest name, before anything is drawn. These names used to be skipped,
  `plot_regions` with a warning, so a region silently went missing from the
  animation, the table or the plot. Fix the name. A single string, such as
  `show_regions="goal"`, now means that one region; it used to be read
  letter by letter.

- `graph_turn_sequence(env, position_bins)` no longer takes `start_bin` and
  `end_bin`. They were documented as orienting the path but were never used:
  the order of `position_bins` already sets the direction of travel, and the
  result did not depend on them. Drop the two arguments from the call.

- A built-in layout engine's `connectivity` graph (for example
  `create_layout("RegularGrid", ...).connectivity`) is now made from the
  engine's `binning()` when you first read it, so it is the same graph as
  `env.connectivity`. It can no longer be assigned, and editing it no longer
  changes the engine's bins or an environment built from it; return other
  neighbor pairs from your own engine's `binning()` instead. Reading it
  before `build()` raises `LayoutNotBuiltError` instead of returning `None`.
  Its attributes now match `env.connectivity`: on a track, edge `vector`
  and `angle_2d` point from the lower to the higher bin (along a segment
  they pointed the other way), `source_grid_flat_index` counts the gap cells
  of a track with `edge_spacing > 0`, a 3-D track has no `angle_2d`, the
  angular-seam edges of a polar grid number their `edge_id` by pair row,
  and values are Python numbers instead of NumPy scalars. Neighbor pairs,
  their order and their distances are unchanged.

- `neurospatial.GraphValidationError` and
  `neurospatial.layout.validation.validate_connectivity_graph` are removed.
  Layout engines no longer build graphs, and an environment's graph is made
  from bins that are checked when they are created: a non-finite or negative
  pair distance, a bin paired with itself or a repeated pair raises
  `ValueError`. Code that caught `GraphValidationError` can catch
  `ValueError`, its parent class.

### Changed

- The `neurospatial.layout.LayoutEngine` protocol lists only `build()` and
  `binning()`, the two methods an environment uses. A custom engine needs
  nothing else, and type checkers no longer ask it for `bin_centers`,
  `grid_edges`, `plot()` and the other members that the built-in engines
  offer for working with an engine directly.

- `to_linear()`, `linear_to_nd()` and `with_track_width()` on a polar
  environment now say what is actually wrong: it is not a track (for example
  "to_linear() needs a linearized track ..., but this is a Polar environment,
  which has no track"), with the same advice every other non-track
  environment gets, such as `env.bin_at(positions)` instead of `to_linear`.
  They used to answer with the Cartesian-coordinates refusal. The error is
  still a `TypeError`, but no longer a `CartesianOnlyError`, and these
  three are no longer listed among the Cartesian-only members.

- Positions or points with the wrong number of columns get the same error from
  every operation that maps them to bins (`bin_at`, `contains`, `occupancy`,
  `bin_sequence`, `bin_sequence_with_runs`, `transitions`, `interpolate`,
  `points=` queries such as `distance_to`, geodesic
  `compute_egocentric_distance`). It names the operation, the shape given and
  the shape needed, for example `Environment.occupancy got positions with
  shape (500, 3), but this environment's bins are 2-D, so positions needs
  shape (n_positions, 2).`, and says how to fix it, such as reshaping a 1-D
  array on a 1-D environment or using the original coordinates instead of
  linearized ones. Rate maps and other encoding functions that report several
  input problems together describe this one in the same words. It is still a
  `ValueError`; only the wording changed.

- `Environment.from_bins(binning)` is annotated as taking a `Binning`. An
  object from another package that an installed snapshot adapter converts
  still works at runtime, but a static type checker now flags the call; wrap
  it as `typing.cast(Binning, obj)` there.

- `nearest_bin` on an egocentric polar environment measures physical
  distance. It places the bin centers and the `(distance, angle)` rows in the
  animal's plane, `(distance * cos(angle), distance * sin(angle))`, before
  comparing them, so its distances, `max_distance` and the default
  far-point limit are in the environment's length units and angles a whole
  turn apart name one direction. It used to compare the stored columns as if
  centimeters and radians were one unit, and could return a bin on the other
  side of the animal. Results change for polar environments only.
  `bin_center_tree()` is still the tree over the stored columns.

- Every `Environment` method that returns bin indices returns `np.intp`, the
  dtype NumPy uses for indexing, as `bin_at` already did. `bin_sequence`,
  `bin_sequence_with_runs(...).bins`, `components` and `rings` returned
  `int32`, and `nearest_bin` was declared `int64`; all are now `np.intp`
  (64-bit on 64-bit systems). The values are unchanged. `neighbors` and
  `path_between` still return lists of Python ints.

- `env.smooth` takes the same field layout as `env.diffuse`: one field of
  shape `(n_bins,)` or a batch of shape `(n_bins, n_fields)`, one column per
  field. It used to refuse any 2-D array. An array with its bins on the
  second axis, such as population rates `(n_units, n_bins)`, is refused by
  both with `Fix: pass field.T` (or `fields.T`) instead of a bare shape
  error, and passing a `SpatialRateResult` or `SpatialRatesResult` raises
  `TypeError` naming the array to smooth:
  `env.smooth(result.firing_rates.T, bandwidth=...).T`.

- Common construction mistakes now say what failed, why and how to fix it,
  instead of surfacing an internal error:
  - a transposed `positions` array of shape `(n_dims, n_samples)` in
    `from_samples` or `open_field` raised `Exceeds the limit (4300 digits) for
    integer string conversion`; it now says the grid would need too many bins
    and to pass `positions.T`;
  - non-numeric positions name the conversion to apply
    (`df[['x', 'y']].to_numpy(dtype=float)`);
  - a shapely polygon passed as `arena=` asked to install an adapter package;
    it now says to pass `arena=OpenArena(polygon)`;
  - `Environment.from_bins(env.binning)` shows the two lines that build the
    faces first;
  - `env.rebin(1.5)` raised `TypeError: 'float' object is not iterable`; a
    factor that is not a whole number now raises `ValueError` suggesting
    `rebin(2)` or `rebin(bin_size=...)` with the matching size;
  - `env.apply_transform(np.eye(3))` raised `AttributeError`; a matrix now
    raises `TypeError` saying to wrap it, `AffineND(matrix)`;
  - `linear_track(endpoints=[0, 200])` raised `'int' object is not
    iterable`; it now shows the (x, y) points to pass;
  - a track graph whose nodes lack `pos` (`from_graph`, `maze(track_graph=)`)
    raised `KeyError: 'pos'`; it now names the node and how to set `pos`;
  - a `bin_size` sequence whose length differs from the number of position
    columns, or any sequence with `layout="Hexagonal"`, names the expected
    length instead of an internal argument;
  - `dimension_ranges` that is not one finite `(min, max)` pair per position
    column raises `ValueError` naming the rule it breaks (the number of
    pairs, a value that is not finite, or a minimum that is not below its
    maximum) and showing the pairs for your positions. A reversed pair such
    as `(100, 0)`, a zero-width pair such as `(50, 50)` or a non-finite pair
    used to build a grid silently;
  - `rebin` on bins that are not a rectangular grid gives a call that
    rebuilds those bins at the new size: `from_samples(..., layout="Hexagonal")`
    for hexagons, `from_graph(env.binning.linearization.graph, ...)` for a
    track, `from_layout(create_layout("TriangularMesh", ...))` for a mesh,
    `env.rebin(bin_size=...)` for a clipped grid, or `env.smooth(field,
    bandwidth=...)` for a coarser scale. A grid with unequal cells says so;
  - `rebin` factors that are not integers (including `2.0`), are 0 or larger
    than the grid, or have the wrong length; a 3-D transform passed to
    `apply_transform` on 2-D bins; three `endpoints` for `linear_track`
    (use `node_positions`); bare numbers in `maze(node_positions=...)`;
    an `edge_spacing` sequence with the wrong number of gaps; `bin_size` as
    text; a list of points passed to `from_polygon`; `grid_edges` that do not
    fit the `from_grid_mask` mask; and a list of names passed to
    `region_membership` all name the call that works. The suggested code in
    every refusal is run by the test suite on the input that was refused.

- An infinite coordinate in the positions given to `from_samples`,
  `open_field` or `environment_from_position` is treated as a missing sample,
  like NaN, which these factories already leave out. It used to raise
  `Requested binning overflows` (grids) or `OverflowError` (hexagons).

- Every lookup of a region by name raises `RegionNotFoundError` with the
  closest existing name, for example `Fix: pass region_name='reward'` for
  `env.bins_in_region("rewrd")`. This covers `env.regions[name]`,
  `bins_in_region`, `subset(region_names=...)`, the `Regions` editing methods
  (`remove`, `update_region`, `region_center`), `region_reward_field`,
  `compute_region_coverage`, `decision_region_entry_time` and
  `compute_decision_analysis`, which used to raise a bare `KeyError` or a
  `ValueError` without a suggestion. `RegionNotFoundError` is both a
  `KeyError` and a `ValueError`, so existing `except` blocks still catch it.

- `spatial_autocorrelation` raises `NotImplementedError` instead of
  `ValueError` on tracks, meshes, 3-D grids and grids with fewer than half of
  their cells active, the type used for every environment whose geometry
  cannot run an analysis. Code that caught `ValueError` around it should
  catch `NotImplementedError`; see the migration page. Rate-map
  `grid_score()` and `summary_table()` still report NaN there.

- `env.occupancy(..., time_allocation="linear")` on bins without cell shapes
  (bins passed to `Environment.from_bins` with only centers and sizes) raises
  `NotImplementedError` instead of a `ValueError` about locators, and its
  message, like the one for tracks, says why linear allocation needs each
  bin's outline and points to `time_allocation="start"`.

- On a track, `bin_at` and `nearest_bin(tie_break="closest_center")` search
  one KD-tree, so a position exactly equidistant from two or more bin centers
  goes to the same bin in both. `bin_at` used a tree of its own and could pick
  another of the equidistant bins; for such positions its answer can differ
  from earlier versions. Every other position gets the same bin as before.

- A track graph attribute that cannot be saved at all, such as a function or
  a lock, is refused when the track environment is built, with a `TypeError`
  naming the node or edge and the attribute. It used to fail later, when the
  environment was pickled or written. Attributes that pickle can store but
  JSON cannot (sets, dates, NaN) still build; `to_file`, `to_dict` and NWB
  export refuse them with a `ValueError` that now names the attribute.

- On a polar environment, wall geometry (`border_score`, view-cell rates with
  walls, `boundary_source=...`) and `time_allocation="linear"` rate maps raise
  `CartesianOnlyError` instead of `NotImplementedError`, like every other
  operation that reads polar bins as Cartesian positions. Wall geometry on a
  3-D environment still raises `NotImplementedError`. The error reference
  page lists the two error types for environments that cannot run an
  analysis.

- The linear track of `simulation.mazes.make_plus_maze` numbers its arms in
  the order they are laid out along the linear position, clockwise from
  north (north 0, east 1, south 2, west 3), as `Environment.maze(kind="plus")`
  does. East and south used to have each other's number in the
  `source_edge_id` of each bin and the `edge_id` of the track graph. Bin
  positions, linear positions and every analysis are unchanged. The
  simulated T and plus mazes are now built with `Environment.maze`.

- `env.plot()` names the axes of a 2-D environment by what the coordinates
  mean, as `env.plot_field()` does: "X Position (cm)" and "Y Position (cm)"
  on a Cartesian environment, and "Distance (cm)" and "Angle (rad)" with a
  free aspect ratio on a polar one. It used to label every environment
  "Dimension 0" and "Dimension 1" (or "X coordinate" and "Y coordinate").

- On an egocentric polar environment, `env.component_keys`, `env.bin_sources`
  and `env.get_bridge_attributes()` give the answers of any environment that
  was not combined: an empty tuple, `None` and an empty table. They used to
  raise `CartesianOnlyError`, although they read only the records that
  `Environment.combine` keeps and never the coordinates.

- `env.occupancy`, `env.bin_sequence`, `env.bin_sequence_with_runs` and
  `env.transitions(times=..., positions=...)` warn when the `max_gap` gate
  and the other interval gates exclude every interval, and when the median
  sampling interval is longer than `max_gap`, which usually means `times` are
  not in seconds (milliseconds turn every frame into a recording gap).
  `occupancy` also warns when no interval keeps any time because every sample
  is NaN or outside, with either `time_allocation`. They used to return an
  empty or mostly empty result silently. The same warning
  comes from `compute_spatial_rate(s)` and the behavior analyses that share
  the gate, which now also warn about a median interval longer than
  `max_gap` when some intervals survive.

- `env.smooth` and `env.diffuse` with `mode="average"` treat NaN bins as
  missing observations instead of zeros. They smooth the field with NaN read
  as 0 and the observed mask with the same operator, then divide, so holes
  are filled from their observed neighbors and a constant field with holes
  stays constant. A bin stays NaN when the share of its smoothing weight on
  observed bins is at most 0.1% of the largest share in its connected part of
  the environment, which includes every bin of a part with no observed bin.
  Binned rate maps and `resample_field(method="diffuse")` use the same
  masked-average code, each with its own far-field floor (a
  bandwidth-dependent one for binned maps, none for resampling), so their
  NaN bins far from observations can differ from `smooth`'s. `mode="transition"` and
  `mode="density"` refuse a field with NaN, because missing bins have no
  defensible mass under a conserving operator: fill them explicitly, or use
  `mode="average"` for an intensive field. `diffuse(backend="jax")` refuses
  a concrete field with NaN and points to `backend="numpy"`. Before, `smooth`
  refused any NaN and `diffuse` silently turned every bin NaN. Fields without
  NaN give exactly the same values as before.

- `compute_decision_analysis` represents each goal region by the region bin
  nearest the centroid of the region's bins, the same bin
  `trials_to_region_arrays` uses. It used the lowest-index bin, which depends
  on how bins are numbered and sits at the region's corner; the new choice
  depends on numbering only for ties, where the lowest index among the
  equally near bins is chosen. The goal labels
  and `distance_to_boundary` change for goal regions that cover more than one
  bin.

- `env.connectivity` is built from the environment's bins for every layout,
  so it carries the same node and edge attributes everywhere (`pos`,
  `source_grid_flat_index`, `original_grid_nd_index`; `distance`, `vector`,
  `edge_id` and, in 2-D, `angle_2d`), plus `pos_1D` and `source_edge_id` on
  tracks. Graph, node and edge attributes that a layout engine adds to its
  own graph are no longer published; keep such values in your own arrays
  indexed by bin ID, or annotate the graph returned by `env.to_networkx()`.

- `Environment.from_bins` on a binning with a triangular-mesh locator numbers
  each node's `source_grid_flat_index` and `original_grid_nd_index` by its
  triangle, as built mesh environments do and as `env.grid_shape` and
  `env.active_mask` describe; it used to give the bin ID. On a binning with a
  track linearization, `from_bins` now adds each node's `pos_1D` (linear
  position) and `source_edge_id` (track-graph edge), as track factories do.

- `env.with_metadata(...)` and `env.with_regions(...)` share their parent's
  analysis cache instead of starting with an empty one, because the bins are
  the same, so renaming an environment keeps the kernels and eigenbases
  already computed. Clearing the cache of one clears it for all of them.
  Derived environments (`subset`, `clip`, `rebin`, `apply_transform`,
  `combine`) and environments loaded from files, dicts, NWB or pickles start
  with their own empty cache. Each cached analysis is computed once even when
  several threads request it at the same time. `smooth`, `diffuse` and
  `get_mrf_basis` calls on one environment and its relabelled copies run one
  at a time, because each holds a lock on the shared cache for the whole call; use
  worker processes for parallel smoothing.

- A factory given an `arena=` that leaves out some bins (for example
  `from_grid_mask(..., arena=OpenArena(polygon))`) selects the bins before
  building the environment, instead of building it, subsetting it and
  building it again. The bins, graph and ranges are unchanged, but
  `layout_type` and `layout_parameters` now describe the factory's layout (for
  example `"MaskedGrid"` with `connect_diagonal_neighbors`) whether or not the
  arena dropped a bin; they used to switch to a subset record.

- `dimension_ranges` is the extent of the bins' cells for every layout, and
  `Environment.subset()` reports the extent of the cells it keeps, with no
  added margin, so a subset never reports a larger range than its parent.
  Hexagonal environments report their hexagons' extent, corners included,
  instead of the range the tiling was asked to cover (a tiling requested over
  `(0, 30)` with 4 cm hexagons reports `(-2, 36)`); triangular meshes report
  their boundary's bounds instead of the bounds of the triangle centroids;
  tracks report the centerline their bins cover, end to end, instead of the
  bin-center bounds (a 200 cm linear track reports `(0, 200)`, not
  `(1, 199)`). Subsets of hexagonal, triangular-mesh and clipped polygon-cell
  environments (`env.clip(...)`) used to add 5% on each side; a mesh subset
  now bounds only its kept triangles, so a subset of one corner of a mesh no
  longer reports the extent of the whole mesh, and a track subset bounds the
  centerline its kept bins cover. `apply_transform` reports the extent of the
  transformed cells rather than the box around the transformed ranges, which
  was wider after a rotation. Grids report the extent of their active cells:
  a grid subset reports the cells it keeps rather than the whole lattice it
  keeps in `grid_edges`, and a grid whose border rows or columns are all
  inactive reports its active cells. Napari draws a grid field over the full
  lattice, so its overlay and scale-bar mapping now read `grid_edges` and
  `grid_shape`, with each pixel centered on its cell (overlays used to be
  stretched by up to half a pixel). Readers of `dimension_ranges` change
  with it on hexagonal, mesh, track and grid-subset environments: plot limits
  (a hexagonal `env.plot()` is framed by its cells' extent, the same limits as
  `env.plot_field()`, with no extra margin), `env.info()`, the automatic scale of `ops.spatial_basis`, the "bandwidth
  spans the environment" warning, the uniform field placement of
  `simulate_session`, and the extent animation overlays and an uncalibrated
  video are drawn over. The same
  applies to the bins an `arena=` removes in `Environment.from_layout`,
  `from_grid_mask` and `from_pixel_mask`. For `Environment.from_bins`, the
  ranges of triangular-mesh bins bound the triangles' vertices rather than
  every stored mesh point. The migration page has a table for each layout.

- A `CartesianOnlyError` from a Cartesian-only method or property, such as
  `subset`, `rebin` or `interpolate`, names the class of the environment that
  refused the call. On a subclass of `Environment`,
  `polar_env.interpolate(field, points)` reports
  `MyEnvironment.interpolate() does not work on a polar environment.`, for the
  bound call and for `Environment.interpolate(polar_env, field, points)` alike.

- Derived environments no longer repeat the warning about unrecognized units.

- **Breaking:** Cartesian and egocentric polar factories now return one public
  `Environment` class. Use `env.coordinate_kind` for coordinate meaning and
  `Environment.from_polar_egocentric(...)` instead of the removed
  `EgocentricPolarEnvironment.create(...)`. Radius remains in length units and
  angle in radians; polar Cartesian-only calls still refuse, including unbound
  calls. Factory-inferred static types preserve those restrictions. Supported
  dictionary, file and NWB records retain their coordinate semantics.

- Points in discarded subset bins map to `-1` rather than a nearby retained bin.
  Non-uniform scaling of hexagonal or mesh environments and general affine maps
  disable finite-volume smoothing with an explanation. Tracks refuse
  non-similarity transforms; rebuild them from transformed graph nodes instead.

- Environment files and dictionaries now use `Environment-v3`; NWB environment
  records use schema `3.0`. They preserve Arena, walls, exact cells and clip
  mappings without rebuilding layouts. Version 2/`2.0` remains readable with an
  explicitly unknown Arena. Version 1/`1.x` must be recreated with a factory.
- Polygon-aware lookup excludes points outside the declared domain and in its
  holes. Linear occupancy subdivides entry, exit and re-entry intervals on
  exact polygon support, and refuses unresolved physical-wall crossings.
  Default encoding and occupancy retain their existing interval-allocation
  conventions.
- Rebinning preserves the original arena and selects coarse bins by center,
  including inferred mask holes. Physical walls remain closed; coarse cells
  that cannot represent them are refused.

- Point lookup consistently rejects incorrect coordinate widths, including
  hexagonal inputs with extra columns and empty or non-finite grid queries.

- `Environment.bin_sizes` now returns an owned, read-only array, like
  `bin_centers`.

- Rewrite all nine tutorials around spatial neuroscience questions, using
  simple technical English, explanations of analysis choices, figure
  interpretation, and instructions for using recorded data. Update the shared
  catalog and keep example-data setup separate from the main analysis.

- Consolidate the 27 feature-oriented notebooks into nine standalone task
  tutorials covering environments, rate maps, behavior, real data, simulation,
  held-out decoding, events, visualization and NWB. Combine the J16 neural and
  maze analyses, add executable specialized recipes in the user guide, and
  update catalog, links, mirrored outputs and full-collection CI.

- **Breaking:** `env.smooth` now defaults to `mode="average"`, preserving the
  units and level of a rate map. Pass `mode="density"` explicitly to retain
  the previous count-to-density default; kernel constructors are unchanged.
- **Breaking:** field detection and population coverage now default to no
  map-mean exclusion (`max_mean_rate=np.inf`). Pass `max_mean_rate=10.0` to
  retain the previous filter. Its unweighted spatial-bin mean is not a
  recording-time mean or a cell-type verdict. Coverage displays report maps
  with segmented fields rather than asserting cell identity.

- Review all 27 tutorials for coherent workflows, current APIs, valid geometry,
  observed support, independent simulation randomness and calibrated scientific
  language. Correct rate-map and direction labels, add held-out decoding,
  preserve real-data transition endpoints/gaps, and make desktop viewers and
  long video exports opt-in. Refresh worked outputs and align prerequisite links.

- Directional place fields, decode summaries, batch scores, behavior metrics
  and run bins, assembly results and trajectory-fit results now own read-only
  copies of their array fields, including after pickle/deepcopy. Directional
  mappings also prevent replacing entries. **Behavior change:** copy arrays
  before editing values. Decode-summary xarray exports remain editable copies.
- Rate and peri-event results store read-only copies of their arrays. This
  covers spatial, directional, view and object-vector results (single-unit
  and population) and both peri-event results. Editing an array a result was
  built from no longer changes it, and editing a result's arrays in place, for
  example `rates.firing_rates /= peak`, raises instead of silently changing
  what the result reports. **Breaking:** copy before modifying values:
  `normalized = np.array(rates.firing_rates) / peak`. `to_xarray()` and
  `to_dataframe()` return editable copies; editing a `to_xarray()` Dataset
  used to edit the result.
- `interpolate(mode="linear")` and `occupancy(time_allocation="linear")` work
  on environments built with `from_polygon`, `from_grid_mask` and
  `from_pixel_mask`, and on grid environments read from NWB. They previously
  required a `RegularGrid` layout. Inactive cells behave as on masked
  `from_samples` environments.
- Polar environments no longer define the twelve Cartesian methods. Accessing
  one raises `CartesianOnlyError`, both a `NotImplementedError` and an
  `AttributeError`, with the same message. `hasattr` reports them as absent and
  type checkers flag the call. Refusals inside shared methods name the call
  that was made.
- **Behavior change:** an environment's geometry is read-only once built:
  its geometry attributes cannot be reassigned, `bin_centers`, `active_mask`
  and `grid_edges` reject writes, and `connectivity` is frozen. Cached kernels and operators cannot see in-place
  edits, so such edits used to return stale results silently. Build a new
  environment (a factory, `subset`, `rebin` or `apply_transform`) instead.
  Factories copy array inputs, so the caller's own arrays stay writable and
  later edits to them do not reach the environment.
- Layout setup synchronizes restored geometry and invalidates all environment
  caches. The polar layout builds physical distances and circular seams before
  environment initialization, and records its `circular_angle` build option.
- Layout engines, including a polar layout with annular-sector cells, supply
  their own finite-volume cell faces. Dense kernels, spectral diffusion,
  smoothness bases, gradients and divergence all use that one capability, and
  the diffusion consumers share one validated operator snapshot. Numerical
  schemes are unchanged.
- **Behavior change:** `compute_diffusion_kernels` rejects directed graphs and
  multigraphs, whose reciprocal or parallel edges were summed into doubled
  conductances. Collapse them first with `nx.Graph(graph)`.

- Extend rate-result `analysis_settings` and `spatial_support` to directional,
  view and object-vector analyses, preserving exact retained intervals and
  family settings through indexing, tables and NetCDF-compatible xarray exports.
  Shared binning preparation keeps counts, occupancy and provenance aligned.
- **Behavior change:** view and smoothed object-vector encoders now use the same
  unit-aware physical bandwidth default as spatial encoding. Polar results
  inherit the supplied environment's distance units; angular smoothing retains
  its existing units and defaults.

- **Behavior change:** spatial encoding defaults to a 5 cm smoothing bandwidth
  in declared metric units (0.05 m / 50 mm), or 2.5 median graph-edge lengths
  for unspecified/custom units. Explicit bandwidths retain their coordinate
  units; bandwidths spanning the environment warn about lost spatial tuning.
- Align unit-indexed metadata by identity in spike containers and population
  results. Reject missing, duplicate or conflicting metadata identities; an
  unnamed default RangeIndex retains positional interpretation.

- **Breaking:** video and HTML animation exports default to `overwrite=False`, refusing existing outputs before rendering or dry-run estimates. HTML checks its default filename and nonempty frames directory before creating directories; ffmpeg uses `-n` unless overwrite is explicit. Environment saves share the same paired-target check and corrected-call wording.

- **Breaking:** rate summary tables lead with rate/information, place classification last, and carry estimator, physical units, direction frame and resolved classification thresholds in `DataFrame.attrs`. Singular/population columns match; constant `method` is no longer a summary column, and labels are documented as heuristic screens.

- **Breaking:** population rate summaries name `n_units` and `max_peak_firing_rate` explicitly. Shared occupancy remains seconds observed once; singular summaries add cheap family metrics and stay safe for all-NaN maps without computing grid/border scores.

- **Breaking:** NWB position, head-direction and units readers return frozen, non-iterable `NWBPosition`, `NWBHeadDirection` and `NWBUnits` holders. Position holders expose normalized declared physical units without a second scaling step or an assumed unit; eager/lazy reads retain their existing conversion and lifetime contracts.

- **Breaking:** simulation validation and summary plotting each take a required `SimulationSession` with keyword-only `unit_ids` selection by label. The raw keyword validation form and row-index selection keywords are removed; unknown labels explain the available labels and corrected call.

- **Breaking:** align `SimulationSession` attributes with analysis inputs: `spike_times`, integer `unit_ids`, and ground truth keyed by unit label. The frozen holder checks that spikes, labels and models align one-to-one; seeded trajectory, spike and duration behavior is unchanged.

- **Breaking:** require tracking positions in spatial rate computation, decoder `fit`/`score`, and `decode_session(_summary)`. Session decoding always encodes tracking and no longer accepts `encoding_models=`; estimator predictions take only spikes and array timestamps and reuse the extracted gap-aware full/streamed model decoders.

- **Breaking:** standardize required trajectory pairs as `(times, positions)` in `approach_rate`, `compute_decision_analysis`, `compute_goal_directed_metrics`, `compute_path_efficiency`, `compute_pre_decision_metrics`, `compute_vte_session`, `compute_vte_trial`, `extract_pre_decision_window`, `goal_bias`, `head_sweep_from_positions`, `instantaneous_goal_alignment`, `mean_square_displacement`, `pre_decision_heading_stats`, `pre_decision_speed_stats`, `segment_by_velocity`, `time_efficiency`, `heading_from_velocity`, `visibility_occupancy` and `generate_population_spikes`. Place `env` first in `compute_region_coverage`, `field_shape_metrics`, `rate_map_coherence`, `map_points_to_bins`, `shuffle_place_fields_circular_2d` and `calibrate_video`. Rename the spike parameters on `bin_spikes_in_time`, `population_peri_event_histogram`, `restrict_spike_trains` and `validate_simulation` to `spike_times`. Analysis entry points take explicit arrays rather than a `PositionLike` adapter.

- **Breaking:** rename the phase-precession callable to `compute_phase_precession`, preserving the sibling `encoding.phase_precession` module for normal imports.

- **Breaking:** rename place-field detection to `has_place_field`; `is_place_cell` now requires a classification criterion. Align free predicates, result methods and batch screens on read-only threshold defaults and inclusive cutoffs, make method thresholds keyword-only, rename spatial `classify(min_spatial_info=)` to `min_info`, and propagate invalid-input errors.

- **Breaking:** default `ObjectVectorCellModel` to allocentric direction tuning without headings, which changes simulated spikes, without an error, for code that relied on heading-relative tuning. Set `direction_frame="egocentric"` to retain heading-relative behavior; model ground truth records the frame and missing-heading errors explain both choices.

- Plot object-vector tuning in its recorded frame: allocentric zero points East and positive angles turn toward North; the corrected egocentric ahead/left orientation is retained. Frame-specific literature and direction conventions are explicit.

- Rename object-vector results to `ObjectVectorRateResult` and `ObjectVectorRatesResult`, and their information accessor to `spatial_information()`. Removed names have no aliases; indexed population results preserve their frame.

- `Environment.combine`, and `subset` and loading of a combined environment,
  check component overlap only for cells whose bounding boxes intersect instead
  of every pair of cells. Combining two 1 m square arenas at 2 cm bins takes
  about 2 s instead of 21 s; the overlap check for two 1 m arenas at 1 cm bins
  takes 0.9 s instead of 141 s. The same overlaps, contacts and error messages
  are reported.

### Added

- `env.coordinate_system` says what it is: `kind` is `"cartesian"` or
  `"polar"`, `circular` says whether the angle axis wraps around (always
  `False` for Cartesian coordinates), and `periodic_axes` lists the axes whose
  two ends are neighbors (`(1,)` for a full-circle polar environment, otherwise
  empty). A custom analysis can ask the descriptor directly instead of
  checking its class. A polar environment's wrap is recorded once, by its
  `EgocentricSurround` arena (`env.arena.circular`), and the descriptor is
  derived from it, so the two can never disagree.

- `Environment.with_track_width(width)` gives a track its measured full width
  and keeps its bins, linear coordinates, name, units, frame and regions. Use
  it on a track built without `track_width`, such as one saved by an earlier
  version, which refuses to look up 2-D positions: after
  `Environment.from_file(path).with_track_width(10.0)`, positions more than
  5 cm from the centerline get bin `-1` and rate maps, occupancy and decoding
  work again. It adds `"with_track_width"` to `env.derivations` and refuses an
  environment that is not a track.

- The cell-shape records in `neurospatial.binning` (`HexagonShapes`,
  `TriangleShapes`, `PolygonCells`) have `extent()`, the `(low, high)` bounds
  of their cells per axis, and `subset(keep)`, the cells selected by a boolean
  mask of one entry per bin, in their original order. These are the same
  rules `env.dimension_ranges` and `env.subset()` use, so explicit bins built
  for `Environment.from_bins()` can be bounded and subset the same way.

- `neurospatial.ops.KernelMode` names the values `mode=` accepts in
  `Environment.compute_kernel`, `smooth` and `diffuse`
  (`"transition"`, `"density"`, `"average"`), for type annotations in your
  own analysis code.

- `neurospatial.layout.ScaleBarConfig` is the scale-bar settings class that
  `Environment.plot(scale_bar=...)` accepts. It is the same class as
  `neurospatial.animation.ScaleBarConfig`, now importable without the
  animation package.

- `neurospatial.encoding.autocorrelation_pixel_size(env)` gives the side, in
  the environment's units, of the square pixels of `spatial_autocorrelation`.
  Pass it as `bin_size` to `grid_scale` and `grid_properties`.

- `Environment.bin_center_tree()` returns the KD-tree over `bin_centers`
  that `nearest_bin` searches, built once and kept in the analysis cache.
  Use it for the `k` nearest bins with their distances.
  `map_probabilities`, `select_basis_centers(method="kmeans")`,
  `heat_kernel_wavelet_basis` and the simulated place cells now search this
  tree instead of building their own, with identical results; repeated
  `map_probabilities` calls onto a 160,000-bin grid take 28 ms instead of
  38 ms.

- `neurospatial.environment.BinSequenceWithRuns`, the result type of
  `Environment.bin_sequence_with_runs()`, is importable from a public path for
  type annotations and `isinstance` checks.

- `Environment.from_layout` accepts your own layout engine after its
  `build(...)`, with no registration. The engine can be a plain class with
  only `build(...)` and `binning()`. Subclasses of built-in engines, which
  used to be refused, now build environments normally.

- `Binning` has a `dimension_ranges` field: the `(low, high)` extent of each
  axis the bins' cells cover. `env.binning.dimension_ranges` equals
  `env.dimension_ranges`, so `Environment.from_bins(env.binning)` keeps the
  ranges. An environment computes the ranges from the cells (grid, outline,
  cell shapes or a track's centerline); only bins without cells, such as
  point bins, use the value set on the `Binning`.

- `subset()` and `rebin()` accept `name=` like `clip()` and
  `apply_transform()`.

- `from_polygon`, `from_grid_mask`, `from_pixel_mask`, `from_graph`,
  `linear_track` and `maze` accept `regions=`, so every Cartesian factory
  takes the same `name=`, `units=`, `frame=` and `regions=` keywords.
  `open_field` accepts `regions=` and `arena=`, and `from_samples` and
  `open_field` accept `dimension_ranges=` as a named option.

- Every arena type has `domain_bounds()`, the lower and upper corners of a box
  that encloses its known Cartesian domain (`None` for an unknown domain, a
  track without a width or a polar surround). `Environment.rebin(bin_size=...)`
  uses it to lay out the new grid.

- `neurospatial.ops.geodesic_distance_between_points` accepts `weight=`, the
  edge attribute that measures length (`"distance"` by default; `None` counts
  edges). `Environment.distance_between` now calls it, so the two always
  agree.

- Expose detached, read-only `env.binning` snapshots and public coordinate,
  diffusion-geometry, diffusion-accuracy, MRF-basis and Gaussian-kernel
  operations. Environment owns analysis caches; numerical consumers use public
  operations or explicit geometry. Existing public geometry imports remain
  stable while owned data and codecs move into an independent `geometry/`
  package with enforced dependency rules.

- Preserve linear interpolation and movement time fractions after rotating,
  reflecting or shearing Cartesian 2-D/3-D grids. Planar grids use exact QuadMesh
  plots/animation updates. Planar linear spike counts and dwell share bounded
  source-plane ownership, including paths along inactive-cell boundaries.
  Napari raster helpers explicitly refuse framed grids.

- Certify new affine numerical transforms from stored physical normals and flux
  distances. Unequal scaling along retained orthogonal grid axes can preserve
  diffusion, smoothing, gradients and divergence; skew faces refuse with a
  geometric reason. Existing approximate-mesh similarity behavior is preserved.

- Retain 2-D/3-D source grid frames through affine transforms and whole-bin
  subsets, without exposing source axes as world-aligned grid edges. Reflected
  and deformed hexagons retain exact polygon cells. New binning codec version 3
  stores the frame and reads earlier versions 1 and 2; partial clipping uses
  exact polygon cells rather than a tensor grid.

- Add optional owned `FaceNormals` data to `Binning`, in authoritative neighbor
  pair order. Native Cartesian grids and planar hexagon/triangle cells capture
  physical directions; subsets, clipping, physical walls and affine transforms
  preserve their alignment. Normals survive copying and JSON/NWB storage, but
  enable new numerical transforms only with the source/target face certificate.
  Codec version 2 introduced normals; current version 3 also stores grid frames
  and reads legacy versions 1 and 2. Older readers reject the new nested version.

- Ray casting and cue line of sight now intersect declared physical walls,
  including holes and internal barriers. Analysis crops no longer occlude
  physical rays. **Breaking:** use `boundary_source="analysis"` explicitly for
  analysis-cell boundaries when physical walls are unknown. No wall hit within
  the search distance returns NaN. View encoding, shuffle tests and simulation
  forward and record the choice; wall hits use observer-side cell ownership.
  Stored frames retain bounded boundary-contact behavior after transforms.
  Viewshed bin coverage remains sampled; unsupported wall/owner geometry refuses.

- Border scores now use declared physical footprint rings and internal walls
  by default, with wall-length coverage and distances to actual walls.
  **Breaking:** explicitly choose `boundary_source="analysis"` for the legacy
  boundary-bin approximation when only analysis geometry is available. Result,
  population and classification methods propagate the choice. Summary tables
  explain unavailable wall metrics without dropping other metrics. Physical
  coverage requires exact planar cells and polygonal wall geometry.

- Add `Environment.rebin(bin_size=...)` for rectangular grids, rebuilding over
  the retained arena with an owned `GridBinningSpec` separate from provenance.
  Grid axes and anchor survive crops, clipping, affine transforms and JSON/NWB
  storage; new sizes are physical lengths along those axes. Other families and
  records without a specification refuse this mode. Integer-factor coarsening
  keeps its truncation warning and now also accepts masked and polygon grids
  with uniform rectangular cells. Fields and fitted models are not transferred.

- Add `Environment.combine` for owned disjoint components, with stable keyed
  bin sources, qualified regions and deterministic mutual-nearest-neighbor
  bridges that respect known physical walls. Path-only bridges do not diffuse;
  explicit geometric conductance couples certified child blocks. Exact lookup,
  component geometry and bridge data survive copy, derivation and JSON/NWB
  storage without live provider objects or bridge re-inference.
  **Breaking:** remove `CompositeEnvironment`; use
  `Environment.combine({"key": env, ...})` and the ordinary environment APIs.
  Overlapping or uncertifiable support and incompatible declarations raise.

- Add `Environment.with_metadata(...)` and `with_regions(...)` for immutable
  environment values. Metadata replacements share geometry and start fresh
  analysis caches. **Breaking:** assign units/frame/name at construction or use
  a replacement; edit a standalone `Regions` builder, then publish it with
  `with_regions`. Environment snapshots own caller data and protect geometry,
  provenance, regions and cached numerical exports through copy and persistence.
- Add `Environment.from_bins(...)` with public `neurospatial.binning` records,
  explicit lookup and explicit face availability. `env.edge_pairs` preserves
  supplied neighbor order and defines gradient/flux order. `to_networkx()`
  returns an independent editable graph. Declared navigation path lengths are
  preserved by rigid transforms and scaled by uniform scaling; other affine
  maps require known paths or rebuilt distances.
- Add installed `SnapshotAdapter` objects in the versioned
  `neurospatial.snapshot_adapters.v1` entry-point group, plus
  `neurospatial.testing.assert_environment_contract`. Custom providers must
  convert into supported built-in data snapshots. Original custom saved records
  require their adapter package; canonical resaves use built-in codecs.
- Preserve hexagonal grid shape and active mask across explicit-bin construction
  and persistence, keeping grid-based spatial autocorrelation available.

- Add continuous Arena domains alongside spatial bins. Tracked positions,
  grid masks and image masks remain direct inputs and infer computational
  support without asserting physical walls. Optional polygons, explicit
  internal walls and measured track widths describe known physical geometry.
- Add `Environment.clip(polygon)` for exact planar cell intersections,
  including holes and split bins, with a read-only immediate-parent
  `clip_mapping`. Plotting and containment work without numerical certification;
  smoothing and spatial operators require certified finite-volume geometry.
  Clipping does not transfer fields or fitted models automatically.
- Add explicit `time_allocation="linear"` to single-unit and population spatial
  rate estimation. Spike counts and dwell time share exact retained entry,
  exit and re-entry intervals; the default remains start allocation.

- Plot actual positions directly on decoded posteriors with
  `DecodingResult.plot(true_times=..., true_positions=..., max_gap=...)`.
  Ground-truth alignment is shared with `error_against`, preserving tracking
  gaps and missing observations without manual coordinate conversions.
- Add `plot_peri_event_response` for a matching raster and PSTH. Both panels
  reuse the result's original event cohort and complete half-open bins,
  including exact boundary handling on the absolute recording clock.
- Add `detect_region_visits` and `region_transitions`, with immutable visit and
  transition records preserving region identity, original sample indices and
  continuous observation runs. Partial visits retain observed-boundary flags;
  entry-to-entry transitions include source dwell time and never cross missing
  tracking, recording gaps or excluded epochs.

- Peri-event results expose read-only `event_indices` identifying retained
  events in the original input. Indexing a population preserves this cohort,
  so rasters, PSTHs and event tables can share exposure filtering without
  duplicating it. Hand-built results use `None` for unavailable provenance.

- Check the complete notebook collection in CI, including paired source and
  documentation parity, local links and an acyclic prerequisite graph. Generate
  one shared catalog from notebook metadata for the repository and docs site.

- Download and cache the real J16 bandit recording from
  Zenodo (10.5281/zenodo.23020757), verifying its
  pinned SHA-256 before loading numeric NPZ arrays without pickle. The real-data
  tutorial fetches this archive explicitly and reuses it offline; data
  instructions include the source, license and citation.

- Select units by label with `sel(unit_id=...)` on `SpikeTrains` and every
  population result (spatial, directional, view, object-vector and peri-event).
  `[]` selects by position on all of them, so `spikes[i]` and `rates[i]` are
  the same neuron. A label passed to `[]`, or a position out of range, raises
  an error that names the `sel` call to use. **Breaking:** `SpikeTrains[...]`
  used to look up a label; replace `st[unit_id]` with `st.sel(unit_id=unit_id)`.
- Build a fitted decoder from hand-built rate maps, such as a simulation's
  ground truth, with `BayesianDecoder.from_models(env, encoding_models,
  unit_ids=..., spatial_support=..., dt=...)`. Without `unit_ids`, spike trains
  pair with models by position. **Breaking:** the `BayesianDecoder` constructor
  now takes configuration only. `encoding_models`, `unit_ids`,
  `spatial_support`, `spike_window` and `analysis_settings` are read-only
  attributes, set by `fit`, `from_rates` or `from_models`; replace
  `BayesianDecoder(env, encoding_models=m, unit_ids=ids)` with
  `BayesianDecoder.from_models(env, m, unit_ids=ids)`.
  `dataclasses.replace(decoder, dt=...)` keeps the fitted models.
  Changing fitted encoding settings (`method`, `bandwidth`, `min_occupancy`,
  `penalty`, `rank`) with `replace` now raises; construct and fit a new decoder.
  Prediction defaults remain replaceable. `from_rates` uses the rate result's
  method, bandwidth and recorded encoding settings.
- Add `Environment.get_divergence_operator()`, the cached finite-volume
  divergence operator that `divergence(env, edge_field)` applies, alongside
  `get_differential_operator()`.
- Preserve spatial `analysis_settings` through population indexing, tables,
  xarray/NetCDF and NWB schema 2.4, including selected/observed windows, gap,
  speed and occupancy gates. Expose `spatial_support` on rate results and fitted
  decoders. Legacy results identify unavailable settings as `None`.
- Add runnable condition-comparison and held-out decoding workflows and a
  scientific-task index linking recording import through analysis and export.

- Add scientific invariants across layout transforms, coordinate-unit conversions, single/population estimation, labelled unit permutations and disk NWB round trips, covering grid, masked, hexagonal and graph layouts.

- Give singular rate results the same metric tables and labeled xarray Dataset export as population results, including GAM slices, unit identity, frame and recording-window metadata. Shared table builders/classifiers retain the established statistics and thresholds without wrapping singular GAM results in artificial populations.

- Preserve NWB units' per-unit `obs_intervals` and their shared acquisition `spike_window`, including subset selection and lazy-spike reads. Population analyses can use this intersection explicitly; absent coverage stays None and disjoint coverage stays an empty window.

- Build fitted spatial decoders from `SpatialRatesResult` with `BayesianDecoder.from_rates`, carrying the environment, precision, unit identity, training support and analysis settings. Generated labels remain positional; caller-supplied labels align by identity, including direct constructors, indexing and dataclass replacement. Prediction excludes bins without supported training observations.

- Accept `BayesianDecoder.fit(unit_ids=)` and preserve caller-supplied unit identity for prediction alignment. Labels supplied alongside a spike group must match its labels and order; the shared resolver rejects duplicates.

- Snapshot all public namespace names and function signatures in one sorted, reviewable file, with a unified diff and explicit regeneration command for intentional API changes.

- Give lazy root exports concrete static types for IDEs and type checkers, and verify each headline class/function with a CI typing contract without eager runtime imports.

- Expose spatial rate computation/results, position decoding/results and peri-event histogram/results lazily at the root, alongside core spatial types and domain namespaces.

- Add opt-in `criterion="shuffle"` to all five raw-array cell predicates, with explicit stream labels and p-value levels. Mode-specific keywords raise together when they would be ignored; result methods and batch classifiers remain threshold screens without retained inputs.

- Add five raw-array circular-shift significance functions for place, head-direction, spatial-view and both object-vector frames. Shared valid-window masks, copied inputs and label-keyed random streams preserve recording coverage and seeded single/population agreement.

- Add seeded circular spike-time shifts over joined valid recording windows. Shifts preserve retained spike counts and compressed-clock circular spacings without placing spikes in gaps.

- Add `is_egocentric_object_vector_cell` for heading-relative tuning. **Breaking:** `is_object_vector_cell` now measures allocentric direction and no longer accepts `headings`; call `is_egocentric_object_vector_cell` for the previous heading-relative verdict; each frame predicate forwards its encoder options and retains the existing information threshold and error behavior.

- Add `compute_object_vector_rate(s)` for allocentric animal-to-object direction alongside the egocentric encoders. Both use the shared polar binning, smoothing and recording-window rules, and record a required `direction_frame` on singular and population results.

### Performance

- `from neurospatial import Environment` no longer imports matplotlib or
  `pandas`; they load when you first plot or ask for a table. The import
  takes about 0.19 s instead of 0.42 s, which helps scripts and worker
  processes that only bin positions.

- Renaming an environment no longer discards its cached analyses: on a
  3,586-bin open field, the first `compute_spatial_rate` call after
  `with_metadata(name=...)` takes 0.51 s instead of 0.88 s.

- Reading graph attributes of `env.connectivity` is about 5 times faster, and
  Dijkstra shortest paths on `env.connectivity` about 2 times faster, because
  attributes are frozen once instead of copied on every read.

- A 200 × 200 grid mask cropped by an `arena=` builds about 2.7 times faster,
  because the bins are selected before the environment is built.

- Building an environment checks its connectivity graph against its bins with
  array operations instead of a Python loop over every bin and pair: for a
  22,500-bin grid the check takes 0.08 s instead of 0.45 s.

- `Environment.occupancy(time_allocation="linear")` on 1-D and 3-D
  rectangular grids splits every interval across the cells its straight path
  crosses at once, instead of one interval at a time: a 20,000-sample 3-D
  trajectory sampled at 5 Hz on a 21 × 21 × 21 grid takes 0.07 s instead of
  1.2 s. Occupancy values are unchanged.

- `Environment.distance_to(metric="euclidean")` and
  `neurospatial.ops.distance_field(metric="euclidean")` find the nearest
  target with a KD-tree instead of a bins × targets distance array: on a
  90,000-bin grid with 22,500 target bins the call takes 0.03 s instead of
  56 s and about 32 GB of temporary memory. Distances can differ from before
  only in the last digit of floating-point rounding.

- `Environment.transitions(times=..., positions=...)` with the default
  `allow_teleports=False` checks that each step joins neighboring bins with a
  sparse adjacency lookup instead of a Python loop: 1,000,000 steps on a
  90,000-bin grid take 0.04 s instead of 0.49 s. The counts are unchanged.

- `nearest_bin` and `interpolate(mode="nearest")` resolve a point that lies
  equally close to several bins by measuring only the bins near that point,
  instead of every bin: for points on the cell corners of a 10,000-bin grid
  this is about 100 times faster and no longer allocates an array the size
  of points × bins. The chosen bin (the lowest-numbered tied bin) is
  unchanged.

- `subset` and `apply_transform` reuse the parent environment's cell faces
  once smoothing or another diffusion analysis has built them, instead of
  building them again: on a 90,000-bin grid `subset` takes 0.43 s instead of
  2.2 s after such an analysis.

- Building an environment is much faster. Layout engines find neighboring
  bins with array operations instead of building a NetworkX graph one bin at
  a time, which the environment then discarded, and the environment builds
  its own read-only graph in one pass. On a 300 × 300 grid mask
  `Environment.from_grid_mask` takes 0.48 s instead of 4.4 s and `subset`
  0.48 s instead of 3.2 s; `Environment.from_samples` with 100,000 positions
  (bin size 1, about 10,000 bins) 0.05 s instead of 0.57 s; a 20,000-bin
  linear track 0.04 s instead of 0.33 s. Bins, neighbor pairs
  (`env.edge_pairs`), their order and their distances are unchanged.

### Fixed

- `simulate_trajectory_ou(boundary_mode="stop")` holds the animal at the
  wall it reached. Before, the clamp used `bin_at`, which returns `-1` for a
  position outside, and moved the animal to the last bin center, often the
  far corner, where it stayed. `boundary_mode="reflect"` used the same clamp
  after a reflection.

- `PlaceCellModel` and `ObjectVectorCellModel` with `metric="geodesic"` give
  a NaN rate for a NaN position, as `metric="euclidean"` does. Before, the
  missing position took the rate at bin 0.

- A sample outside the environment (bin `-1`) is in no bin in
  `trajectory_similarity`, `compute_home_range` and
  `detect_goal_directed_runs`. Before, `-1` indexed the last bin:
  `trajectory_similarity` scored an outside sample as the last bin
  (Hausdorff, DTW) or as a place both trajectories shared (Jaccard),
  `compute_home_range` returned `-1` as a home-range bin and counted outside
  time in the total, and `detect_goal_directed_runs` failed with
  `NodeNotFound: Node -1`. Now `trajectory_similarity` compares only places
  inside the environment and refuses a trajectory with none,
  `compute_home_range` leaves outside samples out, and
  `detect_goal_directed_runs` ends a recording at an outside sample.

- An environment pickled by another version of neurospatial is refused when
  it is loaded, with the advice to save with `env.to_file(path)` and read with
  `Environment.from_file(path)`. Before, a pickle from the previous release
  loaded and then failed on its first `subset`, `apply_transform`, `rebin` or
  `to_dict` with an `AttributeError`. Pickles now carry a format version.

- On a grid rotated or sheared by `apply_transform`, a position exactly on the
  line (or, in 3-D, the face) between two bins now belongs to the same bin as
  on the grid before the transform: the bin above the line, as on any
  rectangular grid. Before, rounding in the rotation put such a position in
  either bin, or outside the environment, so `bin_at`, `bin_sequence`,
  `occupancy(time_allocation="start")` and spike counts could disagree with
  `occupancy(time_allocation="linear")`, which already used this rule. In a
  test lattice of cell centers, cell edges and corners, 16–41% of the points
  changed bins after a rotation. Positions inside a bin are unchanged, and
  the same rule now applies inside environments built with
  `Environment.combine`. Finding bins on a rotated grid takes about 1.4 times
  as long (0.42 s instead of 0.31 s for a million 2-D positions). A rotated
  grid that is then cut with `clip` has polygon cells instead of a grid, and
  points exactly on its cell edges can still fall outside every cell, as
  before.

- `rebin(2)` on a pixel-mask grid that was clipped to a polygon suggests
  the coarser cell size, `env.rebin(bin_size=10)` for 5 cm pixels, instead
  of `env.rebin(bin_size=...)`. Each built-in layout engine now names the
  option that sets its cell size (`size_parameter`, such as `pixel_size` or
  `hexagon_width`), and the suggestion reads that option. Bins from a custom
  engine no longer get a size guessed from a recorded `bin_size`.

- After `apply_transform` scales the bins, a refused `rebin(2)` suggests
  twice their current size. It used to double the size they were built at,
  so hexagons built 4 cm wide and scaled by two were suggested a rebuild at
  `bin_size=8`, the width they already had, instead of 16; tracks, meshes
  and clipped grids had the same error. After a transform that stretches
  some directions more than others, the suggestion gives no size, since
  the cells no longer have one. Files, dictionaries and NWB files save the
  scale as `length_scale`; files saved earlier give no size after a
  transform.

- After `apply_transform`, a track with a known width still says so. The
  `rebin` refusal on such a track suggested an `Environment.from_graph(...)`
  rebuild without `track_width=`, so following it lost the width; it now
  passes the width in the transformed units (20 for a 10 cm track scaled by
  two). The refusal of `time_allocation="linear"` on a transformed track
  also names the track instead of "this arena".

- `apply_transform` accepts a rotation by an angle so small that its sine is
  below the normal floating-point range (about 1e-308), such as 2e-313
  radians. The check that the stored grid frame matches the transform used
  only relative rounding errors, which vanish for such tiny values, so it
  refused the environment it had just built. Its error bound now also counts
  the absolute rounding of these tiny products.

- A grid environment that went through a chain of scalings and translations
  no longer refuses its own inverse transform. Each step mapped the cell edges
  and, separately, the stored bin centers; after a few steps the two disagreed
  by a few rounding units and the inverse failed with "bins.centers and
  locator geometry does not match the binning geometry". The centers of a
  scaled or translated grid are now the midpoints of its mapped edges. Values
  change by at most one rounding unit of the coordinates (7e-15 cm at 35 cm).

- `Environment.animate_fields` warns once when you pass an option the chosen
  backend ignores, such as `crf=` with `backend="html"` or `overwrite=True`
  with napari, naming the backend that uses it and the call that would. With
  `backend="auto"` the warning says which backend was chosen and why.
  Backend options left out use each backend's own default.

- Video export uses constant-quality encoding (`crf=18`) by default, and
  `crf=` reaches the encoder. `Environment.animate_fields` used to send
  `bitrate=5000` on every call, which replaced constant quality, so `crf=`
  was silently ignored. For a short test animation the default output is
  visually the same (SSIM 0.9999 against the old encoding) and less than
  half the size. Passing both `crf=` and `bitrate=` warns that the bitrate
  wins and `crf` is ignored.

- `Environment.animate_fields(..., dry_run=True)` writes nothing on every
  backend and returns `None` after reporting what it would render. The HTML
  backend ignored `dry_run` and wrote `animation.html` into the working
  directory; napari opened a viewer and the widget displayed. The HTML backend
  now also draws `scale_bar=`, which it ignored.

- Time-series overlays are drawn by the video and widget backends of
  `animate_fields`. Those backends never received the frame timestamps the
  panels are aligned to, so the panels were silently left out.

- Static type checkers accept `Environment.from_polygon(polygon, ...)` with a
  shapely `Polygon`; it was annotated as taking the `Polygon` class itself.
  `from_file`, `from_dict` and `from_nwb` (and `neurospatial.io.from_file`,
  `from_dict` and `neurospatial.io.nwb.read_environment`) keep the static type
  `Environment[Any]`, because the stored data decides whether the
  environment is Cartesian or polar; their docstrings now say so and point
  to `env.coordinate_kind`. The `neurospatial.io.nwb` functions, which load
  lazily, now have their declared types instead of `Any`.

- The presets `Environment.open_field`, `Environment.linear_track` and
  `Environment.maze` are easy to find: the `from_samples` docstring, a new
  Presets section in the environments guide, the quickstart and Core Concepts
  point to them. Every Core Concepts example now runs as written (several used
  undefined `data`, `polygon` and `graph` variables), and the quickstart's
  example peak rate (about 12 Hz) matches what the code prints.

- The `Environment` docstring examples run in the documentation tests. The
  skipped ones now execute on fixed example data, so they show real output:
  `from_dict` compares the round trip with `np.array_equal(env.bin_centers,
  reconstructed.bin_centers)` (it asserted `env == reconstructed`, which is
  False now that environments compare by identity), and `to_linear`,
  `subset`, `interpolate`, `smooth`, `transitions` and `to_nwb` show their
  results. `n_dims`, `from_polygon`, `plot_1d` and
  `linearization_properties` no longer document errors or keys that do not
  exist. Only the napari and video examples of `animate_fields` stay
  skipped, because they need a display and ffmpeg.

- `simulate_trajectory_sinusoidal` runs over the whole length of the track,
  in linear position. It used the x range of the track's coordinates, so a
  bent track was shortened to its horizontal extent: an L-shaped 120 + 80 cm
  track oscillated over 120 cm instead of 200 cm.

- `PlaceCellModel(metric="euclidean")` on an egocentric polar environment
  measures the distance between places in the animal's plane,
  `(d cos(angle), d sin(angle))`, so `width` (and its default, three bin
  spacings) is a length. It used to subtract `(distance, angle)` rows as if
  centimeters and radians were one unit. A per-axis `width` on a polar
  environment raises `CartesianOnlyError` with code that passes one width.

- `neurospatial.testing.assert_environment_contract` works on a track built
  without `track_width`: it checks that position lookups refuse with the fix
  that names `track_width`, instead of failing on the refusal. A track with a
  width gets the full lookup checks against `expected_bins`.

- `ops.spatial_basis` sizes its basis by the track length on a linearized
  track. It used the geometric mean of the axis extents, which is zero for a
  straight track (it raised "sigma values must be positive") and unrelated to
  the length of a bent one: an L-shaped 200 cm track was sized as 98 cm.

- On egocentric polar environments, a `(distance, angle)` row whose angle is a
  whole turn outside the angle range finds the bin of that direction. Before,
  `points=` queries, point regions and the polar lookups refused or skipped
  it as outside the environment, and on a full circle the angle `π` itself,
  the same direction as `-π`, was outside. A partial angle range still
  excludes the directions it leaves out.

- `graph_turn_sequence` labelled every turn backwards. In the package's y-up
  frame (x East, y North), heading North and then turning West is a left turn,
  but it was reported as `"right"`; every `"left"` and `"right"` in its output
  is now swapped back to the animal's own side. Recompute saved turn
  sequences. Positions in image pixels, where y points down, must be flipped
  to y-up first, as the docstring now says.

- `env.layout_parameters` reports only options that shaped the bins.
  `Environment.open_field(..., infer_active_bins=False)` recorded
  `fill_holes=True`, its default, although every bin is active and nothing was
  filled; it now records `fill_holes=False`, and `bin_count_threshold`,
  `dilate` and `close_gaps` are recorded as `0`/`False` whenever active bins
  are not inferred. A 1-D grid (from 1-D samples or a 1-D mask) has no
  diagonal neighbors, and neither has any grid `rebin` derives from it, so it
  records `connect_diagonal_neighbors=False` whatever was passed; it used to
  record the default `True`. The bins are unchanged. A 2-D grid one cell wide
  keeps the choice, because refining it creates diagonal neighbors.

- `Environment.from_graph`, `linear_track` and `maze` accept a NumPy scalar
  `edge_spacing` (for example `np.float32(2.0)` read from a config array) as
  they accept a Python number. It used to raise `IndexError` or `TypeError`.

- `spatial_autocorrelation` accepted hexagonal environments and treated the
  hexagon lattice as a rectangular grid. Its rows are offset, so the
  autocorrelogram and the grid score computed from it were wrong. It now
  raises `NotImplementedError` on every environment without a regular 2-D
  grid of rectangular cells (hexagonal, mesh, track, 3-D, or a grid with
  fewer than half of its cells active), and names
  `spatial_autocorrelation_radial`, which works on every layout. Tracks,
  meshes and sparse grids raised `ValueError` before. Rate-map
  `summary_table()` reports a NaN `grid_score` on hexagonal environments.

- `spatial_autocorrelation` reported lags in cell steps. On a grid whose two
  axes have different cell widths (a `from_polygon` grid of a 50 × 37 cm box
  with `bin_size=3` has 2.94 × 2.85 cm cells; small arenas can differ by tens
  of percent), rotating the autocorrelogram for the grid score and measuring
  peak distances for the grid scale mixed the two widths, so both were
  distorted. The autocorrelogram now has square pixels: lags along the wider
  axis are interpolated onto the finer width whenever the two widths differ
  by more than 1/100 of a cell across the grid. Grids fitted with
  `from_samples` differ by up to a few percent between axes, and their grid
  scores move by about 0.004 (at most 0.04 in a sweep of 30 arenas). On a
  120 cm arena with 2 × 3 cm cells, a hexagonal rate map with 30 cm spacing
  now scores 0.99 with scale 29.7 cm, as on 2 × 2 cm cells (1.03, 29.7 cm);
  it scored -0.13 with scale 26.1 cm. A grid with uneven edges along one axis
  (possible through `Environment.from_bins`) has no single lag per step and
  raises the same `NotImplementedError`, naming
  `spatial_autocorrelation_radial`.

- Grid scores change on every grid. `grid_score` rotated the
  autocorrelogram about the array center, `(n - 1) / 2`, while zero lag sits
  at `n // 2`, so on an even-sized map every rotation was half a pixel off
  zero lag; on an odd-sized map the annulus and the peaks used by
  `grid_scale` and `grid_properties` were centered half a pixel off. Scores
  therefore jumped between even and odd arenas (about 0.1 for the same
  hexagonal cell). The metrics now drop an even axis's most negative lag,
  which has no positive partner, and measure everything about the zero-lag
  pixel. Hexagonal maps score higher, mostly on even-sized grids (median
  +0.09 over 30 random `from_samples` arenas; a perfect synthetic
  autocorrelogram scores 1.52 instead of 1.40); noise and shuffled maps
  change by a median of about 0.01-0.02 either way. Grid-cell labels from
  threshold screens can change for borderline cells whose score was near the
  threshold.

- `grid_properties()` averaged the orientation over every detected peak,
  including the spare one it detects beyond `n_peaks`, which can lie on the
  second ring of fields at another angle. Orientation was off by about 5°
  on synthetic hexagonal maps; it now uses the same `n_peaks` nearest peaks
  as the scale and is within about 1-2°.

- `SpatialRateResult.grid_properties()` converted pixels to distance with
  the smallest cell *area*, so its `scale` was in the wrong units (off by a
  factor of the cell width, e.g. 4× for 4 cm cells). It now uses
  `autocorrelation_pixel_size(env)`, the autocorrelogram's pixel side.

- A position sample with a NaN coordinate is a missing sample in every
  per-sample query. `env.nearest_bin` gives it bin
  `-1` and distance NaN, and `env.interpolate` gives it NaN; both used to
  raise, `nearest_bin` with a SciPy error about finite values. A point with an
  infinite coordinate is outside: bin `-1` and distance `inf`, or NaN from
  `interpolate`. `env.interpolate` also accepts a field with NaN bins, such as
  a rate map with unvisited bins, and returns NaN only where a point needs a
  NaN bin's value; it used to refuse the whole field.

- `env.transitions(times=..., positions=...)` counts a pair only when both of
  its samples are inside the environment, so a sample outside or with a NaN
  coordinate adds no transition. It used to raise "Invalid bin indices ...
  -1 (outside) values are not allowed" for any such sample, although the
  docstring said the trajectory input handles outside samples. A precomputed
  `bins=` sequence that holds `-1` is still refused.

- Track environments (`from_graph`, `linear_track`, `maze`) store each
  connectivity edge's `vector` and `angle_2d` from the lower to the higher bin
  ID, as every other layout does. They pointed the other way, so
  `env.get_edge_attributes()` reported track vectors from target to source.
  On a track with gaps between segments (`edge_spacing > 0`), each node's
  `source_grid_flat_index` and `original_grid_nd_index` now name its cell of
  the linear grid described by `env.grid_shape` and `env.active_mask`; they
  used to give the bin's position among the active bins, which pointed at gap
  cells after the first gap.

- The `edge_id` of each connectivity edge is its row in `env.edge_pairs` on
  every layout, so it can index edge values such as gradients and fluxes.
  Edges across the angular seam of a polar environment, and the edges of a
  hexagonal, triangular-mesh or track environment after `subset()`, used to
  keep IDs that did not match their rows.

- `env.boundary_bins` on a hexagonal environment checks all six neighbors of
  each hexagon, so every hexagon next to a hole, a concave notch or a
  `subset()` cut is a boundary bin. It used to check only four of the six
  (east, west and one diagonal pair), so two of the six hexagons around a
  removed interior hexagon were missed. Border scores with
  `boundary_source="analysis"` and `BoundaryCellModel` firing on hexagonal
  environments with holes or irregular outlines change accordingly; full
  rectangular hexagonal arenas are unchanged.

- `env.apply_transform(...)` no longer refuses a valid environment with
  "polygon cell N disagrees with its locator" after a chain of coordinate
  changes, such as a scaled, rotated and reflected map followed by its
  inverse. Clipped environments (`env.clip(...)`) and hexagonal environments
  after a reflection store exact polygon cells, and the check compared them
  with the cells recomputed from the stored transforms at storage precision
  only, although recomputing them adds floating-point rounding. The check now
  allows that rounding, as it already did for bin centers and mesh vertices,
  and still caps it at a quarter of the smallest cell spacing.

- `Environment.open_field(..., fill_holes=False)` works; it used to fail with
  a `TypeError` about multiple values for `fill_holes`.

- NWB reading and writing (`Environment.from_nwb`, `env.to_nwb` and the
  `neurospatial.io.nwb` functions) report "pynwb is required" only when pynwb
  itself is not installed. Any other import failure, such as a missing or
  broken pynwb dependency like `hdmf`, now reaches you unchanged with the name
  of the module that failed, instead of being reported as a missing pynwb.
  Pose and event reading and writing (`read_pose`, `read_events`,
  `write_events`, `write_laps`, `write_region_crossings` and
  `dataframe_to_events_table`) do the same for their extensions:
  "ndx-pose is required" or "ndx-events is required" appears only when that
  extension itself is not installed.

- `Environment.rebin(factor)` on an environment without a stored grid
  specification (for example one built with `Environment.from_bins` without
  `grid_spec`) records the diagonal-neighbor choice it used to connect the
  coarse grid. It previously guessed the choice from the coarse grid's
  connections, which recorded "no diagonals" when the coarse grid was one cell
  wide, so a later `rebin(bin_size=...)` dropped diagonal neighbors.

- Smoothing on an environment transformed by an affine map fitted to noisy
  landmarks (`estimate_transform(..., kind="affine")`) still refuses, because
  the fit absorbs tracking noise as a real stretch or shear. The refusal now
  reports the measured stretch and angle error, explains that the bandwidth
  would not be applied equally in every direction, and names the fix: fit
  `kind="similarity"` (or `"rigid"`) or smooth on the source environment. It
  previously suggested recentering coordinates, which does not help.

- Triangular-mesh environments saved on one operating system now load on
  another. Loading used to rebuild the Delaunay triangulation and refuse the
  file when it differed from the saved triangles, which happens whenever sample
  points are cocircular (any square lattice), because Qhull breaks those ties
  differently across platforms. Point-to-bin lookup now uses the saved
  triangles directly, with the same edge tolerance as before. A point exactly
  on an edge or vertex shared by several triangles goes to the lowest-numbered
  saved triangle. Files whose triangles overlap or reference missing or
  repeated vertices are still refused, now with a message that names the
  triangles.

- `Environment.rebin(factor=...)` keeps a coarse cell whose center lies on the
  arena boundary, such as the cell beside a missing corner of a grid mask or of
  a grid-aligned polygon. Two-dimensional arenas previously required strictly
  interior centers, unlike 1-D and 3-D masks, so a factor of 2 dropped coarse
  cells that held recorded fine bins (3 of 15 bins of a 4-by-4 mask without one
  corner), and positions in them mapped to `-1`. A factor of 2 now merges every
  fine bin of a grid mask inside the (possibly truncated) coarse extent.

- Both `rebin` modes warn when original bins are left without a new bin,
  naming their count, their share of the area and how to cover them. Positions
  in those bins map to `-1`, so their time and spikes silently dropped out of
  totals: about 2.5% of a circular arena's fine bins at `factor=2`, and 56% of
  a 6-by-6 arena at `bin_size=4`.

- `rebin` warns when new bins on both sides of a barrier in the arena outline
  are connected because they meet only outside the arena. In a U-maze with a
  2-unit barrier, 5-unit bins shortened the path between the arm ends from 37
  to 5. The connections are kept, as in `from_polygon` at the same bin size;
  the warning names them and how to fix it: smaller bins, or the barrier's
  edges declared as internal walls.

- Posterior means on polar environments are the probability-weighted centroid
  in the animal's egocentric plane, reported as `(distance, angle)` with the
  angle in the environment's range. A posterior split across the angle seam
  behind the animal now has a `mean_position` behind it, not straight ahead.
  This covers `mean_position`, `DecodingResult.mean_position` and
  `to_dataframe()`, and `decode_position_summary`; a centroid at the animal has
  a NaN angle.

- Domain functions that read bin centers as Cartesian `(x, y)` coordinates now
  raise `CartesianOnlyError` on polar environments instead of returning wrong
  numbers. The refusals cover `field_shape_metrics`,
  `field_shift_distance`, `spatial_autocorrelation`,
  `resample_field`, `map_probabilities`, `spatial_basis`,
  `graph_turn_sequence` and `detect_laps`. Type checkers reject polar
  environments for these functions. Some functions refuse only their Cartesian
  mode and keep a graph-based mode for polar environments:
  `rate_map_centroid(method="euclidean")`,
  `compute_field_emd(metric="euclidean")`,
  `select_basis_centers(method="kmeans")` (also the default `center_method` of
  the basis functions), `path_progress` and `distance_to_region` with
  `metric="euclidean"`, `trajectory_similarity` with `"hausdorff"` or `"dtw"`,
  and `detect_runs_between_regions` with `min_speed`.

- `Environment.from_bins` and environment loading now check supplied
  finite-volume faces for hexagonal and triangular-mesh bins against the cells:
  each face length must equal the shared cell edge (zero when cells share no
  edge) and each face distance the center-to-center distance. Inconsistent
  faces previously passed silently and changed smoothing and diffusion results.

- Ray-cast view rates no longer refuse when cells clipped to a curved or
  slanted wall reach it: a hit is owned by the cell within that cell's
  coordinate rounding of the hit, instead of an overlay of nearly coincident
  edges. Cells that stop short of the wall still refuse, and the message names
  the route that builds wall-reaching cells:
  `Environment.from_bins(grid.clip(polygon).binning, arena=OpenArena(polygon))`.

- Observers tracked exactly on a wall no longer view their own position. A ray
  or cue sight line that leaves the wall into the room reaches the next wall;
  only facing out through the wall or exactly along it gives a zero-distance
  contact. Standing on an internal wall views the side faced. Gaze hits,
  viewsheds, visible cues, view-rate bins and simulated view cells agree.

- Preserve static attribute and protocol checking for the common Environment;
  runtime Cartesian/polar capability filtering no longer admits missing methods.

- Check physical view-map ownership only for samples with selected exposure.
  Missing wall bins raise actionable guidance; excluded/final samples and raw
  geometric hits retain their meanings. Slanted wall cues tolerate local cue
  rounding while earlier crossings and wall overlaps remain blocked.
- Physical border scores require complete represented and observed wall
  coverage, distinguishing missing measurements from measured zero firing.
  Unsupported scores and batch summaries provide setup guidance rather than
  treating an unobserved wall as zero activity.
  When the border score is unavailable, `summary_table()` labels a spatially
  tuned non-grid cell `"undetermined"` instead of `"place"`, since it may be a
  border cell; single-unit and population tables agree.
  Physical border scores refuse combined environments in which a component has
  no measured footprint, naming the component, instead of scoring its fields
  against walls in other components (euclidean returned -1 for a wall-hugging
  field). Declare each component's outline or use `boundary_source="analysis"`.
  A single arena that declares internal walls but not its footprint
  (`OpenArena(..., physical_footprint_known=False)`) is refused the same way:
  a field along its outer edge was scored against the internal walls alone
  (-1, labeled `"place"`; now NaN, labeled `"undetermined"`). Declare the
  measured outline, for example
  `OpenArena(outline, walls=..., physical_footprint_known=True)`.
- The border-score examples in the neuroscience metrics guide raised on their
  position-derived environments, which have no measured walls. They now
  explain the physical-versus-analysis boundary choice, pass
  `boundary_source="analysis"` and run in the documentation tests.
- Prevent automatic combined-environment bridges from crossing a third
  component, including tiny and nearly collinear footprints. Long affine
  histories retain their arena through storage without using the composite
  nesting budget. Clipped diffusion keeps its own certification boundary.

- Omit transient analysis caches before copy/pickle; retain JSON-compatible
  protected graph metadata. Explicit bins now reject wall-crossing supplied
  pairs, keep drawing outlines separate from physical evidence, and validate
  finite-volume faces against their grid geometry across coordinate scales.
  Clipping preserves declared navigation paths and stores equal parent maps once.

- Reject circular region metadata before exhausting the interpreter stack,
  avoiding a test-worker crash on Windows/Python 3.10.

- Compare point regions by coordinate values, avoiding ambiguous NumPy truth
  values on Python 3.10–3.12 and when comparing independent region copies.
- Label place fields from actual graph connections, preserving diagonal links
  and physical walls after derivation and storage. Polygon clipping now retains
  per-parent ownership at floating-point precision and exposes analysis boundary
  bins. Wall-crossing checks also handle exact on-wall trajectory samples.
- Require declared track width for containment. Track and maze presets accept
  `track_width`.
- Reject cyclic environment metadata before recursive decoding, avoiding a
  Windows/Python 3.10 stack crash while retaining shared acyclic records.

- Preserve coincident track centers, native mesh indexing and reserved user
  graph attributes through environment storage. Malformed JSON/NWB/NPZ records
  now report contextual recovery guidance.

- Preserve intended radial/angular bin counts when a polar step divides its
  range but floating-point division rounds above an integer. Polar circularity
  now defaults to automatic: full circles wrap, partial angle ranges remain
  open, and explicitly wrapping a partial range is rejected.

- Preserve dotted recording names in environment file prefixes, so saving
  `rat1.day1` and `rat1.day2` produces independent file pairs. Explicit `.json`
  and `.npz` paths remain accepted.

- Preserve physical diffusion faces on grids with small coordinate units;
  exactly nonzero axis displacements no longer disappear below an absolute
  `1e-9` cutoff. Diagonal navigation edges still carry no shared face area.

- Keep valid gaze hits on slanted physical walls during view encoding, using
  retained wall contact and parent-bin ownership to avoid crossing missing bins.

- Preserve each component’s declared physical walls when environments are combined,
  including shared room boundaries and measured walls alongside unknown domains.

- Subsets retain exact bin measures, selected finite-volume faces, cell shapes
  and absolute track coordinates. Derived environments can be saved and loaded.
- Translations and similarities retain smoothing, differential operators and
  drawing components; positive diagonal grid scaling also keeps rectilinear
  interpolation and updates the grid edges.
- Triangle plots retain exact mapped vertices under every invertible affine
  transform, including shear and unequal scaling. Smoothing still refuses maps
  that lose the required finite-volume face geometry.

- NWB environment read-back retains geometric lookup, smoothing, boundaries,
  plotting and track linearization, with the original graph attributes.

- Plot graph fields against physical linearized coordinates, preserving
  unsupported bins and inserted display gaps in both lines and shaded areas.
- **Behavior change:** `DecodingResult.error_against` now returns NaN for truth
  outside observed runs instead of clamping to endpoints or interpolating
  across dropouts. Plot overlays use the same `max_gap` rule; decoder scoring
  forwards the fitted decoder's gap configuration.

- Match tutorial raster spikes to the PSTH's half-open window, register video
  preview pixel centers with the full affine transform, and label maze endpoint
  proxies used by transition summaries. Check exact spike edges and affine
  registration across notebook sources and documentation mirrors in CI.
- Omit threshold-based cell labels from the NWB tutorial's persistence summary.

- Resolve heading colormaps such as the default `hsv` to angle-dependent RGBA
  colors in Matplotlib exports and persistent artists. Preserve literal colors,
  update colors with heading, and wrap full turns consistently.

- Apply the full pixel-to-world affine transform to rendered video images,
  preserving Y-axis flips, rotation, shear and translation. Export and widget
  paths share the same mapping and keep axes/data limits in physical units.
  Pixel-position regressions check both initial and updated frames.

- Render supported overlays consistently in embedded and external-frame HTML,
  including regions without other overlays. Preserve global frame clocks and
  overlay histories across parallel workers, and redraw video overlays when
  persistent artists cannot represent them. Warn and skip unsupported HTML
  overlays without mutating or reading them.

- Simulation recovery validation requires finite center-error and correlation
  metrics for every selected unit before passing. Return aligned unit labels,
  spike counts, per-unit outcomes and skip reasons; report evaluable counts
  accurately and plot only jointly finite metric pairs.

- Explain the generalized border score's untuned baseline: a uniform map can
  have a strongly positive score. Add analytic flat, boundary-only and central
  controls, and require geometry-specific interpretation of score thresholds.

- Use a normalized center-surround contrast in the graph-convolution example,
  with a positive center and no response to a constant input.

- Render transformed-layout field markers in display units, honor scatter size
  overrides and retain the requested missing-support color without changing
  shared colormaps.

- Apply inclusive minimum sample-count thresholds consistently across regular
  and hexagonal grids; threshold zero still leaves unvisited bins inactive.

- Clarify that `detect_goal_directed_runs` evaluates one candidate per observed
  sample run, and that success means sufficient directed progress rather than
  arrival. Point completed-traversal analyses to `detect_runs_between_regions`.

- Landmark video calibration now includes the raw-pixel Y-axis flip in the
  returned transform, so landmarks and unseen video points map to the intended
  physical coordinates for rigid, similarity and affine fits.
- Windows CI installs FFmpeg with a pinned setup action instead of the
  unreliable Chocolatey package feed. Every test platform verifies that FFmpeg
  and FFprobe are available before running the tests.
- The linear-track validation example records enough repeated traversals and
  spikes for stable field recovery with continuous Poisson event times.
  Documentation CI uses each job's declared Python version instead of running
  both version labels with the repository's default interpreter.
- Partial missing-rate masks retain the position-dependent Poisson
  factorial terms, preventing incorrect posterior probabilities and MAP bins.
- Decoder xarray exports contain independent writable arrays, coordinates and
  metadata, so editing an export cannot corrupt cached position estimates.
- Fitting a decoder requires observed training support without requiring its
  prediction bin width to fit inside the training recordings.
- Phase-precession shuffle statistics use a data-adaptive slope grid that
  resolves narrow peaks on physical tracks, with the same procedure for the
  observed and shuffled pairings. Physically equivalent position-unit and
  slope-bound conversions preserve significance. Searches requiring more than
  20,000 candidates explain how to narrow slope bounds or normalize positions
  instead of silently losing resolution.
- Affine transforms preserve physical cell measures in 1D, 2D and 3D, including
  anisotropic cells and segment-direction length scaling for embedded tracks.
  Transformed grids reload from NWB through the documented non-grid fallback
  instead of attempting reconstruction with stale, untransformed grid edges.
- Circular polar grids with two angular sectors include both shared radial
  interfaces in their finite-volume diffusion conductance.
- **Behavior change:** Poisson spike simulation samples counts and continuous
  event times inside each observed interval, retaining multiple events under
  coarse or nonuniform sampling. The final sample no longer invents a following
  recording interval. Seeded spike trains change; a given seed remains
  reproducible, and refractory filtering still enforces the requested ISI.
- MSD matches the nearest recorded time, excludes outside/missing/disconnected
  geodesic pairs, and returns NaN for lags with no observed displacement pairs.
  Curvature retains the original sample timestamps through stationary periods
  before smoothing within each observed run.
- Position decoding rejects rate results built on different bin coordinates,
  measures, coordinate systems, units, frames or weighted connectivity, even
  when bin counts match or `validate=False`. Equivalent copied environments
  remain supported.
- Rate results own their analysis settings and return independent copies,
  including nested recording windows. Fitted decoders also return independent
  settings. Editing settings can no longer change training support or reported
  provenance. Previously saved rate-result pickles migrate their stored
  settings on restoration; results predating provenance keep settings as `None`.
- Published connectivity graphs reject node, edge and graph attribute edits,
  in addition to topology changes, so edge-distance edits cannot bypass cache
  invalidation. `env.connectivity.copy()` provides an editable graph for
  separate analyses; protection survives pickling and deep copies. Coordinates,
  grid indices and edge vectors are owned immutable tuples, including after
  JSON/file restoration, so nested edits and graph copies cannot corrupt
  boundary caches or transformed edge geometry.
- The introductory decoding examples execute on partially observed space and
  pass rate results directly, preserving support even after zero filling.

- Editing arrays after handing them to neurospatial no longer changes stored
  state. `SpikeTrains` keeps read-only copies of its trains and labels, so
  editing the caller's labels can no longer create duplicate unit IDs. Every
  result's `unit_ids` is a read-only copy. A `BayesianDecoder` keeps read-only
  copies of its rate maps, labels and training support, so editing
  `rates.firing_rates` after `BayesianDecoder.from_rates(rates)` no longer
  changes its predictions. Arrays stay read-only after pickling and deep copies.
- Static type checkers such as mypy accept every documented spike input: a
  list of per-unit arrays, a 1-D or NaN-padded 2-D array, `SpikeTrains` or a
  pynapple `TsGroup`. This covers every population encoder and significance
  test, `population_peri_event_histogram`, `decode_session`,
  `decode_session_summary` and the `BayesianDecoder` methods. Previously,
  passing a plain list or `SpikeTrains` to `BayesianDecoder.fit` failed type
  checking.
- Predicting with an unfitted `BayesianDecoder` now explains that `fit()`
  returns a new decoder and shows the fix,
  `decoder = decoder.fit(spike_times, times, positions)`. The old message told
  users who had already called `fit()` to call it again.
- Rebinned environments load from files and dictionaries again, and rebinning
  preserves diagonal connectivity on small and sparse grids.
- Linear interpolation retains known values at active-bin centers and grid
  lines next to holes; missing corners with positive weight still produce NaN.
- Circular polar grids with one or two angular bins correctly exclude interior
  radial rings from their boundary. The circular flag also survives NWB saves;
  older NWB records without the flag retain their previous open-axis behavior.
- Type checkers report wrong arguments to environment methods, such as
  `env.bin_at("not an array")` or an unknown keyword; the fitted-state check
  previously hid every decorated method's signature. `plot_field` accepts a
  `matplotlib` `Colormap` as well as a colormap name.
- `subset()` of a grid keeps the parent's edges among the selected bins instead
  of reconnecting them by cell adjacency. A 4-connected grid read from NWB, or a
  saved graph with removed edges, previously came back with edges it never had
  (a 4-connected grid became 8-connected), changing geodesic distances and
  smoothing.
- `to_linear()` and `linear_to_nd()` on tracks read from NWB explain that the
  track graph is not stored and how to rebuild it, instead of requesting a bug
  report.
- The Euclidean `distance_to` refusal on polar environments no longer prints
  `()()`.
- Type checkers accept calls such as `env.smooth(field, bandwidth)` and
  `polar_env.to_dict()` on `Environment` and `EgocentricPolarEnvironment`;
  mypy previously reported "Invalid self argument".
- Polar `boundary_bins` treats a circular angular axis as wrapping: bins beside
  the ±π seam are interior, so a full circular grid's boundary is its innermost
  and outermost rings.
- `grid_cell_session` no longer permutes rows of its environment's
  `bin_centers` while shuffling grid-cell phases; the returned environment's
  centers matched neither its graph nor `bin_at`.
- Refresh active-bin lookup indices when rebuilding a masked polar layout, so
  angular seam edges and point lookup use the new mask.
- Keep the saved connectivity of environments read with `from_file`,
  `from_dict` or `read_environment` on their layout. Gradients and divergence
  work on 4-connected grids read from NWB, and `copy()` of a read-back
  environment keeps 4-connectivity and polar seam edges and arc lengths
  instead of reverting to the rebuilt layout's graph.
- Point dense-kernel and mesh-skew warnings at the scientist's call rather
  than library code.
- Correct spatial information and sparsity for missing data in NumPy and JAX,
  including per-neuron support normalization in population metrics. Observed
  zero activity remains distinct from missing rates.
- Correct missing-bin centering in spatial autocorrelation so zero lag is one
  and constant firing-rate offsets cannot change the correlogram.
- Whiten PCA scores before ICA assembly detection, handle null components, and
  compute explained variance for the recovered ICA components.
- Correct reverse explained variance to swap pre- and post-behavior periods
  while retaining the behavioral template.
- **Breaking:** `mutual_information` now computes discrete mutual information
  from a joint position-response histogram in bits per observation. Old rate-map
  calls raise migration guidance; use `information_per_second` for the previous
  Skaggs information-rate calculation in bits/second.

- Exclude unsupported training locations from fitted/session decoding and
  functional decoding of rate results, including silent bins, explicit priors
  and streamed summaries. Explicit raw-array decoding must supply the support
  prior separately; computational zero filling no longer supplies evidence for
  unobserved space in the fitted workflows.
- Preserve missing geodesic steps as NaN instead of indexing the last spatial
  bin with the outside-position sentinel. Report missing pre-decision/heading
  observations as NaN and expose `PreDecisionMetrics.is_valid`; missing evidence
  cannot produce a positive deliberation label.
- Warn when slicing aligned arrays loses boundaries between observation windows;
  document using original arrays with `epochs=` and half-open event cohorts.
  Correct epoch input forms and the home-range area recipe, and execute those
  recipes in documentation tests.

- Correct workflow guidance on raw-second occupancy thresholds and smoothing missing support, with executable occupancy examples.

- Clarify cell-screen thresholds and the scope of synthetic, biological and cross-package validation.

- Label decoding errors as in-sample and interpret sequence shuffle tests relative to their null. Refresh notebook outputs and synchronize the source and documentation copies.

- Preserve containment and inactive bins after affine environment transforms, including chained transforms, and retain polygon-region holes. Reject singular transforms that collapse bin geometry.
- Preserve generated unit-label provenance and spike-recording windows in NWB spatial-rate round trips (schema `2.3`), so reloaded maps retain their decoder pairing and coverage. Older `2.x` files remain readable with their existing defaults.
- Return `NaN` path progress when the current position is disconnected from the start, even if the goal is reachable.
- Compute video calibration scale from transformed axis lengths so rotations and reflections do not change the reported `cm_per_px`.

- Unit labels of mixed types keep their types: `unit_ids=[1, "u"]` no
  longer turns `1` into the string `"1"`. Result keys, indexed results,
  labelled spike groups and shuffle random streams all use the label as
  given, so a unit's seeded null is the same in single and population calls.
- **Behavior change:** spatial rates use only intervals whose two tracking
  samples both lie inside the environment. An interval ending in a dropout
  (NaN) or outside was previously kept in the occupancy while its spikes
  were dropped, biasing rates low (20% isolated NaN frames read a 5 Hz unit
  as 3.97 Hz); it is now excluded from spikes and occupancy alike, which
  recovers 5 Hz without imputing positions.
- The all-intervals-excluded, spikes-outside-tracking, out-of-bin and
  `min_occupancy`-masks-all warnings point at the user's calling line instead
  of a line inside neurospatial, for every rate family.
- `object_vector_cell_significance` and
  `egocentric_object_vector_cell_significance` build the polar grid and each
  frame's polar bin once per call instead of once per shuffle: about 3x
  faster (0.24 s to 0.08 s for 10 units, 20 shuffles, 60,000 samples) with
  identical results.
- Kinematic speeds treat a zero-length interval (duplicate timestamps) as
  undefined (NaN) instead of infinite, so `pre_decision_speed_stats`, goal
  approach rates and heading labels no longer absorb an `inf` or a
  divide-by-zero warning.
- Kinematic speeds from 1-D (linear-track) positions are the track speed.
  `pre_decision_speed_stats` previously broadcast a 1-D difference against
  the time steps and returned an unrelated number.
- **Breaking:** `Environment.bin_center_of` raises `BinIndexOutOfRangeError`
  for indices outside `[0, n_bins)`, like the graph queries. Negative indices
  previously wrapped silently to the last bins, and indices past the end
  raised NumPy's `IndexError`.
- `DecodingSummary.plot` breaks its entropy and MAP lines at recording gaps
  instead of drawing a straight segment across the pause.
- neurospatial exceptions survive pickling, so errors raised in worker
  processes reach the caller with their original message and attributes;
  previously several failed to unpickle or repeated their message.
- `SimulationSession` checks that `times` and `positions` have one row per
  sample and that `ground_truth` is keyed by exactly the `unit_ids`, so
  relabeling units with `dataclasses.replace` must re-key `ground_truth` too.
  Duplicate `unit_ids` are rejected, since two cells sharing a label would
  both be validated against one ground truth.
  `NWBUnits` checks that `unit_ids` and `obs_intervals` align with
  `spike_times`.
- `ObjectVectorRateResult` and `ObjectVectorRatesResult` reject a
  `direction_frame` other than `"allocentric"` or `"egocentric"`; plots
  previously drew any other value as egocentric.
- `spatial_view_cell_significance` computes the gaze geometry once per
  call instead of once per shuffle. With `gaze_model="ray_cast"` on 3,000
  samples, 10 shuffles drop from 6.95 s to 0.64 s with identical observed
  scores, null scores and p-values.
- Object-vector and egocentric `method="binned"` rates keep bins whose
  occupancy equals `min_occupancy`, as every other rate path does, and warn
  when `min_occupancy` masks every occupied bin.
- `ShuffleTestResult` stores `null_scores` as a read-only copy, and its
  p-value documentation names the finite-null denominator the significance
  functions use.
- `DecodingResult(...)` checks that the posterior is 2-D with one column
  per `env` bin and that `times` has one entry per posterior row, instead of
  silently mapping columns to the wrong bin centers. The `times` checks that
  `to_xarray()` used to apply now fire at construction.
- Smoothing, `gradient`, `divergence` and `heat_kernel_wavelet_basis` on a
  non-grid environment read from NWB explain that the round trip does not
  restore cell geometry and name the original layout, instead of advising a
  factory method the user never skipped.
- `BayesianDecoder.fit` (and the session decoders that build an encoding
  model) raise `ValueError` when the speed, gap, window and `min_occupancy`
  gates leave no occupied bin, instead of returning a model that decodes
  every time bin as a uniform posterior.
- `read_units` reads files in which a unit has no `obs_intervals` rows (a
  unit that was never observed); that unit's coverage is an empty `(0, 2)`
  array, so the shared `spike_window` is empty unless it is excluded with
  `unit_ids=`.

- Center continuous posterior image columns on decoder timestamps, preserving MAP/actual overlays, gapped index clocks and explicit extent overrides. Two-column plots use their timestamp spacing; a single column has a documented 1-second display width, with no changes to posterior arrays or clocks.

- Plot native 1D rate maps as labeled lines over physical bin coordinates, including singular/population results and NaN breaks. Existing graph-track and 2D plotting paths are retained; rate arrays are unchanged.

- Name the current `spike_times` keyword in simulation validation's missing-input diagnostic.

- Name nonnumeric times/positions together in shared conversion errors, retaining shape/order problems from the convertible argument and providing Why/Fix guidance without changing valid arrays.

- Share one swap-aware times/positions validator across encoding and environment trajectory APIs. Timestamp shape, finiteness, ordering and sample-count problems are reported together with a corrected call; missing position samples remain supported.

- Explain invalid circular-shift windows with a valid `windows=` call, preserving all row/shape problems from shared interval normalization.

- Resolve heading interpretation thresholds through `HEAD_DIRECTION_THRESHOLDS`, matching the predicate and batch screen while preserving the default text and positional option.

- Align place shuffle windows with inferred speed gates and 1-D trajectory normalization, preserving retained null spike counts. Validate object-vector metric choices before coordinate construction in both frames, with shared public diagnostics.

- Validate required environments before constructing place/view shuffle windows, so missing environments receive the encoder diagnostic and a corrected call. Object-vector Euclidean significance continues to accept `env=None`.

- Normalize allocentric object bearings exactly like zero-heading egocentric bearings at direction-bin boundaries, and make invalid-environment guidance omit headings for allocentric calls. Result help consistently describes the recorded frame.

- Require the documented spatial-information threshold for every place, grid and border label. In the seeded 10-minute noise example, erroneous border labels fall from 19/20 to 0/20; label precedence and threshold values are unchanged.

- Honor requested duration in lap-based sessions, including linear-track and T-maze conveniences. Keep every one-way traversal and fixed pause on a shared half-open recording clock, derive traversal speeds from available time, and reject infeasible durations with guidance. Direct speed-driven lap trajectories and other simulation methods are unchanged.

### Fixed — documentation

- Document singular/population table parity, readable column order, units/threshold metadata and singular xarray identity. The quickstart covers native 1D rate plots; animation guidance explains explicit overwrite opt-in and the current HTML numeric-colorbar limitation with an executable static-scale/timestamp-label recovery. Posterior plotting help records the centered continuous and short-clock edge conventions.

- Include required unit labels in the manual `SimulationSession` constructor example.

- Teach explicit simulator/NWB holder attributes and a complete file → selected epochs → population fields → decoder → summary/overlay workflow. A synthetic HDF5 example executes the published NWB recipe after file close with physical units, nondefault unit IDs and two analysis epochs; synchronized tutorials preserve the simulation-duration contract.

- Preserve the existing shared-validator NumPy examples and parameter sections while adding timestamp/position diagnostics.

- Teach explicit tracking arrays, typed precomputed-rate and count-array decoding handoffs, and one retained event cohort across gap-aware PSTHs, rasters, regressors and positioned tables. Keep event identifiers and selection masks visible, explain each helper's existing edge closure, and synchronize the updated tutorials.

- Correct both decoder tutorial posterior overlays to use spatial bins and the MAP line's plotted time coordinates. Synchronized examples and an executable continuous/gapped-clock recipe preserve physical coordinates for accuracy metrics and separate position plots.

- Explain graph linearization as geometric coordinates rather than direction separation. The synchronized track tutorial and an executable graph/trial recipe use explicit direction labels and recover separate planted field centers within one bin.

- Distinguish standardized assembly activity and EV/REV effect sizes from calibrated significance, document controlled REV correctly, and explain that selected dimensions can have no thresholded core members. Statistical calculations are unchanged.

- Fix the README's simulated-field peak using a complete 5 cm sampling grid and the result's NaN-aware peak lookup. Reference examples use canonical graph, visibility, region, and immutable-data APIs; expected gotcha errors are checked with markers.

### Documentation

- Fix renamed phase-precession imports in executable examples and move the spatial adapter test normalizer import to its owning private module and distinguish independently standardized assembly activation from reactivation strength on a shared template-normalized projection scale.

- Update root/domain imports and hard-break migration guidance, and make population covariance APIs discoverable from the API index. Add an executable simulation → encoding recipe with separate decoding and population-statistics branches, one shared unit selection and explicit baseline-controlled EV/REV interpretation.

- Document recording-window metadata on the spatial result classes now exposed at the root; their existing GLM diagnostics and constructor fields are covered by the public docstring checks.

- Clarify that the head-direction MVL screen weights firing rates and its Rayleigh test weights spike counts, with a rate-times-occupancy fallback. The numerical Rayleigh calculation is unchanged.

- Update cell-criterion migration, README/quickstart, API references, glossary and neuroscience guidance. Execute six new flagship docstrings and synchronize the view, heading and object-vector tutorials; the object-vector tutorial reports its estimator/criterion and adds seeded shuffle results alongside an explicit place-cell control.

- Document finite-count information bias and the measured noise-screen results in every cell predicate, field detector and batch classifier. Explain occupancy bias in Rayleigh tests weighted by spike counts, circular-shift assumptions and the need for calibrated publication verdicts; retain the uniform-spike example and show its shuffle rejection.

- Teach allocentric and egocentric object-vector analysis separately in the guides, migration notes, glossary and synchronized tutorial. Public examples record the frame and use the appropriate heading-free or heading-required call; executable docstrings cover the new APIs.

- Document marker-based pytest checks and current first-run calls. Publish the changelog from one canonical source, preserving copy-only notes and consolidating duplicate release headings.

### Removed

- **Breaking:** remove `Session`, `load_session` and `neurospatial.recording`, plus the unused `PositionLike` trajectory adapter and `validate_simulation`'s raw keyword form. NWB component readers and simulator holders expose arrays for explicit analysis calls; pynapple conversion still returns `(times, positions)`.

- **Breaking:** remove the image-mask `bin_size` keyword shim and both view-result `peak_view_location` methods. Use `pixel_size` and the shared `peak_location()`/`peak_locations()` accessors.

- **Breaking:** remove root decoder/binner, spike-container and epoch-selection re-exports; import them from their domains. Drop public normalizers, event validators, animation internals/protocols, duplicate calibration and renderer helpers, and cache plumbing. Delete the probability-space Poisson wrapper, `integrated_absolute_rotation`, `Affine3D`, `SpikeOverlay` and surrogate compatibility imports.

- Remove the superseded public API curation plans and the test pinning the old root export list; the public API snapshot replaces that export contract.

- Remove the deprecated `detect_cell_types`, `detect_hd_cells`, `detect_view_cells` and `detect_ovcs` methods; use `label_cell_types` or `classify`.

- Remove the deprecated region-crossing positional order. Use `detect_region_crossings(position_bins, times, env, region_name="home")`; `region_name` is required and keyword-only.

### Added — executable documentation

- Require function inputs to be documented under Parameters or Other Parameters; only class constructors may also use Attributes.

- Install widget support in the headless module-doctest CI job so the newly executable widget example runs there too.

- Execute Markdown examples using adjacent `docs-test` markers, cumulative reader state for README/quickstart, fresh seeded blocks for reference fragments, and restored animation setup patches. Replace the indexed snippet manifest and subprocess helper with default pytest coverage.

- Execute flagship docstrings, including NWB geometry and position examples, with only display/video calls skipped. Module doctests write outputs in temporary directories; the NWB workflow enforces its examples separately.

- Check public docstring sections and constructor inputs, and complete the missing examples and parameter documentation.

### Changed — errors that teach

- Corrected-call guidance includes each encoder's and decoder's required arguments and all animation backends. Population egocentric validation reports coordinate and length problems together, and malformed peri-event window bounds fail with guidance before alignment.

- Region fixes show list-valued `end_regions` and handle missing names without a conversion error. Zero-rate posterior errors distinguish all-negative-infinity likelihoods from corrupt NaN inputs.

- Warn visibly when `bin_size` covers every data axis or a peri-event window exceeds 60 seconds, with units and corrected values. Large-grid warnings are now `UserWarning` instead of hidden `ResourceWarning`; the existing hard memory ceiling is unchanged. Length-1 bin-index arrays explain how to unwrap `bin_at` output before a graph query.

- First-run factory, graph-query, encoding, decoding, peri-event, animation and file errors give concrete corrected calls. Segmentation and boundary regressors raise `RegionNotFoundError` with the caller's region argument and available names; NWB readers and direction-label lookups name their corrected arguments.

- Replace `HOW:` labels with `Fix:` in diagnostic messages. Shared timestamp, spike and trajectory validators name their caller, collect all input problems, detect swapped arrays, and check coordinate dimensions before binning. A non-Environment first argument now raises `TypeError` with a corrected call.

- Add `NeurospatialError` as the common base for library exceptions while retaining their standard Python bases. `RegionNotFoundError` also supports `except ValueError` and prints an unquoted `Fix:` line with a suggested region name.
- **Breaking:** raise `BinIndexOutOfRangeError` for graph-query bin indices (formerly `IndexError`; it subclasses `ValueError`, not `IndexError`, so `except IndexError` no longer catches it), `LayoutNotBuiltError` for unbuilt layouts, and `IncompatibleEnvironmentError` for mismatched environment dimensions or decoding bins.

### Changed — behavior analyses respect recording gaps

- Timestamp-based heading rejects malformed position shapes before velocity
  division, preventing a 1-D trajectory from broadcasting into a square matrix.
  The error names the shape and supplies an aligned x/y-coordinate fix.
- Timed home ranges omit bins with zero observed dwell, including at
  `percentile=100`, so rounding in cumulative percentages cannot include an
  isolated sample's bin. Untimed visit counts retain their previous meaning.
- The 13 kinematic/event entry points expose `max_gap=0.5` and `epochs=None`:
  `heading_from_velocity`, `pre_decision_heading_stats`,
  `pre_decision_speed_stats`, `head_sweep_from_positions`,
  `heading_direction_labels`, `compute_path_efficiency`,
  `instantaneous_goal_alignment`, `goal_bias`, `approach_rate`,
  `compute_goal_directed_metrics`, `compute_trajectory_curvature`,
  `compute_home_range`, and `events.add_positions`. Pre-decision and VTE
  composites use their existing window keywords for kinematics too. Examples
  and notebook pairs now pass timestamps to `heading_from_velocity`.
- Speed statistics, approach rates, direction labels, head sweeps, trajectory
  curvature and home-range dwell respect `max_gap` and `epochs`; goal and VTE
  composites forward the same windows. IdPhi sums runs without joining their
  headings, and the two-epoch dwell totals 200 seconds instead of charging the
  pause to a bin. Entirely excluded speed windows return NaN without reduction
  warnings. Samples outside runs have NaN curvature and stationary labels.
  `compute_path_efficiency` reports NaN traveled length, efficiency and angular
  efficiency when any interval is unobserved. Shortest length and wall-clock
  time efficiency/time-to-goal retain their existing meanings.
- **Breaking:** `heading_from_velocity(times, positions)` replaces scalar `dt`, validates
  finite, strictly increasing aligned timestamps, and computes velocity,
  smoothing and circular interpolation separately per observed run. The
  measured 1,911.44 cm/s teleport no longer produces a heading across the pause.
  Isolated samples and runs without moving anchors have NaN headings.
  Shared heading statistics, goal alignment and simulation callers use actual
  timestamps. Smoothing bandwidth retains its existing units of samples.
- `events.add_positions` accepts `max_gap` and `epochs`, interpolating only
  inside closed observed-run spans. Events in pauses, on isolated samples or
  outside the tracked span now receive NaN coordinates; extrapolation ends.
  Events exactly at either observed run endpoint retain that sample's position.
- Kinematic interval velocity masks unobserved intervals with NaN, preserving
  interval alignment and position-units-per-second measurements.
- All 16 segmentation and sequence entry points now expose keyword-only
  `max_gap=0.5` and `epochs=None`: `detect_region_crossings`,
  `detect_runs_between_regions`, `segment_by_velocity`, `detect_laps`,
  `segment_trials`, `detect_goal_directed_runs`, `running_direction_labels`,
  `detect_boundary_crossings`, `extract_pre_decision_window`,
  `compute_pre_decision_metrics`, `compute_decision_analysis`,
  `compute_vte_trial`, `compute_vte_session`, and `Environment.bin_sequence`,
  `bin_sequence_with_runs`, `transitions`. Use `max_gap=None` to disable the
  gap gate for intentionally coarse sampling; `epochs` still applies.
- Position-only analyses warn when `max_gap` or `epochs` exclude every
  interval, for example tracking sampled more slowly than 2 Hz under the
  default `max_gap=0.5`. The warning names the active gate, reports the median
  sampling interval and points at the calling line, so an empty or all-NaN
  result is no longer indistinguishable from a measurement.
- Pre-decision extraction, metrics, composite decision analysis and trial/session
  VTE accept `max_gap` and `epochs`. Pre-decision samples come only from the
  observed run containing the entry time; entries outside every run return an
  empty window. The pause example retains five post-resumption samples instead
  of ten spanning 1000.9 seconds. Session VTE still respects the trial start.
- Runs, trials, velocity epochs, laps, goal-directed candidates and decision
  boundary crossings accept `max_gap` and `epochs` and analyze each observed
  run separately. The measured 98.5→1100 s run and 0→1100 s trial are no longer
  marked successful across a pause, the 98.1→1100 s movement epoch is clipped,
  and the 88.6→1108.6 s region lap and 599.95 s boundary crossing disappear.
  Auto laps keep the first 10% of the whole input as their template. Running
  direction labels forward the same windows and retain one label per sample.
- `detect_region_crossings` accepts `max_gap` and `epochs`, and detects entries
  and exits separately within observed runs. A first post-pause target sample
  no longer invents an entry at 1100 s. The contiguous detector keeps its
  existing crossing timestamps and region-boundary handling.
- `Environment.bin_sequence` and `bin_sequence_with_runs` accept `max_gap`
  and `epochs`, drop samples that touch no observed interval (including
  singleton inputs), and split same-bin runs and deduplication at recording
  breaks. Original run indices and lengths stay aligned with the input.
  Empirical `transitions` gates every intervening interval in a lagged pair;
  the two-epoch fixture gives 9,998 one-step and 9,994 lag-three pairs. Raw
  bin-only transitions retain their previous behavior without a time gate.
- Empirical transitions return a zero CSR matrix when every pair is excluded
  by gaps or epochs, including lagged pairs under the default adjacency filter.

### Changed — decoding and peri-event histograms respect recording gaps

- Peri-event counts use original spike times against absolute event-shifted
  edges, preserving inclusive starts and exclusive stops at nonzero and Unix
  timestamps. The shared precision guard rejects widths that the absolute
  clock cannot represent, with an origin-shift fix for spike/event times.
- Peri-event histograms share the decoding edge and count helpers: every bin
  is half-open and stays inside its requested window. Spikes at the last
  whole-bin edge are excluded, and large timestamp offsets use relative
  rounding instead of an absolute `1e-9` epsilon.
- Conflicting `epochs` and explicit bounds in `bin_spikes_in_time` report
  the argument conflict with separate `Why:` and `Fix:` lines.
- Single-unit and population peri-event histograms accept `epochs` and
  `spike_window`, keeping only events whose entire analysis window is observed.
  Means and SEM use retained events, and results, child results, plots and
  summaries carry `n_events_dropped`. With neither window supplied, all events
  remain eligible; partial drops add no warning.
- Full and streamed decoding results record `spike_window` and
  `spike_window_assumed`, including summaries and NetCDF-compatible xarray
  attributes. Attaching session metadata preserves the single posterior
  allocation; public construction and replacement still copy input arrays.
- Posterior heatmaps with detected recording breaks use time-bin indices and
  dashed gap markers, preserving visible separation instead of stretching
  bins across a pause. Contiguous and single-bin results keep time axes;
  two timestamps alone cannot distinguish a gap from a larger bin width.
- **Breaking:** `BayesianDecoder.fit(epoch=...)` is replaced by keyword-only `epochs` and
  `spike_window`. Training masks the original arrays instead of concatenating
  epoch slices, preserving aligned speed samples and unit identities.
  `predict`, `predict_summary`, and `score` forward the same window keywords
  and the decoder's configured `max_gap` to observed-run decoding.
- `decode_session` and its streamed summary accept `epochs` and
  `spike_window`, and form bins only within observed sample runs. No posterior
  is invented inside a tracking pause. Decode bins retain immobility and
  out-of-bounds periods; speed/bounds restrict encoding only. Full and streamed
  paths use the same half-open edges and counters, including run breaks.
- `bin_spikes_in_time` accepts `epochs=` and tiles each normalized window
  independently. Shared decoding helpers count only half-open bins and clamp
  their edges to window stops. Final-edge spikes are no longer counted, and
  floating-point rounding uses timestamp-relative slack instead of an absolute
  `1e-9` epsilon (which lost a bin near `1e9` seconds). Partial trailing bins
  are dropped, and unrepresentable bin widths raise with an origin-shift fix.

### Changed — recording gaps and time windows in rate maps

- The three frame-family cell predicates propagate invalid `epochs` and
  `spike_window` diagnostics, including combined argument problems and
  `Why:`/`Fix:` guidance, instead of silently classifying malformed inputs
  as `False`. Each analysis still normalizes the windows once.
- Directional, view and egocentric encoders record the normalized
  `spike_window` on singular, population and zero-unit results. Population
  indexing/iteration preserves it, and xarray exports include the assumption
  as an integer and explicit windows as a flat array for NetCDF round-trips.
- `compute_directional_rates`, `compute_view_rates`, and
  `compute_egocentric_rates` share the population-silence heuristic: with no
  `spike_window`, at least five units silent for at least 60 seconds within
  a tracked run produce one warning. The scan respects gaps and epochs,
  includes leading and trailing silence, and ignores invalid directional,
  view or polar bins; silence is not proof of recording coverage.
- `compute_directional_rate(s)`, `compute_view_rate(s)`,
  `compute_egocentric_rate(s)`, `is_head_direction_cell`, `is_spatial_view_cell`,
  and `is_object_vector_cell` accept keyword-only `max_gap`, `epochs`, and
  `spike_window`. They parse windows once and apply one shared analysis mask
  across units, including empty populations. Directional, view and egocentric
  rates now recover a 5 Hz synthetic rate across a 1000-second recording pause
  instead of charging the pause to a single bin. In an unsmoothed example,
  pooled directional/egocentric rates rise from 0.833 Hz to 5.001 Hz and view
  rates from 0.717 Hz to 4.916 Hz; no bin absorbs the 1000-second pause.
- Directional, view and object-vector rates warn, like spatial rates, when
  the gates exclude every interval (for example `epochs` in milliseconds) or
  when more than half of the spikes fall outside the tracked time (for
  example `spike_times` in milliseconds), instead of silently returning an
  empty or truncated map.
- Directional, view and egocentric binning helpers use one interval mask for
  both spike counts and occupancy. Their default `max_gap=0.5` excludes
  recording pauses instead of charging them to one bin, and their normalized
  `epochs` and `spike_window` keywords restrict the same intervals on both
  sides. **Behavior change:** a spike at `times[-1]` is no longer counted;
  the final sample starts no half-open interval.
- Malformed numeric attributes on IntervalSet-like windows now report the
  window argument, conversion problem, `Why:` and `Fix:` together, without
  hiding invalid values in the other time-window argument.
- `compute_spatial_rates` warns once when at least five units are all silent
  for at least 60 seconds of continuously tracked time and no `spike_window`
  was supplied. This recording-outage heuristic includes leading and trailing
  silence and rest, but never combines silence across untracked pauses;
  an explicit window suppresses it. Its absence does not prove coverage.
- Every rate result exposes `spike_window` and the read-only
  `spike_window_assumed` flag in `summary()`. Spatial encoders record the
  normalized acquisition windows, preserving them when indexing a population
  and in direction-conditioned place fields. Spatial population xarray exports
  store the assumption as an integer and supplied windows as a flat array,
  so both round-trip through NetCDF. Frame-family encoders now record their
  acquisition windows as well.
- **Breaking:** `behavior.in_epochs`, `restrict`, and `restrict_spike_trains` share the
  same window parser as spatial analyses. Nested sequences always describe
  `(n, 2)` rows; parallel `(starts, ends)` arrays are no longer an input form,
  and a tuple or list of two 1-D NumPy arrays raises (naming
  `np.column_stack([starts, stops])`) instead of being read as two rows.
  Overlapping or touching rows merge, zero-width and empty window sets raise
  with a fix, and `epochs=None` leaves data unrestricted. Point membership
  still uses the requested `closed=` setting on the normalized windows.
- Spatial rate functions, `is_place_cell`, and `compute_directional_place_fields`
  accept `epochs=` and `spike_window=`. One shared analysis mask excludes the
  same intervals from spike counts and occupancy, including GLM and empty
  populations. Direction-labelled runs use epochs instead of concatenated
  samples, removing artificial occupancy across breaks (1.6 s instead of
  1.9 s in a repeated-label example).
- **Behavior change:** with `max_gap=None`, intervals starting outside the
  environment now exclude their spikes as well as their occupancy. A spike
  exactly at the last position timestamp, `times[-1]`, lies outside all
  half-open sampling intervals and is no longer counted; it is reported as
  time-dropped.
- `Environment.occupancy` accepts `epochs=` as half-open windows in seconds.
  It excludes intervals that cross window boundaries for both start and
  linear time allocation, interval counts, and smoothed occupancy.

### Fixed — numerical and I/O corrections

- **Behavior change:** `pre_decision_heading_stats` now excludes samples below
  `min_speed` using the same uninterpolated velocity and speed as goal
  alignment. On an east/stop/north path, circular variance and resultant length
  are 0.2929 and 0.7071 (formerly 0.1849 and 0.8151 from stationary headings).
  The all-stationary result remains `(0.0, 1.0, 0.0)`.

- **Behavior change:** `instantaneous_goal_alignment` now returns NaN for
  samples below `min_speed`, and `goal_bias` averages moving samples only.
  On an east/stop/north path, all 30 stopped samples are excluded and
  `goal_bias` is 0.5055 (formerly 0.5862 from interpolated stationary headings).
  Fully stationary trajectories still return NaN without raising.

- **Behavior change:** `heading_from_velocity` and `heading_from_body_orientation`
  interpolate missing or low-speed headings along the shorter arc, uniformly
  in angle. Near a 180° turn, filled headings formerly snapped toward the
  endpoints (errors up to 0.785 rad) because interpolation followed the chord.
  Antipodal turns follow the sign of the stored angle difference; unmasked
  NaN headings no longer contaminate neighboring filled samples.

- **Behavior change:** `heat_kernel_wavelet_basis` now uses the finite-volume
  diffusion generator, scaled by squared bin spacing. On uniform Cartesian
  grids its standard deviation is `sqrt(2 * scale)` bins at every bin size
  (scale 1 formerly spread 2.77, 3.91, and 6.17 bins at sizes 1, 2, and 5).
  Layouts without finite-volume geometry now raise `NotImplementedError`
  with a fix. `chebyshev_filter_basis` keeps its graph-hop locality and
  distance-weighted, spectrally rescaled operator.

- **Behavior change:** `gradient`, `divergence`, and
  `compute_differential_operator` now use the finite-volume cell geometry of
  `env.smooth`: gradient divides by edge length, and divergence weights face
  flux by cell volume with positive values at sources. For `f=x`, gradients
  are 1 at bin spacings 1, 2, and 4 (formerly 1, 2.83, and 8);
  `div(grad(x²+y²))` is 4 in grid interiors (formerly -15.31, -122.51,
  and -980.08). The Laplacian has the continuum sign and units Hz/cm² for
  rates in Hz and positions in cm. Layouts without finite-volume geometry,
  including NWB-reconstructed environments, now raise `NotImplementedError`
  with a fix instead of returning physically inconsistent derivatives.
- `decode_position` (and every decoder built on `normalize_to_posterior`)
  floored the prior at `1e-10`, so a bin with zero prior could still win the
  posterior given a large enough likelihood (for example a MAP in an excluded
  bin with posterior `0.999998`). Zero-prior bins now get exactly zero
  posterior, small positive priors are no longer floored, and the
  `handle_degenerate="uniform"` fallback spreads mass only over bins with
  positive prior (NaN when the prior has none).
- `theta_phase` used a transfer-function Butterworth band-pass that is
  numerically unstable for a narrow theta band at typical LFP sampling rates:
  on a clean 8 Hz sine the phase was off by ~1.56 rad at 2-5 kHz and all-NaN
  at 10 kHz and above. It now filters with second-order sections
  (`sosfiltfilt`); the phase error is below 0.01 rad at 2-30 kHz.
- `circular_linear_correlation`, `circular_circular_correlation` and
  `circular_basis_metrics` computed p-values as `1 - cdf`, which cancels to
  exactly `0.0` for strong effects (below about `1e-16`). They now use survival
  functions, so a strong effect reports its true tiny p-value (for example
  `exp(-500)` for a perfect circular-linear correlation over 1000 samples).
- `Environment.from_grid_mask` (and every `MaskedGridLayout`, including
  `env.subset`) accepted nonuniform, decreasing, duplicate or non-finite
  `grid_edges`, then reported wrong bin sizes (for example `[-1, -1]` or
  `[0, 0]`) and used one width per axis for cell volumes and diffusion. It now
  raises `ValueError` naming the axis unless the edges are finite, strictly
  increasing and uniformly spaced (to within float rounding), and raises when
  coordinates are so large that float64 cannot resolve the bin width, with a fix
  to subtract an origin offset.
- **Behavior change:** `BayesianDecoder.predict`, `predict_summary` and `score`
  passed spike trains to the encoding models by position even when both the fit
  and the predict input were labelled, so a reordered or different pynapple
  `TsGroup` silently paired each unit's spikes with another unit's model. When
  both the fit (a labelled group, or `unit_ids=` at construction) and the
  predict input carry labels, trains are now matched by label in any order, and
  a label mismatch raises `ValueError` listing the missing and unexpected
  labels. Otherwise trains are still paired by position, and a unit-count
  mismatch now raises a `ValueError` that says so (previously a Poisson
  likelihood shape error). A predict input that repeats a label raises.
- `compute_directional_rates`, `compute_view_rates` and
  `compute_egocentric_rates` mis-read a pynapple `TsGroup` (it was coerced as
  one train, giving one unit labelled `0`). They now accept it like
  `compute_spatial_rates`: one rate map per unit, with the group's index as
  `unit_ids`.
- **Behavior change:** passing `unit_ids=` together with a labelled `TsGroup`
  silently replaced the group's labels, so in `compute_spatial_rates` a group
  keyed `[10, 20]` with `unit_ids=[20, 10]` attached unit 10's spikes to label
  20. All four population encoders now raise `ValueError` listing both label
  sets unless `unit_ids` equals the group's index.
- **Behavior change:** repeated `unit_ids` were accepted (for example
  `compute_spatial_rates(..., unit_ids=[5, 5])`). Every place that takes
  `unit_ids` now raises `ValueError` naming the repeated labels: the population
  encoders, `population_peri_event_histogram`, the `BayesianDecoder`
  constructor and `fit` (fitted labels) and `predict` input labels, and
  population result constructors (including `summary_table(unit_ids=)`
  relabelling). `SpikeTrains` already rejected them.
- **Behavior change:** `to_pynapple` passed unsorted timestamps to pynapple,
  which sorts the times without reordering the values, so
  `to_pynapple([0, 2, 1], [10, 20, 30])` returned `t=[0, 1, 2]`,
  `d=[10, 20, 30]` (each value paired with another sample's time). It now
  raises `ValueError` for unsorted, repeated or non-finite timestamps, naming
  the first offending indices, and for a `columns` list whose length does not
  match the value columns (pynapple silently renamed them `0, 1, ...`). These
  checks run before pynapple is imported.
- `DirectionalRatesResult.to_xarray()` wrote `attrs["bandwidth"] = None` for
  unsmoothed results (`bandwidth=None`), so `Dataset.to_netcdf()` raised
  `TypeError`. The attribute is now omitted when no smoothing was applied, as
  for spatial results.
- `read_environment` estimated the bin sizes of reloaded non-grid environments
  (graph tracks, hexagonal, triangular mesh) from the nearest neighbour's
  *index* instead of its distance: a Y-track with 2.94 cm² bins came back with
  1640.25. The estimate now uses the median nearest-neighbour spacing.
- Writing a non-grid environment (graph track, hexagonal, triangular mesh) to
  NWB and reading it back lost its geometry: bin sizes were re-estimated
  (2.94 became 8.68 on a Y-track), `grid_edges` came back `None`, and every
  edge's direction `vector` was reversed (81 of 81 on the Y-track). The NWB
  environment schema is now 1.1: it stores the exact bin sizes and graph grid
  edges, and writes each edge so its vector round-trips. Schema 1.0 files
  still read, with estimated bin sizes.
- **Behavior change:** `DecodingResult` kept a reference to the caller's
  posterior and cached `map_estimate` and the other derived properties, so
  editing that array in place (or reassigning `result.posterior`) left the
  cached values stale (for example a MAP of bin 0 while `posterior.argmax`
  was bin 1). `DecodingResult` is now frozen, and its constructor stores
  read-only copies of `posterior` and `times`. Use
  `dataclasses.replace(result, posterior=new)` for a modified result.
  `decode_position` hands over its freshly computed posterior without a copy.
- **Behavior change:** `to_linear` returned wrong positions on graph
  environments whose `edge_order` differs from `graph.edges()` order, such as
  `Environment.maze("w", ...)`: on a W maze with 50 cm segments, `(75, 0)` on
  the base mapped to 175 instead of 75 and `(0, 40)` on the left arm to 114.03
  instead of 140, and about 38% of on-track points disagreed with `bin_at`.
  `track_linearization` looks a point's nearest segment up by its `edge_id`
  attribute but finds it by its position in `graph.edges()`. Graph layouts now
  linearize a copy numbered in `graph.edges()` order, so `to_linear` returns
  along-track distance for `from_graph`, `maze`, `linear_track` and reloaded
  environments. Any `edge_id` on an input graph is ignored, and the caller's
  graph is not modified.
- **Behavior change:** polar tuning plots were drawn mirrored relative to the
  library's angle conventions. `plot_head_direction_tuning` and
  `plot_circular_basis_tuning` drew 0 at the top and clockwise, so a
  North-preferring (π/2) cell pointed East; they now draw angles as in the
  arena, 0 = East (right), π/2 = North (up), counter-clockwise, and reset a
  caller-supplied polar axis to that orientation. `plot_object_vector_tuning`
  drew +π/2 (left of the animal) on the right; it keeps "ahead" at the top
  and now draws left on the left.
- **Behavior change:** `read_position`, `read_head_direction` and `read_pose`
  returned an NWB series' *stored* values and ignored its `conversion` and
  `offset`, which NWB defines as the map to `unit` (`data * conversion +
  offset`). A pixel series stored 0-500 with `unit="meters"`,
  `conversion=0.002`, `offset=0.1` came back as 0-500 instead of 0.1-1.1 m, and
  `environment_from_position` built a 500 × 500 environment. The readers now
  return values in the series' `unit` (head direction converts before the
  degree-to-radian step), and `lazy=True` raises `ValueError` for a series
  whose scaling is not the identity rather than returning unconverted handles.
  The NWB overlays inherit the fix. `environment_from_position` maps NWB unit
  names to the `Environment.units` registry, so pynwb's default `"meters"`
  becomes `"m"` (no unrecognized-units warning), and likewise `cm`, `mm` and
  `px`.
- **Behavior change:** `environment_from_position` silently set
  `env.units = "cm"` when the position series declared no unit (`unit=""`).
  It now emits a `UserWarning` naming the series and saying to pass `units=`,
  then still assumes `"cm"`. Passing `units=` skips the lookup and the
  warning.
- **Behavior change:** `population_peri_event_histogram` raised
  `AxisError: axis -1 is out of bounds` for a pynapple `TsGroup`, because it
  iterated the group (which yields unit keys, not trains). It now accepts a
  labelled group like the population encoders: one PSTH per unit, with the
  group's index as `unit_ids`. A `unit_ids=` passed with a labelled group must
  equal its index, or the call raises `ValueError` listing both.
- **Behavior change:** `compute_vte_session` cut each pre-decision window from
  the whole session, so a window reached back past the trial start into the
  inter-trial interval or the previous trial. A trial starting at 5.0 s that
  entered the decision region at 5.267 s got the window `[4.267, 5.267]` and a
  head sweep of 12.57 rad, all from movement before the trial; the trial's own
  straight run has 0. Windows are now clipped at the trial start, and
  `window_start` reports the clipped start. A trial left with fewer than 3
  samples before its entry is skipped, now with a `UserWarning` (it was
  skipped silently). `compute_vte_trial` is unchanged.
- **Behavior change:** `method="diffusion_kde"` (the default for
  `compute_spatial_rate(s)` and `smooth_rate_map(s)`) reported huge rates in
  bins far from all occupancy: a simulated 50 Hz place cell in a 40 s open-field
  session got 20,554 Hz in a corner the animal never approached, so the map's
  peak (and anything using it) landed there. Far from occupancy both smoothed
  densities fall to the diffusion operator's numerical accuracy (1e-6 of the
  largest value when its eigenbasis is truncated, float64 roundoff otherwise),
  and the ratio was noise over noise. Bins whose smoothed occupancy is within
  1000 times that accuracy of zero, relative to the largest in their connected
  component (typically 4-7 bandwidths from any occupancy), are now NaN, like
  other unsupported bins (`fill_value=` replaces them). The remaining rates
  match the exact dense kernel to within 1e-4 of the peak rate at bandwidths
  5-20.
- **Behavior change:** the simulators' default place-field width was
  `3 * mean(env.bin_sizes)`, but `bin_sizes` holds per-bin volumes (areas in
  2-D), so a 2-D grid with 2 cm bins got 12 cm fields and one with 5 cm bins
  got 75 cm fields. `PlaceCellModel(width=None)`, and through it
  `simulate_session`, `open_field_session`, `linear_track_session` and
  `tmaze_alternation_session`, now default to 3 × the bin spacing (the median
  bin length along a track, otherwise the median distance between neighbouring
  bin centres, which is `bin_size` on a regular grid): 6 cm at 2 cm bins, 15 cm
  at 5 cm bins. A one-bin
  environment now raises `ValueError` asking for `width=`. Likewise
  `validate_simulation`'s default `max_center_error` is 2 × the bin spacing
  (4 cm at 2 cm bins, previously 8 cm).
- **Behavior change:** `method="binned"` with `bandwidth > 0` had the same
  far-field problem: on a corridor visited only at x < 40 cm, with raw rates
  of at most 50 Hz, it reported 295,658 Hz at x = 104 cm (bandwidth 10) and
  237,467 Hz at x = 190 cm (bandwidth 20). Its smoothed rate is a ratio of
  two diffused sums, and far from every visited bin both are noise. Those
  bins are now NaN under the same rule as `diffusion_kde`; the remaining
  rates match the exact dense average to within 1e-4 of the peak rate.

### Added — `method="glm"`: penalized-Poisson GAM estimator (spatial only)

`compute_spatial_rate` and `compute_spatial_rates` gain a fourth estimator,
`method="glm"`: a batched penalized-Poisson generalized additive model that uses
the cached finite-volume eigenbasis as its smoothness-penalty basis. Occupancy
enters as a **log-offset** (never a denominator) and the smoothness penalty `λ`
is chosen by REML, so the fit returns **finite firing rates everywhere** —
including the low-occupancy and unvisited bins where the ratio estimators
(`diffusion_kde` / `gaussian_kde` / `binned`) produce `NaN`.

- **New keyword-only params** on `compute_spatial_rate(s)`: `penalty`
  (fixed `λ ≥ 0`; `None` → choose by REML) and `rank` (requested basis rank;
  out-of-range values are clamped, not rejected, to the effective rank reported
  via `result.rank`). These are **mutually exclusive** with the ratio-method
  params `bandwidth` / `min_occupancy` / `fill_value`, which now default to
  `None` (resolving to their historical `5.0` / `0.0` / preserve-`NaN`
  behavior). Combining a param with the wrong `method` raises `ValueError`.
- **New result fields** on `SpatialRateResult` / `SpatialRatesResult`:
  `coefficients`, `penalty`, `penalty_weights`, `rank`, `deviance`, `converged`,
  `n_iter`, `reml_objective` (all `None` for the ratio methods). `bandwidth` is
  `None` for glm. `summary_table()` gains the GAM scalar columns for glm results.
- **Spatial only.** `compute_view_rate(s)`, `compute_egocentric_rate(s)`, and the
  directional encoders keep their ratio-only `method` set (no `"glm"`).
- The default `method="diffusion_kde"` path is unchanged (byte-for-byte).

### Added — `method="glm"` through NWB persistence and the decoder

`method="glm"` is now usable end-to-end (encode → persist → decode), not just as
a standalone estimator.

- **NWB round-trip.** `write_spatial_rates` / `read_place_field` now persist a
  glm `SpatialRatesResult` **field-for-field**: the scalars `penalty` / `rank` /
  `n_iter` / `converged` / `reml_objective` and the `(rank,)` `penalty_weights`
  vector in the table's JSON metadata blob, and `deviance` `(n_units,)` plus
  `coefficients` (stored `(n_units, rank)`, transposed back to `(rank, n_units)`
  on read) as per-unit table columns. `bandwidth` is stored nullable (`None` for
  glm). Ratio-method results round-trip unchanged, with the GAM fields `None`.
  The spatial-rates schema version bumps `2.0 → 2.1` (glm-diagnostics addition);
  ratio tables are byte-identical to `2.0` apart from the version string. This
  **removes** the earlier `NotImplementedError` `write_spatial_rates` raised on a
  glm result. The clean-break contract still holds: pre-`0.9`
  `"smoothing_method"`-keyed tables do not read.
- **Decoder.** `decode_session`, `decode_session_summary`, and `BayesianDecoder`
  accept `method="glm"` and forward the glm knobs `penalty` / `rank` into the
  encoding step. `bandwidth` and `min_occupancy` are now **nullable** on all
  three (default `None`, resolving to the encoder defaults `5.0` / `0.0` for the
  ratio methods) so they can stay unset for glm. All three validate the
  method-specific params with the **same** validator the encoder uses, so a
  ratio param combined with `method="glm"` (or a glm param with a ratio method,
  or an out-of-domain `penalty` / `rank`) raises the identical `ValueError` at
  the decoder boundary. The ratio decode path is unchanged.

### Added — `method="glm"` optional float32 JAX acceleration

`compute_spatial_rate(s)(method="glm", backend="jax")` now routes the
penalized-Poisson fit + REML `λ` selection through an optional **float32 JAX
mirror** of the NumPy/SciPy core (installed via the `jax` extra; the base install
and `backend="numpy"` continue to run the float64 core). The float64 core remains
the correctness reference: the JAX path is a faithful mirror — same box
constraint `|B·γ| ≤ η_clip`, out-of-domain guard, and REML objective — cast back
to NumPy float64 at the boundary, so `SpatialRatesResult` diagnostics stay
float64 and the public return contract is unchanged.

- **Parity.** At a fixed `penalty` the float32 fit matches the float64 core to
  ~`1e-6` on the firing-rate map (relative L2 and relative error above a rate
  threshold). Under automatic REML it is a touch looser (float32 selects a
  slightly different `λ` near the broad objective minimum), approximately
  `~1e-3` in measured rate/λ comparisons and still scientifically identical.
- **Speed.** Markedly faster on populations — on a 400-bin, 30-unit, rank-60
  problem the **warm** JAX REML fit is ~40–60× faster than the NumPy path on the
  CPU backend (fixed reference workload, fixed seed). The **first** call on an
  unseen shape pays JIT compilation, so a one-off fixed-penalty fit can be slower
  than NumPy; the win is on warm/repeated fits and grows on GPU.
- **Kernel cleanup.** The constant-field warm start is constructed exactly from
  the MRF basis's leading component intercepts and shared across both backends
  and every REML evaluation, eliminating the basis-wide least-squares SVD
  (including its repeated NumPy REML cost). The JAX Newton step uses the SPD
  Hessian's Cholesky factorization and triangular solves rather than generic LU.
- **No new API.** Dispatch is driven entirely by the existing `backend=` keyword
  (`"numpy"` / `"jax"` / `"auto"`); there is no new user flag. JAX stays an
  optional extra (Linux/macOS).

### Added — `method="glm"`: per-neuron smoothing (`pooled=False`)

`compute_spatial_rate(s)(method="glm")` gain a keyword-only `pooled` flag
(default `True`, unchanged behavior). With `pooled=False` REML selects an
**independent smoothing penalty `λ_k` per neuron** instead of one shared `λ`, so
a sharp field is smoothed less and a broad one more. This is purely additive: the
`pooled=True` default is byte-for-byte identical to before apart from one new
scalar diagnostic (`reml_at_boundary`).

- **Cost.** `pooled=False` runs the REML search **once per informative unit**
  (~one REML per neuron), reusing the existing batched Newton fit per distinct
  `λ`; there is no separate per-neuron fast path.
- **Result fields become per-unit vectors.** On the automatic-REML path
  (`penalty=None`, a penalized basis, ≥1 spiking unit) `penalty` /
  `reml_objective` / `reml_at_boundary` widen to `(n_units,)` and a new
  `penalty_selected_by_reml` `(n_units,)` bool mask records provenance. A
  single-unit `compute_spatial_rate` call unwraps these back to scalars.
  `summary_table()` surfaces them; indexing a population result slices them.
- **Fixed penalty beats `pooled`.** Supplying `penalty=<float>` skips REML and
  records one scalar `λ` regardless of `pooled`. `pooled` is likewise a no-op at
  `penalty_rank == 0` (a shared-basis property, so `λ` stays a scalar `None`) and
  when no unit spikes (the all-zero-spike degenerate path, with **no** pooled REML
  run).
- **Zero-spike units** (whose per-unit `λ` is statistically unidentified) take
  the **pooled `λ` over the informative units** as a fallback, flagged
  `penalty_selected_by_reml=False` with `reml_objective=nan` (a documented
  sentinel); their field floors to the rate floor.
- **Boundary diagnostic.** The new `reml_at_boundary` field (scalar for
  `pooled=True`, `(n_units,)` for `pooled=False`, `None` when REML did not run)
  flags a `λ` selected **near** a REML search bound (within `5·xatol`): `λ` itself
  is weakly identified — its optimum may lie at or beyond the bound. The fitted
  field remains finite, but its sensitivity to `λ` should be checked. A single
  warning names the boundary side and the affected units (by their `unit_ids`);
  the applied `λ` is the finite value returned near the bound (the interval is
  never expanded).
- **Validation.** `pooled` is a strict `bool` (rejects `"true"` / `1` / `0` /
  arrays / `None`) and **glm-only** (`pooled=False` with a ratio method raises;
  `pooled=True` with one is the harmless default).
- **NWB.** The spatial-rates schema bumps `2.1 → 2.2`: `pooled` and a scalar
  `reml_at_boundary` join the GAM metadata blob, and the per-unit
  `penalty` / `reml_objective` / `reml_at_boundary` / `penalty_selected_by_reml`
  vectors are persisted as table columns when present. A schema-`2.1` file with
  no `pooled` key reads back **method-conditionally** — `pooled=True` for a glm
  table, `pooled=None` for a ratio table.

### Changed — BREAKING: `smoothing_method` renamed to `method` (estimator axis)

The smoothing/estimator keyword is now uniformly named `method` across every
smoothing encoder, result class, the decoder, and NWB metadata — one name, one
meaning. This is a **hard rename with no alias** (per project policy); callers
that named `smoothing_method=` or read `.smoothing_method` must update. Callers
that never touched it see identical behavior — the default stays
`"diffusion_kde"` (and `"binned"` for the egocentric encoders), and no numerics
change.

- **Renamed keyword `smoothing_method` → `method`** on `compute_spatial_rate`,
  `compute_spatial_rates`, `compute_view_rate`, `compute_view_rates`,
  `compute_egocentric_rate`, `compute_egocentric_rates`, `is_place_cell`,
  `compute_directional_place_fields`, `decode_session`,
  `decode_session_summary`, `BayesianDecoder`, and
  `simulation.validate_simulation`. Each encoder keeps its existing method value
  set and default.
- **Renamed field `smoothing_method` → `method`** on `SpatialRateResult`,
  `SpatialRatesResult`, `ViewRateResult`, and `ViewRatesResult`.
- **NWB:** `write_spatial_rates` now stores the estimator under the metadata key
  `"method"` (was `"smoothing_method"`); the spatial-rates schema version bumps
  to `2.0`. This is a clean break with no back-compatibility shim — spatial-rates
  tables written by earlier versions (schema `1.x`) no longer read.
- **`decode_session_summary`:** because `method` now names the smoothing
  estimator, it is no longer accepted as a per-block decode kwarg. The Poisson
  observation model (the only supported likelihood) is applied unconditionally.
- `method="glm"` (a penalized-Poisson GAM estimator, `compute_spatial_rate(s)`
  only) lands on this renamed axis — see the "Added — `method="glm"`" entry
  above.

### Added — `env.diffuse`: matrix-free diffusion smoothing (0.8.0, performance)

Smoothing no longer materializes the dense `(n_bins, n_bins)` diffusion kernel on
its hot paths. A new method

```python
env.diffuse(fields, bandwidth, *, mode="density", backend="numpy")
```

applies the finite-volume heat operator `H = exp(-t L)` **without building an
`(n, n)` matrix**, using a cached, per-component, bandwidth-aware **truncated
symmetric eigenbasis** of `S = M^{-1/2}(D − W) M^{-1/2}`. Time and memory scale
with `n_bins × rank` (`rank ~ measure(domain)/σ^d`), not `n_bins²`, so smoothing
now scales to large/fine grids. The eigenbasis depends only on geometry, so it is
built once and reused across every bandwidth, mode, and neuron (auto-invalidated
when the environment is mutated). `fields` may be 1-D `(n_bins,)` or a 2-D batch
`(n_bins, n_fields)`.

- **`env.diffuse` is a pure linear operator** — no output clip, no
  renormalization. The always-retained per-component null mode makes mass
  conservation exact under truncation (`mode="transition"` preserves `sum`;
  `mode="average"` preserves a constant field), so smoothing a **signed** field
  is preserved. Positivity, where required, is the consumer's job.
- **`env.smooth` now routes through `env.diffuse`** and gains a documented
  **approximation contract**: it reproduces the dense kernel to within a
  near-lossless truncation tolerance (dropped modes contribute ≤ `tol` (default
  `1e-6`) in the **M-weighted norm**; the raw per-bin error carries a
  volume-conditioning factor `κ(M) = sqrt(max vol / min vol)`, worst on polar
  `r→0` / skewed mesh). On a non-negative input the linear apply may leave
  tolerance-level negatives (bounded relative to the dense output) instead of the
  dense kernel's exact 0-floor — clip the result yourself if you need a strict
  floor.
- **`backend="jax"`** runs the apply **in JAX** (the cached NumPy eigenbasis is
  cast to `jnp`), so `jit` / `grad` / GPU work through the smoothing. The JAX
  `diffusion_kde` encoders no longer round-trip through NumPy.
- The `diffusion_kde`, `binned`, and diffuse-`resample_field` consumers route
  through `env.diffuse`; strict `> 0` support gates (`binned`, resample) derive
  their support from the **W-component structure** (exact, truncation-proof), not
  the smoothed denominator's sign, so truncation cannot emit a spurious `NaN`.
  `diffusion_kde` clips its own output `≥ 0` (decode nonnegativity).

**Breaking:** the `kernel=` parameter is **removed** from
`neurospatial.encoding._smoothing.smooth_rate_map` and `smooth_rate_maps_batch`.
Passing a precomputed dense kernel is obsolete — the cached eigenbasis now
provides the cross-neuron reuse the parameter existed for. There is no
backward-compat shim (per the project default); callers passing `kernel=` should
simply drop it. `env.compute_kernel` is **unchanged** (it still returns the dense
`(n, n)` matrix via the `expm` path, byte-identical, for callers that genuinely
need the matrix), as is `transitions(method="diffusion")`.

### Changed — diffusion smoothing bandwidth is now the true physical σ (breaking, correctness)

The `diffusion_kde` / diffusion-kernel `bandwidth` is now the **true physical
standard deviation (σ)** on every environment layout, independent of bin size or
resolution. Previously the effective smoothing width scaled with bin size (2D
place fields at `bin_size=1` oversmoothed by ~70% for the same `bandwidth`), so
place-field widths and spatial information were not comparable across
resolutions or with other tools. This is a **behavior-changing correctness fix**:
smoothed values change even on uniform grids — that difference *is* the
correction.

- **New finite-volume operator.** The kernel is now the finite-volume heat
  operator `H = exp(-t L)`, `t = σ²/2`, `L = M⁻¹(D − W)` with
  `W[i, j] = A[i, j] / d[i, j]` (`A` = shared-face measure, `d` = center
  distance) and `M` = per-bin cell volumes. On any K-orthogonal layout this has
  the continuum limit `−∇²`, so a point source smooths to physical σ exactly.
- **Impact on existing results.** For the same `bandwidth`, **2D place fields
  now smooth ~1.5–1.7× less**. To approximate the *old* amount of smoothing,
  scale `bandwidth` by roughly `√(bin_size)` (rough, mode-dependent — prefer
  re-choosing `bandwidth` as a physical σ in your units).
- **Correct orientation per consumer.** `compute_kernel(mode="transition")`
  returns `Hᵀ` (column-stochastic, mass-conserving smoothing of extensive
  quantities); `mode="density"` returns `H·M⁻¹` (count → density, integrates
  to 1 under bin volumes); `transitions(method="diffusion")` now returns the
  **row-stochastic** `H` via transpose (correct on non-uniform bin volumes —
  polar, mesh — where it no longer assumes symmetry).
- **Breaking: public `neurospatial.ops.compute_diffusion_kernels` signature.**
  The old Gaussian-weight form
  `compute_diffusion_kernels(graph, bandwidth_sigma, bin_sizes=..., mode=...)`
  is **replaced** by
  `compute_diffusion_kernels(graph, *, volumes, sigma, mode)`. The face measure
  is read from a per-edge `"A"` attribute (single source of truth, like
  `"distance"`); a missing `"A"` on an edge raises, and `A == 0` means no
  diffusion across that edge. `mode` now also accepts `"average"` (row-stochastic
  `H`) at the low level.
- **`min_occupancy` caveat.** `min_occupancy` is documented in seconds, but the
  KDE paths compare it to `occupancy_density = K @ occupancy`, which under the
  new density kernel is seconds-per-cell-volume on non-uniform grids — a
  pre-existing unit mismatch that this change neither fixes nor worsens for
  uniform grids, and that lies **outside** the grid-invariance guarantee.
  Tracked as a follow-up.
- **Out of scope (unchanged):** nonuniform-Cartesian custom `grid_edges` inherit
  the uniform-cell approximation and are excluded from the physical-σ guarantee;
  performance (dense `expm`) is unchanged.

### Added — `mode="average"` intensive-field smoother

- **New public `mode="average"`** on `Environment.compute_kernel` and
  `Environment.smooth` (and the low-level `compute_diffusion_kernels`): the
  row-stochastic heat operator `H` that **averages an intensive field** (a rate
  map or probability *density*). Unlike `density` (`H·M⁻¹`), it carries no
  cell-volume bias on non-uniform bin volumes (polar, mesh), and unlike
  `transition` it is the correct choice for a rate map rather than a total.
  Discrete probability *mass* (a posterior summing to 1) still uses `transition`.
  `env.smooth`'s **default stays `density`** (unchanged behavior for existing
  mode-less calls); `average` is recommended for intensive rate maps.
- **`smooth_rate_map(method="binned")`** and **`resample_field(method="diffuse")`**
  now smooth their intensive fields through the row-stochastic average
  (masked / valid-bin-normalized), removing the volume bias on non-uniform `M`.
  For `resample_field`, this also fixes a down-bias where covered bins adjacent
  to an uncovered region were pulled toward zero, and makes a source `NaN`
  interpolate from valid neighbours instead of propagating. Uniform-grid results
  are unchanged (the per-cell volume factor cancels in the rate/weight ratio).

### Corrected guidance

- **`method="binned"` is not a dense-kernel memory mitigation.**
  Earlier notes (through v0.6.0) recommended `binned` to avoid the dense
  `(n_bins, n_bins)` kernel, but `binned` smooths its rate map through the same
  diffusion kernel (via `env.smooth`), so **all** smoothing methods build a dense
  kernel. The only memory mitigation is reducing the bin count (a larger
  `bin_size`). Docstrings and the high-bin warnings are corrected accordingly.

### UX hardening — fail-loud errors, one count convention, docs & viewer polish (0.8.0)

A broad usability pass: silent failures now raise or warn with actionable
messages, a few APIs were made consistent, and the docs/viewers were polished.

#### Breaking changes

- **`add_positions` is now `add_positions(events, *, times, positions)`** —
  reordered to the canonical `(data, times, positions)` order and made
  **keyword-only**. A bare 1-D trajectory makes both arrays 1-D, so a positional
  swap was shape-indistinguishable and silently mis-interpolated; keyword-only
  removes the ambiguity. *Migration:*
  `add_positions(events, times=times, positions=positions)`.
- **Assembly functions take `(n_time_bins, n_neurons)`** — `detect_assemblies`,
  `assembly_activation`, `pairwise_correlations`, and `reactivation_strength`
  now use the same time-first count convention as `decode_position` and
  `bin_spikes_in_time`, so a binned count matrix feeds them directly.
  *Migration:* pass the default `bin_spikes_in_time(...)` output as-is, or
  transpose an existing `(n_neurons, n_time_bins)` matrix.
- **Behavior result `summary()` returns a `dict`** of scalar metrics (was a
  formatted string) on every `behavior` result class. *Migration:* use
  `str(result)` for the human-readable form.
- **Fail-loud instead of silent sentinels.** `bin_at` on a graph/track
  environment raises on wrong-dimension points (was all `-1`);
  `heading_from_velocity` raises when every sample is below `min_speed` (opt out
  with `allow_all_nan=True`); graph queries reject a multi-point coordinate batch.

#### Added

- **`units=` and `frame=` keyword args** on the factory methods (`from_samples`,
  `open_field`, `linear_track`, `maze`) set the metadata at construction:
  `Environment.from_samples(pos, bin_size=2.0, units="cm")`.
- **Graph queries accept coordinates**, not just bin indices — `neighbors`,
  `path_between`, and `reachable_from` map a coordinate via `bin_at` (integers
  are still treated as indices).
- **`ResultMixin` on every result class** — concise `__repr__`/`_repr_html_`
  (summary metrics, not array dumps) and a scalar `summary()` dict.
- `dir(neurospatial.ops)` now surfaces the lazily-exported ops (autocomplete).

#### Fixed

- **`min_occupancy` thresholds raw occupancy seconds, not smoothed density** —
  fixes silent all-zero place fields on the documented golden path; applied
  identically across `diffusion_kde` / `gaussian_kde` / `binned` (single and
  batch). `min_occupancy=0.0` (the default) is unchanged.
- **Grid allocation is preflighted** — a transposed `(2, N)` trajectory (read as
  N dimensions) raises in ~1 ms instead of OOMing in `histogramdd`/`meshgrid`; a
  likely-transposed array now *warns* rather than false-rejecting a valid
  low-sample N-D environment.
- Non-finite query rows map to the `-1` "outside" sentinel **per row** (one NaN
  no longer collapses the whole occupancy/rate map on graph/masked envs).
- `to_file` refuses to overwrite by default (`overwrite=True` to allow);
  `occupancy()` checks shape before monotonicity (clear swapped-argument error);
  `from_graph` validates the edge `distance` attribute.
- `SpatialRateResult.plot()` documents the real `colorbar`/`colorbar_label`
  kwargs and labels the colorbar "Firing Rate (Hz)"; batch `plot()` raises an
  actionable error when no unit index is given.
- **Interactive-viewer accessibility** — the standalone HTML player ignores
  global keyboard shortcuts when an interactive control has focus, adds
  `:focus-visible` styles, and stops announcing every autoplay frame to screen
  readers; the napari region dock and track builder gain a compact layout,
  visible labels, and accessible controls.

#### Documentation

- Docs version bumped to 0.8.0; dead links and the fork-clone step fixed; a
  value-first, runnable quickstart; grouped/collapsible nav; a new advanced
  architecture page; colorblind-safe `viridis` in examples; example-notebook
  links normalized with an internal link-check step in docs CI.



### Additional release notes

- **BREAKING: `smoothing_method` renamed to `method`.** The smoothing/estimator
  keyword is now uniformly named `method` across every smoothing encoder
  (`compute_spatial_rate(s)`, `compute_view_rate(s)`,
  `compute_egocentric_rate(s)`, `is_place_cell`,
  `compute_directional_place_fields`), the four rate result classes
  (`SpatialRateResult`/`SpatialRatesResult`, `ViewRateResult`/`ViewRatesResult`),
  the decoder (`decode_session`, `decode_session_summary`, `BayesianDecoder`),
  and `simulation.validate_simulation`. This is a hard rename with no alias;
  defaults and numerics are unchanged (`"diffusion_kde"`, or `"binned"` for the
  egocentric encoders). NWB `write_spatial_rates` stores the estimator under the
  key `"method"` (schema bumped to `2.0`); this is a clean break with no
  back-compatibility shim, so spatial-rates tables written by earlier versions
  (schema `1.x`) no longer read. On
  `decode_session_summary`, `method` now names the smoothing estimator, so it is
  no longer accepted as a per-block decode kwarg (Poisson is the only likelihood
  and is applied unconditionally).

- **`method="binned"` is not a dense-kernel memory mitigation.**
  Earlier notes (through 0.6.0) recommended `binned` to avoid the dense
  `(n_bins, n_bins)` kernel, but `binned` smooths its rate map through the same
  diffusion kernel (via `env.smooth`), so all smoothing methods build a dense
  kernel. The only memory mitigation is fewer bins (a larger `bin_size`).

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

- **`units=` / `frame=` keyword args** on `from_samples`, `open_field`,
  `linear_track`, and `maze` set the metadata at construction.

- **Graph queries accept coordinates** (not just bin indices) — `neighbors`,
  `path_between`, `reachable_from` map a coordinate via `bin_at`.

- **`ResultMixin` on every result class** — concise repr/HTML and a scalar
  `summary()` dict.

- `dir(neurospatial.ops)` surfaces the lazily-exported ops (autocomplete).

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

- Version bumped to 0.8.0; dead links and fork-clone fixed; value-first runnable
  quickstart; grouped/collapsible nav; new advanced architecture page; `viridis`
  defaults; example-notebook links normalized + internal link-check in docs CI.

## [0.6.0] - 2026-07-03

### Commit highlights

#### Features
- feat(io/nwb): SpatialRatesResult round-trip (write_spatial_rates/read_place_field) + lazy reads (92cb14a)
- feat(decoding): add immutable BayesianDecoder (fit/predict/predict_summary/score) over the decode core (5ec00da)
- feat(recording): add frozen Session bundle + load_session (from_arrays/from_nwb, with_environment/restrict) (5edf393)
- feat(behavior): add restrict/in_epochs/restrict_spike_trains (array or IntervalSet epochs) (9b940fa)
- feat(encoding): add SpikeTrains ragged-spike container (label access, filter, unit_table) (1eba7ca)
- feat(typing): add PositionLike/SpikeTrainsLike/EnvironmentLike Protocols + pynapple ingress (optional) (b35330c)
- feat(decoding): honor float32 end-to-end through decode_session(_summary) via a dtype knob (fd58811)
- feat(encoding): speed filtering masks both spikes and occupancy in encode path (c514061)
- feat(decoding): add memory-safe summary decode + dtype/time_chunk on decode_position (afc1d68)
- feat: Update documentation and code to reflect 'unit' terminology for v0.6 API consistency (a5520ab)
- feat: Add ux-reviewer agent for evaluating user experience in scientific software (b2d3cdb)
- feat: Update API for detect_region_crossings and enhance documentation (0c0488e)
- feat(environment): experiment-shaped factory presets (open_field, linear_track, maze) (928a921)
- feat(encoding,behavior,decoding): enforce naming contract (classify/label_cell_types/is_place_cell/peak_location; decode_position result arg; deprecation aliases) (e3c9c94)
- feat(results)!: split terminal verbs — dense to_dataframe + new summary_table; PSTH ResultMixin (breaking, D1) (ccc72a6)
- feat(results)!: to_xarray returns a labeled xr.Dataset (breaking, D1) (14fe4f3)
- feat(encoding,events): thread unit_ids through population results and compute functions (c3e29e9)
- feat(decoding): add decode_session() one-call encode->bin->decode golden path (b1fabc4)

### Bug Fixes
- Merge pull request #25 from edeno/phase-3-pynapple-validation (6d8f7e2)
- fix(io/pynapple): validate shapes in to_pynapple (ValueError, not raw AssertionError) (241ed10)
- Merge pull request #24 from edeno/phase-3-trajectory-extra (3d635b1)
- fix(decoding): drop nonexistent neurospatial[trajectory] hints (scikit-image is core) (51e1ac5)
- Merge pull request #23 from edeno/phase-3-nwb-extra-msgs (8f4ac4d)
- fix(io/nwb): ndx-pose/ndx-events ImportError hints name the real [nwb] extra (14a8d93)
- Merge pull request #22 from edeno/phase-3-bugfixes (e5c11d9)
- fix(interop): restrict speed under fit(epoch=); mixed-type unit_ids; correct nwb extra name (56d0b51)
- fix(io/nwb): persist+validate env on rates round-trip (no degenerate env); atomic write; occupancy integrity; lazy length checks + unit caching; add nwb CI job (fefe6c3)
- fix(decoding): score warns/raises on undecodable bins; validate BayesianDecoder config+fitted state; add is_fitted, warn_on_drop, score distance=; error provenance (ff606f3)
- fix(recording): raise on malformed NWB env (not silent None); add environment_name; warn on empty restrict; Position self-validates (822ebc4)
- fix(behavior): raise on ambiguous nested-list epochs; validate closed on empty epochs; doc NaN exclusion (a3d6ee2)
- fix(typing): extract TsGroup trains by unit-id index (not iterate-keys); narrow EnvironmentLike; fix pynapple tests (9fd8d86)
- fix(decoding): route log_poisson_likelihood/poisson_likelihood dt through shared validate_dt (9b939a5)
- fix(decoding): validate dt in decode_position_summary too; test decode_position(_summary) dt guards; unify dt message; fix examples README sync wording (917ad4b)
- fix(decoding): unify dt validation (reject str/bool) via shared helper; clean ValueError for unparseable dtype (726ed71)
- fix(decoding): clean error for non-numeric/bool dt; restrict sync prune to numbered examples (23501df)
- fix(decoding): validate dt (finite, >0) in decode_session(_summary) before grid math (2766022)
- fix(decoding): preserve float32 in result-object handoff; validate time_chunk as positive int (not bool/float/str) (20b0e63)
- fix(decoding): reject time_chunk=None in summary decoders (it defeats the never-materialize-full-posterior contract) (eeb3d80)
- fix(encoding): warn when interval filtering empties the rate map (max_gap/out-of-bounds, not just min_speed) (7536d53)
- fix(decoding): validate prior shape in decode_position_summary (was silently truncating over-long 2D priors) (8b18f59)
- fix(encoding): align spike counts with occupancy on the FULL interval mask (max_gap, out-of-bounds, speed) (87b2164)
- fix(encoding): raise on speed-without-min_speed; warn on all-excluded speed filter; tighten final-sample gate (d13c968)
- fix(decoding): robust scaling test, precise float32 summary reductions, quiet degenerate-row warning (0847aa7)
- fix: address Phase 1 PR review (linear_track NaN guard, unit_table validation, all-NaN peaks, decode_position error, maze kind, honest annotations, docs) (92067e7)
- fix(encoding,docs): add peak_locations() to batch results; correct CLAUDE.md peak/cell-type contract; dedup CHANGELOG (aee29bf)
- fix(environment): order-based (coordinate-independent) maze topology; no track_graph mutation (9d9cc69)
- fix(results): raise (not silently skip) on xarray coord shape mismatch; omit None units (eeac275)
- fix(decoding): validate times up front in decode_session (beginner-grade non-finite error) (ec427eb)
- fix(decoding): warn on out-of-window spike drop in decode_session (units footgun) (09b1dea)
- fix: address Phase 0 PR review findings (E1006 code, docs accuracy, test gaps) (3f779e7)
- fix(decoding): keep decode_session export in neurospatial.decoding only (e6b2030)
- fix(environment): distance_to raises RegionNotFoundError for missing region (KeyError-compatible) (85410c0)
- fix(errors): remove internal-doc (CLAUDE.md) references from user-facing errors (63c8062)
- fix(environment): correct factory examples + error code in bare-Environment() message (3eaa207)
- fix(environment): beginner-grade error for bare Environment() pointing to factories (80d6ae7)
- fix(events): reject Inf event_times in align_spikes_to_events (matches docstring) (a364db1)
- fix(behavior): normalize array-likes at trajectory public boundaries before .ndim (2d4b5f8)
- fix(encoding): address review findings on spike-drop warning (4b3dfa5)
- fix(encoding): warn when spikes are dropped outside trajectory window or inactive bins (0d04d01)

### Documentation
- docs(interop): add Interoperability guide (session-first-optional: pynapple/NWB/Session/SpikeTrains/BayesianDecoder) (7848436)
- docs(plan): add Phase 3 interop/ergonomics design RFC (locks Protocols, Session, BayesianDecoder) (a0dc512)
- docs(decoding): widen likelihood Raises docstrings to match new dt validation (65b0e2c)
- docs(changelog): sync hosted changelog with latest v0.6 fixes (dt/dtype validation, float32 result handoff, time_chunk) (bdf2afd)
- docs(animation): fix broken per-frame memmap rate-map example; clarify jupytext vs docs-mirror sync; soften index parity claim (06f5e91)
- docs(examples): add missing example-23 row to index; correct CI sync guidance; prune stale mirrors in sync script (ebf0397)
- docs: surface decode_session_summary/scale APIs (api index, workflow long-session callout, batch processing, snippet CI) (15dd728)
- docs(changelog): sync hosted changelog with v0.6 scale API (dtype end-to-end, time_chunk hybrid + None rejection, summary speed forwarding) (e3eb03c)
- docs(encoding): correct compute_spatial_rates dtype note for R8 (decode_session now honors float32 end-to-end); test blockwise prior validation (cc7c498)
- docs: teach compute_spatial_rates batch path in README/animation/overlays/metrics (not per-neuron loops) (fe83cc4)
- docs(changelog): smoothing kernels warn and proceed (no hard gate / allow_large) (4a099e2)
- docs(examples): re-sync stale bayesian-decoding mirror from source; add CI sync guard (961ea4c)
- docs(decoding): teach batch/decode_session path not the per-neuron loop; add Phase 2 entries to docs changelog (241a8d5)
- docs/fix(decoding): correct time_chunk memory claim; DecodingSummary map_estimate + post-init validation; test gaps (59b3a56)
- docs(encoding): clarify float32 dtype decode-memory scope; fix batch return annotations (4dbfbc9)
- docs(encoding): teach compute_spatial_rates batch path, not the per-neuron loop (952b861)
- docs(encoding): note joblib warning-swallowing in population_coverage; coerce label_cell_types score dtype (27be8cf)
- docs: v0.6 naming contract in CLAUDE.md + migration guide + breaking-change CHANGELOG (1.6, 1.7) (c91184a)
- docs(encoding): note is_place_cell (field-based) vs classify (info-threshold) divergence (review nit) (d725d9f)
- docs,fix(encoding,events): document unit_ids/unit_table/unit_id; guard ndim; honest unit_id scalar (2a4d1ac)
- docs: sync Phase 0 plan §0.7 to shipped decode_session; archive re-review (d5cb4eb)
- docs(reviews): archive Phase 0 independent review with resolution note (d296b58)
- docs,ci: bump installation.md to v0.5.0; execute nb-20 in notebook CI (0abba24)
- docs: align decode_session example with manual path + execute notebook (review fixes) (b0b1e55)
- docs: teach decode_session as the one-call decode golden path (720547f)
- docs: fix degenerate Workflow 1 example + review nits (Phase 0.6) (4bb7636)
- docs: teach one canonical beginner path; fix broken API snippets (Phase 0.6) (da3ba4c)
- fix(errors): remove internal-doc (CLAUDE.md) references from user-facing errors (63c8062)
- docs(plan): apply independent + maintainer review findings to v0.6 plan (8cf9c92)
- docs(plan): add v0.6 UX implementation plan + review synthesis (9ccf0f7)
- docs: update CHANGELOG.md for v0.5.0 (9552a27)

### Other Changes
- Merge pull request #27 from edeno/chore/release-0.6.0 (ce62eff)
- chore(release): 0.6.0 — bump version, cut CHANGELOG [0.6.0], sync docs; fix stale doc refs (aa3e80f)
- chore(reviews): strip trailing blank line at EOF (git diff --check hygiene) (b393fd3)
- Merge pull request #21 from edeno/phase-3-simplify (19beb30)
- refactor(interop): dedup pynapple adapters, collapse Session spike coercion, avoid double spike/interval normalization (35af797)
- refactor(encoding): store SpikeTrains.trains as an immutable tuple; fix lazy-export comment (229135e)
- test(io): cover to_pynapple TypeError guard for non-result + values=None (3d27e81)
- chore(deps): lock pynapple optional extra (quantities, tabulate) (dd06e7e)
- test(decoding): structural throughput guard for summary decoders (per-block decode count = ceil(n_time/time_chunk), independent of n_neurons) (b0da2c1)
- test(encoding): enforce batch throughput contract — occupancy/kernel computed once, not per-neuron (deterministic call-count) (2aa7c21)
- test(decoding): cover decode_position time_chunk<1 guard; route summary prior-shape check through shared helper (f0d40d7)
- refactor(smoothing): warn (don't hard-gate) on high-bin dense kernels; drop allow_large/_KERNEL_HARD_LIMIT_BINS (a3e74eb)
- refactor(decoding): make decode_session_summary block-alignment check unconditional (survive -O) (08568b9)
- test(encoding): guard population_coverage docstring arg order (55c6bee)
- refactor(environment): sync compute_kernel protocol signature with allow_large (63c6d09)
- refactor: Remove type ignore comments for clarity in various encoding functions (fe31432)
- refactor: Rename methods and update docstrings for clarity in encoding tests (64d30ab)
- feat(encoding,behavior,decoding): enforce naming contract (classify/label_cell_types/is_place_cell/peak_location; decode_position result arg; deprecation aliases) (e3c9c94)
- test(encoding): rename stale to_dataframe/neuron_id test names to summary_table/unit_id (review nit) (7e882c0)
- refactor(decoding): let encoder own the drop warning in decode_session None branch (891a5be)
- refactor(encoding): rename normalize_spike_times -> as_spike_trains (f4cf99a)
- refactor(decoding): polish decode_session per review (times guard, ArrayLike, tests) (9bd454d)
- test: let CLAUDE.md backstop raise on unreadable file (review nit) (5ae370c)
- fix(environment): beginner-grade error for bare Environment() pointing to factories (80d6ae7)
- test(events): strengthen finite-event_times assertion (review nit) (8cad979)
- test(behavior): add list-input normalization tests for traveled_path_length (b81d78b)
- chore(worktrees): ignore .claude/worktrees/ for per-task subagent worktrees (994e796)

**Full Changelog**: https://github.com/edeno/neurospatial/compare/v0.5.0...v0.6.0


All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor releases may still include breaking changes;
these are called out under a dedicated **Breaking changes** heading.

### Changed

- The internal `_build_encoding_model` (shared by `decode_session` /
  `decode_session_summary` / `BayesianDecoder.fit`) gained an optional
  `context` parameter that names the caller in its up-front
  timestamp-validation error. `BayesianDecoder.fit` passes
  `context="BayesianDecoder.fit"`, so an `epoch` that selects too-few training
  samples now raises `"At least 2 samples required for BayesianDecoder.fit,
  got N"` instead of the misleading `decode_session` provenance. The default
  preserves every existing message.
- `decode_session` / `decode_session_summary` now accept `positions=None` when
  `encoding_models` is supplied (the passthrough decode never uses a position
  track). Previously this raised `"as_times_positions received a timestamp array
  but no positions"`; now the position track is normalized only when actually
  needed (the encode step, or a `PositionLike` `times`). Every existing caller
  that passes `positions` is byte-for-byte unchanged. This enables the
  fitted-model decode path used by `BayesianDecoder.predict` /
  `predict_summary`.

### Fixed

- The public likelihood functions `log_poisson_likelihood` and
  `poisson_likelihood` now **validate `dt` via the shared `validate_dt`
  helper**, consistent with the decode entry points. A non-numeric `dt`
  (including a numeric string like `"0.1"`), a `bool` (`dt=True`), and a
  non-finite `dt` (`nan`/`inf`) now raise a clear `ValueError` instead of the
  weaker `dt <= 0` guard (which leaked a raw `TypeError` on strings, silently
  accepted `dt=True` as `1`, and let `nan`/`inf` slip through). The error
  message changed from `"dt must be positive"` to `"dt must be a finite number
  > 0"`.
- `bin_spikes_in_time` now **validates `dt` consistently** with
  `decode_session` / `decode_session_summary` via a shared `validate_dt` helper:
  a non-numeric `dt` (including a numeric string like `"0.1"`) and a `bool`
  (`dt=True`) now raise a clear `ValueError` (`"dt must be a finite number >
  0, ..."`). Previously a numeric string leaked a raw `TypeError` from
  `"0.1" <= 0` and `dt=True` was silently accepted and used as a chunk size of
  `1`. `decode_position` routes through the same helper, so an invalid `dt`
  there also raises the clean message instead of a cryptic downstream error.
- An **unparseable `dtype`** (e.g. `dtype="bogus"`) now raises a clear
  `ValueError` naming `dtype` across the decode/encode entry points
  (`decode_position`, `decode_position_summary`, `compute_spatial_rates`,
  `decode_session`, `decode_session_summary`) instead of a raw NumPy
  `TypeError: data type 'bogus' not understood`.
- `decode_session` and `decode_session_summary` now **validate `dt`** (finite,
  `> 0`) up front with a clear `ValueError`, matching `bin_spikes_in_time`'s
  wording (`"dt must be finite and > 0, got ..."`). Previously the shared
  `_build_encoding_model` built the decode time grid directly, bypassing
  `bin_spikes_in_time`'s guard, so an invalid `dt` leaked a cryptic error:
  `dt=0` → `ZeroDivisionError`, `dt=NaN` → "cannot convert float NaN to
  integer", `dt<0` → a misleading "span smaller than one bin" message, and
  `dt=inf` → a similar cryptic failure.
- `decode_position` now **preserves `float32`** when handed a rate-result object
  (anything exposing `.firing_rates`). Previously the friendly object path
  promoted a `float32` `.firing_rates` to `float64`, silently losing part of the
  `dtype=np.float32` memory win that the raw-array path already delivered. The
  object path now matches the raw-array path byte-for-byte: `float32` stays
  `float32`, `float64` stays `float64`, an integer rate map is promoted to
  `float64`, and a `None` / dict / non-2-D `.firing_rates` still raises the same
  clear `ValueError`.
- `time_chunk` is now validated as a **positive integer (not `bool`)** across
  `normalize_to_posterior`, `decode_position`, `decode_position_summary`, and
  `decode_session_summary`, via a shared validator that raises a clear
  `ValueError` naming the value and its type. Previously a float (`1.5`) or
  string (`"2"`) leaked a raw `TypeError` from `range(...)`/comparison, and
  `True` was silently accepted as a chunk size of `1`.
- The summary decoders `decode_position_summary` and `decode_session_summary`
  now **reject `time_chunk=None`** (raising a clear `ValueError`). Previously a
  `None` value set the streaming block to the full session length, materializing
  the full `(n_time, n_bins)` posterior transiently and defeating the
  memory-safe "never materialize the full posterior" contract these functions
  promise. `time_chunk` must be a positive integer (default `1024`); use
  `decode_position` / `decode_session` if you want the full posterior.

### Added

- NWB `SpatialRatesResult` round-trip + lazy reads (`neurospatial.io.nwb`).
  `write_spatial_rates(nwbfile, result, *, name="spatial_rates", overwrite=False)`
  stores a population result on a **unit axis** — a `DynamicTable` (one row per
  unit) in `analysis/` with a `unit_id` column, a 2-D `firing_rate` column of
  shape `(n_units, n_bins)`, and one column per `unit_table` field; occupancy is
  stored once as a companion `TimeSeries`, `smoothing_method` / `bandwidth` /
  bin counts ride in the table description, and bin centers are shared via the
  existing `bin_centers` dataset. The full `Environment` is now **persisted**
  alongside the rates (via `write_environment` under the derived name
  `f"{name}_environment"`), so it round-trips with its **connectivity edges and
  geometry intact**. The write is **atomic**: all name collisions (the table,
  `f"{name}_occupancy"`, `f"{name}_environment"`) and shape validation are
  resolved before the first object is added, so a duplicate without `overwrite`
  raises before any mutation and `overwrite=True` cleans every companion; a
  later add failure rolls back the partial write. `firing_rates` / `occupancy`
  are defensively copied so the NWB containers never alias the live result, and a
  `unit_table` column named `unit_id` / `firing_rate` (reserved) raises a clear
  `ValueError`. `read_place_field(nwbfile, *, name="spatial_rates", env=None)` is
  the inverse, reconstructing a `SpatialRatesResult` with `firing_rates` /
  `occupancy` / `unit_ids` / `unit_table` / `smoothing_method` / `bandwidth` all
  preserved (**`unit_ids` and `unit_table` links survive** non-default ids and
  non-trivial tables). When `env=` is omitted it restores the **persisted
  environment with full connectivity** (`read_place_field(nwb).env.neighbors(i)`
  matches the original; there is no connectivity-less bin-centers fabrication),
  and it raises if neither an `env=` nor a persisted env is present. A mismatched
  or stale `env=` (its `n_bins` disagreeing with the stored rates) now raises a
  clear `ValueError` instead of silently attaching, the companion occupancy
  length is validated at read against the table's recorded `n_bins`, and a `name`
  pointing at a non-spatial-rates table raises a clear `ValueError`. Shape
  mismatches raise param-named `ValueError`s and duplicate names honor
  `overwrite`. `read_position` / `read_pose` / `read_units` gain a keyword-only
  `lazy=False`: `lazy=True` returns h5py-backed handles (per-unit spike-time
  handles for `read_units`) that materialize only when sliced / `np.asarray`-ed —
  valid only while the backing `NWBFile` is open — while the default eager path
  stays byte-for-byte unchanged. The lazy `read_position` / `read_pose` paths now
  **validate lengths** via the handles' `.shape[0]` (no materialization), so a
  positions/timestamps or per-bodypart mismatch raises exactly as the eager path
  (lazy pose no longer silently misaligns bodyparts of differing length), and the
  lazy per-unit spike-time handle **caches** its materialized array (the ragged
  slice is read and sorted at most once). A `test_nwb.yml` CI job installs the
  `nwb` extra and runs the NWB tests (`tests/nwb` + `tests/test_recording.py`),
  which the default `dev`-only job skips. `neurospatial` still never imports
  pynwb (loaded lazily in `io/nwb`).
- `BayesianDecoder` — an immutable (`frozen`) `fit`/`predict`/`predict_summary`/
  `score` wrapper over the decode core (`neurospatial.decoding`, also exported at
  the top level). `fit(spike_times, times, positions, *, speed=None,
  min_speed=None, epoch=None)` builds encoding models by reusing
  `decode_session`'s internal encoder and returns a **new** fitted decoder
  (never mutates the original); an optional `epoch` restricts the training data
  first (via `behavior.restrict` / `restrict_spike_trains`) for train/test
  splits. `predict` / `predict_summary` delegate to `decode_session` /
  `decode_session_summary` with the fitted models, so a fitted decoder's
  posterior is **byte-exact** with `decode_session` on the same inputs; `score`
  reports decode error (`"median_error"` / `"mean_error"`, lower is better) via
  `DecodingResult.error_against`. `score` gained a `distance=` option
  (`"euclidean"` default or `"geodesic"`, forwarded to `error_against`) so
  scoring can use the environment's graph distance, not only straight-line
  error, and now **does not silently drop undecodable bins**: it **warns**
  (naming the excluded fraction) when any decode time bin is undecodable
  (all-non-finite posterior row, which `nanmedian` / `nanmean` would otherwise
  ignore) and **raises** a clear `ValueError` — instead of returning `nan` — when
  *no* bin is decodable (naming the likely degenerate-encoding-model or
  seconds-vs-milliseconds unit-mismatch cause); the all-decodable path stays
  warning-free. Invalid `metric` / `distance` are validated **before** decoding
  so a typo does not cost a full decode. Construction now validates config and
  fitted state: `dt` (via the shared `validate_dt`) and `dtype` are checked at
  build time, and a directly-injected fitted decoder
  (`BayesianDecoder(env, encoding_models=..., unit_ids=...)`) is checked for
  fitted-state coupling (`unit_ids` present, 2-D models, and a bin/unit-count
  match against `env`) instead of detonating later inside the core. A new
  read-only `is_fitted` property lets callers branch without catching
  `RuntimeError`, and a `warn_on_drop` config field (default `True`) threads to
  both the `fit` encode step and the `predict` / `predict_summary` decode steps
  as a single knob to silence the spikes-out-of-window warnings. Accepts
  `SpikeTrainsLike` / `PositionLike` inputs, and decodes through the
  `Environment`, so geodesic / linearized-track / graph-based decoding works
  (unlike pynapple `decode_1d` / `decode_2d`). Unfitted `predict` /
  `predict_summary` / `score` raise a clear `RuntimeError`.
- `Session` — a frozen **discoverability bundle** (`neurospatial.recording`,
  also exported at the top level) grouping `env` / `position` / `spikes` /
  `epochs` / `metadata`. It is **not a god-object**: it exposes the raw arrays
  (`session.times` / `session.positions` / `session.spikes`) but carries **no**
  heavy analysis methods — compute stays functional, e.g.
  `compute_spatial_rates(session.env, session.spikes, session.times,
  session.positions)`. `position` uniformly exposes `.t` / `.values` (arrays are
  wrapped in a small internal `Position` holder; a pynapple `Tsd` / `TsdFrame`
  already conforms). Constructors: `Session.from_arrays(*, env=None, times,
  positions, spike_times, unit_ids=None, unit_table=None, epochs=None,
  metadata=None)` (accepts arrays or a `PositionLike`, and a list / 2-D array /
  `SpikeTrains` / pynapple `TsGroup` for spikes, threading `unit_ids` /
  `unit_table`) and `Session.from_nwb(path_or_file, *, environment_name=None,
  unit_ids=None, **read_kwargs)`. Frozen / immutable: `with_environment(env)` and
  `restrict(epochs)` return a **new** `Session` and never mutate the original.
  `restrict` slices the position and the spikes to the epochs and is
  **identity-preserving** — restriction trims spikes per unit (never drops
  units), so it rebuilds a `SpikeTrains` carrying the original `unit_ids` /
  `unit_table` unchanged, and records the epochs on the new session. A
  `restrict` that keeps **zero** position samples (epochs that miss the session
  entirely — often a seconds-vs-milliseconds unit mismatch) emits a
  `UserWarning` naming the likely cause and returns the (empty) session. A
  mismatched `position.t` / `position.values` length (also self-enforced by the
  internal `Position` holder, which additionally requires a 1-D `t`), a
  `position` that is not `PositionLike` (missing `.t` / `.values`), or a
  non-`EnvironmentLike` `env` raises a clear `ValueError`.
- `load_session(source, **kwargs)` — dispatches an NWB file path (`str` /
  `os.PathLike`) or an open pynwb `NWBFile` to `Session.from_nwb`; any other
  `source` raises a clear `TypeError` directing you to `Session.from_arrays`.
  `Session.from_nwb` builds the bundle via the existing lazy
  `neurospatial.io.nwb` readers (`read_units` / `read_position` /
  `read_environment`), so `neurospatial.recording` **never imports pynwb /
  pynapple** and `import neurospatial` stays cheap. The environment is read by
  **presence** (membership in `nwbfile.scratch`), not by catching an error: a
  genuinely-absent environment maps to `env=None`, while an environment that is
  **present but unreadable** (malformed / wrong schema) now **raises** instead of
  being silently swallowed to `None`. The new `environment_name=` selector reads
  the standard `spatial_environment` scratch entry by default and can select an
  environment written under a custom name. (The lazily-materialized `lazy=True`
  NWB read path is intentionally deferred to a later phase.)
- `restrict` / `in_epochs` / `restrict_spike_trains` — array-native epoch
  selection ("give me my running periods / trial N") in a new
  `neurospatial.behavior.epochs`. `epochs` accepts `(start, end)` scalars, two
  `(starts, ends)` 1-D arrays, an `(n, 2)` array, **or a pynapple `IntervalSet`**
  — the `IntervalSet` is **duck-typed** (`.start` / `.end`), so this stays
  array-first and **never imports pynapple**. The one genuinely ambiguous form
  — a length-2 pair of length-2 sequences (`[[0, 5], [10, 15]]`), which could
  mean two `(start, end)` interval rows **or** two parallel `(starts, ends)`
  arrays — **raises** `ValueError`; pass an `(n, 2)` NumPy array (interval rows)
  or explicit 1-D `start`/`end` arrays to disambiguate. Endpoints are **inclusive** by
  default (`closed="both"`, matching `behavior.segmentation` and pynapple; also
  `"left"` / `"right"` / `"neither"`). `restrict(times, *arrays, epochs=...)`
  slices `times` and any arrays **aligned to `times`** (e.g. `positions`) by the
  same in-epoch mask, order preserved (`t, pos = restrict(times, positions,
  epochs=run_epochs)`); with no extra arrays it restricts an event-time array by
  its own timestamps (`restrict(spike_train, epochs=...)`).
  `restrict_spike_trains(trains, epochs)` restricts **ragged** per-unit spikes,
  each train by its **own** timestamps, and accepts a plain sequence or a
  `SpikeTrains`. `restrict` is also exported at the top level (`neurospatial.restrict`).
- `SpikeTrains` — a frozen ragged-spike-train container (label access
  `st[unit_id]`, `filter("region == 'CA1'")`, optional per-unit `unit_table`),
  the one justified new container (ragged per-unit spike times genuinely don't
  fit a rectangular array). Exported from both `neurospatial` (top level) and
  `neurospatial.encoding`. It **duck-types as `SpikeTrainsLike`** — it is not a
  `Mapping`, exposes a non-callable `.index` property (the unit ids), and its
  `__iter__` yields the per-unit **train arrays** — so it flows through the
  Phase 3.1 spike-input adapter's iterate branch and into `compute_spatial_rates`
  (and the other batch encoders/decoders) with `unit_ids` preserved. Label
  access (`st[unit_id]`) and the adapter's positional iteration coexist because
  they use different dunders (`__getitem__` vs `__iter__` / `.index`). Frozen
  and immutable: `trains` is stored as a `tuple` so in-place mutation raises,
  `filter` returns a new container and never mutates the original; duplicate
  `unit_ids` raise `ValueError`.
- Input Protocol surface + pynapple ingress/egress (optional). A new
  `neurospatial/_typing.py` defines the structural Protocols that let
  third-party objects flow into the **array-first** scientific core without the
  core ever importing or `isinstance`-checking them:
  - `PositionLike` — a `.t` / `.values` time-series (pynapple `Tsd` / `TsdFrame`
    conform). `compute_spatial_rate` / `compute_spatial_rates` and
    `decode_session` / `decode_session_summary` now accept a `PositionLike` in
    the `times` slot (with `positions` omitted) and normalize it to plain
    `float64` arrays at the boundary via `as_times_positions`. The plain-array
    path is unchanged byte-for-byte.
  - `SpikeTrainsLike` — the accepted spike-input union, plus a real pynapple
    `TsGroup`. A `TsGroup` is a `collections.UserDict`, so **iterating it yields
    the unit-id keys, not the per-unit trains**; `SpikeTrainsLike` is therefore
    the *indexable-by-id* surface (`.index` of unit ids + `group[uid]` returning
    a per-unit `Ts` with `.t`). A new `encoding.as_spike_trains_with_ids`
    extracts trains by indexing each id (never by iterating), so a raw `TsGroup`
    flows correctly into `compute_spatial_rates` / `decode_session`; it surfaces
    the ids without changing `as_spike_trains`'s `list[NDArray]` contract, and
    when a group carries ids and the caller passes no `unit_ids=`, they now flow
    into `SpatialRatesResult.unit_ids` instead of being silently dropped.
  - `EnvironmentLike` — a public re-export of `EnvironmentProtocol`. The
    internal `isinstance(env, Environment)` check in `CompositeEnvironment` is
    replaced by a duck-typed `is_environment_like` check, fixing the surprise
    that the sibling `EgocentricPolarEnvironment` (not an `Environment`
    subclass) was rejected even though it is a legitimate environment.
- pynapple I/O shim behind a new optional `pynapple` extra:
  `neurospatial.io.from_pynapple` (`TsGroup` → `(trains, unit_ids)`;
  `Tsd` / `TsdFrame` → `(times, positions)`; `IntervalSet` → `(start, end)`) and
  `neurospatial.io.to_pynapple` (a decoded MAP track / `times`+`values` →
  `Tsd` / `TsdFrame`). `import pynapple` is lazy inside these functions, so the
  package and the array path import and run with pynapple absent; calling them
  without it raises a clear `ImportError` naming `neurospatial[pynapple]`. The
  scientific modules never import pynapple.
- Speed filtering on the encode path. `compute_spatial_rate` and
  `compute_spatial_rates` gain keyword-only `speed` / `min_speed` parameters
  (also forwarded by `decode_session` / `decode_session_summary`). When
  `min_speed` is set, low-speed periods are excluded using **one shared
  per-interval speed gate** applied to **both** the spike numerator and the
  occupancy denominator, so a `min_speed` knob can never silently bias firing
  rates by filtering only one side. The spike gate matches the occupancy gate
  exactly: occupancy keeps interval `k` iff `speed[k] >= min_speed`, and a
  spike at time `t` is kept iff
  `speed[searchsorted(times, t, "right") - 1] >= min_speed` — the same
  per-interval criterion, so identical intervals drop on both sides. When
  `speed` is omitted it is auto-derived with a **forward difference**
  (`speed[k] = ||positions[k+1] - positions[k]|| / (times[k+1] - times[k])`,
  with `speed[n-1] = speed[n-2]`) to match `env.occupancy`'s
  `time_allocation="start"` interval semantics; pass an explicit `speed` for
  geodesic / linearized-track environments. `min_speed=None` (the default)
  applies no filtering and leaves output byte-for-byte unchanged.
  - Passing `speed` **without** `min_speed` now raises `ValueError` (instead of
    silently ignoring `speed`), mirroring `env.occupancy`, which raises on
    `min_speed` without `speed`.
  - When the interval filter excludes **all** trajectory intervals — whether
    via `min_speed`, `max_gap`, or the out-of-bounds-start rule (e.g. a
    wrong-units threshold) — `compute_spatial_rate` / `compute_spatial_rates`
    now emit one `UserWarning` naming the active gate(s) and suggesting fixes
    (check units; pass `max_gap=None`), instead of silently returning an empty
    rate map. The warning fires once per call (not per neuron in the batch
    path) and is suppressed by `warn_on_drop=False`.
- `decode_session` and `decode_session_summary` gain a keyword-only `dtype`
  parameter (`np.float32` / `np.float64`, default `np.float64`). "Decode in
  this dtype": a single `decode_session(dtype=np.float32)` controls **both**
  the encoding-model working set and the posterior dtype, end-to-end — the
  functions no longer force-promote encoding models back to `float64`. On
  `decode_session_summary` it is now an explicit parameter rather than a
  `decode_kwargs` entry. Default `np.float64` leaves every existing caller
  byte-for-byte unchanged; any other dtype raises `ValueError`.

### Performance

- `decode_session(dtype=np.float32)` / `decode_session_summary(dtype=np.float32)`
  now actually halve the decode working set on the beginner golden path: the
  `float32` rate-map dtype is honored end-to-end (encoding-model working set +
  posterior) instead of being silently promoted back to `float64` inside the
  session helpers. Values match `float64` within `float32` tolerance (the rate
  computation is done in `float64` and only the result is cast, per
  `compute_spatial_rates`).

### Breaking changes

- `to_xarray()` now returns a labeled `xarray.Dataset` instead of an
  `xarray.DataArray` with integer coordinates. This is a clean break: there is
  no `DataArray` shim and no `to_dataset()` alias. Two distinct shapes are
  produced:
  - **Population rate results** (`SpatialRatesResult`, `DirectionalRatesResult`,
    `ViewRatesResult`, `EgocentricRatesResult`) return a `Dataset` with dims
    `("unit_id", "bin")`. `unit_id` is the index coordinate holding the *real*
    per-unit identity labels (`result.unit_ids`), so units are selected by
    label. The `bin` dimension carries non-index `bin_center_x` / `bin_center_y`
    (/ `bin_center_z`) coordinates for Cartesian environments, or
    `bin_center_distance` / `bin_center_angle` for the polar egocentric result.
    The rate matrix is the `firing_rate` data var; `occupancy` is a `("bin",)`
    data var. `attrs` carry `units`, `bandwidth` (where applicable), an `env`
    fingerprint, and `software_version`. Duplicate `unit_ids` now raise
    `ValueError` (label-based selection requires uniqueness).
  - **Decode results** (`DecodingResult`) return a `Dataset` with dims
    `("time", "bin")` (a posterior over space per time bin; no `unit_id` axis).
    The `posterior` data var holds the posterior, with the same `bin_center_*`
    coordinate logic and `units` / `env` / `software_version` attrs.

    Before → after:

    ```python
    # before (DataArray, integer coords)
    da = result.to_xarray()
    da.sel(neuron=0)
    # after (Dataset, real unit_id labels)
    ds = result.to_xarray()
    ds.sel(unit_id=result.unit_ids[0])
    ```

- The two terminal verbs now mean **one** thing on every result class.
  `to_dataframe()` on the batch (plural) encoding results — `SpatialRatesResult`,
  `DirectionalRatesResult`, `ViewRatesResult`, `EgocentricRatesResult` — is now
  **dense tidy**: one row per `(unit, bin)` (single-unit results: one row per
  `bin`), always carrying a `unit_id` column plus the bin-center coordinate
  columns (`bin_center_x`/`y`/`z` for Cartesian, `bin_center_distance`/`angle`
  for polar egocentric, `bin_center_angle` for directional), `firing_rate`, and
  `occupancy`. The **per-unit summary** that `to_dataframe()` used to return
  (one row per neuron with `peak_x`, `peak_rate`, `spatial_info`, `sparsity`,
  `grid_score`, `border_score`, `cell_type`, etc.) has moved to the new
  `summary_table()`, which is `unit_id`-indexed. This is a clean break: there is
  no mode flag and no transition shim. The `neuron_ids=` keyword on the old
  per-unit `to_dataframe()` is replaced by `unit_ids=` on `summary_table()`
  (defaulting to the result's own `unit_ids`).

    Before → after:

    ```python
    # before — to_dataframe() returned one row per neuron with metric columns
    df = result.to_dataframe()           # columns: neuron_id, peak_x, ...
    place = df[df["cell_type"] == "place"]

    # after — summary_table() is the per-unit summary; to_dataframe() is dense
    summary = result.summary_table()     # one row per unit, unit_id-indexed
    place = summary[summary["cell_type"] == "place"]
    dense = result.to_dataframe()         # one row per (unit, bin), carries unit_id
    ```

### Added

- Memory-safe summary decoding for long sessions. `decode_position_summary`
  is a new sibling of `decode_position` that streams over time, computing the
  posterior one time-block at a time and reducing each block to per-time
  scalars/vectors (`map_position` / `map_bin`, `mean_position`,
  `posterior_entropy`, `peak_prob`) without ever materializing the full
  `(n_time, n_bins)` posterior. It returns a new `DecodingSummary` frozen
  dataclass (alongside `DecodingResult`) carrying `ResultMixin` and the
  standard terminal verbs — `to_dataframe()` (one row per time bin),
  `summary()`, `plot()`, and `to_xarray()` (a track `Dataset` with a `time`
  dim and **no** `bin` posterior axis) — sharing accessor names and column
  conventions with `DecodingResult` so user code ports between the two.
  `decode_session_summary` is the matching one-call encode→bin→decode wrapper
  (sibling of `decode_session`). The summary reductions are bit-for-bit
  identical to reducing the full posterior for `map_position` / `map_bin`, and
  match to floating-point tolerance for `mean_position` / `posterior_entropy` /
  `peak_prob`.

- Experiment-shaped factory presets on `Environment` that speak experiment
  vocabulary and delegate to the existing `from_*` factories:
  - `Environment.open_field(positions, bin_size, ...)` — the only
    positions-based preset; delegates to `from_samples` with `fill_holes=True`
    flipped on (a sensible open-arena default that fills interior gaps).
  - `Environment.linear_track(*, endpoints=..., node_positions=..., bin_size)`
    — builds a 1D track graph (`is_linearized_track == True`) from an explicit
    topology (two endpoints for a straight track, or waypoints for a
    piecewise-linear track) and delegates to `from_graph`.
  - `Environment.maze(kind, *, track_graph=..., node_positions=..., bin_size)`
    — assembles the standard W / plus / T track-graph topology (or accepts a
    ready `networkx` graph) and delegates to `from_graph`.

  Track/maze presets require an explicit topology spec; raw positions cannot
  infer a linear/W/plus/T graph, and calling them without a topology raises a
  clear `ValueError`.

- `to_xarray()` on `DirectionalRatesResult`, `ViewRatesResult`, and
  `EgocentricRatesResult` (the directional/view/egocentric population results
  previously had no xarray export). Each returns the labeled `xr.Dataset`
  described under Breaking changes above.

- Durable unit identity on encoding/events results. Every population result
  (`SpatialRatesResult`, `DirectionalRatesResult`, `ViewRatesResult`,
  `EgocentricRatesResult`, `PopulationPeriEventResult`) now carries a
  `unit_ids: np.ndarray` field (plus an optional `unit_table: pd.DataFrame |
  None`), and every single-unit result (`SpatialRateResult`,
  `DirectionalRateResult`, `ViewRateResult`, `EgocentricRateResult`,
  `PeriEventResult`) carries a singular `unit_id`. Indexing or iterating a
  population result stamps the per-unit label onto the child
  (`rates[i].unit_id == rates.unit_ids[i]`, iteration preserves order and
  labels). The batch compute functions (`compute_spatial_rates`,
  `compute_directional_rates`, `compute_view_rates`,
  `compute_egocentric_rates`, `population_peri_event_histogram`) gained a
  keyword-only `unit_ids=` parameter that threads onto the result; a
  wrong-length value raises a clear `ValueError`. Fully back-compatible:
  `unit_ids` defaults to `np.arange(n_units)` and the new fields are
  `compare=False`, so existing callers and equality/hash behavior are
  unchanged.

- `summary_table()` — the per-unit summary terminal verb — on every batch
  encoding result (`SpatialRatesResult`, `DirectionalRatesResult`,
  `ViewRatesResult`, `EgocentricRatesResult`) and on `PopulationPeriEventResult`.
  Returns one row per unit, `unit_id`-indexed, with that result's scalar metric
  columns (peak location/rate, spatial info, grid/border score, cell type,
  preferred direction/distance, etc.). Accepts an optional `unit_ids=` to
  relabel the index.

- PSTH results now carry the uniform result surface. `PeriEventResult` and
  `PopulationPeriEventResult` inherit the canonical `ResultMixin` and implement
  the terminal verbs: `to_dataframe()` (dense — one row per time bin for the
  single-unit result, one row per `(unit, time-bin)` for the population result,
  always carrying `unit_id`), `summary()` (flat dict of headline scalars —
  peak rate/latency, baseline rate; population adds `mean_peak_rate` /
  `population_peak_latency`), `PopulationPeriEventResult.summary_table()` (one
  row per unit with `peak_rate` / `peak_latency` / `baseline_rate`), and
  `plot()` (delegates to `plot_peri_event_histogram`, returns the axis).

- `decode_session(env, spike_times, times, positions, *, dt, ...)` — one-call
  encode→bin→decode golden path in `neurospatial.decoding.session`.  Glues
  `compute_spatial_rates`, `bin_spikes_in_time`, and `decode_position` into a
  single function so beginners can decode position in ≤10 lines.  Exported from
  `neurospatial.decoding` (`from neurospatial.decoding import decode_session`).
  Accepts an optional `encoding_models=` array to bypass the encoding step
  entirely.
  Extra keyword arguments are forwarded verbatim to `decode_position`.

- `as_spike_trains` — public helper that coerces the various spike-input
  formats (1D array, NaN-padded 2D array, list of scalars, list of 1D arrays)
  into the canonical list of per-neuron spike-time arrays. Exported from
  `neurospatial.encoding` (and present in `__all__`); the implementation lives
  in `neurospatial.encoding._spikes`. It standardizes the container shape only
  — spike-time values are never shifted, rescaled, or aligned. (This is the
  previously-internal `normalize_spike_times`, renamed before any public
  release so the name reads as a structural conversion, not a value transform;
  no deprecated alias is kept since the public name was never released.)

- `classify(*, ...)` — a single-type boolean cell-type predicate (returns
  `NDArray[np.bool_]`) on every batch encoding result: `EgocentricRatesResult`
  (OVC, `min_info=0.3`), `ViewRatesResult` (view cell, `min_info=0.5`),
  `DirectionalRatesResult` (HD cell, `min_mvl`/`alpha`), and `SpatialRatesResult`
  (place cell, `min_spatial_info=0.5`). These replace the per-domain
  `detect_ovcs` / `detect_view_cells` / `detect_hd_cells` detectors (now
  deprecated aliases).

- `label_cell_types(...)` on `SpatialRatesResult` — the multi-class string
  labeler (`"place"`/`"grid"`/`"border"`/`"unclassified"`, returns
  `NDArray[np.str_]`). This is the renamed `detect_cell_types` and is kept
  deliberately SEPARATE from the boolean `classify` (different return type).

- `is_place_cell(...)` — a single-neuron place-cell predicate, both as a free
  function in `neurospatial.encoding.spatial` (exported from
  `neurospatial.encoding`) and as a `SpatialRateResult.is_place_cell()` method.
  Mirrors `is_spatial_view_cell` / `is_object_vector_cell` and agrees with
  `detect_place_fields` (returns `True` iff that detector finds ≥1 field).

- `decode_position(env, spike_counts, encoding_models, dt, ...)` now accepts a
  population rate result object (anything exposing a `firing_rates` attribute,
  e.g. `SpatialRatesResult`) directly in place of the raw `(n_neurons, n_bins)`
  array, removing the `np.stack([r.firing_rate ...])` glue between the encoding
  and decoding steps.

- New keyword-only parameter `warn_on_drop: bool = True` on
  `bin_spike_train`, `bin_spike_trains` (`encoding/_binning.py`),
  `compute_spatial_rate`, and `compute_spatial_rates` (`encoding/spatial.py`).
  Set to `False` to intentionally silence all spike-drop warnings (e.g. when
  the caller handles the diagnostic themselves).

### Performance

- `decode_session_summary` now **streams the time-binning** so the full
  `(n_time, n_neurons)` spike-count matrix is never materialized. It builds the
  encoding model once over the whole session (small, `(n_neurons, n_bins)`),
  then bins spikes block-by-block against a contiguous slice of the global time
  grid and decodes + reduces each block via the same shared inner-loop helper as
  `decode_position_summary`. Peak memory is now
  `O(time_chunk × max(n_neurons, n_bins))` plus the `(n_neurons, n_bins)`
  encoding model — **independent of session length** — meeting the
  1 hr / 25 ms / 5000-bin / <500 MB summary-decode DoD golden path (the dense
  `(144000, 1000)` count matrix alone would be ~1.15 GB). The result is
  byte-for-byte identical to the prior materialize-then-stream path; per-block
  counts are binned against the precomputed global edges so they match a single
  global histogram exactly, and boundary spikes are counted exactly once.
  `decode_session` (the full-posterior path) is unchanged.

- `decode_position` gains two keyword-only memory knobs with **no change to its
  return contract** (`.posterior` stays a fully-materialized `ndarray`):
  - `dtype=np.float32` stores and computes the posterior in single precision,
    halving stored and transient memory (parity with float64 to ~1e-6
    relative). Every `DecodingResult` method works unchanged on a float32
    posterior.
  - `time_chunk` is now **hybrid**. `time_chunk=None` (the default) keeps the
    full-matmul path **byte-for-byte unchanged** — the Poisson log-likelihood is
    computed once over the whole window and normalized at once (transient peak
    ~3× the stored posterior). An **explicit `time_chunk=k`** now computes the
    Poisson log-likelihood **blockwise directly into the preallocated
    posterior**, so the full-size log-likelihood and its working copy are never
    materialized — cutting the transient peak to **~1×** over the returned
    posterior (the posterior itself is unavoidably 1×, since `decode_position`
    returns the full dense array). The opt-in path is **tolerance-equal**, not
    byte-exact, to the full path: the per-block likelihood matmul is a different
    BLAS shape than the full matmul, so it differs by ~1e-15 (MAP/argmax
    identical; every row sums to 1). For a path that never holds even the full
    posterior, use `decode_position_summary`.

  For sessions where even the stored dense posterior is too large to hold, use
  the new `decode_position_summary` / `decode_session_summary` (see **Added**),
  which never materialize the full `(n_time, n_bins)` posterior.

- `compute_spatial_rates` gains a keyword-only `dtype` (`np.float32` /
  `np.float64`, default `np.float64`). `dtype=np.float32` halves the stored
  `(n_units, n_bins)` rate-map array. The rate computation (GEMM / division) is
  still performed in float64 and only the final result is cast, so float32
  values match the float64 default within float32 tolerance. `decode_session` /
  `decode_session_summary` now accept their own `dtype` parameter (default
  float64) that honors float32 end-to-end — the encoding-model working set and
  the posterior — so `decode_session(dtype=np.float32)` halves the decode
  working set on the golden path (see the `decode_session` / `decode_session_summary`
  `dtype` entry above). Default `np.float64` leaves every existing caller
  byte-for-byte unchanged; any other dtype raises `ValueError`.

- Documented the dense diffusion-kernel **O(n²) memory cost** and added a loud
  high-bin memory **warning**. The heat kernel `exp(-tL)` of a connected graph
  is dense by construction (every entry > 0), so it always costs
  `n_bins**2 * 8` bytes of float64 memory (≈ 3.2 GB at 20,000 bins).
  `compute_diffusion_kernels` and `env.compute_kernel` now emit a loud
  `UserWarning` (with a GB estimate) above 3,000 bins and then **proceed** —
  there is **no hard limit** and no `allow_large` opt-out. The warning names the
  size, the GB estimate, the dense O(n²) reason, and the fixes (reduce bins or
  use `smoothing_method="binned"`). No numerical results change — this is
  documentation plus a warn-and-proceed guard. The reliable scale wins this
  release remain float32 rate maps and the memory-safe summary decode (above). A
  faster, lower-peak `expm_multiply` / Chebyshev rewrite of the kernel is a
  **deferred stretch goal** and is intentionally **not** part of this release.

- Warned on the dense **Gaussian-KDE** high-bin path the same way as the
  diffusion kernel. `smoothing_method="gaussian_kde"` builds a dense
  `(n_bins, n_bins)` weight matrix (`exp(-d²/2σ²)`, every entry > 0), so it
  carries the same O(n_bins²) memory cost; it now emits a loud `UserWarning`
  (with a GB estimate) above the shared `_LARGE_KERNEL_THRESHOLD` (3,000 bins)
  and proceeds — no hard limit, no `allow_large`. One warn threshold is shared
  by both dense smoothing paths instead of a divergent copy. Also corrected
  stale `encoding/_smoothing.py` docs that wrongly described the diffusion
  kernel as "sparse" / `O(n_bins)` — it is dense `(n_bins, n_bins)`,
  O(n_bins²) per neuron, with a one-time O(n_bins³) matrix-exponential build.
  No numerical results change.

- `SpatialRatesResult.summary_table()` no longer double-computes grid and
  border scores. It previously ran the expensive per-neuron grid/border score
  computation **twice** per call (once for the `grid_score`/`border_score`
  columns and again inside `label_cell_types()`); it now computes each once and
  forwards them. `label_cell_types()` gains optional keyword-only
  `grid_scores=` / `border_scores=` parameters to accept precomputed score
  arrays (validated to be 1-D and length `n_neurons`); when omitted it
  recomputes as before, so existing callers are unchanged.

- `population_coverage()` gains a keyword-only `n_jobs` parameter (default `1`)
  to parallelize per-neuron place-field detection via joblib (`-1` uses all
  CPUs). `n_jobs=1` keeps the sequential path with no joblib overhead; results
  are byte-for-byte identical regardless of `n_jobs` (returned data identical;
  per-neuron exclusion warnings are not surfaced under `n_jobs != 1`).

### Changed

- **Default rate-map output changed for gappy / out-of-bounds data.** With the
  new `max_gap` default of `0.5 s`, `compute_spatial_rate` /
  `compute_spatial_rates` (and `decode_session` / `decode_session_summary`,
  which forward `max_gap`) now exclude spikes inside large tracking gaps and
  out-of-bounds excursions from the numerator so it matches the occupancy
  denominator. See the **Fixed** entry "Firing-rate numerator/denominator
  alignment on the FULL interval mask" for the full rationale and the
  `max_gap=None` opt-out.

- `detect_region_crossings` argument order changed to follow the
  behavioral-segmentation convention:
  `(position_bins, times, env, *, region_name, direction=...)`. The old
  positional order `(position_bins, times, region_name, env, ...)` is still
  accepted for one release (transitional dispatch emits a `DeprecationWarning`)
  and will be removed in 0.7.

- Docs/examples now teach `decode_session` as the one-call decode golden path;
  the manual 3-call path (`compute_spatial_rates` → `bin_spikes_in_time` →
  `decode_position`) is kept as an "Advanced: manual three-call path" section.
  `examples/20_bayesian_decoding` now leads with `decode_session`; a new
  "Workflow 2: Bayesian Decoding (one call)" section was added to
  `docs/user-guide/workflows.md` (with a `workflows_decode_session_golden_path`
  CI snippet in `docs/snippets.yml`); and `README.md` points to
  `from neurospatial.decoding import decode_session` as the position-decoding
  entry point.

- Docs (Phase 0.6 sweep): rewrote Workflow 1 in `docs/user-guide/workflows.md` to use the
  canonical `compute_spatial_rate(env, spike_times, times, positions, ...).firing_rate` idiom
  with `simulate_trajectory_ou` + `PlaceCellModel` fixtures, replacing the hand-rolled
  `np.histogram` / `scipy.ndimage.gaussian_filter` approach. Added a CI snippet entry
  (`workflows_place_field_canonical`) to `docs/snippets.yml`.

### Deprecated

All of the following emit a `DeprecationWarning` and are scheduled for removal
in 0.7. Each old name forwards to its replacement with unchanged behavior.

- `EgocentricRatesResult.detect_ovcs` → `EgocentricRatesResult.classify`.
- `ViewRatesResult.detect_view_cells` → `ViewRatesResult.classify`.
- `DirectionalRatesResult.detect_hd_cells` → `DirectionalRatesResult.classify`.
- `SpatialRatesResult.detect_cell_types` → `SpatialRatesResult.label_cell_types`
  (the multi-class string labeler; not folded into `classify`).
- `ViewRateResult.peak_view_location` → `ViewRateResult.peak_location`.
- `ViewRatesResult.peak_view_location` → `ViewRatesResult.peak_locations`.
- `detect_region_crossings` old positional order
  `(position_bins, times, region_name, env, ...)` → new order
  `(position_bins, times, env, *, region_name, ...)`.

### Fixed

- `decode_position_summary` now validates `prior` shape (was silently
  truncating over-long 2-D priors), matching `decode_position`. A 1-D prior
  must be `(n_bins,)` and a 2-D time-varying prior must be exactly
  `(n_time, n_bins)`; an over-long or short 2-D prior, a wrong-length 1-D
  prior, or a non-1-D/2-D prior now raises `ValueError` before streaming
  instead of silently slicing the prior to the decoded time range.

- **Firing-rate numerator/denominator alignment on the FULL interval mask.**
  A firing-rate map is `spike_counts (numerator) / occupancy (denominator)`
  per bin. `env.occupancy` drops a trajectory interval `k` for **three**
  reasons — `dt[k] > max_gap` (large tracking gap), `speed[k] < min_speed`
  (low speed), and `start_bin[k] < 0` (interval's start sample out of the
  active environment) — but the spike binner previously only filtered by the
  time window and (since the speed-filter work) by speed. A spike inside a
  dropped interval (e.g. a 1 s tracking gap, or an out-of-bounds excursion)
  was therefore **counted in the numerator** while occupancy **excluded that
  interval's time from the denominator**, inflating/biasing the rate. The
  spike numerator and the occupancy denominator now drop the **identical** set
  of intervals via one shared `interval_valid_mask` helper (the single source
  of truth that `env.occupancy` also consumes). `compute_spatial_rate` /
  `compute_spatial_rates` gain a keyword-only `max_gap: float | None = 0.5`
  (matching `env.occupancy`'s default, so occupancy behavior is unchanged)
  that gates **both** sides identically.
  - **Behavior change:** rate maps now differ for sessions that contain large
    tracking gaps (intervals longer than `max_gap`) or out-of-bounds samples —
    previously those rates were inflated. This is a correctness fix. Pass
    `max_gap=None` to disable gap gating on **both** sides (restoring the
    pre-fix, no-gap-gating behavior while keeping numerator and denominator
    aligned).
  - **No more silent empty rate maps.** When the interval filter excludes
    **every** trajectory interval (so occupancy is all-zero and the rate map is
    all-NaN/0), `compute_spatial_rate` / `compute_spatial_rates` now emit a
    single `UserWarning` naming the active gate(s) (`max_gap`, `min_speed`,
    and/or the out-of-bounds-start possibility) and suggesting fixes (check
    units; pass `max_gap=None`). Previously only the `min_speed` gate warned —
    a too-large `max_gap` (or a wrong-units one) and the out-of-bounds-start
    rule emptied the map **silently**. Gated by `warn_on_drop=True` (the
    default); fires once per call (not per neuron in the batch path).

- `decode_session` now warns loudly when most spikes fall outside the decode
  time window `[times.min(), times.max()]` instead of silently dropping them.
  Previously, when `encoding_models=` was passed (which skips
  `compute_spatial_rates`), spikes outside the window were silently discarded
  by `bin_spikes_in_time`'s `np.histogram`, so a milliseconds-vs-seconds unit
  mismatch produced an all-zero count matrix and a plausible-but-wrong
  posterior with no warning.  `decode_session` now owns a single
  `UserWarning` (naming the dropped count/total, percentage, decode window,
  spike range, and units hypothesis) that covers **both** the
  `encoding_models`-provided and `encoding_models=None` branches; the
  encoder's now-redundant duplicate is suppressed.  A new keyword-only
  `warn_on_drop: bool = True` parameter silences it for genuinely sparse
  sessions.

- The bare-`Environment()` error now uses a unique code `[E1006]` (was
  `[E1001]`, which collided with "No active bins found") and points at the
  correct docs host (`https://edeno.github.io/neurospatial/`).  Documented in
  `docs/errors.md`.

- `Environment.distance_to(region_name)` now raises `RegionNotFoundError`
  (from `neurospatial._exceptions`) instead of a bare `KeyError` when the
  named region is absent.  `RegionNotFoundError` subclasses `KeyError`, so all
  existing `except KeyError` blocks keep working without change.

- Removed stale `compute_firing_rate(...)` calls (that function does not exist publicly) from
  `docs/user-guide/workflows.md` (Workflow 3) and `docs/user-guide/spatial-analysis.md`;
  replaced with the correct `compute_spatial_rate(env, spike_times, times, positions, ...).firing_rate`
  API and argument order.

- Replaced 11 broken `env.plot(field, ax=...)` calls in `docs/user-guide/spike-field-primitives.md`
  and `docs/user-guide/rl-primitives.md` with `env.plot_field(field, ax=...)`. The `env.plot()`
  method's first positional argument is `ax`, not a field array, so passing a field there was silently
  broken.

- Updated `examples/20_bayesian_decoding.py` (and the jupytext-paired `.ipynb`) to use the batch
  `compute_spatial_rates(env, spike_times_list, times, positions, ...).firing_rates` for encoding
  models and the canonical `bin_spikes_in_time(spike_trains, dt, t_start, t_stop)` helper for
  time-binned spike counts; regenerated the notebook via `jupytext --sync`.

- Bumped stale version strings from `v0.4.0` to `v0.5.0` in `docs/index.md` (status line and
  BibTeX entry) and `README.md` (dependency table header and BibTeX entry).

- `GraphValidationError` messages in `layout/validation.py` and the wrapping
  `ValueError` in `environment/core.py` no longer reference the internal
  developer guide `CLAUDE.md`.  Error text now points to the public issue
  tracker (`https://github.com/edeno/neurospatial/issues`) instead.  A
  permanent repo-wide backstop test (`tests/test_no_internal_doc_refs.py`)
  asserts that no source file under `src/neurospatial/` contains this string,
  preventing the leak from regressing.

- `Environment.__init__` now raises a beginner-grade `ValueError` when called
  without arguments (i.e. bare `Environment()`).  The old message "layout
  parameter is required" gave no actionable guidance; the new message explains
  that `Environment` must be created through a factory method, shows a concrete
  correct example (`Environment.from_samples(data, bin_size=2.0)`), lists the
  other available factories (`from_polygon`, `from_graph`, `from_grid_mask`,
  `from_pixel_mask`), and links to the online docs — matching the style already
  used by `EnvironmentNotFittedError`.  The exception type remains `ValueError`
  so existing `except ValueError` callers are unaffected.

- `align_spikes_to_events` in `events/alignment.py` now rejects `event_times`
  containing `Inf` values with a descriptive `ValueError`, matching the
  existing `spike_times` Inf check and fulfilling the docstring promise.
  Previously an `Inf` event time silently produced an empty-or-wrong result.

- `behavior.trajectory` public functions (`compute_turn_angles`,
  `compute_step_lengths`, `mean_square_displacement`) and
  `behavior.navigation.traveled_path_length` now coerce `positions` (and
  `times` for MSD) with `np.asarray` at the public boundary before any
  `.ndim`/`.shape` access. Passing a plain Python list no longer raises a
  confusing `AttributeError`; valid list-of-lists inputs succeed, and
  malformed inputs raise a descriptive `TypeError` or `ValueError`.

- `bin_spike_train` and `bin_spike_trains` in `encoding/_binning.py` no
  longer silently drop spikes outside the trajectory time window or spikes
  that map to inactive environment bins.  When the dropped fraction exceeds
  50 % (or all spikes are dropped), a `UserWarning` is emitted naming the
  dropped count, total, both time ranges, and a units-hypothesis hint (e.g.
  spike_times in milliseconds vs. times in seconds).  Two separate messages
  cover the two drop causes (time-window and inactive-bin).  The batch path
  (`bin_spike_trains`, `compute_spatial_rates`) warns exactly **once per
  cause** in the main process — never from joblib worker processes where
  warnings are commonly swallowed.  Default behaviour for in-window spikes
  is byte-for-byte unchanged.

### Documentation

- `population_coverage` docstring example now shows the vectorized batch path
  (`compute_spatial_rates(...).firing_rates`) instead of the per-neuron
  `compute_spatial_rate` loop; the pre-existing arg-order bug
  (`population_coverage(firing_rates, env)`) has been corrected to
  `population_coverage(env, firing_rates)`.
- `population_coverage` shape-mismatch `ValueError` message now steers users to
  `compute_spatial_rates` (the batch function) rather than the slow
  `compute_spatial_rate` + `np.stack` recipe.
- `compute_spatial_rate` (singular) docstring now includes a short note pointing
  many-neuron users to `compute_spatial_rates` (plural), which shares occupancy
  and smoothing-kernel work across the whole population.

### Additional release notes

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

- `EgocentricRatesResult.detect_ovcs(...)` -> `classify(...)`

- `ViewRatesResult.detect_view_cells(...)` -> `classify(...)`

- `DirectionalRatesResult.detect_hd_cells(...)` -> `classify(...)`

- `SpatialRatesResult.detect_cell_types(...)` -> `label_cell_types(...)`

- `ViewRateResult.peak_view_location()` -> `peak_location()`

- `ViewRatesResult.peak_view_location()` -> `peak_locations()`

- `detect_region_crossings(position_bins, times, region_name, env, ...)` ->
  `detect_region_crossings(position_bins, times, env, *, region_name, ...)`

## [v0.5.0] - 2026-06-04

## What's Changed

### Features

### Bug Fixes
- fix(io,regions): atomic writes, NWB metadata round-trip, deep immutability (4307973)
- fix(animation): grid artist reuse, frame_times, figure/capture leaks (8f862d3)
- fix(decoding,stats,events): PSTH window, weighted Rayleigh, posterior NaN (301d2b0)
- fix(simulation): RNG independence, exact Poisson, mixed-mode kwargs (f00c048)
- fix(encoding): bearing wrap, occupancy masking, phase-precession optimizer (1525ffd)
- fix(environment,behavior,annotation): invariants, segmentation, leaks (d5c9f34)
- fix(ops,layout): correctness + safety fixes in core primitives (cefe030)
- docs(plan): mark test-suite remediation complete (9f18cc6)
- fix(encoding): recover egocentric distance tuning in binned rate (ef5090a)
- fix(simulation): place uniform field centers in the interior, not on the boundary (9872487)

### Documentation
- docs(plan): mark test-suite remediation complete (9f18cc6)
- docs: address review follow-up on the fix-all changeset (a2ca036)
- test: reword two docstrings to drop implementation-plan references (28105fe)
- docs: refresh UX-facing onboarding docs (4080f49)
- docs: update CHANGELOG.md for v0.4.0 (c03df84)

### Other Changes
- chore(release): 0.5.0 (#4) (d1afe9d)
- chore: remove completed test-suite remediation plan (8a7a04b)
- test(animation): drive visual regression through production renderer (8e5ed03)
- test: dedup shared test helpers into shared modules (review follow-up) (215c971)
- test: reword two docstrings to drop implementation-plan references (28105fe)
- test: tighten thresholds, strengthen sim recovery, hygiene (Phase 7) (fad0dcb)
- test: replace mocks with real-path tests where they hid integration (Phase 8) (ad3d105)
- test(ops,layout,environment): core-primitives behavioral coverage (Phase 5) (e23365d)
- test(nwb): add NWBHDF5IO disk round-trip tests for writers (Phase 6) (628acb2)
- test(animation): verify real cross-backend color-mapping parity (Phase 4) (b72dc61)
- test(decoding,stats,events): add closed-form + analytic-reference correctness tests (Phase 3) (d4f8da4)
- test(encoding): add end-to-end recovery + NumPy/JAX parity tests (Phase 2) (a8b216e)
- test(encoding): add object_vector_score tests; fix OVC convention doc + CLAUDE.md drift (d73cdc8)

**Full Changelog**: https://github.com/edeno/neurospatial/compare/v0.4.0...v0.5.0

## [0.4.0] - 2026-05-26

This release is the v0.4 UX cleanup: a wide-ranging consolidation of
public-API names, argument orders, return types, error semantics, and
example coverage. There are **no deprecations**; every change is a
clean delete-and-replace. Pin to `<0.4.0` if you need the old surface.

### Breaking changes

#### Parameter renames

- **`distance_metric` / `distance_type` / `use_geodesic` → `metric`** across
  all physical-distance APIs. Legal values are `{"euclidean", "geodesic"}`.
  Affects `Environment.distance_to`, `compute_egocentric_rate(s)`,
  `compute_egocentric_distance`, `compute_spatial_rate(s)`,
  `ObjectVectorCellModel`, `PlaceCellModel`, `BoundaryCellModel`, and the
  boundary / border modules.
- **`smoothing_sigma` / `kernel_bandwidth` → `bandwidth`** across smoothing
  APIs (`compute_spatial_rate(s)`, `compute_view_rate(s)`,
  `compute_egocentric_rate(s)`, `Environment.smooth`, KDE helpers).
- **`velocity_threshold` / `speed_threshold` / `threshold` → `min_speed`**
  across velocity-based behaviour segmentation
  (`segment_by_velocity`, `heading_from_velocity`, etc.).
- **Overlay `data=` → semantic name.** `PositionOverlay(data=...)` →
  `PositionOverlay(positions=...)`; `HeadDirectionOverlay(data=...)` →
  `HeadDirectionOverlay(headings=...)`.

#### Result-class field renames

- **`EgocentricRateResult.ego_env` → `env`**;
  **`ViewRateResult.view_occupancy` → `occupancy`**.
- **`PeriEventResult.firing_rate`** is now a cached attribute, not a method.
  Replace `result.firing_rate()` with `result.firing_rate`.
- **`DecodingResult.uncertainty` → `posterior_entropy`** (matches the free
  function in `decoding/estimates.py`).
- **Singular vs plural method/attribute normalization** on result classes
  (single-neuron results use singular methods; batch results use plural).
  Renames: `SpatialRateResult.peak_firing_rates()` → `peak_firing_rate()`,
  `SpatialRateResult.peak_locations()` → `peak_location()`,
  `ViewRateResult.peak_view_locations()` → `peak_view_location()`.
- **`is_X_cell` method names** normalized to match the free-function names
  (`is_object_vector_cell`, `is_spatial_view_cell`,
  `is_head_direction_cell`).

#### Argument-order canonicalization

- **Encoding functions are now `env`-first**, with the canonical order
  `(env, spike_times, times, positions, headings?, object_positions?, *, ...)`.
  Affects `compute_spatial_rate(s)`, `compute_egocentric_rate(s)`,
  `compute_view_rate(s)`, `detect_place_fields`, `is_spatial_view_cell`,
  `is_object_vector_cell`, `is_border_cell`, and friends.

  ```python
  # v0.3
  compute_spatial_rate(spike_times, times, positions, env)
  compute_egocentric_rate(spike_times, times, positions, headings,
                          object_positions, env=env)
  # v0.4
  compute_spatial_rate(env, spike_times, times, positions)
  compute_egocentric_rate(env, spike_times, times, positions,
                          headings, object_positions)
  ```

- **`compute_directional_rate` / `is_head_direction_cell`** keep the
  heading-domain-native `(spike_times, times, headings, *, ...)`
  signature — this is the documented exception to the env-first rule
  (heading is a circular angular variable, not a spatial position).
  See the function docstrings and `CLAUDE.md` "Canonical Argument
  Order".
- **Egocentric ops** `allocentric_to_egocentric` /
  `egocentric_to_allocentric` reorder to `(positions, headings, targets)`.
- **Behavioural segmentation** functions reordered to
  `(position_bins, times, env, *, ...)`.
- **`distance_to_reward`** in `events.regressors` reordered to
  `(env, times, positions, reward_times, ...)`.
- **`fit_isotonic_trajectory` / `fit_linear_trajectory`** reordered to
  `(env, posterior, times, *, ...)` with a standardized `method`
  keyword.
- **`*` keyword-only separator** added consistently across the public
  API. Numerical parameters and verbose flags become keyword-only.

#### Coordinate / convention changes

- **`Environment.is_1d` → `Environment.is_linearized_track`.** Same
  semantics (a 1-D graph track embedded in 2-D world coordinates); the
  new name resolves the historical "is this n_dims==1 or a 2-D track?"
  ambiguity. Serialized environment metadata uses the new key — pre-v0.4
  saved environments will need to be re-saved.
- **`GridProperties.peak_coords` is now `(x_offset, y_offset)`** instead
  of `(row_offset, col_offset)`. Swap `peak_coords[:, 0]` (was row) for
  `peak_coords[:, 1]` (now y) when reading the second component.
- **`simulate_trajectory_ou(speed_units=...)`** is now required (was
  defaulted). Speed defaults switch from m/s to cm/s. Mismatch between
  `speed_units` and `env.units` raises rather than silently rescaling.

#### Removed (no aliases, no deprecation)

- **`Environment.save` / `Environment.load`.** The pickle path is gone.
  Use `Environment.to_file` / `Environment.from_file` (JSON metadata
  plus npz arrays).
- **`Environment.mask_for_region`.** Use `Environment.region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask(image_mask, pixel_size, ...)` and
  `from_grid_mask(active_mask, grid_edges, ...)`.
- **`path_efficiency` (float-returning).** Use `compute_path_efficiency`
  which returns a `PathEfficiencyResult`.
- **Cross-domain re-exports.** Each public symbol now has exactly one
  canonical import path; the top-level `neurospatial` namespace no
  longer re-exports symbols from `encoding`, `decoding`, etc.

### Added

- **`PlaceFieldsResult` dataclass.** `detect_place_fields` returns a
  frozen dataclass with `fields`, `excluded_reason`, and `n_excluded`
  fields. Still iterable / sized / indexable, so existing `for f in
  detect_place_fields(...)` and `len(...)` patterns keep working.
  Closes the "silent drop when mean rate too high" failure mode.
- **`BinSequenceWithRuns` dataclass + new method.**
  `Environment.bin_sequence` always returns an `ndarray`;
  `Environment.bin_sequence_with_runs` returns a dataclass with `bins`,
  `run_starts`, `run_lengths`.
- **`MSDResult` and friends.** Misc result-type cleanup in trajectory
  analysis: `MSDResult`, `SpatialAutocorrelationResult`,
  `PathEfficiencyResult`.
- **`Environment.is_polar` property and `coordinate_kind` attribute.**
  `from_polar_egocentric` sets `coordinate_kind="polar"`. Methods that
  assume Cartesian (`distance_to`, `distance_between`,
  `Environment.contains`, `apply_transform`, `bin_at` on `(x, y)`
  input) raise on polar environments with a clear error.
  `plot_field` switches axis labels and skips the equal-aspect call so
  egocentric polar firing fields still render correctly.
- **Custom exception classes.** `EnvironmentNotFittedError` (already
  existed) now has a free-function variant; added `RegionNotFoundError`,
  `RegionAlreadyExistsError`, and three more in `_exceptions.py`.
- **`Environment.from_pixel_mask` and `Environment.from_grid_mask`
  factories** (replacing `from_image` / `from_mask`).
- **`Environment._state_version` invalidation token.** Cached
  properties verify the version on access; subset / transform / rebin
  bump it, so stale caches are surfaced loudly instead of returning
  silently-wrong results.
- **`Environment.__str__` returns `info()`** for quick inspection.
- **Glossary page** at [docs/glossary.md](https://edeno.github.io/neurospatial/glossary/) defining 14
  core terms. Linked from `docs/getting-started/core-concepts.md` and
  the README.
- **`docs/api/index.md` expansion.** Structured sections for
  `encoding`, `decoding`, `behavior`, `events`, `ops.egocentric`,
  `ops.visibility`, `ops.basis`, `stats`, `animation`, `io.nwb`.
- **`docs/examples/index.md` rewrite.** Goal → notebook table plus
  full per-notebook entries with Time + Prerequisites.
- **Notebooks 24–27.** Object-vector cells, head-direction tuning,
  peri-event PSTH, and NWB loading round-trip.
- **README "Your First Place Field" front-door example.** Canonical
  pattern using `simulate_trajectory_ou`, `PlaceCellModel`,
  `generate_population_spikes`, and `compute_spatial_rate`.
- **CI doc-snippet test.** `scripts/test_doc_snippets.py` plus
  `.github/workflows/test_docs.yml` re-executes a curated manifest of
  doc snippets on every PR.
- **CI notebook regen test.** `.github/workflows/test_notebooks.yml`
  re-executes `11_place_field_analysis.ipynb` per PR to catch silent
  regressions in the example surface.
- **Shared example styling.** `examples/_style.py` Wong / Okabe-Ito
  palette and fixed figure sizes. Wired into every tutorial notebook
  that previously set matplotlib rcParams inline (01-08, 19, 20,
  22-27). Notebooks 09-18 and 21 had no rcParams blocks and
  intentionally keep matplotlib defaults.

### Changed

- **Silent failures replaced with loud failures.**
  - `subset()` round-trip now returns a `MaskedGrid` instead of a one-off
    `subset` layout kind, so the result is fully serializable.
  - `bin_at` vs `map_points_to_bins` standardize on `-1` for
    out-of-environment samples in trajectory contexts.
  - `detect_place_fields` returns a `PlaceFieldsResult` with
    `excluded_reason` set instead of silently returning `[]`.
  - `batch_grid_scores` / `batch_border_scores` use NaN as the explicit
    failure marker and warn once per batch.
  - Fitted-state checks at entry of `compute_spatial_rate(s)`,
    `compute_egocentric_rate(s)`, `compute_view_rate(s)`,
    `decode_position` raise immediately instead of failing deep in the
    call stack.
  - `spike_times` validation rejects unsorted / negative / non-finite
    values with diagnostic messages.
  - `decode_position(validate=True)` is the default; rejects negative
    spike counts and posteriors that don't sum to 1.
- **Canonical exception types** throughout. Manual "not fitted" checks
  migrated to `EnvironmentNotFittedError`; warning-and-overwrite paths
  in `Regions.__setitem__` now raise.
- **Errors carry units and stack context.** Length-mismatch errors
  from `_binning` include a `context` arg so messages say "in
  compute_spatial_rate: ..."; magnitude errors include the offending
  unit.
- **Warning hygiene.** `UserWarning` for data-quality, `RuntimeWarning`
  for numerical fallbacks, `stacklevel=2` everywhere.
- **Production `print()` calls** replaced with module-level
  `logger.info` / `logger.debug`.
- **`Environment.bin_attributes`, `edge_attributes`,
  `differential_operator`** converted from `@cached_property` to
  methods (`get_bin_attributes()`, etc.) so the cost is visible.
- **`Environment.units`** validated against a small registry (`{"cm",
  "m", "mm", "px", None}`) with a `UserWarning` for unknown values.
  Documented as advisory.
- **Heading convention** documented explicitly in every function that
  takes a `headings` argument (allocentric world-frame: 0 = East,
  +π/2 = North; egocentric for OVC tuning: 0 = ahead, +π/2 = left).
- **`events.__init__`** is now eager (was lazy).
- **Bandit-task notebook** prints the download URL and exits cleanly
  when `data/` is missing; CI no longer fails on the example.

### Fixed

- **`repr(env)` `name=None` bug** for empty-string names. Now uses
  `repr(self.name)` so empty strings are visible as `''`.
- **`Environment._state_version` cache invalidation** prevents
  stale-cache reads after mutating operations.
- **Polar environment misuse** is now an error instead of producing
  silently-wrong distances or transforms.

### Removed

- **`Environment.save` / `Environment.load`** (pickle). Replaced by
  `to_file` / `from_file`.
- **`Environment.mask_for_region`.** Use `region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask` / `from_grid_mask`.
- **`path_efficiency` float-returning function.** Use
  `compute_path_efficiency`.
- **All cross-domain re-exports** from top-level `neurospatial`.

### Major feature additions (v0.3.x development cycle)

The following features were developed during the v0.3.x development
line and ship as part of v0.4.0. The names below reflect the final
v0.4 surface — many of these symbols were introduced under earlier
names that were renamed during the M2 consolidation pass (see
**Breaking changes** above for the v0.3 → v0.4 mapping).

#### Added (features)

- **Spatial View Cells**: Firing-rate fields indexed by gaze location
  - `compute_view_rate()` / `compute_view_rates()` — single / batch
  - `ViewRateResult` frozen dataclass (`firing_rate`, `occupancy`,
    `env`, plus `view_spatial_information()`, `is_spatial_view_cell()`,
    `sparsity()`, `selectivity()` methods)
  - `is_spatial_view_cell()` free-function classifier
  - `SpatialViewCellModel` simulation model with three gaze models
    (`fixed_distance`, `ray_cast`, `boundary`)

- **Visibility and Gaze**: Ray-casting visibility for view cells
  - `FieldOfView` frozen dataclass with species presets
    (`FieldOfView.rat()`, `FieldOfView.primate()`)
  - `ViewshedResult` frozen dataclass (visible bins + visibility
    fraction)
  - `compute_viewed_location()` — gaze-directed location projection
  - `compute_viewshed()` / `compute_viewshed_trajectory()` /
    `compute_view_field()` — observer-position visibility analysis
  - `visible_cues()` — line-of-sight check to cue / landmark positions
  - `visibility_occupancy()` — time each bin was visible

- **Object-Vector Cells**: Firing fields in egocentric polar coordinates
  - `compute_egocentric_rate()` / `compute_egocentric_rates()`
  - `EgocentricRateResult` / `EgocentricRatesResult` frozen dataclasses
    (`firing_rate`, `occupancy`, `env`, plus `preferred_distance()`,
    `preferred_direction()`, `egocentric_spatial_information()`,
    `is_object_vector_cell()` methods)
  - `object_vector_score()` — distance × direction selectivity metric
  - `is_object_vector_cell()` free-function classifier
  - `plot_object_vector_tuning()` — polar heatmap visualization
  - `ObjectVectorCellModel` simulation model with von Mises directional
    tuning
  - `ObjectVectorOverlay` for animation with object–animal vectors

- **Egocentric Reference Frames**: Foundation for object-vector and view cells
  - `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()`
  - `allocentric_to_egocentric()` / `egocentric_to_allocentric()` —
    batch frame transforms
  - `compute_egocentric_bearing()` — angle to targets relative to
    heading (egocentric convention: 0 = ahead, +π/2 = left)
  - `compute_egocentric_distance()` — Euclidean and geodesic distance
  - `heading_from_velocity()` / `heading_from_body_orientation()` —
    derive heading from tracking data
  - `Environment.from_polar_egocentric()` — egocentric polar coordinate
    environment with optional circular connectivity

- **3D Transform Support**: Full N-dimensional affine transformation capabilities
  - New `AffineND` class for N-dimensional affine transforms using (N+1)×(N+1) homogeneous matrices
  - `Affine3D` type alias for convenience (equivalent to `AffineND` with n_dims=3)
  - 3D factory functions: `translate_3d()`, `scale_3d()`, `from_rotation_matrix()`
  - Integration with `scipy.spatial.transform.Rotation` for 3D rotations
  - `estimate_transform()` now auto-detects dimensionality (2D or 3D) from input points
  - `apply_transform_to_environment()` supports N-dimensional environments with validation
  - 45 new tests for 3D transforms including scipy integration
  - Backward compatible: `Affine2D` unchanged for existing 2D workflows

- **N-D Probability Mapping**: `alignment.py` now accepts N×N rotation matrices
  - `map_probabilities_to_nearest_target_bin()` supports 3D environments
  - Updated `_transform_source_bin_centers()` validates rotation matrix dimensionality
  - For 2D: Use `get_2d_rotation_matrix(angle_degrees)`
  - For 3D: Use `scipy.spatial.transform.Rotation.as_matrix()`

#### Changed (internal)

- **Internal**: Refactored `environment.py` (5,335 lines) into modular package structure for improved maintainability
  - Split into 11 focused modules: `core.py` (1,023 lines), `factories.py` (630 lines), `queries.py` (897 lines), `trajectory.py` (1,222 lines), `transforms.py` (634 lines), `fields.py` (564 lines), `metrics.py` (469 lines), `regions.py` (398 lines), `serialization.py` (315 lines), `visualization.py` (211 lines), `decorators.py` (77 lines)
  - Implemented mixin pattern: Environment inherits from all functionality mixins
  - No breaking changes - `from neurospatial import Environment` continues to work
  - All 1,076 tests passing (100% success rate)
  - Improved code organization for easier contribution and maintenance
  - Largest module is trajectory.py at 1,222 lines (down from original analysis.py at 2,104 lines)

#### Documentation

- **3D Support**: Updated dimensionality support documentation to reflect 3D transforms availability
  - Updated `docs/dimensionality_support.md` with 3D transform examples and feature matrix
  - Updated `docs/user-guide/alignment.md` with comprehensive 3D transformation examples
  - Added complete 3D alignment workflow example
  - Updated compatibility matrix: ~75% of neurospatial now works in 3D (up from 70%)

## [0.2.0] - 2025-11-04

### Added

#### Environment Operations (Complete Feature Set)

**Core Analysis Operations (P0)**

- `Environment.occupancy()` - Compute time-in-bin from trajectory data with speed filtering, gap handling, and optional kernel smoothing
- `Environment.bin_sequence()` - Convert trajectories to bin sequences with run-length encoding
- `Environment.transitions()` - Compute empirical transition matrices with adjacency filtering and normalization
- `Environment.components()` - Find connected components in environment graph
- `Environment.reachable_from()` - Compute reachable bins via BFS or geodesic distance

**Smoothing & Resampling (P1)**

- `Environment.smooth()` - Apply diffusion kernel smoothing to arbitrary fields
- `Environment.rebin()` - Conservative grid coarsening with mass/mean aggregation (grid-only)
- `Environment.subset()` - Extract subregions by bins, regions, or polygons

**Interpolation & Field Utilities (P2)**

- `Environment.interpolate()` - Evaluate bin-valued fields at continuous points (nearest/linear modes)
- `Environment.occupancy()` linear mode - Ray-grid intersection for accurate boundary handling (grid-only)
- `field_ops.py` module:
  - `normalize_field()` - Normalize to probability distribution
  - `clamp()` - Bound field values
  - `combine_fields()` - Weighted combination (mean/max/min)
  - `divergence()` - KL/JS divergence and cosine distance

**Utilities & Polish (P3)**

- `Environment.region_membership()` - Vectorized bin-to-region containment checks
- `Environment.distance_to()` - Compute distances to target bins or regions (Euclidean/geodesic)
- `Environment.rings()` - K-hop neighborhoods via BFS layers
- `Environment.copy()` - Deep/shallow copying with cache invalidation
- `spatial.map_points_to_bins()` - Enhanced with `max_distance` and `max_distance_factor` thresholds for deterministic boundary decisions

**Diffusion Kernel Infrastructure**

- `kernels.py` module:
  - `compute_diffusion_kernels()` - Matrix-exponential heat kernel on graphs with volume correction
  - `Environment.compute_kernel()` - Convenience wrapper with caching
  - Support for both transition and density normalization modes

**Documentation**

- `docs/user-guide/spatial-analysis.md` - Comprehensive 1,400+ line guide covering all operations with scientific context
- `docs/examples/08_complete_workflow.ipynb` - Enhanced workflow notebook with movement/navigation analysis
- All methods have NumPy-style docstrings with working examples
- "Why This Matters" sections explaining scientific motivation for key operations

### Changed

- **GraphLayout**: Now supports 1D layouts correctly (conditional `angle_2d`, dynamic `dimension_ranges`)
- **KDTree operations**: Now deterministic by default using `tie_break="lowest_index"`
- **All environment operations**: Use `@check_fitted` decorator for consistent state enforcement
- **Input validation**: Comprehensive validation with diagnostic error messages across all operations
- **Caching**: Object identity-based caching for kernels and spatial queries

### Fixed

- GraphLayout `angle_2d` computation for 1D graphs (was unconditionally assuming 2D)
- GraphLayout `dimension_ranges` now correctly handles 1D case
- Disconnected graph handling in connectivity tests
- Hexagonal layout interpolation edge cases

### Testing

- **1067 tests passing** (up from 614 in v0.1.0)
- **0 skipped tests** (eliminated all 12 previous skips)
- Performance benchmarks: occupancy on 100k samples, large transition matrices, kernel computation
- Integration tests: end-to-end workflows, multi-layout compatibility
- Edge case coverage: empty environments, single bins, disconnected graphs

### Internal

- Systematic debugging skill used to eliminate all test skips
- Test-driven development for all features
- Code review and UX review completed
- Pre-commit hooks for code quality

## [0.1.0] - 2025-11-03

### Added

- **CompositeEnvironment API parity**: Added `bins_in_region()`, `mask_for_region()`, `shortest_path()`, `info()`, `save()`, and `load()` methods to CompositeEnvironment for full API compatibility with Environment class
- **KDTree-optimized spatial queries**: CompositeEnvironment.bin_at() now uses KDTree for O(M log N) performance instead of O(N×M) sequential queries (enabled by default via `use_kdtree_query=True`)
- **Structured logging infrastructure**: New `_logging.py` module with NullHandler by default, enabling optional logging for debugging and workflow tracing
- **Centralized numerical constants**: New `_constants.py` module consolidating all magic numbers (tolerances, KDTree parameters, epsilon values) for consistent behavior
- **Comprehensive type validation**: CompositeEnvironment constructor now validates input types with actionable error messages
- **Graph metadata validation**: Added `validate_connectivity_graph()` to enforce required node/edge attributes from layout engines
- **Dimensionality support documentation**: New `docs/dimensionality_support.md` clarifying 1D/2D/3D feature support with compatibility matrix

### Changed

- **Updated alignment module**: Now uses centralized constants (`IDW_MIN_DISTANCE`, `KDTREE_LEAF_SIZE`)
- **Updated regions module**: Uses `POINT_TOLERANCE` constant for consistent geometric comparisons
- **Enhanced error messages**: CompositeEnvironment now provides detailed diagnostics for dimension mismatches and type errors
- **Clarified 2D-only transforms**: Updated `transforms.py` docstring to explicitly state 2D-only status and suggest scipy for 3D

### Fixed

- Removed unused `type: ignore` comment in `regular_grid.py`
- Fixed potential `KeyError` in logging by renaming `name` parameter to `env_name` (avoids conflict with LogRecord reserved field)

### Documentation

- Added comprehensive dimensionality support guide (1D/2D/3D feature matrix)
- Updated CLAUDE.md with latest patterns and requirements
- Added 18 new tests for CompositeEnvironment type validation
- Added 23 new tests for Environment error path coverage
- Added 28 new tests for graph validation

### Internal

- Consolidated duplicate dimension inference code
- All 614 tests passing
- Ruff and mypy checks passing
- Test coverage: 78%

[Unreleased]: https://github.com/edeno/neurospatial/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/edeno/neurospatial/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/edeno/neurospatial/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/edeno/neurospatial/compare/v0.2.0...v0.4.0
[0.2.0]: https://github.com/edeno/neurospatial/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edeno/neurospatial/releases/tag/v0.1.0

### Additional release notes

- Initial release of neurospatial

- Core `Environment` class with factory methods

- Multiple layout engines (regular grid, hexagonal, triangular, graph-based)

- Region support for defining ROIs

- Composite environment functionality

- Alignment and transformation tools

- Comprehensive test suite

- NumPy-style docstrings throughout

- Automatic active bin detection from data samples

- NetworkX-based connectivity graphs

- 1D linearization for track-based experiments

- Spatial queries (bin_at, neighbors, shortest_path, distance_between)

- Visualization with matplotlib

- Morphological operations (dilation, closing, hole filling)
