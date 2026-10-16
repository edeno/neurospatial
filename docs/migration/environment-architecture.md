# Environment architecture changes

The `Environment` internals were reorganized so that each kind of data has one
owner. Analyses you run with `Environment` do not change. This page lists what
you may need to change in your own code.

## Saving environments: use files, not pickles from older versions

Pickle files of an `Environment` written by an earlier version of neurospatial
cannot be loaded. Pickle still works for sending an environment to worker
processes of the same version (for example with `multiprocessing`), but it is
not meant for storing data.

To keep an environment between sessions, save it with `to_file` (or write it to
an NWB file) and load it back with `from_file`:

```python
env.to_file("arena")
env = Environment.from_file("arena")
```

If you only have an old pickle, load it with the neurospatial version that
wrote it, call `env.to_file(...)` there, and then load that file with the
current version.

## Removed import paths

These private modules no longer exist. Import from the real location, or use the
public API.

| Removed | Use instead |
| --- | --- |
| `neurospatial.environment._coordinates` | `neurospatial.geometry._coordinates` (or `Environment.coordinate_system`) |
| `neurospatial.environment._metadata` | `Environment.name`, `.units`, `.frame` |
| `neurospatial.environment._ownership` | `neurospatial.geometry._ownership` |
| `neurospatial.environment._from_bins` | `Environment.from_bins` |
| `neurospatial.environment.transforms` | `Environment.subset`, `Environment.rebin` |
| `neurospatial.layout._binning`, `_locators`, `_polygon`, ... | `neurospatial.geometry` (same module names) |
| `neurospatial.layout._capabilities` (`RectilinearGridProvider`) | `layout.binning().rectilinear` |
| `neurospatial.environment.trajectory.BinSequenceWithRuns` | `neurospatial.environment.BinSequenceWithRuns` |
| `neurospatial.environment.fields`, `.queries`, `.metrics`, `.trajectory`, `.regions`, `.visualization` | the matching `Environment` method, such as `env.smooth`, `env.neighbors`, `env.boundary_bins`, `env.occupancy`, `env.region_mask` or `env.plot_field` |
| `neurospatial.environment.serialization` | `env.to_file`, `Environment.from_file`, `env.to_dict`, `Environment.from_dict`, `env.to_nwb` (or `neurospatial.io`) |
| `neurospatial.environment.factories`, `.polar`, `.cartesian` | the `Environment` factories, such as `Environment.from_samples` and `Environment.from_polar_egocentric` |

The modules that implement `Environment` methods now have private names
(`neurospatial.environment._fields` and so on) and may change without notice.
Use the methods; the result they return is the same.

`neurospatial.arena` still provides the public arena classes (`Arena`,
`OpenArena`, `MaskArena`, `TrackArena`, `UnknownArena`, `EgocentricSurround`).

## Shapely is required

`neurospatial.environment._HAS_SHAPELY` is removed. Shapely is a required
dependency and is always available.

## Environments are immutable values

An `Environment` cannot be changed after it is built, so it now behaves like a
plain value.

**Equality.** Two environments are equal only when they are the same object.
`env == "session1"` (comparing with the name) no longer works; compare
`env.name == "session1"`. `env.with_metadata(name="other")` returns a new
environment, which does not equal the original. Environments can be put in sets
and used as dictionary keys.

**Copying.** `Environment.copy()` is removed because there is nothing to copy.
`copy.copy(env)` and `copy.deepcopy(env)` return the same object.

| Before | Now |
| --- | --- |
| `env2 = env.copy()` | `env2 = env` (or `env.with_metadata()` for a separate object) |
| `env.copy()` then `with_regions(...)` | `env.with_regions(...)` |

**No "fitted" state.** An environment always has its bins and connectivity, so
`EnvironmentNotFittedError` (error code E1004), `check_fitted` and the
`neurospatial.environment.decorators` module are removed. Calling
`Environment()` directly still raises the E1006 error that points to the
factories. Passing an object that is not an `Environment` to an analysis
function raises a `TypeError` that shows the corrected call.

**Linearization errors.** `to_linear` and `linear_to_nd` raise `TypeError`
(not `AttributeError`) on an environment that is not a linearized track.

## One analysis cache, shared by relabelled copies

An environment computes some analyses once and keeps them: the KD-tree behind
`nearest_bin`, diffusion and Gaussian smoothing kernels, the eigenbases used by
`smooth`, `diffuse` and `get_mrf_basis`, the gradient and divergence
operators, boundary bins and the bin and edge attribute tables. They now live
in one cache per environment.

**`clear_cache()` takes no arguments.** It releases every cached analysis. The
`kdtree=`, `kernels=` and `cached_properties=` options are removed, together
with `neurospatial.ops.binning.clear_kdtree_cache`. Clearing only frees
memory: the next analysis computes what it needs again and gives the same
result.

| Before | Now |
| --- | --- |
| `env.clear_cache(kdtree=True, kernels=False, cached_properties=False)` | `env.clear_cache()` |
| `env.clear_cache(kernels=True)` | `env.clear_cache()` |
| `clear_kdtree_cache(env)` | `env.clear_cache()` |

**Relabelled copies share the cache.** `env.with_metadata(...)` and
`env.with_regions(...)` return environments with the same bins, so they now
share their parent's cache instead of starting empty. Renaming an environment
or replacing its regions keeps the kernels and eigenbases already computed:

```python
rates = compute_spatial_rate(env, spike_times, times, positions)
renamed = env.with_metadata(name="session 2")
rates = compute_spatial_rate(renamed, spike_times, times, positions)  # reuses the cache
```

Calling `clear_cache()` on one of them clears it for all of them. To get an
environment with its own empty cache, derive one (`subset`, `clip`, `rebin`,
`apply_transform`, `combine`) or load one from a file, dict, NWB or pickle.

**Threads.** Threads can share one environment: each cached analysis is
computed once even when several threads ask for it at the same time.
`smooth`, `diffuse` and `get_mrf_basis` calls on one environment and its
relabelled copies run one at a time, because each holds the eigenbasis lock
for the whole call. To smooth in parallel (for example one neuron per
worker), use worker processes; each unpickles its own environment with an
empty cache.

## Polar environments report their own layout type

`Environment.from_polar_egocentric(...)` environments now report
`env.layout_type == "Polar"` (it was `"MaskedGrid"`), and `print(env)` shows
`Polar`. The bins are sectors in distance and egocentric angle, so the old
name suggested a Cartesian grid. To test whether an environment is polar, use
the coordinate kind rather than the layout name:

```python
if env.coordinate_kind == "polar":
    ...
```

A polar environment saved with an earlier version loads unchanged and keeps
the layout name stored in its file.

Whether the angle wraps around at ±π is read only from the environment's
`EgocentricSurround` arena. Files from before arenas were saved
(`Environment-v2` files and NWB environment schema 2.0) do not have one, so a
polar environment stored in them can no longer be read. Load it with the
version that wrote it and save it again, or rebuild it with
`Environment.from_polar_egocentric(..., circular_angle=...)`.

## Custom layout engines supply `binning()`

Each layout engine now hands its geometry to the environment through one
method, `binning()`, which returns a `Binning`: the bins in connectivity edge
order, point lookup, finite-volume faces (or the reason they are missing) and
any grid, track, shape and outline records. All built-in engines implement it,
and the `LayoutEngine` protocol lists it.

If you wrote your own layout engine:

- Implement `binning()` from the engine's built state. Returning only built-in
  records is required, because the environment keeps a copy of the record and
  never calls back into the engine. A subclass of a built-in engine inherits a
  working `binning()`; such subclasses used to be refused and now publish
  normally.
- The environment keeps only what `binning()` returns (plus `connectivity`,
  `dimension_ranges` and the grid attributes). It never calls
  `point_to_bin_index` or `finite_volume_geometry`, so to change lookup or
  faces, change what `binning()` returns. A layout whose methods were replaced
  on the instance is no longer refused.
- An external object that has no `binning()` method still needs an installed
  snapshot adapter (see the interoperability guide).
- `neurospatial.layout._capabilities.RectilinearGridProvider` is removed. Read
  the grid record from `layout.binning().rectilinear`.

## Graph attributes are immutable values

`env.connectivity` was already read-only. Its graph, node and edge attribute
values (including `env.connectivity.graph`) are now stored as immutable
values, so reading them no longer makes a copy:

| Attribute value you stored | What you read from `env.connectivity` |
| --- | --- |
| list, e.g. `["measured", 7]` | tuple, `("measured", 7)` |
| dict | read-only mapping |
| set | frozenset |
| NumPy array | read-only array |

`env.to_networkx()` returns an independent graph with the original types
(lists, dicts, sets and writable arrays), so edit attributes there:

```python
graph = env.to_networkx()
graph.nodes[0]["label"] = "start"
```

## Derived environments keep their labels

`subset`, `clip` and `rebin` describe the same arena in the same coordinates,
so they keep the parent's name, units, frame, arena and named regions.
`apply_transform` keeps the name and units, moves the regions and arena with
the bins, and sets the frame only from the `frame=` you pass, because the new
coordinates are no longer in the parent's frame.

| Call | Before | Now |
| --- | --- | --- |
| `env.subset(...)` | name `""`, regions dropped | name and regions kept |
| `env.clip(polygon)` | regions dropped | regions kept |
| `env.rebin(...)` | name `"<name>_rebinned"` | name kept |
| `env.apply_transform(T)` | name `"<name>_transformed"`, frame `"<frame>_transformed"` (or `"transformed"`) | name kept, frame `None` |
| `env.apply_transform(T, frame="session2")` | not available | frame `"session2"` |

Regions are annotations and may extend beyond the retained bins. To drop them,
call `child.with_regions(Regions())`. To rename a derived environment, pass
`name=` to `subset`, `clip`, `rebin` or `apply_transform`, or call
`with_metadata(name=...)`.

## `layout_parameters` keeps choices, not data

`env.layout_parameters` used to keep every argument given to the layout
engine, including the full position array passed to `from_samples`. Saved
files therefore grew with the length of the recording. It now keeps only the
scalar choices that describe how the bins were built, such as `bin_size`,
`connect_diagonal_neighbors`, `pixel_size` or `circular_angle`. Lists of
numbers come back as tuples.

| Value you used to read | Read it from |
| --- | --- |
| `layout_parameters["graph_definition"]` | `env.binning.linearization.graph` |
| `layout_parameters["edge_order"]` | `env.binning.linearization.edge_order` |
| `layout_parameters["active_mask"]`, `["grid_edges"]` | `env.active_mask`, `env.grid_edges` |
| `layout_parameters["dimension_ranges"]` | `env.dimension_ranges` |
| `layout_parameters["polygon"]`, `["boundary_polygon"]` | `env.arena` (for example `env.arena.polygonal_domain()`) |
| `layout_parameters["positions"]` | keep your own position array |

The parameters never affected any analysis, and loading a saved environment
does not need them: the bins, graph and arena are stored directly.

## Subsets report the extent of the bins they keep

`env.subset(...)` sets `dimension_ranges` to the extent of the bins it keeps,
with no margin. Plot limits follow `dimension_ranges`, so plots of a subset now
fit its kept cells instead of leaving extra space around them.

| Layout of the parent | Before | Now |
| --- | --- | --- |
| Hexagonal | kept hexagons' extent plus 5% on each side | kept hexagons' extent |
| Triangular mesh | kept triangles' extent plus 5% on each side | kept triangles' extent |
| Clipped polygon cells (`env.clip(...)`) | kept polygons' extent plus 5% on each side | kept polygons' extent |
| Rectangular grid (including `from_polygon`, masks and images) | the grid's extent | unchanged |
| Track or other bins without cell shapes | kept bin centers' extent | unchanged |

A mesh subset bounds only the triangles it keeps, so a subset of one corner of
an arena no longer reports the extent of the whole arena. The same rule applies
when an `arena=` passed to `from_layout`, `from_grid_mask` or `from_pixel_mask`
removes bins. `Environment.from_bins` bounds triangular-mesh bins by their
triangles' vertices rather than every stored mesh point. Grid subsets keep the
full grid's extent because they keep its whole lattice, matching
`env.grid_edges`.

If you want space around a subset's cells in a figure, widen the axes limits
after plotting, for example
`(low, high) = child.dimension_ranges[0]; ax.set_xlim(low - 1, high + 1)`.

