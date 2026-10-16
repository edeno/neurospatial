# Environment architecture changes

The `Environment` internals were reorganized so that each kind of data has one
owner. Rate maps, occupancy, smoothing and decoding give the same numbers as
before. This page lists what you may need to change in your own code, with the
old call as a comment above the new one. The examples use `positions` and
`times` from your own recording, and `env` built from them:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

rng = np.random.default_rng(0)
positions = rng.uniform(0, 100, size=(2000, 2))  # x, y in cm
times = np.arange(len(positions)) / 30.0  # 30 Hz tracking, in seconds
env = Environment.from_samples(positions, bin_size=4.0, units="cm")
```

## Saving environments: use files, not pickles from older versions

Pickle files of an `Environment` written by an earlier version of neurospatial
cannot be loaded. Pickle still works for sending an environment to worker
processes of the same version (for example with `multiprocessing`), but it is
not meant for storing data.

To keep an environment between sessions, save it with `to_file` (or write it to
an NWB file) and load it back with `from_file`:

<!-- docs-test: run -->
```python
# Before: pickle.dump(env, open("arena.pkl", "wb"))
env.to_file("arena")
env = Environment.from_file("arena")
```

If you only have an old pickle, load it with the neurospatial version that
wrote it, call `env.to_file(...)` there, and then load that file with the
current version.

## Removed import paths

These modules no longer exist. Use the public API instead.

| Removed | Use instead |
| --- | --- |
| `neurospatial.environment._coordinates` | `neurospatial.environment.CartesianSystem`, `EgocentricPolarSystem`, or `env.coordinate_system` |
| `neurospatial.environment._metadata` | `env.name`, `env.units`, `env.frame` |
| `neurospatial.environment._ownership` | nothing: the environment owns its geometry |
| `neurospatial.environment._from_bins` | `Environment.from_bins` |
| `neurospatial.environment.transforms` | `Environment.subset`, `Environment.rebin` |
| `neurospatial.layout._binning`, `_locators`, `_polygon`, ... | `neurospatial.binning` (`Binning`, `RectilinearLocator`, `PolygonCells`, ...) |
| `neurospatial.layout._capabilities` (`RectilinearGridProvider`) | `layout.binning().rectilinear` |
| `neurospatial.environment.trajectory.BinSequenceWithRuns` | `neurospatial.environment.BinSequenceWithRuns` |
| `neurospatial.environment.fields`, `.queries`, `.metrics`, `.trajectory`, `.regions`, `.visualization` | the matching `Environment` method, such as `env.smooth`, `env.neighbors`, `env.boundary_bins`, `env.occupancy`, `env.region_mask` or `env.plot_field` |
| `neurospatial.environment.serialization` | `env.to_file`, `Environment.from_file`, `env.to_dict`, `Environment.from_dict`, `env.to_nwb` (or `neurospatial.io`) |
| `neurospatial.environment.factories`, `.polar`, `.cartesian` | the `Environment` factories, such as `Environment.from_samples` and `Environment.from_polar_egocentric` |
| `neurospatial.environment._HAS_SHAPELY` | nothing: shapely is a required dependency and always available |

The modules that implement `Environment` methods now have private names
(`neurospatial.environment._fields` and so on) and may change without notice.
Call the methods; they return the same results.

<!-- docs-test: run -->
```python
# Before: from neurospatial.environment.trajectory import BinSequenceWithRuns
from neurospatial.environment import BinSequenceWithRuns

runs = env.bin_sequence_with_runs(times, positions)
assert isinstance(runs, BinSequenceWithRuns)
```

`neurospatial.arena` still provides the public arena classes (`Arena`,
`OpenArena`, `MaskArena`, `TrackArena`, `UnknownArena`, `EgocentricSurround`).

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

<!-- docs-test: run -->
```python
import copy

session = env.with_metadata(name="session1")
# Before: session == "session1"
assert session.name == "session1"

# Before: env2 = env.copy()
env2 = env  # nothing can change it, so share it
assert copy.deepcopy(env) is env

rate_maps = {session: "rate maps for session 1"}  # usable as a dict key
```

**No "fitted" state.** An environment always has its bins and connectivity, so
`EnvironmentNotFittedError` (error code E1004), `check_fitted` and the
`neurospatial.environment.decorators` module are removed. Remove `except
EnvironmentNotFittedError` blocks. Calling `Environment()` directly still
raises the E1006 error that points to the factories. Passing an object that is
not an `Environment` to an analysis function raises a `TypeError` that shows
the corrected call.

**Linearization errors.** `to_linear` and `linear_to_nd` raise `TypeError`
(not `AttributeError`) on an environment that is not a linearized track:

<!-- docs-test: run -->
```python
# Before: except AttributeError:
try:
    linear_position = env.to_linear(positions)
except TypeError:
    linear_position = None  # env is an open field, not a track
```

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

<!-- docs-test: run -->
```python
# Before: env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
# Before: env.clear_cache(kernels=True)
# Before: clear_kdtree_cache(env)
env.clear_cache()
```

**Relabelled copies share the cache.** `env.with_metadata(...)` and
`env.with_regions(...)` return environments with the same bins, so they now
share their parent's cache instead of starting empty. Renaming an environment
or replacing its regions keeps the kernels and eigenbases already computed:

<!-- docs-test: run -->
```python
from neurospatial import compute_spatial_rate

spike_times = np.sort(rng.uniform(times[0], times[-1], 300))
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

<!-- docs-test: run -->
```python
polar_env = Environment.from_polar_egocentric(
    distance_range=(0.0, 50.0),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=5.0,
    angle_bin_size=np.pi / 8,
    circular_angle=True,
)
# Before: polar_env.layout_type == "MaskedGrid"
assert polar_env.coordinate_kind == "polar"
assert env.coordinate_kind == "cartesian"
```

A polar environment saved with an earlier version loads unchanged and keeps
the layout name stored in its file.

Whether the angle wraps around at ±π is read only from the environment's
`EgocentricSurround` arena. Files from before arenas were saved
(`Environment-v2` files and NWB environment schema 2.0) do not have one, so a
polar environment stored in them can no longer be read. Load it with the
version that wrote it and save it again, or rebuild it with
`Environment.from_polar_egocentric` and the same ranges, bin sizes and
`circular_angle`, as above.

## Custom layout engines supply `binning()`

Each layout engine now hands its geometry to the environment through one
method, `binning()`, which returns a `Binning`: the bins in the order of the
connectivity graph's nodes, neighbor pairs in the order of its edges, point
lookup, finite-volume faces (or the reason they are missing) and any grid,
track, shape and outline records. All built-in engines implement it, and the
`LayoutEngine` protocol lists it.

If you wrote your own layout engine:

- Implement `binning()` from the engine's built state. Returning only built-in
  records is required, because the environment keeps a copy of the record and
  never calls back into the engine. A subclass of a built-in engine inherits a
  working `binning()`; such subclasses used to be refused and now work
  normally.
- The environment keeps only what `binning()` returns (plus `connectivity`,
  `dimension_ranges` and the grid attributes). It never calls
  `point_to_bin_index` or `finite_volume_geometry`, so to change lookup or
  faces, change what `binning()` returns. A layout whose methods were replaced
  on the instance is no longer refused.
- An external object that has no `binning()` method still needs an installed
  snapshot adapter (see the interoperability guide).
- `neurospatial.layout._capabilities.RectilinearGridProvider` is removed. Read
  the grid record from the engine's binning:

<!-- docs-test: run -->
```python
from neurospatial.layout import RegularGridLayout, create_layout


class CoarseGrid(RegularGridLayout):
    """A regular grid whose subclass adds no geometry of its own."""


engine = CoarseGrid()
engine.build(bin_size=10.0, dimension_ranges=[(0.0, 100.0), (0.0, 100.0)])
grid_env = Environment.from_layout(engine, units="cm")

layout = create_layout(
    "RegularGrid", bin_size=4.0, dimension_ranges=[(0.0, 100.0), (0.0, 100.0)]
)
# Before: RectilinearGridProvider(layout)
grid = layout.binning().rectilinear
```

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

<!-- docs-test: run -->
```python
# Before: env.connectivity.nodes[0]["label"] = "start"  (refused)
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
`with_metadata(name=...)`:

<!-- docs-test: run -->
```python
from neurospatial import Regions
from neurospatial.ops.transforms import translate

west = env.bin_centers[:, 0] < 50.0
# Before: env.subset(bins=west) dropped the regions and the name
west_env = env.subset(bins=west, name="west half").with_regions(Regions())

edges = np.linspace(0.0, 100.0, 21)
square = Environment.from_grid_mask(np.ones((20, 20), dtype=bool), (edges, edges))
# Before: square.rebin(2) was named "<name>_rebinned"
coarse_env = square.rebin(2, name="coarse")
shifted_env = env.apply_transform(translate(-50.0, -50.0), frame="centered")
```

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

<!-- docs-test: run -->
```python
# Before: env.layout_parameters["dimension_ranges"]
ranges = env.dimension_ranges
bin_size = env.layout_parameters["bin_size"]  # scalar choices are still there
```

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
after plotting:

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt

hex_env = Environment.from_samples(positions, bin_size=8.0, layout="Hexagonal")
child = hex_env.subset(bins=hex_env.bin_centers[:, 0] < 50.0)
fig, ax = plt.subplots()
child.plot(ax=ax)
low, high = child.dimension_ranges[0]
ax.set_xlim(low - 1.0, high + 1.0)
```

## Factory arguments

Every `Environment` factory takes its data positionally and every option by
keyword, following the argument order used across neurospatial. Options that a
layout does not use raise an error instead of being ignored.

### `from_samples` and `open_field`

`bin_size` is keyword-only:

<!-- docs-test: run -->
```python
# Before: env = Environment.from_samples(positions, 2.0)
env = Environment.from_samples(positions, bin_size=2.0)
```

Both factories list all their options. `dimension_ranges=` is a named option;
other names that used to be forwarded to the layout engine now raise
`TypeError`. The options that clean up a regular grid's active-bin mask or set
its diagonal connections (`dilate`, `fill_holes`, `close_gaps`,
`add_boundary_bins`, `connect_diagonal_neighbors`) mean nothing on a hexagonal
layout, so passing any of them with `layout="Hexagonal"` raises `ValueError`.
Remove them from the call; leaving them out gives the same hexagons as before:

<!-- docs-test: run -->
```python
# Before: Environment.from_samples(positions, 4.0, layout="Hexagonal", fill_holes=True)
hex_env = Environment.from_samples(positions, bin_size=4.0, layout="Hexagonal")
```

In the same way, `bin_count_threshold`, `dilate`, `fill_holes` and `close_gaps`
decide which bins the samples make active. With `infer_active_bins=False`
every bin in the extent is active, so passing any of them raises `ValueError`
instead of being ignored. A `layout=` other than `"RegularGrid"` or
`"Hexagonal"` raises `ValueError` (it was `NotImplementedError`); build such a
layout with `create_layout` and pass it to `from_layout`.

`open_field` fills holes by default. To keep unvisited interior bins out, pass
`fill_holes=False` (this used to raise a `TypeError`):

<!-- docs-test: run -->
```python
env = Environment.open_field(positions, bin_size=2.0, fill_holes=False)
```

### Track factories

`from_graph` keeps the graph and edge order positional; `bin_size` and
`edge_spacing` are keywords. `edge_spacing` defaults to `0.0`, the same as in
`linear_track` and `maze`, so a track whose edges join end to end needs no
spacing argument:

<!-- docs-test: run -->
```python
import networkx as nx

graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))
graph.add_node(1, pos=(100.0, 0.0))
graph.add_node(2, pos=(100.0, 50.0))
graph.add_edge(0, 1, distance=100.0)
graph.add_edge(1, 2, distance=50.0)
edge_order = [(0, 1), (1, 2)]

# Before: env = Environment.from_graph(graph, edge_order, 0.0, 2.0)
track = Environment.from_graph(graph, edge_order, bin_size=2.0)
# A gap between arms that do not touch is still a choice you make
track = Environment.from_graph(graph, edge_order, bin_size=2.0, edge_spacing=10.0)
```

`maze` names its maze type by keyword:

<!-- docs-test: run -->
```python
nodes = {
    "center": (0.0, 0.0),
    "north": (0.0, 50.0),
    "south": (0.0, -50.0),
    "east": (50.0, 0.0),
    "west": (-50.0, 0.0),
}
# Before: env = Environment.maze("plus", node_positions=nodes, bin_size=2.0)
plus_maze = Environment.maze(kind="plus", node_positions=nodes, bin_size=2.0)
```

`from_graph`, `linear_track` and `maze` now accept `regions=` like the other
factories.

### Polygon, grid-mask and pixel-mask factories

The bin size of a polygon and the pixel size of an image mask are keywords; the
polygon, the mask and the grid edges stay positional:

<!-- docs-test: run -->
```python
from shapely.geometry import box

arena_polygon = box(0.0, 0.0, 100.0, 100.0)
image_mask = np.ones((40, 60), dtype=bool)
active_mask = np.ones((10, 10), dtype=bool)
grid_edges = (np.linspace(0.0, 100.0, 11), np.linspace(0.0, 100.0, 11))

# Before: Environment.from_polygon(arena_polygon, 2.0)
polygon_env = Environment.from_polygon(arena_polygon, bin_size=2.0)
# Before: Environment.from_pixel_mask(image_mask, 0.5)
image_env = Environment.from_pixel_mask(image_mask, pixel_size=0.5)
mask_env = Environment.from_grid_mask(active_mask, grid_edges)  # unchanged
```

All three accept `regions=`, so every Cartesian factory takes the same `name=`,
`units=`, `frame=` and `regions=` keywords.

### Egocentric polar factory

`from_polar_egocentric` has no data arguments, so every argument is a keyword.
Swapping the distance and angle bin sizes used to build a valid but wrong
environment without an error:

<!-- docs-test: run -->
```python
# Before: Environment.from_polar_egocentric((0, 50), (-np.pi, np.pi), 5.0, np.pi / 8)
polar_env = Environment.from_polar_egocentric(
    distance_range=(0, 50),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=5.0,
    angle_bin_size=np.pi / 8,
)
```

### `from_layout` takes a built layout engine

`from_layout` used to take a layout name and a dictionary of build parameters,
which repeated what `from_samples`, `from_polygon`, `from_grid_mask`,
`from_pixel_mask` and `from_graph` already do with named, checked arguments.
It now takes a layout engine you have built, so its job is the layouts that
have no dedicated factory: a triangular mesh, a grid or hexagonal tiling over
explicit ranges with no position samples, or an engine you wrote (see
[custom layout engines](#custom-layout-engines-supply-binning)).

<!-- docs-test: run -->
```python
from shapely.geometry import box

from neurospatial.layout import create_layout

arena = box(0.0, 0.0, 50.0, 50.0)
# Before:
# env = Environment.from_layout(
#     "TriangularMesh", {"boundary_polygon": arena, "point_spacing": 2.5}
# )
mesh = create_layout("TriangularMesh", boundary_polygon=arena, point_spacing=2.5)
mesh_env = Environment.from_layout(mesh, units="cm")
```

The environment copies the bins from the engine's `binning()`, so changing the
engine afterwards does not change the environment. `env.layout_parameters`
comes from the engine's record of its `build(...)` arguments, including the
defaults it resolved, for example `connect_diagonal_neighbors=True` for a
masked grid. Input data is never recorded: a hexagonal tiling built from
ranges records `hexagon_width`, `infer_active_bins` and `bin_count_threshold`
but no `positions`, and a triangular mesh records only `point_spacing`, with
or without holes in its boundary.

### NWB: loading versus building

`Environment.from_nwb` used to do two different things depending on its
arguments. It now only loads an environment that was saved in the file, and
names the record with `name=`, as `env.to_nwb(nwbfile, name)` does. Building an
environment from the file's Position data is
`neurospatial.io.nwb.environment_from_position`, which takes the same binning
options as `Environment.from_samples`, with `bin_size` as a keyword:

<!-- docs-test: skip needs an NWB file and the nwb extra -->
```python
from neurospatial.io.nwb import environment_from_position

# Before: env = Environment.from_nwb(nwbfile, scratch_name="linear_track")
env = Environment.from_nwb(nwbfile, name="linear_track")
# Before: env = Environment.from_nwb(nwbfile, bin_size=5.0, units="cm")
# Before: env = environment_from_position(nwbfile, 5.0)
env = environment_from_position(nwbfile, bin_size=5.0, units="cm")
```
