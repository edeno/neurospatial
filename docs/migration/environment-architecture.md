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
`env == "session1"` used to compare the name; it now returns `False` without an
error, so `if env == "session1":` silently takes the other branch. Search your
code for `== "` next to an environment and compare the name instead:
`env.name == "session1"`. `env.with_metadata(name="other")` returns a new
environment, which does not equal the original. Environments can be put in sets
and used as dictionary keys.

**Copying.** `Environment.copy()` is removed because there is nothing to copy.
`copy.copy(env)` and `copy.deepcopy(env)` return the same object.

<!-- docs-test: run -->
```python
import copy

session = env.with_metadata(name="session1")
# Before: session == "session1"  (now always False)
assert session != "session1"
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
relabelled copies run one at a time, because each holds a lock on the shared
cache for the whole call. To smooth in parallel (for example one neuron per
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
lookup, the faces between cells (or the reason they are missing) and any grid,
track, shape and outline records. All built-in engines implement it, and the
`LayoutEngine` protocol lists it.

If you wrote your own layout engine:

- Implement `binning()` from the engine's built state. Returning only built-in
  records is required, because the environment keeps a copy of the record and
  never calls back into the engine. A subclass of a built-in engine inherits a
  working `binning()`; such subclasses used to be refused and now work
  normally.
- The environment keeps only what `binning()` returns. It derives the
  connectivity graph, `grid_edges`, `grid_shape`, `active_mask` and
  `dimension_ranges` from the `Binning`, so an engine needs no other
  attribute and attributes added to an engine's own graph are not kept. Set
  `Binning(..., dimension_ranges=...)` when the cells do not show the range
  the bins cover. It never calls `point_to_bin_index`, so to change lookup or
  faces, change what `binning()` returns. A layout whose methods were
  replaced on the instance is no longer refused.
- An engine may also have a `layout_type` category string, and build
  arguments recorded with `neurospatial.layout.base.capture_build_params`
  become `env.layout_parameters`. Both are optional labels.
- An external object that has no `binning()` method still needs an installed
  snapshot adapter (see the
  [interoperability guide](../user-guide/interoperability.md)).
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

An engine does not need to subclass anything. This one places bins at
recorded points and finds the nearest one for each position:

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.binning import Binning, MissingFaces, NearestBinLocator, SpatialBins


class RecordedPoints:
    """Bins at recorded reward-port positions, joined in a ring."""

    def build(self, *, points):
        points = np.asarray(points, dtype=float)
        ring = np.arange(len(points))
        pairs = np.column_stack([ring, np.roll(ring, -1)])
        lengths = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
        self._binning = Binning(
            "RecordedPoints",
            SpatialBins.from_arrays(points, np.ones(len(points)), 2, pairs, lengths),
            NearestBinLocator(points, max_distance=5.0),
            MissingFaces("Recorded points have no cells."),
            dimension_ranges=((0.0, 100.0), (0.0, 100.0)),
        )

    def binning(self):
        return self._binning


ports = RecordedPoints()
ports.build(points=[[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]])
ports_env = Environment.from_layout(ports, units="cm")
assert ports_env.bin_at([[11.0, 9.0], [50.0, 50.0]]).tolist() == [0, -1]
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
west_env = env.subset(mask=west, name="west half").with_regions(Regions())

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
child = hex_env.subset(mask=hex_env.bin_centers[:, 0] < 50.0)
fig, ax = plt.subplots()
child.plot(ax=ax)
low, high = child.dimension_ranges[0]
ax.set_xlim(low - 1.0, high + 1.0)
```

## Factory arguments

Every `Environment` factory takes its data positionally and every option by
keyword, following the argument order used across neurospatial. An option
that the layout would ignore raises an error when you change it from its
default, instead of being ignored.

An old call that passes an option by position raises a `TypeError` that names
the keyword to use and shows the corrected call, so the error tells you how to
update each line:

<!-- docs-test: run -->
```python
try:
    Environment.from_samples(positions, 2.0)
except TypeError as error:
    assert "pass bin_size=2.0 by keyword" in str(error)
```

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
layout, so setting any of them to a value other than its default with
`layout="Hexagonal"` raises `ValueError` (it used to be ignored). Remove
those values from the call; leaving them out gives the same hexagons as
before. The defaults themselves are accepted, and so is `fill_holes=False` in
`open_field` (whose default is `True`), because a hexagonal layout fills no
holes. A parameter set that sweeps `layout` over `["RegularGrid", "Hexagonal"]`
can therefore spell the options out:

<!-- docs-test: run -->
```python
# Before: Environment.from_samples(positions, 4.0, layout="Hexagonal", fill_holes=True)
hex_env = Environment.from_samples(positions, bin_size=4.0, layout="Hexagonal")

grid_options = {"dilate": False, "fill_holes": False, "connect_diagonal_neighbors": True}
for layout in ["RegularGrid", "Hexagonal"]:
    Environment.from_samples(positions, bin_size=4.0, layout=layout, **grid_options)
```

In the same way, `bin_count_threshold`, `dilate`, `fill_holes` and `close_gaps`
decide which bins the samples make active. With `infer_active_bins=False`
every bin in the extent is active, so changing any of them from its default
raises `ValueError` instead of being ignored. A `layout=` other than `"RegularGrid"` or
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

The old calls tell you what to use instead.
`Environment.from_nwb(nwbfile, bin_size=5.0)` raises a `TypeError` that shows
the `environment_from_position(nwbfile, bin_size=5.0)` call, and
`Environment.from_nwb(nwbfile)` on a file with no saved environment raises a
`KeyError` that lists the environments the file does store and, when the file
has Position data, names it and points to `environment_from_position`.

## Which bins a region covers

Every region query uses one rule. A point region, such as a reward port,
covers the bin that contains the point. A polygon region, such as a reward
zone or a maze arm, covers the bins whose centers lie inside it or exactly on
its edge. `bins_in_region`, `region_mask`, `region_membership`,
`distance_to` with a region, `subset(region_names=...)`,
`subset(polygon=...)` and `regions_to_mask` all follow it, so the bins you
count time in are the bins you measure distance to.

| Call | Before | Now |
| --- | --- | --- |
| `env.region_membership()`, point region | no bins | the bin containing the point |
| `regions_to_mask(env, "reward")`, point region | no bins and a warning | the bin containing the point |
| `env.distance_to("reward")`, point region | warning, then `ValueError` | distance to the bin containing the point |
| `env.subset(region_names=["reward"])`, point region | `ValueError` | keeps the bin containing the point |
| `env.bins_in_region("zone")`, centers on the polygon's edge | left out | included |
| `env.subset(polygon=...)` and `subset(region_names=...)`, centers on the edge | left out | included |

A bin center lies exactly on a polygon's edge when the polygon is drawn on the
bin grid, for example `box(0, 0, 50, 100)` on a grid with a center at
`x = 50`. Analyses that use `bins_in_region` then see those edge bins too:
decision-region entry times (`decision_region_entry_time`,
`compute_decision_analysis`, `compute_vte_session`), the goal bins and
decision boundary of `compute_decision_analysis`, `trials_to_region_arrays`
and `distance_to_boundary(region_name=...)`. To keep only the interior, pass
`include_boundary=False` to `bins_in_region`, `region_mask`,
`region_membership` or `subset`:

<!-- docs-test: run -->
```python
from shapely.geometry import box

from neurospatial import Regions

edges = np.linspace(0.0, 100.0, 11)  # bin centers at 5, 15, ..., 95 cm
grid = Environment.from_grid_mask(np.ones((10, 10), dtype=bool), (edges, edges))
regions = Regions()
regions.add("reward", point=(52.0, 48.0))
regions.add("west", polygon=box(0.0, 0.0, 55.0, 100.0))  # edge on x = 55
grid = grid.with_regions(regions)

reward_bins = grid.bins_in_region("reward")  # the bin containing the point
assert np.array_equal(np.flatnonzero(grid.region_membership()[:, 0]), reward_bins)
# Before: grid.bins_in_region("west") left out the column of centers at x = 55
west_bins = grid.bins_in_region("west")
west_interior = grid.bins_in_region("west", include_boundary=False)
assert len(west_bins) == len(west_interior) + 10
```

## Query targets are keywords: `bins=`, `points=` or `region=`

A bin index and a coordinate look the same as numbers, so the graph and
distance queries no longer guess which one you meant. Name the target with a
keyword: `bins=` for bin indices, `points=` for coordinates and, for
`distance_to`, `region=` for a named region. A point means the bin that
contains it, so the distance to a reward port at `(82, 77)` is the distance to
the bin that contains the port.

<!-- docs-test: run -->
```python
positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))  # cm
env = Environment.from_samples(positions, bin_size=4.0, units="cm")
reward_port = np.array([[82.0, 77.0]])  # cm, one row per point
reward_bin = int(env.bin_at(reward_port)[0])
start_bin = int(env.bin_at(positions[:1])[0])

# Before: env.distance_to([reward_bin])
to_reward = env.distance_to(bins=reward_bin)
assert np.array_equal(to_reward, env.distance_to(points=reward_port))
# Before: env.neighbors(reward_bin)
around_reward = env.neighbors(bins=reward_bin)
# Before: env.path_between(start_bin, reward_bin)
route = env.path_between(bins=[start_bin, reward_bin])
# Before: env.distance_between(positions[0], reward_port[0])
route_cm = env.distance_between(points=[positions[0], reward_port[0]])
# Before: env.reachable_from(reward_bin, radius=20.0, metric="geodesic")
near_reward = env.reachable_from(bins=reward_bin, radius=20.0, metric="geodesic")
# Before: env.rings(center_bin=reward_bin, hops=2)
rings = env.rings(bins=reward_bin, hops=2)
```

With a named region, pass `region=`:

<!-- docs-test: run -->
```python
from neurospatial import Regions

positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))  # cm
env = Environment.from_samples(positions, bin_size=4.0, units="cm")
regions = Regions()
regions.add("reward", point=(82.0, 77.0))
env = env.with_regions(regions)
# Before: env.distance_to("reward")
to_reward = env.distance_to(region="reward")
assert np.array_equal(to_reward, env.distance_to(points=[[82.0, 77.0]]))
```

What the old guesses did, and what happens now:

| Call | Before | Now |
| --- | --- | --- |
| `env.distance_to([50.7, 12.3])` | distance to bins 50 and 12 | use `points=[[50.7, 12.3]]`; floats in `bins=` raise `TypeError` |
| `env.neighbors([0, 1])` | neighbors of the point (0, 1) | use `bins=0`, or `points=[[0, 1]]` |
| `env.distance_to(3)` | `TypeError: len() of unsized object` | `env.distance_to(bins=3)` works |
| `env.distance_between(p, far_point)` | `inf` | `ValueError` naming the point outside the environment |
| `polar_env.distance_between(...)` | `CartesianOnlyError` | graph path length between two bins or (distance, angle) points |
| `env.bin_center_of(np.array([10.0, 10.0]))` | centers of bin 10, twice | `TypeError` pointing to `env.bin_at` |
| `distance_to_region(position_bins, env, target_bins=50.7)` | distance to bin 50 | `TypeError`; pass a bin index, for example `int(env.bin_at(goal)[0])` |
| `shortest_path_length(env, start_outside, goal)` | distance from the last bin | `ValueError` naming the point outside |

Other refusals that used to give a number or a different error:

- `env.bin_center_of` takes a 1-D array of bin indices; a 2-D index array
  raises instead of returning an array of shape `(a, b, n_dims)`.
- A polygon region on an environment whose bins are not 2-D raises
  `ValueError` (it was `NotImplementedError`) in `region_membership` and
  `regions_to_mask`.
- `compute_path_efficiency` and `subgoal_efficiency` with
  `metric="geodesic"` raise `ValueError` when the trajectory starts outside
  the environment, instead of measuring from the last bin.
- `geodesic_voronoi_labels` and `distance_to_decision_boundary` raise
  `ValueError` for an empty goal list; `distance_to_decision_boundary` used to
  return `inf`.

`points=` always has shape `(n_points, n_dims)`: wrap one point as
`[[x, y]]`. Points with NaN coordinates or outside the environment raise a
`ValueError` that names the first such row, instead of mapping to `-1`. Queries
that start from one bin (`neighbors`, `reachable_from`, `rings`) take one bin
or one point; `path_between` and `distance_between` take two, the start and the
end.

`subset` selects bins with a boolean mask, so its keyword is `mask=`; `bins=`
always means integer bin indices:

<!-- docs-test: run -->
```python
west = env.bin_centers[:, 0] < 50.0  # one True/False per bin
# Before: env.subset(bins=west)
west_env = env.subset(mask=west)
# From bin indices, build the mask first
north_bins = np.flatnonzero(env.bin_centers[:, 1] > 50.0)
keep = np.zeros(env.n_bins, dtype=bool)
keep[north_bins] = True
north_env = env.subset(mask=keep)
```

## Missing data: NaN is a missing value, never zero

A NaN position sample is a missing sample in every per-sample query: it gets
bin `-1` from `bin_at` and `nearest_bin`, `False` from `contains` and NaN from
`interpolate`, and adds no occupancy time and no transition. `nearest_bin` and
`interpolate` used to raise for it, and `transitions(times=..., positions=...)`
raised for any sample outside the environment. A NaN bin of a field is a bin
with no value: `smooth` and `diffuse` with `mode="average"` average over the
observed bins and fill holes from them, where `smooth` used to refuse the field
and `diffuse` turned every bin NaN. A bin stays NaN when its share of smoothing
weight on observed bins is at most 0.1% of the largest share in its connected
part, for example in a room with no visits. `mode="transition"` and
`mode="density"` refuse NaN, because a missing bin has no defensible mass:
fill it explicitly, or use `mode="average"` if the field is a rate. Fields
without NaN give the same values as before.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

rng = np.random.default_rng(0)
positions = rng.uniform(0, 100, size=(2000, 2))  # cm
env = Environment.from_samples(positions, bin_size=4.0, units="cm")
rate = np.where(env.bin_centers[:, 0] < 70.0, 5.0, np.nan)  # Hz, east unvisited
# Before: env.smooth(rate, 8.0) raised "Field contains NaN values"
smoothed = env.smooth(rate, 8.0)
assert np.allclose(smoothed[np.isfinite(smoothed)], 5.0)
# Before: env.nearest_bin([[np.nan, 10.0]]) raised a SciPy error
assert env.nearest_bin(np.array([[np.nan, 10.0]]))[0] == -1
```

`occupancy`, `bin_sequence`, `bin_sequence_with_runs` and trajectory
`transitions` now warn, like `compute_spatial_rate`, when `max_gap` and the
other interval gates exclude every interval, or when the median sampling
interval is longer than `max_gap`. Timestamps in milliseconds trigger it: each
33 ms frame looks like a 33 s recording gap. Convert `times` to seconds.

## Cartesian-only methods on polar environments

An egocentric polar environment stores `(distance, angle)` bins around the
animal. Methods that read positions as Cartesian `(x, y)`, such as `bin_at`,
`contains`, `occupancy` and `subset`, still refuse it, but the refusal now
happens when you call the method, not when you look it up:

- `hasattr(polar_env, "bin_at")` is `True` and `dir(polar_env)` lists the
  method. Calling it raises `CartesianOnlyError`.
- `CartesianOnlyError` is a `TypeError`, and it is importable as
  `neurospatial.CartesianOnlyError`. It is no longer an `AttributeError` or a
  `NotImplementedError`.
- Every Cartesian-only refusal raises this one error: the methods, the
  Cartesian modes of shared methods such as `distance_to(metric="euclidean")`,
  analysis functions such as `spatial_autocorrelation`, and
  `Environment.combine`.
- The message says what was refused, why it needs Cartesian positions, and what
  to use on a polar environment. When a polar environment has an equivalent,
  the message ends with the code, also in `error.example`. The suggested
  lookups give `-1` for rows outside `env.dimension_ranges`;
  `env.nearest_bin` alone would put them in the nearest edge bin.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import CartesianOnlyError, Environment

polar_env = Environment.from_polar_egocentric(
    distance_range=(0.0, 40.0),  # cm from the animal
    angle_range=(-np.pi, np.pi),
    distance_bin_size=10.0,
    angle_bin_size=np.pi / 4,
    units="cm",
)
points = np.array([[15.0, 0.3], [55.0, 0.0]])  # (distance, angle); 2nd is beyond 40 cm
# Before: if hasattr(polar_env, "bin_at"): ...
# Before: except (AttributeError, NotImplementedError):
try:
    bins = polar_env.bin_at(points)
except CartesianOnlyError as error:
    print(error)  # what, why, and the code to use on a polar environment
    print(error.example)
# To branch before calling, read the coordinate kind instead of hasattr
if polar_env.coordinate_kind == "polar":
    low, high = np.asarray(polar_env.dimension_ranges).T
    inside = ((points >= low) & (points <= high)).all(axis=1)
    bins = np.where(inside, polar_env.nearest_bin(points), -1)
assert bins[1] == -1  # outside the surround, not the nearest edge bin
```

A polar environment now answers `component_keys`, `bin_sources` and
`get_bridge_attributes()` like any environment that was not combined: an empty
tuple, `None` and an empty table.

## `ops` functions that renamed an `Environment` method

Three functions in `neurospatial.ops` only called an `Environment` method
under another name. They are removed; call the method. The arguments and
results are unchanged.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.ops import translate

rng = np.random.default_rng(0)
positions = rng.uniform(0, 100, size=(2000, 2))  # cm
env = Environment.from_samples(positions, bin_size=4.0, units="cm")

# Before: map_points_to_bins(env, positions[:5], tie_break="lowest_index")
bins = env.nearest_bin(positions[:5], tie_break="lowest_index")
# Before: compute_differential_operator(env)
D = env.get_differential_operator()  # (n_bins, n_edges)
# Before: apply_transform_to_environment(env, translate(10.0, 0.0), frame="rig")
shifted = env.apply_transform(translate(10.0, 0.0), frame="rig")
assert np.allclose(shifted.bin_centers, env.bin_centers + [10.0, 0.0])
```

`map_points_to_bins` refused polar environments; `env.nearest_bin` accepts
them and compares the stored distance and angle columns directly.

## `validate_environment` is removed

`neurospatial.layout.validation.validate_environment(env)` re-checked what
every factory already checks when it builds an environment, so it always
passed. Remove the call. Its `strict=True` mode warned when `units` or `frame`
were missing; declare them when you build the environment instead.

<!-- docs-test: run -->
```python
# Before: validate_environment(env)  # warned: no frame
labelled = env.with_metadata(frame="room")
assert (labelled.units, labelled.frame) == ("cm", "room")
```
