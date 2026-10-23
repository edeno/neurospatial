# Environment architecture changes

The `Environment` internals were reorganized so that each kind of data has one
owner. Rate maps, smoothing and decoding give the same numbers as before;
`env.occupancy` changes only for intervals that end at a dropout or outside the
environment. This page lists what you may need to change in your own code,
with the old call as a comment above the new one. The examples use `positions` and
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
cannot be loaded. An environment state from another version raises a
`TypeError` that says to use files instead.
Pickle still works for sending an environment to worker
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

## `neurospatial.ops.binning` is now `neurospatial.ops.mapping`

`neurospatial.binning` holds the `Binning` records that describe an
environment's bins. The point-to-bin utilities had a confusingly similar
name, so `neurospatial.ops.binning` is renamed `neurospatial.ops.mapping`.
The functions are unchanged:

<!-- docs-test: run -->
```python
# Before: from neurospatial.ops.binning import regions_to_mask
from neurospatial.ops.mapping import regions_to_mask, resample_field
from neurospatial.ops import TieBreakStrategy  # also exported from ops
```

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
`nearest_bin`, the search structures behind `bin_at` on tracks, meshes and
clipped environments, diffusion and Gaussian smoothing kernels, the eigenbases used by
`smooth`, `diffuse` and `get_mrf_basis`, the gradient and divergence
operators, boundary bins and the bin and edge attribute tables. They now live
in one cache per environment.

**`clear_cache()` takes no arguments.** It releases every cached analysis. The
`kdtree=`, `kernels=` and `cached_properties=` options are removed, together
with `neurospatial.ops.mapping.clear_kdtree_cache`. Clearing only frees
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

A polar environment saved with an earlier version loads with
`layout_type == "Polar"` too.

Whether the angle wraps around at ±π is read only from the environment's
`EgocentricSurround` arena. Files from before arenas were saved
(`Environment-v2` files and NWB environment schema 2.0) do not have one, so a
polar environment stored in them can no longer be read. Load it with the
version that wrote it and save it again, or rebuild it with
`Environment.from_polar_egocentric` and the same ranges, bin sizes and
`circular_angle`, as above.

## Custom layout engines supply `binning()`

Each layout engine now hands its geometry to the environment through one
method, `binning()`, which returns a `Binning`: the bins, neighbor pairs with
their distances (their order becomes `env.edge_pairs`), point lookup, the
faces between cells (or the reason they are missing) and any grid, track,
shape and outline records. All built-in engines implement it, and the
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
  the bins cover. Lookup and faces come only from `binning()`: the built-in
  engines no longer have `point_to_bin_index`, `finite_volume_geometry` or
  `is_grid_compatible` (see below), and a custom engine does not need them. A layout whose methods were
  replaced on the instance is no longer refused.
- A built-in engine computes its neighbor pairs as arrays in `build()` and
  no longer builds a graph. Its `connectivity` is derived from `binning()` on
  first access, with the same node and edge attributes as `env.connectivity`.
  It can no longer be assigned, and editing it changes neither the engine's
  bins nor an environment built from it. To publish other neighbor pairs,
  return them from your own engine's `binning()`.
- An engine may also have a `layout_type` label, and build
  arguments recorded with `neurospatial.layout.base.capture_build_params`
  become `env.layout_parameters`. Both are optional. The label becomes
  `env.layout_type` and the binning's `kind` (see "One layout label"
  below); without one, the environment uses your binning's `kind`. Name the
  arguments that are input data, such as positions, masks, polygons or
  graphs, once in the decorator, as in
  `@capture_build_params(data=("positions",))`. They are not recorded, and
  any other argument that is not a scalar or a flat sequence of scalars
  raises a `TypeError` when the engine is built, instead of being dropped
  silently.
- Each bin center must be found in its own bin by the binning's locator; the
  environment checks this for every engine when it is built and refuses a
  binning that fails it.
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

from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)


class RecordedPoints:
    """Bins at recorded reward-port positions, joined in a ring."""

    def build(self, *, points):
        points = np.asarray(points, dtype=float)
        ring = np.arange(len(points))
        pairs = np.column_stack([ring, np.roll(ring, -1)])
        lengths = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
        self._binning = Binning(
            "RecordedPoints",
            SpatialBins.from_arrays(points, np.ones(len(points)), 2),
            BinConnections.from_arrays(pairs, lengths),
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

## The connectivity graph is rebuilt from the bins

`env.connectivity` was already read-only. It is now rebuilt from the bins
whenever an environment is built, derived (`subset`, `clip`, `rebin`,
`apply_transform`, `combine`) or loaded, so every environment carries the
same standard attributes: `pos`, `source_grid_flat_index` and
`original_grid_nd_index` on nodes; `distance`, `vector`, `edge_id` and, in
2-D, `angle_2d` on edges; `pos_1D` and `source_edge_id` on tracks. The
values are numbers and tuples, read without a copy. A saved file keeps its
graph for other tools, but loading does not read it back, so attributes
added to a saved graph are dropped. Keep per-bin labels in your own arrays
indexed by bin ID, or annotate an editable copy:

<!-- docs-test: run -->
```python
# Before: env.connectivity.nodes[0]["label"] = "start"  (refused)
graph = env.to_networkx()
graph.nodes[0]["label"] = "start"
labels = np.full(env.n_bins, "", dtype=object)  # or an array indexed by bin
labels[0] = "start"
```

## Neighbor pairs live in `BinConnections`

A `Binning` keeps its bins (`SpatialBins`: centers, measures and
`measure_dim`) and their neighbor pairs (`BinConnections`: pairs, navigation
distances and `is_open`) in separate records, passed in that order. A wall
closes the pairs across it instead of deleting them; `connections.pairs` and
`env.edge_pairs` are the open pairs, as before. The clip mapping is read from
the clipped locator, so `Binning` no longer takes `clip_mapping=`:

<!-- docs-test: run -->
```python
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)

sites = np.array([[0.0, 0.0], [2.0, 0.0]])
# Before: bins = SpatialBins.from_arrays(sites, [1.0, 1.0], 2, [[0, 1]], [2.0])
# Before: Binning("sites", bins, NearestBinLocator(sites), MissingFaces("none"))
bins = SpatialBins.from_arrays(sites, [1.0, 1.0], 2)
connections = BinConnections.from_arrays([[0, 1]], [2.0])
sites_env = Environment.from_bins(
    Binning("sites", bins, connections, NearestBinLocator(sites), MissingFaces("none"))
)
# Before: sites_env.binning.bins.pairs
assert sites_env.binning.connections.pairs.tolist() == [[0, 1]]
```

## Track graph attributes must be saveable

A track environment (`from_graph`, `linear_track`, `maze`) keeps the graph you
built it from, and saves it with the environment. An attribute that cannot be
saved at all, such as a function or a lock, is now refused when the
environment is built, with a `TypeError` that names the node or edge and the
attribute. Before, it failed later, when the environment was pickled for a
worker process or written to a file.

Attributes that pickle can store but JSON cannot, such as sets, dates or NaN,
still build and still pickle. `to_file`, `to_dict` and NWB export refuse them
with a `ValueError` that names the attribute. Store such values as numbers,
strings or lists, or keep them outside the graph:

<!-- docs-test: run -->
```python
import networkx as nx

track = nx.Graph()
track.add_node(0, pos=(0.0, 0.0))
track.add_node(1, pos=(100.0, 0.0))
# Before: track.add_edge(0, 1, distance=100.0, reward=lambda t: t > 5)
track.add_edge(0, 1, distance=100.0, reward_after_s=5.0)
linear = Environment.from_graph(track, [(0, 1)], edge_spacing=0.0, bin_size=4.0)
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

The recorded choices describe the bins that were built. An option the bins
did not use is recorded at the value that does nothing: with
`infer_active_bins=False` every bin is active, so `open_field` records
`fill_holes=False` (it used to record its default `fill_holes=True`), and a
1-D grid records `connect_diagonal_neighbors=False`. An
option that cannot act at all is refused, as `edge_spacing` is on a straight
`linear_track`, which has one edge and no gap between edges:

<!-- docs-test: run -->
```python
# Before: Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=2.0, edge_spacing=5.0)
track = Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=2.0)
assert Environment.open_field(
    positions, bin_size=4.0, infer_active_bins=False
).layout_parameters["fill_holes"] is False
```

## One layout label

A layout used to have several names: `LayoutType.REGULAR_GRID` was
`"RegularGrid"`, but `RegularGridLayout().layout_type` was the category
`"grid"`, saved files also stored a `layout_category`, and
`Environment.from_samples(..., layout="regular-grid")` reported
`env.layout_type == "regular-grid"`. Derived environments invented more:
`"subset"`, `"PolygonCells"` after `clip`, `"combined"`, and
`"RegularGrid_transformed"` after each `apply_transform`.

Now `LayoutType` is the one vocabulary. An engine's `layout_type`, the
binning's `kind` and `env.layout_type` are the same `LayoutType` value. The
user spellings that `create_layout` and `layout=` accept (`"regular-grid"`,
`"regular_grid"`, and the short names `"grid"`, `"polygon"` and `"mesh"`)
are reported in their canonical form. `LayoutType.POLAR` and
`LayoutType.COMBINED` label environments from `from_polar_egocentric` and
`combine`; `create_layout` refuses them and names the factory to use.

Derived environments keep their parent's label, and `env.derivations` lists
what was applied, oldest first:

| Call | `layout_type` before | `layout_type` now | `derivations` |
| --- | --- | --- | --- |
| `grid.subset(...)` | `"MaskedGrid"` | `"RegularGrid"` | `("subset",)` |
| `hexagons.subset(...)` | `"subset"` | `"Hexagonal"` | `("subset",)` |
| `grid.clip(polygon)` | `"PolygonCells"` | `"RegularGrid"` | `("clip",)` |
| `grid.apply_transform(T)` | `"RegularGrid_transformed"` | `"RegularGrid"` | `("apply_transform",)` |
| `polygon_grid.rebin(2)` | `"RegularGrid"` | `"ShapelyPolygon"` | `("rebin",)` |
| `Environment.combine({...})` | `"combined"` | `"Combined"` | `("combine",)` |

`combine` starts a new history, because the result has no single parent.
`env.layout_parameters` keeps the parent's build choices through `subset`,
`clip` and `apply_transform` instead of being replaced by
`{"source": "subset", "original_n_bins": ...}` or gaining
`"transformed": True`. `rebin` builds new bins and records their `bin_size`
and `connect_diagonal_neighbors`, whether you passed a factor or a
`bin_size`. `combine` records no build choices.

<!-- docs-test: run -->
```python
west = env.subset(mask=env.bin_centers[:, 0] < 50.0)
# Before: west.layout_type == "MaskedGrid" and
#         west.layout_parameters == {"source": "subset", ...}
assert west.layout_type == env.layout_type
assert west.derivations == ("subset",)
assert west.layout_parameters == env.layout_parameters
```

Code that compared `engine.layout_type` with a category such as `"grid"` or
`"mask"` should compare with a `LayoutType` value, or better, ask the bins
what it needs (for example `env.binning.rectilinear is not None` for an
axis-aligned grid). Files saved by earlier versions load with the canonical
label: suffixes are dropped, `"subset"` and `"PolygonCells"` take the label
of the bins they were cut from (a grid subset or clip reads as
`"MaskedGrid"`, as such a subset was labelled), polar bins read as
`"Polar"`, and the `"source"` and `"transformed"` markers become
`env.derivations`. Input data that earlier versions kept in the layout
parameters (positions, masks, polygons) is dropped when such a file is read;
the bins themselves are stored in the file.

## Tracks need `track_width` to look up positions

A track's centerline and bin size do not say how far from the centerline a
position is still on the track. Without a width, every position, however far
off the track (a reflection, a tracking jump to the other side of the room),
used to be put in its nearest bin, and its time and spikes were counted there.
A track built without `track_width` now refuses to assign 2-D positions to
bins: `bin_at`, `contains`, `occupancy`, the bin sequences,
`transitions(times=..., positions=...)`, `points=` queries, point regions and
the analyses built on them raise `NotImplementedError`. With a width,
positions farther than half the width from every centerline segment get bin
`-1` and add no occupancy, like positions outside an open-field arena.

Pass the measured full width of the corridor, in the units of your positions,
to `linear_track`, `maze` or `from_graph`. The error message shows how to give
an existing track a width (see the next section).

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

# Before: Environment.linear_track(endpoints=[(0, 0), (200, 0)], bin_size=2.0)
track = Environment.linear_track(
    endpoints=[(0, 0), (200, 0)], bin_size=2.0, track_width=10.0, units="cm"
)
positions = np.array([[100.0, 2.0], [100.0, 500.0]])  # on the track; 5 m away
assert track.bin_at(positions)[1] == -1  # before: the bin at x = 100
```

`linear_to_nd` and graph queries on bin indices (`bins=`) do not need the
width. `to_linear` does not either, but it projects every position onto the
centerline, however far off the track it is, so it is not a way around the
refusal: keep only the samples `bin_at` puts on a track with a width. Simulated mazes from `neurospatial.simulation.mazes` pass
their corridor width to their track. The Sungod, honeycomb and small hex
mazes, whose corridors differ in width, pass the widest one: a sample within
half of it from the centerline counts as on the track, so a narrow corridor
can admit a few samples just outside its walls.

### Tracks saved by earlier versions

A track saved by an earlier version has no width, and neither does a track
you no longer have the build script for. Read it as before and give it its
measured full width with `env.with_track_width(width)`. It keeps the bins,
the linear coordinates, the name, units, frame and regions, so rate maps and
decoders keep their bin order. It also works on a subset or a transformed
track; give the width in the transformed coordinates.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

# A track saved without a width, as earlier versions wrote it
saved = Environment.linear_track(
    endpoints=[(0, 0), (200, 0)], bin_size=2.0, units="cm"
).to_dict()

track = Environment.from_dict(saved)  # or from_file(path), from_nwb(nwbfile)
# Before: track.bin_at(positions) put every position in its nearest bin
track = track.with_track_width(10.0)  # the corridor is 10 cm wide
positions = np.array([[101.0, 2.0], [101.0, 500.0]])  # on the track; 5 m away
assert track.bin_at(positions).tolist() == [50, -1]
# Save it again so the width is stored with it
saved = track.to_dict()
```

## `dimension_ranges` is the extent of the bins' cells

`env.dimension_ranges` is the extent of the cells the bins cover, for every
layout. A parent reports its cells' extent and `env.subset(...)` the extent of
the cells it keeps, with no margin, so a subset never reports a larger range
than its parent. Plot limits follow `dimension_ranges`, so plots fit the cells
instead of cutting them off or leaving extra space around them.

| Layout | Before | Now |
| --- | --- | --- |
| Hexagonal | the range the tiling was asked to cover (data or `dimension_ranges=`), which cut off the outer hexagons | the hexagons' extent, corners included |
| Triangular mesh | the bounds of the triangle centroids | the mesh boundary's bounds |
| Track (`linear_track`, `maze`, `from_graph`) | the bounds of the bin centers, half a bin short at each end | the centerline the bins cover, end to end (the width is not included) |
| Rectangular grid (including `from_polygon`, masks and images), polar | the whole grid, inactive border cells included | the active cells (unchanged when every border row and column has an active cell) |
| Grid subsets | the whole grid the subset keeps (`grid_edges`) | the cells the subset keeps |
| Subsets of hexagonal, mesh and clipped polygon cells (`env.clip(...)`) | kept cells' extent plus 5% on each side | kept cells' extent |
| Track subsets | kept bin centers' extent | the centerline the kept bins cover |

For a 200 cm linear track built with `bin_size=2.0`, `dimension_ranges[0]`
was `(1.0, 199.0)` and is now `(0.0, 200.0)`. A hexagonal environment
requested over `(0, 30)` with `hexagon_width=4.0` now reports
`(-2.0, 36.0)`, the hexagons it built.

A mesh subset bounds only the triangles it keeps, so a subset of one corner of
an arena no longer reports the extent of the whole arena. The same rule applies
when an `arena=` passed to `from_layout`, `from_grid_mask` or `from_pixel_mask`
removes bins. `Environment.from_bins` bounds triangular-mesh bins by their
triangles' vertices rather than every stored mesh point. A grid subset still
keeps the whole lattice in `env.grid_edges` and `env.grid_shape`, which is
what napari draws, but its `dimension_ranges` is the cells it keeps.

Code that reads `dimension_ranges` as a number changes with it on hexagonal,
mesh, track and grid-subset environments: the automatic scale of
`neurospatial.ops.spatial_basis`, the "bandwidth spans the environment"
warning, the uniform field placement of `simulate_session`, axis limits,
`env.info()`, and the extent that animation overlays and an uncalibrated
video are drawn over. To keep the old numbers, pass them yourself, for
example the bin-center bounds:

<!-- docs-test: run -->
```python
track = Environment.linear_track(endpoints=[(0, 0), (200, 0)], bin_size=2.0)
assert track.dimension_ranges[0] == (0.0, 200.0)
# Before: track.dimension_ranges[0] == (1.0, 199.0), the bin-center bounds
centers_low, centers_high = track.bin_centers.min(axis=0), track.bin_centers.max(axis=0)
```

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

An old call raises a `TypeError` that says which keyword applies to the
value you passed and shows the corrected call. It makes the same guess the old
call made, integers as bins and real numbers as points, so check that the
suggestion means what you intended:

<!-- docs-test: run -->
```python
from neurospatial import Regions

positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))  # cm
regions = Regions()
regions.add("reward", point=(82.0, 77.0))
env = Environment.from_samples(positions, bin_size=4.0, units="cm", regions=regions)
try:
    env.neighbors(12)
except TypeError as error:
    assert "env.neighbors(bins=12)" in str(error)
try:
    env.distance_to("reward")
except TypeError as error:
    assert "env.distance_to(region='reward')" in str(error)
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
always means integer bin indices. `env.subset(bins=west)` raises a `TypeError`
that says to pass `mask=`:

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
smoothed = env.smooth(rate, bandwidth=8.0, mode="average")
assert np.allclose(smoothed[np.isfinite(smoothed)], 5.0)
# Before: env.nearest_bin([[np.nan, 10.0]]) raised a SciPy error
assert env.nearest_bin(np.array([[np.nan, 10.0]]))[0] == -1
```

`env.occupancy` gives an interval time only when both of its samples are
inside the environment, the rule `compute_spatial_rate` uses for spike counts
and occupancy. An interval that ends at a dropout or outside the environment
used to keep its time in its start bin, so dividing your own spike counts by
`env.occupancy` used a different denominator than the rate maps. Fill short
dropouts during tracking preprocessing if you want that time back.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

env = Environment.from_samples(np.linspace(0, 100, 101)[:, None], bin_size=10.0)
times = np.array([0.0, 0.1, 0.2, 0.3])
positions = np.array([[15.0], [16.0], [np.nan], [45.0]])  # cm; one dropout
# Before: 0.2 s (the interval ending at the dropout kept its time)
assert np.isclose(env.occupancy(times, positions).sum(), 0.1)
```

`occupancy`, `bin_sequence`, `bin_sequence_with_runs` and trajectory
`transitions` now warn, like `compute_spatial_rate`, when `max_gap` and the
other interval gates exclude every interval, or when the median sampling
interval is longer than `max_gap`. Timestamps in milliseconds trigger it: each
33 ms frame looks like a 33 s recording gap. Convert `times` to seconds.

## Cartesian-only methods on polar environments

An egocentric polar environment stores `(distance, angle)` bins around the
animal. Methods that read positions as Cartesian `(x, y)`, such as `subset`,
`rebin`, `interpolate` and `apply_transform`, still refuse it, but the refusal
now happens when you call the method, not when you look it up:

- `hasattr(polar_env, "subset")` is `True` and `dir(polar_env)` lists the
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
  the message ends with the code, also in `error.example`.

## Polar environments look up `(distance, angle)` rows

`bin_at`, `contains`, `occupancy`, `bin_sequence`, `bin_sequence_with_runs`
and `transitions(times=..., positions=...)` work on an egocentric polar
environment. Each row is `(distance, angle)`, the angle in radians, and a row
gets the bin that contains it, or `-1` (`False`) outside the distance and angle
ranges. The `points=` queries (`distance_to`, `neighbors`, ...) and point
regions use the same lookup. An angle is a direction, so an angle a whole turn
outside the angle range (for example `3.5` on `(-π, π)`) finds the same bin as
the angle inside it, and `π` on a full circle is the bin at `-π`; before, such
rows were outside. `occupancy(time_allocation="linear")` still refuses a polar
environment, because a straight path between two samples is not straight in
`(distance, angle)`.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

polar_env = Environment.from_polar_egocentric(
    distance_range=(0.0, 40.0),  # cm from the animal
    angle_range=(-np.pi, np.pi),
    distance_bin_size=10.0,
    angle_bin_size=np.pi / 4,
    units="cm",
)
points = np.array([[15.0, 0.3], [15.0, 0.3 + 2 * np.pi], [55.0, 0.0]])
# Before: polar_env.bin_at(points) raised CartesianOnlyError, which suggested
# graph operations or building an allocentric Cartesian environment instead
bins = polar_env.bin_at(points)
assert bins[0] == bins[1] >= 0  # one direction, one bin
assert bins[2] == -1  # beyond the 40 cm surround
```

`nearest_bin` on a polar environment now measures the physical distance
between places, in the environment's length units: it compares
`(distance * cos(angle), distance * sin(angle))` positions in the animal's
plane, not the raw `(distance, angle)` columns. It can therefore return a
different bin than before, and `return_dist=True` distances and
`max_distance` are lengths. It is still the nearest bin center, not the bin
that contains the row; use `bin_at` for that.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

polar_env = Environment.from_polar_egocentric(
    distance_range=(0.0, 40.0),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=10.0,
    angle_bin_size=np.pi / 4,
    units="cm",
)
row = np.array([[10.2, 0.0]])  # 10.2 cm straight ahead
# Before: nearest_bin compared cm and radians as one unit
nearest, distance_cm = polar_env.nearest_bin(row, return_dist=True)
containing = polar_env.bin_at(row)  # the 10-20 cm ring, ahead
assert polar_env.bin_centers[containing[0], 0] == 15.0
```

The analysis functions that read positions as the animal's allocentric
positions keep refusing a polar environment, now with their own name in the
message: `compute_spatial_rate`, `compute_spatial_rates`,
`place_cell_significance`, `compute_directional_place_fields`, the view-rate functions,
`compute_decision_analysis`, `compute_vte_session`, the geodesic modes of
`compute_step_lengths`, `mean_square_displacement` and `decoding_error`,
the `distance_to_reward` and `distance_to_boundary` event regressors,
`simulate_trajectory_ou`, and the boundary, object-vector, spatial-view and
geodesic place-cell models. For egocentric rate maps use
`neurospatial.encoding.compute_egocentric_rate`.

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

## Layout engines no longer look up points or build faces themselves

The built-in layout engines had `point_to_bin_index`, `finite_volume_geometry`
and `is_grid_compatible`. An environment never called them; it uses the
engine's `binning()`. They are removed. Ask the environment, or the engine's
binning:

<!-- docs-test: run -->
```python
from neurospatial.layout import create_layout

engine = create_layout(
    "RegularGrid", bin_size=4.0, dimension_ranges=[(0.0, 100.0), (0.0, 100.0)]
)
binning = engine.binning()
# Before: engine.point_to_bin_index(positions[:5])
bins = binning.locator.locate(positions[:5])
# Before: engine.finite_volume_geometry()
faces = binning.faces.build()  # face areas faces.A, cell measures faces.volumes
# Before: engine.is_grid_compatible
is_grid = binning.rectilinear is not None
assert is_grid and bins.min() >= 0
```

For an environment, `env.bin_at(positions)` locates points and
`env.binning.rectilinear is not None` tells you whether the bins form a
rectangular grid.

## An environment's public attributes cannot be set

Assigning any public attribute of an `Environment`, even one the class does
not define, raises an `AttributeError`; it used to attach the value silently.
Keep your own session labels beside the environment:

<!-- docs-test: run -->
```python
# Before: env.session_id = "s1"
sessions = {env.name: {"session_id": "s1"}}
labelled = env.with_metadata(name="s1")  # or set the environment's own name
```

## Simulated plus-maze arms are numbered in linear order

`simulation.mazes.make_plus_maze` now builds its linear track with
`Environment.maze(kind="plus")`, the same builder as every other plus maze.
Its arms are numbered in the order they are laid out along the linear
position, clockwise from north. East and south swap numbers in each bin's
`source_edge_id` and in the `edge_id` of the track graph:

| Arm | Before | Now |
| --- | --- | --- |
| north | 0 | 0 |
| south | 1 | 2 |
| east | 2 | 1 |
| west | 3 | 3 |

Bin positions, linear positions (`pos_1D`) and every analysis are unchanged.
Select an arm by where its bins lie, or by a region, rather than by a
hard-coded number:

<!-- docs-test: run -->
```python
from neurospatial.simulation.mazes import make_plus_maze

track = make_plus_maze(include_track=True).env_track
arm = np.array([track.connectivity.nodes[b]["source_edge_id"] for b in range(track.n_bins)])
# Before: east = arm == 2
east = track.bin_centers[:, 0] > 0
assert set(arm[east]) == {1}
```

## Grid analyses on layouts without square lags raise `NotImplementedError`

The FFT autocorrelogram behind grid scores needs a 2-D grid of rectangular
cells, one width along each axis. `spatial_autocorrelation` used to raise
`ValueError` for tracks, meshes and sparse grids, and accepted hexagonal
environments with wrong results. It now raises `NotImplementedError` for all
of them, naming `spatial_autocorrelation_radial`, which works on every layout.
Polar environments raise `CartesianOnlyError` (a `TypeError`), here and for
wall-based measures such as `border_score`:

<!-- docs-test: run -->
```python
from neurospatial.encoding.grid import (
    spatial_autocorrelation,
    spatial_autocorrelation_radial,
)

track = Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=2.0, units="cm")
rate = np.random.default_rng(1).random(track.n_bins)
# Before: except ValueError:
try:
    autocorr = spatial_autocorrelation(track, rate)
except NotImplementedError:
    distances, correlations = spatial_autocorrelation_radial(track, rate)
```

Rate-map `grid_score()` and `summary_table()` report NaN on these
environments, as before for tracks and meshes.

## Grid orientation is in radians, in the environment's frame

`grid_properties(...).orientation` used to be in degrees, in [0, 60), and
measured in the autocorrelogram's image layout, which reflected the field
lattice about 45°. It is now the lattice angle in the environment's frame:
radians from +x (East) counterclockwise toward +y (North), in [0, π/3),
because a hexagonal lattice repeats every 60°. `orientation_std` is in
radians too. `peak_coords` columns are now x then y offsets; earlier
versions returned them y first, so swap the columns of any saved values.
`GridCellModel(grid_orientation=...)` takes the same angle, so a simulated
grid cell is reported with the orientation it was given:

<!-- docs-test: run -->
```python
from neurospatial.encoding.grid import (
    autocorrelation_pixel_size,
    grid_properties,
    spatial_autocorrelation,
)
from neurospatial.simulation import GridCellModel

open_field = np.random.default_rng(0).uniform(0, 100, size=(5000, 2))  # x, y in cm
arena = Environment.from_samples(open_field, bin_size=2.0, units="cm")
cell = GridCellModel(arena, grid_spacing=30.0, grid_orientation=np.radians(20.0))
autocorr = spatial_autocorrelation(arena, cell.firing_rate(arena.bin_centers))
properties = grid_properties(autocorr, bin_size=autocorrelation_pixel_size(arena))
# Before: the lattice sat at 50° (grid_orientation set the plane waves), and
# properties.orientation reported it reflected about 45°, in degrees
assert abs(np.degrees(properties.orientation) - 20.0) < 2.0
```

To compare with orientations computed by an earlier version, convert them:

<!-- docs-test: run -->
```python
old_orientation_deg = 10.0  # from an earlier version
orientation = np.radians((90.0 - old_orientation_deg) % 60.0)  # 20° in radians
```

The conversion is approximate. Earlier versions also averaged in a spare
peak that could lie on the second ring of fields, so old values were off by
up to about 5°. Recompute orientations from your rate maps when you can.

`GridCellModel(grid_orientation=θ)` used to set the angle of its plane waves,
which put the field lattice π/6 further counterclockwise. To reproduce a
simulation made with an earlier version, add π/6:

<!-- docs-test: run -->
```python
from neurospatial.simulation import GridCellModel

earlier_grid_orientation = 0.0
# Before: GridCellModel(env, grid_spacing=30.0, grid_orientation=earlier_grid_orientation)
cell = GridCellModel(env, grid_spacing=30.0, grid_orientation=earlier_grid_orientation + np.pi / 6)
```

## `graph_turn_sequence` takes the bins only

`start_bin` and `end_bin` were never used: the order of the samples already
sets the direction of travel. Drop them from the call. The labels are now the
animal's own left and right in the y-up frame (heading North and then West
is `"left"`); earlier versions swapped them, so recompute saved sequences:

<!-- docs-test: run -->
```python
from neurospatial.behavior import graph_turn_sequence

position_bins = env.bin_at(positions)
# Before: graph_turn_sequence(env, position_bins, start_bin, end_bin)
turns = graph_turn_sequence(env, position_bins)
```

## Smoothing several fields at once

`env.smooth` now takes a batch of fields with bins first, shape
`(n_bins, n_fields)`, as `env.diffuse` does. Population rate arrays are
stored `(n_units, n_bins)`, so transpose them on the way in and out. Passing
the rate result itself raises `TypeError` with this call:

<!-- docs-test: run -->
```python
rates = np.random.default_rng(1).random((3, env.n_bins))  # (n_units, n_bins), Hz
# Before: np.stack([env.smooth(unit_rates, 5.0) for unit_rates in rates])
smoothed = env.smooth(rates.T, bandwidth=5.0, mode="average").T
```

## Transforms that change lengths name their units

`env.apply_transform(transform)` used to keep the environment's units, even
when the transform was a pixel-to-centimeter calibration. A transform that
scales or shears lengths now needs `units=` when the environment has units.
Translations, rotations and reflections keep the units without it.

<!-- docs-test: run -->
```python
from neurospatial.ops import scale_2d

pixels = env.with_metadata(units="px")
# Before: pixels.apply_transform(scale_2d(0.1)).with_metadata(units="cm")
room = pixels.apply_transform(scale_2d(0.1), units="cm")
assert room.units == "cm"
```

## Smoothing needs `mode=`

`env.smooth`, `env.diffuse` and `env.compute_kernel` take a required `mode=`.
They had different defaults (`smooth` averaged, the other two made densities),
so the same call gave different numbers depending on the method. Choose the
mode by what the field holds: `"average"` for a rate map or other value per
bin, `"transition"` for counts, occupancy or probability mass (keeps the
total), and `"density"` to turn counts into a density. Leaving it out raises a
`TypeError` that lists the three.

<!-- docs-test: run -->
```python
rate = np.random.default_rng(3).random(env.n_bins)  # Hz
# Before: env.smooth(rate, bandwidth=5.0)  (averaged by default)
smoothed = env.smooth(rate, bandwidth=5.0, mode="average")
# Before: env.diffuse(counts, bandwidth=5.0)  (density by default)
counts = np.random.default_rng(4).poisson(2.0, env.n_bins).astype(float)
density = env.diffuse(counts, bandwidth=5.0, mode="density")
```

## Method options are keywords

`Environment` methods take their data positionally and every option by
keyword, like the factories. A bandwidth is an option, so name it; this also
keeps a smoothing width from being read as some other number:

<!-- docs-test: run -->
```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rate = np.random.default_rng(2).random(env.n_bins)  # Hz
# Before: env.smooth(rate, 5.0)
smoothed = env.smooth(rate, bandwidth=5.0, mode="average")
# Before: env.diffuse(counts, 5.0); env.compute_kernel(5.0)
kernel = env.compute_kernel(bandwidth=5.0, mode="density")
# Before: env.plot_field(rate, ax, "magma")
fig, ax = plt.subplots()
env.plot_field(smoothed, ax, cmap="magma")
plt.close(fig)
```

An old positional call raises a `TypeError` that names the keyword and shows
the corrected call:

<!-- docs-test: run -->
```python
rate = np.random.default_rng(2).random(env.n_bins)  # Hz
try:
    env.smooth(rate, 5.0)
except TypeError as error:
    assert "pass bandwidth=5.0 by keyword, as in env.smooth(field, bandwidth=5.0)" in str(
        error
    )
```

`to_nwb(nwbfile, "arena")` is now `to_nwb(nwbfile, name="arena")`, matching
`Environment.from_nwb(nwbfile, name="arena")`.

Coordinates passed to `bin_at`, `contains`, `nearest_bin`, `to_linear` and
`interpolate` are named `positions`, as in the analysis functions. A call
that passes them by position is unchanged:

<!-- docs-test: run -->
```python
# Before: env.bin_at(points_nd=positions); env.nearest_bin(points=positions)
bins = env.bin_at(positions=positions)
nearest = env.nearest_bin(positions)
```

## Draw the trajectory with an overlay

`env.animate_fields(..., overlay_trajectory=positions)` is removed; no backend
drew it. Use a position overlay, which every backend draws:

<!-- docs-test: run -->
```python
from neurospatial.animation import PositionOverlay

# Before: env.animate_fields(fields, frame_times=..., overlay_trajectory=positions)
path = PositionOverlay(positions=positions, times=times, trail_length=30)
# env.animate_fields(fields, frame_times=frame_times, overlays=[path])
```

## Misspelled region names raise

A region name that is not in `env.regions` used to be skipped by
`animate_fields(show_regions=[...])`, `regions_containing_points(region_names=...)`
and `plot_regions(region_names=...)`, so the region was silently missing. Each
now raises `RegionNotFoundError` naming the closest region; fix the name:

<!-- docs-test: run -->
```python
from neurospatial import RegionNotFoundError, Regions
from neurospatial.regions import regions_containing_points

regions = Regions()
regions.add("reward", point=(50.0, 50.0))
# Before: regions_containing_points(positions, regions, region_names=["rewrd"])
# returned a table without the column
try:
    regions_containing_points(positions, regions, region_names=["rewrd"])
except RegionNotFoundError as error:
    print(str(error).splitlines()[-1])  # Fix: pass region_names=['reward']
table = regions_containing_points(positions, regions, region_names=["reward"])
```

## Bin indices are `np.intp`

`bin_sequence`, `bin_sequence_with_runs(...).bins`, `components` and `rings`
return `np.intp` arrays (64-bit on 64-bit systems) instead of `int32`, like
`bin_at` and `nearest_bin`. The values are the same. Code that compares the
dtype or stores the arrays in an `int32` file column should cast explicitly:

<!-- docs-test: run -->
```python
sequence = env.bin_sequence(times, positions)
assert sequence.dtype == np.intp
# Before: sequence was int32; keep that layout with an explicit cast
compact = sequence.astype(np.int32)
```

## Infinite positions are missing samples

`from_samples`, `open_field` and `environment_from_position` treat a row with
an infinite coordinate like a row with NaN: as a missing sample that does not
shape the bins. Such a row used to raise an overflow error, so a recording
with one bad sample can now be used without cleaning it first:

<!-- docs-test: run -->
```python
glitched = positions.copy()
glitched[10] = [np.inf, 50.0]  # a tracking glitch
# Before: Environment.from_samples(glitched, bin_size=4.0) raised "Requested binning overflows"
arena = Environment.from_samples(glitched, bin_size=4.0, units="cm")
```

## `animate_fields` returns a view or a file

`env.animate_fields` builds an `Animation` scene and either opens a view of it
or writes a file. Interactive backends (`"napari"`, `"widget"`,
`"fastplotlib"`) return an `AnimationView`, whose `native` attribute is the
napari viewer, notebook widget or fastplotlib animation; closing the view
releases the video files it opened. Video and HTML exports return the `Path`
they wrote. All backends now play the same scene on the recording clock:

- A field row is held until the next one only up to `max_gap=0.5` seconds;
  across longer gaps the field is missing instead of frozen. Pass
  `max_gap=None` for one field per lap or trial.
- A file's frame rate is the sampling rate of `frame_times` times `speed`,
  capped at `max_playback_fps`; a capped export keeps `speed`, skips rows and
  warns.
- Several fields shown side by side are a mapping from name to field, or a
  list of fields named by `layer_names`, in every backend.
- Every backend draws every overlay type. A combination an environment
  cannot show (for example video in a 3-D environment) raises before drawing
  instead of being skipped with a warning.

Removed options: napari `cache_size`, `chunk_size` and `max_chunks`; widget
`initial_cache_size`, `cache_limit`, `image_format` and `return_container`;
`dry_run`; `frame_labels` outside `backend="fastplotlib"` (every frame is
labeled with its recording time); and `large_session_napari_config`. Napari
and the notebook widget read rows lazily without cache settings.

<!-- docs-test: run -->
```python
# Before: viewer = env.animate_fields(fields, frame_times=frame_times, backend="napari")
#         viewer.layers  (the napari viewer itself)
# Now:    view = env.animate_fields(...); view.native.layers; view.close()
# Before: env.animate_fields(..., backend="html", frame_labels=labels, dry_run=True)
path = env.animate_fields(
    fields, frame_times=frame_times, save_path="animation.html", dpi=40
)
assert path.name == "animation.html"
```

## Misspelled animation options raise

A name that no backend accepts raises `TypeError` with the closest option, so
a typo cannot silently change nothing. An option that another backend uses,
such as `crf=` with `backend="html"`, does nothing, and says so with one
`UserWarning` that names the backend that uses it. `fps=` raises; the frame
rate comes from `frame_times` and `speed=`.

<!-- docs-test: run -->
```python
try:
    env.animate_fields(
        fields, frame_times=frame_times, save_path="typo.html", colorbar_lable="Hz"
    )
except TypeError as error:
    print(str(error).splitlines()[-1])  # Fix: pass colorbar_label= instead of colorbar_lable=
```
