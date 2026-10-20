# Environments

Choose spatial bins from the data you have: tracked positions, an image mask,
a grid mask or a track graph. A polygon is optional. `Environment` keeps the
bins, their connectivity and the continuous domain together so that position
queries and subsequent analyses use the same coordinates.

## Start with tracked positions { #factory-methods }

`from_samples()` is the usual starting point for a recording. Supply positions
with shape `(n_samples, n_dims)` in consistent physical units:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

rng = np.random.default_rng(0)
positions = rng.uniform(0.0, 20.0, size=(200, 2))
env = Environment.from_samples(
    positions,
    bin_size=2.0,
    units="cm",
    name="Tracked session",
)

assert env.arena.domain_known
assert not env.arena.physical_footprint_known
```

By default, bins containing observations are active. `bin_count_threshold`
controls how many samples a bin needs; `dilate`, `fill_holes` and `close_gaps`
can change the estimated support. Set `infer_active_bins=False` when the full
grid is the intended analysis region. `layout="Hexagonal"` selects hexagonal
bins instead of the default rectangular grid.

Inspect `env.grid_edges` and `env.bin_sizes` for the resulting discretization.
Use `from_grid_mask()` when exact physical cell edges are already known.

The inferred domain describes the active cells. An unvisited patch does not
establish a physical wall. Inspect the support before interpreting missing
occupancy as inaccessible space.

## Presets for common apparatus { #presets }

Three presets cover the usual apparatus. Each returns an ordinary
`Environment`, so every analysis below works the same way on it.

`open_field()` bins tracked positions like `from_samples()` but fills interior
holes. A patch in the middle of the arena that the animal happened not to
visit stays in the analysis, as an unvisited bin rather than a gap:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

rng = np.random.default_rng(0)
positions = rng.uniform(0.0, 100.0, size=(5000, 2))  # 100 x 100 cm box
# The animal never crossed an 8 cm patch in the center
positions = positions[np.linalg.norm(positions - 50.0, axis=1) > 8.0]

arena = Environment.open_field(positions, bin_size=5.0, units="cm")
visited = Environment.from_samples(positions, bin_size=5.0, units="cm")

center = np.array([[50.0, 50.0]])
assert arena.contains(center).all()  # the patch is part of the arena
assert not visited.contains(center).any()  # from_samples keeps only visited bins
```

`linear_track()` and `maze()` build tracks from their geometry, not from
positions, and linearize them so place fields can be computed along the track.
`linear_track()` takes the track's endpoints (or a list of corner positions for
a bent track). `maze()` takes a plus, T or W maze from the positions of its
center, junctions and arm ends. `track_width` is the full corridor width;
positions farther from the centerline than half of it are outside the track:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

track = Environment.linear_track(
    endpoints=[(0.0, 0.0), (180.0, 0.0)],
    bin_size=2.0,
    track_width=8.0,
    units="cm",
)
assert track.is_linearized_track
np.testing.assert_allclose(track.to_linear(np.array([[90.0, 2.0]])), [90.0])
assert track.bin_at(np.array([[90.0, 20.0]]))[0] == -1  # 20 cm off the track

plus_maze = Environment.maze(
    kind="plus",
    node_positions={
        "center": (0.0, 0.0),
        "north": (0.0, 50.0),
        "south": (0.0, -50.0),
        "east": (50.0, 0.0),
        "west": (-50.0, 0.0),
    },
    bin_size=2.0,
    track_width=8.0,
    units="cm",
)
assert plus_maze.is_linearized_track
```

For any other track, build a NetworkX graph of the centerline and pass it to
`from_graph()`.

## Start with a mask

### A grid in physical coordinates

`from_grid_mask()` accepts an N-dimensional boolean mask and explicit edges for
each coordinate axis. For a Cartesian 2-D grid, `active_mask[i, j]` selects the
cell between `x_edges[i:i+2]` and `y_edges[j:j+2]`.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

active_mask = np.array([[True, True], [True, False], [False, True]])
x_edges = np.arange(0.0, 8.0, 2.0)
y_edges = np.arange(0.0, 12.0, 4.0)
env = Environment.from_grid_mask(
    active_mask,
    grid_edges=(x_edges, y_edges),
    name="Analysis mask",
    units="cm",
)

assert env.n_bins == active_mask.sum()
np.testing.assert_array_equal(env.bin_sizes, 8.0)
```

This defines a continuous analysis domain from the active cells. It does not
assert that inactive cells are bounded by measured physical walls.

### An image or segmented floor plan

`from_pixel_mask()` accepts image indexing, `image_mask[row, column]`. Positions
remain Cartesian `(x, y)`: columns run along x and rows along y. A scalar
`pixel_size` gives square pixels; `(width, height)` sets x and y calibration.
Use the same origin, scale and coordinate directions for your tracking data.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

image_mask = np.array([[True, False, True], [False, True, False]])
env = Environment.from_pixel_mask(image_mask, pixel_size=(2.0, 3.0), units="cm")

assert env.n_bins == 3
np.testing.assert_array_equal(env.bin_sizes, 6.0)
assert env.contains(np.array([[1.0, 1.5], [5.0, 1.5]])).all()
```

The mask itself supplies the domain; you do not need to trace an additional
polygon. Its physical interpretation depends on how the mask was obtained.
The default `MaskArena` records computational support without claiming a
measured physical footprint.

## Immutable values and labels

Set `name`, `units` and `frame` at construction. These labels and the published
geometry are read-only. Units describe the values already supplied; declaring
`units="cm"` does not calibrate pixels or rescale coordinates.

To rename an analysis or correct a declaration, keep the returned environment:

```python
updated = env.with_metadata(name="Reviewed mask", frame="arena")
```

Omitted labels are preserved; explicit `None` clears `units` or `frame`.
Replacements share the immutable geometry and have separate analysis caches.
Use a factory, `subset`, `clip` or a supported transform for geometry changes.
Build named selections with `Regions` and publish them with `with_regions`;
the [region guide](regions.md) shows that workflow.

## Bring explicit bins

Use `from_bins()` when another tool already defines the bin centers, measures,
neighbor pairs and lookup rule. `SpatialBins` holds the bins and their measures,
`BinConnections` their neighbor pairs and navigation distances. Supply the
measures and distances explicitly; nearest-center lookup does not establish cell
boundaries or faces.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)

centers = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
bins = SpatialBins.from_arrays(
    centers=centers,
    sizes=np.array([1.0, 1.0, 1.0]),  # supplied areas, in cm squared
    measure_dim=2,
)
connections = BinConnections.from_arrays(
    pairs=np.array([[1, 2], [1, 0]]),
    pair_distances=np.array([3.0, 4.0]),  # supplied navigation path lengths
)
env = Environment.from_bins(
    Binning(
        kind="sample_sites",
        bins=bins,
        connections=connections,
        locator=NearestBinLocator(centers, max_distance=0.25),
        faces=MissingFaces("Only centers and navigation distances were measured."),
    ),
    units="cm",
    frame="lab",
)

np.testing.assert_array_equal(env.bin_at([[0.0, 0.0], [10.0, 0.0]]), [0, -1])
np.testing.assert_array_equal(env.edge_pairs, [[1, 2], [1, 0]])
graph = env.to_networkx()
graph.remove_node(0)  # editable export; the environment keeps its own graph
assert env.n_bins == env.connectivity.number_of_nodes() == 3
```

The environment owns its published snapshot. `edge_pairs` determines the order
and orientation of pair-aligned values, including flux when faces are available;
NetworkX edge iteration can have a different order. `MissingFaces` permits lookup
and graph navigation while numerical operations requiring cell faces raise.
Supply `KnownFaces` and consistent geometry when those operations are required.

For supplied path lengths whose paths are unknown, translations, rotations and
reflections preserve the lengths; uniform scaling scales them by the same
factor. Unequal axis scaling and shear raise because scalar path lengths do not
determine how those paths stretch. See [snapshot adapters](interoperability.md#installed-snapshot-adapters)
to convert an external provider into these built-in records.

`env.binning` returns an independent snapshot for inspecting or preparing geometry.
Changing its array shapes, writable flags or values cannot change the environment.
Reading it keeps native face recipes lazy; `from_bins()` builds them and checks
them against the bins, so a Cartesian snapshot rebuilds directly:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

original = Environment.from_grid_mask(
    np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
)
rebuilt = Environment.from_bins(original.binning, arena=original.arena)
np.testing.assert_array_equal(rebuilt.bin_centers, original.bin_centers)
```

`MissingFaces` already states that numerical geometry is unavailable and needs
no conversion. Polar snapshots retain radius/angle coordinates; use environment
storage to restore those, because `from_bins()` constructs Cartesian geometry.

For nearest-center assignment, use `env.nearest_bin(positions)`. It reuses the
environment's query tree and supports tie-breaking and distance limits.
This is different from `bin_at()`, which finds the cell that contains each
position (on a polar environment, each `(distance, angle)` row).
On an egocentric polar environment, `nearest_bin` measures physical distance:
it places the bin centers and the `(distance, angle)` rows in the animal's
plane, `(distance * cos(angle), distance * sin(angle))`, before comparing
them. Its distances and `max_distance` are in the environment's length units.
It is still not a containment test: use `bin_at()` for the bin that holds a
row.

For a custom numerical analysis, use `env.coordinate_system` to interpret the
columns and `env.get_diffusion_geometry()` to inspect conductances, cell measures
and connected diffusion components. These are independent read-only exports.
The environment keeps its expensive geometry and eigenbasis caches private.

`env.compute_gaussian_kernel(bandwidth=...)` returns unnormalized Gaussian weights
that ignore walls; `env.diffuse(field, bandwidth=..., mode=...)` respects the numerical boundary
geometry. After diffusion, `env.diffusion_accuracy(bandwidth=...)` reports its relative
noise floor. A spatial model can request a Markov random field smoothness basis
with `env.get_mrf_basis(occupancy, rank=...)`.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.environment import CartesianSystem

env: Environment[CartesianSystem] = Environment.from_grid_mask(
    np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
)
geometry = env.get_diffusion_geometry()
assert geometry.W.shape == (9, 9)
assert geometry.n_components == 1
weights = env.compute_gaussian_kernel(bandwidth=1.0)
assert not weights.flags.writeable
smoothed = env.diffuse(np.arange(env.n_bins, dtype=float), bandwidth=1.0, mode="average")
assert env.diffusion_accuracy(bandwidth=1.0) <= 1e-6
basis = env.get_mrf_basis(np.ones(env.n_bins), rank=4)
assert basis.B.shape == (env.n_bins, 4)
```

Geometry and Gaussian exports are new read-only objects over the cached,
immutable arrays, so repeated reads do not copy them. Use `.copy()` for
editable arrays or sparse matrices. MRF basis arrays are already independent
and editable. All of these come from the environment's one analysis cache.
`with_metadata` and `with_regions` share it, because their bins are the same;
derived environments and restored files start with an empty one (see
[Performance and caching](performance.md)).

## Add known geometry when available

`env.arena` describes continuous support; `env.bin_at(positions)` and
`env.contains(positions)` also require an active analysis bin. A subset keeps
its parent's arena, so a dropped bin can lie inside the arena while being
outside the subset. `env.arena.contains(positions)` asks about the continuous
domain alone when that domain is known.

A measured polygon can be supplied through `from_polygon()`, or through the
optional `arena=` argument of the sample and mask factories. Explicit polygon
selection keeps whole bins whose centers lie strictly inside the polygon;
lookup also rejects points outside its boundary or inside its holes. A curved or
diagonal wall leaves slivers inside the arena whose grid cells were not kept
(their centers lie outside); a position in a sliver belongs to the neighboring
bin, less than one cell away, so an animal running along the wall stays in
occupancy and spike counts. This selection does not change cell areas. Use
`clip()` below when exact intersected cell areas are needed.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import Point
from neurospatial import Environment

env = Environment.from_polygon(Point(0.0, 0.0).buffer(10.0), bin_size=2.0)
assert env.arena.physical_footprint_known
assert env.bin_at(np.array([[9.9, 2.0]]))[0] == -1
```

`OpenArena` separates a known polygon from explicitly supplied internal walls.
Walls block navigation and diffusion connections between the bins on either
side. A wall that covers part of a cell edge, such as a barrier ending partway
along it, leaves the open part of that edge connected. A measured wall rarely
lies exactly on bin edges, so the factories that build bins (`from_samples`,
`open_field`, `from_polygon`, `from_grid_mask`, `from_pixel_mask` and
`from_layout`) fit the two together with `wall_fit`:

- `wall_fit="snap"` (the default) keeps every bin and moves the wall onto the
  bin edges between the bins it divides, with a warning that says how far it
  moved, at most about half a bin. Paths, occupancy, smoothing and wall
  distances all use the moved wall, and every analysis works as on any grid.
- `wall_fit="split"` keeps the measured wall and cuts each bin it crosses into
  one bin per side. The split bins are exact polygons, so smoothing and other
  finite-volume operations refuse the bins next to a wall's end or along a
  diagonal wall. Use it for binned rates (`method="binned"`) or for
  wall-distance measures such as boundary-vector maps, where the exact wall
  matters.

Name each wall by passing a mapping, `walls={"divider": line}`, and errors,
warnings and saved files refer to it by that name (`arena.wall_names`). Walls
given as a sequence are named `"wall_0"`, `"wall_1"`, and so on.

A wall closes the neighbor pairs on either side of it rather than deleting
them. For a wall on bin edges (including a snapped one), the bins, their order
and the list of pairs (`env.binning.connections`) are the same as without the
wall, and only the pairs' `is_open` flags differ; `wall_fit="split"` makes new
bins, and the pieces on either side of the wall are closed pairs.
Paths, distances, smoothing and gradients use the open pairs, `env.edge_pairs`.
`env.get_blocked_edges()` lists each closed pair and the wall that closes it, a
quick check that a wall separates the bins you meant.

`from_bins`, `rebin` and saved environments keep the bins they are given, so a
wall through one of their bins raises an error naming the wall and the bin.
Declare walls in the original coordinates before clipping or transforming so
their source alignment is retained. A newly declared wall after those operations
can raise if coordinate precision prevents proving its clearance from a cell;
recenter coordinates or declare the wall before deriving the environment.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import LineString, box
from neurospatial import Environment
from neurospatial.arena import OpenArena

arena = OpenArena(box(0, 0, 4, 4), walls={"divider": LineString([(2, 0), (2, 4)])})
env = Environment.from_grid_mask(
    np.ones((4, 4), dtype=bool),
    grid_edges=(np.arange(5.0), np.arange(5.0)),
    arena=arena,
)
left = env.bin_centers[:, 0] < 2
assert all(left[u] == left[v] for u, v in env.connectivity.edges)

# A barrier at x = 2.4 runs through bins: by default it moves to x = 2.
import warnings

barrier = OpenArena(box(0, 0, 4, 4), walls={"barrier": LineString([(2.4, 0), (2.4, 4)])})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    snapped = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0), np.arange(5.0)),
        arena=barrier,
    )
assert "moved the walls onto bin edges by up to 0.4" in str(caught[0].message)
assert snapped.arena.wall_names == ("barrier",)

# The wall closes the pairs across it; the bins and their pair order stay.
blocked = env.get_blocked_edges()
assert set(blocked["wall"]) == {"divider"}
assert all(left[u] != left[v] for u, v in zip(blocked["source"], blocked["target"]))
split = Environment.from_grid_mask(
    np.ones((4, 4), dtype=bool),
    grid_edges=(np.arange(5.0), np.arange(5.0)),
    arena=barrier,
    wall_fit="split",
)
assert (snapped.n_bins, split.n_bins) == (16, 20)
```

For a linearized track, `from_graph(..., track_width=...)` supplies a measured
full width in the graph's coordinate units. Containment uses distance to the
actual centerline segments, with round end caps: a position farther than half
the width from every segment gets bin `-1`. A track built without a width has
no footprint, so `bin_at`, `contains`, `occupancy`, the bin sequences,
`points=` queries and point regions refuse with `NotImplementedError`.
`env.with_track_width(width)` gives such a track, including one saved by an
earlier version, its measured width and keeps its bins; `to_linear`, `linear_to_nd` and the graph
queries on bin indices still work. `to_linear` projects every position onto
the centerline, however far off the track, so check `bin_at(positions) >= 0`
on a track with a width before trusting linear positions.
Egocentric polar environments similarly record their radial and angular
analysis support without inferring physical walls.

## Optional: clip cells to a polygon

`env.subset(...)` selects whole bins. `env.clip(polygon)` intersects exact
planar cells with a polygon and the known arena domain. Polygons can contain
holes or disconnected regions. A source cell split into disconnected pieces
produces multiple output bins.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import MultiPolygon, box
from neurospatial import Environment

parent = Environment.from_grid_mask(
    np.ones((1, 1), dtype=bool), grid_edges=(np.array([0.0, 1.0]),) * 2
)
child = parent.clip(MultiPolygon([box(0, 0, 0.2, 1), box(0.8, 0, 1, 1)]))

np.testing.assert_allclose(child.bin_sizes, [0.2, 0.2])
assert child.clip_mapping is not None
np.testing.assert_array_equal(child.clip_mapping.parent_bin_indices, [0, 0])
assert child.arena.polygonal_domain().equals(parent.arena.polygonal_domain())
```

The mapping identifies each child's bin in the environment passed to the most
recent `clip()` call. Subsets select its rows and transforms preserve them;
a subsequent clip creates a new immediate-parent mapping. The array is
read-only and survives saving and loading.

Clipping preserves the name, units, frame, arena and named regions. It does
not transfer counts, occupancy, rates or fitted decoders. Reassign raw positions
and events to the new bins and fit again. A parent mapping alone cannot say
how activity was distributed between split pieces. Splitting bins also changes
the default equal-per-bin decoding prior; supply an area-weighted prior if the
scientific assumption is uniform physical density.

Containment and plotting retain the exact cell boundaries and holes. Numerical
operations such as smoothing, diffusion and spatial derivatives additionally
require certified finite-volume geometry: convex cells and a single open,
orthogonal interface between neighboring centers. Unsupported geometry raises
`NotImplementedError` with the reason. An arbitrary valid crop can therefore
be queried and plotted even when these numerical operations are unavailable.
A rectangular crop of a regular grid can retain them:

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box
from neurospatial import Environment

parent = Environment.from_grid_mask(
    np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
)
child = parent.clip(box(0.25, 0, 2, 2))
smoothed = child.smooth(np.ones(child.n_bins), bandwidth=0.5, mode="average")
np.testing.assert_allclose(smoothed, 1.0)
```

Clipping or transforming geometry refuses features that cannot be resolved at
the stored coordinate precision. Recenter or rescale coordinates before making
extremely thin cells at a large coordinate origin.

Use `bin_at()` for membership. Nearest-center interpolation remains an
approximation, especially for concave or clipped cells.

## Occupancy and position queries

`env.occupancy(times, positions)` measures time in seconds, rather than counting
samples. Its default assigns each valid interval to its starting bin. Use
`time_allocation="linear"` to split a straight segment over supported cells.
On polygon cells, this handles entry, exit and re-entry, excludes holes, and
does not renormalize the retained duration. Gaps, speed and epoch filters still
apply. A proper crossing of an explicit physical wall raises instead of
interpolating through it; endpoint contact and motion along a wall are allowed
when a bin owns those points.

Spatial encoding uses start allocation by default. Choose
`time_allocation="linear"` explicitly to use the same retained time support
for spike counts and dwell time. This option requires exact planar cells and
also works with environments built directly from samples or masks. Retained
time windows include entry and exclude exit, subject to spatial bin ownership.
Events at an internal shared cell boundary use its locator owner. The final
tracking timestamp remains excluded.
At an exact corner crossing, that owner can have zero dwell time. Linear rate
estimation raises if a spike lands in such a bin: a positive count divided by
zero exposure has no finite firing rate. Inspect the boundary observation and
choose an appropriate spatial or tracking resolution before fitting again.
If absolute timestamps cannot represent a retained time span, subtract the same
origin from tracking, spike times and any time windows before analysis; this raises
explicitly rather than publishing zero-width observed intervals.

In this simulated straight traversal, only four seconds and four spikes lie
in the retained cell. The hole and outside portions contribute to neither
counts nor dwell time:

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import Polygon, box
from neurospatial import Environment, compute_spatial_rates
from neurospatial.decoding import BayesianDecoder

parent = Environment.from_grid_mask(
    np.ones((1, 1), dtype=bool),
    grid_edges=(np.array([0.0, 6.0]), np.array([0.0, 2.0])),
)
region = Polygon(
    box(0, 0, 6, 2).exterior.coords,
    holes=[box(2, 0.5, 4, 1.5).exterior.coords],
)
env = parent.clip(region)
times = np.array([0.0, 8.0])
positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
spikes = [np.arange(0.5, 8.0, 1.0)]
rates = compute_spatial_rates(
    env, spikes, times, positions,
    method="binned", bandwidth=0, time_allocation="linear",
    max_gap=None,  # This synthetic segment is known to be continuous.
)
np.testing.assert_allclose(rates.occupancy, [4.0])
np.testing.assert_allclose(rates.firing_rates, [[1.0]])
decoder = BayesianDecoder.from_rates(rates, dt=1.0)
# Prediction can use its own continuous time grid without tracking positions.
decoded = decoder.predict(spikes, np.arange(0.0, 8.25, 0.25))
np.testing.assert_allclose(decoded.posterior.sum(axis=1), 1.0)
```

For an explicit mask whose cell edges differ from the analysis grid, linear
subdivision currently requires 2-D polygon support. Higher-dimensional masks
must match the grid support for linear allocation; otherwise the method raises
with guidance to use start allocation or a matching grid.

Other common queries are `env.neighbors(bins=bin_index)`,
`env.path_between(bins=[start_bin, end_bin])` and
`env.distance_between(points=[point_a, point_b])`. Each query takes its target
by keyword: `bins=` for bin indices, `points=` for coordinates (one row per
point, mapped to the bin that contains it) and, for `distance_to`, `region=`
for a named region. A bin index and a coordinate look alike as numbers, so the
keyword says which one you mean. See the [API reference](../api/index.md) for
their coordinate and distance conventions.

## Choose physical walls or analysis boundaries

Physical border scores use asserted footprint rings (including holes) and
explicit internal walls. The footprint must be known: internal walls alone do
not say where the outer walls are, so an arena declared with
`OpenArena(..., physical_footprint_known=False)` refuses physical scoring even
when it has internal walls. A crop or a missing position sample does not create a
wall. If only tracking-derived support or an analysis mask is available, request
`boundary_source="analysis"` to use the legacy boundary-bin approximation.
Physical mode is the default. It requires measured wall geometry and retained
cells with finite firing rates along every measured wall segment. Missing wall
observations cannot establish that firing was zero near the wall. Supplying
`OpenArena` establishes wall locations; it does not add bins or observations.
Physical scoring refuses when either geometry or this observation support is
missing. A valid zero rate near the wall still contributes to a negative score
for a field away from the wall.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box
from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import border_score

env = Environment.from_grid_mask(
    np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2,
    arena=OpenArena(box(0, 0, 10, 10)), units="cm",
)
field = (env.bin_centers[:, 0] < 1).astype(float)
score = border_score(env, field, metric="euclidean")
assert 0 < score < 1
approximation = border_score(env, field, boundary_source="analysis")
assert approximation == 1.0
```

Physical coverage is the fraction of declared wall length covered by field
cells. Distance is mean field-center distance to actual walls, normalized by
the wall diameter (maximum point separation). Geodesic mode includes center-to-wall lengths
and paths through retained graph bins. An unavailable route makes the score
NaN; crops can restrict represented routes. Euclidean mode avoids that graph
restriction. This is a generalized screening metric; existing fixed thresholds
are not a calibrated significance test for this geometry.

For a rotated rectangular arena, construct bins in coordinates aligned with the
measured rectangle, then transform the bins and arena together. Transform the
recorded positions into the same frame before computing rates. A change of
frame then preserves the represented wall coverage:

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box
from neurospatial import Environment
from neurospatial.encoding import border_score
from neurospatial.ops.transforms import Affine2D

env = Environment.from_polygon(box(0, 0, 10, 10), bin_size=1.0)
field = (env.bin_centers[:, 0] < 1).astype(float)
score = border_score(env, field, metric="euclidean")
angle = np.pi / 4
transform = Affine2D(np.array([
    [np.cos(angle), -np.sin(angle), 20.0],
    [np.sin(angle), np.cos(angle), 10.0],
    [0.0, 0.0, 1.0],
]))
rotated = env.apply_transform(transform)
np.testing.assert_allclose(border_score(rotated, field, metric="euclidean"), score)
recorded_positions = np.array([[0.5, 0.5], [0.5, 1.5]])
rotated_positions = transform(recorded_positions)
assert np.all(rotated.bin_at(rotated_positions) >= 0)
```

Rebuilding an axis-aligned grid from a rotated polygon can leave slanted wall
segments outside the selected bins. For other arena shapes, use exact cells
that cover the measured walls and collect observations in their wall-adjacent
cells. `clip()` only cuts existing cells; it cannot restore missing bins.

The same choice is available on `result.border_score()`, population
`border_scores()` and `label_cell_types()`, and `summary_table()`. Tables retain
other metrics when wall geometry is unavailable: border scores are NaN, with a
reason in `table.attrs["unavailable_metrics"]` and the chosen source in
`table.attrs["border_boundary_source"]`. A spatially tuned cell that is not a
grid cell is then labeled `"undetermined"` rather than `"place"`, because it may
be a border cell. No analysis approximation is selected
implicitly. Physical border coverage currently requires exact planar cells and
polygonal walls; curved track footprints require a separate representation.
Independently transformed combined components also refuse physical contact
until a shared-frame contact certificate is available; combine before applying
one shared transform, or explicitly choose the analysis approximation.
A combined environment also needs a measured footprint for every component: a
field at the edge of a component without one would otherwise be scored against
walls elsewhere, so physical scoring refuses and names the component.

## Compute gaze and visibility against walls

Ray casting and cue visibility use the same `boundary_source` choice. Physical
mode reads declared footprint rings and internal walls; it does not stop when a
ray crosses an analysis crop or an unvisited bin. Analysis mode explicitly uses
analysis-cell boundaries, constrained by the known domain and internal walls.

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box, LineString
from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.ops.visibility import compute_viewed_location, visible_cues

env = Environment.from_grid_mask(
    np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2,
    arena=OpenArena(box(0, 0, 10, 10), walls=(LineString([(5, 0), (5, 10)]),)),
)
crop = env.subset(mask=env.bin_centers[:, 0] < 4)
hit = compute_viewed_location(
    np.array([[2.5, 4.5]]), np.array([0.0]), env=crop,
    method="ray_cast", max_distance=30,
)
np.testing.assert_allclose(hit, [[5.0, 4.5]])  # real wall, beyond the analysis crop
approximate = compute_viewed_location(
    np.array([[2.5, 4.5]]), np.array([0.0]), env=crop,
    method="ray_cast", max_distance=30, boundary_source="analysis",
)
np.testing.assert_allclose(approximate, [[4.0, 4.5]])
visible, _, _ = visible_cues(
    crop, np.array([2.5, 4.5]), 0.0, np.array([[5.0, 4.5], [7.5, 4.5]]),
)
np.testing.assert_array_equal(visible, [True, False])  # on the wall; behind it
```

`compute_viewed_location()` returns the first wall contact. No hit within
`max_distance`, a nonfinite observation, or an observer outside an asserted
footprint yields NaN. An observer tracked exactly on a wall sees the next wall
when it faces into the room; facing out through the wall, or exactly along it,
gives a zero-distance contact at the observer. On an internal wall, the
observer sees the side it faces. A cue at the first contact is visible; a cue
behind it is occluded. Stored affine frames
use locally bounded projection errors for boundary contact. Physical mode only
uses supplied wall evidence; inferred domains do not add unasserted outer walls.

`compute_viewshed()` still samples rays to determine visible-bin coverage;
wall intersections and cue line of sight use geometric intersections. Missing
analysis bins do not occlude physical rays, and the sampled bins stop strictly
before wall contact. Visibility maps remain an approximation whose angular
coverage depends on `n_rays`.

The source choice is forwarded by the single/population view encoders, shuffle
tests and `SpatialViewCellModel`. With `gaze_model="ray_cast"` or `"boundary"`,
encoding assigns a wall hit to the bin just inside the observer's side. A hit
outside the retained analysis support makes physical view encoding unavailable
and raises with guidance when it contributes to the selected recording time.
The final sample starts no exposure interval, and hits during excluded epochs,
recording windows, long gaps or zero-duration intervals cannot change the map.
Supplying wall geometry cannot add missing bins, and
assigning a distant wall to a nearby bin would change the measured location.
Build cells that reach the wall, or explicitly request the analysis-boundary
approximation. `from_polygon()` keeps whole cells whose centers lie inside, so
its cells stop short of a curved or slanted wall. To reach such a wall, clip a
grid that covers the arena to the measured polygon and attach its walls:
`Environment.from_bins(grid.clip(polygon).binning, arena=OpenArena(polygon))`.
The geometry-only `compute_viewed_location()` still returns the known physical
hit even when its location has no analysis bin.

View encoding requires a retained bin for each observed finite hit; it does not
require gaze observations at every wall. Unviewed bins remain missing, and rays
with no hit or invalid observations remain missing. Spike counts and exposure
share that assignment, and results record the source in `analysis_settings`.
Encoder/model searches cover the known geometry rather than imposing a fixed
100-unit range. Fixed-distance gaze keeps its existing behavior.

Physical encoding of independently transformed combined components requires a
shared certified owner frame; unsupported cases refuse with guidance. Curved
track footprints still require a separate exact wall representation. For a
position-derived environment with supported planar cells, explicitly select
`boundary_source="analysis"` when that approximation is the intended model.

## Change bin size

`env.rebin(bin_size=...)` builds a new rectangular lattice over the bins the
environment has, within its arena: rebinning a subset keeps the selection
rather than bringing back the space it left out. It keeps the original grid
frame, lattice anchor, units, frame label and regions. The argument is a physical length along each retained grid axis;
a tuple gives different lengths along different axes. It can refine or coarsen.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

env = Environment.from_grid_mask(
    np.ones((6, 6), dtype=bool),
    grid_edges=(np.arange(7.0),) * 2, units="cm", frame="lab",
)
crop = env.subset(mask=env.bin_centers[:, 0] < 2)
coarse = crop.rebin(bin_size=2.0)
assert coarse.n_bins == 3  # the cropped column, at the new size
np.testing.assert_allclose(coarse.bin_sizes, 4.0)
assert coarse.rebin(bin_size=1.0).n_bins == crop.n_bins
```

Whole-bin selection keeps new cells that cover the environment's bins and whose
centers lie inside the arena; it does not promise exact clipped areas. Original bins in a new perimeter cell whose center lies outside
the arena take the neighboring new bin, like any position in a sliver. When
original bins are still left without a new bin (beyond the new grid, or in a
cell a subset removed),
`rebin` warns with their count and share of the area, because positions there
map to `-1` and drop out of occupancy and spike totals. Physical walls must
still follow complete new cell faces. The arena outline is not a wall: new bins
on both sides of a barrier narrower than a bin, such as the wall between two
maze arms, stay connected, and `rebin` warns because path lengths and smoothing
then pass through the barrier. Declare the barrier's edges as internal walls,
`OpenArena(polygon, walls=...)`, to remove those connections. Use `clip()`
separately, or a finer grid, where needed. Re-binning
creates new bin IDs and does not copy rates, counts or fitted models.

New rectangular environments store an owned `GridBinningSpec` independently of
provenance. It survives cropping, clipping, transforms and JSON/NWB storage.
Hexagons, tracks, meshes, combined environments and older records without a
supported specification refuse new-size re-binning. Explicit-bin providers can
supply a `GridBinningSpec` describing future rectangular construction.
The existing `env.rebin(factor=2)` remains available for uniform regular grids,
including its truncation warning. Supply exactly one of `factor` and `bin_size`.

## Carry values to a subset or a coarser grid { #bin-mapping }

Two common questions need values moved between environments: does the map
estimated on part of the arena match the full map there, and what is the
occupancy at a coarser resolution? Each derived environment numbers its bins
from 0, so bin `j` of a subset is not bin `j` of the arena.
`child.bin_mapping_from(parent)` pairs the bins from their geometry. It reports
only pairs it can prove: a bin that is the same cell as a parent bin (a subset,
the cells a clip did not cut), or a fine cell that lies wholly inside one cell of
a coarser grid on the same lattice (`rebin`). The simulated session below has a
random trajectory over a 40 x 40 cm box and two cells with random spikes.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates

times = np.arange(18_000) / 30.0  # 10 minutes at 30 Hz
positions = np.random.default_rng(0).uniform(0.0, 40.0, size=(len(times), 2))  # cm
spike_rng = np.random.default_rng(1)
spike_times = [np.sort(spike_rng.choice(times, size=n, replace=False)) for n in (400, 900)]

env = Environment.from_grid_mask(
    np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11) * 4.0,) * 2, units="cm"
)


def binned_rates(where):
    return compute_spatial_rates(
        where, spike_times, times, positions,
        method="binned", bandwidth=0.0, unit_ids=[1, 2],
    )


# Compare the map estimated on the left half with the full map there.
left = env.subset(mask=env.bin_centers[:, 0] < 20)
to_left = left.bin_mapping_from(env)
print(to_left)
full_map, left_map = binned_rates(env), binned_rates(left)
on_left = compare_spatial_rates(
    full_map, left_map, bin_mapping=to_left.pairs, topology="may_differ"
)
assert on_left.status.tolist() == ["scored", "scored"]

# Coarsen counts and occupancy to 8 cm bins.
coarse = env.rebin(2)
to_coarse = coarse.bin_mapping_from(env)
occupancy = to_coarse.sum(full_map.occupancy)  # seconds per 8 cm bin
counts = np.where(
    full_map.occupancy > 0, full_map.firing_rates * full_map.occupancy, 0.0
)  # spikes per 4 cm bin; unvisited bins have no spikes
rates = to_coarse.rates(counts, full_map.occupancy)  # summed spikes / summed time
np.testing.assert_allclose(occupancy, coarse.occupancy(times, positions))
np.testing.assert_allclose(rates, binned_rates(coarse).firing_rates)
```

The printed mapping counts each kind of bin: all 50 bins of the left half pair
one-to-one with arena bins, and the other 50 arena bins are unmapped. Its
`pairs` are rows of `(bin in env, bin in left)`, the orientation
`compare_spatial_rates` takes. After `rebin(2)` every 8 cm bin holds four 4 cm
bins, so summed occupancy and pooled rates equal estimating on the coarse grid
from the same trajectory. Rates are pooled from counts and occupancy, never
averaged: 0 Hz for 100 s and 100 Hz for 1 s pool to 100 spikes / 101 s =
0.99 Hz, not 50 Hz. A smoothed map's value in a bin borrows spikes from its
neighbors, so it is not a count that adds up; estimate smoothed maps again on
the new environment. `to_coarse.category(labels, missing="mixed")` carries
labels such as region names, keeping a label only where all the fine bins in a
coarse bin agree.

The mapping never guesses. Missing values stay missing: a coarse bin with any
NaN fine bin is NaN. A clip's cut cells are listed in `split_parent_bins` and
their pieces in `uncovered_child_bins`, which get NaN, because a count cannot
be divided between pieces without knowing where in the cell each observation
fell. A finer grid, a rotated or offset lattice, or an independently built
environment raises `ValueError`; re-estimate on the new environment from the
original observations, or use `neurospatial.ops.map_probabilities` and
`ops.resample_field`, which assign bins by nearest center and are
approximations. `apply_transform` needs no mapping, because transforms keep bin
IDs. When a truncating `rebin` drops fine bins, occupancy summed from the fine
grid equals the coarse grid's own occupancy only if positions in the dropped
bins are excluded from both: an interval that ends in a dropped bin counts on
the fine grid but not on the coarse one.

## Name cues, monitors and objects { #landmarks }

Landmarks are the things the animal sees or approaches: a cue card, a monitor,
an object, a reward port. Give each one a name and its geometry in the
environment's coordinates: a point, a line (a monitor or a card along a wall)
or a polygon (an object's footprint). Landmarks are not bins, so they may lie
outside the arena, as a cue on the room wall does, and they neither block
movement nor block the view; declare a wall for that. The name is the
landmark's identity: keep it the same in every session, so a cue rotated or an
object moved between sessions is still the same landmark.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.landmarks import Landmark, Landmarks

edges = np.arange(0.0, 101.0, 5.0)
arena = Environment.from_grid_mask(
    np.ones((20, 20), dtype=bool), grid_edges=(edges, edges), units="cm"
)
arena = arena.with_landmarks(
    Landmarks(
        [
            Landmark("cue_card", "point", [50.0, 108.0], {"role": "cue"}),
            Landmark("monitor", "line", [(10.0, 100.0), (90.0, 100.0)]),
            Landmark("object", "polygon", [(20, 20), (30, 20), (30, 30), (20, 30)]),
        ]
    )
)
# Each landmark's nearest point to the animal, shape (n_time, n_landmarks, 2):
# a line or polygon is measured from its closest edge, not its center.
positions = np.array([[50.0, 90.0], [15.0, 40.0]])
nearest = arena.landmarks.nearest_points(positions, ["monitor", "object"])
assert nearest[0, 0].tolist() == [50.0, 100.0]

# Whether projected gaze lands on the monitor, within 2 cm:
gaze = np.array([[60.0, 99.5], [60.0, 80.0]])
assert arena.landmarks.near(gaze, "monitor", tolerance=2.0).tolist() == [True, False]
```

Encoding functions that take `object_positions` accept landmark names
instead of coordinates, and record the names in the result; see
[object vectors](neuroscience-metrics.md#allocentric-and-egocentric-object-vectors).
`subset`, `clip` and `rebin` keep the landmarks, `apply_transform` moves them
with the bins, and `combine` prefixes their names with the component key, as
it does for regions. A polar environment, centered on the animal, holds none.

## Check the environment against the apparatus { #overview }

`env.plot()` draws everything the environment holds: the bins and the
connections between them, the declared walls (red, named), the landmarks
(orange, named) and the regions (in the legend). Compare it with a video frame
or a sketch of the rig before analyzing: a wall on the wrong side or a cue in
the wrong corner shows up here, but not in a rate map. It draws a
hundred-thousand-bin grid in about a second. `show_walls=False`,
`show_landmarks=False` and `show_regions=False` hide each layer.
`env.plot(backend="fastplotlib")` draws the same picture on the GPU, where you
can pan, zoom and rotate a 3-D environment, in Jupyter or from a remote GPU
server; see [the fastplotlib backend](animation.md#fastplotlib-backend-gpu-local-or-remote).

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import LineString, box
from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.landmarks import Landmark, Landmarks

edges = np.arange(0.0, 101.0, 5.0)
divided = Environment.from_grid_mask(
    np.ones((20, 20), dtype=bool),
    grid_edges=(edges, edges),
    units="cm",
    arena=OpenArena(box(0, 0, 100, 100), walls={"divider": LineString([(50, 0), (50, 70)])}),
)
divided = divided.with_landmarks(
    Landmarks(
        [
            Landmark("cue_card", "point", [50.0, 108.0]),
            Landmark("monitor", "line", [(10.0, 100.0), (90.0, 100.0)]),
        ]
    )
)
ax = divided.plot()
```

## Save the environment with the analysis

`env.to_file(path)` writes JSON metadata and NPZ arrays;
`env.to_dict()` supplies an in-memory equivalent, and
`neurospatial.io.nwb.write_environment` stores the same record in an NWB file.
They preserve the arena, named walls, regions, landmarks, exact cells,
coordinate metadata and supported numerical geometry without running a layout
builder on load.

Environments saved by earlier versions of neurospatial do not load: recreate
them from their source inputs and save them again.

## See also

- [Layout engines](layout-engines.md): choose discretization strategies.
- [Regions](regions.md): define named analysis regions.
- [API reference](../api/index.md): full parameters and methods.
