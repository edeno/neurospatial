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
)
env.units = "cm"

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
env = Environment.from_pixel_mask(image_mask, pixel_size=(2.0, 3.0))
env.units = "cm"

assert env.n_bins == 3
np.testing.assert_array_equal(env.bin_sizes, 6.0)
assert env.contains(np.array([[1.0, 1.5], [5.0, 1.5]])).all()
```

The mask itself supplies the domain; you do not need to trace an additional
polygon. Its physical interpretation depends on how the mask was obtained.
The default `MaskArena` records computational support without claiming a
measured physical footprint.

## Add known geometry when available

`env.arena` describes continuous support; `env.bin_at(positions)` and
`env.contains(positions)` also require an active analysis bin. A subset keeps
its parent's arena, so a dropped bin can lie inside the arena while being
outside the subset. `env.arena.contains(positions)` asks about the continuous
domain alone when that domain is known.

A measured polygon can be supplied through `from_polygon()`, or through the
optional `arena=` argument of the sample and mask factories. Explicit polygon
selection keeps whole bins whose centers lie strictly inside the polygon;
lookup also rejects points outside its boundary or inside its holes. This
selection does not change cell areas. Use `clip()` below when exact intersected
cell areas are needed.

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
Walls block navigation and diffusion connections. The current wall policy
requires walls to follow complete cell faces; a wall through a cell interior
or part of a face raises with guidance to refine or align the grid. Align
openings with cell vertices when representing doorways.
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

arena = OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),))
env = Environment.from_grid_mask(
    np.ones((4, 4), dtype=bool),
    grid_edges=(np.arange(5.0), np.arange(5.0)),
    arena=arena,
)
left = env.bin_centers[:, 0] < 2
assert all(left[u] == left[v] for u, v in env.connectivity.edges)
```

For a linearized track, `from_graph(..., track_width=...)` supplies a measured
full width in the graph's coordinate units. Containment uses distance to the
actual centerline segments, with round end caps. Omit the width for an abstract
graph whose physical footprint is unknown; its existing graph lookup remains
available, but `env.arena.contains(...)` cannot establish physical containment.
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
assert child.arena is parent.arena
```

The mapping identifies each child's bin in the environment passed to the most
recent `clip()` call. Subsets select its rows and transforms preserve them;
a subsequent clip creates a new immediate-parent mapping. The array is
read-only and survives saving and loading.

Clipping preserves units, frame and arena, and drops analysis regions. It does
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
smoothed = child.smooth(np.ones(child.n_bins), bandwidth=0.5)
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

Other common queries are `env.neighbors(bin_index)`,
`env.path_between(source_bin, target_bin)` and
`env.distance_between(point_a, point_b)`. See the
[API reference](../api/index.md) for their coordinate and distance conventions.

## Save the environment with the analysis

`env.to_file(path)` writes JSON metadata and NPZ arrays;
`env.to_dict()` supplies an in-memory equivalent. Both use `Environment-v3`.
NWB environment records use schema `3.0`. They preserve the arena, explicit
walls, exact cells, clip mapping, coordinate metadata and supported numerical
geometry without running a layout builder on load.

Version 2 files/dictionaries and NWB schema `2.0` remain readable with an
explicitly unknown arena. Their stored bins and capabilities remain available,
but an old outline or active mask does not establish a measured footprint.
Recreate the environment from its source inputs with a known arena if the
analysis requires that information, then save it again. Version 1 files and
NWB schemas `1.0`/`1.1` must be recreated before loading with this version.

## See also

- [Layout engines](layout-engines.md): choose discretization strategies.
- [Regions](regions.md): define named analysis regions.
- [API reference](../api/index.md): full parameters and methods.
