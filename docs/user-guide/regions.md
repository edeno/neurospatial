# Regions

Use named regions to mark reward zones, maze arms or other areas relevant to an
analysis. A region is an analysis label; it does not change the arena, bin
geometry or physical walls. Regions may extend outside the arena.

## Build regions before publishing them

`Regions` is a mutable builder. An environment stores an immutable snapshot, so
editing the builder later cannot change an encoder or decoder using that
environment.

```python
import numpy as np
from shapely.geometry import Point, Polygon

from neurospatial import Environment, Regions

regions = Regions()
regions.add("start", point=(5.0, 5.0))
regions.add("reward", polygon=Point(25.0, 25.0).buffer(5.0))
regions.add("arm", polygon=Polygon([(0, 0), (10, 0), (10, 30), (0, 30)]))

rng = np.random.default_rng(0)
positions = rng.uniform(0.0, 40.0, size=(200, 2))
env = Environment.from_samples(
    positions, bin_size=5.0, units="cm", frame="arena", regions=regions
)
```

Point coordinates have shape `(n_dims,)`. Polygon regions use a Shapely
`Polygon` and require a 2D environment. Region coordinates must use the same
units and frame as the environment. Metadata must be JSON serializable and is
stored with immutable nested containers.

## Query the published snapshot

```python
names = env.regions.list_names()          # Insertion order
reward = env.regions["reward"]
area = env.regions.area("reward")        # In coordinate units squared
center = env.regions.region_center("reward")
reward_bins = env.bins_in_region("reward")
```

Point coordinates are read-only. Use `region.data.copy()` when you need an
editable array; editing that copy does not update the region.

## Regions on an egocentric polar environment

An environment from `Environment.from_polar_egocentric` takes positions as
`(distance, angle)` rows, angles in radians (0 = ahead, +π/2 = left). Regions
use the same coordinates: a polygon's x is distance and its y is angle, so a
box is a wedge of a distance band.

A polygon is not wrapped around the circle. It selects bins whose center
angles, inside the environment's angle range, lie within it. A wedge behind the
animal that crosses the ±π seam therefore needs two polygons, one at each end
of the range; `region_mask` of both gives the whole wedge. A point region is
located like any position, with its angle wrapped into the range.

```python
from shapely.geometry import box

ego = Environment.from_polar_egocentric(
    distance_range=(0.0, 20.0),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=10.0,
    angle_bin_size=np.pi / 4,
)
behind = Regions()
behind.add("behind_left", polygon=box(0.0, 3 * np.pi / 4, 20.0, np.pi))
behind.add("behind_right", polygon=box(0.0, -np.pi, 20.0, -3 * np.pi / 4))
ego = ego.with_regions(behind)
behind_mask = ego.region_mask(["behind_left", "behind_right"])  # 4 bins
```

## Replace regions on an existing environment

Start from the current values when you want to retain other names. Build all
edits, then publish once:

```python
edited = Regions(env.regions.values())
edited.update_region("start", point=(8.0, 5.0))
edited.remove("arm")
edited.buffer("start", distance=3.0, new_name="start_zone")
updated = env.with_regions(edited)

assert "arm" in env.regions
assert "arm" not in updated.regions
```

`with_regions` replaces the whole collection and returns a new environment.
It preserves the bin geometry and labels. Pass `Regions()` to clear the
collection. Direct edits such as `env.regions.add(...)` raise an error; keep
the returned environment for subsequent analysis.

The builder's `add` and mapping assignment reject duplicate names. Use `set`
to replace a named `Region`, or `update_region` to change an existing region's
geometry or metadata. Standalone builders also support `remove`, `buffer`, JSON
serialization and DataFrame export.

See the [core concepts](../getting-started/core-concepts.md) and
[API reference](../api/index.md) for related operations.
