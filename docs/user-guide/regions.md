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
