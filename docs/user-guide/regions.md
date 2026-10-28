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
units and frame as the environment.

A region is checked when it is built. Coordinates must be finite, and a
polygon must enclose a positive area without crossing itself. A hand-drawn
outline that loops over itself (a "bowtie") is refused with the reason and a
suggested repair (`shapely.make_valid`, or redrawing it), because the bins it
would cover are not well defined. Metadata must be standard JSON: string keys,
finite numbers, and `None` for a missing value. It is stored read-only.

## Query the published snapshot

`env.regions` is a read-only mapping from names to regions, in the order they
were added:

```python
from neurospatial.regions import regions_table

names = list(env.regions)                 # ["start", "reward", "arm"]
reward = env.regions["reward"]
area = reward.area                        # In coordinate units squared
center = reward.center                    # Centroid, shape (2,)
reward_bins = env.bins_in_region("reward")
table = regions_table(env.regions)        # One row per region: kind, area, center
```

A bin belongs to a polygon region when its center lies inside the polygon, and
to a point region when the point lies inside the bin. A polygon's `center` is
its centroid, which can lie outside a non-convex region such as a U-shaped
arm, so it is not necessarily a place the animal can be.

Point coordinates are read-only. Use `region.data.copy()` when you need an
editable array; editing that copy does not update the region.

## Test raw positions against regions

`env.bins_in_region` and `env.region_membership` answer which bins a region
covers. To ask which tracked positions fall inside a region without binning
them, use `regions_containing_points` or `points_in_any_region`:

```python
from neurospatial.regions import points_in_any_region, regions_containing_points

membership = regions_containing_points(
    positions, env.regions, region_names=["reward", "arm"]
)  # DataFrame: one bool column per region, one row per position
fraction_in_reward = membership["reward"].mean()
```

Positions must be finite. A tracking dropout is not a sample outside every
region, so NaN rows are refused; drop or interpolate them first, for example
with `positions[np.isfinite(positions).all(axis=1)]`.

A point region has no area, so testing positions against it needs a distance
you choose, in the regions' units. Pass `point_radius=`, for example the radius
of a reward well:

```python
near_any = points_in_any_region(positions, env.regions, point_radius=3.0)
```

## Regions drawn on a video frame

Regions drawn in LabelMe or CVAT, or traced from a binary mask, start in image
pixels: x is the column and y the row, increasing downward, with pixel
`(x, y)` centred on `(x, y)` as in a video frame array (both tools store
pixel corners; the readers shift their shapes by half a pixel). The
[video annotation guide](video-annotation.md) shows the calibrated path,
`regions_from_labelme(..., calibration=...)` and
`regions_from_cvat(..., calibration=...)`, which converts pixels to
centimeters and flips the y axis. The lower-level readers take any
`pixel_to_world` transform:

```python
import json
from pathlib import Path

from neurospatial.ops.transforms import simple_scale
from neurospatial.regions import load_labelme_json, plot_regions

Path("arena.json").write_text(
    json.dumps(
        {
            "imagePath": "frame_0001.png",
            "shapes": [
                {
                    "label": "reward",
                    "shape_type": "rectangle",
                    "points": [[200.0, 200.0], [300.0, 260.0]],
                }
            ],
        }
    )
)
drawn = load_labelme_json("arena.json", pixel_to_world=simple_scale(px_per_cm=10.0))
assert drawn["reward"].area == 60.0       # cm^2: 10 cm by 6 cm
ax = plot_regions(drawn)                  # Check the outline before using it
```

LabelMe polygons, rectangles, circles and points become regions. An open line
is refused because it encloses no area; store it as a
`neurospatial.Landmark(name, "line", points)` instead. A label used more than
once is numbered `label_0`, `label_1`, ... in file order. A CVAT export with
several annotated images needs `image=` to pick one.

A region loaded without a transform records `metadata["coord_system"] =
"pixels"`. An environment in centimeters refuses it and says how to convert
it, so pixel coordinates cannot silently cover the wrong bins.

## Regions on an egocentric polar environment

An environment from `Environment.from_polar_egocentric` takes positions as
`(distance, angle)` rows, angles in radians (0 = ahead, +π/2 = left). Regions
use the same coordinates: a polygon's x is distance and its y is angle, so a
box is a wedge of a distance band.

Angles are compared modulo a full turn, as for positions. A polygon covers a
bin when the bin's center angle, plus any whole number of turns, lies inside
it, so a wedge behind the animal that crosses the ±π seam is one polygon from
3π/4 to 5π/4. A point region is located like any position, with its angle
wrapped into the range.

```python
from shapely.geometry import box

ego = Environment.from_polar_egocentric(
    distance_range=(0.0, 20.0),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=10.0,
    angle_bin_size=np.pi / 4,
)
behind = Regions()
behind.add("behind", polygon=box(0.0, 3 * np.pi / 4, 20.0, 5 * np.pi / 4))
ego = ego.with_regions(behind)
behind_mask = ego.region_mask("behind")  # 4 bins, on both sides of the seam
```

## Replace regions on an existing environment

Start from the current values when you want to retain other names. Build all
edits, then publish once:

```python
edited = Regions(env.regions.values())
edited.update_region("start", point=(8.0, 5.0))
edited.update_region("reward", metadata={"reward": "milk"})  # Geometry kept
edited.remove("arm")
edited.buffer("start", distance=3.0, new_name="start_zone")
updated = env.with_regions(edited)

assert list(updated.regions) == ["start", "reward", "start_zone"]
assert "arm" in env.regions
```

`with_regions` replaces the whole collection and returns a new environment.
It preserves the bin geometry and labels, and takes any mapping from names to
regions. Pass `Regions()` to clear the collection. `env.regions` cannot be
edited: `env.regions.add(...)` raises an error, and a type checker flags it.
Keep the returned environment for subsequent analysis.

The builder's `add` and mapping assignment reject duplicate names. Use `set`
to replace a named `Region`, or `update_region` to change an existing region's
geometry, its metadata, or both. The region keeps its place in the order, so
the columns of `env.region_membership()` stay put.

## Save and load regions

```python
from neurospatial.regions import regions_from_json, regions_to_json

regions_to_json(env.regions, "regions.json")
restored = regions_from_json("regions.json")
assert list(restored) == list(env.regions)
```

The file is standard JSON tagged with its format version; a file without that
tag is refused rather than guessed at. Environments saved to NWB keep their
regions as well.

See the [core concepts](../getting-started/core-concepts.md) and
[API reference](../api/index.md) for related operations.
