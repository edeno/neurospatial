# One Environment for Cartesian and polar coordinates

Cartesian rooms and tracks and egocentric polar bins now use the same public
`Environment` class. Choose the factory for the data you have. The factory sets
what the coordinates mean; returning the same class does not make angle and
length interchangeable.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

room = Environment.from_grid_mask(
    np.ones((2, 2), dtype=bool),
    grid_edges=(np.arange(3.0),) * 2,
    units="cm",
)
polar = Environment.from_polar_egocentric(
    distance_range=(0.0, 3.0),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=1.0,
    angle_bin_size=np.pi / 2,
    circular_angle=True,
    units="cm",
)
assert type(room) is type(polar) is Environment
assert room.coordinate_kind == "cartesian"
assert polar.coordinate_kind == "polar"
assert hasattr(polar, "bin_at")  # visible; calling it raises CartesianOnlyError
assert polar.smooth(
    np.ones(polar.n_bins), bandwidth=1.0, mode="average"
).shape == (polar.n_bins,)
```

Polar coordinates are `(distance, angle)`: distance uses the declared length
unit and angle is in radians. Zero angle is ahead of the animal; positive angles
point to its left. Shared graph and field operations keep their existing polar
math. Cartesian lookup, affine transforms, clipping and Cartesian trajectory
operations still refuse polar inputs. Domain functions that read bin centers as
Cartesian coordinates, such as `resample_field` and
`spatial_autocorrelation`, refuse them too; functions with a graph-based mode,
such as `rate_map_centroid(method="geodesic")`, keep that mode for polar bins.
Decoded posterior means work on polar environments: `mean_position` is the
centroid in the animal's egocentric plane, returned as `(distance, angle)`.
Type checkers infer that restriction from the factory return type; runtime
checks also cover explicit unbound calls.

An unparameterized `Environment` annotation accepts either coordinate kind for
shared operations. Loading from a file, dictionary or NWB record also returns
an environment whose coordinate type is unknown until the record is read;
the annotation does not promise Cartesian coordinates. Inspect `coordinate_kind`
before choosing coordinate-specific operations. Factory-inferred Cartesian and
polar types retain their stricter static checks.

Replace `EgocentricPolarEnvironment.create(...)` with
`Environment.from_polar_egocentric(...)`. The old public class and import are
removed, with no alias. For metadata dispatch, use `coordinate_kind` instead of
`isinstance`: both coordinate kinds are now instances of `Environment`.

`env.layout` and direct `Environment(layout=...)` construction are removed.
Read supported environment properties such as `bin_centers`, `bin_sizes`,
`grid_edges`, `grid_shape`, `active_mask` and `edge_pairs`. `layout_type` and
`layout_parameters` remain descriptive construction history. They do not decide
which operations are available. Construct from positions, polygons, masks or
tracks using the existing factories; use `from_bins` for explicit owned geometry.

<!-- docs-test: run -->
```python
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)

bins = SpatialBins.from_arrays(
    centers=np.array([[0.0], [1.0]]), sizes=np.ones(2), measure_dim=1
)
connections = BinConnections.from_arrays(
    pairs=np.array([[0, 1]]), pair_distances=np.ones(1)
)
explicit = Environment.from_bins(
    Binning("measured bins", bins, connections, NearestBinLocator(bins.centers),
            MissingFaces("No measured cell interfaces")),
    units="cm",
)
assert explicit.n_bins == 2
assert not hasattr(explicit, "layout")
restored = Environment.from_dict(explicit.to_dict())
np.testing.assert_array_equal(restored.bin_centers, explicit.bin_centers)
```

Explicit bins need `KnownFaces` to enable numerical operations requiring cell
interfaces, or `MissingFaces` with a reason when those interfaces are unknown.
A custom provider must adapt to supported owned snapshots; live callbacks are
not retained. Environments own their input buffers, metadata and regions. Use
`with_metadata`, `with_regions`, `subset`, `clip`, `rebin` or `apply_transform`
to obtain an updated value.

Supported dictionary, file and NWB records still load without running builders.
No new scientific file schema is introduced by this class merge. Current copies,
deep copies and pickles retain ownership. Historical pickles containing removed
class paths or stored layouts are outside this migration contract: export those
objects to a supported dictionary, file or NWB record using the older version,
then load the export with this version.
