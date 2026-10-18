# Combining environments

Use `Environment.combine` when disjoint parts of an experiment already have
separate environments in the same coordinate frame. The result is an ordinary
`Environment`, with the usual lookup, plotting, encoding and storage methods.
It retains the component bins and their physical measures.

## Construct and inspect the combined environment

Give each component a stable key. Keys, rather than display names, identify the
source of each bin. Units and frame labels must match; convert and align the
actual coordinates before combining when needed.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

left = Environment.from_grid_mask(
    np.ones((2, 1), dtype=bool),
    grid_edges=(np.arange(3.0), np.arange(2.0)), units="cm", frame="lab",
)
right = Environment.from_grid_mask(
    np.ones((2, 1), dtype=bool),
    grid_edges=(np.arange(3.0) + 4, np.arange(2.0)), units="cm", frame="lab",
)
env = Environment.combine({"left": left, "right": right}, name="two arms")
assert type(env) is Environment
assert env.component_keys == ("left", "right")
assert env.bin_sources == (("left", 0), ("left", 1), ("right", 0), ("right", 1))
np.testing.assert_array_equal(env.bin_at([[0.5, 0.5], [4.5, 0.5], [3, 0.5]]), [0, 2, -1])
bridges = env.get_bridge_attributes()
assert len(bridges) == 1
assert bridges.iloc[0]["distance"] == 3.0
assert bridges.iloc[0]["conductance"] == 0.0
```

The bins follow input component order. Child edges retain their order, followed
by inferred bridges; `env.edge_pairs` defines array order. Child region names
become `component_key/region_name`, avoiding collisions. Subsets retain original
source IDs and remove bridges whose endpoints disappear. Saved environments own
the component snapshots and bridge results; loading does not infer connections
again or run layout builders.

Positive-measure cell overlap is rejected. Shared boundaries preserve each
child's lookup rule, then use component order to resolve a remaining tie. The
coverage check supports exact planar cells, axis-aligned boxes, transformed 3D
boxes and certifiable nearest/track support. Uncertain mixed geometry or nearly
parallel track clearance raises with guidance. Multiple unrestricted nearest
locators both claim all finite space: supply finite lookup radii, known domains,
or disjoint subsets of the same parent locator. Matching centers or a successful
sample of queries cannot certify disjointness.

## Paths and diffusion

By default, automatic mutual-nearest-neighbor bridges join graph paths between
components. Equal nearest distances choose the lowest local bin ID. Use
`max_mnn_distance` to limit their lengths, or `auto_bridge=False` for disconnected
components. Inference respects known physical walls. Bridges do not fill the gap
for lookup or occupancy and do not assert a measured corridor.

Path-only bridges carry no diffusion. Valid child geometries retain their
independent numerical blocks. To couple them, supply geometric conductance for
specific inferred endpoints, in units `length ** (measure_dim - 2)`:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

components = {
    key: Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0) + shift, np.arange(2.0)),
        units="cm", frame="lab",
    )
    for key, shift in (("left", 0), ("right", 4))
}
env = Environment.combine(components)
field = np.array([1.0, 1.0, 0.0, 0.0])
np.testing.assert_allclose(env.diffuse(field, bandwidth=1.0, mode="average"), field, atol=1e-14)
coupled = Environment.combine(
    components,
    bridge_conductances={(('left', 1), ('right', 0)): 0.5},
)
smoothed = coupled.diffuse(field, bandwidth=1.0, mode="average")
assert smoothed[2] > 0
np.testing.assert_allclose(smoothed @ coupled.bin_sizes, field @ coupled.bin_sizes)
```

A conductance does not specify a physical face or flux density. Spatial gradient
and divergence therefore refuse combinations containing nonphysical bridges.
A retained child without valid faces blocks whole-environment diffusion; removing
it with a subset allows the remaining certified geometry to be analyzed.

Exact planar plotting and linear occupancy use retained cells. Geometry-specific
operations remain conditional: there is no single combined regular grid or global
track coordinate, and rebinning requires a separately supported representation.
Clip children before adding bridges, or combine with `auto_bridge=False` before
clipping: splitting bridge endpoints needs an explicit reconnection rule.

## A corridor joined to an arena

Components can differ in what their bins measure. An approach corridor built as
a track has bins measured by length; the arena it leads into has bins measured
by area. Combining them keeps each bin's own measure, so occupancy, rate maps,
smoothing within each component and decoding all work on the combined
environment, while `bin_sizes` refuses, because a sum or density over lengths
and areas together has no unit:

<!-- docs-test: run -->
```python
import numpy as np
import pytest
from neurospatial import Environment

corridor = Environment.linear_track(
    endpoints=[[-3.0, 5.0], [-23.0, 5.0]], bin_size=2.0, track_width=2.0, units="cm"
)
edges = np.arange(0.0, 11.0, 2.0)
arena = Environment.from_grid_mask(
    np.ones((5, 5), dtype=bool), grid_edges=(edges, edges), units="cm"
)
env = Environment.combine({"corridor": corridor, "arena": arena})
dims = env.binning.bins.measure_dims  # 1 for corridor bins, 2 for arena bins
assert set(dims.tolist()) == {1, 2}
with pytest.raises(ValueError, match="mix measures"):
    env.bin_sizes
in_arena = np.array([key == "arena" for key, _ in env.bin_sources])
np.testing.assert_allclose(env.subset(mask=in_arena).bin_sizes, 4.0)  # cm^2
```

A transform scales each bin by its own dimension: doubling lengths doubles the
corridor bins and quadruples the arena bins. A bridge conductance between a
corridor bin and an arena bin has no single length dimension, so a transform
that rescales lengths refuses it; scale each component before combining.

## Migration

`CompositeEnvironment` has been removed. Replace its constructor with
`Environment.combine({"key": env, ...})`. Use `get_bin_attributes()` and
`get_edge_attributes()` for tables, and `to_file()`/`from_file()` for owned data
storage. Overlapping components, incompatible declarations and uncertified
lookup support now raise instead of silently selecting the first environment.
