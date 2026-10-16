# Geometry access and cache ownership

An `Environment` owns its geometry and the caches used to analyze it. The
geometry is immutable: selection, clipping, rebinning and coordinate transforms
return a new environment. Numerical code can inspect owned data through public
operations without retaining a layout builder or modifying private caches.

Most analysis workflows keep using the existing factories, encoding functions
and environment methods. This change mainly helps users constructing explicit
bins and contributors implementing numerical operations.

## Inspect or reuse bins

`env.binning` returns a detached, read-only snapshot with bin centers, measures,
ordered neighbor pairs, lookup data and optional geometry components. Native
face geometry stays lazy until requested. Mutating an exported array header or
a locator's local query cache cannot change the environment.

Use an editable array copy when preparing different geometry. To reuse a native
Cartesian snapshot in `from_bins`, resolve its face recipe explicitly:

<!-- docs-test: run -->
```python
from dataclasses import replace

import numpy as np
from neurospatial import Environment
from neurospatial.binning import KnownFaces

env = Environment.from_grid_mask(
    np.ones((3, 3), dtype=bool),
    grid_edges=(np.arange(4.0),) * 2,
    units="cm",
)
binning = env.binning
explicit = replace(binning, faces=KnownFaces(binning.faces.build()))
restored = Environment.from_bins(explicit, units=env.units, arena=env.arena)
np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
np.testing.assert_array_equal(restored.edge_pairs, env.edge_pairs)
```

For ordinary copying or persistence, use `env.copy()` or the existing storage
methods. They preserve supported Cartesian and polar geometry directly.
`from_bins` constructs Cartesian environments; it is not a polar restoration API.

## Numerical operations

Numerical consumers use public operations while Environment retains expensive
trees, matrix factorizations and bounded kernel caches.

| Operation | Purpose |
| --- | --- |
| `env.coordinate_system` | Read the immutable Cartesian or polar coordinate descriptor. |
| `env.get_diffusion_geometry()` | Obtain conductances, volumes and connected components for supported finite-volume geometry. |
| `env.diffusion_accuracy(bandwidth)` | Query the current diffusion approximation's numerical noise floor. |
| `env.get_mrf_basis(occupancy, rank=...)` | Resolve an occupancy-supported MRF penalty basis using the environment's cached eigenbasis. |
| `env.compute_gaussian_kernel(bandwidth)` | Obtain cached Gaussian weights without exposing the mutable cache. |

These operations do not grant missing geometric capabilities. Unsupported
diffusion geometry still refuses with guidance. Polar coordinate descriptors
retain the distinction between length and angle.

Use `nearest_bin` for nearest-center queries; Cartesian `bin_at` retains exact
cell ownership. Existing free functions delegate through public operations.
`clear_cache()` releases cached analyses, and copies and pickle records omit
disposable analysis caches. Existing `compute_kernel()` continues to return an
independent editable diffusion matrix.

## Contributor imports

Public imports such as `neurospatial.binning` and `neurospatial.arena` remain
stable. Owned data, locators, validation and codecs now live in `geometry/`;
`layout/` contains construction adapters. Numerical algorithms in `ops/` take
explicit geometry and arrays. Environment adapters coordinate them and own
caches. Package-level import rules enforce this separation.

Compatibility aliases retain supported older private pickle references. New
code should use public geometry exports or the current internal package, rather
than importing records from `layout._*`.
