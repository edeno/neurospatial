# Alignment & Transforms

neurospatial provides tools for transforming and aligning spatial representations between 2D and 3D environments.

## 2D Affine Transformations

Every transform is an `AffineND`; for 2D environments, build one with the
factory functions:

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.ops import scale_2d, translate

# Factory functions (composable)
T = translate(10, 20)
S = scale_2d(1.5, 1.5)

# Compose transformations
transform = T @ S

# Apply to points
points_2d = np.array([[0.0, 0.0], [1.0, 2.0]])
transformed_points = transform(points_2d)
```

## 3D Affine Transformations

For 3D environments, use `AffineND`:

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.ops import from_rotation_matrix, scale_3d, translate_3d
from scipy.spatial.transform import Rotation

# 3D translation
T = translate_3d(10, 20, 30)

# 3D scaling (uniform or anisotropic)
S_uniform = scale_3d(1.5)  # Scale all axes equally
S_aniso = scale_3d(1.5, 2.0, 0.8)  # Different scale per axis

# 3D rotation using scipy
R = Rotation.from_euler('z', 45, degrees=True).as_matrix()
Rot = from_rotation_matrix(R)

# Rotation with translation
transform = from_rotation_matrix(R, translation=[10, 20, 30])

# Compose transformations
combined = T @ Rot @ S_uniform

# Apply to 3D points
points_3d = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
transformed_points = combined(points_3d)
```

## Transform Estimation from Point Correspondences

Estimate transformations from matching point pairs (works for 2D or 3D):

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.ops import estimate_transform

# Given source and destination point correspondences
src_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
dst_points = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5], [5, 5, 6]])

# Estimate rigid transformation (rotation + translation)
T_rigid = estimate_transform(src_points, dst_points, kind="rigid")

# Estimate similarity transformation (rotation + translation + uniform scaling)
T_sim = estimate_transform(src_points, dst_points, kind="similarity")

# Estimate full affine transformation
T_affine = estimate_transform(src_points, dst_points, kind="affine")

# Apply to new points
new_points = np.array([[2, 2, 2], [3, 3, 3]])
aligned_points = T_rigid(new_points)
```

## Transforming Environments

`env.apply_transform(transform)` returns an environment in the new coordinates.
It moves cells, the arena, physical walls and regions together. Bin IDs and the
order of neighbor pairs stay the same, including after reflection. Supply query
positions in the new world coordinates. The name is kept. Name the new
coordinate frame with `frame=` (for example `frame="room"`); without it the
frame is `None`, because the coordinates are no longer in the source frame.

A transform that keeps lengths (a translation, rotation or reflection) keeps
the units. A transform that scales or shears lengths needs `units=` when the
environment has units: a pixel-to-centimeter calibration takes `units="cm"`,
and an arena stretched in the same units takes `units=env.units`. The package
cannot tell the two apart, so it asks instead of keeping a unit that may now be
wrong.

A grid keeps its original coordinate axes internally. Interpolation and linear
movement allocation use those axes; plots transform the cell corners into world
coordinates. This lets a rotation preserve interpolation and draw the actual
cells instead of replacing them with dots.

### Start with an image mask

This simulated example starts with an image mask whose center pixel is excluded.
A mask defines the domain included in the analysis; it does not establish where
physical walls exist. `pixel_size` gives the length of each pixel in the selected
units. Estimate a map from corresponding landmarks, transform the environment
and tracking positions together, then use the usual field and encoding APIs.

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment, compute_spatial_rates
from neurospatial.ops import estimate_transform
from neurospatial.ops.transforms import AffineND

mask = np.ones((3, 3), dtype=bool)
mask[1, 1] = False
source = Environment.from_pixel_mask(mask, pixel_size=1.0, units="cm")
assert not source.arena.physical_footprint_known
assert not source.arena.walls

landmarks = np.array([[0., 0.], [3., 0.], [0., 3.], [3., 3.]])
known_map = AffineND(np.array([[.8, -.6, 10.], [.6, .8, 20.], [0., 0., 1.]]))
transform = estimate_transform(landmarks, known_map(landmarks), kind="affine")
room = source.apply_transform(transform, frame="room")
np.testing.assert_array_equal(room.bin_at(transform(source.bin_centers)), np.arange(source.n_bins))

# Values indexed by bin ID keep their order. Queries use room coordinates.
field = source.bin_centers @ np.array([2., 3.])
queries = np.array([[.5, 1.1], [.5, 1.75]])
np.testing.assert_allclose(
    room.interpolate(field, transform(queries), mode="linear"),
    source.interpolate(field, queries, mode="linear"),
)
ax = room.plot_field(field, colorbar=False)
plt.close(ax.figure)

# Straight movement keeps the same time fractions after an affine map.
times = np.array([0., 2.5])
positions = np.array([[.25, .5], [2.75, .5]])
spikes = [np.array([.25, .75, 1.25, 1.75, 2.25])]
options = {
    "method": "binned", "bandwidth": 0,
    "time_allocation": "linear", "max_gap": None, "warn_on_drop": False,
}
source_rates = compute_spatial_rates(source, spikes, times, positions, **options)
room_rates = compute_spatial_rates(room, spikes, times, transform(positions), **options)
np.testing.assert_allclose(room_rates.occupancy, source_rates.occupancy)
np.testing.assert_allclose(room_rates.firing_rates, source_rates.firing_rates, equal_nan=True)

restored = Environment.from_dict(room.to_dict())
np.testing.assert_allclose(restored.interpolate(field, transform(queries), mode="linear"), [4.3, 6.25])
```

`compute_spatial_rate` and `compute_spatial_rates` use the same retained
movement intervals for spike counts and dwell time when
`time_allocation="linear"`. The example disables smoothing to compare the
allocation directly. Smoothing requires valid numerical face geometry.
Spike encoding with `time_allocation="linear"` requires planar 2-D cells. 3-D
grids support linear occupancy, but for 3-D spike encoding use
`time_allocation="start"`.

### Start with position samples

If only tracking positions are available, `from_samples` can infer which bins
were visited. That estimated domain also asserts no physical walls. You can
transform it with the same API; add measured arena geometry separately when it
is available.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.ops.transforms import AffineND

x = np.linspace(0., 10., 101)
positions = np.column_stack((x, 2. + np.sin(x)))
source = Environment.from_samples(positions, bin_size=.5, units="cm")
reflection = AffineND(np.diag([1., -1., 1.]))
aligned = source.apply_transform(reflection, frame="room")
assert not aligned.arena.physical_footprint_known
assert not aligned.arena.walls
np.testing.assert_array_equal(aligned.bin_at(reflection(source.bin_centers)), np.arange(source.n_bins))
```

### Geometry and numerical capabilities

Exact cell geometry and a valid diffusion calculation are separate capabilities.
The current diffusion solver needs each center-to-center link to be perpendicular
to its shared physical face. A shear generally breaks that condition, even though
lookup, plotting and dwell allocation remain well defined.

| Current cells / transform | Linear field interpolation | Exact 2-D field plot | Smoothing and differential operators |
| --- | --- | --- | --- |
| Cartesian grid, translation or positive axis scaling | Yes | QuadMesh | Retained |
| Cartesian 2-D/3-D grid, rotation or reflection | Yes, in source axes | QuadMesh | Retained |
| Grid, unequal scaling along retained orthogonal axes | Yes | QuadMesh | Enabled when normals and flux distances certify the geometry |
| Grid, shear | Yes | QuadMesh | Refused if shared faces lose orthogonality |
| Whole-bin subset of a transformed grid | Yes; missing interpolation corners stay NaN | QuadMesh with the retained mask | Retains the parent's available geometry |
| Partial polygon clipping | No tensor grid; use nearest mode | Exact polygon cells | Only when the clipping certificate remains valid |
| Transformed triangles or deformed hexagons | No tensor grid; use nearest mode | Exact vertices/polygon cells | Existing similarity support; other maps require a valid face certificate |
| Embedded track | Use existing track operations | Existing track renderer | Similarity transforms only |

Landmarks measured from video carry tracking noise. A general affine fit
(`kind="affine"`) absorbs that noise as a small stretch or shear, so smoothing on
the transformed environment refuses and reports the measured stretch. When the
arena's shape is known, fit `kind="similarity"` instead (or `kind="rigid"` when
both coordinate systems use the same length units). The fitted map keeps right
angles, so smoothing with the bandwidth multiplied by the fitted scale matches
smoothing in the source coordinates.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial import Environment
from neurospatial.ops import estimate_transform

rng = np.random.default_rng(0)
camera = Environment.from_samples(rng.uniform(0, 100, (5000, 2)), bin_size=5.0, units="px")
corners_px = np.array([[0., 0.], [100., 0.], [0., 100.], [100., 100.]])
detected_px = corners_px + rng.normal(0, 0.5, corners_px.shape)  # 0.5 px tracking noise
corners_cm = np.array([[0., 0.], [50., 0.], [0., 50.], [50., 50.]])

to_cm = estimate_transform(detected_px, corners_cm, kind="similarity")
room = camera.apply_transform(to_cm, units="cm")
cm_per_px = np.sqrt(abs(np.linalg.det(to_cm.A[:2, :2])))
field = rng.random(camera.n_bins)
np.testing.assert_allclose(
    room.smooth(field, bandwidth=5.0, mode="average"),
    camera.smooth(field, bandwidth=5.0 / cm_per_px, mode="average"),
)
```

New face certification checks normalized orthogonality residuals against `1e-12`.
Legacy records without stored face normals keep their existing guaranteed
transforms, but cannot gain new numerical support from inferred directions.
Once a transform discards numerical faces, applying its inverse does not restore
them. Rebuild from the original environment to retain that authority.

Floating-point projection can land just across a source center or edge plane.
Interpolation and linear allocation recover a plane only within a calculated
roundoff bound. The bound must remain below one quarter of the smallest source
cell width on that axis; otherwise the operation asks you to recenter or rescale
coordinates. Resolved missing support remains missing.

Matplotlib plots and video, HTML and widget animations draw transformed grid
corners. Napari's current raster/coordinate helpers explicitly refuse these
framed grids. Raster FFT/grid metrics and 2-D circular grid shuffling also keep
their world-aligned-grid restriction; saved source axes are not exposed as world
`grid_edges` or a world-aligned `grid_shape`.

`rebin(bin_size=...)` uses its separate construction specification and physical
axis lengths. It creates new bins over the retained arena; it does not reuse
current bin IDs or transfer fitted values. Partial clipping drops the current
tensor-grid component even when a construction specification remains available.

Transforms update geometry and cell measures, while bin-aligned arrays keep
their indexing. They do not refit models or rescale firing-rate values. A matrix
does not supply a unit label: use `with_metadata(units=..., frame=...)` when the
coordinate conversion changes those labels. Dict/file/NWB round trips retain
source frames and normals. New files use nested binning codec version 3; current
readers accept versions 1 and 2, while older readers reject the new nested
version. The outer Environment-v3 and NWB 3.0 envelopes are unchanged.

## Probability Alignment

Map probability distributions between environments (works for 2D or 3D).
`map_probabilities` moves probability mass: each source bin's mass goes to its
nearest target bin and the masses are summed, so the total is kept. Use it for
posteriors and occupancy probabilities, not for rate maps. To move a rate map
onto other bins, use `resample_field`, which copies each rate.

```python
import numpy as np
from neurospatial.ops import get_2d_rotation_matrix, map_probabilities
from scipy.spatial.transform import Rotation

# 2D alignment with rotation and scaling
aligned_probs_2d = map_probabilities(
    source_env=env1,
    target_env=env2,
    source_probs=probs1,
    source_rotation_matrix=get_2d_rotation_matrix(np.pi / 4),  # radians
    source_scale=1.2
)

# 3D alignment with rotation
R_3d = Rotation.from_euler('xyz', [45, 30, 60], degrees=True).as_matrix()
aligned_probs_3d = map_probabilities(
    source_env=env_3d_1,
    target_env=env_3d_2,
    source_probs=probs_3d,
    source_rotation_matrix=R_3d,
    source_scale=0.9,
    source_translation_vector=[10, 20, 30]
)
```

## Complete Alignment Workflow

Here's a complete example aligning two 3D environments:

```python
import numpy as np
from scipy.spatial.transform import Rotation
from neurospatial import Environment
from neurospatial.ops import estimate_transform

# Create two 3D environments
positions_session1 = np.random.randn(1000, 3) * 20
positions_session2 = positions_session1 @ Rotation.from_euler('z', 30, degrees=True).as_matrix().T + [10, 5, 2]

env1 = Environment.from_samples(positions_session1, bin_size=5.0, name="session1")
env2 = Environment.from_samples(positions_session2, bin_size=5.0, name="session2")

# Define landmark correspondences (e.g., from manual annotation)
landmarks_env1 = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
landmarks_env2 = np.array([[10, 5, 2], [18.66, 10, 2], [1.34, 13.66, 2], [10, 5, 12]])

# Estimate transformation
transform = estimate_transform(landmarks_env1, landmarks_env2, kind="rigid")

# Align env1 to env2's coordinate frame
env1_aligned = env1.apply_transform(transform, name="session1_aligned")

# Now compare neural activity, place fields, etc. in aligned coordinate frame
```

See the [API Reference](../api/index.md) for complete documentation.
