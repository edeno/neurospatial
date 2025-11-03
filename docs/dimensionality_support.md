# Dimensionality Support in neurospatial

**Last Updated:** 2025-11-03 (v0.2.0)

## Summary

**neurospatial supports 1D and 2D spatial environments.** Basic 3D binning works, but some features are 2D-only.

---

## Supported Dimensionalities

### 1D Environments (Linearized Tracks)

**Use case:** Linear tracks, mazes with defined paths, sequential spatial trajectories

**How to create:**
```python
from neurospatial import Environment

# From position data with track structure
env = Environment.from_graph(
    track_graph=graph,
    position=position_data,
    sampling_frequency=30.0
)
```

**Unique features:**
- `env.to_linear(nd_position)` - Convert N-D coordinates to linear position
- `env.linear_to_nd(linear_position)` - Convert linear position to N-D coordinates
- `env.plot_1d()` - 1D visualization
- `env.is_1d == True`

**Typical applications:**
- Linear tracks (e.g., T-maze arms, figure-8 tracks)
- Virtual reality corridors
- Sequential navigation analysis

---

### 2D Environments (Grid-Based Layouts)

**Use case:** Open field navigation, arenas, complex 2D environments

**How to create:**
```python
from neurospatial import Environment
import numpy as np

# From 2D position samples
data_2d = np.random.randn(1000, 2) * 50  # shape (n_samples, 2)
env = Environment.from_samples(data_2d, bin_size=2.0)

# From polygon boundary
from shapely.geometry import Polygon
polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
env = Environment.from_polygon(polygon, bin_size=2.0)

# From image mask
env = Environment.from_image("arena_mask.png", bin_size_cm=2.0, pixels_per_cm=10)
```

**Layout engines available:**
- `RegularGridLayout` - Standard rectangular grids
- `HexagonalLayout` - Hexagonal tessellations
- `TriangularMeshLayout` - Triangular tessellations
- `MaskedGridLayout` - Grids with arbitrary active/inactive regions
- `ImageMaskLayout` - Binary image-based layouts
- `ShapelyPolygonLayout` - Polygon-bounded grids

**Full feature support:**
- ✅ 2D affine transforms (rotation, scaling, translation)
- ✅ Polygon regions
- ✅ Image mask layouts
- ✅ All spatial queries (bin_at, contains, neighbors, shortest_path)
- ✅ Alignment and probability mapping between environments
- ✅ Full visualization suite

**Typical applications:**
- Open field experiments
- Water maze navigation
- Complex 2D arenas with barriers
- Multi-room environments

---

## 3D Support Status

### What Works in 3D

✅ **Basic spatial binning:**
```python
data_3d = np.random.randn(1000, 3) * 50  # shape (n_samples, 3)
env_3d = Environment.from_samples(data_3d, bin_size=2.0)

# These all work:
bins = env_3d.bin_at(points_3d)
mask = env_3d.contains(points_3d)
neighbors = env_3d.neighbors(bin_idx)
path = env_3d.shortest_path(source_bin, target_bin)
dist = env_3d.distance_between(bin1, bin2)
```

✅ **Connectivity graphs** - Full 3D graph support

✅ **Distance calculations** - Euclidean distances in 3D

✅ **Path finding** - Shortest paths through 3D bin connectivity

✅ **Composite environments** - Merging 3D environments

### What Doesn't Work in 3D

❌ **2D affine transforms**
```python
from neurospatial.alignment import get_2d_rotation_matrix

# This is 2D only - will not work for 3D environments
rotation = get_2d_rotation_matrix(angle_degrees=45)  # 2x2 matrix
```

**Alternative for 3D:** Use `scipy.spatial.transform.Rotation` for 3D rotations

❌ **Polygon regions**
```python
# Polygon regions are 2D only
polygon_coords = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
env_3d.regions.add("goal", polygon=polygon_coords)  # Will raise error

bins = env_3d.bins_in_region("goal")  # ValueError: Polygon regions only supported for 2D
```

❌ **Image mask layouts** - Binary images are inherently 2D

❌ **Hexagonal and triangular layouts** - Currently 2D-only tessellations

❌ **3D-specific visualization** - No 3D plotting methods yet

---

## Feature Compatibility Matrix

| Feature | 1D | 2D | 3D | Notes |
|---------|----|----|----|----|
| **Core Functionality** |
| Spatial binning | ✅ | ✅ | ✅ | |
| Connectivity graphs | ✅ | ✅ | ✅ | |
| Distance calculations | ✅ | ✅ | ✅ | |
| Path finding | ✅ | ✅ | ✅ | |
| Composite environments | ✅ | ✅ | ✅ | |
| **Regions** |
| Point regions | ✅ | ✅ | ✅ | |
| Polygon regions | ❌ | ✅ | ❌ | 2D only |
| **Transforms** |
| Affine2D | ❌ | ✅ | ❌ | 2D only |
| Rotation matrices | ❌ | ✅ | ❌ | Use scipy for 3D |
| Scaling | ✅ | ✅ | ✅ | |
| Translation | ✅ | ✅ | ✅ | |
| **Layout Engines** |
| RegularGridLayout | ❌ | ✅ | ✅ | |
| HexagonalLayout | ❌ | ✅ | ❌ | 2D tessellation |
| TriangularMeshLayout | ❌ | ✅ | ❌ | 2D tessellation |
| GraphLayout | ✅ | ❌ | ❌ | 1D linearization |
| MaskedGridLayout | ❌ | ✅ | ✅ | |
| ImageMaskLayout | ❌ | ✅ | ❌ | Images are 2D |
| ShapelyPolygonLayout | ❌ | ✅ | ❌ | Polygons are 2D |
| **Visualization** |
| plot() | ✅ | ✅ | ❌ | |
| plot_1d() | ✅ | ❌ | ❌ | 1D only |
| **Alignment** |
| Probability mapping | ✅ | ✅ | ✅ | |
| 2D rotation alignment | ❌ | ✅ | ❌ | |

---

##  Best Practices

### Choosing Dimensionality

**Use 1D when:**
- Your spatial data follows a defined path or track
- You need linearized position coordinates
- Working with sequential navigation (T-maze, linear track)

**Use 2D when:**
- Your spatial data is in an open 2D arena
- You need polygon-based regions
- Working with standard behavioral experiments (open field, water maze)

**Use 3D when:**
- Your spatial data is truly 3D (flight, swimming with depth, climbing)
- You only need basic binning and connectivity
- You don't need transforms or polygon regions

### Validation

Always verify dimensionality before assuming features work:

```python
env = Environment.from_samples(data, bin_size=2.0)

# Check dimensionality
print(f"Dimensions: {env.n_dims}")  # 1, 2, or 3

# Check if 1D (linearized)
if env.is_1d:
    linear_pos = env.to_linear(nd_position)
else:
    bin_idx = env.bin_at(nd_position)

# Check before using 2D-only features
if env.n_dims == 2:
    # Safe to use polygon regions, 2D transforms, etc.
    env.regions.add("goal", polygon=polygon_coords)
else:
    # Use point regions instead
    env.regions.add("goal", point=[10, 20, 30])
```

---

## Future 3D Support

Full 3D support is planned for future releases (v0.3.0+). Planned features:

- 3D affine transformations
- 3D polyhedron regions (using `trimesh` or similar)
- 3D visualization with `matplotlib 3D` or `plotly`
- 3D-specific layout engines (volumetric tessellations)
- Full documentation and examples for 3D workflows

**Track progress:** See [GitHub Issues](https://github.com/user/neurospatial/issues) for 3D support roadmap

---

## Getting Help

- For 1D/2D questions: See main documentation and examples
- For 3D questions: Check this guide for current limitations
- Report issues: https://github.com/user/neurospatial/issues
- Discussions: https://github.com/user/neurospatial/discussions
