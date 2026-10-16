# Core Concepts

Understanding these core concepts will help you use neurospatial effectively.

## Spatial Discretization

neurospatial converts **continuous spatial coordinates** into **discrete bins** (also called nodes or spatial bins). This discretization is essential for:

- Computing spatial statistics (occupancy, firing rates, etc.)
- Representing spatial relationships in a graph structure
- Performing efficient spatial queries
- Reducing computational complexity

### Example: Continuous to Discrete

```python
import numpy as np
from neurospatial import Environment

# Continuous coordinates (cm)
continuous_position = np.array([[10.5, 12.3]])

# Create discretized environment
env = Environment.from_samples(
    positions=continuous_position,
    bin_size=2.0  # 2 cm bins
)

# Map to discrete bin index
bin_index = env.bin_at(continuous_position)[0]
print(f"Continuous position {continuous_position[0]} → Bin {bin_index}")

# Get bin center (representative coordinate)
bin_center = env.bin_centers[bin_index]
print(f"Bin {bin_index} center: {bin_center}")
```

## Bins and Active Bins

### Bins

A **bin** is a discrete spatial unit with:

- A unique integer index
- A center coordinate in N-dimensional space
- Optional size/extent information
- Connectivity to neighboring bins

### Active vs Inactive Bins

**Active bins** are bins considered part of the environment. **Inactive bins** are excluded (e.g., walls, obstacles, or unvisited regions).

```python
# By default, only bins that contain at least one sample are active
env = Environment.from_samples(data, bin_size=2.0)
print(f"Visited bins: {env.n_bins}")

# Require several samples before a bin counts as visited
env_strict = Environment.from_samples(data, bin_size=2.0, bin_count_threshold=5)
print(f"Bins with at least 5 samples: {env_strict.n_bins}")

# Keep every bin in the sampled extent, visited or not
env_all = Environment.from_samples(data, bin_size=2.0, infer_active_bins=False)
print(f"All bins in the extent: {env_all.n_bins}")
```

### Why Active/Inactive Matters

In neuroscience, you often want to exclude:

- Areas the animal didn't visit
- Physical barriers (walls, obstacles)
- Regions outside the experimental arena

This prevents:

- Dividing by zero in firing rate calculations
- Including irrelevant bins in statistics
- Wasting computation on empty space

## Connectivity Graphs

Every environment includes a **connectivity graph** (NetworkX `Graph`) that defines spatial relationships between bins.

### Graph Structure

- **Nodes**: Represent bins (indexed 0, 1, 2, ...)
- **Edges**: Connect neighboring bins
- **Attributes**: Store spatial metadata

```python
# Access the connectivity graph (a NetworkX Graph)
G = env.connectivity

print(f"Number of nodes (bins): {G.number_of_nodes()}")
print(f"Number of edges (neighbor links): {G.number_of_edges()}")

# Neighboring bins of a given bin
print(f"Neighbors of bin 0: {env.neighbors(bins=0)}")
```

Nodes and edges also carry metadata — bin positions, inter-bin distances, and
displacement vectors — that power shortest paths and geodesic distances. Those
mandatory attributes are documented in
[Architecture & Internals](../advanced/architecture.md#mandatory-graph-metadata),
which you only need when reading the graph directly or writing a custom layout.

### Why Graphs?

Graphs provide:

1. **Flexible topology**: Handle complex shapes, holes, and disconnected regions
2. **Efficient queries**: NetworkX algorithms for paths, distances, neighborhoods
3. **Manifold distances**: Geodesic distances that respect environment boundaries
4. **Extensibility**: Add custom node/edge attributes for your analysis

## Layout Engines

**Layout engines** define **how** continuous space is discretized into bins. They implement the `LayoutEngine` protocol.

### Available Layout Engines

| Engine | Description | Use Case |
|--------|-------------|----------|
| `RegularGridLayout` | Rectangular/cuboid grids | Standard spatial binning |
| `HexagonalLayout` | Hexagonal tessellation | Uniform neighbor distances |
| `GraphLayout` | 1D linearized tracks | Maze/track experiments |
| `MaskedGridLayout` | Grid with active mask | Arbitrary active regions |
| `ImageMaskLayout` | Binary image masking | Image-based boundaries |
| `ShapelyPolygonLayout` | Polygon-bounded grid | Circular/custom arenas |
| `TriangularMeshLayout` | Triangular tessellation | Alternative to grids |

### Automatic Selection

Factory methods automatically choose the appropriate engine:

```python
# Regular grid (automatic)
env_grid = Environment.from_samples(data, bin_size=2.0)

# Hexagonal (explicit)
env_hex = Environment.from_samples(
    data, bin_size=2.0, layout="Hexagonal"
)

# Polygon-bounded (automatic)
env_poly = Environment.from_polygon(polygon, bin_size=2.0)

# 1D linearized (automatic)
env_1d = Environment.from_graph(
    graph=graph,
    edge_order=edge_order,
    edge_spacing=0.0,
    bin_size=2.0,
)
```

Layout engines are wired into `Environment` through a **protocol** (structural
typing), so you can add a custom discretization strategy without subclassing.
That extension point, and the raw graph metadata each engine must produce, are
covered in [Architecture & Internals](../advanced/architecture.md).

## 1D vs N-D Environments

### N-D Environments (Grids)

Most environments are **N-dimensional** (typically 2D or 3D):

- Use grid-based layouts
- Spatial queries in original coordinate space
- Multiple neighbors per bin (4-8 in 2D, 6-26 in 3D)

```python
# 2D environment
env_2d = Environment.from_samples(data_2d, bin_size=2.0)
print(f"Dimensionality: {env_2d.n_dims}D")
print(f"Is 1D: {env_2d.is_linearized_track}")  # False

# Spatial queries use 2D coordinates
point_2d = np.array([[10.0, 15.0]])
bin_idx = env_2d.bin_at(point_2d)[0]
```

### 1D Environments (Linearized Tracks)

**1D linearized environments** represent tracks/mazes as a single dimension:

- Use `GraphLayout` engine
- Linearize complex 2D/3D geometries into 1D
- Essential for track-based experiments (T-maze, plus maze, etc.)

```python
import networkx as nx

# Define track structure
graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))
graph.add_node(1, pos=(10.0, 0.0))
graph.add_node(2, pos=(20.0, 0.0))
graph.add_edge(0, 1, edge_id=0, distance=10.0)
graph.add_edge(1, 2, edge_id=1, distance=10.0)

# Create 1D environment
env_1d = Environment.from_graph(
    graph=graph,
    edge_order=[(0, 1), (1, 2)],
    edge_spacing=0.0,
    bin_size=2.0
)

print(f"Is 1D: {env_1d.is_linearized_track}")  # True

# Convert 2D position to 1D linear coordinate
position_2d = np.array([[15.0, 0.0]])
position_1d = env_1d.to_linear(position_2d)
print(f"2D position {position_2d[0]} → 1D position {position_1d[0]:.2f}")

# Convert back
position_2d_reconstructed = env_1d.linear_to_nd(position_1d)
```

**When to use 1D:**

- Track-based experiments (mazes, linear tracks)
- Analyzing sequences along specific paths
- Decoding position in 1D space
- Reducing dimensionality for analysis

## Published values and coordinate labels

An environment's geometry, name, units, frame and named regions are read-only.
Set labels when constructing it:

```python
env = Environment.from_samples(data, bin_size=2.0, units="cm", frame="arena")
```

Units describe the coordinates supplied; setting `units="cm"` does not convert
pixels into centimeters. Calibrate coordinates before construction. To correct
a declaration or rename an analysis, keep the returned environment:

```python
renamed = env.with_metadata(name="session_1_checked")
```

This shares the published geometry and gives the new environment its own
analysis caches. It does not change `env` or rescale coordinates. Use factories,
`subset`, `clip` or a supported transform to create different geometry.

## Regions of Interest (ROIs)

**Regions** are named spatial areas within an environment:

- Immutable `Region` objects (point or polygon)
- Built with a mutable `Regions` collection and snapshotted by the environment
- JSON serialization support

```python
from shapely.geometry import Point
from neurospatial import Regions

regions = Regions(env.regions.values())
regions.add("StartLocation", point=(0.0, 0.0))
regions.add("RewardZone", polygon=Point(10.0, 10.0).buffer(5.0))
env = env.with_regions(regions)

# Access regions
print(env.regions.list_names())  # ['StartLocation', 'RewardZone']
print(env.regions["RewardZone"].kind)  # 'polygon'

# Region operations
area = env.regions.area("RewardZone")
center = env.regions.region_center("RewardZone")

# Edit the builder and publish a replacement; env keeps its original snapshot.
regions.update_region("StartLocation", point=(5.0, 5.0))
updated = env.with_regions(regions)
```

`with_regions` replaces the complete collection. Start with
`Regions(env.regions.values())` to retain existing names, or `Regions()` for a
new collection. Regions are analysis selections and do not define physical
walls. See the [region guide](../user-guide/regions.md) for editing and buffering.

## Construct a complete environment

Factory methods return environments with validated geometry and connectivity,
ready for spatial queries. There is no public fit step or empty environment to
initialize later.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment

positions = np.array([[0., 0.], [0., 10.], [10., 0.], [10., 10.]])
env = Environment.from_samples(positions, bin_size=2.0)
bin_idx = env.bin_at(positions[:1])
```

Calling the bare constructor raises E1006 immediately:

<!-- docs-test: raises ValueError -->
```python
from neurospatial import Environment

Environment()
```

## Factory Methods vs Constructor

**Always use factory methods**, not the bare constructor:

| Factory Method | Purpose |
|----------------|---------|
| `Environment.from_samples()` | Create from data points |
| `Environment.from_graph()` | Create 1D linearized track |
| `Environment.from_polygon()` | Grid bounded by polygon |
| `Environment.from_grid_mask()` | Pre-defined N-D boolean mask |
| `Environment.from_pixel_mask()` | Binary image mask |
| `Environment.from_layout()` | A built layout engine (triangular mesh, your own engine) |

Factory methods ensure:

- Proper initialization and fitting
- Correct layout engine selection
- Validation of parameters
- Sensible defaults

## Summary

Key takeaways:

1. **Discretization**: Continuous space → discrete bins
2. **Active bins**: Only include relevant spatial regions
3. **Connectivity graphs**: Define spatial relationships with mandatory metadata
4. **Layout engines**: Protocol-based, swappable discretization strategies
5. **1D vs N-D**: Different query methods for linearized vs grid environments
6. **Regions**: Named, immutable ROIs within environments
7. **Factory methods**: Always use these, not bare constructor

## See Also

- **[Architecture & Internals](../advanced/architecture.md)**: Mandatory graph
  metadata and the layout-engine protocol — for reading the graph directly or
  writing a custom layout engine.

## Next Steps

- **[User Guide](../user-guide/index.md)**: Detailed feature guides
- **[API Reference](../api/index.md)**: Complete API documentation
- **[Examples](../examples/index.md)**: Real-world use cases
- **[Glossary](../glossary.md)**: Spatial-coding vocabulary reference
