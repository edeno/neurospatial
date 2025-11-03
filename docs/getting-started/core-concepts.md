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
    data_samples=continuous_position,
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
# All bins are active by default
env = Environment.from_samples(data, bin_size=2.0)
print(f"All bins active: {env.n_bins}")

# Automatically infer active bins from data
env_auto = Environment.from_samples(
    data,
    bin_size=2.0,
    infer_active_bins=True,  # Only bins with data
    bin_count_threshold=1    # Minimum samples per bin
)
print(f"Auto-detected active bins: {env_auto.n_bins}")
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
# Access the connectivity graph
G = env.connectivity

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Node attributes (mandatory)
node_0_attrs = G.nodes[0]
print(f"Node 0 position: {node_0_attrs['pos']}")
print(f"Node 0 grid index: {node_0_attrs['original_grid_nd_index']}")

# Edge attributes (mandatory)
edge_attrs = G.edges[0, 1]
print(f"Edge (0,1) distance: {edge_attrs['distance']}")
print(f"Edge (0,1) vector: {edge_attrs['vector']}")
```

### Mandatory Graph Metadata

neurospatial enforces **mandatory attributes** for correctness:

**Node attributes:**

- `'pos'`: Tuple of N-D coordinates
- `'source_grid_flat_index'`: Flat index in original grid
- `'original_grid_nd_index'`: N-D grid index tuple

**Edge attributes:**

- `'distance'`: Euclidean distance between bin centers
- `'vector'`: Displacement vector (as tuple)
- `'edge_id'`: Unique integer edge identifier
- `'angle_2d'`: Angle in 2D (optional, for 2D layouts)

These attributes enable spatial queries like shortest paths, distance calculations, and neighbor finding.

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
    data, bin_size=2.0, layout_type="hexagonal"
)

# Polygon-bounded (automatic)
env_poly = Environment.from_polygon(polygon, bin_size=2.0)

# 1D linearized (automatic)
env_1d = Environment.from_graph(graph, edge_order, bin_size=2.0)
```

### Protocol-Based Design

Layout engines implement a **protocol** (not inheritance):

```python
# Layout engines must provide:
# - bin_centers: NDArray of shape (n_bins, n_dims)
# - connectivity: nx.Graph with mandatory attributes
# - dimension_ranges: List of (min, max) tuples
# - is_1d: bool (True for linearized layouts)
# - build(): Method to construct the layout
# - point_to_bin_index(): Map points to bins
# - bin_sizes(): Compute bin sizes
# - plot(): Visualize the layout
```

This design allows:

- Custom layout engines without modifying core code
- Type checking with protocols
- Maximum flexibility

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
print(f"Is 1D: {env_2d.is_1d}")  # False

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
    bin_size=2.0
)

print(f"Is 1D: {env_1d.is_1d}")  # True

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

## Regions of Interest (ROIs)

**Regions** are named spatial areas within an environment:

- Immutable `Region` objects (point or polygon)
- Managed by `Regions` container (dict-like)
- JSON serialization support

```python
from shapely.geometry import Point, Polygon

# Add point region
env.regions.add("StartLocation", point=(0.0, 0.0))

# Add polygon region (buffered point = circle)
reward_zone = Point(10.0, 10.0).buffer(5.0)
env.regions.add("RewardZone", polygon=reward_zone)

# Add polygon region (custom shape)
corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
custom_zone = Polygon(corners)
env.regions.add("CustomZone", polygon=custom_zone)

# Access regions
print(env.regions.list_names())  # ['StartLocation', 'RewardZone', 'CustomZone']
print(env.regions["RewardZone"].type)  # 'polygon'

# Region operations
area = env.regions.area("RewardZone")
center = env.regions.region_center("RewardZone")

# Regions are immutable - use update to modify
env.regions.update_region("StartLocation", point=(5.0, 5.0))
```

## The Fitted State Pattern

`Environment` uses a **fitted state pattern** to ensure proper initialization:

```python
# Factory methods automatically fit the environment
env = Environment.from_samples(data, bin_size=2.0)
print(env._is_fitted)  # True

# Can now call spatial query methods
bin_idx = env.bin_at(point)  # Works

# Don't use bare constructor (not fitted)
# env_bad = Environment()  # Not fitted
# env_bad.bin_at(point)  # RuntimeError!
```

Methods requiring a fitted environment use the `@check_fitted` decorator to prevent errors.

## Factory Methods vs Constructor

**Always use factory methods**, not the bare constructor:

| Factory Method | Purpose |
|----------------|---------|
| `Environment.from_samples()` | Create from data points |
| `Environment.from_graph()` | Create 1D linearized track |
| `Environment.from_polygon()` | Grid bounded by polygon |
| `Environment.from_mask()` | Pre-defined N-D boolean mask |
| `Environment.from_image()` | Binary image mask |
| `Environment.from_layout()` | Direct layout specification |

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

## Next Steps

- **[User Guide](../user-guide/index.md)**: Detailed feature guides
- **[API Reference](../api/index.md)**: Complete API documentation
- **[Examples](../examples/index.md)**: Real-world use cases
