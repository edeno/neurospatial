# Environments

The `Environment` class is the main interface for working with discretized spatial environments in neurospatial.

## Overview

An `Environment` wraps a layout engine and provides:

- Spatial queries (point-to-bin mapping, neighbors, paths, distances)
- Region management (named ROIs)
- Visualization capabilities
- Serialization support

## Factory Methods

Always create environments using factory methods, not the bare constructor:

### from_samples()

Create an environment from data samples (most common):

```python
import numpy as np
from neurospatial import Environment

# Position data (n_samples, n_dims)
data = np.random.uniform(0, 100, (1000, 2))

env = Environment.from_samples(
    data_samples=data,
    bin_size=5.0,
    name="MyEnvironment"
)
```

**Parameters:**

- `data_samples`: Array of shape (n_samples, n_dims)
- `bin_size`: Size of bins (required)
- `infer_active_bins`: Automatically detect active regions (default: False)
- `bin_count_threshold`: Minimum samples per active bin
- `layout_type`: "regular", "hexagonal", or "triangular"
- `dilate`, `fill_holes`, `close`: Morphological operations

See [API Reference](../api/index.md) for complete parameter documentation.

### from_polygon()

Create an environment bounded by a Shapely polygon:

```python
from shapely.geometry import Polygon

# Circular arena
theta = np.linspace(0, 2*np.pi, 100)
boundary = np.column_stack([50*np.cos(theta), 50*np.sin(theta)])
polygon = Polygon(boundary)

env = Environment.from_polygon(
    polygon=polygon,
    bin_size=2.5,
    name="CircularArena"
)
```

### from_graph()

Create a 1D linearized track environment:

```python
import networkx as nx

# Define track topology
graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))
graph.add_node(1, pos=(50.0, 0.0))
graph.add_edge(0, 1, edge_id=0, distance=50.0)

env = Environment.from_graph(
    graph=graph,
    edge_order=[(0, 1)],
    bin_size=2.0,
    name="LinearTrack"
)
```

### from_mask()

Create from a pre-defined N-D boolean mask:

```python
# Create custom mask
mask = np.zeros((50, 50), dtype=bool)
mask[10:40, 10:40] = True  # Active region

env = Environment.from_mask(
    mask=mask,
    bin_size=2.0,
    dimension_ranges=[(0, 100), (0, 100)],
    name="MaskedEnvironment"
)
```

### from_image()

Create from a binary image file:

```python
env = Environment.from_image(
    image_path="arena_mask.png",
    bin_size=2.0,
    dimension_ranges=[(0, 100), (0, 100)],
    name="ImageEnvironment"
)
```

## Spatial Queries

### bin_at()

Map points to bin indices:

```python
points = np.array([
    [10.0, 20.0],
    [30.0, 40.0],
    [50.0, 60.0]
])

bin_indices = env.bin_at(points)
# Returns: array of bin indices
```

### contains()

Check if points are within the environment:

```python
is_inside = env.contains(points)
# Returns: boolean array
```

### neighbors()

Get neighboring bins:

```python
neighbors = env.neighbors(bin_index=42)
# Returns: list of neighbor bin indices
```

### distance_between()

Calculate distance between bins:

```python
distance = env.distance_between(bin_a=10, bin_b=20)
# Returns: float distance
```

### shortest_path()

Find shortest path between bins:

```python
path = env.shortest_path(start_bin=0, end_bin=50)
# Returns: list of bin indices forming path
```

## Properties

Access environment properties:

```python
# Basic properties
print(env.n_bins)           # Number of bins
print(env.n_dims)           # Number of dimensions
print(env.is_1d)            # True for linearized environments
print(env.name)             # Environment name

# Spatial information
print(env.bin_centers)      # Array of shape (n_bins, n_dims)
print(env.dimension_ranges) # List of (min, max) tuples
print(env.connectivity)     # NetworkX graph

# Layout-specific (if available)
if hasattr(env.layout, 'grid_shape'):
    print(env.layout.grid_shape)
```

## 1D Environments

Linearized track environments have additional methods:

```python
# Check if environment is 1D
if env.is_1d:
    # Convert N-D coordinates to 1D
    linear_position = env.to_linear(nd_position)

    # Convert 1D back to N-D
    nd_position = env.linear_to_nd(linear_position)
```

## Visualization

Plot the environment:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
env.plot(ax=ax)
plt.show()
```

## Common Patterns

### Pattern: Occupancy Map

```python
# Assign positions to bins
position_data = load_tracking_data()  # (n_timepoints, 2)
bin_indices = env.bin_at(position_data)

# Compute occupancy
occupancy, _ = np.histogram(
    bin_indices,
    bins=np.arange(env.n_bins + 1)
)

# Visualize
fig, ax = plt.subplots()
env.plot(ax=ax)
# Overlay occupancy heatmap...
```

### Pattern: Spatial Smoothing

```python
# Smooth data using neighbor relationships
def spatial_smooth(values, env, sigma=1.0):
    smoothed = np.zeros_like(values)
    for bin_idx in range(env.n_bins):
        neighbors = env.neighbors(bin_idx)
        neighbor_values = values[neighbors]
        smoothed[bin_idx] = np.mean(neighbor_values)
    return smoothed
```

## See Also

- [Layout Engines](layout-engines.md): Understanding discretization strategies
- [Regions](regions.md): Defining ROIs within environments
- [API Reference](../api/index.md): Complete API documentation
