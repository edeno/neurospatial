# Quickstart

This guide provides a quick introduction to neurospatial's main features. You'll learn how to create environments, query spatial bins, and define regions of interest.

## Basic Environment Creation

The most common use case is creating an environment from spatial data samples:

```python
import numpy as np
from neurospatial import Environment

# Simulate position tracking data (x, y coordinates in cm)
# Shape: (n_timepoints, 2)
position_data = np.array([
    [10.0, 10.0],
    [12.0, 11.0],
    [15.0, 15.0],
    [18.0, 12.0],
    [20.0, 10.0],
    [22.0, 15.0],
    [25.0, 20.0],
    # ... more positions
])

# Create environment with 2 cm bins
env = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.0,  # Required parameter
    name="OpenField"
)

# Inspect the environment
print(f"Environment: {env.name}")
print(f"Number of bins: {env.n_bins}")
print(f"Dimensions: {env.n_dims}D")
print(f"Spatial extent: {env.dimension_ranges}")
```

## Spatial Queries

Once you have an environment, you can perform various spatial queries:

```python
# Map a point to its bin index
point = np.array([[15.0, 15.0]])
bin_idx = env.bin_at(point)[0]
print(f"Point {point[0]} is in bin {bin_idx}")

# Get bin center coordinates
center = env.bin_centers[bin_idx]
print(f"Bin {bin_idx} center: {center}")

# Check if a point is in the environment
is_inside = env.contains(point)[0]
print(f"Point is inside environment: {is_inside}")

# Find neighbors of a bin
neighbors = env.neighbors(bin_idx)
print(f"Bin {bin_idx} has {len(neighbors)} neighbors: {neighbors}")

# Calculate distance between bins
bin_a, bin_b = 0, 10
distance = env.distance_between(bin_a, bin_b)
print(f"Distance between bins {bin_a} and {bin_b}: {distance:.2f}")

# Find shortest path
path = env.shortest_path(bin_a, bin_b)
print(f"Shortest path: {path}")
```

## Computing Occupancy

A common neuroscience workflow is computing time spent in each bin:

```python
# Assign all positions to bins
bin_indices = env.bin_at(position_data)

# Compute occupancy histogram
occupancy, _ = np.histogram(
    bin_indices,
    bins=np.arange(env.n_bins + 1)
)

print(f"Time in bin 0: {occupancy[0]} samples")
print(f"Total occupied bins: {np.sum(occupancy > 0)}")
```

## Defining Regions of Interest

You can define named regions (ROIs) within your environment:

```python
from shapely.geometry import Point

# Add a circular reward zone (5 cm radius)
reward_polygon = Point(15.0, 15.0).buffer(5.0)
env.regions.add("RewardZone", polygon=reward_polygon)

# Add a point marker for start location
env.regions.add("StartLocation", point=(10.0, 10.0))

# Query regions
print(f"Number of regions: {len(env.regions)}")
print(f"Region names: {env.regions.list_names()}")

# Get region statistics
area = env.regions.area("RewardZone")
center = env.regions.region_center("RewardZone")
print(f"Reward zone area: {area:.2f} cm²")
print(f"Reward zone center: {center}")
```

## Visualizing the Environment

Visualize your environment with matplotlib:

```python
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the environment
env.plot(ax=ax)

# Overlay trajectory
ax.plot(
    position_data[:, 0],
    position_data[:, 1],
    'r-', alpha=0.5, linewidth=1,
    label='Trajectory'
)

ax.set_title(env.name)
ax.legend()
plt.show()
```

## Automatic Active Bin Detection

For sparse data, you can automatically detect active regions:

```python
# Create environment with automatic active bin detection
env_auto = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.0,
    infer_active_bins=True,  # Enable automatic detection
    bin_count_threshold=1,   # Minimum samples per bin
    dilate=True,             # Expand active region
    fill_holes=True,         # Fill gaps
    name="OpenField_Auto"
)

print(f"Active bins: {env_auto.n_bins}")
```

## Creating Masked Environments

Create environments bounded by polygons:

```python
from shapely.geometry import Polygon

# Define a circular arena (40 cm radius)
theta = np.linspace(0, 2*np.pi, 100)
boundary = np.column_stack([
    40 * np.cos(theta),
    40 * np.sin(theta)
])
polygon = Polygon(boundary)

# Create environment bounded by polygon
env_circle = Environment.from_polygon(
    polygon=polygon,
    bin_size=2.5,
    name="CircularArena"
)

print(f"Circular arena bins: {env_circle.n_bins}")
```

## Working with Different Layout Types

neurospatial supports multiple layout engines:

```python
# Hexagonal layout (more uniform neighbor distances)
env_hex = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.0,
    layout_type="hexagonal",
    name="HexEnvironment"
)

# Triangular mesh
env_tri = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.0,
    layout_type="triangular",
    name="TriEnvironment"
)
```

## Next Steps

Now that you understand the basics, explore:

- **[Core Concepts](core-concepts.md)**: Deeper understanding of bins, graphs, and layout engines
- **[User Guide](../user-guide/index.md)**: Detailed guides for specific features
- **[API Reference](../api/index.md)**: Complete API documentation
- **[Examples](../examples/index.md)**: Real-world use cases with Jupyter notebooks

## Common Patterns

### Pattern 1: Spatial Firing Rate Map

```python
# Compute spike counts per bin
spike_bin_indices = env.bin_at(spike_positions)
spike_counts, _ = np.histogram(
    spike_bin_indices,
    bins=np.arange(env.n_bins + 1)
)

# Compute occupancy
position_bin_indices = env.bin_at(position_data)
occupancy, _ = np.histogram(
    position_bin_indices,
    bins=np.arange(env.n_bins + 1)
)

# Calculate firing rate (spikes/sec, assuming 30 Hz sampling)
sampling_rate = 30.0  # Hz
time_per_bin = occupancy / sampling_rate
firing_rate = np.divide(
    spike_counts,
    time_per_bin,
    where=time_per_bin > 0
)
```

### Pattern 2: Distance to Target

```python
# Find bin containing target location
target = np.array([[20.0, 20.0]])
target_bin = env.bin_at(target)[0]

# Calculate distance from every bin to target
distances = np.array([
    env.distance_between(bin_idx, target_bin)
    for bin_idx in range(env.n_bins)
])

print(f"Mean distance to target: {np.mean(distances):.2f} cm")
```

### Pattern 3: Region Occupancy

```python
# Define multiple zones
zones = {
    "Zone1": Point(10.0, 10.0).buffer(5.0),
    "Zone2": Point(25.0, 25.0).buffer(5.0),
}

for name, polygon in zones.items():
    env.regions.add(name, polygon=polygon)

# Compute time in each zone
for name in zones.keys():
    # Get bins in this region
    region_bins = []
    for bin_idx in range(env.n_bins):
        bin_point = Point(env.bin_centers[bin_idx])
        if env.regions[name].polygon.contains(bin_point):
            region_bins.append(bin_idx)

    # Count samples
    time_in_region = np.sum(np.isin(position_bin_indices, region_bins))
    print(f"Time in {name}: {time_in_region} samples")
```
