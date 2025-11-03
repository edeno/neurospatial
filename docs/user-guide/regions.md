# Regions

Regions allow you to define and manage named spatial areas (ROIs) within your environment.

## Creating Regions

```python
from shapely.geometry import Point, Polygon

# Point region
env.regions.add("StartLocation", point=(0.0, 0.0))

# Polygon region (circular)
reward_zone = Point(10.0, 10.0).buffer(5.0)
env.regions.add("RewardZone", polygon=reward_zone)

# Polygon region (custom)
corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
custom = Polygon(corners)
env.regions.add("CustomZone", polygon=custom)
```

## Managing Regions

```python
# List regions
names = env.regions.list_names()

# Access region
region = env.regions["RewardZone"]

# Remove region
del env.regions["RewardZone"]
# or
env.regions.remove("RewardZone")

# Update region (creates new immutable instance)
env.regions.update_region("StartLocation", point=(5.0, 5.0))
```

## Region Operations

```python
# Compute area
area = env.regions.area("RewardZone")

# Get center
center = env.regions.region_center("RewardZone")

# Buffer region
buffered = env.regions.buffer("RewardZone", distance=2.0)
```

See the [API Reference](../api/index.md) for complete documentation.
