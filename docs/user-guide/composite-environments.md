# Composite Environments

`CompositeEnvironment` allows you to merge multiple environments with automatic bridge inference.

## Basic Usage

```python
from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment

# Create multiple environments
env1 = Environment.from_samples(data1, bin_size=2.0, name="Arena1")
env2 = Environment.from_samples(data2, bin_size=2.0, name="Arena2")

# Merge environments
composite = CompositeEnvironment(
    environments=[env1, env2],
    names=["Arena1", "Arena2"]
)
```

## Automatic Bridge Inference

CompositeEnvironment automatically infers connections (bridges) between environments based on mutual nearest neighbors.

See the [API Reference](../api/index.md) for complete documentation.
