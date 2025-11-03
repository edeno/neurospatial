# Alignment & Transforms

neurospatial provides tools for transforming and aligning spatial representations between environments.

## Affine Transformations

```python
from neurospatial.transforms import Affine2D

# Create transformation
transform = Affine2D()
transform.rotate(np.pi/4)  # 45 degrees
transform.translate(10, 20)
transform.scale(1.5, 1.5)

# Apply to points
transformed_points = transform.transform(points)
```

## Probability Alignment

Map probability distributions between environments:

```python
from neurospatial.alignment import map_probabilities_to_nearest_target_bin

# Align distributions from source to target environment
aligned_probs = map_probabilities_to_nearest_target_bin(
    source_env=env1,
    target_env=env2,
    source_probabilities=probs1
)
```

See the [API Reference](../api/index.md) for complete documentation.
