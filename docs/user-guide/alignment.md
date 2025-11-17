# Alignment & Transforms

neurospatial provides tools for transforming and aligning spatial representations between 2D and 3D environments.

## 2D Affine Transformations

For 2D environments, use the `Affine2D` class:

```python
from neurospatial.transforms import Affine2D, translate, rotate, scale

# Factory functions (composable)
transform = translate(10, 20)
R = rotate(np.pi/4)  # 45 degrees
S = scale(1.5, 1.5)

# Compose transformations
transform = T @ R @ S

# Apply to points
transformed_points = transform(points_2d)
```

## 3D Affine Transformations (v0.3.0+)

For 3D environments, use `AffineND` or the convenience alias `Affine3D`:

```python
from neurospatial.transforms import translate_3d, scale_3d, from_rotation_matrix
from scipy.spatial.transform import Rotation

# 3D translation
transform = translate_3d(10, 20, 30)

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
transformed_points = combined(points_3d)
```

## Transform Estimation from Point Correspondences

Estimate transformations from matching point pairs (works for 2D or 3D):

```python
from neurospatial import estimate_transform

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
aligned_points = T_rigid(new_points)
```

## Transforming Environments

Apply transformations to entire environments:

```python
from neurospatial import Environment, apply_transform_to_environment
from neurospatial.transforms import translate_3d

# Create 3D environment
env_3d = Environment.from_samples(positions_3d, bin_size=5.0)

# Transform the environment
transform = translate_3d(100, 200, 50)
env_transformed = apply_transform_to_environment(env_3d, T, name="shifted")

# Transformations update:
# - Bin centers
# - Dimension ranges
# - Edge distances
# - Regions (if any)
```

## Probability Alignment

Map probability distributions between environments (works for 2D or 3D):

```python
from neurospatial.alignment import map_probabilities, get_2d_rotation_matrix
from scipy.spatial.transform import Rotation

# 2D alignment with rotation and scaling
aligned_probs_2d = map_probabilities(
    source_env=env1,
    target_env=env2,
    source_probabilities=probs1,
    source_rotation_matrix=get_2d_rotation_matrix(angle_degrees=45),
    source_scale=1.2
)

# 3D alignment with rotation (v0.3.0+)
R_3d = Rotation.from_euler('xyz', [45, 30, 60], degrees=True).as_matrix()
aligned_probs_3d = map_probabilities(
    source_env=env_3d_1,
    target_env=env_3d_2,
    source_probabilities=probs_3d,
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
from neurospatial import Environment, estimate_transform, apply_transform_to_environment

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
env1_aligned = apply_transform_to_environment(env1, T, name="session1_aligned")

# Now compare neural activity, place fields, etc. in aligned coordinate frame
```

See the [API Reference](../api/index.md) for complete documentation.
