"""
Low-level operations for power users.

This module provides primitive operations for spatial analysis
that can be composed into higher-level functionality.

Submodules
----------
binning : Point-to-bin mapping, region masks
distance : Distance fields, pairwise distances
normalize : Field normalization, clamping
smoothing : Diffusion kernels, kernel application
graph : Graph convolution, neighborhood reduction
calculus : Spatial gradient, divergence
transforms : Affine transforms, calibration
alignment : Probability mapping, similarity transforms
egocentric : Heading computation, allocentric/egocentric transforms
visibility : Viewshed, gaze, line-of-sight
basis : GLM spatial basis functions
"""

# Binning operations
# Alignment operations
from neurospatial.ops.alignment import (
    ProbabilityMappingParams,
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities,
)
from neurospatial.ops.binning import (
    TieBreakStrategy,
    clear_kdtree_cache,
    map_points_to_bins,
    regions_to_mask,
    resample_field,
)

# Calculus operations (gradient, divergence)
from neurospatial.ops.calculus import (
    compute_differential_operator,
    divergence,
    gradient,
)

# Distance operations
from neurospatial.ops.distance import (
    distance_field,
    euclidean_distance_matrix,
    geodesic_distance_between_points,
    geodesic_distance_matrix,
    neighbors_within,
    pairwise_distances,
)

# Graph operations
from neurospatial.ops.graph import (
    convolve,
    neighbor_reduce,
)

# Normalize operations
from neurospatial.ops.normalize import (
    clamp,
    combine_fields,
    normalize_field,
)

# Smoothing operations
from neurospatial.ops.smoothing import (
    apply_kernel,
    compute_diffusion_kernels,
)

# Transform operations
from neurospatial.ops.transforms import (
    Affine2D,
    Affine3D,
    AffineND,
    SpatialTransform,
    VideoCalibration,
    apply_transform_to_environment,
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
    convert_to_cm,
    convert_to_pixels,
    estimate_transform,
    flip_y,
    flip_y_data,
    from_rotation_matrix,
    identity,
    identity_nd,
    scale_2d,
    scale_3d,
    simple_scale,
    translate,
    translate_3d,
)

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Alignment
    "ProbabilityMappingParams",
    "apply_similarity_transform",
    "get_2d_rotation_matrix",
    "map_probabilities",
    # Binning
    "TieBreakStrategy",
    "clear_kdtree_cache",
    "map_points_to_bins",
    "regions_to_mask",
    "resample_field",
    # Calculus
    "compute_differential_operator",
    "divergence",
    "gradient",
    # Distance
    "distance_field",
    "euclidean_distance_matrix",
    "geodesic_distance_between_points",
    "geodesic_distance_matrix",
    "neighbors_within",
    "pairwise_distances",
    # Graph
    "convolve",
    "neighbor_reduce",
    # Normalize
    "clamp",
    "combine_fields",
    "normalize_field",
    # Smoothing
    "apply_kernel",
    "compute_diffusion_kernels",
    # Transforms - Core classes
    "Affine2D",
    "Affine3D",
    "AffineND",
    "SpatialTransform",
    "VideoCalibration",
    # Transforms - 2D factories
    "flip_y",
    "identity",
    "scale_2d",
    "translate",
    # Transforms - 3D factories
    "from_rotation_matrix",
    "identity_nd",
    "scale_3d",
    "translate_3d",
    # Transforms - Calibration
    "calibrate_from_landmarks",
    "calibrate_from_scale_bar",
    "simple_scale",
    # Transforms - Helpers
    "convert_to_cm",
    "convert_to_pixels",
    "flip_y_data",
    # Transforms - Estimation
    "apply_transform_to_environment",
    "estimate_transform",
]
