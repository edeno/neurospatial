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

__all__ = [
    "TieBreakStrategy",
    "apply_kernel",
    "clamp",
    "clear_kdtree_cache",
    "combine_fields",
    "compute_differential_operator",
    "compute_diffusion_kernels",
    "convolve",
    "distance_field",
    "divergence",
    "euclidean_distance_matrix",
    "geodesic_distance_between_points",
    "geodesic_distance_matrix",
    "gradient",
    "map_points_to_bins",
    "neighbor_reduce",
    "neighbors_within",
    "normalize_field",
    "pairwise_distances",
    "regions_to_mask",
    "resample_field",
]
