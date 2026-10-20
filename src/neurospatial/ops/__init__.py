"""
Low-level operations for power users.

This module provides primitive operations for spatial analysis
that can be composed into higher-level functionality.

Submodules
----------
mapping : Point-to-bin mapping, region masks, field resampling
distance : Distance fields, pairwise distances
normalize : Field normalization, clamping
diffusion : Finite-volume heat operator and its eigenbasis apply-path
smoothing : Diffusion-kernel primitive, kernel application
graph : Graph convolution, neighborhood reduction
calculus : Spatial gradient, divergence
transforms : Affine transforms, calibration
alignment : Probability mapping, similarity transforms
boundary_vectors : Distance and direction to the arena's walls
egocentric : Allocentric/egocentric transforms
feature_sampling : Feature-map values at positions, by bin containment
visibility : Viewshed, gaze, line-of-sight
basis : GLM spatial basis functions
"""

# Alignment operations
from neurospatial.ops._linear_allocation import (
    _linear_grid_occupancy as _linear_grid_occupancy,
)
from neurospatial.ops._polygon_allocation import (
    _polygon_linear_occupancy as _polygon_linear_occupancy,
)
from neurospatial.ops._polygon_allocation import (
    _RetainedLinearInterval as _RetainedLinearInterval,
)
from neurospatial.ops._polygon_allocation import (
    _uses_polygon_allocation as _uses_polygon_allocation,
)
from neurospatial.ops._transitions import _count_transitions as _count_transitions
from neurospatial.ops._transitions import (
    _diffusion_transitions as _diffusion_transitions,
)
from neurospatial.ops._transitions import (
    _random_walk_transitions as _random_walk_transitions,
)
from neurospatial.ops._types import KernelMode
from neurospatial.ops.alignment import (
    ProbabilityMappingParams,
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities,
)

# Basis functions (GLM spatial basis)
from neurospatial.ops.basis import (
    chebyshev_filter_basis,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    plot_basis_functions,
    select_basis_centers,
    spatial_basis,
)

# Distance and direction to walls
from neurospatial.ops.boundary_vectors import (
    boundary_directions,
    compute_boundary_distances,
    nearest_boundary,
)

# Calculus operations (gradient, divergence)
from neurospatial.ops.calculus import (
    _compute_differential_operator as _compute_differential_operator,
)
from neurospatial.ops.calculus import (
    _compute_divergence_operator as _compute_divergence_operator,
)
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.diffusion import _HEAT_KERNEL_RANK_TOL as _HEAT_KERNEL_RANK_TOL
from neurospatial.ops.diffusion import _RESOLUTION_MARGIN as _RESOLUTION_MARGIN
from neurospatial.ops.diffusion import MRFBasis
from neurospatial.ops.diffusion import _above_component_floor as _above_component_floor
from neurospatial.ops.diffusion import (
    _component_support_mask as _component_support_mask,
)
from neurospatial.ops.diffusion import _diffusion_apply as _diffusion_apply
from neurospatial.ops.diffusion import _diffusion_apply_rtol as _diffusion_apply_rtol
from neurospatial.ops.diffusion import _DiffusionEigenbasis as _DiffusionEigenbasis
from neurospatial.ops.diffusion import _ensure_global_modes as _ensure_global_modes
from neurospatial.ops.diffusion import _live_components as _live_components
from neurospatial.ops.diffusion import (
    _masked_diffusion_average as _masked_diffusion_average,
)
from neurospatial.ops.diffusion import _MRFEigenbasis as _MRFEigenbasis
from neurospatial.ops.diffusion import _null_mode_indices as _null_mode_indices
from neurospatial.ops.diffusion import _resolve_basis as _resolve_basis
from neurospatial.ops.diffusion import _select_live_basis as _select_live_basis
from neurospatial.ops.diffusion import _symmetric_conjugate as _symmetric_conjugate
from neurospatial.ops.diffusion import _target_live_rank as _target_live_rank
from neurospatial.ops.diffusion import _validate_occupancy as _validate_occupancy

# Distance operations
from neurospatial.ops.distance import (
    distance_field,
    euclidean_distance_matrix,
    geodesic_distance_between_points,
    geodesic_distance_matrix,
    neighbors_within,
    pairwise_distances,
)

# Egocentric operations
from neurospatial.ops.egocentric import (
    EgocentricFrame,
    allocentric_to_egocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    egocentric_to_allocentric,
)
from neurospatial.ops.egocentric import (
    _velocity_heading_and_speed as _velocity_heading_and_speed,
)
from neurospatial.ops.egocentric import _wrap_bearing as _wrap_bearing

# Feature maps sampled at positions
from neurospatial.ops.feature_sampling import (
    SpatialFeatureResult,
    sample_egocentric_features,
    sample_spatial_features,
)

# Graph operations
from neurospatial.ops.graph import (
    graph_convolve,
    neighbor_reduce,
)

# Point-to-bin mapping, region masks and field resampling
from neurospatial.ops.mapping import TieBreakStrategy, regions_to_mask, resample_field
from neurospatial.ops.mapping import _nearest_bin_query as _nearest_bin_query
from neurospatial.ops.mapping import _regions_mask as _regions_mask
from neurospatial.ops.mapping import _typical_bin_spacing as _typical_bin_spacing
from neurospatial.ops.mapping import (
    _validate_nearest_options as _validate_nearest_options,
)

# Normalize operations
from neurospatial.ops.normalize import (
    clamp,
    combine_fields,
    normalize_field,
)
from neurospatial.ops.smoothing import _gaussian_kernel as _gaussian_kernel

# Smoothing operations
from neurospatial.ops.smoothing import apply_kernel, compute_diffusion_kernels

# Transform operations
from neurospatial.ops.transforms import (
    Affine2D,
    AffineND,
    SpatialTransform,
    VideoCalibration,
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

# Visibility operations
from neurospatial.ops.visibility import (
    FieldOfView,
    ViewshedResult,
    compute_view_field,
    compute_viewed_location,
    compute_viewshed,
    compute_viewshed_trajectory,
    visibility_occupancy,
    visible_cues,
)

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    "MRFBasis",
    # Alignment
    "ProbabilityMappingParams",
    "apply_similarity_transform",
    "get_2d_rotation_matrix",
    "map_probabilities",
    # Binning
    "TieBreakStrategy",
    "regions_to_mask",
    "resample_field",
    # Boundary vectors
    "boundary_directions",
    "compute_boundary_distances",
    "nearest_boundary",
    # Calculus
    "divergence",
    "gradient",
    # Distance
    "distance_field",
    "euclidean_distance_matrix",
    "geodesic_distance_between_points",
    "geodesic_distance_matrix",
    "neighbors_within",
    "pairwise_distances",
    # Egocentric
    "EgocentricFrame",
    "allocentric_to_egocentric",
    "compute_egocentric_bearing",
    "compute_egocentric_distance",
    "egocentric_to_allocentric",
    # Feature sampling
    "SpatialFeatureResult",
    "sample_egocentric_features",
    "sample_spatial_features",
    # Graph
    "graph_convolve",
    "neighbor_reduce",
    # Normalize
    "clamp",
    "combine_fields",
    "normalize_field",
    # Smoothing
    "KernelMode",
    "apply_kernel",
    "compute_diffusion_kernels",
    # Transforms - Core classes
    "Affine2D",
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
    "estimate_transform",
    # Visibility
    "FieldOfView",
    "ViewshedResult",
    "compute_view_field",
    "compute_viewed_location",
    "compute_viewshed",
    "compute_viewshed_trajectory",
    "visibility_occupancy",
    "visible_cues",
    # Basis functions
    "chebyshev_filter_basis",
    "geodesic_rbf_basis",
    "heat_kernel_wavelet_basis",
    "plot_basis_functions",
    "select_basis_centers",
    "spatial_basis",
]
