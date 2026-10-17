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
egocentric : Heading computation, allocentric/egocentric transforms
visibility : Viewshed, gaze, line-of-sight
basis : GLM spatial basis functions
"""

# Alignment operations
# Visibility operations - imported lazily to avoid circular imports
# These modules depend on Environment, which creates a circular import when
# ops/__init__.py is imported during Environment initialization.
# Use direct imports: from neurospatial.ops.visibility import FieldOfView
import importlib as _importlib

from neurospatial.ops._linear_allocation import (
    linear_grid_occupancy as linear_grid_occupancy,
)
from neurospatial.ops._polygon_allocation import (
    RetainedLinearInterval as RetainedLinearInterval,
)
from neurospatial.ops._polygon_allocation import (
    polygon_linear_occupancy as polygon_linear_occupancy,
)
from neurospatial.ops._polygon_allocation import source_grid as source_grid
from neurospatial.ops._polygon_allocation import (
    uses_polygon_allocation as uses_polygon_allocation,
)
from neurospatial.ops._transitions import count_transitions as count_transitions
from neurospatial.ops._transitions import diffusion_transitions as diffusion_transitions
from neurospatial.ops._transitions import (
    random_walk_transitions as random_walk_transitions,
)
from neurospatial.ops._types import KernelMode as KernelMode
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

# Calculus operations (gradient, divergence)
from neurospatial.ops.calculus import (
    compute_differential_operator as compute_differential_operator,
)
from neurospatial.ops.calculus import (
    compute_divergence_operator as compute_divergence_operator,
)
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.diffusion import HEAT_KERNEL_RANK_TOL as HEAT_KERNEL_RANK_TOL
from neurospatial.ops.diffusion import RESOLUTION_MARGIN as RESOLUTION_MARGIN
from neurospatial.ops.diffusion import DiffusionEigenbasis as DiffusionEigenbasis
from neurospatial.ops.diffusion import MRFBasis
from neurospatial.ops.diffusion import MRFEigenbasis as MRFEigenbasis
from neurospatial.ops.diffusion import above_component_floor as above_component_floor
from neurospatial.ops.diffusion import component_support_mask as component_support_mask
from neurospatial.ops.diffusion import diffusion_apply as diffusion_apply
from neurospatial.ops.diffusion import diffusion_apply_rtol as diffusion_apply_rtol
from neurospatial.ops.diffusion import ensure_global_modes as ensure_global_modes
from neurospatial.ops.diffusion import heat_kernel_from_W as heat_kernel_from_W
from neurospatial.ops.diffusion import live_components as live_components
from neurospatial.ops.diffusion import (
    masked_diffusion_average as masked_diffusion_average,
)
from neurospatial.ops.diffusion import null_mode_indices as null_mode_indices
from neurospatial.ops.diffusion import resolve_basis as resolve_basis
from neurospatial.ops.diffusion import select_live_basis as select_live_basis
from neurospatial.ops.diffusion import symmetric_conjugate as symmetric_conjugate
from neurospatial.ops.diffusion import target_live_rank as target_live_rank
from neurospatial.ops.diffusion import validate_occupancy as validate_occupancy

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
    heading_from_body_orientation,
    heading_from_velocity,
)
from neurospatial.ops.egocentric import (
    velocity_heading_and_speed as velocity_heading_and_speed,
)
from neurospatial.ops.egocentric import wrap_bearing as wrap_bearing

# Graph operations
from neurospatial.ops.graph import (
    graph_convolve,
    neighbor_reduce,
)

# Point-to-bin mapping, region masks and field resampling
from neurospatial.ops.mapping import TieBreakStrategy, regions_to_mask, resample_field
from neurospatial.ops.mapping import nearest_bin_query as nearest_bin_query
from neurospatial.ops.mapping import regions_mask as regions_mask
from neurospatial.ops.mapping import typical_bin_spacing as typical_bin_spacing
from neurospatial.ops.mapping import (
    validate_nearest_options as validate_nearest_options,
)

# Normalize operations
from neurospatial.ops.normalize import (
    clamp,
    combine_fields,
    normalize_field,
)

# Smoothing operations
from neurospatial.ops.smoothing import apply_kernel, compute_diffusion_kernels
from neurospatial.ops.smoothing import gaussian_kernel as gaussian_kernel

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


def __getattr__(name: str):
    """Lazy import for visibility module to avoid circular imports."""
    _visibility_exports = {
        "FieldOfView",
        "ViewshedResult",
        "compute_view_field",
        "compute_viewed_location",
        "compute_viewshed",
        "compute_viewshed_trajectory",
        "visibility_occupancy",
        "visible_cues",
    }
    if name in _visibility_exports:
        _visibility = _importlib.import_module("neurospatial.ops.visibility")
        return getattr(_visibility, name)
    msg = f"module 'neurospatial.ops' has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List module attributes, including lazily-exported names, for autocomplete.

    Merges the eagerly-imported module globals with everything in ``__all__``
    (which includes the lazily-loaded visibility exports) so interactive tools
    and ``dir()`` surface the full public surface, not just the already-imported
    names.
    """
    return sorted(set(globals()) | set(__all__))


# Names imported here but not listed in ``__all__`` are shared with other
# neurospatial packages. They are internal, not public API.
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
    "heading_from_body_orientation",
    "heading_from_velocity",
    # Graph
    "graph_convolve",
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
