"""
Neural encoding analysis.

This module provides tools for analyzing how neurons represent space,
including place cells, grid cells, head direction cells, border cells,
object-vector cells, and spatial view cells.

Submodules
----------
place : Place cell analysis
grid : Grid cell analysis
head_direction : Head direction cell analysis
border : Border/boundary cell analysis
object_vector : Object-vector cell analysis
spatial_view : Spatial view cell analysis
phase_precession : Theta phase precession analysis
population : Population-level metrics
"""

# Border/boundary cell analysis
from neurospatial.encoding.border import (
    border_score,
    compute_region_coverage,
)

# Grid cell analysis
from neurospatial.encoding.grid import (
    GridProperties,
    grid_orientation,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)

# Head direction cell analysis
from neurospatial.encoding.head_direction import (
    HeadDirectionMetrics,
    circular_mean,
    head_direction_metrics,
    head_direction_tuning_curve,
    is_head_direction_cell,
    mean_resultant_length,
    plot_head_direction_tuning,
    rayleigh_test,
)

# Object-vector cell analysis
from neurospatial.encoding.object_vector import (
    ObjectVectorFieldResult,
    ObjectVectorMetrics,
    compute_object_vector_field,
    compute_object_vector_tuning,
    is_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)

# Phase precession analysis
from neurospatial.encoding.phase_precession import (
    PhasePrecessionResult,
    has_phase_precession,
    phase_precession,
    plot_phase_precession,
)

# Place cell analysis
from neurospatial.encoding.place import (
    DirectionalPlaceFields,
    compute_directional_place_fields,
    compute_field_emd,
    compute_place_field,
    detect_place_fields,
    field_centroid,
    field_shape_metrics,
    field_shift_distance,
    field_size,
    field_stability,
    in_out_field_ratio,
    information_per_second,
    mutual_information,
    rate_map_coherence,
    selectivity,
    skaggs_information,
    sparsity,
    spatial_coverage_single_cell,
    spikes_to_field,
)

# Spatial view cell analysis
from neurospatial.encoding.spatial_view import (
    FieldOfView,
    SpatialViewFieldResult,
    SpatialViewMetrics,
    compute_spatial_view_field,
    compute_viewed_location,
    compute_viewshed,
    is_spatial_view_cell,
    spatial_view_cell_metrics,
    visibility_occupancy,
)

__all__ = [  # noqa: RUF022 - organized by category
    # Border/boundary cell analysis
    "border_score",
    "compute_region_coverage",
    # Phase precession analysis
    "PhasePrecessionResult",
    "has_phase_precession",
    "phase_precession",
    "plot_phase_precession",
    # Object-vector cell analysis
    "ObjectVectorFieldResult",
    "ObjectVectorMetrics",
    "compute_object_vector_field",
    "compute_object_vector_tuning",
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
    # Grid cell analysis
    "GridProperties",
    "grid_orientation",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "periodicity_score",
    "spatial_autocorrelation",
    # Head direction cell analysis
    "HeadDirectionMetrics",
    "circular_mean",
    "head_direction_metrics",
    "head_direction_tuning_curve",
    "is_head_direction_cell",
    "mean_resultant_length",
    "plot_head_direction_tuning",
    "rayleigh_test",
    # Spatial view cell analysis
    "SpatialViewFieldResult",
    "SpatialViewMetrics",
    "compute_spatial_view_field",
    "spatial_view_cell_metrics",
    "is_spatial_view_cell",
    "compute_viewed_location",
    "compute_viewshed",
    "visibility_occupancy",
    "FieldOfView",
    # Place cell analysis
    "DirectionalPlaceFields",
    "compute_directional_place_fields",
    "compute_field_emd",
    "compute_place_field",
    "detect_place_fields",
    "field_centroid",
    "field_shape_metrics",
    "field_shift_distance",
    "field_size",
    "field_stability",
    "in_out_field_ratio",
    "information_per_second",
    "mutual_information",
    "rate_map_coherence",
    "selectivity",
    "skaggs_information",
    "sparsity",
    "spatial_coverage_single_cell",
    "spikes_to_field",
]
