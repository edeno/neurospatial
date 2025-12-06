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

__all__ = [  # noqa: RUF022 - organized by category
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
