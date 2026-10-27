"""
Neural encoding analysis.

This module provides tools for analyzing how neurons represent space,
including place cells, grid cells, head direction cells, border cells,
object-vector cells, and spatial view cells, and how the same cells' rate
maps compare between behavioral conditions (``compare_spatial_rates``).

Submodules
----------
spatial : Spatial rate analysis
grid : Grid cell analysis
directional : Head direction and directional-rate analysis
border : Border/boundary cell analysis
boundary_vector : Boundary-vector and egocentric boundary cell rate maps
egocentric : Object-vector and egocentric-rate analysis
view : Spatial view cell analysis
phase_precession : Theta phase precession analysis
population : Population-level metrics
"""

# Comparing rate maps between conditions
from neurospatial.encoding._comparison import (
    RateComparisonResult,
    compare_spatial_rates,
)

# Border/boundary cell analysis
# Field metrics (from _field_metrics and _metrics modules)
from neurospatial.encoding._field_metrics import (
    compute_field_emd,
    field_shape_metrics,
    field_shift_distance,
    field_size,
    field_stability,
    in_out_field_ratio,
    rate_map_centroid,
    rate_map_coherence,
)
from neurospatial.encoding._metrics import (
    information_per_second,
    mutual_information,
    selectivity,
    sparsity,
    spatial_coverage_single_cell,
    spatial_information,
)
from neurospatial.encoding._smoothing import (
    _validate_spatial_method_params as _validate_spatial_method_params,
)
from neurospatial.encoding._validation import (
    _validate_env_argument as _validate_env_argument,
)
from neurospatial.encoding._validation import _validate_times as _validate_times
from neurospatial.encoding.border import (
    border_score,
    compute_region_coverage,
    observed_wall_fraction,
    untuned_border_score,
)

# Directional rate (head direction cells)
from neurospatial.encoding.boundary_vector import (
    BoundaryVectorRateResult,
    BoundaryVectorRatesResult,
    compute_boundary_vector_rate,
    compute_boundary_vector_rates,
)
from neurospatial.encoding.directional import (
    DirectionalRateResult,
    DirectionalRatesResult,
    compute_directional_rate,
    compute_directional_rates,
    head_direction_cell_significance,
    is_head_direction_cell,
    plot_head_direction_tuning,
)

# Egocentric rate (object vector cells)
from neurospatial.encoding.egocentric import (
    ObjectVectorRateResult,
    ObjectVectorRatesResult,
    compute_egocentric_rate,
    compute_egocentric_rates,
    compute_object_vector_rate,
    compute_object_vector_rates,
    egocentric_object_vector_cell_significance,
    is_egocentric_object_vector_cell,
    is_object_vector_cell,
    object_vector_cell_significance,
    object_vector_score,
    plot_object_vector_tuning,
)

# Grid cell analysis
from neurospatial.encoding.grid import (
    GridProperties,
    autocorrelation_pixel_size,
    grid_properties,
    grid_scale,
    grid_score,
    spatial_autocorrelation,
)

# Phase precession analysis
from neurospatial.encoding.phase_precession import (
    PhasePrecessionResult,
    compute_phase_precession,
    has_phase_precession,
    plot_phase_precession,
    theta_phase,
)

# Population-level metrics
from neurospatial.encoding.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
    rate_map_correlations,
)

# Spatial rate (place/grid/border cells)
from neurospatial.encoding.spatial import (
    DirectionalPlaceFields,
    PlaceFieldsResult,
    SpatialRateResult,
    SpatialRatesResult,
    compute_directional_place_fields,
    compute_spatial_rate,
    compute_spatial_rates,
    detect_place_fields,
    has_place_field,
    is_place_cell,
    place_cell_significance,
)
from neurospatial.encoding.spike_trains import SpikeTrains

# View rate (spatial view cells)
from neurospatial.encoding.view import (
    ViewRateResult,
    ViewRatesResult,
    compute_view_rate,
    compute_view_rates,
    is_spatial_view_cell,
    spatial_view_cell_significance,
)

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__ = [  # noqa: RUF022 - organized by category
    "egocentric_object_vector_cell_significance",
    "object_vector_cell_significance",
    "spatial_view_cell_significance",
    "head_direction_cell_significance",
    "place_cell_significance",
    # Spike-time normalization
    # Ragged-spike-train container
    "SpikeTrains",
    # Border/boundary cell analysis
    "border_score",
    "compute_region_coverage",
    "observed_wall_fraction",
    "untuned_border_score",
    # Boundary-vector rate (boundary-vector and egocentric boundary cells)
    "BoundaryVectorRateResult",
    "BoundaryVectorRatesResult",
    "compute_boundary_vector_rate",
    "compute_boundary_vector_rates",
    # Directional rate (head direction cells)
    "DirectionalRateResult",
    "DirectionalRatesResult",
    "compute_directional_rate",
    "compute_directional_rates",
    # Egocentric rate (object vector cells)
    "ObjectVectorRateResult",
    "ObjectVectorRatesResult",
    "compute_egocentric_rate",
    "compute_egocentric_rates",
    "compute_object_vector_rate",
    "compute_object_vector_rates",
    # Spatial rate (place/grid/border cells)
    "SpatialRateResult",
    "SpatialRatesResult",
    "compute_spatial_rate",
    "compute_spatial_rates",
    # View rate (spatial view cells)
    "ViewRateResult",
    "ViewRatesResult",
    "compute_view_rate",
    "compute_view_rates",
    # Phase precession analysis
    "PhasePrecessionResult",
    "has_phase_precession",
    "compute_phase_precession",
    "plot_phase_precession",
    "theta_phase",
    # Object-vector cell analysis
    "is_egocentric_object_vector_cell",
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
    # Grid cell analysis
    "GridProperties",
    "autocorrelation_pixel_size",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "spatial_autocorrelation",
    # Head direction cell analysis
    "is_head_direction_cell",
    "plot_head_direction_tuning",
    # Spatial view cell analysis
    "is_spatial_view_cell",
    # Place cell analysis
    "DirectionalPlaceFields",
    "PlaceFieldsResult",
    "compute_directional_place_fields",
    "compute_field_emd",
    "detect_place_fields",
    "is_place_cell",
    "has_place_field",
    "rate_map_centroid",
    "field_shape_metrics",
    "field_shift_distance",
    "field_size",
    "field_stability",
    "compare_spatial_rates",
    "RateComparisonResult",
    "in_out_field_ratio",
    "information_per_second",
    "mutual_information",
    "rate_map_coherence",
    "selectivity",
    "sparsity",
    "spatial_information",
    "spatial_coverage_single_cell",
    # Population-level metrics
    "PopulationCoverageResult",
    "population_coverage",
    "plot_population_coverage",
    "field_density_map",
    "count_place_cells",
    "field_overlap",
    "rate_map_correlations",
    "population_vector_correlation",
]
