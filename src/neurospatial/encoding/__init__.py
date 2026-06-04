"""
Neural encoding analysis.

This module provides tools for analyzing how neurons represent space,
including place cells, grid cells, head direction cells, border cells,
object-vector cells, and spatial view cells.

Submodules
----------
spatial : Spatial rate analysis
grid : Grid cell analysis
directional : Head direction and directional-rate analysis
border : Border/boundary cell analysis
egocentric : Object-vector and egocentric-rate analysis
view : Spatial view cell analysis
phase_precession : Theta phase precession analysis
population : Population-level metrics
"""

# Spike-time normalization (public helper used by decode_session and internal callers)
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
from neurospatial.encoding._spikes import as_spike_trains
from neurospatial.encoding.border import (
    border_score,
    compute_region_coverage,
)

# Directional rate (head direction cells)
from neurospatial.encoding.directional import (
    DirectionalRateResult,
    DirectionalRatesResult,
    compute_directional_rate,
    compute_directional_rates,
    is_head_direction_cell,
    plot_head_direction_tuning,
)

# Egocentric rate (object vector cells)
from neurospatial.encoding.egocentric import (
    EgocentricRateResult,
    EgocentricRatesResult,
    compute_egocentric_rate,
    compute_egocentric_rates,
    is_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)

# Grid cell analysis
from neurospatial.encoding.grid import (
    GridProperties,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)

# Phase precession analysis
from neurospatial.encoding.phase_precession import (
    PhasePrecessionResult,
    has_phase_precession,
    phase_precession,
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
)

# View rate (spatial view cells)
from neurospatial.encoding.view import (
    ViewRateResult,
    ViewRatesResult,
    compute_view_rate,
    compute_view_rates,
    is_spatial_view_cell,
)

__all__ = [  # noqa: RUF022 - organized by category
    # Spike-time normalization
    "as_spike_trains",
    # Border/boundary cell analysis
    "border_score",
    "compute_region_coverage",
    # Directional rate (head direction cells)
    "DirectionalRateResult",
    "DirectionalRatesResult",
    "compute_directional_rate",
    "compute_directional_rates",
    # Egocentric rate (object vector cells)
    "EgocentricRateResult",
    "EgocentricRatesResult",
    "compute_egocentric_rate",
    "compute_egocentric_rates",
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
    "phase_precession",
    "plot_phase_precession",
    "theta_phase",
    # Object-vector cell analysis
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
    # Grid cell analysis
    "GridProperties",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "periodicity_score",
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
    "rate_map_centroid",
    "field_shape_metrics",
    "field_shift_distance",
    "field_size",
    "field_stability",
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
    "population_vector_correlation",
]
