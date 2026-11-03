"""
Neural encoding analysis.

This module provides tools for analyzing how neurons represent space,
including place cells, grid cells, head direction cells, border cells,
object-vector cells, and spatial view cells, and how the same cells' rate
maps compare between behavioral conditions (``compare_spatial_rates``).

Start Here
----------
compute_spatial_rate, compute_spatial_rates : Where does the neuron fire?
    ``compute_spatial_rate(env, spike_times, times, positions)`` returns a
    :class:`SpatialRateResult`; pass a list of spike trains to
    ``compute_spatial_rates`` for a population. Bins the animal never
    visited are NaN, not zero.

has_place_field, detect_place_fields : Does the rate map have a field, and where?
    ``has_place_field`` returns a bool; ``detect_place_fields(env,
    rate_result)`` returns the field bins. Neither says the neuron is a place
    cell.

is_place_cell, place_cell_significance : Is it a place cell?
    ``is_place_cell`` is a circular-shift shuffle verdict for one neuron
    (``criterion="threshold"`` is the fast screen); ``place_cell_significance``
    tests a population. Result methods ``.is_place_cell()`` and ``.classify()``
    are threshold screens only.

spatial_cell_type_significance : Which cells are place, grid or border cells?
    ``rates.label_cell_types()`` is the fast screen.

compute_directional_rate, is_head_direction_cell : Is it a head direction cell?
    Take ``spike_times, times, headings`` (radians); no environment.

compute_object_vector_rate, compute_view_rate : Is it tuned to objects or to
    where the animal looks?
    Object-vector and spatial-view cells. ``is_object_vector_cell`` and
    ``is_spatial_view_cell`` are the matching shuffle verdicts.

compute_tuning_curve, compare_spatial_rates : Other tuning curves, and rate maps
    compared between conditions

spatial_information, border_score, spatial_autocorrelation, grid_score : Metrics
    of a rate map you already have
    ``grid_score`` takes the 2-D autocorrelogram from ``spatial_autocorrelation``,
    not the rate map.

See the "Which place-cell function?" table in the neuroscience-metrics user
guide for a longer decision table. Every name above is public at
``neurospatial.encoding``.
"""

# Comparing rate maps between conditions
from neurospatial.encoding._comparison import (
    RateComparisonResult,
    compare_spatial_rates,
)

# Border/boundary cell analysis
# Field metrics (from _field_metrics and _metrics modules)
from neurospatial.encoding._field_metrics import (
    field_emd,
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
    observed_wall_fraction,
    region_coverage,
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
    spatial_autocorrelation_radial,
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
    CellTypeSignificanceResult,
    DirectionalPlaceFields,
    PlaceCellSignificanceResult,
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
    spatial_cell_type_significance,
)
from neurospatial.encoding.spike_trains import SpikeTrains

# View rate (spatial view cells)
# Tuning curves (firing rate against one behavioral variable)
from neurospatial.encoding.tuning import (
    TuningCurveResult,
    TuningCurvesResult,
    compute_tuning_curve,
    compute_tuning_curves,
)
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
    "spatial_cell_type_significance",
    "CellTypeSignificanceResult",
    "PlaceCellSignificanceResult",
    # Spike-time normalization
    # Ragged-spike-train container
    "SpikeTrains",
    # Border/boundary cell analysis
    "border_score",
    "region_coverage",
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
    # Tuning curves (firing rate against one behavioral variable)
    "TuningCurveResult",
    "TuningCurvesResult",
    "compute_tuning_curve",
    "compute_tuning_curves",
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
    "spatial_autocorrelation_radial",
    # Head direction cell analysis
    "is_head_direction_cell",
    "plot_head_direction_tuning",
    # Spatial view cell analysis
    "is_spatial_view_cell",
    # Place cell analysis
    "DirectionalPlaceFields",
    "PlaceFieldsResult",
    "compute_directional_place_fields",
    "field_emd",
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


def __dir__() -> list[str]:
    """List the public names, so ``dir()`` shows what the documentation teaches."""
    return sorted(__all__)
