"""
Neuroscience metrics for spatial analysis.

This module provides standard neuroscience metrics for place cells, boundary
cells, population-level analyses, and trajectory characterization validated
against field-standard packages (opexebo, neurocode, buzcode) and ecology
literature (Traja, adehabitatHR).

Modules
-------
place_fields
    Place field detection and single-cell spatial metrics.
population
    Population-level metrics for analyzing spatial representations.
boundary_cells
    Boundary cell metrics including border score.
(trajectory moved to neurospatial.behavior.trajectory)
    Trajectory metrics re-exported here for backward compatibility.
path_efficiency
    Path efficiency metrics for spatial navigation analysis.
goal_directed
    Goal-directed navigation metrics (goal bias, approach rate).
decision_analysis
    Spatial decision analysis at choice points (pre-decision metrics, boundary crossings).
vte
    Vicarious Trial and Error (VTE) detection and analysis.

"""

from __future__ import annotations

# Decision analysis from behavior.decisions
# VTE metrics from behavior.decisions
from neurospatial.behavior.decisions import (
    DecisionAnalysisResult,
    DecisionBoundaryMetrics,
    PreDecisionMetrics,
    VTESessionResult,
    VTETrialResult,
    classify_vte,
    compute_decision_analysis,
    compute_pre_decision_metrics,
    compute_vte_index,
    compute_vte_session,
    compute_vte_trial,
    decision_region_entry_time,
    detect_boundary_crossings,
    distance_to_decision_boundary,
    extract_pre_decision_window,
    geodesic_voronoi_labels,
    head_sweep_from_positions,
    head_sweep_magnitude,
    integrated_absolute_rotation,
    normalize_vte_scores,
    pre_decision_heading_stats,
    pre_decision_speed_stats,
)
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
    angular_efficiency,
    approach_rate,
    compute_goal_directed_metrics,
    compute_path_efficiency,
    goal_bias,
    goal_direction,
    goal_vector,
    instantaneous_goal_alignment,
    path_efficiency,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    traveled_path_length,
)

# Trajectory metrics have been moved to neurospatial.behavior.trajectory
# Re-export from new location for backward compatibility
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)

# Boundary cell metrics - from encoding module (consolidated from metrics.boundary_cells)
from neurospatial.encoding.border import border_score, compute_region_coverage

# Grid cell metrics - from encoding module (consolidated from metrics.grid_cells)
from neurospatial.encoding.grid import (
    GridProperties,
    grid_orientation,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)

# Head direction metrics - from encoding module (consolidated from metrics.head_direction)
from neurospatial.encoding.head_direction import (
    HeadDirectionMetrics,
    head_direction_metrics,
    head_direction_tuning_curve,
    is_head_direction_cell,
    plot_head_direction_tuning,
)

# Object-vector cell metrics - from encoding module (consolidated from metrics.object_vector_cells)
from neurospatial.encoding.object_vector import (
    ObjectVectorMetrics,
    compute_object_vector_tuning,
    is_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)

# Phase precession metrics - from encoding module (consolidated from metrics.phase_precession)
from neurospatial.encoding.phase_precession import (
    PhasePrecessionResult,
    has_phase_precession,
    phase_precession,
    plot_phase_precession,
)

# Place field metrics - from encoding module (consolidated from metrics.place_fields)
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

# Population metrics - from encoding module (consolidated from metrics.population)
from neurospatial.encoding.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
)

# Spatial view cell metrics - from encoding module (consolidated from metrics.spatial_view_cells)
from neurospatial.encoding.spatial_view import (
    SpatialViewMetrics,
    is_spatial_view_cell,
    spatial_view_cell_metrics,
)

# NOTE: Circular statistics have been moved to neurospatial.stats.circular
# Import from there: from neurospatial.stats.circular import rayleigh_test, ...

__all__ = [
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
    "DirectionalPlaceFields",
    "GoalDirectedMetrics",
    "GridProperties",
    "HeadDirectionMetrics",
    "ObjectVectorMetrics",
    "PathEfficiencyResult",
    "PhasePrecessionResult",
    "PopulationCoverageResult",
    "PreDecisionMetrics",
    "SpatialViewMetrics",
    "SubgoalEfficiencyResult",
    "VTESessionResult",
    "VTETrialResult",
    "angular_efficiency",
    "approach_rate",
    "border_score",
    "classify_vte",
    "compute_decision_analysis",
    "compute_directional_place_fields",
    "compute_field_emd",
    "compute_goal_directed_metrics",
    "compute_home_range",
    "compute_object_vector_tuning",
    "compute_path_efficiency",
    "compute_place_field",
    "compute_pre_decision_metrics",
    "compute_region_coverage",
    "compute_step_lengths",
    "compute_turn_angles",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "count_place_cells",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "detect_place_fields",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "field_centroid",
    "field_density_map",
    "field_overlap",
    "field_shape_metrics",
    "field_shift_distance",
    "field_size",
    "field_stability",
    "geodesic_voronoi_labels",
    "goal_bias",
    "goal_direction",
    "goal_vector",
    "grid_orientation",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "has_phase_precession",
    "head_direction_metrics",
    "head_direction_tuning_curve",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "in_out_field_ratio",
    "information_per_second",
    "instantaneous_goal_alignment",
    "integrated_absolute_rotation",
    "is_head_direction_cell",
    "is_object_vector_cell",
    "is_spatial_view_cell",
    "mean_square_displacement",
    "mutual_information",
    "normalize_vte_scores",
    "object_vector_score",
    "path_efficiency",
    "periodicity_score",
    "phase_precession",
    "plot_head_direction_tuning",
    "plot_object_vector_tuning",
    "plot_phase_precession",
    "plot_population_coverage",
    "population_coverage",
    "population_vector_correlation",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
    "rate_map_coherence",
    "selectivity",
    "shortest_path_length",
    "skaggs_information",
    "sparsity",
    "spatial_autocorrelation",
    "spatial_coverage_single_cell",
    "spatial_view_cell_metrics",
    "spikes_to_field",
    "subgoal_efficiency",
    "time_efficiency",
    "traveled_path_length",
]
