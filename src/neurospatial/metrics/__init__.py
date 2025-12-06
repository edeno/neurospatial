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

# Trajectory metrics have been moved to neurospatial.behavior.trajectory
# Re-export from new location for backward compatibility
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_turn_angles,
    mean_square_displacement,
)
from neurospatial.metrics.boundary_cells import border_score, compute_region_coverage
from neurospatial.metrics.decision_analysis import (
    DecisionAnalysisResult,
    DecisionBoundaryMetrics,
    PreDecisionMetrics,
    compute_decision_analysis,
    compute_pre_decision_metrics,
    decision_region_entry_time,
    detect_boundary_crossings,
    distance_to_decision_boundary,
    extract_pre_decision_window,
    geodesic_voronoi_labels,
    pre_decision_heading_stats,
    pre_decision_speed_stats,
)
from neurospatial.metrics.goal_directed import (
    GoalDirectedMetrics,
    approach_rate,
    compute_goal_directed_metrics,
    goal_bias,
    goal_direction,
    goal_vector,
    instantaneous_goal_alignment,
)
from neurospatial.metrics.grid_cells import (
    GridProperties,
    grid_orientation,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)
from neurospatial.metrics.head_direction import (
    HeadDirectionMetrics,
    head_direction_metrics,
    head_direction_tuning_curve,
    is_head_direction_cell,
    plot_head_direction_tuning,
)
from neurospatial.metrics.object_vector_cells import (
    ObjectVectorMetrics,
    compute_object_vector_tuning,
    is_object_vector_cell,
    object_vector_score,
    plot_object_vector_tuning,
)
from neurospatial.metrics.path_efficiency import (
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
    angular_efficiency,
    compute_path_efficiency,
    path_efficiency,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    traveled_path_length,
)
from neurospatial.metrics.phase_precession import (
    PhasePrecessionResult,
    has_phase_precession,
    phase_precession,
    plot_phase_precession,
)
from neurospatial.metrics.place_fields import (
    compute_field_emd,
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
)
from neurospatial.metrics.population import (
    PopulationCoverageResult,
    count_place_cells,
    field_density_map,
    field_overlap,
    plot_population_coverage,
    population_coverage,
    population_vector_correlation,
)
from neurospatial.metrics.spatial_view_cells import (
    SpatialViewMetrics,
    is_spatial_view_cell,
    spatial_view_cell_metrics,
)
from neurospatial.metrics.vte import (
    VTESessionResult,
    VTETrialResult,
    classify_vte,
    compute_vte_index,
    compute_vte_session,
    compute_vte_trial,
    head_sweep_from_positions,
    head_sweep_magnitude,
    integrated_absolute_rotation,
    normalize_vte_scores,
)

# NOTE: Circular statistics have been moved to neurospatial.stats.circular
# Import from there: from neurospatial.stats.circular import rayleigh_test, ...

__all__ = [
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
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
    "compute_field_emd",
    "compute_goal_directed_metrics",
    "compute_home_range",
    "compute_object_vector_tuning",
    "compute_path_efficiency",
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
    "subgoal_efficiency",
    "time_efficiency",
    "traveled_path_length",
]
