"""
Behavioral analysis.

This module provides tools for analyzing animal behavior including
trajectory analysis, behavioral segmentation, navigation metrics,
decision-making analysis, VTE (Vicarious Trial and Error), and reward-related
computations.

Submodules
----------
trajectory : Step lengths, turn angles, MSD, home range, curvature
segmentation : Laps, trials, region crossings, runs
navigation : Path efficiency, goal-directed metrics, path progress
decisions : Decision analysis, choice points, Voronoi boundaries
vte : VTE (Vicarious Trial and Error) detection, head sweeping
reward : Reward field computations
"""

from neurospatial.behavior.decisions import (
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
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
    angular_efficiency,
    approach_rate,
    compute_goal_directed_metrics,
    compute_path_efficiency,
    cost_to_goal,
    distance_to_region,
    goal_bias,
    goal_direction,
    goal_pair_direction_labels,
    goal_vector,
    graph_turn_sequence,
    heading_direction_labels,
    instantaneous_goal_alignment,
    path_efficiency,
    path_progress,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    time_to_goal,
    traveled_path_length,
    trials_to_region_arrays,
)
from neurospatial.behavior.reward import (
    goal_reward_field,
    region_reward_field,
)
from neurospatial.behavior.segmentation import (
    Crossing,
    Lap,
    Run,
    Trial,
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_runs_between_regions,
    segment_by_velocity,
    segment_trials,
    trajectory_similarity,
)
from neurospatial.behavior.trajectory import (
    compute_home_range,
    compute_step_lengths,
    compute_trajectory_curvature,
    compute_turn_angles,
    mean_square_displacement,
)
from neurospatial.behavior.vte import (
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

__all__ = [  # noqa: RUF022
    # decisions module
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
    "PreDecisionMetrics",
    "compute_decision_analysis",
    "compute_pre_decision_metrics",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "geodesic_voronoi_labels",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
    # vte module
    "VTESessionResult",
    "VTETrialResult",
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "integrated_absolute_rotation",
    "normalize_vte_scores",
    # navigation module
    "GoalDirectedMetrics",
    "PathEfficiencyResult",
    "SubgoalEfficiencyResult",
    "angular_efficiency",
    "approach_rate",
    "compute_goal_directed_metrics",
    "compute_path_efficiency",
    "cost_to_goal",
    "distance_to_region",
    "goal_bias",
    "goal_direction",
    "goal_pair_direction_labels",
    "goal_vector",
    "graph_turn_sequence",
    "heading_direction_labels",
    "instantaneous_goal_alignment",
    "path_efficiency",
    "path_progress",
    "shortest_path_length",
    "subgoal_efficiency",
    "time_efficiency",
    "time_to_goal",
    "traveled_path_length",
    "trials_to_region_arrays",
    # segmentation module
    "Crossing",
    "Lap",
    "Run",
    "Trial",
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "detect_runs_between_regions",
    "segment_by_velocity",
    "segment_trials",
    "trajectory_similarity",
    # reward module
    "goal_reward_field",
    "region_reward_field",
    # trajectory module
    "compute_home_range",
    "compute_step_lengths",
    "compute_trajectory_curvature",
    "compute_turn_angles",
    "mean_square_displacement",
]
