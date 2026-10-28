"""
Behavioral analysis.

This module provides tools for analyzing animal behavior including
trajectory analysis, behavioral segmentation, navigation metrics,
decision-making analysis, VTE (Vicarious Trial and Error), and reward-related
computations.

Submodules
----------
trajectory : Speed, movement and body heading, step lengths, turn angles,
    MSD, home range, curvature
epochs : Restrict times / arrays / spike trains to time intervals (epochs);
    convert per-sample labels to epochs
segmentation : Laps, trials, region crossings, runs
navigation : Path efficiency, goal-directed metrics, path progress,
    distance to regions, boundaries and rewards
decisions : Decision analysis, choice points, Voronoi boundaries
vte : VTE (Vicarious Trial and Error) detection, head sweeping
reward : Reward field computations
resampling : Distance resampling
visits : Observed region visits and entry-to-entry transitions
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
from neurospatial.behavior.epochs import (
    LabelEpochsResult,
    in_epochs,
    labels_to_epochs,
    restrict,
    restrict_spike_trains,
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
    distance_to_boundary,
    distance_to_region,
    distance_to_reward,
    goal_bias,
    goal_direction,
    goal_pair_direction_labels,
    goal_vector,
    graph_turn_sequence,
    heading_direction_labels,
    instantaneous_goal_alignment,
    path_progress,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    time_to_goal,
    traveled_path_length,
    trials_to_region_arrays,
)
from neurospatial.behavior.resampling import (
    ResampledPathResult,
    resample_by_distance,
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
    laps_to_direction_labels,
    running_direction_labels,
    runs_to_direction_labels,
    segment_by_velocity,
    segment_trials,
    trajectory_similarity,
)
from neurospatial.behavior.trajectory import (
    CumulativeDistanceResult,
    MSDResult,
    compute_cumulative_distance,
    compute_home_range,
    compute_movement_angular_velocity,
    compute_planar_curvature,
    compute_planar_turn_angles,
    compute_speed,
    compute_step_lengths,
    compute_trajectory_curvature,
    compute_turn_angles,
    heading_from_body_orientation,
    heading_from_velocity,
    mean_square_displacement,
)
from neurospatial.behavior.visits import (
    RegionTransition,
    RegionVisit,
    detect_region_visits,
    region_transitions,
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
    # epochs module
    "LabelEpochsResult",
    "in_epochs",
    "labels_to_epochs",
    "restrict",
    "restrict_spike_trains",
    # vte module
    "VTESessionResult",
    "VTETrialResult",
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
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
    "distance_to_boundary",
    "distance_to_region",
    "distance_to_reward",
    "goal_bias",
    "goal_direction",
    "goal_pair_direction_labels",
    "goal_vector",
    "graph_turn_sequence",
    "heading_direction_labels",
    "instantaneous_goal_alignment",
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
    "laps_to_direction_labels",
    "running_direction_labels",
    "runs_to_direction_labels",
    "segment_by_velocity",
    "segment_trials",
    "trajectory_similarity",
    # visits module
    "RegionTransition",
    "RegionVisit",
    "detect_region_visits",
    "region_transitions",
    # resampling module
    "ResampledPathResult",
    "resample_by_distance",
    # reward module
    "goal_reward_field",
    "region_reward_field",
    # trajectory module
    "CumulativeDistanceResult",
    "MSDResult",
    "compute_cumulative_distance",
    "compute_home_range",
    "compute_movement_angular_velocity",
    "compute_planar_curvature",
    "compute_planar_turn_angles",
    "compute_speed",
    "compute_step_lengths",
    "compute_trajectory_curvature",
    "compute_turn_angles",
    "heading_from_body_orientation",
    "heading_from_velocity",
    "mean_square_displacement",
]
