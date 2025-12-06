"""
Behavioral analysis.

This module provides tools for analyzing animal behavior including
trajectory analysis, behavioral segmentation, navigation metrics,
decision-making analysis, and reward-related computations.

Submodules
----------
trajectory : Step lengths, turn angles, MSD, home range, curvature
segmentation : Laps, trials, region crossings, runs
navigation : Path efficiency, goal-directed metrics, path progress
decisions : VTE, decision analysis, choice points
reward : Reward field computations
"""

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

__all__ = [  # noqa: RUF022
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
    # trajectory module
    "compute_home_range",
    "compute_step_lengths",
    "compute_trajectory_curvature",
    "compute_turn_angles",
    "mean_square_displacement",
]
