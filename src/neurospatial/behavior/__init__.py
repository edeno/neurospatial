"""
Behavioral analysis.

This module provides tools for analyzing animal behavior including
trajectory analysis, behavioral segmentation, navigation metrics,
decision-making analysis, VTE (Vicarious Trial and Error), and reward-related
computations.

Start Here
----------
compute_speed, heading_from_velocity : How fast and which way did the animal move?
    ``compute_speed(times, positions)`` and ``heading_from_velocity(times,
    positions)`` return one value per sample; both leave gaps in the tracking
    as NaN. Other trajectory measures: ``step_lengths``,
    ``compute_cumulative_distance``, ``mean_square_displacement``,
    ``compute_home_range``.

segment_by_velocity, detect_laps, detect_trials, detect_region_visits : When did
    the behavior happen?
    Detectors return :class:`Intervals` you can pass as ``epochs=`` to an
    encoding or decoding function. ``detect_laps`` and ``detect_trials`` take
    the environment first; ``segment_by_velocity`` takes ``times, positions``.

in_epochs, restrict, restrict_spike_trains : Keep only the data inside some
    intervals
    Select samples or spikes inside intervals such as running bouts or trials.

distance_to_goal, path_progress, goal_bias, compute_path_efficiency : How well did
    the animal navigate to a goal?
    Goal functions take ``env, times, positions, intervals`` and ``goal=``.
    ``compute_path_efficiency`` and ``compute_goal_directed_metrics`` return a
    result object per trial.

compute_decision_analysis, compute_vte_session : What happened at a choice point?
    ``compute_decision_analysis`` takes a ``decision_region`` and
    ``goal_regions``; ``compute_vte_session`` takes the decision ``region=``
    and the ``trials=`` to score. Both report pre-decision speed, heading and
    head sweeps (vicarious trial and error).

region_reward_field, goal_reward_field : Reward fields over the environment

The group names are public at ``neurospatial.behavior``; import every name in
this list from there. Result classes (``Lap``, ``Trial``, ``RegionVisit``,
``PathEfficiencyResult``, ``VTESessionResult``) are listed in ``__all__``.
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
    distance_to_boundary,
    distance_to_goal,
    distance_to_reward,
    goal_bias,
    goal_direction,
    goal_vector,
    graph_turn_sequence,
    heading_direction_labels,
    headings_to_direction_labels,
    instantaneous_goal_alignment,
    path_progress,
    shortest_path_length,
    subgoal_efficiency,
    time_efficiency,
    time_to_goal,
    traveled_path_length,
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
    Trial,
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    segment_by_velocity,
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
    heading_from_body_orientation,
    heading_from_velocity,
    mean_square_displacement,
    smooth_positions,
    step_lengths,
)
from neurospatial.behavior.trials import detect_trials, trials_between_visits
from neurospatial.behavior.visits import (
    RegionVisit,
    detect_region_visits,
)
from neurospatial.behavior.vte import (
    VTESessionResult,
    VTETrialResult,
    classify_vte,
    compute_vte_session,
    compute_vte_trial,
    head_sweep_from_positions,
    head_sweep_magnitude,
    normalize_vte_scores,
    vte_index,
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
    "in_epochs",
    "labels_to_epochs",
    "restrict",
    "restrict_spike_trains",
    # vte module
    "VTESessionResult",
    "VTETrialResult",
    "classify_vte",
    "vte_index",
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
    "distance_to_boundary",
    "distance_to_goal",
    "distance_to_reward",
    "goal_bias",
    "goal_direction",
    "goal_vector",
    "graph_turn_sequence",
    "heading_direction_labels",
    "headings_to_direction_labels",
    "instantaneous_goal_alignment",
    "path_progress",
    "shortest_path_length",
    "subgoal_efficiency",
    "time_efficiency",
    "time_to_goal",
    "traveled_path_length",
    # segmentation module
    "Crossing",
    "Lap",
    "Trial",
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "segment_by_velocity",
    "trajectory_similarity",
    # visits module
    "RegionVisit",
    "detect_region_visits",
    # trials module
    "detect_trials",
    "trials_between_visits",
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
    "step_lengths",
    "heading_from_body_orientation",
    "heading_from_velocity",
    "mean_square_displacement",
    "smooth_positions",
]


def __dir__() -> list[str]:
    """List the public names, so ``dir()`` shows what the documentation teaches."""
    return sorted(__all__)
