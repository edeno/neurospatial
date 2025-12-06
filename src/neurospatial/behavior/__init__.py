"""
Behavioral analysis.

This module provides tools for analyzing animal behavior including
trajectory analysis, behavioral segmentation, navigation metrics,
decision-making analysis, and reward-related computations.

Submodules
----------
trajectory : Step lengths, turn angles, MSD, home range, curvature
segmentation : Laps, trials, region crossings, runs
navigation : Path efficiency, goal-directed metrics
decisions : VTE, decision analysis, choice points
reward : Reward field computations
"""

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
