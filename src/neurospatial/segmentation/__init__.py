"""Behavioral segmentation and trajectory analysis.

This module provides functions for segmenting trajectories based on regions,
velocity, and behavioral epochs.

Functions
---------
detect_region_crossings
    Detect entry and exit events for a spatial region
detect_runs_between_regions
    Detect runs from source region to target region
segment_by_velocity
    Segment trajectory into movement and rest periods
detect_laps
    Detect laps on circular tracks
segment_trials
    Segment trajectory into behavioral trials (T-maze, Y-maze, etc.)
trajectory_similarity
    Compare similarity between two trajectories
detect_goal_directed_runs
    Detect goal-directed navigation segments
"""

from neurospatial.segmentation.laps import Lap, detect_laps
from neurospatial.segmentation.regions import (
    Crossing,
    Run,
    detect_region_crossings,
    detect_runs_between_regions,
    segment_by_velocity,
)
from neurospatial.segmentation.similarity import (
    detect_goal_directed_runs,
    trajectory_similarity,
)
from neurospatial.segmentation.trials import Trial, segment_trials

__all__ = [
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
]
