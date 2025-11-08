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
"""

from neurospatial.segmentation.regions import (
    Crossing,
    Run,
    detect_region_crossings,
    detect_runs_between_regions,
    segment_by_velocity,
)

__all__ = [
    "Crossing",
    "Run",
    "detect_region_crossings",
    "detect_runs_between_regions",
    "segment_by_velocity",
]
