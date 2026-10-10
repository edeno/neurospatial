"""Animation module for neurospatial.

This module provides multi-backend animation capabilities for visualizing
spatial fields over time (place field learning, replay sequences, value
function evolution).

Available backends:
- Napari: GPU-accelerated interactive viewer (large-scale exploration)
- Video (MP4): Parallel video export (publications, presentations)
- HTML: Standalone interactive files (sharing, remote viewing)
- Jupyter Widget: Notebook integration (quick exploration)

Public API
----------
PositionOverlay, BodypartOverlay, HeadDirectionOverlay, ObjectVectorOverlay : class
    Sample-aligned spatial overlays.
EventOverlay, TimeSeriesOverlay, VideoOverlay : class
    Discrete events, scalar signals and calibrated video overlays.
ScaleBarConfig : class
    Scale-bar settings shared by renderers.
Skeleton : class
    Immutable pose skeleton, with mouse, rat and simple presets.
calibrate_video : function
    Calibrate video coordinates using the canonical ops.VideoCalibration type.
subsample_frames, estimate_colormap_range_from_subset, large_session_napari_config : function
    Frame selection, colormap estimation and settings for large sessions.
"""

from neurospatial.animation.calibration import calibrate_video
from neurospatial.animation.config import ScaleBarConfig
from neurospatial.animation.core import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
    subsample_frames,
)
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.animation.skeleton import (
    MOUSE_SKELETON,
    RAT_SKELETON,
    SIMPLE_SKELETON,
    Skeleton,
)

__all__: list[str] = [
    "MOUSE_SKELETON",
    "RAT_SKELETON",
    "SIMPLE_SKELETON",
    "BodypartOverlay",
    "EventOverlay",
    "HeadDirectionOverlay",
    "ObjectVectorOverlay",
    "PositionOverlay",
    "ScaleBarConfig",
    "Skeleton",
    "TimeSeriesOverlay",
    "VideoOverlay",
    "calibrate_video",
    "estimate_colormap_range_from_subset",
    "large_session_napari_config",
    "subsample_frames",
]
