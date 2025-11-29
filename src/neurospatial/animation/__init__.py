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
subsample_frames : function
    Subsample frames to target frame rate for large-scale sessions
estimate_colormap_range_from_subset : function
    Estimate colormap vmin/vmax from random subset of frames
large_session_napari_config : function
    Get recommended napari settings for large datasets
Skeleton : class
    Immutable skeleton definition for pose tracking
MOUSE_SKELETON, RAT_SKELETON, SIMPLE_SKELETON : Skeleton
    Common skeleton presets
VideoOverlay : class
    Video background overlay for displaying recorded footage
EventOverlay : class
    Overlay for discrete timestamped events (spikes, rewards, zone entries)
SpikeOverlay : class
    Convenience alias for EventOverlay for neural spike visualization
VideoCalibration : class
    Coordinate transform from video pixels to environment cm
VideoReaderProtocol : Protocol
    Interface for video readers (for type checking custom implementations)
calibrate_video : function
    Convenience function to calibrate video to environment coordinates

Extensibility
-------------
OverlayProtocol : Protocol
    Protocol for creating custom overlays. Implement ``times``, ``interp``,
    and ``convert_to_data()`` to create custom overlay types.
PositionData, BodypartData, HeadDirectionData, VideoData, EventData : dataclass
    Internal data containers returned by ``convert_to_data()``. Custom overlays
    should return one of these types.
"""

from neurospatial.animation._video_io import VideoReaderProtocol
from neurospatial.animation.calibration import calibrate_video
from neurospatial.animation.core import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
    subsample_frames,
)
from neurospatial.animation.overlays import (
    BodypartData,
    EventData,
    EventOverlay,
    HeadDirectionData,
    OverlayProtocol,
    PositionData,
    SpikeOverlay,
    VideoData,
    VideoOverlay,
)
from neurospatial.animation.skeleton import (
    MOUSE_SKELETON,
    RAT_SKELETON,
    SIMPLE_SKELETON,
    Skeleton,
)
from neurospatial.transforms import VideoCalibration

__all__: list[str] = [
    "MOUSE_SKELETON",
    "RAT_SKELETON",
    "SIMPLE_SKELETON",
    "BodypartData",
    "EventData",
    "EventOverlay",
    "HeadDirectionData",
    "OverlayProtocol",
    "PositionData",
    "Skeleton",
    "SpikeOverlay",
    "VideoCalibration",
    "VideoData",
    "VideoOverlay",
    "VideoReaderProtocol",
    "calibrate_video",
    "estimate_colormap_range_from_subset",
    "large_session_napari_config",
    "subsample_frames",
]
