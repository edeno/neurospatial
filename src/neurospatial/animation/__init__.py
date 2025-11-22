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
Skeleton : class
    Immutable skeleton definition for pose tracking
MOUSE_SKELETON, RAT_SKELETON, SIMPLE_SKELETON : Skeleton
    Common skeleton presets
VideoOverlay : class
    Video background overlay for displaying recorded footage
VideoCalibration : class
    Coordinate transform from video pixels to environment cm
VideoReaderProtocol : Protocol
    Interface for video readers (for type checking custom implementations)
calibrate_video : function
    Convenience function to calibrate video to environment coordinates
"""

from neurospatial.animation._video_io import VideoReaderProtocol
from neurospatial.animation.calibration import calibrate_video
from neurospatial.animation.core import subsample_frames
from neurospatial.animation.overlays import VideoOverlay
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
    "Skeleton",
    "VideoCalibration",
    "VideoOverlay",
    "VideoReaderProtocol",
    "calibrate_video",
    "subsample_frames",
]
