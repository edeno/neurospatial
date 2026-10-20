"""Animation module for neurospatial.

This module provides multi-backend animation capabilities for visualizing
spatial fields over time (place field learning, replay sequences, value
function evolution).

Available backends:
- Napari: GPU-accelerated interactive viewer (large-scale exploration)
- Video (MP4): Parallel video export (publications, presentations)
- HTML: Standalone interactive files (sharing, remote viewing)
- Jupyter Widget: Notebook integration (quick exploration)
- Fastplotlib: GPU drawing in Jupyter (also from a remote GPU server) or a
  desktop window, for 2-D and 3-D environments

Public API
----------
Animation, AnimationField : class
    Named field streams and synchronized overlays shared by viewing and export.
AnimationView, EventInspector : class
    Recording-time controls and identified event selection and annotation.
PositionOverlay, BodypartOverlay, HeadDirectionOverlay, ObjectVectorOverlay : class
    Sample-aligned spatial overlays.
EventOverlay, TimeSeriesOverlay, VideoOverlay : class
    Discrete events, scalar signals and calibrated video overlays.
SpikeRasterOverlay, EpochOverlay : class
    Spikes of many units as a raster, and intervals such as ripples, on the
    animation's clock (scene views/export and fastplotlib).
FastplotlibAnimation : class
    The interactive animation ``animate_fields(backend="fastplotlib")`` returns.
ScaleBarConfig : class
    Scale-bar settings shared by renderers.
Skeleton : class
    Immutable pose skeleton, with mouse, rat and simple presets.
calibrate_video : function
    Calibrate video coordinates using the canonical ops.VideoCalibration type.
subsample_frames, estimate_colormap_range_from_subset, large_session_napari_config : function
    Frame selection, colormap estimation and settings for large sessions.
"""

from neurospatial.animation._video_io import _VideoReader as _VideoReader
from neurospatial.animation._view import AnimationView
from neurospatial.animation.backends.fastplotlib_backend import FastplotlibAnimation
from neurospatial.animation.calibration import calibrate_video
from neurospatial.animation.config import ScaleBarConfig
from neurospatial.animation.core import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
    subsample_frames,
)
from neurospatial.animation.events import EventInspector
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EpochOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.animation.scene import Animation, AnimationField
from neurospatial.animation.skeleton import (
    MOUSE_SKELETON,
    RAT_SKELETON,
    SIMPLE_SKELETON,
    Skeleton,
)

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__: list[str] = [
    "MOUSE_SKELETON",
    "RAT_SKELETON",
    "SIMPLE_SKELETON",
    "Animation",
    "AnimationField",
    "AnimationView",
    "BodypartOverlay",
    "EpochOverlay",
    "EventInspector",
    "EventOverlay",
    "FastplotlibAnimation",
    "HeadDirectionOverlay",
    "ObjectVectorOverlay",
    "PositionOverlay",
    "ScaleBarConfig",
    "Skeleton",
    "SpikeRasterOverlay",
    "TimeSeriesOverlay",
    "VideoOverlay",
    "calibrate_video",
    "estimate_colormap_range_from_subset",
    "large_session_napari_config",
    "subsample_frames",
]
