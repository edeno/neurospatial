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
"""

from neurospatial.animation.core import subsample_frames
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
    "Skeleton",
    "subsample_frames",
]
