"""Track graph annotation entry point.

This module provides the public API for interactive track graph annotation,
allowing users to build track graphs on video frames using napari.

The output integrates with `Environment.from_graph()` for creating 1D
linearized track environments.

Examples
--------
>>> # Annotate track graph on video frame
>>> from neurospatial.annotation import annotate_track_graph
>>> result = annotate_track_graph("maze.mp4")  # doctest: +SKIP
>>> env = result.to_environment(bin_size=2.0)  # doctest: +SKIP

>>> # With calibration for pixel-to-cm conversion
>>> from neurospatial.transforms import VideoCalibration
>>> result = annotate_track_graph("maze.mp4", calibration=calib)  # doctest: +SKIP
>>> # node_positions now in cm
"""

from __future__ import annotations

# Re-export TrackGraphResult from helpers module for public API
from neurospatial.annotation._track_helpers import TrackGraphResult

__all__ = [
    "TrackGraphResult",
]
