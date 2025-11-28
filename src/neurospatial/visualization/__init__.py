"""Visualization utilities for neurospatial.

This module provides visualization helpers including scale bars for
publication-quality figures.
"""

from neurospatial.visualization.scale_bar import (
    ScaleBarConfig,
    add_scale_bar_to_axes,
    compute_nice_length,
    configure_napari_scale_bar,
    format_scale_label,
)

__all__ = [
    "ScaleBarConfig",
    "add_scale_bar_to_axes",
    "compute_nice_length",
    "configure_napari_scale_bar",
    "format_scale_label",
]
