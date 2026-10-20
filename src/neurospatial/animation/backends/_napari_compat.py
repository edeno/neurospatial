"""Feature-detected napari APIs for the supported 0.5–0.9 releases.

Keep the one necessary Qt lifecycle hook here. Rendering uses public models.
"""

from __future__ import annotations

from inspect import getattr_static
from typing import Any


def canvas_overlay(viewer: Any, name: str) -> Any:
    """Resolve canvas overlays without touching deprecated 0.9 aliases."""
    canvas = getattr(viewer, "canvas", None)
    overlays = getattr(canvas, "overlays", None)
    if overlays is not None and hasattr(overlays, name):
        return getattr(overlays, name)
    return getattr(viewer, "text_overlay" if name == "text" else name)


def scene_camera(viewer: Any) -> Any:
    """Resolve the public scene camera, falling back on pre-0.9 viewers."""
    scene = getattr(viewer, "scene", None)
    if scene is not None and hasattr(scene, "camera"):
        return scene.camera
    return viewer.camera


def canvas_grid(viewer: Any) -> Any:
    """Resolve the public grid model on modern and legacy viewers."""
    # Static detection avoids inventing features on dynamic adapter objects.
    canvas = (
        getattr(viewer, "canvas", None)
        if getattr_static(viewer, "canvas", None) is not None
        else None
    )
    return (
        canvas.grid if canvas is not None and hasattr(canvas, "grid") else viewer.grid
    )


def spatial_unit_label(units: str | None) -> str | None:
    """Return an interpretable physical unit; preserve freeform labels elsewhere."""
    if units is None:
        return None
    from pint import Unit
    from pint.errors import PintError

    try:
        Unit(units)
    except (PintError, ValueError, TypeError):
        return None
    return units


def qt_window(viewer: Any) -> Any:
    """Access the native window solely for accepted-close lifecycle handling.

    Napari has no public accepted-window-close event in the supported releases.
    """
    return getattr(getattr(viewer, "window", None), "_qt_window", None)


def screen_point_options(size: float) -> dict[str, Any]:
    """Clamp marker diameter in logical canvas pixels, independent of zoom.

    ``size`` alone is expressed in data coordinates by every supported napari
    version. The public Points canvas limits are available in 0.5 through 0.9;
    feature detection keeps errors explicit for incompatible point adapters.
    Scientific bin-cell layers must not use this display-marker helper.
    """
    from napari.layers import Points

    if not hasattr(Points, "canvas_size_limits"):
        raise RuntimeError(
            "Screen-sized overlays require napari Points.canvas_size_limits "
            "(napari >=0.5,<0.10)."
        )
    return {"size": size, "canvas_size_limits": (size, size)}


def set_spatial_units(layer: Any, units: str | None, spatial_ndim: int) -> None:
    """Declare physical axes while keeping any leading frame axes unitless."""
    from napari.layers import Layer

    units = spatial_unit_label(units)
    if units is not None and hasattr(Layer, "units"):
        spatial_ndim = min(layer.ndim, spatial_ndim)
        layer.units = ("dimensionless",) * (layer.ndim - spatial_ndim) + (
            units,
        ) * spatial_ndim
