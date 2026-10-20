"""Napari scale bars, and the shared matplotlib scale-bar helpers for animation.

This module configures napari's built-in scale bar
(:func:`configure_napari_scale_bar`) so it keeps the environment's units. The
matplotlib scale bar and its "nice" 1-2-5 length rule live in
``neurospatial.layout``, below the environment, so that
``Environment.plot`` can draw them; this module re-exports
``ScaleBarConfig``, ``add_scale_bar_to_axes``, ``compute_nice_length`` and
``format_scale_label`` for the animation backends.

Imports
-------
``ScaleBarConfig`` can be imported from either location:

>>> from neurospatial.animation.config import ScaleBarConfig  # Direct
>>> from neurospatial.animation import ScaleBarConfig  # Via animation package

Note: This module provides **visualization scale bars** (showing physical scale on plots).
This is distinct from the existing **calibration scale bar** feature
(`calibrate_from_scale_bar()`, `calibrate_video(scale_bar=...)`) which is used for
video-to-environment coordinate transforms.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from neurospatial._logging import logger
from neurospatial.animation.backends._napari_compat import (
    canvas_overlay,
    scene_camera,
    spatial_unit_label,
)
from neurospatial.layout import (
    ScaleBarConfig,
)
from neurospatial.layout import (
    _add_scale_bar_to_axes as add_scale_bar_to_axes,
)
from neurospatial.layout import (
    _compute_nice_length as compute_nice_length,
)
from neurospatial.layout import (
    _format_scale_label as format_scale_label,
)

if TYPE_CHECKING:
    import napari

# Constants for napari scale bar patching
_SCALE_CHANGE_THRESHOLD = 1e-4  # Threshold for detecting zoom changes in log10 space
_VISPY_IDENTITY_YZW = [1, 1, 1]  # Y, Z, W components of vispy 4D transform

__all__ = [
    "ScaleBarConfig",
    "add_scale_bar_to_axes",
    "compute_nice_length",
    "configure_napari_scale_bar",
    "format_scale_label",
]


def _patch_napari_scale_bar_format(viewer: napari.Viewer) -> None:
    """Patch napari's scale bar to preserve user-specified units.

    By default, napari's scale bar uses pint format string "{:g~#P}" which
    auto-converts units (e.g., cm → mm) because the "#" flag triggers
    pint's "compact" formatting. This patch replaces the format string
    with "{:g~P}" which preserves the original unit.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to patch.

    Notes
    -----
    This is a workaround for napari's behavior of auto-converting units.
    The patch modifies the VispyScaleBarOverlay's _on_zoom_change method
    to use a different format string.

    The patch is applied per-viewer and does not affect other viewers.

    **Compatibility**: Supports the legacy and napari 0.9 formatting hooks.
    The patch relies on napari's
    internal API (``_vispy.overlays.scale_bar.VispyScaleBarOverlay``) which
    may change in future versions. If the patch fails, the scale bar will
    still work but will use napari's default unit conversion (cm → mm).

    .. todo::
       If napari adds a public API for format string customization
       (e.g., ``viewer.scale_bar.format_string``), migrate to that instead
       of monkey-patching internal methods.
    """
    try:
        # Access the vispy scale bar overlay
        # napari stores overlays in viewer.window._qt_viewer.canvas._overlay_base
        # but we can access the vispy overlay through the internal structures
        from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay

        # Napari 0.9 groups multiple visuals per viewer overlay. Keep both
        # representations inside this optional formatting compatibility hook.
        canvas = viewer.window._qt_viewer.canvas
        grouped = getattr(canvas, "_viewer_overlay_to_visual", None)
        visuals = (
            [visual for group in grouped.values() for visual in group]
            if grouped is not None
            else list(canvas._overlay_to_visual.values())
        )
        for overlay in visuals:
            if isinstance(overlay, VispyScaleBarOverlay):
                # 0.9 computes units from layers and has a different rendering
                # hook. Wrap the final draw so zoom/resize events retain the
                # original physical unit without replacing connected handlers.
                if hasattr(overlay, "_on_rendering_change"):
                    if getattr(overlay, "_neurospatial_unit_format", False):
                        return
                    draw = overlay._on_rendering_change

                    def draw_in_original_units(*, _overlay=overlay, _draw=draw):
                        length = _overlay._current_length * _overlay._scale
                        _overlay.node.text.text = f"{length * _overlay._unit:g~P}"
                        _draw()

                    overlay._on_rendering_change = draw_in_original_units
                    overlay._neurospatial_unit_format = True
                    overlay._on_size_or_zoom_change(force=True)
                    return

                def patched_on_zoom_change(*, force: bool = False, _overlay=overlay):
                    """Patched zoom handler that uses unit-preserving format."""
                    import numpy as np

                    # If scale has not changed, do not redraw
                    scale = 1 / scene_camera(_overlay.viewer).zoom
                    scale_diff = abs(np.log10(_overlay._scale) - np.log10(scale))
                    if scale_diff < _SCALE_CHANGE_THRESHOLD and not force:
                        return
                    _overlay._scale = scale

                    # Store the original unit BEFORE any conversion
                    original_unit = _overlay._unit.units

                    scale_canvas2world = _overlay._scale
                    target_canvas_pixels = _overlay._target_length
                    target_world_pixels = scale_canvas2world * target_canvas_pixels

                    if _overlay.overlay.length is not None:
                        target_canvas_pixels = (
                            _overlay.overlay.length / scale_canvas2world
                        )
                        new_dim = _overlay.overlay.length * original_unit
                    else:
                        target_world_pixels_rounded, new_dim = (
                            _overlay._calculate_best_length(target_world_pixels)
                        )
                        target_canvas_pixels = (
                            target_world_pixels_rounded / scale_canvas2world
                        )
                        # _calculate_best_length calls to_compact() which converts
                        # cm → mm. Convert back to original unit to preserve user's choice.
                        new_dim = new_dim.to(original_unit)

                    scale = target_canvas_pixels
                    # Vispy uses 4D transform: [X_scale, Y, Z, W]
                    _overlay.node.transform.scale = [scale, *_VISPY_IDENTITY_YZW]

                    # Use {:g~P} instead of {:g~#P} to preserve units
                    # The "#" flag triggers pint's compact formatting (cm→mm)
                    _overlay.node.text.text = f"{new_dim:g~P}"

                    _overlay.x_size = scale
                    # Call parent's _on_position_change
                    from napari._vispy.overlays.base import VispyCanvasOverlay

                    VispyCanvasOverlay._on_position_change(_overlay)

                # Replace the method
                overlay._on_zoom_change = patched_on_zoom_change
                # Trigger a refresh to apply the new format
                overlay._on_zoom_change(force=True)
                break
    except ImportError as e:
        # Napari internal module structure changed
        logger.debug(
            f"Scale bar unit preservation patch failed (import): {e}. "
            "Scale bar will use napari's default unit conversion."
        )
    except AttributeError as e:
        # Expected attributes missing from overlay
        logger.debug(
            f"Scale bar unit preservation patch failed (API): {e}. "
            "Scale bar will use napari's default unit conversion."
        )


def configure_napari_scale_bar(
    viewer: napari.Viewer,
    units: str | None = None,
    config: ScaleBarConfig | None = None,
) -> None:
    """Configure napari's native scale bar.

    Napari has built-in scale bar support with automatic sizing based on
    the current zoom level and layer scale.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    units : str | None
        Unit label (e.g., "cm", "um"). If None, no unit shown.
    config : ScaleBarConfig | None
        Configuration object. Note: ``config.length`` is ignored because
        napari auto-sizes based on layer scale and zoom level.

    Notes
    -----
    **Napari scale bar auto-sizing**: Unlike matplotlib where we set an
    explicit length, napari calculates the scale bar length dynamically
    based on the current view. This is actually beneficial for interactive
    use but means ``ScaleBarConfig.length`` has no effect in napari.

    **Layer scale**: For napari's scale bar to show correct units, the
    image layer must have the correct ``scale`` attribute set. This is
    typically set via ``EnvScale`` in ``animation/transforms.py``. If scale
    is (1, 1), the scale bar shows pixels regardless of ``unit`` setting.

    Supported config attributes:

    - position: Maps to napari's position names
    - color: Scale bar color
    - font_size: Text size

    Ignored config attributes (matplotlib-only):

    - length, background, background_alpha, pad, sep, label_top, box_style

    Examples
    --------
    >>> import napari
    >>> viewer = napari.Viewer(show=False)  # doctest: +SKIP
    >>> configure_napari_scale_bar(viewer, units="cm")  # doctest: +SKIP
    >>> viewer.close()  # doctest: +SKIP
    """
    if config is None:
        config = ScaleBarConfig()

    # Position mapping: our API -> napari names
    position_map = {
        "lower right": "bottom_right",
        "lower left": "bottom_left",
        "upper right": "top_right",
        "upper left": "top_left",
    }

    scale_bar = canvas_overlay(viewer, "scale_bar")
    if units is not None and spatial_unit_label(units) is None:
        scale_bar.visible = False
        warnings.warn(
            f"Napari cannot interpret the environment's unit label {units!r}; "
            "the scale bar is hidden. Use recognized physical units to display it.",
            UserWarning,
            stacklevel=2,
        )
        return
    scale_bar.visible = True
    scale_bar.position = position_map.get(config.position, "bottom_right")
    scale_bar.font_size = config.font_size

    if units and getattr(viewer, "canvas", None) is None:
        # Since 0.9 the scale-bar unit setter is a deprecated no-op. Scene
        # adapters declare units on their owned layers instead.
        scale_bar.unit = units

    if config.color:
        scale_bar.color = config.color

    # Monkey-patch napari's scale bar to preserve user-specified units
    # By default, napari uses format string "{:g~#P}" which converts cm→mm
    # We patch _on_zoom_change to use "{:g~P}" which preserves the unit
    _patch_napari_scale_bar_format(viewer)
