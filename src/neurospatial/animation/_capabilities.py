"""What each scene view can draw, checked before any view or frame is drawn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData
    from neurospatial.animation.scene import Animation


_OVERLAY_NAMES = {
    "positions": "PositionOverlay",
    "bodypart_sets": "BodypartOverlay",
    "head_directions": "HeadDirectionOverlay",
    "events": "EventOverlay",
    "videos": "VideoOverlay",
    "timeseries": "TimeSeriesOverlay",
    "object_vectors": "ObjectVectorOverlay",
    "spike_rasters": "SpikeRasterOverlay",
    "epochs": "EpochOverlay",
}


@dataclass(frozen=True)
class _Drawn:
    """What a view leaves out of a scene of one dimension, why, and the fix."""

    left_out: frozenset[str] = frozenset()
    why: str = ""
    fix: str = ""


_EVERYTHING = _Drawn()
# Every view, by the environment dimensions it draws. The widget's scene
# renderer also renders frames and exports, so a scene must not hold
# overlays it leaves out.
_VIEWS: Mapping[str, Mapping[int, _Drawn]] = MappingProxyType(
    {
        "widget": {
            1: _Drawn(
                frozenset({"head_directions", "videos"}),
                why="its spatial plot has position on x and field amplitude on y, "
                "so angular arrows and image coordinates have no defined spatial "
                "plane",
                fix="Use a 2D environment for spatial headings/video, or "
                "TimeSeriesOverlay to show heading angles beside the 1D field.",
            ),
            2: _EVERYTHING,
            3: _EVERYTHING,
        },
        "napari": {2: _EVERYTHING, 3: _EVERYTHING},
        "fastplotlib": {
            1: _Drawn(
                frozenset(
                    {"bodypart_sets", "head_directions", "object_vectors", "videos"}
                ),
                why="it plots position along the track against time, which has "
                "no place for body parts, arrows or video",
                fix="Use a 2D environment to draw them.",
            ),
            2: _EVERYTHING,
            3: _EVERYTHING,
        },
    }
)


def _left_out(view: str, data: OverlayData, n_dims: int) -> list[str]:
    left_out = _VIEWS[view][n_dims].left_out
    return [name for name in _OVERLAY_NAMES if getattr(data, name) and name in left_out]


def require_scene_overlays(data: OverlayData, n_dims: int) -> None:
    """Refuse overlays that the scene's own renderer cannot draw.

    ``render_frame``, ``export`` and the widget view share that renderer.
    """
    missing = _left_out("widget", data, n_dims)
    if missing:
        drawn = _VIEWS["widget"][n_dims]
        names = ", ".join(_OVERLAY_NAMES[name] for name in missing)
        raise ValueError(
            f"A {n_dims}D scene cannot draw {names}: {drawn.why}. {drawn.fix}"
        )


def _views_drawing(n_dims: int, overlays: frozenset[str] = frozenset()) -> list[str]:
    """The views that draw scenes of ``n_dims`` with all of ``overlays``."""
    return [
        view
        for view, dims in _VIEWS.items()
        if n_dims in dims and not overlays & dims[n_dims].left_out
    ]


def _require_view_draws(backend: str, scene: Animation) -> None:
    n_dims = scene.env.n_dims
    dims = _VIEWS[backend]
    if n_dims not in dims:
        views = " or ".join(
            f"Animation.show(backend={view!r})" for view in _views_drawing(n_dims)
        )
        raise ValueError(
            f"The {backend} view cannot draw {n_dims}D scenes. "
            f"Use {views} for this environment."
        )
    missing = _left_out(backend, scene._overlay_data, n_dims)
    if missing:
        drawn = dims[n_dims]
        names = ", ".join(_OVERLAY_NAMES[name] for name in missing)
        drawing = _views_drawing(n_dims, frozenset(missing))
        fix = (
            f"Use Animation.show(backend={drawing[0]!r}), which draws them, or "
            "Animation.export()."
            if drawing
            else drawn.fix
        )
        raise ValueError(
            f"The {backend} view of a {n_dims}D scene cannot draw {names}: "
            f"{drawn.why}. {fix}"
        )


def checked_speed(speed: Any) -> float:
    """``speed``, a positive finite playback multiplier, as a float."""
    try:
        value = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "speed must be a positive finite playback multiplier."
        ) from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("speed must be a positive finite playback multiplier.")
    return value


def checked_max_playback_fps(max_playback_fps: Any) -> int:
    """``max_playback_fps``, a positive integer, as an int."""
    if (
        isinstance(max_playback_fps, bool)
        or not isinstance(max_playback_fps, (int, np.integer))
        or max_playback_fps <= 0
    ):
        raise ValueError("max_playback_fps must be a positive integer.")
    return int(max_playback_fps)


def _checked_fps(fps: Any) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fps must be a positive finite number of steps a second."
        ) from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("fps must be a positive finite number of steps a second.")
    return value


_PLAYBACK_CHECKS = {
    "speed": checked_speed,
    "max_playback_fps": checked_max_playback_fps,
    "fps": _checked_fps,
}
_SCENE_OPTIONS = {
    "widget": frozenset({"dpi", "figsize", "speed", "max_playback_fps"}),
    "napari": frozenset(
        {"viewer", "show", "ndisplay", "title", "fps", "speed", "layout"}
    ),
    "fastplotlib": frozenset(
        {
            "speed",
            "max_playback_fps",
            "frame_labels",
            "layout",
            "layer_names",
            "field_rendering",
            "volume_threshold",
            "slice_plane",
        }
    ),
}


def require_scene_view(scene: Animation, backend: str, options: dict[str, Any]) -> None:
    """Check the entire request before importing or allocating a native viewer."""
    if backend not in _SCENE_OPTIONS:
        raise ValueError("Use backend='widget', 'napari', or 'fastplotlib'.")
    unknown = set(options) - _SCENE_OPTIONS[backend]
    if unknown:
        raise TypeError(
            f"Unknown {backend} view options: {', '.join(sorted(unknown))}."
        )
    _require_view_draws(backend, scene)
    for name, check in _PLAYBACK_CHECKS.items():
        if name in options:
            check(options[name])
    layout = options.get("layout")
    if layout is not None and layout not in {"horizontal", "vertical", "grid"}:
        raise ValueError("layout must be 'horizontal', 'vertical' or 'grid'.")
    if backend == "napari" and options.get("ndisplay", scene.env.n_dims) not in (2, 3):
        raise ValueError("ndisplay must be 2 or 3 for a napari scene view.")
    if backend == "fastplotlib" and scene.scale_bar:
        raise ValueError(
            "The fastplotlib scene backend cannot draw scale_bar. "
            "Use backend='widget' or backend='napari', or omit scale_bar."
        )
