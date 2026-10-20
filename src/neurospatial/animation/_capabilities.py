"""One capability table for validation of the existing backend adapters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData
    from neurospatial.animation.scene import Animation


@dataclass(frozen=True)
class _BackendCapabilities:
    overlays: frozenset[str]
    multi_field: bool
    dimensions: frozenset[int]
    interactive: bool


_SPATIAL = frozenset({"positions", "bodypart_sets", "head_directions", "events"})
_ALL = _SPATIAL | {
    "videos",
    "timeseries",
    "object_vectors",
    "spike_rasters",
    "epochs",
}
_CAPABILITIES = MappingProxyType(
    {
        "napari": _BackendCapabilities(
            _SPATIAL | {"videos", "timeseries"}, True, frozenset({2, 3}), True
        ),
        "fastplotlib": _BackendCapabilities(_ALL, True, frozenset({1, 2, 3}), True),
        "napari_scene": _BackendCapabilities(_ALL, True, frozenset({2, 3}), True),
        "video": _BackendCapabilities(
            _SPATIAL | {"videos", "timeseries"}, False, frozenset({1, 2}), False
        ),
        "widget": _BackendCapabilities(
            _SPATIAL | {"videos", "timeseries"}, False, frozenset({1, 2}), True
        ),
        "html": _BackendCapabilities(_SPATIAL, False, frozenset({1, 2}), False),
        # Animation's notebook and export paths use the same complete renderer.
        "scene": _BackendCapabilities(_ALL, True, frozenset({1, 2, 3}), True),
    }
)
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


def unsupported_overlays(
    backend: str, data: OverlayData, *, n_dims: int | None = None
) -> list[str]:
    """Internal data collection names the backend cannot draw."""
    supported = _CAPABILITIES[backend].overlays
    if backend == "scene" and n_dims == 1:
        supported -= {"head_directions", "videos"}
    if backend == "fastplotlib" and n_dims == 1:
        supported -= {"bodypart_sets", "head_directions", "object_vectors", "videos"}
    return [
        name for name in _OVERLAY_NAMES if getattr(data, name) and name not in supported
    ]


def require_overlays(
    backend: str, data: OverlayData, *, n_dims: int | None = None
) -> None:
    """Refuse a scene view that would omit requested data."""
    missing = unsupported_overlays(backend, data, n_dims=n_dims)
    if missing and backend == "scene" and n_dims == 1:
        names = ", ".join(_OVERLAY_NAMES[name] for name in missing)
        raise ValueError(
            f"A 1D scene cannot draw {names}: its spatial plot has position "
            "on x and field amplitude on y, so angular arrows and image "
            "coordinates have no defined spatial plane. Use a 2D environment "
            "for spatial headings/video, or TimeSeriesOverlay to show heading "
            "angles beside the 1D field."
        )
    if missing:
        names = ", ".join(_OVERLAY_NAMES[name] for name in missing)
        raise ValueError(
            f"The {backend} backend cannot draw {names}. "
            "Use Animation.show(backend='widget') for a complete notebook view "
            "or Animation.export() to render the scene."
        )


def require_scene_rendering(scene: Animation) -> None:
    """Validate content shared by complete-scene rendering and its widgets."""
    require_overlays("scene", scene._overlay_data, n_dims=scene.env.n_dims)
    if scene.env.n_dims == 1 and scene.show_regions:
        raise ValueError(
            "1D scene region display is unavailable. Use a 2D environment "
            "to display regions, or omit show_regions."
        )


def warn_unsupported_overlays(backend: str, data: OverlayData) -> OverlayData:
    """Warn and remove unsupported collections at the legacy boundary."""
    missing = unsupported_overlays(backend, data)
    if not missing:
        return data
    names = " or ".join(_OVERLAY_NAMES[name] for name in missing)
    warnings.warn(
        f"The {backend} backend does not draw {names}, so it is skipped; "
        "the other overlays are drawn. Use backend='fastplotlib' to show it, "
        "or Animation.show(backend='widget') / Animation.export() for a "
        "complete scene.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )
    if "videos" in missing:
        for video in data.videos:
            close = getattr(video.reader, "close", None)
            if video._owns_reader and close is not None:
                close()
    changes: dict[str, Any] = {name: [] for name in missing}
    return replace(data, **changes)


_SCENE_OPTIONS = {
    "widget": frozenset({"dpi", "figsize", "speed", "max_playback_fps"}),
    "napari": frozenset(
        {"viewer", "show", "ndisplay", "title", "fps", "speed", "layout"}
    ),
    "fastplotlib": frozenset(
        {
            "speed",
            "max_playback_fps",
            "cmap",
            "vmin",
            "vmax",
            "frame_labels",
            "title",
            "show_regions",
            "region_alpha",
            "window_seconds",
            "layout",
            "layer_names",
            "colorbar_label",
            "track",
            "dry_run",
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
    capability_name = {"widget": "scene", "napari": "napari_scene"}.get(
        backend, backend
    )
    capabilities = _CAPABILITIES[capability_name]
    if scene.env.n_dims not in capabilities.dimensions:
        raise ValueError(
            f"The {backend} backend cannot draw {scene.env.n_dims}D scenes. "
            "Use Animation.show(backend='widget') for this environment."
        )
    if len(scene.fields) > 1 and not capabilities.multi_field:
        raise ValueError(f"The {backend} backend cannot draw multiple fields.")
    if capability_name == "scene":
        require_scene_rendering(scene)
    else:
        require_overlays(capability_name, scene._overlay_data, n_dims=scene.env.n_dims)
    layout = options.get("layout")
    if layout is not None and layout not in {"horizontal", "vertical", "grid"}:
        raise ValueError("layout must be 'horizontal', 'vertical' or 'grid'.")
    show_regions = options.get("show_regions", scene.show_regions)
    region_names: list[str] = []
    if show_regions:
        from neurospatial.animation._region_names import shown_region_names

        region_names = shown_region_names(scene.env, show_regions)
    if backend == "napari" and options.get("ndisplay", scene.env.n_dims) not in (2, 3):
        raise ValueError("ndisplay must be 2 or 3 for a napari scene view.")
    if backend == "napari" and layout is not None and (scene.overlays or show_regions):
        raise ValueError(
            "Napari grid layout separates overlay and region layers from the "
            "spatial fields. Omit layout to keep overlays and regions registered."
        )
    if backend == "fastplotlib":
        from neurospatial.environment._fastplotlib_volume import resolve_field_rendering

        resolve_field_rendering(
            scene.env,
            options.get("field_rendering", "auto"),
            volume_threshold=options.get("volume_threshold"),
            slice_plane=options.get("slice_plane"),
        )
        track = options.get("track", scene.track)
        if track is not None:
            from neurospatial.animation._tracks import _checked_track

            _checked_track(scene.env, track)
        if options.get("dry_run", False):
            raise ValueError(
                "Animation.show() creates a view; dry_run is not supported."
            )
        if scene.scale_bar:
            raise ValueError(
                "The fastplotlib scene backend cannot draw scale_bar. "
                "Use backend='widget' or backend='napari', or omit scale_bar."
            )
        if scene.env.n_dims == 1 and show_regions:
            raise ValueError(
                "The fastplotlib backend cannot draw regions on a 1D time panel. "
                "Use a 2D environment for region display, or omit show_regions."
            )
    if backend == "napari" and show_regions:
        for name in region_names:
            region = scene.env.regions[name]
            if region.kind not in {"point", "polygon"}:
                raise ValueError(
                    f"Napari cannot draw region {name!r} of kind {region.kind!r}. "
                    "Omit show_regions or use backend='widget'."
                )
            if region.kind == "polygon" and region.data.interiors:
                raise ValueError(
                    "Napari region polygons with holes require a raster mask; "
                    "omit show_regions or use backend='widget'."
                )
