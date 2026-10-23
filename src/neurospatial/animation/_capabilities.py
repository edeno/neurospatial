"""What each scene view and the scene renderer can draw, checked before drawing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData
    from neurospatial.animation.scene import Animation


@dataclass(frozen=True)
class _BackendCapabilities:
    overlays: frozenset[str]
    multi_field: bool
    dimensions: frozenset[int]


_ALL = frozenset(
    {
        "positions",
        "bodypart_sets",
        "head_directions",
        "events",
        "videos",
        "timeseries",
        "object_vectors",
        "spike_rasters",
        "epochs",
    }
)
_CAPABILITIES = MappingProxyType(
    {
        "fastplotlib": _BackendCapabilities(_ALL, True, frozenset({1, 2, 3})),
        "napari_scene": _BackendCapabilities(_ALL, True, frozenset({2, 3})),
        # Animation's notebook and export paths use the same complete renderer.
        "scene": _BackendCapabilities(_ALL, True, frozenset({1, 2, 3})),
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
    """Validate content shared by complete-scene rendering and its widgets.

    ``Animation`` already refused regions it cannot draw in 1-D.
    """
    require_overlays("scene", scene._overlay_data, n_dims=scene.env.n_dims)


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
    region_names: list[str] = []
    if scene.show_regions:
        from neurospatial.animation._region_names import shown_region_names

        region_names = shown_region_names(scene.env, scene.show_regions)
    if backend == "napari" and options.get("ndisplay", scene.env.n_dims) not in (2, 3):
        raise ValueError("ndisplay must be 2 or 3 for a napari scene view.")
    if backend == "fastplotlib" and scene.scale_bar:
        raise ValueError(
            "The fastplotlib scene backend cannot draw scale_bar. "
            "Use backend='widget' or backend='napari', or omit scale_bar."
        )
    if backend == "napari" and scene.show_regions:
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
