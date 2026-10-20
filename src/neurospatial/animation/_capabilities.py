"""One capability table for validation of the existing backend adapters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData


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
        "video": _BackendCapabilities(
            _SPATIAL | {"videos", "timeseries"}, False, frozenset({1, 2}), False
        ),
        "widget": _BackendCapabilities(
            _SPATIAL | {"videos", "timeseries"}, False, frozenset({1, 2}), True
        ),
        "html": _BackendCapabilities(_SPATIAL, False, frozenset({1, 2}), False),
        # Animation's notebook and export paths use the same complete renderer.
        "scene": _BackendCapabilities(_ALL, True, frozenset({1, 2}), True),
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


def unsupported_overlays(backend: str, data: OverlayData) -> list[str]:
    """Internal data collection names the backend cannot draw."""
    supported = _CAPABILITIES[backend].overlays
    return [
        name for name in _OVERLAY_NAMES if getattr(data, name) and name not in supported
    ]


def require_overlays(backend: str, data: OverlayData) -> None:
    """Refuse a scene view that would omit requested data."""
    missing = unsupported_overlays(backend, data)
    if missing:
        names = ", ".join(_OVERLAY_NAMES[name] for name in missing)
        raise ValueError(
            f"The {backend} backend cannot draw {names}. "
            "Use Animation.show(backend='widget') for a complete notebook view "
            "or backend='fastplotlib' for a GPU view."
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
