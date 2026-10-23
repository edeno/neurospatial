"""Private editable polygon rings and their stable parent associations."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial.annotation._types import RegionType
from neurospatial.ops.transforms import VideoCalibration


@dataclass
class PolygonDraft:
    """An exterior and its explicitly associated interior rings."""

    vertices: NDArray[np.float64]
    name: str
    role: RegionType
    interiors: list[NDArray[np.float64]]
    metadata: dict[str, Any]


def add_polygon(
    layer: Any,
    polygon: Polygon,
    name: str,
    role: RegionType,
    calibration: VideoCalibration | None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append a polygon as independently editable rings with stable links."""
    import pandas as pd

    from neurospatial.annotation._helpers import (
        rebuild_features,
        sync_colors_from_features,
    )
    from neurospatial.regions.core import _thaw_metadata

    if polygon.geom_type != "Polygon" or not polygon.is_valid:
        raise ValueError(f"Initial polygon '{name}' must be a valid Polygon.")
    existing = layer.features.copy()
    identifier = uuid4().hex
    rings = [polygon.exterior, *polygon.interiors]
    vertices = []
    for ring in rings:
        coords = np.asarray(ring.coords)
        if calibration is not None:
            coords = calibration.transform_cm_to_px(coords)
        vertices.append(coords[:, ::-1])
    layer.add(vertices, shape_type="polygon")
    rows = rebuild_features(
        [role] * len(rings),
        [name, *[f"{name} interior {i + 1}" for i in range(len(rings) - 1)]],
    )
    rows["_polygon_id"] = [identifier, *[uuid4().hex for _ in rings[1:]]]
    rows["_parent_id"] = ["", *[identifier for _ in rings[1:]]]
    combined = pd.concat([existing, rows], ignore_index=True)
    for column in ("_polygon_id", "_parent_id"):
        combined[column] = combined[column].fillna("")
    layer.features = combined
    layer.metadata.setdefault("_polygon_metadata", {})[identifier] = _thaw_metadata(
        metadata or {}
    )
    guard_identity_defaults(layer)
    sync_colors_from_features(layer)


def guard_identity_defaults(layer: Any) -> None:
    """Keep new drawings from inheriting the selected polygon's identity."""
    for column in ("_polygon_id", "_parent_id"):
        layer.feature_defaults[column] = ""
    if layer.metadata.get("_polygon_defaults_guard"):
        return

    def clear_identity_defaults(event: Any) -> None:
        # Mutate defaults only: current_properties' setter can overwrite the
        # selected shapes themselves in select mode.
        for column in ("_polygon_id", "_parent_id"):
            if column in layer.feature_defaults:
                layer.feature_defaults[column] = ""

    layer.events.current_properties.connect(clear_identity_defaults)
    layer.events.highlight.connect(clear_identity_defaults)
    layer.metadata["_polygon_defaults_guard"] = True


def polygon_drafts(
    layer: Any,
    data: list[NDArray[np.float64]],
    names: list[str],
    roles: list[RegionType],
) -> list[PolygonDraft]:
    """Collect rings by stable ID, excluding orphan rings after parent deletion."""
    features = layer.features
    ids = list(features.get("_polygon_id", [""] * len(data)))
    parents = list(features.get("_parent_id", [""] * len(data)))
    identities = [identifier for identifier in ids if identifier]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Copied polygon rings have duplicate identities. Delete the copied shapes and draw new polygons before saving."
        )
    output = []
    for index, (vertices, name, role) in enumerate(
        zip(data, names, roles, strict=True)
    ):
        if parents[index]:
            continue
        output.append(
            PolygonDraft(
                vertices,
                name,
                role,
                [
                    data[j]
                    for j, parent in enumerate(parents)
                    if ids[index] and parent == ids[index]
                ],
                deepcopy(
                    layer.metadata.get("_polygon_metadata", {}).get(ids[index], {})
                ),
            )
        )
    return output
