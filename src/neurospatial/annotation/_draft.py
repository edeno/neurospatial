"""Private editable polygon rings and their stable parent associations."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial.annotation._types import RegionType
from neurospatial.ops.transforms import VideoCalibration

_IDENTITY_COLUMNS = ("_polygon_id", "_parent_id")


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

    # Invalid geometry still loads, so it can be repaired; saving validates it.
    if polygon.geom_type != "Polygon":
        raise TypeError(
            f"Initial polygon '{name}' must be a shapely Polygon, "
            f"not {polygon.geom_type}."
        )
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
    for column in _IDENTITY_COLUMNS:
        combined[column] = combined[column].fillna("")
    layer.features = combined
    layer.metadata.setdefault("_polygon_metadata", {})[identifier] = _thaw_metadata(
        metadata or {}
    )
    reset_identity_defaults(layer)
    if not layer.metadata.get("_polygon_defaults_guard"):
        # Mutate defaults only: current_properties' setter can overwrite the
        # selected shapes themselves in select mode.
        layer.events.current_properties.connect(
            lambda event: reset_identity_defaults(layer)
        )
        layer.events.highlight.connect(lambda event: reset_identity_defaults(layer))
        # Napari's paste emits only a features event, before any other edit.
        layer.events.features.connect(lambda event: _identify_copies(layer))
        layer.metadata["_polygon_defaults_guard"] = True
    sync_colors_from_features(layer)


def reset_identity_defaults(layer: Any) -> None:
    """Keep new drawings from inheriting a selected ring's identity.

    Napari rebuilds ``feature_defaults`` from the last row whenever
    ``features`` is assigned, so call this after every such assignment.
    """
    for column in _IDENTITY_COLUMNS:
        if column in layer.feature_defaults:
            layer.feature_defaults[column] = ""


def _identify_copies(layer: Any) -> None:
    """Give pasted copies their own identities, keeping their ring links.

    A row repeating an earlier identity is a copy. A copied interior ring
    follows the most recent copy of its exterior; one copied without its
    exterior becomes a standalone polygon.
    """
    features = layer.features
    if "_polygon_id" not in features:
        return
    ids = [str(identifier) for identifier in features["_polygon_id"]]
    parents = [str(parent) for parent in features["_parent_id"]]
    seen: set[str] = set()
    copies: dict[str, str] = {}
    copied = False
    polygon_metadata = layer.metadata.setdefault("_polygon_metadata", {})
    for index, identifier in enumerate(ids):
        if not identifier or identifier not in seen:
            seen.add(identifier)
            continue
        copied = True
        ids[index] = copies[identifier] = uuid4().hex
        seen.add(ids[index])
        parents[index] = copies.get(parents[index], "")
        if identifier in polygon_metadata:
            polygon_metadata[ids[index]] = deepcopy(polygon_metadata[identifier])
    if copied:
        features = features.copy()
        features["_polygon_id"] = ids
        features["_parent_id"] = parents
        layer.features = features
        reset_identity_defaults(layer)


def annotation_roles(features: Any) -> list[RegionType]:
    """Return the role each row adds to the annotation counts.

    An environment's interior ring is one of its holes. A region's interior
    ring is part of that region and adds nothing.
    """
    roles = [cast("RegionType", str(role)) for role in features.get("role", [])]
    if "_parent_id" not in features:
        return roles
    return [
        role if not parent else "hole"
        for role, parent in zip(roles, features["_parent_id"], strict=True)
        if not parent or role == "environment"
    ]


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
