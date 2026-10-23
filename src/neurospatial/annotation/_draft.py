"""Private editable polygon rings and their stable parent associations."""

from collections.abc import Mapping, Sequence
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


def add_polygons(
    layer: Any,
    polygons: Sequence[tuple[Polygon, str, RegionType, Mapping[str, Any] | None]],
    calibration: VideoCalibration | None,
) -> None:
    """Append polygons as independently editable rings with stable links.

    Parameters
    ----------
    layer : napari.layers.Shapes
        The annotation shapes layer.
    polygons : sequence of (Polygon, str, RegionType, mapping or None)
        Each polygon with its name, role and region metadata. Coordinates are
        centimeters with calibration and video pixels without it.
    calibration : VideoCalibration or None
        Transform from centimeters to video pixels.

    """
    import pandas as pd

    from neurospatial.annotation._helpers import (
        rebuild_features,
        sync_colors_from_features,
    )
    from neurospatial.regions.core import _thaw_metadata

    # Invalid geometry still loads, so it can be repaired; saving validates it.
    for polygon, name, _, _ in polygons:
        if polygon.geom_type != "Polygon":
            raise TypeError(
                f"Initial polygon '{name}' must be a shapely Polygon, "
                f"not {polygon.geom_type}."
            )
    if not polygons:
        return
    vertices: list[NDArray[np.float64]] = []
    roles: list[RegionType] = []
    names: list[str] = []
    ids: list[str] = []
    parents: list[str] = []
    polygon_metadata = layer.metadata.setdefault("_polygon_metadata", {})
    for polygon, name, role, metadata in polygons:
        identifier = uuid4().hex
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coords = np.asarray(ring.coords)
            if calibration is not None:
                coords = calibration.transform_cm_to_px(coords)
            vertices.append(coords[:, ::-1])
        roles += [role] * len(rings)
        names += [name, *[f"{name} interior {i}" for i in range(1, len(rings))]]
        ids += [identifier, *[uuid4().hex for _ in rings[1:]]]
        parents += ["", *[identifier] * (len(rings) - 1)]
        polygon_metadata[identifier] = _thaw_metadata(metadata or {})
    if not layer.metadata.get("_polygon_defaults_guard"):
        # Mutate defaults only: current_properties' setter can overwrite the
        # selected shapes themselves in select mode.
        layer.events.current_properties.connect(
            lambda event: reset_identity_defaults(layer)
        )
        layer.events.highlight.connect(lambda event: reset_identity_defaults(layer))
        # Every features assignment, paste and removal emits this event.
        layer.events.features.connect(lambda event: _on_features_changed(layer))
        layer.metadata["_polygon_defaults_guard"] = True
    existing = layer.features.copy()
    layer.add(vertices, shape_type="polygon")
    rows = rebuild_features(roles, names)
    rows["_polygon_id"] = ids
    rows["_parent_id"] = parents
    combined = pd.concat([existing, rows], ignore_index=True)
    for column in _IDENTITY_COLUMNS:
        combined[column] = combined[column].fillna("")
    layer.features = combined
    sync_colors_from_features(layer)


def reset_identity_defaults(layer: Any) -> None:
    """Keep new drawings from inheriting a selected ring's identity."""
    for column in _IDENTITY_COLUMNS:
        if column in layer.feature_defaults:
            layer.feature_defaults[column] = ""


def _on_features_changed(layer: Any) -> None:
    """Identify pasted copies, then clear identity defaults.

    Napari rebuilds ``feature_defaults`` from the last row whenever
    ``features`` is assigned, which would give the next drawing that row's
    identity. Paste emits only this event, before any other edit.
    """
    _identify_copies(layer)
    reset_identity_defaults(layer)


def _identify_copies(layer: Any) -> None:
    """Give pasted copies their own identities, keeping their ring links.

    A row repeating an earlier identity is a copy. Each paste emits its own
    features event, so the copies found here came from one paste, in the
    selection's order rather than row order. A copied interior ring follows
    the copy of its exterior pasted with it; one copied without its exterior
    becomes a standalone polygon.
    """
    features = layer.features
    if "_polygon_id" not in features:
        return
    ids = [str(identifier) for identifier in features["_polygon_id"]]
    parents = [str(parent) for parent in features["_parent_id"]]
    seen: set[str] = set()
    copies: dict[str, str] = {}
    copied_rows: list[int] = []
    polygon_metadata = layer.metadata.setdefault("_polygon_metadata", {})
    for index, identifier in enumerate(ids):
        if not identifier or identifier not in seen:
            seen.add(identifier)
            continue
        copied_rows.append(index)
        ids[index] = copies[identifier] = uuid4().hex
        seen.add(ids[index])
        if identifier in polygon_metadata:
            polygon_metadata[ids[index]] = deepcopy(polygon_metadata[identifier])
    # Link rings only once every copy has its identity: an interior ring can
    # be pasted before its exterior.
    for index in copied_rows:
        parents[index] = copies.get(parents[index], "")
    copied = bool(copied_rows)
    if copied:
        features = features.copy()
        features["_polygon_id"] = ids
        features["_parent_id"] = parents
        layer.features = features


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
    interiors: dict[str, list[NDArray[np.float64]]] = {}
    for vertices, parent in zip(data, parents, strict=True):
        if parent:
            interiors.setdefault(parent, []).append(vertices)
    polygon_metadata = layer.metadata.get("_polygon_metadata", {})
    output = []
    for vertices, name, role, identifier, parent in zip(
        data, names, roles, ids, parents, strict=True
    ):
        if parent:
            continue
        output.append(
            PolygonDraft(
                vertices,
                name,
                role,
                interiors.get(identifier, []) if identifier else [],
                deepcopy(polygon_metadata.get(identifier, {})),
            )
        )
    return output
