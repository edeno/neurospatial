"""Private conversion between portable drafts and editable napari layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np

from neurospatial.annotation._draft import _IDENTITY_COLUMNS, guard_identity_defaults
from neurospatial.annotation._helpers import rebuild_features, sync_colors_from_features
from neurospatial.annotation._types import RegionType
from neurospatial.annotation.project import AnnotationShape
from neurospatial.regions.core import _thaw_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence


def capture_shapes(shapes_layer: Any) -> tuple[AnnotationShape, ...]:
    """Capture pixel ``(x, y)`` shapes, assigning missing stable identities.

    The caller must first flush pending widget edits. Only domain features and
    JSON polygon metadata enter snapshots; runtime deletion markers and widget
    callbacks remain local to the layer.
    """
    features = shapes_layer.features.copy()
    if len(features) != len(shapes_layer.data):
        raise ValueError("Cannot capture project: shape features are not synchronized.")
    added = [column for column in _IDENTITY_COLUMNS if column not in features]
    for column in added:
        features[column] = ""
    blank_ids = features["_polygon_id"].isna() | (features["_polygon_id"] == "")
    blank_parents = features["_parent_id"].isna()
    changed = bool(added) or bool(blank_ids.any()) or bool(blank_parents.any())
    if changed:
        features.loc[blank_ids, "_polygon_id"] = [
            uuid4().hex for _ in range(int(blank_ids.sum()))
        ]
        features["_parent_id"] = features["_parent_id"].fillna("")
    with shapes_layer.events.blocker_all():
        if changed:
            shapes_layer.features = features
        guard_identity_defaults(shapes_layer)
    metadata = shapes_layer.metadata.get("_polygon_metadata", {})
    return tuple(
        AnnotationShape(
            np.asarray(vertices, dtype=float)[:, ::-1],
            shape_type=str(shape_type),
            name=name,
            role=cast("RegionType", str(role)),
            identifier=identifier,
            parent_id=parent,
            metadata=metadata.get(identifier, {}),
        )
        for vertices, shape_type, name, role, identifier, parent in zip(
            shapes_layer.data,
            shapes_layer.shape_type,
            features["name"].tolist(),
            features["role"].tolist(),
            features["_polygon_id"].tolist(),
            features["_parent_id"].tolist(),
            strict=True,
        )
    )


def _validate_shapes_for_display(shapes: Sequence[AnnotationShape]) -> None:
    """Preflight native display cardinalities without importing napari.

    Draft storage deliberately permits fewer vertices. The native polygon/path
    models display at least two vertices, lines exactly two, and rectangles and
    ellipses two defining vertices or four corners. Do not synthesize geometry
    to make a saved shape displayable.
    """
    for index, shape in enumerate(shapes):
        count = len(shape.vertices)
        if shape.shape_type in {"polygon", "path"}:
            valid = count >= 2
            required = "at least 2 vertices"
        elif shape.shape_type == "line":
            valid = count == 2
            required = "exactly 2 vertices"
        else:
            valid = count in {2, 4}
            required = "2 defining vertices or 4 corners"
        if not valid:
            label = shape.name or shape.identifier
            raise ValueError(
                f"Cannot display annotation shape {index + 1} ({label!r}): "
                f"{shape.shape_type} has {count} vertices; napari requires {required}. "
                "Replace this shape with an AnnotationShape containing the required "
                "vertices before resuming. The saved geometry has not been changed."
            )


def restore_shapes(shapes: Sequence[AnnotationShape], shapes_layer: Any) -> None:
    """Replace geometry and domain features while keeping runtime callbacks.

    All layer events are blocked during replacement. Existing widgets should
    explicitly synchronize from the layer after this function returns.
    """
    _validate_shapes_for_display(shapes)
    features = rebuild_features(
        [shape.role for shape in shapes], [shape.name for shape in shapes]
    )
    features["_polygon_id"] = [shape.identifier for shape in shapes]
    features["_parent_id"] = [shape.parent_id for shape in shapes]
    with shapes_layer.events.blocker_all():
        shapes_layer.selected_data = set()
        shapes_layer.data = [
            (shape.vertices[:, ::-1].copy(), shape.shape_type) for shape in shapes
        ]
        shapes_layer.features = features
        shapes_layer.metadata["_polygon_metadata"] = {
            shape.identifier: _thaw_metadata(shape.metadata) for shape in shapes
        }
        guard_identity_defaults(shapes_layer)
        sync_colors_from_features(shapes_layer)
