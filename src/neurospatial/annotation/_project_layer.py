"""Private conversion between portable drafts and editable napari layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from neurospatial.annotation._draft import guard_identity_defaults
from neurospatial.annotation._helpers import rebuild_features, sync_colors_from_features
from neurospatial.annotation._types import RegionType
from neurospatial.annotation.project import AnnotationShape
from neurospatial.regions.core import _thaw_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence


def capture_shapes(
    shapes_layer: Any, previous: Sequence[AnnotationShape] = ()
) -> tuple[AnnotationShape, ...]:
    """Capture pixel ``(x, y)`` shapes without changing the layer.

    The caller must first flush pending widget edits; every processed shape
    already has its ring identity. Only domain features and JSON polygon
    metadata enter snapshots; runtime deletion markers and widget callbacks
    remain local to the layer. A shape equal to one in ``previous`` is that
    same object, so unchanged shapes are shared between history steps.
    """
    features = shapes_layer.features
    if len(features) != len(shapes_layer.data):
        raise ValueError("Cannot capture project: shape features are not synchronized.")
    if not features["_polygon_id"].astype(bool).all():
        raise ValueError(
            "Cannot capture project: a shape has not been processed yet. "
            "Flush pending annotation edits before saving."
        )
    metadata = shapes_layer.metadata.get("_polygon_metadata", {})
    earlier = {shape.identifier: shape for shape in previous}
    captured = (
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
    return tuple(
        earlier[shape.identifier] if earlier.get(shape.identifier) == shape else shape
        for shape in captured
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

    All layer events are blocked during replacement, so the layer is redrawn
    explicitly afterwards. Existing widgets should explicitly synchronize from
    the layer after this function returns.
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
    # The canvas redraws on the blocked data event; without this, it keeps
    # showing the replaced shapes.
    shapes_layer.refresh()
