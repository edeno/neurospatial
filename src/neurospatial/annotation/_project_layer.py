"""Private conversion between portable drafts and editable napari layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

import numpy as np

from neurospatial.annotation._draft import guard_identity_defaults
from neurospatial.annotation._helpers import rebuild_features, sync_colors_from_features
from neurospatial.annotation._types import MultipleBoundaryStrategy, RegionType
from neurospatial.annotation.project import AnnotationProject, AnnotationShape
from neurospatial.regions.core import _thaw_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from neurospatial.ops.transforms import VideoCalibration


def capture_shapes(shapes_layer: Any) -> tuple[AnnotationShape, ...]:
    """Capture pixel ``(x, y)`` shapes, assigning missing stable identities.

    The caller must first flush pending widget edits. Only domain features and
    JSON polygon metadata enter snapshots; runtime deletion markers and widget
    callbacks remain local to the layer.
    """
    features = shapes_layer.features.copy()
    count = len(shapes_layer.data)
    if len(features) != count:
        raise ValueError("Cannot capture project: shape features are not synchronized.")
    changed = False
    for column in ("_polygon_id", "_parent_id"):
        if column not in features:
            features[column] = ""
            changed = True
    for index in range(count):
        identifier = features.iloc[index]["_polygon_id"]
        if (
            identifier == ""
            or identifier is None
            or (isinstance(identifier, float) and np.isnan(identifier))
        ):
            features.iloc[index, features.columns.get_loc("_polygon_id")] = uuid4().hex
            changed = True
        parent = features.iloc[index]["_parent_id"]
        if parent is None or (isinstance(parent, float) and np.isnan(parent)):
            features.iloc[index, features.columns.get_loc("_parent_id")] = ""
            changed = True
    with shapes_layer.events.blocker_all():
        if changed:
            shapes_layer.features = features
        guard_identity_defaults(shapes_layer)
    metadata = shapes_layer.metadata.get("_polygon_metadata", {})
    return tuple(
        AnnotationShape(
            np.asarray(vertices, dtype=float)[:, ::-1],
            shape_type=str(shape_type),
            name=features.iloc[i]["name"],
            role=cast("RegionType", str(features.iloc[i]["role"])),
            identifier=features.iloc[i]["_polygon_id"],
            parent_id=features.iloc[i]["_parent_id"],
            metadata=metadata.get(features.iloc[i]["_polygon_id"], {}),
        )
        for i, (vertices, shape_type) in enumerate(
            zip(shapes_layer.data, shapes_layer.shape_type, strict=True)
        )
    )


def capture_project(
    frame: NDArray[np.uint8],
    shapes_layer: Any,
    *,
    video_path: str | Path | None,
    frame_index: int,
    calibration: VideoCalibration | None,
    mode: Literal["environment", "regions", "both"],
    bin_size: float | None,
    simplify_tolerance: float | None,
    multiple_boundaries: MultipleBoundaryStrategy,
) -> AnnotationProject:
    """Capture a flushed layer with its reference image and workflow settings."""
    return AnnotationProject(
        frame,
        shapes=capture_shapes(shapes_layer),
        video_path=video_path,
        frame_index=frame_index,
        calibration=calibration,
        mode=mode,
        bin_size=bin_size,
        simplify_tolerance=simplify_tolerance,
        multiple_boundaries=multiple_boundaries,
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


def restore_project_shapes(project: AnnotationProject, shapes_layer: Any) -> None:
    """Restore a saved layer before creating its annotation widget."""
    restore_shapes(project.shapes, shapes_layer)
