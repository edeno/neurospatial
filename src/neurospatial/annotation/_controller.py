"""Controller for applying annotation state to napari shapes layer.

This module provides a thin adapter between pure annotation state and
the napari Shapes layer, keeping side-effects isolated.

The controller handles:
- Shape deletion with feature/state synchronization
- Shape renaming with uniqueness enforcement
- Mode synchronization (role colors and defaults)

The widget uses the controller for layer mutations while handling
UI-specific concerns (status messages, list updates, dialogs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast
from uuid import uuid4

from neurospatial.annotation._draft import annotation_roles
from neurospatial.annotation._helpers import (
    REGION_TYPE_COLORS,
    rebuild_features,
    select_shapes,
    sync_colors_from_features,
)
from neurospatial.annotation._state import AnnotationModeState, make_unique_name
from neurospatial.annotation._types import RegionType

if TYPE_CHECKING:
    from collections.abc import Callable

    import napari


class UpdateFeaturesResult(NamedTuple):
    """Result from update_features_for_new_shapes().

    Attributes
    ----------
    assigned_name : str
        The name assigned to the last new shape.
    name_was_modified : bool
        True if name was changed due to duplicate.
    last_role : str
        The role of the last new shape (or "" if no shapes added).

    """

    assigned_name: str
    name_was_modified: bool
    last_role: str


class RenameResult(NamedTuple):
    """Result from rename_shape().

    Attributes
    ----------
    assigned_name : str
        The name that was actually assigned (may differ if duplicate).
    name_was_modified : bool
        True if name was changed due to duplicate.

    """

    assigned_name: str
    name_was_modified: bool


class ShapesLayerController:
    """Adapter applying annotation state to napari shapes layer.

    Isolates all napari Shapes layer interactions to this class,
    making the widget logic cleaner and state changes explicit.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer to control.
    state : AnnotationModeState
        The annotation state to sync with the layer.

    Examples
    --------
    >>> controller = ShapesLayerController(shapes_layer, state)  # doctest: +SKIP
    >>> controller.apply_mode()  # doctest: +SKIP
    >>> names = controller.get_existing_names()  # doctest: +SKIP

    """

    def __init__(
        self,
        shapes_layer: napari.layers.Shapes,
        state: AnnotationModeState,
        *,
        before_mutation: Callable[[], None] | None = None,
        after_mutation: Callable[[], None] | None = None,
    ) -> None:
        self.shapes = shapes_layer
        self.state = state
        self._before_mutation = before_mutation
        self._after_mutation = after_mutation

    def apply_mode(self) -> None:
        """Apply current state mode to layer defaults.

        Updates:
        - feature_defaults["role"] to current region type
        - feature_defaults["name"] to default name for region type
        - current_face_color to region type color
        - the drawing tool: a path for walls, a polygon for areas; landmarks
          keep the tool in use, since their shape sets their kind
        """
        region_type = self.state.region_type
        self.shapes.feature_defaults["role"] = region_type
        self.shapes.feature_defaults["name"] = self.state.default_name()
        self.shapes.current_face_color = REGION_TYPE_COLORS.get(region_type, "yellow")
        if region_type == "wall":
            self.shapes.mode = "add_path"
        elif region_type != "landmark":
            self.shapes.mode = "add_polygon"

    def get_existing_names(self) -> list[str]:
        """Get list of existing shape names.

        Returns
        -------
        list of str
            Names of all shapes in the layer.

        """
        if self.shapes is None or len(self.shapes.data) == 0:
            return []
        features = self.shapes.features
        if features is None or len(features) == 0:
            return []
        return [str(n) for n in features.get("name", [])]

    def get_existing_region_types(self) -> list[RegionType]:
        """Get list of existing annotation region types.

        Returns
        -------
        list of RegionType
            One region type per annotation. An environment's interior ring
            counts as a hole; a region's interior ring is part of that region.

        """
        if self.shapes is None or len(self.shapes.data) == 0:
            return []
        features = self.shapes.features
        if features is None or len(features) == 0:
            return []
        return annotation_roles(features)

    def shape_count(self) -> int:
        """Get current number of shapes in the layer."""
        if self.shapes is None:
            return 0
        return len(self.shapes.data)

    def has_existing_environment(self) -> bool:
        """Check if an environment shape already exists.

        Returns
        -------
        bool
            True if at least one shape has region_type "environment".

        """
        region_types = self.get_existing_region_types()
        return any(r == "environment" for r in region_types)

    def sync_state_from_layer(self) -> None:
        """Sync state counts from current layer data.

        Call this after external changes to the layer (e.g., deletion).
        """
        region_types = self.get_existing_region_types()
        self.state.sync_counts_from_region_types(region_types)

    def update_features_for_new_shapes(
        self,
        prev_count: int,
        name_override: str | None = None,
    ) -> UpdateFeaturesResult:
        """Update features for newly added shapes.

        Parameters
        ----------
        prev_count : int
            Number of shapes before the addition.
        name_override : str, optional
            Name provided by user. If empty/None, auto-generates name.

        Returns
        -------
        UpdateFeaturesResult
            Named tuple with assigned_name, name_was_modified, and last_role.

        """
        current_count = self.shape_count()
        delta = current_count - prev_count

        if delta <= 0:
            return UpdateFeaturesResult("", False, "")

        # Number auto-names after the annotations that still exist.
        self.state.sync_counts_from_region_types(
            annotation_roles(self.shapes.features.iloc[:prev_count])
        )

        # Get current features (napari may have auto-populated with defaults)
        features_len = (
            len(self.shapes.features) if self.shapes.features is not None else 0
        )

        # Keep existing features (for shapes that existed before)
        # Only read up to prev_count to avoid auto-populated placeholder values
        region_types: list[RegionType] = []
        names: list[str] = []
        if features_len > 0:
            for i in range(min(prev_count, features_len)):
                region_types.append(
                    cast("RegionType", str(self.shapes.features["role"].iloc[i])),
                )
                names.append(str(self.shapes.features["name"].iloc[i]))

        # Track if name was modified
        name_was_modified = False
        assigned_name = ""

        # Add entries for new shapes (from prev_count to current_count)
        for index in range(prev_count, current_count):
            # Clipboard rows already carry an identity and parent association;
            # keep their role, and their name unless it is already taken.
            if (
                "_polygon_id" in self.shapes.features
                and self.shapes.features["_polygon_id"].iloc[index]
            ):
                row = self.shapes.features.iloc[[index]]
                role = cast("RegionType", str(row["role"].iloc[0]))
                region_types.append(role)
                assigned_name = str(row["name"].iloc[0])
                if not row["_parent_id"].iloc[0]:
                    assigned_name = make_unique_name(assigned_name, names)
                names.append(assigned_name)
                for counted in annotation_roles(row):
                    self.state.record_shape_added(counted)
                continue
            region_types.append(self.state.region_type)

            # Determine name
            if name_override and name_override.strip():
                requested_name = name_override.strip()
            else:
                # Auto-generate name based on role
                requested_name = self.state.generate_auto_name(names)

            # Ensure uniqueness
            unique_name = make_unique_name(requested_name, names)
            if unique_name != requested_name:
                name_was_modified = True
            names.append(unique_name)
            assigned_name = unique_name

            # Update state counts
            self.state.record_shape_added(self.state.region_type)

        # Apply updated features
        updated = rebuild_features(region_types, names)
        for column in self.shapes.features:
            if str(column).startswith("_"):
                updated[column] = list(
                    self.shapes.features[column].iloc[:current_count]
                )
        self.shapes.features = updated
        sync_colors_from_features(self.shapes)

        # Update layer defaults to match assigned name
        self.shapes.feature_defaults["name"] = assigned_name

        # Get the region_type of the last shape added
        last_role = str(self.state.region_type)

        return UpdateFeaturesResult(assigned_name, name_was_modified, last_role)

    def delete_shapes_by_indices(self, indices_to_delete: set[int]) -> int:
        """Delete shapes at specified indices.

        Parameters
        ----------
        indices_to_delete : set of int
            Indices of shapes to remove.

        Returns
        -------
        int
            Number of shapes deleted.

        """
        if not indices_to_delete:
            return 0

        # A pending boundary decision can remove/reorder rows. Carry the
        # requested identities through that decision instead of reusing indices.
        marker = f"_delete_target_{uuid4().hex}"
        features = self.shapes.features.copy()
        features[marker] = [i in indices_to_delete for i in range(len(features))]
        self.shapes.features = features
        self.shapes.feature_defaults[marker] = False
        try:
            if self._before_mutation is not None:
                self._before_mutation()
            indices_to_delete = {
                i
                for i, selected in enumerate(self.shapes.features.get(marker, []))
                if selected
            }
        finally:
            self.shapes.features = self.shapes.features.drop(
                columns=marker, errors="ignore"
            )
        if not indices_to_delete:
            return 0

        if "_parent_id" in self.shapes.features:
            parent_ids = set(
                self.shapes.features["_polygon_id"].iloc[list(indices_to_delete)]
            ) - {""}
            indices_to_delete = indices_to_delete | {
                i
                for i, parent in enumerate(self.shapes.features["_parent_id"])
                if parent in parent_ids
            }
        delete_count = len(indices_to_delete)
        indices_to_keep = [
            i for i in range(len(self.shapes.data)) if i not in indices_to_delete
        ]

        # Preserve layer state
        old_mode = self.shapes.mode

        with self.shapes.events.data.blocker():
            if not indices_to_keep:
                # Delete all shapes
                self.shapes.data = []
                self.shapes.features = rebuild_features([], [])
            else:
                select_shapes(self.shapes, indices_to_keep)

        self.shapes.selected_data = set()
        self.shapes.mode = old_mode
        self.shapes.refresh()

        # Notify the widget synchronously: queued napari events may arrive after
        # another drawing and must not mistake the deletion for an addition.
        if self._after_mutation is not None:
            self._after_mutation()

        # Update state counts
        self.sync_state_from_layer()

        return delete_count

    def rename_shape(self, idx: int, new_name: str) -> RenameResult:
        """Rename a shape, ensuring uniqueness.

        Parameters
        ----------
        idx : int
            Index of the shape to rename.
        new_name : str
            Desired new name.

        Returns
        -------
        RenameResult
            Named tuple with assigned_name and name_was_modified.

        """
        features_df = self.shapes.features.copy()

        # Get existing names excluding this shape
        other_names = [
            str(features_df.loc[i, "name"]) for i in range(len(features_df)) if i != idx
        ]

        # Ensure unique name
        unique_name = make_unique_name(new_name, other_names)
        features_df.loc[idx, "name"] = unique_name

        self.shapes.features = features_df
        self.shapes.refresh()

        return RenameResult(unique_name, unique_name != new_name)
