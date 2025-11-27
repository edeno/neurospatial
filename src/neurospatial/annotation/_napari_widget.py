"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from magicgui.widgets import (
    Container,
    Label,
    LineEdit,
    PushButton,
    RadioButtons,
    Select,
)
from numpy.typing import NDArray

from neurospatial.annotation._controller import ShapesLayerController
from neurospatial.annotation._helpers import (
    REGION_TYPE_CATEGORIES,
    REGION_TYPE_COLOR_CYCLE,
    REGION_TYPE_COLORS,
    rebuild_features,
    sync_face_colors_from_features,
)
from neurospatial.annotation._state import AnnotationModeState
from neurospatial.annotation._types import RegionType

if TYPE_CHECKING:
    import napari


def create_annotation_widget(
    viewer: napari.Viewer,
    shapes_layer_name: str = "Annotations",
    initial_mode: RegionType = "environment",
) -> Container:
    """Create annotation control widget using magicgui.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    shapes_layer_name : str, default="Annotations"
        Name of the Shapes layer to manage.
    initial_mode : str, default="environment"
        Starting annotation mode: "environment" or "region".
        Use "region" when user only needs to annotate regions
        without an environment boundary.

    Returns
    -------
    Container
        Magicgui container widget with annotation controls.

    """

    # Get shapes layer
    def get_shapes() -> napari.layers.Shapes | None:
        try:
            return viewer.layers[shapes_layer_name]
        except KeyError:
            return None

    # --- Widget Components ---
    instructions = Label(
        value=(
            "─── WORKFLOW ───\n"
            "1. Draw environment boundary (cyan)\n"
            "2. Press M → add holes (red) if needed\n"
            "3. Press M → draw named regions (yellow)\n"
            "4. Save and Close when finished\n"
            "\n"
            "─── DRAWING ───\n"
            "• Click points to draw polygon, ENTER to finish\n"
            "• Name your polygon in the input field below\n"
            "\n"
            "─── SHORTCUTS ───\n"
            "• M = cycle modes (environment → hole → region)\n"
            "• E = environment mode\n"
            "• R = region mode\n"
            "• 3 = move shape\n"
            "• 4 = edit vertices\n"
            "• Delete = remove shape\n\n"
            "Note: Holes are subtracted from the environment boundary.\n"
        ),
    )

    # Determine initial state based on mode
    start_in_region_mode = initial_mode == "region"
    start_in_hole_mode = initial_mode == "hole"

    # Mode indicator shows current annotation type with colored background
    # Initial text (will be styled after widget creation)
    if start_in_region_mode:
        mode_indicator = Label(value="Drawing: NAMED REGION")
    elif start_in_hole_mode:
        mode_indicator = Label(value="Drawing: HOLE IN ENVIRONMENT")
    else:
        mode_indicator = Label(value="Drawing: ENVIRONMENT BOUNDARY")

    # Annotation count status
    annotation_status = Label(value="Annotations: 0 environment, 0 regions")

    # Radio buttons for mode selection (more visible than dropdown)
    # Use simple string values - display enhancement via labels
    role_selector = RadioButtons(  # type: ignore[call-arg]
        choices=["environment", "hole", "region"],
        value=initial_mode,
        orientation="vertical",
        label="Annotation Type:",
    )

    # Name input - always visible for discoverability
    # Default values: "arena" for environment, "" for holes/regions (auto-named if empty)
    initial_name = "" if start_in_region_mode or start_in_hole_mode else "arena"

    name_input = LineEdit(
        value=initial_name,
        label="Next Shape Name:",
        visible=True,  # Always visible - UX: recognition over recall
        tooltip="Name for the next shape you draw. Press Enter after selecting a shape to rename it.",
    )
    # Set placeholder text (magicgui uses native widget underneath)
    name_input.native.setPlaceholderText("Type name, press Enter to apply")

    # Shapes list for tracking annotations
    shapes_list = Select(
        choices=[],
        label="Annotations:",
        allow_multiple=False,
    )

    apply_btn = PushButton(
        text="Apply Name to Selected",
        tooltip="Rename selected shape(s) to current name",
        enabled=False,  # Disabled until shape selected
    )
    delete_btn = PushButton(
        text="Delete Selected",
        tooltip="Remove selected shape(s). Shortcut: Delete key",
        enabled=False,  # Disabled until shape selected
    )
    save_btn = PushButton(
        text="Save and Close",
        tooltip="Close viewer and return annotations to Python. Shortcut: Escape",
    )

    # --- State Management ---
    # Pure state object for tracking mode and counts (testable without napari)
    state = AnnotationModeState(region_type=initial_mode)

    def get_controller() -> ShapesLayerController | None:
        """Get controller for current shapes layer, or None if layer doesn't exist."""
        shapes = get_shapes()
        if shapes is None:
            return None
        return ShapesLayerController(shapes, state)

    # --- Helper Functions ---
    def update_mode_indicator(region_type: str) -> None:
        """Update mode indicator label with visual color feedback.

        Sets both the text and background color to match the shape color,
        making the current mode immediately visible at a glance.
        """
        color = REGION_TYPE_COLORS.get(region_type, "yellow")

        # Include naming hint in the indicator - UX: visibility of system status
        if region_type == "environment":
            mode_indicator.value = "Drawing: ENVIRONMENT (name above)"
        elif region_type == "hole":
            mode_indicator.value = "Drawing: HOLE (name above)"
        else:
            mode_indicator.value = "Drawing: REGION (name above)"

        # Set background color to match shape color for visual association
        # Use dark text for light backgrounds, white for red
        text_color = "white" if region_type == "hole" else "black"
        mode_indicator.native.setStyleSheet(
            f"background-color: {color}; color: {text_color}; "
            f"padding: 5px; font-weight: bold; border-radius: 3px;",
        )

    def update_annotation_status() -> None:
        """Update annotation count display from current state."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            state.sync_counts_from_region_types([])
        else:
            region_types = [
                cast("RegionType", str(r)) for r in shapes.features.get("role", [])
            ]
            state.sync_counts_from_region_types(region_types)
        annotation_status.value = state.status_text()

    def update_shapes_list() -> None:
        """Refresh the shapes list from layer data in creation order."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            shapes_list.choices = ()
            return

        # Build list of (label, value) tuples in creation order
        # Magicgui Select expects (label, value) format where label is displayed
        # Using tuples ensures index extraction works even if names contain colons
        features = shapes.features
        choices: list[tuple[str, int]] = []
        for i in range(len(shapes.data)):
            name = features["name"].iloc[i] if i < len(features) else f"shape_{i}"
            role = features["role"].iloc[i] if i < len(features) else "region"
            # Show index for easy reference: "#1: name (role)"
            display_label = f"#{i + 1}: {name} ({role})"
            choices.append((display_label, i))

        # Preserve creation order - no sorting
        shapes_list.choices = tuple(choices)

    def select_shape_in_layer(idx: int) -> None:
        """Select a shape in the layer by index."""
        shapes = get_shapes()
        if shapes is not None and 0 <= idx < len(shapes.data):
            shapes.selected_data = {idx}

    def update_button_states() -> None:
        """Enable/disable buttons based on current selection state."""
        shapes = get_shapes()
        # Handle None selected_data (older napari versions) by treating as empty
        has_selection = (
            shapes is not None
            and getattr(shapes, "selected_data", None) is not None
            and len(shapes.selected_data) > 0
        )
        delete_btn.enabled = has_selection
        apply_btn.enabled = has_selection

    def cycle_annotation_mode() -> None:
        """Cycle between environment, hole, and region modes using state."""
        state.cycle_region_type()
        role_selector.value = state.region_type
        name_input.value = state.default_name()

    # --- Event Handlers ---
    @role_selector.changed.connect
    def on_role_changed(region_type_str: str) -> None:
        """Update feature_defaults and state when region type selector changes."""
        # Sync state with UI (handles both programmatic and user changes)
        # Cast validated - role_selector only allows valid RegionType values

        state.region_type = cast("RegionType", region_type_str)
        # Controller handles layer defaults synchronization
        controller = get_controller()
        if controller is not None:
            controller.apply_mode()
        update_mode_indicator(region_type_str)
        name_input.value = state.default_name()

    @name_input.changed.connect
    def on_name_changed(name: str) -> None:
        """Update feature_defaults when name changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["name"] = name

    @shapes_list.changed.connect
    def on_shapes_list_selection(selection: Any) -> None:
        """Select shape in layer when selected in list."""
        if selection is None:
            return
        # Select widget returns a list even with allow_multiple=False
        if isinstance(selection, list):
            selection = selection[0] if selection else None
        if selection is None:
            return
        # With (value, label) tuples, selection is already the index value
        idx = selection
        if not isinstance(idx, int):
            return
        select_shape_in_layer(idx)
        # Update name input to match selected shape
        shapes = get_shapes()
        if shapes is not None and idx < len(shapes.features):
            name_input.value = shapes.features["name"].iloc[idx]

    @apply_btn.clicked.connect
    def apply_to_selected() -> None:
        """Apply current name to selected shapes using controller."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        controller = get_controller()
        if controller is None:
            return

        count = len(shapes.selected_data)
        requested_name = name_input.value
        applied_name = None
        name_was_modified = False

        # Controller handles renaming with uniqueness enforcement
        for idx in shapes.selected_data:
            if 0 <= idx < len(shapes.features):
                assigned_name, was_modified = controller.rename_shape(
                    idx,
                    requested_name,
                )
                applied_name = assigned_name
                if was_modified:
                    name_was_modified = True

        # Widget handles UI updates
        update_shapes_list()

        # Status feedback - warn if name was modified due to duplicate
        if count == 1 and applied_name:
            if name_was_modified:
                viewer.status = (
                    f"Renamed to '{applied_name}' ('{requested_name}' already exists)"
                )
            else:
                viewer.status = f"Renamed to '{applied_name}'"
        else:
            viewer.status = f"Renamed {count} shapes"

    # Connect Enter key in name input to apply name to selected
    # Use event filter to prevent Enter from propagating to napari's canvas

    from qtpy import QtCore

    qt_core: Any = QtCore

    class EnterKeyFilter(qt_core.QObject):
        """Event filter to handle Enter key in name input without propagating to napari."""

        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt override
            if event.type() == qt_core.QEvent.KeyPress and event.key() in (
                qt_core.Qt.Key_Return,
                qt_core.Qt.Key_Enter,
            ):
                apply_to_selected()
                return True  # Consume the event, don't propagate
            return False  # Let other events through

    _enter_filter = EnterKeyFilter(name_input.native)
    name_input.native.installEventFilter(_enter_filter)

    @delete_btn.clicked.connect
    def delete_selected() -> None:
        """Delete selected shapes using controller."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        controller = get_controller()
        if controller is None:
            return

        # Controller handles layer mutations and state sync
        delete_count = controller.delete_shapes_by_indices(set(shapes.selected_data))

        # Widget handles UI updates
        update_shapes_list()
        update_annotation_status()
        viewer.status = f"Deleted {delete_count} shape(s)"

    @save_btn.clicked.connect
    def save_and_close() -> None:
        """Close viewer to return control to Python, with confirmation if empty."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            # Show confirmation dialog when no annotations exist
            # Use napari window as parent for proper modality
            from qtpy.QtWidgets import QMessageBox

            parent = viewer.window._qt_window
            reply = QMessageBox.question(
                parent,
                "No Annotations",
                "Close without saving any annotations?\n\n"
                "Python will receive an empty result.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        viewer.close()

    # --- Keyboard Shortcuts ---
    @viewer.bind_key("m")
    def toggle_mode(viewer: napari.Viewer) -> None:
        """Cycle between environment, hole, and region modes."""
        cycle_annotation_mode()
        # Status bar feedback
        if role_selector.value == "environment":
            viewer.status = "Mode: ENVIRONMENT - draw boundary polygon (cyan)"
        elif role_selector.value == "hole":
            viewer.status = "Mode: HOLE - draw exclusion area inside boundary (red)"
        else:
            viewer.status = "Mode: REGION - enter name, then draw polygon (yellow)"

    @viewer.bind_key("e")
    def set_environment_mode(viewer: napari.Viewer) -> None:
        """Switch directly to environment (boundary) mode."""
        role_selector.value = "environment"
        viewer.status = "Mode: ENVIRONMENT - draw boundary polygon (cyan)"

    @viewer.bind_key("r")
    def set_region_mode(viewer: napari.Viewer) -> None:
        """Switch directly to region mode."""
        role_selector.value = "region"
        viewer.status = "Mode: REGION - enter name, then draw polygon (yellow)"

    @viewer.bind_key("Escape")
    def finish_annotation(viewer: napari.Viewer) -> None:
        """Finish annotation and close viewer."""
        save_and_close()

    # --- Data Change Callback ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure
    _pending_update = [False]  # Flag to prevent duplicate scheduled updates

    def _process_data_change() -> None:
        """Handle shape additions: update list, auto-switch mode, update status.

        This is called via QTimer.singleShot(0, ...) to accumulate rapid events
        and process once, reducing UI latency.
        """
        _pending_update[0] = False  # Reset flag - we're now processing

        shapes = get_shapes()
        if shapes is None:
            return

        current_count = len(shapes.data)
        delta = current_count - _prev_shape_count[0]

        if delta > 0:
            # New shape(s) added - ensure features are set
            # Napari's feature_defaults sometimes doesn't work properly

            # Build new features list from scratch to avoid concat issues
            features_len = len(shapes.features) if shapes.features is not None else 0
            roles = (
                [str(r) for r in shapes.features["role"]] if features_len > 0 else []
            )
            names = (
                [str(n) for n in shapes.features["name"]] if features_len > 0 else []
            )

            # Check if user is trying to add a second environment boundary
            # Only count environments from shapes that existed BEFORE this addition
            # (features might include auto-populated entries from napari)
            pre_existing_count = _prev_shape_count[0]
            existing_env_count = sum(
                1
                for i, r in enumerate(roles)
                if i < pre_existing_count and str(r) == "environment"
            )
            if state.region_type == "environment" and existing_env_count > 0:
                # Show dialog asking if they want to replace the existing boundary
                from qtpy.QtWidgets import QMessageBox

                parent = viewer.window._qt_window
                reply = QMessageBox.question(
                    parent,
                    "Environment Already Exists",
                    "An environment boundary already exists.\n\n"
                    "Replace it with this new boundary?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    # Delete the existing environment boundary
                    env_indices = [
                        i for i, r in enumerate(roles) if str(r) == "environment"
                    ]
                    # Remove from data, roles, names (in reverse order)
                    new_data = [
                        shapes.data[i]
                        for i in range(len(shapes.data) - delta)
                        if i not in env_indices
                    ]
                    # Also include the new shape(s)
                    new_data.extend(shapes.data[-delta:])
                    roles = [r for i, r in enumerate(roles) if i not in env_indices]
                    names = [n for i, n in enumerate(names) if i not in env_indices]
                    # Block data events during assignment to prevent re-triggering
                    with shapes.events.data.blocker():
                        shapes.data = new_data
                    current_count = len(shapes.data)
                    viewer.status = "Replaced existing environment boundary"
                else:
                    # Delete the just-drawn shape
                    # Must defer to avoid reentrancy with napari's _finish_drawing()
                    # Update count IMMEDIATELY to prevent dialog re-triggering
                    # (napari may fire multiple data events for single shape)
                    target_count = _prev_shape_count[0]  # What we want after deletion
                    _prev_shape_count[0] = current_count  # Prevent re-entry

                    def delete_shape() -> None:
                        nonlocal _prev_shape_count
                        # Delete back to target count
                        if len(shapes.data) > target_count:
                            # Block data events during deletion to prevent re-triggering
                            with shapes.events.data.blocker():
                                shapes.data = shapes.data[:target_count]
                                # Also truncate features to match data length
                                if len(shapes.features) > target_count:
                                    truncated_roles = [
                                        str(r)
                                        for r in shapes.features["role"][:target_count]
                                    ]
                                    truncated_names = [
                                        str(n)
                                        for n in shapes.features["name"][:target_count]
                                    ]
                                    shapes.features = rebuild_features(
                                        cast("list[RegionType]", truncated_roles),
                                        truncated_names,
                                    )
                                    sync_face_colors_from_features(shapes)
                            # Update UI to reflect the deletion
                            update_shapes_list()
                            update_annotation_status()
                        _prev_shape_count[0] = len(shapes.data)

                    qt_core.QTimer.singleShot(0, delete_shape)
                    viewer.status = "Shape removed (environment already exists)"
                    return  # Exit early - deferred deletion will handle cleanup

            # Use controller to update features for new shapes
            controller = get_controller()
            if controller is None:
                return

            # Get name from input (controller handles auto-naming if empty)
            name_override = name_input.value.strip() or None
            result = controller.update_features_for_new_shapes(
                _prev_shape_count[0],
                name_override=name_override,
            )

            # Widget handles UI updates
            update_shapes_list()
            update_annotation_status()

            # Status feedback for shape added
            last_name = result.assigned_name
            last_role = result.last_role
            duplicate_note = (
                f" ('{name_override}' already exists)"
                if result.name_was_modified
                else ""
            )
            if last_role == "environment":
                viewer.status = (
                    f"Added environment '{last_name}'{duplicate_note} - "
                    "edit name above, or press M for holes/regions"
                )
            elif last_role == "hole":
                viewer.status = (
                    f"Added hole '{last_name}'{duplicate_note} - "
                    "edit name above if needed"
                )
            else:
                viewer.status = (
                    f"Added region '{last_name}'{duplicate_note} - "
                    "edit name above if needed"
                )

            # Update name input to show the assigned name
            name_input.value = last_name

        elif delta < 0:
            # Shape was deleted externally (e.g., via Delete key)
            update_shapes_list()
            update_annotation_status()

        _prev_shape_count[0] = current_count

    def on_data_changed(event: Any) -> None:
        """Throttle data change events using QTimer.singleShot.

        Napari's Shapes.events.data may fire several times for a single shape
        addition. This wrapper schedules the actual processing to run once
        after the event storm settles, reducing UI latency.
        """
        if not _pending_update[0]:
            _pending_update[0] = True
            qt_core.QTimer.singleShot(0, _process_data_change)

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        _prev_shape_count[0] = len(shapes.data)

        # Initialize shape defaults from widget state (ensures consistency)
        # This is the single source of truth for annotation mode
        shapes.feature_defaults["role"] = role_selector.value
        if role_selector.value == "environment":
            shapes.feature_defaults["name"] = "arena"
            shapes.current_face_color = REGION_TYPE_COLORS["environment"]
        else:
            shapes.feature_defaults["name"] = ""
            shapes.current_face_color = REGION_TYPE_COLORS["region"]

        # Note: shapes.events.selected_data doesn't exist in napari
        # (see napari issue #6886). Use highlight event as proxy for selection.
        @shapes.events.highlight.connect
        def on_highlight_changed(event: Any) -> None:
            """Enable/disable buttons when selection highlight changes."""
            update_button_states()

    # Apply initial styling to mode indicator (after widget is created)
    update_mode_indicator(initial_mode)

    # --- Build Container ---
    widget = Container(
        widgets=[
            instructions,
            mode_indicator,
            annotation_status,
            role_selector,
            name_input,
            shapes_list,
            apply_btn,
            delete_btn,
            save_btn,
        ],
        labels=False,
    )

    return widget


def setup_shapes_layer_for_annotation(
    viewer: napari.Viewer,
    initial_mode: RegionType = "environment",
) -> napari.layers.Shapes:
    """Create and configure a Shapes layer optimized for annotation.

    Uses napari best practices:
    - Features-based coloring for role distinction
    - Text labels showing shape names
    - Initialized feature_defaults for new shapes

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    initial_mode : str, default="environment"
        Starting annotation mode: "environment" or "region".

    Returns
    -------
    napari.layers.Shapes
        Configured shapes layer ready for annotation.

    """
    import pandas as pd

    # Initialize with empty features DataFrame (categorical for color cycling)
    features = pd.DataFrame(
        {
            "role": pd.Categorical([], categories=REGION_TYPE_CATEGORIES),
            "name": pd.Series([], dtype=str),
        },
    )

    shapes = viewer.add_shapes(
        name="Annotations",
        shape_type="polygon",
        # Features-based coloring
        features=features,
        face_color="role",
        face_color_cycle=REGION_TYPE_COLOR_CYCLE,
        edge_color="white",
        edge_width=2,
        # Text labels on shapes
        text={
            "string": "{name}",
            "size": 10,
            "color": "white",
            "anchor": "upper_left",
            "translation": [5, 5],
        },
    )

    # Set defaults for new shapes based on initial mode
    shapes.feature_defaults["role"] = initial_mode
    if initial_mode == "region":
        shapes.feature_defaults["name"] = ""
        shapes.current_face_color = REGION_TYPE_COLORS["region"]
    else:
        shapes.feature_defaults["name"] = "arena"
        shapes.current_face_color = REGION_TYPE_COLORS["environment"]

    # Start in polygon drawing mode
    shapes.mode = "add_polygon"

    return shapes


def get_annotation_data(
    shapes_layer: napari.layers.Shapes | None,
) -> tuple[list[NDArray[np.float64]], list[str], list[RegionType]]:
    """Extract annotation data from shapes layer.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer containing annotations.

    Returns
    -------
    shapes_data : list of NDArray
        List of polygon vertex arrays in napari (row, col) format.
    names : list of str
        Name for each shape.
    region_types : list of RegionType
        Region type for each shape ("environment", "hole", or "region").

    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return [], [], []

    data = list(shapes_layer.data)

    # Get from features (preferred) or properties (fallback)
    if hasattr(shapes_layer, "features") and len(shapes_layer.features) > 0:
        features = shapes_layer.features
        names = list(features.get("name", [f"region_{i}" for i in range(len(data))]))
        region_types = list(features.get("role", ["region"] * len(data)))
    else:
        props = shapes_layer.properties
        names = list(props.get("name", [f"region_{i}" for i in range(len(data))]))
        region_types = list(props.get("role", ["region"] * len(data)))

    # Ensure lists match data length exactly
    # Truncate if features have extra entries (can happen with napari sync issues)
    names = names[: len(data)]
    region_types = region_types[: len(data)]

    # Pad if features are missing entries
    while len(names) < len(data):
        names.append(f"region_{len(names)}")
    while len(region_types) < len(data):
        region_types.append("region")

    # Cast region_types to RegionType (validated at runtime by napari widget constraints)
    from typing import cast

    return (
        data,
        [str(n) for n in names],
        [cast("RegionType", str(r)) for r in region_types],
    )


def add_initial_boundary_to_shapes(
    shapes_layer: napari.layers.Shapes,
    boundary: Any,
    calibration: Any = None,
) -> None:
    """Add pre-drawn boundary polygon to shapes layer for editing.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The annotation shapes layer (may already contain shapes from initial_regions).
    boundary : Polygon
        Shapely Polygon. Coordinate system depends on calibration:
        - With calibration: environment units (cm), Y-up origin
        - Without calibration: video pixels (x, y), Y-down origin
    calibration : VideoCalibration, optional
        Transform from environment coords (cm) to video pixels.
        If None, boundary coords are assumed to be in video pixels already.

    Notes
    -----
    Mirrors the pattern in _add_initial_regions() for consistency.
    Preserves existing shapes/features and prepends the boundary to front.

    """
    from shapely import get_coordinates

    # Preserve existing features before adding new shape
    existing_roles = list(shapes_layer.features.get("role", []))
    existing_names = list(shapes_layer.features.get("name", []))

    # Get polygon exterior vertices
    coords = get_coordinates(boundary.exterior)

    # Transform to pixels if calibration provided
    # NOTE: transform_cm_to_px handles Y-flip internally - don't double-flip!
    if calibration is not None:
        coords = calibration.transform_cm_to_px(coords)

    # Convert to napari (row, col) order
    coords_rc = coords[:, ::-1]

    # Add to shapes layer (appends to existing shapes)
    shapes_layer.add([coords_rc], shape_type="polygon")

    # Extend features with the new boundary shape
    # NOTE: Boundary should be first in the list for proper widget mode handling
    new_region_types: list[RegionType] = [
        "environment",
        *[cast("RegionType", str(r)) for r in existing_roles],
    ]
    new_names = ["arena", *[str(n) for n in existing_names]]
    shapes_layer.features = rebuild_features(new_region_types, new_names)

    # Reorder data so boundary is first (environment boundary should be drawn first)
    # This ensures the widget's mode logic works correctly
    if len(existing_roles) > 0:
        # Move the last shape (just added) to the front
        data = list(shapes_layer.data)
        data = [data[-1], *data[:-1]]
        shapes_layer.data = data

    # Sync face colors from features
    sync_face_colors_from_features(shapes_layer)
