"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import napari
    import pandas as pd

# Role categories - order determines color cycle mapping
# Environment first since users typically define boundary first
ROLE_CATEGORIES = ["environment", "hole", "region"]

# Color scheme for role-based visualization
ROLE_COLORS = {
    "environment": "cyan",
    "hole": "red",
    "region": "yellow",
}
ROLE_COLOR_CYCLE = [ROLE_COLORS[cat] for cat in ROLE_CATEGORIES]


def rebuild_features(roles: list[str], names: list[str]) -> pd.DataFrame:
    """
    Create a fresh features DataFrame with proper categorical types.

    Centralizes feature DataFrame construction to ensure consistency
    across all shape update operations.

    Parameters
    ----------
    roles : list of str
        Role for each shape ("environment" or "region").
    names : list of str
        Name for each shape.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical 'role' and string 'name' columns.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "role": pd.Categorical(roles, categories=ROLE_CATEGORIES),
            "name": pd.Series(names, dtype=str),
        }
    )


def sync_face_colors_from_features(shapes_layer: napari.layers.Shapes) -> None:
    """
    Update face colors to match feature roles.

    Napari's face_color_cycle doesn't always work reliably when features
    are updated programmatically. This function explicitly syncs colors.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        Shapes layer to update.
    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return

    roles = shapes_layer.features.get("role", [])
    face_colors = [ROLE_COLORS.get(str(r), "yellow") for r in roles]
    shapes_layer.face_color = face_colors


def make_unique_name(base_name: str, existing_names: list[str]) -> str:
    """
    Generate a unique name by appending a suffix if needed.

    Parameters
    ----------
    base_name : str
        The desired name.
    existing_names : list of str
        Names that are already in use.

    Returns
    -------
    str
        A unique name (base_name if available, or base_name_N if not).
    """
    if base_name not in existing_names:
        return base_name

    # Find next available suffix
    counter = 2
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    return f"{base_name}_{counter}"


def create_annotation_widget(
    viewer: napari.Viewer,
    shapes_layer_name: str = "Annotations",
    initial_mode: str = "environment",
) -> Container:
    """
    Create annotation control widget using magicgui.

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
    def get_shapes():
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
            "4. Escape or Save and Close when finished\n"
            "\n"
            "─── DRAWING ───\n"
            "• Click points to draw polygon, ENTER to finish\n"
            "• Unnamed shapes auto-named: hole_1, region_1, ...\n"
            "\n"
            "─── SHORTCUTS ───\n"
            "• M = cycle modes (environment → hole → region)\n"
            "• 3 = move shape\n"
            "• 4 = edit vertices\n"
            "• Delete = remove shape\n"
            "• Escape = save and close\n"
            "• Ctrl+Z = undo"
        )
    )

    # Determine initial state based on mode
    start_in_region_mode = initial_mode == "region"

    # Mode indicator shows current annotation type with action-oriented wording
    if start_in_region_mode:
        mode_indicator = Label(value="Drawing: NAMED REGION (yellow polygon)")
    else:
        mode_indicator = Label(value="Drawing: ENVIRONMENT BOUNDARY (cyan polygon)")

    # Annotation count status
    annotation_status = Label(value="Annotations: 0 environment, 0 regions")

    # Radio buttons for mode selection (more visible than dropdown)
    # Use simple string values - display enhancement via labels
    role_selector = RadioButtons(
        choices=["environment", "hole", "region"],
        value=initial_mode,  # type: ignore[call-arg]
        orientation="vertical",
        label="Annotation Type:",
    )

    # Name input - visible only for region mode (progressive disclosure)
    name_input = LineEdit(
        value="" if start_in_region_mode else "arena",
        label="Shape Name:",
        visible=start_in_region_mode,
        tooltip="Enter name for next shape, or select a shape and press Enter to rename",
    )
    # Set placeholder text (magicgui uses native widget underneath)
    name_input.native.setPlaceholderText("Enter region name...")

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

    # --- Helper Functions ---
    def update_mode_indicator(role: str):
        """Update mode indicator label with action-oriented wording."""
        if role == "environment":
            mode_indicator.value = "Drawing: ENVIRONMENT BOUNDARY (cyan polygon)"
        elif role == "hole":
            mode_indicator.value = "Drawing: HOLE IN ENVIRONMENT (red polygon)"
        else:
            mode_indicator.value = "Drawing: NAMED REGION (yellow polygon)"

    def update_annotation_status():
        """Update annotation count display."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            annotation_status.value = "Annotations: 0 environment, 0 holes, 0 regions"
            return

        features = shapes.features
        env_count = sum(1 for r in features["role"] if str(r) == "environment")
        hole_count = sum(1 for r in features["role"] if str(r) == "hole")
        region_count = sum(1 for r in features["role"] if str(r) == "region")
        annotation_status.value = f"Annotations: {env_count} environment, {hole_count} holes, {region_count} regions"

    def update_shapes_list():
        """Refresh the shapes list from layer data in creation order."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            shapes_list.choices = []
            return

        # Build list of (label, value) tuples in creation order
        # Magicgui Select expects (label, value) format where label is displayed
        # Using tuples ensures index extraction works even if names contain colons
        features = shapes.features
        choices = []
        for i in range(len(shapes.data)):
            name = features["name"].iloc[i] if i < len(features) else f"shape_{i}"
            role = features["role"].iloc[i] if i < len(features) else "region"
            # Show index for easy reference: "#1: name (role)"
            display_label = f"#{i + 1}: {name} ({role})"
            choices.append((display_label, i))

        # Preserve creation order - no sorting
        shapes_list.choices = choices

    def select_shape_in_layer(idx: int):
        """Select a shape in the layer by index."""
        shapes = get_shapes()
        if shapes is not None and 0 <= idx < len(shapes.data):
            shapes.selected_data = {idx}

    def update_button_states():
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

    def cycle_annotation_mode():
        """Cycle between environment, hole, and region modes."""
        if role_selector.value == "environment":
            role_selector.value = "hole"
            name_input.value = ""
            name_input.visible = False  # Hide for hole (auto-named)
        elif role_selector.value == "hole":
            role_selector.value = "region"
            name_input.value = ""
            name_input.visible = True  # Show name input for regions
        else:
            role_selector.value = "environment"
            name_input.value = "arena"
            name_input.visible = False  # Hide for environment (uses "arena")

    # --- Event Handlers ---
    @role_selector.changed.connect
    def on_role_changed(role: str):
        """Update feature_defaults when role selector changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["role"] = role
            # Update colors for visual feedback
            shapes.current_face_color = ROLE_COLORS.get(role, "yellow")
        update_mode_indicator(role)
        # Progressive disclosure: show name input only for regions
        if role == "region":
            name_input.visible = True
            if not name_input.value:  # Clear if switching to region
                name_input.value = ""
        elif role == "hole":
            name_input.visible = False
            name_input.value = ""  # Holes are auto-named
        else:  # environment
            name_input.visible = False
            name_input.value = "arena"

    @name_input.changed.connect
    def on_name_changed(name: str):
        """Update feature_defaults when name changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["name"] = name

    @shapes_list.changed.connect
    def on_shapes_list_selection(selection):
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
    def apply_to_selected():
        """Apply current name to selected shapes."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        count = len(shapes.selected_data)
        # Use DataFrame operations for safer feature updates
        features_df = shapes.features.copy()
        applied_name = None
        for idx in shapes.selected_data:
            if 0 <= idx < len(features_df):
                # Get existing names excluding this shape
                other_names = [
                    str(features_df.loc[i, "name"])
                    for i in range(len(features_df))
                    if i != idx
                ]
                # Ensure unique name
                requested_name = name_input.value
                unique_name = make_unique_name(requested_name, other_names)
                features_df.loc[idx, "name"] = unique_name
                applied_name = unique_name
                # Track if name was changed due to duplicate
                name_was_modified = unique_name != requested_name

        shapes.features = features_df
        shapes.refresh()
        update_shapes_list()
        # Status feedback - warn if name was modified due to duplicate
        if count == 1 and applied_name:
            if name_was_modified:
                viewer.status = (
                    f"Renamed to '{applied_name}' ('{name_input.value}' already exists)"
                )
            else:
                viewer.status = f"Renamed to '{applied_name}'"
        else:
            viewer.status = f"Renamed {count} shapes"

    # Connect Enter key in name input to apply name to selected
    # Use event filter to prevent Enter from propagating to napari's canvas
    from qtpy.QtCore import QEvent, QObject

    class EnterKeyFilter(QObject):
        """Event filter to handle Enter key in name input without propagating to napari."""

        def eventFilter(self, obj, event):  # noqa: N802 - Qt override
            if event.type() == QEvent.KeyPress:
                from qtpy.QtCore import Qt

                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    apply_to_selected()
                    return True  # Consume the event, don't propagate
            return False  # Let other events through

    _enter_filter = EnterKeyFilter(name_input.native)
    name_input.native.installEventFilter(_enter_filter)

    @delete_btn.clicked.connect
    def delete_selected():
        """Delete selected shapes."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        # Preserve layer state
        old_mode = shapes.mode
        delete_count = len(shapes.selected_data)

        # Get indices to keep (not selected)
        indices_to_delete = set(shapes.selected_data)
        indices_to_keep = [
            i for i in range(len(shapes.data)) if i not in indices_to_delete
        ]

        # Block data events during multi-assignment to prevent redundant redraws
        with shapes.events.data.blocker():
            if not indices_to_keep:
                # Delete all shapes
                shapes.data = []
                shapes.features = rebuild_features([], [])
            else:
                # Keep only non-deleted shapes and features
                new_data = [shapes.data[i] for i in indices_to_keep]
                new_roles = [
                    str(shapes.features["role"].iloc[i]) for i in indices_to_keep
                ]
                new_names = [
                    str(shapes.features["name"].iloc[i]) for i in indices_to_keep
                ]

                shapes.data = new_data
                shapes.features = rebuild_features(new_roles, new_names)
                sync_face_colors_from_features(shapes)

        shapes.selected_data = set()
        shapes.mode = old_mode  # Restore layer mode
        shapes.refresh()
        update_shapes_list()
        update_annotation_status()
        # Status feedback
        viewer.status = f"Deleted {delete_count} shape(s)"

    @save_btn.clicked.connect
    def save_and_close():
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
    def toggle_mode(viewer):
        """Cycle between environment, hole, and region modes."""
        cycle_annotation_mode()
        # Status bar feedback
        if role_selector.value == "environment":
            viewer.status = "Mode: ENVIRONMENT - draw boundary polygon (cyan)"
        elif role_selector.value == "hole":
            viewer.status = "Mode: HOLE - draw exclusion area inside boundary (red)"
        else:
            viewer.status = "Mode: REGION - enter name, then draw polygon (yellow)"

    @viewer.bind_key("Escape")
    def finish_annotation(viewer):
        """Finish annotation and close viewer."""
        save_and_close()

    # --- Data Change Callback ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure
    _pending_update = [False]  # Flag to prevent duplicate scheduled updates

    def _process_data_change():
        """
        Handle shape additions: update list, auto-switch mode, update status.

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
            if role_selector.value == "environment" and existing_env_count > 0:
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
                    from qtpy.QtCore import QTimer

                    # Update count IMMEDIATELY to prevent dialog re-triggering
                    # (napari may fire multiple data events for single shape)
                    target_count = _prev_shape_count[0]  # What we want after deletion
                    _prev_shape_count[0] = current_count  # Prevent re-entry

                    def delete_shape():
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
                                        truncated_roles, truncated_names
                                    )
                                    sync_face_colors_from_features(shapes)
                            # Update UI to reflect the deletion
                            update_shapes_list()
                            update_annotation_status()
                        _prev_shape_count[0] = len(shapes.data)

                    QTimer.singleShot(0, delete_shape)
                    viewer.status = "Shape removed (environment already exists)"
                    return  # Exit early - deferred deletion will handle cleanup

            # Add entries for any new shapes (handles multiple additions)
            name_was_modified = False
            requested_name = None
            while len(roles) < current_count:
                roles.append(role_selector.value)
                # Use current name, or generate fallback if empty
                current_name = name_input.value.strip()
                if not current_name:
                    # Fallback name if user didn't enter one
                    role = role_selector.value
                    if role == "environment":
                        current_name = "arena"
                    elif role == "hole":
                        hole_count = sum(1 for r in roles if str(r) == "hole")
                        current_name = f"hole_{hole_count}"
                    else:
                        region_count = sum(1 for r in roles if str(r) == "region")
                        current_name = f"region_{region_count}"
                # Ensure unique name to prevent overwrites in Regions container
                requested_name = current_name
                unique_name = make_unique_name(current_name, names)
                if unique_name != requested_name:
                    name_was_modified = True
                names.append(unique_name)

            # Use centralized feature builder
            shapes.features = rebuild_features(roles, names)
            sync_face_colors_from_features(shapes)

            update_shapes_list()
            update_annotation_status()

            # Status feedback for shape added (no auto-switch - user controls mode with M)
            last_role = str(shapes.features["role"].iloc[-1])
            last_name = str(shapes.features["name"].iloc[-1])
            # Warn if name was modified due to duplicate
            duplicate_note = (
                f" ('{requested_name}' already exists)" if name_was_modified else ""
            )
            if last_role == "environment":
                viewer.status = (
                    f"Added environment '{last_name}'{duplicate_note} "
                    "(press M to add holes or regions)"
                )
            elif last_role == "hole":
                viewer.status = f"Added hole '{last_name}'{duplicate_note}"
            else:
                viewer.status = f"Added region '{last_name}'{duplicate_note}"
                # Clear name for next region (encourage meaningful naming)
                name_input.value = ""
                shapes.feature_defaults["name"] = ""

        elif delta < 0:
            # Shape was deleted externally (e.g., via Delete key)
            update_shapes_list()
            update_annotation_status()

        _prev_shape_count[0] = current_count

    def on_data_changed(event):
        """
        Throttle data change events using QTimer.singleShot.

        Napari's Shapes.events.data may fire several times for a single shape
        addition. This wrapper schedules the actual processing to run once
        after the event storm settles, reducing UI latency.
        """
        from qtpy.QtCore import QTimer

        if not _pending_update[0]:
            _pending_update[0] = True
            QTimer.singleShot(0, _process_data_change)

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        _prev_shape_count[0] = len(shapes.data)

        # Initialize shape defaults from widget state (ensures consistency)
        # This is the single source of truth for annotation mode
        shapes.feature_defaults["role"] = role_selector.value
        if role_selector.value == "environment":
            shapes.feature_defaults["name"] = "arena"
            shapes.current_face_color = ROLE_COLORS["environment"]
        else:
            shapes.feature_defaults["name"] = ""
            shapes.current_face_color = ROLE_COLORS["region"]

        # Note: shapes.events.selected_data doesn't exist in napari
        # (see napari issue #6886). Use highlight event as proxy for selection.
        @shapes.events.highlight.connect
        def on_highlight_changed(event):
            """Enable/disable buttons when selection highlight changes."""
            update_button_states()

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
    initial_mode: str = "environment",
) -> napari.layers.Shapes:
    """
    Create and configure a Shapes layer optimized for annotation.

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
            "role": pd.Categorical([], categories=ROLE_CATEGORIES),
            "name": pd.Series([], dtype=str),
        }
    )

    shapes = viewer.add_shapes(
        name="Annotations",
        shape_type="polygon",
        # Features-based coloring
        features=features,
        face_color="role",
        face_color_cycle=ROLE_COLOR_CYCLE,
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
        shapes.current_face_color = ROLE_COLORS["region"]
    else:
        shapes.feature_defaults["name"] = "arena"
        shapes.current_face_color = ROLE_COLORS["environment"]

    # Start in polygon drawing mode
    shapes.mode = "add_polygon"

    return shapes


def get_annotation_data(
    shapes_layer: napari.layers.Shapes | None,
) -> tuple[list[NDArray[np.float64]], list[str], list[str]]:
    """
    Extract annotation data from shapes layer.

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
    roles : list of str
        Role for each shape ("environment" or "region").
    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return [], [], []

    data = list(shapes_layer.data)

    # Get from features (preferred) or properties (fallback)
    if hasattr(shapes_layer, "features") and len(shapes_layer.features) > 0:
        features = shapes_layer.features
        names = list(features.get("name", [f"region_{i}" for i in range(len(data))]))
        roles = list(features.get("role", ["region"] * len(data)))
    else:
        props = shapes_layer.properties
        names = list(props.get("name", [f"region_{i}" for i in range(len(data))]))
        roles = list(props.get("role", ["region"] * len(data)))

    # Ensure lists match data length exactly
    # Truncate if features have extra entries (can happen with napari sync issues)
    names = names[: len(data)]
    roles = roles[: len(data)]

    # Pad if features are missing entries
    while len(names) < len(data):
        names.append(f"region_{len(names)}")
    while len(roles) < len(data):
        roles.append("region")

    return data, [str(n) for n in names], [str(r) for r in roles]
