"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
from magicgui.widgets import (
    Container,
    Label,
    LineEdit,
    PushButton,
    RadioButtons,
    Select,
    Widget,
)
from numpy.typing import NDArray

from neurospatial.annotation._controller import (
    ShapesLayerController,
    _reset_identity_defaults,
)
from neurospatial.annotation._helpers import (
    REGION_TYPE_CATEGORIES,
    REGION_TYPE_COLOR_CYCLE,
    REGION_TYPE_COLORS,
    shapes_with_types,
    sync_colors_from_features,
)
from neurospatial.annotation._helpers import (
    rebuild_features as rebuild_features,
)
from neurospatial.annotation._state import AnnotationModeState
from neurospatial.annotation._types import RegionType

if TYPE_CHECKING:
    import napari

# What the widget says each annotation type draws.
_MODE_LABELS: dict[str, str] = {
    "environment": "Drawing: ENVIRONMENT (name below)",
    "hole": "Drawing: HOLE (name below)",
    "region": "Drawing: REGION (name below)",
    "wall": "Drawing: WALL as a path (name below)",
    "landmark": "Drawing: LANDMARK - ellipse = point, path = line, polygon = area",
}
_MODE_STATUS: dict[str, str] = {
    "environment": "Mode: ENVIRONMENT - draw boundary polygon (cyan)",
    "hole": "Mode: HOLE - draw exclusion area inside boundary (red)",
    "region": "Mode: REGION - enter name, then draw polygon (yellow)",
    "wall": "Mode: WALL - enter name, then draw the wall as a path (magenta)",
    "landmark": (
        "Mode: LANDMARK - enter name, then draw an ellipse (a point), "
        "a path (a line) or a polygon (an area) (orange)"
    ),
}


class _AnnotationWidget(Container):
    """Annotation controls with an explicit pending-edit save boundary."""

    def __init__(
        self,
        *,
        flush_pending: Callable[[], None],
        sync_from_layer: Callable[[], None],
        next_annotation: Callable[[], tuple[RegionType, str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.flush_pending = flush_pending
        self.sync_from_layer = sync_from_layer
        self.next_annotation = next_annotation


def create_annotation_widget(
    viewer: napari.Viewer,
    shapes_layer_name: str = "Annotations",
    initial_mode: RegionType = "environment",
) -> _AnnotationWidget:
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
        """
        Retrieve the shapes layer by name.

        Returns
        -------
        napari.layers.Shapes or None
            The shapes layer if it exists, otherwise None.
        """
        try:
            return viewer.layers[shapes_layer_name]
        except KeyError:
            return None

    # --- Widget Components ---
    # Compact three-step hint (always visible). The verbose workflow and the
    # full shortcut list live in a collapsible section below so they do not
    # consume vertical space by default and the main controls stay above the
    # fold at a normal (~390 px) dock width.
    instructions = Label(
        value=(
            "Quick start:\n"
            "1. Type a name in 'Next Shape Name' (applies to the next shape)\n"
            "2. Draw the shape (click points, ENTER to finish)\n"
            "3. Save your project, or click 'Done' to return results\n"
            "Rename later: select a shape, edit the name, press Enter."
        ),
    )
    instructions.native.setWordWrap(True)

    # Collapsible details: full workflow + keyboard shortcuts, hidden by default.
    # Use "and" (not "&") -- Qt treats "&" as a mnemonic and would underline the
    # next character, rendering the label as "Workflow shortcuts".
    shortcuts_toggle = PushButton(text="▸ Workflow and shortcuts")
    shortcuts_toggle.native.setStyleSheet("text-align: left;")
    shortcuts_toggle.tooltip = "Show or hide the full workflow and keyboard shortcuts"

    shortcuts = Label(
        value=(
            "Workflow\n"
            "• Draw environment boundary (cyan)\n"
            "• Press M → add holes (red) if needed\n"
            "• Press M → draw named regions (yellow)\n"
            "• Press M → draw walls as paths (magenta)\n"
            "• Press M → draw landmarks (orange): an ellipse marks a point\n"
            "  such as a reward port, a path a line such as a monitor, a\n"
            "  polygon or rectangle an object\n"
            "• Save Project to keep editing; Done to return results\n"
            "\n"
            "Drawing\n"
            "• Click points to draw polygon, ENTER to finish\n"
            "• Name your shape in the input field below\n"
            "\n"
            "Shortcuts\n"
            "• M = cycle modes (environment → hole → region → wall → landmark)\n"
            "• E = environment mode\n"
            "• R = region mode\n"
            "• 3 = move shape\n"
            "• 4 = edit vertices\n"
            "• Delete = remove shape\n"
            "\n"
            "Note: Holes are subtracted from the environment boundary;\n"
            "walls are clipped to it."
        ),
        visible=False,
    )
    shortcuts.native.setWordWrap(True)

    @shortcuts_toggle.clicked.connect
    def _toggle_shortcuts() -> None:
        """Expand or collapse the detailed workflow/shortcuts section."""
        shortcuts.visible = not shortcuts.visible
        shortcuts_toggle.text = (
            "▾ Workflow & shortcuts" if shortcuts.visible else "▸ Workflow & shortcuts"
        )

    # Mode indicator shows current annotation type with colored background;
    # update_mode_indicator sets its text and style after widget creation.
    mode_indicator = Label(value="")

    # --- State Management ---
    # Pure state object for tracking mode and counts (testable without napari)
    state = AnnotationModeState(region_type=initial_mode)

    # Annotation count status (matches AnnotationModeState.status_text format)
    annotation_status = Label(value=state.status_text())

    # Radio buttons for mode selection (more visible than dropdown)
    # Use simple string values - display enhancement via labels
    role_selector = RadioButtons(  # type: ignore[call-arg]
        choices=list(REGION_TYPE_CATEGORIES),
        value=initial_mode,
        orientation="vertical",
        label="Annotation Type:",
    )

    # Name input - always visible for discoverability
    # Default values: "arena" for environment, "" otherwise (auto-named if empty)
    name_input = LineEdit(
        value=state.default_name(),
        label="Next Shape Name:",
        visible=True,  # Always visible - UX: recognition over recall
        tooltip="Name for the next shape you draw. Press Enter after selecting a shape to rename it.",
    )
    # Set placeholder text (magicgui uses native widget underneath)
    name_input.native.setPlaceholderText("Type name, press Enter to apply")

    def annotation_choices(widget: Any = None) -> tuple[tuple[str, int], ...]:
        """Derive choices from domain data when napari refreshes magicgui."""
        shapes = get_shapes()
        if shapes is None:
            return ()
        features = shapes.features
        choices = []
        for index in range(len(shapes.data)):
            name = (
                features["name"].iloc[index]
                if index < len(features)
                else f"shape_{index}"
            )
            role = features["role"].iloc[index] if index < len(features) else "region"
            choices.append((f"#{index + 1}: {name} ({role})", index))
        return tuple(choices)

    # A callable keeps choices intact when adding calibration or prompt layers
    # causes napari to reset every categorical widget in the dock.
    shapes_list = Select(
        choices=annotation_choices,
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
        text="Done",
        tooltip="Validate annotations and return the results to Python.",
    )

    def get_controller() -> ShapesLayerController | None:
        """Get controller for current shapes layer, or None if layer doesn't exist."""
        shapes = get_shapes()
        if shapes is None:
            return None
        return ShapesLayerController(
            shapes,
            state,
            before_mutation=flush_pending,
            after_mutation=mutation_finished,
        )

    # --- Helper Functions ---
    def update_mode_indicator(region_type: str) -> None:
        """Update mode indicator label with visual color feedback.

        Sets both the text and background color to match the shape color,
        making the current mode immediately visible at a glance.
        """
        color = REGION_TYPE_COLORS.get(region_type, "yellow")

        # Include naming hint in the indicator - UX: visibility of system status
        mode_indicator.value = _MODE_LABELS.get(region_type, _MODE_LABELS["region"])

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
        shapes_list.reset_choices()

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
            """
            Filter Qt events to intercept Enter key presses.

            Parameters
            ----------
            obj : QObject
                The object that received the event.
            event : QEvent
                The event to filter.

            Returns
            -------
            bool
                True if the event was handled (consumed), False to pass it through.
            """
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
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        viewer.window._qt_window.close()

    # --- Keyboard Shortcuts ---
    @viewer.bind_key("m")
    def toggle_mode(viewer: napari.Viewer) -> None:
        """Cycle between environment, hole, region, wall and landmark modes."""
        cycle_annotation_mode()
        viewer.status = _MODE_STATUS[role_selector.value]

    @viewer.bind_key("e")
    def set_environment_mode(viewer: napari.Viewer) -> None:
        """Switch directly to environment (boundary) mode."""
        role_selector.value = "environment"
        viewer.status = _MODE_STATUS["environment"]

    @viewer.bind_key("r")
    def set_region_mode(viewer: napari.Viewer) -> None:
        """Switch directly to region mode."""
        role_selector.value = "region"
        viewer.status = _MODE_STATUS["region"]

    # --- Data Change Callback ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure
    _pending_update = [False]  # Flag to prevent duplicate scheduled updates
    _deferred_changes: list[Callable[[], None]] = []

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
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # Delete the existing environment boundary
                    env_indices = [
                        i for i, r in enumerate(roles) if str(r) == "environment"
                    ]
                    # Keep the earlier shapes but the old boundary, then the new
                    # shape(s); each keeps its type
                    n_before = len(shapes.data) - delta
                    kept = [i for i in range(n_before) if i not in env_indices]
                    kept_indices = [*kept, *range(n_before, len(shapes.data))]
                    kept_features = shapes.features.iloc[kept_indices].reset_index(
                        drop=True
                    )
                    new_data = shapes_with_types(shapes, kept_indices)
                    roles = [r for i, r in enumerate(roles) if i not in env_indices]
                    names = [n for i, n in enumerate(names) if i not in env_indices]
                    # Block data events during assignment to prevent re-triggering
                    with shapes.events.data.blocker():
                        shapes.data = new_data
                        shapes.features = kept_features
                    _prev_shape_count[0] = len(kept)
                    current_count = len(shapes.data)
                    viewer.status = "Replaced existing environment boundary"
                else:
                    # Keep the rejection pending until drawing has finished.
                    # A save flush drains it synchronously before any snapshot.
                    marker = f"_rejected_boundary_{uuid4().hex}"
                    features = shapes.features.copy()
                    features[marker] = [
                        i >= _prev_shape_count[0] for i in range(current_count)
                    ]
                    shapes.features = features
                    shapes.feature_defaults[marker] = False
                    _prev_shape_count[0] = current_count

                    def delete_shape() -> None:
                        features_before = shapes.features.copy()
                        kept = [
                            i
                            for i, rejected in enumerate(shapes.features[marker])
                            if not rejected
                        ]
                        features = (
                            shapes.features.iloc[kept]
                            .drop(columns=marker)
                            .reset_index(drop=True)
                        )
                        with shapes.events.data.blocker():
                            shapes.data = shapes_with_types(shapes, kept)
                            shapes.features = features
                            sync_colors_from_features(shapes)
                        _prev_shape_count[0] -= sum(
                            i < _prev_shape_count[0]
                            for i, rejected in enumerate(features_before[marker])
                            if rejected
                        )
                        update_shapes_list()
                        update_annotation_status()
                        _reset_identity_defaults(shapes)

                    _deferred_changes.append(delete_shape)
                    qt_core.QTimer.singleShot(0, flush_pending)
                    viewer.status = "Shape removed (environment already exists)"
                    return

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
                    "edit name above, or press M for holes, regions, walls "
                    "and landmarks"
                )
            else:
                viewer.status = (
                    f"Added {last_role} '{last_name}'{duplicate_note} - "
                    "edit name above if needed"
                )

            # Update name input to show the assigned name
            name_input.value = last_name

        if delta <= 0:
            # Delete interior handles whose parent was removed with napari's tool.
            if "_parent_id" in shapes.features:
                remaining = set(shapes.features["_polygon_id"])
                orphans = {
                    i
                    for i, parent in enumerate(shapes.features["_parent_id"])
                    if parent and parent not in remaining
                }
                controller = get_controller()
                if orphans and controller is not None:
                    controller.delete_shapes_by_indices(orphans)
                    current_count = len(shapes.data)
            update_shapes_list()
            update_annotation_status()

        _prev_shape_count[0] = current_count

    def flush_pending() -> None:
        """Finish queued bookkeeping and rejected drawings before a snapshot."""
        while _pending_update[0] or _deferred_changes:
            if _deferred_changes:
                _deferred_changes.pop(0)()
            elif _pending_update[0]:
                _process_data_change()

    def mutation_finished() -> None:
        shapes = get_shapes()
        if shapes is not None:
            _prev_shape_count[0] = len(shapes.data)
            update_shapes_list()
            update_annotation_status()

    def on_data_changed(event: Any) -> None:
        """Track native removal boundaries without losing queued additions."""
        shapes = get_shapes()
        if shapes is None:
            return
        action = str(getattr(event, "action", ""))
        if action == "removing":
            # Napari has already captured its removal indices. Do not mutate
            # data here. Count the surviving old rows so a queued addition is
            # still recognized even when the total count stays unchanged.
            removed = set(getattr(event, "data_indices", ()))
            _prev_shape_count[0] -= sum(i < _prev_shape_count[0] for i in removed)
            return
        if action in {"adding", "changing"}:
            return
        if action == "removed":
            # Process only bookkeeping here; structural boundary decisions and
            # rejected drawings remain deferred until napari finishes removal.
            has_environment_conflict = (
                state.region_type == "environment"
                and len(shapes.data) > _prev_shape_count[0]
                and "environment"
                in list(shapes.features["role"].iloc[: _prev_shape_count[0]])
            )
            if not has_environment_conflict:
                _process_data_change()
                return
        if not _pending_update[0]:
            _pending_update[0] = True
            qt_core.QTimer.singleShot(0, flush_pending)

    def on_features_changed(event: Any) -> None:
        """Process pasted shapes, which napari announces with features only."""
        shapes = get_shapes()
        if shapes is None or _pending_update[0]:
            return
        count = _prev_shape_count[0]
        if (
            len(shapes.data) > count
            and "_polygon_id" in shapes.features
            and any(shapes.features["_polygon_id"].iloc[count:])
        ):
            _pending_update[0] = True
            qt_core.QTimer.singleShot(0, flush_pending)

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        shapes.events.features.connect(on_features_changed)
        _prev_shape_count[0] = len(shapes.data)

        # Initialize shape defaults from widget state (ensures consistency)
        # This is the single source of truth for annotation mode
        shapes.feature_defaults["role"] = role_selector.value
        shapes.feature_defaults["name"] = state.default_name()
        shapes.current_face_color = REGION_TYPE_COLORS[role_selector.value]
        _reset_identity_defaults(shapes)

        # Note: shapes.events.selected_data doesn't exist in napari
        # (see napari issue #6886). Use highlight event as proxy for selection.
        @shapes.events.highlight.connect
        def on_highlight_changed(event: Any) -> None:
            """Enable/disable buttons when selection highlight changes."""
            update_button_states()

    # Apply initial styling to mode indicator (after widget is created)
    update_mode_indicator(initial_mode)

    # --- Build Container ---
    # Explicit element type: the widgets are a heterogeneous mix of magicgui
    # value widgets whose invariant generic parameters otherwise make mypy widen
    # the inferred list element type to ``object``.
    container_widgets: list[Widget] = [
        instructions,
        shortcuts_toggle,
        shortcuts,
        mode_indicator,
        annotation_status,
        role_selector,
        name_input,
        shapes_list,
        apply_btn,
        delete_btn,
        save_btn,
    ]
    widget = _AnnotationWidget(
        flush_pending=flush_pending,
        sync_from_layer=mutation_finished,
        next_annotation=lambda: (state.region_type, name_input.value.strip()),
        widgets=container_widgets,
        labels=True,
    )
    mutation_finished()

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
) -> tuple[list[NDArray[np.float64]], list[str], list[RegionType], list[str]]:
    """Extract annotation data from shapes layer.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer containing annotations.

    Returns
    -------
    shapes_data : list of NDArray
        List of vertex arrays in napari (row, col) format.
    names : list of str
        Name for each shape.
    region_types : list of RegionType
        Annotation type for each shape.
    shape_types : list of str
        Napari shape type for each shape, such as "polygon", "path" or
        "ellipse"; a landmark's kind follows from it.

    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return [], [], [], []

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
        [str(shape_type) for shape_type in shapes_layer.shape_type],
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
    Preserves existing shapes/features and prepends the boundary's editable rings.

    """
    from neurospatial.annotation._draft import add_polygon

    previous_count = len(shapes_layer.data)
    add_polygon(shapes_layer, boundary, "arena", "environment", calibration)
    order = [*range(previous_count, len(shapes_layer.data)), *range(previous_count)]
    features = shapes_layer.features.iloc[order].reset_index(drop=True)
    shapes_layer.data = shapes_with_types(shapes_layer, order)
    shapes_layer.features = features
    sync_colors_from_features(shapes_layer)
