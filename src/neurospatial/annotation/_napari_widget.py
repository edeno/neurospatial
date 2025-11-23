"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    Label,
    LineEdit,
    PushButton,
    Select,
)
from numpy.typing import NDArray

if TYPE_CHECKING:
    import napari

# Role categories - order determines color cycle mapping
# Environment first since users typically define boundary first
ROLE_CATEGORIES = ["environment", "region"]

# Color scheme for role-based visualization
ROLE_COLORS = {
    "environment": "cyan",
    "region": "yellow",
}
ROLE_COLOR_CYCLE = [ROLE_COLORS[cat] for cat in ROLE_CATEGORIES]  # ["cyan", "yellow"]


def create_annotation_widget(
    viewer: napari.Viewer,
    shapes_layer_name: str = "Annotations",
) -> Container:
    """
    Create annotation control widget using magicgui.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    shapes_layer_name : str, default="Annotations"
        Name of the Shapes layer to manage.

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
            "2. Press R → draw named regions (yellow)\n"
            "3. Click Done when finished\n"
            "\n"
            "─── DRAWING ───\n"
            "Click points to draw polygon, ENTER to finish\n"
            "\n"
            "─── SHORTCUTS ───\n"
            "E = environment mode\n"
            "R = region mode\n"
            "3 = move shape\n"
            "4 = edit vertices\n"
            "Delete = remove shape\n"
            "Ctrl+Z = undo"
        )
    )

    # Mode indicator shows current annotation type with action-oriented wording
    mode_indicator = Label(value="Drawing: ENVIRONMENT BOUNDARY (cyan polygon)")

    # Annotation count status
    annotation_status = Label(value="Annotations: 0 environment, 0 regions")

    role_selector = ComboBox(
        choices=["environment", "region"],
        value="environment",  # Default to environment first
        label="Type:",
    )

    name_input = LineEdit(value="arena", label="Shape Name:")

    # Shapes list for tracking annotations
    shapes_list = Select(
        choices=[],
        label="Annotations:",
        allow_multiple=False,
    )

    # Opacity control
    opacity_slider = FloatSlider(
        value=0.5,
        min=0.1,
        max=1.0,
        step=0.1,
        label="Opacity:",
    )

    apply_btn = PushButton(text="Apply Name to Selected")
    delete_btn = PushButton(text="Delete Selected")
    save_btn = PushButton(text="Done")

    # --- Helper Functions ---
    def update_mode_indicator(role: str):
        """Update mode indicator label with action-oriented wording."""
        if role == "environment":
            mode_indicator.value = "Drawing: ENVIRONMENT BOUNDARY (cyan polygon)"
        else:
            mode_indicator.value = "Drawing: NAMED REGION (yellow polygon)"

    def update_annotation_status():
        """Update annotation count display."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            annotation_status.value = "Annotations: 0 environment, 0 regions"
            return

        features = shapes.features
        env_count = sum(1 for r in features["role"] if str(r) == "environment")
        region_count = sum(1 for r in features["role"] if str(r) == "region")
        annotation_status.value = (
            f"Annotations: {env_count} environment, {region_count} regions"
        )

    def update_shapes_list():
        """Refresh the shapes list from layer data."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            shapes_list.choices = []
            return

        # Build list of shape descriptions
        features = shapes.features
        choices = []
        for i in range(len(shapes.data)):
            name = features["name"].iloc[i] if i < len(features) else f"shape_{i}"
            role = features["role"].iloc[i] if i < len(features) else "region"
            choices.append(f"{i}: {name} ({role})")
        shapes_list.choices = choices

    def select_shape_in_layer(idx: int):
        """Select a shape in the layer by index."""
        shapes = get_shapes()
        if shapes is not None and 0 <= idx < len(shapes.data):
            shapes.selected_data = {idx}

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

    @name_input.changed.connect
    def on_name_changed(name: str):
        """Update feature_defaults when name changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["name"] = name

    @shapes_list.changed.connect
    def on_shapes_list_selection(selection):
        """Select shape in layer when selected in list."""
        if not selection:
            return
        # Select widget returns a list even with allow_multiple=False
        if isinstance(selection, list):
            selection = selection[0] if selection else None
        if not selection:
            return
        # Extract index from "idx: name (role)" format
        try:
            idx = int(str(selection).split(":")[0])
            select_shape_in_layer(idx)
            # Update name input to match selected shape
            shapes = get_shapes()
            if shapes is not None and idx < len(shapes.features):
                name_input.value = shapes.features["name"].iloc[idx]
        except (ValueError, IndexError):
            pass

    @opacity_slider.changed.connect
    def on_opacity_changed(opacity: float):
        """Update shapes layer opacity."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.opacity = opacity

    @apply_btn.clicked.connect
    def apply_to_selected():
        """Apply current name to selected shapes."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        # Use DataFrame operations for safer feature updates
        features_df = shapes.features.copy()
        for idx in shapes.selected_data:
            if 0 <= idx < len(features_df):
                features_df.loc[idx, "name"] = name_input.value

        shapes.features = features_df
        shapes.refresh()
        update_shapes_list()

    @delete_btn.clicked.connect
    def delete_selected():
        """Delete selected shapes."""
        import pandas as pd

        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        # Get indices to keep (not selected)
        indices_to_delete = set(shapes.selected_data)
        indices_to_keep = [
            i for i in range(len(shapes.data)) if i not in indices_to_delete
        ]

        if not indices_to_keep:
            # Delete all shapes
            shapes.data = []
            shapes.features = pd.DataFrame(
                {
                    "role": pd.Categorical([], categories=ROLE_CATEGORIES),
                    "name": pd.Series([], dtype=str),
                }
            )
        else:
            # Keep only non-deleted shapes and features
            new_data = [shapes.data[i] for i in indices_to_keep]
            new_roles = [str(shapes.features["role"].iloc[i]) for i in indices_to_keep]
            new_names = [str(shapes.features["name"].iloc[i]) for i in indices_to_keep]

            shapes.data = new_data
            shapes.features = pd.DataFrame(
                {
                    "role": pd.Categorical(new_roles, categories=ROLE_CATEGORIES),
                    "name": pd.Series(new_names, dtype=str),
                }
            )
            # Update face colors
            face_colors = [ROLE_COLORS.get(r, "yellow") for r in new_roles]
            shapes.face_color = face_colors

        shapes.selected_data = set()
        shapes.refresh()
        update_shapes_list()

    @save_btn.clicked.connect
    def save_and_close():
        """Close viewer to return control to Python, with confirmation if empty."""
        shapes = get_shapes()
        if shapes is None or len(shapes.data) == 0:
            # Show confirmation dialog when no annotations exist
            from qtpy.QtWidgets import QMessageBox

            reply = QMessageBox.question(
                None,
                "No Annotations",
                "No annotations were drawn. Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        viewer.close()

    # --- Keyboard Shortcuts ---
    @viewer.bind_key("e")
    def set_environment_mode(viewer):
        """Set mode to draw environment boundary."""
        role_selector.value = "environment"
        name_input.value = "arena"
        update_mode_indicator("environment")
        # Status bar feedback for mode switch
        viewer.status = "Mode: ENVIRONMENT - draw boundary polygon (cyan)"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "environment"
            shapes.feature_defaults["name"] = "arena"
            shapes.current_face_color = ROLE_COLORS["environment"]

    @viewer.bind_key("r")
    def set_region_mode(viewer):
        """Set mode to draw named region."""
        role_selector.value = "region"
        # Clear name to prompt user to enter a meaningful name
        name_input.value = ""
        update_mode_indicator("region")
        # Status bar feedback for mode switch
        viewer.status = "Mode: REGION - enter a name, then draw polygon (yellow)"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "region"
            shapes.feature_defaults["name"] = ""
            shapes.current_face_color = ROLE_COLORS["region"]

    # --- Data Change Callback ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure

    def on_data_changed(event):
        """Handle shape additions: update list, auto-switch mode, update status."""
        import pandas as pd

        shapes = get_shapes()
        if shapes is None:
            return

        current_count = len(shapes.data)
        if current_count > _prev_shape_count[0]:
            # New shape was added - ensure features are set for the new shape
            # Napari's feature_defaults sometimes doesn't work properly

            # Build new features list from scratch to avoid concat issues
            features_len = len(shapes.features) if shapes.features is not None else 0
            roles = list(shapes.features["role"]) if features_len > 0 else []
            names = list(shapes.features["name"]) if features_len > 0 else []

            # Add entries for any new shapes
            while len(roles) < current_count:
                roles.append(role_selector.value)
                # Use current name, or generate fallback if empty
                current_name = name_input.value.strip()
                if not current_name:
                    # Fallback name if user didn't enter one
                    role = role_selector.value
                    if role == "environment":
                        current_name = "arena"
                    else:
                        region_count = sum(1 for r in roles if str(r) == "region")
                        current_name = f"region_{region_count}"
                names.append(current_name)

            # Create fresh DataFrame
            features_df = pd.DataFrame(
                {
                    "role": pd.Categorical(roles, categories=ROLE_CATEGORIES),
                    "name": pd.Series(names, dtype=str),
                }
            )
            shapes.features = features_df

            # Explicitly set face colors based on role (cycle doesn't work reliably)
            face_colors = [ROLE_COLORS.get(str(r), "yellow") for r in roles]
            shapes.face_color = face_colors

            update_shapes_list()
            update_annotation_status()

            # Check if we just drew an environment boundary - auto-switch to region mode
            last_role = str(features_df["role"].iloc[-1])
            if last_role == "environment":
                # Auto-switch to region mode for next shape
                role_selector.value = "region"
                update_mode_indicator("region")
                shapes.feature_defaults["role"] = "region"
                shapes.current_face_color = ROLE_COLORS["region"]
                # Status bar feedback for auto-switch
                viewer.status = "Switched to REGION mode - enter a name for each region"
                # Clear name to prompt user to enter meaningful region name
                new_name = ""
            else:
                # Clear name for next region (encourage meaningful naming)
                new_name = ""

            name_input.value = new_name
            shapes.feature_defaults["name"] = new_name

        elif current_count < _prev_shape_count[0]:
            # Shape was deleted externally (e.g., via Delete key)
            update_shapes_list()
            update_annotation_status()

        _prev_shape_count[0] = current_count

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        _prev_shape_count[0] = len(shapes.data)

    # --- Build Container ---
    widget = Container(
        widgets=[
            instructions,
            mode_indicator,
            annotation_status,
            role_selector,
            name_input,
            shapes_list,
            opacity_slider,
            apply_btn,
            delete_btn,
            save_btn,
        ],
        labels=False,
    )

    return widget


def setup_shapes_layer_for_annotation(viewer: napari.Viewer) -> napari.layers.Shapes:
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

    # Set defaults for new shapes - environment first
    shapes.feature_defaults["role"] = "environment"
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

    # Ensure lists match data length
    while len(names) < len(data):
        names.append(f"region_{len(names)}")
    while len(roles) < len(data):
        roles.append("region")

    return data, [str(n) for n in names], [str(r) for r in roles]
