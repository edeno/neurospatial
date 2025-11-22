"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton

if TYPE_CHECKING:
    import napari

# Role categories - order determines color cycle mapping
ROLE_CATEGORIES = ["region", "environment"]

# Color scheme for role-based visualization
# Order must match ROLE_CATEGORIES
ROLE_COLORS = {
    "region": "yellow",
    "environment": "cyan",
}
ROLE_COLOR_CYCLE = [ROLE_COLORS[cat] for cat in ROLE_CATEGORIES]  # ["yellow", "cyan"]


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
            "Draw polygons: E=environment, R=region\n"
            "Edit: 3=move shape, 4=edit vertices, Delete=remove\n"
            "Click shape to select, then Apply to change metadata"
        )
    )

    role_selector = ComboBox(
        choices=["region", "environment"],
        value="region",
        label="New shape role:",
    )

    name_input = LineEdit(value="region_0", label="New shape name:")

    apply_btn = PushButton(text="Apply to Selected")
    save_btn = PushButton(text="Save and Close")

    # --- Event Handlers ---
    @role_selector.changed.connect
    def on_role_changed(role: str):
        """Update feature_defaults when role selector changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["role"] = role
            # Update colors for visual feedback
            shapes.current_face_color = ROLE_COLORS.get(role, "yellow")

    @name_input.changed.connect
    def on_name_changed(name: str):
        """Update feature_defaults when name changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["name"] = name

    @apply_btn.clicked.connect
    def apply_to_selected():
        """Apply current role/name to selected shapes."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        # Use DataFrame operations for safer feature updates
        features_df = shapes.features.copy()
        for idx in shapes.selected_data:
            if 0 <= idx < len(features_df):
                features_df.loc[idx, "role"] = role_selector.value
                features_df.loc[idx, "name"] = name_input.value

        shapes.features = features_df
        shapes.refresh()

    @save_btn.clicked.connect
    def save_and_close():
        """Close viewer to return control to Python."""
        viewer.close()

    # --- Keyboard Shortcuts ---
    @viewer.bind_key("e")
    def set_environment_mode(viewer):
        """Set mode to draw environment boundary."""
        role_selector.value = "environment"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "environment"
            shapes.current_face_color = ROLE_COLORS["environment"]

    @viewer.bind_key("r")
    def set_region_mode(viewer):
        """Set mode to draw named region."""
        role_selector.value = "region"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "region"
            shapes.current_face_color = ROLE_COLORS["region"]

    # --- Data Change Callback for Auto-increment ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure

    def on_data_changed(event):
        """Auto-increment name when a new shape is added."""
        shapes = get_shapes()
        if shapes is None:
            return

        current_count = len(shapes.data)
        if current_count > _prev_shape_count[0]:
            # New shape was added - update name for next shape
            new_name = f"region_{current_count}"
            name_input.value = new_name
            shapes.feature_defaults["name"] = new_name
        _prev_shape_count[0] = current_count

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        _prev_shape_count[0] = len(shapes.data)

    # --- Build Container ---
    widget = Container(
        widgets=[
            instructions,
            role_selector,
            name_input,
            apply_btn,
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

    # Set defaults for new shapes
    shapes.feature_defaults["role"] = "region"
    shapes.feature_defaults["name"] = "region_0"
    shapes.current_face_color = ROLE_COLORS["region"]

    # Start in polygon drawing mode
    shapes.mode = "add_polygon"

    return shapes


def get_annotation_data(
    shapes_layer: napari.layers.Shapes,
) -> tuple[list, list[str], list[str]]:
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
