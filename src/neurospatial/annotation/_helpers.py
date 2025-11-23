"""Shared helpers for annotation module.

This module contains utilities shared between the widget and controller,
extracted to avoid circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from neurospatial.annotation._types import Role

if TYPE_CHECKING:
    import napari
    import pandas as pd

# Role categories - order determines color cycle mapping
# Environment first since users typically define boundary first
# Derived from Role type alias for single source of truth
ROLE_CATEGORIES: list[Role] = list(get_args(Role))

# Color scheme for role-based visualization
# Use str keys for runtime flexibility (values come from pandas DataFrames as str)
ROLE_COLORS: dict[str, str] = {
    "environment": "cyan",
    "hole": "red",
    "region": "yellow",
}
ROLE_COLOR_CYCLE = [ROLE_COLORS[cat] for cat in ROLE_CATEGORIES]


def rebuild_features(roles: list[Role], names: list[str]) -> pd.DataFrame:
    """
    Create a fresh features DataFrame with proper categorical types.

    Centralizes feature DataFrame construction to ensure consistency
    across all shape update operations.

    Parameters
    ----------
    roles : list of Role
        Role for each shape ("environment", "hole", or "region").
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
