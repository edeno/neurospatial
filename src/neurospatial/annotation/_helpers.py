"""Shared helpers for annotation module.

This module contains utilities shared between the widget and controller,
extracted to avoid circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from neurospatial.annotation._types import RegionType

if TYPE_CHECKING:
    import napari
    import pandas as pd

# RegionType categories - order determines color cycle mapping
# Environment first since users typically define boundary first
# Derived from RegionType type alias for single source of truth
REGION_TYPE_CATEGORIES: list[RegionType] = list(get_args(RegionType))

# Color scheme for region type visualization
# Use str keys for runtime flexibility (values come from pandas DataFrames as str)
#
# Accessibility note: The current cyan/red/yellow scheme may be difficult for
# users with red-green colorblindness. However, the text labels (shown in the
# widget and on shapes) provide additional identification. Future versions may
# add configurable color schemes or patterns for better accessibility.
REGION_TYPE_COLORS: dict[str, str] = {
    "environment": "cyan",
    "hole": "red",
    "region": "yellow",
}
REGION_TYPE_COLOR_CYCLE = [REGION_TYPE_COLORS[cat] for cat in REGION_TYPE_CATEGORIES]


def rebuild_features(region_types: list[RegionType], names: list[str]) -> pd.DataFrame:
    """
    Create a fresh features DataFrame with proper categorical types.

    Centralizes feature DataFrame construction to ensure consistency
    across all shape update operations.

    Parameters
    ----------
    region_types : list of RegionType
        Type for each shape ("environment", "hole", or "region").
    names : list of str
        Name for each shape.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical 'region_type' (stored as 'role') and string 'name' columns.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "role": pd.Categorical(region_types, categories=REGION_TYPE_CATEGORIES),
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

    region_types = shapes_layer.features.get("role", [])
    face_colors = [REGION_TYPE_COLORS.get(str(r), "yellow") for r in region_types]
    shapes_layer.face_color = face_colors
