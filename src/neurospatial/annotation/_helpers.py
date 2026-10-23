"""Shared helpers for annotation module.

This module contains utilities shared between the widget and controller,
extracted to avoid circular dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, get_args

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
    "wall": "magenta",
    "landmark": "orange",
}
# Walls and landmarks are often paths, which have no face: they show their
# type by their line color. Areas keep a white outline over their fill.
LINE_COLORED_TYPES = frozenset({"wall", "landmark"})
REGION_TYPE_COLOR_CYCLE = [REGION_TYPE_COLORS[cat] for cat in REGION_TYPE_CATEGORIES]


def rebuild_features(region_types: list[RegionType], names: list[str]) -> pd.DataFrame:
    """Create a fresh features DataFrame with proper categorical types.

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
        },
    )


def shapes_with_types(
    shapes_layer: napari.layers.Shapes, indices: Iterable[int]
) -> list[tuple[Any, str]]:
    """The shapes at ``indices``, each as ``(vertices, shape_type)``.

    Assign the result to ``shapes_layer.data`` to remove or reorder shapes.
    Given vertices alone, napari keeps the layer's old shape types in their
    old order, so a deleted or moved shape would pass its type on: a path
    wall after a deleted region would become that region's polygon.
    """
    return [(shapes_layer.data[i], shapes_layer.shape_type[i]) for i in indices]


def select_shapes(shapes_layer: napari.layers.Shapes, indices: Sequence[int]) -> None:
    """Keep only the shapes at ``indices``, in that order, with their features.

    Each kept shape keeps its type, feature row and role color.
    """
    features = shapes_layer.features.iloc[list(indices)].reset_index(drop=True)
    shapes_layer.data = shapes_with_types(shapes_layer, indices)
    shapes_layer.features = features
    sync_colors_from_features(shapes_layer)


def sync_colors_from_features(shapes_layer: napari.layers.Shapes) -> None:
    """Update face and edge colors to match feature roles.

    Napari's face_color_cycle doesn't always work reliably when features
    are updated programmatically. This function explicitly syncs colors.
    Walls and landmarks also take their role's color for their edges, so a
    path shows what it is.

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
    shapes_layer.edge_color = [
        REGION_TYPE_COLORS[str(r)] if str(r) in LINE_COLORED_TYPES else "white"
        for r in region_types
    ]
