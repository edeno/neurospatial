"""Track graph builder widget and layer setup for napari.

This module provides napari layers and widgets for interactively building
track graphs on video frames. The track graph can be used with
`Environment.from_graph()` to create 1D linearized track environments.

Notes
-----
This module requires napari to be installed. All napari-dependent code
is imported lazily to allow the rest of the annotation module to work
without napari.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from napari.layers import Points, Shapes


# =============================================================================
# Colorblind-Safe Color Palette
# =============================================================================
# Based on Tab10 colormap, verified for colorblind accessibility
# See: https://jfly.uni-koeln.de/color/

NODE_COLOR: str = "#1f77b4"
"""Blue - Default color for track nodes."""

EDGE_COLOR: str = "#ff7f0e"
"""Orange - Color for track edges."""

START_NODE_COLOR: str = "#2ca02c"
"""Green - Color for the designated start node."""

SELECTED_COLOR: str = "#d62728"
"""Red - Highlight color for selected items."""

PREVIEW_COLOR: str = "#7f7f7f"
"""Gray - Color for edge preview line (dashed)."""


# =============================================================================
# Layer Setup
# =============================================================================


def setup_track_layers(viewer: napari.Viewer) -> tuple[Shapes, Points]:
    """Create napari layers for track graph annotation.

    Creates two layers in the viewer:
    1. Shapes layer for track edges (middle z-order)
    2. Points layer for track nodes (top z-order, interactive)

    The layers are ordered so that nodes are on top and clickable,
    while edges appear below for visual clarity.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to add layers to.

    Returns
    -------
    edges_layer : napari.layers.Shapes
        Shapes layer for drawing track edges as paths.
    nodes_layer : napari.layers.Points
        Points layer for placing track nodes.

    Examples
    --------
    >>> import napari
    >>> viewer = napari.Viewer()  # doctest: +SKIP
    >>> edges, nodes = setup_track_layers(viewer)  # doctest: +SKIP
    >>> nodes.add([[100, 100], [200, 150]])  # Add waypoints  # doctest: +SKIP
    >>> edges.add_paths([[[100, 100], [200, 150]]])  # Connect them  # doctest: +SKIP
    """
    # Edges layer (middle - below nodes for clickability)
    # Uses Shapes with path type for line segments between nodes
    edges_layer = viewer.add_shapes(
        name="Track Edges",
        shape_type="path",
        edge_color=EDGE_COLOR,
        edge_width=3,
    )

    # Nodes layer (top - interactive, clickable)
    # Points layer for track waypoints
    # Use border_width_is_relative=False for absolute pixel border widths
    nodes_layer = viewer.add_points(
        name="Track Nodes",
        size=15,
        face_color=NODE_COLOR,
        border_color="white",
        border_width=2,
        border_width_is_relative=False,
    )

    return edges_layer, nodes_layer
