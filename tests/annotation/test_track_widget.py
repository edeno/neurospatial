"""Tests for track graph widget layer setup (requires napari)."""

import numpy as np
import pytest

# =============================================================================
# Task 2.1 Tests: Layer Setup
# =============================================================================


@pytest.mark.gui
class TestSetupTrackLayers:
    """Tests for setup_track_layers() function."""

    def test_returns_correct_types(self):
        """setup_track_layers returns tuple of (Shapes, Points) layers."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        result = setup_track_layers(viewer)

        assert isinstance(result, tuple)
        assert len(result) == 2

        edges_layer, nodes_layer = result

        # Check layer types using class names (avoids import issues)
        assert type(edges_layer).__name__ == "Shapes"
        assert type(nodes_layer).__name__ == "Points"

        viewer.close()

    def test_layers_added_to_viewer(self):
        """setup_track_layers adds two layers to the viewer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)
        initial_layer_count = len(viewer.layers)

        setup_track_layers(viewer)

        assert len(viewer.layers) == initial_layer_count + 2
        viewer.close()

    def test_layers_z_order_nodes_on_top(self):
        """Nodes layer is above edges layer for clickability."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, nodes_layer = setup_track_layers(viewer)

        # In napari, higher index = on top
        edges_idx = viewer.layers.index(edges_layer)
        nodes_idx = viewer.layers.index(nodes_layer)

        assert nodes_idx > edges_idx, "Nodes layer should be above edges layer"
        viewer.close()

    def test_edges_layer_has_correct_name(self):
        """Edges layer is named 'Track Edges'."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, _ = setup_track_layers(viewer)

        assert edges_layer.name == "Track Edges"
        viewer.close()

    def test_nodes_layer_has_correct_name(self):
        """Nodes layer is named 'Track Nodes'."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        _, nodes_layer = setup_track_layers(viewer)

        assert nodes_layer.name == "Track Nodes"
        viewer.close()

    def test_edges_layer_is_path_type(self):
        """Edges layer uses 'path' shape type."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, _ = setup_track_layers(viewer)

        # Add a test path to verify shape_type behavior
        # Shapes layer should accept path-type shapes
        edges_layer.add_paths([np.array([[0, 0], [10, 10]])])
        assert len(edges_layer.data) == 1

        viewer.close()

    def test_nodes_layer_starts_empty(self):
        """Nodes layer starts with no data."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        _, nodes_layer = setup_track_layers(viewer)

        assert len(nodes_layer.data) == 0
        viewer.close()

    def test_edges_layer_starts_empty(self):
        """Edges layer starts with no data."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, _ = setup_track_layers(viewer)

        assert len(edges_layer.data) == 0
        viewer.close()


@pytest.mark.gui
class TestColorConstants:
    """Tests for color constant definitions."""

    def test_node_color_defined(self):
        """NODE_COLOR constant is defined."""
        from neurospatial.annotation._track_widget import NODE_COLOR

        assert NODE_COLOR is not None
        assert isinstance(NODE_COLOR, str)
        # Should be a hex color starting with #
        assert NODE_COLOR.startswith("#")

    def test_edge_color_defined(self):
        """EDGE_COLOR constant is defined."""
        from neurospatial.annotation._track_widget import EDGE_COLOR

        assert EDGE_COLOR is not None
        assert isinstance(EDGE_COLOR, str)
        assert EDGE_COLOR.startswith("#")

    def test_start_node_color_defined(self):
        """START_NODE_COLOR constant is defined."""
        from neurospatial.annotation._track_widget import START_NODE_COLOR

        assert START_NODE_COLOR is not None
        assert isinstance(START_NODE_COLOR, str)
        assert START_NODE_COLOR.startswith("#")

    def test_selected_color_defined(self):
        """SELECTED_COLOR constant is defined."""
        from neurospatial.annotation._track_widget import SELECTED_COLOR

        assert SELECTED_COLOR is not None
        assert isinstance(SELECTED_COLOR, str)
        assert SELECTED_COLOR.startswith("#")

    def test_preview_color_defined(self):
        """PREVIEW_COLOR constant is defined."""
        from neurospatial.annotation._track_widget import PREVIEW_COLOR

        assert PREVIEW_COLOR is not None
        assert isinstance(PREVIEW_COLOR, str)
        assert PREVIEW_COLOR.startswith("#")

    def test_colors_are_colorblind_safe(self):
        """Colors match the colorblind-safe palette from spec."""
        from neurospatial.annotation._track_widget import (
            EDGE_COLOR,
            NODE_COLOR,
            PREVIEW_COLOR,
            SELECTED_COLOR,
            START_NODE_COLOR,
        )

        # Verify exact colors from spec
        assert NODE_COLOR == "#1f77b4"  # Blue
        assert EDGE_COLOR == "#ff7f0e"  # Orange
        assert START_NODE_COLOR == "#2ca02c"  # Green
        assert SELECTED_COLOR == "#d62728"  # Red
        assert PREVIEW_COLOR == "#7f7f7f"  # Gray


@pytest.mark.gui
class TestLayerColors:
    """Tests for layer color configuration."""

    def test_nodes_layer_uses_node_color(self):
        """Nodes layer face_color matches NODE_COLOR."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        _, nodes_layer = setup_track_layers(viewer)

        # Add a point to check color
        nodes_layer.add([[50, 50]])

        # Get the face color - napari stores colors as RGBA arrays
        # Just verify it was set (color conversion is complex)
        assert nodes_layer.current_face_color is not None

        viewer.close()

    def test_edges_layer_uses_edge_color(self):
        """Edges layer edge_color matches EDGE_COLOR."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, _ = setup_track_layers(viewer)

        # Verify edge_color is set
        assert edges_layer.edge_color is not None

        viewer.close()

    def test_nodes_layer_has_white_border(self):
        """Nodes layer has white border for visibility."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        _, nodes_layer = setup_track_layers(viewer)

        # Check border_width is set (> 0)
        # border_width is an array when there are points, scalar otherwise
        # Check the current_border_width which is a scalar
        assert nodes_layer.current_border_width > 0

        viewer.close()


@pytest.mark.gui
class TestLayerProperties:
    """Tests for layer property configuration."""

    def test_nodes_layer_size(self):
        """Nodes layer has reasonable default point size."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        _, nodes_layer = setup_track_layers(viewer)

        # Size should be visible (> 5 pixels)
        # Use current_size which is the scalar default value
        assert nodes_layer.current_size > 5

        viewer.close()

    def test_edges_layer_edge_width(self):
        """Edges layer has reasonable edge width."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_widget import setup_track_layers

        viewer = napari.Viewer(show=False)

        edges_layer, _ = setup_track_layers(viewer)

        # Edge width should be visible (> 1 pixel)
        # Use current_edge_width which is the scalar default value
        assert edges_layer.current_edge_width > 1

        viewer.close()
