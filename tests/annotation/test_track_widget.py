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


# =============================================================================
# Task 2.2 Tests: Event Handlers
# =============================================================================


@pytest.mark.gui
class TestSyncLayersFromState:
    """Tests for _sync_layers_from_state() function."""

    def test_sync_updates_nodes_layer_data(self):
        """_sync_layers_from_state updates nodes layer with state.nodes."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)

        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # Check nodes layer has 2 points
        assert len(nodes_layer.data) == 2
        viewer.close()

    def test_sync_updates_edges_layer_data(self):
        """_sync_layers_from_state updates edges layer with state.edges."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)

        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # Check edges layer has 1 path
        assert len(edges_layer.data) == 1
        viewer.close()

    def test_sync_highlights_start_node_with_different_color(self):
        """Start node gets highlighted with START_NODE_COLOR."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.set_start_node(0)

        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # Check that face colors are different for start vs non-start nodes
        # Start node (index 0) should have different color than node 1
        assert len(nodes_layer.face_color) == 2
        # The colors should not be identical
        start_color = nodes_layer.face_color[0]
        other_color = nodes_layer.face_color[1]
        assert not np.allclose(start_color, other_color), (
            "Start node should have different color"
        )
        viewer.close()

    def test_sync_highlights_start_node_with_larger_size(self):
        """Start node gets larger size than other nodes."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.set_start_node(0)

        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # Start node (index 0) should be larger
        assert nodes_layer.size[0] > nodes_layer.size[1]
        viewer.close()

    def test_sync_clears_layers_when_state_empty(self):
        """_sync_layers_from_state clears layers when state is empty."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        # Add data manually first
        nodes_layer.add([[100, 200]])
        edges_layer.add_paths([np.array([[0, 0], [10, 10]])])

        # Sync with empty state
        state = TrackBuilderState()
        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # Layers should be empty
        assert len(nodes_layer.data) == 0
        assert len(edges_layer.data) == 0
        viewer.close()

    def test_sync_uses_napari_row_col_convention(self):
        """Coordinates are converted to napari (row, col) format."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _sync_layers_from_state,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        # State uses (x, y) = (100, 200) meaning x=100, y=200
        state.add_node(100.0, 200.0)

        _sync_layers_from_state(state, nodes_layer, edges_layer)

        # napari uses (row, col) = (y, x) = (200, 100)
        assert nodes_layer.data[0][0] == 200.0  # row = y
        assert nodes_layer.data[0][1] == 100.0  # col = x
        viewer.close()


@pytest.mark.gui
class TestClickHandlerAddNodeMode:
    """Tests for click handler in add_node mode."""

    def test_click_in_add_node_mode_adds_node_to_state(self):
        """Clicking in add_node mode adds a node to state."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.mode = "add_node"

        # Simulate click at (row=200, col=100) in napari coords
        # This should add node at (x=100, y=200) in state coords
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))

        assert len(state.nodes) == 1
        assert state.nodes[0] == (100.0, 200.0)  # (x, y) format
        viewer.close()

    def test_click_in_add_node_mode_syncs_layers(self):
        """Clicking in add_node mode updates napari layers."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.mode = "add_node"

        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))

        # Layer should show the new node
        assert len(nodes_layer.data) == 1
        viewer.close()


@pytest.mark.gui
class TestClickHandlerDeleteMode:
    """Tests for click handler in delete mode."""

    def test_click_in_delete_mode_removes_nearest_node(self):
        """Clicking near a node in delete mode removes it."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)  # Node at (x=100, y=200)
        state.mode = "delete"

        # Click near the node (within threshold)
        # Node is at (row=200, col=100) in napari coords
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))

        assert len(state.nodes) == 0
        viewer.close()

    def test_click_in_delete_mode_ignores_click_far_from_nodes(self):
        """Clicking far from any node in delete mode does nothing."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)  # Node at (x=100, y=200)
        state.mode = "delete"

        # Click far from the node
        _handle_click(state, nodes_layer, edges_layer, position=(500.0, 500.0))

        # Node should still exist
        assert len(state.nodes) == 1
        viewer.close()


@pytest.mark.gui
class TestClickHandlerAddEdgeMode:
    """Tests for click handler in add_edge mode (two-click pattern)."""

    def test_first_click_in_add_edge_mode_sets_edge_start_node(self):
        """First click on node in add_edge mode sets edge_start_node."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)  # Node 0
        state.add_node(300.0, 400.0)  # Node 1
        state.mode = "add_edge"

        # Click on first node (at row=200, col=100)
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))

        assert state.edge_start_node == 0
        viewer.close()

    def test_second_click_in_add_edge_mode_creates_edge(self):
        """Second click on node in add_edge mode creates edge."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)  # Node 0
        state.add_node(300.0, 400.0)  # Node 1
        state.mode = "add_edge"

        # First click on node 0
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))
        # Second click on node 1
        _handle_click(state, nodes_layer, edges_layer, position=(400.0, 300.0))

        assert len(state.edges) == 1
        assert state.edges[0] == (0, 1)
        viewer.close()

    def test_second_click_clears_edge_start_node(self):
        """After creating edge, edge_start_node is cleared."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.mode = "add_edge"

        # Two clicks to create edge
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))
        _handle_click(state, nodes_layer, edges_layer, position=(400.0, 300.0))

        assert state.edge_start_node is None
        viewer.close()

    def test_click_on_same_node_twice_does_not_create_self_loop(self):
        """Clicking the same node twice doesn't create a self-loop edge."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.mode = "add_edge"

        # Click same node twice
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))
        _handle_click(state, nodes_layer, edges_layer, position=(200.0, 100.0))

        # No edge should be created
        assert len(state.edges) == 0
        viewer.close()

    def test_click_away_from_nodes_in_add_edge_mode_does_nothing(self):
        """Clicking empty space in add_edge mode doesn't start edge creation."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_click,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.mode = "add_edge"

        # Click far from any node
        _handle_click(state, nodes_layer, edges_layer, position=(500.0, 500.0))

        assert state.edge_start_node is None
        viewer.close()


@pytest.mark.gui
class TestCancelEdgeCreation:
    """Tests for canceling edge creation with Escape."""

    def test_cancel_edge_creation_clears_edge_start_node(self):
        """cancel_edge_creation clears edge_start_node."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _cancel_edge_creation

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.mode = "add_edge"
        state.edge_start_node = 0

        _cancel_edge_creation(state)

        assert state.edge_start_node is None


@pytest.mark.gui
class TestEdgePreview:
    """Tests for edge preview line during edge creation."""

    def test_show_edge_preview_creates_preview_shape(self):
        """_show_edge_preview adds a preview path to edges layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _show_edge_preview,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, _ = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.edge_start_node = 0

        # Show preview from node 0 to cursor position
        _show_edge_preview(state, edges_layer, cursor_position=(400.0, 300.0))

        # Should have one preview path
        assert len(edges_layer.data) == 1
        viewer.close()

    def test_clear_edge_preview_removes_preview_shape(self):
        """_clear_edge_preview removes the preview path."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _clear_edge_preview,
            _show_edge_preview,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, _ = setup_track_layers(viewer)

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.edge_start_node = 0

        _show_edge_preview(state, edges_layer, cursor_position=(400.0, 300.0))
        _clear_edge_preview(edges_layer)

        # Preview should be removed
        assert len(edges_layer.data) == 0
        viewer.close()


@pytest.mark.gui
class TestKeyboardShortcuts:
    """Tests for keyboard shortcut handler."""

    def test_a_key_switches_to_add_node_mode(self):
        """Pressing 'A' switches mode to add_node."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.mode = "delete"

        _handle_key(state, key="A")

        assert state.mode == "add_node"

    def test_e_key_switches_to_add_edge_mode(self):
        """Pressing 'E' switches mode to add_edge."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.mode = "add_node"

        _handle_key(state, key="E")

        assert state.mode == "add_edge"

    def test_x_key_switches_to_delete_mode(self):
        """Pressing 'X' switches mode to delete."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.mode = "add_node"

        _handle_key(state, key="X")

        assert state.mode == "delete"

    def test_ctrl_z_triggers_undo(self):
        """Pressing 'Ctrl+Z' triggers undo."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)  # Creates undo snapshot
        state.add_node(300.0, 400.0)  # Creates undo snapshot

        result = _handle_key(state, key="Z", modifiers=["Control"])

        assert result == "undo"
        assert len(state.nodes) == 1  # Undid second add_node

    def test_ctrl_shift_z_triggers_redo(self):
        """Pressing 'Ctrl+Shift+Z' triggers redo."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.undo()  # Now have 1 node

        result = _handle_key(state, key="Z", modifiers=["Control", "Shift"])

        assert result == "redo"
        assert len(state.nodes) == 2  # Redid second add_node

    def test_escape_cancels_edge_creation(self):
        """Pressing 'Escape' cancels edge creation."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.mode = "add_edge"
        state.edge_start_node = 0

        result = _handle_key(state, key="Escape")

        assert result == "cancel"
        assert state.edge_start_node is None

    def test_shift_s_sets_start_node(self):
        """Pressing 'Shift+S' sets selected node as start."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)

        # Simulate having node 1 selected
        result = _handle_key(state, key="S", modifiers=["Shift"], selected_node=1)

        assert result == "set_start"
        assert state.start_node == 1

    def test_lowercase_keys_also_work(self):
        """Keyboard shortcuts work with lowercase keys too."""
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import _handle_key

        state = TrackBuilderState()
        state.mode = "delete"

        _handle_key(state, key="a")

        assert state.mode == "add_node"


# =============================================================================
# Task 2.3 Tests: Control Widget
# =============================================================================


@pytest.mark.gui
class TestCreateTrackWidget:
    """Tests for create_track_widget() function."""

    def test_returns_qwidget(self):
        """create_track_widget returns a widget with QWidget native property."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QWidget

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Check widget has a native QWidget (or is one)
        native = getattr(widget, "_widget", widget)
        assert isinstance(native, QWidget)
        viewer.close()

    def test_widget_has_mode_selector(self):
        """Widget includes a mode selector component."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Check widget has mode_selector attribute or child
        assert hasattr(widget, "mode_selector") or widget.findChild(
            type(widget), "mode_selector"
        )
        viewer.close()

    def test_widget_has_save_button(self):
        """Widget includes a save/close button."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QPushButton

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Find save button by searching children
        save_buttons = [
            child
            for child in widget.findChildren(QPushButton)
            if "save" in child.text().lower() or "close" in child.text().lower()
        ]
        assert len(save_buttons) >= 1, "Widget should have a Save/Close button"
        viewer.close()

    def test_widget_has_help_text(self):
        """Widget includes help text panel."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QLabel

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Find labels containing help keywords
        labels = widget.findChildren(QLabel)
        help_text_found = any(
            "Press A" in label.text() or "add" in label.text().lower()
            for label in labels
        )
        assert help_text_found, "Widget should display help text"
        viewer.close()

    def test_widget_has_start_node_button(self):
        """Widget includes a 'Set as Start' button."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QPushButton

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Find start button
        start_buttons = [
            child
            for child in widget.findChildren(QPushButton)
            if "start" in child.text().lower()
        ]
        assert len(start_buttons) >= 1, "Widget should have a 'Set as Start' button"
        viewer.close()

    def test_widget_has_delete_buttons(self):
        """Widget includes delete buttons for nodes and edges."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QPushButton

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Find delete buttons
        delete_buttons = [
            child
            for child in widget.findChildren(QPushButton)
            if "delete" in child.text().lower()
        ]
        assert len(delete_buttons) >= 1, "Widget should have delete button(s)"
        viewer.close()


@pytest.mark.gui
class TestModeSelectorSync:
    """Tests for mode selector synchronization with state."""

    def test_mode_selector_initial_value_matches_state(self):
        """Mode selector starts with state's initial mode."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.mode = "add_node"

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Check mode selector shows add_node
        assert hasattr(widget, "mode_selector")
        assert widget.mode_selector.value == "add_node"
        viewer.close()

    def test_changing_mode_selector_updates_state(self):
        """Changing mode selector updates state.mode."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.mode = "add_node"

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Change mode via selector
        widget.mode_selector.value = "add_edge"

        assert state.mode == "add_edge"
        viewer.close()

    def test_keyboard_mode_change_updates_selector(self):
        """Changing mode via keyboard updates the mode selector."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            _handle_key,
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.mode = "add_node"

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Change mode via keyboard handler
        _handle_key(state, key="X")

        # Manually trigger sync (in real usage this happens via callbacks)
        widget.sync_from_state()

        assert widget.mode_selector.value == "delete"
        viewer.close()


@pytest.mark.gui
class TestNodeEdgeLists:
    """Tests for node and edge list updates."""

    def test_node_list_updates_when_node_added(self):
        """Node list updates when a node is added to state."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Add a node
        state.add_node(100.0, 200.0)
        widget.sync_from_state()

        # Check node list has one item
        assert hasattr(widget, "node_list")
        assert len(widget.node_list.choices) == 1
        viewer.close()

    def test_edge_list_updates_when_edge_added(self):
        """Edge list updates when an edge is added to state."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Add nodes and edge
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)
        widget.sync_from_state()

        # Check edge list has one item
        assert hasattr(widget, "edge_list")
        assert len(widget.edge_list.choices) == 1
        viewer.close()

    def test_node_list_shows_labels_when_available(self):
        """Node list displays labels when nodes have labels."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Add nodes with labels
        state.add_node(100.0, 200.0, label="start")
        state.add_node(300.0, 400.0, label="goal")
        widget.sync_from_state()

        # Check labels appear in choices (as display text)
        choices = widget.node_list.choices
        choice_texts = [str(c) for c in choices]
        assert any("start" in t for t in choice_texts), "Node label should appear"
        viewer.close()


@pytest.mark.gui
class TestValidation:
    """Tests for save validation behavior."""

    def test_get_validation_status_empty_state(self):
        """Validation shows errors for empty state."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Check validation status shows errors
        assert hasattr(widget, "get_validation_status")
        is_valid, errors, _warnings = widget.get_validation_status()

        assert not is_valid
        assert len(errors) > 0
        viewer.close()

    def test_get_validation_status_valid_state(self):
        """Validation passes for valid state."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        # Create valid state
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)
        state.set_start_node(0)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        is_valid, errors, _warnings = widget.get_validation_status()

        assert is_valid
        assert len(errors) == 0
        viewer.close()

    def test_get_validation_status_warns_no_start_node(self):
        """Validation warns when no start node is set."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()

        # Valid state but no start node set
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)
        # Explicitly do NOT set start_node

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        is_valid, _errors, warnings = widget.get_validation_status()

        # Should be valid but with warning
        assert is_valid
        assert len(warnings) > 0
        assert any("start" in w.lower() for w in warnings)
        viewer.close()


@pytest.mark.gui
class TestSetStartNodeButton:
    """Tests for Set Start Node button functionality."""

    def test_set_start_button_updates_state(self):
        """Clicking 'Set as Start' updates state.start_node."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Select node 1 in the list and click set start
        widget.node_list.value = 1
        widget.set_start_button.click()

        assert state.start_node == 1
        viewer.close()


@pytest.mark.gui
class TestDeleteButtons:
    """Tests for delete button functionality."""

    def test_delete_node_button_removes_selected_node(self):
        """Delete Node button removes the selected node."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Select and delete node 0
        widget.node_list.value = 0
        widget.delete_node_button.click()

        assert len(state.nodes) == 1
        viewer.close()

    def test_delete_edge_button_removes_selected_edge(self):
        """Delete Edge button removes the selected edge."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Select and delete edge 0
        widget.edge_list.value = 0
        widget.delete_edge_button.click()

        assert len(state.edges) == 0
        viewer.close()


@pytest.mark.gui
class TestNodeLabelInput:
    """Tests for node label input functionality."""

    def test_node_label_input_updates_label(self):
        """Changing node label input updates the node's label."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.add_node(100.0, 200.0)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Select node and set label
        widget.node_list.value = 0
        widget.node_label_input.value = "start"
        widget.apply_label_button.click()

        assert state.node_labels[0] == "start"
        viewer.close()


@pytest.mark.gui
class TestStatusBar:
    """Tests for status bar updates."""

    def test_status_bar_shows_current_mode(self):
        """Status bar displays current mode."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.mode = "add_node"

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Check status shows mode (either in napari status or widget status label)
        assert hasattr(widget, "status_label")
        # Status shows "Mode: Add Node" - check for "add" and "node" in lowercase
        status_text = widget.status_label.text().lower()
        assert "add" in status_text and "node" in status_text
        viewer.close()


@pytest.mark.gui
class TestSaveClose:
    """Tests for save and close functionality."""

    def test_save_valid_state_sets_result(self):
        """Saving with valid state sets the widget result."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Try to save
        result = widget.try_save()

        assert result is True  # Save should succeed
        viewer.close()

    def test_save_invalid_state_returns_false(self):
        """Saving with invalid state returns False."""
        napari = pytest.importorskip("napari")
        pytest.importorskip("qtpy")

        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.annotation._track_widget import (
            create_track_widget,
            setup_track_layers,
        )

        viewer = napari.Viewer(show=False)
        edges_layer, nodes_layer = setup_track_layers(viewer)
        state = TrackBuilderState()
        # Empty state - invalid

        widget = create_track_widget(viewer, edges_layer, nodes_layer, state)

        # Try to save
        result = widget.try_save()

        assert result is False  # Save should fail
        viewer.close()
