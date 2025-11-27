"""Tests for TrackBuilderState - track graph annotation state management."""

from __future__ import annotations

import pytest


class TestTrackBuilderStateImport:
    """Tests for importing TrackBuilderState."""

    def test_import(self) -> None:
        """TrackBuilderState should be importable from module."""
        from neurospatial.annotation._track_state import TrackBuilderState

        assert TrackBuilderState is not None


class TestTrackBuilderStateInitialization:
    """Tests for TrackBuilderState initialization."""

    def test_default_state(self) -> None:
        """Default state has expected initial values."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        assert state.mode == "add_node"
        assert state.nodes == []
        assert state.edges == []
        assert state.node_labels == []
        assert state.start_node is None
        assert state.edge_start_node is None
        assert state.undo_stack == []
        assert state.redo_stack == []

    def test_custom_initialization(self) -> None:
        """State can be initialized with custom values."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState(
            mode="add_edge",
            nodes=[(0.0, 0.0), (10.0, 10.0)],
            edges=[(0, 1)],
            node_labels=["start", "end"],
            start_node=0,
        )

        assert state.mode == "add_edge"
        assert len(state.nodes) == 2
        assert len(state.edges) == 1
        assert state.node_labels == ["start", "end"]
        assert state.start_node == 0


class TestNodeOperations:
    """Tests for node add/delete operations."""

    def test_add_node_returns_index(self) -> None:
        """add_node returns the index of the new node."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        idx0 = state.add_node(0.0, 0.0)
        idx1 = state.add_node(10.0, 10.0)
        idx2 = state.add_node(20.0, 20.0)

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2
        assert len(state.nodes) == 3

    def test_add_node_stores_position(self) -> None:
        """add_node stores the correct position."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        state.add_node(5.5, 10.2)

        assert state.nodes[0] == (5.5, 10.2)

    def test_add_node_with_label(self) -> None:
        """add_node with label stores the label."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        state.add_node(0.0, 0.0, label="start")
        state.add_node(10.0, 10.0, label="goal")
        state.add_node(20.0, 20.0)  # No label

        assert state.node_labels[0] == "start"
        assert state.node_labels[1] == "goal"
        assert state.node_labels[2] == ""  # Default empty string

    def test_delete_node_removes_node(self) -> None:
        """delete_node removes the node from the list."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)

        state.delete_node(1)

        assert len(state.nodes) == 2
        assert state.nodes[0] == (0.0, 0.0)
        assert state.nodes[1] == (20.0, 20.0)

    def test_delete_node_removes_connected_edges(self) -> None:
        """delete_node removes all edges connected to the deleted node."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)
        state.add_edge(0, 1)  # Edge to be removed (connected to node 1)
        state.add_edge(1, 2)  # Edge to be removed (connected to node 1)
        state.add_edge(0, 2)  # Edge to be kept

        state.delete_node(1)

        # Only edge (0, 2) remains, but reindexed to (0, 1)
        assert len(state.edges) == 1
        assert state.edges[0] == (0, 1)

    def test_delete_node_reindexes_remaining_edges(self) -> None:
        """delete_node reindexes edges after node removal."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)  # Index 0
        state.add_node(10.0, 10.0)  # Index 1
        state.add_node(20.0, 20.0)  # Index 2
        state.add_node(30.0, 30.0)  # Index 3
        state.add_edge(2, 3)  # Edge between nodes 2 and 3

        # Delete node 0 - nodes 2,3 become 1,2
        state.delete_node(0)

        assert state.edges[0] == (1, 2)

    def test_delete_node_updates_start_node_when_deleted(self) -> None:
        """delete_node sets start_node to None if the start node is deleted."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.set_start_node(0)

        state.delete_node(0)

        assert state.start_node is None

    def test_delete_node_decrements_start_node_when_earlier_deleted(self) -> None:
        """delete_node decrements start_node if earlier node is deleted."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)  # Index 0
        state.add_node(10.0, 10.0)  # Index 1
        state.add_node(20.0, 20.0)  # Index 2
        state.set_start_node(2)

        state.delete_node(0)  # Node 2 becomes node 1

        assert state.start_node == 1

    def test_delete_node_removes_label(self) -> None:
        """delete_node removes the label for the deleted node."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0, label="first")
        state.add_node(10.0, 10.0, label="second")
        state.add_node(20.0, 20.0, label="third")

        state.delete_node(1)

        assert len(state.node_labels) == 2
        assert state.node_labels[0] == "first"
        assert state.node_labels[1] == "third"

    def test_find_nearest_node_within_threshold(self) -> None:
        """find_nearest_node returns index when node is within threshold."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)

        result = state.find_nearest_node(9.5, 10.5, threshold=2.0)

        assert result == 1

    def test_find_nearest_node_outside_threshold_returns_none(self) -> None:
        """find_nearest_node returns None when no node within threshold."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        result = state.find_nearest_node(100.0, 100.0, threshold=5.0)

        assert result is None

    def test_find_nearest_node_returns_closest(self) -> None:
        """find_nearest_node returns the closest node when multiple within threshold."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)  # Distance = sqrt(2) ≈ 1.41
        state.add_node(2.0, 2.0)  # Distance = sqrt(2) ≈ 1.41, but from (1,1)
        state.add_node(1.0, 0.0)  # Distance = 1.0 from (1,1)

        result = state.find_nearest_node(1.0, 1.0, threshold=5.0)

        # Node at (0,0) is closest (distance ≈ 1.41)
        # Actually let me recalculate: from (1,1):
        # - (0,0): sqrt(1+1) = sqrt(2) ≈ 1.41
        # - (2,2): sqrt(1+1) = sqrt(2) ≈ 1.41
        # - (1,0): sqrt(0+1) = 1.0
        # So node 2 at (1,0) is closest
        assert result == 2

    def test_find_nearest_node_empty_state(self) -> None:
        """find_nearest_node returns None when no nodes exist."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        result = state.find_nearest_node(0.0, 0.0, threshold=10.0)

        assert result is None

    def test_set_start_node(self) -> None:
        """set_start_node designates a node as start."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        state.set_start_node(1)

        assert state.start_node == 1

    def test_set_start_node_invalid_index(self) -> None:
        """set_start_node raises error for invalid index."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)

        with pytest.raises(IndexError):
            state.set_start_node(5)


class TestEdgeOperations:
    """Tests for edge add/delete operations."""

    def test_add_edge_success(self) -> None:
        """add_edge successfully adds a valid edge."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        result = state.add_edge(0, 1)

        assert result is True
        assert len(state.edges) == 1
        assert state.edges[0] == (0, 1)

    def test_add_edge_rejects_self_loop(self) -> None:
        """add_edge rejects edges where node1 == node2."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)

        result = state.add_edge(0, 0)

        assert result is False
        assert len(state.edges) == 0

    def test_add_edge_rejects_duplicate(self) -> None:
        """add_edge rejects duplicate edges."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)

        result = state.add_edge(0, 1)

        assert result is False
        assert len(state.edges) == 1

    def test_add_edge_rejects_duplicate_reversed(self) -> None:
        """add_edge rejects edges that are reversed duplicates."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)

        result = state.add_edge(1, 0)  # Reversed

        assert result is False
        assert len(state.edges) == 1

    def test_add_edge_invalid_node_index(self) -> None:
        """add_edge returns False for invalid node indices."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)

        result = state.add_edge(0, 5)

        assert result is False
        assert len(state.edges) == 0

    def test_delete_edge(self) -> None:
        """delete_edge removes edge by index."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)
        state.add_edge(0, 1)
        state.add_edge(1, 2)

        state.delete_edge(0)

        assert len(state.edges) == 1
        assert state.edges[0] == (1, 2)

    def test_delete_edge_invalid_index(self) -> None:
        """delete_edge raises error for invalid index."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        with pytest.raises(IndexError):
            state.delete_edge(0)


class TestUndoRedo:
    """Tests for undo/redo functionality."""

    def test_undo_restores_previous_state(self) -> None:
        """undo restores the state before the last action."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)  # This will be undone

        result = state.undo()

        assert result is True
        assert len(state.nodes) == 1
        assert state.nodes[0] == (0.0, 0.0)

    def test_redo_restores_next_state(self) -> None:
        """redo restores the state after undo."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.undo()

        result = state.redo()

        assert result is True
        assert len(state.nodes) == 2
        assert state.nodes[1] == (10.0, 10.0)

    def test_undo_empty_stack_returns_false(self) -> None:
        """undo returns False when no history exists."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        result = state.undo()

        assert result is False

    def test_redo_empty_stack_returns_false(self) -> None:
        """redo returns False when no redo history exists."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)

        result = state.redo()

        assert result is False

    def test_new_action_clears_redo_stack(self) -> None:
        """A new action after undo clears the redo stack."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.undo()  # Now we have redo available
        state.add_node(20.0, 20.0)  # New action clears redo

        result = state.redo()

        assert result is False

    def test_undo_stack_depth_limit(self) -> None:
        """Undo stack respects max depth limit."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state._max_undo_depth = 5

        # Add more nodes than the limit
        for i in range(10):
            state.add_node(float(i), float(i))

        # Undo stack should only have max_depth items
        assert len(state.undo_stack) <= 5

    def test_undo_edge_operation(self) -> None:
        """Undo works for edge operations."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)

        state.undo()

        assert len(state.edges) == 0

    def test_undo_delete_node(self) -> None:
        """Undo restores deleted node and connected edges."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)
        state.delete_node(1)

        state.undo()

        assert len(state.nodes) == 2
        assert len(state.edges) == 1
        assert state.edges[0] == (0, 1)

    def test_undo_set_start_node(self) -> None:
        """Undo restores previous start_node value."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.set_start_node(0)
        state.set_start_node(1)

        state.undo()

        assert state.start_node == 0

    def test_multiple_undo_redo_cycle(self) -> None:
        """Multiple undo/redo operations work correctly."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)

        state.undo()  # Remove node 2
        state.undo()  # Remove node 1

        assert len(state.nodes) == 1

        state.redo()  # Add back node 1

        assert len(state.nodes) == 2
        assert state.nodes[1] == (10.0, 10.0)


class TestValidation:
    """Tests for validation methods."""

    def test_is_valid_for_save_requires_nodes(self) -> None:
        """is_valid_for_save returns error when fewer than 2 nodes."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)  # Only 1 node

        is_valid, errors, _warnings = state.is_valid_for_save()

        assert is_valid is False
        assert "Need at least 2 nodes" in errors

    def test_is_valid_for_save_requires_edges(self) -> None:
        """is_valid_for_save returns error when no edges."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        # No edges added

        is_valid, errors, _warnings = state.is_valid_for_save()

        assert is_valid is False
        assert "Need at least 1 edge" in errors

    def test_is_valid_for_save_warns_no_start_node(self) -> None:
        """is_valid_for_save warns when no start node set."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)
        # No start_node set

        is_valid, errors, warnings = state.is_valid_for_save()

        assert is_valid is True  # Valid, but with warning
        assert len(errors) == 0
        assert any("start node" in w.lower() for w in warnings)

    def test_is_valid_for_save_success(self) -> None:
        """is_valid_for_save returns True for valid state."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)
        state.set_start_node(0)

        is_valid, errors, warnings = state.is_valid_for_save()

        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_get_effective_start_node_returns_set_value(self) -> None:
        """get_effective_start_node returns the set value if set."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.set_start_node(1)

        assert state.get_effective_start_node() == 1

    def test_get_effective_start_node_defaults_to_zero(self) -> None:
        """get_effective_start_node defaults to 0 if not set."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        assert state.get_effective_start_node() == 0

    def test_get_effective_start_node_empty_state(self) -> None:
        """get_effective_start_node returns None for empty state."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        assert state.get_effective_start_node() is None


class TestToTrackGraph:
    """Tests for to_track_graph method."""

    def test_to_track_graph_has_pos_attributes(self) -> None:
        """to_track_graph creates graph with 'pos' node attributes."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)

        graph = state.to_track_graph()

        assert graph.nodes[0]["pos"] == (0.0, 0.0)
        assert graph.nodes[1]["pos"] == (10.0, 10.0)

    def test_to_track_graph_has_edges(self) -> None:
        """to_track_graph creates graph with correct edges."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_node(20.0, 20.0)
        state.add_edge(0, 1)
        state.add_edge(1, 2)

        graph = state.to_track_graph()

        assert graph.has_edge(0, 1)
        assert graph.has_edge(1, 2)
        assert not graph.has_edge(0, 2)

    def test_to_track_graph_empty_state(self) -> None:
        """to_track_graph returns empty graph for empty state."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        graph = state.to_track_graph()

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0


class TestSnapshot:
    """Tests for snapshot/restore functionality."""

    def test_snapshot_captures_state(self) -> None:
        """_snapshot captures current state."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)
        state.add_edge(0, 1)
        state.node_labels[0] = "start"
        state.set_start_node(0)

        snapshot = state._snapshot()

        assert snapshot["nodes"] == [(0.0, 0.0), (10.0, 10.0)]
        assert snapshot["edges"] == [(0, 1)]
        assert snapshot["node_labels"] == ["start", ""]
        assert snapshot["start_node"] == 0

    def test_restore_snapshot_restores_state(self) -> None:
        """_restore_snapshot restores state from snapshot."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        snapshot = state._snapshot()

        # Modify state
        state.add_node(20.0, 20.0)
        state.add_edge(0, 1)

        # Restore
        state._restore_snapshot(snapshot)

        assert len(state.nodes) == 2
        assert len(state.edges) == 0

    def test_snapshot_is_deep_copy(self) -> None:
        """_snapshot creates independent copy (modifying state doesn't affect snapshot)."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)

        snapshot = state._snapshot()

        # Modify original state (without triggering undo save)
        state.nodes.append((10.0, 10.0))

        # Snapshot should be unchanged
        assert len(snapshot["nodes"]) == 1


class TestModeHandling:
    """Tests for mode state handling."""

    def test_initial_mode_is_add_node(self) -> None:
        """Initial mode is 'add_node'."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        assert state.mode == "add_node"

    def test_mode_can_be_changed(self) -> None:
        """Mode can be changed to any valid value."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()

        state.mode = "add_edge"
        assert state.mode == "add_edge"

        state.mode = "delete"
        assert state.mode == "delete"

        state.mode = "add_node"
        assert state.mode == "add_node"

    def test_edge_start_node_for_two_click_pattern(self) -> None:
        """edge_start_node tracks first click in edge creation."""
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 10.0)

        # Simulate first click
        state.edge_start_node = 0

        assert state.edge_start_node == 0

        # After edge creation, should be cleared
        state.edge_start_node = None

        assert state.edge_start_node is None
