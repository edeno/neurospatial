"""State management for track graph annotation.

This module provides TrackBuilderState, a pure state object for managing
track graph construction (nodes, edges, labels) with undo/redo support.
The state is napari-independent and fully testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx

from neurospatial.annotation._track_types import TrackGraphMode


@dataclass
class TrackBuilderState:
    """Track graph builder state.

    A pure state object managing nodes, edges, and labels for track graph
    construction. Includes undo/redo functionality with snapshot-based history.

    Attributes
    ----------
    mode : TrackGraphMode
        Current interaction mode: "add_node", "add_edge", or "delete".
    nodes : list[tuple[float, float]]
        List of node positions as (x, y) tuples.
    edges : list[tuple[int, int]]
        List of edges as (node1_idx, node2_idx) tuples.
    node_labels : list[str]
        Labels for each node (empty string if unlabeled).
    start_node : int or None
        Designated start node for linearization (used by infer_edge_layout).
    edge_start_node : int or None
        Transient state for two-click edge creation (first clicked node).
    undo_stack : list[dict]
        Stack of state snapshots for undo operations.
    redo_stack : list[dict]
        Stack of state snapshots for redo operations.

    Examples
    --------
    >>> state = TrackBuilderState()
    >>> idx0 = state.add_node(0.0, 0.0, label="start")
    >>> idx1 = state.add_node(10.0, 10.0, label="end")
    >>> state.add_edge(idx0, idx1)
    True
    >>> state.set_start_node(0)
    >>> is_valid, errors, warnings = state.is_valid_for_save()
    >>> is_valid
    True
    """

    mode: TrackGraphMode = "add_node"
    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)

    # Linearization control
    start_node: int | None = None

    # Edge creation state (transient for two-click pattern)
    edge_start_node: int | None = None

    # Undo/redo stacks
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)
    _max_undo_depth: int = 50

    def _snapshot(self) -> dict[str, Any]:
        """Create serializable snapshot of mutable state.

        Returns
        -------
        dict
            Snapshot containing nodes, edges, node_labels, and start_node.
            Uses explicit deep copies to ensure independence from live state.
        """
        return {
            "nodes": [tuple(n) for n in self.nodes],
            "edges": [tuple(e) for e in self.edges],
            "node_labels": list(self.node_labels),
            "start_node": self.start_node,
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot.

        Parameters
        ----------
        snapshot : dict
            Snapshot created by _snapshot().
        """
        self.nodes = [tuple(n) for n in snapshot["nodes"]]
        self.edges = [tuple(e) for e in snapshot["edges"]]
        self.node_labels = list(snapshot["node_labels"])
        self.start_node = snapshot["start_node"]

    def _save_for_undo(self) -> None:
        """Save current state before mutation.

        Clears redo stack since new action invalidates future history.
        Enforces max undo depth limit.
        """
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()
        # Limit stack depth
        if len(self.undo_stack) > self._max_undo_depth:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore previous state.

        Returns
        -------
        bool
            True if undo was possible, False if undo stack was empty.
        """
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._snapshot())
        self._restore_snapshot(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        """Restore next state (after undo).

        Returns
        -------
        bool
            True if redo was possible, False if redo stack was empty.
        """
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._snapshot())
        self._restore_snapshot(self.redo_stack.pop())
        return True

    def add_node(self, x: float, y: float, label: str | None = None) -> int:
        """Add a node at the specified position.

        Parameters
        ----------
        x : float
            X coordinate of the node.
        y : float
            Y coordinate of the node.
        label : str, optional
            Label for the node (e.g., "start", "goal", "junction").

        Returns
        -------
        int
            Index of the newly added node.
        """
        self._save_for_undo()
        self.nodes.append((x, y))
        self.node_labels.append(label if label is not None else "")
        return len(self.nodes) - 1

    def delete_node(self, idx: int) -> None:
        """Delete a node and all connected edges.

        Reindexes remaining nodes and updates edge references.
        Also updates start_node if affected by the deletion.

        Parameters
        ----------
        idx : int
            Index of the node to delete.

        Raises
        ------
        IndexError
            If idx is out of range.
        """
        if idx < 0 or idx >= len(self.nodes):
            raise IndexError(f"Node index {idx} out of range")

        self._save_for_undo()

        # Remove node and its label
        self.nodes.pop(idx)
        self.node_labels.pop(idx)

        # Remove edges connected to deleted node and reindex remaining edges
        new_edges = []
        for n1, n2 in self.edges:
            if n1 == idx or n2 == idx:
                # Edge connected to deleted node - remove it
                continue
            # Reindex: decrement indices > deleted index
            new_n1 = n1 - 1 if n1 > idx else n1
            new_n2 = n2 - 1 if n2 > idx else n2
            new_edges.append((new_n1, new_n2))
        self.edges = new_edges

        # Update start_node if affected
        if self.start_node is not None:
            if self.start_node == idx:
                self.start_node = None
            elif self.start_node > idx:
                self.start_node -= 1

    def set_start_node(self, idx: int) -> None:
        """Designate a node as the start node for linearization.

        The start node determines the origin for track linearization algorithms
        (used by track_linearization.infer_edge_layout).

        Parameters
        ----------
        idx : int
            Index of the node to set as start.

        Raises
        ------
        IndexError
            If idx is out of range.
        """
        if idx < 0 or idx >= len(self.nodes):
            raise IndexError(f"Node index {idx} out of range")

        self._save_for_undo()
        self.start_node = idx

    def find_nearest_node(self, x: float, y: float, threshold: float) -> int | None:
        """Find the nearest node within threshold distance.

        Parameters
        ----------
        x : float
            X coordinate of query point.
        y : float
            Y coordinate of query point.
        threshold : float
            Maximum distance to consider a node "near".

        Returns
        -------
        int or None
            Index of nearest node within threshold, or None if no node found.
        """
        if not self.nodes:
            return None

        min_dist = float("inf")
        nearest_idx = None

        for i, (node_x, node_y) in enumerate(self.nodes):
            dist = math.sqrt((x - node_x) ** 2 + (y - node_y) ** 2)
            if dist <= threshold and dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def add_edge(self, node1: int, node2: int) -> bool:
        """Add an edge between two nodes.

        Validates that the edge is not a self-loop, not a duplicate,
        and both node indices are valid.

        Parameters
        ----------
        node1 : int
            Index of the first node.
        node2 : int
            Index of the second node.

        Returns
        -------
        bool
            True if edge was added successfully, False otherwise.
        """
        # Validate node indices
        if node1 < 0 or node1 >= len(self.nodes):
            return False
        if node2 < 0 or node2 >= len(self.nodes):
            return False

        # Reject self-loops
        if node1 == node2:
            return False

        # Check for duplicate (including reversed)
        edge = (node1, node2)
        edge_rev = (node2, node1)
        if edge in self.edges or edge_rev in self.edges:
            return False

        self._save_for_undo()
        self.edges.append(edge)
        return True

    def delete_edge(self, edge_idx: int) -> None:
        """Delete an edge by index.

        Parameters
        ----------
        edge_idx : int
            Index of the edge to delete.

        Raises
        ------
        IndexError
            If edge_idx is out of range.
        """
        if edge_idx < 0 or edge_idx >= len(self.edges):
            raise IndexError(f"Edge index {edge_idx} out of range")

        self._save_for_undo()
        self.edges.pop(edge_idx)

    def to_track_graph(self) -> nx.Graph:
        """Build NetworkX graph from current state.

        Creates a graph with node 'pos' attributes set to node coordinates.
        For internal validation only; final output uses transformed coordinates.

        Returns
        -------
        nx.Graph
            Graph with nodes having 'pos' attribute.
        """
        graph = nx.Graph()

        # Add nodes with positions
        for i, (x, y) in enumerate(self.nodes):
            graph.add_node(i, pos=(x, y))

        # Add edges
        for n1, n2 in self.edges:
            graph.add_edge(n1, n2)

        return graph

    def validate(self) -> dict[str, Any]:
        """Validate graph using track_linearization.check_track_graph_validity.

        Builds temporary graph in pixel coordinates for validation.
        If track_linearization is not installed, performs basic validation
        (at least 2 nodes and 1 edge).

        Returns
        -------
        dict
            Validation report with keys 'valid', 'errors', 'warnings'.

        Notes
        -----
        Full validation requires the track_linearization package. Without it,
        only basic structural checks are performed (node/edge count).
        """
        try:
            from track_linearization import check_track_graph_validity

            graph = self.to_track_graph()
            return cast("dict[str, Any]", check_track_graph_validity(graph))
        except ImportError:
            # Return basic validation if track_linearization not available
            return {
                "valid": len(self.nodes) >= 2 and len(self.edges) >= 1,
                "errors": [],
                "warnings": [],
            }

    def is_valid_for_save(self) -> tuple[bool, list[str], list[str]]:
        """Check if state is valid for saving.

        Returns
        -------
        tuple[bool, list[str], list[str]]
            (is_valid, errors, warnings) where:
            - is_valid: True if state can be saved
            - errors: List of blocking error messages
            - warnings: List of non-blocking warning messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        if len(self.nodes) < 2:
            errors.append("Need at least 2 nodes")
        if len(self.edges) < 1:
            errors.append("Need at least 1 edge")
        if self.start_node is None and len(self.nodes) > 0:
            warnings.append("No start node set. Defaulting to Node 0.")

        return len(errors) == 0, errors, warnings

    def get_effective_start_node(self) -> int | None:
        """Get start node, defaulting to 0 if not explicitly set.

        This method implements the fallback logic used during graph export:
        if no start node has been explicitly set via set_start_node(),
        defaults to node 0 (the first created node, assumed track origin).

        Returns
        -------
        int or None
            start_node if set, else 0 if nodes exist, else None.
        """
        if self.start_node is not None:
            return self.start_node
        if len(self.nodes) > 0:
            return 0
        return None
