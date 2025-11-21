"""Tests for the Skeleton dataclass and factory methods.

This module tests the Skeleton class including:
- Basic creation and validation
- Factory methods (from_edge_list, from_ndx_pose, from_movement)
- Serialization (to_dict, from_dict)
- Common presets (MOUSE_SKELETON, RAT_SKELETON, SIMPLE_SKELETON)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neurospatial.animation.skeleton import (
    MOUSE_SKELETON,
    RAT_SKELETON,
    SIMPLE_SKELETON,
    Skeleton,
)


class TestSkeletonBasicCreation:
    """Test Skeleton dataclass basic creation."""

    def test_basic_creation(self):
        """Test creating a Skeleton with required fields."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c")),
        )
        assert skeleton.name == "test"
        assert skeleton.nodes == ("a", "b", "c")
        assert skeleton.edges == (("a", "b"), ("b", "c"))
        assert skeleton.edge_color == "white"
        assert skeleton.edge_width == 1.0
        assert skeleton.node_colors is None

    def test_creation_with_styling(self):
        """Test creating a Skeleton with custom styling."""
        skeleton = Skeleton(
            name="styled",
            nodes=("a", "b"),
            edges=(("a", "b"),),
            edge_color="red",
            edge_width=3.0,
            node_colors={"a": "yellow", "b": "blue"},
        )
        assert skeleton.edge_color == "red"
        assert skeleton.edge_width == 3.0
        assert skeleton.node_colors == {"a": "yellow", "b": "blue"}

    def test_n_nodes_property(self):
        """Test n_nodes property."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"),),
        )
        assert skeleton.n_nodes == 3

    def test_n_edges_property(self):
        """Test n_edges property."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c")),
        )
        assert skeleton.n_edges == 2

    def test_frozen_immutability(self):
        """Test that Skeleton is immutable (frozen)."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"),),
        )
        with pytest.raises(FrozenInstanceError):
            skeleton.name = "modified"


class TestSkeletonValidation:
    """Test Skeleton validation in __post_init__."""

    def test_empty_nodes_raises(self):
        """Test that empty nodes raises ValueError."""
        with pytest.raises(ValueError, match="at least one node"):
            Skeleton(name="test", nodes=(), edges=())

    def test_duplicate_nodes_raises(self):
        """Test that duplicate nodes raises ValueError."""
        with pytest.raises(ValueError, match="Duplicate node names"):
            Skeleton(name="test", nodes=("a", "a", "b"), edges=())

    def test_invalid_edge_node_raises(self):
        """Test that edges referencing unknown nodes raises ValueError."""
        with pytest.raises(ValueError, match="unknown node"):
            Skeleton(
                name="test",
                nodes=("a", "b"),
                edges=(("a", "c"),),  # "c" not in nodes
            )

    def test_invalid_node_colors_raises(self):
        """Test that node_colors with unknown keys raises ValueError."""
        with pytest.raises(ValueError, match="unknown nodes"):
            Skeleton(
                name="test",
                nodes=("a", "b"),
                edges=(("a", "b"),),
                node_colors={"a": "red", "c": "blue"},  # "c" not in nodes
            )


class TestSkeletonFromEdgeList:
    """Test Skeleton.from_edge_list() factory method."""

    def test_infers_nodes_from_edges(self):
        """Test that nodes are inferred from edge list."""
        skeleton = Skeleton.from_edge_list([("a", "b"), ("b", "c")])
        assert skeleton.nodes == ("a", "b", "c")  # Alphabetically sorted

    def test_respects_node_order(self):
        """Test that explicit node_order is respected."""
        skeleton = Skeleton.from_edge_list(
            [("a", "b")],
            node_order=["b", "a"],
        )
        assert skeleton.nodes == ("b", "a")

    def test_missing_node_in_order_raises(self):
        """Test that missing node in node_order raises ValueError."""
        with pytest.raises(ValueError, match="missing nodes"):
            Skeleton.from_edge_list([("a", "b")], node_order=["a"])

    def test_custom_name(self):
        """Test that custom name is used."""
        skeleton = Skeleton.from_edge_list(
            [("a", "b")],
            name="custom_name",
        )
        assert skeleton.name == "custom_name"

    def test_passes_styling_kwargs(self):
        """Test that styling kwargs are passed through."""
        skeleton = Skeleton.from_edge_list(
            [("a", "b")],
            edge_color="red",
            edge_width=3.0,
        )
        assert skeleton.edge_color == "red"
        assert skeleton.edge_width == 3.0


class TestSkeletonSerialization:
    """Test Skeleton serialization methods."""

    def test_to_dict(self):
        """Test to_dict() produces correct structure."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"),),
            edge_color="red",
            edge_width=2.0,
        )
        d = skeleton.to_dict()

        assert d["name"] == "test"
        assert d["nodes"] == ["a", "b"]
        assert d["edges"] == [["a", "b"]]
        assert d["edge_color"] == "red"
        assert d["edge_width"] == 2.0

    def test_from_dict(self):
        """Test from_dict() reconstructs Skeleton."""
        data = {
            "name": "test",
            "nodes": ["a", "b"],
            "edges": [["a", "b"]],
            "edge_color": "blue",
            "edge_width": 3.0,
        }
        skeleton = Skeleton.from_dict(data)

        assert skeleton.name == "test"
        assert skeleton.nodes == ("a", "b")
        assert skeleton.edges == (("a", "b"),)
        assert skeleton.edge_color == "blue"
        assert skeleton.edge_width == 3.0

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip preserves data."""
        original = Skeleton(
            name="roundtrip",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
            edge_color="white",
            edge_width=1.5,
            node_colors={"head": "red"},
            metadata={"species": "mouse"},
        )
        d = original.to_dict()
        restored = Skeleton.from_dict(d)

        assert restored.name == original.name
        assert restored.nodes == original.nodes
        assert restored.edges == original.edges
        assert restored.edge_color == original.edge_color
        assert restored.edge_width == original.edge_width
        assert restored.node_colors == original.node_colors
        assert restored.metadata == original.metadata


class TestSkeletonWithColors:
    """Test Skeleton.with_colors() method."""

    def test_creates_copy_with_new_edge_color(self):
        """Test that with_colors creates a new Skeleton with updated edge_color."""
        original = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"),),
            edge_color="white",
        )
        modified = original.with_colors(edge_color="red")

        assert modified.edge_color == "red"
        assert original.edge_color == "white"  # Original unchanged

    def test_creates_copy_with_new_edge_width(self):
        """Test that with_colors creates a new Skeleton with updated edge_width."""
        original = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"),),
            edge_width=1.0,
        )
        modified = original.with_colors(edge_width=3.0)

        assert modified.edge_width == 3.0
        assert original.edge_width == 1.0  # Original unchanged

    def test_preserves_structure(self):
        """Test that with_colors preserves skeleton structure."""
        original = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
        )
        modified = original.with_colors(edge_color="blue")

        assert modified.name == original.name
        assert modified.nodes == original.nodes
        assert modified.edges == original.edges


class TestSkeletonPresets:
    """Test predefined skeleton presets."""

    def test_mouse_skeleton_valid(self):
        """Test MOUSE_SKELETON is valid and has correct structure."""
        assert MOUSE_SKELETON.name == "mouse"
        assert MOUSE_SKELETON.n_nodes == 8
        assert MOUSE_SKELETON.n_edges == 7
        assert "nose" in MOUSE_SKELETON.nodes
        assert "tail_tip" in MOUSE_SKELETON.nodes

    def test_rat_skeleton_valid(self):
        """Test RAT_SKELETON is valid and has correct structure."""
        assert RAT_SKELETON.name == "rat"
        assert RAT_SKELETON.n_nodes == 8
        assert RAT_SKELETON.n_edges == 7
        assert "nose" in RAT_SKELETON.nodes
        assert "tail_tip" in RAT_SKELETON.nodes

    def test_simple_skeleton_valid(self):
        """Test SIMPLE_SKELETON is valid and has correct structure."""
        assert SIMPLE_SKELETON.name == "simple"
        assert SIMPLE_SKELETON.n_nodes == 3
        assert SIMPLE_SKELETON.n_edges == 2
        assert SIMPLE_SKELETON.nodes == ("nose", "body", "tail")

    def test_presets_are_frozen(self):
        """Test that presets cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            MOUSE_SKELETON.name = "modified"

        with pytest.raises(FrozenInstanceError):
            RAT_SKELETON.edge_color = "red"

        with pytest.raises(FrozenInstanceError):
            SIMPLE_SKELETON.edge_width = 5.0


class TestSkeletonStr:
    """Test Skeleton string representation."""

    def test_str_format(self):
        """Test __str__ returns formatted string."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c")),
        )
        s = str(skeleton)
        assert "Skeleton" in s
        assert "'test'" in s
        assert "3 nodes" in s
        assert "2 edges" in s
