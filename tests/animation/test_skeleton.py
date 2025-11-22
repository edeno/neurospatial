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


class TestSkeletonEdgeNormalization:
    """Test Skeleton edge canonicalization and deduplication.

    Edges are undirected for visualization purposes, so:
    - ("a", "b") and ("b", "a") should be treated as the same edge
    - Duplicates (including reversed duplicates) should be removed
    - Edges should be stored in canonical form: sorted lexicographically
    """

    def test_reversed_edge_is_canonicalized(self):
        """Test that reversed edges are converted to canonical form."""
        # ("b", "a") should become ("a", "b") since "a" < "b" lexicographically
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("b", "a"),),
        )
        assert skeleton.edges == (("a", "b"),)

    def test_already_canonical_edge_unchanged(self):
        """Test that already canonical edges are unchanged."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"),),
        )
        assert skeleton.edges == (("a", "b"),)

    def test_mixed_canonical_and_reversed_edges(self):
        """Test skeleton with mix of canonical and reversed edges."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("c", "b")),  # Second edge reversed
        )
        # Both edges should be in canonical form
        assert ("a", "b") in skeleton.edges
        assert ("b", "c") in skeleton.edges

    def test_duplicate_edges_deduplicated(self):
        """Test that exact duplicate edges are removed."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"), ("a", "b")),  # Exact duplicate
        )
        assert skeleton.edges == (("a", "b"),)
        assert skeleton.n_edges == 1

    def test_reversed_duplicate_edges_deduplicated(self):
        """Test that reversed duplicates are deduplicated."""
        # ("a", "b") and ("b", "a") are the same edge
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"), ("b", "a")),
        )
        assert skeleton.edges == (("a", "b"),)
        assert skeleton.n_edges == 1

    def test_multiple_duplicates_reduced_to_one(self):
        """Test that multiple duplicates (including reversed) become one edge."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b"),
            edges=(("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")),
        )
        assert skeleton.edges == (("a", "b"),)
        assert skeleton.n_edges == 1

    def test_edge_order_preserved_after_dedup(self):
        """Test that edge order is preserved based on first occurrence."""
        skeleton = Skeleton(
            name="test",
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("c", "b"), ("b", "a")),  # Last is dup of first
        )
        # Should have edges in order of first occurrence (canonicalized)
        assert skeleton.edges == (("a", "b"), ("b", "c"))

    def test_string_comparison_for_canonical_form(self):
        """Test that string comparison is used for canonical form."""
        # "nose" < "tail" lexicographically
        skeleton = Skeleton(
            name="test",
            nodes=("tail", "nose"),
            edges=(("tail", "nose"),),
        )
        assert skeleton.edges == (("nose", "tail"),)

    def test_case_sensitive_comparison(self):
        """Test that comparison is case-sensitive (uppercase < lowercase in ASCII)."""
        # 'A' (65) < 'a' (97) in ASCII
        skeleton = Skeleton(
            name="test",
            nodes=("A", "a"),
            edges=(("a", "A"),),  # "a" > "A"
        )
        assert skeleton.edges == (("A", "a"),)

    def test_self_loop_edges_preserved(self):
        """Test that self-loops (same start/end) are preserved."""
        skeleton = Skeleton(
            name="test",
            nodes=("a",),
            edges=(("a", "a"),),
        )
        assert skeleton.edges == (("a", "a"),)

    def test_duplicate_self_loops_deduplicated(self):
        """Test that duplicate self-loops are deduplicated."""
        skeleton = Skeleton(
            name="test",
            nodes=("a",),
            edges=(("a", "a"), ("a", "a")),
        )
        assert skeleton.edges == (("a", "a"),)
        assert skeleton.n_edges == 1

    def test_from_edge_list_normalizes_edges(self):
        """Test that from_edge_list also normalizes edges."""
        skeleton = Skeleton.from_edge_list(
            [("b", "a"), ("c", "b"), ("a", "b")],  # Reversed and dup
        )
        # Should be canonicalized and deduplicated
        assert ("a", "b") in skeleton.edges
        assert ("b", "c") in skeleton.edges
        assert skeleton.n_edges == 2

    def test_from_dict_normalizes_edges(self):
        """Test that from_dict also normalizes edges."""
        data = {
            "name": "test",
            "nodes": ["a", "b"],
            "edges": [["b", "a"], ["a", "b"]],  # Reversed and dup
        }
        skeleton = Skeleton.from_dict(data)
        assert skeleton.edges == (("a", "b"),)
        assert skeleton.n_edges == 1

    def test_complex_skeleton_normalization(self):
        """Test normalization on a more realistic skeleton."""
        skeleton = Skeleton(
            name="test",
            nodes=("nose", "neck", "body", "tail"),
            edges=(
                ("nose", "neck"),
                ("neck", "body"),  # Will be reversed to ("body", "neck")
                ("tail", "body"),  # Will be reversed to ("body", "tail")
                ("neck", "nose"),  # Duplicate of first edge (reversed)
            ),
        )
        # Should have 3 unique edges in canonical form
        # Canonical: lexicographic order: "body" < "neck" < "nose" < "tail"
        assert skeleton.n_edges == 3
        assert ("neck", "nose") in skeleton.edges  # "neck" < "nose"
        assert ("body", "neck") in skeleton.edges  # "body" < "neck"
        assert ("body", "tail") in skeleton.edges  # "body" < "tail"
