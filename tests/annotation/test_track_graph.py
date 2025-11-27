"""Tests for track_graph module entry point.

This module tests the public API entry point for track graph annotation,
including TrackGraphResult and annotate_track_graph().

Task 3.1: Tests for TrackGraphResult import and to_environment method
Task 3.4: Tests for annotate_track_graph entry point (to be added)
"""

from __future__ import annotations

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Task 3.1: Tests for TrackGraphResult import from track_graph.py
# -----------------------------------------------------------------------------


class TestTrackGraphResultImport:
    """Tests for TrackGraphResult import from track_graph.py."""

    def test_import_from_track_graph_module(self) -> None:
        """TrackGraphResult should be importable from track_graph.py."""
        from neurospatial.annotation.track_graph import TrackGraphResult

        assert TrackGraphResult is not None

    def test_is_same_as_helpers_version(self) -> None:
        """TrackGraphResult should be the same class as in _track_helpers."""
        from neurospatial.annotation._track_helpers import (
            TrackGraphResult as HelpersResult,
        )
        from neurospatial.annotation.track_graph import TrackGraphResult

        # Should be the exact same class (re-export, not copy)
        assert TrackGraphResult is HelpersResult

    def test_has_expected_fields(self) -> None:
        """TrackGraphResult from track_graph.py should have all expected fields."""
        from neurospatial.annotation.track_graph import TrackGraphResult

        expected_fields = {
            "track_graph",
            "node_positions",
            "edges",
            "edge_order",
            "edge_spacing",
            "node_labels",
            "start_node",
            "pixel_positions",
        }

        assert set(TrackGraphResult._fields) == expected_fields


class TestTrackGraphResultToEnvironment:
    """Tests for TrackGraphResult.to_environment() method via track_graph.py import."""

    def test_to_environment_method_exists(self) -> None:
        """TrackGraphResult should have to_environment method."""
        from neurospatial.annotation.track_graph import TrackGraphResult

        assert hasattr(TrackGraphResult, "to_environment")

    def test_to_environment_raises_without_graph(self) -> None:
        """to_environment should raise ValueError if track_graph is None."""
        from neurospatial.annotation.track_graph import TrackGraphResult

        result = TrackGraphResult(
            track_graph=None,
            node_positions=[],
            edges=[],
            edge_order=[],
            edge_spacing=np.array([]),
            node_labels=[],
            start_node=None,
            pixel_positions=[],
        )

        with pytest.raises(ValueError, match="no track graph"):
            result.to_environment(bin_size=2.0)

    def test_to_environment_creates_valid_environment(self) -> None:
        """to_environment should create valid Environment from track_graph import."""
        pytest.importorskip("track_linearization")

        from track_linearization import infer_edge_layout

        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )
        from neurospatial.annotation.track_graph import TrackGraphResult

        # Create a simple track graph
        node_positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        edges = [(0, 1), (1, 2)]
        graph = build_track_graph_from_positions(node_positions, edges)
        edge_order, edge_spacing = infer_edge_layout(graph, start_node=0)

        result = TrackGraphResult(
            track_graph=graph,
            node_positions=node_positions,
            edges=edges,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            node_labels=["start", "middle", "end"],
            start_node=0,
            pixel_positions=node_positions,
        )

        env = result.to_environment(bin_size=2.0)

        # Verify environment was created
        assert env is not None
        assert env.n_bins > 0

    def test_to_environment_edge_spacing_override(self) -> None:
        """to_environment should allow edge_spacing override."""
        pytest.importorskip("track_linearization")

        from track_linearization import infer_edge_layout

        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )
        from neurospatial.annotation.track_graph import TrackGraphResult

        # Create a track graph with two edges
        node_positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        edges = [(0, 1), (1, 2)]
        graph = build_track_graph_from_positions(node_positions, edges)
        edge_order, edge_spacing = infer_edge_layout(graph, start_node=0)

        result = TrackGraphResult(
            track_graph=graph,
            node_positions=node_positions,
            edges=edges,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            node_labels=["a", "b", "c"],
            start_node=0,
            pixel_positions=node_positions,
        )

        # Should work with float override
        env1 = result.to_environment(bin_size=2.0, edge_spacing=5.0)
        assert env1 is not None

        # Should work with list override
        env2 = result.to_environment(bin_size=2.0, edge_spacing=[3.0])
        assert env2 is not None

    def test_to_environment_with_name(self) -> None:
        """to_environment should pass name to Environment."""
        pytest.importorskip("track_linearization")

        from track_linearization import infer_edge_layout

        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )
        from neurospatial.annotation.track_graph import TrackGraphResult

        # Create a simple track graph
        node_positions = [(0.0, 0.0), (10.0, 0.0)]
        edges = [(0, 1)]
        graph = build_track_graph_from_positions(node_positions, edges)
        edge_order, edge_spacing = infer_edge_layout(graph, start_node=0)

        result = TrackGraphResult(
            track_graph=graph,
            node_positions=node_positions,
            edges=edges,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            node_labels=[],
            start_node=0,
            pixel_positions=node_positions,
        )

        env = result.to_environment(bin_size=2.0, name="test_track")

        assert env.name == "test_track"
