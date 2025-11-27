"""Tests for track graph building helpers."""

from __future__ import annotations

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Tests for transform_nodes_to_output
# -----------------------------------------------------------------------------


class TestTransformNodesToOutput:
    """Tests for transform_nodes_to_output function."""

    def test_import(self) -> None:
        """Function should be importable from module."""
        from neurospatial.annotation._track_helpers import transform_nodes_to_output

        assert transform_nodes_to_output is not None

    def test_without_calibration_returns_pixels(self) -> None:
        """Without calibration, should return pixel coordinates unchanged."""
        from neurospatial.annotation._track_helpers import transform_nodes_to_output

        nodes_px = [(100.0, 200.0), (300.0, 400.0)]
        result = transform_nodes_to_output(nodes_px, calibration=None)

        assert result == nodes_px

    def test_with_calibration_transforms_to_cm(self) -> None:
        """With calibration, should apply pixel-to-cm transform."""
        from neurospatial.annotation._track_helpers import transform_nodes_to_output
        from neurospatial.transforms import (
            VideoCalibration,
            calibrate_from_scale_bar,
        )

        # Create calibration: 100 pixels = 50 cm, so cm_per_px = 0.5
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        calibration = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        nodes_px = [(0.0, 480.0), (100.0, 480.0)]  # Bottom of frame (y=480)
        result = transform_nodes_to_output(nodes_px, calibration)

        # After Y-flip and scale: (0, 480) -> (0, 0) -> (0, 0) cm
        # (100, 480) -> (100, 0) -> (50, 0) cm
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        # Check that x coordinate was scaled (100 px -> 50 cm)
        assert np.isclose(result[1][0], 50.0, atol=0.1)

    def test_empty_nodes_returns_empty(self) -> None:
        """Empty node list should return empty list."""
        from neurospatial.annotation._track_helpers import transform_nodes_to_output

        result = transform_nodes_to_output([], calibration=None)
        assert result == []

    def test_single_node(self) -> None:
        """Single node should work correctly."""
        from neurospatial.annotation._track_helpers import transform_nodes_to_output

        nodes_px = [(50.0, 100.0)]
        result = transform_nodes_to_output(nodes_px, calibration=None)

        assert result == nodes_px


# -----------------------------------------------------------------------------
# Tests for build_track_graph_from_positions
# -----------------------------------------------------------------------------


class TestBuildTrackGraphFromPositions:
    """Tests for build_track_graph_from_positions function."""

    def test_import(self) -> None:
        """Function should be importable from module."""
        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )

        assert build_track_graph_from_positions is not None

    def test_creates_graph_with_nodes(self) -> None:
        """Should create graph with nodes having 'pos' attribute."""
        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )

        node_positions = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        edges = [(0, 1), (1, 2)]

        graph = build_track_graph_from_positions(node_positions, edges)

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2

        # Check node positions
        for i, pos in enumerate(node_positions):
            assert graph.nodes[i]["pos"] == pos

    def test_edges_have_distance_attribute(self) -> None:
        """Edges should have 'distance' attribute computed correctly."""
        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )

        node_positions = [(0.0, 0.0), (3.0, 4.0)]  # Distance = 5.0
        edges = [(0, 1)]

        graph = build_track_graph_from_positions(node_positions, edges)

        # Edge 0-1 should have distance 5.0 (3-4-5 triangle)
        edge_data = graph.edges[0, 1]
        assert "distance" in edge_data
        assert np.isclose(edge_data["distance"], 5.0)

    def test_edges_have_edge_id_attribute(self) -> None:
        """Edges should have 'edge_id' attribute."""
        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )

        node_positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        edges = [(0, 1), (1, 2)]

        graph = build_track_graph_from_positions(node_positions, edges)

        # Check each edge has edge_id
        for u, v in edges:
            assert "edge_id" in graph.edges[u, v]

    def test_empty_edges_creates_nodes_only(self) -> None:
        """Empty edges list should create graph with nodes but no edges."""
        from neurospatial.annotation._track_helpers import (
            build_track_graph_from_positions,
        )

        node_positions = [(0.0, 0.0), (10.0, 0.0)]
        edges: list[tuple[int, int]] = []

        graph = build_track_graph_from_positions(node_positions, edges)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0


# -----------------------------------------------------------------------------
# Tests for build_track_graph_result
# -----------------------------------------------------------------------------


class TestBuildTrackGraphResult:
    """Tests for build_track_graph_result function."""

    def test_import(self) -> None:
        """Function should be importable from module."""
        from neurospatial.annotation._track_helpers import build_track_graph_result

        assert build_track_graph_result is not None

    def test_result_has_all_fields(self) -> None:
        """Result should have all TrackGraphResult fields."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0, label="start")
        state.add_node(10.0, 0.0, label="middle")
        state.add_node(20.0, 0.0, label="end")
        state.add_edge(0, 1)
        state.add_edge(1, 2)
        state.set_start_node(0)

        result = build_track_graph_result(state, calibration=None)

        # Check all expected fields exist
        assert hasattr(result, "track_graph")
        assert hasattr(result, "node_positions")
        assert hasattr(result, "edges")
        assert hasattr(result, "edge_order")
        assert hasattr(result, "edge_spacing")
        assert hasattr(result, "node_labels")
        assert hasattr(result, "start_node")
        assert hasattr(result, "pixel_positions")

    def test_pixel_positions_preserved(self) -> None:
        """pixel_positions should contain original pixel coordinates."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration=None)

        assert result.pixel_positions == [(100.0, 200.0), (300.0, 400.0)]

    def test_node_positions_without_calibration(self) -> None:
        """Without calibration, node_positions should match pixel_positions."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(100.0, 200.0)
        state.add_node(300.0, 400.0)
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration=None)

        assert result.node_positions == result.pixel_positions

    def test_edges_from_state(self) -> None:
        """edges should match state.edges."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_node(20.0, 0.0)
        state.add_edge(0, 1)
        state.add_edge(1, 2)

        result = build_track_graph_result(state, calibration=None)

        assert result.edges == [(0, 1), (1, 2)]

    def test_node_labels_from_state(self) -> None:
        """node_labels should match state.node_labels."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0, label="start")
        state.add_node(10.0, 0.0, label="end")
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration=None)

        assert result.node_labels == ["start", "end"]

    def test_start_node_from_state(self) -> None:
        """start_node should use get_effective_start_node()."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_edge(0, 1)
        state.set_start_node(1)

        result = build_track_graph_result(state, calibration=None)

        assert result.start_node == 1

    def test_start_node_defaults_to_zero(self) -> None:
        """start_node should default to 0 if not set."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_edge(0, 1)
        # Don't set start_node explicitly

        result = build_track_graph_result(state, calibration=None)

        assert result.start_node == 0  # Default

    def test_empty_state_returns_none_graph(self) -> None:
        """Empty state (< 2 nodes) should return None for track_graph."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        # Only 1 node, no edges

        result = build_track_graph_result(state, calibration=None)

        assert result.track_graph is None
        assert result.edge_order == []
        assert len(result.edge_spacing) == 0

    def test_insufficient_edges_returns_none_graph(self) -> None:
        """State with < 1 edge should return None for track_graph."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        # No edges added

        result = build_track_graph_result(state, calibration=None)

        assert result.track_graph is None

    def test_track_graph_has_correct_structure(self) -> None:
        """track_graph should have proper node/edge attributes."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration=None)

        assert result.track_graph is not None
        assert result.track_graph.number_of_nodes() == 2
        assert result.track_graph.number_of_edges() == 1

        # Check node positions
        assert result.track_graph.nodes[0]["pos"] == (0.0, 0.0)
        assert result.track_graph.nodes[1]["pos"] == (10.0, 0.0)

    def test_edge_order_computed(self) -> None:
        """edge_order should be computed from infer_edge_layout."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_node(20.0, 0.0)
        state.add_edge(0, 1)
        state.add_edge(1, 2)

        result = build_track_graph_result(state, calibration=None)

        # edge_order should be a list of tuples
        assert isinstance(result.edge_order, list)
        assert len(result.edge_order) == 2

    def test_edge_spacing_computed(self) -> None:
        """edge_spacing should be computed from infer_edge_layout."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_node(20.0, 0.0)
        state.add_edge(0, 1)
        state.add_edge(1, 2)

        result = build_track_graph_result(state, calibration=None)

        # edge_spacing should be a numpy array
        assert isinstance(result.edge_spacing, np.ndarray)

    def test_with_calibration_transforms_positions(self) -> None:
        """With calibration, node_positions should be in cm."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState
        from neurospatial.transforms import (
            VideoCalibration,
            calibrate_from_scale_bar,
        )

        # Create calibration: 100 pixels = 50 cm
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        calibration = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        state = TrackBuilderState()
        # Nodes at bottom of frame for easier Y-flip reasoning
        state.add_node(0.0, 480.0)
        state.add_node(100.0, 480.0)
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration)

        # pixel_positions unchanged
        assert result.pixel_positions[0] == (0.0, 480.0)
        assert result.pixel_positions[1] == (100.0, 480.0)

        # node_positions transformed - x scaled, y flipped
        # After flip: y=480 -> y=0 (bottom to origin)
        # After scale: x=100px -> x=50cm
        assert np.isclose(result.node_positions[1][0], 50.0, atol=0.1)

    def test_track_graph_uses_transformed_positions(self) -> None:
        """track_graph node positions should match transformed node_positions."""
        from neurospatial.annotation._track_helpers import build_track_graph_result
        from neurospatial.annotation._track_state import TrackBuilderState

        state = TrackBuilderState()
        state.add_node(0.0, 0.0)
        state.add_node(10.0, 0.0)
        state.add_edge(0, 1)

        result = build_track_graph_result(state, calibration=None)

        assert result.track_graph is not None
        # Graph positions should match node_positions
        for i, pos in enumerate(result.node_positions):
            assert result.track_graph.nodes[i]["pos"] == pos


# -----------------------------------------------------------------------------
# Tests for TrackGraphResult (defined in track_graph.py but tested here)
# -----------------------------------------------------------------------------


class TestTrackGraphResult:
    """Tests for TrackGraphResult NamedTuple."""

    def test_import(self) -> None:
        """TrackGraphResult should be importable."""
        from neurospatial.annotation._track_helpers import TrackGraphResult

        assert TrackGraphResult is not None

    def test_is_namedtuple(self) -> None:
        """TrackGraphResult should be a NamedTuple."""
        from neurospatial.annotation._track_helpers import TrackGraphResult

        # NamedTuples have _fields attribute
        assert hasattr(TrackGraphResult, "_fields")

    def test_has_expected_fields(self) -> None:
        """TrackGraphResult should have all expected fields."""
        from neurospatial.annotation._track_helpers import TrackGraphResult

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

    def test_to_environment_method(self) -> None:
        """TrackGraphResult should have to_environment method."""
        from neurospatial.annotation._track_helpers import TrackGraphResult

        assert hasattr(TrackGraphResult, "to_environment")

    def test_to_environment_raises_without_graph(self) -> None:
        """to_environment should raise if track_graph is None."""
        from neurospatial.annotation._track_helpers import TrackGraphResult

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

    def test_to_environment_creates_environment(self) -> None:
        """to_environment should create valid Environment."""
        pytest.importorskip("track_linearization")

        from track_linearization import infer_edge_layout

        from neurospatial.annotation._track_helpers import (
            TrackGraphResult,
            build_track_graph_from_positions,
        )

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
