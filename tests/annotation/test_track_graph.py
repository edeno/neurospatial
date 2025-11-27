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


# -----------------------------------------------------------------------------
# Task 3.2: Tests for annotate_track_graph entry point
# -----------------------------------------------------------------------------


class TestAnnotateTrackGraphInputValidation:
    """Tests for annotate_track_graph input validation."""

    def test_requires_video_or_image(self) -> None:
        """annotate_track_graph should raise ValueError if neither video_path nor image provided."""
        from neurospatial.annotation.track_graph import annotate_track_graph

        with pytest.raises(
            ValueError, match="Either video_path or image must be provided"
        ):
            annotate_track_graph()

    def test_frame_index_ignored_for_image(self, monkeypatch) -> None:
        """frame_index should be ignored when image is provided directly."""
        _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # frame_index should be ignored when image is provided - should not raise
        result = annotate_track_graph(image=image, frame_index=999)
        assert result is not None

    def test_image_array_accepted(self, monkeypatch) -> None:
        """annotate_track_graph should accept image array directly."""
        _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not raise - accepts image array
        result = annotate_track_graph(image=image)

        # Should return a TrackGraphResult
        assert result is not None
        assert hasattr(result, "track_graph")


class TestAnnotateTrackGraphWithMockViewer:
    """Tests for annotate_track_graph with mocked napari viewer."""

    def test_viewer_title_set(self, monkeypatch) -> None:
        """annotate_track_graph should set viewer title."""
        mock_napari = _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotate_track_graph(image=image)

        assert mock_napari._viewer.title == "Track Graph Builder"

    def test_image_added_to_viewer(self, monkeypatch) -> None:
        """annotate_track_graph should add image as bottom layer."""
        mock_napari = _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotate_track_graph(image=image)

        # Check that add_image was called
        assert len(mock_napari._viewer.images) == 1
        assert mock_napari._viewer.images[0]["name"] == "video_frame"

    def test_layers_setup_called(self, monkeypatch) -> None:
        """annotate_track_graph should set up track layers."""
        mock_napari = _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotate_track_graph(image=image)

        # Check that shapes and points layers were added
        assert len(mock_napari._viewer.shapes) >= 1  # Edges layer
        assert len(mock_napari._viewer.points) >= 1  # Nodes layer

    def test_widget_docked(self, monkeypatch) -> None:
        """annotate_track_graph should dock the control widget."""
        mock_napari = _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotate_track_graph(image=image)

        # Check that widget was docked
        assert mock_napari._viewer.dock_widgets_added >= 1


class TestAnnotateTrackGraphInitialData:
    """Tests for annotate_track_graph with initial data."""

    def test_initial_nodes_populate_state(self, monkeypatch) -> None:
        """annotate_track_graph should populate state with initial_nodes."""
        captured_state = []
        _setup_mocks(monkeypatch, captured_state=captured_state)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        initial_nodes = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        annotate_track_graph(image=image, initial_nodes=initial_nodes)

        # Check state was populated
        assert len(captured_state) == 1
        state = captured_state[0]
        assert len(state.nodes) == 3
        assert state.nodes[0] == (10.0, 20.0)
        assert state.nodes[1] == (30.0, 40.0)
        assert state.nodes[2] == (50.0, 60.0)

    def test_initial_edges_populate_state(self, monkeypatch) -> None:
        """annotate_track_graph should populate state with initial_edges."""
        captured_state = []
        _setup_mocks(monkeypatch, captured_state=captured_state)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        initial_nodes = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        initial_edges = [(0, 1), (1, 2)]

        annotate_track_graph(
            image=image, initial_nodes=initial_nodes, initial_edges=initial_edges
        )

        assert len(captured_state) == 1
        state = captured_state[0]
        assert len(state.edges) == 2
        assert (0, 1) in state.edges
        assert (1, 2) in state.edges

    def test_initial_node_labels_populate_state(self, monkeypatch) -> None:
        """annotate_track_graph should populate state with initial_node_labels."""
        captured_state = []
        _setup_mocks(monkeypatch, captured_state=captured_state)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        initial_nodes = np.array([[0.0, 0.0], [10.0, 0.0]])
        initial_labels = ["start", "end"]

        annotate_track_graph(
            image=image,
            initial_nodes=initial_nodes,
            initial_node_labels=initial_labels,
        )

        assert len(captured_state) == 1
        state = captured_state[0]
        assert len(state.node_labels) == 2
        assert state.node_labels[0] == "start"
        assert state.node_labels[1] == "end"


class TestAnnotateTrackGraphResult:
    """Tests for annotate_track_graph result construction."""

    def test_returns_track_graph_result(self, monkeypatch) -> None:
        """annotate_track_graph should return TrackGraphResult."""
        _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import (
            TrackGraphResult,
            annotate_track_graph,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = annotate_track_graph(image=image)

        assert isinstance(result, TrackGraphResult)

    def test_result_has_all_fields(self, monkeypatch) -> None:
        """annotate_track_graph result should have all expected fields."""
        _setup_mocks(monkeypatch)

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = annotate_track_graph(image=image)

        # Check all fields exist
        assert hasattr(result, "track_graph")
        assert hasattr(result, "node_positions")
        assert hasattr(result, "edges")
        assert hasattr(result, "edge_order")
        assert hasattr(result, "edge_spacing")
        assert hasattr(result, "node_labels")
        assert hasattr(result, "start_node")
        assert hasattr(result, "pixel_positions")


class TestAnnotateTrackGraphCalibration:
    """Tests for annotate_track_graph with calibration."""

    def test_coordinates_in_pixels_without_calibration(self, monkeypatch) -> None:
        """Without calibration, coordinates should be in pixels."""
        # Custom setup to add nodes in the widget callback
        mock_napari = _make_mock_napari_module()

        import sys

        sys.modules["napari"] = mock_napari
        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.napari", mock_napari, raising=False
        )

        def mock_setup_track_layers(viewer):
            edges_layer = _MockShapesLayer("Track Edges")
            nodes_layer = _MockPointsLayer("Track Nodes")
            mock_napari._viewer.shapes.append(edges_layer)
            mock_napari._viewer.points.append(nodes_layer)
            return edges_layer, nodes_layer

        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.setup_track_layers",
            mock_setup_track_layers,
        )

        def mock_create_widget(viewer, edges, nodes, state):
            # Add some nodes to state before capturing
            state.add_node(100.0, 200.0)
            state.add_node(300.0, 400.0)
            return _MockWidget()

        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.create_track_widget",
            mock_create_widget,
        )

        from neurospatial.annotation.track_graph import annotate_track_graph

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = annotate_track_graph(image=image)

        # Without calibration, node_positions should be same as pixel_positions
        assert result.node_positions == result.pixel_positions

    def test_pixel_positions_preserved_with_calibration(self, monkeypatch) -> None:
        """With calibration, pixel_positions should preserve original coords."""
        # Custom setup to add nodes in the widget callback
        mock_napari = _make_mock_napari_module()

        import sys

        sys.modules["napari"] = mock_napari
        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.napari", mock_napari, raising=False
        )

        def mock_setup_track_layers(viewer):
            edges_layer = _MockShapesLayer("Track Edges")
            nodes_layer = _MockPointsLayer("Track Nodes")
            mock_napari._viewer.shapes.append(edges_layer)
            mock_napari._viewer.points.append(nodes_layer)
            return edges_layer, nodes_layer

        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.setup_track_layers",
            mock_setup_track_layers,
        )

        def mock_create_widget(viewer, edges, nodes, state):
            state.add_node(100.0, 200.0)
            state.add_node(300.0, 400.0)
            return _MockWidget()

        monkeypatch.setattr(
            "neurospatial.annotation.track_graph.create_track_widget",
            mock_create_widget,
        )

        from neurospatial.annotation.track_graph import annotate_track_graph
        from neurospatial.transforms import VideoCalibration, scale_2d

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create a simple calibration (scale by 0.5)
        transform = scale_2d(0.5, 0.5)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        result = annotate_track_graph(image=image, calibration=calibration)

        # pixel_positions should have original pixel values
        assert result.pixel_positions[0] == (100.0, 200.0)
        assert result.pixel_positions[1] == (300.0, 400.0)


# -----------------------------------------------------------------------------
# Mock classes for testing without napari
# -----------------------------------------------------------------------------


def _setup_mocks(monkeypatch, captured_state=None):
    """Set up all mocks needed for annotate_track_graph tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    captured_state : list, optional
        If provided, the state passed to create_track_widget will be appended.
    """
    mock_napari = _make_mock_napari_module()

    # Patch napari import inside the function
    import sys

    sys.modules["napari"] = mock_napari
    monkeypatch.setattr(
        "neurospatial.annotation.track_graph.napari", mock_napari, raising=False
    )

    # Patch setup_track_layers to return mock layers
    def mock_setup_track_layers(viewer):
        edges_layer = _MockShapesLayer("Track Edges")
        nodes_layer = _MockPointsLayer("Track Nodes")
        mock_napari._viewer.shapes.append(edges_layer)
        mock_napari._viewer.points.append(nodes_layer)
        return edges_layer, nodes_layer

    monkeypatch.setattr(
        "neurospatial.annotation.track_graph.setup_track_layers",
        mock_setup_track_layers,
    )

    # Patch create_track_widget to return mock widget
    def mock_create_widget(viewer, edges, nodes, state):
        if captured_state is not None:
            captured_state.append(state)
        return _MockWidget()

    monkeypatch.setattr(
        "neurospatial.annotation.track_graph.create_track_widget",
        mock_create_widget,
    )

    return mock_napari


def _make_mock_napari_module():
    """Create a mock napari module with a fresh viewer."""
    viewer = _MockViewer()
    return _MockNapari(viewer)


class _MockViewer:
    """Mock napari viewer for testing."""

    def __init__(self):
        self.title = ""
        self.status = ""
        self.images = []
        self.shapes = []
        self.points = []
        self.dock_widgets_added = 0
        self._window = _MockWindow(self)

    @property
    def window(self):
        return self._window

    def add_image(self, data, name="", rgb=False):
        self.images.append({"data": data, "name": name, "rgb": rgb})
        return _MockLayer(name)

    def add_shapes(self, data=None, name="", **kwargs):
        layer = _MockShapesLayer(name)
        self.shapes.append(layer)
        return layer

    def add_points(self, data=None, name="", **kwargs):
        layer = _MockPointsLayer(name)
        self.points.append(layer)
        return layer

    def close(self):
        pass


class _MockWindow:
    """Mock napari window."""

    def __init__(self, viewer):
        self._viewer = viewer

    def add_dock_widget(self, widget, name="", area="right"):
        self._viewer.dock_widgets_added += 1

    def resize(self, w, h):
        pass


class _MockLayer:
    """Mock napari layer."""

    def __init__(self, name):
        self.name = name
        self.data = []


class _MockShapesLayer(_MockLayer):
    """Mock napari Shapes layer."""

    def add_paths(self, paths, **kwargs):
        self.data.extend(paths)


class _MockPointsLayer(_MockLayer):
    """Mock napari Points layer."""

    @property
    def face_color(self):
        return np.array([[1, 0, 0, 1]])

    @face_color.setter
    def face_color(self, value):
        pass

    @property
    def size(self):
        return np.array([10])

    @size.setter
    def size(self, value):
        pass

    @property
    def mode(self):
        return "pan_zoom"

    @mode.setter
    def mode(self, value):
        pass


class _MockNapari:
    """Mock napari module."""

    def __init__(self, viewer):
        self._viewer = viewer

    def Viewer(self, title=""):
        self._viewer.title = title
        return self._viewer

    def run(self):
        # Don't block - just return immediately
        pass


class _MockWidget:
    """Mock widget for testing."""

    def __init__(self):
        self._widget = None

    @property
    def native(self):
        return self
