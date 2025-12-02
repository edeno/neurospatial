"""Tests for windowed overlay loading functionality.

Tests the WindowedOverlayManager and WindowedTracksManager classes that
provide progressive loading for large datasets (>50K frames) to avoid
napari's O(n) performance issues with large Points/Tracks layers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip all tests if napari not available
pytest.importorskip("napari")


class TestWindowedOverlayConstants:
    """Test windowed overlay loading constants."""

    def test_threshold_constant_exists(self):
        """Test WINDOWED_OVERLAY_THRESHOLD is defined."""
        from neurospatial.animation.backends.napari_backend import (
            WINDOWED_OVERLAY_THRESHOLD,
        )

        assert isinstance(WINDOWED_OVERLAY_THRESHOLD, int)
        assert WINDOWED_OVERLAY_THRESHOLD == 50_000

    def test_window_radius_constant_exists(self):
        """Test OVERLAY_WINDOW_RADIUS is defined."""
        from neurospatial.animation.backends.napari_backend import (
            OVERLAY_WINDOW_RADIUS,
        )

        assert isinstance(OVERLAY_WINDOW_RADIUS, int)
        assert OVERLAY_WINDOW_RADIUS == 1000

    def test_update_threshold_constant_exists(self):
        """Test OVERLAY_WINDOW_UPDATE_THRESHOLD is defined."""
        from neurospatial.animation.backends.napari_backend import (
            OVERLAY_WINDOW_UPDATE_THRESHOLD,
        )

        assert isinstance(OVERLAY_WINDOW_UPDATE_THRESHOLD, int)
        assert OVERLAY_WINDOW_UPDATE_THRESHOLD == 250


class TestWindowedOverlayManager:
    """Tests for WindowedOverlayManager class."""

    @pytest.fixture
    def mock_viewer(self):
        """Create mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.events.current_step.connect = MagicMock()
        viewer.dims.events.current_step.disconnect = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def mock_layer(self):
        """Create mock napari layer."""
        layer = MagicMock()
        layer.events.blocker = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        layer.metadata = {}
        return layer

    @pytest.fixture
    def sample_points_data(self):
        """Create sample points data (time, y, x)."""
        n_frames = 10000
        return np.column_stack(
            [
                np.arange(n_frames),  # time
                np.random.rand(n_frames) * 100,  # y
                np.random.rand(n_frames) * 100,  # x
            ]
        )

    def test_init_connects_event_handler(
        self, mock_viewer, mock_layer, sample_points_data
    ):
        """Test that initialization connects to dims events."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=sample_points_data,
            window_radius=100,
            update_threshold=25,
        )

        mock_viewer.dims.events.current_step.connect.assert_called_once()
        assert manager._connected is True

    def test_init_sets_initial_window(
        self, mock_viewer, mock_layer, sample_points_data
    ):
        """Test that initialization sets initial window at frame 0."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=sample_points_data,
            window_radius=100,
            update_threshold=25,
        )

        # Window center should be 0 after _update_window(0)
        assert manager.window_center == 0
        # Layer data should have been set
        assert mock_layer.data is not None

    def test_disconnect_method(self, mock_viewer, mock_layer, sample_points_data):
        """Test disconnect() method properly disconnects event handler."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=sample_points_data,
            window_radius=100,
            update_threshold=25,
        )

        assert manager._connected is True
        manager.disconnect()
        assert manager._connected is False
        mock_viewer.dims.events.current_step.disconnect.assert_called_once()

    def test_disconnect_is_idempotent(
        self, mock_viewer, mock_layer, sample_points_data
    ):
        """Test calling disconnect() multiple times is safe."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=sample_points_data,
            window_radius=100,
            update_threshold=25,
        )

        manager.disconnect()
        manager.disconnect()  # Should not raise
        # disconnect should only be called once
        assert mock_viewer.dims.events.current_step.disconnect.call_count == 1

    def test_update_window_filters_data_correctly(self, mock_viewer, mock_layer):
        """Test _update_window correctly filters data to window."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        # Create data with known time values
        n_frames = 1000
        data = np.column_stack(
            [
                np.arange(n_frames),  # time 0-999
                np.arange(n_frames),  # y
                np.arange(n_frames),  # x
            ]
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=data,
            window_radius=100,
            update_threshold=25,
        )

        # Update to center at frame 500
        manager._update_window(500)

        # Check that layer.data was set with correct window
        set_data = mock_layer.data
        assert len(set_data) == 200  # 100 on each side
        assert set_data[0, 0] == 400  # Start at 500-100
        assert set_data[-1, 0] == 599  # End at 500+100-1

    def test_update_window_handles_boundaries(self, mock_viewer, mock_layer):
        """Test _update_window handles start/end boundaries correctly."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        n_frames = 1000
        data = np.column_stack(
            [
                np.arange(n_frames),
                np.arange(n_frames),
                np.arange(n_frames),
            ]
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=data,
            window_radius=100,
            update_threshold=25,
        )

        # Update to center at frame 0 (near start)
        manager._update_window(0)
        # Window should be [0, 100), not [-100, 100)
        set_data = mock_layer.data
        assert set_data[0, 0] == 0
        assert len(set_data) == 100

        # Update to center at frame 999 (near end)
        manager._update_window(999)
        set_data = mock_layer.data
        assert set_data[-1, 0] == 999
        assert set_data[0, 0] == 899

    def test_on_frame_change_respects_threshold(
        self, mock_viewer, mock_layer, sample_points_data
    ):
        """Test that small frame changes don't trigger window updates."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=sample_points_data,
            window_radius=100,
            update_threshold=25,
        )

        initial_center = manager.window_center

        # Small change (< threshold) should not update
        mock_viewer.dims.current_step = [initial_center + 10]
        manager._on_frame_change(None)
        assert manager.window_center == initial_center

        # Large change (> threshold) should update
        mock_viewer.dims.current_step = [initial_center + 50]
        manager._on_frame_change(None)
        assert manager.window_center == initial_center + 50

    def test_empty_window_clears_layer(self, mock_viewer, mock_layer):
        """Test that empty window results in empty layer data."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedOverlayManager,
        )

        # Create sparse data with gap
        data = np.column_stack(
            [
                [0, 1, 2, 1000, 1001, 1002],  # time with gap
                [0, 1, 2, 0, 1, 2],  # y
                [0, 1, 2, 0, 1, 2],  # x
            ]
        ).astype(np.float64)

        manager = WindowedOverlayManager(
            viewer=mock_viewer,
            layer=mock_layer,
            full_data=data,
            window_radius=10,
            update_threshold=5,
        )

        # Update to center in the gap (e.g., frame 500)
        manager._update_window(500)

        # Should set empty array
        set_data = mock_layer.data
        assert len(set_data) == 0


class TestWindowedTracksManager:
    """Tests for WindowedTracksManager class."""

    @pytest.fixture
    def mock_viewer(self):
        """Create mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.events.current_step.connect = MagicMock()
        viewer.dims.events.current_step.disconnect = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    @pytest.fixture
    def mock_tracks_layer(self):
        """Create mock napari Tracks layer."""
        layer = MagicMock()
        layer.events.blocker = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        layer.metadata = {}
        return layer

    @pytest.fixture
    def sample_tracks_data(self):
        """Create sample tracks data (track_id, time, y, x)."""
        n_frames = 10000
        return np.column_stack(
            [
                np.zeros(n_frames),  # track_id
                np.arange(n_frames),  # time
                np.random.rand(n_frames) * 100,  # y
                np.random.rand(n_frames) * 100,  # x
            ]
        )

    def test_init_connects_event_handler(
        self, mock_viewer, mock_tracks_layer, sample_tracks_data
    ):
        """Test that initialization connects to dims events."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedTracksManager,
        )

        manager = WindowedTracksManager(
            viewer=mock_viewer,
            layer=mock_tracks_layer,
            full_data=sample_tracks_data,
            colormaps_dict={},
            window_radius=100,
            update_threshold=25,
        )

        mock_viewer.dims.events.current_step.connect.assert_called_once()
        assert manager._connected is True

    def test_disconnect_method(
        self, mock_viewer, mock_tracks_layer, sample_tracks_data
    ):
        """Test disconnect() method properly disconnects event handler."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedTracksManager,
        )

        manager = WindowedTracksManager(
            viewer=mock_viewer,
            layer=mock_tracks_layer,
            full_data=sample_tracks_data,
            colormaps_dict={},
            window_radius=100,
            update_threshold=25,
        )

        assert manager._connected is True
        manager.disconnect()
        assert manager._connected is False
        mock_viewer.dims.events.current_step.disconnect.assert_called_once()

    def test_time_column_is_1(self, mock_viewer, mock_tracks_layer, sample_tracks_data):
        """Test that tracks manager uses time_column=1 (track format)."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedTracksManager,
        )

        manager = WindowedTracksManager(
            viewer=mock_viewer,
            layer=mock_tracks_layer,
            full_data=sample_tracks_data,
            colormaps_dict={},
            window_radius=100,
            update_threshold=25,
        )

        assert manager.time_column == 1

    def test_update_window_updates_features(self, mock_viewer, mock_tracks_layer):
        """Test _update_window updates both data and features."""
        from neurospatial.animation.backends.napari_backend import (
            WindowedTracksManager,
        )

        n_frames = 1000
        data = np.column_stack(
            [
                np.zeros(n_frames),  # track_id
                np.arange(n_frames),  # time
                np.arange(n_frames),  # y
                np.arange(n_frames),  # x
            ]
        )

        manager = WindowedTracksManager(
            viewer=mock_viewer,
            layer=mock_tracks_layer,
            full_data=data,
            colormaps_dict={"color": MagicMock()},
            window_radius=100,
            update_threshold=25,
        )

        # Update to center at frame 500
        manager._update_window(500)

        # Check both data and features were set
        assert mock_tracks_layer.data is not None
        assert mock_tracks_layer.features is not None

        # Features should have 'color' key with matching length
        features = mock_tracks_layer.features
        assert "color" in features
        assert len(features["color"]) == len(mock_tracks_layer.data)


class TestWindowedLoadingIntegration:
    """Integration tests for windowed loading with actual napari viewer."""

    @pytest.fixture
    def simple_env(self):
        """Create simple 2D environment for testing."""
        from neurospatial import Environment

        positions = np.random.rand(100, 2) * 50
        return Environment.from_samples(positions, bin_size=2.0)

    def test_large_dataset_uses_windowed_loading(self, simple_env):
        """Test that datasets above threshold use windowed loading."""

        from neurospatial import PositionOverlay
        from neurospatial.animation.backends.napari_backend import (
            WINDOWED_OVERLAY_THRESHOLD,
        )

        # Create data above threshold
        n_frames = WINDOWED_OVERLAY_THRESHOLD + 10000
        trajectory = np.column_stack(
            [
                25 + 10 * np.cos(np.linspace(0, 10 * np.pi, n_frames)),
                25 + 10 * np.sin(np.linspace(0, 10 * np.pi, n_frames)),
            ]
        )
        frame_times = np.arange(n_frames) / 30.0
        fields = np.ones((n_frames, simple_env.n_bins))

        position_overlay = PositionOverlay(
            data=trajectory,
            times=frame_times,
            color="cyan",
            size=8.0,
            trail_length=30,
        )

        viewer = simple_env.animate_fields(
            fields,
            frame_times=frame_times,
            backend="napari",
            overlays=[position_overlay],
            title="Test",
        )

        try:
            # Check that windowed managers were created
            for layer in viewer.layers:
                if "Position" in layer.name:
                    assert "windowed_manager" in layer.metadata, (
                        f"Layer {layer.name} should have windowed manager"
                    )
                    manager = layer.metadata["windowed_manager"]
                    assert hasattr(manager, "disconnect")
                    # Verify window is smaller than full data
                    assert len(layer.data) < n_frames
        finally:
            viewer.close()

    def test_small_dataset_does_not_use_windowed_loading(self, simple_env):
        """Test that datasets below threshold don't use windowed loading."""

        from neurospatial import PositionOverlay

        # Create data below threshold
        n_frames = 1000  # Well below 50K threshold
        trajectory = np.column_stack(
            [
                25 + 10 * np.cos(np.linspace(0, 2 * np.pi, n_frames)),
                25 + 10 * np.sin(np.linspace(0, 2 * np.pi, n_frames)),
            ]
        )
        frame_times = np.arange(n_frames) / 30.0
        fields = np.ones((n_frames, simple_env.n_bins))

        position_overlay = PositionOverlay(
            data=trajectory,
            times=frame_times,
            color="cyan",
            size=8.0,
            trail_length=30,
        )

        viewer = simple_env.animate_fields(
            fields,
            frame_times=frame_times,
            backend="napari",
            overlays=[position_overlay],
            title="Test",
        )

        try:
            # Check that windowed managers were NOT created
            for layer in viewer.layers:
                if "Position" in layer.name:
                    assert "windowed_manager" not in layer.metadata, (
                        f"Layer {layer.name} should NOT have windowed manager"
                    )
        finally:
            viewer.close()

    def test_windowed_manager_disconnect_on_cleanup(self, simple_env):
        """Test that windowed managers can be disconnected for cleanup."""

        from neurospatial import PositionOverlay
        from neurospatial.animation.backends.napari_backend import (
            WINDOWED_OVERLAY_THRESHOLD,
        )

        n_frames = WINDOWED_OVERLAY_THRESHOLD + 1000
        trajectory = np.column_stack(
            [
                25 + 10 * np.cos(np.linspace(0, 4 * np.pi, n_frames)),
                25 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_frames)),
            ]
        )
        frame_times = np.arange(n_frames) / 30.0
        fields = np.ones((n_frames, simple_env.n_bins))

        position_overlay = PositionOverlay(
            data=trajectory,
            times=frame_times,
            color="cyan",
            size=8.0,
            trail_length=30,
        )

        viewer = simple_env.animate_fields(
            fields,
            frame_times=frame_times,
            backend="napari",
            overlays=[position_overlay],
            title="Test",
        )

        try:
            # Disconnect all managers
            for layer in viewer.layers:
                if "windowed_manager" in layer.metadata:
                    manager = layer.metadata["windowed_manager"]
                    assert manager._connected is True
                    manager.disconnect()
                    assert manager._connected is False
        finally:
            viewer.close()
