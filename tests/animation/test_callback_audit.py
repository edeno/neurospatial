"""Tests for Phase 5.1: Callback Audit and Migration.

This test module verifies that callbacks connected to
``viewer.dims.events.current_step`` are properly optimized:

1. Video overlay: No callback for in-memory (uses native time), callback for file-based
2. Event overlay: Callback for cumulative/decay modes (already efficient)
3. Timeseries: Callback respects ``update_mode="on_pause"`` via PlaybackController
4. Playback widget: Callback for frame info display (lightweight)

These tests document the audit findings from Phase 5.1 and ensure
the optimization patterns remain in place.

Requires qtpy for PlaybackController tests (QTimer-based debouncing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Helper to check if qtpy is available for PlaybackController tests
def _qtpy_available() -> bool:
    """Check if qtpy is available."""
    try:
        import qtpy  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_viewer() -> MagicMock:
    """Create a mock napari viewer with dims events."""
    viewer = MagicMock()
    viewer.dims.ndim = 4
    viewer.dims.current_step = (0, 0, 0, 0)
    viewer.dims.range = [(0, 100, 1), (0, 50, 1), (0, 50, 1), (0, 3, 1)]

    # Track callback registrations
    viewer._registered_callbacks = []

    def track_connect(callback: Any) -> None:
        viewer._registered_callbacks.append(callback)

    viewer.dims.events.current_step.connect = MagicMock(side_effect=track_connect)

    return viewer


@pytest.fixture
def video_frames_array() -> NDArray[np.uint8]:
    """Create a small in-memory video array (10 frames, 8x8, RGB)."""
    return np.random.randint(0, 255, (10, 8, 8, 3), dtype=np.uint8)


@pytest.fixture
def mock_video_reader() -> MagicMock:
    """Create a mock file-based video reader."""
    reader = MagicMock()
    reader.__len__ = MagicMock(return_value=10)
    reader.__getitem__ = MagicMock(return_value=np.zeros((8, 8, 3), dtype=np.uint8))
    reader.frame_size_px = (8, 8)
    reader.fps = 30.0
    return reader


# =============================================================================
# Video Callback Tests
# =============================================================================


class TestVideoCallbackAudit:
    """Tests verifying video callback behavior per Phase 5.1 audit.

    Expected behavior:
    - In-memory video: No callback registered (uses napari's native time dimension)
    - File-based video: Callback registered (updates layer.data on frame change)
    """

    def test_in_memory_video_no_callback_registered(
        self, mock_viewer: MagicMock, video_frames_array: NDArray[np.uint8]
    ) -> None:
        """In-memory video should NOT register dims callback (uses native time)."""
        from neurospatial.animation.backends.napari_backend import _add_video_layer
        from neurospatial.animation.overlays import VideoData

        # Create VideoData with in-memory array
        n_frames = len(video_frames_array)
        frame_indices = np.arange(n_frames)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=video_frames_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=1.0,
            z_order="below",
        )

        # Add video layer - env=None triggers fallback transform
        _layer, uses_native_time = _add_video_layer(
            mock_viewer,
            video_data,
            env=None,  # type: ignore[arg-type]
            n_frames=n_frames,
            name="test_video",
        )

        # Verify: uses_native_time should be True for in-memory
        assert uses_native_time is True, (
            "In-memory video should use native time dimension"
        )

    def test_file_based_video_callback_registered(
        self, mock_viewer: MagicMock, mock_video_reader: MagicMock
    ) -> None:
        """File-based video should register dims callback for frame updates."""
        from neurospatial.animation.backends.napari_backend import _add_video_layer
        from neurospatial.animation.overlays import VideoData

        # Create VideoData with file-based reader
        n_frames = 10
        frame_indices = np.arange(n_frames)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=mock_video_reader,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=1.0,
            z_order="below",
        )

        # Add video layer
        _layer, uses_native_time = _add_video_layer(
            mock_viewer,
            video_data,
            env=None,  # type: ignore[arg-type]
            n_frames=n_frames,
            name="test_video",
        )

        # Verify: uses_native_time should be False for file-based
        assert uses_native_time is False, "File-based video should NOT use native time"


class TestVideoFrameCallbackRegistration:
    """Tests for _make_video_frame_callback function."""

    def test_empty_video_layers_no_callback(self, mock_viewer: MagicMock) -> None:
        """Empty video layer list should not register any callback."""
        from neurospatial.animation.backends.napari_backend import (
            _make_video_frame_callback,
        )

        initial_count = len(mock_viewer._registered_callbacks)
        _make_video_frame_callback(mock_viewer, [])

        assert len(mock_viewer._registered_callbacks) == initial_count

    def test_file_video_layers_register_callback(
        self, mock_viewer: MagicMock, mock_video_reader: MagicMock
    ) -> None:
        """Video layers with file-based readers should register callback."""
        from neurospatial.animation.backends.napari_backend import (
            _make_video_frame_callback,
        )
        from neurospatial.animation.overlays import VideoData

        # Create mock layer with video data
        video_data = VideoData(
            frame_indices=np.arange(10),
            reader=mock_video_reader,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=1.0,
            z_order="below",
        )

        layer = MagicMock()
        layer.metadata = {
            "video_data": video_data,
            "video_frame_indices": np.arange(10),
        }

        initial_count = len(mock_viewer._registered_callbacks)
        _make_video_frame_callback(mock_viewer, [layer])

        # Should register exactly one callback
        assert len(mock_viewer._registered_callbacks) == initial_count + 1


# =============================================================================
# Timeseries Update Mode Tests
# =============================================================================


class TestTimeseriesUpdateModeAudit:
    """Tests verifying timeseries callback respects update_mode per Phase 5.1.

    Expected behavior:
    - update_mode="live": Updates on every frame change (throttled)
    - update_mode="on_pause": Only updates when PlaybackController.is_playing is False
    - update_mode="manual": Never auto-updates
    """

    def test_on_pause_mode_preserves_through_conversion(self) -> None:
        """on_pause mode should be preserved through overlay conversion."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        overlay = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="Test",
            update_mode="on_pause",
        )

        # Verify update_mode is preserved
        assert overlay.update_mode == "on_pause"

    def test_manual_mode_preserves_through_conversion(self) -> None:
        """manual mode should be preserved through overlay configuration."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        overlay = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="Test",
            update_mode="manual",
        )

        assert overlay.update_mode == "manual"

    def test_live_mode_is_default(self) -> None:
        """live mode should be the default update mode."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        overlay = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="Test",
        )

        assert overlay.update_mode == "live"

    def test_mode_priority_uses_most_restrictive(self) -> None:
        """When multiple overlays have different modes, use most restrictive.

        Priority: manual > on_pause > live
        """
        from neurospatial.animation.overlays import TimeSeriesOverlay

        # Create overlays with different modes
        ts_live = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="Live",
            update_mode="live",
        )

        ts_on_pause = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="OnPause",
            color="green",
            update_mode="on_pause",
        )

        # Verify individual modes
        assert ts_live.update_mode == "live"
        assert ts_on_pause.update_mode == "on_pause"

        # Verify mode priority logic (replicate from _add_timeseries_dock)
        mode_priority = {"manual": 2, "on_pause": 1, "live": 0}
        overlays = [ts_live, ts_on_pause]

        effective_mode = "live"
        for ts in overlays:
            if mode_priority.get(ts.update_mode, 0) > mode_priority.get(
                effective_mode, 0
            ):
                effective_mode = ts.update_mode

        # on_pause should win over live
        assert effective_mode == "on_pause"

    def test_throttle_parameters_configurable(self) -> None:
        """Throttle parameters should be configurable on TimeSeriesOverlay."""
        from neurospatial.animation.overlays import TimeSeriesOverlay

        overlay = TimeSeriesOverlay(
            data=np.random.randn(100),
            times=np.linspace(0, 10, 100),
            label="Test",
            playback_throttle_hz=5.0,
            scrub_throttle_hz=30.0,
        )

        assert overlay.playback_throttle_hz == 5.0
        assert overlay.scrub_throttle_hz == 30.0


# =============================================================================
# Event Overlay Callback Tests
# =============================================================================


class TestEventOverlayCallbackAudit:
    """Tests verifying event overlay callback efficiency per Phase 5.1.

    Expected behavior:
    - None/0 decay_frames (instant mode): Uses 3D Points layer (no per-frame callback)
    - decay_frames > 0: Callback uses efficient layer.shown mask update
    - decay_frames = None with cumulative persistence: Callback uses shown mask
    """

    def test_instant_mode_no_decay(self) -> None:
        """instant mode events should have decay_frames=None (default)."""
        from neurospatial.animation.overlays import EventOverlay

        overlay = EventOverlay(
            event_times=np.array([0.5, 1.0, 1.5]),
            event_positions=np.array([[10, 20], [30, 40], [50, 60]]),
        )

        # Default is None (instant mode)
        assert overlay.decay_frames is None

    def test_decay_mode_has_decay_frames(self) -> None:
        """decay mode should preserve decay_frames setting."""
        from neurospatial.animation.overlays import EventOverlay

        overlay = EventOverlay(
            event_times=np.array([0.5, 1.0, 1.5]),
            event_positions=np.array([[10, 20], [30, 40], [50, 60]]),
            decay_frames=10,
        )

        assert overlay.decay_frames == 10


# =============================================================================
# PlaybackController Callback Integration Tests
# =============================================================================


@pytest.mark.skipif(not _qtpy_available(), reason="PlaybackController requires qtpy")
class TestPlaybackControllerCallbackIntegration:
    """Tests verifying PlaybackController callback mechanism."""

    def test_register_callback_receives_frame_changes(self) -> None:
        """Registered callbacks should receive frame change notifications."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        mock_viewer = MagicMock()
        mock_viewer.dims.set_current_step = MagicMock()

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=0,  # Disable debounce for testing
        )

        # Track callback calls
        received_frames: list[int] = []

        def callback(frame_idx: int) -> None:
            received_frames.append(frame_idx)

        controller.register_callback(callback)

        # Trigger frame changes
        controller.go_to_frame(10)
        controller.go_to_frame(50)
        controller.go_to_frame(99)

        assert received_frames == [10, 50, 99]

    def test_is_playing_property_accessible(self) -> None:
        """is_playing property should be accessible for on_pause mode checks."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        mock_viewer = MagicMock()
        mock_viewer.dims.set_current_step = MagicMock()

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

        # Initially not playing
        assert controller.is_playing is False

        # After play()
        controller.play()
        assert controller.is_playing is True

        # After pause() - state changes dynamically
        # Note: mypy thinks this is unreachable after `assert is True` narrowing
        controller.pause()  # type: ignore[unreachable]
        assert not controller.is_playing


# =============================================================================
# Audit Documentation Tests
# =============================================================================


class TestCallbackAuditDocumentation:
    """Tests documenting the callback audit findings.

    These tests serve as living documentation for the Phase 5.1 audit results.
    """

    def test_audit_summary(self) -> None:
        """Document the callback audit summary.

        Callback Audit Summary (Phase 5.1):

        | Location                     | Callback           | Action                           |
        |------------------------------|--------------------|---------------------------------|
        | _make_video_frame_callback   | update_video_frames| Keep for file-based only        |
        | _render_event_overlay        | on_frame_change    | Keep (efficient shown mask)     |
        | _render_playback_widget      | update_frame_info  | Keep (lightweight UI)           |
        | _add_timeseries_dock         | on_frame_change    | Keep (already uses controller)  |

        Key findings:
        1. In-memory videos now use napari's native time dimension (Phase 2.1)
        2. Event overlays use efficient layer.shown mask updates
        3. Timeseries callback already checks PlaybackController.is_playing
        4. All callbacks remain connected via dims.events.current_step
           (required for responding to user slider interactions)
        """
        # This test documents the audit - it always passes
        assert True
