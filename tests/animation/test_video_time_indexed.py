"""Tests for time-indexed video layer optimization (Phase 2.1).

Tests the optimization that uses napari's native time dimension for in-memory
video arrays, eliminating per-frame layer.data updates.

**Key behavior**:
- In-memory arrays (np.ndarray): Create 4D Image layer with time dimension
  → napari handles frame selection natively, no callback needed
- File-based readers (VideoReaderProtocol): Use callback approach for streaming
  → layer.data updated on each frame change

This optimization reduces per-frame overhead from ~2-3ms to near-zero for
in-memory video overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Skip all tests if napari not available
pytest.importorskip("napari")


# =============================================================================
# Protocol for VideoReader (file-based video source)
# =============================================================================


class MockVideoReader(Protocol):
    """Protocol for file-based video readers."""

    def __getitem__(self, idx: int) -> NDArray[np.uint8]: ...

    def __len__(self) -> int: ...


class SimpleVideoReader:
    """Simple mock video reader for testing file-based video path.

    Implements VideoReaderProtocol interface for testing.
    """

    def __init__(self, frames: NDArray[np.uint8]) -> None:
        self._frames = frames

    @property
    def n_frames(self) -> int:
        """Total number of frames."""
        return len(self._frames)

    @property
    def frame_size_px(self) -> tuple[int, int]:
        """Frame dimensions as (width, height) in pixels."""
        # frames shape is (n_frames, height, width, 3)
        h, w = self._frames.shape[1:3]
        return (w, h)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        frame: NDArray[np.uint8] = self._frames[idx]
        return frame

    def __len__(self) -> int:
        return len(self._frames)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    from neurospatial import Environment

    positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_fields(simple_env: Environment) -> list[NDArray[np.float64]]:
    """Create simple field sequence for testing (10 frames)."""
    rng = np.random.default_rng(42)
    return [rng.random(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def video_frames_array() -> NDArray[np.uint8]:
    """Create in-memory video array (n_frames, height, width, 3)."""
    rng = np.random.default_rng(42)
    n_frames = 10
    height, width = 64, 64
    return rng.integers(0, 256, (n_frames, height, width, 3), dtype=np.uint8)


@pytest.fixture
def video_reader(video_frames_array: NDArray[np.uint8]) -> SimpleVideoReader:
    """Create file-based video reader mock."""
    return SimpleVideoReader(video_frames_array)


# =============================================================================
# Tests for In-Memory Video (Native Time Dimension)
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_uses_4d_array(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
):
    """Test in-memory video creates 4D Image layer with time dimension.

    When video source is np.ndarray, the full array should be passed to
    add_image with shape (n_frames, height, width, 3), letting napari
    handle frame selection via dims[0].
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Create VideoData with in-memory array
    frame_indices = np.arange(len(video_frames_array))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array,  # np.ndarray - in-memory
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Verify add_image was called
    assert mock_viewer.add_image.called

    # Get the data passed to add_image
    call_args = mock_viewer.add_image.call_args
    image_data = call_args[0][0]

    # Should be 4D array (n_frames, height, width, 3)
    assert image_data.ndim == 4, f"Expected 4D array, got {image_data.ndim}D"
    assert image_data.shape == video_frames_array.shape, (
        f"Expected shape {video_frames_array.shape}, got {image_data.shape}"
    )


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_no_frame_callback(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
):
    """Test in-memory video does NOT register frame change callback.

    When using native time dimension, napari handles frame selection
    automatically - no callback needed for layer.data updates.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Track callback registrations
    callbacks_registered = []
    original_connect = MagicMock(side_effect=lambda cb: callbacks_registered.append(cb))
    mock_viewer.dims.events.current_step.connect = original_connect

    # Create VideoData with in-memory array
    frame_indices = np.arange(len(video_frames_array))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check that no video-related callback was registered
    # Other callbacks (e.g., for position overlays) may still be registered,
    # but the video frame callback (update_video_frames) should not be
    for callback in callbacks_registered:
        # Video callback has characteristic name
        callback_name = getattr(callback, "__name__", str(callback))
        assert "update_video_frames" not in callback_name, (
            "Video frame callback should NOT be registered for in-memory video"
        )


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_layer_rgb_mode(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
):
    """Test in-memory video layer is created with rgb=True."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    frame_indices = np.arange(len(video_frames_array))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    call_kwargs = mock_viewer.add_image.call_args[1]
    assert call_kwargs.get("rgb") is True, "Video layer should have rgb=True"


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_opacity_applied(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
):
    """Test in-memory video layer has correct opacity."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    alpha = 0.65
    frame_indices = np.arange(len(video_frames_array))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=alpha,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    call_kwargs = mock_viewer.add_image.call_args[1]
    assert call_kwargs.get("opacity") == pytest.approx(alpha)


# =============================================================================
# Tests for File-Based Video (Callback Approach)
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_file_based_video_uses_callback(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_reader,
):
    """Test file-based video uses callback for frame updates.

    When video source is VideoReaderProtocol (not np.ndarray), we need
    to use the callback approach to stream frames on demand.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Track callback registrations
    callbacks_registered = []
    original_connect = MagicMock(side_effect=lambda cb: callbacks_registered.append(cb))
    mock_viewer.dims.events.current_step.connect = original_connect

    # Create VideoData with file-based reader
    frame_indices = np.arange(len(video_reader))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_reader,  # VideoReaderProtocol - file-based
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should register a callback for frame updates
    assert mock_viewer.dims.events.current_step.connect.called, (
        "File-based video should register frame change callback"
    )


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_file_based_video_uses_single_frame_layer(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_reader,
):
    """Test file-based video creates single-frame Image layer.

    File-based video should create a 3D layer (height, width, 3) that
    gets updated via callback, not a 4D array.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    frame_indices = np.arange(len(video_reader))
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_reader,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Get the data passed to add_image
    call_args = mock_viewer.add_image.call_args
    image_data = call_args[0][0]

    # Should be 3D array (height, width, 3) - single frame
    assert image_data.ndim == 3, (
        f"Expected 3D array for file-based video, got {image_data.ndim}D"
    )


# =============================================================================
# Tests for Frame Index Mapping with Time Dimension
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_reordered_by_frame_indices(
    mock_viewer_class,
    simple_env,
    simple_fields,
):
    """Test in-memory video is reordered according to frame_indices.

    When frame_indices maps animation frames to different video frames
    (e.g., for time alignment), the 4D array should be reordered so
    napari's dims[0] directly indexes the correct video frame.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Create video with 5 frames
    rng = np.random.default_rng(42)
    video_frames = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)

    # Frame indices that map 10 animation frames to 5 video frames (with repeats)
    # Animation frame 0 -> video frame 0
    # Animation frame 1 -> video frame 0
    # Animation frame 2 -> video frame 1
    # ... etc
    frame_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )

    # Extend fields to 10 frames
    extended_fields = [rng.random(simple_env.n_bins) for _ in range(10)]
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, extended_fields, overlay_data=overlay_data)

    # Get the data passed to add_image
    call_args = mock_viewer.add_image.call_args
    image_data = call_args[0][0]

    # Should have 10 frames (matching animation), reordered according to frame_indices
    assert image_data.shape[0] == 10, (
        f"Expected 10 animation frames, got {image_data.shape[0]}"
    )

    # Verify correct mapping: animation frame 0 should equal video frame 0
    assert np.array_equal(image_data[0], video_frames[0])
    # Animation frame 1 should also equal video frame 0 (according to frame_indices)
    assert np.array_equal(image_data[1], video_frames[0])
    # Animation frame 2 should equal video frame 1
    assert np.array_equal(image_data[2], video_frames[1])


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_handles_missing_frames(
    mock_viewer_class,
    simple_env,
    simple_fields,
):
    """Test in-memory video handles frame_indices with -1 (missing frames).

    Missing frames (indicated by -1 in frame_indices) should be filled
    with blank/black frames in the 4D array.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    rng = np.random.default_rng(42)
    video_frames = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)

    # Frame indices with -1 for missing frames
    frame_indices = np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1])

    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )

    extended_fields = [rng.random(simple_env.n_bins) for _ in range(10)]
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, extended_fields, overlay_data=overlay_data)

    call_args = mock_viewer.add_image.call_args
    image_data = call_args[0][0]

    # Should have 10 frames
    assert image_data.shape[0] == 10

    # Missing frames (indices 5-9) should be black/zeros
    for i in range(5, 10):
        assert np.all(image_data[i] == 0), f"Missing frame {i} should be black (zeros)"


# =============================================================================
# Tests for Mixed Scenarios
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_multiple_videos_mixed_types(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
    video_reader,
):
    """Test rendering multiple videos with mixed in-memory and file-based sources."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    n_frames = len(video_frames_array)
    frame_indices = np.arange(n_frames)

    # In-memory video
    video_data_memory = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array,
        transform_to_env=None,
        env_bounds=(0.0, 10.0, 0.0, 5.0),
        alpha=0.8,
        z_order="below",
    )

    # File-based video
    video_data_file = VideoData(
        frame_indices=frame_indices,
        reader=video_reader,
        transform_to_env=None,
        env_bounds=(10.0, 20.0, 5.0, 10.0),
        alpha=0.6,
        z_order="above",
    )

    overlay_data = OverlayData(videos=[video_data_memory, video_data_file])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Both videos should create Image layers
    assert mock_viewer.add_image.call_count >= 2


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_with_position_overlay(
    mock_viewer_class,
    simple_env,
    simple_fields,
    video_frames_array,
):
    """Test in-memory video works correctly with position overlay."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, PositionData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    n_frames = len(simple_fields)
    frame_indices = np.arange(n_frames)

    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames_array[:n_frames],  # Match field count
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )

    positions = np.array([[5.0 + i, 5.0] for i in range(n_frames)])
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=5)

    overlay_data = OverlayData(videos=[video_data], positions=[pos_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Both video (4D) and position layers should be created
    assert mock_viewer.add_image.called
    # Position overlay creates tracks and/or points
    assert mock_viewer.add_tracks.called or mock_viewer.add_points.called


# =============================================================================
# Tests for Edge Cases
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_in_memory_video_single_frame(
    mock_viewer_class,
    simple_env,
    simple_fields,
):
    """Test in-memory video with single frame still works."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, VideoData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    rng = np.random.default_rng(42)
    # Single frame video
    video_frames = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)

    # All animation frames map to the single video frame
    frame_indices = np.zeros(len(simple_fields), dtype=np.int_)

    video_data = VideoData(
        frame_indices=frame_indices,
        reader=video_frames,
        transform_to_env=None,
        env_bounds=(0.0, 20.0, 0.0, 10.0),
        alpha=0.8,
        z_order="below",
    )
    overlay_data = OverlayData(videos=[video_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    call_args = mock_viewer.add_image.call_args
    image_data = call_args[0][0]

    # Should have n_animation_frames copies of the single video frame
    assert image_data.shape[0] == len(simple_fields)


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_empty_video_list_no_error(
    mock_viewer_class,
    simple_env,
    simple_fields,
):
    """Test that empty video list doesn't cause errors."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(videos=[])

    # Should not raise
    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # No video layers should be created
    # (add_image might be called for field layer, but not for video)
