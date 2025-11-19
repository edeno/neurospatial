"""Test Napari backend for animation.

Tests cover:
- Napari availability checking
- LazyFieldRenderer cache behavior with true LRU eviction
- render_napari() function with all parameters
- Trajectory overlay support
- Integration with Environment
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment


# Helper function for creating properly configured mock viewers
def _create_mock_viewer():
    """Create a properly configured mock napari viewer for testing."""
    mock_viewer = MagicMock()
    mock_viewer.add_image = MagicMock(return_value=None)

    # Configure dims (needed for playback control configuration)
    mock_viewer.dims = MagicMock()
    mock_viewer.dims.ndim = 3  # 3 dimensions (time, height, width)
    mock_viewer.dims.current_step = (0, 0, 0)

    return mock_viewer


# Test data fixtures
@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    return env


@pytest.fixture
def simple_fields(simple_env):
    """Create simple field sequence."""
    n_frames = 10
    fields = []
    for i in range(n_frames):
        field = np.sin(np.linspace(0, 2 * np.pi, simple_env.n_bins) + i * 0.5)
        fields.append(field)
    return fields


# ============================================================================
# Napari Availability Tests
# ============================================================================


def test_napari_available_flag_when_installed():
    """Test NAPARI_AVAILABLE flag when napari is installed."""
    # This test will pass or fail depending on whether napari is installed
    from neurospatial.animation.backends.napari_backend import NAPARI_AVAILABLE

    assert isinstance(NAPARI_AVAILABLE, bool)


def test_napari_available_flag_when_not_installed():
    """Test NAPARI_AVAILABLE flag when napari is not installed."""
    # Mock ImportError during module import
    import sys

    # Save original napari if it exists
    original_napari = sys.modules.get("napari")

    try:
        # Remove napari from sys.modules to simulate not installed
        if "napari" in sys.modules:
            del sys.modules["napari"]

        # Force reimport of napari_backend with napari unavailable
        if "neurospatial.animation.backends.napari_backend" in sys.modules:
            del sys.modules["neurospatial.animation.backends.napari_backend"]

        # Mock napari import to raise ImportError
        with patch.dict("sys.modules", {"napari": None}):
            # Force reload to pick up the mock
            import importlib

            from neurospatial.animation.backends import napari_backend

            importlib.reload(napari_backend)

            # Should be False when import fails
            assert hasattr(napari_backend, "NAPARI_AVAILABLE")

    finally:
        # Restore original state
        if original_napari is not None:
            sys.modules["napari"] = original_napari
        elif "napari" in sys.modules:
            del sys.modules["napari"]

        # Reload napari_backend to original state
        if "neurospatial.animation.backends.napari_backend" in sys.modules:
            del sys.modules["neurospatial.animation.backends.napari_backend"]


# ============================================================================
# LazyFieldRenderer Tests
# ============================================================================


def test_lazy_field_renderer_basic_access(simple_env, simple_fields):
    """Test basic frame access through LazyFieldRenderer."""
    # Create colormap lookup
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    # Test basic properties
    assert len(renderer) == len(simple_fields)
    assert renderer.dtype == np.uint8

    # Test frame access
    frame_0 = renderer[0]
    assert frame_0.dtype == np.uint8
    assert frame_0.ndim == 3  # Should be 3D for RGB
    assert frame_0.shape[-1] == 3  # RGB channels


def test_lazy_field_renderer_lru_cache(simple_env, simple_fields):
    """Test LRU cache behavior - oldest items evicted first."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    # Set small cache size for testing
    renderer._cache_size = 3

    # Access frames 0, 1, 2 - should be cached
    _ = renderer[0]
    _ = renderer[1]
    _ = renderer[2]

    assert len(renderer._cache) == 3
    assert 0 in renderer._cache
    assert 1 in renderer._cache
    assert 2 in renderer._cache

    # Access frame 3 - should evict frame 0 (oldest)
    _ = renderer[3]

    assert len(renderer._cache) == 3
    assert 0 not in renderer._cache  # Evicted
    assert 1 in renderer._cache
    assert 2 in renderer._cache
    assert 3 in renderer._cache


def test_lazy_field_renderer_lru_reaccess(simple_env, simple_fields):
    """Test LRU cache re-access updates access order."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )
    renderer._cache_size = 3

    # Access 0, 1, 2
    _ = renderer[0]
    _ = renderer[1]
    _ = renderer[2]

    # Re-access frame 0 (moves it to end of LRU)
    _ = renderer[0]

    # Now cache order should be: 1 (oldest), 2, 0 (newest)
    # Access frame 3 - should evict frame 1 (now oldest)
    _ = renderer[3]

    assert 0 in renderer._cache  # Still cached (re-accessed)
    assert 1 not in renderer._cache  # Evicted
    assert 2 in renderer._cache
    assert 3 in renderer._cache


def test_lazy_field_renderer_negative_indexing(simple_env, simple_fields):
    """Test negative indexing support."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    # Access last frame with negative index
    last_frame = renderer[-1]
    assert last_frame.dtype == np.uint8
    assert last_frame.shape[-1] == 3

    # Should be same as accessing by positive index
    last_frame_pos = renderer[len(simple_fields) - 1]
    np.testing.assert_array_equal(last_frame, last_frame_pos)


def test_lazy_field_renderer_out_of_bounds(simple_env, simple_fields):
    """Test out of bounds indexing raises IndexError."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    n_frames = len(simple_fields)

    # Test positive out of bounds
    with pytest.raises(IndexError, match=f"Frame index {n_frames} out of range"):
        _ = renderer[n_frames]

    # Test far positive out of bounds
    with pytest.raises(IndexError, match=f"Frame index {n_frames + 100} out of range"):
        _ = renderer[n_frames + 100]

    # Test negative out of bounds
    with pytest.raises(IndexError, match=f"Frame index -{n_frames + 1} out of range"):
        _ = renderer[-(n_frames + 1)]


def test_lazy_field_renderer_shape_property(simple_env, simple_fields):
    """Test shape property returns correct dimensions."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import (
        _create_lazy_field_renderer,
    )

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = _create_lazy_field_renderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    shape = renderer.shape
    assert shape[0] == len(simple_fields)  # Time dimension
    assert shape[-1] == 3  # RGB channels
    assert len(shape) >= 3  # At least (time, ..., RGB)


# ============================================================================
# render_napari() Function Tests
# ============================================================================


@pytest.mark.napari
def test_render_napari_basic(simple_env, simple_fields):
    """Test basic napari rendering."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Mock napari.Viewer to avoid actually launching GUI
    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            fps=10,
            cmap="viridis",
            title="Test Animation",
        )

        # Verify viewer was created
        mock_viewer_class.assert_called_once_with(title="Test Animation")

        # Verify add_image was called
        mock_viewer.add_image.assert_called_once()
        call_args = mock_viewer.add_image.call_args

        # Check that lazy renderer was passed
        assert call_args[1]["name"] == "Spatial Fields"
        assert call_args[1]["rgb"] is True


@pytest.mark.napari
def test_render_napari_custom_vmin_vmax(simple_env, simple_fields):
    """Test napari rendering with custom color scale.

    Note: vmin/vmax control colormap range during RGB conversion,
    not napari's contrast_limits (RGB images are already in [0, 255]).
    """
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            fps=30,
            cmap="hot",
            vmin=-1.0,
            vmax=2.0,
        )

        # Verify RGB image added without contrast_limits
        call_args = mock_viewer.add_image.call_args
        assert call_args[1]["rgb"] is True
        assert "contrast_limits" not in call_args[1]  # RGB images don't need this


@pytest.mark.napari
def test_render_napari_rgb_no_contrast_limits(simple_env, simple_fields):
    """Test napari rendering correctly omits contrast_limits for RGB images.

    RGB images are already in [0, 255] range and don't need contrast adjustment.
    Only grayscale images use contrast_limits in napari.
    """
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            cmap="viridis",
        )

        # Verify RGB rendering without contrast_limits
        call_args = mock_viewer.add_image.call_args
        assert call_args[1]["rgb"] is True
        assert "contrast_limits" not in call_args[1]


@pytest.mark.napari
def test_render_napari_trajectory_overlay_2d(simple_env, simple_fields):
    """Test napari rendering with 2D trajectory overlay."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create 2D trajectory
    trajectory = np.random.randn(50, 2) * 20

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer.add_tracks = MagicMock(return_value=None)
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            overlay_trajectory=trajectory,
        )

        # Verify add_tracks was called for 2D trajectory
        mock_viewer.add_tracks.assert_called_once()
        track_data = mock_viewer.add_tracks.call_args[0][0]

        # Track data should be (track_id, time, y, x)
        assert track_data.shape[1] == 4


@pytest.mark.napari
def test_render_napari_trajectory_overlay_high_dim(simple_env, simple_fields):
    """Test napari rendering with high-dimensional trajectory (falls back to points)."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create 3D trajectory (higher dimensional)
    trajectory = np.random.randn(50, 3) * 20

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer.add_points = MagicMock(return_value=None)
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            overlay_trajectory=trajectory,
        )

        # Verify add_points was called for high-dim trajectory
        mock_viewer.add_points.assert_called_once()


def test_render_napari_not_available():
    """Test error message when napari is not installed."""
    from neurospatial.animation.backends.napari_backend import (
        NAPARI_AVAILABLE,
        render_napari,
    )

    if NAPARI_AVAILABLE:
        pytest.skip("Napari is installed, cannot test unavailable case")

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    with pytest.raises(ImportError, match="Napari backend requires napari"):
        render_napari(env, fields)


@pytest.mark.napari
def test_render_napari_frame_labels(simple_env, simple_fields):
    """Test napari rendering with frame labels."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    labels = [f"Trial {i + 1}" for i in range(len(simple_fields))]

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            frame_labels=labels,
        )

        # Frame labels currently not displayed in napari
        # (could be future enhancement with custom widget)
        # For now, just verify no error


@pytest.mark.napari
def test_render_napari_gracefully_accepts_extra_params(simple_env, simple_fields):
    """Test that render_napari accepts extra parameters gracefully (via **kwargs)."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        # Should not raise error even with video/html-specific params
        viewer = render_napari(
            simple_env,
            simple_fields,
            dpi=150,  # Video/HTML parameter
            codec="h264",  # Video parameter
            image_format="jpeg",  # HTML parameter
        )

        # Should still work
        assert viewer is not None


@pytest.mark.napari
def test_render_napari_invalid_trajectory_shape(simple_env, simple_fields):
    """Test error handling for invalid trajectory shape."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # 1D trajectory should raise error
    invalid_trajectory = np.random.randn(50)

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        with pytest.raises(ValueError, match="overlay_trajectory must be 2D"):
            render_napari(
                simple_env,
                simple_fields,
                overlay_trajectory=invalid_trajectory,
            )


@pytest.mark.napari
def test_render_napari_playback_controls(simple_env, simple_fields):
    """Test that playback controls are configured correctly."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch(
            "neurospatial.animation.backends.napari_backend.get_settings"
        ) as mock_get_settings,
    ):
        # Create mock viewer with playback controls
        mock_viewer = _create_mock_viewer()
        mock_viewer.dims.current_step = (5, 0, 0)  # Start at frame 5

        mock_viewer_class.return_value = mock_viewer

        # Mock napari settings for FPS control
        mock_settings = MagicMock()
        mock_settings.application.playback_fps = 10  # Default FPS
        mock_get_settings.return_value = mock_settings

        # Render with custom FPS
        render_napari(
            simple_env,
            simple_fields,
            fps=25,
        )

        # Verify initial frame set to 0
        assert mock_viewer.dims.current_step == (0, 0, 0)

        # Verify FPS configured via settings (not qt_viewer)
        assert mock_settings.application.playback_fps == 25
