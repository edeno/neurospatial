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

# Mark napari GUI tests to run in same worker (prevent Qt crashes)
pytestmark = pytest.mark.xdist_group(name="napari_gui")


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
    import importlib
    import sys

    # Save original modules
    original_napari = sys.modules.get("napari")
    original_backend = sys.modules.get("neurospatial.animation.backends.napari_backend")

    try:
        # Mock napari import to raise ImportError
        with patch.dict("sys.modules", {"napari": None}):
            # Delete backend module WITHIN patch context to force re-import with mocked napari
            if "neurospatial.animation.backends.napari_backend" in sys.modules:
                del sys.modules["neurospatial.animation.backends.napari_backend"]

            # Import with patched napari (avoids reload() which has test isolation issues)
            napari_backend = importlib.import_module(
                "neurospatial.animation.backends.napari_backend"
            )

            # Should be False when napari is not available
            assert hasattr(napari_backend, "NAPARI_AVAILABLE")
            assert napari_backend.NAPARI_AVAILABLE is False

    finally:
        # Restore original state
        if original_napari is not None:
            sys.modules["napari"] = original_napari
        elif "napari" in sys.modules:
            del sys.modules["napari"]

        if original_backend is not None:
            sys.modules["neurospatial.animation.backends.napari_backend"] = (
                original_backend
            )
        elif "neurospatial.animation.backends.napari_backend" in sys.modules:
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
    """Test napari rendering with frame labels in enhanced playback widget."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    labels = [f"Trial {i + 1}" for i in range(len(simple_fields))]

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch(
            "neurospatial.animation.backends.napari_backend._add_speed_control_widget"
        ) as mock_add_widget,
    ):
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        render_napari(
            simple_env,
            simple_fields,
            frame_labels=labels,
            fps=25,
        )

        # Verify enhanced playback widget was called with frame_labels
        mock_add_widget.assert_called_once()
        call_args = mock_add_widget.call_args

        # Check positional arg (viewer)
        assert call_args[0][0] == mock_viewer

        # Check keyword args (initial_fps and frame_labels)
        assert call_args[1]["initial_fps"] == 25
        assert call_args[1]["frame_labels"] == labels


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
def test_render_napari_playback_controls(simple_env, simple_fields):
    """Test that playback controls are configured correctly."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch("napari.settings.get_settings") as mock_get_settings,
        patch(
            "neurospatial.animation.backends.napari_backend._add_speed_control_widget"
        ),
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


@pytest.mark.napari
def test_speed_control_widget_added(simple_env, simple_fields):
    """Test that interactive speed control widget is added to viewer."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch("napari.settings.get_settings") as mock_get_settings,
        patch(
            "neurospatial.animation.backends.napari_backend._add_speed_control_widget"
        ) as mock_add_widget,
    ):
        # Create mock viewer
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        # Mock napari settings
        mock_settings = MagicMock()
        mock_settings.application.playback_fps = 10
        mock_get_settings.return_value = mock_settings

        # Render napari viewer
        render_napari(simple_env, simple_fields, fps=30)

        # Verify speed control widget was called (with frame_labels=None by default)
        mock_add_widget.assert_called_once_with(
            mock_viewer, initial_fps=30, frame_labels=None
        )


@pytest.mark.napari
def test_speed_control_widget_graceful_fallback(simple_env, simple_fields):
    """Test that napari backend works even if magicgui is not available."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import (
        _add_speed_control_widget,
    )

    # Create mock viewer
    mock_viewer = _create_mock_viewer()
    mock_viewer.window = MagicMock()
    mock_viewer.window.add_dock_widget = MagicMock()

    # Mock magicgui import failure
    with patch(
        "builtins.__import__", side_effect=ImportError("magicgui not available")
    ):
        # Should not raise error - gracefully handles missing magicgui
        _add_speed_control_widget(mock_viewer, initial_fps=30)

        # Widget should not have been added
        mock_viewer.window.add_dock_widget.assert_not_called()


@pytest.mark.napari
def test_speed_control_widget_high_fps(simple_env, simple_fields):
    """Test that speed control widget works with high FPS values (>120)."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch("napari.settings.get_settings") as mock_get_settings,
        patch(
            "neurospatial.animation.backends.napari_backend._add_speed_control_widget"
        ) as mock_add_widget,
    ):
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        # Mock napari settings
        mock_settings = MagicMock()
        mock_settings.application.playback_fps = 10
        mock_get_settings.return_value = mock_settings

        # Render with high FPS (250 Hz - common for neuroscience recordings)
        render_napari(simple_env, simple_fields, fps=250)

        # Verify speed control widget was called with high FPS
        mock_add_widget.assert_called_once_with(
            mock_viewer, initial_fps=250, frame_labels=None
        )


@pytest.mark.napari
def test_spacebar_keyboard_shortcut(simple_env, simple_fields):
    """Test that spacebar is bound to toggle playback."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with (
        patch(
            "neurospatial.animation.backends.napari_backend.napari.Viewer"
        ) as mock_viewer_class,
        patch("napari.settings.get_settings") as mock_get_settings,
    ):
        # Create mock viewer
        mock_viewer = _create_mock_viewer()
        mock_viewer.bind_key = MagicMock()  # Mock bind_key method
        mock_viewer_class.return_value = mock_viewer

        # Mock napari settings
        mock_settings = MagicMock()
        mock_settings.application.playback_fps = 10
        mock_get_settings.return_value = mock_settings

        # Render napari viewer (spacebar bound inside _add_speed_control_widget)
        render_napari(simple_env, simple_fields, fps=30)

        # Verify spacebar was bound (inside _add_speed_control_widget or fallback)
        mock_viewer.bind_key.assert_called_once()
        # Get the call arguments
        call_args = mock_viewer.bind_key.call_args
        # Check that "Space" was passed as the key
        assert call_args[0][0] == "Space"


# ============================================================================
# Unknown kwargs warning test
# ============================================================================


@pytest.mark.napari
def test_render_napari_warns_on_unknown_kwargs(simple_env, simple_fields):
    """Test that render_napari emits warning for unknown kwargs."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        # Should emit UserWarning about unknown kwargs
        with pytest.warns(UserWarning, match="unknown keyword arguments"):
            render_napari(
                simple_env,
                simple_fields,
                unknown_param=42,
                another_unknown="test",
            )


@pytest.mark.napari
def test_render_napari_warning_lists_unknown_keys(simple_env, simple_fields):
    """Test that unknown kwargs warning includes the key names."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    with patch(
        "neurospatial.animation.backends.napari_backend.napari.Viewer"
    ) as mock_viewer_class:
        mock_viewer = _create_mock_viewer()
        mock_viewer_class.return_value = mock_viewer

        # Check that specific key names appear in warning (alphabetically sorted)
        with pytest.warns(UserWarning, match="bar_param.*foo_param"):
            render_napari(
                simple_env,
                simple_fields,
                foo_param=1,
                bar_param=2,
            )


# ============================================================================
# _EnvScale class tests
# ============================================================================


def test_env_scale_caches_correct_values(simple_env):
    """Test _EnvScale caches correct values from environment."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    scale = _EnvScale(simple_env)

    # Check bounds match environment
    (x_min, x_max), (y_min, y_max) = simple_env.dimension_ranges
    assert scale.x_min == x_min
    assert scale.x_max == x_max
    assert scale.y_min == y_min
    assert scale.y_max == y_max

    # Check grid shape matches
    n_x, n_y = simple_env.layout.grid_shape
    assert scale.n_x == n_x
    assert scale.n_y == n_y

    # Check scale factors are computed correctly
    expected_x_scale = (n_x - 1) / (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    expected_y_scale = (n_y - 1) / (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    assert scale.x_scale == pytest.approx(expected_x_scale)
    assert scale.y_scale == pytest.approx(expected_y_scale)


def test_env_scale_from_env_returns_none_for_none():
    """Test from_env returns None when env is None."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    assert _EnvScale.from_env(None) is None


def test_env_scale_from_env_returns_none_for_missing_attrs():
    """Test from_env returns None when env lacks required attributes."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    # Object without dimension_ranges
    class FakeEnvNoDimensions:
        pass

    assert _EnvScale.from_env(FakeEnvNoDimensions()) is None

    # Object with dimension_ranges but no layout.grid_shape
    class FakeEnvNoGridShape:
        def __init__(self):
            self.dimension_ranges = [(0, 10), (0, 10)]
            self.layout = object()  # No grid_shape attribute

    assert _EnvScale.from_env(FakeEnvNoGridShape()) is None


def test_env_scale_from_env_returns_scale_for_valid_env(simple_env):
    """Test from_env returns _EnvScale for valid environment."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    scale = _EnvScale.from_env(simple_env)
    assert scale is not None
    assert isinstance(scale, _EnvScale)


def test_make_env_scale_wrapper(simple_env):
    """Test _make_env_scale convenience function."""
    from neurospatial.animation.backends.napari_backend import (
        _EnvScale,
        _make_env_scale,
    )

    # Returns None for None
    assert _make_env_scale(None) is None

    # Returns _EnvScale for valid env
    scale = _make_env_scale(simple_env)
    assert isinstance(scale, _EnvScale)


def test_env_scale_uses_slots():
    """Test _EnvScale uses __slots__ for memory efficiency."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    assert hasattr(_EnvScale, "__slots__")
    assert "x_min" in _EnvScale.__slots__
    assert "x_scale" in _EnvScale.__slots__


def test_env_scale_repr(simple_env):
    """Test _EnvScale __repr__ provides useful debug info."""
    from neurospatial.animation.backends.napari_backend import _EnvScale

    scale = _EnvScale(simple_env)
    repr_str = repr(scale)

    # Should contain class name and key values
    # Note: The public class is named EnvScale (re-exported as _EnvScale in napari_backend)
    assert "EnvScale" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "grid=" in repr_str


# ============================================================================
# __array_priority__ tests
# ============================================================================


def test_lazy_field_renderer_has_array_priority(simple_env, simple_fields):
    """Test LazyFieldRenderer has __array_priority__ attribute."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import LazyFieldRenderer

    # Create colormap lookup
    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = LazyFieldRenderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    assert hasattr(renderer, "__array_priority__")
    assert renderer.__array_priority__ == 1000


def test_chunked_lazy_field_renderer_has_array_priority(simple_env, simple_fields):
    """Test ChunkedLazyFieldRenderer has __array_priority__ attribute."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import ChunkedLazyFieldRenderer

    # Create colormap lookup
    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = ChunkedLazyFieldRenderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    assert hasattr(renderer, "__array_priority__")
    assert renderer.__array_priority__ == 1000


# ============================================================================
# dtype property tests
# ============================================================================


def test_lazy_field_renderer_dtype_returns_np_dtype(simple_env, simple_fields):
    """Test LazyFieldRenderer.dtype returns np.dtype not type."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import LazyFieldRenderer

    # Create colormap lookup
    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = LazyFieldRenderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    # Should be np.dtype, not type
    assert isinstance(renderer.dtype, np.dtype)
    assert renderer.dtype == np.dtype(np.uint8)


def test_chunked_lazy_field_renderer_dtype_returns_np_dtype(simple_env, simple_fields):
    """Test ChunkedLazyFieldRenderer.dtype returns np.dtype not type."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.napari_backend import ChunkedLazyFieldRenderer

    # Create colormap lookup
    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    renderer = ChunkedLazyFieldRenderer(
        simple_env, simple_fields, cmap_lookup, vmin=0.0, vmax=1.0
    )

    # Should be np.dtype, not type
    assert isinstance(renderer.dtype, np.dtype)
    assert renderer.dtype == np.dtype(np.uint8)


# ============================================================================
# Playback Widget Throttling Tests
# ============================================================================


@pytest.mark.napari
def test_playback_widget_scrubbing_updates_immediately():
    """Test frame info updates immediately when scrubbing (not playing).

    When the user is manually scrubbing through frames (playback stopped),
    the frame info should update on every frame change, not be throttled.
    This ensures responsive feedback during manual navigation.
    """
    pytest.importorskip("napari")
    pytest.importorskip("magicgui")

    from unittest.mock import MagicMock, patch

    from neurospatial.animation.backends.napari_backend import _add_speed_control_widget

    # Create mock viewer
    mock_viewer = MagicMock()
    mock_viewer.dims.ndim = 3
    mock_viewer.dims.range = [(0, 100, 100)]  # 100 frames

    # Track frame info updates
    frame_info_values = []

    # We need to capture the update_frame_info callback registered with dims events
    captured_callback = None

    def capture_connect(callback):
        nonlocal captured_callback
        captured_callback = callback

    mock_viewer.dims.events.current_step.connect = capture_connect

    # Mock magicgui to capture the widget and its frame_info attribute
    mock_widget = MagicMock()
    mock_widget.frame_info = MagicMock()

    # Track frame_info.value assignments
    def track_value_assignment(value):
        frame_info_values.append(value)

    mock_widget.frame_info.value = property(
        lambda self: "", lambda self, v: track_value_assignment(v)
    )

    # Use property setter simulation
    mock_widget.frame_info = type(
        "MockFrameInfo",
        (),
        {"value": property(lambda s: "", lambda s, v: frame_info_values.append(v))},
    )()

    with (
        patch("magicgui.magicgui", return_value=lambda f: mock_widget),
        patch("napari.settings.get_settings") as mock_settings,
    ):
        mock_settings.return_value.application.playback_fps = 30
        _add_speed_control_widget(mock_viewer, initial_fps=60)

    # Verify callback was captured
    assert captured_callback is not None, "update_frame_info callback not captured"

    # Clear initial frame info update
    frame_info_values.clear()

    # Simulate scrubbing: manually change frames (is_playing should be False by default)
    # At high FPS (60), update_interval = 60 // 30 = 2, so without fix every other frame skips
    mock_viewer.dims.current_step = (1, 0, 0)
    captured_callback()

    mock_viewer.dims.current_step = (2, 0, 0)
    captured_callback()

    mock_viewer.dims.current_step = (3, 0, 0)
    captured_callback()

    mock_viewer.dims.current_step = (4, 0, 0)
    captured_callback()

    mock_viewer.dims.current_step = (5, 0, 0)
    captured_callback()

    # When scrubbing (not playing), ALL frame changes should result in updates
    # No throttling should occur
    assert len(frame_info_values) == 5, (
        f"Expected 5 updates when scrubbing, got {len(frame_info_values)}. "
        "Scrubbing (not playing) should not be throttled."
    )
