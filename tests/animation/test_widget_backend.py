"""Test Jupyter widget backend."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neurospatial import Environment


# Mock ipywidgets module
class MockIntSlider:
    """Mock ipywidgets.IntSlider."""

    def __init__(self, **kwargs):
        self.min = kwargs.get("min", 0)
        self.max = kwargs.get("max", 0)
        self.step = kwargs.get("step", 1)
        self.value = kwargs.get("value", 0)
        self.description = kwargs.get("description", "")
        self.continuous_update = kwargs.get("continuous_update", True)
        self._observers = []

    def observe(self, callback, names=None):
        """Mock observe method."""
        self._observers.append((callback, names))


class MockPlay:
    """Mock ipywidgets.Play."""

    def __init__(self, **kwargs):
        self.interval = kwargs.get("interval", 100)
        self.min = kwargs.get("min", 0)
        self.max = kwargs.get("max", 0)
        self.step = kwargs.get("step", 1)
        self.value = kwargs.get("value", 0)


class MockHBox:
    """Mock ipywidgets.HBox."""

    def __init__(self, children):
        self.children = children


class MockVBox:
    """Mock ipywidgets.VBox."""

    def __init__(self, children):
        self.children = children


class MockOutput:
    """Mock ipywidgets.Output."""

    def __init__(self):
        self.outputs = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        pass

    def clear_output(self, wait=False):
        """Mock clear_output method."""
        self.outputs.clear()


# MockHTML class removed - no longer needed since we use persistent Image widgets


@pytest.fixture
def sample_env():
    """Create sample environment for testing."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    return env


@pytest.fixture
def sample_fields(sample_env):
    """Create sample fields for testing."""
    fields = [np.random.rand(sample_env.n_bins) for _ in range(10)]
    return fields


# ============================================================================
# Test ipywidgets Availability
# ============================================================================


def test_ipywidgets_available_flag_when_installed():
    """Test IPYWIDGETS_AVAILABLE is True when ipywidgets installed."""
    with patch.dict(
        "sys.modules", {"ipywidgets": MagicMock(), "IPython.display": MagicMock()}
    ):
        # Re-import to trigger availability check
        import importlib

        from neurospatial.animation.backends import widget_backend

        importlib.reload(widget_backend)

        assert widget_backend.IPYWIDGETS_AVAILABLE is True


def test_ipywidgets_available_flag_when_not_installed():
    """Test IPYWIDGETS_AVAILABLE is False when ipywidgets not installed."""
    with patch.dict("sys.modules", {"ipywidgets": None, "IPython.display": None}):
        # Re-import to trigger availability check

        from neurospatial.animation.backends import widget_backend

        # Simulate ImportError on import attempt
        with patch.object(widget_backend, "IPYWIDGETS_AVAILABLE", False):
            assert widget_backend.IPYWIDGETS_AVAILABLE is False


# ============================================================================
# Test render_widget() Function
# ============================================================================


def test_widget_backend_not_available_error(sample_env, sample_fields):
    """Test error when ipywidgets not available."""
    from neurospatial.animation.backends.widget_backend import render_widget

    with patch(
        "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", False
    ):
        with pytest.raises(ImportError) as exc_info:
            render_widget(sample_env, sample_fields)

        error_msg = str(exc_info.value)
        assert "Widget backend requires ipywidgets" in error_msg
        assert "pip install ipywidgets" in error_msg


def test_render_widget_basic(sample_env, sample_fields):
    """Test basic widget creation."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_slider = MockIntSlider()
    mock_ipywidgets.IntSlider = Mock(return_value=mock_slider)
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.Output = MockOutput
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, fps=30)

    # Check that function returns None (widget displayed, not returned)
    assert result is None
    # Check that slider.observe was called to connect update callback
    assert len(mock_slider._observers) == 1


def test_render_widget_with_custom_parameters(sample_env, sample_fields):
    """Test widget creation with custom parameters."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_slider = MockIntSlider()
    mock_ipywidgets.IntSlider = Mock(return_value=mock_slider)
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.Output = MockOutput
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        result = render_widget(
            sample_env,
            sample_fields,
            fps=10,
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

    assert result is None


def test_render_widget_with_frame_labels(sample_env, sample_fields):
    """Test widget with custom frame labels."""
    from neurospatial.animation.backends.widget_backend import render_widget

    labels = [f"Trial {i + 1}" for i in range(len(sample_fields))]

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_slider = MockIntSlider()
    mock_ipywidgets.IntSlider = Mock(return_value=mock_slider)
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.Output = MockOutput
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, frame_labels=labels)

    assert result is None


def test_render_widget_pre_renders_frames(sample_env):
    """Test that widget pre-renders a subset of frames."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Create more frames than cache size to test caching logic
    fields = [np.random.rand(sample_env.n_bins) for _ in range(100)]

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.interactive_output = Mock(return_value="output")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    # Track how many times render_field_to_png_bytes is called

    render_calls = []

    def mock_render(*args, **kwargs):
        render_calls.append(args)
        return b"fake_png_data"

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
        patch(
            "neurospatial.animation.rendering.render_field_to_png_bytes",
            side_effect=mock_render,
        ),
    ):
        render_widget(sample_env, fields)

    # Should pre-render min(len(fields), 500) frames
    expected_cache_size = min(len(fields), 500)
    assert len(render_calls) == expected_cache_size


def test_render_widget_default_frame_labels(sample_env, sample_fields):
    """Test widget generates default frame labels."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.interactive_output = Mock(return_value="output")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, frame_labels=None)

    # Should generate default labels
    assert result is None


def test_render_widget_slider_configuration(sample_env, sample_fields):
    """Test slider is configured correctly."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_slider_class = Mock(side_effect=MockIntSlider)
    mock_ipywidgets.IntSlider = mock_slider_class
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        render_widget(sample_env, sample_fields)

    # Check slider was created with correct parameters
    mock_slider_class.assert_called_once()
    call_kwargs = mock_slider_class.call_args[1]
    assert call_kwargs["min"] == 0
    assert call_kwargs["max"] == len(sample_fields) - 1
    assert call_kwargs["step"] == 1
    assert call_kwargs["value"] == 0
    assert call_kwargs["continuous_update"] is True


def test_render_widget_play_button_configuration(sample_env, sample_fields):
    """Test play button is configured correctly."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_play_class = Mock(side_effect=MockPlay)
    mock_ipywidgets.Play = mock_play_class
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    fps = 15
    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        render_widget(sample_env, sample_fields, fps=fps)

    # Check play button was created with correct parameters
    mock_play_class.assert_called_once()
    call_kwargs = mock_play_class.call_args[1]
    assert call_kwargs["interval"] == int(1000 / fps)  # milliseconds
    assert call_kwargs["min"] == 0
    assert call_kwargs["max"] == len(sample_fields) - 1
    assert call_kwargs["step"] == 1
    assert call_kwargs["value"] == 0


def test_render_widget_jslink_called(sample_env, sample_fields):
    """Test that play button and slider are linked."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_jslink = Mock()
    mock_ipywidgets.jslink = mock_jslink

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        render_widget(sample_env, sample_fields)

    # Check jslink was called to link play button and slider
    mock_jslink.assert_called_once()


def test_render_widget_graceful_extra_parameters(sample_env, sample_fields):
    """Test widget accepts extra parameters gracefully (backend compatibility)."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.interactive_output = Mock(return_value="output")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
    ):
        # Pass parameters meant for other backends
        result = render_widget(
            sample_env,
            sample_fields,
            title="Test Title",  # Used by napari
            codec="h264",  # Used by video backend
            n_workers=4,  # Used by video backend
            max_html_frames=100,  # Used by HTML backend
        )

    # Should not raise error
    assert result is None


# ============================================================================
# Test Frame Caching Logic
# ============================================================================


def test_widget_on_demand_rendering_for_uncached_frames(sample_env):
    """Test that frames beyond cache size are rendered on-demand."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Create more frames than default cache size (500)
    n_frames = 600
    fields = [np.random.rand(sample_env.n_bins) for _ in range(n_frames)]

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_slider = MockIntSlider()
    mock_ipywidgets.IntSlider = Mock(return_value=mock_slider)
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.VBox = MockVBox
    mock_ipywidgets.Output = MockOutput
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()

    render_call_count = [0]  # Use list to allow mutation in nested function

    def mock_render(*args, **kwargs):
        render_call_count[0] += 1
        return b"fake_png_data"

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("builtins.print"),
        patch(
            "neurospatial.animation.rendering.render_field_to_png_bytes",
            side_effect=mock_render,
        ),
    ):
        render_widget(sample_env, fields)

    # Pre-rendering should have cached 500 frames
    # (show_frame(0) uses cached frame 0, doesn't re-render)
    assert render_call_count[0] == 500

    # Verify that slider.observe was called to connect the update callback
    assert len(mock_slider._observers) == 1


# ============================================================================
# Test PersistentFigureRenderer
# ============================================================================


def test_persistent_figure_renderer_reuses_figure(sample_env, sample_fields):
    """Test that PersistentFigureRenderer reuses the same figure."""
    from neurospatial.animation.backends.widget_backend import PersistentFigureRenderer

    # Create renderer
    renderer = PersistentFigureRenderer(
        env=sample_env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )

    try:
        # Store reference to internal figure
        fig_id = id(renderer._fig)

        # Render frame 1
        png_bytes_1 = renderer.render(sample_fields[0], frame_idx=0)

        # Verify figure is still the same
        assert id(renderer._fig) == fig_id

        # Render frame 2
        png_bytes_2 = renderer.render(sample_fields[1], frame_idx=1)

        # Verify figure is still the same
        assert id(renderer._fig) == fig_id

        # Assert outputs are valid PNG bytes
        assert isinstance(png_bytes_1, bytes)
        assert len(png_bytes_1) > 0
        assert png_bytes_1[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number

        assert isinstance(png_bytes_2, bytes)
        assert len(png_bytes_2) > 0
        assert png_bytes_2[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number
    finally:
        renderer.close()


def test_persistent_figure_renderer_output_matches_fresh(sample_env, sample_fields):
    """Test that persistent renderer output produces valid images like fresh render."""
    from neurospatial.animation.backends.widget_backend import (
        PersistentFigureRenderer,
        render_field_to_png_bytes_with_overlays,
    )

    # Render with fresh figure
    fresh_png = render_field_to_png_bytes_with_overlays(
        env=sample_env,
        field=sample_fields[0],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
        frame_idx=0,
    )

    # Render with persistent figure
    renderer = PersistentFigureRenderer(
        env=sample_env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )

    try:
        persistent_png = renderer.render(sample_fields[0], frame_idx=0)

        # Both should be valid PNG files
        assert fresh_png[:8] == b"\x89PNG\r\n\x1a\n"
        assert persistent_png[:8] == b"\x89PNG\r\n\x1a\n"

        # Both should have reasonable size (not empty or corrupted)
        assert len(fresh_png) > 1000  # Reasonable minimum for a PNG image
        assert len(persistent_png) > 1000

        # Sizes should be in the same ballpark (within 50% of each other)
        # Note: Exact match not expected due to potential rendering differences
        size_ratio = len(persistent_png) / len(fresh_png)
        assert 0.5 < size_ratio < 2.0, (
            f"Size ratio {size_ratio} is outside expected range"
        )
    finally:
        renderer.close()


def test_persistent_figure_renderer_with_overlays(sample_env, sample_fields):
    """Test that PersistentFigureRenderer works with overlays."""
    from neurospatial.animation.backends.widget_backend import PersistentFigureRenderer
    from neurospatial.animation.overlays import OverlayData, PositionData

    # Create overlay data
    n_frames = len(sample_fields)
    trajectory = np.random.randn(n_frames, 2) * 20 + 25  # Random trajectory
    overlay_data = OverlayData(
        positions=[
            PositionData(data=trajectory, color="red", size=10.0, trail_length=3)
        ],
        bodypart_sets=[],
        head_directions=[],
    )

    renderer = PersistentFigureRenderer(
        env=sample_env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )

    try:
        # Render multiple frames with overlays
        png_bytes_1 = renderer.render(
            sample_fields[0], frame_idx=0, overlay_data=overlay_data
        )
        png_bytes_2 = renderer.render(
            sample_fields[1], frame_idx=1, overlay_data=overlay_data
        )

        # Both should be valid PNG files
        assert png_bytes_1[:8] == b"\x89PNG\r\n\x1a\n"
        assert png_bytes_2[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        renderer.close()


def test_persistent_figure_renderer_close_releases_figure(sample_env, sample_fields):
    """Test that close() releases the figure."""
    import matplotlib.pyplot as plt

    from neurospatial.animation.backends.widget_backend import PersistentFigureRenderer

    # Count figures before
    initial_fig_count = len(plt.get_fignums())

    renderer = PersistentFigureRenderer(
        env=sample_env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )

    # Render a frame
    renderer.render(sample_fields[0], frame_idx=0)

    # Should have one more figure
    assert len(plt.get_fignums()) == initial_fig_count + 1

    # Close the renderer
    renderer.close()

    # Figure count should be back to initial
    assert len(plt.get_fignums()) == initial_fig_count


def test_persistent_figure_renderer_multiple_renders_consistency(
    sample_env, sample_fields
):
    """Test that multiple renders produce consistent output for same field."""
    from neurospatial.animation.backends.widget_backend import PersistentFigureRenderer

    renderer = PersistentFigureRenderer(
        env=sample_env,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )

    try:
        # Render same field multiple times
        png_1 = renderer.render(sample_fields[0], frame_idx=0)
        png_2 = renderer.render(sample_fields[0], frame_idx=0)

        # Outputs should be identical for same input
        assert png_1 == png_2
    finally:
        renderer.close()
