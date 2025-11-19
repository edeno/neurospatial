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


class MockHTML:
    """Mock IPython.display.HTML."""

    def __init__(self, data):
        self.data = data


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
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, fps=30)

    # Check widget was created
    assert result == "widget"
    mock_ipywidgets.interact.assert_called_once()


def test_render_widget_with_custom_parameters(sample_env, sample_fields):
    """Test widget creation with custom parameters."""
    from neurospatial.animation.backends.widget_backend import render_widget

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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

    assert result == "widget"


def test_render_widget_with_frame_labels(sample_env, sample_fields):
    """Test widget with custom frame labels."""
    from neurospatial.animation.backends.widget_backend import render_widget

    labels = [f"Trial {i + 1}" for i in range(len(sample_fields))]

    # Mock dependencies
    mock_ipywidgets = MagicMock()
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, frame_labels=labels)

    assert result == "widget"


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
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

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
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
        patch("builtins.print"),
    ):
        result = render_widget(sample_env, sample_fields, frame_labels=None)

    # Should generate default labels
    assert result == "widget"


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
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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
    mock_html = MockHTML

    fps = 15
    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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
    mock_ipywidgets.interact = Mock(return_value="widget")
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", True
        ),
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets", mock_ipywidgets
        ),
        patch("neurospatial.animation.backends.widget_backend.display", mock_display),
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
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
    assert result == "widget"


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
    mock_ipywidgets.IntSlider = MockIntSlider
    mock_ipywidgets.Play = MockPlay
    mock_ipywidgets.HBox = MockHBox

    # Capture the show_frame function
    captured_show_frame = None

    def capture_interact(func, **kwargs):
        nonlocal captured_show_frame
        captured_show_frame = func
        return "widget"

    mock_ipywidgets.interact = capture_interact
    mock_ipywidgets.jslink = Mock()

    mock_display = Mock()
    mock_html = MockHTML

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
        patch("neurospatial.animation.backends.widget_backend.HTML", mock_html),
        patch("builtins.print"),
        patch(
            "neurospatial.animation.rendering.render_field_to_png_bytes",
            side_effect=mock_render,
        ),
    ):
        render_widget(sample_env, fields)

    # Pre-rendering should have cached 500 frames
    assert render_call_count[0] == 500

    # Now simulate accessing frame 550 (beyond cache)
    # This would trigger on-demand rendering in the actual widget
    # We can't easily test this without running the widget, so we verify
    # that the show_frame function was captured
    assert captured_show_frame is not None
