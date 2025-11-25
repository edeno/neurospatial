"""Tests for widget backend overlay rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.backends.widget_backend import render_widget
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)
from neurospatial.animation.skeleton import Skeleton


@pytest.fixture
def simple_env():
    """Create a simple 2D environment for testing."""
    positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    return env


@pytest.fixture
def simple_fields(simple_env):
    """Create simple test fields."""
    rng = np.random.default_rng(42)
    n_frames = 10
    return [rng.random(simple_env.n_bins) for _ in range(n_frames)]


@pytest.fixture
def position_overlay_data():
    """Create test position overlay data."""
    rng = np.random.default_rng(42)
    n_frames = 10
    positions = rng.random((n_frames, 2)) * 10
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=3)]
    )


@pytest.fixture
def bodypart_overlay_data():
    """Create test bodypart overlay data."""
    rng = np.random.default_rng(42)
    n_frames = 10
    nose = rng.random((n_frames, 2)) * 10
    tail = rng.random((n_frames, 2)) * 10
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    return OverlayData(
        bodypart_sets=[
            BodypartData(
                bodyparts={"nose": nose, "tail": tail},
                skeleton=skeleton,
                colors={"nose": "red", "tail": "blue"},
            )
        ]
    )


@pytest.fixture
def head_direction_overlay_data():
    """Create test head direction overlay data."""
    rng = np.random.default_rng(42)
    n_frames = 10
    angles = rng.random(n_frames) * 2 * np.pi
    return OverlayData(
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=5.0)]
    )


@pytest.fixture
def all_overlays_data():
    """Create overlay data with all types."""
    rng = np.random.default_rng(42)
    n_frames = 10
    positions = rng.random((n_frames, 2)) * 10
    nose = rng.random((n_frames, 2)) * 10
    tail = rng.random((n_frames, 2)) * 10
    angles = rng.random(n_frames) * 2 * np.pi

    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    return OverlayData(
        positions=[
            PositionData(data=positions, color="red", size=10.0, trail_length=3)
        ],
        bodypart_sets=[
            BodypartData(
                bodyparts={"nose": nose, "tail": tail},
                skeleton=skeleton,
                colors={"nose": "red", "tail": "blue"},
            )
        ],
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=5.0)],
    )


# ============================================================================
# Parameter Acceptance Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
def test_render_widget_accepts_overlay_data_parameter(
    mock_ipywidgets, mock_display, simple_env, simple_fields, position_overlay_data
):
    """Test render_widget accepts overlay_data parameter."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # This should not raise an error
    render_widget(
        simple_env,
        simple_fields,
        overlay_data=position_overlay_data,
        initial_cache_size=5,
    )

    # Verify display was called
    assert mock_display.called


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
def test_render_widget_accepts_show_regions_parameter(
    mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test render_widget accepts show_regions parameter."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # This should not raise an error
    render_widget(
        simple_env,
        simple_fields,
        show_regions=True,
        initial_cache_size=5,
    )

    # Verify display was called
    assert mock_display.called


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
def test_render_widget_accepts_region_alpha_parameter(
    mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test render_widget accepts region_alpha parameter."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # This should not raise an error
    render_widget(
        simple_env,
        simple_fields,
        region_alpha=0.5,
        initial_cache_size=5,
    )

    # Verify display was called
    assert mock_display.called


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
def test_render_widget_without_overlays_still_works(
    mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test render_widget works without overlays (backward compatibility)."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Should work without overlay parameters
    render_widget(simple_env, simple_fields, initial_cache_size=5)

    # Verify display was called
    assert mock_display.called


# ============================================================================
# Overlay Rendering Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_position_overlay_renders_correctly(
    mock_render_func,
    mock_ipywidgets,
    mock_display,
    simple_env,
    simple_fields,
    position_overlay_data,
):
    """Test position overlay rendering calls correct function."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=position_overlay_data,
        initial_cache_size=5,
    )

    # Verify render function was called with overlay_data
    assert mock_render_func.called
    # Check that overlay_data was passed to render function
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["overlay_data"] == position_overlay_data


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_bodypart_overlay_renders_correctly(
    mock_render_func,
    mock_ipywidgets,
    mock_display,
    simple_env,
    simple_fields,
    bodypart_overlay_data,
):
    """Test bodypart overlay rendering calls correct function."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=bodypart_overlay_data,
        initial_cache_size=5,
    )

    # Verify render function was called with overlay_data
    assert mock_render_func.called
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["overlay_data"] == bodypart_overlay_data


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_head_direction_overlay_renders_correctly(
    mock_render_func,
    mock_ipywidgets,
    mock_display,
    simple_env,
    simple_fields,
    head_direction_overlay_data,
):
    """Test head direction overlay rendering calls correct function."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=head_direction_overlay_data,
        initial_cache_size=5,
    )

    # Verify render function was called with overlay_data
    assert mock_render_func.called
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["overlay_data"] == head_direction_overlay_data


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_all_overlays_render_correctly(
    mock_render_func,
    mock_ipywidgets,
    mock_display,
    simple_env,
    simple_fields,
    all_overlays_data,
):
    """Test all overlay types render together."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=all_overlays_data,
        initial_cache_size=5,
    )

    # Verify render function was called with all overlay types
    assert mock_render_func.called
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["overlay_data"] == all_overlays_data


# ============================================================================
# Region Rendering Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_show_regions_passed_to_render_function(
    mock_render_func, mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test show_regions parameter passed to render function."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        show_regions=True,
        initial_cache_size=5,
    )

    # Verify show_regions was passed
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["show_regions"] is True


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_region_alpha_passed_to_render_function(
    mock_render_func, mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test region_alpha parameter passed to render function."""
    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function to return fake PNG bytes
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        show_regions=True,  # Need to enable regions for region_alpha to matter
        region_alpha=0.5,
        initial_cache_size=5,
    )

    # Verify region_alpha was passed
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert call_kwargs["region_alpha"] == 0.5


# ============================================================================
# LRU Cache Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_lru_cache_works_with_overlays(
    mock_render_func,
    mock_ipywidgets,
    mock_display,
    simple_env,
    simple_fields,
    position_overlay_data,
):
    """Test LRU cache works correctly with overlays."""
    # Mock ipywidgets components
    mock_image = MagicMock()
    mock_html = MagicMock()
    mock_slider = MagicMock()
    mock_play = MagicMock()
    mock_vbox = MagicMock()
    mock_hbox = MagicMock()

    mock_ipywidgets.Image.return_value = mock_image
    mock_ipywidgets.HTML.return_value = mock_html
    mock_ipywidgets.IntSlider.return_value = mock_slider
    mock_ipywidgets.Play.return_value = mock_play
    mock_ipywidgets.VBox.return_value = mock_vbox
    mock_ipywidgets.HBox.return_value = mock_hbox
    mock_ipywidgets.jslink = MagicMock()

    # Mock render function to return unique PNG bytes per frame
    call_count = [0]

    def render_side_effect(*args, **kwargs):
        call_count[0] += 1
        return f"png_frame_{call_count[0]}".encode()

    mock_render_func.side_effect = render_side_effect

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=position_overlay_data,
        initial_cache_size=5,
    )

    # Verify render function was called 5 times for initial cache
    assert mock_render_func.call_count == 5


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_cache_limit_respected_with_overlays(
    mock_render_func, mock_ipywidgets, mock_display, simple_env, position_overlay_data
):
    """Test cache limit is respected when rendering overlays."""
    rng = np.random.default_rng(42)
    # Create more frames than cache limit
    n_frames = 20
    fields = [rng.random(simple_env.n_bins) for _ in range(n_frames)]

    # Mock ipywidgets components
    mock_image = MagicMock()
    mock_html = MagicMock()
    mock_slider = MagicMock()
    mock_play = MagicMock()
    mock_vbox = MagicMock()
    mock_hbox = MagicMock()

    mock_ipywidgets.Image.return_value = mock_image
    mock_ipywidgets.HTML.return_value = mock_html
    mock_ipywidgets.IntSlider.return_value = mock_slider
    mock_ipywidgets.Play.return_value = mock_play
    mock_ipywidgets.VBox.return_value = mock_vbox
    mock_ipywidgets.HBox.return_value = mock_hbox
    mock_ipywidgets.jslink = MagicMock()

    # Mock render function
    mock_render_func.return_value = b"fake_png_data"

    # Render with small cache limit
    render_widget(
        simple_env,
        fields,
        overlay_data=position_overlay_data,
        initial_cache_size=10,
        cache_limit=10,
    )

    # Verify initial cache was created (10 frames)
    assert mock_render_func.call_count == 10


# ============================================================================
# Multi-Overlay Tests
# ============================================================================


@patch("neurospatial.animation.backends.widget_backend.display")
@patch("neurospatial.animation.backends.widget_backend.ipywidgets")
@patch(
    "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
)
def test_multiple_position_overlays(
    mock_render_func, mock_ipywidgets, mock_display, simple_env, simple_fields
):
    """Test multiple position overlays (multi-animal) render correctly."""
    rng = np.random.default_rng(42)
    # Create multi-animal position data
    n_frames = 10
    animal1_positions = rng.random((n_frames, 2)) * 10
    animal2_positions = rng.random((n_frames, 2)) * 10

    multi_overlay_data = OverlayData(
        positions=[
            PositionData(
                data=animal1_positions, color="red", size=10.0, trail_length=3
            ),
            PositionData(
                data=animal2_positions, color="blue", size=10.0, trail_length=3
            ),
        ]
    )

    # Mock ipywidgets components
    mock_ipywidgets.Image = MagicMock
    mock_ipywidgets.HTML = MagicMock
    mock_ipywidgets.IntSlider = MagicMock
    mock_ipywidgets.Play = MagicMock
    mock_ipywidgets.VBox = MagicMock
    mock_ipywidgets.HBox = MagicMock
    mock_ipywidgets.jslink = MagicMock

    # Mock render function
    mock_render_func.return_value = b"fake_png_data"

    render_widget(
        simple_env,
        simple_fields,
        overlay_data=multi_overlay_data,
        initial_cache_size=5,
    )

    # Verify render function received multi-animal overlay data
    assert mock_render_func.called
    call_kwargs = mock_render_func.call_args_list[0][1]
    assert len(call_kwargs["overlay_data"].positions) == 2
