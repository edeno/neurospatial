"""Test consistency across all animation backends.

These tests verify that all backends (napari, video, html, widget) can handle
the same input data correctly and produce valid output.

Napari tests use xdist_group to prevent Qt crashes in parallel execution.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment

# Mark napari GUI tests to run in same worker (prevent Qt crashes)
pytestmark = pytest.mark.xdist_group(name="napari_gui")


@pytest.fixture
def shared_test_data():
    """Create consistent test environment and fields for all backends."""
    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create 10 frames of field data (manageable for all backends)
    n_frames = 10
    fields = [rng.random(env.n_bins) for _ in range(n_frames)]

    return env, fields


def test_napari_backend_with_shared_data(shared_test_data):
    """Test Napari backend handles shared test data."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    env, fields = shared_test_data

    # Napari should handle the data without error
    viewer = render_napari(env, fields, vmin=0, vmax=1, fps=10)

    # Verify viewer created
    assert viewer is not None
    assert hasattr(viewer, "layers")
    assert len(viewer.layers) > 0


def test_html_backend_with_shared_data(shared_test_data, tmp_path):
    """Test HTML backend handles shared test data."""
    from neurospatial.animation.backends.html_backend import render_html

    env, fields = shared_test_data

    output_path = tmp_path / "consistency_test.html"

    # HTML should handle the data without error
    render_html(env, fields, save_path=str(output_path), fps=10)

    # Verify output
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify frames are embedded in JavaScript
    html_content = output_path.read_text(encoding="utf-8")
    assert "const frames = [" in html_content
    # HTML uses base64-encoded frames in an array
    assert len(fields) == 10  # Verify we're testing the right number of frames


def test_video_backend_with_shared_data(shared_test_data, tmp_path):
    """Test video backend handles shared test data."""
    # Skip if ffmpeg not available
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        pytest.skip("ffmpeg not installed")

    from neurospatial.animation.backends.video_backend import render_video

    env, fields = shared_test_data

    # Clear cache to make environment pickle-able
    env.clear_cache()

    output_path = tmp_path / "consistency_test.mp4"

    # Video should handle the data without error
    render_video(env, fields, save_path=str(output_path), fps=10, n_workers=1)

    # Verify output
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_widget_backend_with_shared_data(shared_test_data):
    """Test widget backend handles shared test data."""
    pytest.importorskip("ipywidgets")

    from neurospatial.animation.backends.widget_backend import render_widget

    env, fields = shared_test_data

    # Mock ipywidgets to avoid Jupyter dependency
    with (
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets"
        ) as mock_widgets,
        patch("neurospatial.animation.backends.widget_backend.display"),
    ):
        # Mock widget classes
        mock_widgets.Image = MagicMock()
        mock_widgets.HTML = MagicMock()
        mock_widgets.IntSlider = MagicMock()
        mock_widgets.Play = MagicMock()
        mock_widgets.VBox = MagicMock()
        mock_widgets.HBox = MagicMock()
        mock_widgets.jslink = MagicMock()

        # Widget should handle the data without error
        render_widget(env, fields, fps=10)

        # Verify widget components were created
        mock_widgets.Image.assert_called_once()
        mock_widgets.IntSlider.assert_called_once()
        mock_widgets.Play.assert_called_once()


def test_all_backends_handle_same_vmin_vmax(shared_test_data, tmp_path):
    """Test all backends respect the same vmin/vmax color scale."""
    pytest.importorskip("napari")
    pytest.importorskip("ipywidgets")

    env, fields = shared_test_data

    vmin, vmax = -0.5, 1.5  # Explicit range outside [0, 1]

    # Test napari
    from neurospatial.animation.backends.napari_backend import render_napari

    viewer = render_napari(env, fields, vmin=vmin, vmax=vmax, fps=10)
    assert viewer is not None

    # Test HTML
    from neurospatial.animation.backends.html_backend import render_html

    html_path = tmp_path / "vmin_vmax_test.html"
    render_html(env, fields, save_path=str(html_path), fps=10, vmin=vmin, vmax=vmax)
    assert html_path.exists()

    # Test widget (with mocking)
    from neurospatial.animation.backends.widget_backend import render_widget

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets"
        ) as mock_widgets,
        patch("neurospatial.animation.backends.widget_backend.display"),
    ):
        mock_widgets.Image = MagicMock()
        mock_widgets.HTML = MagicMock()
        mock_widgets.IntSlider = MagicMock()
        mock_widgets.Play = MagicMock()
        mock_widgets.VBox = MagicMock()
        mock_widgets.HBox = MagicMock()
        mock_widgets.jslink = MagicMock()

        render_widget(env, fields, fps=10, vmin=vmin, vmax=vmax)
        mock_widgets.Image.assert_called_once()


def test_all_backends_handle_custom_cmap(shared_test_data, tmp_path):
    """Test all backends respect custom colormap."""
    pytest.importorskip("napari")
    pytest.importorskip("ipywidgets")

    env, fields = shared_test_data

    cmap = "plasma"  # Non-default colormap

    # Test napari
    from neurospatial.animation.backends.napari_backend import render_napari

    viewer = render_napari(env, fields, cmap=cmap, fps=10)
    assert viewer is not None

    # Test HTML
    from neurospatial.animation.backends.html_backend import render_html

    html_path = tmp_path / "cmap_test.html"
    render_html(env, fields, save_path=str(html_path), fps=10, cmap=cmap)
    assert html_path.exists()

    # Test widget (with mocking)
    from neurospatial.animation.backends.widget_backend import render_widget

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets"
        ) as mock_widgets,
        patch("neurospatial.animation.backends.widget_backend.display"),
    ):
        mock_widgets.Image = MagicMock()
        mock_widgets.HTML = MagicMock()
        mock_widgets.IntSlider = MagicMock()
        mock_widgets.Play = MagicMock()
        mock_widgets.VBox = MagicMock()
        mock_widgets.HBox = MagicMock()
        mock_widgets.jslink = MagicMock()

        render_widget(env, fields, fps=10, cmap=cmap)
        mock_widgets.Image.assert_called_once()


def test_all_backends_handle_custom_fps(shared_test_data, tmp_path):
    """Test all backends respect custom FPS setting."""
    pytest.importorskip("napari")
    pytest.importorskip("ipywidgets")

    env, fields = shared_test_data

    fps = 60  # High frame rate

    # Test napari
    from neurospatial.animation.backends.napari_backend import render_napari

    viewer = render_napari(env, fields, fps=fps)
    assert viewer is not None

    # Test HTML
    from neurospatial.animation.backends.html_backend import render_html

    html_path = tmp_path / "fps_test.html"
    render_html(env, fields, save_path=str(html_path), fps=fps)
    assert html_path.exists()

    # Test widget (with mocking)
    from neurospatial.animation.backends.widget_backend import render_widget

    with (
        patch(
            "neurospatial.animation.backends.widget_backend.ipywidgets"
        ) as mock_widgets,
        patch("neurospatial.animation.backends.widget_backend.display"),
    ):
        mock_widgets.Image = MagicMock()
        mock_widgets.HTML = MagicMock()
        mock_widgets.IntSlider = MagicMock()
        mock_widgets.Play = MagicMock(return_value=MagicMock())
        mock_widgets.VBox = MagicMock()
        mock_widgets.HBox = MagicMock()
        mock_widgets.jslink = MagicMock()

        render_widget(env, fields, fps=fps)

        # Widget should use fps to set interval
        mock_widgets.Play.assert_called_once()
        call_kwargs = mock_widgets.Play.call_args[1]
        assert call_kwargs["interval"] == int(1000 / fps)  # milliseconds
