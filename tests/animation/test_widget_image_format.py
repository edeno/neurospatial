"""Tests for widget backend image format support (PNG vs JPEG).

This test module verifies:
1. PNG output (default, lossless)
2. JPEG output (optional, lossy, smaller files)
3. Error handling for invalid formats
4. Proper image signatures in output bytes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import OverlayData, PositionData

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    return env


@pytest.fixture
def simple_field(simple_env: Environment) -> np.ndarray:
    """Create a simple test field."""
    rng = np.random.default_rng(42)
    return rng.random(simple_env.n_bins)


@pytest.fixture
def position_overlay_data() -> OverlayData:
    """Create test position overlay data."""
    rng = np.random.default_rng(42)
    n_frames = 10
    positions = rng.random((n_frames, 2)) * 10
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=3)]
    )


# ============================================================================
# render_field_to_png_bytes_with_overlays Tests
# ============================================================================


class TestRenderFieldWithOverlaysImageFormat:
    """Tests for image_format parameter in render_field_to_png_bytes_with_overlays."""

    def test_default_format_is_png(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test default image format is PNG."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
        )

        # PNG signature
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_explicit_png_format(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test explicit PNG format returns PNG bytes."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            image_format="png",
        )

        # PNG signature
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_format(self, simple_env: Environment, simple_field: np.ndarray):
        """Test JPEG format returns JPEG bytes."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            image_format="jpeg",
        )

        # JPEG signature (FFD8FF)
        assert result[:3] == b"\xff\xd8\xff"

    def test_jpeg_format_case_insensitive(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test JPEG format is case insensitive."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            image_format="JPEG",
        )

        # JPEG signature
        assert result[:3] == b"\xff\xd8\xff"

    def test_both_formats_produce_valid_output(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test both PNG and JPEG produce valid non-empty output."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        png_result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=150,
            frame_idx=0,
            image_format="png",
        )

        jpeg_result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=150,
            frame_idx=0,
            image_format="jpeg",
        )

        # Both formats should produce non-empty valid output
        assert len(png_result) > 0
        assert len(jpeg_result) > 0
        # Verify correct signatures
        assert png_result[:8] == b"\x89PNG\r\n\x1a\n"
        assert jpeg_result[:3] == b"\xff\xd8\xff"

    def test_invalid_format_raises_error(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test invalid format raises ValueError."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        with pytest.raises(ValueError, match="image_format must be 'png' or 'jpeg'"):
            render_field_to_png_bytes_with_overlays(
                simple_env,
                simple_field,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                dpi=50,
                frame_idx=0,
                image_format="gif",  # Invalid format
            )

    def test_jpeg_with_overlays(
        self,
        simple_env: Environment,
        simple_field: np.ndarray,
        position_overlay_data: OverlayData,
    ):
        """Test JPEG format works with overlays."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            overlay_data=position_overlay_data,
            image_format="jpeg",
        )

        # JPEG signature
        assert result[:3] == b"\xff\xd8\xff"
        # Should have non-trivial size
        assert len(result) > 1000


# ============================================================================
# PersistentFigureRenderer Tests
# ============================================================================


class TestPersistentFigureRendererImageFormat:
    """Tests for image_format in PersistentFigureRenderer."""

    def test_default_format_is_png(self, simple_env: Environment):
        """Test default image format is PNG."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(5)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            result = renderer.render(fields[0], frame_idx=0)
            # PNG signature
            assert result[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            renderer.close()

    def test_jpeg_format(self, simple_env: Environment):
        """Test JPEG format returns JPEG bytes."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(5)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            image_format="jpeg",
        )

        try:
            result = renderer.render(fields[0], frame_idx=0)
            # JPEG signature
            assert result[:3] == b"\xff\xd8\xff"
        finally:
            renderer.close()

    def test_jpeg_persists_across_renders(self, simple_env: Environment):
        """Test JPEG format persists across multiple renders."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(5)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            image_format="jpeg",
        )

        try:
            for i in range(3):
                result = renderer.render(fields[i], frame_idx=i)
                # JPEG signature for all frames
                assert result[:3] == b"\xff\xd8\xff"
        finally:
            renderer.close()

    def test_invalid_format_raises_error(self, simple_env: Environment):
        """Test invalid format raises ValueError in PersistentFigureRenderer."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        with pytest.raises(ValueError, match="image_format must be 'png' or 'jpeg'"):
            PersistentFigureRenderer(
                env=simple_env,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                dpi=50,
                image_format="gif",  # Invalid format
            )


# ============================================================================
# render_widget Tests
# ============================================================================


class TestRenderWidgetImageFormat:
    """Tests for image_format in render_widget."""

    @patch("neurospatial.animation.backends.widget_backend.display")
    @patch("neurospatial.animation.backends.widget_backend.ipywidgets")
    def test_render_widget_accepts_image_format_parameter(
        self, mock_ipywidgets, mock_display, simple_env: Environment
    ):
        """Test render_widget accepts image_format parameter."""
        from neurospatial.animation.backends.widget_backend import render_widget

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(10)]

        # Mock ipywidgets components
        mock_ipywidgets.Image = MagicMock
        mock_ipywidgets.HTML = MagicMock
        mock_ipywidgets.IntSlider = MagicMock
        mock_ipywidgets.Play = MagicMock
        mock_ipywidgets.VBox = MagicMock
        mock_ipywidgets.HBox = MagicMock
        mock_ipywidgets.jslink = MagicMock

        # Should not raise
        render_widget(
            simple_env,
            fields,
            initial_cache_size=5,
            image_format="jpeg",
        )

        assert mock_display.called

    @patch("neurospatial.animation.backends.widget_backend.display")
    @patch("neurospatial.animation.backends.widget_backend.ipywidgets")
    @patch(
        "neurospatial.animation.backends.widget_backend.render_field_to_png_bytes_with_overlays"
    )
    def test_render_widget_passes_image_format_to_renderer(
        self,
        mock_render,
        mock_ipywidgets,
        mock_display,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test render_widget passes image_format to render function."""
        from neurospatial.animation.backends.widget_backend import render_widget

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(10)]

        # Mock ipywidgets
        mock_ipywidgets.Image = MagicMock
        mock_ipywidgets.HTML = MagicMock
        mock_ipywidgets.IntSlider = MagicMock
        mock_ipywidgets.Play = MagicMock
        mock_ipywidgets.VBox = MagicMock
        mock_ipywidgets.HBox = MagicMock
        mock_ipywidgets.jslink = MagicMock

        mock_render.return_value = b"fake_jpeg_data"

        # Pass overlay_data so render_field_to_png_bytes_with_overlays is called
        render_widget(
            simple_env,
            fields,
            initial_cache_size=5,
            image_format="jpeg",
            overlay_data=position_overlay_data,
        )

        # Verify image_format was passed
        call_kwargs = mock_render.call_args_list[0][1]
        assert call_kwargs["image_format"] == "jpeg"


# ============================================================================
# JPEG Requirements Tests
# ============================================================================


class TestJPEGRequirements:
    """Tests for JPEG-specific requirements (PIL)."""

    def test_jpeg_works_when_pillow_available(
        self, simple_env: Environment, simple_field: np.ndarray
    ):
        """Test that JPEG works correctly when Pillow is available."""
        # This test verifies JPEG rendering works with the installed Pillow
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )

        # Pillow is installed in the test environment, so this should work
        result = render_field_to_png_bytes_with_overlays(
            simple_env,
            simple_field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            image_format="jpeg",
        )

        # Verify JPEG was produced
        assert result[:3] == b"\xff\xd8\xff"
        assert len(result) > 100  # Non-trivial JPEG output
