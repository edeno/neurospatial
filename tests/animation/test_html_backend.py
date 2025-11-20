"""Test HTML export backend."""

from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.backends.html_backend import render_html


def test_html_export_basic(tmp_path):
    """Test basic HTML player generation."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "test.html"

    result = render_html(
        env,
        fields,
        save_path=str(output_path),
        frame_labels=["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5"],
    )

    # Check file created
    assert output_path.exists()
    assert isinstance(result, Path)
    assert result == output_path

    # Check HTML content
    html = output_path.read_text(encoding="utf-8")
    assert "<html" in html
    assert "data:image/png;base64," in html
    assert "Trial 1" in html

    # Should have embedded frames (they're in a JavaScript array)
    # Check that all frame labels appear
    assert "Trial 2" in html
    assert "Trial 3" in html
    assert "Trial 4" in html
    assert "Trial 5" in html


def test_html_export_without_labels(tmp_path):
    """Test HTML export generates default labels."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    output_path = tmp_path / "test.html"

    render_html(
        env,
        fields,
        save_path=str(output_path),
    )

    html = output_path.read_text(encoding="utf-8")
    # Should have default labels
    assert "Frame 1" in html
    assert "Frame 2" in html
    assert "Frame 3" in html


def test_html_export_max_frames_limit(tmp_path):
    """Test HTML export enforces max frame limit."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Try to export 501 frames (exceeds default limit of 500)
    fields = [np.random.rand(env.n_bins) for _ in range(501)]

    output_path = tmp_path / "large.html"

    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        render_html(
            env,
            fields,
            save_path=str(output_path),
        )

    # Check error message is helpful
    error_msg = str(excinfo.value)
    assert "500" in error_msg  # Mentions the limit
    assert "501" in error_msg  # Mentions actual frame count
    assert "Subsample" in error_msg or "subsample" in error_msg  # Suggests solution
    assert "video" in error_msg or "Video" in error_msg  # Suggests alternative


def test_html_export_override_frame_limit(tmp_path):
    """Test HTML export can override frame limit."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Try to export 501 frames with higher limit
    fields = [np.random.rand(env.n_bins) for _ in range(501)]

    output_path = tmp_path / "large.html"

    # Should succeed with higher limit
    render_html(
        env,
        fields,
        save_path=str(output_path),
        max_html_frames=600,
    )

    assert output_path.exists()


def test_html_export_large_file_warning(tmp_path):
    """Test HTML export warns about large file size."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Many frames with high DPI - will produce large HTML (but under 500 frame limit)
    # estimated_mb = 200 * 0.1 * (200/100)^2 = 80 MB > 50 MB threshold
    fields = [np.random.rand(env.n_bins) for _ in range(200)]

    output_path = tmp_path / "large.html"

    # Should complete but warn about size
    with pytest.warns(UserWarning) as warning_info:
        render_html(
            env,
            fields,
            save_path=str(output_path),
            dpi=200,  # High DPI to trigger warning
        )

    # Check warning message
    assert len(warning_info) > 0
    warning_msg = str(warning_info[0].message)
    assert "large file" in warning_msg.lower() or "MB" in warning_msg

    # File should still be created
    assert output_path.exists()
    file_size_mb = output_path.stat().st_size / 1e6

    # Should be large (rough check)
    assert file_size_mb > 1.0


def test_html_player_javascript_controls(tmp_path):
    """Test HTML player has JavaScript controls."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "test.html"

    render_html(
        env,
        fields,
        save_path=str(output_path),
    )

    html = output_path.read_text(encoding="utf-8")

    # Check for control buttons
    assert "play" in html.lower()
    assert "pause" in html.lower()
    assert "prev" in html.lower() or "previous" in html.lower()
    assert "next" in html.lower()

    # Check for slider
    assert "slider" in html.lower() or "range" in html

    # Check for speed control
    assert "speed" in html.lower()

    # Check for JavaScript
    assert "<script>" in html
    assert "</script>" in html

    # Check for frame counter
    assert "frame" in html.lower()


def test_html_player_keyboard_shortcuts(tmp_path):
    """Test HTML player has keyboard shortcuts."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "test.html"

    render_html(
        env,
        fields,
        save_path=str(output_path),
    )

    html = output_path.read_text(encoding="utf-8")

    # Check for keyboard event handlers
    assert "keydown" in html.lower() or "onkeydown" in html.lower()
    assert "ArrowLeft" in html or "ArrowRight" in html


def test_html_export_custom_parameters(tmp_path):
    """Test HTML export respects custom parameters."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    output_path = tmp_path / "custom.html"

    render_html(
        env,
        fields,
        save_path=str(output_path),
        fps=60,
        cmap="hot",
        title="Custom Title",
        dpi=50,
    )

    html = output_path.read_text(encoding="utf-8")

    # Check title appears
    assert "Custom Title" in html

    # Check fps is used (60 appears in JavaScript)
    assert "60" in html


def test_html_export_vmin_vmax(tmp_path):
    """Test HTML export respects vmin/vmax."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create fields with known range
    fields = [np.random.rand(env.n_bins) * 100 for _ in range(3)]

    output_path = tmp_path / "test.html"

    # Export with custom vmin/vmax
    render_html(
        env,
        fields,
        save_path=str(output_path),
        vmin=0.0,
        vmax=50.0,  # Clamp range
    )

    # Should complete without error
    assert output_path.exists()


def test_html_export_auto_save_path():
    """Test HTML export creates default filename if save_path is None."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    # Call with save_path=None (should create animation.html)
    result = render_html(
        env,
        fields,
        save_path=None,
    )

    # Should create animation.html in current directory
    assert result.exists()
    assert result.name == "animation.html"

    # Clean up
    result.unlink()


def test_html_export_image_format_parameter(tmp_path):
    """Test HTML export accepts image_format parameter."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    output_path = tmp_path / "test.html"

    # Should accept image_format parameter (even if not used for PNG)
    render_html(
        env,
        fields,
        save_path=str(output_path),
        image_format="png",
    )

    assert output_path.exists()


def test_html_export_gracefully_accepts_unused_parameters(tmp_path):
    """Test HTML export gracefully accepts parameters for other backends."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    output_path = tmp_path / "test.html"

    # Should not error on video-specific or napari-specific parameters
    render_html(
        env,
        fields,
        save_path=str(output_path),
        n_workers=4,  # Video-specific
        contrast_limits=(0, 1),  # Napari-specific
    )

    assert output_path.exists()


def test_html_title_escaping(tmp_path):
    """Test HTML title properly escapes dangerous characters."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    fields = [np.random.rand(env.n_bins) for _ in range(3)]

    output_path = tmp_path / "test.html"

    # Title with HTML/JS that should be escaped
    render_html(
        env,
        fields,
        str(output_path),
        title="Test <script>alert('xss')</script> & Title",
    )

    html = output_path.read_text(encoding="utf-8")

    # Should escape HTML entities
    assert "&lt;script&gt;" in html
    assert "&amp;" in html

    # Should NOT contain raw HTML/JS
    assert "<script>alert" not in html
    assert "alert('xss')" not in html


def test_html_export_non_embedded_mode(tmp_path):
    """Test HTML export with embed=False (disk-backed frames)."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(10)]

    output_path = tmp_path / "non_embedded.html"

    result = render_html(
        env,
        fields,
        save_path=str(output_path),
        embed=False,
        n_workers=1,
    )

    # Check HTML file created
    assert output_path.exists()
    assert isinstance(result, Path)

    # Check HTML is lightweight (not embedded)
    html = output_path.read_text(encoding="utf-8")
    assert "<html" in html

    # Should NOT contain embedded base64 images
    assert "data:image/png;base64," not in html

    # Should reference external frame files
    assert "frame_" in html
    assert ".png" in html

    # Check frames directory created
    frames_dir = output_path.with_suffix("")  # Default: "non_embedded/"
    assert frames_dir.exists()
    assert frames_dir.is_dir()

    # Check frame files exist
    png_files = list(frames_dir.glob("frame_*.png"))
    assert len(png_files) == 10

    # Check files are valid PNGs (non-zero size)
    for png_file in png_files:
        assert png_file.stat().st_size > 0


def test_html_export_non_embedded_custom_frames_dir(tmp_path):
    """Test HTML export with embed=False and custom frames_dir."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "custom.html"
    frames_dir = tmp_path / "my_frames"

    render_html(
        env,
        fields,
        save_path=str(output_path),
        embed=False,
        frames_dir=str(frames_dir),
        n_workers=1,
    )

    # Check HTML file created
    assert output_path.exists()

    # Check custom frames directory created
    assert frames_dir.exists()
    assert frames_dir.is_dir()

    # Check frame files exist in custom directory
    png_files = list(frames_dir.glob("frame_*.png"))
    assert len(png_files) == 5


def test_html_export_non_embedded_with_labels(tmp_path):
    """Test HTML export with embed=False and frame labels."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]
    labels = [f"Trial {i + 1}" for i in range(5)]

    output_path = tmp_path / "labeled.html"

    render_html(
        env,
        fields,
        save_path=str(output_path),
        embed=False,
        frame_labels=labels,
        n_workers=1,
    )

    # Check HTML file created
    assert output_path.exists()

    # Check labels appear in HTML
    html = output_path.read_text(encoding="utf-8")
    for label in labels:
        assert label in html


def test_html_export_non_embedded_auto_workers(tmp_path):
    """Test HTML export with embed=False and n_workers=None (auto-select)."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "auto_workers.html"

    # n_workers=None should auto-select based on CPU count
    render_html(
        env,
        fields,
        save_path=str(output_path),
        embed=False,
        n_workers=None,
    )

    # Check HTML file created
    assert output_path.exists()

    # Check frames directory created
    frames_dir = output_path.with_suffix("")
    assert frames_dir.exists()

    # Check frame files exist
    png_files = list(frames_dir.glob("frame_*.png"))
    assert len(png_files) == 5


def test_html_export_non_embedded_negative_workers(tmp_path):
    """Test HTML export with embed=False raises error for negative n_workers."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    fields = [np.random.rand(env.n_bins) for _ in range(5)]

    output_path = tmp_path / "test.html"

    # Should raise ValueError for negative n_workers
    with pytest.raises(ValueError, match="n_workers must be positive"):
        render_html(
            env,
            fields,
            save_path=str(output_path),
            embed=False,
            n_workers=-1,
        )
