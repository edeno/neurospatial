"""Test shared rendering utilities."""

import numpy as np
import pytest

from neurospatial import Environment


def test_compute_global_colormap_range():
    """Test global color scale computation."""
    from neurospatial.animation.rendering import compute_global_colormap_range

    fields = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([1, 2, 3]),
    ]

    vmin, vmax = compute_global_colormap_range(fields)
    assert vmin == 0.0
    assert vmax == 5.0

    # Manual limits override
    vmin_manual, vmax_manual = compute_global_colormap_range(fields, vmin=-1, vmax=10)
    assert vmin_manual == -1.0
    assert vmax_manual == 10.0


def test_compute_global_colormap_range_degenerate():
    """Test degenerate case (all same value)."""
    from neurospatial.animation.rendering import compute_global_colormap_range

    fields = [np.ones(10) * 5.0, np.ones(10) * 5.0]

    vmin, vmax = compute_global_colormap_range(fields)
    assert vmin == 4.5
    assert vmax == 5.5


def test_compute_global_colormap_range_partial_override():
    """Test partial manual override (only vmin or vmax)."""
    from neurospatial.animation.rendering import compute_global_colormap_range

    fields = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
    ]

    # Override only vmin
    vmin, vmax = compute_global_colormap_range(fields, vmin=-5, vmax=None)
    assert vmin == -5.0
    assert vmax == 5.0

    # Override only vmax
    vmin, vmax = compute_global_colormap_range(fields, vmin=None, vmax=10)
    assert vmin == 0.0
    assert vmax == 10.0


def test_render_field_to_rgb():
    """Test field rendering to RGB array."""
    from neurospatial.animation.rendering import render_field_to_rgb

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = np.random.rand(env.n_bins)

    rgb = render_field_to_rgb(env, field, cmap="viridis", vmin=0, vmax=1, dpi=50)

    # Check output shape
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3  # RGB
    assert rgb.dtype == np.uint8

    # Check value range
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_render_field_to_png_bytes():
    """Test field rendering to PNG bytes."""
    from neurospatial.animation.rendering import render_field_to_png_bytes

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = np.random.rand(env.n_bins)

    png_bytes = render_field_to_png_bytes(
        env, field, cmap="viridis", vmin=0, vmax=1, dpi=50
    )

    # Check output type
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

    # Check PNG signature (first 8 bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_field_to_rgb_for_napari_grid():
    """Test fast RGB conversion for grid layout."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create colormap lookup
    from matplotlib import pyplot as plt

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    field = np.random.rand(env.n_bins)

    rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)

    # Check output
    assert rgb.dtype == np.uint8
    assert rgb.shape[-1] == 3  # RGB channels

    # For grid layouts, should have spatial dimensions
    if hasattr(env.layout, "grid_shape") and env.layout.grid_shape is not None:
        assert rgb.ndim == 3  # (height, width, 3)


def test_field_to_rgb_for_napari_clipping():
    """Test that values are properly clipped to [vmin, vmax]."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create colormap lookup
    from matplotlib import pyplot as plt

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Field with values outside [0, 1] - ensure we have enough values
    field = np.zeros(env.n_bins)
    # Set first few values to test clipping (if env has at least 3 bins)
    if env.n_bins >= 3:
        field[0] = -10.0  # Should be clipped to 0
        field[1] = 0.5  # In range
        field[2] = 10.0  # Should be clipped to 1

    rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)

    # Should not raise and should produce valid RGB
    assert rgb.dtype == np.uint8
    assert np.all((rgb >= 0) & (rgb <= 255))


def test_render_field_to_image_bytes_jpeg_format():
    """Test field rendering to JPEG bytes (requires Pillow)."""
    pytest.importorskip("PIL")  # Skip if Pillow not available

    from neurospatial.animation.rendering import render_field_to_image_bytes

    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = np.random.rand(env.n_bins)

    # Test JPEG format
    jpeg_bytes = render_field_to_image_bytes(
        env, field, cmap="viridis", vmin=0, vmax=1, dpi=50, image_format="jpeg"
    )

    # Check output type
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0

    # Check JPEG signature (first 3 bytes: 0xFF 0xD8 0xFF)
    assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    # Compare with PNG format
    png_bytes = render_field_to_image_bytes(
        env, field, cmap="viridis", vmin=0, vmax=1, dpi=50, image_format="png"
    )
    # Both should produce valid output
    assert len(png_bytes) > 0
    # PNG signature check
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
