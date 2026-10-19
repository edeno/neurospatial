"""Test shared rendering utilities."""

import numpy as np
import pytest

from neurospatial import Environment


def test_render_field_to_rgb():
    """Test field rendering to RGB array."""
    from neurospatial.animation.rendering import render_field_to_rgb

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = rng.random(env.n_bins)

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

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = rng.random(env.n_bins)

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

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create colormap lookup
    from matplotlib import pyplot as plt

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    field = rng.random(env.n_bins)

    rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)

    # Check output
    assert rgb.dtype == np.uint8
    assert rgb.shape[-1] == 3  # RGB channels

    # For grid layouts, should have spatial dimensions
    if hasattr(env, "grid_shape") and env.grid_shape is not None:
        assert rgb.ndim == 3  # (height, width, 3)


def test_field_to_rgb_for_napari_clipping():
    """Test that values are properly clipped to [vmin, vmax]."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
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


def test_field_to_rgb_for_napari_wrong_shape():
    """field_to_rgb_for_napari rejects a field whose length != env.n_bins."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    from matplotlib import pyplot as plt

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Field with the wrong number of values must raise an informative error.
    wrong_field = rng.random(env.n_bins + 10)

    with pytest.raises(ValueError) as exc_info:
        field_to_rgb_for_napari(env, wrong_field, cmap_lookup, vmin=0, vmax=1)

    error_msg = str(exc_info.value)
    assert "Field has" in error_msg
    assert "values but environment has" in error_msg
    assert str(env.n_bins) in error_msg
    assert "Expected shape:" in error_msg


def test_render_field_to_image_bytes_jpeg_format():
    """Test field rendering to JPEG bytes (requires Pillow)."""
    pytest.importorskip("PIL")  # Skip if Pillow not available

    from neurospatial.animation.rendering import render_field_to_image_bytes

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field = rng.random(env.n_bins)

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


class TestFieldToRgbForNapariZeroRange:
    """Test zero-range guard in field_to_rgb_for_napari."""

    def test_vmin_equals_vmax_returns_uniform_color(self):
        """Test that vmin == vmax (constant field) produces uniform color."""
        pytest.importorskip("matplotlib")

        from neurospatial.animation.rendering import field_to_rgb_for_napari

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create colormap lookup
        from matplotlib import pyplot as plt

        cmap_obj = plt.get_cmap("viridis")
        cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        # Constant field (all same values)
        field = np.full(env.n_bins, 5.0)

        # vmin == vmax: zero range
        rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=5.0, vmax=5.0)

        # Should not raise and should produce valid RGB
        assert rgb.dtype == np.uint8
        assert np.all((rgb >= 0) & (rgb <= 255))

        # All active pixels should map to same color (middle of colormap)
        # Since normalized = 0.5, index = 127
        expected_color = cmap_lookup[127]

        # For grid layouts, inactive bins are black [0,0,0]. Filter those out.
        rgb_flat = rgb.reshape(-1, 3)
        # Find non-black pixels (active bins)
        non_black_mask = np.any(rgb_flat != 0, axis=1)

        if np.any(non_black_mask):
            # All active pixels should have the expected color
            active_pixels = rgb_flat[non_black_mask]
            assert np.all(active_pixels == expected_color), (
                f"Expected all active pixels to be {expected_color}, "
                f"but got unique values: {np.unique(active_pixels, axis=0)}"
            )

    def test_zero_range_no_divide_by_zero_warning(self):
        """Test that zero range does not produce divide-by-zero warning."""
        pytest.importorskip("matplotlib")
        import warnings

        from neurospatial.animation.rendering import field_to_rgb_for_napari

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create colormap lookup
        from matplotlib import pyplot as plt

        cmap_obj = plt.get_cmap("viridis")
        cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        field = rng.random(env.n_bins)

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0.5, vmax=0.5)

            # Should not have RuntimeWarning about divide by zero
            divide_warnings = [
                warning
                for warning in w
                if "divide" in str(warning.message).lower()
                or "invalid" in str(warning.message).lower()
            ]
            assert len(divide_warnings) == 0

        # Should still produce valid output
        assert rgb.dtype == np.uint8


class TestValidateFrameLabels:
    """Test _validate_frame_labels helper function."""

    def test_valid_labels_returned_unchanged(self):
        """Test that valid frame_labels are returned unchanged."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Frame 1", "Frame 2", "Frame 3"]
        result = _validate_frame_labels(labels, n_frames=3, backend_name="test")

        assert result == labels
        assert result is labels  # Same object reference

    def test_none_returns_none(self):
        """Test that None frame_labels returns None."""
        from neurospatial.animation.rendering import _validate_frame_labels

        result = _validate_frame_labels(None, n_frames=10, backend_name="test")

        assert result is None

    def test_mismatched_length_raises_value_error(self):
        """Test that mismatched length raises ValueError with helpful message."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Frame 1", "Frame 2"]  # 2 labels

        with pytest.raises(ValueError) as exc_info:
            _validate_frame_labels(labels, n_frames=5, backend_name="html")

        error_msg = str(exc_info.value)
        # Check WHAT/WHY/HOW format
        assert "WHAT:" in error_msg
        assert "WHY:" in error_msg
        assert "Fix:" in error_msg
        # Check specific details
        assert "2" in error_msg  # frame_labels length
        assert "5" in error_msg  # n_frames
        assert "html" in error_msg  # backend_name

    def test_error_message_includes_backend_name(self):
        """Test error message includes the backend name."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Label"]

        for backend_name in ["html", "video", "widget"]:
            with pytest.raises(ValueError) as exc_info:
                _validate_frame_labels(labels, n_frames=10, backend_name=backend_name)
            assert backend_name in str(exc_info.value)


class TestRenderFieldToImageBytesFigureLeak:
    """render_field_to_image_bytes must not leak figures on the error path."""

    def test_no_figure_leak_when_plot_field_raises(self, monkeypatch):
        """A raising plot_field must still close the figure (no leak)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.animation.rendering import render_field_to_image_bytes

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        field = rng.random(env.n_bins)

        # Force the body to raise after the figure has been created.
        def _boom(*args, **kwargs):
            raise RuntimeError("plot_field exploded")

        monkeypatch.setattr(type(env), "plot_field", _boom)

        fignums_before = set(plt.get_fignums())

        with pytest.raises(RuntimeError, match="plot_field exploded"):
            render_field_to_image_bytes(env, field, "viridis", 0, 1)

        # The figure created inside the function must have been closed.
        assert set(plt.get_fignums()) == fignums_before


def test_field_to_rgb_for_napari_orientation():
    """A field peak at a known env corner lands at the predicted napari pixel."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    # Dense 4x4 grid covering 0..30 cm so every bin is active and indexable.
    xs = np.linspace(0, 30, 16)
    ys = np.linspace(0, 30, 16)
    xx, yy = np.meshgrid(xs, ys)
    env = Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=10.0
    )
    assert env.grid_shape == (4, 4)

    # Identity colormap: index i -> (i, i, i). Channel 0 == colormap index.
    cmap_lookup = np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)
    centers = env.bin_centers

    def bright_pixel(flat_bin_index):
        field = np.zeros(env.n_bins)
        field[flat_bin_index] = 1.0  # maps to colormap index 255
        rgb = field_to_rgb_for_napari(env, field, cmap_lookup, vmin=0, vmax=1)
        assert rgb.shape == (4, 4, 3)
        return np.unravel_index(int(np.argmax(rgb[..., 0])), rgb[..., 0].shape)

    # (min x, max y) -> top-left (row 0, col 0)
    top_left = int(np.lexsort((centers[:, 0], -centers[:, 1]))[0])
    assert bright_pixel(top_left) == (0, 0)

    # (min x, min y) -> bottom-left (row 3, col 0): napari flips Y downward.
    bottom_left = int(np.lexsort((centers[:, 0], centers[:, 1]))[0])
    assert bright_pixel(bottom_left) == (3, 0)

    # (max x, max y) -> top-right (row 0, col 3)
    top_right = int(np.lexsort((-centers[:, 0], -centers[:, 1]))[0])
    assert bright_pixel(top_right) == (0, 3)


def test_animation_color_limits_ignore_a_few_extreme_frames():
    """The limits are percentiles of the frames' extremes, not all values."""
    from neurospatial.animation.rendering import animation_colormap_range

    rng = np.random.default_rng(0)
    # A sharp bump in a few of 1000 bins each frame, as a posterior.
    fields = np.zeros((200, 1000))
    peaks = rng.uniform(0.2, 0.3, 200)
    fields[np.arange(200), rng.integers(0, 1000, 200)] = peaks
    fields[7, 3] = 50.0  # one outlier frame
    fields[:, 500] = np.nan
    low, high = animation_colormap_range(fields, None, None)
    assert low == 0.0
    # Close to the bumps' heights, not the outlier, and not the 99th
    # percentile of all values (0), which every bump would pass.
    assert np.percentile(peaks, 95) < high < 0.31
    assert np.mean(np.nanmax(fields, axis=1) > high) <= 0.01 + 1 / 200
    assert animation_colormap_range(fields, -1.0, None)[0] == -1.0
    # A posterior's floor, 1e-56, is zero; a floor near the top is kept.
    assert animation_colormap_range(fields + 1e-56, None, None)[0] == 0.0
    assert animation_colormap_range(fields + 0.1, None, None)[0] == pytest.approx(0.1)
    assert animation_colormap_range(np.full((3, 4), np.nan), None, None) == (0.0, 1.0)
    assert animation_colormap_range([np.full(4, 2.0)] * 3, None, None) == (1.5, 2.5)


def test_typical_peak_is_the_median_frame_maximum():
    from neurospatial.animation.rendering import frame_extremes, typical_peak

    fields = np.zeros((5, 4))
    fields[:, 1] = [0.1, 0.2, 0.3, 0.4, 9.0]  # one extreme frame
    fields[2, 3] = np.nan
    assert typical_peak(frame_extremes(fields)[1]) == pytest.approx(0.3)
    assert typical_peak(np.full(3, np.nan)) == 1.0


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # Non-finite values are ignored, frame by frame.
        ([np.array([np.nan, 0.0, 5.0]), np.array([0.0, np.inf, 5.0])], (0.0, 5.0)),
        ([np.array([-np.inf, 1.0, 3.0]), np.array([1.0, np.nan, 3.0])], (1.0, 3.0)),
        # One finite value: half a unit either side.
        ([np.array([np.nan, 5.0]), np.array([np.inf, np.nan])], (4.5, 5.5)),
        # Nothing finite, or no frames at all.
        ([np.array([np.nan, np.inf]), np.array([-np.inf, np.nan])], (0.0, 1.0)),
        ([], (0.0, 1.0)),
    ],
)
def test_animation_color_limits_use_finite_values_only(fields, expected):
    from neurospatial.animation.rendering import animation_colormap_range

    assert animation_colormap_range(fields, None, None) == expected


def test_long_recordings_use_every_nth_frame_for_color_limits():
    from neurospatial.animation.rendering import frame_extremes

    fields = np.zeros((250_001, 2), dtype=np.float32)
    fields[1, 0] = 99.0  # not a sampled frame
    with pytest.warns(UserWarning, match="sample_stride=5"):
        _, highs = frame_extremes(fields)
    assert len(highs) == 50_001 and highs.max() == 0.0
