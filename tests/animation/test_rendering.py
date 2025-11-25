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
    if hasattr(env.layout, "grid_shape") and env.layout.grid_shape is not None:
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


class TestComputeGlobalColormapRangeNaN:
    """Test NaN-robustness of compute_global_colormap_range."""

    def test_arrays_with_nan_values_return_correct_range(self):
        """Test arrays with NaN values return correct range (ignoring NaNs)."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Fields with NaN values mixed in
        fields = [
            np.array([np.nan, 1.0, 2.0]),
            np.array([3.0, np.nan, 5.0]),
            np.array([0.0, 2.0, np.nan]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Should ignore NaN and compute range from valid values [0.0, 5.0]
        assert vmin == 0.0
        assert vmax == 5.0

    def test_arrays_with_all_nan_return_safe_defaults(self):
        """Test arrays with all NaN values return safe defaults (0.0, 1.0)."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # All-NaN fields
        fields = [
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Should return safe default range
        assert vmin == 0.0
        assert vmax == 1.0

    def test_arrays_with_inf_values_handled_correctly(self):
        """Test arrays with non-finite values (inf) are handled correctly."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Fields with inf values
        fields = [
            np.array([np.inf, 1.0, 2.0]),
            np.array([3.0, -np.inf, 5.0]),
            np.array([0.0, 2.0, np.inf]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Should ignore inf and compute range from finite values [0.0, 5.0]
        assert vmin == 0.0
        assert vmax == 5.0

    def test_arrays_with_all_inf_return_safe_defaults(self):
        """Test arrays with all inf values return safe defaults (0.0, 1.0)."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # All-inf fields
        fields = [
            np.array([np.inf, np.inf, -np.inf]),
            np.array([np.inf, -np.inf]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Should return safe default range
        assert vmin == 0.0
        assert vmax == 1.0

    def test_mixed_nan_and_inf_values(self):
        """Test arrays with mixed NaN and inf values."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Mixed NaN, inf, and valid values
        fields = [
            np.array([np.nan, np.inf, 2.0]),
            np.array([3.0, -np.inf, np.nan]),
            np.array([1.0, np.nan, np.inf]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Should ignore NaN and inf, compute range from [1.0, 3.0]
        assert vmin == 1.0
        assert vmax == 3.0

    def test_single_finite_value_with_nan(self):
        """Test array with single finite value among NaNs."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Single valid value among NaN/inf
        fields = [
            np.array([np.nan, 5.0, np.nan]),
            np.array([np.nan, np.nan, np.inf]),
        ]

        vmin, vmax = compute_global_colormap_range(fields)

        # Single value: should expand range around it
        assert vmin == 4.5
        assert vmax == 5.5

    def test_empty_fields_list(self):
        """Test empty fields list returns safe defaults."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Empty fields list
        fields: list[np.ndarray] = []

        vmin, vmax = compute_global_colormap_range(fields)

        # Should return safe default range
        assert vmin == 0.0
        assert vmax == 1.0


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
        assert "HOW:" in error_msg
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
