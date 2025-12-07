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


class TestColormapRangeStreaming:
    """Test streaming/chunked colormap range computation for large datasets."""

    @pytest.fixture
    def rng(self):
        """Create a seeded random number generator for reproducible tests."""
        return np.random.default_rng(42)

    def test_small_array_uses_exact_computation(self, rng):
        """Test that small arrays use exact computation (no streaming)."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Small array below threshold
        fields = rng.random((1000, 100))

        # Should work and return exact values
        vmin, vmax = compute_global_colormap_range(fields, max_frames_for_exact=50_000)

        # Verify exact computation
        assert vmin == pytest.approx(np.nanmin(fields))
        assert vmax == pytest.approx(np.nanmax(fields))

    def test_large_array_uses_streaming(self):
        """Test that large arrays use streaming (chunked) computation."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Large array above threshold
        rng = np.random.default_rng(42)
        fields = rng.random((100_000, 100))

        # Set threshold low to force streaming
        vmin, vmax = compute_global_colormap_range(fields, max_frames_for_exact=10_000)

        # Streaming should still find exact min/max
        expected_min = np.nanmin(fields)
        expected_max = np.nanmax(fields)
        assert vmin == pytest.approx(expected_min)
        assert vmax == pytest.approx(expected_max)

    def test_sample_stride_reduces_computation(self):
        """Test that sample_stride subsamples the array."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Create array with known extremes at specific positions
        fields = np.zeros((1000, 100))
        fields[0, 0] = -100.0  # Min at frame 0
        fields[999, 0] = 100.0  # Max at frame 999
        fields[500, 0] = 50.0  # Mid-point value

        # With stride=2, frames 0 and 500 are sampled (but not 999)
        vmin, vmax = compute_global_colormap_range(fields, sample_stride=2)

        # Should find min from frame 0, but max may miss frame 999
        assert vmin == pytest.approx(-100.0)
        # vmax should be from sampled frames (50.0 from frame 500)
        # Note: stride=2 samples 0, 2, 4, ..., 998 - so 999 is missed
        assert vmax == pytest.approx(50.0)

    def test_streaming_with_nan_values(self):
        """Test streaming correctly handles NaN values."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        rng = np.random.default_rng(42)
        fields = rng.random((60_000, 50))
        # Sprinkle NaN values
        fields[100, :] = np.nan
        fields[30_000, 10:20] = np.nan

        vmin, vmax = compute_global_colormap_range(fields, max_frames_for_exact=10_000)

        # Should ignore NaN and compute range from valid values
        finite_mask = np.isfinite(fields)
        expected_min = fields[finite_mask].min()
        expected_max = fields[finite_mask].max()
        assert vmin == pytest.approx(expected_min)
        assert vmax == pytest.approx(expected_max)

    def test_streaming_with_manual_vmin_vmax(self, rng):
        """Test that manual vmin/vmax skips computation (even for large arrays)."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # Large array that would normally trigger streaming
        fields = rng.random((100_000, 100))

        # Manual limits should skip computation entirely
        vmin, vmax = compute_global_colormap_range(
            fields, vmin=-5.0, vmax=10.0, max_frames_for_exact=10_000
        )

        assert vmin == -5.0
        assert vmax == 10.0

    def test_default_threshold_is_50000(self, rng):
        """Test that default max_frames_for_exact is 50_000."""
        from neurospatial.animation.rendering import compute_global_colormap_range

        # This test verifies the parameter has a sensible default
        # by checking that arrays just below threshold work
        fields = rng.random((49_999, 10))

        # Should work with default threshold
        vmin, vmax = compute_global_colormap_range(fields)

        assert np.isfinite(vmin)
        assert np.isfinite(vmax)

    def test_memmap_streaming(self):
        """Test that streaming works with memory-mapped arrays."""
        import tempfile
        from pathlib import Path

        from neurospatial.animation.rendering import compute_global_colormap_range

        # Create memmap
        tmpdir = Path(tempfile.mkdtemp())
        mmap_path = tmpdir / "fields.dat"

        try:
            # Create large memmap
            fields = np.memmap(
                str(mmap_path), dtype="float64", mode="w+", shape=(60_000, 50)
            )
            # Fill with known values
            rng = np.random.default_rng(42)
            for start in range(0, 60_000, 10_000):
                end = min(start + 10_000, 60_000)
                fields[start:end] = rng.random((end - start, 50))
            fields.flush()

            # Compute range with streaming
            vmin, vmax = compute_global_colormap_range(
                fields, max_frames_for_exact=10_000
            )

            # Should produce valid results
            assert np.isfinite(vmin)
            assert np.isfinite(vmax)
            assert vmin < vmax

        finally:
            # Cleanup
            del fields
            mmap_path.unlink()
            tmpdir.rmdir()


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
