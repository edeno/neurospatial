"""Tests for scale bar visualization utilities.

Tests follow TDD approach - written before implementation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial.animation.config import (
    ScaleBarConfig,
    add_scale_bar_to_axes,
    compute_nice_length,
    format_scale_label,
)

# Default seed for reproducible tests
DEFAULT_SEED = 42


@pytest.fixture
def rng():
    """Create a seeded random number generator for reproducible tests."""
    return np.random.default_rng(DEFAULT_SEED)


class TestScaleBarConfig:
    """Test ScaleBarConfig dataclass."""

    def test_default_creation(self):
        """Test creating config with all defaults."""
        config = ScaleBarConfig()
        assert config.length is None
        assert config.position == "lower right"
        assert config.color is None
        assert config.background == "white"
        assert config.background_alpha == 0.7
        assert config.font_size == 10
        assert config.pad == 0.5
        assert config.sep == 5.0
        assert config.label_top is True
        assert config.box_style == "round"
        assert config.show_label is True

    def test_custom_creation(self):
        """Test creating config with custom values."""
        config = ScaleBarConfig(
            length=20.0,
            position="upper left",
            color="white",
            background="black",
            background_alpha=0.5,
            font_size=14,
        )
        assert config.length == 20.0
        assert config.position == "upper left"
        assert config.color == "white"
        assert config.background == "black"
        assert config.background_alpha == 0.5
        assert config.font_size == 14

    def test_frozen(self):
        """Test that config is immutable (frozen dataclass)."""
        config = ScaleBarConfig()
        with pytest.raises(AttributeError):
            config.length = 10.0


class TestComputeNiceLength:
    """Test nice length computation using 1-2-5 rule."""

    def test_basic_extents(self):
        """Test common extent values."""
        assert compute_nice_length(100) == 20  # 20% of 100
        assert compute_nice_length(50) == 10  # 20% of 50

    def test_follows_125_rule(self):
        """Results should be 1, 2, or 5 x 10^n.

        Uses tolerance for floating-point robustness.
        """
        for extent in [7, 13, 27, 73, 156, 0.3, 0.7]:
            length = compute_nice_length(extent)
            # Compute mantissa (normalized to [1, 10))
            mantissa = length / (10 ** np.floor(np.log10(length)))
            # Check with tolerance for floating-point robustness
            valid_mantissas = (1.0, 2.0, 5.0)
            assert any(np.isclose(mantissa, v, atol=1e-9) for v in valid_mantissas), (
                f"extent={extent} produced length={length}, mantissa={mantissa}"
            )

    def test_target_fraction(self):
        """Test different target fractions."""
        assert compute_nice_length(100, target_fraction=0.1) == 10
        assert compute_nice_length(100, target_fraction=0.25) == 20

    def test_small_extents(self):
        """Test with small extents (common in normalized data)."""
        assert compute_nice_length(1.0) == 0.2
        assert compute_nice_length(0.1) == 0.02

    def test_large_extents(self):
        """Test with large extents (e.g., meters)."""
        assert compute_nice_length(10000) == 2000

    def test_zero_extent_raises(self):
        """Zero extent should raise ValueError."""
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(0.0)

    def test_negative_extent_raises(self):
        """Negative extent should raise ValueError."""
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(-10.0)

    def test_inf_extent_raises(self):
        """Infinite extent should raise ValueError."""
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(np.inf)

    def test_nan_extent_raises(self):
        """NaN extent should raise ValueError."""
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(np.nan)


class TestAutoContrast:
    """Test _auto_contrast_color function."""

    def test_dark_background_selects_white(self):
        """Dark background should select white text."""
        from neurospatial.animation.config import _auto_contrast_color

        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        assert _auto_contrast_color(ax) == "white"
        plt.close(fig)

    def test_light_background_selects_black(self):
        """Light background should select black text."""
        from neurospatial.animation.config import _auto_contrast_color

        fig, ax = plt.subplots()
        ax.set_facecolor("white")
        assert _auto_contrast_color(ax) == "black"
        plt.close(fig)

    def test_gray_background(self):
        """Mid-gray background (luminance ~0.5) should pick one."""
        from neurospatial.animation.config import _auto_contrast_color

        fig, ax = plt.subplots()
        ax.set_facecolor((0.5, 0.5, 0.5))  # RGB gray
        color = _auto_contrast_color(ax)
        assert color in ("white", "black")
        plt.close(fig)


class TestFormatScaleLabel:
    """Test scale label formatting."""

    def test_with_units(self):
        """Test formatting with units."""
        assert format_scale_label(10, "cm") == "10 cm"
        assert format_scale_label(2.5, "m") == "2.5 m"

    def test_without_units(self):
        """Test formatting without units."""
        assert format_scale_label(10, None) == "10"

    def test_integer_display(self):
        """Whole numbers should not show decimal."""
        assert format_scale_label(10.0, "cm") == "10 cm"  # Not "10.0 cm"
        assert format_scale_label(5.0, "m") == "5 m"

    def test_decimal_display(self):
        """Non-whole numbers should show decimal."""
        assert format_scale_label(2.5, "cm") == "2.5 cm"
        assert format_scale_label(0.1, "m") == "0.1 m"


class TestAddScaleBarToAxes:
    """Test matplotlib scale bar rendering."""

    def test_adds_artist(self):
        """Test that scale bar is added to axes."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        artist = add_scale_bar_to_axes(ax, extent=100, units="cm")
        # AnchoredSizeBar is added to ax.artists
        assert len(ax.artists) > 0 or artist in ax.get_children()
        plt.close(fig)

    def test_custom_config(self):
        """Test with custom configuration."""
        config = ScaleBarConfig(length=25, position="upper left", color="red")
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
        assert len(ax.artists) > 0
        plt.close(fig)

    def test_no_background(self):
        """Test with no background box."""
        config = ScaleBarConfig(box_style=None)
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
        plt.close(fig)

    def test_all_positions(self):
        """Test all four position options."""
        positions = ["lower right", "lower left", "upper right", "upper left"]
        for position in positions:
            config = ScaleBarConfig(position=position)
            fig, ax = plt.subplots()
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

            add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
            assert len(ax.artists) > 0
            plt.close(fig)

    def test_auto_length(self):
        """Test auto-calculated length (config.length=None)."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # With length=None, should auto-calculate
        config = ScaleBarConfig(length=None)
        add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
        assert len(ax.artists) > 0
        plt.close(fig)


class TestPlotFieldWithScaleBar:
    """Test scale bar integration in plot_field.

    Uses fixtures from tests/conftest.py:
    - small_2d_env: 10x10 cm grid (25 bins)
    - medium_2d_env: 50x50 cm grid (625 bins)
    - small_1d_env: 10 cm linear track (5 bins)
    """

    def test_scale_bar_bool(self, small_2d_env, rng):
        """Test scale_bar=True adds scale bar."""
        field = rng.random(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=True)
        # Check that at least one artist was added (scale bar)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_config(self, small_2d_env, rng):
        """Test scale_bar=ScaleBarConfig works."""
        config = ScaleBarConfig(length=5.0, color="white", position="upper left")
        field = rng.random(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=config)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_with_colorbar(self, small_2d_env, rng):
        """Test scale bar works alongside colorbar."""
        field = rng.random(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=True, colorbar=True)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_1d_env(self, small_1d_env, rng):
        """Test scale bar with 1D environments.

        Note: 1D environments that have grid_shape with 1 dimension will raise
        NotImplementedError from plot_field() because pcolormesh requires 2D grids.
        This is a pre-existing limitation, not related to scale bar functionality.
        """
        field = rng.random(small_1d_env.n_bins)
        # 1D graph layouts go through the 1D plotting path, not grid path
        # Check if it's a 1D layout that can be plotted
        if small_1d_env.layout.is_1d:
            # 1D layouts use line plots, scale bar still works
            small_1d_env.plot_field(field, scale_bar=True)
            plt.close()
        else:
            # Non-1D with 1D grid shape will fail
            with pytest.raises(NotImplementedError):
                small_1d_env.plot_field(field, scale_bar=True)
            plt.close("all")

    def test_no_scale_bar_default(self, small_2d_env, rng):
        """Test scale_bar=False (default) adds no scale bar."""
        field = rng.random(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=False)
        # No additional artists (beyond the field itself)
        assert len(ax.artists) == 0
        plt.close()

    def test_scale_bar_with_units(self, small_2d_env, rng):
        """Test scale bar respects env.units."""
        small_2d_env.units = "cm"
        field = rng.random(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=True)
        assert len(ax.artists) > 0
        plt.close()


class TestPlotWithScaleBar:
    """Test scale bar in env.plot()."""

    def test_plot_with_scale_bar(self, small_2d_env):
        """Test env.plot(scale_bar=True)."""
        ax = small_2d_env.plot(scale_bar=True)
        assert len(ax.artists) > 0
        plt.close()

    def test_plot_with_regions_and_scale_bar(self, small_2d_env):
        """Test scale bar works with regions displayed."""
        small_2d_env.regions.add("center", point=(5.0, 5.0))
        ax = small_2d_env.plot(scale_bar=True, show_regions=True)
        assert len(ax.artists) > 0
        plt.close()


class TestAnimateFieldsWithScaleBar:
    """Test scale bar in animate_fields (Milestone 3).

    Uses fixtures from tests/conftest.py:
    - small_2d_env: 10x10 cm grid (25 bins)
    """

    def test_animate_fields_accepts_scale_bar_bool(self, small_2d_env, rng):
        """Test animate_fields() accepts scale_bar=True parameter."""
        fields = [rng.random(small_2d_env.n_bins) for _ in range(3)]
        frame_times = np.linspace(0, 1.0, 3)  # 3 frames over 1 second

        # Should not raise - parameter is accepted
        # Use html backend since it doesn't require external deps
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            small_2d_env.animate_fields(
                fields,
                frame_times=frame_times,
                backend="html",
                save_path=f.name,
                scale_bar=True,
            )

    def test_animate_fields_accepts_scale_bar_config(self, small_2d_env, rng):
        """Test animate_fields() accepts ScaleBarConfig parameter."""
        from neurospatial.animation.config import ScaleBarConfig

        fields = [rng.random(small_2d_env.n_bins) for _ in range(3)]
        frame_times = np.linspace(0, 1.0, 3)  # 3 frames over 1 second
        config = ScaleBarConfig(length=5.0, position="upper left", color="white")

        # Should not raise - parameter is accepted
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            small_2d_env.animate_fields(
                fields,
                frame_times=frame_times,
                backend="html",
                save_path=f.name,
                scale_bar=config,
            )

    def test_video_backend_with_scale_bar(self, small_2d_env, rng):
        """Test video backend renders scale bar in frames."""
        pytest.importorskip("subprocess")

        # Check if ffmpeg is available
        from neurospatial.animation.backends.video_backend import check_ffmpeg_available

        if not check_ffmpeg_available():
            pytest.skip("ffmpeg not available")

        fields = [rng.random(small_2d_env.n_bins) for _ in range(3)]
        frame_times = np.linspace(0, 1.0, 3)  # 3 frames over 1 second

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            small_2d_env.animate_fields(
                fields,
                frame_times=frame_times,
                backend="video",
                save_path=f.name,
                scale_bar=True,
            )
            # Video should be created (validates parameter flows through)
            from pathlib import Path

            assert Path(f.name).exists()

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="napari not installed",
    )
    def test_napari_backend_with_scale_bar(self, small_2d_env, rng):
        """Test napari backend configures native scale bar."""
        fields = [rng.random(small_2d_env.n_bins) for _ in range(3)]
        frame_times = np.linspace(0, 1.0, 3)  # 3 frames over 1 second

        # This should configure napari's native scale bar
        viewer = small_2d_env.animate_fields(
            fields,
            frame_times=frame_times,
            backend="napari",
            scale_bar=True,
            show=False,
        )

        try:
            # Verify napari's scale bar is enabled
            assert viewer.scale_bar.visible is True
        finally:
            viewer.close()
