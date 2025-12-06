"""Tests for animation.config module (scale bar configuration).

These tests verify that scale bar functionality is importable from the new
animation.config module after the move from visualization.scale_bar.

This test file follows TDD - created before implementation to verify:
1. All functions are importable from the new animation.config path
2. Re-exports through animation/__init__.py work
3. Functionality is identical to the original implementation
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestAnimationConfigImports:
    """Test that all scale bar functions are importable from animation.config."""

    def test_import_scale_bar_config(self):
        """Test ScaleBarConfig is importable from animation.config."""
        from neurospatial.animation.config import ScaleBarConfig

        config = ScaleBarConfig()
        assert config.position == "lower right"

    def test_import_add_scale_bar_to_axes(self):
        """Test add_scale_bar_to_axes is importable from animation.config."""
        from neurospatial.animation.config import add_scale_bar_to_axes

        assert callable(add_scale_bar_to_axes)

    def test_import_compute_nice_length(self):
        """Test compute_nice_length is importable from animation.config."""
        from neurospatial.animation.config import compute_nice_length

        assert callable(compute_nice_length)

    def test_import_configure_napari_scale_bar(self):
        """Test configure_napari_scale_bar is importable from animation.config."""
        from neurospatial.animation.config import configure_napari_scale_bar

        assert callable(configure_napari_scale_bar)

    def test_import_format_scale_label(self):
        """Test format_scale_label is importable from animation.config."""
        from neurospatial.animation.config import format_scale_label

        assert callable(format_scale_label)


class TestAnimationInitReExports:
    """Test that scale bar functions are re-exported from animation/__init__.py."""

    def test_import_scale_bar_config_from_animation(self):
        """Test ScaleBarConfig is importable from animation."""
        from neurospatial.animation import ScaleBarConfig

        config = ScaleBarConfig()
        assert config.position == "lower right"

    def test_import_add_scale_bar_to_axes_from_animation(self):
        """Test add_scale_bar_to_axes is importable from animation."""
        from neurospatial.animation import add_scale_bar_to_axes

        assert callable(add_scale_bar_to_axes)

    def test_import_compute_nice_length_from_animation(self):
        """Test compute_nice_length is importable from animation."""
        from neurospatial.animation import compute_nice_length

        assert callable(compute_nice_length)

    def test_import_configure_napari_scale_bar_from_animation(self):
        """Test configure_napari_scale_bar is importable from animation."""
        from neurospatial.animation import configure_napari_scale_bar

        assert callable(configure_napari_scale_bar)

    def test_import_format_scale_label_from_animation(self):
        """Test format_scale_label is importable from animation."""
        from neurospatial.animation import format_scale_label

        assert callable(format_scale_label)


class TestModuleStructure:
    """Test module structure and __all__ exports."""

    def test_config_module_all(self):
        """Test animation.config has correct __all__ exports."""
        import neurospatial.animation.config as config_module

        expected_exports = {
            "ScaleBarConfig",
            "add_scale_bar_to_axes",
            "compute_nice_length",
            "configure_napari_scale_bar",
            "format_scale_label",
        }
        assert set(config_module.__all__) == expected_exports

    def test_animation_module_includes_config_exports(self):
        """Test animation.__all__ includes scale bar exports."""
        import neurospatial.animation as animation_module

        # Scale bar exports should be in animation's __all__
        for name in [
            "ScaleBarConfig",
            "add_scale_bar_to_axes",
            "compute_nice_length",
            "configure_napari_scale_bar",
            "format_scale_label",
        ]:
            assert name in animation_module.__all__, f"{name} not in animation.__all__"


class TestFunctionality:
    """Test that moved functions work correctly."""

    def test_compute_nice_length_basic(self):
        """Test compute_nice_length returns correct values."""
        from neurospatial.animation.config import compute_nice_length

        assert compute_nice_length(100) == 20
        assert compute_nice_length(50) == 10

    def test_format_scale_label_basic(self):
        """Test format_scale_label works correctly."""
        from neurospatial.animation.config import format_scale_label

        assert format_scale_label(10, "cm") == "10 cm"
        assert format_scale_label(2.5, "m") == "2.5 m"
        assert format_scale_label(10, None) == "10"

    def test_scale_bar_config_creation(self):
        """Test ScaleBarConfig creation with defaults and custom values."""
        from neurospatial.animation.config import ScaleBarConfig

        # Defaults
        config = ScaleBarConfig()
        assert config.length is None
        assert config.position == "lower right"
        assert config.font_size == 10

        # Custom
        config = ScaleBarConfig(length=20.0, position="upper left")
        assert config.length == 20.0
        assert config.position == "upper left"

    def test_scale_bar_config_frozen(self):
        """Test ScaleBarConfig is immutable."""
        from neurospatial.animation.config import ScaleBarConfig

        config = ScaleBarConfig()
        with pytest.raises(AttributeError):
            config.length = 10.0

    def test_add_scale_bar_to_axes(self):
        """Test add_scale_bar_to_axes adds artist."""
        from neurospatial.animation.config import add_scale_bar_to_axes

        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        artist = add_scale_bar_to_axes(ax, extent=100, units="cm")
        assert len(ax.artists) > 0 or artist in ax.get_children()
        plt.close(fig)

    def test_auto_contrast_color(self):
        """Test _auto_contrast_color internal function."""
        from neurospatial.animation.config import _auto_contrast_color

        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        assert _auto_contrast_color(ax) == "white"

        ax.set_facecolor("white")
        assert _auto_contrast_color(ax) == "black"
        plt.close(fig)

    def test_compute_nice_length_edge_cases(self):
        """Test compute_nice_length edge cases."""
        from neurospatial.animation.config import compute_nice_length

        # Zero extent should raise
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(0.0)

        # Negative extent should raise
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(-10.0)

        # Inf should raise
        with pytest.raises(ValueError, match="must be a positive finite number"):
            compute_nice_length(np.inf)

    def test_compute_nice_length_125_rule(self):
        """Test compute_nice_length follows 1-2-5 rule."""
        from neurospatial.animation.config import compute_nice_length

        for extent in [7, 13, 27, 73, 156, 0.3, 0.7]:
            length = compute_nice_length(extent)
            mantissa = length / (10 ** np.floor(np.log10(length)))
            valid_mantissas = (1.0, 2.0, 5.0)
            assert any(np.isclose(mantissa, v, atol=1e-9) for v in valid_mantissas)
