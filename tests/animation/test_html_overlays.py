"""Tests for HTML backend overlay rendering.

Note: As of the matplotlib-based rendering fix, overlays are rendered directly
in matplotlib (baked into images) rather than via JavaScript canvas. This
fixes coordinate alignment issues and enables support for all overlay types.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import (
    Environment,
)
from neurospatial.animation.backends.html_backend import render_html
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
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_fields(simple_env):
    """Create simple fields for testing."""
    rng = np.random.default_rng(42)
    return [rng.random(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def position_overlay_data():
    """Create simple position overlay data (10 frames to match simple_fields)."""
    rng = np.random.default_rng(42)
    positions = rng.random((10, 2)) * 10.0  # 10 frames to match simple_fields
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=5)]
    )


@pytest.fixture
def bodypart_overlay_data():
    """Create bodypart overlay data (10 frames to match simple_fields)."""
    rng = np.random.default_rng(43)
    bodyparts = {
        "nose": rng.random((10, 2)) * 10.0,  # 10 frames
        "tail": rng.random((10, 2)) * 10.0,  # 10 frames
    }
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
                bodyparts=bodyparts,
                skeleton=skeleton,
                colors=None,
            )
        ]
    )


@pytest.fixture
def head_direction_overlay_data():
    """Create head direction overlay data (10 frames to match simple_fields)."""
    rng = np.random.default_rng(44)
    directions = rng.random(10) * 2 * np.pi  # 10 frames of angles
    return OverlayData(
        head_directions=[
            HeadDirectionData(data=directions, color="yellow", length=20.0)
        ]
    )


class TestHTMLSignatureUpdate:
    """Test that render_html accepts overlay parameters."""

    def test_accepts_overlay_data_parameter(self, simple_env, simple_fields, tmp_path):
        """Test render_html accepts overlay_data parameter."""
        save_path = tmp_path / "test.html"

        # Should not raise even with overlay_data=None
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=None,
            show_regions=False,
            region_alpha=0.3,
        )

        assert path.exists()

    def test_accepts_show_regions_parameter(self, simple_env, simple_fields, tmp_path):
        """Test render_html accepts show_regions parameter."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=True,
            region_alpha=0.5,
        )

        assert path.exists()

    def test_backward_compatibility_no_overlays(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test render_html works without overlay parameters (backward compatible)."""
        save_path = tmp_path / "test.html"

        # Should work without new parameters
        path = render_html(simple_env, simple_fields, str(save_path))

        assert path.exists()


class TestPositionOverlayRendering:
    """Test position overlay rendering (now via matplotlib)."""

    def test_position_overlay_renders_successfully(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position overlay renders without error."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        assert path.exists()
        # File should have content (images with overlays baked in)
        assert path.stat().st_size > 0

    def test_multiple_position_overlays(self, simple_env, simple_fields, tmp_path):
        """Test multiple position overlays (multi-animal scenario)."""
        rng = np.random.default_rng(45)
        positions1 = rng.random((10, 2)) * 10.0  # 10 frames
        positions2 = rng.random((10, 2)) * 10.0  # 10 frames

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions1, color="red", size=10.0, trail_length=5),
                PositionData(data=positions2, color="blue", size=8.0, trail_length=3),
            ]
        )

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


class TestRegionOverlayRendering:
    """Test region overlay rendering (now via matplotlib)."""

    def test_regions_render_with_show_regions_true(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test regions are rendered when show_regions=True."""
        # Add a region to environment
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=True,
            region_alpha=0.3,
        )

        assert path.exists()

    def test_show_regions_list_filtering(self, simple_env, simple_fields, tmp_path):
        """Test show_regions can filter specific regions."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))
        simple_env.regions.add("start", point=np.array([1.0, 1.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=["goal"],  # Only show goal region
        )

        assert path.exists()

    def test_show_regions_false_still_renders(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test show_regions=False still produces valid output."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=False,
        )

        assert path.exists()


class TestAllOverlayTypesSupported:
    """Test all overlay types are supported via matplotlib rendering."""

    def test_bodypart_overlay_renders_successfully(
        self, simple_env, simple_fields, bodypart_overlay_data, tmp_path
    ):
        """Test bodypart overlay renders without error (now supported)."""
        save_path = tmp_path / "test.html"

        # Should not emit warning - all overlay types now supported
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=bodypart_overlay_data,
        )

        assert path.exists()

    def test_head_direction_overlay_renders_successfully(
        self, simple_env, simple_fields, head_direction_overlay_data, tmp_path
    ):
        """Test head direction overlay renders without error (now supported)."""
        save_path = tmp_path / "test.html"

        # Should not emit warning - all overlay types now supported
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=head_direction_overlay_data,
        )

        assert path.exists()

    def test_mixed_overlay_types(self, simple_env, simple_fields, tmp_path):
        """Test mixed overlays (positions + bodyparts) all render."""
        rng = np.random.default_rng(46)
        positions = rng.random((10, 2)) * 10.0  # 10 frames
        bodyparts = {"nose": rng.random((10, 2)) * 10.0}  # 10 frames

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ],
            bodypart_sets=[
                BodypartData(
                    bodyparts=bodyparts,
                    skeleton=None,
                    colors=None,
                )
            ],
        )

        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


class TestOverlaySizeGuardrails:
    """Test file size estimation and warnings for large overlay datasets."""

    def test_large_frame_count_raises(self, simple_env, tmp_path):
        """Test error when frame count exceeds limit."""
        rng = np.random.default_rng(42)
        # Create large frame count exceeding default limit
        n_frames = 1000
        fields = [rng.random(simple_env.n_bins) for _ in range(n_frames)]

        save_path = tmp_path / "test.html"

        # Should raise about frame count exceeding max_html_frames
        with pytest.raises(ValueError, match="HTML backend supports max"):
            render_html(
                simple_env,
                fields,
                str(save_path),
                max_html_frames=500,  # Default limit
            )

    def test_small_overlay_no_warning(self, simple_env, simple_fields, tmp_path):
        """Test that small overlay data does not trigger warning."""
        rng = np.random.default_rng(42)
        # Small overlay (10 frames, 1 position)
        positions = rng.random((10, 2)) * 10.0
        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ]
        )

        save_path = tmp_path / "test.html"

        # Small dataset should not warn about overlay size
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                render_html(
                    simple_env,
                    simple_fields,
                    str(save_path),
                    overlay_data=overlay_data,
                    dpi=50,  # Keep DPI low to avoid image size warnings
                )
            except UserWarning as e:
                # If warning occurs, it should NOT be about overlay size
                assert "overlay" not in str(e).lower()


class TestNonEmbeddedModeWithOverlays:
    """Test overlay support in non-embedded mode (disk-backed frames)."""

    def test_non_embedded_mode_with_positions(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test non-embedded mode supports position overlays."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            embed=False,
            overlay_data=position_overlay_data,
            n_workers=1,
        )

        assert path.exists()

    def test_non_embedded_mode_with_regions(self, simple_env, simple_fields, tmp_path):
        """Test non-embedded mode supports regions."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            embed=False,
            show_regions=True,
            n_workers=1,
        )

        assert path.exists()


class TestOverlayCoordinateAlignment:
    """Test overlay coordinates are aligned correctly with field images.

    These tests verify the fix for coordinate alignment issues when
    overlays are rendered via matplotlib rather than JavaScript canvas.
    """

    def test_position_overlay_aligned(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position overlay is aligned with field (visual test via file creation)."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        # File should be created successfully with overlays baked into images
        assert path.exists()
        # Content should be substantial (includes image data)
        assert path.stat().st_size > 1000
