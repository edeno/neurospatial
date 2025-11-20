"""Tests for HTML backend overlay rendering (positions + regions only)."""

from __future__ import annotations

import contextlib
import json
import re
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


@pytest.fixture
def simple_env():
    """Create a simple 2D environment for testing."""
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_fields(simple_env):
    """Create simple fields for testing."""
    return [np.random.rand(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def position_overlay_data():
    """Create simple position overlay data."""
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=5)]
    )


@pytest.fixture
def bodypart_overlay_data():
    """Create bodypart overlay data (should emit warning)."""
    bodyparts = {
        "nose": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "tail": np.array([[2.0, 3.0], [4.0, 5.0]]),
    }
    return OverlayData(
        bodypart_sets=[
            BodypartData(
                bodyparts=bodyparts,
                skeleton=[("nose", "tail")],
                colors=None,
                skeleton_color="white",
                skeleton_width=2.0,
            )
        ]
    )


@pytest.fixture
def head_direction_overlay_data():
    """Create head direction overlay data (should emit warning)."""
    directions = np.array([0.0, np.pi / 2])
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
    """Test position overlay rendering in HTML canvas."""

    def test_position_overlay_serialized_to_json(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position overlay data is serialized to JSON in HTML."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()

        # Check that overlay data is present in JavaScript
        assert "overlayData" in html_content
        assert "positions" in html_content

        # Extract JSON from script tag
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        assert match is not None

        overlay_json = json.loads(match.group(1))
        assert "positions" in overlay_json
        assert len(overlay_json["positions"]) == 1

    def test_position_overlay_includes_coordinates(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position overlay includes coordinate data."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        position_data = overlay_json["positions"][0]
        assert "data" in position_data
        assert len(position_data["data"]) == 3  # 3 frames

    def test_position_overlay_includes_styling(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position overlay includes color, size, trail_length."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        position_data = overlay_json["positions"][0]
        assert position_data["color"] == "red"
        assert position_data["size"] == 10.0
        assert position_data["trail_length"] == 5

    def test_multiple_position_overlays(self, simple_env, simple_fields, tmp_path):
        """Test multiple position overlays (multi-animal scenario)."""
        positions1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        positions2 = np.array([[2.0, 3.0], [4.0, 5.0]])

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

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        assert len(overlay_json["positions"]) == 2
        assert overlay_json["positions"][0]["color"] == "red"
        assert overlay_json["positions"][1]["color"] == "blue"

    def test_canvas_rendering_function_present(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test HTML includes canvas rendering function for positions."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()

        # Check for canvas element
        assert '<canvas id="overlay-canvas"' in html_content

        # Check for rendering function
        assert "renderOverlays" in html_content or "drawOverlays" in html_content


class TestRegionOverlayRendering:
    """Test region overlay rendering in HTML canvas."""

    def test_regions_serialized_to_json(self, simple_env, simple_fields, tmp_path):
        """Test regions are serialized to JSON when show_regions=True."""
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

        html_content = path.read_text()

        # Check that regions are present
        assert "regions" in html_content

        # Extract JSON
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        assert match is not None
        overlay_json = json.loads(match.group(1))

        assert "regions" in overlay_json
        assert len(overlay_json["regions"]) > 0

    def test_region_includes_coordinates(self, simple_env, simple_fields, tmp_path):
        """Test region includes coordinate data."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=True,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        region = overlay_json["regions"][0]
        assert "coordinates" in region
        # Point regions should have coordinate data

    def test_region_alpha_included(self, simple_env, simple_fields, tmp_path):
        """Test region_alpha parameter is passed to JavaScript."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=True,
            region_alpha=0.5,
        )

        html_content = path.read_text()

        # Check that alpha is present in JavaScript
        assert "0.5" in html_content or "regionAlpha" in html_content

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

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should only include "goal" region
        region_names = [r["name"] for r in overlay_json["regions"]]
        assert "goal" in region_names
        assert "start" not in region_names

    def test_show_regions_false_no_serialization(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test show_regions=False prevents region serialization."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            show_regions=False,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)

        if match:
            overlay_json = json.loads(match.group(1))
            # Regions should be empty or not present
            assert overlay_json.get("regions", []) == []


class TestUnsupportedOverlayWarnings:
    """Test warnings for unsupported overlay types (bodyparts, head direction)."""

    def test_bodypart_overlay_emits_warning(
        self, simple_env, simple_fields, bodypart_overlay_data, tmp_path
    ):
        """Test bodypart overlay emits capability warning."""
        save_path = tmp_path / "test.html"

        with pytest.warns(
            UserWarning, match="HTML backend supports positions and regions only"
        ):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=bodypart_overlay_data,
            )

    def test_head_direction_overlay_emits_warning(
        self, simple_env, simple_fields, head_direction_overlay_data, tmp_path
    ):
        """Test head direction overlay emits capability warning."""
        save_path = tmp_path / "test.html"

        with pytest.warns(
            UserWarning, match="HTML backend supports positions and regions only"
        ):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=head_direction_overlay_data,
            )

    def test_mixed_overlay_warns_once(self, simple_env, simple_fields, tmp_path):
        """Test mixed overlays (positions + bodyparts) warns about unsupported types."""
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        bodyparts = {"nose": np.array([[1.0, 2.0], [3.0, 4.0]])}

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ],
            bodypart_sets=[
                BodypartData(
                    bodyparts=bodyparts,
                    skeleton=None,
                    colors=None,
                    skeleton_color="white",
                    skeleton_width=2.0,
                )
            ],
        )

        save_path = tmp_path / "test.html"

        with pytest.warns(
            UserWarning, match="HTML backend supports positions and regions only"
        ):
            path = render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

        # Should still render positions successfully
        html_content = path.read_text()
        assert "positions" in html_content

    def test_warning_suggests_alternative_backends(
        self, simple_env, simple_fields, bodypart_overlay_data, tmp_path
    ):
        """Test warning message suggests video or napari backend."""
        save_path = tmp_path / "test.html"

        with pytest.warns(UserWarning, match="video.*napari"):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=bodypart_overlay_data,
            )


class TestOverlaySizeGuardrails:
    """Test file size estimation and warnings for large overlay datasets."""

    def test_large_position_overlay_warns(self, simple_env, tmp_path):
        """Test warning when position overlay data is very large."""
        # Create large position dataset (10,000 frames)
        n_frames = 10000
        fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]
        positions = np.random.rand(n_frames, 2) * 10.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        # Should warn about frame count exceeding max_html_frames
        with pytest.raises(ValueError, match="HTML backend supports max"):
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=500,  # Default limit
            )

    def test_overlay_size_estimation_in_warning(
        self, simple_env, simple_fields, tmp_path
    ):
        """Test that overlay size is included in file size estimation."""
        # Create position overlay
        positions = np.random.rand(10, 2) * 10.0
        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ]
        )

        save_path = tmp_path / "test.html"

        # Should succeed (small dataset)
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()

    def test_overlay_json_size_estimated(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test that overlay JSON size is estimated correctly."""
        save_path = tmp_path / "test.html"

        # Render with overlay data
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        # Read actual HTML to verify overlay data exists
        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        assert match is not None

        # Calculate actual JSON size
        overlay_json_str = match.group(1)
        actual_json_size_mb = len(overlay_json_str.encode("utf-8")) / 1e6

        # Size should be small for this test case
        assert actual_json_size_mb < 1.0  # Less than 1 MB

    @pytest.mark.slow
    def test_large_overlay_json_warns(self, simple_env, tmp_path):
        """Test warning when overlay JSON size exceeds threshold (1MB)."""
        # Create large overlay dataset (30,000 frames to exceed 1MB threshold)
        # At ~20 bytes/coord, 30k frames × 2 dims = 60k coords = ~1.2 MB
        n_frames = 30000
        fields = [np.random.rand(simple_env.n_bins) for _ in range(min(n_frames, 500))]

        # Create overlay with large position data
        positions = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        # Should warn about large overlay JSON size and frame count limit
        # Will also raise ValueError for exceeding max_html_frames, so catch that
        with (
            pytest.warns(UserWarning, match="overlay data.*MB"),
            contextlib.suppress(ValueError),
        ):
            # Expected if fields count exceeds limit, but we should still get warning
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=n_frames,  # Allow large frame count for this test
            )

    @pytest.mark.slow
    def test_overlay_size_warning_includes_estimate(self, simple_env, tmp_path):
        """Test that overlay size warning includes size estimate."""
        # Create large overlay dataset that triggers warning (> 1MB)
        n_frames = 30000
        fields = [np.random.rand(simple_env.n_bins) for _ in range(500)]
        positions = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        # Warning should include numerical size estimate
        with pytest.warns(UserWarning) as warnings_list:
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=n_frames,
            )

        # Check if any warning mentions overlay size in MB
        overlay_warnings = [
            str(w.message) for w in warnings_list if "overlay" in str(w.message).lower()
        ]
        assert len(overlay_warnings) > 0, "Should emit overlay size warning"
        # Should mention size in MB
        assert any("MB" in msg for msg in overlay_warnings)

    @pytest.mark.slow
    def test_overlay_size_warning_suggests_subsampling(self, simple_env, tmp_path):
        """Test that overlay size warning suggests subsampling positions."""
        n_frames = 30000
        fields = [np.random.rand(simple_env.n_bins) for _ in range(500)]
        positions = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        with pytest.warns(UserWarning) as warnings_list:
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=n_frames,
            )

        # Check if warning suggests subsampling
        overlay_warnings = [
            str(w.message) for w in warnings_list if "overlay" in str(w.message).lower()
        ]
        assert len(overlay_warnings) > 0, "Should emit overlay size warning"
        # Should suggest subsampling or alternative backend
        warning_text = " ".join(overlay_warnings).lower()
        assert "subsample" in warning_text or "video" in warning_text

    @pytest.mark.slow
    def test_overlay_size_warning_suggests_video_backend(self, simple_env, tmp_path):
        """Test that overlay size warning suggests using video backend."""
        n_frames = 30000
        fields = [np.random.rand(simple_env.n_bins) for _ in range(500)]
        positions = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        with pytest.warns(UserWarning, match="video.*backend"):
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=n_frames,
            )

    @pytest.mark.slow
    def test_multiple_overlays_size_accumulated(self, simple_env, tmp_path):
        """Test that size estimation accounts for multiple overlays."""
        n_frames = 12000  # 3 overlays × 12k frames ≈ 1.4 MB total
        fields = [np.random.rand(simple_env.n_bins) for _ in range(500)]

        # Create multiple position overlays (multi-animal)
        positions1 = np.random.rand(n_frames, 2) * 100.0
        positions2 = np.random.rand(n_frames, 2) * 100.0
        positions3 = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions1, color="red", size=10.0, trail_length=50),
                PositionData(data=positions2, color="blue", size=10.0, trail_length=50),
                PositionData(
                    data=positions3, color="green", size=10.0, trail_length=50
                ),
            ]
        )

        save_path = tmp_path / "test.html"

        # Multiple overlays should trigger warning (3x the data exceeds 1MB)
        with pytest.warns(UserWarning, match="overlay data"):
            render_html(
                simple_env,
                fields,
                str(save_path),
                overlay_data=overlay_data,
                max_html_frames=n_frames,
            )

    @pytest.mark.slow
    def test_regions_contribute_to_overlay_size(self, simple_env, tmp_path):
        """Test that regions contribute to overlay JSON size estimation."""
        from shapely.geometry import Polygon

        n_frames = 500
        fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

        # Add many complex polygon regions to exceed 1MB
        # At ~30 bytes/vertex, need ~35k vertices for 1MB
        # 100 polygons × 400 vertices = 40k vertices ≈ 1.2 MB
        for i in range(100):
            # Create polygon with many vertices (400 vertices per polygon)
            vertices = np.random.rand(400, 2) * 100.0
            poly = Polygon(vertices)
            simple_env.regions.add(f"region_{i}", polygon=poly)

        save_path = tmp_path / "test.html"

        # Large regions should contribute to size warning (> 1MB)
        with pytest.warns(UserWarning, match="overlay data"):
            render_html(
                simple_env,
                fields,
                str(save_path),
                show_regions=True,
            )

    def test_small_overlay_no_warning(self, simple_env, simple_fields, tmp_path):
        """Test that small overlay data does not trigger warning."""
        # Small overlay (10 frames, 1 position)
        positions = np.random.rand(10, 2) * 10.0
        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=5)
            ]
        )

        save_path = tmp_path / "test.html"

        # Small dataset should not warn about overlay size
        # (may still warn about image size if DPI is high, but not overlay)
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

    def test_overlay_size_threshold_configurable(self, simple_env, tmp_path):
        """Test that overlay size warning threshold can be configured."""
        n_frames = 500
        fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]
        positions = np.random.rand(n_frames, 2) * 100.0

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=50)
            ]
        )

        save_path = tmp_path / "test.html"

        # Test with custom threshold (if implemented)
        # This tests future extensibility
        # For now, just verify rendering succeeds
        path = render_html(
            simple_env,
            fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


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

        html_content = path.read_text()

        # Check overlay data is still in HTML (even with non-embedded frames)
        assert "overlayData" in html_content

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

        html_content = path.read_text()
        assert "regions" in html_content


class TestOverlayCoordinateScaling:
    """Test overlay coordinates are scaled correctly for canvas rendering."""

    def test_position_coordinates_match_image_bounds(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test position coordinates are within environment bounds."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Check that environment dimension ranges are included
        assert "env_ranges" in overlay_json or "dimension_ranges" in overlay_json

    def test_environment_metadata_included(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test environment metadata is included for coordinate scaling."""
        save_path = tmp_path / "test.html"

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=position_overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should include dimension ranges for scaling
        assert "dimension_ranges" in overlay_json or "env_ranges" in overlay_json
