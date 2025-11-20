"""Integration tests for animation with overlay system.

This module tests end-to-end workflows for animating spatial fields with overlays
across all backends (napari, video, html, widget). Tests verify:

- Each backend correctly renders overlays
- Cross-backend consistency (same config produces equivalent output)
- Multi-animal support (multiple overlays of same type)
- Mixed overlay types in single animation
- Proper integration from Environment.animate_fields() to backend rendering
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neurospatial import (
    BodypartOverlay,
    Environment,
    HeadDirectionOverlay,
    PositionOverlay,
)

# =============================================================================
# Fixtures for Integration Tests
# =============================================================================


@pytest.fixture
def simple_env():
    """Create a simple 2D environment for testing."""
    positions = np.random.uniform(0, 100, (200, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    env.frame = "test"
    return env


@pytest.fixture
def spatial_fields():
    """Create simple spatial fields for animation (20 frames, varying bins)."""

    # Return function to generate fields for any environment
    def _make_fields(n_bins: int, n_frames: int = 20) -> np.ndarray:
        return np.random.rand(n_frames, n_bins)

    return _make_fields


@pytest.fixture
def position_overlay_data():
    """Create position overlay with 20 time points."""
    data = np.random.uniform(0, 100, (20, 2))
    times = np.linspace(0, 2.0, 20)
    return PositionOverlay(
        data=data,
        times=times,
        color="red",
        size=10.0,
        trail_length=5,
    )


@pytest.fixture
def bodypart_overlay_data():
    """Create bodypart overlay with skeleton (20 time points)."""
    bodyparts = {
        "nose": np.random.uniform(0, 100, (20, 2)),
        "head": np.random.uniform(0, 100, (20, 2)),
        "body": np.random.uniform(0, 100, (20, 2)),
        "tail": np.random.uniform(0, 100, (20, 2)),
    }
    skeleton = [("nose", "head"), ("head", "body"), ("body", "tail")]
    times = np.linspace(0, 2.0, 20)

    return BodypartOverlay(
        data=bodyparts,
        times=times,
        skeleton=skeleton,
        colors={"nose": "yellow", "head": "orange", "body": "red", "tail": "blue"},
        skeleton_color="white",
        skeleton_width=2.0,
    )


@pytest.fixture
def head_direction_overlay_data():
    """Create head direction overlay (20 time points)."""
    angles = np.linspace(0, 2 * np.pi, 20)
    times = np.linspace(0, 2.0, 20)

    return HeadDirectionOverlay(
        data=angles,
        times=times,
        color="yellow",
        length=10.0,
    )


@pytest.fixture
def env_with_regions(simple_env):
    """Add regions to simple environment."""
    simple_env.regions.add("goal", point=np.array([75.0, 75.0]))
    simple_env.regions.add("start", point=np.array([25.0, 25.0]))
    return simple_env


# =============================================================================
# Test Class: Napari Backend Integration
# =============================================================================


class TestNapariBackendIntegration:
    """Test Napari backend end-to-end with all overlay types."""

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_napari_with_position_overlay(
        self, mock_napari, simple_env, spatial_fields, position_overlay_data
    ):
        """Test Napari backend renders position overlay correctly."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3  # time + 2 spatial dimensions
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with position overlay
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[position_overlay_data],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer
        mock_napari.Viewer.assert_called_once()

        # Verify position layers added (tracks + points)
        assert mock_viewer.add_tracks.called or mock_viewer.add_points.called

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_napari_with_bodypart_overlay(
        self, mock_napari, simple_env, spatial_fields, bodypart_overlay_data
    ):
        """Test Napari backend renders bodypart overlay with skeleton."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with bodypart overlay
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[bodypart_overlay_data],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify bodypart layers added (points for keypoints, shapes for skeleton)
        assert mock_viewer.add_points.called
        assert mock_viewer.add_shapes.called

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_napari_with_head_direction_overlay(
        self, mock_napari, simple_env, spatial_fields, head_direction_overlay_data
    ):
        """Test Napari backend renders head direction overlay."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with head direction overlay
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[head_direction_overlay_data],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify vectors layer added
        assert mock_viewer.add_vectors.called

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_napari_with_all_overlays(
        self,
        mock_napari,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
        head_direction_overlay_data,
    ):
        """Test Napari backend with all overlay types simultaneously."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with all overlay types
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[
                position_overlay_data,
                bodypart_overlay_data,
                head_direction_overlay_data,
            ],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify all layer types added
        assert mock_viewer.add_points.called  # For position and bodyparts
        assert mock_viewer.add_shapes.called  # For skeleton
        assert mock_viewer.add_vectors.called  # For head direction

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_napari_with_regions(self, mock_napari, env_with_regions, spatial_fields):
        """Test Napari backend renders regions."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(env_with_regions.n_bins, n_frames=20)

        # Call animate_fields with show_regions=True
        result = env_with_regions.animate_fields(
            fields,
            backend="napari",
            show_regions=True,
            region_alpha=0.3,
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify shapes layer added for regions
        assert mock_viewer.add_shapes.called


# =============================================================================
# Test Class: Video Backend Integration
# =============================================================================


class TestVideoBackendIntegration:
    """Test Video backend end-to-end with all overlay types."""

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    def test_video_with_position_overlay(
        self,
        mock_subprocess,
        mock_parallel,
        simple_env,
        spatial_fields,
        position_overlay_data,
    ):
        """Test video backend renders position overlay correctly."""
        # Mock successful rendering and encoding
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            # Call animate_fields with position overlay
            _ = simple_env.animate_fields(
                fields,
                backend="video",
                overlays=[position_overlay_data],
                save_path=str(save_path),
                fps=10,
                n_workers=1,  # Serial for pickle safety in tests
            )

            # Verify parallel rendering called
            assert mock_parallel.called

            # Verify overlay_data passed to parallel renderer
            call_args = mock_parallel.call_args
            assert "overlay_data" in call_args[1]

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    def test_video_with_bodypart_overlay(
        self,
        mock_subprocess,
        mock_parallel,
        simple_env,
        spatial_fields,
        bodypart_overlay_data,
    ):
        """Test video backend renders bodypart overlay with skeleton."""
        # Mock successful rendering and encoding
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            # Call animate_fields with bodypart overlay
            _ = simple_env.animate_fields(
                fields,
                backend="video",
                overlays=[bodypart_overlay_data],
                save_path=str(save_path),
                fps=10,
                n_workers=1,
            )

            # Verify parallel rendering called
            assert mock_parallel.called

            # Verify overlay_data passed
            call_args = mock_parallel.call_args
            assert "overlay_data" in call_args[1]
            overlay_data = call_args[1]["overlay_data"]
            assert len(overlay_data.bodypart_sets) > 0

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    def test_video_with_all_overlays(
        self,
        mock_subprocess,
        mock_parallel,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
        head_direction_overlay_data,
    ):
        """Test video backend with all overlay types simultaneously."""
        # Mock successful rendering and encoding
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            # Call animate_fields with all overlay types
            _ = simple_env.animate_fields(
                fields,
                backend="video",
                overlays=[
                    position_overlay_data,
                    bodypart_overlay_data,
                    head_direction_overlay_data,
                ],
                save_path=str(save_path),
                fps=10,
                n_workers=1,
            )

            # Verify parallel rendering called
            assert mock_parallel.called

            # Verify all overlay types in overlay_data
            call_args = mock_parallel.call_args
            overlay_data = call_args[1]["overlay_data"]
            assert len(overlay_data.positions) > 0
            assert len(overlay_data.bodypart_sets) > 0
            assert len(overlay_data.head_directions) > 0

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    def test_video_with_regions(
        self, mock_subprocess, mock_parallel, env_with_regions, spatial_fields
    ):
        """Test video backend renders regions."""
        # Mock successful rendering and encoding
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(env_with_regions.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            # Call animate_fields with show_regions=True
            _ = env_with_regions.animate_fields(
                fields,
                backend="video",
                show_regions=True,
                region_alpha=0.3,
                save_path=str(save_path),
                fps=10,
                n_workers=1,
            )

            # Verify parallel rendering called
            assert mock_parallel.called

            # Verify show_regions passed
            call_args = mock_parallel.call_args
            assert call_args[1]["show_regions"] is True
            assert call_args[1]["region_alpha"] == 0.3


# =============================================================================
# Test Class: HTML Backend Integration
# =============================================================================


class TestHTMLBackendIntegration:
    """Test HTML backend end-to-end (positions + regions only)."""

    def test_html_with_position_overlay(
        self, simple_env, spatial_fields, position_overlay_data
    ):
        """Test HTML backend renders position overlay correctly."""
        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_animation.html"

            # Call animate_fields with position overlay
            _ = simple_env.animate_fields(
                fields,
                backend="html",
                overlays=[position_overlay_data],
                save_path=str(save_path),
                fps=10,
            )

            # Verify HTML file created
            assert save_path.exists()

            # Read and verify content contains position data
            html_content = save_path.read_text()
            assert (
                "position" in html_content.lower() or "overlay" in html_content.lower()
            )

    def test_html_with_regions(self, env_with_regions, spatial_fields):
        """Test HTML backend renders regions."""
        # Create fields
        fields = spatial_fields(env_with_regions.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_animation.html"

            # Call animate_fields with show_regions=True
            _ = env_with_regions.animate_fields(
                fields,
                backend="html",
                show_regions=True,
                region_alpha=0.3,
                save_path=str(save_path),
                fps=10,
            )

            # Verify HTML file created
            assert save_path.exists()

            # Read and verify content
            html_content = save_path.read_text()
            assert len(html_content) > 0

    def test_html_warns_unsupported_overlays(
        self, simple_env, spatial_fields, bodypart_overlay_data
    ):
        """Test HTML backend warns when bodypart/head direction overlays provided."""
        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_animation.html"

            # Call animate_fields with bodypart overlay (unsupported)
            with pytest.warns(
                UserWarning, match=r"HTML backend.*supports.*position.*region"
            ):
                _ = simple_env.animate_fields(
                    fields,
                    backend="html",
                    overlays=[bodypart_overlay_data],
                    save_path=str(save_path),
                    fps=10,
                )

            # Verify HTML file still created (just without bodypart overlay)
            assert save_path.exists()


# =============================================================================
# Test Class: Widget Backend Integration
# =============================================================================


class TestWidgetBackendIntegration:
    """Test Widget backend end-to-end with overlays."""

    def test_widget_with_position_overlay(
        self, simple_env, spatial_fields, position_overlay_data
    ):
        """Test widget backend renders position overlay correctly."""
        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with position overlay - should execute without error
        _ = simple_env.animate_fields(
            fields,
            backend="widget",
            overlays=[position_overlay_data],
            fps=10,
        )

        # Widget backend returns None but executes successfully
        # (Widget is displayed in notebook, not returned as object)
        assert True  # Test passes if no exception raised

    def test_widget_with_all_overlays(
        self,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
        head_direction_overlay_data,
    ):
        """Test widget backend with all overlay types."""
        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with all overlay types - should execute without error
        _ = simple_env.animate_fields(
            fields,
            backend="widget",
            overlays=[
                position_overlay_data,
                bodypart_overlay_data,
                head_direction_overlay_data,
            ],
            fps=10,
        )

        # Widget backend returns None but executes successfully
        # (Widget is displayed in notebook, not returned as object)
        assert True  # Test passes if no exception raised


# =============================================================================
# Test Class: Multi-Animal Scenarios
# =============================================================================


class TestMultiAnimalScenarios:
    """Test multi-animal support (multiple overlays of same type)."""

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_multiple_position_overlays(self, mock_napari, simple_env, spatial_fields):
        """Test rendering multiple position overlays (multi-animal)."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Create multiple position overlays (3 animals)
        overlays = [
            PositionOverlay(
                data=np.random.uniform(0, 100, (20, 2)),
                times=np.linspace(0, 2.0, 20),
                color=color,
                size=10.0,
                trail_length=5,
            )
            for color in ["red", "blue", "green"]
        ]

        # Call animate_fields with multiple position overlays
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=overlays,
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify multiple position layers added
        assert mock_viewer.add_points.call_count >= 3

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_multiple_bodypart_overlays(self, mock_napari, simple_env, spatial_fields):
        """Test rendering multiple bodypart overlays (multi-animal)."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Create multiple bodypart overlays (2 animals)
        bodyparts_animal1 = {
            "nose": np.random.uniform(0, 100, (20, 2)),
            "tail": np.random.uniform(0, 100, (20, 2)),
        }
        bodyparts_animal2 = {
            "nose": np.random.uniform(0, 100, (20, 2)),
            "tail": np.random.uniform(0, 100, (20, 2)),
        }

        overlays = [
            BodypartOverlay(
                data=bodyparts_animal1,
                times=np.linspace(0, 2.0, 20),
                skeleton=[("nose", "tail")],
                skeleton_color="red",
            ),
            BodypartOverlay(
                data=bodyparts_animal2,
                times=np.linspace(0, 2.0, 20),
                skeleton=[("nose", "tail")],
                skeleton_color="blue",
            ),
        ]

        # Call animate_fields with multiple bodypart overlays
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=overlays,
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify multiple bodypart layers added
        assert mock_viewer.add_points.call_count >= 2


# =============================================================================
# Test Class: Cross-Backend Consistency
# =============================================================================


class TestCrossBackendConsistency:
    """Test that same overlay config produces consistent results across backends."""

    def test_overlay_data_consistent_across_backends(
        self,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
    ):
        """Test overlay conversion produces same internal data for all backends."""
        # Import the conversion function
        from neurospatial.animation.overlays import (
            _build_frame_times,
            _convert_overlays_to_data,
        )

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)
        n_frames = len(fields)

        # Build frame times
        frame_times = _build_frame_times(fps=10, n_frames=n_frames, frame_times=None)

        # Convert overlays
        overlay_data = _convert_overlays_to_data(
            overlays=[position_overlay_data, bodypart_overlay_data],
            frame_times=frame_times,
            n_frames=n_frames,
            env=simple_env,
        )

        # Verify overlay data is consistent
        assert len(overlay_data.positions) == 1
        assert len(overlay_data.bodypart_sets) == 1
        assert overlay_data.positions[0].data.shape == (n_frames, 2)

        # Verify bodypart data has correct shape
        for _bodypart_name, bodypart_array in overlay_data.bodypart_sets[
            0
        ].bodyparts.items():
            assert bodypart_array.shape == (n_frames, 2)

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_same_config_napari_and_video(
        self,
        mock_napari,
        mock_subprocess,
        mock_parallel,
        simple_env,
        spatial_fields,
        position_overlay_data,
    ):
        """Test same overlay config works for both Napari and video backends."""
        # Setup mocks with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Test with Napari
        result_napari = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[position_overlay_data],
            fps=10,
        )
        assert result_napari is mock_viewer

        # Test with Video
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            _ = simple_env.animate_fields(
                fields,
                backend="video",
                overlays=[position_overlay_data],
                save_path=str(save_path),
                fps=10,
                n_workers=1,
            )

            # Verify both backends processed overlays
            assert mock_viewer.add_points.called or mock_viewer.add_tracks.called
            assert mock_parallel.called


# =============================================================================
# Test Class: Mixed Overlay Types
# =============================================================================


class TestMixedOverlayTypes:
    """Test animations with mixed overlay types in single call."""

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_position_and_bodypart_together(
        self,
        mock_napari,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
    ):
        """Test rendering position and bodypart overlays together."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with mixed overlays
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[position_overlay_data, bodypart_overlay_data],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify both overlay types rendered
        assert mock_viewer.add_points.called
        assert mock_viewer.add_shapes.called  # For skeleton

    @patch("neurospatial.animation.backends.napari_backend.napari")
    def test_all_three_overlay_types_together(
        self,
        mock_napari,
        simple_env,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
        head_direction_overlay_data,
    ):
        """Test rendering all three overlay types simultaneously."""
        # Setup mock viewer with proper dims configuration
        mock_viewer = MagicMock()
        mock_viewer.dims.ndim = 3
        mock_viewer.dims.ndisplay = 2
        mock_viewer.dims.current_step = (0, 0, 0)
        mock_napari.Viewer.return_value = mock_viewer

        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Call animate_fields with all overlay types
        result = simple_env.animate_fields(
            fields,
            backend="napari",
            overlays=[
                position_overlay_data,
                bodypart_overlay_data,
                head_direction_overlay_data,
            ],
            fps=10,
        )

        # Verify viewer created
        assert result is mock_viewer

        # Verify all three overlay types rendered
        assert mock_viewer.add_points.called  # Position and bodyparts
        assert mock_viewer.add_shapes.called  # Skeleton
        assert mock_viewer.add_vectors.called  # Head direction

    @patch("neurospatial.animation._parallel.parallel_render_frames")
    @patch("neurospatial.animation.backends.video_backend.subprocess.run")
    def test_mixed_overlays_with_regions_video(
        self,
        mock_subprocess,
        mock_parallel,
        env_with_regions,
        spatial_fields,
        position_overlay_data,
        bodypart_overlay_data,
    ):
        """Test video backend with mixed overlays plus regions."""
        # Mock successful rendering
        mock_parallel.return_value = None
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create fields
        fields = spatial_fields(env_with_regions.n_bins, n_frames=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_video.mp4"

            # Call animate_fields with mixed overlays and regions
            _ = env_with_regions.animate_fields(
                fields,
                backend="video",
                overlays=[position_overlay_data, bodypart_overlay_data],
                show_regions=True,
                region_alpha=0.3,
                save_path=str(save_path),
                fps=10,
                n_workers=1,
            )

            # Verify parallel rendering called with all parameters
            assert mock_parallel.called
            call_args = mock_parallel.call_args[1]
            assert call_args["show_regions"] is True
            assert call_args["region_alpha"] == 0.3
            assert call_args["overlay_data"] is not None


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_mismatched_dimensions_raises_error(self, simple_env, spatial_fields):
        """Test that overlay with wrong dimensions raises error."""
        # Create fields
        fields = spatial_fields(simple_env.n_bins, n_frames=20)

        # Create 3D position overlay for 2D environment
        invalid_overlay = PositionOverlay(
            data=np.random.uniform(0, 100, (20, 3)),  # 3D, but env is 2D
            times=np.linspace(0, 2.0, 20),
            color="red",
        )

        # Try to animate - should raise dimension mismatch error
        with pytest.raises(ValueError, match=r"dimension|shape"):
            simple_env.animate_fields(
                fields,
                backend="napari",
                overlays=[invalid_overlay],
                fps=10,
            )
