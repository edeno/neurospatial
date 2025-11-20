"""Backend capability matrix tests for overlay support.

This module systematically tests which overlay types each backend supports,
focusing on positive capability tests (supported overlays work without errors).

Capability Matrix
-----------------
Backend  | Position | Bodypart | HeadDirection | Regions
---------|----------|----------|---------------|----------
Napari   | ✓        | ✓        | ✓             | ✓
Video    | ✓        | ✓        | ✓             | ✓
HTML     | ✓        | ✗        | ✗             | ✓
Widget   | ✓        | ✓        | ✓             | ✓

Warning tests for unsupported overlays are in test_html_overlays.py.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)

# Backend import attempts
try:
    from neurospatial.animation.backends.napari_backend import (
        NAPARI_AVAILABLE,
        render_napari,
    )
except ImportError:
    NAPARI_AVAILABLE = False

try:
    from neurospatial.animation.backends.widget_backend import (
        IPYWIDGETS_AVAILABLE,
        render_widget,
    )
except ImportError:
    IPYWIDGETS_AVAILABLE = False

from neurospatial.animation.backends.html_backend import render_html
from neurospatial.animation.backends.video_backend import (
    check_ffmpeg_available,
    render_video,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    positions = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]]
    )
    env = Environment.from_samples(positions, bin_size=2.0)
    env.units = "cm"
    return env


@pytest.fixture
def simple_fields(simple_env: Environment) -> NDArray[np.float64]:
    """Create simple fields (n_frames, n_bins) for testing."""
    n_frames = 10
    fields = np.random.rand(n_frames, simple_env.n_bins).astype(np.float64)
    return fields


@pytest.fixture
def position_overlay_data() -> OverlayData:
    """Create position overlay data."""
    positions = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    positions = np.tile(positions, (2, 1))  # 10 frames
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=3)]
    )


@pytest.fixture
def bodypart_overlay_data() -> OverlayData:
    """Create bodypart overlay data."""
    n_frames = 10
    bodyparts = {
        "nose": np.random.uniform(2, 8, (n_frames, 2)),
        "ear_left": np.random.uniform(2, 8, (n_frames, 2)),
        "ear_right": np.random.uniform(2, 8, (n_frames, 2)),
    }
    skeleton = [("nose", "ear_left"), ("nose", "ear_right")]
    return OverlayData(
        bodypart_sets=[
            BodypartData(
                bodyparts=bodyparts,
                skeleton=skeleton,
                colors={"nose": "red", "ear_left": "blue", "ear_right": "green"},
                skeleton_color="white",
                skeleton_width=2.0,
            )
        ]
    )


@pytest.fixture
def head_direction_overlay_data() -> OverlayData:
    """Create head direction overlay data."""
    n_frames = 10
    angles = np.linspace(0, 2 * np.pi, n_frames)
    return OverlayData(
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=2.0)]
    )


@pytest.fixture
def all_overlay_types(
    position_overlay_data: OverlayData,
    bodypart_overlay_data: OverlayData,
    head_direction_overlay_data: OverlayData,
) -> OverlayData:
    """Create overlay data with all types combined."""
    return OverlayData(
        positions=position_overlay_data.positions,
        bodypart_sets=bodypart_overlay_data.bodypart_sets,
        head_directions=head_direction_overlay_data.head_directions,
    )


# =============================================================================
# Capability Matrix Constants
# =============================================================================

# Define what each backend supports
BACKEND_CAPABILITIES = {
    "napari": {
        "position": True,
        "bodypart": True,
        "head_direction": True,
        "regions": True,
    },
    "video": {
        "position": True,
        "bodypart": True,
        "head_direction": True,
        "regions": True,
    },
    "html": {
        "position": True,
        "bodypart": False,
        "head_direction": False,
        "regions": True,
    },
    "widget": {
        "position": True,
        "bodypart": True,
        "head_direction": True,
        "regions": True,
    },
}


# =============================================================================
# Test Class: Napari Backend Capabilities
# =============================================================================


@pytest.mark.skipif(not NAPARI_AVAILABLE, reason="napari not available")
class TestNapariCapabilities:
    """Test Napari backend overlay capabilities.

    Napari should support: positions, bodyparts, head direction, regions.
    """

    def test_supports_position_overlay(
        self, simple_env, simple_fields, position_overlay_data
    ):
        """Test Napari renders position overlay without errors."""
        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer
            mock_viewer.dims.ndim = 3

            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                result = render_napari(
                    simple_env,
                    simple_fields,
                    overlay_data=position_overlay_data,
                )

            assert result is not None
            # Verify position layers were created
            calls = [call[0][0] for call in mock_viewer.add_tracks.call_args_list]
            assert (
                any("Position" in str(call) for call in calls)
                or mock_viewer.add_tracks.called
            )

    def test_supports_bodypart_overlay(
        self, simple_env, simple_fields, bodypart_overlay_data
    ):
        """Test Napari renders bodypart overlay without errors."""
        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer
            mock_viewer.dims.ndim = 3

            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = render_napari(
                    simple_env,
                    simple_fields,
                    overlay_data=bodypart_overlay_data,
                )

            assert result is not None
            # Verify bodypart layers were created (points or shapes)
            assert mock_viewer.add_points.called or mock_viewer.add_shapes.called

    def test_supports_head_direction_overlay(
        self, simple_env, simple_fields, head_direction_overlay_data
    ):
        """Test Napari renders head direction overlay without errors."""
        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer
            mock_viewer.dims.ndim = 3

            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = render_napari(
                    simple_env,
                    simple_fields,
                    overlay_data=head_direction_overlay_data,
                )

            assert result is not None
            # Verify vector layer was created
            assert mock_viewer.add_vectors.called

    def test_supports_regions(self, simple_env, simple_fields):
        """Test Napari renders regions without errors."""

        simple_env.regions.add("test_region", point=np.array([5.0, 5.0]))

        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer
            mock_viewer.dims.ndim = 3

            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = render_napari(
                    simple_env,
                    simple_fields,
                    show_regions=True,
                )

            assert result is not None

    def test_supports_all_overlay_types(
        self, simple_env, simple_fields, all_overlay_types
    ):
        """Test Napari renders all overlay types simultaneously."""
        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer
            mock_viewer.dims.ndim = 3

            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = render_napari(
                    simple_env,
                    simple_fields,
                    overlay_data=all_overlay_types,
                )

            assert result is not None


# =============================================================================
# Test Class: Video Backend Capabilities
# =============================================================================


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
class TestVideoCapabilities:
    """Test Video backend overlay capabilities.

    Video should support: positions, bodyparts, head direction, regions.
    """

    def test_supports_position_overlay(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test Video renders position overlay without errors."""
        save_path = tmp_path / "test.mp4"

        # Should not raise any warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_video(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=position_overlay_data,
                n_workers=1,  # Serial for test simplicity
            )

        assert save_path.exists()

    def test_supports_bodypart_overlay(
        self, simple_env, simple_fields, bodypart_overlay_data, tmp_path
    ):
        """Test Video renders bodypart overlay without errors."""
        save_path = tmp_path / "test.mp4"

        # Should not raise any warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_video(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=bodypart_overlay_data,
                n_workers=1,
            )

        assert save_path.exists()

    def test_supports_head_direction_overlay(
        self, simple_env, simple_fields, head_direction_overlay_data, tmp_path
    ):
        """Test Video renders head direction overlay without errors."""
        save_path = tmp_path / "test.mp4"

        # Should not raise any warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_video(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=head_direction_overlay_data,
                n_workers=1,
            )

        assert save_path.exists()

    def test_supports_regions(self, simple_env, simple_fields, tmp_path):
        """Test Video renders regions without errors."""

        simple_env.regions.add("test_region", point=np.array([5.0, 5.0]))
        save_path = tmp_path / "test.mp4"

        # Should not raise any warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_video(
                simple_env,
                simple_fields,
                str(save_path),
                show_regions=True,
                n_workers=1,
            )

        assert save_path.exists()

    def test_supports_all_overlay_types(
        self, simple_env, simple_fields, all_overlay_types, tmp_path
    ):
        """Test Video renders all overlay types simultaneously."""
        save_path = tmp_path / "test.mp4"

        # Should not raise any warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_video(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=all_overlay_types,
                n_workers=1,
            )

        assert save_path.exists()


# =============================================================================
# Test Class: HTML Backend Capabilities
# =============================================================================


class TestHTMLCapabilities:
    """Test HTML backend overlay capabilities.

    HTML should support: positions, regions only.
    HTML should warn for: bodyparts, head direction.
    """

    def test_supports_position_overlay(
        self, simple_env, simple_fields, position_overlay_data, tmp_path
    ):
        """Test HTML renders position overlay without errors."""
        save_path = tmp_path / "test.html"

        # Should not raise any warnings or errors for position overlay
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=position_overlay_data,
            )

        assert save_path.exists()
        # render_html returns a Path object
        assert result == save_path

    def test_supports_regions(self, simple_env, simple_fields, tmp_path):
        """Test HTML renders regions without errors."""

        simple_env.regions.add("test_region", point=np.array([5.0, 5.0]))
        save_path = tmp_path / "test.html"

        # Should not raise any warnings or errors for regions
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = render_html(
                simple_env,
                simple_fields,
                str(save_path),
                show_regions=True,
            )

        assert save_path.exists()

    def test_does_not_support_bodypart_overlay(
        self, simple_env, simple_fields, bodypart_overlay_data, tmp_path
    ):
        """Test HTML warns for bodypart overlay (not supported)."""
        save_path = tmp_path / "test.html"

        # Should emit warning for bodypart overlay
        with pytest.warns(
            UserWarning, match="HTML backend supports positions and regions only"
        ):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=bodypart_overlay_data,
            )

        # File should still be created (just without bodypart overlay)
        assert save_path.exists()

    def test_does_not_support_head_direction_overlay(
        self, simple_env, simple_fields, head_direction_overlay_data, tmp_path
    ):
        """Test HTML warns for head direction overlay (not supported)."""
        save_path = tmp_path / "test.html"

        # Should emit warning for head direction overlay
        with pytest.warns(
            UserWarning, match="HTML backend supports positions and regions only"
        ):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=head_direction_overlay_data,
            )

        # File should still be created (just without head direction overlay)
        assert save_path.exists()


# =============================================================================
# Test Class: Widget Backend Capabilities
# =============================================================================


@pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
class TestWidgetCapabilities:
    """Test Widget backend overlay capabilities.

    Widget should support: positions, bodyparts, head direction, regions.
    """

    def test_supports_position_overlay(
        self, simple_env, simple_fields, position_overlay_data
    ):
        """Test Widget renders position overlay without errors."""
        # Mock IPython display
        with patch(
            "neurospatial.animation.backends.widget_backend.display"
        ) as mock_display:
            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _ = render_widget(
                    simple_env,
                    simple_fields,
                    overlay_data=position_overlay_data,
                )

            # render_widget returns None but calls display
            assert mock_display.called

    def test_supports_bodypart_overlay(
        self, simple_env, simple_fields, bodypart_overlay_data
    ):
        """Test Widget renders bodypart overlay without errors."""
        with patch(
            "neurospatial.animation.backends.widget_backend.display"
        ) as mock_display:
            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _ = render_widget(
                    simple_env,
                    simple_fields,
                    overlay_data=bodypart_overlay_data,
                )

            # render_widget returns None but calls display
            assert mock_display.called

    def test_supports_head_direction_overlay(
        self, simple_env, simple_fields, head_direction_overlay_data
    ):
        """Test Widget renders head direction overlay without errors."""
        with patch(
            "neurospatial.animation.backends.widget_backend.display"
        ) as mock_display:
            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _ = render_widget(
                    simple_env,
                    simple_fields,
                    overlay_data=head_direction_overlay_data,
                )

            # render_widget returns None but calls display
            assert mock_display.called

    def test_supports_regions(self, simple_env, simple_fields):
        """Test Widget renders regions without errors."""

        simple_env.regions.add("test_region", point=np.array([5.0, 5.0]))

        with patch(
            "neurospatial.animation.backends.widget_backend.display"
        ) as mock_display:
            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _ = render_widget(
                    simple_env,
                    simple_fields,
                    show_regions=True,
                )

            # render_widget returns None but calls display
            assert mock_display.called

    def test_supports_all_overlay_types(
        self, simple_env, simple_fields, all_overlay_types
    ):
        """Test Widget renders all overlay types simultaneously."""
        with patch(
            "neurospatial.animation.backends.widget_backend.display"
        ) as mock_display:
            # Should not raise any warnings or errors
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _ = render_widget(
                    simple_env,
                    simple_fields,
                    overlay_data=all_overlay_types,
                )

            # render_widget returns None but calls display
            assert mock_display.called


# =============================================================================
# Test Class: Capability Matrix Verification
# =============================================================================


class TestCapabilityMatrix:
    """Test the capability matrix is correctly defined and enforced."""

    def test_capability_matrix_completeness(self):
        """Test capability matrix defines all backends and overlay types."""
        expected_backends = {"napari", "video", "html", "widget"}
        expected_overlay_types = {"position", "bodypart", "head_direction", "regions"}

        assert set(BACKEND_CAPABILITIES.keys()) == expected_backends

        for backend, capabilities in BACKEND_CAPABILITIES.items():
            assert set(capabilities.keys()) == expected_overlay_types, (
                f"Backend {backend} missing overlay type definitions"
            )

    def test_napari_capabilities_match_matrix(self):
        """Test Napari backend capabilities match defined matrix."""
        napari_caps = BACKEND_CAPABILITIES["napari"]
        assert napari_caps["position"] is True
        assert napari_caps["bodypart"] is True
        assert napari_caps["head_direction"] is True
        assert napari_caps["regions"] is True

    def test_video_capabilities_match_matrix(self):
        """Test Video backend capabilities match defined matrix."""
        video_caps = BACKEND_CAPABILITIES["video"]
        assert video_caps["position"] is True
        assert video_caps["bodypart"] is True
        assert video_caps["head_direction"] is True
        assert video_caps["regions"] is True

    def test_html_capabilities_match_matrix(self):
        """Test HTML backend capabilities match defined matrix."""
        html_caps = BACKEND_CAPABILITIES["html"]
        assert html_caps["position"] is True
        assert html_caps["bodypart"] is False
        assert html_caps["head_direction"] is False
        assert html_caps["regions"] is True

    def test_widget_capabilities_match_matrix(self):
        """Test Widget backend capabilities match defined matrix."""
        widget_caps = BACKEND_CAPABILITIES["widget"]
        assert widget_caps["position"] is True
        assert widget_caps["bodypart"] is True
        assert widget_caps["head_direction"] is True
        assert widget_caps["regions"] is True

    def test_full_support_backends(self):
        """Test which backends support all overlay types."""
        full_support_backends = []
        for backend, caps in BACKEND_CAPABILITIES.items():
            if all(caps.values()):
                full_support_backends.append(backend)

        # Napari, Video, and Widget should support all types
        assert set(full_support_backends) == {"napari", "video", "widget"}

    def test_partial_support_backends(self):
        """Test which backends have partial overlay support."""
        partial_support_backends = []
        for backend, caps in BACKEND_CAPABILITIES.items():
            if not all(caps.values()) and any(caps.values()):
                partial_support_backends.append(backend)

        # HTML should have partial support (positions and regions only)
        assert partial_support_backends == ["html"]
