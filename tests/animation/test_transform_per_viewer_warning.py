"""Tests for per-viewer transform fallback warning behavior.

These tests verify that transform fallback warnings are tracked per-viewer,
not globally per-session. This ensures:
1. Each viewer receives the warning once (if fallback is needed)
2. Multiple viewers can each receive their own warning
3. Repeated calls within the same viewer don't duplicate warnings
"""

from __future__ import annotations

import warnings as warnings_module
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


class TestTransformSuppressWarningParameter:
    """Tests for the suppress_warning parameter in transform functions."""

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self) -> None:
        """Reset the fallback warning flag before each test."""
        from neurospatial.animation.transforms import reset_transform_warning

        reset_transform_warning()
        yield
        reset_transform_warning()

    def test_transform_coords_suppress_warning_parameter_exists(self) -> None:
        """Test that transform_coords_for_napari accepts suppress_warning parameter."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        # Should not raise TypeError about unexpected keyword argument
        result = transform_coords_for_napari(coords, None, suppress_warning=True)
        assert result.shape == coords.shape

    def test_transform_coords_suppresses_warning_when_true(self) -> None:
        """Test that suppress_warning=True prevents warning emission."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        # With suppress_warning=True, no warning should be emitted
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            transform_coords_for_napari(coords, None, suppress_warning=True)

        # Filter for UserWarnings about fallback
        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    def test_transform_coords_warns_when_not_suppressed(self) -> None:
        """Test that warning is emitted when suppress_warning=False (default)."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        # With suppress_warning=False (default), warning should be emitted
        with pytest.warns(UserWarning, match="falling back"):
            transform_coords_for_napari(coords, None, suppress_warning=False)

    def test_transform_direction_suppress_warning_parameter_exists(self) -> None:
        """Test that transform_direction_for_napari accepts suppress_warning parameter."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        direction = np.array([[1.0, 0.0]])

        # Should not raise TypeError about unexpected keyword argument
        result = transform_direction_for_napari(direction, None, suppress_warning=True)
        assert result.shape == direction.shape

    def test_transform_direction_suppresses_warning_when_true(self) -> None:
        """Test that suppress_warning=True prevents warning emission for direction."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        direction = np.array([[1.0, 0.0]])

        # With suppress_warning=True, no warning should be emitted
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            transform_direction_for_napari(direction, None, suppress_warning=True)

        # Filter for UserWarnings about fallback
        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    def test_transform_direction_warns_when_not_suppressed(self) -> None:
        """Test that warning is emitted when suppress_warning=False (default)."""
        from neurospatial.animation.transforms import (
            reset_transform_warning,
            transform_direction_for_napari,
        )

        # Reset to ensure fresh warning
        reset_transform_warning()
        direction = np.array([[1.0, 0.0]])

        # With suppress_warning=False (default), warning should be emitted
        with pytest.warns(UserWarning, match="falling back"):
            transform_direction_for_napari(direction, None, suppress_warning=False)


class TestPerViewerWarningTracking:
    """Tests for per-viewer warning state management in napari backend."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer with metadata dict."""
        viewer = MagicMock()
        viewer.metadata = {}
        return viewer

    @pytest.fixture
    def another_mock_viewer(self) -> MagicMock:
        """Create another mock napari viewer with metadata dict."""
        viewer = MagicMock()
        viewer.metadata = {}
        return viewer

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self) -> None:
        """Reset the fallback warning flag before each test."""
        from neurospatial.animation.transforms import reset_transform_warning

        reset_transform_warning()
        yield
        reset_transform_warning()

    def test_check_viewer_warned_returns_false_initially(
        self, mock_viewer: MagicMock
    ) -> None:
        """Test that _check_viewer_warned returns False for new viewer."""
        from neurospatial.animation.backends.napari_backend import _check_viewer_warned

        assert _check_viewer_warned(mock_viewer) is False

    def test_mark_viewer_warned_sets_metadata(self, mock_viewer: MagicMock) -> None:
        """Test that _mark_viewer_warned sets the metadata flag."""
        from neurospatial.animation.backends.napari_backend import _mark_viewer_warned

        _mark_viewer_warned(mock_viewer)

        assert mock_viewer.metadata.get("_transform_fallback_warned") is True

    def test_check_viewer_warned_returns_true_after_marked(
        self, mock_viewer: MagicMock
    ) -> None:
        """Test that _check_viewer_warned returns True after marking."""
        from neurospatial.animation.backends.napari_backend import (
            _check_viewer_warned,
            _mark_viewer_warned,
        )

        _mark_viewer_warned(mock_viewer)

        assert _check_viewer_warned(mock_viewer) is True

    def test_separate_viewers_have_separate_warning_state(
        self, mock_viewer: MagicMock, another_mock_viewer: MagicMock
    ) -> None:
        """Test that different viewers track warnings independently."""
        from neurospatial.animation.backends.napari_backend import (
            _check_viewer_warned,
            _mark_viewer_warned,
        )

        # Mark first viewer as warned
        _mark_viewer_warned(mock_viewer)

        # First viewer should be marked, second should not
        assert _check_viewer_warned(mock_viewer) is True
        assert _check_viewer_warned(another_mock_viewer) is False


class TestPerViewerWarningIntegration:
    """Integration tests for per-viewer warning behavior in napari backend."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer with metadata dict."""
        viewer = MagicMock()
        viewer.metadata = {}
        return viewer

    @pytest.fixture
    def another_mock_viewer(self) -> MagicMock:
        """Create another mock napari viewer."""
        viewer = MagicMock()
        viewer.metadata = {}
        return viewer

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self) -> None:
        """Reset the fallback warning flag before each test."""
        from neurospatial.animation.transforms import reset_transform_warning

        reset_transform_warning()
        yield
        reset_transform_warning()

    def test_transform_with_viewer_tracking_warns_once(
        self, mock_viewer: MagicMock
    ) -> None:
        """Test that _transform_with_viewer_tracking warns only once per viewer."""
        from neurospatial.animation.backends.napari_backend import (
            _transform_coords_with_viewer,
        )

        coords = np.array([[5.0, 5.0]])

        # First call should warn (using None as env triggers fallback)
        with pytest.warns(UserWarning, match="falling back"):
            _transform_coords_with_viewer(coords, None, mock_viewer)

        # Second call should NOT warn (same viewer, already warned)
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            _transform_coords_with_viewer(coords, None, mock_viewer)

        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    def test_different_viewers_each_warn_once(
        self,
        mock_viewer: MagicMock,
        another_mock_viewer: MagicMock,
    ) -> None:
        """Test that different viewers each receive their own warning."""
        from neurospatial.animation.backends.napari_backend import (
            _transform_coords_with_viewer,
        )
        from neurospatial.animation.transforms import reset_transform_warning

        coords = np.array([[5.0, 5.0]])

        # First viewer should warn
        with pytest.warns(UserWarning, match="falling back"):
            _transform_coords_with_viewer(coords, None, mock_viewer)

        # Reset module-level flag to allow second viewer to also warn
        # (this tests per-viewer tracking independent of module-level flag)
        reset_transform_warning()

        # Second viewer should ALSO warn (different viewer)
        with pytest.warns(UserWarning, match="falling back"):
            _transform_coords_with_viewer(coords, None, another_mock_viewer)

    def test_transform_direction_with_viewer_tracking(
        self, mock_viewer: MagicMock
    ) -> None:
        """Test per-viewer tracking for direction transforms."""
        from neurospatial.animation.backends.napari_backend import (
            _transform_direction_with_viewer,
        )

        direction = np.array([[1.0, 0.0]])

        # First call should warn (using None as env triggers fallback)
        with pytest.warns(UserWarning, match="falling back"):
            _transform_direction_with_viewer(direction, None, mock_viewer)

        # Second call should NOT warn
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            _transform_direction_with_viewer(direction, None, mock_viewer)

        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0


class TestModuleLevelFallbackBehavior:
    """Tests for module-level fallback warning behavior (for non-napari usage)."""

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self) -> None:
        """Reset the fallback warning flag before each test."""
        from neurospatial.animation.transforms import reset_transform_warning

        reset_transform_warning()
        yield
        reset_transform_warning()

    def test_module_level_still_warns_once_per_session(self) -> None:
        """Test that without viewer tracking, module-level still warns once per session."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        # First call warns
        with pytest.warns(UserWarning, match="falling back"):
            transform_coords_for_napari(coords, None)

        # Second call does NOT warn (module-level flag set)
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            transform_coords_for_napari(coords, None)

        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    def test_no_warning_when_env_is_valid(self, simple_env: Environment) -> None:
        """Test that no warning is emitted when env has required attributes."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        # Valid env should not trigger warning at all
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            transform_coords_for_napari(coords, simple_env)

        fallback_warnings = [
            w for w in warning_list if "falling back" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    @pytest.fixture
    def simple_env(self) -> Environment:
        """Create a simple environment for testing."""
        from neurospatial import Environment

        positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        return Environment.from_samples(positions, bin_size=1.0)
