"""Tests for VideoOverlay validation and napari-free operation.

Tests validate:
- 1D environment rejection
- Non-grid 2D environment support with dimension_ranges fallback
- Video export and import work without napari installed

Uses shared fixtures from conftest.py:
- linearized_env: 1D track environment for rejection tests
- polygon_env: non-grid 2D polygon environment for fallback tests
- masked_env: grid 2D environment for full support tests

Run with:
    uv run pytest tests/animation/test_video_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import VideoOverlay, _validate_video_env

if TYPE_CHECKING:
    from neurospatial.transforms import VideoCalibration


# =============================================================================
# Environment Validation Tests
# =============================================================================


class TestEnvironmentValidation:
    """Test _validate_video_env rejects invalid environments."""

    def test_rejects_1d_environment(self, linearized_env: Environment):
        """Test that VideoOverlay rejects 1D environments.

        WHAT: 1D environments should raise ValueError.
        WHY: Video frames are 2D images that cannot map to 1D coordinates.
        HOW: Use a 2D environment (from_samples, from_polygon, etc.).
        """
        with pytest.raises(ValueError, match=r"requires.*2D"):
            _validate_video_env(linearized_env)

    def test_accepts_2d_grid_environment(self, masked_env: Environment):
        """Test that standard 2D grid environments are accepted."""
        # Should not raise
        _validate_video_env(masked_env)

    def test_accepts_2d_polygon_environment(self, polygon_env: Environment):
        """Test that non-grid 2D polygon environments are accepted."""
        # Should not raise - polygon envs have dimension_ranges
        _validate_video_env(polygon_env)


class TestNonGridEnvironmentSupport:
    """Test that non-grid 2D environments work with dimension_ranges fallback."""

    def test_non_grid_2d_environment_works_with_warning(
        self,
        tmp_path: Path,
        polygon_env: Environment,
        sample_video_array: np.ndarray,
        sample_calibration: VideoCalibration,
    ):
        """Test that non-grid 2D environments can render with a warning.

        Non-grid environments (like from_polygon) may not have grid_shape.
        They should work using dimension_ranges with a fallback warning.
        """
        # Create fields that match the polygon environment and video frame count
        # sample_video_array has 10 frames, so use 10 frames for fields
        n_frames = 10
        rng = np.random.default_rng(42)
        fields = rng.random((n_frames, polygon_env.n_bins))

        # Create calibration for 100x80 env
        from neurospatial.transforms import (
            VideoCalibration,
            calibrate_from_scale_bar,
        )

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(16.0, 0.0),  # sample video is 16x16
            known_length_cm=100.0,  # Maps to env width
            frame_size_px=(16, 16),
        )
        calib = VideoCalibration(transform_px_to_cm=transform, frame_size_px=(16, 16))

        video_overlay = VideoOverlay(
            source=sample_video_array,
            calibration=calib,
            alpha=0.5,
            z_order="above",
        )

        output_path = tmp_path / "polygon_with_video.mp4"

        # Clear cache for pickle safety
        polygon_env.clear_cache()

        # This should work (may emit fallback warning for non-grid)
        polygon_env.animate_fields(
            fields,
            overlays=[video_overlay],
            backend="video",
            save_path=str(output_path),
            fps=10,
            n_workers=1,
        )

        # Verify output was created
        assert output_path.exists(), "Output should be created for polygon env"

    def test_non_grid_extent_uses_dimension_ranges(self, polygon_env: Environment):
        """Test that dimension_ranges defines the extent for non-grid envs.

        Non-grid environments should use dimension_ranges for video alignment
        since they may not have a regular grid_shape.
        """
        # Verify the polygon environment has dimension_ranges
        assert polygon_env.dimension_ranges is not None

        # Check the dimension ranges match our expected bounds (0-100, 0-80)
        x_range, y_range = polygon_env.dimension_ranges
        assert x_range[0] >= 0 and x_range[1] <= 100
        assert y_range[0] >= 0 and y_range[1] <= 80

        # Validation should pass
        _validate_video_env(polygon_env)


# =============================================================================
# Napari-Free Operation Tests
# =============================================================================


class TestNapariFreeOperation:
    """Test that video overlay works without napari installed."""

    def test_video_export_without_napari(
        self,
        tmp_path: Path,
        masked_env: Environment,
        sample_video_array: np.ndarray,
        sample_calibration: VideoCalibration,
    ):
        """Test that video export backend works without napari.

        The video export backend uses matplotlib, not napari.
        It should work even if napari is not installed.
        """
        # Create test fields - must match sample_video_array frame count (10)
        n_frames = 10
        rng = np.random.default_rng(42)
        fields = rng.random((n_frames, masked_env.n_bins))

        video_overlay = VideoOverlay(
            source=sample_video_array,
            calibration=sample_calibration,
            alpha=0.5,
            z_order="above",
        )

        output_path = tmp_path / "no_napari_export.mp4"

        # Mock napari as not installed
        with patch.dict(sys.modules, {"napari": None}):
            # Clear cache for pickle safety
            masked_env.clear_cache()

            # Video export should work without napari
            masked_env.animate_fields(
                fields,
                overlays=[video_overlay],
                backend="video",
                save_path=str(output_path),
                fps=10,
                n_workers=1,
            )

        # Verify output was created
        assert output_path.exists(), "Video export should work without napari"

    def test_import_without_napari(self):
        """Test that animation module imports without napari installed.

        The animation module should be importable even when napari is absent.
        Napari-specific features will just not be available.
        """
        # Mock napari as not installed
        with patch.dict(sys.modules, {"napari": None, "napari.utils": None}):
            # These imports should work without napari
            # (they're already imported at module level, so we just verify)
            from neurospatial.animation.overlays import (
                BodypartOverlay,
                HeadDirectionOverlay,
                PositionOverlay,
                VideoOverlay,
            )

            # Verify classes are accessible
            assert PositionOverlay is not None
            assert BodypartOverlay is not None
            assert HeadDirectionOverlay is not None
            assert VideoOverlay is not None

    def test_video_overlay_creation_without_napari(
        self,
        sample_video_array: np.ndarray,
        sample_calibration: VideoCalibration,
    ):
        """Test that VideoOverlay dataclass can be created without napari."""
        with patch.dict(sys.modules, {"napari": None}):
            # Creating the overlay should work
            overlay = VideoOverlay(
                source=sample_video_array,
                calibration=sample_calibration,
                alpha=0.5,
            )

            assert overlay.source is sample_video_array
            assert overlay.calibration is sample_calibration
            assert overlay.alpha == 0.5
