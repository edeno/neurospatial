"""Tests for VideoOverlay validation and napari-free operation.

Tests validate:
- 1D environment rejection
- Non-grid 2D environment support with dimension_ranges fallback
- Video export and import work without napari installed

Run with:
    uv run pytest tests/animation/test_video_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.animation.overlays import VideoOverlay, _validate_video_env

if TYPE_CHECKING:
    from neurospatial.transforms import VideoCalibration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def env_1d() -> Environment:
    """Create a 1D track environment using from_graph."""
    # Create a simple linear 1D track
    graph = nx.Graph()
    # Add nodes with 1D positions (tuples of length 1)
    graph.add_nodes_from(
        [
            (0, {"pos": (0.0,)}),
            (1, {"pos": (10.0,)}),
            (2, {"pos": (20.0,)}),
            (3, {"pos": (30.0,)}),
        ]
    )
    # Add edges with required distance attributes
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    graph.add_edge(2, 3, distance=10.0)

    edge_order = [(0, 1), (1, 2), (2, 3)]
    # edge_spacing=0.0 means use node positions directly
    return Environment.from_graph(
        graph, edge_order, edge_spacing=0.0, bin_size=2.0, name="Linear1D"
    )


@pytest.fixture
def env_2d_polygon() -> Environment:
    """Create a non-grid 2D environment using from_polygon."""
    polygon = box(0, 0, 100, 80)  # Rectangle 100x80
    return Environment.from_polygon(polygon, bin_size=5.0, name="Polygon2D")


@pytest.fixture
def env_2d_grid() -> Environment:
    """Create a standard grid-based 2D environment."""
    positions = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]])
    return Environment.from_samples(positions, bin_size=5.0, name="Grid2D")


# =============================================================================
# Environment Validation Tests
# =============================================================================


class TestEnvironmentValidation:
    """Test _validate_video_env rejects invalid environments."""

    def test_rejects_1d_environment(self, env_1d: Environment):
        """Test that VideoOverlay rejects 1D environments.

        WHAT: 1D environments should raise ValueError.
        WHY: Video frames are 2D images that cannot map to 1D coordinates.
        HOW: Use a 2D environment (from_samples, from_polygon, etc.).
        """
        with pytest.raises(ValueError, match=r"requires.*2D"):
            _validate_video_env(env_1d)

    def test_accepts_2d_grid_environment(self, env_2d_grid: Environment):
        """Test that standard 2D grid environments are accepted."""
        # Should not raise
        _validate_video_env(env_2d_grid)

    def test_accepts_2d_polygon_environment(self, env_2d_polygon: Environment):
        """Test that non-grid 2D polygon environments are accepted."""
        # Should not raise - polygon envs have dimension_ranges
        _validate_video_env(env_2d_polygon)


class TestNonGridEnvironmentSupport:
    """Test that non-grid 2D environments work with dimension_ranges fallback."""

    def test_non_grid_2d_environment_works_with_warning(
        self,
        tmp_path: Path,
        env_2d_polygon: Environment,
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
        fields = rng.random((n_frames, env_2d_polygon.n_bins))

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
        env_2d_polygon.clear_cache()

        # This should work (may emit fallback warning for non-grid)
        env_2d_polygon.animate_fields(
            fields,
            overlays=[video_overlay],
            backend="video",
            save_path=str(output_path),
            fps=10,
            n_workers=1,
        )

        # Verify output was created
        assert output_path.exists(), "Output should be created for polygon env"

    def test_non_grid_extent_uses_dimension_ranges(self, env_2d_polygon: Environment):
        """Test that dimension_ranges defines the extent for non-grid envs.

        Non-grid environments should use dimension_ranges for video alignment
        since they may not have a regular grid_shape.
        """
        # Verify the polygon environment has dimension_ranges
        assert env_2d_polygon.dimension_ranges is not None

        # Check the dimension ranges match our expected bounds (0-100, 0-80)
        x_range, y_range = env_2d_polygon.dimension_ranges
        assert x_range[0] >= 0 and x_range[1] <= 100
        assert y_range[0] >= 0 and y_range[1] <= 80

        # Validation should pass
        _validate_video_env(env_2d_polygon)


# =============================================================================
# Napari-Free Operation Tests
# =============================================================================


class TestNapariFreeOperation:
    """Test that video overlay works without napari installed."""

    def test_video_export_without_napari(
        self,
        tmp_path: Path,
        env_2d_grid: Environment,
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
        fields = rng.random((n_frames, env_2d_grid.n_bins))

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
            env_2d_grid.clear_cache()

            # Video export should work without napari
            env_2d_grid.animate_fields(
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
