"""Tests for annotation core functions."""

import numpy as np
import pytest
import shapely.geometry as shp

from neurospatial.annotation.core import (
    AnnotationResult,
    _add_initial_regions,
    annotate_video,
)
from neurospatial.regions import Region, Regions
from neurospatial.transforms import Affine2D, VideoCalibration


class TestAnnotateVideoValidation:
    """Tests for annotate_video parameter validation (no GUI required)."""

    def test_bin_size_required_for_environment_mode(self, tmp_path):
        """bin_size is required when mode='environment'."""
        # Create a dummy video file (won't actually be read due to early validation)
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with pytest.raises(
            ValueError, match="bin_size is required when mode='environment'"
        ):
            annotate_video(str(video_file), mode="environment", bin_size=None)

    def test_bin_size_required_for_both_mode(self, tmp_path):
        """bin_size is required when mode='both'."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with pytest.raises(ValueError, match="bin_size is required when mode='both'"):
            annotate_video(str(video_file), mode="both", bin_size=None)

    def test_bin_size_not_required_for_regions_mode(self, tmp_path):
        """bin_size not required when mode='regions' (validation passes, video read fails)."""
        pytest.importorskip("napari")  # Skip if napari not installed

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        # With mode='regions', no bin_size validation error
        # But it will fail on video read (empty file) - ValueError from VideoReader
        with pytest.raises(ValueError, match="Could not open video file"):
            annotate_video(str(video_file), mode="regions", bin_size=None)

    def test_video_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing video file."""
        pytest.importorskip("napari")  # Skip if napari not installed

        missing_file = tmp_path / "nonexistent.mp4"

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            annotate_video(str(missing_file), bin_size=5.0)


class TestAnnotationResult:
    """Tests for AnnotationResult named tuple."""

    def test_creation(self):
        """Create AnnotationResult with environment and regions."""
        regions = Regions([])
        result = AnnotationResult(environment=None, regions=regions)

        assert result.environment is None
        assert isinstance(result.regions, Regions)

    def test_unpacking(self):
        """Unpack AnnotationResult as tuple."""
        regions = Regions([])
        result = AnnotationResult(environment=None, regions=regions)

        env, regs = result

        assert env is None
        assert regs is regions


@pytest.mark.gui  # Skip in headless CI with: pytest -m "not gui"
class TestAddInitialRegions:
    """Tests for _add_initial_regions helper."""

    def test_adds_polygon_regions(self):
        """Add existing polygon regions to shapes layer."""
        napari = pytest.importorskip("napari")

        # Create regions
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        region = Region(
            name="test",
            kind="polygon",
            data=poly,
            metadata={"role": "region"},
        )
        regions = Regions([region])

        # Create shapes layer (no initial features needed - _add_initial_regions sets them)
        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration=None)

        assert len(shapes.data) == 1
        # Check features DataFrame (not properties)
        assert shapes.features["name"].iloc[0] == "test"
        assert shapes.features["role"].iloc[0] == "region"
        viewer.close()

    def test_skips_point_regions(self):
        """Skip non-polygon regions."""
        napari = pytest.importorskip("napari")

        point_region = Region(
            name="point",
            kind="point",
            data=np.array([50.0, 50.0]),
        )
        regions = Regions([point_region])

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration=None)

        assert len(shapes.data) == 0
        viewer.close()

    def test_transforms_with_calibration(self):
        """Transform coordinates back to pixels when calibration provided."""
        napari = pytest.importorskip("napari")

        # 2x scale calibration (cm -> pixels means inverse)
        scale_matrix = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Region in "cm" (world coords after 2x scale)
        poly = shp.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        region = Region(name="scaled", kind="polygon", data=poly)
        regions = Regions([region])

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration)

        # Coordinates should be back in pixels (0.5x scale = inverse of 2x)
        coords = shapes.data[0]
        # After cm_to_px (0.5x) and row/col swap, (20, 20) cm -> (10, 10) px -> (10, 10) napari
        assert coords.max() == pytest.approx(10.0)
        viewer.close()
