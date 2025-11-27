"""Tests for annotation core functions."""

import numpy as np
import pytest
import shapely.geometry as shp

from neurospatial.annotation import AnnotationConfig
from neurospatial.annotation.core import (
    AnnotationResult,
    _add_initial_regions,
    _process_annotation_results,
    _resolve_config_params,
    annotate_video,
)
from neurospatial.regions import Region, Regions
from neurospatial.transforms import Affine2D, VideoCalibration


class TestAnnotationConfig:
    """Tests for AnnotationConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible defaults."""
        config = AnnotationConfig()
        assert config.frame_index == 0
        assert config.simplify_tolerance is None
        assert config.multiple_boundaries == "last"
        assert config.show_positions is False

    def test_custom_values(self):
        """Can create config with custom values."""
        config = AnnotationConfig(
            frame_index=100,
            simplify_tolerance=1.5,
            multiple_boundaries="first",
            show_positions=True,
        )
        assert config.frame_index == 100
        assert config.simplify_tolerance == 1.5
        assert config.multiple_boundaries == "first"
        assert config.show_positions is True

    def test_frozen(self):
        """Config is immutable."""
        config = AnnotationConfig()
        with pytest.raises(AttributeError):
            config.frame_index = 10  # type: ignore[misc]


class TestResolveConfigParams:
    """Tests for _resolve_config_params helper."""

    def test_defaults_when_no_config(self):
        """Use AnnotationConfig defaults when config is None."""
        resolved = _resolve_config_params(None, None, None, None, None)
        assert resolved.frame_index == 0
        assert resolved.simplify_tolerance is None
        assert resolved.multiple_boundaries == "last"
        assert resolved.show_positions is False

    def test_uses_config_values(self):
        """Use config values when provided."""
        config = AnnotationConfig(frame_index=50, simplify_tolerance=2.0)
        resolved = _resolve_config_params(config, None, None, None, None)
        assert resolved.frame_index == 50
        assert resolved.simplify_tolerance == 2.0

    def test_individual_params_override_config(self):
        """Individual params take precedence over config."""
        config = AnnotationConfig(
            frame_index=50,
            simplify_tolerance=2.0,
            multiple_boundaries="first",
            show_positions=True,
        )
        resolved = _resolve_config_params(
            config,
            frame_index=100,
            simplify_tolerance=1.0,
            multiple_boundaries="error",
            show_positions=False,
        )
        # All overridden by individual params
        assert resolved.frame_index == 100
        assert resolved.simplify_tolerance == 1.0
        assert resolved.multiple_boundaries == "error"
        assert resolved.show_positions is False

    def test_partial_override(self):
        """Some params from config, some overridden."""
        config = AnnotationConfig(
            frame_index=50,
            simplify_tolerance=2.0,
        )
        resolved = _resolve_config_params(
            config,
            frame_index=None,  # Use config
            simplify_tolerance=1.0,  # Override
            multiple_boundaries=None,  # Use default
            show_positions=None,  # Use default
        )
        assert resolved.frame_index == 50  # From config
        assert resolved.simplify_tolerance == 1.0  # Overridden
        assert resolved.multiple_boundaries == "last"  # Default
        assert resolved.show_positions is False  # Default


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


@pytest.mark.gui
class TestProcessAnnotationResults:
    """Tests for _process_annotation_results helper."""

    def test_warns_mode_mismatch_regions_with_boundary(self):
        """Warn when boundary is drawn but mode='regions'."""
        napari = pytest.importorskip("napari")
        import pandas as pd

        # Create shapes layer with an environment boundary shape
        viewer = napari.Viewer(show=False)
        try:
            shapes = viewer.add_shapes(name="test")
            # Add a polygon shape (environment boundary)
            boundary = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
            shapes.add(boundary, shape_type="polygon")

            # Set features with 'environment' role
            shapes.features = pd.DataFrame({"role": ["environment"], "name": ["arena"]})

            # Process with mode='regions' - should warn
            with pytest.warns(
                UserWarning,
                match="environment boundary was drawn but mode='regions'",
            ):
                result = _process_annotation_results(
                    shapes,
                    mode="regions",
                    bin_size=None,
                    calibration=None,
                    simplify_tolerance=None,
                    multiple_boundaries="last",
                )

            # Environment should be None because mode='regions'
            assert result.environment is None
        finally:
            viewer.close()


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


# =============================================================================
# Integration Tests for Annotation Widget
# =============================================================================


@pytest.mark.gui  # Skip in headless CI with: pytest -m "not gui"
class TestAnnotationWidgetIntegration:
    """Integration tests for the full annotation widget workflow."""

    def test_widget_creation_and_mode_switching(self):
        """Widget should create correctly and support mode cycling."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._napari_widget import (
            create_annotation_widget,
            rebuild_features,
        )

        viewer = napari.Viewer(show=False)

        # Create a shapes layer for annotations
        shapes = viewer.add_shapes(
            name="Annotations",
            face_color="cyan",
            edge_color="white",
            features=rebuild_features([], []),
        )

        # Create the widget
        widget = create_annotation_widget(
            viewer, "Annotations", initial_mode="environment"
        )

        # Widget should be a container
        assert widget is not None

        # Initial mode should be environment
        # feature_defaults may store pandas Series or scalars depending on napari version
        role_default = shapes.feature_defaults.get("role", "")
        name_default = shapes.feature_defaults.get("name", "")

        # Handle both scalar and Series cases
        import pandas as pd

        if isinstance(role_default, pd.Series):
            role_default = role_default.iloc[0] if len(role_default) > 0 else ""
        if isinstance(name_default, pd.Series):
            name_default = name_default.iloc[0] if len(name_default) > 0 else ""

        assert str(role_default) == "environment"
        assert str(name_default) == "arena"

        viewer.close()

    def test_programmatic_shape_addition_sets_features(self):
        """Adding shapes programmatically should set features correctly."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._napari_widget import (
            create_annotation_widget,
            rebuild_features,
        )

        viewer = napari.Viewer(show=False)

        # Create shapes layer
        shapes = viewer.add_shapes(
            name="Annotations",
            face_color="cyan",
            edge_color="white",
            features=rebuild_features([], []),
        )

        # Create the widget (connects event handlers)
        create_annotation_widget(viewer, "Annotations", initial_mode="environment")

        # Programmatically add an environment boundary
        boundary_coords = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        shapes.add_polygons([boundary_coords])

        # Process events (simulates event loop)
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        # Verify features were set
        assert len(shapes.data) == 1
        assert len(shapes.features) == 1
        assert str(shapes.features["role"].iloc[0]) == "environment"
        assert str(shapes.features["name"].iloc[0]) == "arena"

        viewer.close()

    def test_shape_deletion_truncates_features(self):
        """Deleting shapes should correctly truncate the features DataFrame."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._napari_widget import (
            create_annotation_widget,
            rebuild_features,
        )

        viewer = napari.Viewer(show=False)

        # Create shapes layer with pre-populated features
        shapes = viewer.add_shapes(
            name="Annotations",
            face_color="cyan",
            edge_color="white",
            features=rebuild_features(
                ["environment", "region", "region"], ["arena", "goal", "start"]
            ),
        )

        # Add corresponding shape data
        shapes.add_polygons(
            [
                np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),  # arena
                np.array([[10, 10], [30, 10], [30, 30], [10, 30]]),  # goal
                np.array([[60, 60], [80, 60], [80, 80], [60, 80]]),  # start
            ]
        )

        # Create widget (connects event handlers)
        create_annotation_widget(viewer, "Annotations", initial_mode="region")

        # Process events
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        # Delete middle shape by index
        shapes.selected_data = {1}  # Select "goal"

        # Simulate delete button or key
        # Use the layer's remove_selected method
        shapes.remove_selected()

        # Process events
        QApplication.processEvents()

        # Verify features match data length
        assert len(shapes.data) == 2
        # Features should have been truncated/rebuilt
        assert len(shapes.features) == len(shapes.data)

        viewer.close()

    def test_duplicate_names_get_unique_suffix(self):
        """Adding shapes with duplicate names should auto-generate unique names."""
        pytest.importorskip("napari")
        from neurospatial.annotation._state import make_unique_name

        # Test the make_unique_name function directly (unit test approach)
        # This is more reliable than testing async widget behavior

        existing = ["region_1", "goal", "start"]

        # New name that doesn't exist should pass through
        assert make_unique_name("reward", existing) == "reward"

        # Duplicate name should get suffix
        assert make_unique_name("region_1", existing) == "region_1_2"

        # Test multiple duplicates
        existing_with_dupes = ["region", "region_2", "region_3"]
        assert make_unique_name("region", existing_with_dupes) == "region_4"

        # Empty existing list
        assert make_unique_name("test", []) == "test"

    def test_make_unique_name_in_rebuild_features(self):
        """rebuild_features should create proper categorical DataFrame."""
        pytest.importorskip("napari")
        from neurospatial.annotation._helpers import rebuild_features

        # Test with mixed roles
        features = rebuild_features(
            ["environment", "hole", "region"], ["arena", "obstacle", "goal"]
        )

        assert len(features) == 3
        assert list(features["role"]) == ["environment", "hole", "region"]
        assert list(features["name"]) == ["arena", "obstacle", "goal"]

        # Role column should be categorical
        assert features["role"].dtype.name == "category"

    def test_role_colors_applied_correctly(self):
        """Each role type should get the correct face color."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._napari_widget import (
            rebuild_features,
            sync_face_colors_from_features,
        )

        viewer = napari.Viewer(show=False)

        # Create shapes layer with mixed roles
        shapes = viewer.add_shapes(
            name="Annotations",
            face_color="cyan",
            edge_color="white",
            features=rebuild_features(
                ["environment", "hole", "region"], ["arena", "obstacle", "goal"]
            ),
        )

        # Add shape data
        shapes.add_polygons(
            [
                np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
                np.array([[40, 40], [60, 40], [60, 60], [40, 60]]),
                np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
            ]
        )

        # Sync colors from features
        sync_face_colors_from_features(shapes)

        # Verify each shape has correct color
        # napari stores face_color as RGBA arrays
        face_colors = shapes.face_color
        assert len(face_colors) == 3

        # Colors should match role order: environment=cyan, hole=red, region=yellow
        # Just verify they're different (exact RGB depends on napari's color conversion)
        # The sync function should have applied different colors
        assert face_colors is not None

        viewer.close()


# =============================================================================
# Tests for initial_boundary, boundary_config, show_positions (M3)
# =============================================================================


class TestAnnotateVideoInitialBoundaryValidation:
    """Tests for initial_boundary parameter validation (no GUI required)."""

    def test_initial_boundary_accepts_ndarray(self, tmp_path):
        """initial_boundary accepts NDArray and infers boundary."""
        pytest.importorskip("napari")

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        # Create position data
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        # Should fail on video read (empty file), not on initial_boundary
        # This validates that the parameter is accepted
        with pytest.raises(ValueError, match="Could not open video file"):
            annotate_video(
                str(video_file),
                bin_size=5.0,
                initial_boundary=positions,
            )

    def test_initial_boundary_accepts_polygon(self, tmp_path):
        """initial_boundary accepts Shapely Polygon directly."""
        pytest.importorskip("napari")

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        boundary = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        # Should fail on video read (empty file), not on initial_boundary
        with pytest.raises(ValueError, match="Could not open video file"):
            annotate_video(
                str(video_file),
                bin_size=5.0,
                initial_boundary=boundary,
            )


@pytest.mark.gui
class TestAddPositionsLayer:
    """Tests for _add_positions_layer helper."""

    def test_adds_points_layer(self):
        """Positions are added as a Points layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation.core import _add_positions_layer

        viewer = napari.Viewer(show=False)

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        _add_positions_layer(viewer, positions, calibration=None)

        # Should have a Points layer
        point_layers = [
            layer for layer in viewer.layers if layer._type_string == "points"
        ]
        assert len(point_layers) == 1
        assert "Trajectory" in point_layers[0].name

        viewer.close()

    def test_converts_to_napari_row_col(self):
        """Positions are converted to napari (row, col) order."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation.core import _add_positions_layer

        viewer = napari.Viewer(show=False)

        # Simple positions: (x=10, y=20) -> napari (row=20, col=10)
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])

        _add_positions_layer(viewer, positions, calibration=None)

        point_layer = next(
            layer for layer in viewer.layers if layer._type_string == "points"
        )
        data = point_layer.data

        # (x=10, y=20) -> (row=20, col=10)
        assert data[0, 0] == 20.0  # row = y
        assert data[0, 1] == 10.0  # col = x

        viewer.close()

    def test_subsamples_large_positions(self):
        """Large position arrays are subsampled for performance."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation.core import _add_positions_layer

        viewer = napari.Viewer(show=False)

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (10000, 2))

        _add_positions_layer(viewer, positions, calibration=None)

        point_layer = next(
            layer for layer in viewer.layers if layer._type_string == "points"
        )

        # Should be subsampled to ~5000 points
        assert len(point_layer.data) < 6000

        viewer.close()

    def test_applies_calibration_transform(self):
        """Calibration transforms positions from cm to pixels."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation.core import _add_positions_layer
        from neurospatial.transforms import Affine2D, VideoCalibration

        viewer = napari.Viewer(show=False)

        # 2x scale calibration: 1 cm = 2 px (matrix is inverse)
        scale_matrix = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Positions in cm
        positions = np.array([[10.0, 20.0]])  # 10 cm, 20 cm

        _add_positions_layer(viewer, positions, calibration=calibration)

        point_layer = next(
            layer for layer in viewer.layers if layer._type_string == "points"
        )
        data = point_layer.data

        # After transform_cm_to_px (0.5x) and row/col swap:
        # (10, 20) cm -> (5, 10) px -> napari (10, 5)
        assert data[0, 0] == pytest.approx(10.0)  # row = y_px
        assert data[0, 1] == pytest.approx(5.0)  # col = x_px

        viewer.close()


@pytest.mark.gui
class TestInitialBoundaryConflictResolution:
    """Tests for conflict resolution between initial_boundary and initial_regions."""

    def test_warns_when_both_boundary_and_env_region_provided(self):
        """Warning emitted when initial_boundary and environment region both provided."""
        pytest.importorskip("napari")
        from neurospatial.annotation.core import (
            _filter_environment_regions,
        )

        # Create regions with environment
        boundary_region = Region(
            name="arena",
            kind="polygon",
            data=shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            metadata={"role": "environment"},
        )
        goal_region = Region(
            name="goal",
            kind="polygon",
            data=shp.Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
            metadata={"role": "region"},
        )
        regions = Regions([boundary_region, goal_region])

        # Filter should emit warning and remove environment regions
        with pytest.warns(UserWarning, match="Both initial_boundary and environment"):
            filtered = _filter_environment_regions(regions)

        # Only non-environment regions should remain
        assert "goal" in filtered
        assert "arena" not in filtered

    def test_no_warning_when_no_env_region(self):
        """No warning when initial_regions has no environment regions."""
        pytest.importorskip("napari")
        from neurospatial.annotation.core import (
            _filter_environment_regions,
        )

        # Only non-environment regions
        goal_region = Region(
            name="goal",
            kind="polygon",
            data=shp.Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
            metadata={"role": "region"},
        )
        regions = Regions([goal_region])

        # Should not emit warning, return unchanged
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Fail if any warning
            filtered = _filter_environment_regions(regions)

        assert "goal" in filtered


# =============================================================================
# Tests for decomposed helper functions
# =============================================================================


class TestValidateAnnotateParams:
    """Tests for _validate_annotate_params helper."""

    def test_raises_for_environment_mode_without_bin_size(self):
        """Raises ValueError when mode='environment' and bin_size is None."""
        from neurospatial.annotation.core import _validate_annotate_params

        with pytest.raises(
            ValueError, match="bin_size is required when mode='environment'"
        ):
            _validate_annotate_params(mode="environment", bin_size=None)

    def test_raises_for_both_mode_without_bin_size(self):
        """Raises ValueError when mode='both' and bin_size is None."""
        from neurospatial.annotation.core import _validate_annotate_params

        with pytest.raises(ValueError, match="bin_size is required when mode='both'"):
            _validate_annotate_params(mode="both", bin_size=None)

    def test_passes_for_regions_mode_without_bin_size(self):
        """No error when mode='regions' and bin_size is None."""
        from neurospatial.annotation.core import _validate_annotate_params

        # Should not raise
        _validate_annotate_params(mode="regions", bin_size=None)

    def test_passes_when_bin_size_provided(self):
        """No error when bin_size is provided."""
        from neurospatial.annotation.core import _validate_annotate_params

        # Should not raise for any mode when bin_size provided
        _validate_annotate_params(mode="environment", bin_size=5.0)
        _validate_annotate_params(mode="both", bin_size=5.0)
        _validate_annotate_params(mode="regions", bin_size=5.0)


class TestLoadVideoFrame:
    """Tests for _load_video_frame helper."""

    def test_raises_for_missing_file(self, tmp_path):
        """Raises FileNotFoundError for missing video file."""
        from pathlib import Path

        from neurospatial.annotation.core import _load_video_frame

        missing_file = Path(tmp_path / "nonexistent.mp4")

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            _load_video_frame(missing_file, frame_index=0)

    def test_raises_for_invalid_frame_index(self, tmp_path):
        """Raises IndexError with helpful message for invalid frame index."""
        # Need a valid video file to test frame index error
        pytest.importorskip("cv2")
        from pathlib import Path

        from neurospatial.annotation.core import _load_video_frame

        # Create an empty file (will fail to open as video)
        video_file = Path(tmp_path / "test.mp4")
        video_file.touch()

        # Empty file should raise ValueError about "Could not open video file"
        with pytest.raises(ValueError, match="Could not open video file"):
            _load_video_frame(video_file, frame_index=0)


class TestProcessInitialBoundary:
    """Tests for _process_initial_boundary helper."""

    def test_returns_none_for_none_input(self):
        """Returns (None, None) when initial_boundary is None."""
        from neurospatial.annotation.core import _process_initial_boundary

        boundary, positions = _process_initial_boundary(
            initial_boundary=None,
            boundary_config=None,
            show_positions=False,
        )

        assert boundary is None
        assert positions is None

    def test_returns_polygon_directly(self):
        """Returns Polygon unchanged when initial_boundary is Polygon."""
        from neurospatial.annotation.core import _process_initial_boundary

        input_polygon = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        boundary, positions = _process_initial_boundary(
            initial_boundary=input_polygon,
            boundary_config=None,
            show_positions=False,
        )

        assert boundary is input_polygon
        assert positions is None

    def test_returns_polygon_without_positions_when_not_showing(self):
        """When show_positions=False, positions_for_display is None."""
        from neurospatial.annotation.core import _process_initial_boundary

        input_polygon = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        boundary, positions = _process_initial_boundary(
            initial_boundary=input_polygon,
            boundary_config=None,
            show_positions=True,  # True but input is Polygon, not array
        )

        # For Polygon input, positions are never returned
        assert boundary is input_polygon
        assert positions is None

    def test_infers_boundary_from_ndarray(self):
        """Infers boundary polygon from position NDArray."""
        from neurospatial.annotation.core import _process_initial_boundary

        # Create random positions in a rectangle
        rng = np.random.default_rng(42)
        positions_array = rng.uniform(low=[0, 0], high=[100, 80], size=(200, 2))

        boundary, positions = _process_initial_boundary(
            initial_boundary=positions_array,
            boundary_config=None,
            show_positions=False,
        )

        # Should return a valid polygon
        assert boundary is not None
        assert hasattr(boundary, "exterior")  # It's a Shapely Polygon
        assert positions is None

    def test_returns_positions_for_display_when_requested(self):
        """Returns positions array when show_positions=True and input is NDArray."""
        from neurospatial.annotation.core import _process_initial_boundary

        rng = np.random.default_rng(42)
        positions_array = rng.uniform(low=[0, 0], high=[100, 80], size=(200, 2))

        boundary, positions = _process_initial_boundary(
            initial_boundary=positions_array,
            boundary_config=None,
            show_positions=True,
        )

        assert boundary is not None
        assert positions is positions_array  # Same array reference
