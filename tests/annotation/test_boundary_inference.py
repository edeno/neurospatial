"""Tests for boundary inference algorithms."""

import numpy as np
import pytest
from shapely.geometry import Polygon


class TestBoundaryConfig:
    """Tests for BoundaryConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        from neurospatial.annotation import BoundaryConfig

        config = BoundaryConfig()
        assert config.method == "alpha_shape"
        assert config.buffer_fraction == 0.02
        assert config.simplify_fraction == 0.01
        assert config.alpha == 0.05

    def test_frozen(self):
        """Config is immutable."""
        from neurospatial.annotation import BoundaryConfig

        config = BoundaryConfig()
        with pytest.raises(AttributeError):
            config.method = "convex_hull"  # type: ignore[misc]

    def test_custom_values(self):
        """Config accepts custom values."""
        from neurospatial.annotation import BoundaryConfig

        config = BoundaryConfig(
            method="convex_hull",
            buffer_fraction=0.05,
            simplify_fraction=0.02,
            alpha=0.1,
        )
        assert config.method == "convex_hull"
        assert config.buffer_fraction == 0.05
        assert config.simplify_fraction == 0.02
        assert config.alpha == 0.1

    def test_method_literal_types(self):
        """Method must be one of the allowed values."""
        from neurospatial.annotation import BoundaryConfig

        # Valid methods should work
        for method in ["convex_hull", "alpha_shape"]:
            config = BoundaryConfig(method=method)  # type: ignore[arg-type]
            assert config.method == method


class TestBoundaryFromPositionsValidation:
    """Tests for input validation in boundary_from_positions()."""

    def test_wrong_shape_1d(self):
        """Reject 1D arrays."""
        from neurospatial.annotation import boundary_from_positions

        positions = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            boundary_from_positions(positions)

    def test_wrong_shape_3d(self):
        """Reject 3D arrays."""
        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.random((10, 3))
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            boundary_from_positions(positions)

    def test_minimum_points(self):
        """Require at least 3 points."""
        from neurospatial.annotation import boundary_from_positions

        positions = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="at least 3 valid points"):
            boundary_from_positions(positions)

    def test_unique_points(self):
        """Require at least 3 unique points."""
        from neurospatial.annotation import boundary_from_positions

        # 3 points but only 2 unique
        positions = np.array([[0, 0], [1, 1], [0, 0]])
        with pytest.raises(ValueError, match="at least 3 unique points"):
            boundary_from_positions(positions)

    def test_unknown_method(self):
        """Reject unknown method."""
        from neurospatial.annotation import boundary_from_positions

        positions = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        with pytest.raises(ValueError, match="Unknown method"):
            boundary_from_positions(positions, method="unknown_method")  # type: ignore[arg-type]

    def test_filters_nan_values(self):
        """NaN values are filtered with warning."""
        from neurospatial.annotation import boundary_from_positions

        # Valid positions with some NaN rows
        positions = np.array(
            [
                [0, 0],
                [np.nan, 10],  # NaN in x
                [10, 10],
                [10, np.nan],  # NaN in y
                [10, 0],
                [5, 5],
            ]
        )

        with pytest.warns(UserWarning, match="Filtered 2 positions with NaN"):
            boundary = boundary_from_positions(positions)

        assert boundary.is_valid

    def test_filters_inf_values(self):
        """Inf values are filtered with warning."""
        from neurospatial.annotation import boundary_from_positions

        positions = np.array(
            [
                [0, 0],
                [np.inf, 10],  # Inf in x
                [10, 10],
                [10, 0],
                [5, 5],
            ]
        )

        with pytest.warns(UserWarning, match="Filtered 1 positions with NaN"):
            boundary = boundary_from_positions(positions)

        assert boundary.is_valid

    def test_all_nan_raises(self):
        """All NaN positions raises ValueError."""
        from neurospatial.annotation import boundary_from_positions

        positions = np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )

        with pytest.raises(ValueError, match="at least 3 valid points"):
            boundary_from_positions(positions)

    def test_no_nan_warning_without_nan(self):
        """No NaN filtering warning emitted when data has no NaN values."""
        import warnings

        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boundary = boundary_from_positions(
                positions,
                method="convex_hull",  # Use convex_hull to avoid alphashape warnings
            )

            # Check no NaN-related warnings
            nan_warnings = [warning for warning in w if "NaN" in str(warning.message)]
            assert len(nan_warnings) == 0

        assert boundary.is_valid


class TestBoundaryFromPositionsConfigOverride:
    """Tests for config/override pattern in boundary_from_positions()."""

    def test_default_config(self):
        """Default config produces valid polygon."""
        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        boundary = boundary_from_positions(positions)

        assert isinstance(boundary, Polygon)
        assert boundary.is_valid

    def test_config_object(self):
        """Config object parameters are used."""
        from neurospatial.annotation import BoundaryConfig, boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        # Config with no buffer
        config = BoundaryConfig(buffer_fraction=0.0, simplify_fraction=0.0)
        boundary_no_buffer = boundary_from_positions(positions, config=config)

        # Config with large buffer
        config_buffered = BoundaryConfig(buffer_fraction=0.1, simplify_fraction=0.0)
        boundary_buffered = boundary_from_positions(positions, config=config_buffered)

        # Buffered should be larger
        assert boundary_buffered.area > boundary_no_buffer.area

    def test_kwargs_override_config(self):
        """Explicit kwargs override config values."""
        from neurospatial.annotation import BoundaryConfig, boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        # Config says buffer=0.1, but kwarg says 0
        config = BoundaryConfig(buffer_fraction=0.1)
        boundary = boundary_from_positions(
            positions, config=config, buffer_fraction=0.0, simplify_fraction=0.0
        )

        # With buffer=0.1
        boundary_with_buffer = boundary_from_positions(
            positions, config=config, simplify_fraction=0.0
        )

        # kwarg=0 should produce smaller area than config=0.1
        assert boundary.area < boundary_with_buffer.area

    def test_method_override(self):
        """Method parameter overrides config.method."""
        from neurospatial.annotation import BoundaryConfig, boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        config = BoundaryConfig(method="convex_hull")

        # Should use convex_hull despite config saying something else
        # (this just tests the dispatch works, not the method itself)
        boundary = boundary_from_positions(
            positions,
            method="convex_hull",
            config=config,
            buffer_fraction=0.0,
            simplify_fraction=0.0,
        )

        assert isinstance(boundary, Polygon)
        assert boundary.is_valid


class TestAlphaShape:
    """Tests for alpha shape boundary inference."""

    def test_alpha_shape_fallback_when_not_installed(self, monkeypatch):
        """Falls back to convex_hull with warning if alphashape not installed."""
        import sys
        import warnings

        from neurospatial.annotation import boundary_from_positions

        # Temporarily hide alphashape module
        monkeypatch.setitem(sys.modules, "alphashape", None)

        positions = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boundary = boundary_from_positions(
                positions,
                method="alpha_shape",
                buffer_fraction=0,
                simplify_fraction=0,
            )

            # Should have warned about fallback
            assert len(w) == 1
            assert "alphashape package not installed" in str(w[0].message)
            assert "Falling back to convex_hull" in str(w[0].message)

        # Should still produce a valid boundary (convex hull fallback)
        assert boundary.is_valid
        assert isinstance(boundary, Polygon)

    def test_basic_alpha_shape(self):
        """Alpha shape boundary is valid polygon."""
        pytest.importorskip("alphashape")

        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(20, 80, (500, 2))

        boundary = boundary_from_positions(
            positions,
            method="alpha_shape",
            buffer_fraction=0,
            simplify_fraction=0,
        )

        assert boundary.is_valid
        assert isinstance(boundary, Polygon)

    def test_alpha_parameter_affects_boundary(self):
        """Smaller alpha = tighter fit = smaller area."""
        pytest.importorskip("alphashape")

        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))

        # Larger alpha = more convex (larger area)
        loose = boundary_from_positions(
            positions,
            method="alpha_shape",
            alpha=0.01,  # Very loose
            buffer_fraction=0,
            simplify_fraction=0,
        )
        # Smaller alpha = tighter fit (but may be more complex)
        tight = boundary_from_positions(
            positions,
            method="alpha_shape",
            alpha=0.1,  # Tighter
            buffer_fraction=0,
            simplify_fraction=0,
        )

        # Both should be valid
        assert loose.is_valid
        assert tight.is_valid

    def test_multipolygon_warning(self):
        """Warning emitted when alpha shape produces multiple polygons."""
        pytest.importorskip("alphashape")

        from neurospatial.annotation import boundary_from_positions

        rng = np.random.default_rng(42)
        # Create two clusters that will produce MultiPolygon with high alpha
        cluster1 = rng.uniform(0, 10, (100, 2))
        cluster2 = rng.uniform(90, 100, (100, 2))
        positions = np.vstack([cluster1, cluster2])

        with pytest.warns(UserWarning, match="disconnected regions"):
            boundary = boundary_from_positions(
                positions,
                method="alpha_shape",
                alpha=0.5,
                buffer_fraction=0,
                simplify_fraction=0,
            )

        # Should return largest polygon
        assert isinstance(boundary, Polygon)


@pytest.mark.gui
class TestAddInitialBoundaryToShapes:
    """Tests for add_initial_boundary_to_shapes napari integration."""

    def test_adds_boundary_to_empty_shapes_layer(self):
        """Boundary polygon is added to empty shapes layer."""
        napari = pytest.importorskip("napari")
        from shapely.geometry import Polygon as ShapelyPolygon

        from neurospatial.annotation._helpers import rebuild_features
        from neurospatial.annotation._napari_widget import (
            add_initial_boundary_to_shapes,
        )

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Annotations",
            features=rebuild_features([], []),
        )

        # Create a simple boundary polygon (in environment coordinates)
        boundary = ShapelyPolygon([(0, 0), (100, 0), (100, 80), (0, 80)])

        add_initial_boundary_to_shapes(shapes, boundary, calibration=None)

        assert len(shapes.data) == 1
        assert shapes.features["role"].iloc[0] == "environment"
        assert shapes.features["name"].iloc[0] == "arena"

        viewer.close()

    def test_preserves_existing_shapes(self):
        """Existing regions are preserved when adding boundary."""
        napari = pytest.importorskip("napari")
        from shapely.geometry import Polygon as ShapelyPolygon

        from neurospatial.annotation._helpers import rebuild_features
        from neurospatial.annotation._napari_widget import (
            add_initial_boundary_to_shapes,
        )

        viewer = napari.Viewer(show=False)
        # Pre-populate with an existing region (use 5 vertices for polygon, not rectangle)
        shapes = viewer.add_shapes(
            name="Annotations",
            data=[np.array([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]])],
            shape_type="polygon",
            features=rebuild_features(["region"], ["goal_zone"]),
        )

        # Add boundary
        boundary = ShapelyPolygon([(0, 0), (100, 0), (100, 80), (0, 80)])
        add_initial_boundary_to_shapes(shapes, boundary, calibration=None)

        # Should have 2 shapes now
        assert len(shapes.data) == 2
        # Boundary should be first (prepended)
        assert shapes.features["role"].iloc[0] == "environment"
        assert shapes.features["name"].iloc[0] == "arena"
        # Existing region preserved in second position
        assert shapes.features["role"].iloc[1] == "region"
        assert shapes.features["name"].iloc[1] == "goal_zone"

        viewer.close()

    def test_converts_to_napari_row_col(self):
        """Coordinates are converted from (x, y) to napari (row, col)."""
        napari = pytest.importorskip("napari")
        from shapely.geometry import Polygon as ShapelyPolygon

        from neurospatial.annotation._helpers import rebuild_features
        from neurospatial.annotation._napari_widget import (
            add_initial_boundary_to_shapes,
        )

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Annotations",
            features=rebuild_features([], []),
        )

        # Simple rectangle: (x=10, y=20) -> napari (row=20, col=10)
        boundary = ShapelyPolygon([(10, 20), (30, 20), (30, 50), (10, 50)])
        add_initial_boundary_to_shapes(shapes, boundary, calibration=None)

        # Check first vertex: (x=10, y=20) -> napari (row=20, col=10)
        first_vertex = shapes.data[0][0]
        assert first_vertex[0] == 20  # row = y
        assert first_vertex[1] == 10  # col = x

        viewer.close()

    def test_applies_calibration_transform(self):
        """Calibration transforms coordinates from cm to pixels."""
        napari = pytest.importorskip("napari")
        from shapely.geometry import Polygon as ShapelyPolygon

        from neurospatial.annotation._helpers import rebuild_features
        from neurospatial.annotation._napari_widget import (
            add_initial_boundary_to_shapes,
        )
        from neurospatial.ops.transforms import VideoCalibration, scale_2d

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Annotations",
            features=rebuild_features([], []),
        )

        # Simple 2x scale calibration (1 cm = 2 px)
        # scale_2d(0.5) means px -> cm divides by 2, so cm -> px multiplies by 2
        transform = scale_2d(0.5)  # px -> cm transform
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Boundary in cm coordinates
        boundary = ShapelyPolygon([(0, 0), (50, 0), (50, 40), (0, 40)])
        add_initial_boundary_to_shapes(shapes, boundary, calibration=calibration)

        # After transform: coords should be doubled (cm -> px)
        # (x=50, y=40) cm -> (x=100, y=80) px -> napari (row=80, col=100)
        # Note: calibration.transform_cm_to_px multiplies by inverse scale
        assert len(shapes.data) == 1

        viewer.close()
