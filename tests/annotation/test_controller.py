"""Tests for ShapesLayerController (requires napari)."""

import numpy as np
import pandas as pd
import pytest

from neurospatial.annotation._helpers import rebuild_features
from neurospatial.annotation._state import AnnotationModeState


def _get_feature_default(shapes, key: str) -> str:
    """Get feature default value, handling both scalar and Series cases."""
    value = shapes.feature_defaults.get(key, "")
    if isinstance(value, pd.Series):
        return str(value.iloc[0]) if len(value) > 0 else ""
    return str(value)


@pytest.mark.gui  # Skip in headless CI with: pytest -m "not gui"
class TestShapesLayerControllerApplyMode:
    """Tests for apply_mode() which syncs layer defaults with state."""

    def test_apply_mode_sets_feature_defaults(self):
        """apply_mode sets feature_defaults role and name from state."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="hole")

        controller = ShapesLayerController(shapes, state)
        controller.apply_mode()

        assert _get_feature_default(shapes, "role") == "hole"
        assert _get_feature_default(shapes, "name") == ""  # holes have empty default
        viewer.close()

    def test_apply_mode_sets_face_color(self):
        """apply_mode sets current_face_color to role color."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="hole")

        controller = ShapesLayerController(shapes, state)
        controller.apply_mode()

        # Hole color is red
        assert shapes.current_face_color == "red"
        viewer.close()

    def test_apply_mode_environment_has_arena_default(self):
        """apply_mode for environment sets 'arena' as default name."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)
        controller.apply_mode()

        assert _get_feature_default(shapes, "role") == "environment"
        assert _get_feature_default(shapes, "name") == "arena"
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerDeleteShapes:
    """Tests for delete_shapes_by_indices()."""

    def test_delete_single_shape(self):
        """Delete one shape by index."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
            ],
            features=rebuild_features(["environment", "hole"], ["arena", "hole_1"]),
        )
        state = AnnotationModeState(
            region_type="environment", environment_count=1, hole_count=1
        )

        controller = ShapesLayerController(shapes, state)
        deleted = controller.delete_shapes_by_indices({0})

        assert deleted == 1
        assert len(shapes.data) == 1
        # Remaining shape should be hole_1
        assert shapes.features["name"].iloc[0] == "hole_1"
        assert shapes.features["role"].iloc[0] == "hole"
        # State should be synced
        assert state.environment_count == 0
        assert state.hole_count == 1
        viewer.close()

    def test_delete_multiple_shapes(self):
        """Delete multiple shapes by indices."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
                np.array([[40, 0], [50, 0], [50, 10], [40, 10]]),
            ],
            features=rebuild_features(
                ["environment", "hole", "region"],
                ["arena", "hole_1", "region_1"],
            ),
        )
        state = AnnotationModeState(
            region_type="environment", environment_count=1, hole_count=1, region_count=1
        )

        controller = ShapesLayerController(shapes, state)
        deleted = controller.delete_shapes_by_indices({0, 2})

        assert deleted == 2
        assert len(shapes.data) == 1
        # Only hole_1 should remain
        assert shapes.features["name"].iloc[0] == "hole_1"
        assert state.environment_count == 0
        assert state.hole_count == 1
        assert state.region_count == 0
        viewer.close()

    def test_delete_all_shapes(self):
        """Delete all shapes clears layer and resets counts."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["environment"], ["arena"]),
        )
        state = AnnotationModeState(region_type="environment", environment_count=1)

        controller = ShapesLayerController(shapes, state)
        deleted = controller.delete_shapes_by_indices({0})

        assert deleted == 1
        assert len(shapes.data) == 0
        assert state.environment_count == 0
        viewer.close()

    def test_delete_empty_indices_returns_zero(self):
        """Delete with empty set returns 0 and doesn't modify layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["environment"], ["arena"]),
        )
        state = AnnotationModeState(region_type="environment", environment_count=1)

        controller = ShapesLayerController(shapes, state)
        deleted = controller.delete_shapes_by_indices(set())

        assert deleted == 0
        assert len(shapes.data) == 1
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerRenameShape:
    """Tests for rename_shape()."""

    def test_rename_shape_basic(self):
        """Rename shape with new unique name."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["region_1"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)
        assigned_name, was_modified = controller.rename_shape(0, "goal")

        assert assigned_name == "goal"
        assert was_modified is False
        assert shapes.features["name"].iloc[0] == "goal"
        viewer.close()

    def test_rename_shape_enforces_uniqueness(self):
        """Rename to existing name gets suffix."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
            ],
            features=rebuild_features(["region", "region"], ["goal", "start"]),
        )
        state = AnnotationModeState(region_type="region", region_count=2)

        controller = ShapesLayerController(shapes, state)
        # Try to rename "start" to "goal" - should become "goal_2"
        assigned_name, was_modified = controller.rename_shape(1, "goal")

        assert assigned_name == "goal_2"
        assert was_modified is True
        assert shapes.features["name"].iloc[1] == "goal_2"
        viewer.close()

    def test_rename_to_same_name_allowed(self):
        """Renaming shape to its own name doesn't add suffix."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["goal"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)
        # Rename "goal" to "goal" - should keep as "goal"
        assigned_name, was_modified = controller.rename_shape(0, "goal")

        assert assigned_name == "goal"
        assert was_modified is False
        viewer.close()

    def test_rename_to_empty_string(self):
        """Renaming with empty string assigns empty name (edge case)."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["goal"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)
        # Rename to empty string - assigns empty name
        assigned_name, was_modified = controller.rename_shape(0, "")

        assert assigned_name == ""
        assert was_modified is False  # Empty to empty is not modified
        assert shapes.features["name"].iloc[0] == ""
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerGetters:
    """Tests for getter methods."""

    def test_get_existing_names(self):
        """get_existing_names returns list of shape names."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
            ],
            features=rebuild_features(["environment", "hole"], ["arena", "hole_1"]),
        )
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)
        names = controller.get_existing_names()

        assert names == ["arena", "hole_1"]
        viewer.close()

    def test_get_existing_names_empty_layer(self):
        """get_existing_names returns empty list for empty layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test", features=rebuild_features([], []))
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)
        names = controller.get_existing_names()

        assert names == []
        viewer.close()

    def test_get_existing_region_types(self):
        """get_existing_region_types returns list of shape region types."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
            ],
            features=rebuild_features(["environment", "hole"], ["arena", "hole_1"]),
        )
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)
        region_types = controller.get_existing_region_types()

        assert region_types == ["environment", "hole"]
        viewer.close()

    def test_shape_count(self):
        """shape_count returns number of shapes."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
            ],
            features=rebuild_features(["environment", "hole"], ["arena", "hole_1"]),
        )
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)

        assert controller.shape_count() == 2
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerSyncState:
    """Tests for sync_state_from_layer()."""

    def test_sync_state_from_layer(self):
        """sync_state_from_layer updates state counts from layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
                np.array([[40, 0], [50, 0], [50, 10], [40, 10]]),
            ],
            features=rebuild_features(
                ["environment", "hole", "hole"],
                ["arena", "hole_1", "hole_2"],
            ),
        )
        # State has wrong counts initially
        state = AnnotationModeState(
            region_type="environment", environment_count=0, hole_count=0
        )

        controller = ShapesLayerController(shapes, state)
        controller.sync_state_from_layer()

        assert state.environment_count == 1
        assert state.hole_count == 2
        assert state.region_count == 0
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerUpdateFeatures:
    """Tests for update_features_for_new_shapes()."""

    def test_update_features_adds_role_and_name(self):
        """New shapes get role and auto-generated name."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="region", region_count=0)

        controller = ShapesLayerController(shapes, state)
        # Set up feature_defaults (like widget does)
        controller.apply_mode()

        # Simulate napari adding a shape
        shapes.add_polygons([np.array([[0, 0], [10, 0], [10, 10], [0, 10]])])

        # Controller updates features
        result = controller.update_features_for_new_shapes(0, name_override=None)

        assert result.assigned_name == "region_1"
        assert result.name_was_modified is False
        assert result.last_role == "region"
        assert len(shapes.features) == 1
        assert shapes.features["role"].iloc[0] == "region"
        assert state.region_count == 1
        viewer.close()

    def test_update_features_with_name_override(self):
        """Name override is used instead of auto-generated name."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="region", region_count=0)

        controller = ShapesLayerController(shapes, state)
        controller.apply_mode()
        shapes.add_polygons([np.array([[0, 0], [10, 0], [10, 10], [0, 10]])])

        result = controller.update_features_for_new_shapes(0, name_override="goal")

        assert result.assigned_name == "goal"
        assert result.name_was_modified is False
        assert result.last_role == "region"
        viewer.close()

    def test_update_features_handles_duplicate_name(self):
        """Duplicate name gets suffix."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["goal"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)
        controller.apply_mode()
        shapes.add_polygons([np.array([[20, 0], [30, 0], [30, 10], [20, 10]])])

        result = controller.update_features_for_new_shapes(1, name_override="goal")

        assert result.assigned_name == "goal_2"
        assert result.name_was_modified is True
        viewer.close()

    def test_update_features_no_change_when_no_new_shapes(self):
        """Returns empty result when no shapes were added."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["goal"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)

        # prev_count equals current count - no new shapes
        result = controller.update_features_for_new_shapes(1, name_override=None)

        assert result.assigned_name == ""
        assert result.name_was_modified is False
        assert result.last_role == ""
        viewer.close()


@pytest.mark.gui
class TestShapesLayerControllerEnvironmentCheck:
    """Tests for has_existing_environment()."""

    def test_has_existing_environment_true(self):
        """Returns True when environment exists."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["environment"], ["arena"]),
        )
        state = AnnotationModeState(region_type="environment", environment_count=1)

        controller = ShapesLayerController(shapes, state)

        assert controller.has_existing_environment() is True
        viewer.close()

    def test_has_existing_environment_false(self):
        """Returns False when no environment exists."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            data=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
            features=rebuild_features(["region"], ["goal"]),
        )
        state = AnnotationModeState(region_type="region", region_count=1)

        controller = ShapesLayerController(shapes, state)

        assert controller.has_existing_environment() is False
        viewer.close()

    def test_has_existing_environment_empty_layer(self):
        """Returns False for empty layer."""
        napari = pytest.importorskip("napari")
        from neurospatial.annotation._controller import ShapesLayerController

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(
            name="Test",
            features=rebuild_features([], []),
        )
        state = AnnotationModeState(region_type="environment")

        controller = ShapesLayerController(shapes, state)

        assert controller.has_existing_environment() is False
        viewer.close()
