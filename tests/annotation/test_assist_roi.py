"""Native napari ROI selection invalidates inference without stealing tools."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def roi():
    napari = pytest.importorskip("napari")
    from neurospatial.annotation._assist_roi import AnnotationROIControl

    viewer = napari.Viewer(show=False)
    annotation = viewer.add_shapes(
        [np.array([[0, 0], [0, 8], [5, 8]])], shape_type="polygon"
    )
    changes = []
    widget = AnnotationROIControl(viewer, (7, 13), on_change=lambda: changes.append(1))
    yield widget, viewer, annotation, changes
    widget.dispose()
    widget.close()
    viewer.close()


def _rectangle(y0=2.1, x0=4.2, y1=5.9, x1=10.0):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])


def test_initial_full_frame_preserves_annotation_selection(roi):
    widget, viewer, annotation, changes = roi
    assert widget.bounds is None
    assert widget.layer not in viewer.layers
    assert viewer.layers.selection.active is annotation
    assert not changes


def test_drawing_invalidates_before_rectangle_exists_then_uses_integer_bounds(roi):
    widget, viewer, _, changes = roi
    widget.draw_button.click()
    assert changes
    assert viewer.layers.selection.active is widget.layer
    assert widget.layer.mode == "add_rectangle"
    with pytest.raises(ValueError, match="Finish drawing"):
        _ = widget.bounds
    previous = len(changes)
    widget.layer.add_rectangles(_rectangle())
    assert len(changes) > previous
    assert widget.bounds == (4, 2, 11, 7)
    assert widget.layer.mode == "select"


def test_edit_remove_and_full_image_are_explicit_and_notify(roi):
    widget, _, _, changes = roi
    widget.draw_button.click()
    widget.layer.add_rectangles(_rectangle())
    before = len(changes)
    widget.layer.data = [_rectangle(-2, 3, 5, 15)]
    assert len(changes) > before
    assert widget.bounds == (3, 0, 13, 6)
    before = len(changes)
    widget.layer.data = []
    assert len(changes) > before
    with pytest.raises(ValueError, match="Draw one"):
        _ = widget.bounds
    before = len(changes)
    widget.clear_button.click()
    assert len(changes) > before
    assert widget.bounds is None
    assert not widget.layer.visible


def test_partial_draw_and_multiple_rectangles_do_not_fall_back_to_full_frame(roi):
    widget, _, _, _ = roi
    widget.draw_button.click()
    widget.layer.add_rectangles(_rectangle())
    widget.layer.events.data(action="changing", value=widget.layer.data)
    with pytest.raises(ValueError, match="Finish drawing or editing"):
        _ = widget.bounds
    widget.layer.events.data(action="changed", value=widget.layer.data)
    assert widget.bounds == (4, 2, 11, 7)
    widget.layer.add_rectangles(_rectangle(1, 1, 3, 3))
    with pytest.raises(ValueError, match="Draw one"):
        _ = widget.bounds


def test_edit_callback_does_not_select_roi_after_switching_tools(roi):
    widget, viewer, annotation, changes = roi
    widget.draw_button.click()
    widget.layer.add_rectangles(_rectangle())
    viewer.layers.selection.active = annotation
    annotation.mode = "add_polygon"
    before = len(changes)
    widget.layer.data = [_rectangle(1, 3, 5, 9)]
    assert len(changes) > before
    assert viewer.layers.selection.active is annotation
    assert annotation.mode == "add_polygon"


def test_removing_layer_keeps_active_selection_invalid(roi):
    widget, viewer, _, changes = roi
    widget.draw_button.click()
    widget.layer.add_rectangles(_rectangle())
    before = len(changes)
    viewer.layers.remove(widget.layer)
    assert len(changes) > before
    with pytest.raises(ValueError, match="Draw one"):
        _ = widget.bounds


def test_dispose_removes_callbacks_layer_and_is_idempotent(roi):
    widget, viewer, annotation, changes = roi
    widget.draw_button.click()
    widget.layer.add_rectangles(_rectangle())
    before = len(changes)
    widget.dispose()
    widget.dispose()
    assert widget.layer not in viewer.layers
    assert annotation in viewer.layers
    widget.layer.data = [_rectangle()]
    widget.draw_button.click()
    widget.clear_button.click()
    assert len(changes) == before
