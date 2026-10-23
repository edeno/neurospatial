"""Native Qt checks for the editable mask and accepted polygon preview."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation.segmentation import SegmentationProposal

pytestmark = pytest.mark.gui


@pytest.fixture
def preview():
    napari = pytest.importorskip("napari")
    from neurospatial.annotation._outline_preview import OutlinePreviewControl

    viewer = napari.Viewer(show=False)
    annotations = viewer.add_shapes(
        [np.array([[0, 0], [0, 8], [7, 8]])], shape_type="polygon"
    )
    points = viewer.add_points(np.empty((0, 2)), ndim=2)
    points.mode = "add"
    changes = []
    control = OutlinePreviewControl(viewer, on_change=lambda: changes.append(True))
    yield control, viewer, annotations, points, changes
    control.dispose()
    viewer.close()


def _proposal():
    mask = np.zeros((16, 24), dtype=bool)
    for row in range(1, 13):
        mask[row, 1 : 9 + row // 3] = True
    mask[5:7, 4:6] = False
    mask[10:14, 17:21] = True
    return SegmentationProposal(
        mask, score=0.7, metadata={"provider": {"name": "fake"}}
    )


def _assert_paths_match(control):
    assert control.outline is not None
    assert control.polygon_layer is not None
    assert not control.polygon_layer.editable
    rings = control.outline.rings
    assert len(control.polygon_layer.data) == len(rings)
    assert set(control.polygon_layer.shape_type) == {"path"}
    for actual, expected in zip(control.polygon_layer.data, rings, strict=True):
        assert_array_equal(actual, expected[:, ::-1])
        assert_array_equal(actual[0], actual[-1])


def test_paths_include_holes_and_preserve_active_prompt_tool(preview):
    control, viewer, annotations, points, changes = preview
    original = annotations.data[0].copy()
    proposal = _proposal()
    control.display(proposal)
    _assert_paths_match(control)
    assert control.outline.simplify_tolerance_px == pytest.approx(1.4)
    assert len(control.polygon_layer.data) == 3
    assert control.outline.proposal.metadata == proposal.metadata
    assert control.mask_layer.contour == 2
    assert control.mask_layer.opacity == 1.0
    assert control.mask_layer.mode == "pan_zoom"
    assert not control.mask_layer.visible
    assert viewer.layers.selection.active is points
    assert points.mode == "add"
    assert (
        viewer.layers.index(control.mask_layer)
        < viewer.layers.index(control.polygon_layer)
        < viewer.layers.index(points)
    )
    assert_array_equal(annotations.data[0], original)
    assert changes


def test_detail_changes_reuse_exact_vectorization_and_layers(preview, monkeypatch):
    from neurospatial.annotation import _outline_preview

    control, viewer, _annotations, points, _changes = preview
    conversions = []
    original = _outline_preview.polygon_outline

    def count(proposal):
        conversions.append(proposal)
        return original(proposal)

    monkeypatch.setattr(_outline_preview, "polygon_outline", count)
    proposal = _proposal()
    control.display(proposal)
    control.detail_slider.setValue(0)
    exact = control.outline
    mask_layer, polygon_layer = control.mask_layer, control.polygon_layer
    control.detail_slider.setValue(10)
    assert control.outline.simplify_tolerance_px == 1.0
    assert control.outline.vertex_count < exact.vertex_count
    _assert_paths_match(control)
    control.detail_slider.setValue(20)
    control.detail_slider.setValue(0)
    assert control.snapshot() is exact
    assert len(conversions) == 1
    assert viewer.layers.selection.active is points
    assert points.mode == "add"
    control.display(proposal)
    assert len(conversions) == 2
    assert control.mask_layer is mask_layer
    assert control.polygon_layer is polygon_layer


def test_polygon_paths_stay_uneditable_after_data_and_dimension_changes(preview):
    control, viewer, _annotations, _points, _changes = preview
    control.display(_proposal())
    paths = control.polygon_layer
    paths.data = list(paths.data)
    assert not paths.editable
    viewer.dims.ndisplay = 3
    assert not paths.editable
    viewer.dims.ndisplay = 2
    assert not paths.editable
    control.snapshot()
    _assert_paths_match(control)


def test_paint_updates_mask_authoritatively_and_snapshot_catches_silent_edits(preview):
    control, viewer, _annotations, _points, _changes = preview
    source = _proposal()
    control.display(source)
    control.detail_slider.setValue(10)
    control.edit()
    assert viewer.layers.selection.active is control.mask_layer
    assert control.mask_layer.mode == "paint"
    assert control.mask_layer.contour == 0
    assert control.mask_layer.opacity == 0.5
    assert control.mask_layer.visible
    painted = np.array(control.mask_layer.data, copy=True)
    painted[14, 0] = 1
    control.mask_layer.data = painted
    assert control.outline is None
    snapshot = control.snapshot()
    assert_array_equal(snapshot.proposal.mask, painted != 0)
    assert snapshot.proposal.score == source.score
    assert snapshot.proposal.metadata == source.metadata
    _assert_paths_match(control)
    # napari paint mutates in place, and acceptance must also catch missing events.
    control.mask_layer.data[15, 0] = 1
    assert control.snapshot().proposal.mask[15, 0]
    assert not source.mask[15, 0]
    control.mask_layer.events.paint()
    assert control.outline is None
    assert control._timer.isActive()
    control.snapshot()
    assert not control._timer.isActive()


def test_simplification_failure_keeps_slider_available_for_exact_recovery(
    preview, monkeypatch
):
    from neurospatial.annotation import _outline_preview

    control, _viewer, _annotations, _points, _changes = preview
    control.display(_proposal())
    control.detail_slider.setValue(0)
    exact = control.outline
    original = _outline_preview.simplify_polygon_outline

    def reject_positive_tolerance(outline, tolerance):
        if tolerance > 0:
            raise ValueError("Simplification could not preserve valid geometry.")
        return original(outline, tolerance)

    monkeypatch.setattr(
        _outline_preview, "simplify_polygon_outline", reject_positive_tolerance
    )
    control.detail_slider.setValue(10)
    assert control.outline is None
    assert control.detail_slider.isEnabled()
    assert control._exact is exact
    assert "Simplification could not" in control.detail_status.text()
    assert len(control.polygon_layer.data) == 0
    control.detail_slider.setValue(0)
    assert control.outline is exact
    _assert_paths_match(control)


def test_paint_events_debounce_into_displayed_geometry(preview):
    from qtpy.QtTest import QTest

    control, _viewer, _annotations, _points, _changes = preview
    control.display(_proposal())
    control.mask_layer.data[15, 0] = 1
    control.mask_layer.events.paint()
    assert control.outline is None
    QTest.qWait(200)
    assert control.outline.proposal.mask[15, 0]
    _assert_paths_match(control)


def test_empty_mask_rejects_acceptance_and_remains_paintable(preview):
    control, _viewer, _annotations, _points, _changes = preview
    control.display(_proposal())
    control.mask_layer.data[:] = 0
    with pytest.raises(ValueError, match="empty"):
        control.snapshot()
    assert control.outline is None
    assert not control.detail_slider.isEnabled()
    assert len(control.polygon_layer.data) == 0
    assert "empty" in control.detail_status.text()
    control.edit()
    assert control.mask_layer.mode == "paint"
    control.mask_layer.data[2:5, 3:7] = 1
    assert control.snapshot().proposal.mask.sum() == 12
    assert control.detail_slider.isEnabled()
    _assert_paths_match(control)


@pytest.mark.parametrize("layer_name", ["mask_layer", "polygon_layer"])
def test_removing_preview_layer_invalidates_and_show_can_recover(preview, layer_name):
    control, viewer, _annotations, _points, _changes = preview
    control.display(_proposal())
    viewer.layers.remove(getattr(control, layer_name))
    assert control.outline is None
    assert not control.detail_slider.isEnabled()
    with pytest.raises(ValueError, match="removed"):
        control.snapshot()
    control.display(_proposal())
    assert control.outline is not None
    _assert_paths_match(control)


def test_invalidate_cancels_debounce_and_dispose_detaches_callbacks(preview):
    from qtpy.QtTest import QTest

    control, viewer, annotations, points, changes = preview
    control.display(_proposal())
    mask_layer = control.mask_layer
    mask_layer.events.paint()
    control.invalidate(keep_preview=True)
    assert not control._timer.isActive()
    assert control.outline is None
    assert control.mask_layer is mask_layer
    assert control.polygon_layer.opacity == 0.3
    assert not control.detail_slider.isEnabled()
    with pytest.raises(ValueError, match="removed"):
        control.snapshot()
    control.display(_proposal())
    control.mask_layer.events.paint()
    control.dispose()
    before = len(changes)
    mask_layer.events.data()
    QTest.qWait(150)
    assert len(changes) == before
    assert not control._timer.isActive()
    assert control.mask_layer is control.polygon_layer is None
    assert list(viewer.layers) == [annotations, points]
    control.dispose()


def test_dispose_after_docked_widgets_are_deleted(preview):
    control, viewer, _annotations, _points, changes = preview
    viewer.window.add_dock_widget(control, name="Polygon detail")
    control.display(_proposal())
    viewer.close()
    before = len(changes)
    control.dispose()
    assert len(changes) == before
    assert control.outline is None
