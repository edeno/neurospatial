"""Native Qt checks for assisted annotation preview and calibration boundaries."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytestmark = pytest.mark.gui


def _wait_until(predicate):
    from qtpy.QtWidgets import QApplication

    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert predicate()


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, *args):
        for callback in self.callbacks[:]:
            callback(*args)


class _Worker:
    def __init__(self, provider, frame, prompt):
        self.provider = provider
        self.frame = frame
        self.prompt = prompt
        self.returned = _Signal()
        self.errored = _Signal()
        self.finished = _Signal()
        self.started = False
        self.quit_called = False

    def start(self):
        self.started = True

    def quit(self):
        self.quit_called = True

    def complete(self):
        try:
            self.returned.emit(self.provider.predict(self.frame, self.prompt))
        except Exception as exc:
            self.errored.emit(exc)
        finally:
            self.finished.emit()


class _Provider:
    def __init__(self):
        self.calls = []
        self.error = None

    def predict(self, frame, prompt):
        from neurospatial.annotation.segmentation import SegmentationProposal

        self.calls.append((frame, prompt))
        if self.error is not None:
            raise self.error
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[1:4, 2:6] = True
        return SegmentationProposal(mask, score=0.7, metadata={"source": "fake"})


@pytest.fixture
def assist(monkeypatch):
    napari = pytest.importorskip("napari")
    from neurospatial.annotation import _assist_widget

    workers = []
    real_predict = _assist_widget._predict

    def defer(*args):
        worker = _Worker(*args)
        workers.append(worker)
        return worker

    monkeypatch.setattr(_assist_widget, "_predict", defer)
    viewer = napari.Viewer(show=False)
    # Existing annotations must retain their pixel geometry and selection.
    annotations = viewer.add_shapes(
        [np.array([[0, 0], [0, 8], [7, 8]])], shape_type="polygon"
    )
    provider = _Provider()
    accepted = []
    calibration = []
    bins = []
    widget = _assist_widget.AnnotationAssistWidget(
        viewer,
        np.zeros((8, 12, 3), dtype=np.uint8),
        provider=provider,
        calibration=None,
        bin_size=2.0,
        on_calibration=calibration.append,
        on_bin_size=bins.append,
        on_accept=accepted.append,
    )
    state = {
        "real_predict": real_predict,
        "widget": widget,
        "viewer": viewer,
        "provider": provider,
        "workers": workers,
        "accepted": accepted,
        "calibrations": calibration,
        "bins": bins,
        "annotations": annotations,
    }
    yield state
    widget.dispose()
    if not state.get("dock_closed", False):
        widget.close()
    viewer.close()


def test_success_snapshots_xy_prompts_and_accepts_current_painted_mask(assist):
    widget = assist["widget"]
    assert len(assist["viewer"].layers) == 1
    assert not widget.tolerance.isVisible()
    widget._inside.data = np.array([[2.0, 3.0]])
    widget._outside.data = np.array([[6.0, 10.0]])
    widget._suggest()
    worker = assist["workers"][-1]
    assert worker.started
    assert not assist["provider"].calls
    assert not widget.suggest_button.isEnabled()
    widget._suggest()
    assert len(assist["workers"]) == 1
    assert_array_equal(worker.prompt.points, [[3, 2], [10, 6]])
    assert_array_equal(worker.prompt.labels, [1, 0])
    worker.complete()
    assert widget.suggest_button.isEnabled()
    assert widget.accept_button.isEnabled()
    edited = np.zeros((8, 12), dtype=np.uint8)
    edited[5:7, 7:10] = 1
    widget._preview.data = edited
    widget._accept()
    proposal = assist["accepted"][0]
    assert_array_equal(proposal.mask, edited != 0)
    assert proposal.score == 0.7
    assert proposal.metadata == {"source": "fake"}
    assert widget._preview is None
    assert len(assist["viewer"].layers) == 1
    assert len(widget._inside.data) == len(widget._outside.data) == 0


def test_live_clicks_coalesce_and_keep_prompt_tool_active(assist):
    widget, viewer = assist["widget"], assist["viewer"]
    widget._activate(widget._inside)
    widget._inside.add([2.0, 3.0])
    widget._inside.add([3.0, 4.0])
    widget._activate(widget._outside)
    widget._outside.add([6.0, 10.0])
    _wait_until(lambda: len(assist["workers"]) == 1)
    worker = assist["workers"][0]
    assert_array_equal(worker.prompt.points, [[3, 2], [4, 3], [10, 6]])
    worker.complete()
    preview = widget._preview
    assert preview.contour == 2
    assert viewer.layers.selection.active is widget._outside
    assert widget._outside.mode == "add"
    assert viewer.layers.index(preview) < viewer.layers.index(widget._inside)
    assert viewer.layers.index(preview) < viewer.layers.index(widget._outside)

    # Keep the previous outline visible, but never accept it for new prompts.
    widget._outside.add([5.0, 9.0])
    assert widget._preview is preview
    assert not widget.accept_button.isEnabled()
    widget._accept()
    assert not assist["accepted"]
    _wait_until(lambda: len(assist["workers"]) == 2)
    assist["workers"][-1].complete()
    assert widget._preview is preview
    assert widget.accept_button.isEnabled()
    assert viewer.layers.selection.active is widget._outside
    widget.edit_button.click()
    assert viewer.layers.selection.active is preview
    assert preview.mode == "paint"
    assert preview.contour == 0


def test_cleanup_toggle_restores_raw_prediction_without_inference_and_accepts_preview(
    assist, monkeypatch
):
    from neurospatial.annotation.segmentation import SegmentationProposal

    widget = assist["widget"]
    mask = np.zeros((8, 12), dtype=bool)
    mask[1:5, 2:7] = True
    mask[3, 4] = False
    mask[6:8, 10:12] = True
    raw = SegmentationProposal(mask, metadata={"model": "fake"})
    monkeypatch.setattr(assist["provider"], "predict", lambda *_: raw)
    widget._activate(widget._inside)
    widget._inside.add([2.0, 3.0])
    widget._suggest()
    assist["workers"][-1].complete()
    assert widget.remove_fragments.isChecked()
    assert not widget._preview.data[6:8, 10:12].any()
    assert not widget._preview.data[3, 4]
    assert "Removed 1" in widget.status.text()
    preview = widget._preview
    widget.remove_fragments.setChecked(False)
    assert widget._preview is preview
    assert_array_equal(preview.data, mask)
    widget.remove_fragments.setChecked(True)
    assert len(assist["workers"]) == 1
    assert assist["viewer"].layers.selection.active is widget._inside
    # Manual painting after cleanup is authoritative: don't filter it again
    # at acceptance, or we'd silently remove a region the user just painted.
    widget._edit_preview()
    painted = np.array(preview.data, copy=True)
    painted[7, 0] = 1
    preview.data = painted
    widget._accept()
    assert_array_equal(assist["accepted"][0].mask, painted != 0)
    assert assist["accepted"][0].metadata["annotation_cleanup"]["removed_pixels"] == 4
    assert_array_equal(raw.mask, mask)


def test_cleanup_cannot_restore_stale_preview_after_prompt_changes(assist):
    widget = assist["widget"]
    widget._inside.add([2.0, 3.0])
    widget._suggest()
    assist["workers"][-1].complete()
    widget._inside.add([3.0, 4.0])
    widget.remove_fragments.setChecked(False)
    widget.remove_fragments.setChecked(True)
    assert not widget.accept_button.isEnabled()
    assert not widget.edit_button.isEnabled()
    widget._accept()
    assert not assist["accepted"]


def test_live_edits_during_inference_run_only_latest_prompt_after_finish(assist):
    widget = assist["widget"]
    widget._inside.add([2.0, 3.0])
    _wait_until(lambda: len(assist["workers"]) == 1)
    widget._inside.add([3.0, 4.0])
    widget._outside.add([6.0, 10.0])
    _wait_until(lambda: not widget._preview_timer.isActive())
    assert len(assist["workers"]) == 1
    assist["workers"][0].complete()
    assert widget._preview is None
    assert not widget.accept_button.isEnabled()
    _wait_until(lambda: len(assist["workers"]) == 2)
    worker = assist["workers"][1]
    assert_array_equal(worker.prompt.points, [[3, 2], [4, 3], [10, 6]])
    assert_array_equal(worker.prompt.labels, [1, 1, 0])
    worker.complete()
    assert widget.accept_button.isEnabled()


@pytest.mark.parametrize("action", ["clear", "discard", "disable", "dispose"])
def test_pending_live_preview_can_be_cancelled(assist, action):
    from qtpy.QtTest import QTest

    widget = assist["widget"]
    widget._inside.add([2.0, 3.0])
    _wait_until(lambda: len(assist["workers"]) == 1)
    widget._outside.add([6.0, 10.0])
    if action == "clear":
        widget._clear_prompts()
    elif action == "discard":
        widget._invalidate()
    elif action == "disable":
        widget.live_preview.setChecked(False)
        assert "Click Update outline" in widget.status.text()
    else:
        widget.dispose()
    assist["workers"][0].complete()
    QTest.qWait(250)
    assert len(assist["workers"]) == 1
    assert widget._preview is None
    assert not assist["accepted"]


def test_removing_last_inside_point_clears_preview_without_empty_request(assist):
    from qtpy.QtTest import QTest

    widget = assist["widget"]
    widget._inside.add([2.0, 3.0])
    _wait_until(lambda: len(assist["workers"]) == 1)
    assist["workers"][0].complete()
    widget._inside.selected_data = {0}
    widget._inside.remove_selected()
    QTest.qWait(250)
    assert widget._preview is None
    assert not widget.accept_button.isEnabled()
    assert len(assist["workers"]) == 1


def test_manual_mode_waits_for_update_button(assist):
    from qtpy.QtTest import QTest

    widget = assist["widget"]
    widget.live_preview.setChecked(False)
    widget._inside.add([2.0, 3.0])
    QTest.qWait(250)
    assert not assist["workers"]
    widget.suggest_button.click()
    assist["workers"][0].complete()
    assert widget.accept_button.isEnabled()


def test_error_allows_retry_and_accept_error_preserves_preview(assist):
    widget = assist["widget"]
    widget._inside.data = np.array([[2.0, 3.0]])
    assist["provider"].error = RuntimeError("choose another seed")
    widget._suggest()
    assist["workers"][-1].complete()
    assert "choose another seed" in widget.status.text()
    assert widget.suggest_button.isEnabled()
    assist["provider"].error = None
    widget._suggest()
    assist["workers"][-1].complete()
    preview = widget._preview

    def reject(_proposal):
        raise ValueError("Name the region first")

    widget._on_accept = reject
    widget._accept()
    assert "Name the region first" in widget.status.text()
    assert widget._preview is preview
    assert widget.accept_button.isEnabled()
    widget._preview.data = np.zeros((8, 12), dtype=np.uint8)
    widget._accept()
    assert "empty" in widget.status.text()
    assert not assist["accepted"]


@pytest.mark.parametrize("invalidate", ["changed", "clear", "discard"])
def test_stale_worker_results_cannot_reappear_or_be_accepted(assist, invalidate):
    widget = assist["widget"]
    widget._inside.data = np.array([[2.0, 3.0]])
    widget._suggest()
    if invalidate == "changed":
        widget._inside.data = np.array([[4.0, 5.0]])
    elif invalidate == "clear":
        widget._clear_prompts()
    else:
        widget._invalidate()
    assist["workers"][-1].complete()
    widget._accept()
    assert not assist["accepted"]
    assert widget._preview is None
    assert not widget.accept_button.isEnabled()
    assert widget.suggest_button.isEnabled()


def test_calibration_swaps_coordinates_once_and_keeps_explicit_bin_size(assist):
    widget = assist["widget"]
    original = assist["annotations"].data[0].copy()
    widget._scale.data = np.array([[1.0, 2.0], [1.0, 10.0]])
    widget.known_distance.setValue(4.0)
    widget._apply_calibration()
    calibration = assist["calibrations"][-1]
    assert calibration.frame_size_px == (12, 8)
    assert calibration.cm_per_px == pytest.approx(0.5)
    assert_array_equal(
        calibration.transform_px_to_cm(np.array([[2.0, 1.0], [10.0, 1.0]])),
        [[1.0, 3.5], [5.0, 3.5]],
    )
    assert widget.bin_size.value() == 2.0
    assert widget.bin_size.suffix() == " cm"
    assert "now in cm" in widget.units_status.text()
    assert_array_equal(assist["annotations"].data[0], original)
    assert assist["bins"] == []
    widget.sync_bin_size(3.0)
    assert assist["bins"] == []
    widget.bin_size.setValue(4.0)
    assert assist["bins"] == [4.0]
    widget._clear_calibration()
    assert assist["calibrations"][-1] is None
    assert widget.bin_size.suffix() == " pixels"
    assert widget.bin_size.value() == 4.0


@pytest.mark.parametrize(
    "points", [[], [[1, 2]], [[1, 2], [1, 2]], [[1, 2], [np.nan, 3]]]
)
def test_invalid_scale_does_not_change_calibration(assist, points):
    widget = assist["widget"]
    widget._scale.data = np.asarray(points, dtype=float).reshape((-1, 2))
    widget._apply_calibration()
    assert assist["calibrations"] == []
    assert "Could not calibrate" in widget.status.text()


def test_dispose_detaches_layers_and_signals_and_ignores_late_completion(assist):
    widget = assist["widget"]
    widget._activate(widget._inside)
    widget._activate(widget._scale)
    widget._inside.data = np.array([[2.0, 3.0]])
    widget._suggest()
    worker = assist["workers"][-1]
    generation = widget._generation
    widget.dispose()
    assert worker.quit_called
    assert len(assist["viewer"].layers) == 1
    assert widget._generation > generation
    generation = widget._generation
    widget._inside.data = np.array([[2.0, 3.0]])
    assert widget._generation == generation
    worker.complete()
    assert widget._preview is None
    assert not assist["accepted"]
    widget.dispose()


def test_region_only_bin_control_is_disabled():
    napari = pytest.importorskip("napari")
    from neurospatial.annotation._assist_widget import AnnotationAssistWidget
    from neurospatial.annotation.segmentation import RegionGrowProvider

    viewer = napari.Viewer(show=False)
    widget = AnnotationAssistWidget(
        viewer,
        np.zeros((8, 12, 3), dtype=np.uint8),
        provider=RegionGrowProvider(),
        calibration=None,
        bin_size=None,
        on_calibration=lambda _: None,
        on_bin_size=lambda _: None,
        on_accept=lambda _: None,
    )
    try:
        assert not widget.bin_size.isEnabled()
        assert widget.tolerance.value() == pytest.approx(0.1)
        assert len(viewer.layers) == 0
    finally:
        widget.dispose()
        widget.close()
        viewer.close()


def test_real_worker_inference_runs_off_gui_thread(assist, monkeypatch):
    from neurospatial.annotation import _assist_widget

    main_thread = threading.get_ident()
    inference_threads = []
    layer_threads = []
    provider = assist["provider"]
    original_predict = provider.predict

    def predict(frame, prompt):
        inference_threads.append(threading.get_ident())
        return original_predict(frame, prompt)

    monkeypatch.setattr(provider, "predict", predict)
    monkeypatch.setattr(_assist_widget, "_predict", assist["real_predict"])
    widget = assist["widget"]
    widget._activate(widget._inside)
    assist["viewer"].layers.events.inserted.connect(
        lambda _event: layer_threads.append(threading.get_ident())
    )
    widget._inside.data = np.array([[2.0, 3.0]])
    _wait_until(lambda: widget._preview is not None and widget._worker is None)
    assert widget._worker is None
    assert widget._preview is not None
    assert len(inference_threads) == 1
    assert inference_threads[0] != main_thread
    assert layer_threads == [main_thread]


def test_cleanup_after_docked_viewer_closes(assist):
    widget = assist["widget"]
    viewer = assist["viewer"]
    viewer.window.add_dock_widget(widget, name="Assisted annotation")
    widget._activate(widget._inside)
    widget._inside.data = np.array([[2.0, 3.0]])
    widget._suggest()
    viewer.close()
    assist["dock_closed"] = True
    widget.dispose()
    assist["workers"][-1].complete()
    assert widget._disposed
    assert widget._preview is None
