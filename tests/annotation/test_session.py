"""Closing an annotation window returns a validated result or keeps the draft."""

import warnings

import pytest


@pytest.fixture
def viewer():
    napari = pytest.importorskip("napari")
    viewer = napari.Viewer()
    yield viewer
    viewer.close()


def _annotate(viewer, prepare_result):
    """Stand in for a public annotation function calling the session."""
    from neurospatial.annotation._session import run_annotation_session

    return run_annotation_session(viewer, prepare_result)


@pytest.mark.gui
def test_result_warnings_reach_the_caller_after_close(viewer, monkeypatch):
    import napari
    from napari.utils.notifications import notification_manager

    def run():
        # napari.run shows warnings raised inside its event loop as toasts.
        with notification_manager:
            assert viewer.window._qt_window.close()

    def prepare_result():
        warnings.warn("No annotations were drawn.", UserWarning, stacklevel=2)
        return "result"

    monkeypatch.setattr(napari, "run", run)
    with pytest.warns(UserWarning, match="No annotations were drawn") as record:
        assert _annotate(viewer, prepare_result) == "result"
    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.gui
def test_discarding_an_unsavable_draft_closes_and_raises(viewer, monkeypatch):
    import napari

    from neurospatial.annotation import _session

    asked = []
    window = viewer.window._qt_window

    def run():
        assert window.close()

    def prepare_result():
        raise ValueError("Need at least 2 nodes")

    monkeypatch.setattr(napari, "run", run)
    monkeypatch.setattr(
        _session, "_keep_editing", lambda parent, error: asked.append(str(error))
    )
    with pytest.raises(
        RuntimeError,
        match="Annotations were discarded without saving: Need at least 2 nodes",
    ):
        _annotate(viewer, prepare_result)
    assert asked == ["Need at least 2 nodes"]
    assert not window.isVisible()


@pytest.mark.gui
def test_session_waits_when_an_ipython_event_loop_owns_qt(viewer, monkeypatch):
    import napari
    from qtpy.QtCore import QTimer

    window = viewer.window._qt_window
    # With %gui qt, napari.run returns at once and leaves the window open.
    monkeypatch.setattr(napari, "run", lambda: None)
    QTimer.singleShot(0, window.close)
    assert _annotate(viewer, lambda: "result") == "result"
    assert not window.isVisible()


@pytest.mark.gui
@pytest.mark.parametrize(
    ("button", "keep"), [("Keep Editing", True), ("Discard and Close", False)]
)
def test_unsavable_draft_dialog_offers_keep_or_discard(viewer, button, keep):
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication, QPushButton

    from neurospatial.annotation._session import _keep_editing

    shown = []

    def choose():
        dialog = QApplication.activeModalWidget()
        shown.append(dialog.text())
        next(b for b in dialog.findChildren(QPushButton) if b.text() == button).click()

    QTimer.singleShot(0, choose)
    assert _keep_editing(viewer.window._qt_window, ValueError("bad ring")) is keep
    assert shown == ["Cannot save annotations: bad ring"]


@pytest.mark.gui
def test_cancelled_close_keeps_the_session_running(viewer, monkeypatch):
    import napari
    from qtpy.QtCore import QTimer

    window = viewer.window._qt_window
    close_event = type(window).closeEvent
    attempts = []
    after_cancel = []

    def confirm_close(self, event):
        # Stand in for napari's confirm-close dialog: cancel the first close.
        attempts.append(1)
        if len(attempts) == 1:
            event.ignore()
            return
        close_event(self, event)

    def check_then_close():
        after_cancel.append(window.isVisible())
        window.close()

    monkeypatch.setattr(type(window), "closeEvent", confirm_close)
    monkeypatch.setattr(napari, "run", lambda: None)
    QTimer.singleShot(0, window.close)
    QTimer.singleShot(50, check_then_close)
    prepared = []

    def prepare_result():
        prepared.append(1)
        return len(prepared)

    # The second close's result is returned, not the cancelled first one's.
    assert _annotate(viewer, prepare_result) == 2
    assert after_cancel == [True]
    assert not window.isVisible()


@pytest.mark.gui
def test_view_is_fitted_once_the_window_has_its_final_size(viewer, monkeypatch):
    import napari
    from qtpy.QtWidgets import QApplication

    fitted = []
    monkeypatch.setattr(
        type(viewer), "reset_view", lambda self, *args, **kwargs: fitted.append(True)
    )

    def run():
        QApplication.processEvents()
        assert fitted
        assert viewer.window._qt_window.close()

    monkeypatch.setattr(napari, "run", run)
    assert _annotate(viewer, lambda: "result") == "result"
