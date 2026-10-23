"""Validate annotation results before Qt tears down the editable viewer."""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import napari

_Result = TypeVar("_Result")


def _keep_editing(parent: Any, error: Exception) -> bool:
    """Ask whether to return to an unsavable draft or discard it and close."""
    from qtpy.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Cannot Save Annotations")
    box.setText(f"Cannot save annotations: {error}")
    box.setInformativeText(
        "Keep editing to fix the problem, or discard the annotations and close."
    )
    keep = box.addButton("Keep Editing", QMessageBox.ButtonRole.RejectRole)
    discard = box.addButton("Discard and Close", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    box.exec()
    return box.clickedButton() is not discard


def run_annotation_session(
    viewer: "napari.Viewer", prepare_result: Callable[[], _Result]
) -> _Result:
    """Run the viewer, retaining its draft whenever preparing a result fails.

    A result is prepared when the window closes. If that fails, the user keeps
    editing or discards the draft, which raises here. Warnings raised while
    preparing the result are re-issued to the caller once the window is closed,
    since napari shows warnings raised inside its event loop only in the window.
    """
    import napari
    from qtpy.QtCore import (  # type: ignore[attr-defined]
        QEvent,
        QEventLoop,
        QObject,
        QTimer,
    )

    results: list[_Result] = []
    caught: list[warnings.WarningMessage] = []
    discarded: list[Exception] = []
    window = viewer.window._qt_window
    loop = QEventLoop()
    # napari's own close handler runs after this filter and can still cancel
    # the close, for example from its confirm-close dialog. An accepted close
    # hides the window while that close is being delivered.
    closing = [False]

    def close_delivered() -> None:
        closing[0] = False

    class DraftCloseGuard(QObject):
        def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
            if event is None:
                return False
            if event.type() == QEvent.Type.Hide and closing[0]:
                loop.quit()
            if event.type() == QEvent.Type.Close:
                # Only the attempt that actually closes the window counts.
                results.clear()
                caught.clear()
                discarded.clear()
                try:
                    with warnings.catch_warnings(record=True) as recorded:
                        result = prepare_result()
                except Exception as exc:
                    if _keep_editing(window, exc):
                        event.ignore()
                        return True
                    discarded[:] = [exc]
                else:
                    results[:] = [result]
                    caught[:] = recorded
                closing[0] = True
                QTimer.singleShot(0, close_delivered)
            return False

    guard = DraftCloseGuard(window)
    window.installEventFilter(guard)
    # The camera was fitted before the docks took their width, which hid part
    # of the frame. Fit it again once the shown window has its final size.
    QTimer.singleShot(0, viewer.reset_view)
    try:
        napari.run()
        if window.isVisible():
            # An IPython or PyCharm event loop makes napari.run return at once.
            loop.exec()
    finally:
        window.removeEventFilter(guard)
        viewer.close()
    if discarded:
        raise RuntimeError(
            f"Annotations were discarded without saving: {discarded[0]}"
        ) from discarded[0]
    if not results:
        raise RuntimeError("Annotation window closed without a validated result.")
    for warning in caught:
        warnings.warn(warning.message, stacklevel=3)
    return results[0]
