"""Validate annotation results before Qt tears down the editable viewer."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import napari

_Result = TypeVar("_Result")


def run_annotation_session(
    viewer: "napari.Viewer", prepare_result: Callable[[], _Result]
) -> _Result:
    """Run the viewer, retaining its draft whenever preparing a result fails."""
    import napari
    from napari.utils.notifications import show_error
    from qtpy.QtCore import QEvent, QObject

    results: list[_Result] = []

    class DraftCloseGuard(QObject):
        def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
            if event is not None and event.type() == QEvent.Type.Close:
                try:
                    result = prepare_result()
                except Exception as exc:
                    show_error(f"Cannot save annotations: {exc}")
                    event.ignore()
                    return True
                results[:] = [result]
            return False

    window = viewer.window._qt_window
    guard = DraftCloseGuard(window)
    window.installEventFilter(guard)
    try:
        napari.run()
    finally:
        window.removeEventFilter(guard)
        viewer.close()
    if not results:
        raise RuntimeError("Annotation window closed without a validated result.")
    return results[0]
