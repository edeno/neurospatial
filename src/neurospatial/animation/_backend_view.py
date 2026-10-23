"""Typed boundary between a scene view and existing native viewers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._resources import _add_cleanup_note, _raise_cleanup_errors

FrameCapture = Callable[[float], NDArray[np.uint8]]
CaptureSession = Callable[[], AbstractContextManager[FrameCapture]]


class _CloseNotification:
    """Notify owners once after native resources have been released."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []
        self._closed = False

    def add(self, callback: Callable[[], None]) -> None:
        if self._closed:
            callback()
        else:
            self._callbacks.append(callback)

    def notify(self) -> None:
        if self._closed:
            return
        self._closed = True
        callbacks, self._callbacks = self._callbacks, []
        failures: list[BaseException] = []
        for callback in callbacks:
            try:
                callback()
            except BaseException as error:
                failures.append(error)
        _raise_cleanup_errors("Native view close notification failed.", failures)


class _ViewController(Protocol):
    """A viewer's controls use absolute recording seconds."""

    @property
    def time(self) -> float: ...

    def seek(self, time: float) -> None: ...

    def capture_session(self) -> AbstractContextManager[FrameCapture]: ...

    def close(self) -> None: ...

    def _add_close_callback(self, callback: Callable[[], None]) -> None: ...


class _BackendView(Protocol):
    """The sole backend contract consumed by AnimationView."""

    @property
    def native(self) -> Any: ...

    @property
    def time(self) -> float: ...

    @property
    def capture_session(self) -> CaptureSession | None: ...

    def seek(self, time: float) -> None: ...

    def close(self) -> None: ...

    def _add_close_callback(self, callback: Callable[[], None]) -> None: ...


class _NativeBackendView:
    """Expose an existing typed controller and its native escape hatch."""

    def __init__(self, native: Any, controller: _ViewController) -> None:
        self.native = native
        self._controller = controller

    @property
    def time(self) -> float:
        return self._controller.time

    @property
    def capture_session(self) -> CaptureSession:
        return self._controller.capture_session

    def seek(self, time: float) -> None:
        self._controller.seek(time)

    def close(self) -> None:
        self._controller.close()

    def _add_close_callback(self, callback: Callable[[], None]) -> None:
        self._controller._add_close_callback(callback)


class _CallbackBackendView:
    """A viewer controlled through plain callables, such as the notebook widget."""

    def __init__(
        self,
        native: Any,
        *,
        initial_time: float,
        seek: Callable[[float], None],
        close: Callable[[], None],
        read_time: Callable[[], float] | None,
        capture_session: CaptureSession | None,
    ) -> None:
        self.native = native
        self._time = initial_time
        self._seek, self._close, self._read_time = seek, close, read_time
        self.capture_session = capture_session

    @property
    def time(self) -> float:
        return self._time if self._read_time is None else float(self._read_time())

    def seek(self, time: float) -> None:
        self._seek(time)
        self._time = time

    def close(self) -> None:
        self._close()

    def _add_close_callback(self, callback: Callable[[], None]) -> None:
        # Callables carry no native close event; close() is the only path.
        pass


def _close_after_error(close: Callable[[], None], error: BaseException) -> None:
    """Preserve a construction/capture failure while still attempting cleanup."""
    try:
        close()
    except BaseException as cleanup:
        _add_cleanup_note(error, f"View cleanup also failed: {cleanup!r}")
