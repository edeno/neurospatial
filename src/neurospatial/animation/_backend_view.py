"""Typed boundary between a scene view and existing native viewers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FrameCapture = Callable[[float], NDArray[np.uint8]]
CaptureSession = Callable[[], AbstractContextManager[FrameCapture]]


class _ViewController(Protocol):
    """A viewer's controls use absolute recording seconds."""

    @property
    def time(self) -> float: ...

    def seek(self, time: float) -> None: ...

    def capture_session(self) -> AbstractContextManager[FrameCapture]: ...

    def close(self) -> None: ...


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


class _CallbackBackendView:
    """Compatibility adapter for the original AnimationView constructor."""

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
