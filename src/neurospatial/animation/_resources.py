"""Ownership of decoder resources, separate from scientific overlay data."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast


class _CleanupError(RuntimeError):
    """Python 3.10 equivalent exposing every independent cleanup failure."""

    def __init__(self, message: str, exceptions: Sequence[BaseException]) -> None:
        super().__init__(message)
        self.exceptions = tuple(exceptions)

    def __repr__(self) -> str:
        # Cleanup notes use repr, so retain nested causes just like ExceptionGroup.
        return f"{type(self).__name__}({self.args[0]!r}, {self.exceptions!r})"


def _cleanup_error(message: str, failures: Sequence[BaseException]) -> BaseException:
    """Create a grouped error without requiring Python 3.11 or a backport."""
    group = getattr(builtins, "BaseExceptionGroup", None)
    if group is not None:
        return cast("BaseException", group(message, list(failures)))
    return _CleanupError(message, failures)


def _raise_cleanup_errors(message: str, failures: list[BaseException]) -> None:
    if failures:
        raise _cleanup_error(message, failures)


def _add_cleanup_note(error: BaseException, message: str) -> None:
    """Retain an original failure while recording cleanup diagnostics."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)
    else:
        notes = list(getattr(error, "__notes__", ()))
        notes.append(message)
        setattr(error, "__notes__", notes)  # noqa: B010 - Python 3.10 typing compatibility


class _OwnedReaders:
    """Close explicitly owned readers once, including after partial failures."""

    def __init__(self, readers: Iterable[Any] = ()) -> None:
        self._readers: dict[int, Any] = {}
        self._closed = False
        for reader in readers:
            self.add(reader)

    def add(self, reader: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot add readers to a closed resource owner.")
        if callable(getattr(reader, "close", None)):
            self._readers.setdefault(id(reader), reader)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        readers, self._readers = self._readers, {}
        failures = []
        for reader in readers.values():
            try:
                reader.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise _cleanup_error(
                "Failed to close owned animation video readers.", failures
            )


def _bind_cleanup(handle: Any, cleanup: Callable[[], None]) -> Any:
    """Close a returned native handle before its separately owned resources."""
    original_close = handle.close
    closed = False

    def close(*args: Any, **kwargs: Any) -> Any:
        nonlocal closed
        if closed:
            return None
        closed = True
        result = None
        failures = []
        try:
            result = original_close(*args, **kwargs)
        except BaseException as exc:
            failures.append(exc)
        try:
            cleanup()
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise _cleanup_error(
                "Failed to close the animation and its resources.", failures
            )
        return result

    object.__setattr__(handle, "close", close)
    return handle
