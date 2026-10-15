"""Immutable declarations attached to an environment's coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, cast, overload

_T = TypeVar("_T")


class _Unset(Enum):
    VALUE = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset.VALUE


@dataclass(frozen=True)
class _Metadata:
    name: str
    units: str | None
    frame: str | None

    def __post_init__(self) -> None:
        for key in ("name", "units", "frame"):
            value = getattr(self, key)
            if not isinstance(value, str) and not (key != "name" and value is None):
                expected = "str" if key == "name" else "str or None"
                raise TypeError(f"Environment.{key} must be a {expected}.")


class _MetadataView(Generic[_T]):
    """A non-shadowable declaration backed by one metadata record."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> _MetadataView[_T]: ...

    @overload
    def __get__(self, obj: object, owner: type | None = None) -> _T: ...

    def __get__(
        self, obj: object | None, owner: type | None = None
    ) -> _MetadataView[_T] | _T:
        if obj is None:
            return self
        return cast("_T", getattr(cast("Any", obj)._metadata, self.name))

    def __set__(self, obj: object, value: object) -> None:
        raise AttributeError(
            f"Environment.{self.name} is read-only; use env.with_metadata({self.name}=...)."
        )

    def __delete__(self, obj: object) -> None:
        self.__set__(obj, None)
