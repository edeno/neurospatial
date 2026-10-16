"""Installed adapters from external spatial data to built-in owned snapshots.

Adapters are objects declared in the ``neurospatial.snapshot_adapters.v1``
entry-point group. Publication retains their normalized data, never their
provider objects or callbacks. Saved normalized records use built-in codecs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["SnapshotAdapter"]

_SnapshotRole = Literal["locator", "track", "arena", "binning"]
_ROLES = ("locator", "track", "arena", "binning")
_ENTRY_POINT_GROUP = "neurospatial.snapshot_adapters.v1"


@dataclass(frozen=True, eq=False)
class SnapshotAdapter:
    """Adapt one external provider type and saved format to built-in data.

    Parameters
    ----------
    name : str
        Globally unique custom codec name. Built-in names are reserved.
    version : int
        Positive integer version of the custom payload accepted by ``decode``.
        This is distinct from the entry-point API version in the group name.
    role : {"locator", "track", "arena", "binning"}
        Snapshot role. Results must be supported built-in lookup records,
        ``TrackLinearization``, built-in Arena data, or ``Binning``, respectively.
    provider_type : type
        Exact external type adapted by ``snapshot``. Subclasses need their own
        declaration; the same type can have different adapters for distinct roles.
    snapshot : callable
        Pure ``snapshot(provider)`` function returning built-in data. It must
        not mutate the provider or retain provider callbacks in its result.
    decode : callable
        Pure ``decode(metadata, arrays, prefix)`` function returning built-in
        data from this custom codec/version. Array keys retain the reader's
        namespace. The callback receives detached metadata and array buffers.

    Notes
    -----
    Register the adapter object itself as an installed entry point, rather than
    a factory. Its package is required to read original custom payloads. After
    publication and saving, canonical built-in records need no adapter package.
    Adapters do not register custom query algorithms or layout builders.
    """

    name: str
    version: int
    role: _SnapshotRole
    provider_type: type[Any]
    snapshot: Callable[[Any], object]
    decode: Callable[[Mapping[str, Any], Mapping[str, NDArray[Any]], str], object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("SnapshotAdapter.name must be a nonempty codec name.")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("SnapshotAdapter.version must be a positive integer.")
        if self.role not in _ROLES:
            raise ValueError(f"SnapshotAdapter.role must be one of {_ROLES!r}.")
        if not isinstance(self.provider_type, type):
            raise TypeError("SnapshotAdapter.provider_type must be a Python type.")
        if not callable(self.snapshot) or not callable(self.decode):
            raise TypeError("SnapshotAdapter.snapshot and decode must be callable.")


def _arena_types() -> tuple[type[Any], ...]:
    from neurospatial.geometry._arena import (
        EgocentricSurround,
        MaskArena,
        OpenArena,
        TrackArena,
        UnknownArena,
        _CombinedArena,
        _TransformedArena,
    )

    return (
        UnknownArena,
        OpenArena,
        MaskArena,
        TrackArena,
        EgocentricSurround,
        _TransformedArena,
        _CombinedArena,
    )


@lru_cache(maxsize=1)
def _snapshot_adapters() -> tuple[SnapshotAdapter, ...]:
    """Discover installed objects once, rejecting ambiguous declarations."""
    from neurospatial.geometry._codecs import _CODECS

    reserved = set(_CODECS) | {kind.__name__ for kind in _arena_types()}
    adapters: list[SnapshotAdapter] = []
    names: set[str] = set()
    providers: set[tuple[type[Any], str]] = set()
    entries = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    for entry in sorted(entries, key=lambda item: (item.name, item.value)):
        try:
            adapter = entry.load()
        except ImportError as exc:
            raise ValueError(
                f"Cannot import snapshot adapter entry point {entry.value!r}. "
                "Install its package and required dependencies."
            ) from exc
        if not isinstance(adapter, SnapshotAdapter):
            raise TypeError(
                f"Snapshot entry point {entry.value!r} must expose a "
                "SnapshotAdapter object, rather than a class or factory."
            )
        if adapter.name in reserved:
            raise ValueError(
                f"Snapshot adapter {adapter.name!r} conflicts with a built-in codec. "
                "Choose a globally unique adapter name."
            )
        if adapter.name in names:
            raise ValueError(
                f"duplicate snapshot adapter name {adapter.name!r}. "
                "Rename the adapter or remove the conflicting installed package."
            )
        provider_role = (adapter.provider_type, adapter.role)
        if provider_role in providers:
            raise ValueError(
                "duplicate snapshot adapter for provider "
                f"{adapter.provider_type.__module__}.{adapter.provider_type.__qualname__} "
                f"and role {adapter.role!r}. Keep one installed adapter per provider role."
            )
        names.add(adapter.name)
        providers.add(provider_role)
        adapters.append(adapter)
    return tuple(adapters)


def _normalize_snapshot(value: object, role: _SnapshotRole) -> object:
    """Validate an adapter result and rebuild owned canonical data records."""
    if role == "arena":
        from neurospatial.geometry._arena import Arena, decode_arena, encode_arena

        if type(value) not in _arena_types():
            raise ValueError("snapshot adapters must return a built-in arena record.")
        meta, arrays = encode_arena(cast("Arena", value))
        return decode_arena(meta, arrays)

    from neurospatial.geometry._binning import Binning, TrackLinearization
    from neurospatial.geometry._codecs import _LOCATOR_TYPES, decode, encode

    expected = {
        "locator": _LOCATOR_TYPES,
        "track": (TrackLinearization,),
        "binning": (Binning,),
    }
    if type(value) not in expected[role]:
        raise ValueError(f"snapshot adapters must return a built-in {role} record.")
    meta, arrays = encode(value, "snapshot")
    return decode(meta, arrays, "snapshot")


def _snapshot_provider(provider: object, role: _SnapshotRole) -> object:
    """Convert a registered external provider to owned built-in scientific data."""
    for adapter in _snapshot_adapters():
        if type(provider) is adapter.provider_type and role == adapter.role:
            return _normalize_snapshot(adapter.snapshot(provider), role)
    hint = (
        " A layout engine instead supplies its geometry from a binning() "
        "method that returns a Binning."
        if role == "binning"
        else ""
    )
    raise TypeError(
        f"No installed snapshot adapter for {type(provider).__name__} "
        f"with role {role!r}. Install the provider's adapter package or supply "
        f"supported built-in snapshot data.{hint}"
    )


def _decode_snapshot(
    meta: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str
) -> object:
    """Read an installed custom format and return only built-in snapshot data."""
    name = meta.get("codec")
    for adapter in _snapshot_adapters():
        if adapter.name != name:
            continue
        version = meta.get("codec_version")
        if type(version) is not int or version != adapter.version:
            raise ValueError(
                f"Cannot load {name!r} codec version {version!r}; the installed "
                f"adapter supports version {adapter.version}. Install an adapter "
                "package supporting the saved version."
            )
        detached_meta = copy.deepcopy(dict(meta))
        detached_arrays = {
            key: np.array(value, copy=True) for key, value in arrays.items()
        }
        try:
            value = adapter.decode(detached_meta, detached_arrays, prefix)
            return _normalize_snapshot(value, adapter.role)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(
                f"invalid {name} snapshot payload at {prefix}: {exc}"
            ) from exc
    raise ValueError(
        f"cannot load {name!r}: no codec or installed snapshot adapter is "
        "registered for it. Fix: install the package providing this snapshot "
        "adapter, then load the original record and save canonical built-in data."
    )
