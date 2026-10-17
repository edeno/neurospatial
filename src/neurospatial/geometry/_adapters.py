"""Declaration of an installed snapshot adapter.

Adapters are objects declared in the ``neurospatial.snapshot_adapters.v1``
entry-point group. Publication retains their normalized data, never their
provider objects or callbacks. Saved normalized records use built-in codecs;
``_codecs`` discovers installed adapters and normalizes their data.
``SnapshotAdapter`` is public as ``neurospatial.extensions.SnapshotAdapter``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

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
