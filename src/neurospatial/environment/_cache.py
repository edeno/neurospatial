"""One memo of the analyses derived from an environment's published geometry.

Every value here depends only on geometry that never changes after
publication, so a value computed once stays correct for the environment's
lifetime. ``with_metadata`` and ``with_regions`` copies keep the same geometry
and therefore share their parent's cache; every derivation, reader and
unpickled environment starts with an empty one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from neurospatial import _constants
from neurospatial._lru import lru_get, lru_put

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from scipy import sparse
    from scipy.spatial import cKDTree

    from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)


class CacheKey(Generic[_T]):
    """Name of one cached analysis whose value has type ``_T``.

    Parameters
    ----------
    name : str
        Label shown in ``repr``; keys compare by identity.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"CacheKey({self.name!r})"


class RecentKey(Generic[_K, _T]):
    """Name of a family of cached values, keeping only the most recently used.

    Parameters
    ----------
    name : str
        Label shown in ``repr``; keys compare by identity.
    limit : int
        Maximum number of values kept; the least recently used go first.
    """

    __slots__ = ("limit", "name")

    def __init__(self, name: str, *, limit: int) -> None:
        self.name = name
        self.limit = limit

    def __repr__(self) -> str:
        return f"RecentKey({self.name!r}, limit={self.limit})"


class AnalysisCache:
    """Thread-safe memo of geometry-derived analyses for one environment.

    Each key has its own lock. The first request for a key computes its value
    while holding that key's lock, so concurrent first requests compute it
    once and every caller receives the same object. A computation may request
    other keys (a kernel needs the diffusion geometry, which needs the cell
    faces): it takes their locks, and no global lock is held while computing.
    Keys depend on each other without cycles, so threads cannot wait on each
    other in a loop. The locks are re-entrant, so a computation that asks for
    its own key fails with ``RecursionError`` instead of hanging.

    Pickling and copying a cache produce a new empty cache: computed values
    and locks are never serialized.
    """

    __slots__ = ("_guard", "_locks", "_values")

    def __init__(self) -> None:
        self._values: dict[object, Any] = {}
        self._locks: dict[object, threading.RLock] = {}
        self._guard = threading.Lock()

    def __reduce__(self) -> tuple[type[AnalysisCache], tuple[()]]:
        """Rebuild as an empty cache; values and locks are not serialized."""
        return (AnalysisCache, ())

    def _lock(self, key: object) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.RLock()
            return lock

    def get(self, key: CacheKey[_T], build: Callable[[], _T]) -> _T:
        """Return the cached value of ``key``, computing it on first request.

        Parameters
        ----------
        key : CacheKey
            Analysis to look up.
        build : callable
            Computes the value from the published geometry. Called at most
            once per key until :meth:`clear`, even under concurrent requests.

        Returns
        -------
        object
            The cached value; every caller receives the same object.
        """
        missing = _MISSING
        value = self._values.get(key, missing)
        if value is not missing:
            return cast("_T", value)
        with self._lock(key):
            value = self._values.get(key, missing)
            if value is missing:
                value = build()
                self._values[key] = value
            return cast("_T", value)

    def get_recent(
        self, key: RecentKey[_K, _T], item: _K, build: Callable[[], _T]
    ) -> _T:
        """Return one member of a bounded family, computing it on first request.

        Parameters
        ----------
        key : RecentKey
            Family to look up, for example the diffusion kernels.
        item : hashable
            Member of the family, for example ``(bandwidth, mode)``.
        build : callable
            Computes the member. Members of one family are computed one at a
            time; a member is computed again only after it was dropped.

        Returns
        -------
        object
            The cached member; the family keeps the ``key.limit`` most
            recently used members.
        """
        with self._lock(key):
            family = self._values.get(key)
            if family is None:
                family = self._values[key] = {}
            value = lru_get(family, item)
            if value is None:
                value = build()
                lru_put(family, item, value, limit=key.limit)
            return cast("_T", value)

    def exclusive(self, key: CacheKey[Any]) -> AbstractContextManager[object]:
        """Let one thread at a time use a value that grows in place.

        Eigenbasis holders gain modes while they are used. Read and grow such
        a value inside ``with cache.exclusive(key):`` so a concurrent caller
        never sees it half updated. The same thread may request ``key`` again
        inside the block.

        Parameters
        ----------
        key : CacheKey
            Value to reserve.

        Returns
        -------
        context manager
            Holds the key's lock for the duration of the ``with`` block.
        """
        return self._lock(key)

    def clear(self) -> None:
        """Drop every cached value; later requests compute them again."""
        with self._guard:
            self._values.clear()

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` currently holds a computed value."""
        return key in self._values

    def __len__(self) -> int:
        """Number of keys that currently hold a computed value."""
        return len(self._values)


_MISSING = object()

KDTREE: CacheKey[cKDTree] = CacheKey("kdtree")
FACE_GEOMETRY: CacheKey[FaceGeometry] = CacheKey("face_geometry")
DIFFUSION_GEOMETRY: CacheKey[DiffusionGeometry] = CacheKey("diffusion_geometry")
DIFFUSION_EIGENBASIS: CacheKey[DiffusionEigenbasisCache] = CacheKey(
    "diffusion_eigenbasis"
)
MRF_EIGENBASIS: CacheKey[MRFEigenbasisCache] = CacheKey("mrf_eigenbasis")
DIFFERENTIAL_OPERATOR: CacheKey[sparse.csc_matrix] = CacheKey("differential_operator")
DIVERGENCE_OPERATOR: CacheKey[sparse.csc_matrix] = CacheKey("divergence_operator")
BOUNDARY_BINS: CacheKey[NDArray[Any]] = CacheKey("boundary_bins")
BIN_ATTRIBUTES: CacheKey[pd.DataFrame] = CacheKey("bin_attributes")
EDGE_ATTRIBUTES: CacheKey[pd.DataFrame] = CacheKey("edge_attributes")
DIFFUSION_KERNELS: RecentKey[tuple[float, KernelMode], NDArray[Any]] = RecentKey(
    "diffusion_kernels", limit=_constants.MAX_CACHED_KERNELS
)
GAUSSIAN_KERNELS: RecentKey[float, NDArray[Any]] = RecentKey(
    "gaussian_kernels", limit=_constants.MAX_CACHED_KERNELS
)
