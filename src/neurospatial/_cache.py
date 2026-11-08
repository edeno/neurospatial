"""The memo classes behind every environment's analysis cache.

:class:`AnalysisCache` holds the analyses derived from an environment's
published geometry. :class:`CacheKey` and :class:`RecentKey` name its
entries. They live below every package so that any layer can name an entry;
the environment's own keys are in :mod:`neurospatial.environment._cache`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import Any, Generic, TypeVar, cast

from neurospatial._lru import lru_get, lru_put

_T = TypeVar("_T")
_R = TypeVar("_R")
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

    Values are never edited after they are stored. A value that grows, such
    as an eigenbasis that gains modes, is an immutable record replaced whole
    by :meth:`update`, so a reader always sees a complete value.

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
        return self._once(key, build)

    def lookup_index(self, locator: object, build: Callable[[], _T]) -> _T:
        """Return a locator's search structure, building it on first request.

        Locators call this for the KD-tree, triangle index or polygon tree
        they search, so these live and are cleared with the environment's
        other analyses. Each locator has its own entry, so a derived
        environment whose locator wraps its parent's builds the parent's
        structure again in its own cache.

        Parameters
        ----------
        locator : object
            The locator the structure belongs to, compared by identity.
        build : callable
            Builds the structure from the locator's data. Called at most once
            per locator until :meth:`clear`, even under concurrent requests.

        Returns
        -------
        object
            The cached structure; every caller receives the same object.
        """
        return self._once((_LOOKUP_INDEX, locator), build)

    def _once(self, key: object, build: Callable[[], _T]) -> _T:
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

    def update(
        self,
        key: CacheKey[_T],
        initial: Callable[[], _T],
        step: Callable[[_T], tuple[_T, _R]],
    ) -> _R:
        """Replace the value of ``key`` with the one ``step`` returns.

        ``step`` receives the current value (``initial()`` on first use) and
        returns the value to store, often the same object, and a result for
        the caller. It runs while holding the key's lock, so concurrent
        updates of one key run one at a time and each sees the value the
        previous one stored: work that ``step`` records, such as an
        eigensolve, is done once.

        Parameters
        ----------
        key : CacheKey
            Value to advance.
        initial : callable
            Builds the starting value when ``key`` holds none.
        step : callable
            Maps the current value to ``(new_value, result)``. It must not
            modify the current value.

        Returns
        -------
        object
            The result ``step`` returned.
        """
        with self._lock(key):
            current = self._values.get(key, _MISSING)
            value, result = step(initial() if current is _MISSING else current)
            self._values[key] = value
            return result

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
_LOOKUP_INDEX = object()
