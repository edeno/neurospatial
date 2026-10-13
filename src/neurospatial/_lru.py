"""Least-recently-used bookkeeping for small per-environment caches.

The caches are plain dicts, whose insertion order records recency. This module
has no internal dependencies, so every package tier can use it.
"""

from __future__ import annotations

from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def lru_get(cache: dict[K, V], key: K) -> V | None:
    """Return ``cache[key]`` and mark it most recently used, or ``None``.

    Parameters
    ----------
    cache : dict
        Cache whose insertion order is its recency order.
    key : hashable
        Entry to look up.

    Returns
    -------
    value or None
        The cached value, or ``None`` when ``key`` is absent.
    """
    value = cache.pop(key, None)
    if value is not None:
        cache[key] = value
    return value


def lru_put(cache: dict[K, V], key: K, value: V, *, limit: int) -> None:
    """Store ``value`` as most recently used, keeping at most ``limit`` entries.

    Parameters
    ----------
    cache : dict
        Cache whose insertion order is its recency order.
    key : hashable
        Entry to store.
    value : object
        Value to cache.
    limit : int
        Maximum number of entries; the least recently used are dropped first.
    """
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > limit:
        del cache[next(iter(cache))]
