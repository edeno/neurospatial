"""Recency bookkeeping for the per-environment kernel caches."""

from neurospatial._lru import lru_get, lru_put


def test_put_drops_least_recently_used_beyond_limit():
    cache: dict[str, int] = {}
    for key, value in [("a", 1), ("b", 2), ("c", 3)]:
        lru_put(cache, key, value, limit=2)
    assert cache == {"b": 2, "c": 3}


def test_get_marks_entry_most_recently_used():
    cache = {"a": 1, "b": 2}
    assert lru_get(cache, "a") == 1
    lru_put(cache, "c", 3, limit=2)
    assert list(cache) == ["a", "c"]


def test_get_missing_key_returns_none_and_keeps_cache():
    cache = {"a": 1}
    assert lru_get(cache, "b") is None
    assert cache == {"a": 1}


def test_put_existing_key_replaces_without_evicting():
    cache = {"a": 1, "b": 2}
    lru_put(cache, "a", 10, limit=2)
    assert cache == {"b": 2, "a": 10}
