"""Internal utilities for stats module.

This module contains shared utility functions used across the stats submodules.
These functions are private (underscore prefix) and not part of the public API.
"""

from __future__ import annotations

import zlib

import numpy as np


def _ensure_rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    """Convert rng parameter to a Generator instance.

    Parameters
    ----------
    rng : np.random.Generator | int | None
        Random number generator, seed, or None.

    Returns
    -------
    np.random.Generator
        A random number generator instance.
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _entropy(rng: np.random.Generator | int | None) -> int:
    """The entropy every random stream of one call derives from.

    A generator contributes one draw, so the call advances it; a seed or
    ``None`` gives the entropy of ``np.random.SeedSequence(rng)``.
    """
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(2**63))
    return int(np.random.SeedSequence(rng).entropy)  # type: ignore[arg-type]


def _stream(entropy: int, *key: int) -> np.random.Generator:
    """An independent random stream derived from ``entropy`` and ``key``."""
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=key))


def _name_key(name: str) -> int:
    """A stable spawn-key entry for ``name``, independent of other names."""
    return zlib.crc32(name.encode())
