"""Private random-stream helpers shared by seeded analyses.

Every random stream of one call derives from one entropy value, so a seed
reproduces the whole call and adding a stream leaves the others unchanged.
"""

from __future__ import annotations

import zlib
from typing import cast

import numpy as np


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


def _seed_sequence(rng: np.random.Generator | int | None) -> np.random.SeedSequence:
    """The seed sequence behind ``rng``, as ``np.random.default_rng`` reads it.

    An ``int`` or ``None`` gives ``np.random.SeedSequence(rng)``. A generator
    gives the sequence it was seeded from, so ``np.random.default_rng(0)``
    derives the same streams as ``0``.
    """
    if isinstance(rng, np.random.Generator):
        return cast("np.random.SeedSequence", rng.bit_generator.seed_seq)
    return np.random.SeedSequence(rng)
