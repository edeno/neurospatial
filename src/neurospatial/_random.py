"""Private random-stream helpers shared by seeded analyses.

``rng`` is read as ``np.random.default_rng(rng)``. A call that needs several
random streams spawns one child of the seed sequence behind ``rng`` and
derives every stream from it, so ``rng=0`` and a fresh
``np.random.default_rng(0)`` give the same streams, a generator gives new
streams on each call, a seed reproduces the whole call, and adding a stream
leaves the others unchanged.
"""

from __future__ import annotations

import zlib
from typing import cast

import numpy as np


def _seed_sequence(rng: np.random.Generator | int | None) -> np.random.SeedSequence:
    """The seed sequence behind ``rng``, as ``np.random.default_rng`` reads it.

    An ``int`` or ``None`` gives ``np.random.SeedSequence(rng)``. A generator
    gives the sequence it was seeded from, so ``np.random.default_rng(0)``
    derives the same streams as ``0``.
    """
    if isinstance(rng, np.random.Generator):
        return cast("np.random.SeedSequence", rng.bit_generator.seed_seq)
    return np.random.SeedSequence(rng)


def _root(rng: np.random.Generator | int | None) -> np.random.SeedSequence:
    """The seed sequence every random stream of one call derives from.

    The next child spawned from :func:`_seed_sequence`: ``rng=0`` and a
    fresh ``np.random.default_rng(0)`` give the same root, and each call with
    one generator spawns a new child, so its streams differ.
    """
    return cast("np.random.SeedSequence", _seed_sequence(rng).spawn(1)[0])


def _recorded_seed(
    rng: np.random.Generator | int | None, root: np.random.SeedSequence
) -> int | None:
    """The int that, passed as ``rng``, gives ``root`` again.

    The seed itself, or the entropy drawn for ``None``. A generator's root
    depends on how many children it has spawned, which no int seed repeats,
    so none is recorded.
    """
    if isinstance(rng, np.random.Generator):
        return None
    return int(cast("int", root.entropy))


def _stream(root: np.random.SeedSequence, *key: int) -> np.random.Generator:
    """An independent random stream derived from ``root`` and ``key``."""
    return np.random.default_rng(
        np.random.SeedSequence(
            root.entropy,
            spawn_key=(*root.spawn_key, *key),
            pool_size=root.pool_size,
        )
    )


def _name_key(name: str) -> int:
    """A stable spawn-key entry for ``name``, independent of other names."""
    return zlib.crc32(name.encode())
