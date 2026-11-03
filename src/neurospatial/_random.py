"""Private random-stream helpers shared by seeded analyses.

``rng`` is read as ``np.random.default_rng(rng)``. A call that needs several
random streams draws one entropy value from it,
``int(np.random.default_rng(rng).integers(2**63))``, and derives every
stream from ``np.random.SeedSequence(entropy)`` and a key. So ``rng=0`` and a
fresh ``np.random.default_rng(0)`` give the same streams, a generator's
state counts (the draw advances it, so successive calls differ, and two
generators in one state agree), and adding a stream leaves the others
unchanged. The simulators keep their own derivation (:func:`_seed_sequence`).
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


def _entropy(rng: np.random.Generator | int | None) -> int:
    """The entropy every random stream of one call derives from.

    The first draw of ``np.random.default_rng(rng)``: an int ``k`` and a
    fresh ``np.random.default_rng(k)`` give the same value, a generator
    gives a value set by its state and advances it by one draw, and
    ``None`` gives fresh entropy.
    """
    return int(np.random.default_rng(rng).integers(2**63))


def _root(rng: np.random.Generator | int | None) -> np.random.SeedSequence:
    """The seed sequence every random stream of one call derives from.

    ``np.random.SeedSequence(_entropy(rng))``; see :func:`_entropy`.
    """
    return np.random.SeedSequence(_entropy(rng))


def _recorded_seed(root: np.random.SeedSequence) -> int:
    """The entropy of ``root``, recorded as a table's ``rng_entropy``.

    Stream ``key`` of the call is
    ``np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=key))``
    (:func:`_stream`), so the entropy determines every draw of the call.
    Passing it back as ``rng`` gives a new entropy (its own first draw), not
    the same streams.
    """
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
