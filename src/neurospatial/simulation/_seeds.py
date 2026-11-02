"""Seed derivation shared by the simulators."""

from __future__ import annotations

import numpy as np

from neurospatial._random import _seed_sequence


def _stream_seeds(rng: np.random.Generator | int | None, n_streams: int) -> list[int]:
    """Seeds of ``n_streams`` independent streams spawned from ``rng``.

    Each is the first 32-bit word of a child of the seed sequence behind
    ``rng``, so the streams are independent and the same ``rng`` gives the
    same seeds. ``None`` draws fresh entropy, leaving the streams
    nondeterministic but still mutually independent.
    """
    return [
        int(child.generate_state(1)[0])
        for child in _seed_sequence(rng).spawn(n_streams)
    ]


def _concrete_rng(rng: np.random.Generator | int | None) -> np.random.Generator | int:
    """``rng``, with ``None`` replaced by the fresh entropy it would draw.

    A call that records its ``rng`` then records a value that reproduces it.
    """
    if rng is None:
        return int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    return rng


def _recorded_rng(rng: np.random.Generator | int | None) -> int | None:
    """What a result records about ``rng``: the int, else ``None``.

    A generator cannot be stored, and a session made from it is not
    reproducible from a record, so none is recorded.
    """
    return None if isinstance(rng, np.random.Generator) else rng
