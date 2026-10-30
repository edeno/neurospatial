"""Seed derivation shared by the simulators."""

from __future__ import annotations

import numpy as np


def _stream_seeds(seed: int | None, n_streams: int) -> list[int]:
    """Seeds of ``n_streams`` independent streams spawned from ``seed``.

    Each is the first 32-bit word of a child of ``np.random.SeedSequence(seed)``,
    so the streams are independent and the same ``seed`` gives the same seeds.
    ``None`` draws fresh entropy, leaving the streams nondeterministic but still
    mutually independent.
    """
    return [
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(seed).spawn(n_streams)
    ]
