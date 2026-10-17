"""Internal utilities for stats module.

This module contains shared utility functions used across the stats submodules.
These functions are private (underscore prefix) and not part of the public API.
"""

from __future__ import annotations

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
