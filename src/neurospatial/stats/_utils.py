"""Private random-stream and rate helpers.

Shared by the stats, encoding, replay and reactivation packages; not part of
the public API.
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


def _binomial_rate(
    n_called: int, n_total: int, confidence: float
) -> tuple[float, float, float]:
    """Rate ``n_called / n_total`` with its exact (Clopper-Pearson) interval.

    Parameters
    ----------
    n_called, n_total : int
        Successes and trials.
    confidence : float
        Confidence level of the interval, in (0, 1).

    Returns
    -------
    rate, low, high : float
        All NaN when ``n_total`` is 0.
    """
    if not n_total:
        return np.nan, np.nan, np.nan
    from scipy.stats import binomtest

    interval = binomtest(n_called, n_total).proportion_ci(
        confidence_level=confidence, method="exact"
    )
    return n_called / n_total, float(interval.low), float(interval.high)


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
