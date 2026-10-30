"""Posteriors shared by the replay tests."""

from __future__ import annotations

import numpy as np


def wrapped_bump(env, centers, sigma):
    """Posterior rows of a Gaussian bump wrapped around the track's ends."""
    position = env.bin_centers[:, 0]
    length = float(env.dimension_ranges[0][1] - env.dimension_ranges[0][0])
    distance = (position[None, :] - np.asarray(centers)[:, None] + length / 2) % length
    rows = np.exp(-0.5 * ((distance - length / 2) / sigma) ** 2)
    return rows / rows.sum(axis=1, keepdims=True)


def template_stream(seed, name):
    """The random stream one template draws from: the seed and its name."""
    import zlib

    entropy = np.random.SeedSequence(seed).entropy
    return np.random.default_rng(
        np.random.SeedSequence(entropy, spawn_key=(zlib.crc32(name.encode()),))
    )
