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
