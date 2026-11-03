"""The decoding operation's chance level, in the metric of its error."""

from __future__ import annotations

import numpy as np
import pytest
from operations import chance_errors


def _env(samples: np.ndarray):
    from neurospatial import Environment

    return Environment.from_samples(samples, bin_size=4.0, units="cm")


@pytest.fixture(scope="module")
def u_track() -> np.ndarray:
    """Samples on a U: two 100 cm arms 24 cm apart, joined at the bottom."""
    rng = np.random.default_rng(3)
    up = rng.uniform(0, 100, 3000)
    left = np.column_stack([rng.uniform(0, 6, 3000), up])
    right = np.column_stack([rng.uniform(30, 36, 3000), up])
    base = np.column_stack([rng.uniform(0, 36, 1000), rng.uniform(0, 6, 1000)])
    return np.vstack([left, right, base])


def test_geodesic_chance_exceeds_straight_line_on_a_u_track(u_track):
    env = _env(u_track)
    errors, disconnected = chance_errors(env, u_track, seed=0)
    assert disconnected == 0.0
    assert np.all(np.isfinite(errors))
    order = np.random.default_rng(1).permutation(len(u_track))
    straight = np.linalg.norm(u_track - u_track[order], axis=1)
    # Arm to arm is 24 cm straight across but about 200 cm along the track.
    assert np.median(errors) > 1.3 * np.median(straight)  # 63 vs 39 cm here


def test_chance_shuffles_within_each_disconnected_part():
    rng = np.random.default_rng(4)
    first = rng.uniform(0, 40, (2000, 2))
    second = rng.uniform(0, 40, (2000, 2)) + np.array([200.0, 0.0])
    samples = np.vstack([first, second])
    env = _env(samples)
    errors, disconnected = chance_errors(env, samples, seed=0)
    # A global shuffle puts about half the pairs in the other part (inf apart);
    # within-part pairs stay finite and no longer than a box's diagonal path.
    assert 0.4 < disconnected < 0.6
    assert np.all(np.isfinite(errors))
    assert np.max(errors) < 2 * 40 * 1.1
