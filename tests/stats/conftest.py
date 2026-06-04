"""Shared fixtures for the stats test subsystem."""

import numpy as np
import pytest


@pytest.fixture
def concentrated_angles():
    """A concentrated sample of 20 angles wrapped to ``[0, 2*pi)``.

    Drawn from a von Mises distribution (``mu=0.5``, ``kappa=4.0``) with a
    fixed seed so weighted-Rayleigh regression tests are deterministic.
    """
    rng = np.random.default_rng(0)
    angles = rng.vonmises(mu=0.5, kappa=4.0, size=20)
    return angles % (2 * np.pi)
