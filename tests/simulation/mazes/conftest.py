"""Shared fixtures for maze tests."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def rng():
    """Provide seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_positions(rng):
    """Generate sample position data for testing."""
    return rng.uniform(0, 100, (500, 2))


@pytest.fixture
def sample_env(sample_positions):
    """Create a simple 2D environment for testing."""
    env = Environment.from_samples(sample_positions, bin_size=5.0)
    env.units = "cm"
    return env
