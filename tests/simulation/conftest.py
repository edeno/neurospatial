"""Shared fixtures for simulation tests."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def simple_2d_env():
    """Create a simple 2D square environment for testing.

    Returns
    -------
    Environment
        100x100 cm square arena with 2 cm bins.
    """
    # Create a grid of sample points
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    samples = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(samples, bin_size=2.0)
    env.units = "cm"
    env.frame = "test_arena"
    return env


@pytest.fixture
def simple_1d_env():
    """Create a simple 1D linear track for testing.

    Returns
    -------
    Environment
        200 cm linear track with 2 cm bins.
    """
    # Create sample points along a line
    samples = np.linspace(0, 200, 100).reshape(-1, 1)

    # Note: 1D environments require GraphLayout, which we'll implement later
    # For now, this will create a regular 1D grid
    env = Environment.from_samples(samples, bin_size=2.0)
    env.units = "cm"
    env.frame = "linear_track"
    return env


@pytest.fixture
def rng():
    """Create a deterministic random number generator for reproducible tests.

    Returns
    -------
    np.random.Generator
        Seeded random number generator.
    """
    return np.random.default_rng(42)


@pytest.fixture
def sample_positions():
    """Create sample trajectory positions for testing.

    Returns
    -------
    ndarray, shape (1000, 2)
        Random positions in a 100x100 arena.
    """
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, size=(1000, 2))


@pytest.fixture
def sample_times():
    """Create sample time points for testing.

    Returns
    -------
    ndarray, shape (1000,)
        Time points at 100 Hz sampling rate (10 seconds total).
    """
    return np.linspace(0, 10, 1000)
