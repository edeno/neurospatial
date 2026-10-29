"""Shared fixtures for maze tests."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation import mazes
from neurospatial.simulation.mazes import MazeEnvironments

MAZE_FACTORIES = sorted(
    name
    for name in dir(mazes)
    if name.startswith("make_")
    and (name.endswith("maze") or name == "make_linear_track")
)
"""Names of the maze factories in ``neurospatial.simulation.mazes``."""


@functools.cache
def build_default_maze(name: str) -> MazeEnvironments:
    """Build a maze at its defaults once per test process.

    A built maze is immutable, so tests can share it; the slowest mazes take
    over a second to build.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the rat HexMaze's large-grid advisory
        return getattr(mazes, name)()


@pytest.fixture(scope="session", params=MAZE_FACTORIES)
def maze_name(request):
    """Each maze factory's name, in turn."""
    return request.param


@pytest.fixture(scope="session")
def default_maze(maze_name):
    """Each maze built at its defaults, shared across test modules."""
    return build_default_maze(maze_name)


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
    env = env.with_metadata(units="cm")
    return env
