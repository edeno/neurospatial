"""Tests for maze base classes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments


class TestMazeDims:
    """Tests for MazeDims base class."""

    def test_is_frozen_dataclass(self):
        """MazeDims should be a frozen (immutable) dataclass."""
        from dataclasses import is_dataclass

        dims = MazeDims()
        # Verify it's a dataclass
        assert is_dataclass(dims)
        # Verify it's frozen (can't add new attributes)
        with pytest.raises(FrozenInstanceError):
            dims.new_attr = "value"  # type: ignore[attr-defined]

    def test_default_construction(self):
        """MazeDims can be constructed with no arguments."""
        dims = MazeDims()
        assert dims is not None

    def test_repr(self):
        """MazeDims has a reasonable repr."""
        dims = MazeDims()
        assert "MazeDims" in repr(dims)


class TestMazeEnvironments:
    """Tests for MazeEnvironments container class."""

    def test_construction_with_2d_only(self, sample_env):
        """MazeEnvironments can be constructed with only env_2d."""
        maze = MazeEnvironments(env_2d=sample_env)
        assert maze.env_2d is sample_env
        assert maze.env_track is None

    def test_construction_with_both_envs(self, sample_env):
        """MazeEnvironments can hold both 2D and track environments."""
        # Use same env as track for simplicity (normally would be different)
        maze = MazeEnvironments(env_2d=sample_env, env_track=sample_env)
        assert maze.env_2d is sample_env
        assert maze.env_track is sample_env

    def test_requires_env_2d(self):
        """MazeEnvironments requires env_2d argument."""
        with pytest.raises(TypeError):
            MazeEnvironments()  # type: ignore[call-arg]

    def test_env_track_defaults_to_none(self, sample_env):
        """env_track should default to None."""
        maze = MazeEnvironments(env_2d=sample_env)
        assert maze.env_track is None

    def test_is_dataclass(self, sample_env):
        """MazeEnvironments should be a dataclass."""
        from dataclasses import is_dataclass

        maze = MazeEnvironments(env_2d=sample_env)
        assert is_dataclass(maze)

    def test_repr(self, sample_env):
        """MazeEnvironments has a reasonable repr."""
        maze = MazeEnvironments(env_2d=sample_env)
        repr_str = repr(maze)
        assert "MazeEnvironments" in repr_str
        assert "env_2d" in repr_str
