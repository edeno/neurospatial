"""Tests for Barnes Maze implementation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurospatial.simulation.mazes._base import MazeDims


class TestBarnesDims:
    """Tests for BarnesDims dataclass."""

    def test_inherits_from_maze_dims(self):
        """BarnesDims should inherit from MazeDims."""
        from neurospatial.simulation.mazes.barnes import BarnesDims

        dims = BarnesDims()
        assert isinstance(dims, MazeDims)

    def test_default_values(self):
        """BarnesDims should have correct default values (Barnes 1979)."""
        from neurospatial.simulation.mazes.barnes import BarnesDims

        dims = BarnesDims()
        assert dims.diameter == 120.0
        assert dims.n_holes == 18  # Original Barnes 1979 design
        assert dims.hole_radius == 2.5

    def test_is_frozen(self):
        """BarnesDims should be frozen (immutable)."""
        from neurospatial.simulation.mazes.barnes import BarnesDims

        dims = BarnesDims()
        with pytest.raises(FrozenInstanceError):
            dims.diameter = 150.0  # type: ignore[misc]

    def test_custom_values(self):
        """BarnesDims should accept custom values."""
        from neurospatial.simulation.mazes.barnes import BarnesDims

        dims = BarnesDims(diameter=150.0, n_holes=20, hole_radius=3.0)
        assert dims.diameter == 150.0
        assert dims.n_holes == 20
        assert dims.hole_radius == 3.0


class TestMakeBarnesMaze:
    """Tests for make_barnes_maze factory function."""

    def test_returns_maze_environments(self):
        """make_barnes_maze should return MazeEnvironments."""
        from neurospatial.simulation.mazes._base import MazeEnvironments
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert isinstance(maze, MazeEnvironments)

    def test_env_2d_is_valid_environment(self):
        """env_2d should be a valid, fitted Environment."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        # Should have n_bins > 0 if fitted
        assert maze.env_2d.n_bins > 0
        assert maze.env_2d._is_fitted

    def test_env_2d_has_correct_units(self):
        """env_2d should have units set to 'cm'."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert maze.env_2d.units == "cm"

    def test_env_2d_is_circular(self):
        """env_2d should have a circular platform extent."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0)
        maze = make_barnes_maze(dims=dims)

        # Get spatial extent from bin_centers
        bin_centers = maze.env_2d.bin_centers
        x_min, x_max = bin_centers[:, 0].min(), bin_centers[:, 0].max()
        y_min, y_max = bin_centers[:, 1].min(), bin_centers[:, 1].max()

        # X and Y extent should both be approximately equal to diameter
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        expected_extent = dims.diameter

        # Allow 20% tolerance for binning effects
        assert x_extent > expected_extent * 0.7
        assert x_extent < expected_extent * 1.3
        assert y_extent > expected_extent * 0.7
        assert y_extent < expected_extent * 1.3

        # Should be roughly circular (aspect ratio near 1.0)
        aspect_ratio = x_extent / y_extent
        assert 0.8 < aspect_ratio < 1.2

    def test_platform_centered_at_origin(self):
        """Platform should be centered at origin (0, 0)."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        bin_centers = maze.env_2d.bin_centers

        # Center of mass should be near origin
        center_x = bin_centers[:, 0].mean()
        center_y = bin_centers[:, 1].mean()

        assert abs(center_x) < 5.0
        assert abs(center_y) < 5.0

    def test_default_18_holes(self):
        """Barnes maze should have 18 holes by default (original design)."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()

        # Should have hole_0 through hole_17
        for i in range(18):
            assert f"hole_{i}" in maze.env_2d.regions

        # hole_18 should not exist
        assert "hole_18" not in maze.env_2d.regions

    def test_custom_number_of_holes(self):
        """Custom n_holes should be respected."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(n_holes=12)
        maze = make_barnes_maze(dims=dims)

        # Should have hole_0 through hole_11
        for i in range(12):
            assert f"hole_{i}" in maze.env_2d.regions

        # hole_12 should not exist
        assert "hole_12" not in maze.env_2d.regions

    def test_holes_on_perimeter(self):
        """Holes should be positioned on the perimeter of the platform."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0, n_holes=18, hole_radius=2.5)
        maze = make_barnes_maze(dims=dims)

        radius = dims.diameter / 2.0
        expected_hole_radius = radius - dims.hole_radius

        # Check each hole is at the expected radius from origin
        for i in range(dims.n_holes):
            hole = maze.env_2d.regions[f"hole_{i}"]
            assert hole.kind == "point"

            # Distance from origin
            dist = np.sqrt(hole.data[0] ** 2 + hole.data[1] ** 2)

            # Should be close to expected_hole_radius
            assert abs(dist - expected_hole_radius) < 0.1

    def test_holes_evenly_distributed(self):
        """Holes should be evenly distributed around perimeter."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=120.0, n_holes=18)
        maze = make_barnes_maze(dims=dims)

        # Get angles of all holes
        angles = []
        for i in range(dims.n_holes):
            hole = maze.env_2d.regions[f"hole_{i}"]
            angle = np.arctan2(hole.data[1], hole.data[0])
            angles.append(angle)

        # Sort angles
        angles = np.sort(angles)

        # Check angular spacing is consistent
        expected_spacing = 2 * np.pi / dims.n_holes
        for i in range(len(angles) - 1):
            spacing = angles[i + 1] - angles[i]
            assert abs(spacing - expected_spacing) < 0.01

        # Check wrap-around spacing
        wrap_spacing = (angles[0] + 2 * np.pi) - angles[-1]
        assert abs(wrap_spacing - expected_spacing) < 0.01

    def test_has_escape_hole_region(self):
        """Should have an 'escape_hole' region."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert "escape_hole" in maze.env_2d.regions

    def test_escape_hole_is_one_of_holes(self):
        """escape_hole should match one of the numbered holes."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        escape_hole = maze.env_2d.regions["escape_hole"]

        # escape_hole should match hole_0 by default
        hole_0 = maze.env_2d.regions["hole_0"]
        assert np.allclose(escape_hole.data, hole_0.data, atol=0.01)

    def test_custom_escape_hole_index(self):
        """Custom escape_hole_index should be respected."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        # Set escape hole to hole_9
        maze = make_barnes_maze(escape_hole_index=9)
        escape_hole = maze.env_2d.regions["escape_hole"]
        hole_9 = maze.env_2d.regions["hole_9"]

        assert np.allclose(escape_hole.data, hole_9.data, atol=0.01)

    def test_env_track_is_none(self):
        """Barnes maze should not have a track graph (open field)."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        maze = make_barnes_maze()
        assert maze.env_track is None

    def test_custom_bin_size(self):
        """Custom bin_size should affect discretization."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims()

        maze_fine = make_barnes_maze(dims=dims, bin_size=1.0)
        maze_coarse = make_barnes_maze(dims=dims, bin_size=5.0)

        # Finer bins should result in more bins
        assert maze_fine.env_2d.n_bins > maze_coarse.env_2d.n_bins

    def test_custom_dims(self):
        """Custom dimensions should be respected."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        dims = BarnesDims(diameter=200.0, n_holes=24)
        maze = make_barnes_maze(dims=dims)

        # Check that platform extent is larger
        bin_centers = maze.env_2d.bin_centers
        x_extent = bin_centers[:, 0].max() - bin_centers[:, 0].min()
        y_extent = bin_centers[:, 1].max() - bin_centers[:, 1].min()

        # Should be larger than default (120 cm)
        assert x_extent > 140  # Larger than 120
        assert y_extent > 140

        # Should have 24 holes
        assert "hole_23" in maze.env_2d.regions
        assert "hole_24" not in maze.env_2d.regions

    def test_none_dims_uses_defaults(self):
        """dims=None should use default dimensions."""
        from neurospatial.simulation.mazes.barnes import BarnesDims, make_barnes_maze

        maze_default = make_barnes_maze(dims=None)
        maze_explicit = make_barnes_maze(dims=BarnesDims())

        # Both should have similar number of bins
        assert maze_default.env_2d.n_bins == maze_explicit.env_2d.n_bins

        # Both should have 18 holes
        assert "hole_17" in maze_default.env_2d.regions
        assert "hole_17" in maze_explicit.env_2d.regions


class TestBarnesMazeDocstrings:
    """Tests for docstrings and examples."""

    def test_make_barnes_maze_has_docstring(self):
        """make_barnes_maze should have a docstring."""
        from neurospatial.simulation.mazes.barnes import make_barnes_maze

        assert make_barnes_maze.__doc__ is not None
        assert len(make_barnes_maze.__doc__) > 100  # Reasonable length

    def test_barnes_dims_has_docstring(self):
        """BarnesDims should have a docstring."""
        from neurospatial.simulation.mazes.barnes import BarnesDims

        assert BarnesDims.__doc__ is not None
